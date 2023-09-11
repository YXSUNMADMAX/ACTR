r'''
    modified training script of CATs
    https://github.com/SunghwanHong/Cost-Aggregation-transformers
'''

import argparse
import os
import pickle
import random
import time
from os import path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
from termcolor import colored
from torch.utils.data import DataLoader
from models.ACTR import ACTR
import utils_training.optimize as optimize
from utils_training.evaluation import Evaluator
from utils_training.utils import parse_list, load_checkpoint, save_checkpoint, boolean_string
from data import download


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Argument parsing
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser(description='CATs Training Script')
    # Paths
    parser.add_argument('--name_exp', type=str,
                        default=time.strftime('%Y_%m_%d_%H_%M'),
                        help='name of the experiment to save')
    parser.add_argument('--snapshots', type=str, default='./snapshots')
    parser.add_argument('--pretrained', dest='pretrained', default=None,
                       help='path to pre-trained model')
    parser.add_argument('--start_epoch', type=int, default=-1,
                        help='start epoch')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size')
    parser.add_argument('--n_threads', type=int, default=24,
                        help='number of parallel threads for dataloaders')
    parser.add_argument('--seed', type=int, default=2021,
                        help='Pseudo-RNG seed')
                        
    parser.add_argument('--datapath', type=str, default='/home/dyzhao/SC_Dataset')
    parser.add_argument('--benchmark', type=str, default='pfpascal', choices=['pfpascal', 'spair'])
    parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=3e-5, metavar='LR',
                        help='learning rate (default: 3e-5)')
    parser.add_argument('--lr-backbone', type=float, default=4e-6, metavar='LR',
                        help='learning rate (default: 4e-6)')
    parser.add_argument('--scheduler', type=str, default='step', choices=['step', 'cosine'])
    parser.add_argument('--step', type=str, default='[100, 150, 250]')
    parser.add_argument('--step_gamma', type=float, default=0.5)
    parser.add_argument('--lr_decay_rate', default=0.99, type=float)

    parser.add_argument('--feature-size', type=int, default=64)
    parser.add_argument('--feature-proj-dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--mlp-ratio', type=int, default=4)
    parser.add_argument('--freeze', type=boolean_string, nargs='?', const=True, default=True)
    parser.add_argument('--augmentation', type=boolean_string, nargs='?', const=True, default=True)
    parser.add_argument('--ibot_ckp_file', type=str, default='./') ### donot forget this argument for iBOT ckp!!!!

    # Seed
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Initialize Evaluator
    Evaluator.initialize(args.benchmark, args.alpha)
    
    # Dataloader
    download.download_dataset(args.datapath, args.benchmark)
    train_dataset = download.load_dataset(args.benchmark, args.datapath, args.thres, device, 'trn', args.augmentation, args.feature_size)
    val_dataset = download.load_dataset(args.benchmark, args.datapath, args.thres, device, 'val', args.augmentation, args.feature_size)
    train_dataloader = DataLoader(train_dataset,
        batch_size=args.batch_size,
        num_workers=args.n_threads,
        shuffle=True)
    val_dataloader = DataLoader(val_dataset,
        batch_size=args.batch_size,
        num_workers=args.n_threads,
        shuffle=False)

    test_dataset = download.load_dataset(args.benchmark, args.datapath, args.thres, device, 'test', False,
                                         args.feature_size)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=args.n_threads,
                                 shuffle=False)

    # Model
    if args.freeze:
        print('Backbone frozen!')
    model = ACTR(
        ibot_ckp_file=args.ibot_ckp_file, feature_size=args.feature_size,
        depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, freeze=args.freeze)

    param_model = []
    param_backbone = []
    param_refine = []

    for name, param in model.named_parameters():
        if ('feature_extraction' not in name) and ('swin' not in name) and ('proj_query' not in name):
            param_model.append(param)
        elif 'feature_extraction' in name:
            param_backbone.append(param)
        else:
            param_refine.append(param)

    optimizer = optim.AdamW([{'params': param_model, 'lr': args.lr}, {'params': param_backbone, 'lr': args.lr_backbone}
                             , {'params': param_refine, 'lr': args.lr}],
                weight_decay=args.weight_decay)

    # Scheduler
    scheduler = \
        lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6, verbose=True)\
        if args.scheduler == 'cosine' else\
        lr_scheduler.MultiStepLR(optimizer, milestones=parse_list(args.step), gamma=args.step_gamma, verbose=True)

    if args.pretrained:
        # reload from pre_trained_model
        model, optimizer, scheduler, start_epoch, best_val = load_checkpoint(model, optimizer, scheduler,
                                                                 filename=args.pretrained)
        # now individually transfer the optimizer parts...
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        cur_snapshot = os.path.basename(os.path.dirname(args.pretrained))
    else:
        if not os.path.isdir(args.snapshots):
            os.mkdir(args.snapshots)

        cur_snapshot = args.name_exp
        if not osp.isdir(osp.join(args.snapshots, cur_snapshot)):
            os.makedirs(osp.join(args.snapshots, cur_snapshot))

        with open(osp.join(args.snapshots, cur_snapshot, 'args.pkl'), 'wb') as f:
            pickle.dump(args, f)

        best_val = 0
        start_epoch = 0

    # create summary writer
    save_path = osp.join(args.snapshots, cur_snapshot)
    train_writer = SummaryWriter(os.path.join(save_path, 'train'))
    val_writer = SummaryWriter(os.path.join(save_path, 'val'))
    test_writer = SummaryWriter(os.path.join(save_path, 'test'))

    model = nn.DataParallel(model)
    model = model.to(device)

    train_started = time.time()

    for epoch in range(start_epoch, args.epochs):
        scheduler.step(epoch)
        train_loss = optimize.train_epoch(model,
                                 optimizer,
                                 train_dataloader,
                                 device,
                                 epoch,
                                 train_writer)
        train_writer.add_scalar('train loss', train_loss, epoch)
        train_writer.add_scalar('learning_rate', scheduler.get_lr()[0], epoch)
        train_writer.add_scalar('learning_rate_backbone', scheduler.get_lr()[1], epoch)
        print(colored('==> ', 'green') + 'Train average loss:', train_loss)

        val_loss_grid, val_mean_pck = optimize.validate_epoch(model,
                                                       val_dataloader,
                                                       device,
                                                       epoch=epoch)

        test_loss_grid, test_mean_pck = optimize.test_epoch(model,
                                                          test_dataloader,
                                                          device,
                                                          epoch=0)

        print(colored('==> ', 'blue') + 'Val average grid loss :',
              val_loss_grid)
        print('mean PCK is {}'.format(val_mean_pck))
        # print(colored('==> ', 'blue') + 'epoch :', epoch + 1)
        val_writer.add_scalar('mean PCK', val_mean_pck, epoch)
        val_writer.add_scalar('val loss', val_loss_grid, epoch)

        print(colored('==> ', 'blue') + 'Test average grid loss :',
              test_loss_grid)
        print('mean PCK is {}'.format(test_mean_pck))
        test_writer.add_scalar('mean test PCK', test_mean_pck, epoch)
        test_writer.add_scalar('test loss', test_loss_grid, epoch)

        print(colored('==> ', 'blue') + 'epoch :', epoch + 1)


        is_best = test_mean_pck > best_val
        best_val = max(test_mean_pck, best_val)
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.module.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(),
                         'best_loss': best_val},
                        is_best, save_path, 'epoch_{}.pth'.format(epoch + 1))

    print(args.seed, 'Training took:', time.time()-train_started, 'seconds')
