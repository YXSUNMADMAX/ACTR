import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from utils_training.utils import flow2kps
from utils_training.evaluation import Evaluator
import os

def EPE(input_flow, target_flow, sparse=True, mean=True, sum=False):
    EPE_map = torch.norm(target_flow - input_flow, 2, 1)
    batch_size = EPE_map.size(0)
    if sparse:
        mask = (target_flow[:, 0] == 0) & (target_flow[:, 1] == 0)
        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    elif sum:
        return EPE_map.sum()
    else:
        return EPE_map.sum() / torch.sum(~mask)

def train_epoch(net, optimizer, train_loader, device, epoch, train_writer):
    n_iter = epoch * len(train_loader)
    net.train()
    running_total_loss = 0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, mini_batch in pbar:
        optimizer.zero_grad()

        flow_gt_src = mini_batch["flow_src"].to(device)

        source = F.interpolate(mini_batch["src_img"].to(device), 256, None, 'bilinear', False)
        target = F.interpolate(mini_batch["trg_img"].to(device), 256, None, 'bilinear', False)

        flow = net(source, target)
        Loss = EPE(flow, flow_gt_src)
        Loss.backward()
        optimizer.step()

        running_total_loss += Loss.item()
        train_writer.add_scalar("train_loss_per_iter", Loss.item(), n_iter)
        n_iter += 1
        pbar.set_description(
            " Training: R_total_loss: %.3f/%.3f"
            % (running_total_loss / (i + 1), Loss.item())
        )
        # if running_total_loss / (i + 1) > 17.0 and i>20:
        #     raise RuntimeError
    running_total_loss /= len(train_loader)

    return running_total_loss


def validate_epoch(net, val_loader, device, epoch):
    net.eval()
    running_total_loss = 0

    if not os.path.isdir("./vis_val/"):
        os.makedirs("./vis_val/")
    if not os.path.isdir("./vis_val/" + str(epoch)):
        os.makedirs("./vis_val/" + str(epoch))

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        pck_array = []
        for i, mini_batch in pbar:
            flow_gt = mini_batch["flow_src"].to(device)

            ### To make sure the input size is 256x256
            source = F.interpolate(mini_batch["src_img"].to(device), 256, None, 'bilinear', False)
            target = F.interpolate(mini_batch["trg_img"].to(device), 256, None, 'bilinear', False)
            pred_flow = net(source, target)
            estimated_kps = flow2kps(
                mini_batch["src_kps"].to(device),
                pred_flow,
                mini_batch["n_pts"].to(device),
            )
            eval_result = Evaluator.eval_kps_transfer(estimated_kps.cpu(), mini_batch)
            Loss = EPE(pred_flow, flow_gt)
            pck_array += eval_result["pck"]

            running_total_loss += Loss.item()
            pbar.set_description(
                " Validation R_total_loss: %.3f/%.3f"
                % (running_total_loss / (i + 1), Loss.item())
            )
        mean_pck = sum(pck_array) / len(pck_array)
        if mean_pck <= 1:
            raise RuntimeError
        print("####### Val mean pck: %.3f #######" % mean_pck)

    return running_total_loss / len(val_loader), mean_pck


def test_epoch(net, val_loader, device, epoch):
    net.eval()
    running_total_loss = 0

    if not os.path.isdir("./vis_test/"):
        os.makedirs("./vis_test/")

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        pck_array = []
        for i, mini_batch in pbar:
            flow_gt = mini_batch["flow_src"].to(device)
            ### To make sure the input size is 256x256
            source = F.interpolate(mini_batch["src_img"].to(device), 256, None, 'bilinear', False)
            target = F.interpolate(mini_batch["trg_img"].to(device), 256, None, 'bilinear', False)
            pred_flow = net(source, target)
            estimated_kps = flow2kps(
                mini_batch["src_kps"].to(device),
                pred_flow,
                mini_batch["n_pts"].to(device),
            )
            eval_result = Evaluator.eval_kps_transfer(estimated_kps.cpu(), mini_batch)
            Loss = EPE(pred_flow, flow_gt)
            pck_array += eval_result["pck"]

            running_total_loss += Loss.item()
            pbar.set_description(
                " Test R_total_loss: %.3f/%.3f"
                % (running_total_loss / (i + 1), Loss.item())
            )
        mean_pck = sum(pck_array) / len(pck_array)
        print("####### Test mean pck: %.3f #######" % mean_pck)

    return running_total_loss / len(val_loader), mean_pck
