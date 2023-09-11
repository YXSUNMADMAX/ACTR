import os
from functools import partial
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, trunc_normal_
from models.feature_backbones import vision_transformer
from models.mod import unnormalise_and_convert_mapping_to_flow
from models.base.swin import SwinTransformer2d, TransformerWarper2d
import torch
from torch import nn

class SpatialAttention_heat(nn.Module):
    def __init__(self):
        super(SpatialAttention_heat, self).__init__()
        self.sigmoid = nn.Sigmoid()
        # self.weight = nn.Parameter(torch.ones(1, 1, 384), requires_grad=False)
        self.weight = nn.Parameter(torch.ones(1, 1, 768), requires_grad=True)

    def forward(self, attm, x):
        attm = self.sigmoid(attm)
        x1 = x * self.weight * attm
        x = x + x1
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        x = x + x * out
        return x

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y, mode):

        if mode == "self":
            B, N, C = x.shape
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = (
                qkv[0],
                qkv[1],
                qkv[2],
            )  # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x, attn
        else:
            B, N, C = x.shape
            qkvx = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            qx, kx, vx = qkvx[0], qkvx[1], qkvx[2]

            qkvy = (
                self.qkv(y)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            qy, ky, vy = qkvy[0], qkvy[1], qkvy[2]

            attn = (qx @ ky.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ vy).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x, attn.sum(1)

class MultiscaleBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        last=False,
    ):
        super().__init__()
        self.sattn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.cattn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.mlp2 = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.last = last
        self.gelu = nn.GELU()
        self.spatial_attention = SpatialAttention_heat()
        self.pos_embed = nn.Parameter(torch.zeros(1, 256 + 1, dim))

    def arf(self, x):
        exp = torch.exp
        zero = torch.zeros_like(x)
        tmp = (exp(x) - exp(-1 * x)) / (exp(x) + exp(-1 * x - 4))
        return torch.where(tmp < 0, zero, tmp)

    def forward(self, ins):
        """
        Multi-level aggregation
        """
        src, tgt, attm = ins
        B, N, C = src.shape
        src = src + self.pos_embed

        srct, _ = self.sattn(self.norm1(src), None, "self")
        tgtt, _ = self.sattn(self.norm1(tgt), None, "self")

        src = src + self.drop_path(srct)
        tgt = tgt + self.drop_path(tgtt)

        src = src + self.drop_path(self.mlp(self.norm2(srct)))
        tgt = tgt + self.drop_path(self.mlp(self.norm2(tgtt)))

        srct, attn_src = self.cattn(self.norm1(src), self.norm1(tgt), "cross")

        srct = self.arf(srct)

        src = src + self.drop_path(srct)
        src = src + self.drop_path(self.mlp(self.norm2(srct)))

        attm.append(attn_src[:, 1:, 1:])

        if self.last:
            return src.contiguous().view(B, N, C), tgt.contiguous().view(B, N, C), attm

        with torch.no_grad():
            tgt_heat = attn_src.clone()
        tgt_heat = tgt_heat.view(B, N, N).mean(1, keepdim=True)
        tgt_heat = tgt_heat.view(B, N, 1)
        tgt = self.spatial_attention(tgt_heat, tgt)

        return src.contiguous().view(B, N, C), tgt.contiguous().view(B, N, C), attm


class TransformerAggregator(nn.Module):
    def __init__(
        self,
        img_size=224,
        embed_dim=2048,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.pos_embed_x = nn.Parameter(torch.zeros(1, embed_dim // 2, 1, img_size))
        self.pos_embed_y = nn.Parameter(torch.zeros(1, embed_dim // 2, img_size, 1))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        self.blocks = nn.Sequential(
            *[
                MultiscaleBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    last=False,
                )
                for i in range(depth - 1)
            ]
        )

        self.last_block = nn.Sequential(
            *[
                MultiscaleBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[depth - 1],
                    norm_layer=norm_layer,
                    last=True,
                )
                for i in range(1)
            ]
        )

        self.proj = nn.Linear(embed_dim, img_size**2)
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed_x, std=0.02)
        trunc_normal_(self.pos_embed_y, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, source, target):
        src = source
        tgt = target
        src, tgt, attm = self.blocks((src, tgt, []))
        feat_src, feat_trg, corr_src = self.last_block((src, tgt, attm))
        return corr_src, feat_src

class FeatureExtractionHyperPixel_VIT(nn.Module):
    def __init__(self, feature_size, ibot_ckp_file, freeze=True):
        super().__init__()
        self.backbone = vision_transformer.__dict__['vit_base'](patch_size=16, num_classes=0)
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if os.path.isfile(ibot_ckp_file):
            state_dict = torch.load(ibot_ckp_file, map_location="cpu")
            dino_ckp_key = 'teacher'
            # dino_ckp_key = 'model'
            if dino_ckp_key is not None and dino_ckp_key in state_dict:
                print(f"Take key {dino_ckp_key} in provided checkpoint dict")
                state_dict = state_dict[dino_ckp_key]
            for k, v in state_dict.items():
                print(k)
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = self.backbone.load_state_dict(state_dict, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(ibot_ckp_file, msg))
        else:
            print('No iBot ckp loaded!!!')
            raise RuntimeError

        self.feature_size = feature_size

    def forward(self, img):
        r"""Extract desired a list of intermediate features"""
        # b, c, h, w = img.shape
        feat = self.backbone(img)
        return feat


class ACTR(nn.Module):
    def __init__(self, ibot_ckp_file='./', feature_size=128, depth=6, num_heads=8, mlp_ratio=4, freeze=False):
        super().__init__()
        self.feature_size = feature_size
        self.feature_size_model = 16
        self.decoder_embed_dim = 768
        self.feature_extraction = FeatureExtractionHyperPixel_VIT(feature_size, ibot_ckp_file, freeze)
        self.learn_parm = nn.Parameter(torch.tensor((1), dtype=torch.float, requires_grad=False))
        self.x_normal = np.linspace(-1, 1, self.feature_size_model)
        self.x_normal = nn.Parameter(
            torch.tensor(self.x_normal, dtype=torch.float, requires_grad=False)
        )
        self.y_normal = np.linspace(-1, 1, self.feature_size_model)
        self.y_normal = nn.Parameter(
            torch.tensor(self.y_normal, dtype=torch.float, requires_grad=False)
        )
        self.decoder = TransformerAggregator(
            img_size=self.feature_size_model,
            embed_dim=self.decoder_embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        self.refine_swin_decoder_1_1 = nn.Sequential(
            TransformerWarper2d(
                SwinTransformer2d(
                    img_size=(64, 64), embed_dim=96, window_size=4, num_heads=[8], depths=[1]
                )
            ),
            ChannelAttention(96),
        )

        self.refine_swin_decoder_1_2 = nn.Sequential(
            TransformerWarper2d(
                SwinTransformer2d(
                    img_size=(64, 64), embed_dim=96, window_size=4, num_heads=[8], depths=[1]
                )
            ),
            ChannelAttention(96),
        )

        self.refine_swin_decoder_2_1 = nn.Sequential(
            TransformerWarper2d(
                SwinTransformer2d(
                    img_size=(64, 64), embed_dim=96, window_size=8, num_heads=[8], depths=[1]
                )
            ),
            ChannelAttention(96),
        )

        self.refine_swin_decoder_2_2 = nn.Sequential(
            TransformerWarper2d(
                SwinTransformer2d(
                    img_size=(64, 64), embed_dim=96, window_size=8, num_heads=[8], depths=[1]
                )
            ),
            ChannelAttention(96),
        )

        self.refine2 = nn.Sequential(
            nn.Conv2d(96, 32, (3, 3), padding=(1, 1), bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 2, (3, 3), padding=(1, 1), bias=True),
        )

        self.dropout2d = nn.Dropout2d(p=0.5)

        self.refine_proj_query_feat = nn.Sequential(
            nn.Conv2d(self.decoder_embed_dim, 94, 1), nn.ReLU()
        )

    def softmax_with_temperature(self, x, beta, d=1):
        r"""SFNet: Learning Object-aware Semantic Flow (Lee et al.)"""
        M, _ = x.max(dim=d, keepdim=True)
        x = x - M  # subtract maximum value for stability
        exp_x = torch.exp(x / beta)
        exp_x_sum = exp_x.sum(dim=d, keepdim=True)
        return exp_x / exp_x_sum

    def soft_argmax(self, corr, beta=0.02):
        r"""SFNet: Learning Object-aware Semantic Flow (Lee et al.)"""
        b, _, h, w = corr.size()

        corr = self.softmax_with_temperature(corr, beta=beta, d=1)
        corr = corr.view(-1, h, w, h, w)  # (target hxw) x (source hxw)

        grid_x = corr.sum(dim=1, keepdim=False)  # marginalize to x-coord.
        x_normal = self.x_normal.expand(b, w)
        x_normal = x_normal.view(b, w, 1, 1)
        grid_x = (grid_x * x_normal).sum(dim=1, keepdim=True)  # b x 1 x h x w

        grid_y = corr.sum(dim=2, keepdim=False)  # marginalize to y-coord.
        y_normal = self.y_normal.expand(b, h)
        y_normal = y_normal.view(b, h, 1, 1)
        grid_y = (grid_y * y_normal).sum(dim=1, keepdim=True)  # b x 1 x h x w
        return grid_x, grid_y

    def apply_dropout(self, dropout, *feats):
        sizes = [x.shape[-2:] for x in feats]
        max_size = max(sizes)
        resized_feats = [F.interpolate(x, size=max_size, mode="nearest") for x in feats]

        channel_list = [x.size(1) for x in feats]
        feats = dropout(torch.cat(resized_feats, dim=1))
        feats = torch.split(feats, channel_list, dim=1)
        recoverd_feats = [
            F.interpolate(x, size=size, mode="nearest") for x, size in zip(feats, sizes)
        ]
        return recoverd_feats

    def forward(self, source, target):
        Size = 64
        B, _, _, _ = source.shape
        src_feat =self.feature_extraction(source)
        tgt_feat = self.feature_extraction(target)

        corrs, en_src_feat = self.decoder(src_feat, tgt_feat)

        flow_all = torch.zeros([B, 2, 64, 64]).to(en_src_feat.device)

        refine_feat = self.refine_proj_query_feat(
            en_src_feat[:, 1:, :].transpose(-1, -2).view(B, -1, 16, 16)
        )

        refine_feat = F.interpolate(refine_feat, Size, None, "bilinear", True)
        refine_feat = self.apply_dropout(self.dropout2d, refine_feat)[0]

        for it in corrs:
            curr_flow = it
            grid_x, grid_y = self.soft_argmax(curr_flow.view(B, 16, 16, -1).permute(0, 3, 1, 2))
            flow = torch.cat((grid_x, grid_y), dim=1)
            flow = unnormalise_and_convert_mapping_to_flow(flow)
            flow_up = F.interpolate(flow, Size, None, "nearest") * Size / 16
            x_t = torch.cat([flow_up, refine_feat], dim=1)
            x_8 = self.refine_swin_decoder_2_1(x_t)
            x_4 = self.refine_swin_decoder_1_1(x_t)
            x_t = x_8 + x_4
            x_8 = self.refine_swin_decoder_2_2(x_t)
            x_4 = self.refine_swin_decoder_1_2(x_t)
            x = x_8 + x_4
            flow_refine = self.refine2(x)
            curr_flow = flow_up + flow_refine
            flow_all = flow_all + curr_flow
        flow_all = flow_all / len(corrs)
        return flow_all


if __name__ == "__main__":
    trg = torch.rand((3, 3, 256, 256))
    src = torch.rand((3, 3, 256, 256))
    b, c, h, w = trg.shape
    print(b, c, h, w)
    model = ACTR(num_heads=8)
    total = sum([param.nelement() for param in model.parameters()])
    print("######## Number of parameter: %.2fM ########" % (total / 1e6))
    print(model(trg, src).shape)
