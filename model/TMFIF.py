import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional, Tuple
from model.DBFA import DBFA
from model.ADWF import ADWF
from model.BGIF import BGIF


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias   = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)
        assert self.data_format in ["channels_last", "channels_first"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            mean = x.mean(1, keepdim=True)
            var  = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            return self.weight[:, None, None] * x + self.bias[:, None, None]

class LateFusionHead(nn.Module):
    def __init__(self, in_dim_per_modal=256, num_modal=3, num_classes=4):
        super().__init__()
        fused_dim = in_dim_per_modal * num_modal
        self.norm = nn.LayerNorm(fused_dim)
        self.fc   = nn.Linear(fused_dim, num_classes)

    def forward(self, embs: Tuple[torch.Tensor, ...]):
        z = torch.cat(embs, dim=1)
        z = self.norm(z)
        logits = self.fc(z)
        return logits

class DWAHead(nn.Module):

    def __init__(self, dim: int = 256, num_classes: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(4*dim)
        self.fc1  = nn.Linear(4*dim, dim)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2  = nn.Linear(dim, num_classes)

    def forward(self, e_fuse, t_rgb, t_dep, t_ir):
        z = torch.cat([e_fuse, t_rgb, t_dep, t_ir], dim=-1)  # [B,4C]
        z = self.norm(z)
        z = self.fc1(z)
        z = self.act(z)
        z = self.drop(z)
        logits = self.fc2(z)
        return logits

class TMFIF(nn.Module):

    def __init__(self,
                 num_classes=4,
                 num_heads=4,

                 rgb_in_chans=3,
                 rgb_cfg=dict(patch_size=4, embed_dim=64, depths=(2,2,2,2),
                              num_heads=(2,4,4,4), conv_depths=(1,1,2,1),
                              conv_dims=(48,96,192,384), drop_path_rate=0.1,
                              conv_drop_path_rate=0.1, use_checkpoint=False,
                              # local_attn=("none","se","lka","lka")),
                              local_attn=("se", "se", "se", "se")),
                 rgb_dmp_cfg=dict(dmp_mode="post_stage", dmp_kernel=3, dmp_stride=1, dmp_padding=1, dmp_eps=0.0),

                 ecg_in_chans=1,
                 ecg_cfg=dict(patch_size=4, embed_dim=64, depths=(2,2,2,2),
                              num_heads=(2,4,4,4), conv_depths=(1,1,2,1),
                              conv_dims=(48,96,192,384), drop_path_rate=0.1,
                              conv_drop_path_rate=0.1, use_checkpoint=False,
                              local_attn=("se", "se", "se", "se")),

                 depth_in_chans=1,
                 depth_cfg=dict(patch_size=4, embed_dim=64, depths=(2,2,2,2),
                             num_heads=(2,4,4,4), conv_depths=(1,1,2,1),
                             conv_dims=(48,96,192,384), drop_path_rate=0.1,
                             conv_drop_path_rate=0.1, use_checkpoint=False,
                             local_attn=("se", "se", "se", "se")),
                 out_dim=256,
                 gate_temperature=1.0,
                 gate_min_clamp=0.0,
                 dropout=0.1):
        super().__init__()


        rgb_all_cfg = dict(**rgb_cfg, **rgb_dmp_cfg)
        self.enc_rgb =DBFA(out_dim=out_dim, in_chans1=3, **rgb_all_cfg)
        ecg_all_cfg = dict(**ecg_cfg, **dict(dmp_mode="none"))
        self.enc_ecg =DBFA(out_dim=out_dim, in_chans2=1, **ecg_all_cfg)
        depth_all_cfg  = dict(**depth_cfg,  **dict(dmp_mode="none"))
        self.enc_dep  =DBFA(out_dim=out_dim, in_chans2=1,  **depth_all_cfg)


        self.tri_weighter = ADWF(dim=out_dim, temperature=gate_temperature, min_clamp=gate_min_clamp)
        self.ref_rgb = BGIF(dim=out_dim, num_heads=num_heads, dropout=dropout, fuse_mode="gate")
        self.ref_ecg = BGIF(dim=out_dim, num_heads=num_heads, dropout=dropout, fuse_mode="gate")
        self.ref_depth  = BGIF(dim=out_dim, num_heads=num_heads, dropout=dropout, fuse_mode="gate")

        self.head = DWAHead(dim=out_dim, num_classes=num_classes, dropout=dropout)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)

    @torch.no_grad()
    def extract_embeddings(self, x_rgb, x_ecg, x_dep):
        e_rgb = self.enc_rgb(x_rgb)
        e_ecg = self.enc_ecg(x_ecg)
        e_depth  = self.enc_depth(x_dep)
        return e_rgb, e_ecg, e_depth

    def forward(self, x_rgb, x_ecg, x_dep, mask: Optional[torch.Tensor] = None):

        e_rgb = self.enc_rgb(x_rgb)
        e_ecg = self.enc_ecg(x_ecg)
        e_dep = self.enc_dep(x_dep)

        e_fuse, w = self.tri_weighter(e_rgb, e_ecg, e_dep, mask=mask)

        t_rgb = self.ref_rgb(e_rgb, e_fuse)
        t_ecg = self.ref_ecg(e_ecg, e_fuse)
        t_dep  = self.ref_depth(e_dep,  e_fuse)

        logits = self.head(e_fuse, t_rgb, t_ecg, t_dep)
        return logits

