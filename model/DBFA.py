import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional, Tuple
from model.HSFE import HSFE
from model.DMP import DirectionalMaxPool2d
from model.DCAB import DCABlock

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

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size[0]-1)*(2*window_size[1]-1), num_heads)
        )
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C//self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size[0]*self.window_size[1], self.window_size[0]*self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_//nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def window_partition(x, window_size: int):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0,1,3,2,4,5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0,1,3,2,4,5).contiguous().view(B, H, W, -1)
    return x

class Global_block(nn.Module):

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size),
            num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.fc1   = nn.Linear(dim, dim)
        self.act   = act_layer()

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1,2))
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size).view(-1, self.window_size*self.window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1,2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H*W, C)
        x = self.fc1(x)
        x = self.act(x)
        x = shortcut + self.drop_path(x)
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2

        self.blocks = nn.ModuleList([
            Global_block(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            ) for i in range(depth)
        ])
        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample is not None else None

    def create_mask(self, x, H, W):
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt; cnt += 1
        mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2

        attn_mask = self.create_mask(x, H, W)
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        return x, H, W

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0], 0, 0))
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class PatchMerging(nn.Module):

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        dim = dim // 2
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W
        x = x.view(B, H, W, C)
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class PatchEmbedFromFeat(nn.Module):

    def __init__(self, in_chans, embed_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=1, bias=False)
        self.norm = norm_layer(embed_dim)
    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)     # [B, HW, C]
        x = self.norm(x)
        return x, H, W

class DBFA(nn.Module):

    def __init__(self, out_dim=256, in_chans1=3,in_chans2=1,
                 patch_size=4, embed_dim=64, depths=(2,2,2,2), num_heads=(2,4,4,4),
                 window_size=7, qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, conv_depths=(1,1,2,1), conv_dims=(48,96,192,384),
                 conv_drop_path_rate=0.1,
                 local_attn: Tuple[str, str, str, str] = ("none", "se", "lka", "lka"),
                 se_reduction: int = 16,
                 layer_scale_init: float = 1e-6,
                 dmp_mode: str = "post_stage",
                 dmp_kernel: int = 3, dmp_stride: int = 1, dmp_padding: int = 1, dmp_eps: float = 0.0):
        super().__init__()
        assert dmp_mode in ("none", "post_stage", "downsample")
        assert len(local_attn) == 4
        self.dmp_mode = dmp_mode
        self.local_attn = local_attn
        self.se_reduction = se_reduction
        self.layer_scale_init = layer_scale_init

        C_sh = conv_dims[0]
        mid_ch = max(C_sh // 2, 16)
        self.shared_stem = HSFE(in_chans1,conv_dims=(48, 96, 192, 384))

        self.downsample_layers = nn.ModuleList()
        for i in range(3):
            if self.dmp_mode == "downsample":
                self.downsample_layers.append(nn.Sequential(
                    LayerNorm(conv_dims[i], eps=1e-6, data_format="channels_first"),
                    DirectionalMaxPool2d(kernel_size=2, stride=2, padding=0, eps=dmp_eps),  # /2
                    nn.Conv2d(conv_dims[i], conv_dims[i+1], kernel_size=1, stride=1, bias=True),
                ))
            else:
                self.downsample_layers.append(nn.Sequential(
                    LayerNorm(conv_dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(conv_dims[i], conv_dims[i+1], kernel_size=2, stride=2),
                ))

        dp_rates = [x.item() for x in torch.linspace(0, conv_drop_path_rate, sum(conv_depths))]
        cur = 0
        self.stages = nn.ModuleList()
        for i in range(0, 4):
            stage_blocks = []
            for j in range(conv_depths[i]):
                stage_blocks.append(
                    DCABlock(
                        dim=conv_dims[i],
                        drop_rate=dp_rates[cur + j],
                        attn_type=local_attn[i],
                        se_reduction=se_reduction,
                        layer_scale_init=layer_scale_init
                    )
                )
            self.stages.append(nn.Sequential(*stage_blocks))
            cur += conv_depths[i]

        if self.dmp_mode == "post_stage":
            self.post_stage_dmp = nn.ModuleList([
                DirectionalMaxPool2d(kernel_size=dmp_kernel, stride=dmp_stride, padding=dmp_padding, eps=dmp_eps)
                for _ in range(4)
            ])
        else:
            self.post_stage_dmp = None

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        self.patch_from_shared = PatchEmbedFromFeat(
            in_chans=C_sh,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else nn.Identity
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers1 = BasicLayer(int(embed_dim*1), depths[0], num_heads[0], window_size,
                                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                                  drop_path=dpr[:depths[0]], norm_layer=norm_layer,
                                  downsample=None, use_checkpoint=use_checkpoint)
        self.layers2 = BasicLayer(int(embed_dim*2), depths[1], num_heads[1], window_size,
                                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                                  drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                                  norm_layer=norm_layer, downsample=PatchMerging, use_checkpoint=use_checkpoint)
        self.layers3 = BasicLayer(int(embed_dim*4), depths[2], num_heads[2], window_size,
                                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                                  drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                                  norm_layer=norm_layer, downsample=PatchMerging, use_checkpoint=use_checkpoint)
        self.layers4 = BasicLayer(int(embed_dim*8), depths[3], num_heads[3], window_size,
                                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                                  drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                                  norm_layer=norm_layer, downsample=PatchMerging, use_checkpoint=use_checkpoint)

        concat_dim = conv_dims[-1] + self.num_features
        self.head_norm = nn.LayerNorm(concat_dim, eps=1e-6)
        self.proj = nn.Linear(concat_dim, out_dim, bias=False)
        self.proj_act = nn.GELU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.2)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, imgs):
        x_shared = self.shared_stem(imgs)

        x_s, Hs, Ws = self.patch_from_shared(x_shared)
        x_s, Hs, Ws = self.layers1(x_s, Hs, Ws)
        x_s, Hs, Ws = self.layers2(x_s, Hs, Ws)
        x_s, Hs, Ws = self.layers3(x_s, Hs, Ws)
        x_s, Hs, Ws = self.layers4(x_s, Hs, Ws)
        x_s = torch.transpose(x_s, 1, 2).view(imgs.size(0), -1, Hs, Ws)

        x_c = x_shared
        x_c = self.stages[0](x_c)
        if self.post_stage_dmp is not None: x_c = self.post_stage_dmp[0](x_c)

        x_c = self.downsample_layers[0](x_c)
        x_c = self.stages[1](x_c)
        if self.post_stage_dmp is not None: x_c = self.post_stage_dmp[1](x_c)

        x_c = self.downsample_layers[1](x_c)
        x_c = self.stages[2](x_c)
        if self.post_stage_dmp is not None: x_c = self.post_stage_dmp[2](x_c)

        x_c = self.downsample_layers[2](x_c)
        x_c = self.stages[3](x_c)
        if self.post_stage_dmp is not None: x_c = self.post_stage_dmp[3](x_c)

        gap_s = x_s.mean(dim=(-2, -1))
        gap_c = x_c.mean(dim=(-2, -1))
        feat  = torch.cat([gap_c, gap_s], dim=1)
        feat  = self.head_norm(feat)
        emb   = self.proj_act(self.proj(feat))
        return emb