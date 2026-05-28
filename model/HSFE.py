import torch
import torch.nn as nn
import torch.nn.functional as F
from model.DCRB import DCRB

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

class stem(nn.Module):

    def __init__(self, in_chans, mid_chans, out_chans):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_chans, mid_chans, kernel_size=3, stride=2, padding=1, bias=False),
            LayerNorm(mid_chans, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(mid_chans, out_chans, kernel_size=3, stride=2, padding=1, bias=False),
            LayerNorm(out_chans, eps=1e-6, data_format="channels_first"),
        )

    def forward(self, x):
        return self.block(x)


class HSFE(nn.Module):

    def __init__(self, in_chans, conv_dims=(48,96,192,384),

                 dmp_kernel: int = 3, dmp_stride: int = 1, dmp_padding: int = 1, dmp_eps: float = 0.0):
        super().__init__()
        C_sh = conv_dims[0]
        mid_ch = max(C_sh // 2, 16)

        self.stem_1ch = stem(in_chans=1, mid_chans=mid_ch, out_chans=C_sh)
        self.stem_3ch = stem(in_chans=3, mid_chans=mid_ch, out_chans=C_sh)


        self.blocks = nn.Sequential(
            DCRB(dim=C_sh, drop_rate=0.0),
            DCRB(dim=C_sh, drop_rate=0.0),
        )

    def forward(self, imgs):

        if imgs.shape[1] == 1:
            x_shared = self.stem_1ch(imgs)
        elif imgs.shape[1] == 3:
            x_shared = self.stem_3ch(imgs)
        else:
            raise ValueError(f"不支持的输入通道数: {imgs.shape[1]}，只支持1或3通道")

        x_shared = self.blocks(x_shared)

        return x_shared