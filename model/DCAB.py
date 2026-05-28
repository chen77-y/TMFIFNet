import torch
import torch.nn as nn
import torch.nn.functional as F
from model.SE import SqueezeExcite,LargeKernelAttention

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

class DCABlock(nn.Module):

    def __init__(
        self,
        dim: int,
        drop_rate: float = 0.,
        attn_type: str = 'none',
        se_reduction: int = 16,
        layer_scale_init: float = 1e-6
    ):
        super().__init__()
        assert attn_type in {'none', 'se', 'lka'}
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm   = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv = nn.Linear(dim, dim)
        self.act    = nn.GELU()
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None

        if attn_type == 'se':
            self.attn = SqueezeExcite(dim, reduction=se_reduction)
        elif attn_type == 'lka':
            self.attn = LargeKernelAttention(dim)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = self.pwconv(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.attn(x)
        return shortcut + self.drop_path(x)

