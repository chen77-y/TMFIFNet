import torch
import torch.nn as nn

class SqueezeExcite(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        hidden = max(dim // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w


class LargeKernelAttention(nn.Module):
    def __init__(self, dim, k1=5, k2=7, d2=3):
        super().__init__()
        pad2 = (k2 // 2) * d2
        self.dw1 = nn.Conv2d(dim, dim, k1, padding=k1//2, groups=dim)
        self.dw2 = nn.Conv2d(dim, dim, k2, padding=pad2, dilation=d2, groups=dim)
        self.pw  = nn.Conv2d(dim, dim, 1)
        self.act = nn.Sigmoid()
    def forward(self, x):
        attn = self.pw(self.dw2(self.dw1(x)))
        return x * self.act(attn)