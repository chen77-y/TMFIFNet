import torch
import torch.nn as nn
import torch.nn.functional as F

class DirectionalMaxPool2d1(nn.Module):

    def __init__(self, kernel_size=2, stride=None, padding=0, dilation=1, ceil_mode=False, eps=0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size,
                                 stride=stride if stride is not None else kernel_size,
                                 padding=padding, dilation=dilation, ceil_mode=ceil_mode)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.eps > 0:
            s = torch.where(torch.abs(x) < self.eps, torch.zeros_like(x), torch.sign(x))
        else:
            s = torch.sign(x)
        s = s.detach()
        Dp = F.relu(s)
        Dm = F.relu(-s)

        Vp = x * Dp
        Vm = x * Dm

        Vp = self.pool(Vp)
        Vm = self.pool(Vm)
        Dp_p = self.pool(Dp)
        Dm_p = self.pool(Dm)

        out = Vp * Dp_p + Vm * Dm_p
        return out



class DirectionalMaxPool2d(nn.Module):

    def __init__(self, kernel_size=2, stride=None, padding=0, dilation=1, ceil_mode=False, eps=0.0):
        super().__init__()
        self.max_pool = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride if stride is not None else kernel_size,
            padding=padding, dilation=dilation, ceil_mode=ceil_mode
        )
        self.min_pool = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride if stride is not None else kernel_size,
            padding=padding, dilation=dilation, ceil_mode=ceil_mode
        )
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.eps > 0:
            s = torch.where(torch.abs(x) < self.eps, torch.zeros_like(x), torch.sign(x))
        else:
            s = torch.sign(x)
        s = s.detach()

        Dp = F.relu(s)
        Dm = F.relu(-s)

        Vp = x * Dp
        Vm = x * Dm

        Vp_p = self.max_pool(Vp)

        Vm_neg = -Vm
        Vm_neg_p = self.max_pool(Vm_neg)
        Vm_p = -Vm_neg_p

        Dp_p = self.max_pool(Dp)
        Dm_p = self.max_pool(Dm)

        out = Vp_p * Dp_p + Vm_p * Dm_p
        return out