import torch
import torch.nn as nn
from typing import Optional, Tuple

class ADWF(nn.Module):
    def __init__(self, dim: int, hidden_ratio: float = 0.5, temperature: float = 1.0, min_clamp: float = 0.0):
        super().__init__()
        h = max(8, int(dim * hidden_ratio))
        self.proj = nn.Sequential(
            nn.LayerNorm(3*dim),
            nn.Linear(3*dim, h),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(h, 3)
        )
        self.temperature = float(temperature)
        self.min_clamp  = float(min_clamp)

    def forward(self, e_rgb: torch.Tensor, e_dep: torch.Tensor, e_ir: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        z = torch.cat([e_rgb, e_dep, e_ir], dim=-1)
        logits = self.proj(z) / max(1e-6, self.temperature)
        if mask is not None:
            logits = logits + torch.log(mask + 1e-6)
        w = torch.softmax(logits, dim=-1)
        if self.min_clamp > 0.0:
            w = torch.clamp(w, min=self.min_clamp)
            w = w / w.sum(dim=-1, keepdim=True)

        e_fuse = w[:,0:1]*e_rgb + w[:,1:2]*e_dep + w[:,2:3]*e_ir
        return e_fuse, w