import torch
import torch.nn as nn

class CrossAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, local_feat, global_feat):
        attn_output, _ = self.attn(local_feat, global_feat, global_feat)
        attn_output = attn_output + local_feat
        attn_output = self.norm(attn_output)
        output = self.fc(attn_output)
        output = self.act(output)
        return output

class PairwiseCA(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                         dropout=dropout, batch_first=True)
        self.ln  = nn.LayerNorm(dim)

    def forward(self, ea: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
        q = ea.unsqueeze(1); k = eb.unsqueeze(1); v = eb.unsqueeze(1)
        y, _ = self.mha(q, k, v, need_weights=False)
        return self.ln(y.squeeze(1))

class PairSymmetricFuse(nn.Module):
    def __init__(self, dim, mode="gate"):
        super().__init__()
        assert mode in {"avg","concat","gate"}
        self.mode = mode
        if mode == "concat":
            self.proj = nn.Linear(2*dim, dim)
        if mode == "gate":
            self.gate = nn.Linear(2*dim, dim)
            self.proj = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, y_ab, y_ba):
        if self.mode == "avg":
            y = 0.5*(y_ab + y_ba)
        elif self.mode == "concat":
            y = self.proj(torch.cat([y_ab, y_ba], dim=-1))
        else:
            g = torch.sigmoid(self.gate(torch.cat([y_ab, y_ba], dim=-1)))
            y = self.proj(g*y_ab + (1-g)*y_ba)
        return self.ln(y)

class BGIF(nn.Module):

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0, fuse_mode: str = "gate"):
        super().__init__()
        self.ca_ab = PairwiseCA(dim=dim, num_heads=num_heads, dropout=dropout)
        self.ca_ba = PairwiseCA(dim=dim, num_heads=num_heads, dropout=dropout)
        self.sym   = PairSymmetricFuse(dim=dim, mode=fuse_mode)

    def forward(self, e_i: torch.Tensor, e_fuse: torch.Tensor) -> torch.Tensor:
        y_ab = self.ca_ab(e_i, e_fuse)
        y_ba = self.ca_ba(e_fuse, e_i)
        t_i  = self.sym(y_ab, y_ba)
        return t_i

