"""SwiGLU MLP: W_2 [SiLU(G) ⊙ U], [G | U] = W_1 x."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, d, ff, bias=True):
        super().__init__()
        self.a = nn.Linear(d, 2 * ff, bias=bias)                       # [G | U]
        self.c = nn.Linear(ff, d, bias=bias)
        self.track, self.stats = False, {}                             # teşhis: kapı girdisinin dağılımı

    def forward(self, x):
        g, up = self.a(x).chunk(2, dim=-1)
        if self.track:
            gd = g.detach().float()
            self.stats = {"u_std": gd.std().item(), "sat_frac": (gd.abs() > 3).float().mean().item()}
        return self.c(F.silu(g) * up)
