"""MLP: rasyonel MLP (ana), cebirsel GLU (alternatif), SwiGLU (karşılaştırma tabanı).

Üçü de ~8D² parametre: rasyonel D×4D + 4D×D; GLU / SwiGLU 3 × D × 8D/3.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RationalMLP(nn.Module):
    """Çıktı = W_2 [u / sqrt(1 + u²)] + b_2,  u = W_1 x + b_1  (doküman 2.2).  Nöron çıktısı (-1, 1) aralığında."""

    def __init__(self, d, ff, bias=True):
        super().__init__()
        self.a = nn.Linear(d, ff, bias=bias)                           # a·S + b
        self.c = nn.Linear(ff, d, bias=bias)                           # c·[...] + b_2
        self.track = False                                             # teşhis: doyma istatistiği
        self.stats = {}

    def forward(self, x):
        u = self.a(x)
        if self.track:
            ud = u.detach().float()
            self.stats = {"u_std": ud.std().item(), "sat_frac": (ud.abs() > 3).float().mean().item()}
        return self.c(u * torch.rsqrt(1.0 + u * u))


def algebraic_sigmoid(u):
    """σ_a(u) = ½ (1 + u / sqrt(1 + u²)) — sigmoid'in üstel olmayan karşılığı."""
    return 0.5 * (1.0 + u * torch.rsqrt(1.0 + u * u))


class GatedMLP(nn.Module):
    """kind="glu": W_2 [(G σ_a(G)) ⊙ U] (cebirsel GLU) | kind="swiglu": W_2 [SiLU(G) ⊙ U] (karşılaştırma tabanı)."""

    def __init__(self, d, ff, kind, bias=True):
        super().__init__()
        assert kind in ("glu", "swiglu")
        self.kind = kind
        self.a = nn.Linear(d, 2 * ff, bias=bias)
        self.c = nn.Linear(ff, d, bias=bias)
        self.track, self.stats = False, {}

    def forward(self, x):
        g, up = self.a(x).chunk(2, dim=-1)
        act = g * algebraic_sigmoid(g) if self.kind == "glu" else F.silu(g)
        return self.c(act * up)


def build_mlp(cfg):
    ff = cfg.ff_dim()
    if cfg.mlp == "rational":
        return RationalMLP(cfg.d_model, ff, cfg.mlp_bias)
    return GatedMLP(cfg.d_model, ff, cfg.mlp, cfg.mlp_bias)
