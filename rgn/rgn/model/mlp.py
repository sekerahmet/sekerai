"""MLP: rasyonel MLP, rasyonel kapı (rglu), cebirsel GLU, SwiGLU (karşılaştırma tabanı).

Hepsi ~8D² parametre: rasyonel D×4D + 4D×D; kapılılar 3 × D × 8D/3.
Ölçüldü (SS, 1000. adım, softmax dikkat): rasyonel MLP SwiGLU'dan +0,10 kayıp — kapısız ve doyan (|u| > 3 %10–18).
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

    def forward(self, x, extra=None):
        u = self.a(x)
        if extra is not None:
            u = u + extra
        if self.track:
            ud = u.detach().float()
            self.stats = {"u_std": ud.std().item(), "sat_frac": (ud.abs() > 3).float().mean().item()}
        return self.c(u * torch.rsqrt(1.0 + u * u))


def algebraic_sigmoid(u):
    """σ_a(u) = ½ (1 + u / sqrt(1 + u²)) — sigmoid'in üstel olmayan karşılığı."""
    return 0.5 * (1.0 + u * torch.rsqrt(1.0 + u * u))


class GatedMLP(nn.Module):
    """Kapılı MLP: W_2 [f(G) ⊙ U].
        kind="rglu"   f(G) = G / sqrt(1 + G²)   rasyonel kapı (rasyonel MLP'nin fonksiyonu kapı olarak; çıktı U ile sınırsız)
        kind="glu"    f(G) = G σ_a(G)           cebirsel GLU (SwiGLU'nun üstel olmayan karşılığı)
        kind="swiglu" f(G) = SiLU(G)            karşılaştırma tabanı"""

    def __init__(self, d, ff, kind, bias=True):
        super().__init__()
        assert kind in ("rglu", "glu", "swiglu")
        self.kind = kind
        self.a = nn.Linear(d, 2 * ff, bias=bias)
        self.c = nn.Linear(ff, d, bias=bias)
        self.track, self.stats = False, {}

    def forward(self, x, extra=None):
        """extra: kapı girdisine eklenen terim (örn. dikkatin kanıt miktarı)."""
        g, up = self.a(x).chunk(2, dim=-1)
        if extra is not None:
            g = g + extra
        if self.track:
            gd = g.detach().float()
            self.stats = {"u_std": gd.std().item(), "sat_frac": (gd.abs() > 3).float().mean().item()}
        if self.kind == "rglu":
            act = g * torch.rsqrt(1.0 + g * g)
        elif self.kind == "glu":
            act = g * algebraic_sigmoid(g)
        else:
            act = F.silu(g)
        return self.c(act * up)


def build_mlp(cfg):
    ff = cfg.ff_dim()
    if cfg.mlp == "rational":
        return RationalMLP(cfg.d_model, ff, cfg.mlp_bias)
    return GatedMLP(cfg.d_model, ff, cfg.mlp, cfg.mlp_bias)
