"""Normalizasyon: RMSNorm (ön-norm, final norm, softmax tabanının QK-norm'u) ve birim küre (RGN'nin q̂, k̂'sı)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """x / sqrt(ort(x²) + eps) * g — fp32'de hesaplanır, girdinin dtype'ında döner."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight.to(x.dtype)


def unit(x):
    """Birim küre (kazançsız L2): ||x|| = 1.  Uzaklığın matris çarpımına inmesi için şart (doküman 3.1)."""
    return F.normalize(x.float(), dim=-1).to(x.dtype)
