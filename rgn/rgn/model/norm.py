"""RMSNorm (ön-norm, final norm, QK-norm)."""
import torch
import torch.nn as nn


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
