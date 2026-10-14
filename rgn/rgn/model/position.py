"""Konum: RoPE (q̂, k̂ döndürmesi; boy korunur) ve head başına başlangıç dizileri.

Konum çarpanları (alibi / additive) çekim paydasının parçası olduğu için rgn/model/attention.py:denominator'dadır.
"""
import torch


def rope_tables(n, d_head, base=10000.0, device=None):
    """-> cos, sin (n, d_head) fp32.  Bir kez hesaplanır; ileri geçişte trigonometrik işlem yapılmaz."""
    inv = 1.0 / (base ** (torch.arange(0, d_head, 2, device=device, dtype=torch.float32) / d_head))
    ang = torch.outer(torch.arange(n, device=device, dtype=torch.float32), inv)
    ang = torch.cat([ang, ang], dim=-1)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """x (B, H, N, d_head), cos / sin (N, d_head).  ||x|| = 1 ise sonuç da birim vektördür."""
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos.to(x.dtype) + torch.cat([-x2, x1], dim=-1) * sin.to(x.dtype)


def geometric(lo, hi, n):
    """lo'dan hi'ye n elemanlı geometrik dizi (head başına γ / λ başlangıcı)."""
    return torch.tensor([lo * (hi / lo) ** (i / max(n - 1, 1)) for i in range(n)], dtype=torch.float32)
