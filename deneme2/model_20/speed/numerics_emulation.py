# -*- coding: utf-8 -*-
"""numerics_emulation -- GPU'nun TF32 ve bf16 sayisalligini CPU'da taklit eder (yalniz test ve olcum icin; egitim kodu
bunu KULLANMAZ).

    tf32_emulation(mode)     her fp32 matmul'un (ileri VE geri) iki girdisi TF32'ye yuvarlanir, toplama fp32:
                             cuBLAS'in allow_tf32 davranisi.  mode "nearest" | "truncate" (donanim alt 13 biti yok sayarsa)
                             | None (ayni kod yolu, yuvarlama yok: taban)
    cuda_bf16_autocast()     CPU autocast(bf16) + F.normalize'in normu fp32: CUDA autocast politikasinin bizim
                             modellerdeki op'lar icin aynisi (CUDA'da linalg_vector_norm fp32 listesinde, CPU'da degil)
    bf16_emulation()         ayni sonuc, hizli: bf16'ya yuvarlanmis degerlerle fp32 GEMM (AVX2'de bf16 GEMM cok yavas)

Taklidin siniri: matmul'da fp32 toplama sirasi cuBLAS'inkiyle ayni degil (en son bit); SDPA taklidi math yolu.
"""
import contextlib
import math

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

_MATMULS = {torch.matmul, torch.Tensor.matmul, torch.Tensor.__matmul__, torch.mm, torch.bmm, torch.Tensor.mm,
            torch.Tensor.bmm}


def round_tf32(x, mode):
    """fp32 -> TF32 (10 bit mantissa), fp32 olarak.  nearest: en yakina, esitlikte cift; truncate: alt 13 bit silinir."""
    if mode is None:
        return x
    i = x.contiguous().view(torch.int32)
    if mode == "nearest":
        i = i + (0x0FFF + ((i >> 13) & 1))
    elif mode != "truncate":
        raise ValueError(mode)
    return (i & -8192).view(torch.float32)


class _RoundedMatmul(torch.autograd.Function):
    """a @ b; ileri ve geride her GEMM'in iki girdisi TF32'ye yuvarlanir (cuBLAS TF32 gibi)."""

    @staticmethod
    def forward(ctx, a, b, mode):
        ra, rb = round_tf32(a, mode), round_tf32(b, mode)
        ctx.save_for_backward(ra, rb)
        ctx.mode = mode
        return torch.matmul(ra, rb)

    @staticmethod
    def backward(ctx, g):
        ra, rb = ctx.saved_tensors
        rg = round_tf32(g, ctx.mode)
        if rb.dim() == 2 and ra.dim() > 2:              # (..., k) @ (k, n): agirlik gradyani tek GEMM (torch'un katlamasi)
            ga = rg @ rb.T
            gb = ra.reshape(-1, ra.shape[-1]).T @ rg.reshape(-1, rg.shape[-1])
        else:
            ga = (rg @ rb.transpose(-1, -2)).sum_to_size(ra.shape)
            gb = (ra.transpose(-1, -2) @ rg).sum_to_size(rb.shape)
        return ga, gb, None


def _sdpa_math(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, matmul=None):
    assert attn_mask is None and dropout_p == 0.0 and not enable_gqa
    scale = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
    s = matmul(q, k.transpose(-1, -2)) * scale
    if is_causal:
        L, S = q.shape[-2], k.shape[-2]
        s = s.masked_fill(torch.ones(L, S, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    return matmul(torch.softmax(s, -1), v)


class tf32_emulation(TorchFunctionMode):
    """with tf32_emulation("nearest"): ...  -- @, matmul, mm, bmm, F.linear ve SDPA (math) TF32 yuvarlamali."""

    def __init__(self, mode="nearest"):
        super().__init__()
        self.mode = mode

    def _mm(self, a, b):
        return _RoundedMatmul.apply(a, b, self.mode)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in _MATMULS:
            return self._mm(args[0], args[1])
        if func is F.linear:
            x, w = args[0], args[1]
            b = args[2] if len(args) > 2 else kwargs.get("bias")
            y = self._mm(x, w.T)
            return y if b is None else y + b
        if func is F.scaled_dot_product_attention:
            return _sdpa_math(*args, matmul=self._mm, **kwargs)
        return func(*args, **kwargs)


def _to_bf16_values(t):
    """bf16'ya yuvarlanmis degerler, fp32 tensor olarak (CPU'da bf16 GEMM yavas; fp32 GEMM ayni toplama semantigi)."""
    return t.to(torch.bfloat16).float()


def _bf16_mm(a, b):
    return (_to_bf16_values(a) @ _to_bf16_values(b)).to(torch.bfloat16)


class bf16_emulation(TorchFunctionMode):
    """CUDA autocast(bf16)'in bu projedeki op'lardaki sonucu, hizli: GEMM girdileri bf16'ya yuvarlanir, toplama fp32,
    cikti bf16 (ileri VE geri: cast'lerin geri yayilimi ayni yuvarlamayi yapar).  normalize ve cross_entropy fp32; SDPA
    CUDA math yolu gibi bf16 girdiden fp32 hesap, bf16 cikti.  cuda_bf16_autocast ile ayni sonucu verir (test)."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in _MATMULS:
            return _bf16_mm(args[0], args[1])
        if func is F.linear:
            x, w = args[0], args[1]
            b = args[2] if len(args) > 2 else kwargs.get("bias")
            y = _to_bf16_values(x) @ _to_bf16_values(w).T
            return (y if b is None else y + _to_bf16_values(b)).to(torch.bfloat16)
        if func is F.scaled_dot_product_attention:
            q, k, v = (_to_bf16_values(t) for t in args[:3])
            return func(q, k, v, *args[3:], **kwargs).to(torch.bfloat16)
        if func in (F.normalize, F.cross_entropy) and args[0].dtype == torch.bfloat16:
            return func(args[0].float(), *args[1:], **kwargs)
        return func(*args, **kwargs)


def _normalize_fp32_norm(input, p=2.0, dim=1, eps=1e-12, out=None):
    """CUDA autocast'ta F.normalize: linalg_vector_norm fp32 politikasinda -> norm fp32, bolum fp32'ye yukselir."""
    assert out is None
    denom = torch.linalg.vector_norm(input, p, dim, keepdim=True, dtype=torch.float32).clamp_min(eps).expand_as(input)
    return input / denom


@contextlib.contextmanager
def cuda_bf16_autocast():
    """CPU'da CUDA autocast(bf16) politikasi (bu projedeki op'lar icin): matmul/SDPA bf16, norm/log_softmax/nll fp32."""
    original = F.normalize
    F.normalize = _normalize_fp32_norm
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            yield
    finally:
        F.normalize = original
