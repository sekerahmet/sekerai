"""Dikkat: ters kare çekim (rgn) ve karşılaştırma tabanı (softmax).

Çekim ağırlığı (doküman 2.1, 2.4), k = i - j >= 0:
    s_ij = q̂_i · k̂_j,   d_ij = ||q̂_i - k̂_j||² = 2 - 2 s_ij ∈ [0, 4]   (birim küre)
    payda = 1 + γ_h d^p + β_h k                              pos: additive (ANA: anlam uzaklığı + sıra uzaklığı)
          = (1 + γ_h d^p)(1 + β_h k)                         pos: alibi | rope+alibi   (çarpımsal)
          = 1 + γ_h d^p                                      pos: rope | none
    w_ij = 1 / payda,   O_i = Σ_j w_ij V_j / Σ_j w_ij        (maksimum takibi yok)

Uygulamalar:
    dense  başvuru: N×N açık hesap (CPU, testler, teşhis).  Bellek O(N²).
    flex   FlexAttention, score_mod = -log(payda): softmax(-log payda) = w / Σw birebir aynı ağırlık.  CUDA'da birleşik
           çekirdek, N×N belleğe yazılmaz.  KALİTE deneyi içindir: içeride exp / log var; hız iddiası (V7) özel çekirdek
           ister (doküman 3.3).
"""
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from rgn.model.norm import RMSNorm, unit
from rgn.model.position import apply_rope, geometric

_POS_MODE = {"additive": "add", "alibi": "mul", "rope+alibi": "mul"}


def denominator(s, gamma, beta, k, pos, power):
    """Çekim paydası (>= 1).  s: q̂·k̂; gamma / beta: yayınlanabilir head değerleri; k = i - j (>= 0).
    Hem dense (tensörler) hem flex score_mod (eleman) yolunda aynı fonksiyon kullanılır."""
    d = torch.clamp(2.0 - 2.0 * s, min=0.0)
    if power == 2:
        d = d * d
    den = 1.0 + gamma * d
    mode = _POS_MODE.get(pos)
    if mode == "add":
        den = den + beta * k
    elif mode == "mul":
        den = den * (1.0 + beta * k)
    return den


def rgn_dense(q, k, v, gamma, beta, pos, power, return_weights=False):
    """q (B, H, N, dh), k / v (B, Hkv, N, dh) -> O (B, H, N, dh) [, α (B, H, N, N)]."""
    H, N = q.shape[1], q.shape[2]
    if k.shape[1] != H:
        rep = H // k.shape[1]
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    s = q @ k.transpose(-2, -1)
    idx = torch.arange(N, device=q.device)
    rel = idx[:, None] - idx[None, :]
    beta_b = None if beta is None else beta.view(1, -1, 1, 1)
    den = denominator(s, gamma.view(1, -1, 1, 1), beta_b, rel.clamp_min(0).to(s.dtype), pos, power)
    w = (1.0 / den).masked_fill(rel < 0, 0.0)
    z = w.sum(dim=-1, keepdim=True)                     # köşegen > 0 olduğundan z > 0
    out = (w @ v) / z
    return (out, w / z) if return_weights else out


_FLEX = {}
FLEX_BLOCK = 128           # flex yalnız N bu sayının katıyken (eğitim / değerlendirme); diğer uzunluklar (üretim) dense


def _flex_attention():
    if "fn" not in _FLEX:
        from torch.nn.attention.flex_attention import flex_attention
        for k in ("recompile_limit", "cache_size_limit"):        # eğitim, değerlendirme, teşhis, profil şekilleri
            if getattr(torch._dynamo.config, k, 64) < 64:
                setattr(torch._dynamo.config, k, 64)
        _FLEX["fn"] = torch.compile(flex_attention, dynamic=False)
    return _FLEX["fn"]


def _causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def _causal_block_mask(n, device):
    key = ("mask", n, str(device))
    if key not in _FLEX:
        from torch.nn.attention.flex_attention import create_block_mask
        _FLEX[key] = create_block_mask(_causal, None, None, n, n, device=device)
    return _FLEX[key]


def rgn_flex(q, k, v, gamma, beta, pos, power):
    """FlexAttention yolu (CUDA).  γ ve β score_mod'a yakalanan tensörlerdir; gradyan alırlar (PyTorch >= 2.6)."""
    def score_mod(score, b, h, q_idx, kv_idx):
        # k = i - j 0'da kırpılır (dense ile aynı): maskeli konumlarda (j > i) payda 0 / negatif olmasın — flex'in geri
        # geçişi yakalanan γ / β'nın gradyanını maskeden bağımsız topladığı için aksi halde NaN üretir.
        k = torch.clamp(q_idx - kv_idx, min=0)
        return -torch.log(denominator(score, gamma[h], None if beta is None else beta[h], k, pos, power))

    return _flex_attention()(q, k, v, score_mod=score_mod, block_mask=_causal_block_mask(q.shape[2], q.device),
                             scale=1.0, enable_gqa=k.shape[1] != q.shape[1])


class Attention(nn.Module):
    """Çekim (rgn) ya da softmax dikkati; GQA, konum terimi (β) ya da RoPE."""

    def __init__(self, cfg):
        super().__init__()
        d, H, Hkv = cfg.d_model, cfg.n_heads, cfg.n_kv_heads
        self.cfg, self.H, self.Hkv, self.dh = cfg, H, Hkv, d // H
        self.w_q = nn.Linear(d, d, bias=False)                         # W_Q
        self.w_kv = nn.Linear(d, 2 * Hkv * self.dh, bias=False)        # W_K, W_V
        self.w_o = nn.Linear(d, d, bias=False)                         # W_O (başları birleştirir)
        self.use_rope = cfg.pos in ("rope", "rope+alibi")
        self.beta_hat = None
        if cfg.attn == "softmax":
            self.q_norm, self.k_norm = RMSNorm(self.dh), RMSNorm(self.dh)
            return
        # γ = γ_min + γ̂² (artan sırada): her zaman pozitif, "ölü" bölgeye kilitlenmez (doküman 2.3, 4.2)
        g_hat = (geometric(*cfg.gamma_range, H) - cfg.gamma_min).clamp_min(0).sqrt()
        if cfg.learn_gamma:
            self.gamma_hat = nn.Parameter(g_hat)
        else:
            self.register_buffer("gamma_hat", g_hat)
        if cfg.pos in _POS_MODE:
            # β = β̂² (azalan sırada: ilk head yerel, son head neredeyse global); β = 0 ile başlatılmaz, β̂ = 0'da türev 0
            self.beta_hat = nn.Parameter(geometric(*cfg.beta_range, H).sqrt())

    def gamma(self):
        return self.cfg.gamma_min + self.gamma_hat ** 2

    def beta(self):
        return None if self.beta_hat is None else self.beta_hat ** 2

    def impl(self, device):
        c = self.cfg.attn_impl
        if self.cfg.attn == "softmax":
            return "sdpa"
        if c == "auto":
            return "flex" if device.type == "cuda" else "dense"
        return c

    def _flex_ok(self, n):
        return n >= FLEX_BLOCK and n % FLEX_BLOCK == 0

    def project(self, x, rope):
        """x (B, N, d) -> q (B, H, N, dh), k / v (B, Hkv, N, dh); rgn'de q̂, k̂ birim küre üzerinde."""
        B, N, _ = x.shape
        q = self.w_q(x).view(B, N, self.H, self.dh).transpose(1, 2)
        k, v = self.w_kv(x).view(B, N, 2, self.Hkv, self.dh).permute(2, 0, 3, 1, 4)
        if self.cfg.attn == "rgn":
            dt = torch.float32 if self.cfg.rgn_fp32 else q.dtype
            q, k, v = unit(q.to(dt)), unit(k.to(dt)), v.to(dt)
        else:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.use_rope:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        return q, k, v

    def _core(self, q, k, v):
        impl = self.impl(q.device)
        if impl == "sdpa":
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=self.Hkv != self.H)
        args = (self.gamma(), self.beta(), self.cfg.pos, self.cfg.power)
        ctx = torch.autocast(q.device.type, enabled=False) if self.cfg.rgn_fp32 else contextlib.nullcontext()
        with ctx:
            if impl == "flex" and self._flex_ok(q.shape[2]):
                return rgn_flex(q, k, v, *args)
            return rgn_dense(q, k, v, *args)

    def forward(self, x, rope=None):
        B, N, D = x.shape
        q, k, v = self.project(x, rope)
        out = self._core(q, k, v)
        return self.w_o(out.to(x.dtype).transpose(1, 2).reshape(B, N, D))

    @torch.no_grad()
    def weights(self, x, rope=None):
        """Teşhis: dikkat ağırlıkları α (B, H, N, N), her satırın toplamı 1 (dense hesap)."""
        q, k, v = self.project(x, rope)
        if self.cfg.attn == "rgn":
            with torch.autocast(q.device.type, enabled=False):
                return rgn_dense(q.float(), k.float(), v.float(), self.gamma(), self.beta(), self.cfg.pos,
                                 self.cfg.power, return_weights=True)[1]
        if self.Hkv != self.H:
            k = k.repeat_interleave(self.H // self.Hkv, dim=1)
        s = (q.float() @ k.float().transpose(-2, -1)) / self.dh ** 0.5
        n = s.shape[-1]
        s = s.masked_fill(~torch.ones(n, n, dtype=torch.bool, device=s.device).tril(), float("-inf"))
        return torch.softmax(s, dim=-1)
