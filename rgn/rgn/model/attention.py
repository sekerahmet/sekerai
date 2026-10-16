"""Dikkat: softmax + RoPE + QK-norm, GQA.  Yerel (Y) katman yalnız son `window` token'a bakar; genel (G) katman tam
causal.  CUDA'da yerel katman ve hikâye maskesi FlexAttention blok maskesiyle (dışarıda kalan bloklar hiç hesaplanmaz),
diğer durumlarda SDPA.  İsteğe bağlı çıktı kapısı: çıktı = kapı · Σ pay·v, kapı = 1 + r(z) ∈ (0, 2), r(z) = z/√(1+z²),
sıfır başlatmada 1 (taban ile aynı).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from rgn.model.norm import RMSNorm
from rgn.model.position import apply_rope

_FLEX = {}
FLEX_BLOCK = 128           # flex yalnız N bu sayının katıyken (eğitim / değerlendirme); diğer uzunluklar (üretim) SDPA


def _flex_attention():
    if "fn" not in _FLEX:
        from torch.nn.attention.flex_attention import flex_attention
        for k in ("recompile_limit", "cache_size_limit"):        # eğitim, değerlendirme, teşhis, profil şekilleri
            if getattr(torch._dynamo.config, k, 64) < 64:
                setattr(torch._dynamo.config, k, 64)
        _FLEX["fn"] = torch.compile(flex_attention, dynamic=False)
    return _FLEX["fn"]


def _window_block_mask(n, window, device):
    key = (n, window, str(device))
    if key not in _FLEX:
        from torch.nn.attention.flex_attention import create_block_mask

        def mask_mod(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & (q_idx - kv_idx < window)
        _FLEX[key] = create_block_mask(mask_mod, None, None, n, n, device=device, _compile=True)
    return _FLEX[key]


def _story_mod(doc, sent, kind, window):
    """Hikâye dizilimi maskesi (True = bakılır): aynı hikâye, causal; Y katmanı: window > 0 ise son window token,
    window == 0 ise yalnız kendi cümlesi + hikâye başı (sent -1).  Dolgu (doc -1, sent -2) yalnız dolguya bakar."""
    def mm(b, h, q, kv):
        ok = (doc[b, q] == doc[b, kv]) & (q >= kv)
        if kind == "Y":
            ok = ok & ((q - kv < window) if window else ((sent[b, q] == sent[b, kv]) | (sent[b, kv] == -1)))
        return ok
    return mm


def story_masks(doc, sent, window):
    """doc, sent (B, N) -> {"Y": maske, "G": maske}.  CUDA'da FlexAttention BlockMask, CPU'da yoğun (B, 1, N, N) bool."""
    B, N = doc.shape
    if doc.is_cuda:
        from torch.nn.attention.flex_attention import create_block_mask
        return {k: create_block_mask(_story_mod(doc, sent, k, window), B, None, N, N, device=doc.device, _compile=True)
                for k in "YG"}
    b = torch.arange(B)[:, None, None]
    q, kv = torch.arange(N)[None, :, None], torch.arange(N)[None, None, :]
    return {k: _story_mod(doc, sent, k, window)(b, 0, q, kv).unsqueeze(1) for k in "YG"}


class Attention(nn.Module):
    """Softmax dikkati; window > 0: yerel katman (son window token)."""

    def __init__(self, cfg, window=0):
        super().__init__()
        d, H, Hkv = cfg.d_model, cfg.n_heads, cfg.n_kv_heads
        self.cfg, self.H, self.Hkv, self.dh = cfg, H, Hkv, cfg.dh()
        self.window = int(window)
        self.w_q = nn.Linear(d, H * self.dh, bias=False)
        self.w_kv = nn.Linear(d, 2 * Hkv * self.dh, bias=False)
        self.w_o = nn.Linear(H * self.dh, d, bias=False)
        self.q_norm, self.k_norm = RMSNorm(self.dh), RMSNorm(self.dh)
        # kapı: head başına (H) ya da her çıktı sayısı (d); RGNModel sıfırla başlatır (kapı 1)
        self.w_aux = nn.Linear(d, {"": 0, "head": H, "elem": H * self.dh}[cfg.attn_gate], bias=False) if cfg.attn_gate else None

    def project(self, x, rope):
        """x (B, N, d) -> q (B, H, N, dh), k / v (B, Hkv, N, dh); q, k RMSNorm + RoPE."""
        B, N, _ = x.shape
        q = self.w_q(x).view(B, N, self.H, self.dh).transpose(1, 2)
        k, v = self.w_kv(x).view(B, N, 2, self.Hkv, self.dh).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        return apply_rope(q, *rope), apply_rope(k, *rope), v

    def _mask(self, n, device):
        """Yerel katman maskesi (True = bakılır): 0 <= i - j < window."""
        rel = torch.arange(n, device=device)[:, None] - torch.arange(n, device=device)[None, :]
        return (rel >= 0) & (rel < self.window)

    def _core(self, q, k, v, mask=None):
        s, gqa = self.dh ** -0.5, self.Hkv != self.H
        if mask is not None:                                           # hikâye dizilimi
            if isinstance(mask, torch.Tensor):
                return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=gqa, scale=s)
            return _flex_attention()(q, k, v, block_mask=mask, scale=s, enable_gqa=gqa)
        if not self.window:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=gqa, scale=s)
        n = q.shape[2]
        if q.is_cuda and n >= FLEX_BLOCK and n % FLEX_BLOCK == 0:     # blok atlamalı yerel pencere
            return _flex_attention()(q, k, v, block_mask=_window_block_mask(n, self.window, q.device), scale=s,
                                     enable_gqa=gqa)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=self._mask(n, q.device), enable_gqa=gqa, scale=s)

    def gate(self, x):
        """x (B, N, d) -> kapı (B, H, N, 1 | dh) ya da None."""
        if self.w_aux is None:
            return None
        B, N, _ = x.shape
        z = self.w_aux(x).float()
        g = 1.0 + z * torch.rsqrt(1.0 + z * z)
        return (g.view(B, N, self.H, 1) if self.cfg.attn_gate == "head" else g.view(B, N, self.H, self.dh)).transpose(1, 2)

    def forward(self, x, rope, mask=None, vv=None):
        """x (B, N, d) -> (B, N, d).  mask: hikâye dizilimi maskesi (story_masks) ya da None.  vv (B, N, Hkv·dh):
        kelimeye özel V vektörü (V'ye eklenir) ya da None."""
        B, N, D = x.shape
        q, k, v = self.project(x, rope)
        if vv is not None:
            v = v + vv.view(B, N, self.Hkv, self.dh).transpose(1, 2).to(v.dtype)
        out = self._core(q, k, v, mask)
        g = self.gate(x)
        if g is not None:
            out = out * g.to(out.dtype)
        return self.w_o(out.to(x.dtype).transpose(1, 2).reshape(B, N, self.H * self.dh))

    @torch.no_grad()
    def weights(self, x, rope):
        """Teşhis: dikkat payları (B, H, N, N), satır toplamı 1 (açık hesap)."""
        q, k, _ = self.project(x, rope)
        if self.Hkv != self.H:
            k = k.repeat_interleave(self.H // self.Hkv, dim=1)
        s = (q.float() @ k.float().transpose(-2, -1)) / self.dh ** 0.5
        n = s.shape[-1]
        keep = self._mask(n, s.device) if self.window else torch.ones(n, n, dtype=torch.bool, device=s.device).tril()
        return torch.softmax(s.masked_fill(~keep, float("-inf")), dim=-1)
