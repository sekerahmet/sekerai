"""Dikkat: ters kare çekim (rgn) ve karşılaştırma tabanı (softmax).

Çekim ağırlığı (doküman 2.1, 2.4), k = i - j >= 0:
    s_ij = q̂_i · k̂_j,   d_ij = ||q̂_i - k̂_j||² = 2 - 2 s_ij ∈ [0, 4]   (birim küre)
    payda = 1 + γ_h d^p                                      pos: rope (VARSAYILAN; q̂, k̂ döndürülmüş) | none
          = 1 + γ_h d^p + β_h k                              pos: additive (anlam uzaklığı + sıra uzaklığı)
          = (1 + γ_h d^p)(1 + β_h k)                         pos: alibi | rope+alibi   (çarpımsal)
    w_ij = 1 / payda,   O_i = Σ_j w_ij V_j / Σ_j w_ij        (maksimum takibi yok)

Keskinlik üssü k (sharpness): w_ij = payda(γ/k)^(-k) = (1 + γ d / k)^(-k).  k = 1 ters kare çekim; k → ∞ iken w → e^(-γd)
(Cauchy -> Gauss / softmax çekirdeği).  Tam eşleşmedeki eğim k'den bağımsız (-γ); kuyruk inceldikçe en iyi / ilgisiz oranı
(1 + 2γ/k)^k olur (k = 1'de 1 + 2γ).  Ölçüldü (SS, 15 Ekim): k = 1'de dikkat top1 0,047, transformer 0,263.

Uygulamalar:
    dense  başvuru: N×N açık hesap (CPU, testler, teşhis).  Bellek O(N²).
    flex   FlexAttention, score_mod = -log(payda): softmax(-log payda) = w / Σw birebir aynı ağırlık.  CUDA'da birleşik
           çekirdek, N×N belleğe yazılmaz.  KALİTE deneyi içindir: içeride exp / log var; hız iddiası (V7) özel çekirdek
           ister (doküman 3.3).
    triton Flash-RGN (rgn/model/triton_attention.py): birleşik ileri + geri çekirdek, maksimum takibi yok.  pos rope / none,
           power 1; CUDA'da varsayılan.
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


def rgn_dense(q, k, v, gamma, beta, pos, power, sharpness=1, return_weights=False, av_bf16=False, window=0):
    """q (B, H, N, dh), k / v (B, Hkv, N, dh) -> O (B, H, N, dh) [, α (B, H, N, N)].
    sharpness: keskinlik üssü k — sayı (tüm head'ler) ya da (H,) tensör (head başına, öğrenilebilir).
    av_bf16: w·V çarpımı bf16 Tensor Core'da (w ∈ [0, 1]; FlashAttention'daki P·V gibi).  q·kᵀ fp32 kalır (yakın
    eşleşmede 2 - 2 q·k iptal hatası, doküman 3.5).
    window > 0: yerel katman — i yalnız i - window < j <= i'ye bakar."""
    H, N = q.shape[1], q.shape[2]
    if k.shape[1] != H:
        rep = H // k.shape[1]
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    s = q @ k.transpose(-2, -1)
    idx = torch.arange(N, device=q.device)
    rel = idx[:, None] - idx[None, :]
    beta_b = None if beta is None else beta.view(1, -1, 1, 1)
    K = sharpness.view(1, -1, 1, 1) if torch.is_tensor(sharpness) else sharpness
    den = denominator(s, gamma.view(1, -1, 1, 1) / K, beta_b, rel.clamp_min(0).to(s.dtype), pos, power)
    w = (1.0 / den) if (not torch.is_tensor(K) and K == 1) else den.pow(-K)
    w = w.masked_fill((rel < 0) | (rel >= window) if window else rel < 0, 0.0)
    z = w.sum(dim=-1, keepdim=True).clamp_min(1e-30)    # köşegen > 0; k <= 8'de alt taşma yok, yine de 0/0'a karşı
    if av_bf16 and not return_weights:
        out = (w.to(torch.bfloat16) @ v.to(torch.bfloat16)).float() / z
    else:
        out = (w @ v) / z
    return (out, w / z) if return_weights else out


_TRITON = {}


def triton_ok():
    """Triton çekirdeği yüklenebiliyor mu (CUDA ortamında bir kez denenir)."""
    if "ok" not in _TRITON:
        try:
            from rgn.model.triton_attention import rgn_triton
            _TRITON.update(ok=True, fn=rgn_triton)
        except Exception as e:                                      # triton yok / sürüm uyumsuz: dense'e düş
            _TRITON.update(ok=False, err=repr(e))
    return _TRITON["ok"]


def rgn_triton(q, k, v, gamma, sharpness=1, qk_prec="tf32x3", window=0, int_k=False):
    assert triton_ok(), _TRITON.get("err")
    return _TRITON["fn"](q, k, v, gamma, sharpness, qk_prec, window, int_k)


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


def _window_block_mask(n, window, device):
    key = ("win", n, window, str(device))
    if key not in _FLEX:
        from torch.nn.attention.flex_attention import create_block_mask

        def mask_mod(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & (q_idx - kv_idx < window)
        _FLEX[key] = create_block_mask(mask_mod, None, None, n, n, device=device)
    return _FLEX[key]


def softmax_window(q, k, v, window, scale):
    """Softmax yerel katman (CUDA): FlexAttention blok maskesi — pencere dışındaki bloklar ileri ve geri geçişte hiç
    hesaplanmaz (SDPA'nın N×N maskeli yolu bütün blokları hesaplar)."""
    return _flex_attention()(q, k, v, block_mask=_window_block_mask(q.shape[2], window, q.device), scale=scale,
                             enable_gqa=k.shape[1] != q.shape[1])


@torch.no_grad()
def locality(q, k, window, scale, taus):
    """Konum özeti (saf gözlem, gradyansız) -> (B, N, T·H):  L_τ,i = Σ_j pay_ij · e^(−(i−j)/τ) ∈ (0, 1].
    pay = softmax(q·k·ölçek) (causal, yerel katmanda pencere) — dikkatin kendi payları.  "Dikkatimin ne kadarı ~τ token
    içinden geldi": 1 = tamamen yakından, 0 = tamamen uzaktan.  Tom örneği (he -> Tom 12 geri %80): L_4 ≈ 0,08,
    L_32 ≈ 0,72; önceki-token head'i: L_4 ≈ 0,72.  Açık N×N hesap (yalnız ileri, gradyansız)."""
    B, H, N, _ = q.shape
    if k.shape[1] != H:
        k = k.repeat_interleave(H // k.shape[1], 1)
    s = (q.float() @ k.float().transpose(-2, -1)) * scale
    d = (torch.arange(N, device=q.device)[:, None] - torch.arange(N, device=q.device)[None, :]).float()
    keep = (d >= 0) & (d < window) if window else d >= 0
    p = torch.softmax(s.masked_fill(~keep, float("-inf")), -1)
    w = torch.stack([torch.exp(-d.clamp_min(0) / t) for t in taus])               # (T, N, N)
    loc = torch.einsum("bhij,tij->bthi", p, w)                                     # (B, T, H, N)
    return loc.reshape(B, len(taus) * H, N).transpose(1, 2)


def rgn_flex(q, k, v, gamma, beta, pos, power, sharpness=1):
    """FlexAttention yolu (CUDA).  γ ve β score_mod'a yakalanan tensörlerdir; gradyan alırlar (PyTorch >= 2.6)."""
    def score_mod(score, b, h, q_idx, kv_idx):
        # k = i - j 0'da kırpılır (dense ile aynı): maskeli konumlarda (j > i) payda 0 / negatif olmasın — flex'in geri
        # geçişi yakalanan γ / β'nın gradyanını maskeden bağımsız topladığı için aksi halde NaN üretir.
        k = torch.clamp(q_idx - kv_idx, min=0)
        K = sharpness[h] if torch.is_tensor(sharpness) else sharpness
        den = denominator(score, gamma[h] / K, None if beta is None else beta[h], k, pos, power)
        return -K * torch.log(den)

    return _flex_attention()(q, k, v, score_mod=score_mod, block_mask=_causal_block_mask(q.shape[2], q.device),
                             scale=1.0, enable_gqa=k.shape[1] != q.shape[1])


class Attention(nn.Module):
    """Çekim (rgn) ya da softmax dikkati; GQA, konum terimi (β) ya da RoPE.  window > 0: yerel katman (son window token)."""

    def __init__(self, cfg, window=0):
        super().__init__()
        d, H, Hkv = cfg.d_model, cfg.n_heads, cfg.n_kv_heads
        self.cfg, self.H, self.Hkv, self.dh = cfg, H, Hkv, d // H
        self.window = int(window)
        self.w_q = nn.Linear(d, d, bias=False)                         # W_Q
        self.w_kv = nn.Linear(d, 2 * Hkv * self.dh, bias=False)        # W_K, W_V
        self.w_o = nn.Linear(d, d, bias=False)                         # W_O (başları birleştirir)
        self.use_rope = cfg.pos in ("rope", "rope+alibi")
        self.beta_hat = None
        # önem / odak / kapı ham değerleri tek projeksiyondan (RGNModel sıfırla başlatır)
        self.aux_sizes = [n for n in (Hkv if cfg.attn_importance else 0, H if cfg.attn_focus else 0,
                                      {"": 0, "head": H, "elem": d}[cfg.attn_gate])]
        self.w_aux = nn.Linear(d, sum(self.aux_sizes), bias=False) if sum(self.aux_sizes) else None
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
        self.sharp_rho = None
        if cfg.sharpness_learn:
            # k_h = lo + (hi - lo) σ_a(ρ_h), σ_a(u) = ½(1 + u/√(1+u²)) (üstel işlem yok); başlangıç k_h = sharpness_init
            lo, hi = cfg.sharpness_range
            t = 2 * (cfg.sharpness_init - lo) / (hi - lo) - 1               # σ_a⁻¹: u = t / √(1 - t²)
            self.sharp_rho = nn.Parameter(torch.full((H,), t / (1 - t * t) ** 0.5))

    def gamma(self):
        """γ = γ_min + γ̂², k'ye bağlı tavanla: (1 + 4γ/k)^k <= 2^120 -> en uzak anahtarın ağırlığı fp32'de alt taşmaz
        (bir satırın bütün ağırlıkları sıfıra düşüp çıktı sessizce sıfır olmasın).  k <= 8'de tavan ~65.000 (bağlamaz),
        k = 16'da ~720."""
        g = self.cfg.gamma_min + self.gamma_hat ** 2
        K = self.sharpness()
        if torch.is_tensor(K):
            K = K.detach()
            return torch.minimum(g, K * (torch.pow(2.0, 120.0 / K) - 1.0) / 4.0)
        return g.clamp(max=K * (2.0 ** (120.0 / K) - 1.0) / 4.0)

    def beta(self):
        return None if self.beta_hat is None else self.beta_hat ** 2

    def sharpness(self):
        """Keskinlik üssü: sabit sayı ya da head başına öğrenilen (H,) tensör."""
        if self.sharp_rho is None:
            return self.cfg.sharpness
        lo, hi = self.cfg.sharpness_range
        r = self.sharp_rho
        k = lo + (hi - lo) * 0.5 * (1.0 + r * torch.rsqrt(1.0 + r * r))
        if self.cfg.sharpness_int:
            k = k + (torch.round(k) - k).detach()                       # ileri tamsayı, geri sürekli (straight-through)
        return k

    def impl(self, device, n):
        """Kullanılacak çekirdek: sdpa (softmax), triton, dense ya da flex.  triton / flex yalnız CUDA'da; flex N blok
        boyunun katıyken."""
        c = self.cfg.attn_impl
        if self.cfg.attn == "softmax":
            return "sdpa"
        cuda = device.type == "cuda"
        if c == "auto" and cuda and self.cfg.pos in ("rope", "none") and self.cfg.power == 1 and triton_ok():
            return "triton"
        if c == "triton":
            return "triton" if cuda else "dense"
        if c == "auto":
            c = "dense" if (not cuda or n <= self.cfg.dense_max_len) else "flex"
        if c == "flex" and not (device.type == "cuda" and n >= FLEX_BLOCK and n % FLEX_BLOCK == 0):
            c = "dense"
        return c

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

    def _aux(self, x):
        """x (B, N, d) -> {"imp": (B, Hkv, N), "focus": (B, H, N), "gate": (B, H, N, 1 | dh)}; yalnız açık olanlar.
        Hepsi r(z) = z / √(1 + z²) ile (üstel işlem yok); sıfır başlatmada önem 0, odak 1, kapı 1."""
        if self.w_aux is None:
            return {}
        B, N, _ = x.shape
        parts = torch.split(self.w_aux(x).float(), self.aux_sizes, dim=-1)
        r = lambda z: z * torch.rsqrt(1.0 + z * z)                                     # noqa: E731
        out = {}
        z_imp, z_foc, z_gate = parts
        if self.cfg.attn_importance:
            out["imp"] = (self.cfg.importance_max * r(z_imp)).transpose(1, 2)
        if self.cfg.attn_focus:
            t = 2.0 / self.cfg.focus_max - 1.0                                           # σ_a(z0) = 1 / focus_max
            z0 = t / (1.0 - t * t) ** 0.5
            out["focus"] = (self.cfg.focus_max * 0.5 * (1.0 + r(z_foc + z0))).transpose(1, 2)
        if self.cfg.attn_gate:
            g = 1.0 + r(z_gate)                                                          # 2 σ_a(z)
            out["gate"] = (g.view(B, N, self.H, 1) if self.cfg.attn_gate == "head"
                           else g.view(B, N, self.H, self.dh)).transpose(1, 2)
        return out

    def _augment(self, q, k, v, imp):
        """Önemi hızlı çekirdeğe (SDPA / flash) taşımak için ek boyut: q'ya sabit 1, k'ya önem / ölçek; 8'in katına
        sıfırla doldurulur (flash).  q'·k'·ölçek = q·k·ölçek + önem_j."""
        pad = 8
        z = lambda t, n: t.new_zeros(*t.shape[:-1], n)                                  # noqa: E731
        q = torch.cat([q, torch.ones_like(q[..., :1]), z(q, pad - 1)], -1)
        k = torch.cat([k, (imp * self.dh ** 0.5).unsqueeze(-1).to(k.dtype), z(k, pad - 1)], -1)
        v = torch.cat([v, z(v, pad)], -1)
        return q, k, v

    def _core(self, q, k, v):
        impl = self.impl(q.device, q.shape[2])
        if impl == "sdpa":
            s = self.dh ** -0.5                     # açık ölçek: ek boyutlar (önem) varsayılan 1/√boyut'u değiştirmesin
            if self.window:
                n, dh = q.shape[2], q.shape[-1]
                if q.is_cuda and n >= FLEX_BLOCK and n % FLEX_BLOCK == 0 and dh & (dh - 1) == 0:
                    return softmax_window(q, k, v, self.window, s)           # blok atlamalı (hızlı)
                return F.scaled_dot_product_attention(q, k, v, attn_mask=self._mask(q.shape[2], q.device),
                                                      enable_gqa=self.Hkv != self.H, scale=s)
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=self.Hkv != self.H, scale=s)
        args = (self.gamma(), self.beta(), self.cfg.pos, self.cfg.power, self.sharpness())
        ctx = torch.autocast(q.device.type, enabled=False) if self.cfg.rgn_fp32 else contextlib.nullcontext()
        with ctx:
            if impl == "triton":
                return rgn_triton(q, k, v, args[0], args[4], self.cfg.rgn_qk_prec, self.window, self.cfg.sharpness_int)
            if impl == "flex":
                assert not self.window, "yerel pencere flex yolunda yok"
                return rgn_flex(q, k, v, *args)
            return rgn_dense(q, k, v, *args, av_bf16=self.cfg.rgn_av_bf16 and q.is_cuda, window=self.window)

    def _mask(self, n, device):
        """Softmax yerel katman maskesi (True = bakılır): 0 <= i - j < window."""
        rel = torch.arange(n, device=device)[:, None] - torch.arange(n, device=device)[None, :]
        return (rel >= 0) & (rel < self.window)

    def forward(self, x, rope=None):
        """-> çıktı (B, N, d); attn_locality açıksa (çıktı, konum özeti (B, N, T·H))."""
        B, N, D = x.shape
        q, k, v = self.project(x, rope)
        aux = self._aux(x)
        if "focus" in aux:
            q = q * aux["focus"].unsqueeze(-1).to(q.dtype)
        if "imp" in aux:
            q, k, v = self._augment(q, k, v, aux["imp"])
        loc = None
        if self.cfg.attn_locality:
            taus = self.cfg.locality_tau_local if self.window else self.cfg.locality_tau_global
            loc = locality(q, k, self.window, self.dh ** -0.5, taus)
        out = self._core(q, k, v)[..., :self.dh]
        if "gate" in aux:
            out = out * aux["gate"].to(out.dtype)
        y = self.w_o(out.to(x.dtype).transpose(1, 2).reshape(B, N, D))
        return (y, loc) if loc is not None else y

    @torch.no_grad()
    def weights(self, x, rope=None):
        """Teşhis: dikkat ağırlıkları α (B, H, N, N), her satırın toplamı 1 (dense hesap)."""
        q, k, v = self.project(x, rope)
        if self.cfg.attn == "rgn":
            with torch.autocast(q.device.type, enabled=False):
                return rgn_dense(q.float(), k.float(), v.float(), self.gamma(), self.beta(), self.cfg.pos,
                                 self.cfg.power, self.sharpness(), return_weights=True, window=self.window)[1]
        aux = self._aux(x)
        if "focus" in aux:
            q = q * aux["focus"].unsqueeze(-1).to(q.dtype)
        if self.Hkv != self.H:
            k = k.repeat_interleave(self.H // self.Hkv, dim=1)
        s = (q.float() @ k.float().transpose(-2, -1)) / self.dh ** 0.5
        if "imp" in aux:
            s = s + aux["imp"].repeat_interleave(self.H // self.Hkv, dim=1).unsqueeze(-2)
        n = s.shape[-1]
        keep = self._mask(n, s.device) if self.window else torch.ones(n, n, dtype=torch.bool, device=s.device).tril()
        s = s.masked_fill(~keep, float("-inf"))
        return torch.softmax(s, dim=-1)
