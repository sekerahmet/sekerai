"""operators -- Model S Parca 1: denklemin sag tarafi, ayri ayri degerlendirilen operatorler (TASARIM.md s1, s3).

    du/dt = A(u) + F(u),   P: izdusum (RMSNorm; operatorler P(u)'da degerlendirilir)

Her operator ARTIS dondurur (du), durumu degistirmez; zaman adimi semasi (integrator) artisi kendi katsayisiyla ekler.
    P  Projection     her token vektorunu yaricap kisitina izdusurur: u / rms(u) * g
    A  AttentionOp    x ekseninde yerel olmayan integral: sum_x' gamma(x, x') V(x'), gamma = softmax(<Q(x), K(x')> / sqrt(hd))
                      (nedensel ya da verilen maske), RoPE konum, QK izdusumu (fp32), GQA (k / v daha az head)
    F  FeedForwardOp  her x'te ayri: SwiGLU (down(silu(gate) * up))
"""
import dataclasses

import torch
import torch.nn.functional as F_


@dataclasses.dataclass
class Context:
    """Operatorlerin ortak girdisi: pos (B, T) RoPE konumu; mask None (nedensel), bool (B, T, T) ya da FlexAttention BlockMask."""
    pos: torch.Tensor
    mask: object = None


class Projection(torch.nn.Module):
    """P: u -> u / rms(u) * g (RMSNorm; makalede LN = kisita izdusum)."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.g = torch.nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, u):
        uf = u.float()
        return (uf * torch.rsqrt(uf.pow(2).mean(-1, keepdim=True) + self.eps)).to(u.dtype) * self.g


def rope(x, pos, base=10000.0):
    """x (B, H, T, hd), pos (B, T) -> yarim boyut ciftlerinde donme (konum pos)."""
    half = x.shape[-1] // 2
    freq = base ** (-torch.arange(half, device=x.device, dtype=torch.float) / half)
    ang = pos[:, None, :, None].float() * freq
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    a, b = x[..., :half], x[..., half:]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], -1)


class AttentionOp(torch.nn.Module):
    """A: du(x) = W_o sum_x' gamma(x, x') V(x').  Q, K, V = P(u) W_{q,k,v}; Q, K head basina izdusum (fp32) ve RoPE."""

    def __init__(self, d, heads, kv_heads=None, rope_on=True, head_dim=None):
        """head_dim (varsayilan d / heads): heads x head_dim > d genis attention (Parca 3, birlesik asama)."""
        super().__init__()
        assert head_dim or d % heads == 0, "d heads'e bolunmeli"
        self.d, self.heads, self.hd = d, heads, int(head_dim or d // heads)
        self.inner = heads * self.hd
        self.kv_heads = int(kv_heads or heads)
        assert heads % self.kv_heads == 0, "kv_heads heads'in boleni olmali"
        self.rope_on = rope_on
        self.P = Projection(d)
        self.w_q = torch.nn.Linear(d, self.inner, bias=False)
        self.w_kv = torch.nn.Linear(d, 2 * self.kv_heads * self.hd, bias=False)
        self.w_o = torch.nn.Linear(self.inner, d, bias=False)
        self.q_P, self.k_P = Projection(self.hd), Projection(self.hd)

    def qkv(self, u, pos):
        """-> q (B, H, T, hd), k / v (B, Hkv, T, hd)."""
        B, T, _ = u.shape
        h = self.P(u)
        q = self.w_q(h).view(B, T, self.heads, self.hd).transpose(1, 2)
        k, v = self.w_kv(h).view(B, T, 2, self.kv_heads, self.hd).permute(2, 0, 3, 1, 4)
        with torch.autocast(u.device.type, enabled=False):
            q, k = self.q_P(q.float()).to(v.dtype), self.k_P(k.float()).to(v.dtype)
        if self.rope_on:
            q, k = rope(q, pos), rope(k, pos)
        return q, k, v

    @staticmethod
    def kernel(q, k, v, mask):
        """sum_x' softmax(<q, k> / sqrt(hd)) v; mask None nedensel, bool (B, T, T) ya da BlockMask."""
        gqa = k.shape[1] != q.shape[1]
        if mask is None:
            return F_.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=gqa)
        if torch.is_tensor(mask):
            if gqa:
                r = q.shape[1] // k.shape[1]
                k, v = k.repeat_interleave(r, 1), v.repeat_interleave(r, 1)
            return F_.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
        from torch.nn.attention.flex_attention import flex_attention
        bs = mask.BLOCK_SIZE[0]                         # 128 disinda varsayilan cekirdek hata veriyor (baseline ile ayni)
        return flex_attention(q, k, v, block_mask=mask, enable_gqa=gqa, kernel_options=None if bs == 128 else dict.fromkeys(
            ("BLOCK_M", "BLOCK_N", "BLOCK_M1", "BLOCK_N1", "BLOCK_M2", "BLOCK_N2"), bs))

    def forward(self, u, ctx):
        q, k, v = self.qkv(u, ctx.pos)
        return self.out(self.kernel(q, k, v, ctx.mask))

    def out(self, a):
        """a (B, H, T, hd) -> du (B, T, d)."""
        B, _, T, _ = a.shape
        return self.w_o(a.transpose(1, 2).reshape(B, T, self.inner))


class MixOp(torch.nn.Module):
    """M: A ve F tek operator (Strata; kullanici, 11 Ekim: "işin matematiğini hızlandıran kolaylaştıran basitleştiren").  F, cekirdegi
    birim matris olan bir A'dir: A(u) + F(u) = sum_x' [gamma(x, x') W_ov + delta(x, x') F] u(x').  Bu yuzden:
        z = W_in P(u) = [q | k | v | yerel];  du = W_out [ sum_x' gamma(x, x') V(x')  |  silu(g) * up ]
    values "linear": V = v (= u + A(Pu) + F(Pu), paralel blok; ayni agirlikla birebir); "swiglu": V = silu(v_g) * v_u
    (attention dogrusal olmayan ozellikleri karistirir).  Katman basina 1 izdusum, 2 matmul, 1 attention; bolme yok."""

    def __init__(self, d, heads, hidden, kv_heads=None, head_dim=None, values="swiglu", rope_on=True):
        super().__init__()
        assert values in ("linear", "swiglu"), "values: linear | swiglu"
        assert head_dim or d % heads == 0, "d heads'e bolunmeli"
        self.d, self.heads, self.hd = d, heads, int(head_dim or d // heads)
        self.kv_heads = int(kv_heads or heads)
        assert heads % self.kv_heads == 0, "kv_heads heads'in boleni olmali"
        self.inner, self.kvd, self.hidden, self.values, self.rope_on = heads * self.hd, self.kv_heads * self.hd, hidden, values, rope_on
        self.split = [self.inner, self.kvd, self.kvd * (2 if values == "swiglu" else 1), 2 * hidden]
        self.P = Projection(d)
        self.w_in = torch.nn.Linear(d, sum(self.split), bias=False)
        self.w_out = torch.nn.Linear(self.inner + hidden, d, bias=False)
        self.q_P, self.k_P = Projection(self.hd), Projection(self.hd)

    def qkv_local(self, u, pos):
        """-> q (B, H, T, hd), k / v (B, Hkv, T, hd), yerel ozellik (B, T, hidden)."""
        B, T, _ = u.shape
        q, k, v, loc = self.w_in(self.P(u)).split(self.split, -1)
        if self.values == "swiglu":
            vg, vu = v.chunk(2, -1)
            v = F_.silu(vg) * vu
        q = q.view(B, T, self.heads, self.hd).transpose(1, 2)
        k, v = (t.view(B, T, self.kv_heads, self.hd).transpose(1, 2) for t in (k, v))
        with torch.autocast(u.device.type, enabled=False):
            q, k = self.q_P(q.float()).to(v.dtype), self.k_P(k.float()).to(v.dtype)
        if self.rope_on:
            q, k = rope(q, pos), rope(k, pos)
        g, up = loc.chunk(2, -1)
        return q, k, v, F_.silu(g) * up

    def out(self, a, loc):
        B, _, T, _ = a.shape
        return self.w_out(torch.cat([a.transpose(1, 2).reshape(B, T, self.inner), loc], -1))

    def forward(self, u, ctx):
        q, k, v, loc = self.qkv_local(u, ctx.pos)
        return self.out(AttentionOp.kernel(q, k, v, ctx.mask), loc)


def mix_hidden(d, inner, kvd, hidden, values):
    """Esit matris parametresi (= esit FLOP): ayri A + F (heads, kv_heads, hidden) ile ayni toplam -> MixOp yerel genisligi
    (64'un kati, asagi).  linear: hidden aynen; swiglu: v'nin ikinci yarisi yerel genislikten duser (d 1024, tam kv: 2.736 -> 2.368)."""
    if values == "linear":
        return hidden
    base = d * (inner + 2 * kvd) + inner * d + 3 * d * hidden          # inner = q boyu, kvd = k (= v) boyu
    attn = d * (inner + 3 * kvd) + inner * d
    return max(64, int((base - attn) / (3 * d)) // 64 * 64)


class FeedForwardOp(torch.nn.Module):
    """F: du(x) = W_down(silu(W_gate P(u)) * W_up P(u)), her x'te ayri."""

    def __init__(self, d, hidden):
        super().__init__()
        self.P = Projection(d)
        self.w_gate_up = torch.nn.Linear(d, 2 * hidden, bias=False)
        self.w_down = torch.nn.Linear(hidden, d, bias=False)

    def forward(self, u, ctx=None):
        g, up = self.w_gate_up(self.P(u)).chunk(2, -1)
        return self.w_down(F_.silu(g) * up)
