"""Kelimeye özel nöronun birleşik Triton çekirdeği (ileri + geri; kullanıcı, 16 Ekim: "özel çekirdek mantıklı").

Token t, satırı r = loc[t] (adımda geçen satırların kopyası `sub` içinde), girdi x_t (d):
    g, u, w = sub[r, 0:d], sub[r, d:2d], sub[r, 2d:3d]
    s1 = g·x, s2 = u·x, a = SiLU(s1)·s2,  çıktı = a·w
Ara (T, 3d) tensör belleğe yazılmaz (eski yol: satırlar önce okunup T × 3d'lik bir tensöre kopyalanıyordu).
Geri (çıktı gradyanı δ):
    da = δ·w,  ds1 = da·s2·SiLU'(s1),  ds2 = da·SiLU(s1)
    dx = ds1·g + ds2·u                                   (token başına çekirdek)
    dg_r = Σ_{t: loc=r} ds1_t x_t,  du_r = Σ ds2_t x_t,  dw_r = Σ a_t δ_t   (token'lar satıra göre sıralı; satır en çok
                                                         CH token'lık parçalara bölünür, parça toplamı atomik — senkronsuz)
CPU'da (testler, değerlendirme) düz PyTorch karşılığı: reference().
"""
import torch
import torch.nn.functional as F


def reference(x, loc, sub):
    """x (T, d), loc (T,), sub (R, 3d) -> (T, d).  Düz PyTorch (otomatik türev)."""
    d = x.shape[-1]
    w = F.embedding(loc, sub).view(-1, 3, d).to(x.dtype)
    a = F.silu((w[:, 0] * x).sum(-1)) * (w[:, 1] * x).sum(-1)
    return a.unsqueeze(-1) * w[:, 2]


_K = {}
CH = 32                     # satır gradyanı: parça başına en çok token (sık satır paralel parçalara bölünür)


def _kernels():
    if _K:
        return _K
    import triton
    import triton.language as tl

    @triton.jit
    def fwd(X, LOC, SUB, OUT, S, sx, ss, so, D: tl.constexpr):
        t = tl.program_id(0)
        r = tl.load(LOC + t).to(tl.int64)
        o = tl.arange(0, D)
        x = tl.load(X + t * sx + o).to(tl.float32)
        b = SUB + r * ss
        s1 = tl.sum(tl.load(b + o).to(tl.float32) * x, 0)
        s2 = tl.sum(tl.load(b + D + o).to(tl.float32) * x, 0)
        a = s1 * tl.sigmoid(s1) * s2
        tl.store(OUT + t * so + o, (a * tl.load(b + 2 * D + o).to(tl.float32)).to(OUT.dtype.element_ty))
        tl.store(S + 2 * t, s1)
        tl.store(S + 2 * t + 1, s2)

    @triton.jit
    def bwd_tok(DO, X, LOC, SUB, S, DX, C, sdo, sx, ss, sdx, D: tl.constexpr):
        t = tl.program_id(0)
        r = tl.load(LOC + t).to(tl.int64)
        o = tl.arange(0, D)
        b = SUB + r * ss
        do = tl.load(DO + t * sdo + o).to(tl.float32)
        s1 = tl.load(S + 2 * t)
        s2 = tl.load(S + 2 * t + 1)
        sg = tl.sigmoid(s1)
        silu = s1 * sg
        da = tl.sum(do * tl.load(b + 2 * D + o).to(tl.float32), 0)
        ds1 = da * s2 * sg * (1.0 + s1 * (1.0 - sg))
        ds2 = da * silu
        dx = ds1 * tl.load(b + o).to(tl.float32) + ds2 * tl.load(b + D + o).to(tl.float32)
        tl.store(DX + t * sdx + o, dx.to(DX.dtype.element_ty))
        tl.store(C + 3 * t, ds1)
        tl.store(C + 3 * t + 1, ds2)
        tl.store(C + 3 * t + 2, silu * s2)

    @triton.jit
    def bwd_row(DO, X, C, ORDER, CROW, CLO, CHI, DSUB, sdo, sx, sds, D: tl.constexpr):
        """Parça başına (bir satırın en çok CH ardışık token'ı): toplam fp32 DSUB'a atomik eklenir.  Sık satır birçok
        paralel parçaya bölünür (tek programda uzun döngü yok)."""
        c = tl.program_id(0)
        r = tl.load(CROW + c)
        if r >= 0:
            lo = tl.load(CLO + c)
            hi = tl.load(CHI + c)
            o = tl.arange(0, D)
            ag = tl.zeros([D], dtype=tl.float32)
            au = tl.zeros([D], dtype=tl.float32)
            aw = tl.zeros([D], dtype=tl.float32)
            for i in range(lo, hi):
                t = tl.load(ORDER + i).to(tl.int64)
                x = tl.load(X + t * sx + o).to(tl.float32)
                ag += tl.load(C + 3 * t) * x
                au += tl.load(C + 3 * t + 1) * x
                aw += tl.load(C + 3 * t + 2) * tl.load(DO + t * sdo + o).to(tl.float32)
            b = DSUB + r.to(tl.int64) * sds
            tl.atomic_add(b + o, ag)
            tl.atomic_add(b + D + o, au)
            tl.atomic_add(b + 2 * D + o, aw)

    _K.update(fwd=fwd, bwd_tok=bwd_tok, bwd_row=bwd_row)
    return _K


@torch.library.custom_op("rgn::private_fwd", mutates_args=())
def private_fwd(x: torch.Tensor, loc: torch.Tensor, sub: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    T, D = x.shape
    out = torch.empty_like(x)
    s = torch.empty((T, 2), device=x.device, dtype=torch.float32)
    _kernels()["fwd"][(T,)](x, loc, sub, out, s, x.stride(0), sub.stride(0), out.stride(0), D=D, num_warps=4)
    return out, s


@private_fwd.register_fake
def _(x, loc, sub):
    return torch.empty_like(x), x.new_empty((x.shape[0], 2), dtype=torch.float32)


@torch.library.custom_op("rgn::private_bwd", mutates_args=())
def private_bwd(do: torch.Tensor, x: torch.Tensor, loc: torch.Tensor, sub: torch.Tensor, s: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
    T, D = x.shape
    R = sub.shape[0]
    do = do.contiguous()
    k = _kernels()
    dx = torch.empty_like(x)
    c = torch.empty((T, 3), device=x.device, dtype=torch.float32)
    k["bwd_tok"][(T,)](do, x, loc, sub, s, dx, c, do.stride(0), x.stride(0), sub.stride(0), dx.stride(0), D=D,
                       num_warps=4)
    order = torch.argsort(loc, stable=True)
    # satır sınırları senkronsuz (bincount CUDA'da eşitleme yapar): sıralı satır numaralarında arama
    seg = torch.searchsorted(loc[order], torch.arange(R + 1, device=x.device))
    cnt = seg[1:] - seg[:-1]
    nch = (cnt + CH - 1) // CH                                   # satır başına parça
    cum = torch.cumsum(nch, 0)
    cmax = R + T // CH + 1                                       # parça sayısının üst sınırı (senkronsuz ızgara)
    cid = torch.arange(cmax, device=x.device)
    crow = torch.searchsorted(cum, cid, right=True)
    ok = crow < R
    crow_c = crow.clamp_max(R - 1)
    first = cum[crow_c] - nch[crow_c]
    clo = seg[crow_c] + (cid - first) * CH
    chi = torch.minimum(clo + CH, seg[crow_c + 1])
    crow = torch.where(ok, crow, torch.full_like(crow, -1))
    dsub = torch.zeros(sub.shape, device=x.device, dtype=torch.float32)
    k["bwd_row"][(cmax,)](do, x, c, order, crow, clo, chi, dsub, do.stride(0), x.stride(0), dsub.stride(0), D=D,
                          num_warps=4)
    return dx, dsub.to(sub.dtype)


@private_bwd.register_fake
def _(do, x, loc, sub, s):
    return torch.empty_like(x), torch.empty_like(sub)


def _setup(ctx, inputs, output):
    x, loc, sub = inputs
    ctx.save_for_backward(x, loc, sub, output[1])


def _backward(ctx, do, ds):
    x, loc, sub, s = ctx.saved_tensors
    dx, dsub = private_bwd(do, x, loc, sub, s)
    return dx, None, dsub


private_fwd.register_autograd(_backward, setup_context=_setup)


def private_neuron(x, loc, sub):
    """x (T, d), loc (T,) satır sırası, sub (R, 3d) -> (T, d).  CUDA'da Triton çekirdeği, yoksa reference()."""
    if x.is_cuda:
        return private_fwd(x.contiguous(), loc.contiguous(), sub.contiguous())[0]
    return reference(x, loc, sub)
