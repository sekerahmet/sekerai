"""Kelimeye özel nöronun birleşik Triton çekirdeği (ileri + geri).

Token t, satırı r = loc[t] (adımda geçen satırların kopyası `sub` içinde), girdi x_t (d):
    g, u, w = sub[r, 0:d], sub[r, d:2d], sub[r, 2d:3d]
    s1 = g·x, s2 = u·x, a = SiLU(s1)·s2,  çıktı = a·w
Gruplu (groups G > 1): d, G ardışık parçaya bölünür (her biri d/G); parça k kendi iç çarpımlarıyla kendi kapısını
hesaplar ve yalnız kendi parçasına yazar: s1_k = g_k·x_k, s2_k = u_k·x_k, a_k = SiLU(s1_k)·s2_k, çıktı_k = a_k·w_k.
Kelime başına G nöron; satır ve işlem aynı (3d).  G = 1 tek nöron.
Ara (T, 3d) tensör belleğe yazılmaz.
Geri (çıktı gradyanı δ), parça başına:
    da = δ·w,  ds1 = da·s2·SiLU'(s1),  ds2 = da·SiLU(s1)
    dx = ds1·g + ds2·u                                   (token başına çekirdek)
    dg_r = Σ_{t: loc=r} ds1_t x_t,  du_r = Σ ds2_t x_t,  dw_r = Σ a_t δ_t   (token'lar satıra göre sıralı; satır en çok
                                                         CH token'lık parçalara bölünür, parça toplamı atomik — senkronsuz)
Çekirdekte vektörler (GP, BG) bloğu: GP = G'yi kapsayan 2'nin kuvveti, BG = d/G'yi kapsayan; fazlası maskelenir.
CPU'da (testler, değerlendirme) düz PyTorch karşılığı: reference().
"""
import torch
import torch.nn.functional as F


def neuron(w, x, groups=1):
    """w (..., 3, d) satırlar, x (..., d) -> (..., d): grup başına a_k = SiLU(g_k·x_k)(u_k·x_k), çıktı_k = a_k·w_k."""
    s = x.shape
    w = w.view(*s[:-1], 3, groups, s[-1] // groups)
    x = x.view(*s[:-1], groups, s[-1] // groups)
    a = F.silu((w[..., 0, :, :] * x).sum(-1)) * (w[..., 1, :, :] * x).sum(-1)
    return (a.unsqueeze(-1) * w[..., 2, :, :]).reshape(s)


def reference(x, loc, sub, groups=1):
    """x (T, d), loc (T,), sub (R, 3d) -> (T, d).  Düz PyTorch (otomatik türev)."""
    d = x.shape[-1]
    return neuron(F.embedding(loc, sub).view(-1, 3, d).to(x.dtype), x, groups)


_K = {}
CH = 32                     # satır gradyanı: parça başına en çok token (sık satır paralel parçalara bölünür)


def _kernels():
    if _K:
        return _K
    import triton
    import triton.language as tl

    @triton.jit
    def _idx(D: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, BG: tl.constexpr):
        """(GP, BG) blok: eleman konumu o = k·(D/G) + j ve maskesi; grup vektörü kv (GP,) ve maskesi."""
        DG: tl.constexpr = D // G
        kk = tl.arange(0, GP)[:, None]
        jj = tl.arange(0, BG)[None, :]
        kv = tl.arange(0, GP)
        return kk * DG + jj, (kk < G) & (jj < DG), kv, kv < G

    @triton.jit
    def fwd(X, LOC, SUB, OUT, S, sx, ss, so, D: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, BG: tl.constexpr):
        t = tl.program_id(0).to(tl.int64)
        r = tl.load(LOC + t).to(tl.int64)
        o, m, kv, mk = _idx(D, G, GP, BG)
        x = tl.load(X + t * sx + o, mask=m, other=0.0).to(tl.float32)
        b = SUB + r * ss
        s1 = tl.sum(tl.load(b + o, mask=m, other=0.0).to(tl.float32) * x, 1)
        s2 = tl.sum(tl.load(b + D + o, mask=m, other=0.0).to(tl.float32) * x, 1)
        a = s1 * tl.sigmoid(s1) * s2
        w = tl.load(b + 2 * D + o, mask=m, other=0.0).to(tl.float32)
        tl.store(OUT + t * so + o, (a[:, None] * w).to(OUT.dtype.element_ty), mask=m)
        tl.store(S + t * (2 * G) + kv, s1, mask=mk)
        tl.store(S + t * (2 * G) + G + kv, s2, mask=mk)

    @triton.jit
    def bwd_tok(DO, X, LOC, SUB, S, DX, C, sdo, sx, ss, sdx, D: tl.constexpr, G: tl.constexpr, GP: tl.constexpr,
                BG: tl.constexpr):
        t = tl.program_id(0).to(tl.int64)
        r = tl.load(LOC + t).to(tl.int64)
        o, m, kv, mk = _idx(D, G, GP, BG)
        b = SUB + r * ss
        do = tl.load(DO + t * sdo + o, mask=m, other=0.0).to(tl.float32)
        s1 = tl.load(S + t * (2 * G) + kv, mask=mk, other=0.0)
        s2 = tl.load(S + t * (2 * G) + G + kv, mask=mk, other=0.0)
        sg = tl.sigmoid(s1)
        silu = s1 * sg
        da = tl.sum(do * tl.load(b + 2 * D + o, mask=m, other=0.0).to(tl.float32), 1)
        ds1 = da * s2 * sg * (1.0 + s1 * (1.0 - sg))
        ds2 = da * silu
        dx = (ds1[:, None] * tl.load(b + o, mask=m, other=0.0).to(tl.float32)
              + ds2[:, None] * tl.load(b + D + o, mask=m, other=0.0).to(tl.float32))
        tl.store(DX + t * sdx + o, dx.to(DX.dtype.element_ty), mask=m)
        tl.store(C + t * (3 * G) + kv, ds1, mask=mk)
        tl.store(C + t * (3 * G) + G + kv, ds2, mask=mk)
        tl.store(C + t * (3 * G) + 2 * G + kv, silu * s2, mask=mk)

    @triton.jit
    def bwd_row(DO, X, C, ORDER, CROW, CLO, CHI, DSUB, sdo, sx, sds, D: tl.constexpr, G: tl.constexpr,
                GP: tl.constexpr, BG: tl.constexpr):
        """Parça başına (bir satırın en çok CH ardışık token'ı): toplam fp32 DSUB'a atomik eklenir.  Sık satır birçok
        paralel parçaya bölünür (tek programda uzun döngü yok)."""
        c = tl.program_id(0)
        r = tl.load(CROW + c)
        if r >= 0:
            lo = tl.load(CLO + c)
            hi = tl.load(CHI + c)
            o, m, kv, mk = _idx(D, G, GP, BG)
            ag = tl.zeros([GP, BG], dtype=tl.float32)
            au = tl.zeros([GP, BG], dtype=tl.float32)
            aw = tl.zeros([GP, BG], dtype=tl.float32)
            for i in range(lo, hi):
                t = tl.load(ORDER + i).to(tl.int64)
                x = tl.load(X + t * sx + o, mask=m, other=0.0).to(tl.float32)
                cb = C + t * (3 * G)
                ag += tl.load(cb + kv, mask=mk, other=0.0)[:, None] * x
                au += tl.load(cb + G + kv, mask=mk, other=0.0)[:, None] * x
                aw += (tl.load(cb + 2 * G + kv, mask=mk, other=0.0)[:, None]
                       * tl.load(DO + t * sdo + o, mask=m, other=0.0).to(tl.float32))
            b = DSUB + r.to(tl.int64) * sds
            tl.atomic_add(b + o, ag, mask=m)
            tl.atomic_add(b + D + o, au, mask=m)
            tl.atomic_add(b + 2 * D + o, aw, mask=m)

    _K.update(fwd=fwd, bwd_tok=bwd_tok, bwd_row=bwd_row)
    return _K


def _bd(d):
    """Çekirdek blok genişliği: d'yi kapsayan 2'nin kuvveti (d 768 -> 1024, fazlası maskelenir)."""
    return 1 << (d - 1).bit_length()


def _meta(D, groups):
    """Çekirdek sabitleri: D, G, GP (G'yi kapsayan 2'nin kuvveti), BG (D / G'yi kapsayan)."""
    return dict(D=D, G=groups, GP=_bd(groups), BG=_bd(D // groups))


def make_plan(loc, R):
    """Geri geçişin satır planı (senkronsuz; bincount CUDA'da eşitleme yapar): token'lar satıra göre sıralı (order), satır
    en çok CH token'lık parçalara bölünür (crow: parçanın satırı, -1 boş; clo / chi: order içindeki aralık).  Satır
    sırası her katmanda aynı olduğundan mikro adım başına bir kez hesaplanır (trainer)."""
    T = loc.shape[0]
    order = torch.argsort(loc, stable=True)
    seg = torch.searchsorted(loc[order], torch.arange(R + 1, device=loc.device))
    nch = (seg[1:] - seg[:-1] + CH - 1) // CH                     # satır başına parça
    cum = torch.cumsum(nch, 0)
    cid = torch.arange(R + T // CH + 1, device=loc.device)         # parça sayısının üst sınırı (senkronsuz ızgara)
    crow = torch.searchsorted(cum, cid, right=True)
    rc = crow.clamp_max(R - 1)
    clo = seg[rc] + (cid - (cum[rc] - nch[rc])) * CH
    chi = torch.minimum(clo + CH, seg[rc + 1])
    crow = torch.where(crow < R, crow, torch.full_like(crow, -1))
    return order, crow, clo, chi


@torch.library.custom_op("rgn::private_fwd", mutates_args=())
def private_fwd(x: torch.Tensor, loc: torch.Tensor, sub: torch.Tensor, order: torch.Tensor, crow: torch.Tensor,
                clo: torch.Tensor, chi: torch.Tensor, groups: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    T, D = x.shape
    assert sub.shape[1] == 3 * D and D % groups == 0, "satır 3d; d grup sayısına bölünmeli"
    out = torch.empty_like(x)
    s = torch.empty((T, 2 * groups), device=x.device, dtype=torch.float32)
    _kernels()["fwd"][(T,)](x, loc, sub, out, s, x.stride(0), sub.stride(0), out.stride(0), **_meta(D, groups),
                            num_warps=4)
    return out, s


@private_fwd.register_fake
def _(x, loc, sub, order, crow, clo, chi, groups=1):
    return torch.empty_like(x), x.new_empty((x.shape[0], 2 * groups), dtype=torch.float32)


@torch.library.custom_op("rgn::private_bwd", mutates_args=())
def private_bwd(do: torch.Tensor, x: torch.Tensor, loc: torch.Tensor, sub: torch.Tensor, s: torch.Tensor,
                order: torch.Tensor, crow: torch.Tensor, clo: torch.Tensor, chi: torch.Tensor, groups: int = 1
                ) -> tuple[torch.Tensor, torch.Tensor]:
    T, D = x.shape
    do = do.contiguous()
    k = _kernels()
    dx = torch.empty_like(x)
    c = torch.empty((T, 3 * groups), device=x.device, dtype=torch.float32)
    k["bwd_tok"][(T,)](do, x, loc, sub, s, dx, c, do.stride(0), x.stride(0), sub.stride(0), dx.stride(0),
                       **_meta(D, groups), num_warps=4)
    dsub = torch.zeros(sub.shape, device=x.device, dtype=torch.float32)
    k["bwd_row"][(crow.shape[0],)](do, x, c, order, crow, clo, chi, dsub, do.stride(0), x.stride(0), dsub.stride(0),
                                   **_meta(D, groups), num_warps=4)
    return dx, dsub.to(sub.dtype)


@private_bwd.register_fake
def _(do, x, loc, sub, s, order, crow, clo, chi, groups=1):
    return torch.empty_like(x), torch.empty_like(sub)


def _setup(ctx, inputs, output):
    x, loc, sub, order, crow, clo, chi, groups = inputs
    ctx.save_for_backward(x, loc, sub, output[1], order, crow, clo, chi)
    ctx.groups = groups


def _backward(ctx, do, ds):
    x, loc, sub, s, order, crow, clo, chi = ctx.saved_tensors
    dx, dsub = private_bwd(do, x, loc, sub, s, order, crow, clo, chi, ctx.groups)
    return dx, None, dsub, None, None, None, None, None


private_fwd.register_autograd(_backward, setup_context=_setup)


def private_neuron(x, loc, sub, plan=None, groups=1):
    """x (T, d), loc (T,) satır sırası, sub (R, 3d) -> (T, d).  CUDA'da Triton çekirdeği (plan: make_plan(loc, R);
    verilmezse burada hesaplanır), yoksa reference().  groups: kelime başına nöron (d'nin parçaları)."""
    if x.is_cuda:
        loc = loc.contiguous()
        plan = plan if plan is not None else make_plan(loc, sub.shape[0])
        # girdi kopyanın dtype'ında (eğitimde bf16): çekirdek fp32 biriktirir; geri geçiş için saklanan x yarı boyda
        return private_fwd(x.to(sub.dtype).contiguous(), loc, sub.contiguous(), *plan, groups)[0].to(x.dtype)
    return reference(x, loc, sub, groups)
