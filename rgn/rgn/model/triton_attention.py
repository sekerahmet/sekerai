"""Flash-RGN: çekim dikkatinin birleşik Triton çekirdeği (ileri + geri).  pos rope / none, power 1.

N×N ağırlık matrisi belleğe yazılmaz (dense yolun O(N²) belleği ve 4-5 ayrı çekirdeği yerine tek geçiş).  Softmax'tan
farkı: w ∈ (0, 1] olduğundan satır maksimumu takibi ve yeniden ölçekleme yok:

    s = q̂·k̂ᵀ,  d = max(2 - 2s, 0),  den = 1 + (γ/K) d,  w = den^(-K)
    acc += w·V,  z += Σ_j w,  O = acc / z                       (z geri geçiş için saklanır)

Geri geçiş (D_i = dO_i·O_i, w maskeli):
    dw = (dO·Vᵀ - D) / z
    dV = Σ_i (w/z)ᵀ dO
    ds = 2γ · dw · w / den                       (d > 0 iken; d kırpılınca 0 — dense ile aynı)
    dQ = ds K,   dK = dsᵀ Q
    dγ = -Σ dw · w · d / den
    dK_h = Σ dw · w · (-ln den + (den - 1) / den)     (yalnız öğrenilen k)

Hassasiyet: q·kᵀ varsayılan 'tf32x3' (üç TF32 Tensor Core çarpımı, fp32'ye yakın; yakın eşleşmede 2 - 2s iptal hatası —
doküman 3.5; ModelConfig.rgn_qk_prec), w·V ve gradyan çarpımları 'tf32' (w ∈ [0, 1]).
Ölçüldü (RTX PRO 6000 Blackwell, B 64, H 8, N 512, dh 64, ileri + geri): tf32x3 2,6 ms, tf32 1,5 ms, hepsi tf32x3
7,6 ms; dense (fp32) 19,5 ms; softmax SDPA bf16 0,54 ms.  Tam adım (SS d512, 128 × 512): dense 182 ms -> triton
tf32x3 145 ms / tf32 130 ms, bellek 29,9 -> 20,2 GB; transformer 112 ms.  K = 1, 2, 4, 8 tam sayıları kuvvet zinciriyle (log / exp yok).
"""
import torch
import triton
import triton.language as tl

AV_PREC = "tf32"
# Çekirdek başına (BM, BN, num_warps, num_stages).  fp32 bloklar paylaşılan belleği (sm_120: 99 KB) çabuk doldurur.
# Tarandı (B 64, H 8, N 512, dh 64; tf32x3 ileri + geri 2,57 -> 1,71 ms, tf32 1,52 -> 1,33 ms).
CFG = {"fwd": (64, 32, 4, 2), "kv": (32, 32, 4, 3), "q": (64, 32, 4, 2)}


@triton.jit
def _weight(s, g, K, K_INT: tl.constexpr):
    """-> d, den, w   (g = γ / K)."""
    d = tl.maximum(2.0 - 2.0 * s, 0.0)
    den = 1.0 + g * d
    if K_INT == 1:
        w = 1.0 / den
    elif K_INT == 2:
        r = 1.0 / den
        w = r * r
    elif K_INT == 4:
        r = 1.0 / den
        r = r * r
        w = r * r
    elif K_INT == 8:
        r = 1.0 / den
        r = r * r
        r = r * r
        w = r * r
    else:
        w = tl.exp2(-K * tl.log2(den))
    return d, den, w


@triton.jit
def _fwd_kernel(Q, Kp, V, G, KV, O, Z,
                sqb, sqh, sqn, skb, skh, skn, svb, svh, svn, sob, soh, son,
                H, N,
                D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, K_INT: tl.constexpr,
                QK: tl.constexpr, AV: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + b * sqb + h * sqh + offs_m[:, None] * sqn + offs_d[None, :], mask=offs_m[:, None] < N, other=0.0)
    Kh = tl.load(KV + h)
    g = tl.load(G + h) / Kh
    acc = tl.zeros([BM, D], dtype=tl.float32)
    z = tl.zeros([BM], dtype=tl.float32)
    hi = tl.minimum((pid_m + 1) * BM, N)
    for start_n in range(0, hi, BN):
        offs_n = start_n + tl.arange(0, BN)
        kv_mask = offs_n[:, None] < N
        k = tl.load(Kp + b * skb + h * skh + offs_n[:, None] * skn + offs_d[None, :], mask=kv_mask, other=0.0)
        v = tl.load(V + b * svb + h * svh + offs_n[:, None] * svn + offs_d[None, :], mask=kv_mask, other=0.0)
        s = tl.dot(q, tl.trans(k), input_precision=QK)
        d, den, w = _weight(s, g, Kh, K_INT)
        valid = (offs_m[:, None] >= offs_n[None, :]) & (offs_n[None, :] < N)
        w = tl.where(valid, w, 0.0)
        z += tl.sum(w, 1)
        acc += tl.dot(w, v, input_precision=AV)
    z = tl.maximum(z, 1e-30)
    out = acc / z[:, None]
    tl.store(O + b * sob + h * soh + offs_m[:, None] * son + offs_d[None, :], out, mask=offs_m[:, None] < N)
    tl.store(Z + bh * N + offs_m, z, mask=offs_m < N)


@triton.jit
def _bwd_kv_kernel(Q, Kp, V, G, KV, DO, Z, DL, DK, DV, PG, PK,
                   sqb, sqh, sqn, skb, skh, skn, svb, svh, svn, sdb, sdh, sdn,
                   H, N, NB,
                   D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, K_INT: tl.constexpr, NEED_K: tl.constexpr,
                   QK: tl.constexpr, AV: tl.constexpr):
    """Anahtar bloğu başına: dK, dV ve head parametrelerinin (γ, k) blok kısmi toplamları."""
    pid_n = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    kv_mask = offs_n[:, None] < N
    k = tl.load(Kp + b * skb + h * skh + offs_n[:, None] * skn + offs_d[None, :], mask=kv_mask, other=0.0)
    v = tl.load(V + b * svb + h * svh + offs_n[:, None] * svn + offs_d[None, :], mask=kv_mask, other=0.0)
    Kh = tl.load(KV + h)
    gam = tl.load(G + h)
    g = gam / Kh
    dk = tl.zeros([BN, D], dtype=tl.float32)
    dv = tl.zeros([BN, D], dtype=tl.float32)
    pg = tl.zeros([BN], dtype=tl.float32)
    pk = tl.zeros([BN], dtype=tl.float32)
    lo = (pid_n * BN // BM) * BM
    for start_m in range(lo, N, BM):
        offs_m = start_m + tl.arange(0, BM)
        m_mask = offs_m < N
        q = tl.load(Q + b * sqb + h * sqh + offs_m[:, None] * sqn + offs_d[None, :], mask=m_mask[:, None], other=0.0)
        do = tl.load(DO + b * sdb + h * sdh + offs_m[:, None] * sdn + offs_d[None, :], mask=m_mask[:, None], other=0.0)
        zi = tl.load(Z + bh * N + offs_m, mask=m_mask, other=1.0)
        di = tl.load(DL + bh * N + offs_m, mask=m_mask, other=0.0)
        s = tl.dot(q, tl.trans(k), input_precision=QK)
        d, den, w = _weight(s, g, Kh, K_INT)
        valid = (offs_m[:, None] >= offs_n[None, :]) & m_mask[:, None] & (offs_n[None, :] < N)
        w = tl.where(valid, w, 0.0)
        dp = tl.dot(do, tl.trans(v), input_precision=AV)
        dw = tl.where(valid, (dp - di[:, None]) / zi[:, None], 0.0)
        dv += tl.dot(tl.trans(w / zi[:, None]), do, input_precision=AV)
        t = dw * w / den
        ds = tl.where(d > 0.0, 2.0 * gam * t, 0.0)
        dk += tl.dot(tl.trans(ds), q, input_precision=AV)
        pg += tl.sum(-t * d, 0)
        if NEED_K:
            pk += tl.sum(dw * w * ((den - 1.0) / den - tl.log(den)), 0)
    tl.store(DK + bh * N * D + offs_n[:, None] * D + offs_d[None, :], dk, mask=kv_mask)
    tl.store(DV + bh * N * D + offs_n[:, None] * D + offs_d[None, :], dv, mask=kv_mask)
    tl.store(PG + bh * NB + pid_n, tl.sum(pg, 0))
    if NEED_K:
        tl.store(PK + bh * NB + pid_n, tl.sum(pk, 0))


@triton.jit
def _bwd_q_kernel(Q, Kp, V, G, KV, DO, Z, DL, DQ,
                  sqb, sqh, sqn, skb, skh, skn, svb, svh, svn, sdb, sdh, sdn,
                  H, N,
                  D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, K_INT: tl.constexpr,
                  QK: tl.constexpr, AV: tl.constexpr):
    """Sorgu bloğu başına: dQ."""
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    m_mask = offs_m < N
    q = tl.load(Q + b * sqb + h * sqh + offs_m[:, None] * sqn + offs_d[None, :], mask=m_mask[:, None], other=0.0)
    do = tl.load(DO + b * sdb + h * sdh + offs_m[:, None] * sdn + offs_d[None, :], mask=m_mask[:, None], other=0.0)
    zi = tl.load(Z + bh * N + offs_m, mask=m_mask, other=1.0)
    di = tl.load(DL + bh * N + offs_m, mask=m_mask, other=0.0)
    Kh = tl.load(KV + h)
    gam = tl.load(G + h)
    g = gam / Kh
    dq = tl.zeros([BM, D], dtype=tl.float32)
    hi = tl.minimum((pid_m + 1) * BM, N)
    for start_n in range(0, hi, BN):
        offs_n = start_n + tl.arange(0, BN)
        kv_mask = offs_n[:, None] < N
        k = tl.load(Kp + b * skb + h * skh + offs_n[:, None] * skn + offs_d[None, :], mask=kv_mask, other=0.0)
        v = tl.load(V + b * svb + h * svh + offs_n[:, None] * svn + offs_d[None, :], mask=kv_mask, other=0.0)
        s = tl.dot(q, tl.trans(k), input_precision=QK)
        d, den, w = _weight(s, g, Kh, K_INT)
        valid = (offs_m[:, None] >= offs_n[None, :]) & m_mask[:, None] & (offs_n[None, :] < N)
        w = tl.where(valid, w, 0.0)
        dp = tl.dot(do, tl.trans(v), input_precision=AV)
        dw = tl.where(valid, (dp - di[:, None]) / zi[:, None], 0.0)
        ds = tl.where(d > 0.0, 2.0 * gam * dw * w / den, 0.0)
        dq += tl.dot(ds, k, input_precision=AV)
    tl.store(DQ + bh * N * D + offs_m[:, None] * D + offs_d[None, :], dq, mask=m_mask[:, None])


def _strides(t):
    return t.stride(0), t.stride(1), t.stride(2)


PRECS = ("tf32x3", "tf32", "ieee")       # özel işleme kod olarak geçer (Inductor şema eşlemesi str argümanı tanımıyor)


def _prec(device, qk):
    """Triton yorumlayıcısında (CPU, TRITON_INTERPRET=1) Tensor Core yok: tam fp32."""
    return ("ieee", "ieee") if device.type != "cuda" else (PRECS[qk], AV_PREC)


@torch.library.custom_op("rgn::gravity_fwd", mutates_args=())
def gravity_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gamma: torch.Tensor, kvec: torch.Tensor,
                k_int: int, qk_prec: int) -> tuple[torch.Tensor, torch.Tensor]:
    B, H, N, D = q.shape
    o = torch.empty((B, H, N, D), device=q.device, dtype=torch.float32)
    z = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
    qk, av = _prec(q.device, qk_prec)
    bm, bn, nw, ns = CFG["fwd"]
    _fwd_kernel[(triton.cdiv(N, bm), B * H)](
        q, k, v, gamma, kvec, o, z, *_strides(q), *_strides(k), *_strides(v), *_strides(o), H, N,
        D=D, BM=bm, BN=bn, K_INT=k_int, QK=qk, AV=av, num_warps=nw, num_stages=ns)
    return o, z


@gravity_fwd.register_fake
def _(q, k, v, gamma, kvec, k_int, qk_prec):
    B, H, N, D = q.shape
    return q.new_empty((B, H, N, D), dtype=torch.float32), q.new_empty((B, H, N), dtype=torch.float32)


@torch.library.custom_op("rgn::gravity_bwd", mutates_args=())
def gravity_bwd(do: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, z: torch.Tensor,
                gamma: torch.Tensor, kvec: torch.Tensor, k_int: int, qk_prec: int
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, N, D = q.shape
    do = do.float()
    if do.stride(-1) != 1:
        do = do.contiguous()
    dl = (do * o).sum(-1).contiguous()                          # D_i = dO_i · O_i
    dq = torch.empty((B, H, N, D), device=q.device, dtype=torch.float32)
    dk = torch.empty_like(dq)
    dv = torch.empty_like(dq)
    bm, bn, nw, ns = CFG["kv"]
    nb = triton.cdiv(N, bn)
    pg = torch.empty((B, H, nb), device=q.device, dtype=torch.float32)
    pk = torch.zeros((B, H, nb), device=q.device, dtype=torch.float32)
    qk, av = _prec(q.device, qk_prec)
    common = (*_strides(q), *_strides(k), *_strides(v), *_strides(do))
    _bwd_kv_kernel[(nb, B * H)](
        q, k, v, gamma, kvec, do, z, dl, dk, dv, pg, pk, *common, H, N, nb,
        D=D, BM=bm, BN=bn, K_INT=k_int, NEED_K=k_int == 0, QK=qk, AV=av, num_warps=nw, num_stages=ns)
    bm, bn, nw, ns = CFG["q"]
    _bwd_q_kernel[(triton.cdiv(N, bm), B * H)](
        q, k, v, gamma, kvec, do, z, dl, dq, *common, H, N,
        D=D, BM=bm, BN=bn, K_INT=k_int, QK=qk, AV=av, num_warps=nw, num_stages=ns)
    return dq, dk, dv, pg.sum((0, 2)), pk.sum((0, 2))


@gravity_bwd.register_fake
def _(do, q, k, v, o, z, gamma, kvec, k_int, qk_prec):
    B, H, N, D = q.shape
    e = q.new_empty((B, H, N, D), dtype=torch.float32)
    return e, torch.empty_like(e), torch.empty_like(e), gamma.new_empty((H,)), gamma.new_empty((H,))


def _setup_context(ctx, inputs, output):
    q, k, v, gamma, kvec, k_int, qk_prec = inputs
    o, z = output
    ctx.save_for_backward(q, k, v, o, z, gamma, kvec)
    ctx.k_int, ctx.qk_prec = k_int, qk_prec


def _backward(ctx, do, dz):
    q, k, v, o, z, gamma, kvec = ctx.saved_tensors
    dq, dk, dv, dg, dkv = gravity_bwd(do, q, k, v, o, z, gamma, kvec, ctx.k_int, ctx.qk_prec)
    return dq, dk, dv, dg, (dkv if ctx.k_int == 0 else None), None, None


gravity_fwd.register_autograd(_backward, setup_context=_setup_context)


def rgn_triton(q, k, v, gamma, sharpness=1, qk_prec="tf32x3"):
    """q (B, H, N, dh), k / v (B, Hkv, N, dh) -> O (B, H, N, dh) fp32.  rgn_dense(pos rope / none, power 1) ile aynı
    sonuç.  sharpness: sayı ya da (H,) tensör (öğrenilen k; gradyan alır).  qk_prec: q·kᵀ çarpımı — tf32x3 (fp32'ye
    yakın) | tf32 (hızlı) | ieee (tam fp32, yavaş)."""
    H, D = q.shape[1], q.shape[3]
    assert D in (16, 32, 64, 128), "head boyutu 16 / 32 / 64 / 128"
    if k.shape[1] != H:
        rep = H // k.shape[1]
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    q, k, v = (t.float() if t.stride(-1) == 1 else t.float().contiguous() for t in (q, k, v))
    gamma = gamma.float().contiguous()
    if torch.is_tensor(sharpness):
        kvec, k_int = sharpness.float().contiguous(), 0
    else:
        kvec = torch.full((H,), float(sharpness), device=q.device)
        k_int = int(sharpness) if float(sharpness).is_integer() else -1     # -1: genel üs yolu, gradyan yok
    return gravity_fwd(q, k, v, gamma, kvec, k_int, PRECS.index(qk_prec))[0]
