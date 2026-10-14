"""Çekim dikkati: formül, satır toplamı, geri geçiş (doküman 3.4) ve flex = dense (CUDA)."""
import torch
import torch.nn.functional as F

from _util import require_cuda
from rgn.model.attention import denominator, rgn_dense, rgn_flex
from rgn.model.position import geometric


def _inputs(B=2, H=4, Hkv=2, N=12, dh=8, dtype=torch.float64, device="cpu", seed=0):
    g = torch.Generator().manual_seed(seed)
    q = F.normalize(torch.randn(B, H, N, dh, generator=g, dtype=dtype), dim=-1).to(device)
    k = F.normalize(torch.randn(B, Hkv, N, dh, generator=g, dtype=dtype), dim=-1).to(device)
    v = torch.randn(B, Hkv, N, dh, generator=g, dtype=dtype).to(device)
    gamma = geometric(4.0, 64.0, H).to(dtype).to(device)
    beta = geometric(1.0, 1 / 256, H).to(dtype).to(device)
    return q, k, v, gamma, beta


def test_formula_matches_explicit_loop():
    """Dense sonuç, kullanıcı formülünün doğrudan döngüyle hesabına eşit: w = 1/(1 + γ||q-k||² + β(i-j))."""
    q, k, v, gamma, beta = _inputs()
    out = rgn_dense(q, k, v, gamma, beta, "additive", 1)
    B, H, N, _ = q.shape
    rep = H // k.shape[1]
    for b in range(B):
        for h in range(H):
            for i in range(N):
                w = torch.stack([1 / (1 + gamma[h] * (q[b, h, i] - k[b, h // rep, j]).pow(2).sum() + beta[h] * (i - j))
                                 for j in range(i + 1)])
                o = (w[:, None] * v[b, h // rep, :i + 1]).sum(0) / w.sum()
                assert torch.allclose(out[b, h, i], o, atol=1e-10)


def test_sharpness_formula_and_k1_identity():
    """k = 1 eski formülle bit düzeyinde aynı; k = 4 açık formül (1 + γ d / 4)^-4 ile aynı; k büyüdükçe e^(-γd)'ye yaklaşır."""
    q, k, v, gamma, beta = _inputs()
    a1 = rgn_dense(q, k, v, gamma, beta, "rope", 1, 1, return_weights=True)[1]
    a0 = rgn_dense(q, k, v, gamma, beta, "rope", 1, return_weights=True)[1]
    assert torch.equal(a0, a1)
    rep = q.shape[1] // k.shape[1]
    kk = k.repeat_interleave(rep, 1)
    d = (2 - 2 * q @ kk.transpose(-2, -1)).clamp_min(0)
    small = geometric(0.5, 2.0, gamma.numel()).to(gamma.dtype)          # yakınsama (göreli hata ~ (γd)² / 2k) için
    for K, g, ref_fn in ((4, gamma, lambda g: (1 + g * d / 4) ** -4), (1024, small, lambda g: torch.exp(-g * d))):
        w = ref_fn(g.view(1, -1, 1, 1)).tril()
        ref = w / w.sum(-1, keepdim=True)
        got = rgn_dense(q, k, v, g, beta, "rope", 1, K, return_weights=True)[1]
        tol = 1e-10 if K == 4 else 1e-2
        assert torch.allclose(got, ref, atol=tol), (K, (got - ref).abs().max().item())


def test_learnable_sharpness_per_head():
    """Head başına tensör k: tüm head'ler 4 iken sabit k = 4 ile aynı; farklı k'ler head head açık formülle aynı; ρ gradyan alır."""
    q, k, v, gamma, beta = _inputs()
    four = torch.full((q.shape[1],), 4.0, dtype=q.dtype)
    assert torch.allclose(rgn_dense(q, k, v, gamma, beta, "rope", 1, four), rgn_dense(q, k, v, gamma, beta, "rope", 1, 4),
                          atol=1e-12)
    ks = torch.tensor([1.0, 2.5, 4.0, 7.5], dtype=q.dtype)
    a = rgn_dense(q, k, v, gamma, beta, "rope", 1, ks, return_weights=True)[1]
    for h in range(4):
        b = rgn_dense(q[:, h:h + 1], k[:, h // 2:h // 2 + 1], v[:, h // 2:h // 2 + 1], gamma[h:h + 1], beta[h:h + 1],
                      "rope", 1, float(ks[h]), return_weights=True)[1]
        assert torch.allclose(a[:, h:h + 1], b, atol=1e-12), h


def test_weights_sum_to_one_and_causal():
    q, k, v, gamma, beta = _inputs()
    for pos in ("additive", "alibi", "rope", "none"):
        _, a = rgn_dense(q, k, v, gamma, beta, pos, 1, return_weights=True)
        assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)))
        lower = torch.ones(a.shape[-1], a.shape[-1], dtype=torch.bool).tril()
        assert (a[..., ~lower] == 0).all() and (a[..., lower] > 0).all()


def test_denominator_at_least_one():
    s = torch.linspace(-1, 1, 101)
    for pos in ("additive", "alibi", "none"):
        for power in (1, 2):
            den = denominator(s[:, None], torch.tensor(512.0), torch.tensor(0.5), torch.arange(50.0)[None], pos, power)
            assert (den >= 1).all()


def test_backward_formulas_match_autograd():
    """Elle türetilmiş geri geçiş (toplamsal β): D_i = dO_i·O_i, dw = (dO·V - D)/Z, ds = 2γ w² dw,
    dγ = -Σ d w² dw, dβ = -Σ k w² dw, dV = Σ (w/Z) dO."""
    N, dh = 9, 8
    torch.manual_seed(1)
    q = F.normalize(torch.randn(N, dh, dtype=torch.float64), dim=-1).requires_grad_()
    k = F.normalize(torch.randn(N, dh, dtype=torch.float64), dim=-1).requires_grad_()
    v = torch.randn(N, dh, dtype=torch.float64, requires_grad=True)
    gamma = torch.tensor([7.0], dtype=torch.float64, requires_grad=True)
    beta = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    O = rgn_dense(q[None, None], k[None, None], v[None, None], gamma, beta, "additive", 1)[0, 0]
    G = torch.randn_like(O)
    (O * G).sum().backward()
    with torch.no_grad():
        idx = torch.arange(N)
        rel = (idx[:, None] - idx[None, :]).double()
        mask = rel >= 0
        s = q @ k.T
        d = 2 - 2 * s
        w = (1 / (1 + gamma * d + beta * rel.clamp_min(0))) * mask
        Z = w.sum(-1, keepdim=True)
        D = (G * O).sum(-1, keepdim=True)
        dw = ((G @ v.T) - D) / Z * mask
        ds = 2 * gamma * w ** 2 * dw
        checks = {"dQ": (ds @ k, q.grad), "dK": (ds.T @ q, k.grad), "dV": ((w / Z).T @ G, v.grad),
                  "dγ": (-(d * w ** 2 * dw).sum(), gamma.grad[0]), "dβ": (-(rel.clamp_min(0) * w ** 2 * dw).sum(), beta.grad[0])}
    for name, (manual, auto) in checks.items():
        err = ((manual - auto).abs().max() / auto.abs().max()).item()
        assert err < 1e-10, (name, err)


def test_flex_matches_dense_cuda():
    require_cuda()
    q, k, v, gamma, beta = _inputs(B=2, H=4, Hkv=2, N=256, dh=64, dtype=torch.float32, device="cuda")
    for pos, K in (("additive", 1), ("alibi", 1), ("none", 1), ("none", 4), ("additive", 2)):
        a = rgn_dense(q, k, v, gamma, beta, pos, 1, K)
        b = rgn_flex(q, k, v, gamma, beta, pos, 1, K)
        assert torch.allclose(a, b, atol=2e-4, rtol=1e-3), (pos, K, (a - b).abs().max().item())


def test_flex_gradients_reach_gamma_beta_cuda():
    """flex geri geçişi γ / β gradyanını maskeden bağımsız toplar: maskeli konumda payda 0 olursa NaN (β = 1 olan head,
    çarpımsal modda j = i + 1).  Gradyanlar sonlu ve dense ile aynı olmalı."""
    require_cuda()
    q, k, v, gamma, beta = _inputs(N=256, dh=64, dtype=torch.float32, device="cuda")
    assert beta.max().item() == 1.0
    for t in (q, k, v, gamma, beta):
        t.requires_grad_()
    for pos in ("additive", "alibi"):
        rgn_flex(q, k, v, gamma, beta, pos, 1).sum().backward()
        grads = [t.grad.clone() for t in (q, k, v, gamma, beta)]
        for t in (q, k, v, gamma, beta):
            t.grad = None
        rgn_dense(q, k, v, gamma, beta, pos, 1).sum().backward()
        for name, gf, t in zip("qkvγβ", grads, (q, k, v, gamma, beta)):
            assert torch.isfinite(gf).all(), (pos, name)
            assert torch.allclose(gf, t.grad, rtol=1e-2, atol=1e-4), (pos, name, (gf - t.grad).abs().max().item())
            t.grad = None
