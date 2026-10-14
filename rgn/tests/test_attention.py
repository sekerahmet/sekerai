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
    for pos in ("additive", "alibi", "none"):
        a = rgn_dense(q, k, v, gamma, beta, pos, 1)
        b = rgn_flex(q, k, v, gamma, beta, pos, 1)
        assert torch.allclose(a, b, atol=2e-4, rtol=1e-3), (pos, (a - b).abs().max().item())


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
