"""Model: nedensellik, sonlu gradyan, sıra duyarlılığı, parametre sayısı (doküman 1.3), başlangıç doyma istatistiği."""
import itertools

import torch

from _util import tiny
from rgn.config import ModelConfig
from rgn.model import RGNModel

VARIANTS = [dict(attn="rgn", pos="rope", mlp="rational", sharpness=4)] + [dict(attn="rgn", pos=p, mlp=m, power=pw)
            for p, m, pw in itertools.product(("additive", "alibi", "rope", "rope+alibi", "none"),
                                              ("rational", "rglu", "glu", "swiglu"), (1, 2))]
VARIANTS += [dict(attn="softmax", pos="rope", mlp=m, attn_impl="auto") for m in ("rational", "swiglu")]


def test_causal_and_finite_gradients():
    torch.manual_seed(0)
    for v in VARIANTS:
        m = RGNModel(tiny(**v))
        idx = torch.randint(0, 97, (2, 16))
        loss = m(idx, idx)
        loss.backward()
        for n, p in m.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), (v, n)
        logits = m(idx)
        idx2 = idx.clone()
        idx2[:, -1] = (idx2[:, -1] + 1) % 97
        assert torch.allclose(logits[:, :-1], m(idx2)[:, :-1], atol=1e-5), ("nedensellik", v)


def test_local_global_layers():
    """Y-Y-G dizilimi: interleave son katman G; yerel katmanda pencere, genel katmanda konum global_pos; ileri + geri çalışır;
    yerel katman ağırlıkları pencere dışında 0."""
    cfg = tiny(n_layers=3, local_window=8, global_layers=1, global_pos="none", sharpness=4)
    assert cfg.layer_kinds() == "YYG"
    assert tiny(n_layers=8, local_window=8, global_layers=3).layer_kinds() == "YYGYGYYG"
    m = RGNModel(cfg)
    assert [b.attn.window for b in m.blocks] == [8, 8, 0]
    assert [b.attn.use_rope for b in m.blocks] == [True, True, False] and m.use_rope
    idx = torch.randint(0, 97, (2, 32))
    m(idx, idx).backward()
    x = m.embed(idx)
    a = m.blocks[0].attn.weights(m.blocks[0].norm1(x), m.rope(32))
    rel = torch.arange(32)[:, None] - torch.arange(32)[None, :]
    assert (a[..., rel >= 8] == 0).all() and (a[..., (rel >= 0) & (rel < 8)] > 0).all()
    r = RGNModel(tiny(n_layers=3, local_window=8, global_layers=1, mlp="glu", ff_ratio=(1, 3)))
    r(torch.randint(0, 97, (2, 16)), torch.randint(0, 97, (2, 16))).backward()
    assert r.cfg.ff_dim("Y") == 64 and r.cfg.ff_dim("G") == 256         # d 64: 1·4·64/3 = 85 -> 64; 3·4·64/3 = 256
    s = RGNModel(tiny(attn="softmax", mlp="swiglu", attn_impl="auto", n_layers=3, local_window=8, global_layers=1))
    s(idx, idx).backward()
    a = s.blocks[0].attn.weights(s.blocks[0].norm1(s.embed(idx)), s.rope(32))
    assert (a[..., rel >= 8] == 0).all()


def test_integer_sharpness_and_gamma_cap():
    """Tamsayı k: ileri değerler tamsayı, ρ gradyan alır (straight-through); genel katman tavanı sharpness_range_g;
    γ tavanı (1 + 4γ/k)^k <= 2^120."""
    cfg = tiny(n_layers=3, local_window=8, global_layers=1, sharpness_learn=True, sharpness_int=True,
               sharpness_range=(1.0, 8.0), sharpness_range_g=(1.0, 16.0), sharpness_init=4.0)
    m = RGNModel(cfg)
    assert [b.attn.cfg.sharpness_range for b in m.blocks] == [(1.0, 8.0), (1.0, 8.0), (1.0, 16.0)]
    for b in m.blocks:
        b.attn.sharp_rho.data.uniform_(-3, 3)
        k = b.attn.sharpness()
        assert torch.equal(k, k.round())
    idx = torch.randint(0, 97, (2, 16))
    m(idx, idx).backward()
    assert all((b.attn.sharp_rho.grad.abs() > 0).any() for b in m.blocks)
    a = m.blocks[2].attn
    a.gamma_hat.data.fill_(100.0)                                   # γ ~ 1e4: k = 16'da tavan bağlar
    g, k = a.gamma().detach(), a.sharpness().detach()
    assert ((1 + 4 * g.double() / k.double()) ** k.double() <= 2.0 ** 120 * 1.0001).all()


def test_importance_focus_gate():
    """Önem / odak / kapı: sıfır başlatmada çıktı tabanla birebir aynı; SDPA (ek boyutlu) yol açık formülle aynı;
    hepsi gradyan alır."""
    base = dict(attn="softmax", mlp="swiglu", attn_impl="auto", n_kv_heads=4)
    torch.manual_seed(0)
    m0 = RGNModel(tiny(**base))
    torch.manual_seed(0)
    m1 = RGNModel(tiny(**base, attn_importance=True, attn_focus=True, attn_gate="elem"))
    sd = {k: v for k, v in m1.state_dict().items() if "w_aux" not in k}
    m0.load_state_dict(sd)
    idx = torch.randint(0, 97, (2, 16))
    assert torch.allclose(m0(idx), m1(idx), atol=1e-5)
    for b in m1.blocks:
        b.attn.w_aux.weight.data.normal_(0, 0.5)
    m1(idx, idx).backward()
    assert all((b.attn.w_aux.weight.grad.abs() > 0).any() for b in m1.blocks)
    a = m1.blocks[0].attn                                       # SDPA yolu = açık formül (weights) ile aynı çıktı
    x = m1.blocks[0].norm1(m1.embed(idx))
    q, k, v = a.project(x, m1.rope(16))
    w = a.weights(x, m1.rope(16))
    aux = a._aux(x)
    ref = (w @ v) * aux["gate"]
    q2 = q * aux["focus"].unsqueeze(-1)
    q2, k2, v2 = a._augment(q2, k, v, aux["imp"])
    got = a._core(q2, k2, v2)[..., :a.dh] * aux["gate"]
    assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max()


def test_evidence():
    """Kanıt: sıfır başlatmada tabanla aynı; açık hesap = ln((1/n) Σ e^puan); E gradyan alır; yerel pencerede n = W."""
    from rgn.model.attention import softmax_lse
    base = dict(attn="softmax", mlp="swiglu", attn_impl="auto", n_kv_heads=4, n_layers=3, local_window=8, global_layers=1)
    torch.manual_seed(0)
    m0 = RGNModel(tiny(**base))
    torch.manual_seed(0)
    m1 = RGNModel(tiny(**base, attn_evidence=True, attn_gate="head"))
    m0.load_state_dict({k: v for k, v in m1.state_dict().items() if "ev_" not in k and "w_aux" not in k})
    idx = torch.randint(0, 97, (2, 16))
    assert torch.allclose(m0(idx), m1(idx), atol=1e-5)
    m1(idx, idx).backward()
    assert all((b.ev_proj.weight.grad.abs() > 0).any() for b in m1.blocks)
    from rgn.model.attention import evidence
    g = torch.Generator().manual_seed(1)
    q, k, v = (torch.randn(1, 2, 12, 8, generator=g) for _ in range(3))
    for W in (0, 4):
        out, lse = softmax_lse(q, k, v, W, 0.5)
        ev = evidence(q, k, lse, W, 0.5)                          # (1, 12, 4): güç (2 head), odaklanma (2 head)
        s = (q @ k.transpose(-2, -1)) * 0.5
        for i in range(12):
            js = [j for j in range(i + 1) if not W or i - j < W]
            row = s[0, :, i, js]
            assert torch.allclose(lse[0, :, i], torch.logsumexp(row, -1), atol=1e-5), (W, i)
            power = torch.logsumexp(row, -1) - torch.log(torch.tensor(float(len(js))))
            focus = torch.logsumexp(row - row.mean(-1, keepdim=True), -1) - torch.log(torch.tensor(float(len(js))))
            assert torch.allclose(ev[0, i, :2], power, atol=1e-5) and torch.allclose(ev[0, i, 2:], focus, atol=1e-5), (W, i)
            assert (ev[0, i, 2:] >= -1e-6).all()


def test_learnable_sharpness_model():
    m = RGNModel(tiny(sharpness_learn=True, sharpness_init=4.0))
    for blk in m.blocks:
        k = blk.attn.sharpness()
        assert torch.allclose(k, torch.full_like(k, 4.0), atol=1e-5), k
    idx = torch.randint(0, 97, (2, 16))
    m(idx, idx).backward()
    for blk in m.blocks:
        assert (blk.attn.sharp_rho.grad.abs() > 0).all()


def test_gamma_beta_receive_gradient():
    m = RGNModel(tiny(pos="additive"))
    idx = torch.randint(0, 97, (2, 16))
    m(idx, idx).backward()
    for blk in m.blocks:
        assert (blk.attn.gamma_hat.grad.abs() > 0).all()
        assert (blk.attn.beta_hat.grad.abs() > 0).all()


def test_order_sensitivity():
    """Tek katman: önceki iki kelimenin yeri değişince son çıktı değişmeli; konumsuz (pos none) değişmemeli."""
    torch.manual_seed(0)
    s, t = torch.tensor([[5, 9, 20]]), torch.tensor([[9, 5, 20]])
    for pos, expect_change in (("additive", True), ("alibi", True), ("rope", True), ("none", False)):
        m = RGNModel(tiny(n_layers=1, pos=pos))
        diff = (m(s)[0, -1] - m(t)[0, -1]).abs().max().item()
        assert (diff > 1e-6) == expect_change, (pos, diff)


def test_param_count_rgn_ss():
    """Doküman 1.3: RGN-SS (D 512, 8 blok, 8 baş, d_ff 2048, V 50.304, RoPE — β parametresi yok)."""
    with torch.device("meta"):
        m = RGNModel(ModelConfig(d_model=512, n_layers=8, n_heads=8, n_kv_heads=8))
    emb = 50304 * 512
    attn = 512 * 512 * 4 + 8                         # W_Q, W_K, W_V, W_O + γ̂
    mlp = 512 * 2048 + 2048 + 2048 * 512 + 512        # W_1, b_1, W_2, b_2
    block = attn + mlp + 2 * 512                      # + iki RMSNorm
    expected = emb + 8 * block + 512
    assert m.num_params() == expected == 50_950_720, m.num_params()


def test_mlp_parameter_parity():
    """Rasyonel MLP ve SwiGLU ~8D² parametre (karşılaştırmalar aynı bütçeyle)."""
    with torch.device("meta"):
        a = RGNModel(ModelConfig(mlp="rational")).num_params()
        b = RGNModel(ModelConfig(mlp="swiglu")).num_params()
    assert abs(a - b) / a < 0.02, (a, b)


def test_initial_u_not_saturated():
    torch.manual_seed(0)
    m = RGNModel(ModelConfig(vocab_size=1000, max_len=256))
    for blk in m.blocks:
        blk.mlp.track = True
    with torch.no_grad():
        m(torch.randint(0, 1000, (2, 128)))
    for blk in m.blocks:
        assert 0.3 < blk.mlp.stats["u_std"] < 0.6 and blk.mlp.stats["sat_frac"] < 1e-3, blk.mlp.stats
