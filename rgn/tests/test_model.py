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
