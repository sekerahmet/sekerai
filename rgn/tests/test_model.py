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
