from pathlib import Path
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


def test_locality():
    """Konum özeti: sıfır başlatmada tabanla aynı; açık formül L_τ = Σ pay e^(−(i−j)/τ); gradyansız (gözlem); E gradyan
    alır; yerel / genel τ'lar; Tom örneği kâğıt üstü değerlerle aynı."""
    from rgn.model.attention import locality
    base = dict(attn="softmax", mlp="swiglu", attn_impl="auto", n_kv_heads=4, n_layers=3, local_window=8, global_layers=1)
    torch.manual_seed(0)
    m0 = RGNModel(tiny(**base))
    torch.manual_seed(0)
    m1 = RGNModel(tiny(**base, attn_locality=True))
    m0.load_state_dict({k: v for k, v in m1.state_dict().items() if "loc_" not in k})
    idx = torch.randint(0, 97, (2, 16))
    assert torch.allclose(m0(idx), m1(idx), atol=1e-5)
    m1(idx, idx).backward()
    assert all((b.loc_proj.weight.grad.abs() > 0).any() for b in m1.blocks)
    assert [b.loc_proj.in_features for b in m1.blocks] == [12, 12, 12]
    g = torch.Generator().manual_seed(1)
    q = torch.randn(1, 2, 10, 8, generator=g, requires_grad=True)
    k = torch.randn(1, 2, 10, 8, generator=g)
    for W in (0, 4):
        L = locality(q, k, W, 0.5, (2.0, 16.0))                      # (1, 10, 4): τ 2 (2 head), τ 16 (2 head)
        assert not L.requires_grad
        s = (q.detach() @ k.transpose(-2, -1)) * 0.5
        for i in range(10):
            js = [j for j in range(i + 1) if not W or i - j < W]
            p = torch.softmax(s[0, :, i, js], -1)
            for ti, t in enumerate((2.0, 16.0)):
                ref = (p * torch.exp(-torch.tensor([float(i - j) for j in js]) / t)).sum(-1)
                assert torch.allclose(L[0, i, ti * 2:(ti + 1) * 2], ref, atol=1e-5), (W, i, t)
    # Tom örneği: "he" (12) -> Tom (0) %80, "It" (6) %20 -> L_4 ≈ 0,084, L_32 ≈ 0,716
    import math
    sc = torch.full((13,), -1e4)
    sc[0], sc[6] = math.log(0.8), math.log(0.2)
    qq = torch.zeros(1, 1, 13, 13)
    kk = torch.zeros(1, 1, 13, 13)
    qq[0, 0, 12] = 1.0
    kk[0, 0, :, :] = torch.diag(sc)                                 # q_12 · k_j = sc[j]
    L = locality(qq, kk, 0, 1.0, (4.0, 32.0))
    assert abs(L[0, 12, 0].item() - 0.084) < 2e-3 and abs(L[0, 12, 1].item() - 0.716) < 2e-3, L[0, 12]


def test_observer():
    """Gözlemci: sıfır başlatmada ana çıktı tabanla aynı; gözlemci kaybı yalnız gözlemci başını eğitir (girdi koparılmış);
    özellikler [0, 1]; obs_feat ana kayıptan gradyan alır."""
    import json as _j
    import tempfile
    from rgn.paths import ROOT
    top = list(range(10))
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=ROOT) as f:
        _j.dump(top, f)
    rel = Path(f.name).name
    try:
        base = dict(attn="softmax", mlp="swiglu", attn_impl="auto", n_kv_heads=4, n_layers=3)
        torch.manual_seed(0)
        m0 = RGNModel(tiny(**base))
        torch.manual_seed(0)
        m1 = RGNModel(tiny(**base, observer_layer=2, observer_vocab=rel))
        m0.load_state_dict({k: v for k, v in m1.state_dict().items() if not k.startswith("obs_")})
        idx = torch.randint(0, 97, (2, 16))
        assert torch.allclose(m0(idx), m1(idx), atol=1e-5)
        main, aux = m1(idx, idx)
        aux.backward(retain_graph=True)
        assert m1.obs_head.weight.grad.abs().sum() > 0
        assert all(p.grad is None or p.grad.abs().sum() == 0 for n, p in m1.named_parameters() if n.startswith("blocks."))
        m1.zero_grad()
        m1.obs_feat.weight.data.normal_(0, 0.1)
        m1(idx, idx)[0].backward()
        assert m1.obs_feat.weight.grad.abs().sum() > 0 and m1.obs_head.weight.grad is None
        _, fe = m1.observe(torch.randn(2, 16, 64))
        assert ((fe >= 0) & (fe <= 1)).all()
        assert int(m1.obs_map[5]) == 5 and int(m1.obs_map[50]) == 10       # sık token kendi sınıfı, diğerleri "diğer"
    finally:
        Path(f.name).unlink()


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


def test_mlp_only_layers():
    """İlk N katman yalnız MLP: o katmanda token'lar arası bilgi yok (yalnız-M model konumdan bağımsız), dikkat
    parametresi yok, teşhis çalışır."""
    from rgn.train.diagnostics import diagnose
    import contextlib
    base = dict(attn="softmax", attn_impl="sdpa", n_kv_heads=4, local_window=4, global_layers=1, n_layers=3)
    m0, m2 = RGNModel(tiny(**base)), RGNModel(tiny(**base, mlp_only_layers=2))
    assert m2.blocks[0].attn is None and m2.blocks[1].attn is None and m2.blocks[2].attn is not None
    assert m2.num_params() < m0.num_params()
    idx = torch.randint(0, 97, (2, 12))
    x = m2.embed(idx)
    y1 = m2.blocks[0](x)
    x2 = x.clone()
    x2[:, :5] = torch.randn_like(x2[:, :5])
    assert torch.allclose(m2.blocks[0](x2)[:, 5:], y1[:, 5:])           # önceki token'lar değişince çıktı aynı
    loss = m2(idx, idx)
    loss.backward()
    d = diagnose(m2, idx, contextlib.nullcontext())
    assert len(d["layers"]) == 1


def test_vocab_neurons():
    """Sözlüğe bölüştürülmüş nöronlar: her token k farklı nöron, her nöron eşit sayıda token; iki katman ayrı dağıtım;
    yalnız token'ın nöronları gradyan alır."""
    from rgn.model.rgn import vocab_neuron_table
    t = vocab_neuron_table(97, 16, 3, seed=0)
    assert t.shape == (97, 3) and all(len(set(r.tolist())) == 3 for r in t)
    load = torch.bincount(t.flatten(), minlength=16)
    assert load.max() - load.min() <= 1                                   # 97·3/16 ≈ 18,2 token / nöron
    assert not torch.equal(t, vocab_neuron_table(97, 16, 3, seed=1))
    m = RGNModel(tiny(attn="softmax", attn_impl="sdpa", n_kv_heads=4, local_window=4, global_layers=1, n_layers=3,
                      mlp_only_layers=2, mlp_vocab_neurons=2, mlp="swiglu", d_ff=16))
    idx = torch.full((1, 6), 5)
    m(idx, idx).backward()
    w = m.blocks[0].mlp.c.weight.grad                                      # (d, ff): yalnız token 5'in 2 nöronu
    used = (w.abs().sum(0) > 0).nonzero().flatten().tolist()
    assert sorted(used) == sorted(m.blocks[0].vocab_neurons[5].tolist())


def test_token_route():
    """Tam rota: tek token'lık batch'te yalnız token'ın head'leri (q satırları) ve nöronları gradyan alır; parametre aynı."""
    base = dict(attn="softmax", attn_impl="sdpa", n_kv_heads=4, local_window=4, global_layers=1, n_layers=2,
                mlp="swiglu", d_ff=16)
    m0, m = RGNModel(tiny(**base)), RGNModel(tiny(**base, route_heads=1, route_neurons=3))
    assert m.num_params() == m0.num_params()
    idx = torch.full((1, 6), 5)
    m(idx, idx).backward()
    for blk in m.blocks:
        h = int(blk.route_h_tbl[5, 0])
        gq = blk.attn.w_q.weight.grad.view(4, 16, -1).abs().sum((1, 2))          # head başına q gradyanı
        assert gq[h] > 0 and all(gq[i] == 0 for i in range(4) if i != h)
        used = (blk.mlp.c.weight.grad.abs().sum(0) > 0).nonzero().flatten().tolist()
        assert sorted(used) == sorted(blk.route_n_tbl[5].tolist())


def test_private_neurons():
    """Kelimeye özel nöronlar: başlangıçta taban ile aynı çıktı (d = 0); tek kelimelik batch'te yalnız o kelimenin satırı
    gradyan alır; özel tablo Muon'a girmez."""
    import json as _j
    import tempfile
    from rgn.paths import ROOT
    from rgn.train.optim import param_groups
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=ROOT) as f:
        _j.dump({"tokens": [3, 5, 9]}, f)
    rel = Path(f.name).name
    try:
        base = dict(attn="softmax", attn_impl="sdpa", n_kv_heads=4, mlp="swiglu", n_layers=2)
        torch.manual_seed(0)
        m0 = RGNModel(tiny(**base))
        torch.manual_seed(0)
        m = RGNModel(tiny(**base, private_neurons=1, private_vocab=rel))
        m0.load_state_dict({k: v for k, v in m.state_dict().items() if "private" not in k and k != "priv_map"})
        idx = torch.tensor([[5, 5, 5, 5]])
        assert torch.allclose(m0(idx), m(idx), atol=1e-5)
        m.blocks[0].private.data.normal_(0, 0.1)
        m(idx, idx).backward()
        g = m.blocks[0].private.grad.abs().sum(1)
        assert g[2] > 0 and float(g.sum() - g[2]) == 0                     # token 5 -> satır 2
        mu, de, nd = param_groups(m, True)                                  # özel tablo AdamW / Muon'a girmez
        assert not any(p is m.blocks[0].private for p in mu + de + nd)
        from rgn.config import TrainConfig
        from rgn.train.optim import build_optimizer
        opt = build_optimizer(m, TrainConfig())
        before = m.blocks[0].private.detach().clone()
        m.blocks[0].private.grad = torch.ones_like(before)                  # bütün satırlarda gradyan olsa da
        opt.row_adam.set_rows(torch.tensor([2]))                            # yalnız bu adımda geçen satır güncellenir
        opt.row_adam.step()
        ch = (m.blocks[0].private.detach() - before).abs().sum(1)
        assert ch[2] > 0 and float(ch.sum() - ch[2]) == 0
    finally:
        Path(f.name).unlink()
