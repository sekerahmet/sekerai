"""Model: nedensellik, sonlu gradyan, YYG düzeni, parametre sayısı, kapı, kelimeye özel nöron (tam tablo / satır kopyası /
çekirdek eşdeğerliği, satır planı)."""
import torch

from _util import private_vocab, require_cuda, tiny
from rgn.config import ModelConfig
from rgn.model import RGNModel


def _variants(rel):
    return [dict(), dict(attn_gate="head"), dict(attn_gate="elem"), dict(local_window=0),
            dict(private_neurons=1, private_vocab=rel), dict(private_neurons=1, private_vocab=rel, shared_mlp=False),
            dict(shared_mlp=False), dict(private_value=1, private_vocab=rel),
            dict(private_neurons=1, private_value=1, private_vocab=rel)]


def test_causal_and_finite_gradients():
    torch.manual_seed(0)
    with private_vocab(range(1, 40)) as rel:
        for v in _variants(rel):
            m = RGNModel(tiny(**v))
            for _, _, p in m.private_tables():
                p.data.normal_(0, 0.1)
            idx = torch.randint(0, 97, (2, 16))
            m(idx, idx).backward()
            for n, p in m.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(), (v, n)
            idx2 = idx.clone()
            idx2[:, -1] = (idx2[:, -1] + 1) % 97
            assert torch.allclose(m(idx)[:, :-1], m(idx2)[:, :-1], atol=1e-5), ("nedensellik", v)


def test_layer_kinds_and_window():
    """YYG dizilimi; yerel katmanda pencere dışı pay 0, genel katman tam causal."""
    assert tiny(n_layers=8, global_layers=3).layer_kinds() == "YYGYGYYG"
    assert tiny(local_window=0).layer_kinds() == "GGG"
    m = RGNModel(tiny())
    assert [b.attn.window for b in m.blocks] == [8, 8, 0]
    idx = torch.randint(0, 97, (2, 32))
    x = m.embed(idx)
    rel = torch.arange(32)[:, None] - torch.arange(32)[None, :]
    a = m.blocks[0].attn.weights(m.blocks[0].norm1(x), m.rope(32))
    assert (a[..., rel >= 8] == 0).all() and (a[..., (rel >= 0) & (rel < 8)] > 0).all()
    g = m.blocks[2].attn.weights(m.blocks[2].norm1(x), m.rope(32))
    assert (g[..., rel >= 0] > 0).all() and (g[..., rel < 0] == 0).all()


def test_gqa_global_only():
    """n_kv_heads_global: yalnız genel (G) katmanlarda GQA; yerel katmanlar n_kv_heads."""
    m = RGNModel(tiny(n_heads=4, n_kv_heads=4, n_kv_heads_global=1))
    assert [b.attn.Hkv for b in m.blocks] == [4, 4, 1] and m.cfg.layer_kinds() == "YYG"
    idx = torch.randint(0, 97, (2, 16))
    m(idx, idx).backward()
    assert all(p.grad is not None for p in m.parameters())


def test_wide_attention():
    """head_dim: dikkat iç boyutu d'den bağımsız (16 head × 16 = 256 > d 64); dikkat parametresi buna göre büyür."""
    m0, m = RGNModel(tiny()), RGNModel(tiny(n_heads=16, n_kv_heads=16, head_dim=16))
    a = m.blocks[0].attn
    assert a.w_q.weight.shape == (256, 64) and a.w_kv.weight.shape == (512, 64) and a.w_o.weight.shape == (64, 256)
    assert m.num_params() > m0.num_params()
    idx = torch.randint(0, 97, (2, 16))
    m(idx, idx).backward()
    assert all(p.grad is not None for p in m.parameters())


def test_order_sensitivity():
    torch.manual_seed(0)
    m = RGNModel(tiny(n_layers=1, global_layers=1, local_window=0))
    s, t = torch.tensor([[5, 9, 20]]), torch.tensor([[9, 5, 20]])
    assert (m(s)[0, -1] - m(t)[0, -1]).abs().max().item() > 1e-6


def test_param_count_ss():
    """SS tabanı (d 512, 8 katman, SwiGLU d_ff 1344, V 50.304, paylaşımlı gömme)."""
    with torch.device("meta"):
        m = RGNModel(ModelConfig())
    d, ff = 512, 1344
    attn = 4 * d * d + 2 * 64                                  # W_Q, W_K, W_V, W_O + QK-norm
    mlp = d * 2 * ff + 2 * ff + ff * d + d                     # W_1, b_1, W_2, b_2
    expected = 50304 * d + 8 * (attn + mlp + 2 * d) + d
    assert m.num_params() == expected, (m.num_params(), expected)


def test_gate_starts_as_identity():
    """Kapı sıfır başlatmada 1: çıktı kapısız modelle aynı; kapı gradyan alır."""
    torch.manual_seed(0)
    m0 = RGNModel(tiny())
    torch.manual_seed(0)
    m1 = RGNModel(tiny(attn_gate="head"))
    m0.load_state_dict({k: v for k, v in m1.state_dict().items() if "w_aux" not in k})
    idx = torch.randint(0, 97, (2, 12))
    assert torch.allclose(m0(idx), m1(idx), atol=1e-6)
    m1(idx, idx).backward()
    assert m1.blocks[0].attn.w_aux.weight.grad.abs().sum() > 0


def test_private_neurons():
    """Başlangıçta özel nöronsuz modelle aynı çıktı (w = 0); tek kelimelik batch'te yalnız o kelimenin satırı gradyan alır;
    bileşen dökümü (parts) akışı birebir verir."""
    with private_vocab([3, 5, 9]) as rel:
        torch.manual_seed(0)
        m0 = RGNModel(tiny())
        torch.manual_seed(0)
        m = RGNModel(tiny(private_neurons=1, private_vocab=rel))
        m0.load_state_dict({k: v for k, v in m.state_dict().items() if "private" not in k and k != "priv_map"})
        idx = torch.tensor([[5, 5, 5, 5]])
        assert torch.allclose(m0(idx), m(idx), atol=1e-5)
        for b in m.private_blocks():
            b.private.data.normal_(0, 0.1)
        m(idx, idx).backward()
        g = m.blocks[0].private.grad.abs().sum(1)
        assert g[2] > 0 and float(g.sum() - g[2]) == 0                    # token 5 -> satır 2
        with torch.no_grad():
            parts = []
            h = m.hidden(idx, parts=parts)
            x = parts[0]["embed"] + sum(d["attn"] + d["mlp"] + d["private"] for d in parts[1:])
            assert torch.allclose(m.norm_f(x), h, atol=1e-5)
        m2 = RGNModel(tiny(private_neurons=1, private_vocab=rel, shared_mlp=False))   # ortak MLP yok
        assert all(b.mlp is None for b in m2.blocks)
        assert m2.num_params() == m.num_params() - sum(p.numel() for b in m.blocks for p in b.mlp.parameters())
        with torch.no_grad():
            parts = []
            h = m2.hidden(idx, parts=parts)
            x = parts[0]["embed"] + sum(d["attn"] + d["private"] for d in parts[1:])
            assert torch.allclose(m2.norm_f(x), h, atol=1e-5)


def test_private_row_copy_matches_full():
    """Eğitim yolu (yalnız geçen satırların kopyası) tam tablo yoluyla aynı kaybı ve aynı satır gradyanlarını verir
    (nöron, V vektörü, ikisi birlikte)."""
    with private_vocab(range(1, 40)) as rel:
        for kw in (dict(private_neurons=1), dict(private_value=1), dict(private_neurons=1, private_value=1),
                   dict(private_neurons=2), dict(ple_dim=8)):
            torch.manual_seed(0)
            m = RGNModel(tiny(private_vocab=rel, **kw))
            tables = m.private_tables()
            for _, _, p in tables:
                p.data.normal_(0, 0.1)
            idx = torch.randint(0, 97, (2, 12))
            l_full = m(idx, idx)
            l_full.backward()
            uniq = torch.unique(m.priv_map[idx])
            sub = [p.detach().index_select(0, uniq).requires_grad_() for _, _, p in tables]
            l_sub = m(idx, idx, priv=(torch.searchsorted(uniq, m.priv_map[idx]), sub, None))
            l_sub.backward()
            assert torch.allclose(l_full, l_sub, atol=1e-6), kw
            for (_, _, p), s in zip(tables, sub):
                assert torch.allclose(p.grad.index_select(0, uniq), s.grad, atol=1e-6), kw


def test_private_value_gated_layers():
    """Seçili katmanlarda V vektörü + kapı: kapı başlangıçta 1 (V vektörü 0 iken tabanla aynı), yalnız seçili
    katmanların tablosu var; kapı gradyan alır; satır kopyası yolu tam tabloyla aynı."""
    with private_vocab(range(1, 40)) as rel:
        torch.manual_seed(0)
        m0 = RGNModel(tiny())
        torch.manual_seed(0)
        m = RGNModel(tiny(private_value=1, private_vocab=rel, private_value_layers=(1, 3), private_value_gate=True))
        assert [b.private_v is not None for b in m.blocks] == [True, False, True]
        assert [b.vgate is not None for b in m.blocks] == [True, False, True]
        m0.load_state_dict({k: v for k, v in m.state_dict().items()
                            if "private" not in k and "vgate" not in k and k != "priv_map"})
        idx = torch.randint(0, 97, (2, 12))
        assert torch.allclose(m0(idx), m(idx), atol=1e-6)
        for _, _, p in m.private_tables():
            p.data.normal_(0, 0.1)
        l_full = m(idx, idx)
        l_full.backward()
        assert m.blocks[0].vgate.weight.grad.abs().sum() > 0
        uniq = torch.unique(m.priv_map[idx])
        sub = [p.detach().index_select(0, uniq).requires_grad_() for _, _, p in m.private_tables()]
        assert torch.allclose(l_full, m(idx, idx, priv=(torch.searchsorted(uniq, m.priv_map[idx]), sub, None)), atol=1e-6)


def test_private_top():
    """private_top: yalnız en sık K kelimenin satırı var; diğerleri ortak satır 0."""
    import json
    from _util import ROOT
    with private_vocab([3, 5, 9, 11]) as rel:
        f = ROOT / rel
        d = json.loads(f.read_text(encoding="utf-8"))
        d["counts"] = [10, 500, 70, 500]
        f.write_text(json.dumps(d), encoding="utf-8")
        m = RGNModel(tiny(private_neurons=1, private_vocab=rel, private_top=2))
        assert m.blocks[0].private.shape[0] == 3                           # 2 kelime + ortak
        assert m.priv_map[5] == 1 and m.priv_map[11] == 2 and m.priv_map[3] == 0 and m.priv_map[9] == 0
        m = RGNModel(tiny(private_neurons=1, private_vocab=rel, private_skip=1, private_top=3))   # 2. ve 3. sıradakiler
        assert m.blocks[0].private.shape[0] == 3
        assert m.priv_map[9] == 1 and m.priv_map[11] == 2 and m.priv_map[5] == 0 and m.priv_map[3] == 0


def test_private_value():
    """V vektörü: başlangıçta (0) tabanla aynı çıktı; tek kelimelik batch'te yalnız o kelimenin satırı gradyan alır."""
    with private_vocab([3, 5, 9]) as rel:
        torch.manual_seed(0)
        m0 = RGNModel(tiny())
        torch.manual_seed(0)
        m = RGNModel(tiny(private_value=1, private_vocab=rel))
        assert all(b.private is None and b.private_v.shape == (4, m.cfg.n_kv_heads * 16) for b in m.blocks)
        m0.load_state_dict({k: v for k, v in m.state_dict().items() if "private" not in k and k != "priv_map"})
        idx = torch.tensor([[5, 3, 5, 9]])
        assert torch.allclose(m0(idx), m(idx), atol=1e-6)
        m(idx[:, :1].repeat(1, 4), idx).backward()
        g = m.blocks[1].private_v.grad.abs().sum(1)
        assert g[2] > 0 and float(g.sum() - g[2]) == 0                     # token 5 -> satır 2


def test_make_plan_covers_tokens():
    """Satır planı: her token tam bir kez, her parça tek satırın en çok CH token'ı; boş satırlar ve çok sık satır dahil."""
    from rgn.model.private_kernel import CH, make_plan
    g = torch.Generator().manual_seed(0)
    for trial in range(50):
        R = int(torch.randint(1, 40, (1,), generator=g))
        T = int(torch.randint(1, 600, (1,), generator=g))
        loc = torch.randint(0, R, (T,), generator=g)
        if trial % 3 == 0:
            loc[:T // 2] = 0                                              # çok sık satır
        order, crow, clo, chi = make_plan(loc, R)
        seen = torch.zeros(T, dtype=torch.long)
        for r, lo, hi in zip(crow.tolist(), clo.tolist(), chi.tolist()):
            if r < 0:
                continue
            assert 0 < hi - lo <= CH
            toks = order[lo:hi]
            assert (loc[toks] == r).all()
            seen[toks] += 1
        assert (seen == 1).all(), trial


def test_private_kernel_cuda():
    """Birleşik Triton çekirdeği (CUDA): ileri ve geri, düz PyTorch karşılığıyla aynı (sık satırlar dahil; d 2'nin kuvveti
    olmayan genişlik dahil; gruplu nöron: G 2 / 4, parça 2'nin kuvveti olmayan dahil)."""
    require_cuda()
    from rgn.model.private_kernel import make_plan, private_fwd, reference
    torch.manual_seed(0)
    for D, G in ((512, 1), (768, 1), (512, 4), (768, 4), (512, 2)):
        T, R = 4096, 300
        loc = torch.randint(0, R, (T,), device="cuda")
        loc[:1500] = 7                                                 # çok sık bir satır ("the" gibi)
        x = torch.randn(T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        sub = (torch.randn(R, 3 * D, device="cuda") * 0.05).to(torch.bfloat16).requires_grad_()
        out = private_fwd(x, loc, sub, *make_plan(loc, R), G)[0]
        g = torch.randn_like(out)
        (out.float() * g.float()).sum().backward()
        dx, ds = x.grad.clone(), sub.grad.clone()
        xr, sr = x.detach().float().requires_grad_(), sub.detach().float().requires_grad_()
        ref = reference(xr, loc, sr, G)
        (ref * g.float()).sum().backward()
        rel = lambda a, b: float((a.float() - b).norm() / b.norm().clamp_min(1e-12))  # noqa: E731
        assert rel(out, ref.detach()) < 2e-2 and rel(dx, xr.grad) < 2e-2 and rel(ds, sr.grad) < 2e-2, (
            D, G, rel(out, ref.detach()), rel(dx, xr.grad), rel(ds, sr.grad))


def test_private_groups_reference():
    """Gruplu nöron (CPU): G 1 eski tek nöron formülü; G 4 parça başına ayrı kapı (açık döngüyle aynı)."""
    from rgn.model.private_kernel import reference
    torch.manual_seed(0)
    T, D, R = 10, 16, 5
    x, loc, sub = torch.randn(T, D), torch.randint(0, R, (T,)), torch.randn(R, 3 * D)
    w = sub[loc].view(T, 3, D)
    one = torch.nn.functional.silu((w[:, 0] * x).sum(-1)) * (w[:, 1] * x).sum(-1)
    assert torch.allclose(reference(x, loc, sub), one[:, None] * w[:, 2], atol=1e-6)
    out = reference(x, loc, sub, 4)
    for k in range(4):
        s = slice(4 * k, 4 * k + 4)
        a = torch.nn.functional.silu((w[:, 0, s] * x[:, s]).sum(-1)) * (w[:, 1, s] * x[:, s]).sum(-1)
        assert torch.allclose(out[:, s], a[:, None] * w[:, 2, s], atol=1e-6)


def test_kv_cache_matches_full():
    """Önbellekli üretim = önbelleksiz: her adımın logit'i tam ileri geçişle aynı; yerel pencere döner tamponu (istem +
    yeni > pencere), genel tampon, GQA, kapı, kelimeye özel nöron ve V vektörü."""
    from rgn.model.decode import StaticCache
    with private_vocab(range(1, 60)) as rel:
        for kw in (dict(), dict(local_window=0), dict(n_heads=4, n_kv_heads=1), dict(attn_gate="head"),
                   dict(private_neurons=1, private_vocab=rel), dict(private_value=1, private_value_gate=True,
                                                                     private_value_std=0.05, private_vocab=rel),
                   dict(private_neurons=1, private_vocab=rel, shared_mlp=False), dict(n_heads=4, n_kv_heads=4, n_kv_heads_global=1),
            dict(n_kv_heads_global=1, private_value=1, private_value_std=0.05, private_vocab=rel),
            dict(ple_dim=8, private_vocab=rel), dict(ple_dim=8, private_neurons=1, private_vocab=rel),
            dict(private_neurons=1, private_groups=4, private_vocab=rel),
            dict(y_windows=[4, 8]), dict(y_windows=[4, 8], n_heads=4, n_kv_heads=1)):
            torch.manual_seed(0)
            m = RGNModel(tiny(**kw)).eval()
            for _, _, p in m.private_tables():
                p.data.normal_(0, 0.1)
            idx = torch.randint(0, 97, (2, 11))
            cache = StaticCache(m, 2, positions=14)
            logits = cache.prefill(idx)
            seq = idx
            for i in range(14):
                with torch.no_grad():
                    ref = m(seq)[:, -1].float()
                assert torch.allclose(logits, ref, atol=1e-4), (kw, i, (logits - ref).abs().max().item())
                nxt = logits.argmax(-1)
                seq = torch.cat([seq, nxt[:, None]], 1)
                logits = cache.step(nxt).clone()
            assert torch.equal(m.generate(idx, 9), m.generate(idx, 9, cache=False)), kw


def test_prefill_padding():
    """Dolgulu prefill (istem FLEX_BLOCK'un katına sağdan dolgulanır) = dolgusuz tam ileri geçiş: son logit ve sonraki
    adımlar; istem boyu blok katı olan, olmayan, pencereden kısa / uzun; GQA ve özel nöron."""
    from rgn.model.decode import StaticCache
    with private_vocab(range(1, 60)) as rel:
        for kw in (dict(n_heads=4, n_kv_heads=2), dict(private_neurons=1, private_vocab=rel)):
            torch.manual_seed(0)
            m = RGNModel(tiny(max_len=256, **kw)).eval()
            for T in (5, 127, 128, 130):
                idx = torch.randint(0, 97, (2, T))
                cache = StaticCache(m, 2, positions=4)
                logits, seq = cache.prefill(idx), idx
                for _ in range(3):
                    with torch.no_grad():
                        ref = m(seq)[:, -1].float()
                    assert torch.allclose(logits, ref, atol=1e-4), (kw, T, (logits - ref).abs().max().item())
                    nxt = logits.argmax(-1)
                    seq = torch.cat([seq, nxt[:, None]], 1)
                    logits = cache.step(nxt).clone()


def test_kv_cache_cuda():
    """CUDA'da önbellekli üretim (torch.compile + CUDA graph, tampon havuzu) önbelleksizle aynı greedy token'ları verir
    (fp32; özel nöronlu YYG ve GQA); aynı tamponlarla ikinci çağrı da aynı."""
    require_cuda()
    with private_vocab(range(1, 60)) as rel:
        for kw in (dict(private_neurons=1, private_vocab=rel), dict(n_heads=4, n_kv_heads=1)):
            torch.manual_seed(0)
            m = RGNModel(tiny(**kw)).cuda().eval()
            for _, _, p in m.private_tables():
                p.data.normal_(0, 0.1)
            for seed in (0, 1):                                            # ikinci tur: havuzdaki graph tekrar kullanılır
                idx = torch.randint(0, 97, (3, 20), generator=torch.Generator().manual_seed(seed)).cuda()
                assert torch.equal(m.generate(idx, 24), m.generate(idx, 24, cache=False)), (kw, seed)
