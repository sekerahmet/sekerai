# -*- coding: utf-8 -*-
"""tests_y -- model_y'nin kapilari: model (BlockModel ve ayarlari) ve genel egitim.  CPU, saniyeler.
Egitim verisi testin icinde uretilir (synthetic_data): ag, Drive ve egitim klasoru gerekmez.

    python tests_y.py
"""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
torch.set_num_threads(1)

import train_y as TR  # noqa: E402
from model_y import deviation, scale_for  # noqa: E402

RESULTS = []


def check(name, ok, note=""):
    RESULTS.append(bool(ok))
    print("  %-78s %s  %s" % (name, "GECTI" if ok else "KALDI", note))


def synthetic_data(n_words=232, n_rules=8, rows=1024, seed=0):
    """Deterministik test verisi: '<eos> r x f_r(x) f_r(f_r(x)) ... <eos>', 6-12 adim (dizi 10-16 token, dolgu var).
    f_r kural token'ina ozgu, sabit noktasiz bir permutasyon: sonraki token onceki token ve dizinin basindaki kuraldan
    belirli -- ogrenilebilir yapi (bigram tabani ln n_rules, kurala bakan model 0'a iner), gurultu degil.  Sozluk 242."""
    g = torch.Generator().manual_seed(seed)
    words, rules = ["w%03d" % i for i in range(n_words)], ["r%d" % i for i in range(n_rules)]
    order = torch.randperm(n_words, generator=g)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(n_words)
    shifts = 1 + torch.randperm(n_words - 1, generator=g)[:n_rules]          # 1 .. n-1: f_r(x) != x
    train = []
    for _ in range(rows):
        r, x, k = (int(torch.randint(lo, hi, (1,), generator=g)) for lo, hi in ((0, n_rules), (0, n_words), (6, 13)))
        seq = [rules[r], words[x]]
        for _ in range(k):
            x = int(order[(rank[x] + shifts[r]) % n_words])
            seq.append(words[x])
        train.append(seq)
    return dict(vocab=["<pad>", "<eos>"] + rules + words, train=train)


def sequences(data):
    """Her dizi <eos> ... <eos> -> (ids, mask), sagdan <pad>."""
    ix = {w: i for i, w in enumerate(data["vocab"])}
    return TR.pad([[ix["<eos>"]] + [ix[t] for t in s] + [ix["<eos>"]] for s in data["train"]], data["vocab"])


def reference_blocks(m, ids):
    """Model Y formulu, float64, modelden bagimsiz: h = PL; her turda Canon (x = h + sum_k w_k h_(t-k)), head basina
    attention (q, k birim, RoPE karmasik sayi carpimiyla; getirilen W_value dilimi), h <- norm(h + a_A (norm(W_context c) -
    h)), FactUnits SwiGLU (girdi sqrt(d) h), h <- norm(h + a_F (norm(olgu) - h)); cikis e^tau <h, PL>.  Sabit attention
    olcegi, PL girdisi ve phi'siz model icin."""
    assert not m.blocks[0].attention.attention_log_scale and not hasattr(m, "input_embedding") and not m.output_link
    unit = lambda v: v / v.norm(dim=-1, keepdim=True)
    dd = lambda t: t.detach().double()
    PL = unit(dd(m.tokens.fixed_points) + dd(m.tokens.shift))
    h = PL[ids]
    T, d = ids.shape[-1], h.shape[-1]
    future = torch.ones(T, T, dtype=torch.bool).triu(1)
    for t in range(m.turns):
        b = m.blocks[t % len(m.blocks)]                          # tur t -> Block t mod LAYERS
        at = b.attention
        w = dd(b.canon_weights)
        full = torch.cat([torch.zeros(*h.shape[:-2], 3, d, dtype=h.dtype), h], -2)
        x = h + sum(w[k] * full[..., 3 - k:3 - k + T, :] for k in range(4))
        dh = d // at.heads
        theta = torch.arange(T, dtype=torch.float64)[:, None] * at.rope_base ** (-torch.arange(0, dh, 2, dtype=torch.float64) / dh)
        rot = lambda y: torch.view_as_real(torch.view_as_complex(y.reshape(*y.shape[:-1], dh // 2, 2).contiguous())
                                           * torch.polar(torch.ones_like(theta), theta)).flatten(-2)
        heads = []
        for j in range(at.heads):                                # head j: W_query / W_key / W_value'nun j. satir dilimi
            sl = slice(j * dh, (j + 1) * dh)
            q, k = unit(x @ dd(at.W_query)[sl].T), unit(x @ dd(at.W_key)[sl].T)
            if at.rope:
                q, k = rot(q), rot(k)
            a = torch.softmax((at.scale * q @ k.transpose(-1, -2)).masked_fill(future, float("-inf")), -1)
            heads.append(a @ (x @ dd(at.W_value)[sl].T))
        h = unit(h + dd(m.alpha_attention[t]) * (unit(torch.cat(heads, -1) @ dd(at.W_context).T) - h))
        f = m.extra_facts[t - m.layers] if not m.shared_facts and t >= m.layers else b.facts
        if t == 0 and not m.first_turn_facts:
            continue
        xs = h * d ** 0.5
        gate, up = xs @ dd(f.W_fact_in).T, xs @ dd(f.W_fact_up).T
        u = gate / (1 + torch.exp(-gate)) * up                   # SiLU(gate) * up
        h = unit(h + dd(m.alpha_facts[t]) * (unit(u @ dd(f.W_fact_out).T) - h))
    return torch.exp(dd(m.log_output_scale)) * h @ PL.T


def t_step3():
    import functools
    import types
    import model_y
    tp = model_y.TokenPoints(3, d=4, anchor=0.05)
    with torch.no_grad():
        tp.fixed_points.copy_(torch.eye(4)[:3])
        tp.shift[0] = torch.tensor([math.cos(math.radians(17)) - 1, math.sin(math.radians(17)), 0, 0])
    dev = deviation(types.SimpleNamespace(tokens=tp))
    tp.anchor_loss().backward()
    gerr = float((tp.shift.grad - 2 * 0.05 * tp.shift.detach()).abs().max())
    check("noktalar: scale = ln(0,99 (n-1) / 0,01) (n 74), elle girilmez; capa gradyani = 2 . anchor . shift; 17 derece "
          "dondurulen nokta 17 derece okunur (deviation), digerleri 0",
          abs(scale_for(74) - math.log(0.99 * 73 / 0.01)) < 1e-12 and gerr < 1e-6 and abs(float(dev[0]) - 17) < 1e-3
          and float(dev[1:].abs().max()) < 1e-3, "%.4f" % float(dev[0]))
    TURNS = 2
    # tasarim formulu testleri: 2 tur, PL girdisi, sabit attention olcegi, phi'siz, tek parca kayip, paylasimli FactUnits
    BlockModel = functools.partial(model_y.BlockModel, turns=2, heads=2, attention_log_scale=False, loss_chunk=0,
                                   output_link=False, shared_facts=True, input_embedding=False, first_turn_facts=True)
    n, d = 12, 8
    g = torch.Generator().manual_seed(9)
    ids = torch.randint(0, n, (3, 9), generator=g)

    for kw in (dict(layers=1), dict(layers=TURNS)):
        m = BlockModel(n, d=d, units=10, **kw).double()
        with torch.no_grad():
            for p_ in m.parameters():
                if p_.requires_grad:
                    p_.copy_(0.5 * torch.randn(p_.shape, generator=g, dtype=torch.float64))
        err = float((m.logits(ids).detach() - reference_blocks(m, ids)).abs().max())
        check("model: skor = tasarim formulu (bagimsiz float64; Canon, 2 head, RoPE, normalized update, SwiGLU), %s" % kw,
              err < 1e-10, "fark %.1e" % err)

    base_logits = m.logits(ids).detach()
    changed = ids.clone()
    changed[:, 5] = (changed[:, 5] + 1) % n
    after = m.logits(changed).detach()
    check("model: nedensellik iki turda da -- konum 5 degisince 0-4 aynen kalir",
          float((after[:, :5] - base_logits[:, :5]).abs().max()) < 1e-10 and float((after[:, 5:] - base_logits[:, 5:]).abs().max()) > 1e-3)

    vocab = ["<pad>", "<eos>"] + [str(i) for i in range(n - 2)]
    rows = [[1, 3, 4, 5, 1], [1, 6, 7, 1], [1, 8, 9, 10, 11, 3, 1]]
    padded, mask = TR.pad(rows, vocab)
    lp = m.logits(padded).detach()
    err = max(float((lp[i, :len(r)] - m.logits(torch.tensor([r])).detach()[0]).abs().max()) for i, r in enumerate(rows))
    check("model: sagdaki dolgu sonucu degistirmez", err < 1e-10, "fark %.1e" % err)

    import itertools
    drawn = ("fixed_points", "W_query", "W_key", "W_context", "W_fact_in", "W_fact_up", "W_fact_out")   # tohumdan cekilenler
    starts = [(s_, k_, tuple(t_.flatten()[:8].tolist())) for s_ in range(4)
              for k_, t_ in BlockModel(238, seed=s_, layers=TURNS).state_dict().items() if k_.split(".")[-1] in drawn]
    shared_start = [(a_[:2], b_[:2]) for a_, b_ in itertools.combinations(starts, 2) if a_[2] == b_[2]]
    m0 = BlockModel(238)
    raw_q = torch.randn(64, 64, generator=torch.Generator().manual_seed(10)) / 8          # Block 0'in tohumu 100 x 0 + 10
    raw_f = torch.randn(170, 64, generator=torch.Generator().manual_seed(11)) / 8         # FactUnits'inki + 1
    ref = [[-0.134501650929451, -0.13766998052597046, -0.029936080798506737],
           [-0.10216674953699112, -0.06944606453180313, -0.10333619266748428],
           [-0.06384969502687454, 0.1285339742898941, -0.04414394870400429]]
    kept = [m0.tokens.fixed_points[0, :3].tolist(), raw_q[0, :3].tolist(), raw_f[0, :3].tolist()]
    check("model: tohumlar rastgele sayi paylasmaz (0-3 arasinda ayni baslayan tensor yok); tohum 0'in cekilisi degismedi "
          "(W_query ve W_fact_in kurede: cekilenin satirlari birim)",
          not shared_start and all(abs(a_ - b_) < 1e-7 for ka, kb in zip(kept, ref) for a_, b_ in zip(ka, kb))
          and torch.allclose(m0.blocks[0].attention.W_query, torch.nn.functional.normalize(raw_q, dim=1), atol=1e-6)
          and torch.allclose(m0.blocks[0].facts.W_fact_in, torch.nn.functional.normalize(raw_f, dim=1), atol=1e-6),
          str(shared_start[:3]))

    sh, se = BlockModel(n, d=d, units=10, layers=1), BlockModel(n, d=d, units=10, layers=TURNS)
    per_block = sum(p_.numel() for p_ in sh.blocks[0].parameters())
    count = lambda mm: sum(p_.numel() for p_ in mm.parameters())
    tb = sh.turn_blocks()
    check("model: layers=1 tek Block'u her turda kullanir, layers=turns her tura ayri Block",
          len(sh.blocks) == 1 and len(tb) == TURNS and all(b is tb[0] for b in tb) and len(se.blocks) == TURNS
          and count(se) - count(sh) == (TURNS - 1) * per_block and len(sh.hidden(ids)) == TURNS + 1)

    # ROPE=True: attention'da q ve k konuma gore dondurulur
    cnt = lambda mm: sum(p_.numel() for p_ in mm.parameters() if p_.requires_grad)
    mr = BlockModel(n, d=d, units=10, rope=False).double()
    with torch.no_grad():
        for p_ in mr.parameters():
            if p_.requires_grad:
                p_.copy_(0.5 * torch.randn(p_.shape, generator=g, dtype=torch.float64))
    err = float((mr.logits(ids).detach() - reference_blocks(mr, ids)).abs().max())
    check("model, rope=False: skor = tasarim formulu (RoPE'suz)", err < 1e-10, "fark %.1e" % err)
    mr = BlockModel(n, d=d, units=10)
    at, x = mr.blocks[0].attention, mr.hidden(ids)[0]
    q_, k_ = at.queries_keys(x)
    v_ = (x @ at.W_value.T).unflatten(-1, (at.heads, -1)).transpose(-3, -2)
    err_w = float(((at.weights(x) @ v_).transpose(-3, -2).flatten(-2) - at(x)).abs().max())
    rb = mr.logits(ids).detach()
    ch = ids.clone()
    ch[:, 5] = (ch[:, 5] + 1) % n
    ra = mr.logits(ch).detach()
    check("model, rope: varsayilan True; sayi degismez; q, k boyu 1 kalir; weights() ileri hesapla ayni; nedensellik",
          mr.rope and all(b.attention.rope for b in mr.blocks)
          and cnt(BlockModel(238, rope=False)) == cnt(BlockModel(238))
          and float((q_.norm(dim=-1) - 1).abs().max()) < 1e-5 and float((k_.norm(dim=-1) - 1).abs().max()) < 1e-5
          and err_w < 1e-5
          and float((ra[:, :5] - rb[:, :5]).abs().max()) < 1e-5 and float((ra[:, 5:] - rb[:, 5:]).abs().max()) > 1e-3,
          "weights farki %.1e" % err_w)

    data = synthetic_data()
    sids, smask = sequences(data)
    nv = len(data["vocab"])
    mr, curve_r = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=20, log_at=(0, 20), rope=True)
    md, _ = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=1, log_at=())
    mo, _ = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=1, log_at=(), rope=False)
    check("adim 3, rope=True: train_seq ayari modele ulasir, kayip iner; train_seq varsayilani (None) BlockModel'de ROPE (True); "
          "rope=False kapatir",
          all(b.attention.rope for b in mr.blocks) and curve_r[-1]["nll"] < curve_r[0]["nll"]
          and all(b.attention.rope for b in md.blocks) and all(not b.attention.rope for b in mo.blocks),
          "%.3f -> %.3f" % (curve_r[0]["nll"], curve_r[-1]["nll"]))
    # batches: mini-batch; model_kw: model ayarlari
    import copy
    whole, _ = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=5, log_at=(), schedule="wsd")
    via, _ = TR.train_seq("shared", None, None, nv, steps=5, log_at=(), batches=lambda step: (sids[:64], smask[:64]),
                          schedule="wsd")
    pick = lambda step: (sids[(step * 16) % 480:(step * 16) % 480 + 32], smask[(step * 16) % 480:(step * 16) % 480 + 32])
    packs = {}
    keep = lambda step, model, opt: packs.setdefault(step, dict(step=step, model=copy.deepcopy(model.state_dict()),
                                                                   optimizer=copy.deepcopy(opt.state_dict())))
    full, curve_b = TR.train_seq("shared", None, None, nv, steps=6, log_at=(0, 6), batches=pick, save_every=2, save=keep)
    resumed, _ = TR.train_seq("shared", None, None, nv, steps=6, log_at=(), batches=pick, checkpoint=packs[4])
    try:
        TR.train_seq("shared", None, None, nv, steps=1, log_at=())
        refused = False
    except AssertionError:
        refused = True
    same = lambda a, b: all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    check("train_seq, batches: her adimda butun veri = batches'siz egitim, bit duzeyinde; parcali egitimde 4. adim paketinden "
          "surdurulen = kesintisiz; ids ve batches yoksa reddeder",
          same(whole, via) and same(full, resumed) and refused and curve_b[-1]["nll"] < curve_b[0]["nll"],
          "%.3f -> %.3f" % (curve_b[0]["nll"], curve_b[-1]["nll"]))
    mk, _ = TR.train_seq("shared", sids[:8], smask[:8], nv, steps=1, log_at=(), model_kw=dict(d=32, turns=3, layers=1, units=64,
                                                                                          first_turn_facts=True))
    check("train_seq, model_kw: ayarlar modele ulasir (d 32, 3 tur, 64 birim)",
          tuple(mk.blocks[0].attention.W_query.shape) == (32, 32) and mk.turns == 3 and len(mk.hidden(sids[:2])) == 4
          and tuple(mk.blocks[0].facts.W_fact_in.shape) == (64, 32))

    before = BlockModel(nv).tokens.fixed_points.clone()
    m, curve = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=20, log_at=(0, 20))
    check("adim 3: egitimde PF bit duzeyinde degismez, W_context 0'dan ayrilir, kayip iner",
          torch.equal(m.tokens.fixed_points, before) and curve[-1]["W_context"] > 0 and curve[-1]["nll"] < curve[0]["nll"])

    a1, _ = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=3, log_at=(), seed=0)
    a2, _ = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=3, log_at=(), seed=1)
    check("egitim tarifi: seed modeli degistirir (PF dahil), seed=0 varsayilanla ayni",
          not torch.equal(a1.tokens.fixed_points, a2.tokens.fixed_points)
          and torch.equal(a1.tokens.fixed_points, before))

    # tarif: Muon (gizli matrisler) + Adam, takvim, compile varsayilan acik; masked_nll; RoPE en az fp32
    import inspect
    from model_y import apply_rope, masked_nll
    sig = inspect.signature(TR.train_seq).parameters
    check("tarif varsayilanlari (ana kosu): schedule 'coherence', son inis 'log', weight_ema 0,999, cooldown 0,2, compile True",
          sig["schedule"].default == TR.SCHEDULE == "coherence"
          and sig["final_cooldown_shape"].default == TR.FINAL_COOLDOWN_SHAPE == "log"
          and sig["weight_ema"].default == TR.WEIGHT_EMA == 0.999
          and sig["cooldown"].default == TR.COOLDOWN == 0.2 and sig["compile"].default is True)
    c_on, _ = TR.train_seq("shared", sids[:8], smask[:8], nv, steps=2, log_at=(), compile=True)
    c_off, _ = TR.train_seq("shared", sids[:8], smask[:8], nv, steps=2, log_at=(), compile=False)
    check("compile CPU'da kendiliginden kapali (if, train_seq): compile=True hata vermez, compile=False ile bit duzeyinde ayni; "
          "COMPILE_LOCK (iplikler arasi derleme kilidi) egitimden sonra serbest",
          all(torch.equal(a, b) for a, b in zip(c_on.state_dict().values(), c_off.state_dict().values()))
          and not TR.COMPILE_LOCK.locked())

    mb = BlockModel(nv)
    with torch.no_grad():
        for p_ in mb.parameters():
            if p_.requires_grad:
                p_.copy_(0.3 * torch.randn(p_.shape, generator=g))
    ids_, mask_ = sids[:16], smask[:16]
    lg = mb.logits(ids_[:, :-1])
    old = torch.nn.functional.cross_entropy(lg[mask_[:, 1:]], ids_[:, 1:][mask_[:, 1:]])
    new = masked_nll(lg, ids_[:, 1:], mask_[:, 1:])
    ga = torch.autograd.grad(old, [mb.blocks[0].attention.W_context, mb.tokens.shift], retain_graph=True)
    gb = torch.autograd.grad(new, [mb.blocks[0].attention.W_context, mb.tokens.shift])
    gerr = max(float((a_ - b_).abs().max()) for a_, b_ in zip(ga, gb))
    check("masked_nll = logits[valid] ile eski kayip (deger ve gradyan), sekil sabit", abs(float(old - new)) < 1e-6 and gerr < 1e-7,
          "fark %.1e, gradyan %.1e" % (abs(float(old - new)), gerr))

    xr = torch.randn(2, 600, 64, generator=g)
    T_, dh_ = 600, 64
    fr = 10000.0 ** (-torch.arange(0, dh_, 2, dtype=torch.float32) / dh_)
    an = torch.arange(T_, dtype=torch.float32)[:, None] * fr[None, :]
    x1_, x2_ = xr[..., 0::2], xr[..., 1::2]
    ref_ = torch.stack([x1_ * an.cos() - x2_ * an.sin(), x1_ * an.sin() + x2_ * an.cos()], -1).flatten(-2)
    bf = float((apply_rope(xr.bfloat16()).float() - ref_).abs().max())
    check("RoPE: fp32'de onceki formulle birebir; bf16 girdide acilar fp32'de (konum > 256 dogru)",
          torch.equal(apply_rope(xr), ref_) and bf < 0.05, "bf16 fark %.3f" % bf)

    Gm = torch.randn(64, 256, generator=g)
    sv = torch.linalg.svdvals(TR.Muon.orthogonalize(Gm))
    check("Muon: Newton-Schulz ciktisinin tekil degerleri ~1 (yari-ortogonal)", float((sv - 1).abs().max()) < 0.35,
          "tekil deger %.3f..%.3f" % (float(sv.min()), float(sv.max())))
    stack = [torch.randn(64, 256, generator=g) for _ in range(3)]
    batch_same = all(torch.equal(x, TR.Muon.orthogonalize(y)) for x, y in zip(TR.Muon.orthogonalize(torch.stack(stack)), stack))
    check("Muon: Newton-Schulz ayni boydaki matrislerin yigininda her matrise ayri (tek is parcacigi: bit duzeyinde ayni)",
          batch_same)
    ob = TR.Muon.orthogonalize(Gm, precision="bf16")
    of = TR.Muon.orthogonalize(Gm)
    svb = torch.linalg.svdvals(ob)
    rel = float((ob - of).norm() / of.norm())
    ns_opts = []
    nb, _ = TR.train_seq("shared", sids[:8], smask[:8], nv, steps=2, log_at=(), save_every=1,
                         save=lambda step, model, opt: ns_opts.append(opt), newton_schulz_precision="bf16")
    ns_set = ns_opts[-1].newton_schulz_precision
    TR.train_seq("shared", sids[:8], smask[:8], nv, steps=1, log_at=(), save_every=1,
                 save=lambda step, model, opt: ns_opts.append(opt))
    try:
        TR.train_seq("shared", sids[:8], smask[:8], nv, steps=1, log_at=(), newton_schulz_precision="fp16")
        refused_k = False
    except AssertionError:
        refused_k = True
    check("NEWTON_SCHULZ_PRECISION: varsayilan fp32; bf16 Newton-Schulz tekil degerleri ~1, fp32'ye yakin, tipi G'ninki, "
          "train_seq optimizer'a iletir; gecersiz deger reddedilir",
          TR.NEWTON_SCHULZ_PRECISION == "fp32" and float((svb - 1).abs().max()) < 0.35
          and rel < 0.05 and ob.dtype == Gm.dtype and ns_set == "bf16" and ns_opts[-1].newton_schulz_precision == "fp32"
          and not any(torch.isnan(p_).any() for p_ in nb.parameters()) and refused_k,
          "bf16 tekil %.3f..%.3f, fp32'den fark %.3f" % (float(svb.min()), float(svb.max()), rel))

    opts, lrs = [], {}

    def grab(step, model, opt):
        opts.append(opt)
        lrs[step] = opt.param_groups[0]["lr"]
    mm, curve_m = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=10, log_at=(0, 10), save_every=1, save=grab,
                               schedule="wsd")
    names_m = {id(p_): k.split(".")[-1] for k, p_ in mm.named_parameters()}
    in_muon = sorted({names_m[id(p_)] for g_ in opts[0].param_groups if g_["use_muon"] for p_ in g_["params"]})
    in_adam = sorted({names_m[id(p_)] for g_ in opts[0].param_groups if not g_["use_muon"] for p_ in g_["params"]})
    wsd = [0.01 * (1.0 if t < 8 else TR.LR_FLOOR + (1 - TR.LR_FLOOR) * (1 - math.sqrt((t - 8) / 2))) for t in range(1, 11)]
    check("Muon + WSD: Muon'da W_context, W_fact_in, W_fact_up, W_fact_out, W_value; Adam'da noktalar, W_query, "
          "W_key, cikis olcegi; "
          "lr 8. adima kadar sabit, sonra 1 - sqrt ile LR x LR_FLOOR'a; kayip iner",
          isinstance(opts[0], TR.Muon) and in_muon == ["W_context", "W_fact_in", "W_fact_out", "W_fact_up", "W_value"]
          and in_adam == ["W_key", "W_query", "alpha_attention", "alpha_facts", "canon_weights", "input_embedding",
                          "link_q", "link_u", "log_output_scale", "shift"]
          and all(abs(lrs[t] - w) < 1e-12 for t, w in zip(range(1, 11), wsd)) and curve_m[-1]["nll"] < curve_m[0]["nll"],
          "%s | %s | lr %s" % (in_muon, in_adam, [round(lrs[t], 5) for t in range(1, 11)]))

    import copy
    packs_m = {}
    keep_m = lambda step, model, opt: packs_m.setdefault(step, dict(step=step, model=copy.deepcopy(model.state_dict()),
                                                                     optimizer=copy.deepcopy(opt.state_dict())))
    full_m, _ = TR.train_seq("shared", sids[:40], smask[:40], nv, steps=6, log_at=(), save_every=2, save=keep_m)
    res_m, _ = TR.train_seq("shared", sids[:40], smask[:40], nv, steps=6, log_at=(), checkpoint=packs_m[4])
    check("Muon + WSD: 4. adim paketinden surdurulen = kesintisiz, bit duzeyinde",
          all(torch.equal(a_, b_) for a_, b_ in zip(full_m.state_dict().values(), res_m.state_dict().values())))

    packs_s = {}
    keep_s = lambda step, model, opt: packs_s.setdefault(step, dict(step=step, model=copy.deepcopy(model.state_dict()),
                                                                     optimizer=copy.deepcopy(opt.state_dict())))

    def stop_at_3(step, model, nll):
        if step == 3:
            raise RuntimeError("durdur")
    try:
        TR.train_seq("shared", sids[:40], smask[:40], nv, steps=6, log_at=(), every=1, callback=stop_at_3, save_every=2,
                     save=keep_s)
        stopped = False
    except RuntimeError:
        stopped = True
    res_s, _ = TR.train_seq("shared", sids[:40], smask[:40], nv, steps=6, log_at=(), checkpoint=packs_s[3])
    check("durdurma: callback hata atinca o adimin paketi de yazilir (adim 3, save_every 2); ondan surdurulen = kesintisiz, "
          "bit duzeyinde",
          stopped and sorted(packs_s) == [2, 3]
          and all(torch.equal(a_, b_) for a_, b_ in zip(full_m.state_dict().values(), res_s.state_dict().values())),
          str(sorted(packs_s)))


@torch.no_grad()
def cached_scores(m, prompts, tails):
    """AttentionCache ile, token'lar SIRAYLA verilerek (acgozlu degil): satir basina istemin son konumunun ve tails'in
    her token'indan sonraki skorlar, (B, k + 1, n); ve istem hesabinin butun skorlari."""
    from model_y import AttentionCache
    L = [len(p) for p in prompts]
    ids = torch.zeros(len(prompts), max(L), dtype=torch.long)
    for i, p in enumerate(prompts):
        ids[i, :len(p)] = torch.tensor(p)
    caches = [AttentionCache(torch.tensor(L), max(L) + len(tails[0]) + 1) for _ in range(m.turns)]
    first = m.logits(ids, caches)
    rows = [first[torch.arange(len(prompts)), torch.tensor(L) - 1]]
    for j in range(len(tails[0])):
        rows.append(m.logits(torch.tensor([t[j] for t in tails])[:, None], caches)[:, 0])
    return torch.stack(rows, 1), first


def t_generate_cached():
    from model_y import BlockModel, apply_rope
    g = torch.Generator().manual_seed(21)
    pos = torch.tensor([[0], [5], [16]])
    x = torch.randn(3, 17, 12, generator=g)
    at_pos = apply_rope(x[torch.arange(3), pos[:, 0]][:, None], pos)
    err_p = float((at_pos[:, 0] - apply_rope(x)[torch.arange(3), pos[:, 0]]).abs().max())
    check("apply_rope: positions verilince o konumdaki donusun aynisi", err_p < 1e-6, "fark %.1e" % err_p)

    # onbellekli uretim = tam yeniden hesap: rastgele agirlik, butun Block ayarlari, farkli uzunlukta istemler
    n = 40
    prompts = [torch.randint(0, n, (int(k),), generator=g).tolist() for k in torch.randint(1, 21, (12,), generator=g)]
    worst, same, runs = 0.0, True, 0
    for kw in (dict(), dict(layers=4), dict(rope=False)):
        m = BlockModel(n, d=16, units=24, t_max=64, **kw).double()      # float64: fp32 yuvarlamasi SwiGLU'da 1e-4'u asiyor
        with torch.no_grad():
            for p_ in m.parameters():
                if p_.requires_grad:
                    p_.copy_(0.5 * torch.randn(p_.shape, generator=g))
        old, new = TR.generate(m, prompts, 15, cached=False), TR.generate(m, prompts, 15)
        alone = [TR.generate(m, [p], 15)[0] for p in prompts[:4]]
        same &= old == new and alone == new[:4]
        scores, first = cached_scores(m, prompts, new)
        for i, p in enumerate(prompts):
            full = m.logits(torch.tensor([p + new[i]])).detach()[0]
            worst = max(worst, float((scores[i] - full[len(p) - 1:]).abs().max()),
                        float((first[i, :len(p)] - full[:len(p)]).abs().max()))
        runs += 1
    check("generate: onbellekli (AttentionCache) = tam yeniden hesap -- ayni token'lar, %d ayar x 12 istem (1-20 token) x 15; "
          "tek basina = batch icinde; her konumun skoru float64'te 1e-10 icinde" % runs, same and worst < 1e-10,
          "en buyuk fark %.1e" % worst)
    edge = BlockModel(n, d=16, units=24, t_max=64)
    check("generate: n = 0 bos, n = 1 istem hesabinin son konumu",
          TR.generate(edge, prompts[:3], 0) == [[], [], []]
          and TR.generate(edge, prompts[:3], 1) == TR.generate(edge, prompts[:3], 1, cached=False))
    from model_y import AttentionCache
    ring_m = BlockModel(n, d=16, units=24, t_max=64)
    L = torch.tensor([len(p) for p in prompts])
    caches = [AttentionCache(L, int(L.max()) + 15) for _ in range(ring_m.turns)]
    padded = torch.zeros(len(prompts), int(L.max()), dtype=torch.long)
    for i, p in enumerate(prompts):
        padded[i, :len(p)] = torch.tensor(p)
    with torch.no_grad():
        ring_m.logits(padded, caches)
        for _ in range(14):
            ring_m.logits(torch.zeros(len(prompts), 1, dtype=torch.long), caches)
    check("generate: Canon onbellegi 3 yuvali halka -- tur basina (istem sayisi, 3, d), istem boyundan ve uretilen token "
          "sayisindan bagimsiz (esitlik yukarida, 1-20 token'lik istemlerle)",
          all(tuple(c.canon_inputs.shape) == (len(prompts), 3, 16) for c in caches), str(tuple(caches[0].canon_inputs.shape)))

    # egitilmis agirlik: test verisinde 60 adim; farkli boyda dizi baslari ve uzun devam
    data = synthetic_data()
    sids, smask = sequences(data)
    nv = len(data["vocab"])
    trained, curve = TR.train_seq("shared", sids[:256], smask[:256], nv, steps=60, log_at=(0, 60))
    qs = [sids[i, :2 + i % 8].tolist() for i in range(96)]
    old, new = TR.generate(trained, qs, 24, cached=False), TR.generate(trained, qs, 24)
    scores, _ = cached_scores(trained, qs[:24], new[:24])
    err = max(float((scores[i] - trained.logits(torch.tensor([q + new[i]])).detach()[0, len(q) - 1:]).abs().max())
              for i, q in enumerate(qs[:24]))
    check("generate, egitilmis Model X (test verisi, 256 dizi, 60 adim): 96 istem (dizi basi, 2-9 token) x 24 token "
          "onbellekli = tam yeniden hesap; skorlar tolerans icinde", old == new and err < 1e-4 and curve[-1]["nll"] < curve[0]["nll"],
          "fark %.1e  kayip %.3f -> %.3f" % (err, curve[0]["nll"], curve[-1]["nll"]))


def t_normalized_update():
    """Normalized update ve kure agirliklari: alpha ve agirliklarin baslangici; blok ciktisi ne kadar buyurse buyusun durum
    alpha kadar doner; agirliklar her adimdan sonra birim; surdurme ayni; son turun alpha'si (LAST_FACTS_ALPHA_INIT).
    Tasarim formulu: t_step3."""
    import copy
    import functools
    from model_y import ALPHA_INIT, LAST_FACTS_ALPHA_INIT, BlockModel
    BlockModel = functools.partial(BlockModel, input_embedding=False, first_turn_facts=True,   # tur 1 FactUnits'li,
                                   attention_log_scale=False)                                     # sabit attention olcegi
    data = synthetic_data()
    sids, smask = sequences(data)
    nv = len(data["vocab"])

    both = BlockModel(nv, output_link=False)
    b0 = both.blocks[0]
    unit_rows = lambda w: float((w.norm(dim=1) - 1).abs().max())
    unit_cols = lambda w: float((w.norm(dim=0) - 1).abs().max())
    check("alpha (tur, d) = ALPHA_INIT (son turun FactUnits'i LAST_FACTS_ALPHA_INIT), W_context / W_fact_out rastgele, satir / "
          "sutunlar birim",
          both.alpha_attention.shape == (both.turns, 64) and bool((both.alpha_attention == ALPHA_INIT).all())
          and bool((both.alpha_facts[:-1] == ALPHA_INIT).all()) and bool((both.alpha_facts[-1] == LAST_FACTS_ALPHA_INIT).all())
          and max(unit_rows(b0.attention.W_query), unit_rows(b0.attention.W_key), unit_rows(b0.facts.W_fact_in),
                  unit_cols(b0.attention.W_context), unit_cols(b0.facts.W_fact_out)) < 1e-6)
    g = torch.Generator().manual_seed(41)

    # FactUnits ciktisi 1000 kat buyutulse de durum alpha kadar doner
    def turn_cos(model):
        with torch.no_grad():
            hs = model.hidden(sids[:16])
            return min(float((a * b).sum(-1)[smask[:16]].min()) for a, b in zip(hs[:-1], hs[1:]))
    fixed = BlockModel(nv, last_facts_alpha_init=ALPHA_INIT)   # butun alpha'lar 0,1
    with torch.no_grad():
        fixed.blocks[0].attention.W_context.copy_(torch.randn(64, 64, generator=g) / 8)
        fixed.blocks[0].facts.W_fact_out.copy_(1000 * torch.randn(fixed.blocks[0].facts.W_fact_out.shape, generator=g) / 16)
    c_fixed = turn_cos(fixed)
    check("normalized update: FactUnits ciktisi 1000 kat buyukken tur basina cos(h_once, h_sonra) >= 0,95 (alpha 0,1)",
          c_fixed >= 0.95, "en kucuk cos %.3f" % c_fixed)

    # egitim: kayip iner, agirliklar her adimdan sonra birim, alpha Adam'da ve ogreniyor; surdurme bit duzeyinde
    opts = []
    grab = lambda step, model, opt: opts.append(opt)
    kw = {}
    trained, curve = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=20, log_at=(0, 20), save_every=1, save=grab, **kw)
    tb = trained.blocks[1]                                 # varsayilanda blocks[0]'in FactUnits'i yok
    names = {id(p_): k for k, p_ in trained.named_parameters()}
    in_adam = sorted({names[id(p_)] for g_ in opts[-1].param_groups if not g_["use_muon"] for p_ in g_["params"]})
    check("egitim (normalized update + kure agirliklari, Muon): kayip iner; 20 adimdan sonra satir / sutunlar birim; "
          "alpha Adam'da ve baslangictan ayrildi",
          curve[-1]["nll"] < curve[0]["nll"] and "alpha_attention" in in_adam and "alpha_facts" in in_adam
          and max(unit_rows(tb.attention.W_query), unit_rows(tb.attention.W_key), unit_rows(tb.facts.W_fact_in),
                  unit_cols(tb.attention.W_context), unit_cols(tb.facts.W_fact_out)) < 1e-5
          and float((trained.alpha_facts - BlockModel(nv, **kw).alpha_facts).abs().max()) > 1e-4,
          "%.3f -> %.3f  alpha %.4f..%.4f" % (curve[0]["nll"], curve[-1]["nll"], float(trained.alpha_facts.min()),
                                              float(trained.alpha_facts.max())))
    packs = {}
    keep = lambda step, model, opt: packs.setdefault(step, dict(step=step, model=copy.deepcopy(model.state_dict()),
                                                                   optimizer=copy.deepcopy(opt.state_dict())))
    full, _ = TR.train_seq("shared", sids[:40], smask[:40], nv, steps=6, log_at=(), save_every=2, save=keep, **kw)
    res, _ = TR.train_seq("shared", sids[:40], smask[:40], nv, steps=6, log_at=(), checkpoint=packs[4], **kw)
    qs = [[1] + sids[i, 1:9].tolist() for i in range(12)]
    check("normalized update + kure agirliklari: 4. adim paketinden surdurulen = kesintisiz, bit duzeyinde; onbellekli uretim = "
          "tam yeniden hesap",
          all(torch.equal(a, b) for a, b in zip(full.state_dict().values(), res.state_dict().values()))
          and TR.generate(trained, qs, 10) == TR.generate(trained, qs, 10, cached=False))

    # LAST_FACTS_ALPHA_INIT: ALPHA_INIT verilince eski baslatma birebir; 1,0'da son durum girdi noktasindan ayrilir, adim-0
    # kaybi ln n + s^2 / (2d) (h rastgele yon: <h, p> ~ N(0, 1/d)), girdiyi tekrar etme kaybolur
    old_init, new_init = (BlockModel(nv, last_facts_alpha_init=ALPHA_INIT, shared_facts=True),   # esikler paylasimli
                          BlockModel(nv, last_facts_alpha_init=1.0, shared_facts=True))       # FactUnits ile olculdu
    before = {k: v.clone() for k, v in new_init.state_dict().items()}
    before["alpha_facts"].fill_(ALPHA_INIT)                               # 28 Eylul oncesi kod: torch.full(ALPHA_INIT)
    same_old = list(before) == list(old_init.state_dict()) and all(torch.equal(before[k], v)
                                                                    for k, v in old_init.state_dict().items())
    rows, gv = [], torch.Generator().manual_seed(7)
    big = torch.randint(0, 8004, (4, 33), generator=gv)
    for n_, d_, units_, ids_, mask_, tol in ((nv, 64, 170, sids[:32], smask[:32], 0.35),
                                             (8004, 384, 1024, big, torch.ones_like(big, dtype=torch.bool), 0.1)):
        out_ = {}
        for init in (1.0, ALPHA_INIT):
            m_ = BlockModel(n_, d=d_, units=units_, last_facts_alpha_init=init, shared_facts=True)
            with torch.no_grad():
                keep_ = mask_[:, 1:]
                repeat = float((m_.logits(ids_[:, :-1]).argmax(-1) == ids_[:, :-1])[keep_].float().mean())
                out_[init] = (float(m_.loss(ids_, mask_)[1]), repeat)
        want = math.log(n_) + m_.scale ** 2 / (2 * d_)
        rows.append((n_, d_, want, out_, abs(out_[1.0][0] - want) < tol and out_[ALPHA_INIT][0] > want + 2
                     and out_[ALPHA_INIT][1] > 0.9 and out_[1.0][1] < 0.05))
    check("last_facts_alpha_init: varsayilan 0,1 (ana kosu); ALPHA_INIT ile eski baslatma bit duzeyinde; 1,0'da adim-0 "
          "kaybi ln n + s^2/(2d)'ye yakin "
          "(test verisi n 242 d 64, 0,35; n 8004 d 384 rastgele token, 0,1), 0,1'de 2 nat ustunde; girdiyi "
          "tekrar (argmax = girdi token'i) 0,1'de > %90, 1,0'da < %5",
          LAST_FACTS_ALPHA_INIT == 0.1 and same_old and all(r[-1] for r in rows),
          "  ".join("n %d d %d: formul %.3f / 1,0 kayip %.3f tekrar %.3f / 0,1 kayip %.3f tekrar %.3f" % (
              n_, d_, want, o[1.0][0], o[1.0][1], o[ALPHA_INIT][0], o[ALPHA_INIT][1])
              for n_, d_, want, o, _ in rows))


def t_canon():
    """Canon-A: w 0'dan; attention girdisi resmi kodun hesabiyla (PhysicsLM4
    canon_helper: x + conv1d(groups=d, cekirdek 4, soldan 3 dolgu), aktivasyon ve bias yok) ayni; nedensel; egitim ve
    surdurme; onbellekli uretim (canon_cache) tam hesapla ayni."""
    import copy
    import torch.nn.functional as F
    from model_y import BlockModel
    data = synthetic_data()
    sids, smask = sequences(data)
    nv = len(data["vocab"])
    ids = sids[:6, :20]

    zero = BlockModel(nv)
    check("canon: canon_weights (4, d) 0'dan", tuple(zero.blocks[0].canon_weights.shape) == (4, 64)
          and not zero.blocks[0].canon_weights.any())

    # resmi kodun hesabi: out = x + conv1d(x, W, padding=3, groups=d)[..., :T], W[:, 0, 3 - k] = w_k
    g = torch.Generator().manual_seed(51)
    m = BlockModel(nv)
    with torch.no_grad():
        m.blocks[0].canon_weights.copy_(torch.randn(4, 64, generator=g))
    seen = []
    hook = m.blocks[0].attention.register_forward_pre_hook(lambda mod, args, kwargs: seen.append(args[0].detach()),
                                                            with_kwargs=True)
    with torch.no_grad():
        m.logits(ids)
    hook.remove()
    with torch.no_grad():
        x = m.tokens.points()[ids]                                  # tur 1'in girdisi: PL
        W = m.blocks[0].canon_weights.flip(0).T.unsqueeze(1)        # (d, 1, 4): W[:, 0, 3 - k] = w_k
        ref = x + F.conv1d(x.transpose(1, 2), W, padding=3, groups=64)[..., :x.shape[1]].transpose(1, 2)
    err = float((seen[0] - ref).abs().max())
    check("canon: attention girdisi (tur 1) = resmi kodun hesabi x + conv1d(gruplu, cekirdek 4, soldan dolgu)", err < 1e-5,
          "fark %.1e" % err)

    # nedensellik: konum 5'teki token degisince 0-4'un skoru ayni, 5 ve sonrasi degisir
    alt = ids.clone()
    alt[:, 5] = (alt[:, 5] + 1) % nv
    with torch.no_grad():
        a_, b_ = m.logits(ids), m.logits(alt)
    check("canon: nedensel -- konum 5 degisince 0-4 bit duzeyinde ayni, 5 ve sonrasi degisir",
          torch.equal(a_[:, :5], b_[:, :5]) and not torch.equal(a_[:, 5:], b_[:, 5:]))

    opts = []
    grab = lambda step, model, opt: opts.append(opt)
    trained, curve = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=20, log_at=(0, 20), save_every=1, save=grab)
    names = {id(p_): k for k, p_ in trained.named_parameters()}
    in_adam = [names[id(p_)] for g_ in opts[-1].param_groups if not g_["use_muon"] for p_ in g_["params"]]
    packs = {}
    keep = lambda step, model, opt: packs.setdefault(step, dict(step=step, model=copy.deepcopy(model.state_dict()),
                                                                   optimizer=copy.deepcopy(opt.state_dict())))
    full, _ = TR.train_seq("shared", sids[:40], smask[:40], nv, steps=6, log_at=(), save_every=2, save=keep)
    res, _ = TR.train_seq("shared", sids[:40], smask[:40], nv, steps=6, log_at=(), checkpoint=packs[4])
    qs = [[1] + sids[i, 1:1 + k].tolist() for i, k in enumerate((1, 2, 3, 5, 8, 8, 12, 16))]   # 2-17 token: Canon
    cached_new = TR.generate(trained, qs, 6)                                                 # istem basini da gorur
    scores, _ = cached_scores(trained, qs, cached_new)
    err_c = max(float((scores[i] - trained.logits(torch.tensor([q + cached_new[i]])).detach()[0, len(q) - 1:]).abs().max())
                for i, q in enumerate(qs))
    check("canon: egitimde kayip iner, canon_weights 0'dan ayrilir ve Adam'da; surdurme bit duzeyinde; onbellekli uretim "
          "(canon_cache) tam hesapla ayni token'lar ve skorlar (istem 2-17 token)",
          curve[-1]["nll"] < curve[0]["nll"] and bool(trained.blocks[0].canon_weights.abs().max() > 0)
          and "blocks.0.canon_weights" in in_adam
          and all(torch.equal(a, b) for a, b in zip(full.state_dict().values(), res.state_dict().values()))
          and cached_new == TR.generate(trained, qs, 6, cached=False) and err_c < 1e-4,
          "%.3f -> %.3f  |w| %.4f  skor farki %.1e" % (curve[0]["nll"], curve[-1]["nll"],
                                                     float(trained.blocks[0].canon_weights.abs().max()), err_c))


def t_layers():
    """LAYERS: paylasilan blokta farkli Block sayisi, turlar sirayla (A B A B).  layers=1 bugunku model; layers=2, turns=4
    iki Block'u donusumlu kullanir; ayri blok (separate) Model X2 anahtarlariyla egitilir."""
    import copy
    from model_y import BlockModel
    data = synthetic_data()
    sids, smask = sequences(data)
    nv = len(data["vocab"])
    ids = sids[:6, :20]

    base, explicit = BlockModel(nv), BlockModel(nv, layers=2, turns=4)
    with torch.no_grad():
        same = torch.equal(base.logits(ids), explicit.logits(ids))
    check("layers: varsayilan = 2 x 2 (layers=2, turns=4; kullanici, 28 Eylul), bit duzeyinde ayni",
          same and list(base.state_dict()) == list(explicit.state_dict()) and len(base.blocks) == 2 and base.turns == 4)

    m = BlockModel(nv, layers=2, turns=4, shared_facts=True)
    tb = m.turn_blocks()
    four = BlockModel(nv, turns=4, layers=1, shared_facts=True)
    count = lambda mm: sum(p_.numel() for p_ in mm.parameters())
    per_block = sum(p_.numel() for p_ in m.blocks[0].parameters())
    check("layers: layers=2, turns=4 -> iki Block, sira A B A B; parametre = tek Block'lu 4 tur + bir Block; alpha (4, d)",
          len(m.blocks) == 2 and [tb.index(b) for b in tb] == [0, 1, 0, 1] and tb[0] is tb[2] and tb[1] is tb[3]
          and tb[0] is not tb[1] and count(m) - count(four) == per_block
          and tuple(m.alpha_facts.shape) == (4, 64) and len(m.hidden(ids)) == 5)

    # B'yi bozunca tur 1 (A) ayni, tur 2 (B) ve sonrasi degisir
    alt = copy.deepcopy(m)
    with torch.no_grad():
        alt.blocks[1].attention.W_query.add_(0.5)
        ha, hb = m.hidden(ids), alt.hidden(ids)
    check("layers: tur 1 yalniz A'yi kullanir (B degisince h1 bit duzeyinde ayni), tur 2 B'yi kullanir (h2 degisir)",
          torch.equal(ha[1], hb[1]) and not torch.equal(ha[2], hb[2]))

    refused = []
    for kw in (dict(layers=2, turns=3), dict(layers=3, turns=4)):
        try:
            BlockModel(nv, **kw)
            refused.append(False)
        except AssertionError:
            refused.append(True)
    check("layers: turns layers'in kati degilse reddedilir", all(refused))

    opts = []
    grab = lambda step, model, opt: opts.append(opt)
    kw = dict(layers=2, turns=4, first_turn_facts=True)
    trained, curve = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=20, log_at=(0, 20), save_every=1, save=grab, model_kw=kw)
    names = {id(p_): k for k, p_ in trained.named_parameters()}
    groups = {names[id(p_)]: g_["use_muon"] for g_ in opts[-1].param_groups for p_ in g_["params"]}
    unit = all(torch.allclose(b.attention.W_query.norm(dim=1), torch.ones(64), atol=1e-5)
               and torch.allclose(b.facts.W_fact_out.norm(dim=0), torch.ones(b.facts.W_fact_out.shape[1]), atol=1e-5)
               for b in trained.blocks)
    packs = {}
    keep = lambda step, model, opt: packs.setdefault(step, dict(step=step, model=copy.deepcopy(model.state_dict()),
                                                                   optimizer=copy.deepcopy(opt.state_dict())))
    full, _ = TR.train_seq("shared", sids[:40], smask[:40], nv, steps=6, log_at=(), save_every=2, save=keep, model_kw=kw)
    res, _ = TR.train_seq("shared", sids[:40], smask[:40], nv, steps=6, log_at=(), checkpoint=packs[4], model_kw=kw)
    check("layers: layers=2 egitimde kayip iner; iki Block'un matrisleri ayni optimizer grubunda ve kurede; canon her "
          "Block'ta ayri ogrenir; surdurme bit duzeyinde",
          curve[-1]["nll"] < curve[0]["nll"] and unit
          and all(groups["blocks.0." + k] == groups["blocks.1." + k]
                  for k in ("attention.W_query", "attention.W_context", "facts.W_fact_in", "canon_weights"))
          and not torch.equal(trained.blocks[0].canon_weights, trained.blocks[1].canon_weights)
          and all(torch.equal(a, b) for a, b in zip(full.state_dict().values(), res.state_dict().values())),
          "%.3f -> %.3f" % (curve[0]["nll"], curve[-1]["nll"]))

    sep, curve_s = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=20, log_at=(0, 20), model_kw=dict(layers=4))
    check("layers: her tura ayri Block (layers = turns) Model X2 anahtarlariyla (normalized_update, sphere_weights, canon) "
          "egitilir",
          curve_s[-1]["nll"] < curve_s[0]["nll"] and len(sep.blocks) == sep.turns and sep.canon
          and sep.normalized_update and sep.sphere_weights
          and all(torch.allclose(b.attention.W_key.norm(dim=1), torch.ones(64), atol=1e-5) for b in sep.blocks),
          "%.3f -> %.3f" % (curve_s[0]["nll"], curve_s[-1]["nll"]))

    plain = BlockModel(nv, layers=2, turns=4)
    qs = [[1] + sids[i, 1:9].tolist() for i in range(8)]
    check("layers: layers=2 onbellekli uretim tam hesapla ayni (tur basina onbellek)",
          TR.generate(plain, qs, 6) == TR.generate(plain, qs, 6, cached=False))


def t_coherence():
    """SCHEDULE "coherence": mini-batch'te adim iki yarida, birlestirilen gradyan tam batch'inki (ilk adim wsd ile ayni);
    tam batch'te rho = 1 (lr sabit, sonda final_cooldown inisi); mini-batch'te c, rho, ortalama gecerli ve lr = LR x
    ortalama; surdurme bit duzeyinde (ortalama optimizer'in grup kaydinda); lr_floor yoksa reddedilir."""
    import copy
    data = synthetic_data()
    sids, smask = sequences(data)
    nv = len(data["vocab"])

    def batches(step):                                     # adimin fonksiyonu: 16 cumle
        rows = torch.randperm(len(sids), generator=torch.Generator().manual_seed(1000 + step))[:16]
        return sids[rows], smask[rows]

    a, _ = TR.train_seq("shared", None, None, nv, steps=1, log_at=(), batches=batches, schedule="coherence")
    b, _ = TR.train_seq("shared", None, None, nv, steps=1, log_at=(), batches=batches, schedule="wsd")
    err = max(float((x - y).abs().max()) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    c0 = a.coherence
    check("coherence: ilk adim (ortalama 1) wsd ile ayni guncelleme -- iki yarinin birlesik gradyani = tam batch; "
          "c, rho gecerli", err < 1e-4 and -1 <= c0["c"] <= 1 and 0 <= c0["rho"] <= 1 and c0["lr"] == TR.LR,
          "fark %.1e  c %.3f rho %.3f" % (err, c0["c"], c0["rho"]))

    lrs = {}
    grab = lambda step, model, opt: lrs.setdefault(step, opt.param_groups[0]["lr"])
    full, _ = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=40, log_at=(), save_every=1, save=grab, schedule="coherence",
                           final_cooldown=0.05, final_cooldown_shape="sqrt")      # 29 Eylul oncesi varsayilan: son %5, 1 - sqrt
    floor = TR.LR_FLOOR
    want = {37: TR.LR, 38: TR.LR, 39: TR.LR * (floor + (1 - floor) * (1 - math.sqrt(0.5))), 40: TR.LR * floor}
    check("coherence, tam batch: rho olculmez (= 1), lr sabit; son %5'te 1 - sqrt ile x LR_FLOOR'a",
          all(abs(lrs[t] - v) < 1e-12 for t, v in want.items()) and all(lrs[t] == TR.LR for t in range(1, 38))
          and not getattr(full, "coherence", None), str({t: round(lrs[t], 6) for t in (1, 37, 38, 39, 40)}))

    lin = {}
    grab_l = lambda step, model, opt: lin.setdefault(step, opt.param_groups[0]["lr"])
    TR.train_seq("shared", sids[:64], smask[:64], nv, steps=40, log_at=(), save_every=1, save=grab_l, schedule="coherence",
                 final_cooldown=0.5, lr_floor=0.0, final_cooldown_shape="linear")
    want_l = {t: TR.LR * (1 - (t - 20) / 20) for t in range(20, 41)}
    check("coherence, final_cooldown_shape linear: son %50'de dogrusal x 0'a (D2Z)",
          all(abs(lin[t] - v) < 1e-12 for t, v in want_l.items())
          and all(lin[t] == TR.LR for t in range(1, 20)), str({t: round(lin[t], 6) for t in (19, 20, 30, 39, 40)}))

    lg = {}
    TR.train_seq("shared", sids[:64], smask[:64], nv, steps=40, log_at=(), save_every=1,
                 save=lambda step, model, opt: lg.setdefault(step, opt.param_groups[0]["lr"]), schedule="coherence",
                 final_cooldown=0.5, lr_floor=0.0, final_cooldown_shape="log")
    k_ = TR.LOG_COOLDOWN_KAPPA
    want_g = {t: TR.LR * (1 - math.log(1 + (t - 20) / 20 / k_) / math.log(1 + 1 / k_)) for t in range(20, 41)}
    check("coherence, final_cooldown_shape log (nGPT 2026, varsayilan): son %50'de 1 - log(1 + p/k) / log(1 + 1/k) ile 0'a, "
          "k 0,05; inisin %10'unda tepenin %64'u, yarisinda %21'i",
          TR.FINAL_COOLDOWN_SHAPE == "log" and k_ == 0.05 and all(abs(lg[t] - v) < 1e-12 for t, v in want_g.items()) and all(lg[t] == TR.LR for t in range(1, 20))
          and abs(lg[22] / TR.LR - 0.639) < 1e-3 and abs(lg[30] / TR.LR - 0.212) < 1e-3, str({t: round(lg[t], 6) for t in (19, 20, 22, 30, 40)}))

    lrs_m, means = {}, {}
    grab_m = lambda step, model, opt: (lrs_m.setdefault(step, opt.param_groups[0]["lr"]),
                                       means.setdefault(step, opt.param_groups[0]["coherence_mean"]))
    mm, curve = TR.train_seq("shared", None, None, nv, steps=30, log_at=(0, 30), batches=batches, save_every=1, save=grab_m,
                             schedule="coherence", coherence_window=5, final_cooldown=0.05, final_cooldown_shape="sqrt")
    follows = all(abs(lrs_m[t] - TR.LR * min(max(means[t], floor), 1.0)) < 1e-12 for t in range(1, 28))
    check("coherence, mini-batch: ortalama 1'den ayrilir, lr = LR x ortalama (alt sinir LR_FLOOR); kayip iner",
          follows and 0 < means[27] < 1 and curve[-1]["nll"] < curve[0]["nll"],
          "ortalama %.3f  lr %.5f  %.3f -> %.3f" % (means[27], lrs_m[27], curve[0]["nll"], curve[-1]["nll"]))

    packs = {}
    keep = lambda step, model, opt: packs.setdefault(step, dict(step=step, model=copy.deepcopy(model.state_dict()),
                                                                   optimizer=copy.deepcopy(opt.state_dict())))
    kw = dict(batches=batches, schedule="coherence", coherence_window=3, final_cooldown=0.05, final_cooldown_shape="sqrt")
    whole, _ = TR.train_seq("shared", None, None, nv, steps=8, log_at=(), save_every=2, save=keep, **kw)
    res, _ = TR.train_seq("shared", None, None, nv, steps=8, log_at=(), checkpoint=packs[4], **kw)
    check("coherence: 4. adim paketinden surdurulen = kesintisiz, bit duzeyinde (ortalama pakette)",
          all(torch.equal(x, y) for x, y in zip(whole.state_dict().values(), res.state_dict().values()))
          and packs[4]["optimizer"]["param_groups"][0]["coherence_mean"] != 1.0)

    # tek gradyan kopyasi (g2 = (g1 + g2) - g1): her adimin c, rho, ortalama ve lr'si ile kirpilmis birlesik gradyani, ayni
    # agirliklardan iki ayri kopyayla (eski yol) bagimsiz hesapla ayni.  save(s): agirlik s. guncellemeden once, p.grad ve
    # model.coherence (s - 1). adimin
    rec = {}
    grab_r = lambda step, model, opt: rec.setdefault(step, dict(
        model=copy.deepcopy(model.state_dict()), coherence=dict(model.coherence),
        grad=[p_.grad.clone() for p_ in model.parameters() if p_.requires_grad]))
    kw = dict(batches=batches, schedule="coherence", coherence_window=3, final_cooldown=0.05, final_cooldown_shape="sqrt")
    run_r, curve_r = TR.train_seq("shared", None, None, nv, steps=6, log_at=tuple(range(7)), save_every=1, save=grab_r, **kw)
    start_r, _ = TR.train_seq("shared", None, None, nv, steps=0, log_at=(), **kw)
    ref_m = copy.deepcopy(run_r)
    named_r = [(k, p_) for k, p_ in ref_m.named_parameters() if p_.requires_grad]
    mean_ref, worst = 1.0, dict(rho=0.0, c=0.0, mean=0.0, lr=0.0, grad=0.0)
    signal = total = None                                   # yansiz ortalama: pay ve payda ayri (ilk olcumde 1 x boy)
    ratio_mean, ratio_gap = 1.0, 0.0                        # eski tahminci (oranlarin ortalamasi): farki gosterir
    for t in range(5):
        ref_m.load_state_dict(start_r.state_dict() if t == 0 else rec[t]["model"])
        ids_t, mask_t = batches(t)
        gs = []
        for h in (0, 1):
            ref_m.zero_grad()
            with TR.sdpa_kernel([TR.SDPBackend.MATH]):
                ref_m.loss(ids_t[h::2].contiguous(), mask_t[h::2].contiguous())[0].backward()
            gs.append([p_.grad.clone() for _, p_ in named_r])
        w0, w1 = (float(mask_t[h::2, 1:].sum()) for h in (0, 1))
        comb = [(w0 * a_ + w1 * b_) / (w0 + w1) for a_, b_ in zip(*gs)]
        dot = na = nb = 0.0
        for (k, p_), a_, b_ in zip(named_r, *gs):
            kind = k.split(".")[-1]
            if kind in ("W_query", "W_key", "W_fact_in", "W_fact_up", "W_value"):
                a_, b_ = (v - (v * p_.detach()).sum(1, keepdim=True) * p_.detach() for v in (a_, b_))
            elif kind in ("W_context", "W_fact_out"):
                a_, b_ = (v - (v * p_.detach()).sum(0, keepdim=True) * p_.detach() for v in (a_, b_))
            dot, na, nb = dot + float((a_ * b_).sum()), na + float((a_ * a_).sum()), nb + float((b_ * b_).sum())
        rho = max(dot, 0.0) / (max(dot, 0.0) + (na + nb - 2 * dot) / 4 + 1e-30)
        lr_ref = TR.LR * min(max(mean_ref, TR.LR_FLOOR), 1.0)
        now_s, now_t = max(dot, 0.0), max(dot, 0.0) + (na + nb - 2 * dot) / 4
        if total is None:
            signal, total = mean_ref * now_t, now_t
        signal, total = signal + (now_s - signal) / 3, total + (now_t - total) / 3
        mean_ref = signal / (total + 1e-30)
        ratio_mean += (rho - ratio_mean) / 3
        ratio_gap = max(ratio_gap, abs(ratio_mean - mean_ref))
        norm = float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g_) for g_ in comb])))
        comb = [g_ * min(TR.GRAD_CLIP / (norm + 1e-6), 1.0) for g_ in comb]       # clip_grad_norm_ gibi
        got = rec[t + 1]["coherence"]
        worst = dict(rho=max(worst["rho"], abs(got["rho"] - rho)),
                     c=max(worst["c"], abs(got["c"] - dot / (na * nb + 1e-30) ** 0.5)),
                     mean=max(worst["mean"], abs(got["mean"] - mean_ref)), lr=max(worst["lr"], abs(got["lr"] - lr_ref) / lr_ref),
                     grad=max(worst["grad"], max(float((x - y).abs().max() / (y.abs().max() + 1e-30)) for x, y in
                                                 zip(rec[t + 1]["grad"], comb))))
    check("coherence, tek gradyan kopyasi: 5 adimda rho, c, lr ve kirpilmis birlesik gradyan iki kopyali bagimsiz hesapla "
          "ayni (rho / lr ~1e-7; ayni guncelleme -> ayni egri); kayip iner",
          worst["rho"] < 1e-6 and worst["c"] < 1e-6 and worst["lr"] < 1e-6 and worst["grad"] < 1e-5
          and curve_r[-1]["nll"] < curve_r[0]["nll"], "  ".join("%s %.1e" % kv for kv in worst.items() if kv[0] != "mean"))
    g4 = packs[4]["optimizer"]["param_groups"][0]
    check("coherence, yansiz ortalama: ortalama(rho) = EMA(dot+) / EMA(dot+ + |g1 - g2|^2 / 4) (bagimsiz hesapla ~1e-7; ilk "
          "olcumde eski tahminciyle ayni baslar), oranlarin ortalamasindan farkli; pay ve payda pakette (surdurme yukarida)",
          worst["mean"] < 1e-6 and ratio_gap > 1e-6 and g4.get("coherence_signal") is not None
          and abs(g4["coherence_mean"] - g4["coherence_signal"] / (g4["coherence_total"] + 1e-30)) < 1e-15,
          "fark %.1e  eski tahminciden en buyuk ayrilma %.1e" % (worst["mean"], ratio_gap))



def t_weight_ema():
    """WEIGHT_EMA: ortalama <- d x ortalama + (1 - d) x agirlik, kurede satir / sutunlar yeniden birim; surdurme (ortalama
    optimizer durumunda) bit duzeyinde; gecersiz d reddedilir."""
    import copy
    import torch.nn.functional as F
    data = synthetic_data()
    sids, smask = sequences(data)
    nv = len(data["vocab"])

    start, _ = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=0, log_at=())
    one, _ = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=1, log_at=(), weight_ema=0.5)
    em = one.weight_ema["model"]
    b0, b1, be = start.blocks[0], one.blocks[0], em.blocks[0]
    with torch.no_grad():
        q = F.normalize(0.5 * b0.attention.W_query + 0.5 * b1.attention.W_query, dim=1)
        ctx = F.normalize(0.5 * b0.attention.W_context + 0.5 * b1.attention.W_context, dim=0)
        sh = 0.5 * start.tokens.shift + 0.5 * one.tokens.shift
    err = max(float((q - be.attention.W_query).abs().max()), float((ctx - be.attention.W_context).abs().max()),
              float((sh - em.tokens.shift).abs().max()))
    check("weight_ema: bir adim sonra ortalama = norm(d x baslangic + (1 - d) x agirlik) (satirlar W_query, sutunlar "
          "W_context; shift normalizesiz); ortalama model agirliktan farkli", err < 1e-6
          and not torch.equal(be.attention.W_query, b1.attention.W_query), "fark %.1e" % err)

    def batches(step):
        rows = torch.randperm(len(sids), generator=torch.Generator().manual_seed(2000 + step))[:16]
        return sids[rows], smask[rows]
    packs = {}
    keep = lambda step, model, opt: packs.setdefault(step, dict(step=step, model=copy.deepcopy(model.state_dict()),
                                                                   optimizer=copy.deepcopy(opt.state_dict())))
    kw = dict(batches=batches, schedule="coherence", weight_ema=0.9, final_cooldown=0.05, final_cooldown_shape="sqrt")
    whole, _ = TR.train_seq("shared", None, None, nv, steps=8, log_at=(), save_every=2, save=keep, **kw)
    res, _ = TR.train_seq("shared", None, None, nv, steps=8, log_at=(), checkpoint=packs[4], **kw)
    same = lambda a, b: all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    unit = all(torch.allclose(b.attention.W_query.norm(dim=1), torch.ones(64), atol=1e-5)
               for b in whole.weight_ema["model"].blocks)
    try:
        TR.train_seq("shared", sids[:8], smask[:8], nv, steps=1, log_at=(), weight_ema=1.5)
        refused = False
    except AssertionError:
        refused = True
    check("weight_ema: 4. adim paketinden surdurulen = kesintisiz, bit duzeyinde (model VE ortalama); ortalamanin satirlari "
          "birim; d = 1,5 reddedilir",
          same(whole, res) and same(whole.weight_ema["model"], res.weight_ema["model"]) and unit and refused)


def t_heads():
    """HEADS (> 1): head basina q, k birim + RoPE, V'nin dilimi, head'ler yan yana -> bagimsiz hesapla ayni; nedensel;
    W_value birim baslar, kurede satirlari birim, Muon'da; egitim, surdurme; uretim."""
    import copy
    import torch.nn.functional as F
    from model_y import BlockModel, CausalAttention, apply_rope
    data = synthetic_data()
    sids, smask = sequences(data)
    nv = len(data["vocab"])
    ids = sids[:6, :20]

    default, four = BlockModel(nv), BlockModel(nv, heads=4)
    with torch.no_grad():
        same = torch.equal(four.logits(ids), default.logits(ids))
    check("heads: varsayilan 4 (heads=4 ile bit duzeyinde ayni, W_value var)",
          same and default.heads == 4 and any("W_value" in k for k in default.state_dict()))

    g = torch.Generator().manual_seed(61)
    at = CausalAttention(64, rope=True, heads=4).double()
    with torch.no_grad():
        for p_ in at.parameters():
            p_.copy_(torch.randn(p_.shape, generator=g, dtype=torch.float64))
        x = torch.randn(3, 20, 64, generator=g, dtype=torch.float64)
        out, T, dh = at(x), 20, 16
        parts = []
        for h in range(4):                                  # head h: W_query / W_key / W_value'nun h. satir dilimi
            sl = slice(h * dh, (h + 1) * dh)
            q = apply_rope(F.normalize(x @ at.W_query[sl].T, dim=-1), None, at.rope_base)
            k = apply_rope(F.normalize(x @ at.W_key[sl].T, dim=-1), None, at.rope_base)
            s = (at.scale * q @ k.transpose(-1, -2)).masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
            parts.append(torch.softmax(s, -1) @ (x @ at.W_value[sl].T))
        err = float((out - torch.cat(parts, -1)).abs().max())
    check("heads: 4 head'li attention = bagimsiz hesap (head basina q, k birim + RoPE, V dilimi, yan yana)", err < 1e-10,
          "fark %.1e" % err)

    m4 = BlockModel(nv, heads=4)
    alt = ids.clone()
    alt[:, 5] = (alt[:, 5] + 1) % nv
    with torch.no_grad():
        a_, b_ = m4.logits(ids), m4.logits(alt)
    refused = []
    for make in (lambda: CausalAttention(64, heads=5), lambda: CausalAttention(12, heads=4, rope=True),
                 lambda: CausalAttention(64, heads=1)):
        try:
            make()
            refused.append(False)
        except AssertionError:
            refused.append(True)
    check("heads: nedensel; W_value birim baslar (d x d); reddedilir: d head'e bolunmezse, RoPE'de tek "
          "sayili head boyu, tek head",
          torch.equal(a_[:, :5], b_[:, :5]) and not torch.equal(a_[:, 5:], b_[:, 5:])
          and all(torch.equal(b.attention.W_value, torch.eye(64)) for b in m4.blocks) and all(refused))

    opts = []
    grab = lambda step, model, opt: opts.append(opt)
    kw = dict(model_kw=dict(heads=4))
    trained, curve = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=20, log_at=(0, 20), save_every=1, save=grab, **kw)
    names = {id(p_): k for k, p_ in trained.named_parameters()}
    in_muon = {names[id(p_)].split(".")[-1] for g_ in opts[-1].param_groups if g_["use_muon"] for p_ in g_["params"]}
    unit = all(torch.allclose(b.attention.W_value.norm(dim=1), torch.ones(64), atol=1e-5) for b in trained.blocks)
    packs = {}
    keep = lambda step, model, opt: packs.setdefault(step, dict(step=step, model=copy.deepcopy(model.state_dict()),
                                                                   optimizer=copy.deepcopy(opt.state_dict())))
    full, _ = TR.train_seq("shared", sids[:40], smask[:40], nv, steps=6, log_at=(), save_every=2, save=keep, **kw)
    res, _ = TR.train_seq("shared", sids[:40], smask[:40], nv, steps=6, log_at=(), checkpoint=packs[4], **kw)
    qs = [[1] + sids[i, 1:9].tolist() for i in range(6)]

    def greedy(model, prompt, k):                           # elle acgozlu uretim: her adimda logits'in en buyugu
        x = list(prompt)
        with torch.no_grad():
            for _ in range(k):
                x.append(int(model.logits(torch.tensor([x]))[0, -1].argmax()))
        return x[len(prompt):]
    check("heads: egitimde kayip iner, W_value Muon'da ve satirlari birim; surdurme bit duzeyinde; onbellekli uretim "
          "elle acgozlu uretimle ayni", curve[-1]["nll"] < curve[0]["nll"] and "W_value" in in_muon and unit
          and all(torch.equal(a, b) for a, b in zip(full.state_dict().values(), res.state_dict().values()))
          and TR.generate(trained, qs, 5) == [greedy(trained, q, 5) for q in qs],
          "%.3f -> %.3f" % (curve[0]["nll"], curve[-1]["nll"]))


def t_fact_activation():
    """FactUnits (SwiGLU): u = SiLU(W_fact_in x) * (W_fact_up x), esik yok; W_fact_up Muon'da ve satirlari birim; surdurme
    bit duzeyinde; onbellekli uretim; relu / reglu reddedilir."""
    import copy
    import functools
    from model_y import FACT_UNITS, BlockModel, FactUnits
    BlockModel = functools.partial(BlockModel, input_embedding=False, first_turn_facts=True)   # butun turlarda FactUnits
    data = synthetic_data()
    sids, smask = sequences(data)
    nv = len(data["vocab"])
    ids = sids[:6, :20]

    base = BlockModel(nv)
    names = {k.split(".")[-1] for k in dict(base.named_parameters())}
    try:
        BlockModel(nv, fact_activation="relu")
        refused = False
    except AssertionError:
        refused = True
    check("FactUnits: birim 170 (8/3 x D); W_fact_up var, fact_threshold yok; fact_activation='relu' reddedilir",
          FACT_UNITS == 170 and base.blocks[0].facts.W_fact_in.shape[0] == 170 and "W_fact_up" in names
          and "fact_threshold" not in names and refused)

    g = torch.Generator().manual_seed(71)
    f = FactUnits(16, 12).double()
    with torch.no_grad():
        for p_ in f.parameters():
            p_.copy_(torch.randn(p_.shape, generator=g, dtype=torch.float64))
        x = torch.randn(5, 16, generator=g, dtype=torch.float64)
        gate, up = x @ f.W_fact_in.T, x @ f.W_fact_up.T
        ref = (gate / (1 + torch.exp(-gate)) * up) @ f.W_fact_out.T
        err = float((f(x) - ref).abs().max())
    check("FactUnits: cikti = W_fact_out (SiLU(W_fact_in x) * (W_fact_up x)) (bagimsiz float64)", err < 1e-12,
          "fark %.1e" % err)

    opts = []
    grab = lambda step, model, opt: opts.append(opt)
    kw = dict(model_kw=dict(units=170, first_turn_facts=True))
    trained, curve = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=20, log_at=(0, 20), save_every=1, save=grab, **kw)
    nm = {id(p_): k.split(".")[-1] for k, p_ in trained.named_parameters()}
    in_muon = {nm[id(p_)] for g_ in opts[-1].param_groups if g_["use_muon"] for p_ in g_["params"]}
    unit = all(torch.allclose(b.facts.W_fact_up.norm(dim=1), torch.ones(170), atol=1e-5) for b in trained.blocks)
    packs = {}
    keep = lambda step, model, opt: packs.setdefault(step, dict(step=step, model=copy.deepcopy(model.state_dict()),
                                                                   optimizer=copy.deepcopy(opt.state_dict())))
    full, _ = TR.train_seq("shared", sids[:40], smask[:40], nv, steps=6, log_at=(), save_every=2, save=keep, **kw)
    res, _ = TR.train_seq("shared", sids[:40], smask[:40], nv, steps=6, log_at=(), checkpoint=packs[4], **kw)
    qs = [[1] + sids[i, 1:1 + k].tolist() for i, k in enumerate((1, 3, 5, 8, 12, 16))]
    check("FactUnits: egitimde kayip iner; W_fact_up Muon'da ve satirlari birim; surdurme bit duzeyinde; "
          "onbellekli uretim tam hesapla ayni",
          curve[-1]["nll"] < curve[0]["nll"] and "W_fact_up" in in_muon and unit
          and all(torch.equal(a, b) for a, b in zip(full.state_dict().values(), res.state_dict().values()))
          and TR.generate(trained, qs, 6) == TR.generate(trained, qs, 6, cached=False),
          "%.3f -> %.3f" % (curve[0]["nll"], curve[-1]["nll"]))


def t_learn_output_scale():
    """Ogrenilen cikis olcegi: skor = e^tau <h, PL>, tau = log_output_scale ln(scale)'dan; Adam'da (Muon'da degil), weight
    decay yok, weight EMA ortalar; egitim, surdurme, onbellekli uretim; konus eski config'leri (internals_y._build) acar,
    sabit olcekli (learn_output_scale=False) config'i acik hatayla reddeder.  Kosucunun sinav gunlugu (olcek):
    tests_simplestories."""
    import copy
    import json
    import tempfile
    from model_y import BlockModel
    data = synthetic_data()
    sids, smask = sequences(data)
    nv = len(data["vocab"])
    ids = sids[:6, :20]
    val = lambda t: float(t.detach())

    on = BlockModel(nv)
    check("cikis olcegi: tau = log_output_scale ln(scale)'dan baslar", val(on.log_output_scale) == float(torch.tensor(math.log(on.scale))))

    # skor = e^tau <h, PL>: tau = ln 15 iken tau = ln(scale)'daki skor x 15 / scale (float64)
    a = BlockModel(nv, d=32, units=48).double()
    with torch.no_grad():
        a.log_output_scale.fill_(math.log(a.scale))                # float64'te tam ln(scale)
        z0 = a.logits(ids)
        a.log_output_scale.fill_(math.log(15.0))
        err = float((a.logits(ids) - 15.0 / a.scale * z0).abs().max())
    check("cikis olcegi: skor = e^tau phi(<h, PL>) (tau = ln 15: baslangic skoru x 15 / scale, float64)", err < 1e-10,
          "fark %.1e" % err)

    opts = []
    grab = lambda step, model, opt: opts.append(opt)
    trained, curve = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=20, log_at=(0, 20), save_every=1, save=grab)
    names = {id(p_): k for k, p_ in trained.named_parameters()}
    in_adam = {names[id(p_)] for g_ in opts[-1].param_groups if not g_["use_muon"] for p_ in g_["params"]}
    in_muon = {names[id(p_)] for g_ in opts[-1].param_groups if g_["use_muon"] for p_ in g_["params"]}
    start, _ = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=0, log_at=())
    one, _ = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=1, log_at=(), weight_ema=0.5)
    ema_err = abs(val(one.weight_ema["model"].log_output_scale) - 0.5 * (val(start.log_output_scale) + val(one.log_output_scale)))
    moved = val(trained.log_output_scale) - val(on.log_output_scale)
    check("cikis olcegi: Adam'da (Muon'da degil); weight EMA ortalar; 20 adimda deger degisir, kayip iner",
          "log_output_scale" in in_adam and "log_output_scale" not in in_muon
          and ema_err < 1e-6 and abs(moved) > 1e-3 and curve[-1]["nll"] < curve[0]["nll"],
          "olcek %.3f -> %.3f  kayip %.3f -> %.3f" % (math.exp(val(on.log_output_scale)), math.exp(val(trained.log_output_scale)),
                                                     curve[0]["nll"], curve[-1]["nll"]))

    packs = {}
    keep = lambda step, model, opt: packs.setdefault(step, dict(step=step, model=copy.deepcopy(model.state_dict()),
                                                                   optimizer=copy.deepcopy(opt.state_dict())))
    full, _ = TR.train_seq("shared", sids[:40], smask[:40], nv, steps=6, log_at=(), save_every=2, save=keep)
    res, _ = TR.train_seq("shared", sids[:40], smask[:40], nv, steps=6, log_at=(), checkpoint=packs[4])
    qs = [[1] + sids[i, 1:1 + k].tolist() for i, k in enumerate((1, 3, 5, 8, 12, 16))]
    new = TR.generate(trained, qs, 6)
    scores, _ = cached_scores(trained, qs, new)
    err_c = max(float((scores[i] - trained.logits(torch.tensor([q + new[i]])).detach()[0, len(q) - 1:]).abs().max())
                for i, q in enumerate(qs))
    check("cikis olcegi: 4. adim paketinden surdurulen = kesintisiz, bit duzeyinde (tau pakette); onbellekli uretim "
          "= tam hesap (token'lar ve skorlar)",
          "log_output_scale" in packs[4]["model"]
          and all(torch.equal(x, y) for x, y in zip(full.state_dict().values(), res.state_dict().values()))
          and new == TR.generate(trained, qs, 6, cached=False) and err_c < 1e-4, "skor farki %.1e" % err_c)

    # konus: yeni kosu ve anahtarsiz eski config acilir; sabit olcekli kosu acik hatayla reddedilir
    sys.path.insert(0, os.path.join(HERE, "train_tinystories"))
    import konus_y as K
    tmp = tempfile.mkdtemp()
    legacy = BlockModel(nv, output_link=False, shared_facts=True, input_embedding=False, first_turn_facts=True,
                        rope_base=10000, attention_log_scale=False)   # 29 Eylul oncesi kosu (RoPE tabani sabit 10.000)
    for sub, model_ in (("old", on), ("new", trained), ("legacy", legacy)):
        os.makedirs(os.path.join(tmp, sub))
        kw_ = dict(d=64, turns=4, layers=2, heads=4, fact_activation="swiglu", units=170, t_max=512)
        if sub != "legacy":                                  # kosucular (29 Eylul'den) iki anahtari hep yazar
            kw_.update(output_link=model_.output_link, shared_facts=model_.shared_facts,
                       input_embedding=hasattr(model_, "input_embedding"), first_turn_facts=model_.first_turn_facts,
                       input_embedding_sphere=model_.input_embedding_sphere,
                       rope_base=model_.blocks[0].attention.rope_base,    # kosucular (30 Eylul'den) tabani sayi yazar
                       attention_log_scale=model_.blocks[0].attention.attention_log_scale)
        kw_["learn_output_scale"] = sub != "old"
        json.dump(dict(setting="shared", model_kw=kw_, stream_norm=True, layer_norm=False, rope=True, normalized_update=True,
                       sphere_weights=True, canon=True, steps=20), open(os.path.join(tmp, sub, "config.json"), "w"))
        torch.save(model_.state_dict(), os.path.join(tmp, sub, "model.pt"))
    try:
        K.load_model(os.path.join(tmp, "old"), nv)
        refused = False
    except AssertionError:
        refused = True
    lnew, _ = K.load_model(os.path.join(tmp, "new"), nv)
    lleg, _ = K.load_model(os.path.join(tmp, "legacy"), nv)      # anahtarsiz config: phi kapali, FactUnits paylasimli
    with torch.no_grad():
        loaded = (torch.equal(lnew.logits(ids), trained.logits(ids)) and not lleg.output_link and lleg.shared_facts
                  and torch.equal(lleg.logits(ids), legacy.logits(ids)))
    check("konus (internals_y._build): yeni kosu ve anahtarsiz (29 Eylul oncesi) config acilir (phi kapali, paylasimli "
          "FactUnits); learn_output_scale=False config'i acik hatayla reddedilir", loaded and refused)


def t_matmul_precision():
    """MATMUL_PRECISION: varsayilan "bf16"; CPU'da etkisiz (fp32 / bf16 bit duzeyinde ayni egitim, float32 matmul
    ayari degismez); gecersiz deger reddedilir.  Kosucunun config'e yazmasi ve train_seq'e iletmesi: tests_simplestories.
    GPU yolu (autocast bf16, TF32) yerelde sinanamaz."""
    import inspect
    data = synthetic_data()
    sids, smask = sequences(data)
    nv = len(data["vocab"])
    before = torch.get_float32_matmul_precision()
    runs = {p: TR.train_seq("shared", sids[:40], smask[:40], nv, steps=4, log_at=(), matmul_precision=p)[0]
            for p in ("fp32", "bf16")}
    same = all(torch.equal(a, b) for p in ("bf16",)
               for a, b in zip(runs["fp32"].state_dict().values(), runs[p].state_dict().values()))
    try:
        TR.train_seq("shared", sids[:8], smask[:8], nv, steps=1, log_at=(), matmul_precision="fp16")
        refused = False
    except AssertionError:
        refused = True
    check("matmul_precision: varsayilan 'bf16'; CPU'da fp32 / bf16 bit duzeyinde ayni egitim, float32 matmul ayari "
          "degismez; 'fp16' reddedilir",
          TR.MATMUL_PRECISION == "bf16" and inspect.signature(TR.train_seq).parameters["matmul_precision"].default == "bf16"
          and same and torch.get_float32_matmul_precision() == before and refused)


def t_loss_chunk():
    """LOSS_CHUNK: egitim kaybi sozluk parcalariyla, gradyan ileri hesapta.  float64'te parca 4096 / 7 / 1 ile kayip ve butun
    gradyanlar tek parca (0) yoluyla ayni; fp32 egitim egrisi yakin; surdurme bit duzeyinde; tam logits tablosu olusmaz
    (tepe tensor ve geri yayilima saklanan tensor).  Kosucunun config'e yazmasi: tests_simplestories."""
    import copy
    from model_y import LOSS_CHUNK, BlockModel
    data = synthetic_data()
    sids, smask = sequences(data)
    nv = len(data["vocab"])
    ids, mask = sids[:8, :24], smask[:8, :24]

    worst, missing = 0.0, []
    g = torch.Generator().manual_seed(91)
    for kw in (dict(),):
        base = BlockModel(nv, d=32, units=48, loss_chunk=0, output_link=False, **kw).double()   # parcali yol phi'siz
        with torch.no_grad():
            for p_ in base.parameters():
                p_.add_(0.1 * torch.randn(p_.shape, generator=g, dtype=torch.float64))
        total0, nll0 = base.loss(ids, mask)
        total0.backward()
        ref = {k: p_.grad for k, p_ in base.named_parameters()}
        for chunk in (4096, 7, 1):
            m = copy.deepcopy(base)
            m.loss_chunk = chunk
            m.zero_grad()
            total, nll = m.loss(ids, mask)
            total.backward()
            missing += [k for k, p_ in m.named_parameters() if p_.grad is None]
            worst = max([worst, abs(float(nll.detach() - nll0.detach()))]
                        + [float((p_.grad - ref[k]).abs().max()) for k, p_ in m.named_parameters() if p_.grad is not None])
    check("loss_chunk: varsayilan 4096; float64'te parca 4096 / 7 / 1 ile kayip ve BUTUN gradyanlar (tau, shift dahil) tek "
          "parca yoluyla <= 1e-10 ayni -- dolgulu batch",
          LOSS_CHUNK == 4096 and not missing and worst < 1e-10 and bool((~mask).any()), "en buyuk fark %.1e" % worst)

    c0 = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=20, log_at=tuple(range(21)),
                      model_kw=dict(loss_chunk=0, output_link=False))[1]
    c1 = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=20, log_at=tuple(range(21)), model_kw=dict(output_link=False))[1]
    c7 = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=20, log_at=tuple(range(21)),
                      model_kw=dict(loss_chunk=7, output_link=False))[1]
    gap = max(abs(a["nll"] - b["nll"]) for c in (c1, c7) for a, b in zip(c0, c)) / c0[-1]["nll"]
    packs = {}
    keep = lambda step, model, opt: packs.setdefault(step, dict(step=step, model=copy.deepcopy(model.state_dict()),
                                                                   optimizer=copy.deepcopy(opt.state_dict())))
    kw = dict(model_kw=dict(loss_chunk=7, output_link=False))
    full, _ = TR.train_seq("shared", sids[:40], smask[:40], nv, steps=6, log_at=(), save_every=2, save=keep, **kw)
    res, _ = TR.train_seq("shared", sids[:40], smask[:40], nv, steps=6, log_at=(), checkpoint=packs[4], **kw)
    check("loss_chunk: fp32 CPU'da 20 adim egri tek parca yoluna yakin (4096 ve 7); parcali (7) egitimde 4. adim paketinden "
          "surdurulen = kesintisiz, bit duzeyinde",
          gap < 1e-4 and c1[-1]["nll"] < c1[0]["nll"]
          and all(torch.equal(x, y) for x, y in zip(full.state_dict().values(), res.state_dict().values())),
          "en buyuk goreli fark %.1e  kayip %.3f -> %.3f" % (gap, c1[0]["nll"], c1[-1]["nll"]))

    # tam logits tablosu (N x V) olusmuyor: buyuk sozlukte en buyuk tek ayirma ve geri yayilima saklanan en buyuk tensor
    V, gr = 4000, torch.Generator().manual_seed(5)
    ids_v = torch.randint(0, V, (16, 30), generator=gr)
    mask_v = torch.ones(16, 30, dtype=torch.bool)
    mask_v[:8, 20:] = False
    table = ids_v.shape[0] * (ids_v.shape[1] - 1) * V
    peak, saved = {}, {}
    for chunk in (0, 256):
        mv = BlockModel(V, loss_chunk=chunk, output_link=False)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as prof:
            mv.loss(ids_v, mask_v)[0].backward()
        peak[chunk] = max(e.self_cpu_memory_usage for e in prof.events()) / 4          # fp32 eleman
        sizes = []
        with torch.autograd.graph.saved_tensors_hooks(lambda t: (sizes.append(t.numel()), t)[1], lambda t: t):
            mv.loss(ids_v, mask_v)
        saved[chunk] = max(sizes)
    check("loss_chunk: tam logits tablosu olusmaz -- V 4000, N 464: tek parcada en buyuk ayirma ve saklanan tensor = N x V; "
          "parcali (256) yolda ikisi de < N x V / 3 (en buyugu P ve gradyani, V x d)",
          peak[0] >= table and saved[0] >= table and peak[256] < table / 3 and saved[256] < table / 3,
          "tablo %d  tepe %d / %d  saklanan %d / %d" % (table, peak[0], peak[256], saved[0], saved[256]))


def t_muon_tangent():
    """Muon'un kure tegeti: Muon'a giren Nesterov birlesimi once kure tegetine (satirlari birim: W_value, W_fact_in,
    W_fact_up; sutunlari birim: W_context, W_fact_out).  Bir adim bagimsiz elle hesapla ayni; izdusulen girdi agirlikla dik;
    egitim, surdurme."""
    import copy
    import torch.nn.functional as F
    data = synthetic_data()
    sids, smask = sequences(data)
    nv = len(data["vocab"])
    ids, mask = sids[:40], smask[:40]
    rows, cols = ("W_value", "W_fact_in", "W_fact_up"), ("W_context", "W_fact_out")

    start, _ = TR.train_seq("shared", ids, mask, nv, steps=0, log_at=())
    ref = copy.deepcopy(start)
    ref.zero_grad()
    with TR.sdpa_kernel([TR.SDPBackend.MATH]):
        ref.loss(ids, mask)[0].backward()
    torch.nn.utils.clip_grad_norm_([p_ for p_ in ref.parameters() if p_.requires_grad], TR.GRAD_CLIP)
    seen, saved = [], TR.Muon.__dict__["orthogonalize"]              # staticmethod nesnesi: aynen geri konur
    real = saved.__func__
    TR.Muon.orthogonalize = staticmethod(lambda G, steps=5, precision="fp32": (seen.append(G.clone()),
                                                                            real(G, steps, precision))[1])
    worst, dots = 0.0, []
    try:
        m, _ = TR.train_seq("shared", ids, mask, nv, steps=1, log_at=())
        flat = [x for G in seen for x in (G.unbind(0) if G.dim() == 3 else [G])]   # Muon ayni boyu yigin halinde verir
        for (k, p0), p1 in zip(ref.named_parameters(), m.parameters()):
            kind = k.split(".")[-1]
            if kind not in rows + cols:
                continue
            axis = 1 if kind in rows else 0
            v = p0.grad.add(p0.grad, alpha=0.95)                # ilk adim: tampon = g, Nesterov g + 0,95 g
            v = v - (v * p0.detach()).sum(axis, keepdim=True) * p0.detach()
            G = next((x for x in flat if x.shape == v.shape and torch.allclose(x, v, rtol=1e-5, atol=1e-7)), None)
            dots.append(float("inf") if G is None else float((G * p0.detach()).sum(axis).abs().max() / G.norm(dim=axis).max()))
            new = p0.detach().clone().add_(real(v), alpha=-TR.LR * 0.2 * max(p0.shape) ** 0.5)
            worst = max(worst, float((F.normalize(new, dim=axis) - p1.detach()).abs().max()))
    finally:
        TR.Muon.orthogonalize = saved
    check("muon kure tegeti: bir adim bagimsiz elle hesapla ayni (once teget, sonra ortogonallestirme); izdusulen girdi "
          "agirligin satir / sutunlariyla dik (goreli < 1e-5)",
          worst < 1e-6 and dots and max(dots) < 1e-5, "fark %.1e  en buyuk <G, w> %.1e" % (worst, max(dots) if dots else -1))

    opts = []
    grab = lambda step, model, opt: opts.append(opt)
    trained, curve = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=20, log_at=(0, 20), save_every=1, save=grab)
    names = {id(p_): k.split(".")[-1] for k, p_ in trained.named_parameters()}
    axes = {names[id(p_)]: a for p_, a in opts[-1].tangent_axis.items()}
    packs = {}
    keep = lambda step, model, opt: packs.setdefault(step, dict(step=step, model=copy.deepcopy(model.state_dict()),
                                                                   optimizer=copy.deepcopy(opt.state_dict())))
    full, _ = TR.train_seq("shared", ids, mask, nv, steps=6, log_at=(), save_every=2, save=keep)
    res, _ = TR.train_seq("shared", ids, mask, nv, steps=6, log_at=(), checkpoint=packs[4])
    check("muon kure tegeti: tablo tek yerde (coherence ile ortak): satir W_query, W_key, W_value, W_fact_in, W_fact_up; sutun "
          "W_context, W_fact_out; 20 adimda kayip iner; surdurme bit duzeyinde",
          axes == dict(W_query=1, W_key=1, W_value=1, W_fact_in=1, W_fact_up=1, W_context=0, W_fact_out=0)
          and curve[-1]["nll"] < curve[0]["nll"]
          and all(torch.equal(x, y) for x, y in zip(full.state_dict().values(), res.state_dict().values())),
          "%.3f -> %.3f  %s" % (curve[0]["nll"], curve[-1]["nll"], sorted(axes.items())))


def t_output_link():
    """OUTPUT_LINK: skor = s phi(c), phi(c) = c (1 + c (q + c (q^2/3 + u))); varsayilan False; q = u = 0'da dogrusal skorla
    ayni; phi' >= 0 (u her adimdan sonra kirpilir); kayip tam tablo yolundan; Adam'da; egitim, surdurme, onbellekli uretim."""
    import copy
    import inspect
    from model_y import OUTPUT_LINK, BlockModel, masked_nll
    data = synthetic_data()
    sids, smask = sequences(data)
    nv = len(data["vocab"])
    ids = sids[:6, :20]

    base, link = BlockModel(nv, output_link=False), BlockModel(nv, output_link=True)
    with torch.no_grad():
        err0 = float((base.logits(ids) - link.logits(ids)).abs().max())
    rest = {k: v for k, v in link.state_dict().items() if k not in ("link_q", "link_u")}
    check("output_link: varsayilan True (29 Eylul); link_q, link_u 0'dan, geri kalan agirliklar ayni; q = u = 0'da skor dogrusal "
          "skorla ayni (yuvarlama icinde)",
          OUTPUT_LINK is True and inspect.signature(BlockModel).parameters["output_link"].default is True
          and not base.output_link and link.output_link and "link_q" not in base.state_dict()
          and list(rest) == list(base.state_dict()) and all(torch.equal(rest[k], v) for k, v in base.state_dict().items())
          and float(link.link_q) == 0.0 and float(link.link_u) == 0.0 and err0 < 1e-5, "fark %.1e" % err0)

    a = BlockModel(nv, d=32, units=48, output_link=True).double()
    q, u = -0.7, 0.3
    with torch.no_grad():
        a.link_q.fill_(q)
        a.link_u.fill_(u)
        P = a.tokens.points()
        c = a.hidden(ids)[-1] @ P.T
        s = a.scale * torch.exp(a.log_output_scale - math.log(a.scale))
        err = float((a.logits(ids) - s * (c + q * c ** 2 + (q * q / 3 + u) * c ** 3)).abs().max())
    grid = torch.linspace(-1, 1, 2001, dtype=torch.float64)
    check("output_link: skor = s (c + q c^2 + (q^2/3 + u) c^3) (float64); u >= 0'da phi' = (1 + q c)^2 + 3 u c^2 >= 0",
          err < 1e-10 and bool((((1 + q * grid) ** 2 + 3 * u * grid ** 2) >= 0).all()), "fark %.1e" % err)

    m = BlockModel(nv, output_link=True)
    with torch.no_grad():
        m.link_q.fill_(0.5)
        m.link_u.fill_(0.2)
    _, nll = m.loss(sids[:8], smask[:8])
    ref = masked_nll(m.logits(sids[:8, :-1]), sids[:8, 1:], smask[:8, 1:])
    mc = BlockModel(nv, d=32, units=48, loss_chunk=7, output_link=True).double()
    with torch.no_grad():
        mc.link_q.fill_(-1.3)
        mc.link_u.fill_(0.2)
    mf = copy.deepcopy(mc)
    mf.loss_chunk = 0
    _, lc = mc.loss(sids[:6].clone(), smask[:6])
    _, lf = mf.loss(sids[:6].clone(), smask[:6])
    lc.backward()
    lf.backward()
    gerr = max(float((p_.grad - r_.grad).abs().max()) for p_, r_ in zip(mc.parameters(), mf.parameters())
               if p_.grad is not None)
    check("output_link: parcali kayip (29 Eylul) = tam tablo, deger ve butun gradyanlar (link_q, link_u, olcek, noktalar "
          "dahil) float64'te <= 1e-10", float((lc - lf).abs()) < 1e-10 and gerr < 1e-10 and mc.link_q.grad is not None,
          "deger %.1e  gradyan %.1e" % (float((lc - lf).abs()), gerr))
    breaks = {}
    for link_ in (True, False):
        torch._dynamo.reset()
        ex = torch._dynamo.explain(BlockModel(nv, d=32, units=48, loss_chunk=7, output_link=link_).loss)(sids[:4], smask[:4])
        breaks[link_] = ex.graph_break_count
    torch._dynamo.reset()
    check("output_link: parcali kayip compile'da tek grafik (graph break yok; phi'de float(q) kiriyordu: GPU'da 166 -> 274 "
          "ms/adim, 29 Eylul)", breaks == {True: 0, False: 0}, str(breaks))
    check("output_link: LOSS_CHUNK > 0 iken kayip (parcali yol) = masked_nll(logits)",
          m.loss_chunk > 0 and float((nll - ref).abs()) < 1e-6, "fark %.1e" % float((nll - ref).abs()))

    opts = []
    grab = lambda step, model, opt: opts.append(opt)
    kw = dict(model_kw=dict(output_link=True))
    trained, curve = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=20, log_at=(0, 20), save_every=1, save=grab, **kw)
    nm = {id(p_): k for k, p_ in trained.named_parameters()}
    in_adam = {nm[id(p_)] for g_ in opts[-1].param_groups if not g_["use_muon"] for p_ in g_["params"]}
    packs = {}
    keep = lambda step, model, opt: packs.setdefault(step, dict(step=step, model=copy.deepcopy(model.state_dict()),
                                                                   optimizer=copy.deepcopy(opt.state_dict())))
    full, _ = TR.train_seq("shared", sids[:40], smask[:40], nv, steps=6, log_at=(), save_every=2, save=keep, **kw)
    res, _ = TR.train_seq("shared", sids[:40], smask[:40], nv, steps=6, log_at=(), checkpoint=packs[4], **kw)
    bad = copy.deepcopy(packs[4])
    bad["model"]["link_u"] = torch.tensor(-0.5)
    clamped, _ = TR.train_seq("shared", sids[:40], smask[:40], nv, steps=5, log_at=(), checkpoint=bad, **kw)
    qs = [[1] + sids[i, 1:1 + k].tolist() for i, k in enumerate((1, 3, 5, 8, 12, 16))]
    check("output_link: Adam'da; egitimde kayip iner, q oynar; u eksiden baslatilinca bir adimda >= 0'a kirpilir; surdurme "
          "bit duzeyinde; onbellekli uretim = tam hesap",
          {"link_q", "link_u"} <= in_adam and curve[-1]["nll"] < curve[0]["nll"] and abs(float(trained.link_q)) > 0
          and float(trained.link_u) >= 0 and float(clamped.link_u) >= 0
          and all(torch.equal(x, y) for x, y in zip(full.state_dict().values(), res.state_dict().values()))
          and TR.generate(trained, qs, 6) == TR.generate(trained, qs, 6, cached=False),
          "q %.4f u %.4f  kirpilan u %.4f  kayip %.3f -> %.3f" % (float(trained.link_q), float(trained.link_u),
                                                               float(clamped.link_u), curve[0]["nll"], curve[-1]["nll"]))


def t_shared_facts():
    """SHARED_FACTS: varsayilan False (29 Eylul'den; True: ek FactUnits yok); False'ta her turun kendi FactUnits'i, attention paylasimli
    kalir: attention'i ikiser bagli ayri bloklu modelle ayni skor (float64); parametre + (turns - layers) FactUnits;
    ek FactUnits kurede ve Muon'da; egitim, surdurme, onbellekli uretim."""
    import copy
    import inspect
    from model_y import SHARED_FACTS, BlockModel
    data = synthetic_data()
    sids, smask = sequences(data)
    nv = len(data["vocab"])
    kw = dict(d=16, units=16, layers=2, turns=4, heads=2, input_embedding=False, first_turn_facts=True)
    base, split = BlockModel(nv, shared_facts=True, **kw), BlockModel(nv, **kw)
    cnt = lambda m: sum(p_.numel() for p_ in m.parameters() if p_.requires_grad)
    fu = sum(p_.numel() for p_ in split.extra_facts[0].parameters())
    check("shared_facts: varsayilan False (29 Eylul; True'da ek FactUnits yok); False'ta turns - layers ek FactUnits, parametre tam o kadar "
          "artar; ek FactUnits'in satirlari birim, W_fact_out sifir degil",
          SHARED_FACTS is False and inspect.signature(BlockModel).parameters["shared_facts"].default is False
          and not hasattr(base, "extra_facts") and len(split.extra_facts) == 2 and cnt(split) - cnt(base) == 2 * fu
          and float((split.extra_facts[1].W_fact_in.norm(dim=1) - 1).abs().max()) < 1e-6
          and float(split.extra_facts[1].W_fact_out.abs().max()) > 0, "ek %d parametre" % (cnt(split) - cnt(base)))

    g = torch.Generator().manual_seed(3)
    split = split.double()
    with torch.no_grad():                                  # sifirdan farkli alpha, Canon, bag: kontrol bos gecmesin
        for p_ in split.parameters():
            p_.add_(0.1 * torch.randn(p_.shape, generator=g, dtype=p_.dtype))
    sep = BlockModel(nv, **dict(kw, layers=4)).double()   # her tura ayri Block: attention'i split'ten ikiser kopya
    with torch.no_grad():
        for k in ("alpha_attention", "alpha_facts", "log_output_scale", "link_q", "link_u"):
            getattr(sep, k).copy_(getattr(split, k))
        sep.tokens.load_state_dict(split.tokens.state_dict())
        for i, b in enumerate(sep.blocks):
            src = split.blocks[i % 2]
            b.attention.load_state_dict(src.attention.state_dict())
            b.canon_weights.copy_(src.canon_weights)
            b.facts.load_state_dict((src.facts if i < 2 else split.extra_facts[i - 2]).state_dict())
    ids = sids[:4, :16]
    with torch.no_grad():
        err = float((split.logits(ids) - sep.logits(ids)).abs().max())
        base_err = float((split.logits(ids) - BlockModel(nv, **kw).double().logits(ids)).abs().max())
    check("shared_facts=False: skor = attention'i ikiser bagli (tur i ve i + layers), FactUnits'i tur basina ayri modelinki "
          "(float64); paylasimli modelden farkli", err < 1e-10 and base_err > 1e-3, "fark %.1e" % err)

    opts = []
    grab = lambda step, model, opt: opts.append(opt)
    mkw = dict(model_kw=dict(kw, shared_facts=False))
    trained, curve = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=20, log_at=(0, 20), save_every=1, save=grab, **mkw)
    nm = {id(p_): k for k, p_ in trained.named_parameters()}
    in_muon = {nm[id(p_)] for g_ in opts[-1].param_groups if g_["use_muon"] for p_ in g_["params"]}
    packs = {}
    keep = lambda step, model, opt: packs.setdefault(step, dict(step=step, model=copy.deepcopy(model.state_dict()),
                                                                   optimizer=copy.deepcopy(opt.state_dict())))
    full, _ = TR.train_seq("shared", sids[:40], smask[:40], nv, steps=6, log_at=(), save_every=2, save=keep, **mkw)
    res, _ = TR.train_seq("shared", sids[:40], smask[:40], nv, steps=6, log_at=(), checkpoint=packs[4], **mkw)
    qs = [[1] + sids[i, 1:1 + k].tolist() for i, k in enumerate((1, 3, 5, 8))]
    check("shared_facts=False: ek FactUnits Muon'da ve her adimdan sonra kurede; kayip iner; surdurme bit duzeyinde; "
          "onbellekli uretim = tam hesap",
          {"extra_facts.%d.%s" % (j, w) for j in (0, 1) for w in ("W_fact_in", "W_fact_up", "W_fact_out")} <= in_muon
          and float((trained.extra_facts[0].W_fact_in.norm(dim=1) - 1).abs().max()) < 1e-5
          and curve[-1]["nll"] < curve[0]["nll"]
          and all(torch.equal(x, y) for x, y in zip(full.state_dict().values(), res.state_dict().values()))
          and TR.generate(trained, qs, 6) == TR.generate(trained, qs, 6, cached=False),
          "kayip %.3f -> %.3f" % (curve[0]["nll"], curve[-1]["nll"]))


def t_input_embedding():
    """INPUT_EMBEDDING / FIRST_TURN_FACTS: girdi tablosu baslangicta (PF'den) PL'li modelle ayni; input_states elle hesapla
    ayni; tur 1 FactUnits'siz model = internals'in F1 atlamasi (ayni tohum); egitim, Adam, surdurme; onbellekli uretim
    (sagdan dolgulu istemler; istem sonra tek token) tam hesapla ayni; compile'da graph break yok; ayarsiz config eski
    model."""
    import copy
    import torch.nn.functional as F
    import internals_y as I
    from model_y import AttentionCache, BlockModel
    data = synthetic_data()
    sids, smask = sequences(data)
    nv = len(data["vocab"])
    ids = sids[:6, :20]
    full_kw = dict(input_embedding=True, first_turn_facts=False)

    base, emb = BlockModel(nv, input_embedding=False), BlockModel(nv, input_embedding=True)
    with torch.no_grad():
        z0 = base.logits(ids)
        free0 = BlockModel(nv, input_embedding=True, input_embedding_sphere=False)
        same = torch.equal(z0, free0.logits(ids)) and float((z0 - emb.logits(ids)).abs().max()) < 1e-5
    check("input_embedding: tablo baslangicta (girdi PF'den) PL'li modelle ayni skor (kuresizde bit duzeyinde, kurede "
          "yuvarlama farki)", same and not hasattr(base, "input_embedding"))

    g = torch.Generator().manual_seed(71)
    with torch.no_grad():
        emb.input_embedding.copy_(torch.randn(emb.input_embedding.shape, generator=g))
        err = float((emb.input_states(ids) - F.normalize(emb.input_embedding[ids], dim=-1)).abs().max())
    check("input_embedding: input_states = norm(E[token]) (elle)", err < 1e-6, "fark %.1e" % err)

    torch.manual_seed(0)
    plain, skip = BlockModel(nv, first_turn_facts=True), BlockModel(nv, first_turn_facts=False)
    kept = BlockModel(nv, first_turn_facts=False, shared_facts=True)
    with torch.no_grad():
        err_s = float((skip.logits(ids) - I._logits(plain, ids, dict(skip_facts=[0]))).abs().max())
        err_k = float((kept.logits(ids) - I._logits(BlockModel(nv, shared_facts=True, first_turn_facts=True), ids, dict(skip_facts=[0]))).abs().max())
    n_plain = sum(p.numel() for p in plain.parameters())
    n_skip = sum(p.numel() for p in skip.parameters())
    check("first_turn_facts=False: tur 1 FactUnits'siz model = internals'in F1 atlamasi (ayni tohum); ayri takimda tur 1'in "
          "takimi yok (parametre azalir), paylasimda takim durur (tur 3 kullaniyor)",
          err_s < 1e-5 and err_k < 1e-5 and skip.blocks[0].facts is None and kept.blocks[0].facts is not None
          and n_plain - n_skip == sum(p.numel() for p in plain.blocks[0].facts.parameters()),
          "fark %.1e / %.1e  parametre %d -> %d" % (err_s, err_k, n_plain, n_skip))

    opts = []
    grab = lambda step, model, opt: opts.append(opt)
    trained, curve = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=20, log_at=(0, 20), save_every=1, save=grab,
                                  model_kw=full_kw)
    names = {id(p_): k for k, p_ in trained.named_parameters()}
    in_adam = [names[id(p_)] for g_ in opts[-1].param_groups if not g_["use_muon"] for p_ in g_["params"]]
    packs = {}
    keep = lambda step, model, opt: packs.setdefault(step, dict(step=step, model=copy.deepcopy(model.state_dict()),
                                                                   optimizer=copy.deepcopy(opt.state_dict())))
    full, _ = TR.train_seq("shared", sids[:40], smask[:40], nv, steps=6, log_at=(), save_every=2, save=keep, model_kw=full_kw)
    res, _ = TR.train_seq("shared", sids[:40], smask[:40], nv, steps=6, log_at=(), checkpoint=packs[4], model_kw=full_kw)
    free, _ = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=20, log_at=(),
                           model_kw=dict(full_kw, input_embedding_sphere=False))
    rows_s, rows_f = trained.input_embedding.detach().norm(dim=1), free.input_embedding.detach().norm(dim=1)
    check("input_embedding_sphere: girdi tablosunun satirlari egitimde birim boy (varsayilan); False'ta serbest, boy buyur "
          "(gradyan satira dik)",
          float((rows_s - 1).abs().max()) < 1e-5 and trained.input_embedding_sphere and not free.input_embedding_sphere
          and float(rows_f.max()) > 1.01 and float((BlockModel(nv).tokens.fixed_points.norm(dim=1) - 1).abs().max()) < 1e-5,
          "birim %.1e  serbest en buyuk %.3f" % (float((rows_s - 1).abs().max()), float(rows_f.max())))

    check("input_embedding: egitimde kayip iner; input_embedding Adam'da; tur 1'in takimi state_dict'te yok; surdurme bit "
          "duzeyinde",
          curve[-1]["nll"] < curve[0]["nll"] and "input_embedding" in in_adam
          and not any(k_.startswith("blocks.0.facts.") for k_ in trained.state_dict())
          and all(torch.equal(a, b) for a, b in zip(full.state_dict().values(), res.state_dict().values())),
          "%.3f -> %.3f" % (curve[0]["nll"], curve[-1]["nll"]))

    qs = [[1] + sids[i, 1:1 + k].tolist() for i, k in enumerate((1, 2, 5, 8, 12))]
    cached_new = TR.generate(trained, qs, 6)
    scores, _ = cached_scores(trained, qs, cached_new)
    err_g = max(float((scores[i] - trained.logits(torch.tensor([q + cached_new[i]])).detach()[0, len(q) - 1:]).abs().max())
                for i, q in enumerate(qs))
    with torch.no_grad():                                  # sinavin yolu: istem son token'i haric, sonra tek token
        L = torch.tensor([len(q) for q in qs])
        W = int(L.max())
        x = torch.zeros(len(qs), W, dtype=torch.long)
        for i, q in enumerate(qs):
            x[i, :len(q)] = torch.tensor(q)
        rows = torch.arange(len(qs))
        caches = [AttentionCache(L - 1, W + 4) for _ in range(trained.turns)]
        trained.hidden(x[:, :W - 1], caches)
        z_c = trained.logits(x[rows, L - 1][:, None], caches)[:, 0]
        z_f = trained.logits(x)[rows, L - 1]
    err_h = float((z_c - z_f).abs().max())
    check("input_embedding: onbellekli uretim = tam hesap (sagdan dolgulu istemler); istem - 1 sonra tek token yolu ayni",
          cached_new == TR.generate(trained, qs, 6, cached=False) and err_g < 1e-4 and err_h < 1e-4,
          "skor farki %.1e  istem yolu %.1e" % (err_g, err_h))

    torch._dynamo.reset()
    ex = torch._dynamo.explain(BlockModel(nv, d=32, units=48, loss_chunk=7, **dict(full_kw)).loss)(sids[:4], smask[:4])
    torch._dynamo.reset()
    old = BlockModel(nv, **I._model_kw(dict(model_kw=dict(d=16, turns=2, input_embedding=True))))
    uu = I.unit_usage(BlockModel(nv, first_turn_facts=False).double().eval(), [sids[0, :int(smask[0].sum())].tolist()],
                      exclude_last=False)
    try:
        I._plan(BlockModel(nv, first_turn_facts=False), dict(skip_facts=[0]))
        refused = False
    except AssertionError:
        refused = True
    std_skip = I._standard_cases(6, 4, False)
    check("internals: FIRST_TURN_FACTS=False'ta tur 1'e FactUnits mudahalesi reddedilir; varsayilan ablate listesinde F1 yok",
          refused and "F1" not in std_skip and "F2" in std_skip and "F1" in I._standard_cases(6, 4))

    check("input_embedding: kayip compile'da tek grafik; ayarsiz config eski model (tablo kureye cekilmez, tur 1 FactUnits'li); "
          "unit_usage tur 1'i saymaz",
          ex.graph_break_count == 0 and not old.input_embedding_sphere and old.first_turn_facts
          and [r["turn"] for r in uu["turns"]][0] != I._turn_label(BlockModel(nv), 0) and len(uu["turns"]) == 3,
          "graph break %d" % ex.graph_break_count)


def t_internals():
    """internals_y (kalici analiz araci): elle ileri hesap = model.logits (float64; 5 model: varsayilan, shared_facts=False,
    girdi tablosu + tur 1 FactUnits'siz, output_link=False, dort Block; SDPA ve acik softmax yolu, sagdan dolgulu
    batch); trace'in durumlari, acilari ve lens'i modelin kendi hidden'i ve Block'uyla;
    point_drift, attention_stats, unit_usage bagimsiz hesapla; ablate: hicbir sey = taban (Δ 0), alpha'yi ayni degerle
    vermek = taban, kapatmalar agirligi degistirilmis modelin logits'iyle ayni (Δnll, Δacc, se), head ortalamasi elle;
    over_checkpoints'in yedekten kurdugu EMA = train_seq'in model.weight_ema'si (muon, adam, adamw); tuned_lens: birim
    cevirici = logit lens, ogrenilen cevirici ayri hikayelerde iyi, son durumda KL 0, model degismez."""
    import copy
    import json
    import shutil
    import tempfile
    import numpy as np
    import torch.nn.functional as F
    import internals_y as I
    from model_y import BlockModel
    data = synthetic_data()
    sids, smask = sequences(data)
    nv = len(data["vocab"])
    stories = [sids[i, :int(smask[i].sum())].tolist() for i in range(6)]
    others = [sids[i, :int(smask[i].sum())].tolist() for i in range(6, 12)]
    base_kw = dict(d=16, units=24, layers=2, turns=4, heads=2, shared_facts=True, input_embedding=False, first_turn_facts=True)   # paylasimli, PL girdili;
                                                                                  # ayrik ve tablolu hali acikca

    def perturbed(seed=5, **kw):
        m = BlockModel(nv, **dict(base_kw, **kw)).double()
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():                              # sifirdan farkli alpha, Canon, bag, esik: kontrol bos gecmesin
            for p_ in m.parameters():
                p_.add_(0.1 * torch.randn(p_.shape, generator=g, dtype=p_.dtype))
        return m.eval()

    variants = [("varsayilan", {}), ("shared_facts=False", dict(shared_facts=False)),
                ("output_link=False", dict(output_link=False)),
                ("girdi tablosu + tur 1 FactUnits'siz", dict(input_embedding=True, first_turn_facts=False)),
                ("dort Block", dict(layers=4))]
    errs = {}
    with torch.no_grad():
        for name, kw in variants:
            m = perturbed(**kw)
            ref = [m.logits(torch.tensor([s[:-1]]))[0] for s in stories]
            tr = I.trace(m, stories, keep_logits=True)
            explicit = [I._logits(m, torch.tensor([s[:-1]]), taps=dict(attention=lambda t, a: None))[0] for s in stories]
            errs[name] = max([float((a - b).abs().max()) for a, b in zip(tr["logits"] + explicit, ref + ref)]
                             + [tr["check_logits"]])
    worst = max(errs, key=errs.get)
    check("internals: elle ileri hesap = model.logits (float64; %d model ayari, dolgulu batch, SDPA ve acik softmax)" % len(variants),
          max(errs.values()) < 1e-10, "en buyuk %.1e (%s)" % (errs[worst], worst))

    m = perturbed()
    s0 = stories[0]
    ids = torch.tensor([s0[:-1]])
    y = torch.tensor(s0[1:])
    tr = I.trace(m, [s0], exclude_last=False)
    with torch.no_grad():
        hid = m.hidden(ids)
        states = [hid[0]]
        for t, blk in enumerate(m.turn_blocks()):          # alpha_F = 0: Block yalniz attention alt adimini yapar
            states += [blk(hid[t], alpha_attention=m.alpha_attention[t], alpha_facts=torch.zeros_like(m.alpha_facts[t])),
                       hid[t + 1]]
        P = m.tokens.points()
        c_ = lambda h: h @ P.T
        head = lambda h: torch.exp(m.log_output_scale) * (c_(h) + m.link_q * c_(h) ** 2
                                                          + (m.link_q ** 2 / 3 + m.link_u) * c_(h) ** 3)
        rank_ok, p_err, ang_err = True, 0.0, 0.0
        row = tr["stories"][0]
        for j, (name, h) in enumerate(zip(tr["states"], states)):
            z = head(h[0])
            zt = z[torch.arange(len(y)), y]
            rank_ok &= (1 + (z > zt[:, None]).sum(-1)).tolist() == row["rank"][name]
            p_err = max(p_err, float((torch.softmax(z, -1)[torch.arange(len(y)), y] - torch.tensor(row["p"][name])).abs().max()))
            if j:
                ang = torch.rad2deg(torch.acos((h[0] * states[j - 1][0]).sum(-1).clamp(-1, 1)))
                ang_err = max(ang_err, float((ang - torch.tensor(row["angle"][name], dtype=ang.dtype)).abs().max()))
        z = m.logits(ids)[0]
        last = tr["summary"][tr["states"][-1]]
        sum_err = max(abs(last["nll"] - float(F.cross_entropy(z, y))), abs(last["acc"] - float((z.argmax(-1) == y).double().mean())))
    check("internals trace: durumlar = model.hidden ve alpha_F = 0'li Block; lens sirasi ve p(hedef) modelin cikis formuluyle; "
          "aci = acos(<h, h'>); son durumun nll / acc'si model.logits'ten",
          rank_ok and p_err < 1e-6 and ang_err < 1e-3 and sum_err < 1e-10 and tr["states"][:3] == ["h0", "A1", "F1"],
          "p %.1e  aci %.1e derece  nll/acc %.1e" % (p_err, ang_err, sum_err))

    counts = np.random.default_rng(0).integers(1, 50, nv)
    counts[:5] = 0
    pd = I.point_drift(m, counts, data["vocab"], bands=(0, 16, 64, nv))
    with torch.no_grad():
        PL, PF = m.tokens.points(), m.tokens.fixed_points
        ang = torch.rad2deg(torch.acos((PL * PF).sum(-1).clamp(-1, 1))).numpy()
        nn_ = lambda X: (X @ X.T).fill_diagonal_(-2).max(-1).values.numpy()
    order = np.argsort(-counts, kind="stable")
    band_err = max(abs(b["angle_median"] - float(np.median(ang[order[b["lo"]:b["hi"]]]))) for b in pd["bands"])
    # PF float32'de birimlenip float64'e cevrildi: boyu 1'den ~1e-7 sapar -> acos ile kiris formulu ~1e-5 derece ayrisir
    check("internals point_drift: aci = acos(<PF, PL>) (float64); komsu kosinusu tam tablodan (PL ve PF); siklik bantlari; "
          "gorulmeyen token'lar", abs(pd["overall"]["angle_median"] - float(np.median(ang))) < 1e-4 and band_err < 1e-4
          and abs(pd["overall"]["nn_cos_pl_median"] - float(np.median(nn_(PL)))) < 1e-12
          and abs(pd["overall"]["nn_cos_pf_median"] - float(np.median(nn_(PF)))) < 1e-12
          and [(b["lo"], b["hi"]) for b in pd["bands"]] == [(0, 16), (16, 64), (64, nv)] and pd["unseen"]["n"] == 5
          and len(pd["examples"]) > 0, "bant %.1e" % band_err)

    st = I.attention_stats(m, [s0], exclude_last=False)
    with torch.no_grad():
        err, T = 0.0, ids.shape[1]
        back = (torch.arange(T)[:, None] - torch.arange(T)[None, :]).clamp(min=0).double()
        weights = []
        for t, blk in enumerate(m.turn_blocks()):
            h = hid[t]
            full = F.pad(h, (0, 0, 3, 0))
            x = h + sum(blk.canon_weights[k] * full[:, 3 - k:3 - k + T] for k in range(4))
            a = blk.attention.weights(x)[0]                                        # (H, T, T)
            weights.append(a)
            want = dict(entropy=-(a * a.clamp_min(1e-300).log()).sum(-1).mean(-1),
                        last4=torch.stack([a[:, i, max(0, i - 3):i + 1].sum(-1) for i in range(T)], -1).mean(-1),
                        distance=(a * back).sum(-1).mean(-1), first=a[..., 0].mean(-1))
            for k, v in want.items():
                err = max(err, float((v - torch.tensor(st[k][t], dtype=v.dtype)).abs().max()))
            err = max(err, abs(float(((x - h).norm(dim=-1) / h.norm(dim=-1)).mean()) - float(st["canon"][t])))
        reuse = torch.minimum(weights[0], weights[2]).sum(-1).mean(-1)
        err = max(err, float((reuse - torch.tensor(st["reuse"][0]["overlap"], dtype=reuse.dtype)).abs().max()))
    check("internals attention_stats: entropi, son 4, uzaklik, konum 0, Canon payi ve ayni Block'un iki turunun ortusmesi "
          "CausalAttention.weights'ten", err < 1e-10 and [r["turns"] for r in st["reuse"]] == [(0, 2), (1, 3)],
          "fark %.1e" % err)

    m2 = perturbed(shared_facts=False)
    uu = I.unit_usage(m2, [s0], exclude_last=False)
    ush = I.unit_usage(m, [s0], exclude_last=False)
    with torch.no_grad():
        hid2 = m2.hidden(ids)
        err = 0.0
        for t, blk in enumerate(m2.turn_blocks()):
            mid = blk(hid2[t], alpha_attention=m2.alpha_attention[t], alpha_facts=torch.zeros_like(m2.alpha_facts[t]))[0]
            f = blk.facts if t < 2 else m2.extra_facts[t - 2]
            xs = mid * 16 ** 0.5
            u = F.silu(xs @ f.W_fact_in.T) * (xs @ f.W_fact_up.T)
            share = (u ** 2).sum(0) / (u ** 2).sum()
            err = max(err, float((share - torch.tensor(uu["turns"][t]["share"], dtype=share.dtype)).abs().max()),
                      abs(float((u ** 2).sum(-1).mean()) - uu["turns"][t]["energy"]),
                      abs(float((u @ f.W_fact_out.T).pow(2).sum(-1).mean().sqrt()) - uu["turns"][t]["out_rms"]))
    check("internals unit_usage: enerji payi, E|u|^2, |W_out u| FactUnits formuluyle (shared_facts=False: tur >= layers "
          "kendi takimi); takimlar paylasimda [1+3, 2+4], ayrikta turn basina",
          err < 1e-10 and [tm["turns"] for tm in uu["teams"]] == [[0], [1], [2], [3]]
          and [tm["turns"] for tm in ush["teams"]] == [[0, 2], [1, 3]] and len(ush["teams"][0]["between"]) == 1,
          "fark %.1e" % err)

    dh = 16 // 2
    with torch.no_grad():                                  # tur 3 (blok 0) head ciktilarinin ve Canon ekinin ortalamasi, referansta
        tot, cnt = torch.zeros(2, dh, dtype=torch.float64), 0
        ctot = torch.zeros(16, dtype=torch.float64)
        blk = m.blocks[0]
        for s in others:
            h = m.hidden(torch.tensor([s[:-1]]))[2]
            T = h.shape[1]
            full = F.pad(h, (0, 0, 3, 0))
            mix = sum(blk.canon_weights[k] * full[:, 3 - k:3 - k + T] for k in range(4))
            x = h + mix
            v = (x @ blk.attention.W_value.T).unflatten(-1, (2, -1)).transpose(1, 2)[0]   # (H, T, dh)
            c = blk.attention.weights(x)[0] @ v
            tot += c[:, :T - 1].sum(1)                                                    # son hedef haric
            ctot += mix[0, :T - 1].sum(0)
            cnt += T - 1
        mean, cmean = tot / cnt, ctot / cnt
    cases = [("none", {}),
             ("alpha ayni", dict(alpha_attention={t: m.alpha_attention[t].clone() for t in range(4)},
                                 alpha_facts={t: m.alpha_facts[t].clone() for t in range(4)})),
             ("A2", dict(skip_attention=[1])), ("F1", dict(skip_facts=[0])),
             ("C blok 0", dict(canon={0: torch.zeros(4), 2: torch.zeros(4)})),
             ("head 1 sifir, blok 1", dict(heads={(1, 1): torch.zeros(dh), (3, 1): torch.zeros(dh)})),
             ("H3 ortalama", dict(heads={(2, 0): None, (2, 1): None})),
             ("H3 ortalama elle", dict(heads={(2, 0): mean[0], (2, 1): mean[1]})),
             ("CM3 ortalama", dict(canon_mean={2: None})), ("CM3 ortalama elle", dict(canon_mean={2: cmean}))]
    ab = I.ablate(m, stories, cases, reference=others)

    def per_story(model):                                  # model.logits'ten hikaye basina (nll, dogru, hedef); son hedef haric
        out = []
        for s in stories:
            z = model.logits(torch.tensor([s[:-1]]))[0][:-1]
            t_ = torch.tensor(s[1:-1])
            out.append((float(F.cross_entropy(z, t_, reduction="sum")), float((z.argmax(-1) == t_).sum()), len(t_)))
        return np.array(out)

    def edited(fn):
        c = copy.deepcopy(m)
        with torch.no_grad():
            fn(c)
        return c

    with torch.no_grad():
        B0 = per_story(m)
        refs = {"A2": edited(lambda c: c.alpha_attention[1].zero_()), "F1": edited(lambda c: c.alpha_facts[0].zero_()),
                "C blok 0": edited(lambda c: c.blocks[0].canon_weights.zero_()),
                "head 1 sifir, blok 1": edited(lambda c: c.blocks[1].attention.W_value[dh:].zero_())}
    rows = {r["case"]: r for r in ab["rows"]}
    N = B0[:, 2].sum()
    err = abs(ab["base"]["nll"] - B0[:, 0].sum() / N) + abs(ab["base"]["acc"] - B0[:, 1].sum() / N)
    for name, model in refs.items():
        with torch.no_grad():
            Bc = per_story(model)
        for j, key in ((0, "nll"), (1, "acc")):
            d = Bc[:, j] - B0[:, j]
            mean_d = d.sum() / N
            err = max(err, abs(rows[name]["d_" + key] - mean_d),
                      abs(rows[name]["se_" + key] - float(np.sqrt(((d - mean_d * B0[:, 2]) ** 2).sum()) / N)))
    zero = max(abs(rows[k][f]) for k in ("none", "alpha ayni") for f in ("d_nll", "d_acc", "se_nll", "se_acc"))
    mean_err = max(abs(rows["H3 ortalama"][k] - rows["H3 ortalama elle"][k]) for k in ("d_nll", "d_acc"))
    mean_err = max(mean_err, *(abs(rows["CM3 ortalama"][k] - rows["CM3 ortalama elle"][k]) for k in ("d_nll", "d_acc")))
    moved = min(abs(rows[k]["d_nll"]) for k in list(refs) + ["CM3 ortalama"])
    check("internals ablate: hicbir sey = taban ve alpha'yi ayni degerle vermek = taban (fark 0); A / F atlama, Canon kapatma, "
          "head sifirlama = agirligi degistirilmis modelin logits'i (dnll, dacc, se); head ve Canon eki ortalamasi referans "
          "hikayelerinden (elle hesapla ayni)",
          zero < 1e-12 and err < 1e-10 and mean_err < 1e-10 and moved > 1e-4 and "BAGIMLILIK" in ab["note"]
          and ab["check_logits"] < 1e-10, "sifir %.1e  fark %.1e  ortalama %.1e  en kucuk |d| %.1e" % (zero, err, mean_err, moved))

    name, words, build = I._parse_case((4, 2, 2), "A2+H3.1+C+aF4*0.5")
    case = build(m)
    _, words_cm, build_cm = I._parse_case((4, 2, 2), "CM3+C1")
    case_cm, case_all = build_cm(m), I._parse_case((4, 2, 2), "CM")[2](m)
    std = I._standard_cases(4, 2)
    check("internals CLI mudahale yazimi (turlar 1'den): 'A2+H3.1+C+aF4*0.5' -> skip_attention [1], heads (2, 1), butun Canon, "
          "alpha_facts[3] x 0,5; 'CM3+C1' -> canon_mean {2: None} + canon {0: 0}; 'CM' butun turlar; varsayilan listede C ve CM",
          case["skip_attention"] == [1] and list(case["heads"]) == [(2, 1)] and sorted(case["canon"]) == [0, 1, 2, 3]
          and torch.equal(case["alpha_facts"][3], m.alpha_facts[3] * 0.5)
          and case_cm["canon_mean"] == {2: None} and list(case_cm["canon"]) == [0] and not case_cm["canon"][0].any()
          and sorted(case_all["canon_mean"]) == [0, 1, 2, 3] and std[-10:] == ["C1", "C2", "C3", "C4", "C", "CM1", "CM2", "CM3",
                                                                             "CM4", "CM"],
          "%s | %s" % (words, words_cm))

    tmp = tempfile.mkdtemp(prefix="internals_")
    same = lambda a, b: list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)
    notes, ok = [], True
    try:
        for label, extra in (("muon", {}),):
            run = os.path.join(tmp, label)
            os.makedirs(run)
            mk = dict(base_kw, shared_facts=False, output_link=True)     # colab config'i model_kw'yi tam yazar
            emas, lasts = {}, {}

            def save(step, model, opt, run=run, emas=emas, lasts=lasts):
                torch.save(dict(step=step, model=model.state_dict(), optimizer=opt.state_dict()),
                           os.path.join(run, "checkpoint_t%06d.pt" % step))
                emas[step] = copy.deepcopy(model.weight_ema["model"].state_dict())
                lasts[step] = copy.deepcopy(model.state_dict())
            trained, _ = TR.train_seq("shared", sids[:40], smask[:40], nv, steps=6, log_at=(), save_every=2, save=save,
                                      weight_ema=0.9, model_kw=mk, **extra)
            config = dict(name=label, setting="shared", seed=0, vocab=nv, model_kw=mk, stream_norm=True, layer_norm=False,
                          rope=True, normalized_update=True, sphere_weights=True, canon=True, optimizer="muon",
                          weight_decay=0.0)
            with open(os.path.join(run, "config.json"), "w") as fh:
                json.dump(config, fh)
            grab = lambda m_: {k: v.clone() for k, v in m_.state_dict().items()}
            got = I.over_checkpoints(run, grab)
            got_last = I.over_checkpoints(run, grab, steps=[4], weights="last")
            torch.save(trained.weight_ema["model"].state_dict(), os.path.join(run, "model_weight_ema.pt"))
            this = ([r["step"] for r in got] == [2, 4, 6] and all(same(r["result"], emas[r["step"]]) for r in got)
                    and same(got_last[0]["result"], lasts[4]) and not same(emas[6], lasts[6])
                    and same(I._load_model(run, "ema").state_dict(), trained.weight_ema["model"].state_dict()))
            notes.append("%s %s" % (label, "tamam" if this else "TUTMADI"))
            ok &= this
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    check("internals over_checkpoints: yedekten kurulan EMA = train_seq'in model.weight_ema'si, bit duzeyinde (adim 2/4/6; "
          "Muon gruplari); 'last' = o adimin agirligi; model_weight_ema.pt", ok, ", ".join(notes))

    fit = [sids[i, :int(smask[i].sum())].tolist() for i in range(12, 76)]
    before = {k: v.clone() for k, v in m.state_dict().items()}
    flags = [p_.requires_grad for p_ in m.parameters()]
    tl0 = I.tuned_lens(m, stories, None, steps=0)
    tr = I.trace(m, stories)
    same_id = all(tl0["summary"][n]["tuned"] == tl0["summary"][n]["logit"] for n in tl0["states"])
    lens_err = max(max(abs(tl0["summary"][n]["logit"][k] - tr["summary"][n][k]) for k in ("nll", "acc", "median_rank"))
                   for n in tr["states"])
    check("internals tuned_lens: birim cevirici (0 adim) = logit lens, butun sayilar birebir; logit lens trace'inkiyle ayni "
          "(nll, acc, medyan sira; float64)", same_id and lens_err < 1e-10 and tl0["states"] == tr["states"],
          "fark %.1e" % lens_err)

    tl = I.tuned_lens(m, stories, fit, steps=120, lr=1e-2, batch=8)
    inner = tl["states"][:-1]
    worse = [n for n in inner if tl["summary"][n]["tuned"]["kl"] >= tl["summary"][n]["logit"]["kl"]]
    last = tl["summary"][tl["states"][-1]]["tuned"]["kl"]
    fell = all(b_ < a_ for a_, b_ in zip(tl["fit"]["kl_first"][:-1], tl["fit"]["kl_last"][:-1]))
    untouched = (all(torch.equal(before[k], v) for k, v in m.state_dict().items())
                 and [p_.requires_grad for p_ in m.parameters()] == flags and all(p_.grad is None for p_ in m.parameters()))
    # son durumda birim tam en iyi (gradyan ~1e-17) ama Adam olcekten bagimsiz: en iyinin cevresinde ~lr kadar titrer
    check("internals tuned_lens (d 16): ayri hikayelerde ogrenilen cevirici her ara durumda logit lens'ten iyi (KL); son "
          "durumda KL < 1e-4; egitim KL iner; model degismez (agirlik, requires_grad, grad)",
          not worse and abs(last) < 1e-4 and fell and untouched and tl["fit"]["stories"] == 64
          and len(tl["example"]["top"]["h0"]) == len(stories[0]) - 1,
          "KL ara durum ortalamasi logit %.3f -> tuned %.3f; son %.1e; kotu %s" % (
              np.mean([tl["summary"][n]["logit"]["kl"] for n in inner]), np.mean([tl["summary"][n]["tuned"]["kl"] for n in inner]),
              last, worse))


def _packed_rows(segments_by_row, pad=0):
    """Satir basina parcalar (id listeleri) -> (ids, mask, document_positions, parcalarin (satir, baslangic, boy)).
    Parca kendi konumuyla 0'dan; dolgu sagda, mask disi, konumu artarak devam eder."""
    T = max(sum(len(s) for s in segs) for segs in segments_by_row)
    ids = torch.full((len(segments_by_row), T), pad, dtype=torch.long)
    mask = torch.zeros(len(segments_by_row), T, dtype=torch.bool)
    pos = torch.zeros(len(segments_by_row), T, dtype=torch.long)
    where = []
    for r, segs in enumerate(segments_by_row):
        at = 0
        for s in segs:
            ids[r, at:at + len(s)] = torch.tensor(s)
            pos[r, at:at + len(s)] = torch.arange(len(s))
            where.append((r, at, len(s)))
            at += len(s)
        mask[r, :at] = True
        pos[r, at:] = torch.arange(1, T - at + 1) + (pos[r, at - 1] if at else -1)
    return ids, mask, pos, where


def t_packing():
    """Maskeli paketleme (FineWeb): paketli pencerede her belgenin skoru, kaybi ve gradyani o belgenin tek basina hesabiyla
    ayni (float64, yogun maske); flex yolu (CPU'da yalniz ileri) yogun maskeyle ayni; tek belgeli paket = paketsiz hesap;
    RoPE goreli (8.192 konumda da); compile'da graph break yok; train_seq paketli batch'le (coherence yarilari) ve surdurme;
    onbellek paketli girdiyi reddeder.  Kendi kucuk verisi (rastgele token)."""
    import copy
    import internals_y as I
    from model_y import BlockModel, build_document_mask, same_document_causal
    V = 40
    g = torch.Generator().manual_seed(11)
    docs = [[0] + torch.randint(1, V, (L,), generator=g).tolist() for L in (4, 9, 6, 12, 3)]   # [eos] + metin
    # satir 0: uc tam belge; satir 1: d3'un devam parcasi (pencere basinda, konum 0'dan) + d4, sagda dolgu
    rows = [[docs[0], docs[1], docs[2]], [docs[3][5:], docs[4]]]
    ids, mask, pos, where = _packed_rows(rows)
    segs = [s for r in rows for s in r]

    def perturbed_model(**kw):
        torch.manual_seed(3)
        m = BlockModel(V, **dict(dict(d=16, units=24, t_max=64), **kw)).double()
        with torch.no_grad():                              # Canon, alpha, phi 0'dan baslar: katkilari gorunsun
            for p_ in m.parameters():
                p_.add_(0.3 * torch.randn(p_.shape, generator=torch.Generator().manual_seed(p_.numel()), dtype=torch.float64))
        return m

    def per_document(m, chunk):
        """Parca parca tek basina: skorlar ve kayip toplami; parcanin son konumu paketteki sonraki token'i tahmin eder."""
        zs, total = [], 0.0
        for k, (r, at, L) in enumerate(where):
            z = m.logits(torch.tensor([segs[k]]))[0]
            zs.append(z)
            nxt = int(ids[r, at + L]) if at + L < ids.shape[1] and mask[r, at + L] else None
            y = torch.tensor(segs[k][1:] + ([nxt] if nxt is not None else []))
            total = total + torch.nn.functional.cross_entropy(z[:len(y)], y, reduction="sum")
        return zs, total

    configs = (dict(), dict(input_embedding=False, first_turn_facts=True, shared_facts=True),
               dict(rope=False, output_link=False, loss_chunk=0),
               dict(d=32, layers=6, turns=12, heads=16, t_max=16384, attention_log_scale=True))   # FineWeb 6x2 bicimi
    errs, gerrs, ierrs = [], [], []
    for kw in configs:
        m = perturbed_model(**kw)
        with torch.no_grad():
            z = m.logits(ids, document_positions=pos)
            zs, _ = per_document(m, kw)
            errs.append(max(float((z[r, at:at + L] - zs[k]).abs().max()) for k, (r, at, L) in enumerate(where)))
            ierrs.append(float((I._logits(m, torch.tensor([segs[1]])) - zs[1][None]).abs().max()))
        n_targets = int(mask[:, 1:].sum())
        _, nll = m.loss(ids, mask, document_positions=pos)       # capasiz kayip: belge toplamlariyla karsilastirilir
        grads = torch.autograd.grad(nll * n_targets, [p_ for p_ in m.parameters() if p_.requires_grad], allow_unused=True)
        _, ref = per_document(m, kw)
        rgrads = torch.autograd.grad(ref, [p_ for p_ in m.parameters() if p_.requires_grad], allow_unused=True)
        gerrs.append(max(float((a - b).abs().max()) for a, b in zip(grads, rgrads) if a is not None)
                     + abs(float(nll.detach()) * n_targets - float(ref.detach())))
    check("paketleme: paketli pencerede her belgenin skoru, kaybi ve BUTUN gradyanlari = belge tek basina (float64, yogun "
          "maske; 4 ayar: varsayilan, eski tasarim, RoPE'suz parcasiz kayip, 6 blok x 2 (12 tur, 16 head, t_max "
          "16.384, log-n); devam parcasi ve dolgu dahil); internals'in elle ileri hesabi = model (tek belge)",
          max(errs) < 1e-10 and max(gerrs) < 1e-10 and max(ierrs) < 1e-10,
          "skor %.1e  kayip+gradyan %.1e  internals %.1e" % (max(errs), max(gerrs), max(ierrs)))

    m = perturbed_model()
    T = ids.shape[1]
    dense = build_document_mask(pos)
    t = torch.arange(T)
    by_hand = torch.zeros(2, T, T, dtype=torch.bool)
    for r in range(2):
        for i in range(T):
            for j in range(T):
                by_hand[r, i, j] = j <= i and j >= i - int(pos[r, i])
    flex_mask = build_document_mask(pos, flex=True)
    m_flex = copy.deepcopy(m)
    m_flex.document_mask = lambda p: build_document_mask(p, flex=True)
    with torch.no_grad():
        err_flex = float((m_flex.logits(ids, document_positions=pos) - m.logits(ids, document_positions=pos)).abs().max())
    check("paketleme: yogun maske = elle (j <= t, ayni belge) ve flex'in mask_mod'undan; flex_attention yolu (CPU'da yalniz "
          "ileri, GPU'da compile + flex Colab'da dogrulanacak) yogun maskeyle ayni skor (float64)",
          torch.equal(dense, by_hand) and type(flex_mask).__name__ == "BlockMask" and err_flex < 1e-10
          and torch.equal(same_document_causal(t - pos)(torch.arange(2)[:, None, None], None, t[None, :, None],
                                                        t[None, None, :]), dense),
          "flex fark %.1e" % err_flex)

    single = torch.tensor([docs[1] + docs[2]])
    arange = torch.arange(single.shape[1])[None]
    fm = BlockModel(V, d=16, units=24, t_max=64)
    with torch.no_grad():
        for p_ in fm.parameters():
            p_.add_(0.3 * torch.randn(p_.shape, generator=torch.Generator().manual_seed(p_.numel())))
        z_none, z_one = fm.logits(single), fm.logits(single, document_positions=arange)
    one_mask = torch.ones_like(single, dtype=torch.bool)
    l_none = fm.loss(single, one_mask)[0]
    g_none = torch.autograd.grad(l_none, list(fm.parameters()), allow_unused=True)
    l_one = fm.loss(single, one_mask, document_positions=arange)[0]
    g_one = torch.autograd.grad(l_one, list(fm.parameters()), allow_unused=True)
    diff = max(float((a - b).abs().max()) for a, b in zip(g_none, g_one) if a is not None)
    check("paketleme: tek belgeli pencere (konum 0..T-1) = paketsiz hesap, bit duzeyinde (float32; skor, kayip, gradyan)",
          torch.equal(z_none, z_one) and float(l_none) == float(l_one) and diff == 0,
          "skor %.1e  gradyan %.1e  bit duzeyinde %s" % (float((z_none - z_one).abs().max()), diff,
                                                         torch.equal(z_none, z_one) and float(l_none) == float(l_one)))

    big = BlockModel(V, d=48, heads=12, units=24, t_max=8192, attention_log_scale=False).double()
    x = torch.tensor([docs[3] + docs[1]])
    near, far = torch.arange(x.shape[1])[None], torch.arange(x.shape[1])[None] + 8150
    with torch.no_grad():
        for p_ in big.parameters():
            p_.add_(0.3 * torch.randn(p_.shape, generator=torch.Generator().manual_seed(p_.numel()), dtype=torch.float64))
        h_near = big.blocks[0].attention(big.input_states(x), document_positions=near)
        h_far = big.blocks[0].attention(big.input_states(x), document_positions=far)
    check("paketleme: t_max 8.192, 12 head (head boyu 4): RoPE goreli -- konumlar 8.150 kaydirilinca attention ayni (float64)",
          float((h_near - h_far).abs().max()) < 1e-9 and abs(big.blocks[0].attention.scale - math.log(0.99 * 8191 / 0.01)) < 1e-12,
          "fark %.1e" % float((h_near - h_far).abs().max()))

    torch._dynamo.reset()
    fl = BlockModel(V, d=16, units=24, t_max=64, loss_chunk=7)
    ex = torch._dynamo.explain(fl.loss)(ids, mask, pos)
    torch._dynamo.reset()
    fl.document_mask = lambda p: build_document_mask(p, flex=True)
    with torch.no_grad():                                  # flex'in CPU'da geri yayilimi yok: iz yalniz ileri hesapla
        ex_flex = torch._dynamo.explain(fl.loss)(ids, mask, pos)
    torch._dynamo.reset()
    check("paketleme: paketli kayip compile'da tek grafik (yogun maske; flex yolu ileri hesapta) -- GPU'da compile + flex "
          "Colab'da dogrulanacak", ex.graph_break_count == 0 and ex_flex.graph_break_count == 0,
          "graph break %d / flex %d" % (ex.graph_break_count, ex_flex.graph_break_count))

    windows = [_packed_rows([[docs[0], docs[1]], [docs[2], docs[4], docs[0]]]),
               _packed_rows([[docs[3], docs[4]], [docs[1][2:], docs[2]]])]
    batches = lambda s: windows[s % 2][:3]
    kw = dict(d=16, units=24, t_max=64, loss_chunk=7)
    runs = {}
    for sched in ("wsd", "coherence"):
        packs = {}
        keep = lambda step, model, opt: packs.setdefault(step, dict(step=step, model=copy.deepcopy(model.state_dict()),
                                                                       optimizer=copy.deepcopy(opt.state_dict())))
        full, curve = TR.train_seq("shared", None, None, V, steps=6, log_at=(0, 6), batches=batches, schedule=sched,
                                   save_every=2, save=keep, model_kw=kw)
        res, _ = TR.train_seq("shared", None, None, V, steps=6, log_at=(), batches=batches, schedule=sched,
                              checkpoint=packs[4], model_kw=kw)
        runs[sched] = (all(torch.equal(a, b) for a, b in zip(full.state_dict().values(), res.state_dict().values()))
                       and all(math.isfinite(c["nll"]) for c in curve))
    check("paketleme: train_seq batch (ids, mask, document_positions) ile (wsd; coherence iki yarisi konumlarla); surdurme "
          "bit duzeyinde", all(runs.values()), str(runs))

    refused = []
    for attempt in (lambda: m.hidden(ids, [None] * m.turns, document_positions=pos),):
        try:
            attempt()
            refused.append(False)
        except AssertionError:
            refused.append(True)
    check("paketleme: onbellekli uretim paketli pencereyi reddeder (internals CLI FineWeb'i tek belgeyle okur: "
          "tests_fineweb)", all(refused), str(refused))


def t_attention_log_scale():
    """ATTENTION_LOG_SCALE: sorgu basina scale_for(n), n = gordugu anahtar sayisi (belge ici konum + 1, en az 2).  Varsayilan
    kapali; t_max 512'de konum 511 sabit olcekli modelle ayni; elle hesap (float64; paketsiz / paketli, 4 head / tek head);
    onbellekli uretim = tam hesap; internals'in elle ileri hesabi = model; compile'da graph break yok."""
    import internals_y as I
    from model_y import ATTENTION_LOG_SCALE, BlockModel, CausalAttention, build_document_mask, scale_for
    V = 40
    g = torch.Generator().manual_seed(21)
    fixed = CausalAttention(16, t_max=512, seed=5, rope=True, heads=4)
    logn = CausalAttention(16, t_max=512, seed=5, rope=True, heads=4, attention_log_scale=True)
    x = torch.nn.functional.normalize(torch.randn(1, 512, 16, generator=g), dim=-1)
    with torch.no_grad():
        a, b = fixed(x), logn(x)
    check("attention_log_scale: varsayilan acik (ana kosu); t_max 512'de n = 512 (konum 511) sabit olcekle ayni, erken konumda "
          "farkli",
          ATTENTION_LOG_SCALE is True and BlockModel(V).blocks[0].attention.attention_log_scale
          and float((a[0, 511] - b[0, 511]).abs().max()) < 1e-6 and float((a[0, 100] - b[0, 100]).abs().max()) > 1e-4,
          "511: %.1e  100: %.1e" % (float((a[0, 511] - b[0, 511]).abs().max()), float((a[0, 100] - b[0, 100]).abs().max())))

    errs = []
    for heads in (4,):
        at = CausalAttention(16, t_max=64, seed=6, rope=True, heads=heads, attention_log_scale=True).double()
        ref_at = CausalAttention(16, t_max=64, seed=6, rope=True, heads=heads).double()
        with torch.no_grad():
            for m_ in (at, ref_at):
                m_.W_value.copy_(torch.randn(16, 16, generator=torch.Generator().manual_seed(3), dtype=torch.float64))
        h = torch.randn(2, 13, 16, generator=g, dtype=torch.float64)
        pos = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2], [4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7]])
        for p in (None, pos):
            with torch.no_grad():
                got = at(h) if p is None else at(h, document_positions=p)
                q, k = ref_at.queries_keys(h, p)                        # birim q, k (RoPE'lu), sabit olceksiz
                v = (h @ ref_at.W_value.T).unflatten(-1, (heads, -1)).transpose(-3, -2)
                t = torch.arange(13)
                n = (t + 1 if p is None else p + 1).double()
                s = torch.tensor([[scale_for(max(int(c), 2)) for c in row] for row in n.reshape(-1, 13)],
                                 dtype=torch.float64).reshape(n.shape)
                s = s if p is None else s[:, None]
                allowed = torch.ones(13, 13, dtype=torch.bool).tril() if p is None else build_document_mask(p)[:, None]
                z = (s[..., None] * (q @ k.transpose(-1, -2))).masked_fill(~allowed, float("-inf"))
                want = torch.softmax(z, -1) @ v
                want = want.transpose(-3, -2).flatten(-2)
            errs.append(float((got - want).abs().max()))
    check("attention_log_scale: skor = scale_for(max(n, 2)) <q, k>, n = konum + 1 (elle, float64; paketsiz ve paketli, 4 head)", max(errs) < 1e-12, "fark %.1e" % max(errs))

    torch.manual_seed(4)
    m = BlockModel(V, d=16, units=24, t_max=64, attention_log_scale=True)
    with torch.no_grad():
        for p_ in m.parameters():
            p_.add_(0.3 * torch.randn(p_.shape, generator=torch.Generator().manual_seed(p_.numel())))
    qs = [[0] + torch.randint(1, V, (k,), generator=g).tolist() for k in (1, 3, 6, 9)]
    new = TR.generate(m, qs, 6)
    scores, _ = cached_scores(m, qs, new)
    err_g = max(float((scores[i] - m.logits(torch.tensor([q + new[i]])).detach()[0, len(q) - 1:]).abs().max())
                for i, q in enumerate(qs))
    md = m.double()
    ids = torch.tensor([qs[3] + new[3]])
    with torch.no_grad():
        err_i = float((I._logits(md, ids) - md.logits(ids)).abs().max())
    big = BlockModel(V, d=32, layers=6, turns=12, heads=16, units=24, t_max=16384, attention_log_scale=True).double()
    with torch.no_grad():
        for p_ in big.parameters():
            p_.add_(0.3 * torch.randn(p_.shape, generator=torch.Generator().manual_seed(p_.numel()), dtype=torch.float64))
        seen = []
        err_t = float((I._logits(big, ids, taps=dict(attention=lambda t, a: seen.append(t))) - big.logits(ids)).abs().max())
        err_s = float((I._logits(big, ids) - big.logits(ids)).abs().max())
    check("attention_log_scale: onbellekli uretim = tam hesap (sagdan dolgulu istemler, n = konum + 1); internals'in elle "
          "ileri hesabi = model.logits (float64; SDPA ve acik softmax yolu, 6 blok x 2 = 12 tur dahil)",
          new == TR.generate(m, qs, 6, cached=False) and err_g < 1e-4 and err_i < 1e-10 and err_t < 1e-10
          and err_s < 1e-10 and seen == list(range(12)),
          "skor farki %.1e  internals %.1e / 6x2 %.1e %.1e" % (err_g, err_i, err_s, err_t))

    torch._dynamo.reset()
    fl = BlockModel(V, d=16, units=24, t_max=64, loss_chunk=7, attention_log_scale=True)
    ids = torch.randint(0, V, (3, 12), generator=g)
    mask = torch.ones_like(ids, dtype=torch.bool)
    pos = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6]] * 3)
    ex = torch._dynamo.explain(fl.loss)(ids, mask)
    torch._dynamo.reset()
    ex_p = torch._dynamo.explain(fl.loss)(ids, mask, pos)
    torch._dynamo.reset()
    check("attention_log_scale: kayip compile'da tek grafik (paketsiz ve paketli)",
          ex.graph_break_count == 0 and ex_p.graph_break_count == 0,
          "graph break %d / %d" % (ex.graph_break_count, ex_p.graph_break_count))


def t_rope_base():
    """ROPE_BASE "auto": rope_base_for(head boyu, t_max) Men 2024 olcutu -- secilen tabanda B(m) >= 0 her m <= 2 x baglam, bir
    kucuk aday tutmuyor (bagimsiz hesap, math.cos); SimpleStories (96, 512) 10.000'de kalir; model tabani attention'a ulasir
    (elle RoPE ayni); eski config (rope_base yok) 10.000 ile kurulur."""
    import internals_y as I
    from model_y import ROPE_BASE, BlockModel, CausalAttention, apply_rope, rope_base_for

    def reach(base, dh, L):                              # B(m) >= 0 kalan en uzun m (<= L); bagimsiz: saf Python
        th = [base ** (-2 * i / dh) for i in range(dh // 2)]
        for m in range(L + 1):
            if sum(math.cos(m * t) for t in th) < 0:
                return m - 1
        return L

    grid = [c * 10.0 ** k for k in range(4, 13) for c in (1, 2, 5)]
    rows = []
    for dh, L, want in ((96, 512, 1e4), (128, 8192, 5e5), (64, 8192, 2e6)):
        b = rope_base_for(dh, L)
        prev = grid[grid.index(b) - 1] if grid.index(b) else None
        rows.append((dh, L, b, want, reach(b, dh, 2 * L) == 2 * L, prev is None or reach(prev, dh, 2 * L) < 2 * L))
    check("rope_base_for: B(m) >= 0 her m <= 2 x baglam, bir kucuk aday tutmuyor; (96, 512) 10.000, (128, 8.192) 5e5, "
          "(64, 8.192) 2e6", all(r[2] == r[3] and r[4] and r[5] for r in rows),
          " ".join("%d/%d->%.0e" % r[:3] for r in rows))

    V = 30
    g = torch.Generator().manual_seed(41)
    x = torch.nn.functional.normalize(torch.randn(1, 20, 32, generator=g), dim=-1).double()
    at = CausalAttention(32, t_max=64, seed=3, rope=True, heads=4).double()
    fixed = CausalAttention(32, t_max=64, seed=3, rope=True, heads=4, rope_base=10000).double()
    with torch.no_grad():
        q, k = at.queries_keys(x)
        qh = torch.nn.functional.normalize((x @ at.W_query.T).unflatten(-1, (4, -1)).transpose(-3, -2), dim=-1)
        err = float((q - apply_rope(qh, None, rope_base_for(8, 64))).abs().max())
    m_auto, m_old = BlockModel(V, d=32, heads=4, units=8, t_max=64), I._build(dict(vocab=V, stream_norm=True, layer_norm=False,
        rope=True, normalized_update=True, sphere_weights=True, canon=True, model_kw=dict(d=32, heads=4, units=8, t_max=64)))
    check("ROPE_BASE: varsayilan auto, model rope_base_for(head boyu, t_max) kullanir (elle RoPE ayni); sayi verilince o; "
          "eski config (rope_base yok) 10.000; apply_rope varsayilani 10.000",
          ROPE_BASE == "auto" and at.rope_base == rope_base_for(8, 64) != 10000 and fixed.rope_base == 10000.0 and err < 1e-12
          and all(b.attention.rope_base == rope_base_for(8, 64) for b in m_auto.blocks)
          and all(b.attention.rope_base == 10000.0 for b in m_old.blocks)
          and torch.equal(apply_rope(x), apply_rope(x, None, 10000.0)),
          "auto %.0e  fark %.1e" % (at.rope_base, err))


def t_micro_batches():
    """MICRO_BATCHES (gradyan birikimi): coherence'ta 2 parca = bugunku iki yari (bit duzeyinde); 4 parca ve coherence'siz 2 /
    4 parca tek batch'in egitimiyle ayni (float32 yuvarlamasi); paketli batch'le; surdurme bit duzeyinde; coherence'ta tek
    sayi reddedilir."""
    import copy
    V = 30
    g = torch.Generator().manual_seed(31)
    steps_data = []
    for _ in range(3):
        ids = torch.randint(0, V, (8, 14), generator=g)
        mask = torch.ones_like(ids, dtype=torch.bool)
        mask[1, 9:] = mask[6, 4:] = False                    # farkli hedef sayilari: agirliklar esit degil
        steps_data.append((ids, mask))
    batches = lambda s: steps_data[s % 3]
    kw = dict(d=16, units=24, t_max=64, loss_chunk=7)

    def train(sched, k, draw=batches, steps=6, **extra):
        m, _ = TR.train_seq("shared", None, None, V, steps=steps, log_at=(), batches=draw, schedule=sched, micro_batches=k,
                            model_kw=kw, **extra)
        return m.state_dict()

    def gap(a, b):
        return max(float((a[k_] - b[k_]).abs().max()) for k_ in a)

    base_c, base_w = train("coherence", 1), train("wsd", 1)
    two_c = train("coherence", 2)
    diffs = dict(coherence4=gap(base_c, train("coherence", 4)), wsd2=gap(base_w, train("wsd", 2)),
                 wsd4=gap(base_w, train("wsd", 4)))
    check("micro_batches: coherence'ta 2 parca = iki yari (bit duzeyinde, 6 adim); 4 parca ve coherence'siz 2 / 4 parca tek "
          "batch'le ayni egitim (float32, Muon dahil)",
          all(torch.equal(base_c[k_], two_c[k_]) for k_ in base_c) and max(diffs.values()) < 1e-4,
          " ".join("%s %.1e" % kv for kv in diffs.items()))

    packs = {}
    keep = lambda step, model, opt: packs.setdefault(step, dict(step=step, model=copy.deepcopy(model.state_dict()),
                                                               optimizer=copy.deepcopy(opt.state_dict())))
    pos = torch.tensor([list(range(6)) + list(range(8))] * 8)
    packed = lambda s: steps_data[s % 3] + (pos,)
    full = train("coherence", 4, draw=packed, save_every=2, save=keep)
    res, _ = TR.train_seq("shared", None, None, V, steps=6, log_at=(), batches=packed, schedule="coherence", micro_batches=4,
                          checkpoint=packs[4], model_kw=kw)
    try:
        train("coherence", 3, steps=1)
        refused = False
    except AssertionError:
        refused = True
    check("micro_batches: paketli batch'le (4 parca, coherence); surdurme bit duzeyinde; coherence'ta tek sayi reddedilir",
          all(torch.equal(full[k_], v) for k_, v in res.state_dict().items()) and refused)


def _run_one(name):
    """Tek testi calistirir; ciktisi, sonuclari ve suresi (paralel kosucu icin)."""
    import contextlib
    import io
    import time
    buf = io.StringIO()
    del RESULTS[:]
    t0 = time.time()
    with contextlib.redirect_stdout(buf):
        try:
            globals()[name]()
        except Exception:                                  # istisna o testin kaldisi; havuz dusmez
            import traceback
            print(traceback.format_exc())
            RESULTS.append(False)
    return name, list(RESULTS), buf.getvalue(), time.time() - t0


if __name__ == "__main__":
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1, help="paralel surec sayisi (her surec tek is parcacigi)")
    ap.add_argument("--only", default="", help="yalniz bu testler, virgulle: t_coherence,t_output_link")
    args = ap.parse_args()
    names = [f.__name__ for f in (t_step3, t_generate_cached, t_normalized_update, t_canon,
              t_layers, t_coherence, t_weight_ema, t_heads, t_fact_activation, t_learn_output_scale, t_matmul_precision,
              t_loss_chunk, t_muon_tangent, t_output_link, t_shared_facts, t_input_embedding, t_internals, t_packing, t_attention_log_scale,
              t_micro_batches, t_rope_base)]
    if args.only:
        want = [n.strip() for n in args.only.split(",") if n.strip()]
        assert set(want) <= set(names), "bilinmeyen test: %s" % sorted(set(want) - set(names))
        names = [n for n in names if n in want]
    print("tests (model_y)  %d test, %d surec" % (len(names), args.jobs))
    t0 = time.time()
    if args.jobs > 1:                                  # spawn: her surec tests_y'yi yeniden yukler (tek is parcacigi)
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(args.jobs)
        outs = pool.imap(_run_one, names, chunksize=1)
    else:
        outs = (_run_one(n) for n in names)
    total, times = [], []
    for name, res, text, secs in outs:
        print(text, end="", flush=True)
        print("  -- %s: %d/%d GECTI, %.0f sn" % (name, sum(res), len(res), secs), flush=True)
        total += res
        times.append((secs, name))
    if args.jobs > 1:
        pool.close()
    print("\nen uzun: " + ", ".join("%s %.0f sn" % (n, t) for t, n in sorted(times, reverse=True)[:5]))
    print("toplam %.0f sn (duvar saati)" % (time.time() - t0))
    print("\n%d GECTI   %d KALDI" % (sum(total), len(total) - sum(total)))
    sys.exit(0 if all(total) else 1)
