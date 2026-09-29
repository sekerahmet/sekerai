# -*- coding: utf-8 -*-
"""tests_y -- model_y'nin kapilari: model (Adim 1-3, Model X, deneme ayarlari, transformer) ve genel egitim.  CPU, saniyeler.
Egitim verisi olarak akrabalik verisi kullanilir (train_kinship/); verinin ve sinavin kendi testleri orada (tests_kinship.py).

    python tests_y.py
"""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "train_kinship"))
torch.set_num_threads(1)

import data_y as D  # noqa: E402
import exam_kinship as EK  # noqa: E402
import train_y as TR  # noqa: E402
from model_y import BigramModel, deviation, scale_for  # noqa: E402

RESULTS = []


def check(name, ok, note=""):
    RESULTS.append(bool(ok))
    print("  %-78s %s  %s" % (name, "GECTI" if ok else "KALDI", note))


def reference_logits(PF, shift, W, scale, ids):
    """Tasarim formulu, float64, modelden bagimsiz: PL = norm(PF+shift), q = norm(W PL_i), skor_j = scale <q, PL_j>."""
    PL = PF.double() + shift.double()
    PL = PL / PL.norm(dim=-1, keepdim=True)
    q = PL[ids] @ W.double().T
    q = q / q.norm(dim=-1, keepdim=True)
    return scale * q @ PL.T


def perturbed(n=12, d=6, seed=1, **kw):
    g = torch.Generator().manual_seed(seed)
    m = BigramModel(n, d=d, **kw)
    with torch.no_grad():
        m.next.W_next.copy_(torch.randn(d, d, generator=g))
        if m.tokens.learn:
            m.tokens.shift.copy_(0.3 * torch.randn(n, d, generator=g))
    return m, g


TOY_ANGLES = (0.0, 90.0, 180.0, 270.0)
TOY_PAIRS = ((0, 2), (1, 2), (2, 3))          # Alice->Smith, Tom->Smith, Smith->'s


def toy_points():
    r = torch.deg2rad(torch.tensor(TOY_ANGLES, dtype=torch.float64))
    return torch.stack([torch.cos(r), torch.sin(r)], -1)


def toy_wall(scale, grid=3600):
    """Sabit noktalarda en dusuk kayip, egitimden bagimsiz: W'nin 1. sutunu Alice'in, 2. sutunu Tom'un q yonu;
    Smith = -Alice oldugu icin q_Smith = -q_Alice.  Iki yon aci taramasiyla."""
    P = toy_points()
    a = torch.linspace(0, 2 * math.pi, grid + 1, dtype=torch.float64)[:-1]
    U = torch.stack([torch.cos(a), torch.sin(a)], -1)                      # aday yonler
    nll = lambda q, t: -torch.log_softmax(scale * q @ P.T, -1)[:, t]
    alice_smith = nll(U, 2) + nll(-U, 3)                                   # ayni yon iki ciftte
    tom = nll(U, 2)
    return float((alice_smith.min() + tom.min()) / 3)


def toy(learn, steps=3000):
    """Konusmadaki ornek: 4 token, 2 boyut, 0/90/180/270 derece; Alice->Smith, Tom->Smith, Smith->'s."""
    m = BigramModel(4, d=2, learn_points=learn)
    with torch.no_grad():
        m.tokens.fixed_points.copy_(toy_points().float())
    inputs, targets = torch.tensor([p[0] for p in TOY_PAIRS]), torch.tensor([p[1] for p in TOY_PAIRS])
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=0.05)
    for _ in range(steps):
        total, nll = m.loss(inputs, targets)
        opt.zero_grad()
        total.backward()
        opt.step()
    return m.loss(inputs, targets)[1].item()


def t_model():
    n = 12
    m = BigramModel(n, d=6)
    ids = torch.arange(n)
    want = math.log(0.99 * (n - 1) / 0.01)
    check("model: baslangicta PL = PF; scale = ln(0,99 (n-1) / 0,01), elle girilmez",
          torch.allclose(m.tokens.points(), m.tokens.fixed_points, atol=1e-7) and abs(m.scale - want) < 1e-12
          and abs(scale_for(74) - math.log(0.99 * 73 / 0.01)) < 1e-12, "n=74: %.3f" % scale_for(74))

    before = m.logits(ids).detach()
    with torch.no_grad():
        m.next.W_next.mul_(3.0)
    after = m.logits(ids).detach()
    q = m.next(m.tokens.points()[ids]).detach()
    check("model: W'nin boyu skoru degistirmez (W x 3), q'nun boyu hep 1",
          float((after - before).abs().max()) < 1e-5 and float((q.norm(dim=-1) - 1).abs().max()) < 1e-6)

    for kw in (dict(learn_points=True), dict(learn_points=False)):
        m, g = perturbed(n, **kw)
        ids = torch.randint(0, n, (40,), generator=g)
        ref = reference_logits(m.tokens.fixed_points, m.tokens.shift, m.next.W_next, m.scale, ids)
        err = float((m.logits(ids).detach().double() - ref).abs().max())
        check("model: skor = tasarim formulu (bagimsiz float64 referans), %s" % ("ogrenilen" if kw["learn_points"] else "sabit"),
              err < 1e-5, "fark %.1e" % err)

    m, g = perturbed(n)
    inputs, targets = torch.randint(0, n, (40,), generator=g), torch.randint(0, n, (40,), generator=g)
    m.zero_grad()
    m.loss(inputs, targets)[1].backward()
    P = m.tokens.points().detach().double()
    Wd = m.next.W_next.detach().double()
    pi = torch.softmax(reference_logits(m.tokens.fixed_points, m.tokens.shift.detach(), Wd, m.scale, inputs), -1)
    raw = P[inputs] @ Wd.T
    size = raw.norm(dim=-1, keepdim=True)
    qh = raw / size
    g = m.scale * ((pi @ P) - P[targets])                         # s (p_ort - p_hedef): kuredeki q'ya gore
    g = (g - (g * qh).sum(-1, keepdim=True) * qh) / size          # norm'dan geri: yalniz q'ya dik kismi, 1/|Wp| ile
    formula = g.T @ P[inputs] / len(inputs)
    err = float((m.next.W_next.grad.double() - formula).abs().max())
    check("model: dL/dW = [s (p_ort - p_hedef)]_dik / |W p| . p_i^T (normdan gecen kural)", err < 1e-5, "fark %.1e" % err)

    m, _ = perturbed(n, learn_points=True, anchor=0.05)
    m.zero_grad()
    m.tokens.anchor_loss().backward()
    err = float((m.tokens.shift.grad - 2 * 0.05 * m.tokens.shift.detach()).abs().max())
    check("model: capa gradyani = 2 . anchor . shift", err < 1e-6, "fark %.1e" % err)

    d = D.build()
    inputs, targets = EK.token_pairs(d)
    nv = len(d["vocab"])
    ok = []
    for name in ("fixed", "free"):
        before = BigramModel(nv, **TR.SETTINGS[name]).tokens.fixed_points.clone()
        m, curve = TR.train(name, inputs, targets, nv, steps=20, log_at=(0, 20))
        ok.append(torch.equal(m.tokens.fixed_points, before) and curve[-1]["nll"] < curve[0]["nll"])
        if name == "fixed":
            ok.append(float(m.tokens.shift.abs().max()) == 0 and float(deviation(m).max()) < 1e-3)
        else:
            ok.append(float(m.tokens.shift.abs().max()) > 0 and float(deviation(m).max()) > 0)
    check("model: egitimde PF bit duzeyinde degismez; sabitte PL = PF, serbestte PL kayar; kayip iner", all(ok))

    m = BigramModel(3, d=4)
    with torch.no_grad():
        m.tokens.fixed_points.copy_(torch.eye(4)[:3])
        m.tokens.shift[0] = torch.tensor([math.cos(math.radians(17)) - 1, math.sin(math.radians(17)), 0, 0])
    dev = deviation(m)
    check("okuma: 17 derece dondurulen nokta 17 derece okunur, digerleri 0",
          abs(float(dev[0]) - 17) < 1e-3 and float(dev[1:].abs().max()) < 1e-3, "%.4f" % float(dev[0]))

    floor = TR.bigram_floor(torch.tensor([0, 0, 1]), torch.tensor([1, 2, 0]), 3)
    check("okuma: bigram tavani sayimlardan (0 -> 1|2 esit, 1 -> 0)", abs(floor - 2 * math.log(2) / 3) < 1e-12)

    wall, free, floor = toy(False), toy(True), toy_wall(scale_for(4))
    check("ornek: sabit noktalar aci taramasinin duvarinda, ogrenilen noktalar gecer",
          abs(wall - floor) < 0.01 and free < 0.05, "sabit %.3f (duvar %.3f)  ogrenilen %.3f" % (wall, floor, free))


def reference_sequence(m, ids):
    """Adim 2 formulu, float64, modelden bagimsiz: a_tj = softmax_{j<=t}(s_att cos(W_query x_t, W_key x_j)), c_t = sum a_tj x_j,
    skor = scale <norm(W_next x_t + W_context c_t), PL>."""
    PL = m.tokens.fixed_points.double() + m.tokens.shift.detach().double()
    PL = PL / PL.norm(dim=-1, keepdim=True)
    x = PL[ids]
    unit = lambda v: v / v.norm(dim=-1, keepdim=True)
    lk = m.attention
    q, k = unit(x @ lk.W_query.detach().double().T), unit(x @ lk.W_key.detach().double().T)
    T = ids.shape[-1]
    s = lk.scale * q @ k.transpose(-1, -2)
    s = s.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
    a = torch.softmax(s, -1)
    raw = x @ m.next.W_next.detach().double().T + (a @ x) @ lk.W_context.detach().double().T
    return m.scale * unit(raw) @ PL.T, a


def t_step2():
    from model_y import SequenceModel, T_MAX
    n, d = 12, 6
    g = torch.Generator().manual_seed(5)
    ids = torch.randint(0, n, (3, 9), generator=g)

    one, seq = BigramModel(n, d=d), SequenceModel(n, d=d, attention=False)
    err = float((seq.logits(ids).detach() - one.logits(ids.reshape(-1)).detach().reshape(3, 9, n)).abs().max())
    check("adim 2: attention kapaliyken her konumda Adim 1 ile ayni", err < 1e-6, "fark %.1e" % err)

    on = SequenceModel(n, d=d, attention=True)
    check("adim 2: baslangicta (W_context = 0) attention hicbir seyi degistirmez",
          torch.equal(on.logits(ids).detach(), seq.logits(ids).detach()))
    check("adim 2: attention olcegi = ln(0,99 (T_MAX-1) / 0,01)",
          abs(on.attention.scale - math.log(0.99 * (T_MAX - 1) / 0.01)) < 1e-12, "%.3f" % on.attention.scale)

    with torch.no_grad():
        on.attention.W_context.copy_(torch.randn(d, d, generator=g))
        on.tokens.shift.copy_(0.3 * torch.randn(n, d, generator=g))
    ref, a_ref = reference_sequence(on, ids)
    err = float((on.logits(ids).detach().double() - ref).abs().max())
    a = on.attention.weights(on.tokens.points()[ids]).detach().double()
    werr = float((a - a_ref).abs().max())
    check("adim 2: skor (SDPA yolu) = tasarim formulu (bagimsiz float64); okuma agirliklari da ayni",
          err < 1e-5 and werr < 1e-6 and float(a.triu(1).abs().max()) == 0 and float((a.sum(-1) - 1).abs().max()) < 1e-6,
          "skor %.1e  agirlik %.1e" % (err, werr))

    base = on.logits(ids).detach()
    changed = ids.clone()
    changed[:, 5] = (changed[:, 5] + 1) % n
    after = on.logits(changed).detach()
    check("adim 2: nedensellik -- konum 5 degisince 0-4 aynen kalir, 5 ve sonrasi degisir",
          float((after[:, :5] - base[:, :5]).abs().max()) < 1e-6 and float((after[:, 5:] - base[:, 5:]).abs().max()) > 1e-3)

    vocab = ["<pad>", "<eos>"] + [str(i) for i in range(n - 2)]
    rows = [[1, 3, 4, 5, 1], [1, 6, 7, 1], [1, 8, 9, 10, 11, 3, 1]]
    padded, mask = TR.pad(rows, vocab)
    lp = on.logits(padded).detach()
    alone = [on.logits(torch.tensor([r])).detach()[0] for r in rows]
    err = max(float((lp[i, :len(r)] - alone[i]).abs().max()) for i, r in enumerate(rows))
    nll = on.loss(padded, mask)[1].item()
    parts = [F_ce(on.logits(torch.tensor([r[:-1]])).detach()[0], torch.tensor(r[1:])) for r in rows]
    want = sum(p * (len(r) - 1) for p, r in zip(parts, rows)) / sum(len(r) - 1 for r in rows)
    check("adim 2: sagdaki dolgu sonucu degistirmez; kayip yalniz gercek hedeflerin ortalamasi",
          err < 1e-6 and abs(nll - want) < 1e-6, "fark %.1e  kayip %.6f / %.6f" % (err, nll, want))

    data = D.build()
    sids, smask = EK.sequences(data)
    nv = len(data["vocab"])
    before = SequenceModel(nv).tokens.fixed_points.clone()
    m, curve = TR.train_seq("step2", sids[:64], smask[:64], nv, steps=20, log_at=(0, 20))   # egitim tesisati: 64 dizi yeter (tam veri 2.560)
    check("adim 2: egitimde PF bit duzeyinde degismez, W_context 0'dan ayrilir, kayip iner",
          torch.equal(m.tokens.fixed_points, before) and curve[-1]["W_context"] > 0 and curve[-1]["nll"] < curve[0]["nll"])


def reference_blocks(m, ids):
    """Adim 3 formulu, float64, modelden bagimsiz: h = PL; her turda attention (durumlari getirir), W_context, FactUnits;
    cikis h (W_next yok).
    rope: q ve k karmasik sayi carpimiyla dondurulur (apply_rope'tan bagimsiz)."""
    unit = lambda v: v / v.norm(dim=-1, keepdim=True)
    dd = lambda t: t.detach().double()
    PL = unit(dd(m.tokens.fixed_points) + dd(m.tokens.shift))
    h = PL[ids]
    T = ids.shape[-1]
    future = torch.ones(T, T, dtype=torch.bool).triu(1)
    blocks = [m.blocks[0]] * m.turns if m.shared else list(m.blocks)
    def ln(v, mod):                                              # LayerNorm, elle: (v - ort) / sqrt(var + eps) . kazanc + kayma
        mu, var = v.mean(-1, keepdim=True), v.var(-1, unbiased=False, keepdim=True)
        return (v - mu) / torch.sqrt(var + mod.eps) * dd(mod.weight) + dd(mod.bias)

    for b in blocks:
        at = b.attention
        na = (lambda v, mod=getattr(b, "norm_attention", None): ln(v, mod)) if m.layer_norm else unit
        nf = (lambda v, mod=getattr(b, "norm_facts", None): ln(v, mod)) if m.layer_norm else unit
        x = h if m.stream_norm else na(h)                       # akis normalize edilmiyorsa okunan kopya normalize
        q, k = unit(x @ dd(at.W_query).T), unit(x @ dd(at.W_key).T)
        if getattr(m, "rope", False):
            dh = q.shape[-1]
            theta = torch.arange(T, dtype=torch.float64)[:, None] * 10000.0 ** (-torch.arange(0, dh, 2, dtype=torch.float64) / dh)
            rot = lambda y: torch.view_as_real(torch.view_as_complex(y.reshape(*y.shape[:-1], dh // 2, 2).contiguous())
                                               * torch.polar(torch.ones_like(theta), theta)).flatten(-2)
            q, k = rot(q), rot(k)
        a = torch.softmax((at.scale * q @ k.transpose(-1, -2)).masked_fill(future, float("-inf")), -1)
        added = (a @ x) @ dd(at.W_context).T
        if m.stream_norm:
            h = na(h + added)
            u = torch.relu(h @ dd(b.facts.W_fact_in).T - dd(b.facts.fact_threshold))
            h = nf(h + u @ dd(b.facts.W_fact_out).T)
        else:
            h = h + added
            u = torch.relu(nf(h) @ dd(b.facts.W_fact_in).T - dd(b.facts.fact_threshold))
            h = h + u @ dd(b.facts.W_fact_out).T
    if m.layer_norm:
        return ln(h, m.norm_final) @ PL.T
    scale = torch.exp(dd(m.log_output_scale)) if getattr(m, "learn_output_scale", False) else m.scale   # e^tau ya da sabit
    return scale * (h if m.stream_norm else unit(h)) @ PL.T


def t_step3():
    import functools
    import model_y
    TURNS = 2
    # 28 Eylul oncesi tasarimin testleri: paket (normalized_update, sphere_weights) kapali, tek Block x 2 tur; paket
    # t_normalized_update'te, 2 x 2 t_layers'ta
    BlockModel = functools.partial(model_y.BlockModel, normalized_update=False, sphere_weights=False, canon=False, turns=2,
                                   layers=1, heads=1, fact_activation="relu", learn_output_scale=False, loss_chunk=0,
                                   output_link=False, shared_facts=True)
    n, d = 12, 6
    g = torch.Generator().manual_seed(9)
    ids = torch.randint(0, n, (3, 9), generator=g)

    for shared in (True, False):
        m = BlockModel(n, d=d, units=10, shared=shared)
        P = m.tokens.points().detach()
        err = float((m.logits(ids).detach() - m.scale * P[ids] @ P.T).abs().max())
        check("adim 3: baslangicta (W_context = 0, W_fact_out = 0) durum PL'de kalir, skor = scale <PL_t, PL>, %s" % (
            "shared" if shared else "separate"), err < 1e-5, "fark %.1e" % err)
    check("adim 3: W_next yok", not any("W_next" in k for k in BlockModel(n, d=d, units=10).state_dict()))

    for kw in (dict(shared=True), dict(shared=False)):
        m = BlockModel(n, d=d, units=10, **kw)
        with torch.no_grad():
            for p_ in m.parameters():
                if p_.requires_grad:
                    p_.copy_(0.5 * torch.randn(p_.shape, generator=g))
        err = float((m.logits(ids).detach().double() - reference_blocks(m, ids)).abs().max())
        check("adim 3: skor = tasarim formulu (bagimsiz float64), %s" % kw, err < 1e-4, "fark %.1e" % err)

    base_logits = m.logits(ids).detach()
    changed = ids.clone()
    changed[:, 5] = (changed[:, 5] + 1) % n
    after = m.logits(changed).detach()
    check("adim 3: nedensellik iki turda da -- konum 5 degisince 0-4 aynen kalir",
          float((after[:, :5] - base_logits[:, :5]).abs().max()) < 1e-5 and float((after[:, 5:] - base_logits[:, 5:]).abs().max()) > 1e-3)

    vocab = ["<pad>", "<eos>"] + [str(i) for i in range(n - 2)]
    rows = [[1, 3, 4, 5, 1], [1, 6, 7, 1], [1, 8, 9, 10, 11, 3, 1]]
    padded, mask = TR.pad(rows, vocab)
    lp = m.logits(padded).detach()
    err = max(float((lp[i, :len(r)] - m.logits(torch.tensor([r])).detach()[0]).abs().max()) for i, r in enumerate(rows))
    check("adim 3: sagdaki dolgu sonucu degistirmez", err < 1e-5, "fark %.1e" % err)

    import itertools
    starts = [(s_, k_, tuple(t_.flatten()[:8].tolist())) for s_ in range(4)
              for k_, t_ in BlockModel(238, seed=s_, shared=False).state_dict().items() if t_.abs().sum() > 0]
    shared_start = [(a_[:2], b_[:2]) for a_, b_ in itertools.combinations(starts, 2) if a_[2] == b_[2]]
    m0 = BlockModel(238)
    kept = [m0.tokens.fixed_points[0, :3].tolist(), m0.blocks[0].attention.W_query[0, :3].tolist(),
            m0.blocks[0].facts.W_fact_in[0, :3].tolist()]
    ref = [[-0.134501650929451, -0.13766998052597046, -0.029936080798506737],
           [-0.10216674953699112, -0.06944606453180313, -0.10333619266748428],
           [-0.06384969502687454, 0.1285339742898941, -0.04414394870400429]]
    check("adim 3: tohumlar rastgele sayi paylasmaz (0-3 arasinda ayni baslayan tensor yok); tohum 0'in baslangici degismedi",
          not shared_start and all(abs(a_ - b_) < 1e-7 for ka, kb in zip(kept, ref) for a_, b_ in zip(ka, kb)),
          str(shared_start[:3]))

    sh, se = BlockModel(n, d=d, units=10, shared=True), BlockModel(n, d=d, units=10, shared=False)
    per_block = sum(p_.numel() for p_ in sh.blocks[0].parameters())
    count = lambda mm: sum(p_.numel() for p_ in mm.parameters())
    tb = sh.turn_blocks()
    check("adim 3: shared tek Block'u her turda kullanir, separate her tura ayri Block",
          len(sh.blocks) == 1 and len(tb) == TURNS and all(b is tb[0] for b in tb) and len(se.blocks) == TURNS
          and count(se) - count(sh) == (TURNS - 1) * per_block and len(sh.hidden(ids)) == TURNS + 1)

    # STREAM_NORM=False: akis normalize edilmez
    base, free = BlockModel(n, d=d, units=10), BlockModel(n, d=d, units=10, stream_norm=False)
    err0 = float((base.logits(ids) - free.logits(ids)).detach().abs().max())
    for kw in (dict(shared=True), dict(shared=False)):
        mf = BlockModel(n, d=d, units=10, stream_norm=False, **kw)
        with torch.no_grad():
            for p_ in mf.parameters():
                if p_.requires_grad:
                    p_.copy_(0.5 * torch.randn(p_.shape, generator=g))
        err = float((mf.logits(ids).detach().double() - reference_blocks(mf, ids)).abs().max())
        check("adim 3, stream_norm=False: skor = tasarim formulu (bagimsiz float64; akis normalize edilmez), %s" % kw,
              err < 1e-4, "fark %.1e" % err)
    hs = mf.hidden(ids)
    grows = float(hs[-1].norm(dim=-1).mean()) > 1.01
    bl = mf.logits(ids).detach()
    ch = ids.clone()
    ch[:, 5] = (ch[:, 5] + 1) % n
    af = mf.logits(ch).detach()
    check("adim 3, stream_norm=False: baslangicta varsayilanla ayni skor; varsayilan True; akis boyu 1'den ayrilir; nedensellik",
          err0 < 1e-6 and BlockModel(n, d=d, units=10).stream_norm and grows
          and float((af[:, :5] - bl[:, :5]).abs().max()) < 1e-5 and float((af[:, 5:] - bl[:, 5:]).abs().max()) > 1e-3,
          "baslangic farki %.1e" % err0)

    # LAYER_NORM=True: L2 norm yerine LayerNorm (ogrenilen kazanc ve kayma), cikista sabit scale yok
    for kw in (dict(shared=True, stream_norm=False), dict(shared=False, stream_norm=False), dict(shared=True, stream_norm=True)):
        ml = BlockModel(n, d=d, units=10, layer_norm=True, **kw)
        with torch.no_grad():
            for p_ in ml.parameters():
                if p_.requires_grad:
                    p_.copy_(0.5 * torch.randn(p_.shape, generator=g))
        err = float((ml.logits(ids).detach().double() - reference_blocks(ml, ids)).abs().max())
        check("adim 3, layer_norm=True: skor = tasarim formulu (bagimsiz float64, elle LayerNorm), %s" % kw, err < 1e-4,
              "fark %.1e" % err)
    cnt = lambda mm: sum(p_.numel() for p_ in mm.parameters() if p_.requires_grad)
    check("adim 3, layer_norm=True: varsayilan False; tek blokta 3 LayerNorm (+384 sayi, 238 token), ayri katmanda 5 (+640)",
          not BlockModel(n, d=d, units=10).layer_norm
          and cnt(BlockModel(238, layer_norm=True, stream_norm=False)) - cnt(BlockModel(238)) == 384
          and cnt(BlockModel(238, shared=False, layer_norm=True, stream_norm=False)) - cnt(BlockModel(238, shared=False)) == 640)

    # ROPE=True: attention'da q ve k konuma gore dondurulur
    for kw in (dict(shared=True), dict(shared=False, stream_norm=False, layer_norm=True)):
        mr = BlockModel(n, d=d, units=10, rope=True, **kw)
        with torch.no_grad():
            for p_ in mr.parameters():
                if p_.requires_grad:
                    p_.copy_(0.5 * torch.randn(p_.shape, generator=g))
        err = float((mr.logits(ids).detach().double() - reference_blocks(mr, ids)).abs().max())
        check("adim 3, rope=True: skor = tasarim formulu (bagimsiz float64; RoPE karmasik carpimla), %s" % kw, err < 1e-4,
              "fark %.1e" % err)
    at, x = mr.blocks[0].attention, mr.blocks[0].norm_attention(mr.hidden(ids)[0])
    q_, k_ = at.queries_keys(x)
    err_w = float((at.weights(x) @ x - at(x)).abs().max())
    rb = mr.logits(ids).detach()
    ch = ids.clone()
    ch[:, 5] = (ch[:, 5] + 1) % n
    ra = mr.logits(ch).detach()
    check("adim 3, rope=True: varsayilan True (Model X); sayi degismez; q, k boyu 1 kalir; weights() ileri hesapla ayni; "
          "nedensellik",
          BlockModel(n, d=d, units=10).rope and all(b.attention.rope for b in BlockModel(n, d=d, units=10).blocks)
          and cnt(BlockModel(238, rope=False)) == cnt(BlockModel(238))
          and float((q_.norm(dim=-1) - 1).abs().max()) < 1e-5 and float((k_.norm(dim=-1) - 1).abs().max()) < 1e-5
          and err_w < 1e-5
          and float((ra[:, :5] - rb[:, :5]).abs().max()) < 1e-5 and float((ra[:, 5:] - rb[:, 5:]).abs().max()) > 1e-3,
          "weights farki %.1e" % err_w)

    data = D.build()
    sids, smask = EK.sequences(data)
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
    whole, _ = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=5, log_at=())
    via, _ = TR.train_seq("shared", None, None, nv, steps=5, log_at=(), batches=lambda step: (sids[:64], smask[:64]))
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
    mk, _ = TR.train_seq("shared", sids[:8], smask[:8], nv, steps=1, log_at=(), model_kw=dict(d=32, turns=3, layers=1, units=64))
    mt, _ = TR.train_seq("transformer", sids[:8], smask[:8], nv, steps=1, log_at=(), model_kw=dict(d=32, layers=1))
    check("train_seq, model_kw: ayarlar modele ulasir (BlockModel d 32, 3 tur, 64 birim; transformer d 32, 1 katman)",
          tuple(mk.blocks[0].attention.W_query.shape) == (32, 32) and mk.turns == 3 and len(mk.hidden(sids[:2])) == 4
          and tuple(mk.blocks[0].facts.W_fact_in.shape) == (64, 32)
          and len(mt.layers) == 1 and mt.embedding.weight.shape[1] == 32)

    ml, curve_l = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=20, log_at=(0, 20), stream_norm=False, normalized_update=False, sphere_weights=False, layer_norm=True)
    check("adim 3, layer_norm=True: train_seq ile egitilir, kayip iner", ml.layer_norm and curve_l[-1]["nll"] < curve_l[0]["nll"],
          "%.3f -> %.3f" % (curve_l[0]["nll"], curve_l[-1]["nll"]))
    mf, curve_f = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=20, log_at=(0, 20), stream_norm=False, normalized_update=False, sphere_weights=False)
    check("adim 3, stream_norm=False: train_seq ile egitilir, kayip iner", not mf.stream_norm and curve_f[-1]["nll"] < curve_f[0]["nll"],
          "%.3f -> %.3f" % (curve_f[0]["nll"], curve_f[-1]["nll"]))
    before = BlockModel(nv).tokens.fixed_points.clone()
    m, curve = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=20, log_at=(0, 20))
    check("adim 3: egitimde PF bit duzeyinde degismez, W_context 0'dan ayrilir, kayip iner",
          torch.equal(m.tokens.fixed_points, before) and curve[-1]["W_context"] > 0 and curve[-1]["nll"] < curve[0]["nll"])

    # egitim tarifi: cosine decay + gradient clipping standart; None verilirse eski tarif birebir
    seen = []
    real_adam = torch.optim.Adam

    class Spy(real_adam):
        def step(self, *a, **k):
            seen.append((self.param_groups[0]["lr"], float(torch.nn.utils.clip_grad_norm_(self.param_groups[0]["params"], 1e9))))
            return super().step(*a, **k)
    torch.optim.Adam = Spy
    try:
        TR.train_seq("shared", sids[:64], smask[:64], nv, steps=4, log_at=(), lr=0.01, lr_floor=0.1, grad_clip=0.5, optimizer="adam", schedule="cosine")
        cosine = [0.01 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / 4))) for t in range(4)]
        ok_sched = all(abs(lr_ - c) < 1e-12 for (lr_, _), c in zip(seen, cosine))
        ok_clip = all(g <= 0.5 + 1e-5 for _, g in seen)
        seen.clear()
        TR.train_seq("shared", sids[:64], smask[:64], nv, steps=4, log_at=(), lr=0.01, optimizer="adam", schedule="cosine")
        standard = [0.01 * (TR.LR_FLOOR + (1 - TR.LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * t / 4))) for t in range(4)]
        ok_default = (all(abs(lr_ - c) < 1e-12 for (lr_, _), c in zip(seen, standard))
                      and all(g <= TR.GRAD_CLIP + 1e-5 for _, g in seen))
        seen.clear()
        TR.train_seq("shared", sids[:64], smask[:64], nv, steps=3, log_at=(), lr=0.01, lr_floor=None, grad_clip=None, optimizer="adam", schedule="cosine")
        ok_old = all(lr_ == 0.01 for lr_, _ in seen) and any(g > 0.5 for _, g in seen)
    finally:
        torch.optim.Adam = real_adam
    check("egitim tarifi (optimizer='adam', schedule='cosine', 27 Eylul oncesi): cosine decay ile lr LR'den LR x lr_floor'a iner, gradient boyu grad_clip'i gecmez; standart LR_FLOOR ve "
          "GRAD_CLIP; None verilirse sabit lr", ok_sched and ok_clip and ok_default and ok_old)
    a1, _ = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=3, log_at=(), seed=0)
    a2, _ = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=3, log_at=(), seed=1)
    check("egitim tarifi: seed modeli degistirir (PF dahil), seed=0 varsayilanla ayni",
          not torch.equal(a1.tokens.fixed_points, a2.tokens.fixed_points)
          and torch.equal(a1.tokens.fixed_points, before))

    groups = []
    real_adamw = torch.optim.AdamW

    class SpyW(real_adamw):
        def __init__(self, param_groups, **k):
            super().__init__(param_groups, **k)
            groups.extend(self.param_groups)
    torch.optim.AdamW = SpyW
    try:
        mw, _ = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=1, log_at=(), weight_decay=0.1, optimizer="adam")
    finally:
        torch.optim.AdamW = real_adamw
    names = {id(p_): k.split(".")[-1] for k, p_ in mw.named_parameters()}
    decayed = sorted({names[id(p_)] for g_ in groups if g_["weight_decay"] == 0.1 for p_ in g_["params"]})
    kept = sorted({names[id(p_)] for g_ in groups if g_["weight_decay"] == 0.0 for p_ in g_["params"]})
    check("weight decay: yalniz W_ matrislerine; shift, alpha, Canon agirliklari, cikis olcegi ve cikis bagi (phi) haric",
          decayed == sorted(["W_query", "W_key", "W_context", "W_fact_in", "W_fact_up", "W_fact_out", "W_value"])
          and kept == ["alpha_attention", "alpha_facts", "canon_weights", "link_q", "link_u", "log_output_scale", "shift"],
          "%s | %s" % (decayed, kept))

    # 27 Eylul tarifi: Muon (gizli matrisler) + Adam, WSD takvimi, compile varsayilan acik; masked_nll; RoPE en az fp32
    import inspect
    from model_y import apply_rope, masked_nll
    sig = inspect.signature(TR.train_seq).parameters
    check("tarif varsayilanlari: optimizer 'muon', schedule 'wsd', cooldown 0,2, compile True (kullanici, 27 Eylul)",
          sig["optimizer"].default == TR.OPTIMIZER == "muon" and sig["schedule"].default == TR.SCHEDULE == "wsd"
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
    k_math, _ = TR.train_seq("shared", sids[:8], smask[:8], nv, steps=2, log_at=())
    k_flash, _ = TR.train_seq("shared", sids[:8], smask[:8], nv, steps=2, log_at=(), attention_kernel="flash")
    refused_k = 0
    for bad in (dict(attention_kernel="efficient"), dict(newton_schulz_precision="fp16")):
        try:
            TR.train_seq("shared", sids[:8], smask[:8], nv, steps=1, log_at=(), **bad)
        except AssertionError:
            refused_k += 1
    check("ATTENTION_KERNEL / NEWTON_SCHULZ_PRECISION (29 Eylul): varsayilan math / fp32; bf16 Newton-Schulz tekil degerleri ~1, "
          "fp32'ye yakin, tipi G'ninki, train_seq optimizer'a iletir; CPU'da flash = math (bit duzeyinde); gecersiz deger reddedilir",
          TR.ATTENTION_KERNEL == "math" and TR.NEWTON_SCHULZ_PRECISION == "fp32" and float((svb - 1).abs().max()) < 0.35
          and rel < 0.05 and ob.dtype == Gm.dtype and ns_set == "bf16" and ns_opts[-1].newton_schulz_precision == "fp32"
          and not any(torch.isnan(p_).any() for p_ in nb.parameters())
          and all(torch.equal(a_, b_) for a_, b_ in zip(k_math.state_dict().values(), k_flash.state_dict().values()))
          and refused_k == 2,
          "bf16 tekil %.3f..%.3f, fp32'den fark %.3f" % (float(svb.min()), float(svb.max()), rel))

    opts, lrs = [], {}

    def grab(step, model, opt):
        opts.append(opt)
        lrs[step] = opt.param_groups[0]["lr"]
    mm, curve_m = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=10, log_at=(0, 10), save_every=1, save=grab)
    names_m = {id(p_): k.split(".")[-1] for k, p_ in mm.named_parameters()}
    in_muon = sorted({names_m[id(p_)] for g_ in opts[0].param_groups if g_["use_muon"] for p_ in g_["params"]})
    in_adam = sorted({names_m[id(p_)] for g_ in opts[0].param_groups if not g_["use_muon"] for p_ in g_["params"]})
    wsd = [0.01 * (1.0 if t < 8 else TR.LR_FLOOR + (1 - TR.LR_FLOOR) * (1 - math.sqrt((t - 8) / 2))) for t in range(1, 11)]
    check("Muon + WSD (varsayilan): Muon'da W_context, W_fact_in, W_fact_up, W_fact_out, W_value; Adam'da noktalar, W_query, "
          "W_key, cikis olcegi; "
          "lr 8. adima kadar sabit, sonra 1 - sqrt ile LR x LR_FLOOR'a; kayip iner",
          isinstance(opts[0], TR.Muon) and in_muon == ["W_context", "W_fact_in", "W_fact_out", "W_fact_up", "W_value"]
          and in_adam == ["W_key", "W_query", "alpha_attention", "alpha_facts", "canon_weights", "link_q", "link_u",
                          "log_output_scale", "shift"]
          and all(abs(lrs[t] - w) < 1e-12 for t, w in zip(range(1, 11), wsd)) and curve_m[-1]["nll"] < curve_m[0]["nll"],
          "%s | %s | lr %s" % (in_muon, in_adam, [round(lrs[t], 5) for t in range(1, 11)]))

    import copy
    packs_m = {}
    keep_m = lambda step, model, opt: packs_m.setdefault(step, dict(step=step, model=copy.deepcopy(model.state_dict()),
                                                                     optimizer=copy.deepcopy(opt.state_dict())))
    full_m, _ = TR.train_seq("shared", sids[:40], smask[:40], nv, steps=6, log_at=(), save_every=2, save=keep_m)
    res_m, _ = TR.train_seq("shared", sids[:40], smask[:40], nv, steps=6, log_at=(), checkpoint=packs_m[4])
    ta, _ = TR.train_seq("transformer_novalue", sids[:8], smask[:8], nv, steps=1, log_at=(), save_every=1, save=grab)
    names_t = {id(p_): k for k, p_ in ta.named_parameters()}
    t_muon = sorted({names_t[id(p_)] for g_ in opts[-1].param_groups if g_["use_muon"] for p_ in g_["params"]})
    try:
        TR.train_seq("shared", sids[:8], smask[:8], nv, steps=1, log_at=(), weight_decay=0.1)
        refused_wd = False
    except AssertionError:
        refused_wd = True
    check("Muon + WSD: 4. adim paketinden surdurulen = kesintisiz, bit duzeyinde; transformer'da Muon W_out ve MLP'de; "
          "weight_decay yalniz adam ile",
          all(torch.equal(a_, b_) for a_, b_ in zip(full_m.state_dict().values(), res_m.state_dict().values()))
          and t_muon == sorted("layers.%d.%s.weight" % (i, w) for i in range(2) for w in ("W_out", "W_mlp_in", "W_mlp_out"))
          and refused_wd, str(t_muon))

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


def reference_transformer(m, ids):
    """model_y_transformer formulu, float64, modelden bagimsiz: LayerNorm, RoPE (karmasik sayi carpimiyla), nedensel
    softmax attention, GELU (erf), tied cikis."""
    dd = lambda t: t.detach().double()

    def ln(x, mod):
        mu, var = x.mean(-1, keepdim=True), x.var(-1, unbiased=False, keepdim=True)
        return (x - mu) / torch.sqrt(var + mod.eps) * dd(mod.weight) + dd(mod.bias)

    lin = lambda x, mod: x @ dd(mod.weight).T + dd(mod.bias)
    E = dd(m.embedding.weight)
    h = E[ids]
    B, T = ids.shape
    future = torch.ones(T, T, dtype=torch.bool).triu(1)
    for layer in m.layers:
        H = layer.heads
        dh = h.shape[-1] // H
        x = ln(h, layer.norm_attention)
        split = lambda y: y.view(B, T, H, dh).transpose(1, 2)
        q, k = split(lin(x, layer.W_query)), split(lin(x, layer.W_key))
        v = split(lin(x, layer.W_value)) if layer.W_value is not None else split(x)
        theta = torch.arange(T, dtype=torch.float64)[:, None] * 10000.0 ** (-torch.arange(0, dh, 2, dtype=torch.float64) / dh)
        rot = lambda y: torch.view_as_real(torch.view_as_complex(y.reshape(B, H, T, dh // 2, 2).contiguous())
                                           * torch.polar(torch.ones_like(theta), theta)).flatten(-2)
        if layer.rope:
            q, k = rot(q), rot(k)
        s = (q @ k.transpose(-1, -2) / math.sqrt(dh)).masked_fill(future, float("-inf"))
        a = torch.softmax(s, -1) @ v
        h = h + lin(a.transpose(1, 2).reshape(B, T, -1), layer.W_out)
        u = lin(ln(h, layer.norm_mlp), layer.W_mlp_in)
        h = h + lin(0.5 * u * (1 + torch.erf(u / math.sqrt(2))), layer.W_mlp_out)
    return ln(h, m.norm_final) @ E.T


def t_transformer():
    from model_y_transformer import HEADS, LAYERS, TransformerModel, apply_rope
    count = lambda mm: sum(p_.numel() for p_ in mm.parameters() if p_.requires_grad)
    check("transformer: 238 token, D 64, MLP 256, 2 katman -> 115.328 ogrenilen sayi; head sayisi degistirmez; LAYERS 2, HEADS 1",
          count(TransformerModel(238)) == 115328 and count(TransformerModel(238, heads=4)) == 115328
          and LAYERS == 2 and HEADS == 1, str(count(TransformerModel(238))))

    n, d = 12, 8
    g = torch.Generator().manual_seed(13)
    ids = torch.randint(0, n, (3, 9), generator=g)
    check("transformer, value_matrix=False: V yok -> 238 token'da 107.008 sayi (katman basina 4.160 eksik)",
          count(TransformerModel(238, value_matrix=False)) == 107008
          and all(layer.W_value is None for layer in TransformerModel(238, value_matrix=False).layers),
          str(count(TransformerModel(238, value_matrix=False))))
    for heads, vm, rope in ((1, True, True), (4, True, True), (1, False, True), (1, False, False)):
        m = TransformerModel(n, d=d, units=12, heads=heads, value_matrix=vm, rope=rope)
        with torch.no_grad():
            for p_ in m.parameters():
                p_.copy_(0.5 * torch.randn(p_.shape, generator=g))
        err = float((m.logits(ids).detach().double() - reference_transformer(m, ids)).abs().max())
        check("transformer: skor = tasarim formulu (bagimsiz float64; RoPE karmasik carpimla), %d head%s%s" % (
              heads, "" if vm else ", V yok", "" if rope else ", rope=False"), err < 1e-4, "fark %.1e" % err)
    m = TransformerModel(n, d=d, units=12, heads=4)
    with torch.no_grad():
        for p_ in m.parameters():
            p_.copy_(0.5 * torch.randn(p_.shape, generator=g))

    q0, k0 = torch.randn(4, generator=g), torch.randn(4, generator=g)
    rq, rk = apply_rope(q0.expand(1, 1, 7, 4)), apply_rope(k0.expand(1, 1, 7, 4))
    S = (rq @ rk.transpose(-1, -2))[0, 0]
    err = float((S[1:, 1:] - S[:-1, :-1]).abs().max())
    check("transformer: RoPE -- ayni q, k vektorlerinin skoru yalniz konum farkina bagli", err < 1e-5 and float(S.std()) > 1e-3,
          "fark %.1e" % err)

    base_logits = m.logits(ids).detach()
    changed = ids.clone()
    changed[:, 5] = (changed[:, 5] + 1) % n
    after = m.logits(changed).detach()
    vocab = ["<pad>", "<eos>"] + [str(i) for i in range(n - 2)]
    rows = [[1, 3, 4, 5, 1], [1, 6, 7, 1], [1, 8, 9, 10, 11, 3, 1]]
    padded, _ = TR.pad(rows, vocab)
    lp = m.logits(padded).detach()
    err = max(float((lp[i, :len(r)] - m.logits(torch.tensor([r])).detach()[0]).abs().max()) for i, r in enumerate(rows))
    check("transformer: nedensellik (konum 5 degisince 0-4 aynen kalir) ve sagdaki dolgu sonucu degistirmez",
          float((after[:, :5] - base_logits[:, :5]).abs().max()) < 1e-5 and float((after[:, 5:] - base_logits[:, 5:]).abs().max()) > 1e-3
          and err < 1e-5, "dolgu farki %.1e" % err)

    data = D.build()
    sids, smask = EK.sequences(data)
    nv = len(data["vocab"])
    m, curve = TR.train_seq("transformer", sids[:64], smask[:64], nv, steps=20, log_at=(0, 20))
    try:
        TR.train_seq("transformer", sids[:64], smask[:64], nv, steps=1, log_at=(), weight_decay=0.1)
        refused = False
    except AssertionError:
        refused = True
    mv, curve_v = TR.train_seq("transformer_novalue", sids[:64], smask[:64], nv, steps=20, log_at=(0, 20))
    check("transformer: train_seq ayni tarifle egitir, kayip iner ('transformer' ve 'transformer_novalue'); weight decay "
          "istenirse reddedilir (gruplar tanimsiz)",
          isinstance(m, TransformerModel) and curve[-1]["nll"] < curve[0]["nll"] and refused
          and all(layer.W_value is None for layer in mv.layers) and curve_v[-1]["nll"] < curve_v[0]["nll"],
          "%.3f -> %.3f; V'siz %.3f -> %.3f" % (curve[0]["nll"], curve[-1]["nll"], curve_v[0]["nll"], curve_v[-1]["nll"]))

    mr, curve_r = TR.train_seq("transformer_novalue", sids[:64], smask[:64], nv, steps=20, log_at=(0, 20), rope=False)
    check("transformer, rope=False: train_seq ayari modele ulasir (her katmanda rope False), kayip iner; varsayilan True",
          all(not layer.rope for layer in mr.layers) and all(layer.rope for layer in mv.layers)
          and curve_r[-1]["nll"] < curve_r[0]["nll"],
          "%.3f -> %.3f" % (curve_r[0]["nll"], curve_r[-1]["nll"]))


def F_ce(logits, targets):
    return float(torch.nn.functional.cross_entropy(logits, targets))


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
    off = dict(normalized_update=False, sphere_weights=False)          # 28 Eylul oncesi guncelleme (LayerNorm / normsuz akis icin sart)
    for kw in (dict(), dict(heads=1), dict(canon=False), dict(shared=False), dict(rope=False), dict(off),
               dict(off, stream_norm=False),
               dict(off, stream_norm=False, layer_norm=True), dict(off, shared=False, stream_norm=False, layer_norm=True)):
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
    edge = BlockModel(n, d=16, units=24, t_max=64, canon=False)
    check("generate: n = 0 bos, n = 1 istem hesabinin son konumu; transformer eski yoldan",
          TR.generate(edge, prompts[:3], 0) == [[], [], []]
          and TR.generate(edge, prompts[:3], 1) == TR.generate(edge, prompts[:3], 1, cached=False)
          and len(TR.generate(TR.TransformerModel(n, d=16, layers=1), prompts[:3], 4)[2]) == 4)
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

    # egitilmis agirlik: akrabalik verisinde 60 adim; sinav sorulari (<steps> dahil) ve uzun devam
    data = D.build()
    sids, smask = EK.sequences(data)
    nv = len(data["vocab"])
    trained, curve = TR.train_seq("shared", sids[:256], smask[:256], nv, steps=60, log_at=(0, 60), canon=False)
    ix = {w: i for i, w in enumerate(data["vocab"])}
    qs = [[ix[D.EOS]] + [ix[t] for t in e["prompt"]] for e in data["exam"]][:96]
    old, new = TR.generate(trained, qs, 24, cached=False), TR.generate(trained, qs, 24)
    scores, _ = cached_scores(trained, qs[:24], new[:24])
    err = max(float((scores[i] - trained.logits(torch.tensor([q + new[i]])).detach()[0, len(q) - 1:]).abs().max())
              for i, q in enumerate(qs[:24]))
    check("generate, egitilmis Model X (akrabalik, 256 cumle, 60 adim): 96 sinav sorusu x 24 token onbellekli = tam yeniden "
          "hesap; skorlar tolerans icinde", old == new and err < 1e-4 and curve[-1]["nll"] < curve[0]["nll"],
          "fark %.1e  kayip %.3f -> %.3f" % (err, curve[0]["nll"], curve[-1]["nll"]))


def t_normalized_update():
    """Kusur 1a (normalized_update) ve 2a (sphere_weights): kapaliyken bugunku model; formul bagimsiz float64 hesapla ayni;
    blok ciktisi ne kadar buyurse buyusun durum alpha kadar doner; agirliklar her adimdan sonra birim; surdurme ayni."""
    import copy
    import torch.nn.functional as F
    from model_y import ALPHA_INIT, LAST_FACTS_ALPHA_INIT, BlockModel, apply_rope
    data = D.build()
    sids, smask = EK.sequences(data)
    nv = len(data["vocab"])

    # tasarim formulu dogrusal skorla: phi (OUTPUT_LINK, 29 Eylul varsayilan) t_output_link'te
    today, off = (BlockModel(nv, normalized_update=False, sphere_weights=False, output_link=False),
                  BlockModel(nv, normalized_update=False, sphere_weights=False, output_link=False))
    both = BlockModel(nv, normalized_update=True, sphere_weights=True, output_link=False)
    b0 = both.blocks[0]
    unit_rows = lambda w: float((w.norm(dim=1) - 1).abs().max())
    unit_cols = lambda w: float((w.norm(dim=0) - 1).abs().max())
    check("anahtarlar kapali = bugunku model (ayni agirliklar, alpha yok, W_context ve W_fact_out 0); acikken alpha (tur, d) = "
          "ALPHA_INIT (son turun FactUnits'i LAST_FACTS_ALPHA_INIT), W_context / W_fact_out rastgele, satir / sutunlar birim",
          all(torch.equal(a, b) for a, b in zip(today.state_dict().values(), off.state_dict().values()))
          and not hasattr(off, "alpha_attention") and not off.blocks[0].attention.W_context.any()
          and both.alpha_attention.shape == (both.turns, 64) and bool((both.alpha_attention == ALPHA_INIT).all())
          and bool((both.alpha_facts[:-1] == ALPHA_INIT).all()) and bool((both.alpha_facts[-1] == LAST_FACTS_ALPHA_INIT).all())
          and max(unit_rows(b0.attention.W_query), unit_rows(b0.attention.W_key), unit_rows(b0.facts.W_fact_in),
                  unit_cols(b0.attention.W_context), unit_cols(b0.facts.W_fact_out)) < 1e-6)

    # bagimsiz float64 referans (tasarim formulu): rastgele parametrelerle
    g = torch.Generator().manual_seed(41)
    m = BlockModel(nv, d=32, units=48, normalized_update=True, sphere_weights=True, canon=False, heads=1,
                   output_link=False, shared_facts=True).double()
    with torch.no_grad():
        for name, p_ in m.named_parameters():
            if name.startswith("alpha"):
                p_.copy_(0.5 * torch.rand(p_.shape, generator=g, dtype=torch.float64))
            else:
                p_.add_(0.3 * torch.randn(p_.shape, generator=g, dtype=torch.float64))
        m.normalize_weights()
    ids = sids[:6, :20]
    with torch.no_grad():
        P = F.normalize(m.tokens.fixed_points + m.tokens.shift, dim=-1)
        h = P[ids]
        T = ids.shape[1]
        for i in range(m.turns):
            blk = m.blocks[i % len(m.blocks)]                       # A B A B
            at, fu = blk.attention, blk.facts
            q = apply_rope(F.normalize(h @ at.W_query.T, dim=-1))
            k = apply_rope(F.normalize(h @ at.W_key.T, dim=-1))
            s_ = at.scale * q @ k.transpose(-1, -2)
            s_ = s_.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
            c = torch.softmax(s_, -1) @ h
            u = c @ at.W_context.T
            h = F.normalize(h + m.alpha_attention[i] * (F.normalize(u, dim=-1) - h), dim=-1)
            f = (F.silu(32 ** 0.5 * h @ fu.W_fact_in.T) * (32 ** 0.5 * h @ fu.W_fact_up.T)) @ fu.W_fact_out.T
            h = F.normalize(h + m.alpha_facts[i] * (F.normalize(f, dim=-1) - h), dim=-1)
        ref = torch.exp(m.log_output_scale) * h @ P.T                   # ogrenilen cikis olcegi e^tau
        err = float((m.logits(ids) - ref).abs().max())
    check("normalized_update + sphere_weights: skor = tasarim formulu (bagimsiz float64; h <- norm(h + a (norm(u) - h)), "
          "FactUnits girdisi sqrt(d) x kosinus)", err < 1e-10, "fark %.1e" % err)

    # kusur 1: FactUnits ciktisi 1000 kat buyutulse de durum alpha kadar doner (bugunku modelde silinir)
    def turn_cos(model):
        with torch.no_grad():
            hs = model.hidden(sids[:16])
            return min(float((a * b).sum(-1)[smask[:16]].min()) for a, b in zip(hs[:-1], hs[1:]))
    fixed = BlockModel(nv, normalized_update=True, last_facts_alpha_init=ALPHA_INIT)   # butun alpha'lar 0,1
    now = BlockModel(nv, normalized_update=False, sphere_weights=False)             # 28 Eylul oncesi guncelleme
    with torch.no_grad():
        for mm in (fixed, now):
            mm.blocks[0].attention.W_context.copy_(torch.randn(64, 64, generator=g) / 8)
            mm.blocks[0].facts.W_fact_out.copy_(1000 * torch.randn(mm.blocks[0].facts.W_fact_out.shape, generator=g) / 16)
    c_fixed, c_now = turn_cos(fixed), turn_cos(now)
    check("kusur 1: FactUnits ciktisi 1000 kat buyukken normalized_update'te tur basina cos(h_once, h_sonra) >= 0,95 (alpha "
          "0,1); bugunku guncellemede durum siliniyor", c_fixed >= 0.95 and c_now < 0.5, "en kucuk cos %.3f / bugun %.3f"
          % (c_fixed, c_now))

    # egitim: kayip iner, agirliklar her adimdan sonra birim, alpha Adam'da ve ogreniyor; surdurme bit duzeyinde
    opts = []
    grab = lambda step, model, opt: opts.append(opt)
    kw = dict(normalized_update=True, sphere_weights=True)
    trained, curve = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=20, log_at=(0, 20), save_every=1, save=grab, **kw)
    tb = trained.blocks[0]
    names = {id(p_): k for k, p_ in trained.named_parameters()}
    in_adam = sorted({names[id(p_)] for g_ in opts[-1].param_groups if not g_["use_muon"] for p_ in g_["params"]})
    check("egitim (normalized_update + sphere_weights, Muon + WSD): kayip iner; 20 adimdan sonra satir / sutunlar birim; "
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
    try:
        TR.train_seq("transformer", sids[:8], smask[:8], nv, steps=1, log_at=(), normalized_update=True)
        refused = False
    except AssertionError:
        refused = True
    try:
        BlockModel(nv, normalized_update=True, stream_norm=False, layer_norm=True)
        refused_ln = False
    except AssertionError:
        refused_ln = True
    check("normalized_update + sphere_weights: 4. adim paketinden surdurulen = kesintisiz, bit duzeyinde; onbellekli uretim = "
          "tam yeniden hesap; transformer'da ve LayerNorm'la reddedilir",
          all(torch.equal(a, b) for a, b in zip(full.state_dict().values(), res.state_dict().values()))
          and TR.generate(trained, qs, 10) == TR.generate(trained, qs, 10, cached=False) and refused and refused_ln)

    # LAST_FACTS_ALPHA_INIT: ALPHA_INIT verilince eski baslatma birebir; 1,0'da son durum girdi noktasindan ayrilir, adim-0
    # kaybi ln n + s^2 / (2d) (h rastgele yon: <h, p> ~ N(0, 1/d)), girdiyi tekrar etme kaybolur
    old_init, new_init = (BlockModel(nv, last_facts_alpha_init=ALPHA_INIT, shared_facts=True),   # esikler paylasimli
                          BlockModel(nv, shared_facts=True))                                  # FactUnits ile olculdu
    before = {k: v.clone() for k, v in new_init.state_dict().items()}
    before["alpha_facts"].fill_(ALPHA_INIT)                               # 28 Eylul oncesi kod: torch.full(ALPHA_INIT)
    same_old = list(before) == list(old_init.state_dict()) and all(torch.equal(before[k], v)
                                                                    for k, v in old_init.state_dict().items())
    rows, gv = [], torch.Generator().manual_seed(7)
    big = torch.randint(0, 8004, (4, 33), generator=gv)
    for n_, d_, units_, ids_, mask_, tol in ((nv, 64, 170, sids[:32], smask[:32], 0.35),
                                             (8004, 384, 1024, big, torch.ones_like(big, dtype=torch.bool), 0.1)):
        out_ = {}
        for init in (LAST_FACTS_ALPHA_INIT, ALPHA_INIT):
            m_ = BlockModel(n_, d=d_, units=units_, last_facts_alpha_init=init, shared_facts=True)
            with torch.no_grad():
                keep_ = mask_[:, 1:]
                repeat = float((m_.logits(ids_[:, :-1]).argmax(-1) == ids_[:, :-1])[keep_].float().mean())
                out_[init] = (float(m_.loss(ids_, mask_)[1]), repeat)
        want = math.log(n_) + m_.scale ** 2 / (2 * d_)
        rows.append((n_, d_, want, out_, abs(out_[LAST_FACTS_ALPHA_INIT][0] - want) < tol and out_[ALPHA_INIT][0] > want + 2
                     and out_[ALPHA_INIT][1] > 0.9 and out_[LAST_FACTS_ALPHA_INIT][1] < 0.05))
    check("last_facts_alpha_init: ALPHA_INIT ile eski baslatma bit duzeyinde; 1,0'da adim-0 kaybi ln n + s^2/(2d)'ye yakin "
          "(akrabalik n 242 d 64 gercek cumleler, 0,35; n 8004 d 384 rastgele token, 0,1), 0,1'de 2 nat ustunde; girdiyi "
          "tekrar (argmax = girdi token'i) 0,1'de > %90, 1,0'da < %5",
          LAST_FACTS_ALPHA_INIT == 1.0 and same_old and all(r[-1] for r in rows),
          "  ".join("n %d d %d: formul %.3f / 1,0 kayip %.3f tekrar %.3f / 0,1 kayip %.3f tekrar %.3f" % (
              n_, d_, want, o[LAST_FACTS_ALPHA_INIT][0], o[LAST_FACTS_ALPHA_INIT][1], o[ALPHA_INIT][0], o[ALPHA_INIT][1])
              for n_, d_, want, o, _ in rows))


def t_canon():
    """Canon-A: w = 0 iken Canon'suz modelle bit duzeyinde ayni; attention girdisi resmi kodun hesabiyla (PhysicsLM4
    canon_helper: x + conv1d(groups=d, cekirdek 4, soldan 3 dolgu), aktivasyon ve bias yok) ayni; nedensel; egitim ve
    surdurme; onbellekli uretim (canon_cache) tam hesapla ayni."""
    import copy
    import torch.nn.functional as F
    from model_y import BlockModel
    data = D.build()
    sids, smask = EK.sequences(data)
    nv = len(data["vocab"])
    ids = sids[:6, :20]

    plain, zero = BlockModel(nv, canon=False), BlockModel(nv, canon=True)
    with torch.no_grad():
        same = torch.equal(plain.logits(ids), zero.logits(ids))
    check("canon: w = 0 iken Canon'suz Model X1 ile bit duzeyinde ayni skor; canon_weights (4, d) = 0",
          same and tuple(zero.blocks[0].canon_weights.shape) == (4, 64) and not zero.blocks[0].canon_weights.any())

    # resmi kodun hesabi: out = x + conv1d(x, W, padding=3, groups=d)[..., :T], W[:, 0, 3 - k] = w_k
    g = torch.Generator().manual_seed(51)
    m = BlockModel(nv, canon=True)
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
    trained, curve = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=20, log_at=(0, 20), save_every=1, save=grab, canon=True)
    names = {id(p_): k for k, p_ in trained.named_parameters()}
    in_adam = [names[id(p_)] for g_ in opts[-1].param_groups if not g_["use_muon"] for p_ in g_["params"]]
    packs = {}
    keep = lambda step, model, opt: packs.setdefault(step, dict(step=step, model=copy.deepcopy(model.state_dict()),
                                                                   optimizer=copy.deepcopy(opt.state_dict())))
    full, _ = TR.train_seq("shared", sids[:40], smask[:40], nv, steps=6, log_at=(), save_every=2, save=keep, canon=True)
    res, _ = TR.train_seq("shared", sids[:40], smask[:40], nv, steps=6, log_at=(), checkpoint=packs[4], canon=True)
    qs = [[1] + sids[i, 1:1 + k].tolist() for i, k in enumerate((1, 2, 3, 5, 8, 8, 12, 16))]   # 2-17 token: Canon
    cached_new = TR.generate(trained, qs, 6)                                                 # istem basini da gorur
    scores, _ = cached_scores(trained, qs, cached_new)
    err_c = max(float((scores[i] - trained.logits(torch.tensor([q + cached_new[i]])).detach()[0, len(q) - 1:]).abs().max())
                for i, q in enumerate(qs))
    try:
        TR.train_seq("transformer", sids[:8], smask[:8], nv, steps=1, log_at=(), canon=True)
        refused = False
    except AssertionError:
        refused = True
    check("canon: egitimde kayip iner, canon_weights 0'dan ayrilir ve Adam'da; surdurme bit duzeyinde; onbellekli uretim "
          "(canon_cache) tam hesapla ayni token'lar ve skorlar (istem 2-17 token); transformer'da reddedilir",
          curve[-1]["nll"] < curve[0]["nll"] and bool(trained.blocks[0].canon_weights.abs().max() > 0)
          and "blocks.0.canon_weights" in in_adam
          and all(torch.equal(a, b) for a, b in zip(full.state_dict().values(), res.state_dict().values()))
          and cached_new == TR.generate(trained, qs, 6, cached=False) and err_c < 1e-4 and refused,
          "%.3f -> %.3f  |w| %.4f  skor farki %.1e" % (curve[0]["nll"], curve[-1]["nll"],
                                                     float(trained.blocks[0].canon_weights.abs().max()), err_c))


def t_layers():
    """LAYERS: paylasilan blokta farkli Block sayisi, turlar sirayla (A B A B).  layers=1 bugunku model; layers=2, turns=4
    iki Block'u donusumlu kullanir; ayri blok (separate) Model X2 anahtarlariyla egitilir."""
    import copy
    from model_y import BlockModel
    data = D.build()
    sids, smask = EK.sequences(data)
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
    sep4 = BlockModel(nv, shared=False)                    # ayri blokta layers yok sayilir: her tura bir Block
    for kw in (dict(layers=2, turns=3), dict(layers=3, turns=4)):
        try:
            BlockModel(nv, **kw)
            refused.append(False)
        except AssertionError:
            refused.append(True)
    check("layers: turns layers'in kati degilse reddedilir; ayri blokta layers yok sayilir (her tura bir Block, "
          "model.layers = Block sayisi)", all(refused) and len(sep4.blocks) == sep4.turns == sep4.layers)

    opts = []
    grab = lambda step, model, opt: opts.append(opt)
    kw = dict(layers=2, turns=4)
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

    sep, curve_s = TR.train_seq("separate", sids[:64], smask[:64], nv, steps=20, log_at=(0, 20))
    check("layers: ayri blok (separate) Model X2 anahtarlariyla (normalized_update, sphere_weights, canon) egitilir",
          curve_s[-1]["nll"] < curve_s[0]["nll"] and len(sep.blocks) == sep.turns and sep.canon
          and sep.normalized_update and sep.sphere_weights
          and all(torch.allclose(b.attention.W_key.norm(dim=1), torch.ones(64), atol=1e-5) for b in sep.blocks),
          "%.3f -> %.3f" % (curve_s[0]["nll"], curve_s[-1]["nll"]))

    plain = BlockModel(nv, layers=2, turns=4, canon=False)
    qs = [[1] + sids[i, 1:9].tolist() for i in range(8)]
    check("layers: layers=2 onbellekli uretim tam hesapla ayni (tur basina onbellek)",
          TR.generate(plain, qs, 6) == TR.generate(plain, qs, 6, cached=False))


def t_coherence():
    """SCHEDULE "coherence": mini-batch'te adim iki yarida, birlestirilen gradyan tam batch'inki (ilk adim wsd ile ayni);
    tam batch'te rho = 1 (lr sabit, sonda final_cooldown inisi); mini-batch'te c, rho, ortalama gecerli ve lr = LR x
    ortalama; surdurme bit duzeyinde (ortalama optimizer'in grup kaydinda); lr_floor yoksa reddedilir."""
    import copy
    data = D.build()
    sids, smask = EK.sequences(data)
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
    check("coherence, final_cooldown_shape linear: son %50'de dogrusal x 0'a (D2Z); varsayilan linear",
          TR.FINAL_COOLDOWN_SHAPE == "linear" and all(abs(lin[t] - v) < 1e-12 for t, v in want_l.items())
          and all(lin[t] == TR.LR for t in range(1, 20)), str({t: round(lin[t], 6) for t in (19, 20, 30, 39, 40)}))

    fz = {}
    TR.train_seq("shared", sids[:64], smask[:64], nv, steps=40, log_at=(), save_every=1,
                 save=lambda step, model, opt: fz.setdefault(step, opt.param_groups[0]["lr"]), schedule="coherence",
                 final_cooldown=0.5, lr_floor=0.0, final_cooldown_shape="linear", frozen_lr=0.004)
    try:
        TR.train_seq("shared", sids[:8], smask[:8], nv, steps=1, log_at=(), frozen_lr=0.004, schedule="wsd")
        refused_fz = False
    except AssertionError:
        refused_fz = True
    check("coherence, frozen_lr: inis olculen degerden degil verilen lr'den (0,004) dogrusal 0'a; oncesi ayni; varsayilan "
          "None; wsd'de reddedilir",
          TR.FROZEN_LR is None and all(fz[t] == TR.LR for t in range(1, 20)) and refused_fz
          and all(abs(fz[t] - 0.004 * (1 - (t - 20) / 20)) < 1e-12 for t in range(20, 41)),
          str({t: round(fz[t], 6) for t in (19, 20, 30, 40)}))

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
    try:
        TR.train_seq("shared", sids[:8], smask[:8], nv, steps=1, log_at=(), lr_floor=None, schedule="coherence")
        refused = False
    except AssertionError:
        refused = True
    check("coherence: 4. adim paketinden surdurulen = kesintisiz, bit duzeyinde (ortalama pakette); lr_floor yoksa reddedilir",
          all(torch.equal(x, y) for x, y in zip(whole.state_dict().values(), res.state_dict().values()))
          and packs[4]["optimizer"]["param_groups"][0]["coherence_mean"] != 1.0 and refused)

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

    # COHERENCE_POWER: lr carpani ortalama(rho) ^ us; 1,0 bugunku (varsayilan), 0,5 = sqrt; son inisteki dondurulan deger de
    same_1, _ = TR.train_seq("shared", None, None, nv, steps=8, log_at=(), coherence_power=1.0, **kw)
    lrs_p, means_p = {}, {}
    grab_p = lambda step, model, opt: (lrs_p.setdefault(step, opt.param_groups[0]["lr"]),
                                       means_p.setdefault(step, opt.param_groups[0]["coherence_mean"]))
    TR.train_seq("shared", None, None, nv, steps=30, log_at=(), batches=batches, save_every=1, save=grab_p,
                 schedule="coherence", coherence_window=5, coherence_power=0.5, final_cooldown=0.05, final_cooldown_shape="sqrt")
    clamp = lambda x: min(max(x, floor), 1.0)
    frozen = clamp(means_p[28] ** 0.5)                     # final = round(0,95 x 30) = 28
    sqrt_ok = (all(abs(lrs_p[t] - TR.LR * clamp(means_p[t] ** 0.5)) < 1e-12 for t in range(1, 28))
               and all(abs(lrs_p[t] - TR.LR * frozen * (floor + (1 - floor) * (1 - math.sqrt((t - 28) / 2)))) < 1e-12
                       for t in (28, 29, 30)))
    packs_p = {}
    keep_p = lambda step, model, opt: packs_p.setdefault(step, dict(step=step, model=copy.deepcopy(model.state_dict()),
                                                                     optimizer=copy.deepcopy(opt.state_dict())))
    kw_p = dict(kw, coherence_power=0.5)
    whole_p, _ = TR.train_seq("shared", None, None, nv, steps=8, log_at=(), save_every=2, save=keep_p, **kw_p)
    res_p, _ = TR.train_seq("shared", None, None, nv, steps=8, log_at=(), checkpoint=packs_p[4], **kw_p)
    try:
        TR.train_seq("shared", sids[:8], smask[:8], nv, steps=1, log_at=(), coherence_power=0.0)
        refused_p = False
    except AssertionError:
        refused_p = True
    check("coherence_power: varsayilan 1,0 ve 1,0 verilince bit duzeyinde ayni; 0,5'te lr = LR x sqrt(ortalama) (alt sinir "
          "LR_FLOOR), son inis dondurulan sqrt degerinden; 0,5 ile surdurme bit duzeyinde; 0 reddedilir",
          TR.COHERENCE_POWER == 1.0 and all(torch.equal(x, y) for x, y in zip(whole.state_dict().values(), same_1.state_dict().values()))
          and sqrt_ok and means_p[27] < 1
          and all(torch.equal(x, y) for x, y in zip(whole_p.state_dict().values(), res_p.state_dict().values())) and refused_p,
          "ortalama %.3f  lr %.5f (sqrt) / %.5f (dogrusal olsaydi)" % (means_p[27], lrs_p[27], TR.LR * clamp(means_p[27])))


def t_weight_ema():
    """WEIGHT_EMA: ortalama <- d x ortalama + (1 - d) x agirlik, kurede satir / sutunlar yeniden birim; surdurme (ortalama
    optimizer durumunda) bit duzeyinde; gecersiz d reddedilir."""
    import copy
    import torch.nn.functional as F
    data = D.build()
    sids, smask = EK.sequences(data)
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
    """HEADS: 1 = bugunku model (W_value yok); H > 1: head basina q, k birim + RoPE, V'nin dilimi, head'ler yan yana ->
    bagimsiz hesapla ayni; nedensel; W_value birim baslar, kurede satirlari birim, Muon'da; egitim, surdurme; uretim."""
    import copy
    import torch.nn.functional as F
    from model_y import BlockModel, CausalAttention, apply_rope
    data = D.build()
    sids, smask = EK.sequences(data)
    nv = len(data["vocab"])
    ids = sids[:6, :20]

    one, default, four = BlockModel(nv, heads=1), BlockModel(nv), BlockModel(nv, heads=4)
    with torch.no_grad():
        same = torch.equal(four.logits(ids), default.logits(ids))
    check("heads: varsayilan 4 (heads=4 ile bit duzeyinde ayni, W_value var); heads=1'de W_value yok",
          same and default.heads == 4 and any("W_value" in k for k in default.state_dict())
          and not any("W_value" in k for k in one.state_dict()))

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
            q = apply_rope(F.normalize(x @ at.W_query[sl].T, dim=-1))
            k = apply_rope(F.normalize(x @ at.W_key[sl].T, dim=-1))
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
                 lambda: BlockModel(nv, layer_norm=True, stream_norm=False, normalized_update=False)):
        try:
            make()
            refused.append(False)
        except AssertionError:
            refused.append(True)
    check("heads: nedensel; W_value birim baslar (d x d); reddedilir: d head'e bolunmezse, RoPE'de tek "
          "sayili head boyu, sphere_weights + LayerNorm",
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
    """FACT_ACTIVATION: varsayilan "swiglu" u = SiLU(W_fact_in x) * (W_fact_up x), esik yok; "relu" 28 Eylul oncesi;
    ayni parametre icin units 2/3; W_fact_up Muon'da ve satirlari birim; surdurme bit duzeyinde; onbellekli uretim."""
    import copy
    import torch.nn.functional as F
    from model_y import FACT_ACTIVATION, FACT_UNITS, BlockModel, FactUnits
    data = D.build()
    sids, smask = EK.sequences(data)
    nv = len(data["vocab"])
    ids = sids[:6, :20]

    base, sw = BlockModel(nv), BlockModel(nv, fact_activation="swiglu", units=170)
    with torch.no_grad():
        same = torch.equal(base.logits(ids), sw.logits(ids))
    names = {k.split(".")[-1] for k in dict(sw.named_parameters())}
    names_relu = {k.split(".")[-1] for k in dict(BlockModel(nv, fact_activation="relu").named_parameters())}
    fu = lambda m: sum(p_.numel() for k, p_ in m.named_parameters() if ".facts." in k)
    a384, b384 = BlockModel(nv, d=384, units=1536, fact_activation="relu"), BlockModel(nv, d=384, units=1024)
    check("fact_activation: varsayilan swiglu x 170 (8/3 x D), bit duzeyinde ayni; swiglu'da W_fact_up var, fact_threshold yok, relu'da "
          "tersi; d 384'te ReLU 1536 ile SwiGLU 1024 matris sayisi esit (fark yalniz esikler)",
          FACT_ACTIVATION == "swiglu" and FACT_UNITS == 170 and same and "W_fact_up" in names
          and "fact_threshold" not in names and "fact_threshold" in names_relu and "W_fact_up" not in names_relu
          and fu(a384) - fu(b384) == 2 * 1536, "%d / %d" % (fu(a384), fu(b384)))

    g = torch.Generator().manual_seed(71)
    f = FactUnits(16, 12, activation="swiglu").double()
    with torch.no_grad():
        for p_ in f.parameters():
            p_.copy_(torch.randn(p_.shape, generator=g, dtype=torch.float64))
        x = torch.randn(5, 16, generator=g, dtype=torch.float64)
        gate, up = x @ f.W_fact_in.T, x @ f.W_fact_up.T
        ref = (gate / (1 + torch.exp(-gate)) * up) @ f.W_fact_out.T
        err = float((f(x) - ref).abs().max())
    check("fact_activation swiglu: cikti = W_fact_out (SiLU(W_fact_in x) * (W_fact_up x)) (bagimsiz float64)", err < 1e-12,
          "fark %.1e" % err)
    fr = FactUnits(16, 12, activation="reglu").double()
    with torch.no_grad():
        for p_ in fr.parameters():
            p_.copy_(torch.randn(p_.shape, generator=g, dtype=torch.float64))
        gate = torch.clamp(x @ fr.W_fact_in.T - fr.fact_threshold, min=0)
        ref_r = (gate * (x @ fr.W_fact_up.T)) @ fr.W_fact_out.T
        err_r = float((fr(x) - ref_r).abs().max())
    names_r = {k for k, _ in fr.named_parameters()}
    check("fact_activation reglu: cikti = W_fact_out (ReLU(W_fact_in x - esik) * (W_fact_up x)) (bagimsiz float64); "
          "esik ve W_fact_up birlikte", err_r < 1e-12 and {"fact_threshold", "W_fact_up"} <= names_r, "fark %.1e" % err_r)
    rg, curve_r = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=10, log_at=(0, 10),
                               model_kw=dict(fact_activation="reglu", units=170))
    check("fact_activation reglu: egitimde kayip iner; W_fact_up satirlari birim",
          curve_r[-1]["nll"] < curve_r[0]["nll"]
          and all(torch.allclose(b.facts.W_fact_up.norm(dim=1), torch.ones(170), atol=1e-5) for b in rg.blocks),
          "%.3f -> %.3f" % (curve_r[0]["nll"], curve_r[-1]["nll"]))

    opts = []
    grab = lambda step, model, opt: opts.append(opt)
    kw = dict(model_kw=dict(fact_activation="swiglu", units=170))
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
    check("fact_activation swiglu: egitimde kayip iner; W_fact_up Muon'da ve satirlari birim; surdurme bit duzeyinde; "
          "onbellekli uretim tam hesapla ayni",
          curve[-1]["nll"] < curve[0]["nll"] and "W_fact_up" in in_muon and unit
          and all(torch.equal(a, b) for a, b in zip(full.state_dict().values(), res.state_dict().values()))
          and TR.generate(trained, qs, 6) == TR.generate(trained, qs, 6, cached=False),
          "%.3f -> %.3f" % (curve[0]["nll"], curve[-1]["nll"]))


def t_learn_output_scale():
    """LEARN_OUTPUT_SCALE: skor = e^tau <h, PL>, tau = log_output_scale ln(scale)'dan; baslangicta sabit olcekle bit duzeyinde
    ayni; Adam'da (Muon'da degil), weight decay yok, weight EMA ortalar; egitim, surdurme, onbellekli uretim; sinav gunlugu;
    config'inde anahtar olmayan eski kosu False ile yuklenir ve surdurulur; LayerNorm'da parametre yok."""
    import copy
    import inspect
    import json
    import tempfile
    import colab_kinship as C
    from model_y import LEARN_OUTPUT_SCALE, BlockModel
    data = D.build()
    sids, smask = EK.sequences(data)
    nv = len(data["vocab"])
    ids = sids[:6, :20]
    val = lambda t: float(t.detach())

    on, off = BlockModel(nv), BlockModel(nv, learn_output_scale=False)
    with torch.no_grad():
        same = torch.equal(on.logits(ids), off.logits(ids))
    rest = {k: v for k, v in on.state_dict().items() if k != "log_output_scale"}
    ln_model = BlockModel(nv, layer_norm=True, stream_norm=False, normalized_update=False, sphere_weights=False)
    check("learn_output_scale: varsayilan True; tau = ln(scale) baslar, skor sabit olcekli modelle bit duzeyinde ayni, fark "
          "yalniz log_output_scale; LayerNorm'da parametre yok",
          LEARN_OUTPUT_SCALE is True and inspect.signature(BlockModel).parameters["learn_output_scale"].default is True
          and on.learn_output_scale and not off.learn_output_scale and same
          and val(on.log_output_scale) == float(torch.tensor(math.log(on.scale)))
          and list(rest) == list(off.state_dict()) and all(torch.equal(rest[k], v) for k, v in off.state_dict().items())
          and not ln_model.learn_output_scale and "log_output_scale" not in ln_model.state_dict())

    # skor = e^tau <h, PL>: tau = ln 15 iken sabit olcekli modelin skoru x 15 / scale (float64)
    a, b = BlockModel(nv, d=32, units=48).double(), BlockModel(nv, d=32, units=48, learn_output_scale=False).double()
    b.load_state_dict({k: v for k, v in a.state_dict().items() if k != "log_output_scale"})
    with torch.no_grad():
        a.log_output_scale.fill_(math.log(15.0))
        err = float((a.logits(ids) - 15.0 / a.scale * b.logits(ids)).abs().max())
    check("learn_output_scale: skor = e^tau <h, PL> (tau = ln 15: sabit olcekli skor x 15 / scale, float64)", err < 1e-10,
          "fark %.1e" % err)

    opts = []
    grab = lambda step, model, opt: opts.append(opt)
    trained, curve = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=20, log_at=(0, 20), save_every=1, save=grab)
    names = {id(p_): k for k, p_ in trained.named_parameters()}
    in_adam = {names[id(p_)] for g_ in opts[-1].param_groups if not g_["use_muon"] for p_ in g_["params"]}
    in_muon = {names[id(p_)] for g_ in opts[-1].param_groups if g_["use_muon"] for p_ in g_["params"]}
    groups = []
    real_adamw = torch.optim.AdamW

    class SpyW(real_adamw):
        def __init__(self, param_groups, **k):
            super().__init__(param_groups, **k)
            groups.extend(self.param_groups)
    torch.optim.AdamW = SpyW
    try:
        mw, _ = TR.train_seq("shared", sids[:8], smask[:8], nv, steps=1, log_at=(), weight_decay=0.1, optimizer="adam")
    finally:
        torch.optim.AdamW = real_adamw
    no_decay = {k for g_ in groups if g_["weight_decay"] == 0.0 for k, p_ in mw.named_parameters()
                if any(p_ is q_ for q_ in g_["params"])}
    start, _ = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=0, log_at=())
    one, _ = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=1, log_at=(), weight_ema=0.5)
    ema_err = abs(val(one.weight_ema["model"].log_output_scale) - 0.5 * (val(start.log_output_scale) + val(one.log_output_scale)))
    moved = val(trained.log_output_scale) - val(on.log_output_scale)
    check("learn_output_scale: Adam'da (Muon'da degil), weight decay yok; weight EMA ortalar; 20 adimda deger degisir, "
          "kayip iner",
          "log_output_scale" in in_adam and "log_output_scale" not in in_muon and "log_output_scale" in no_decay
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
    check("learn_output_scale: 4. adim paketinden surdurulen = kesintisiz, bit duzeyinde (tau pakette); onbellekli uretim "
          "= tam hesap (token'lar ve skorlar)",
          "log_output_scale" in packs[4]["model"]
          and all(torch.equal(x, y) for x, y in zip(full.state_dict().values(), res.state_dict().values()))
          and new == TR.generate(trained, qs, 6, cached=False) and err_c < 1e-4, "skor farki %.1e" % err_c)

    # eski kosu: config'inde anahtar yok -> False; konus strict yukler, colab_kinship surdurur
    sys.path.insert(0, os.path.join(HERE, "train_tinystories"))
    import konus_y as K
    tmp = tempfile.mkdtemp()
    legacy = BlockModel(nv, learn_output_scale=False, output_link=False, shared_facts=True)   # 29 Eylul oncesi kosu
    for sub, model_ in (("old", off), ("new", trained), ("legacy", legacy)):
        os.makedirs(os.path.join(tmp, sub))
        kw_ = dict(d=64, turns=4, layers=2, heads=4, fact_activation="swiglu", units=170, t_max=512)
        if sub != "legacy":                                  # kosucular (29 Eylul'den) iki anahtari hep yazar
            kw_.update(output_link=model_.output_link, shared_facts=model_.shared_facts)
        if sub == "new":
            kw_["learn_output_scale"] = True
        json.dump(dict(setting="shared", model_kw=kw_, rope=True, normalized_update=True, sphere_weights=True, canon=True,
                       steps=20), open(os.path.join(tmp, sub, "config.json"), "w"))
        torch.save(model_.state_dict(), os.path.join(tmp, sub, "model.pt"))
    lo, _ = K.load_model(os.path.join(tmp, "old"), nv)
    lnew, _ = K.load_model(os.path.join(tmp, "new"), nv)
    lleg, _ = K.load_model(os.path.join(tmp, "legacy"), nv)      # anahtarsiz config: phi kapali, FactUnits paylasimli
    with torch.no_grad():
        loaded = (not lo.learn_output_scale and torch.equal(lo.logits(ids), off.logits(ids)) and lnew.learn_output_scale
                  and torch.equal(lnew.logits(ids), trained.logits(ids)) and not lleg.output_link and lleg.shared_facts
                  and torch.equal(lleg.logits(ids), legacy.logits(ids)))

    s = D.build(step_answers=True)
    kw = dict(steps=2, every=100, device="cpu", save_every=1, model_kw=dict(d=16, units=16))
    out_on = os.path.join(tmp, "on")
    run = C.start("TEST_SCALE", s, out_on, **dict(kw, steps=1))
    run["thread"].join(600)
    log_on = open(os.path.join(out_on, "log.txt"), encoding="utf-8").read()
    ex_on = json.load(open(os.path.join(out_on, "exams.json")))
    out = os.path.join(tmp, "off")
    kw_off = dict(kw, model_kw=dict(d=16, units=16, learn_output_scale=False))
    run = C.start("TEST_SCALE", s, out, **kw_off)
    run["thread"].join(600)
    first = torch.load(os.path.join(out, "model.pt"))
    cfg = json.load(open(os.path.join(out, "config.json")))
    del cfg["learn_output_scale"]                                  # 28 Eylul oncesi config gibi
    power_written = cfg.pop("coherence_power") == 1.0              # COHERENCE_POWER da: yazilir, anahtarsiz eski config 1,0
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    os.remove(os.path.join(out, "checkpoint_t00002.pt"))
    try:
        C.start("TEST_SCALE2", s, out, resume=True, **kw)          # varsayilan (True) ile: ayar farkli
        refused = False
    except RuntimeError:
        refused = True
    run2 = C.start("TEST_SCALE", s, out, resume=True, **kw_off)
    run2["thread"].join(600)
    second = torch.load(os.path.join(out, "model.pt"))
    check("learn_output_scale: sinav gunlugunde 'olcek', exams.json'da output_scale; config'inde anahtar olmayan eski kosu "
          "False ile strict yuklenir (konus) ve surdurulur (colab_kinship, bit duzeyinde), True ile reddedilir; yeni kosu "
          "True ile yuklenir; coherence_power config'te, anahtarsizi 1,0",
          loaded and " | olcek " in log_on and abs(ex_on[0]["output_scale"] - scale_for(len(s["vocab"]))) < 1e-4
          and power_written and refused and run2["done"] and not run2["error"] and all(torch.equal(first[k], second[k]) for k in first),
          str(run2["error"] or ex_on[0].get("output_scale")))


def t_matmul_precision():
    """MATMUL_PRECISION: varsayilan "bf16"; CPU'da etkisiz (fp32 / tf32 / bf16 bit duzeyinde ayni egitim, float32 matmul
    ayari degismez); gecersiz deger reddedilir; colab config'e yazar ve train_seq'e iletir; config'inde anahtar olmayan eski
    kosu "fp32" ile surdurulur.  GPU yolu (autocast bf16, TF32) yerelde sinanamaz."""
    import inspect
    import json
    import tempfile
    import colab_kinship as C
    data = D.build()
    sids, smask = EK.sequences(data)
    nv = len(data["vocab"])
    before = torch.get_float32_matmul_precision()
    runs = {p: TR.train_seq("shared", sids[:40], smask[:40], nv, steps=4, log_at=(), matmul_precision=p)[0]
            for p in ("fp32", "tf32", "bf16")}
    same = all(torch.equal(a, b) for p in ("tf32", "bf16")
               for a, b in zip(runs["fp32"].state_dict().values(), runs[p].state_dict().values()))
    try:
        TR.train_seq("shared", sids[:8], smask[:8], nv, steps=1, log_at=(), matmul_precision="fp16")
        refused = False
    except AssertionError:
        refused = True
    check("matmul_precision: varsayilan 'bf16'; CPU'da fp32 / tf32 / bf16 bit duzeyinde ayni egitim, float32 matmul ayari "
          "degismez; 'fp16' reddedilir",
          TR.MATMUL_PRECISION == "bf16" and inspect.signature(TR.train_seq).parameters["matmul_precision"].default == "bf16"
          and same and torch.get_float32_matmul_precision() == before and refused)

    s = D.build(step_answers=True)
    seen, real = [], TR.train_seq
    TR.train_seq = lambda *a, **k: (seen.append((k.get("matmul_precision"), k.get("muon_tangent"))), real(*a, **k))[1]
    tmp = tempfile.mkdtemp()
    kw = dict(steps=2, every=100, device="cpu", save_every=1, model_kw=dict(d=16, units=16))
    old = dict(kw, matmul_precision="fp32", muon_tangent=False)    # 28 Eylul oncesi tarif
    try:
        run = C.start("TEST_MP", s, tmp + "/b", **kw)
        run["thread"].join(600)
        run = C.start("TEST_MP", s, tmp + "/f", **old)
        run["thread"].join(600)
        cfg_b, cfg = (json.load(open(tmp + "/%s/config.json" % k)) for k in ("b", "f"))
        first = torch.load(tmp + "/f/model.pt")
        for k in ("matmul_precision", "muon_tangent", "coherence_power"):   # 28 Eylul oncesi config gibi
            del cfg[k]
        json.dump(cfg, open(tmp + "/f/config.json", "w"), indent=1)
        os.remove(tmp + "/f/checkpoint_t00002.pt")
        refused = []
        for bad in (kw, dict(old, matmul_precision="bf16"), dict(old, muon_tangent=True)):   # biri farkliysa ret
            try:
                C.start("TEST_MP2", s, tmp + "/f", resume=True, **bad)
                refused.append(False)
            except RuntimeError:
                refused.append(True)
        run2 = C.start("TEST_MP", s, tmp + "/f", resume=True, **old)
        run2["thread"].join(600)
        second = torch.load(tmp + "/f/model.pt")
        log = open(tmp + "/f/log.txt", encoding="utf-8").read()
    finally:
        TR.train_seq = real
    check("matmul_precision (ve muon_tangent, coherence_power), colab_kinship: config'e yazilir (varsayilan bf16 / True / 1,0) "
          "ve train_seq'e iletilir; anahtarsiz eski config fp32 / False / 1,0 ile surdurulur (bit duzeyinde), biri farkliysa "
          "reddedilir",
          cfg_b["matmul_precision"] == "bf16" and cfg_b["muon_tangent"] is True and cfg_b["coherence_power"] == 1.0
          and seen == [("bf16", True), ("fp32", False), ("fp32", False)] and all(refused) and run2["done"]
          and not run2["error"] and "SURDURULDU adim 1'den" in log and all(torch.equal(first[k], second[k]) for k in first),
          "%s %s %s" % (seen, refused, run2["error"] or ""))


def t_loss_chunk():
    """LOSS_CHUNK: egitim kaybi sozluk parcalariyla, gradyan ileri hesapta.  float64'te parca 4096 / 7 / 1 ile kayip ve butun
    gradyanlar tek parca (0) yoluyla ayni; fp32 egitim egrisi yakin; surdurme bit duzeyinde; tam logits tablosu olusmaz
    (tepe tensor ve geri yayilima saklanan tensor); colab config'e yazar, anahtarsiz eski kosu 0 ile surdurulur."""
    import copy
    import json
    import tempfile
    import colab_kinship as C
    from model_y import LOSS_CHUNK, BlockModel
    data = D.build()
    sids, smask = EK.sequences(data)
    nv = len(data["vocab"])
    ids, mask = sids[:8, :24], smask[:8, :24]

    worst, missing = 0.0, []
    g = torch.Generator().manual_seed(91)
    for kw in (dict(), dict(learn_output_scale=False), dict(stream_norm=False, normalized_update=False, sphere_weights=False),
               dict(layer_norm=True, stream_norm=False, normalized_update=False, sphere_weights=False)):
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
          "parca yoluyla <= 1e-10 ayni -- dolgulu batch; ogrenilen / sabit olcek, normsuz akis, LayerNorm",
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

    s = D.build(step_answers=True)
    tmp = tempfile.mkdtemp()
    kw = dict(steps=2, every=100, device="cpu", save_every=1, model_kw=dict(d=16, units=16))
    run = C.start("TEST_LC", s, tmp + "/n", **dict(kw, steps=1))
    run["thread"].join(600)
    cfg_new = json.load(open(tmp + "/n/config.json"))
    kw_old = dict(kw, model_kw=dict(d=16, units=16, loss_chunk=0, output_link=False))
    run = C.start("TEST_LC", s, tmp + "/o", **kw_old)
    run["thread"].join(600)
    first = torch.load(tmp + "/o/model.pt")
    cfg = json.load(open(tmp + "/o/config.json"))
    del cfg["loss_chunk"]                                          # 28 Eylul oncesi config gibi
    json.dump(cfg, open(tmp + "/o/config.json", "w"), indent=1)
    os.remove(tmp + "/o/checkpoint_t00002.pt")
    try:
        C.start("TEST_LC2", s, tmp + "/o", resume=True, **kw)      # varsayilan 4096: ayar farkli
        refused = False
    except RuntimeError:
        refused = True
    run2 = C.start("TEST_LC", s, tmp + "/o", resume=True, **kw_old)
    run2["thread"].join(600)
    second = torch.load(tmp + "/o/model.pt")
    check("loss_chunk, colab_kinship: config'e yazilir (varsayilan 4096); anahtarsiz eski config 0 ile surdurulur (bit "
          "duzeyinde), 4096 ile reddedilir; transformer'da yok",
          cfg_new["loss_chunk"] == 4096 and refused and run2["done"] and not run2["error"]
          and all(torch.equal(first[k], second[k]) for k in first) and not hasattr(TR.TransformerModel(nv), "loss_chunk"),
          str(run2["error"] or ""))


def t_muon_tangent():
    """MUON_TANGENT: Muon'a giren Nesterov birlesimi once kure tegetine (satirlari birim: W_value, W_fact_in, W_fact_up;
    sutunlari birim: W_context, W_fact_out).  Bir adim bagimsiz elle hesapla ayni (False: eski Muon, bit duzeyinde); izdusulen
    girdi agirlikla dik; egitim, surdurme."""
    import copy
    import torch.nn.functional as F
    data = D.build()
    sids, smask = EK.sequences(data)
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
    worst, dots = {}, []
    try:
        for tangent in (False, True):
            seen.clear()
            m, _ = TR.train_seq("shared", ids, mask, nv, steps=1, log_at=(), muon_tangent=tangent)
            worst[tangent] = 0.0
            flat = [x for G in seen for x in (G.unbind(0) if G.dim() == 3 else [G])]   # Muon ayni boyu yigin halinde verir
            for (k, p0), p1 in zip(ref.named_parameters(), m.parameters()):
                kind = k.split(".")[-1]
                if kind not in rows + cols:
                    continue
                axis = 1 if kind in rows else 0
                v = p0.grad.add(p0.grad, alpha=0.95)            # ilk adim: tampon = g, Nesterov g + 0,95 g
                if tangent:
                    v = v - (v * p0.detach()).sum(axis, keepdim=True) * p0.detach()
                    G = next((x for x in flat if x.shape == v.shape and torch.allclose(x, v, rtol=1e-5, atol=1e-7)), None)
                    dots.append(float("inf") if G is None else
                                float((G * p0.detach()).sum(axis).abs().max() / G.norm(dim=axis).max()))
                new = p0.detach().clone().add_(real(v), alpha=-TR.LR * 0.2 * max(p0.shape) ** 0.5)
                worst[tangent] = max(worst[tangent], float((F.normalize(new, dim=axis) - p1.detach()).abs().max()))
    finally:
        TR.Muon.orthogonalize = saved
    check("muon_tangent: varsayilan True; bir adim bagimsiz elle hesapla ayni -- False eski Muon (bit duzeyinde), True "
          "once teget; izdusulen girdi agirligin satir / sutunlariyla dik (goreli < 1e-5)",
          TR.MUON_TANGENT is True and worst[False] == 0.0 and worst[True] < 1e-6 and dots and max(dots) < 1e-5,
          "False %.1e  True %.1e  en buyuk <G, w> %.1e" % (worst[False], worst[True], max(dots) if dots else -1))

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
    check("muon_tangent: tablo tek yerde (coherence ile ortak): satir W_query, W_key, W_value, W_fact_in, W_fact_up; sutun "
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
    data = D.build()
    sids, smask = EK.sequences(data)
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
    data = D.build()
    sids, smask = EK.sequences(data)
    nv = len(data["vocab"])
    kw = dict(d=16, units=16, layers=2, turns=4, heads=2)
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
    sep = BlockModel(nv, shared=False, **kw).double()      # her tura ayri Block: attention'i split'ten ikiser kopya
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


def _bigram_keys(sids, smask, nv, k):
    """Test verisinin en sik k (onceki, token) ikilisi: anahtar onceki * nv + token, artan sirali (bigrams_<K>.npy gibi)."""
    import numpy as np
    counts = {}
    for i in range(sids.shape[0]):
        x = sids[i, :int(smask[i].sum())].tolist()
        for a, b in zip(x[:-1], x[1:]):
            counts[a * nv + b] = counts.get(a * nv + b, 0) + 1
    top = sorted(counts, key=lambda key: (-counts[key], key))[:k]
    return torch.tensor(np.sort(np.array(top, dtype=np.int64)))


def t_input_embedding():
    """INPUT_EMBEDDING / INPUT_BIGRAMS / FIRST_TURN_FACTS (bulgular tablosu #7c): tablolar baslangicta (girdi PF'den, ikili 0)
    PL'li modelle bit duzeyinde ayni; input_states elle hesapla (ikili bulunan / bulunmayan / konum 0, last ile devam); tur 1
    FactUnits'siz model = internals'in F1 atlamasi (ayni tohum); egitim, Adam, surdurme; onbellekli uretim (sagdan dolgulu
    istemler; istem sonra tek token) tam hesapla ayni; compile'da graph break yok; ayarsiz config eski model."""
    import copy
    import torch.nn.functional as F
    import internals_y as I
    from model_y import AttentionCache, BlockModel
    data = D.build()
    sids, smask = EK.sequences(data)
    nv = len(data["vocab"])
    ids = sids[:6, :20]
    K = 40
    keys = _bigram_keys(sids, smask, nv, K)
    full_kw = dict(input_embedding=True, input_bigrams=K, first_turn_facts=False, bigram_keys=keys)

    base, emb = BlockModel(nv), BlockModel(nv, input_embedding=True)
    big = BlockModel(nv, input_embedding=True, input_bigrams=K, bigram_keys=keys)
    with torch.no_grad():
        z0 = base.logits(ids)
        free0 = BlockModel(nv, input_embedding=True, input_embedding_sphere=False)
        same = torch.equal(z0, free0.logits(ids)) and float((z0 - emb.logits(ids)).abs().max()) < 1e-5             and float((z0 - big.logits(ids)).abs().max()) < 1e-5
    check("input_embedding: tablolar baslangicta (girdi PF'den, ikili 0) PL'li modelle ayni skor (kuresizde bit duzeyinde, "
          "kurede yuvarlama farki); "
          "bigram_keys tampon, parametre degil",
          same and "bigram_keys" in dict(big.named_buffers()) and "bigram_keys" not in dict(big.named_parameters())
          and tuple(big.input_bigrams.shape) == (K, 64) and not hasattr(base, "input_embedding"))

    g = torch.Generator().manual_seed(71)
    with torch.no_grad():
        big.input_embedding.copy_(torch.randn(big.input_embedding.shape, generator=g))
        big.input_bigrams.copy_(torch.randn(big.input_bigrams.shape, generator=g))
        got = big.input_states(ids)
        cont = big.input_states(ids[:, 5:], last=ids[:, 4])
        index = {int(k_): r for r, k_ in enumerate(keys.tolist())}
        want, hits = torch.zeros_like(got), 0
        for b in range(ids.shape[0]):
            for t in range(ids.shape[1]):
                v = big.input_embedding[ids[b, t]].clone()
                r = index.get(int(ids[b, t - 1]) * nv + int(ids[b, t])) if t > 0 else None
                if r is not None:
                    v, hits = v + big.input_bigrams[r], hits + 1
                want[b, t] = F.normalize(v, dim=-1)
    err = float((got - want).abs().max())
    err_c = float((cont - got[:, 5:]).abs().max())
    check("input_embedding: input_states = norm(E[token] + ikili satiri) elle (ikili listede yoksa ve konum 0'da yalniz token); "
          "last ile devam = tam hesabin devami", err < 1e-6 and err_c < 1e-6 and 0 < hits < ids.numel() - ids.shape[0],
          "fark %.1e  devam %.1e  ikili %d / %d" % (err, err_c, hits, ids.numel()))

    torch.manual_seed(0)
    plain, skip = BlockModel(nv), BlockModel(nv, first_turn_facts=False)
    kept = BlockModel(nv, first_turn_facts=False, shared_facts=True)
    with torch.no_grad():
        err_s = float((skip.logits(ids) - I._logits(plain, ids, dict(skip_facts=[0]))).abs().max())
        err_k = float((kept.logits(ids) - I._logits(BlockModel(nv, shared_facts=True), ids, dict(skip_facts=[0]))).abs().max())
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
    res, _ = TR.train_seq("shared", sids[:40], smask[:40], nv, steps=6, log_at=(), checkpoint=packs[4],
                          model_kw=dict(full_kw, bigram_keys="iz"))
    free, _ = TR.train_seq("shared", sids[:64], smask[:64], nv, steps=20, log_at=(),
                           model_kw=dict(full_kw, input_embedding_sphere=False))
    rows_s, rows_f = trained.input_embedding.detach().norm(dim=1), free.input_embedding.detach().norm(dim=1)
    check("input_embedding_sphere: girdi tablosunun satirlari egitimde birim boy (varsayilan); False'ta serbest, boy buyur "
          "(gradyan satira dik)",
          float((rows_s - 1).abs().max()) < 1e-5 and trained.input_embedding_sphere and not free.input_embedding_sphere
          and float(rows_f.max()) > 1.01 and float((BlockModel(nv).tokens.fixed_points.norm(dim=1) - 1).abs().max()) < 1e-5,
          "birim %.1e  serbest en buyuk %.3f" % (float((rows_s - 1).abs().max()), float(rows_f.max())))

    check("input_embedding: egitimde kayip iner; input_embedding ve input_bigrams Adam'da, 0'dan ayrilir; bigram_keys degismez; "
          "tur 1'in takimi state_dict'te yok; surdurme (liste pakette, config'te iz) bit duzeyinde",
          curve[-1]["nll"] < curve[0]["nll"] and {"input_embedding", "input_bigrams"} <= set(in_adam)
          and bool(trained.input_bigrams.abs().max() > 0) and torch.equal(trained.bigram_keys, keys)
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
    check("input_embedding: onbellekli uretim = tam hesap (sagdan dolgulu istemler; ikili icin onceki token last_token'dan); "
          "istem - 1 sonra tek token yolu ayni",
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
    """internals_y (kalici analiz araci): elle ileri hesap = model.logits (float64; 9 model: varsayilan, shared_facts=False, girdi tablosu + ikili + tur 1 FactUnits'siz,
    output_link=False, tek head + relu, ayri blok + reglu, normalized_update kapali, akis normsuz, LayerNorm; SDPA ve acik
    softmax yolu, sagdan dolgulu batch); trace'in durumlari, acilari ve lens'i modelin kendi hidden'i ve Block'uyla;
    point_drift, attention_stats, unit_usage bagimsiz hesapla; ablate: hicbir sey = taban (Δ 0), alpha'yi ayni degerle
    vermek = taban, kapatmalar agirligi degistirilmis modelin logits'iyle ayni (Δnll, Δacc, se), head ortalamasi elle;
    over_checkpoints'in yedekten kurdugu EMA = train_seq'in model.weight_ema'si (muon, adam, adamw); tuned_lens: birim
    cevirici = logit lens, ogrenilen cevirici ayri hikayelerde iyi, son durumda KL 0, model degismez; profile_step: parca
    parca adim = train_seq'in adimi (bit duzeyinde), parcalar adimin icinde, kendi kopyasinda."""
    import copy
    import json
    import shutil
    import tempfile
    import numpy as np
    import torch.nn.functional as F
    import internals_y as I
    from model_y import BlockModel
    data = D.build()
    sids, smask = EK.sequences(data)
    nv = len(data["vocab"])
    stories = [sids[i, :int(smask[i].sum())].tolist() for i in range(6)]
    others = [sids[i, :int(smask[i].sum())].tolist() for i in range(6, 12)]
    base_kw = dict(d=16, units=24, layers=2, turns=4, heads=2, shared_facts=True)   # paylasimli tasarim; ayrik hali acikca

    def perturbed(seed=5, **kw):
        m = BlockModel(nv, **dict(base_kw, **kw)).double()
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():                              # sifirdan farkli alpha, Canon, bag, esik: kontrol bos gecmesin
            for p_ in m.parameters():
                p_.add_(0.1 * torch.randn(p_.shape, generator=g, dtype=p_.dtype))
        return m.eval()

    variants = [("varsayilan", {}), ("shared_facts=False", dict(shared_facts=False)),
                ("output_link=False", dict(output_link=False)), ("tek head + relu", dict(heads=1, fact_activation="relu")),
                ("girdi tablosu + ikili + tur 1 FactUnits'siz", dict(input_embedding=True, input_bigrams=20, first_turn_facts=False,
                                                                     bigram_keys=_bigram_keys(sids, smask, nv, 20))),
                ("ayri blok + reglu", dict(shared=False, fact_activation="reglu")),
                ("normalized_update kapali", dict(normalized_update=False, sphere_weights=False)),
                ("akis normsuz", dict(stream_norm=False, normalized_update=False, sphere_weights=False)),
                ("LayerNorm", dict(layer_norm=True, normalized_update=False, sphere_weights=False))]
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
    check("internals: elle ileri hesap = model.logits (float64; 9 model ayari, dolgulu batch, SDPA ve acik softmax)",
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
        for label, extra in (("muon", {}), ("adam", dict(optimizer="adam")), ("adamw", dict(optimizer="adam", weight_decay=0.1))):
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
                          rope=True, normalized_update=True, sphere_weights=True, canon=True,
                          optimizer=extra.get("optimizer", "muon"), weight_decay=extra.get("weight_decay", 0.0))
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
          "muon, adam, adamw); 'last' = o adimin agirligi; model_weight_ema.pt", ok, ", ".join(notes))

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

    class Enough(Exception):
        pass

    cached = [(sids[i:i + 8].clone(), smask[i:i + 8].clone()) for i in (0, 8, 16)]
    kept = [(a_.clone(), b_.clone()) for a_, b_ in cached]
    replica = []
    for sf in (True, False):                               # paylasimli ve tur basina FactUnits
        mk = dict(base_kw, output_link=True, shared_facts=sf)
        ctx = I._profile_setup(dict(setting="shared", vocab=nv, model_kw=mk, weight_ema=0.9), cached, "cpu")
        for j in (1, 2, 3):                                # kurulum batch 0 ile bir adim; tekrar 1, 2, 0
            I._step_parts(ctx, *cached[j % 3], I._Clock("cpu"))
        ref = {}

        def stop(step, model, opt):
            ref.update(model=copy.deepcopy(model.state_dict()), ema=copy.deepcopy(model.weight_ema["model"].state_dict()))
            raise Enough()
        try:                                               # wsd: 80 adimdan once lr sabit
            TR.train_seq("shared", None, None, nv, steps=100, batches=lambda s: cached[s % 3], log_at=(), save_every=4,
                         save=stop, weight_ema=0.9, model_kw=mk)
        except Enough:
            pass
        replica.append(same(ctx["model"].state_dict(), ref["model"]) and same(ctx["ema"]["model"].state_dict(), ref["ema"]))
    check("internals profile_step: parca parca tekrarlanan adim = train_seq'in adimi, bit duzeyinde (wsd, Muon + Adam, clip, "
          "kure, phi kirpma, EMA; 1 + 3 adim; paylasimli / tur basina FactUnits)", all(replica), str(replica))

    precision, rng = torch.get_float32_matmul_precision(), torch.get_rng_state()
    cfg = dict(setting="shared", vocab=nv, model_kw=dict(base_kw, shared_facts=False, output_link=False),
               schedule="coherence", weight_ema=0.9)
    pr = I.profile_step(cfg, cached, steps=3)
    names = [p_["part"] for p_ in pr["parts"]]
    want = ["batch -> cihaz", "ileri + kayip", "zero_grad", "geri yayilim", "coherence: g1 kopyasi",
            "coherence: birlestirme, teget, carpimlar", "clip", "Muon (Newton-Schulz dahil)", "  Newton-Schulz", "Adam",
            "normalize_weights", "EMA"]
    gap = max((s_["total_ms"] - s_["parts_ms"]) / s_["total_ms"] for s_ in pr["per_step"])
    low = min(s_["total_ms"] - s_["parts_ms"] for s_ in pr["per_step"])
    turns = [b_["part"] for b_ in pr["breakdown"]]
    check("internals profile_step (CPU): parcalar train_seq'in sirasiyla (coherence yarilari, Muon / Newton-Schulz / Adam, "
          "EMA); her adimda parcalarin toplami <= adim, fark < %10; dokum tur basina 3 parca; train_seq'in kendi adimi olculdu",
          names == want and 0 <= low and gap < 0.1 and len(turns) == 4 * 3 + 4 and turns[1] == "tur 1 Canon"
          and pr["real_step_ms"] > 0 and pr["flops"]["per_step"] > 0,
          "adim %.1f ms, olculmeyen en cok %%%.1f; train_seq %.1f ms" % (pr["total"]["device_ms"], 100 * gap, pr["real_step_ms"]))
    check("internals profile_step: kendi kopyasinda calisir (train_seq'in kurulumu); batch'ler, grad modu, matmul hassasiyeti "
          "ve RNG degismez", all(torch.equal(a_, c_) and torch.equal(b_, d_) for (a_, b_), (c_, d_) in zip(cached, kept))
          and torch.is_grad_enabled() and torch.get_float32_matmul_precision() == precision
          and torch.equal(torch.get_rng_state(), rng))


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
    names = [f.__name__ for f in (t_model, t_step2, t_step3, t_transformer, t_generate_cached, t_normalized_update, t_canon,
              t_layers, t_coherence, t_weight_ema, t_heads, t_fact_activation, t_learn_output_scale, t_matmul_precision,
              t_loss_chunk, t_muon_tangent, t_output_link, t_shared_facts, t_input_embedding, t_internals)]
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
