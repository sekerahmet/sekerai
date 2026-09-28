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
    m, curve = TR.train_seq("step2", sids, smask, nv, steps=20, log_at=(0, 20))
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
    return m.scale * (h if m.stream_norm else unit(h)) @ PL.T


def t_step3():
    import functools
    import model_y
    TURNS = 2
    # 28 Eylul oncesi tasarimin testleri: paket (normalized_update, sphere_weights) kapali, tek Block x 2 tur; paket
    # t_normalized_update'te, 2 x 2 t_layers'ta
    BlockModel = functools.partial(model_y.BlockModel, normalized_update=False, sphere_weights=False, canon=False, turns=2,
                                   layers=1, heads=1, fact_activation="relu")
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
    mr, curve_r = TR.train_seq("shared", sids, smask, nv, steps=20, log_at=(0, 20), rope=True)
    md, _ = TR.train_seq("shared", sids, smask, nv, steps=1, log_at=())
    mo, _ = TR.train_seq("shared", sids, smask, nv, steps=1, log_at=(), rope=False)
    check("adim 3, rope=True: train_seq ayari modele ulasir, kayip iner; train_seq varsayilani (None) BlockModel'de ROPE (True); "
          "rope=False kapatir",
          all(b.attention.rope for b in mr.blocks) and curve_r[-1]["nll"] < curve_r[0]["nll"]
          and all(b.attention.rope for b in md.blocks) and all(not b.attention.rope for b in mo.blocks),
          "%.3f -> %.3f" % (curve_r[0]["nll"], curve_r[-1]["nll"]))
    # batches: mini-batch; model_kw: model ayarlari
    import copy
    whole, _ = TR.train_seq("shared", sids, smask, nv, steps=5, log_at=())
    via, _ = TR.train_seq("shared", None, None, nv, steps=5, log_at=(), batches=lambda step: (sids, smask))
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

    ml, curve_l = TR.train_seq("shared", sids, smask, nv, steps=20, log_at=(0, 20), stream_norm=False, normalized_update=False, sphere_weights=False, layer_norm=True)
    check("adim 3, layer_norm=True: train_seq ile egitilir, kayip iner", ml.layer_norm and curve_l[-1]["nll"] < curve_l[0]["nll"],
          "%.3f -> %.3f" % (curve_l[0]["nll"], curve_l[-1]["nll"]))
    mf, curve_f = TR.train_seq("shared", sids, smask, nv, steps=20, log_at=(0, 20), stream_norm=False, normalized_update=False, sphere_weights=False)
    check("adim 3, stream_norm=False: train_seq ile egitilir, kayip iner", not mf.stream_norm and curve_f[-1]["nll"] < curve_f[0]["nll"],
          "%.3f -> %.3f" % (curve_f[0]["nll"], curve_f[-1]["nll"]))
    before = BlockModel(nv).tokens.fixed_points.clone()
    m, curve = TR.train_seq("shared", sids, smask, nv, steps=20, log_at=(0, 20))
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
        TR.train_seq("shared", sids, smask, nv, steps=4, log_at=(), lr=0.01, lr_floor=0.1, grad_clip=0.5, optimizer="adam", schedule="cosine")
        cosine = [0.01 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / 4))) for t in range(4)]
        ok_sched = all(abs(lr_ - c) < 1e-12 for (lr_, _), c in zip(seen, cosine))
        ok_clip = all(g <= 0.5 + 1e-5 for _, g in seen)
        seen.clear()
        TR.train_seq("shared", sids, smask, nv, steps=4, log_at=(), lr=0.01, optimizer="adam", schedule="cosine")
        standard = [0.01 * (TR.LR_FLOOR + (1 - TR.LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * t / 4))) for t in range(4)]
        ok_default = (all(abs(lr_ - c) < 1e-12 for (lr_, _), c in zip(seen, standard))
                      and all(g <= TR.GRAD_CLIP + 1e-5 for _, g in seen))
        seen.clear()
        TR.train_seq("shared", sids, smask, nv, steps=3, log_at=(), lr=0.01, lr_floor=None, grad_clip=None, optimizer="adam", schedule="cosine")
        ok_old = all(lr_ == 0.01 for lr_, _ in seen) and any(g > 0.5 for _, g in seen)
    finally:
        torch.optim.Adam = real_adam
    check("egitim tarifi (optimizer='adam', schedule='cosine', 27 Eylul oncesi): cosine decay ile lr LR'den LR x lr_floor'a iner, gradient boyu grad_clip'i gecmez; standart LR_FLOOR ve "
          "GRAD_CLIP; None verilirse sabit lr", ok_sched and ok_clip and ok_default and ok_old)
    a1, _ = TR.train_seq("shared", sids, smask, nv, steps=3, log_at=(), seed=0)
    a2, _ = TR.train_seq("shared", sids, smask, nv, steps=3, log_at=(), seed=1)
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
        mw, _ = TR.train_seq("shared", sids, smask, nv, steps=1, log_at=(), weight_decay=0.1, optimizer="adam")
    finally:
        torch.optim.AdamW = real_adamw
    names = {id(p_): k.split(".")[-1] for k, p_ in mw.named_parameters()}
    decayed = sorted({names[id(p_)] for g_ in groups if g_["weight_decay"] == 0.1 for p_ in g_["params"]})
    kept = sorted({names[id(p_)] for g_ in groups if g_["weight_decay"] == 0.0 for p_ in g_["params"]})
    check("weight decay: yalniz W_ matrislerine; shift, alpha ve Canon agirliklari haric",
          decayed == sorted(["W_query", "W_key", "W_context", "W_fact_in", "W_fact_up", "W_fact_out", "W_value"])
          and kept == ["alpha_attention", "alpha_facts", "canon_weights", "shift"], "%s | %s" % (decayed, kept))

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

    opts, lrs = [], {}

    def grab(step, model, opt):
        opts.append(opt)
        lrs[step] = opt.param_groups[0]["lr"]
    mm, curve_m = TR.train_seq("shared", sids, smask, nv, steps=10, log_at=(0, 10), save_every=1, save=grab)
    names_m = {id(p_): k.split(".")[-1] for k, p_ in mm.named_parameters()}
    in_muon = sorted({names_m[id(p_)] for g_ in opts[0].param_groups if g_["use_muon"] for p_ in g_["params"]})
    in_adam = sorted({names_m[id(p_)] for g_ in opts[0].param_groups if not g_["use_muon"] for p_ in g_["params"]})
    wsd = [0.01 * (1.0 if t < 8 else TR.LR_FLOOR + (1 - TR.LR_FLOOR) * (1 - math.sqrt((t - 8) / 2))) for t in range(1, 11)]
    check("Muon + WSD (varsayilan): Muon'da W_context, W_fact_in, W_fact_up, W_fact_out, W_value; Adam'da noktalar, W_query, "
          "W_key; "
          "lr 8. adima kadar sabit, sonra 1 - sqrt ile LR x LR_FLOOR'a; kayip iner",
          isinstance(opts[0], TR.Muon) and in_muon == ["W_context", "W_fact_in", "W_fact_out", "W_fact_up", "W_value"]
          and in_adam == ["W_key", "W_query", "alpha_attention", "alpha_facts", "canon_weights", "shift"]
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
    m, curve = TR.train_seq("transformer", sids, smask, nv, steps=20, log_at=(0, 20))
    try:
        TR.train_seq("transformer", sids, smask, nv, steps=1, log_at=(), weight_decay=0.1)
        refused = False
    except AssertionError:
        refused = True
    mv, curve_v = TR.train_seq("transformer_novalue", sids, smask, nv, steps=20, log_at=(0, 20))
    check("transformer: train_seq ayni tarifle egitir, kayip iner ('transformer' ve 'transformer_novalue'); weight decay "
          "istenirse reddedilir (gruplar tanimsiz)",
          isinstance(m, TransformerModel) and curve[-1]["nll"] < curve[0]["nll"] and refused
          and all(layer.W_value is None for layer in mv.layers) and curve_v[-1]["nll"] < curve_v[0]["nll"],
          "%.3f -> %.3f; V'siz %.3f -> %.3f" % (curve[0]["nll"], curve[-1]["nll"], curve_v[0]["nll"], curve_v[-1]["nll"]))

    mr, curve_r = TR.train_seq("transformer_novalue", sids, smask, nv, steps=20, log_at=(0, 20), rope=False)
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
    from model_y import ALPHA_INIT, BlockModel, apply_rope
    data = D.build()
    sids, smask = EK.sequences(data)
    nv = len(data["vocab"])

    today, off = BlockModel(nv, normalized_update=False, sphere_weights=False), BlockModel(nv, normalized_update=False,
                                                                                             sphere_weights=False)
    both = BlockModel(nv, normalized_update=True, sphere_weights=True)
    b0 = both.blocks[0]
    unit_rows = lambda w: float((w.norm(dim=1) - 1).abs().max())
    unit_cols = lambda w: float((w.norm(dim=0) - 1).abs().max())
    check("anahtarlar kapali = bugunku model (ayni agirliklar, alpha yok, W_context ve W_fact_out 0); acikken alpha (tur, d) = "
          "ALPHA_INIT, W_context / W_fact_out rastgele, satir / sutunlar birim",
          all(torch.equal(a, b) for a, b in zip(today.state_dict().values(), off.state_dict().values()))
          and not hasattr(off, "alpha_attention") and not off.blocks[0].attention.W_context.any()
          and both.alpha_attention.shape == (both.turns, 64) and bool((both.alpha_facts == ALPHA_INIT).all())
          and max(unit_rows(b0.attention.W_query), unit_rows(b0.attention.W_key), unit_rows(b0.facts.W_fact_in),
                  unit_cols(b0.attention.W_context), unit_cols(b0.facts.W_fact_out)) < 1e-6)

    # bagimsiz float64 referans (tasarim formulu): rastgele parametrelerle
    g = torch.Generator().manual_seed(41)
    m = BlockModel(nv, d=32, units=48, normalized_update=True, sphere_weights=True, canon=False, heads=1).double()
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
        ref = m.scale * h @ P.T
        err = float((m.logits(ids) - ref).abs().max())
    check("normalized_update + sphere_weights: skor = tasarim formulu (bagimsiz float64; h <- norm(h + a (norm(u) - h)), "
          "FactUnits girdisi sqrt(d) x kosinus)", err < 1e-10, "fark %.1e" % err)

    # kusur 1: FactUnits ciktisi 1000 kat buyutulse de durum alpha kadar doner (bugunku modelde silinir)
    def turn_cos(model):
        with torch.no_grad():
            hs = model.hidden(sids[:16])
            return min(float((a * b).sum(-1)[smask[:16]].min()) for a, b in zip(hs[:-1], hs[1:]))
    fixed = BlockModel(nv, normalized_update=True)
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
    trained, curve = TR.train_seq("shared", sids, smask, nv, steps=20, log_at=(0, 20), save_every=1, save=grab, **kw)
    tb = trained.blocks[0]
    names = {id(p_): k for k, p_ in trained.named_parameters()}
    in_adam = sorted({names[id(p_)] for g_ in opts[-1].param_groups if not g_["use_muon"] for p_ in g_["params"]})
    check("egitim (normalized_update + sphere_weights, Muon + WSD): kayip iner; 20 adimdan sonra satir / sutunlar birim; "
          "alpha Adam'da ve baslangictan ayrildi",
          curve[-1]["nll"] < curve[0]["nll"] and "alpha_attention" in in_adam and "alpha_facts" in in_adam
          and max(unit_rows(tb.attention.W_query), unit_rows(tb.attention.W_key), unit_rows(tb.facts.W_fact_in),
                  unit_cols(tb.attention.W_context), unit_cols(tb.facts.W_fact_out)) < 1e-5
          and float((trained.alpha_facts - ALPHA_INIT).abs().max()) > 1e-4,
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
    trained, curve = TR.train_seq("shared", sids, smask, nv, steps=20, log_at=(0, 20), save_every=1, save=grab, canon=True)
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

    m = BlockModel(nv, layers=2, turns=4)
    tb = m.turn_blocks()
    four = BlockModel(nv, turns=4, layers=1)
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
    trained, curve = TR.train_seq("shared", sids, smask, nv, steps=20, log_at=(0, 20), save_every=1, save=grab, model_kw=kw)
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

    sep, curve_s = TR.train_seq("separate", sids, smask, nv, steps=20, log_at=(0, 20))
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
    full, _ = TR.train_seq("shared", sids, smask, nv, steps=40, log_at=(), save_every=1, save=grab, schedule="coherence")
    floor = TR.LR_FLOOR
    want = {37: TR.LR, 38: TR.LR, 39: TR.LR * (floor + (1 - floor) * (1 - math.sqrt(0.5))), 40: TR.LR * floor}
    check("coherence, tam batch: rho olculmez (= 1), lr sabit; son %5'te 1 - sqrt ile x LR_FLOOR'a",
          all(abs(lrs[t] - v) < 1e-12 for t, v in want.items()) and all(lrs[t] == TR.LR for t in range(1, 38))
          and not getattr(full, "coherence", None), str({t: round(lrs[t], 6) for t in (1, 37, 38, 39, 40)}))

    lrs_m, means = {}, {}
    grab_m = lambda step, model, opt: (lrs_m.setdefault(step, opt.param_groups[0]["lr"]),
                                       means.setdefault(step, opt.param_groups[0]["coherence_mean"]))
    mm, curve = TR.train_seq("shared", None, None, nv, steps=30, log_at=(0, 30), batches=batches, save_every=1, save=grab_m,
                             schedule="coherence", coherence_window=5)
    follows = all(abs(lrs_m[t] - TR.LR * min(max(means[t], floor), 1.0)) < 1e-12 for t in range(1, 28))
    check("coherence, mini-batch: ortalama 1'den ayrilir, lr = LR x ortalama (alt sinir LR_FLOOR); kayip iner",
          follows and 0 < means[27] < 1 and curve[-1]["nll"] < curve[0]["nll"],
          "ortalama %.3f  lr %.5f  %.3f -> %.3f" % (means[27], lrs_m[27], curve[0]["nll"], curve[-1]["nll"]))

    packs = {}
    keep = lambda step, model, opt: packs.setdefault(step, dict(step=step, model=copy.deepcopy(model.state_dict()),
                                                                   optimizer=copy.deepcopy(opt.state_dict())))
    kw = dict(batches=batches, schedule="coherence", coherence_window=3)
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


def t_weight_ema():
    """WEIGHT_EMA: ortalama <- d x ortalama + (1 - d) x agirlik, kurede satir / sutunlar yeniden birim; surdurme (ortalama
    optimizer durumunda) bit duzeyinde; gecersiz d reddedilir."""
    import copy
    import torch.nn.functional as F
    data = D.build()
    sids, smask = EK.sequences(data)
    nv = len(data["vocab"])

    start, _ = TR.train_seq("shared", sids, smask, nv, steps=0, log_at=())
    one, _ = TR.train_seq("shared", sids, smask, nv, steps=1, log_at=(), weight_ema=0.5)
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
    kw = dict(batches=batches, schedule="coherence", weight_ema=0.9)
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
    trained, curve = TR.train_seq("shared", sids, smask, nv, steps=20, log_at=(0, 20), save_every=1, save=grab, **kw)
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
    rg, curve_r = TR.train_seq("shared", sids, smask, nv, steps=10, log_at=(0, 10),
                               model_kw=dict(fact_activation="reglu", units=170))
    check("fact_activation reglu: egitimde kayip iner; W_fact_up satirlari birim",
          curve_r[-1]["nll"] < curve_r[0]["nll"]
          and all(torch.allclose(b.facts.W_fact_up.norm(dim=1), torch.ones(170), atol=1e-5) for b in rg.blocks),
          "%.3f -> %.3f" % (curve_r[0]["nll"], curve_r[-1]["nll"]))

    opts = []
    grab = lambda step, model, opt: opts.append(opt)
    kw = dict(model_kw=dict(fact_activation="swiglu", units=170))
    trained, curve = TR.train_seq("shared", sids, smask, nv, steps=20, log_at=(0, 20), save_every=1, save=grab, **kw)
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


if __name__ == "__main__":
    print("tests (model_y)")
    for f in (t_model, t_step2, t_step3, t_transformer, t_generate_cached, t_normalized_update, t_canon,
              t_layers, t_coherence, t_weight_ema, t_heads, t_fact_activation):
        f()
    print("\n%d GECTI   %d KALDI" % (sum(RESULTS), len(RESULTS) - sum(RESULTS)))
    sys.exit(0 if all(RESULTS) else 1)
