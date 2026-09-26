# -*- coding: utf-8 -*-
"""tests_20 -- model_20'nin kapilari.  CPU, saniyeler.

    python tests_20.py
"""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
torch.set_num_threads(1)

import data_20 as D  # noqa: E402
import train_20 as TR  # noqa: E402
from model_20 import BigramModel, deviation, scale_for  # noqa: E402

RESULTS = []
FINGERPRINT = "90024739fb8f"      # veri degisirse burasi bilerek guncellenir


def check(name, ok, note=""):
    RESULTS.append(bool(ok))
    print("  %-78s %s  %s" % (name, "GECTI" if ok else "KALDI", note))


def graph_from_text(train):
    """Egitim METNINDEKI tek iliskili cumlelerden ('X's r is Y.') graf -- data_20'nin ic yapisindan bagimsiz."""
    g = {}
    for s in train:
        text = D.detokenize(s)
        if text.startswith("Who ") or text.count("'s") != 1:
            continue
        left, y = text[:-1].split(" is ")
        x, r = left.split("'s ")
        g.setdefault((x, r), set()).add(y)
    return g


def walk(g, x, path):
    now = {x}
    for r in path:
        now = {y for z in now for y in g.get((z, r), ())}
    return now


# tanimlar burada ayrica yazili (data_20.DERIVED'e bakilmadan)
DEFINITIONS = {
    "grandfather": [("father", "father"), ("mother", "father")],
    "grandmother": [("father", "mother"), ("mother", "mother")],
    "aunt": [("father", "sister"), ("mother", "sister")],
    "uncle": [("father", "brother"), ("mother", "brother")],
    "cousin": [(p, s, c) for p in ("father", "mother") for s in ("sister", "brother") for c in ("son", "daughter")],
}


def t_data():
    d = D.build()
    check("veri: iz sabit (degisirse bilerek guncellenir)", d["fingerprint"] == FINGERPRINT, d["fingerprint"])
    check("veri: iki kurulus birebir ayni", D.build()["train"] == d["train"] and D.build()["exam"] == d["exam"])
    check("veri: audit gecer (tutulan zincir metinde yok, donen zincir yok, adlar benzersiz)", D.audit(d) == 120)

    people = d["people"]
    classes = {}
    for e in d["exam"]:
        classes[e["cls"]] = classes.get(e["cls"], 0) + 1
    ok = (len(people) == 48 and len(d["train"]) == 640 and len(d["vocab"]) == 74
          and classes == dict(memory_base=160, memory_derived=120, chain2=48, chain3=32, named2=32, named3=8))
    check("veri: sayilar (48 kisi, 640 cumle, sozluk 74, sinif sayilari)", ok, str(classes))

    texts = [D.detokenize(s) for s in d["train"]] + [D.detokenize(e["prompt"]) for e in d["exam"]]
    check("veri: tokenize(detokenize(x)) == x butun cumle ve sorularda",
          all(D.tokenize(D.detokenize(s)) == s for s in d["train"]) and
          all(D.tokenize(D.detokenize(e["prompt"])) == e["prompt"] for e in d["exam"]), texts[0])

    vocab = set(d["vocab"])
    ids = [D.encode(s, d["vocab"]) for s in d["train"]]
    eos = d["vocab"].index(D.EOS)
    check("veri: sozluk her token'i kapsar, encode <eos> ile cevirir",
          all(t in vocab for s in d["train"] for t in s) and all(t in vocab for e in d["exam"] for t in e["prompt"]) and
          all(a.split()[0] in vocab for e in d["exam"] for a in e["answers"]) and
          all(i[0] == eos == i[-1] and 0 <= min(i) and max(i) < len(vocab) for i in ids))

    # bagimsiz referans: cevaplar yalniz egitim METNINDEKI tek adimlardan ve buradaki tanimlardan
    g = graph_from_text(d["train"])
    wrong = []
    for e in d["exam"]:
        x, path = e["subject"], e["path"]
        if path[0] in DEFINITIONS and len(path) == 1:
            want = set().union(*(walk(g, x, p) for p in DEFINITIONS[path[0]]))
        else:
            want = walk(g, x, path)
        if want != set(e["answers"]):
            wrong.append((D.detokenize(e["prompt"]), sorted(want), e["answers"]))
    check("veri: her sorunun cevabi yalniz yazili tek adimlardan (metinden) ayni cikar", not wrong, str(wrong[:2]))

    named_written = {}
    for s in d["train"]:
        t = D.detokenize(s)
        if not t.startswith("Who ") and t.count("'s") == 1:
            left, y = t[:-1].split(" is ")
            x, r = left.split("'s ")
            if r in DEFINITIONS:
                named_written.setdefault((x, r), set()).add(y)
    ok = all(ys == set().union(*(walk(g, x, p) for p in DEFINITIONS[r])) for (x, r), ys in named_written.items())
    check("veri: yazili adli olgular (aunt, cousin ...) tanimin birlesimine esit", ok and len(named_written) == 40,
          "%d adli soru" % len(named_written))

    held = set(d["held"])
    households = {}
    for x, p in people.items():
        if p["household"]:
            households.setdefault(p["household"], set()).add(x)
    split_ok = all(h <= held or not h & held for h in households.values()) and len(held) == 8
    leak = [D.detokenize(s) for s in d["train"] if s[0] + " " + s[1] in held and ("'s" in s[3:] or s[3] in DEFINITIONS)]
    leak += [D.detokenize(s) for s in d["train"] if s[0] == "Who" and s[2] + " " + s[3] in held
             and ("'s" in s[5:s.index("?")] or s[5] in DEFINITIONS)]
    check("veri: ayirma hane hane; tutulan torunun turemis iliskisi HIC yazilmadi", split_ok and not leak, str(leak[:2]))

    reverse = []
    pairs = set()
    for s in d["train"]:
        names = [" ".join(s[i:i + 2]) for i in range(len(s) - 1) if " ".join(s[i:i + 2]) in people]
        pairs.update((a, b) for a in names for b in names)
    for e in d["exam"]:
        if not e["cls"].startswith("memory"):
            reverse.append(any((e["subject"], y) in pairs for y in e["answers"]) == e["co_written"])
    check("veri: co_written = cevap ile ozne ayni egitim cumlesinde gecti (metinden)", all(reverse))

    check("veri: HIC YAZILMAMIS siniflarda co_written yalniz kuzen yolunda (kenar haneler)",
          all(not e["co_written"] or e["path"][-1] in ("son", "daughter", "cousin")
              for e in d["exam"] if not e["cls"].startswith("memory")))


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
    inputs, targets = TR.token_pairs(d)
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
    from model_20 import SequenceModel, T_MAX
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
    sids, smask = TR.sequences(data)
    nv = len(data["vocab"])
    before = SequenceModel(nv).tokens.fixed_points.clone()
    m, curve = TR.train_seq("step2", sids, smask, nv, steps=20, log_at=(0, 20))
    check("adim 2: egitimde PF bit duzeyinde degismez, W_context 0'dan ayrilir, kayip iner",
          torch.equal(m.tokens.fixed_points, before) and curve[-1]["W_context"] > 0 and curve[-1]["nll"] < curve[0]["nll"])


def F_ce(logits, targets):
    return float(torch.nn.functional.cross_entropy(logits, targets))


if __name__ == "__main__":
    print("tests (model_20)")
    for f in (t_data, t_model, t_step2):
        f()
    print("\n%d GECTI   %d KALDI" % (sum(RESULTS), len(RESULTS) - sum(RESULTS)))
    sys.exit(0 if all(RESULTS) else 1)
