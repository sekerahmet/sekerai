# -*- coding: utf-8 -*-
"""tests_20 -- model_20'nin kapilari.  CPU, saniyeler.

    python tests_20.py
"""
import json
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
FINGERPRINT = "454d53e81c67"      # veri degisirse burasi bilerek guncellenir (27 Eylul: 32 aile; once 90024739fb8f)


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
    check("veri: audit gecer (tutulan zincir metinde yok, donen zincir yok, adlar benzersiz)", D.audit(d) == 480)

    people = d["people"]
    classes = {}
    for e in d["exam"]:
        classes[e["cls"]] = classes.get(e["cls"], 0) + 1
    ok = (len(people) == 192 and len(d["train"]) == 2560 and len(d["vocab"]) == 242
          and classes == dict(memory_base=640, memory_derived=480, chain2=192, chain3=128, named2=128, named3=32))
    check("veri: sayilar (192 kisi, 2560 cumle, sozluk 242, sinif sayilari)", ok, str(classes))

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
    check("veri: yazili adli olgular (aunt, cousin ...) tanimin birlesimine esit", ok and len(named_written) == 160,
          "%d adli soru" % len(named_written))

    held = set(d["held"])
    households = {}
    for x, p in people.items():
        if p["household"]:
            households.setdefault(p["household"], set()).add(x)
    split_ok = all(h <= held or not h & held for h in households.values()) and len(held) == 32
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

    # Adim 4: yalniz 1R + ara adimli 2R; her adim Adim 0-3 metninde yazili bir 1R cumlesi olmali
    s = D.build(step_answers=True)
    stmts = {D.detokenize(t) for t in d["train"] if t[0] != "Who"}
    bad = []
    trained = [t for t in s["train"] if t[0] == "<steps>"]
    walks = [(e["subject"], e["steps"]) for e in s["exam"] if e["cls"] in ("2R_T", "2R_UT")]
    walks += [(" ".join(t[3:5]), t[t.index("?") + 1:]) for t in trained]
    for who, rest in walks:
        for i in range(0, len(rest), 8):
            txt = D.detokenize(rest[i:i + 8])
            if txt not in stmts or not txt.startswith(who + "'s "):
                bad.append(txt)
            who = txt[:-1].split(" is ")[1]
    held = set(d["held"])
    text = "\n".join(D.detokenize(t) for t in s["train"])
    leak = [e for e in s["exam"] if e["cls"] == "2R_UT" and D.detokenize(e["steps"]) in text]
    short = [t for t in s["train"] if t[0] != "<steps>" and (t.count("'s") > 1 or any(r in t for r in DEFINITIONS))]
    want = {tuple(e["prompt"] + e["steps"]) for e in s["exam"] if e["cls"] == "2R_T"}
    counts = {}
    for e in s["exam"]:
        counts[e["cls"]] = counts.get(e["cls"], 0) + 1
    check("veri, STEP_ANSWERS: egitimde yalniz 1R + 2 adimli ara adimli 2R (kisa 2R, adli, 3R yok); adimlar yazili 1R "
          "cumleleri; 160 ozne, tutulan torun hic ozne degil, 2R_UT adimlari metinde yok; 2R_T egitimde; audit gecer",
          not bad and not leak and not short and want <= {tuple(t) for t in trained} and len(trained) == 960
          and all(t.index("?") == 9 for t in trained)
          and len({" ".join(t[3:5]) for t in trained}) == 160 and not {" ".join(t[3:5]) for t in trained} & held
          and len(s["train"]) == 2240 and D.audit(s) == 192
          and counts == {"1R_T": 640, "2R_T": 192, "2R_UT": 192}, str(bad[:2] or leak[:1] or short[:1] or counts))

    u = D.build(step_answers=True, step_marker=False)
    same_chains = sorted(tuple(t[1:]) for t in trained) == sorted(tuple(t) for t in u["train"] if t[0] == "Who" and t.index("?") == 8)
    same_exam = [(e["cls"], e["prompt"][1:] if e["prompt"][0] == "<steps>" else e["prompt"], e["answers"]) for e in s["exam"]] \
        == [(e["cls"], e["prompt"], e["answers"]) for e in u["exam"]]
    check("veri, STEP_MARKER=False: ayni zincirler ve sinav, yalniz '<steps>' yok (sozluk 237, iz 3898f9b3d551); audit gecer; "
          "varsayilan (True) iz degismedi",
          same_chains and same_exam and "<steps>" not in u["vocab"] and len(u["vocab"]) == 237 and u["fingerprint"] == "3898f9b3d551"
          and D.audit(u) == 192 and s["fingerprint"] == "7ae5615623aa" and len(u["train"]) == 2240, u["fingerprint"])

    L = D.build(step_answers=True, long_1r=True)
    one = [e for e in L["exam"] if e["cls"] == "1R_T"]
    long_qa = [t for t in L["train"] if t[0] == "Who" and t.index("?") == 6]
    textL = "\n".join(D.detokenize(t) for t in L["train"])
    owen = "Who is Owen Evans's father? Owen Evans's father is David Evans."
    check("veri, LONG_1R: 1R soru-cevabi tam cumle (640, tutulanlar dahil); zincirler ve 2R sinavi ayni; 1R_T'de beklenen "
          "cumle; 2R_UT adimlari metinde yok; audit gecer; kapaliyken iz degismedi",
          len(long_qa) == 640 and all(t[7:9] == t[2:4] and t[9:11] == ["'s", t[5]] for t in long_qa) and owen in textL
          and [t for t in L["train"] if t[0] == "<steps>"] == trained
          and [e for e in L["exam"] if e["cls"] != "1R_T"] == [e for e in s["exam"] if e["cls"] != "1R_T"]
          and all(e["steps"] == e["subject"].split() + ["'s", e["path"][0], "is"] + e["answers"][0].split() + ["."] for e in one)
          and not [e for e in L["exam"] if e["cls"] == "2R_UT" and D.detokenize(e["steps"]) in textL]
          and D.audit(L) == 192 and L["long_1r"] and not s["long_1r"] and s["fingerprint"] == "7ae5615623aa",
          L["fingerprint"])


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


def reference_blocks(m, ids):
    """Adim 3 formulu, float64, modelden bagimsiz: h = PL; her turda attention (durumlari getirir), W_context, FactUnits;
    cikis h (W_next yok).  Oneri A acikken: + g · W_copy · Σ a PL, g = sigmoid(W_copy_gate h + copy_gate_bias)."""
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
        a = torch.softmax((at.scale * q @ k.transpose(-1, -2)).masked_fill(future, float("-inf")), -1)
        added = (a @ x) @ dd(at.W_context).T
        if m.copy_path:
            gate = 1 / (1 + torch.exp(-(x @ dd(at.W_copy_gate).T + dd(at.copy_gate_bias))))
            added = added + gate * ((a @ PL[ids]) @ dd(at.W_copy).T)
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
    from model_20 import BlockModel, TURNS
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
    for kw in (dict(shared=True), dict(shared=False), dict(shared=True, copy_path=True)):
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

    data = D.build()
    sids, smask = TR.sequences(data)
    nv = len(data["vocab"])
    ml, curve_l = TR.train_seq("shared", sids, smask, nv, steps=20, log_at=(0, 20), stream_norm=False, layer_norm=True)
    check("adim 3, layer_norm=True: train_seq ile egitilir, kayip iner", ml.layer_norm and curve_l[-1]["nll"] < curve_l[0]["nll"],
          "%.3f -> %.3f" % (curve_l[0]["nll"], curve_l[-1]["nll"]))
    mf, curve_f = TR.train_seq("shared", sids, smask, nv, steps=20, log_at=(0, 20), stream_norm=False)
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
        TR.train_seq("shared", sids, smask, nv, steps=4, log_at=(), lr=0.01, lr_floor=0.1, grad_clip=0.5)
        cosine = [0.01 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / 4))) for t in range(4)]
        ok_sched = all(abs(lr_ - c) < 1e-12 for (lr_, _), c in zip(seen, cosine))
        ok_clip = all(g <= 0.5 + 1e-5 for _, g in seen)
        seen.clear()
        TR.train_seq("shared", sids, smask, nv, steps=4, log_at=(), lr=0.01)
        standard = [0.01 * (TR.LR_FLOOR + (1 - TR.LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * t / 4))) for t in range(4)]
        ok_default = (all(abs(lr_ - c) < 1e-12 for (lr_, _), c in zip(seen, standard))
                      and all(g <= TR.GRAD_CLIP + 1e-5 for _, g in seen))
        seen.clear()
        TR.train_seq("shared", sids, smask, nv, steps=3, log_at=(), lr=0.01, lr_floor=None, grad_clip=None)
        ok_old = all(lr_ == 0.01 for lr_, _ in seen) and any(g > 0.5 for _, g in seen)
    finally:
        torch.optim.Adam = real_adam
    check("egitim tarifi: cosine decay ile lr LR'den LR x lr_floor'a iner, gradient boyu grad_clip'i gecmez; standart LR_FLOOR ve "
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
        mw, _ = TR.train_seq("shared", sids, smask, nv, steps=1, log_at=(), weight_decay=0.1)
    finally:
        torch.optim.AdamW = real_adamw
    names = {id(p_): k.split(".")[-1] for k, p_ in mw.named_parameters()}
    decayed = sorted(names[id(p_)] for g_ in groups if g_["weight_decay"] == 0.1 for p_ in g_["params"])
    kept = sorted(names[id(p_)] for g_ in groups if g_["weight_decay"] == 0.0 for p_ in g_["params"])
    check("weight decay: yalniz W_ matrislerine; shift ve fact_threshold haric",
          decayed == sorted(["W_query", "W_key", "W_context", "W_fact_in", "W_fact_out"])
          and kept == ["fact_threshold", "shift"], "%s | %s" % (decayed, kept))


def t_copy():
    from model_20 import BlockModel
    n, d = 12, 6
    g = torch.Generator().manual_seed(11)
    ids = torch.randint(0, n, (3, 9), generator=g)

    off, on = BlockModel(n, d=d, units=10), BlockModel(n, d=d, units=10, copy_path=True)
    so, sn = off.state_dict(), on.state_dict()
    added = sorted(k.split(".")[-1] for k in set(sn) - set(so))
    same_start = all(torch.equal(so[k], sn[k]) for k in so)
    with torch.no_grad():                        # baslangicta W_context = W_fact_out = 0: iki model zaten ayni olurdu
        for k, p_ in off.named_parameters():
            if p_.requires_grad:
                v = 0.5 * torch.randn(p_.shape, generator=g)
                p_.copy_(v)
                dict(on.named_parameters())[k].copy_(v)
        on.blocks[0].attention.W_copy_gate.copy_(torch.randn(1, d, generator=g))
    err = float((on.logits(ids) - off.logits(ids)).detach().abs().max())
    check("kopya yolu: varsayilan kapali ve parametresi yok; acikken yalniz W_copy, W_copy_gate, copy_gate_bias eklenir, "
          "digerleri ayni baslar; rastgele parametrelerle W_copy = 0 iken skor kapaliyla ayni",
          not any("copy" in k for k in so) and added == ["W_copy", "W_copy_gate", "copy_gate_bias"]
          and same_start and err < 1e-5, "%s, fark %.1e" % (added, err))

    for kw in (dict(shared=True), dict(shared=False)):
        m = BlockModel(n, d=d, units=10, copy_path=True, **kw)
        with torch.no_grad():
            for p_ in m.parameters():
                if p_.requires_grad:
                    p_.copy_(0.5 * torch.randn(p_.shape, generator=g))
        err = float((m.logits(ids).detach().double() - reference_blocks(m, ids)).abs().max())
        check("kopya yolu: skor = tasarim formulu (bagimsiz float64; c' = sum a PL, g = sigmoid(W_copy_gate h + b)), %s" % kw,
              err < 1e-4, "fark %.1e" % err)

    with torch.no_grad():
        for b in m.blocks:
            b.attention.copy_gate_bias.fill_(-50.0)                   # kapi kapali: g ~ 2e-22
        closed = m.logits(ids).detach()
        for b in m.blocks:
            b.attention.W_copy.zero_()
        err = float((closed - m.logits(ids).detach()).abs().max())
    check("kopya yolu: kapi kapaliyken (g ~ 0) kopya katkisi yok, W_copy = 0 ile ayni skor", err < 1e-5, "fark %.1e" % err)

    m = BlockModel(n, d=d, units=10, copy_path=True)
    with torch.no_grad():
        for p_ in m.parameters():
            if p_.requires_grad:
                p_.copy_(0.5 * torch.randn(p_.shape, generator=g))
    base_logits = m.logits(ids).detach()
    changed = ids.clone()
    changed[:, 5] = (changed[:, 5] + 1) % n
    after = m.logits(changed).detach()
    check("kopya yolu: nedensellik -- konum 5 degisince 0-4 aynen kalir",
          float((after[:, :5] - base_logits[:, :5]).abs().max()) < 1e-5 and float((after[:, 5:] - base_logits[:, 5:]).abs().max()) > 1e-3)

    data = D.build()
    sids, smask = TR.sequences(data)
    nv = len(data["vocab"])
    before = BlockModel(nv).tokens.fixed_points.clone()
    m, curve = TR.train_seq("shared", sids, smask, nv, steps=20, log_at=(0, 20), copy_path=True)
    at = m.blocks[0].attention
    check("kopya yolu: egitimde PF bit duzeyinde degismez, W_copy 0'dan ayrilir, kapi degisir, kayip iner",
          torch.equal(m.tokens.fixed_points, before) and at.W_copy.norm().item() > 0 and at.W_copy_gate.norm().item() > 0
          and curve[-1]["nll"] < curve[0]["nll"])

    groups = []
    real_adamw = torch.optim.AdamW

    class SpyW(real_adamw):
        def __init__(self, param_groups, **k):
            super().__init__(param_groups, **k)
            groups.extend(self.param_groups)
    torch.optim.AdamW = SpyW
    try:
        mw, _ = TR.train_seq("shared", sids, smask, nv, steps=1, log_at=(), weight_decay=0.1, copy_path=True)
    finally:
        torch.optim.AdamW = real_adamw
    names = {id(p_): k.split(".")[-1] for k, p_ in mw.named_parameters()}
    decayed = sorted(names[id(p_)] for g_ in groups if g_["weight_decay"] == 0.1 for p_ in g_["params"])
    kept = sorted(names[id(p_)] for g_ in groups if g_["weight_decay"] == 0.0 for p_ in g_["params"])
    check("kopya yolu, weight decay: W_copy ve W_copy_gate'e uygulanir; copy_gate_bias haric",
          decayed == sorted(["W_query", "W_key", "W_context", "W_fact_in", "W_fact_out", "W_copy", "W_copy_gate"])
          and kept == ["copy_gate_bias", "fact_threshold", "shift"], "%s | %s" % (decayed, kept))


def reference_transformer(m, ids):
    """model_20_transformer formulu, float64, modelden bagimsiz: LayerNorm, RoPE (karmasik sayi carpimiyla), nedensel
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
    from model_20_transformer import HEADS, LAYERS, TransformerModel, apply_rope
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
    sids, smask = TR.sequences(data)
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
    try:
        TR.train_seq("shared", sids, smask, nv, steps=1, log_at=(), rope=False)
        refused = False
    except AssertionError:
        refused = True
    check("transformer, rope=False: train_seq ayari modele ulasir (her katmanda rope False), kayip iner; BlockModel'de "
          "reddedilir; varsayilan True",
          all(not layer.rope for layer in mr.layers) and all(layer.rope for layer in mv.layers)
          and curve_r[-1]["nll"] < curve_r[0]["nll"] and refused,
          "%.3f -> %.3f" % (curve_r[0]["nll"], curve_r[-1]["nll"]))


def t_resume():
    import copy
    import tempfile
    import colab_20 as C
    s = D.build(step_answers=True)
    ids, mask = TR.sequences(s)
    ids, mask = ids[:40], mask[:40]
    nv = len(s["vocab"])
    for kw in (dict(), dict(copy_path=True), dict(weight_decay=0.1)):
        packs, seen_full, seen_resumed = {}, [], []

        def keep(step, model, opt):
            packs[step] = dict(step=step, model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(opt.state_dict()))
        full, _ = TR.train_seq("shared", ids, mask, nv, steps=6, log_at=(), save_every=2, save=keep, every=2,
                               callback=lambda st, m, nll: seen_full.append(st), **kw)
        resumed, _ = TR.train_seq("shared", ids, mask, nv, steps=6, log_at=(), checkpoint=packs[4], every=2,
                                  callback=lambda st, m, nll: seen_resumed.append(st), **kw)
        same = all(torch.equal(a, b) for a, b in zip(full.state_dict().values(), resumed.state_dict().values()))
        check("surdurme: 4. adim paketinden surdurulen = kesintisiz 6 adim, bit duzeyinde; paketler 2, 4, 6; sinav tekrarlanmaz %s"
              % (kw or ""), same and sorted(packs) == [2, 4, 6] and seen_full == [0, 2, 4, 6] and seen_resumed == [6],
              "%s %s %s" % (sorted(packs), seen_full, seen_resumed))

    out = tempfile.mkdtemp() + "/r"
    run = C.start("TEST_R", s, out, steps=3, every=100, device="cpu", compile=False, save_every=1)
    run["thread"].join(600)
    files = sorted(f for f in os.listdir(out) if f.startswith("checkpoint_t"))
    os.remove(os.path.join(out, "checkpoint_t00003.pt"))            # 2. adimdan sonra kesilmis gibi
    lines_before = open(os.path.join(out, "log.txt"), encoding="utf-8").read().count("\n")
    run2 = C.start("TEST_R", s, out, steps=3, every=100, device="cpu", compile=False, save_every=1, resume=True)
    run2["thread"].join(600)
    log = open(os.path.join(out, "log.txt"), encoding="utf-8").read()
    try:
        C.start("TEST_R2", s, out, steps=5, every=100, device="cpu", compile=False, save_every=1, resume=True)
        refused = False
    except RuntimeError:
        refused = True
    check("surdurme, colab_20: her adimda checkpoint_tNNNNN.pt; resume=True son paketten devam eder, gunluge yazar, "
          "klasoru tasimaz; ayar farkliysa reddeder",
          files == ["checkpoint_t00001.pt", "checkpoint_t00002.pt", "checkpoint_t00003.pt"] and run2["done"]
          and not run2["error"] and "SURDURULDU adim 2'den" in log and log.count("\n") > lines_before
          and os.path.exists(os.path.join(out, "checkpoint_t00003.pt")) and refused, str(run2["error"] or files))


def t_colab():
    import tempfile
    import colab_20 as C
    from model_20 import BlockModel
    s = D.build(step_answers=True)
    e = next(e for e in s["exam"] if e["cls"] == "2R_UT")
    good = TR.score_steps(list(e["steps"]), e)
    bad = list(e["steps"])
    bad[-3] = "Tom"
    worse = TR.score_steps(bad, e)
    check("adim 4: score_steps -- dogru cevapta SC BC AC EX FC hepsi evet; son ad yanlissa yalniz AC ve EX hayir",
          all(good.values()) and sorted(good) == ["AC", "BC", "EX", "FC", "SC"]
          and not worse["AC"] and not worse["EX"] and worse["BC"] and worse["SC"] and worse["FC"])

    m = BlockModel(len(s["vocab"]), d=16, units=8)
    gr = torch.Generator().manual_seed(5)
    with torch.no_grad():                        # egitilmemis model birim donusum gibi: '?'tan '?' uretir, test bos gecerdi
        for p_ in m.parameters():
            if p_.requires_grad:
                p_.copy_(0.5 * torch.randn(p_.shape, generator=gr))
    prompts = [TR.questions(s, "2R_UT")[i][0] for i in (0, 1)] + [TR.questions(s, "1R_T")[0][0]]
    out = TR.generate(m, prompts, 5)
    manual = []
    with torch.no_grad():
        for p in prompts:
            ids = list(p)
            for _ in range(5):
                ids.append(int(m.logits(torch.tensor([ids]))[0, -1].argmax()))
            manual.append(ids[len(p):])
    check("adim 4: generate toplu uretim = tek tek acgozlu uretim (farkli uzunluklar birlikte, rastgele model)",
          out == manual and len({tuple(o) for o in out}) > 1, str(out))

    seen = []
    ids, mask = TR.sequences(s)
    TR.train_seq("shared", ids[:20], mask[:20], len(s["vocab"]), steps=4, log_at=(), every=2,
                 callback=lambda step, model, nll: seen.append(step))
    check("train_seq: callback her every adimda, guncellemeden once (0, 2, 4)", seen == [0, 2, 4], str(seen))

    out_dir = tempfile.mkdtemp()
    run = C.start("TEST", s, out_dir + "/r", steps=2, every=100, device="cpu", compile=False)
    run["thread"].join(600)
    files = sorted(os.listdir(out_dir + "/r"))
    check("colab_20: CPU'da start -> sinav, model, son olcum dosyalari; pulse/stop hatasiz",
          run["done"] and not run["error"] and files == ["config.json", "exams.json", "final.json", "log.txt", "model.pt"],
          str(run["error"] or files))
    C.pulse(1)
    C.stop()

    L = D.build(step_answers=True, long_1r=True)
    run = C.start("TEST_L", L, out_dir + "/l", steps=1, every=100, device="cpu", compile=False)
    run["thread"].join(600)
    fin = json.load(open(out_dir + "/l/final.json", encoding="utf-8")) if os.path.exists(out_dir + "/l/final.json") else {}
    ex = json.load(open(out_dir + "/l/exams.json")) if os.path.exists(out_dir + "/l/exams.json") else [{}]
    check("colab_20: LONG_1R verisinde 1R_T tam cumleyle puanlanir (AC, EX), final.json'da 1R_T_given0",
          run["done"] and not run["error"] and "1R_T_given0" in fin and "1R_T_EX" in ex[0]
          and len(fin["1R_T_given0"]["rows"]) == 640, str(run["error"] or sorted(fin)))

    run = C.start("TEST_T", s, out_dir + "/t", steps=2, every=100, device="cpu", compile=False, setting="transformer")
    run["thread"].join(600)
    cfg = json.load(open(out_dir + "/t/config.json"))
    check("colab_20: setting='transformer' ile start; config ayarlarin tamamini yazar (varsayilanlar dahil)",
          run["done"] and not run["error"] and cfg["setting"] == "transformer" and cfg["copy_path"] is False
          and cfg["lr"] == TR.LR and cfg["grad_clip"] == TR.GRAD_CLIP and cfg["rope"] is True
          and "final.json" in os.listdir(out_dir + "/t"),
          str(run["error"] or cfg))


def F_ce(logits, targets):
    return float(torch.nn.functional.cross_entropy(logits, targets))


if __name__ == "__main__":
    print("tests (model_20)")
    for f in (t_data, t_model, t_step2, t_step3, t_copy, t_transformer, t_resume, t_colab):
        f()
    print("\n%d GECTI   %d KALDI" % (sum(RESULTS), len(RESULTS) - sum(RESULTS)))
    sys.exit(0 if all(RESULTS) else 1)
