# -*- coding: utf-8 -*-
"""data -- DUZ MATEMATIK: toplama, model_15'in verisiyle (model_17 veri_mat17'den).

Kullanici, 24 Eylul: "bunları kopyala ama matematik kopyala veri olarak ve
adını data yap".

    <eos> 4 7 2 + 1 8 2 = 6 5 4 <eos>
    Kayip YALNIZ cevabin rakamlarinda ve EOS'ta; soru rakamlari rastgele, hedef degil.

Veri model_15'in Drive'daki dosyalarindan (kural 9); bolme AYNEN onlarinki.
Iz yuklerken YENIDEN hesaplanip model_15'in kaydiyla karsilastirilir.

OLCUT model_15'inkiyle AYNI (sor): soru sorulur, EOS'a kadar serbest
uretim, cevap BIREBIR -- erken durmak, fazla rakam, yanlis rakam YANLIS.
Ek olarak ILK rakam: en buyuk basamak butun eldelere bagli, en zor secim.

Kayitli taban (eski mimari, 4.529 parametre, Drive model_15/*/kayit.txt):
    DUR  2 terim   tutulan 0,7770   uzunluk 0,9916
    COK  2+3 terim tutulan 0,4403   3 terimde 0,3693
"""
from __future__ import annotations

import hashlib

import torch
import torch.nn.functional as F

EOS_TOKEN = "<eos>"
PLUS, EQUALS, EOS = 10, 11, 12
N = 13
VOCAB = [str(i) for i in range(10)] + ["+", "=", EOS_TOKEN]

# model_15'in kayit.txt'sindeki izler -- dosya kayarsa load() durur.
FINGERPRINTS = {"dur": "eb4c73ab05178e18", "cok": "8d0f89938c67e0e7"}


def digits(x):
    """Sayinin KENDI rakamlari.  Dolgu YOK."""
    return [int(c) for c in str(x)]


def question(terms):
    """ts: terimler -> d..d + d..d [+ d..d] ="""
    w = []
    for i, x in enumerate(terms):
        if i:
            w.append(PLUS)
        w += digits(x)
    return w + [EQUALS]


def load(path, name):
    """model_15'in veri dosyasi -> (egitim sorulari, tutulan sorular).
    ad: "dur" (2 terim) ya da "cok" (2+3 terim)."""
    d = torch.load(path, weights_only=False)
    hasher = hashlib.sha256()
    for w, _, u in d["eg"] + d["tu"]:
        hasher.update(w.numpy().tobytes())
        hasher.update(u.numpy().tobytes())
    fp = hasher.hexdigest()[:16]
    if fp != FINGERPRINTS[name]:
        raise ValueError(f"veri izi tutmuyor: {fp} != {FINGERPRINTS[name]} ({path})")
    train_q = d["soru_eg"] if "soru_eg" in d else d["cift_eg"]
    heldout_q = d["soru_tu"] if "soru_tu" in d else d["cift_tu"]
    return [tuple(x) for x in train_q], [tuple(x) for x in heldout_q]


def make_windows(questions):
    """(W, M, H), n x T.  W: <eos> soru cevap <eos>, sagdan dolgu.
    M: gercek token'lar.  H: HEDEF konumlar -- cevabin rakamlari ve EOS."""
    seqs = [[EOS] + question(terms) + digits(sum(terms)) + [EOS] for terms in questions]
    T = max(len(d) for d in seqs)
    W = torch.full((len(seqs), T), EOS, dtype=torch.long)
    M = torch.zeros(len(seqs), T, dtype=torch.bool)
    H = torch.zeros(len(seqs), T, dtype=torch.bool)
    for i, (d, terms) in enumerate(zip(seqs, questions)):
        W[i, :len(d)] = torch.tensor(d)
        M[i, :len(d)] = True
        H[i, len(d) - len(digits(sum(terms))) - 1:len(d)] = True
    return W, M, H


def _sample(questions, limit, seed=12345):
    """Olcum alt kumesi RASTGELE, tohum sabit -- her kosu ayni kumeyi olcer."""
    if limit >= len(questions):
        return list(questions)
    g = torch.Generator().manual_seed(seed)
    return [questions[j] for j in torch.randperm(len(questions), generator=g)[:limit]
            .tolist()]


def _generate(m, questions, device, chunk=4096):
    """Serbest uretim: (soru, uretilen K token) ciftleri.  EOS'ta kesmez,
    K = en uzun cevap + 1 adim yurur; nerede durdugunu ask() okur."""
    buckets = {}
    for terms in questions:
        buckets.setdefault(len(question(terms)), []).append(terms)
    K = max(len(digits(sum(terms))) for terms in questions) + 1
    out = []
    with torch.no_grad():
        for group in buckets.values():
            for i in range(0, len(group), chunk):
                part = group[i:i + chunk]
                seq = torch.tensor([[EOS] + question(terms) for terms in part],
                                   device=device)
                for _ in range(K):
                    t = m.scoreboard(seq)[:, -1].argmax(-1)
                    seq = torch.cat([seq, t[:, None]], 1)
                out += [(terms, seq[r, -K:].tolist()) for r, terms in enumerate(part)]
    return out, K


def ask(m, questions, device="cuda", limit=20000):
    """TEK OLCUT: soru soruldu, cevap DOGRU MU.  Doner: accuracy, length_ok, first_digit, n.

      accuracy     cevap BIREBIR ve EOS dogru yerde
      length_ok    rakamlardan bagimsiz, DOGRU YERDE durdu mu
      first_digit  cevabin ILK rakami dogru mu (en buyuk basamak)"""
    questions = _sample(questions, limit)
    out, K = _generate(m, questions, device)
    n_correct = n_length = n_first = 0
    for terms, c in out:
        target = digits(sum(terms))
        stop = c.index(EOS) if EOS in c else K
        n_length += stop == len(target)
        n_correct += stop == len(target) and c[:len(target)] == target
        n_first += c[0] == target[0]
    n = len(out)
    return {"accuracy": n_correct / n, "length_ok": n_length / n, "first_digit": n_first / n, "n": n}


def answer_ce(m, windows, device="cuda", chunk=4096):
    """Cevap token'larinda (rakamlar + EOS) ortalama CE."""
    W, M, H = windows
    total = count = 0.0
    with torch.no_grad():
        for i in range(0, len(W), chunk):
            w, mk = W[i:i + chunk].to(device), M[i:i + chunk].to(device)
            a = H[i:i + chunk, 1:].to(device)
            p = m.scoreboard(w, mk)[:, :-1]
            ce = F.cross_entropy(p.transpose(1, 2), w[:, 1:], reduction="none")
            total += float(ce[a].sum())
            count += int(a.sum())
    return total / max(count, 1)


def make_metric(train_q, heldout_q, device="cuda", limit=2000):
    """train'in bekledigi metric(m, "train"|"heldout", full) -> dict.
      accuracy  ANA OLCUT: cevap birebir dogru (exact match)
      ce        cevap token'larinda kayip (ayni alt kume)
      diag      ANALIZ icin, matematige ozgu: first_digit, length_ok"""
    splits = {"train": train_q, "heldout": heldout_q}

    def f(m, side, full=False):
        s = splits[side]
        subset = _sample(s, len(s) if full else limit)
        r = ask(m, subset, device=device, limit=len(subset))
        return {"accuracy": r["accuracy"], "ce": answer_ce(m, make_windows(subset), device),
                "diag": {"first_digit": r["first_digit"], "length_ok": r["length_ok"]}}

    return f


def breakdown(m, questions, device="cuda", measure="terim"):
    """{anahtar: sor sonucu}.  olcu "terim" (2/3) ya da "hane" (cevap hanesi)."""
    f = (lambda terms: len(terms)) if measure == "terim" else (lambda terms: len(digits(sum(terms))))
    g = {}
    for terms in questions:
        g.setdefault(f(terms), []).append(terms)
    return {a: ask(m, c, device=device, limit=len(c)) for a, c in sorted(g.items())}


PLACES = ("birler", "onlar", "yuzler", "binler")


def digit_check(m, questions, device="cpu", chunk=4096):
    """TANI, olcut degil: HANGI BASAMAK ogrenildi.  Anahtar (cevap hanesi,
    konum); konum SAGDAN (birler, onlar, yuzler, binler) ya da "eos".
      og    dogru onek verilince o token dogru mu (ogretmen zorlamali)
      ser   serbest uretim, SAGA hizali: modelin yazdigi o basamak dogru mu
      cog   taban: o yuvada hep EN SIK token'i soyleyen
      n     soru sayisi"""
    W, M, _ = make_windows(questions)
    L = torch.tensor([len(digits(sum(terms))) for terms in questions])
    a0 = M.sum(1) - L - 1                     # cevabin ilk rakaminin yeri
    preds = []
    with torch.no_grad():
        for i in range(0, len(W), chunk):
            preds.append(m.scoreboard(W[i:i + chunk].to(device), M[i:i + chunk]
                                    .to(device)).argmax(-1).cpu())
    preds = torch.cat(preds).tolist()
    outputs = {terms: c[:c.index(EOS)] if EOS in c else c
           for terms, c in _generate(m, questions, device)[0]}
    stats = {}

    def add(k, teacher, free, h):
        s = stats.setdefault(k, {"teacher": 0, "free": 0, "n": 0, "freq": {}})
        s["teacher"] += teacher
        s["free"] += free
        s["n"] += 1
        s["freq"][h] = s["freq"].get(h, 0) + 1

    for i, terms in enumerate(questions):
        c, b, g, t = digits(sum(terms)), int(a0[i]), outputs[terms], preds[i]
        for q, h in enumerate(c):
            r = len(c) - 1 - q
            add((len(c), PLACES[r]), t[b + q - 1] == h,
                 r < len(g) and g[len(g) - 1 - r] == h, h)
        add((len(c), "eos"), t[b + len(c) - 1] == EOS, len(g) == len(c), EOS)
    return {k: {"teacher": s["teacher"] / s["n"], "free": s["free"] / s["n"],
                "majority": max(s["freq"].values()) / s["n"], "n": s["n"]}
            for k, s in stats.items()}


def digit_table(r, log=print):
    log("  hane  konum         n       teacher  free  majority")
    for L in sorted({k[0] for k in r}):
        for name in PLACES[:L][::-1] + ("eos",):
            v = r[(L, name)]
            log("  %4d  %-7s %8d   %6.4f   %6.4f   %6.4f"
                % (L, name, v["n"], v["teacher"], v["free"], v["majority"]))


def show(m, questions, device="cpu"):
    """GOZLE: soru, dogru cevap, modelin yazdigi."""
    out, _ = _generate(m, list(questions), device)
    for terms, c in out:
        stop = c.index(EOS) if EOS in c else len(c)
        written = "".join(VOCAB[t] for t in c[:stop]) + ("" if stop < len(c) else " (durmadi)")
        print(f"  {' + '.join(map(str, terms))} = {sum(terms):<5}  model: {written}"
              f"  {'DOGRU' if written == str(sum(terms)) else 'yanlis'}")
