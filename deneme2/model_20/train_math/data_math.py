# -*- coding: utf-8 -*-
"""data_math -- toplama verisi: model_15'in "dur" (2 terim) ve "cok" (2 + 3 terim) kumeleri, Model X icin.

Kod model_15/veri_dur.py, veri_cok.py (uretim) ve model_19/data_mat_19.py'den (dizi, iz) KOPYA; import edilmez.
    <eos> 4 7 2 + 1 8 2 = 6 5 4 <eos>          sozluk 13: 0-9, +, =, <eos>; dolgu da <eos> (maskede yok)
    dur: a, b in 0..500, a + b < 1000 (251.000 cift); cevap hanesine gore egitime %100 / %75 / %50, gerisi tutulan
    cok: dur'un ciftleri AYNEN + 251.000 ornek uclu (terimler 0..500), uclular ayri bolunur (4 hane %50)
Veri koddan uretilir (saniyeler) ve iz model_15'in kaydiyla karsilastirilir (kural 9'un amaci: iki kosu ayni veriyi gorsun).
Yedek: Drive'daki model_15 dosyasi (MyDrive/model_20/train_math/veri_<ad>.pt), load() ayni izle.
"""
import hashlib

import numpy as np
import torch

EOS_TOKEN = "<eos>"
PLUS, EQUALS, EOS = 10, 11, 12
N = 13
VOCAB = [str(i) for i in range(10)] + ["+", "=", EOS_TOKEN]

# model_15'in kayit.txt'sindeki izler (Drive model_15/DUR, COK) -- veri kayarsa build()/load() durur.
FINGERPRINTS = {"dur": "eb4c73ab05178e18", "cok": "8d0f89938c67e0e7"}
MAX_TERM = 500                                  # her terim 0..500
TRAIN_SHARE = {1: 1.00, 2: 0.75, 3: 0.50, 4: 0.50}   # cevabin hane sayisina gore egitime giden pay (model_15, 22 Eylul)
TRIPLES = 251_000                               # cok'ta ornek uclu sayisi (ikili sayisi kadar)


def digits(x):
    """Sayinin KENDI rakamlari.  Dolgu YOK."""
    return [int(c) for c in str(x)]


def question(terms):
    """terimler -> d..d + d..d [+ d..d] ="""
    w = []
    for i, x in enumerate(terms):
        if i:
            w.append(PLUS)
        w += digits(x)
    return w + [EQUALS]


def pairs():
    """Iki terimli evren; 500 + 500 = 1000 yok (tek 4 haneli cevapti)."""
    return [(a, b) for a in range(MAX_TERM + 1) for b in range(MAX_TERM + 1) if a + b < 1000]


def triples(seed=0):
    """Ornek uclular, tekrarsiz, belirlenimci (evren 501^3 sayilamaz)."""
    g = torch.Generator().manual_seed(seed)
    out, seen = [], set()
    while len(out) < TRIPLES:
        for r in torch.randint(0, MAX_TERM + 1, (TRIPLES, 3), generator=g).tolist():
            k = tuple(r)
            if k not in seen:
                seen.add(k)
                out.append(k)
                if len(out) == TRIPLES:
                    break
    return out


def split_by_length(questions, seed):
    """Cevabin hane sayisina gore TRAIN_SHARE kadari egitime; soru bazinda."""
    g = torch.Generator().manual_seed(seed)
    buckets = {}
    for terms in questions:
        buckets.setdefault(len(str(sum(terms))), []).append(terms)
    train, heldout = [], []
    for h in sorted(buckets):
        c = buckets[h]
        k = torch.randperm(len(c), generator=g)
        n = round(len(c) * TRAIN_SHARE[h])
        train += [c[j] for j in k[:n].tolist()]
        heldout += [c[j] for j in k[n:].tolist()]
    return train, heldout


def make_split(name, seed=0):
    """(egitim sorulari, tutulan sorular); ikililer ve uclular AYRI bolunur (cok'un ikili yarisi dur'un aynisi)."""
    train, heldout = split_by_length(pairs(), seed)
    if name == "cok":
        t3, h3 = split_by_length(triples(seed), seed + 1)
        train, heldout = train + t3, heldout + h3
    return train, heldout


def examples(questions):
    """model_15'in egitim bicimi: her soru -> hane + 1 sonraki-token ornegi (w, h, u), girdi uzunluguna gore obek.
    Burada YALNIZ iz icin (model_15'in izi bu tensorler uzerinden)."""
    g = {}
    for terms in questions:
        q, c = question(terms), digits(sum(terms))
        for i in range(len(c) + 1):
            w = q + c[:i]
            g.setdefault(len(w), ([], [], []))
            g[len(w)][0].append(w)
            g[len(w)][1].append(c[i] if i < len(c) else EOS)
            g[len(w)][2].append(len(c))
    return [(torch.tensor(w), torch.tensor(h), torch.tensor(u)) for _, (w, h, u) in sorted(g.items())]


def fingerprint(train_examples, heldout_examples):
    """model_15 / data_mat_19 ile ayni hesap: w ve u tensorlerinin baytlari, sha256'nin ilk 16 hanesi."""
    hasher = hashlib.sha256()
    for w, _, u in train_examples + heldout_examples:
        hasher.update(w.numpy().tobytes())
        hasher.update(u.numpy().tobytes())
    return hasher.hexdigest()[:16]


def load(path, name):
    """YEDEK: model_15'in Drive dosyasi -> (egitim sorulari, tutulan sorular); iz tutmazsa DURUR."""
    d = torch.load(path, weights_only=False)
    fp = fingerprint(d["eg"], d["tu"])
    if fp != FINGERPRINTS[name]:
        raise ValueError(f"veri izi tutmuyor: {fp} != {FINGERPRINTS[name]} ({path})")
    train_q = d["soru_eg"] if "soru_eg" in d else d["cift_eg"]
    heldout_q = d["soru_tu"] if "soru_tu" in d else d["cift_tu"]
    return [tuple(x) for x in train_q], [tuple(x) for x in heldout_q]


def make_windows(questions, width=None):
    """(W, M, H), n x T.  W: <eos> soru cevap <eos>, sagdan <eos> dolgu.  M: gercek token'lar.  H: HEDEF konumlar --
    cevabin rakamlari ve son <eos>.  width: sabit genislik (batch'ler icin; None = en uzun dizi)."""
    seqs = [[EOS] + question(terms) + digits(sum(terms)) + [EOS] for terms in questions]
    T = max(len(d) for d in seqs) if width is None else width
    W = torch.full((len(seqs), T), EOS, dtype=torch.long)
    M = torch.zeros(len(seqs), T, dtype=torch.bool)
    H = torch.zeros(len(seqs), T, dtype=torch.bool)
    for i, (d, terms) in enumerate(zip(seqs, questions)):
        W[i, :len(d)] = torch.tensor(d)
        M[i, :len(d)] = True
        H[i, len(d) - len(digits(sum(terms))) - 1:len(d)] = True
    return W, M, H


def build(name, source="code", path=None):
    """Veri: source "code" koddan uretir, "drive" model_15 dosyasini okur (path); iz FINGERPRINTS[name] ile ayni olmali."""
    if source == "code":
        train, heldout = make_split(name)
        fp = fingerprint(examples(train), examples(heldout))
        if fp != FINGERPRINTS[name]:
            raise ValueError("koddan uretilen verinin izi %s != %s -- kod ya da torch ureteci degismis; source='drive' "
                             "ile Drive kopyasi okunur" % (fp, FINGERPRINTS[name]))
    else:
        train, heldout = load(path, name)
    width = max(len(question(t)) + len(digits(sum(t))) + 2 for t in train + heldout)
    return dict(name=name, fingerprint=FINGERPRINTS[name], vocab=VOCAB, train=train, heldout=heldout, width=width,
                windows=make_windows(train, width))


def batches(data, batch_size, seed=0, answer_only=True):
    """train_seq icin adimin fonksiyonu step -> (ids, mask), sekil hep (batch_size, width).  Epok e: egitim sorularinin
    (seed, e) tohumlu permutasyonu sirayla bolunur; artan son parca o epokta kullanilmaz.  mask: answer_only ise H (cevap
    rakamlari + son <eos>), degilse M (butun gercek token'lar)."""
    W, M, H = data["windows"]
    n = len(W)
    per_epoch = n // batch_size
    assert per_epoch > 0, "batch_size (%d) > egitim sorusu (%d)" % (batch_size, n)
    target = H if answer_only else M
    memo = {}

    def batch(step):
        epoch, i = divmod(step, per_epoch)
        if epoch not in memo:                           # epok basina bir permutasyon; RandomState akisi numpy surumuyle degismez
            memo.clear()
            memo[epoch] = torch.from_numpy(np.random.RandomState([seed, epoch]).permutation(n))
        rows = memo[epoch][i * batch_size:(i + 1) * batch_size]
        return W[rows], target[rows]
    return batch


def audit(data):
    """Egitim ve tutulan ayrik, terimler aralikta; cok'ta ikili yari dur'un aynisi.  Bozuksa DURUR.  -> sayimlar."""
    train, heldout = data["train"], data["heldout"]
    assert not set(train) & set(heldout), "tutulan soru egitimde"
    assert all(0 <= x <= MAX_TERM for t in train + heldout for x in t), "terim araligin disinda"
    counts = {}
    for side, qs in (("train", train), ("heldout", heldout)):
        for t in qs:
            k = "%s_%dterm" % (side, len(t))
            counts[k] = counts.get(k, 0) + 1
    if data["name"] == "cok":
        dur_train, dur_heldout = split_by_length(pairs(), 0)
        assert [t for t in train if len(t) == 2] == dur_train and [t for t in heldout if len(t) == 2] == dur_heldout, \
            "cok'un ikili yarisi dur'dan farkli"
    return counts


def report(data):
    out = ["data_math %s  iz %s  sozluk %d  genislik %d  egitim %d  tutulan %d" % (
        data["name"], data["fingerprint"], len(data["vocab"]), data["width"], len(data["train"]), len(data["heldout"])),
        "  %-6s %-5s %9s %9s" % ("terim", "hane", "egitim", "tutulan")]
    table = {}
    for side, qs in (("train", data["train"]), ("heldout", data["heldout"])):
        for t in qs:
            key = (len(t), len(str(sum(t))))
            table.setdefault(key, {"train": 0, "heldout": 0})[side] += 1
    for (terms, h), c in sorted(table.items()):
        out.append("  %-6d %-5d %9d %9d" % (terms, h, c["train"], c["heldout"]))
    return "\n".join(out)
