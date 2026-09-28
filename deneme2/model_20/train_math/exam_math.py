# -*- coding: utf-8 -*-
"""exam_math -- toplamanin olcutu, model_15 / model_19 ile AYNI (data_mat_19'dan kopya; m.scoreboard yerine model.logits).

Soru sorulur (<eos> soru =), <eos>'a kadar serbest acgozlu uretim; cevap BIREBIR ve <eos> dogru yerde olmali -- erken
durmak, fazla rakam, yanlis rakam YANLIS.  Yaninda length_ok (dogru yerde durdu mu) ve first_digit (en buyuk basamak).
Olcum alt kumesi sabit: tohum 12345, ilk limit soru (model_15'in gunlugundeki 20.000'lik egri); sonda butun kume (kayit.txt).
"""
import torch
import torch.nn.functional as F

from data_math import EOS, VOCAB, digits, question
from train_20 import generate

HEALTH = 0.99            # saglik: egitim sorularinin sabit orneginde accuracy (kullanici onayi; gecen en kucuk boyut secilir)
# model_19'un 8 tani sorusu (belge/OLCULENLER.md §1l); ikisi terim araliginin (0..500) disinda
PANEL = ((123, 456), (472, 182), (87, 65), (1, 1), (999, 1), (505, 495), (909, 99, 2), (25, 34, 12))


def _sample(questions, limit, seed=12345):
    """Olcum alt kumesi RASTGELE, tohum sabit -- her kosu ayni kumeyi olcer."""
    if limit >= len(questions):
        return list(questions)
    g = torch.Generator().manual_seed(seed)
    return [questions[j] for j in torch.randperm(len(questions), generator=g)[:limit].tolist()]


@torch.no_grad()
def _generate(m, questions, device, chunk=4096):
    """Serbest uretim: (soru, uretilen K token) ciftleri.  <eos>'ta kesmez, K = en uzun cevap + 1 adim yurur; nerede
    durdugunu ask() okur."""
    buckets = {}
    for terms in questions:
        buckets.setdefault(len(question(terms)), []).append(terms)
    K = max(len(digits(sum(terms))) for terms in questions) + 1
    order = [terms for group in buckets.values() for terms in group]   # soru boyuna gore sira: show()'un satirlari
    out = []
    for i in range(0, len(order), chunk):              # farkli boylar ayni parcada (artimli uretim satir basina konum tutar)
        part = order[i:i + chunk]
        out += list(zip(part, generate(m, [[EOS] + question(terms) for terms in part], K)))
    return out, K


def ask(m, questions, device="cuda", limit=20000):
    """TEK OLCUT: soru soruldu, cevap DOGRU MU.  Doner: accuracy, length_ok, first_digit, n.
      accuracy     cevap BIREBIR ve <eos> dogru yerde
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


@torch.no_grad()
def answer_ce(m, windows, device="cuda", chunk=4096):
    """Cevap token'larinda (rakamlar + <eos>) ortalama CE."""
    W, M, H = windows
    total = count = 0.0
    for i in range(0, len(W), chunk):
        w = W[i:i + chunk].to(device)
        a = H[i:i + chunk, 1:].to(device)
        p = m.logits(w)[:, :-1]
        ce = F.cross_entropy(p.transpose(1, 2), w[:, 1:], reduction="none")
        total += float(ce[a].sum())
        count += int(a.sum())
    return total / max(count, 1)


def breakdown(m, questions, device="cuda", measure="terms"):
    """{anahtar: ask sonucu}.  measure "terms" (2/3 terim) ya da "digits" (cevabin hane sayisi)."""
    f = (lambda terms: len(terms)) if measure == "terms" else (lambda terms: len(digits(sum(terms))))
    g = {}
    for terms in questions:
        g.setdefault(f(terms), []).append(terms)
    return {a: ask(m, c, device=device, limit=len(c)) for a, c in sorted(g.items())}


def show(m, questions, device="cpu"):
    """GOZLE (kural 12): [(soru, dogru cevap, modelin yazdigi, dogru mu)]; yazdigi <eos>'a kadar, durmadiysa isaretli."""
    out, _ = _generate(m, list(questions), device)
    rows = []
    for terms, c in out:
        stop = c.index(EOS) if EOS in c else len(c)
        written = "".join(VOCAB[t] for t in c[:stop]) + ("" if stop < len(c) else " (durmadi)")
        rows.append((" + ".join(map(str, terms)), str(sum(terms)), written, written == str(sum(terms))))
    return rows
