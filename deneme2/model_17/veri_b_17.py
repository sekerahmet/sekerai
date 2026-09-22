# -*- coding: utf-8 -*-
"""veri_b_17 -- CLUTRR, B hali: ISIMLI.  A ile ayni veri, isimler yerinde.

Kullanici, 22 Eylul: *"hazir veriyi bozmayalim diye kullanalim"* --
yeniden adlandirma YOK, siralama karistirma YOK, veri ne ise o.

BICIM.  Hikaye UCLU dizisi, sonra soru:

    <kaynak> <iliski> <hedef>  ... (k kez) ...  ?  <kim> <kim>

Ucluyu zincire sikistirmiyoruz cunku kenarlar her zaman ardisik degil
(sinavin %38'inde degil).  Uzunluk 3k + 3.

A'DAN FARKI.  A yalniz `edge_types` siralamasini veriyordu; hangi
ciftin soruldugu YAZMIYORDU ve sapakli ornekte soru tanimsiz kaliyordu.
B'de soru acik.

BILINEN DURUM, olculdu ve KABUL EDILDI: sinavda gecen 215 ismin 144'u
egitimde HIC gecmiyor (egitim ve dogrulama ayni 78 ismi kullaniyor).
Verinin kendi dagilimi bu; degistirilmedi.

LISANS.  CLUTRR CC-BY-NC 4.0 (ticari DEGIL).
    Sinha, Sodhani, Dong, Pineau, Hamilton -- EMNLP 2019, arXiv 1908.06177
"""
from __future__ import annotations

import ast
import csv
import io
import os

import torch

from veri_17 import BOLME, GOREV, HAM, SORU, indir

KOK = os.path.dirname(os.path.abspath(__file__))


def _oku(bolme, gorev=GOREV):
    d = os.path.join(HAM, gorev) if gorev != GOREV else HAM
    with io.open(os.path.join(d, bolme + ".csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _cevir(e):
    """Bir satir -> (birim listesi, hedef).  `genders` sirasi varlik sirasi."""
    ad = [p.split(":")[0].strip() for p in e["genders"].split(",")]
    ke = ast.literal_eval(e["story_edges"])
    et = ast.literal_eval(e["edge_types"])
    q = ast.literal_eval(e["query_edge"])
    w = []
    for (u, v), r in zip(ke, et):
        w += [ad[u], r, ad[v]]
    return w + [SORU, ad[q[0]], ad[q[1]]], e["target_text"]


def sozluk(gorev=GOREV):
    """Birim listesi ve indeksi.  VERIDEN cikar, elle yazilmaz."""
    birim = set()
    for b in BOLME:
        for e in _oku(b, gorev):
            w, h = _cevir(e)
            birim.update(w)
            birim.add(h)
    ad = sorted(birim - {SORU}) + [SORU]
    return ad, {a: i for i, a in enumerate(ad)}


def obekler(bolme, ix, gorev=GOREV):
    """Doner: {k: (w, h)} -- w (n, 3k+3) girdi, h (n,) hedef.

    Obek yalnizca TENSOR SEKLI; `train_17` her adimda HER obekten pay alir."""
    kova = {}
    for e in _oku(bolme, gorev):
        w, h = _cevir(e)
        k = len(ast.literal_eval(e["edge_types"]))
        kova.setdefault(k, []).append(([ix[t] for t in w], ix[h]))
    return {k: (torch.tensor([a for a, _ in v]),
                torch.tensor([b for _, b in v]))
            for k, v in sorted(kova.items())}


def kur(gorev=GOREV, yaz=print):
    """CLUTRR -> (ad, ix, EG, DG, SI).  Tek cagri."""
    indir(gorev, yaz)
    ad, ix = sozluk(gorev)
    EG = obekler("train", ix, gorev)
    DG = obekler("validation", ix, gorev)
    SI = obekler("test", ix, gorev)
    for et, O in (("egitim", EG), ("dogrulama", DG), ("sinav", SI)):
        n = sum(w.shape[0] for w, _ in O.values())
        uz = [w.shape[1] for w, _ in O.values()]
        yaz("  %-10s %6d ornek   uzunluk %d..%d   k = %s"
            % (et, n, min(uz), max(uz),
               ", ".join("%d:%d" % (k, w.shape[0]) for k, (w, _) in O.items())))
    yaz("  sozluk %d birim" % len(ad))
    return ad, ix, EG, DG, SI


def iz(ad, EG, DG, SI):
    """Parmak izi -- veri kayarsa kapi yakalasin."""
    import hashlib
    h = hashlib.sha256()
    h.update(repr(ad).encode())
    for O in (EG, DG, SI):
        for k in sorted(O):
            w, t = O[k]
            h.update(w.numpy().tobytes())
            h.update(t.numpy().tobytes())
    return h.hexdigest()[:16]


if __name__ == "__main__":
    ad, ix, EG, DG, SI = kur()
    print()
    print("iz", iz(ad, EG, DG, SI))
    print()
    for et, O, k in (("EGITIM", EG, 2), ("EGITIM", EG, 3), ("SINAV", SI, 4)):
        w, t = O[k]
        print("%s k=%d   %s" % (et, k, " ".join(ad[int(x)] for x in w[0])))
        print("            => %s" % ad[int(t[0])])
