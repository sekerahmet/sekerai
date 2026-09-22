# -*- coding: utf-8 -*-
"""veri_b2_17 -- CLUTRR, B2: isimler ANONIM.  Makalenin kendi yontemi.

Sinha ve ark. 2019, bolum 4.1: her varlik adi bir @entity yer tutucusuyla
degistiriliyor, yer tutucu "kucuk, sabit bir havuzdan RASTGELE" cekiliyor.
Ek 1.5: atama her hikayede yeniden rastgelelestirilmeli -- ayni sirayla
atamak modele pozisyon isaretcisi verir ve o isaretciyi kullanir.

NEDEN GEREKTI.  B1 (gercek isimler) OLCULDU ve okunamaz cikti:
    sinav k=2,3   isimlerin %100'u egitimde var    ->  0,9143
    sinav k>=4    isimlerin %19'u var, TEK ornekte
                  bile hepsi tanidik degil          ->  0,0907
Uzunluk ile isim yeniligi tam ayni yerde degisiyordu; ikisi ayrilamadi.
Anonimlestirme isim eksenini kaldirir, geriye uzunluk kalir.

PARAMETRE DONDURULMADI.  Makale sabit rastgele embedding ile raporluyor
ama Ek 1.5'te farkin kucuk, GAT icin "pratik olarak sifir" oldugunu
soyluyor; bizim girdimiz GAT tarafinda.

BICIM B1 ile ayni:  <kaynak> <iliski> <hedef> ... ? <kim> <kim>

LISANS.  CLUTRR CC-BY-NC 4.0 (ticari DEGIL).  arXiv 1908.06177
"""
from __future__ import annotations

import ast
import os
import random
import zlib

import torch

from veri_17 import BOLME, GOREV, SORU, indir
from veri_b_17 import _oku

KOK = os.path.dirname(os.path.abspath(__file__))
HAVUZ = ["@%d" % i for i in range(20)]     # kucuk sabit havuz; en buyuk
TOHUM = 0                                  # ornekte 11 varlik var


def _cevir(e, rs):
    """Bir satir -> (birim listesi, hedef).  Adlar havuzdan RASTGELE."""
    n_ad = len(e["genders"].split(","))
    ke = ast.literal_eval(e["story_edges"])
    et = ast.literal_eval(e["edge_types"])
    q = ast.literal_eval(e["query_edge"])
    ad = rs.sample(HAVUZ, n_ad)            # her ornekte yeniden
    w = []
    for (u, v), r in zip(ke, et):
        w += [ad[u], r, ad[v]]
    return w + [SORU, ad[q[0]], ad[q[1]]], e["target_text"]


def sozluk(gorev=GOREV):
    """Birim listesi ve indeksi.  VERIDEN cikar, elle yazilmaz."""
    iliski, hedef = set(), set()
    for b in BOLME:
        for e in _oku(b, gorev):
            iliski.update(ast.literal_eval(e["edge_types"]))
            hedef.add(e["target_text"])
    ad = sorted(iliski | hedef) + sorted(HAVUZ) + [SORU]
    return ad, {a: i for i, a in enumerate(ad)}


def obekler(bolme, ix, gorev=GOREV, tohum=TOHUM):
    """Doner: {k: (w, h)} -- w (n, 3k+3) girdi, h (n,) hedef.

    Tohum SABIT: ayni bolme her cagrida ayni anonim atamayi alir, yoksa
    parmak izi her kosuda kayar ve iki kosu kiyaslanamaz."""
    rs = random.Random(tohum * 1000003 + zlib.crc32(bolme.encode()))
    kova = {}
    for e in _oku(bolme, gorev):
        w, h = _cevir(e, rs)
        k = len(ast.literal_eval(e["edge_types"]))
        kova.setdefault(k, []).append(([ix[t] for t in w], ix[h]))
    return {k: (torch.tensor([a for a, _ in v]),
                torch.tensor([b for _, b in v]))
            for k, v in sorted(kova.items())}


def kur(gorev=GOREV, yaz=print, tohum=TOHUM):
    """CLUTRR -> (ad, ix, EG, DG, SI).  Tek cagri."""
    indir(gorev, yaz)
    ad, ix = sozluk(gorev)
    EG = obekler("train", ix, gorev, tohum)
    DG = obekler("validation", ix, gorev, tohum)
    SI = obekler("test", ix, gorev, tohum)
    for et, O in (("egitim", EG), ("dogrulama", DG), ("sinav", SI)):
        n = sum(w.shape[0] for w, _ in O.values())
        uz = [w.shape[1] for w, _ in O.values()]
        yaz("  %-10s %6d ornek   uzunluk %d..%d   k = %s"
            % (et, n, min(uz), max(uz),
               ", ".join("%d:%d" % (k, w.shape[0]) for k, (w, _) in O.items())))
    yaz("  sozluk %d birim   (%d yer tutucu)" % (len(ad), len(HAVUZ)))
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


def kapsama(ad, EG, SI):
    """KAPI: her yer tutucu egitimde de sinavda da geciyor mu.

    B1 tam burada battı -- sinav isimlerinin %81'i egitimde yoktu."""
    say = lambda O: set(int(x) for w, _ in O.values() for x in w.reshape(-1))
    e, s = say(EG), say(SI)
    yt = set(i for i, a in enumerate(ad) if a in HAVUZ)
    return sorted(ad[i] for i in (yt & s) - e)


if __name__ == "__main__":
    ad, ix, EG, DG, SI = kur()
    eksik = kapsama(ad, EG, SI)
    print()
    print("iz", iz(ad, EG, DG, SI))
    print("sinavda olup egitimde OLMAYAN yer tutucu:", eksik or "YOK")
    print()
    for et, O, k in (("EGITIM", EG, 2), ("EGITIM", EG, 3),
                     ("SINAV", SI, 4), ("SINAV", SI, 10)):
        w, t = O[k]
        print("%s k=%d" % (et, k))
        print("   " + " ".join(ad[int(x)] for x in w[0]))
        print("   => %s" % ad[int(t[0])])
