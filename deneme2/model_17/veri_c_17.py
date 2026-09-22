# -*- coding: utf-8 -*-
"""veri_c_17 -- CLUTRR, C hali: HIKAYE METNI girdi.

Makalenin metin modelleri (BiLSTM, RN, MAC, BERT) bunu aliyor.
A ve B'de olgular hazir veriliyordu; C'de model once DILDEN olguyu
cikaracak, sonra bilesimi yapacak.  Makalenin kendi sorusu bu.

    @13 is the proud father of the lovely @12 @1 's brother @13 and her
    went to get ice cream . ? @12 @1                          =>  aunt

ISIMLER ANONIM -- B2'deki gibi, makalenin yontemi (Sinha ve ark. 2019,
bolum 4.1): havuzdan rastgele, her ornekte yeniden.

OLCULDU, C'yi A ve B'den ayiran uc sey:
    sozluk       2.568 kelime   (B2: 41)
    uzunluk      egitim 11..133, sinav 15..190   (B2: 9..12 / 9..33)
    yeni kelime  sinavda 305 kelime egitimde YOK  (jeton olarak %2,6)
Son kalem kasitli: makale egitim ve sinavda FARKLI cumle kaliplari
kullaniyor.  Metin modellerinin asil zorlandigi yer orasi.

DOLGU VAR, A ve B'de YOKTU.  Sebep olculdu: uzunluk 88 farkli deger
aliyor (B2'de 2).  Obek basina tek gecis yapmak adim basina ~6.000
sirali bmm demek; dolgu + maske ile ~10 obege iniyor.

LISANS.  CLUTRR CC-BY-NC 4.0 (ticari DEGIL).  arXiv 1908.06177
"""
from __future__ import annotations

import ast
import collections
import os
import random
import zlib
import re

import torch

from veri_17 import BOLME, GOREV, SORU, indir
from veri_b_17 import _oku
from veri_b2_17 import HAVUZ, TOHUM

KOK = os.path.dirname(os.path.abspath(__file__))
PAD = "<dolgu>"
BASAMAK = 16          # uzunluk bir ust katina yuvarlanir -> obek sayisi duser
JETON = re.compile(r"@\d+|[a-zA-Z']+|[.,!?]")


def _cevir(e, rs):
    """clean_story -> anonim kelime dizisi + soru.  Doner: (kelimeler, hedef)."""
    ad = [p.split(":")[0].strip() for p in e["genders"].split(",")]
    q = ast.literal_eval(e["query_edge"])
    yt = dict(zip(ad, rs.sample(HAVUZ, len(ad))))
    s = e["clean_story"]
    for a, t in yt.items():
        s = s.replace("[" + a + "]", " " + t + " ")
    return (JETON.findall(s) + [SORU, yt[ad[q[0]]], yt[ad[q[1]]]],
            e["target_text"])


def _rs(bolme, tohum):
    return random.Random(tohum * 1000003 + zlib.crc32((bolme + "C").encode()))


def sozluk(gorev=GOREV, tohum=TOHUM):
    """Birim listesi ve indeksi.  VERIDEN cikar, elle yazilmaz.

    PAD indeks 0'da: maske ile duser ama yine de sabit bir yeri olsun."""
    birim = set()
    for b in BOLME:
        rs = _rs(b, tohum)
        for e in _oku(b, gorev):
            w, h = _cevir(e, rs)
            birim.update(w)
            birim.add(h)
    ad = [PAD] + sorted(birim - {SORU}) + [SORU]
    return ad, {a: i for i, a in enumerate(ad)}


def obekler(bolme, ix, gorev=GOREV, tohum=TOHUM, basamak=BASAMAK):
    """Doner: {uzunluk: (w, h, maske)}.

    w (n, T) dolgulu girdi, h (n,) hedef, maske (n, T) bool -- gercek
    jetonlarda True.  Dolgu SONDA; maske olmadan dikkat onlari da
    toplar ve her ornekte baska sayida toplar."""
    rs = _rs(bolme, tohum)
    kova = collections.defaultdict(list)
    for e in _oku(bolme, gorev):
        w, h = _cevir(e, rs)
        T = -(-len(w) // basamak) * basamak          # ust kata yuvarla
        kova[T].append(([ix[t] for t in w], ix[h]))
    cik = {}
    for T, v in sorted(kova.items()):
        w = torch.zeros(len(v), T, dtype=torch.long)
        m = torch.zeros(len(v), T, dtype=torch.bool)
        h = torch.tensor([b for _, b in v])
        for i, (a, _) in enumerate(v):
            w[i, :len(a)] = torch.tensor(a)
            m[i, :len(a)] = True
        cik[T] = (w, h, m)
    return cik


def kur(gorev=GOREV, yaz=print, tohum=TOHUM, basamak=BASAMAK):
    """CLUTRR -> (ad, ix, EG, DG, SI).  Tek cagri.

    basamak buyurse obek sayisi duser.  OLCULDU:
        16   9 obek,  sirali derinlik 720,  dolgu %17,4
       144   1 obek,  sirali derinlik 144,  dolgu %75,2
    Sirali derinlik adim suresini belirliyor: gez dongusu birbirine bagli,
    GPU paralellestiremiyor.  Dolgunun FLOP israfi bunun yaninda kucuk
    (model_16'da olculmustu: 27 kucuk gecis -> 1 buyuk gecis, 4,2 kat)."""
    indir(gorev, yaz)
    ad, ix = sozluk(gorev, tohum)
    EG = obekler("train", ix, gorev, tohum, basamak)
    DG = obekler("validation", ix, gorev, tohum, basamak)
    SI = obekler("test", ix, gorev, tohum, basamak)
    for et, O in (("egitim", EG), ("dogrulama", DG), ("sinav", SI)):
        n = sum(w.shape[0] for w, _, _ in O.values())
        dol = sum(int((~m).sum()) for _, _, m in O.values())
        top = sum(m.numel() for _, _, m in O.values())
        yaz("  %-10s %6d ornek   %2d obek   T = %d..%d   dolgu %%%.1f"
            % (et, n, len(O), min(O), max(O), 100 * dol / top))
    yaz("  sozluk %d birim   (%d yer tutucu + dolgu)" % (len(ad), len(HAVUZ)))
    return ad, ix, EG, DG, SI


def iz(ad, EG, DG, SI):
    """Parmak izi -- veri kayarsa kapi yakalasin."""
    import hashlib
    h = hashlib.sha256()
    h.update(repr(ad).encode())
    for O in (EG, DG, SI):
        for k in sorted(O):
            for t in O[k]:
                h.update(t.numpy().tobytes())
    return h.hexdigest()[:16]


def kapsama(ad, EG, SI):
    """Sinavda gecip egitimde HIC gecmeyen kelimeler.

    C'de bu BOS DEGIL ve olmasi gerekiyor: makale egitim ile sinavda
    farkli cumle kaliplari kullaniyor.  Yer tutucular ise bos olmali."""
    gor = lambda O: set(int(x) for w, _, m in O.values()
                        for x in w[m].reshape(-1))
    yok = gor(SI) - gor(EG)
    return sorted(ad[i] for i in yok)


if __name__ == "__main__":
    ad, ix, EG, DG, SI = kur()
    yok = kapsama(ad, EG, SI)
    yt = [a for a in yok if a in HAVUZ]
    print()
    print("iz", iz(ad, EG, DG, SI))
    print("sinavda olup egitimde olmayan: %d kelime" % len(yok))
    print("   bunlarin YER TUTUCU olani: %s" % (yt or "YOK  <- kapi GECTI"))
    print("   ornek: " + " ".join(yok[:14]))
    print()
    for et, O in (("EGITIM", EG), ("SINAV", SI)):
        T = min(O)
        w, h, m = O[T]
        print("%s  obek T=%d" % (et, T))
        print("   " + " ".join(ad[int(x)] for x in w[0][m[0]]))
        print("   => %s   (dolgu %d)" % (ad[int(h[0])], int((~m[0]).sum())))
