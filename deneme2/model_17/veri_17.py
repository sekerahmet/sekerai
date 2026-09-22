# -*- coding: utf-8 -*-
"""veri_17 -- CLUTRR, SIMGESEL hal.  Iliski zinciri -> bilesik iliski.

Kullanici, 22 Eylul: *"model 17'de veri kismini kaldiracagiz, CLUTRR
kullanacagiz"* ve *"once A yapalim"* -- A = YALNIZ ZINCIR, isimsiz.

NE SORUYOR.  Bir akrabalik zinciri veriliyor, bilesigi soruluyor:

    father -> sister            =>  aunt
    mother -> brother -> father -> son -> brother   =>  uncle

Mimarinin iddiasi tam bu: iliski bir MATRIS, bilesim matris carpimi.
    s = M[r_k] ... M[r_2] M[r_1] s0     ->  en yakin E[etiket]

NEDEN ISIMSIZ (A).  OLCULDU: 1.393 farkli zincirin 1.385'i TEK bir
cevaba gidiyor; yani cevap yalniz zincirden cikiyor, isimler bilgi
tasimiyor.  Belirsiz kalan 8 zincir `mother / mother-in-law` ayrimi.
Isimleri koymak (B, 243 birim) ogrenecek gurultu eklerdi.

NEDEN DOGAL DIL DEGIL.  CLUTRR'in hikayeleri MTurk'te insanlara
yazdirilmis ve oyle birakilmis.  Olculdu (9.074 egitim hikayesi):
    ayrik kisaltma " n't "        %11,9
    son cumlede nokta yok          %4,1
    iki cumle arasi nokta yok      %1,8
    bozuk iyelik  "] 'father"      %1,4
Bu gurultu CLUTRR'in KENDI sorusunun parcasi (gurultulu metinden
iliskiyi cikarabiliyor mu).  Bizim sorumuz bilesim; simgesel hal onu
sorar, gurultuyu disarida birakir.

BOLMELER VERIDE HAZIR -- biz kurmuyoruz:
    egitim      k = 2,3     9.074      model BUNLARI gorur
    dogrulama   k = 2,3     2.020      ayni k, farkli ornek -> EZBER
    sinav       k = 2..10   1.146      k >= 4 HIC GORULMEDI -> CIKARIM
Ve sinavdaki zincirlerin %96,7'si egitimde hic gecmiyor; k >= 3'te %100.

LISANS.  CLUTRR CC-BY-NC 4.0 (ticari DEGIL).  Kaynak:
    Sinha, Sodhani, Dong, Pineau, Hamilton -- EMNLP 2019, arXiv 1908.06177
    veri: github.com/kliang5/CLUTRR_huggingface_dataset
"""
from __future__ import annotations

import ast
import csv
import io
import os
import urllib.request

import numpy as np
import torch

KOK = os.path.dirname(os.path.abspath(__file__))
HAM = os.path.join(KOK, "veri_ham")
URL = ("https://raw.githubusercontent.com/kliang5/"
       "CLUTRR_huggingface_dataset/main/")
GOREV = "gen_train23_test2to10"      # egitim k=2,3  sinav k=2..10
BOLME = ("train", "validation", "test")
SORU = "?"                            # zincirin sonu -- "bilesigi ne"


def indir(gorev: str = GOREV, yaz=print) -> None:
    """CSV'leri BIR KEZ indir.  Varsa dokunma (kural 9)."""
    d = os.path.join(HAM, gorev) if gorev != GOREV else HAM
    os.makedirs(d, exist_ok=True)
    for b in BOLME:
        p = os.path.join(d, b + ".csv")
        if os.path.exists(p):
            continue
        yaz(f"  indiriliyor {gorev}/{b}.csv")
        urllib.request.urlretrieve(URL + gorev + "/" + b + ".csv", p)


def _oku(bolme: str, gorev: str = GOREV) -> list[dict]:
    d = os.path.join(HAM, gorev) if gorev != GOREV else HAM
    with io.open(os.path.join(d, bolme + ".csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sozluk(gorev: str = GOREV) -> tuple[list[str], dict]:
    """Birim listesi ve indeksi.  VERIDEN cikar, elle yazilmaz.

    Kenar turleri (14) ile hedef etiketleri (18) 12'sinde ortak;
    `husband`/`wife` yalniz kenarda, `nephew`/`niece` ve dort `-in-law`
    yalniz hedefte.  Birlesim + soru imi = 21 birim."""
    kenar, hedef = set(), set()
    for b in BOLME:
        for e in _oku(b, gorev):
            kenar.update(ast.literal_eval(e["edge_types"]))
            hedef.add(e["target_text"])
    ad = sorted(kenar | hedef) + [SORU]
    return ad, {a: i for i, a in enumerate(ad)}


def obekler(bolme: str, ix: dict, gorev: str = GOREV):
    """Bir bolmeyi UZUNLUGA GORE obeklenmis tensorlere cevir.

    Doner: {k: (w, h)}  --  w (n, k+1) girdi, h (n,) hedef.
    k degisken (2..10) ve tensor dikdortgen olmak zorunda; obek
    yalnizca TENSOR SEKLI, mufredat degil.  `train_17` her adimda
    HER obekten pay alir (kos_15'te olculdu: tek obek secmek kucuk
    obegi acliktan olduruyor)."""
    kova: dict[int, list] = {}
    for e in _oku(bolme, gorev):
        et = ast.literal_eval(e["edge_types"])
        kova.setdefault(len(et), []).append(
            ([ix[r] for r in et] + [ix[SORU]], ix[e["target_text"]]))
    return {k: (torch.tensor([w for w, _ in v]),
                torch.tensor([h for _, h in v]))
            for k, v in sorted(kova.items())}


def kur(gorev: str = GOREV, yaz=print):
    """CLUTRR -> (ad, ix, EG, DG, SI).  Tek cagri, hepsi burada."""
    indir(gorev, yaz)
    ad, ix = sozluk(gorev)
    EG = obekler("train", ix, gorev)
    DG = obekler("validation", ix, gorev)
    SI = obekler("test", ix, gorev)
    for et, O in (("egitim", EG), ("dogrulama", DG), ("sinav", SI)):
        n = sum(w.shape[0] for w, _ in O.values())
        yaz(f"  {et:10} {n:>6} ornek   k = "
            + ", ".join(f"{k}:{w.shape[0]}" for k, (w, _) in O.items()))
    yaz(f"  sozluk {len(ad)} birim   " + " ".join(ad))
    return ad, ix, EG, DG, SI


def iz(ad, EG, DG, SI) -> str:
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
    print("ORNEK  (egitim k=3)")
    w, t = EG[3]
    for i in range(3):
        print("   " + " ".join(ad[int(x)] for x in w[i])
              + "   =>   " + ad[int(t[i])])
    print()
    print("ORNEK  (sinav k=10 -- egitimde HIC gorulmedi)")
    w, t = SI[10]
    for i in range(2):
        print("   " + " ".join(ad[int(x)] for x in w[i])
              + "\n        =>   " + ad[int(t[i])])
