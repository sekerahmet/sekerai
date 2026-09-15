# -*- coding: utf-8 -*-
"""model_b3 — SERT DARBOGAZ.  TEK FARK: dar_sert False -> True.

Onceden kayit: belge/onkayit/model_b3.md
Kullanici karari 16 Eylul: "model b3 gecelim."

    model_b1   veri_okul2  dar_alfa=0.5  dar_sert=False   yumusak Phi
    model_b3   veri_okul2  dar_alfa=0.5  dar_sert=True    sert Phi
                                         ^ TEK FARK

NEDEN model_c DEGIL: aile harfi MEKANIZMA degisince degisir. `model_b`
ailesinin tanimi "dongu turlari arasinda gomme darbogazi" ve bu kol hala
tam olarak o; degisen sey `Phi`nin ICI. Dahasi bu, zaten var olan
`dar_tau` dugmesinin `tau -> 0` LIMITI.

--------------------------------------------------------------------------
BU KOL IKI SEY SORUYOR, BIRINCISI BULGU

    BIRINCIL   Darbogaz YUMUSAK DAGILIMA mi ihtiyac duyuyor, yoksa
               EN YAKIN SOZCUK yetiyor mu?

    YETIYORSA  mekanizma "kopruyu SOZCUGE YUVARLA" demek. DiscoLoop'un
               anlatisi (gurultulu surekli vektoru temiz gomme ile
               HIZALA) sadelesiyor: hizalama degil, YUVARLAMA.
    YETMIYORSA darbogazin isi BELIRSIZLIGI TASIMAK. Model 1. turda
               kopruden emin degilse dagilim o emin olmayisi 2. tura
               tasiyor; sertlestirmek onu siliyor.

    IKINCIL    Darbogazin bedeli ne kadar duser?

--------------------------------------------------------------------------
MIMARI -- tau -> 0 + straight-through (van den Oord 2017, VQ-VAE)

    ILERI:  Phi(h) = W[ argmax( nf(h) Wᵀ ) ]
    GERI:   gradyan nf(h)'ye DOGRUDAN gecer (identity)

Bu bir VEKTOR NICEMLEME: gizli durum sozluk gomme tablosuna yuvarlanir.
VQ-VAE'den farki, kod defteri ayri bir tablo DEGIL -- modelin kendi
`emb`i, ve ana kayipla birlikte egitiliyor.

--------------------------------------------------------------------------
MALIYET -- KOSUDAN ONCE OLCULDU

Gercek egitim adimi (autocast + CE + GradScaler + clip + AdamW), batch
512, dort yol SIRA DONDURULEREK, 7 tur x 25 adim, ortanca:

    yol     ms/adim   a8'e gore   naif'e gore
    a8       29,75      +0,0%       -16,6%
    naif     35,68     +19,9%        +0,0%     <- model_b1
    sdpa     39,56     +33,0%       +10,9%     <- model_b2, GERI ALINDI
    sert     31,72      +6,6%       -11,1%     <- BU KOL

Darbogazin bedeli a8'e gore %19,9'dan %6,6'ya iniyor.

--------------------------------------------------------------------------
BU KOLUN EN BUYUK KUSURU -- ve bilerek kabul edildi

TEK DUGME DEGIL. `dar_sert=True` iki seyi birden degistirir:
    1) ILERI GECIS  yumusak ortalama -> tek gomme
    2) GRADYAN      softmax uzerinden -> identity

Sonuc degisirse hangisinden geldigini AYIRAMAYIZ. Ayirma kolu (ileri
sert, geri softmax) mumkun ama HIZ KAZANCI OLMAZ -- (N,V) matrisi geri
geciste yine kurulur. "Ucuzluk" ve "temiz ayrim" bu tasarimda ayni anda
olmuyor. Karar: once ucuz olan; sonuc model_b1'den FARKLI cikarsa ayrim
kolu kosulur.

VE: VQ egitimi kirilgandir. VQ-VAE'nin caresi commitment loss; bizde YOK
(onkayit §6-2). Kod defteri cokusu de olculmuyor (§6-3).
"""
from __future__ import annotations

import sys, os

_B = os.path.dirname(os.path.abspath(__file__))
_A = os.path.join(os.path.dirname(_B), "model_a")
for _p in (_A, _B):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import model_a as M                                          # noqa: E402
assert hasattr(M, "egit"), (
    f"model_a MODUL degil PAKET olarak yuklendi: {getattr(M,'__file__',None)}")

from model_b import ModelB                                   # noqa: E402
from model_b1 import AYAR as TABAN                           # noqa: E402

AYAR = TABAN.degistir(ad="model_b3", dar_sert=True)
#      ^ SADECE FARK. veri_okul2, dar_alfa=0.5, ood_pay=0.05, wd=0.5,
#        ort_bas=10000 hepsi model_b1'den (dolayisiyla model_a8'den) gelir.

fark_bas = M.fark_bas          # kos.py UC sey ister: AYAR, egit, fark_bas


def egit(ayar=None, **kw):
    return M.egit(ayar or AYAR, model_kur=ModelB, **kw)


if __name__ == "__main__":
    M.fark_bas(TABAN, AYAR)
    egit(AYAR)
