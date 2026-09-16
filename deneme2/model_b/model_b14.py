# -*- coding: utf-8 -*-
"""model_b14 — VERI YOGUNLUGU (phi 4,57 -> ~6,2).

Onceden kayit: belge/onkayit/model_b14.md
Taban model_b13.  TEK DEGISIKLIK:

    model_b13   veri_ad = "veri_okul2"    phi TAVANI 7,04
    model_b14   veri_ad = "veri_okul3"    phi TAVANI 9,57

wd=0.1 + cosine (CLAUDE.md kural 4) ve ident_frac=0.2 model_b13'ten
DEVRALINIR. Yani iki kol arasindaki tek fark GRAF.

--------------------------------------------------------------------------
NEDEN -- model_b13 mazereti kaldirdi, literatur yeri gosterdi

model_b13'te egitim dagilimi ILK KEZ doydu:

    one 1.0000   seen 0.9993   kayip_ana 1.8543 -> 0.0016
    comp 0.0577

Ve `tani_b` dort bolmede de ayni seyi soyledi: 1. hop TEK BASINA
1.0000, 2. hop TEK BASINA 1.0000, ikisi birden 0.06. Yani "model
ogrenemedi" aciklamasi KAPANDI.

Literatur taramasi (16 Eylul) tek bir yere isaret etti -- VERI
YOGUNLUGU:

    Wang ve ark. (2024)            grokking araligi   phi  7 - 18
    arXiv 2505.17923  2-hop large                     phi 10,0  *
    arXiv 2505.17923  2-hop small                     phi 16,0  *
    BIZ (veri_okul2)                                  phi  4,57

    * BEN hesapladim, onlar basmiyor. |E|=500 |R|=20 -> 10.000 olgu,
      "all possible reasoning questions"in %50'si -> 100.000 soru.

Iki bagimsiz kaynak, ayni yer. Ve 2505.17923 bizim ASAMA-1 probe'umuzu
da yapmis, AYNI pozisyonda (son girdi token'i) -- onlarda kopru ORADA:
"the hidden representation of the last input token encodes information
about all necessary bridge entities". Bizde ayni pozisyonda 0.0275.

--------------------------------------------------------------------------
NEDEN YENI ILISKI SEMBOLU YOK -- arama uzayi tuzagi

Ilk tasarim 17 YENI sembol ekliyordu (phi tavani 13,07). arXiv
2505.17923 veri acliginin asil kaynagini ayristirmis:

  "the search space (i.e. |R|^k relation combinations per entity)...
   Fixing 1-hop and 2-hop relations reduces the required training
   budget to x1, while increasing them leads to rapid budget growth."

Yeni sembol |R|'yi 17 -> 34, arama uzayini 289 -> 1156 yapardi.
`veri_okul3` |R|'ye DOKUNMUYOR: var olan 17 sembolu baska TIPLERDE de
gecerli kiliyor (`okul`/`sehir`/`ders` zaten cok alanliydi, o kalip
yayildi).

    kardes + OKUL/SEHIR    komsu + KISI/OKUL    rakip + KISI/SEHIR
    kurucu + SEHIR/DERS    hoca  + OKUL         okul  + DERS
    sehir  + DERS          ders  + SEHIR

    tip basina iliski   KISI  OKUL  SEHIR  DERS
    veri_okul2            10     5      3     2
    veri_okul3            12     8      7     5

--------------------------------------------------------------------------
CONFOUND -- phi TEK BASINA DONMUYOR

Semayi yogunlastirinca OLGU SAYISI da buyuyor: 16.800 -> 21.920
(+%30). Yani ezberlenecek atomik olgu da artiyor ve `one` zorlasiyor.
Kacinilmaz: bir iliskiyi yeni bir tipe acmak o tipteki her varliga
yeni bir olgu ekler. (Ilk tasarimda bu +%78 idi.)

--------------------------------------------------------------------------
!! OLCME IZI DEGISIR -- YENI MERDIVEN

Veri degisince `olcme_izi` (57cf60a5e9af) degisir. Bu kol model_b6..b13
ile AYNI TABLODA okunamaz. Kontrolu `model_b13`tur ve o zaten kosuldu;
kiyas "ayni ayar, FARKLI graf" olarak yapilir, ayni sinav kumesi
uzerinden DEGIL.

MIMARI: bir satir bile degismedi. Ek parametre YOK.
"""
from __future__ import annotations

import os
import sys

_B = os.path.dirname(os.path.abspath(__file__))
_A = os.path.join(os.path.dirname(_B), "model_a")
_K = os.path.dirname(_B)
for _p in (_A, _B, _K):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import model_a as M                                          # noqa: E402
assert hasattr(M, "egit"), (
    f"model_a MODUL degil PAKET olarak yuklendi: {getattr(M,'__file__',None)}")

from model_b import ModelB                                   # noqa: E402
from model_b13 import AYAR as TABAN                          # noqa: E402

AYAR = TABAN.degistir(ad="model_b14", veri_ad="veri_okul3")
#      ^ TEK DUGME. ident_frac=0.2, ident_kip="q1", wd=0.1,
#        sabit_lr=False model_b13'ten DEVRALINIR (CLAUDE.md kural 4).

fark_bas = M.fark_bas


def egit(ayar=None, **kw):
    return M.egit(ayar or AYAR, model_kur=ModelB, **kw)


if __name__ == "__main__":
    M.fark_bas(TABAN, AYAR)
    egit(AYAR)
