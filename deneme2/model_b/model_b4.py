# -*- coding: utf-8 -*-
"""model_b4 — GERI BESLEMELI ORTALAMA YOK.  TEK FARK: ort_bas 10000 -> 0.

Onceden kayit: belge/onkayit/model_b4.md
Kullanici karari 16 Eylul: "model_b4 agirlikli ortalama olmadan yaz."

    model_b1   veri_okul2  dar_alfa=0.5  ort_bas=10000
    model_b4   veri_okul2  dar_alfa=0.5  ort_bas=0
                                         ^ TEK FARK

--------------------------------------------------------------------------
NEDEN

`model_b1`in mimarisi DiscoLoop'un mimarisi (arXiv 2607.00341). Makale
okunarak dogrulandi; bizde olup onda olmayan tek MIMARI oge agirlik
ortalamasi:

    DiscoLoop'ta VAR   dongulu transformer + gomme darbogazi Phi
    DiscoLoop'ta YOK   tau taramasi
    DiscoLoop'ta YOK   sert / VQ surumu
    DiscoLoop'ta YOK   agirlik ortalamasi (Ek B: AdamW + Muon, lineer LR)

Soru: o ortalama gercekten bir sey katiyor mu? Kendi olcumumuz "hayir"
diyor ama DARBOGAZSIZ zeminde:

    pencere 52000-60000          ent
    model_a4   ortalama KAPALI   0.8820
    model_a5   ortalama ACIK     0.8643   k=2000
    model_a6   ortalama ACIK     0.8667   k=200

Ucu de ayni yerde, k'dan da bagimsiz -> "ayirt edilemedi".
Darbogazla birlikte OLCULMEDI.

--------------------------------------------------------------------------
NE KALDIRILIYOR -- ve ne KALDIRILMIYOR

`ort_bas=0` ortalamayi TAMAMEN kaldirmaz. Birincil okuma `pencere_b`
yine 5 anlik goruntunun AGIRLIK ORTALAMASINI aliyor -- o bizim olcme
yontemimiz, ayardan bagimsiz. Kaldirilan yalniz GERI BESLEME: esikte
ortala ve egitim ORTALANMIS agirliktan devam etsin.

Yani bu kol "ortalama var mi yok mu"yu degil:

    ORTALAMAYI EGITIMIN ICINDE MI ALALIM, SONRADAN MI

sorusunu soruyor. Kullanici gozlemi (16 Eylul): egitim sirasinda
ortalama almak, sonradan "neyin ortalamasini alalim" sorusunu ortadan
kaldiriyor -- sonradan ortalamada pencere SECILIR (genislik, nerede),
egitim icinde alinca secim yok. Bu bir TASARIM GEREKCESI; bu kosu
olcumu getiriyor (onkayit §3 ucuncul: egri/pencere orani).

--------------------------------------------------------------------------
BU BIR TEKRAR, ve bilerek

model_a4 vs model_a5 ayni soruyu darbogazsiz sordu ve "ayirt edilemedi"
dedi. Kullanici bunu kosudan ONCE soyledi: "bu deneyimizin tekrari
olur... ama kayitlara gecmesi icin yapalim." Kol KAYIT icin kosuluyor.
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

AYAR = TABAN.degistir(ad="model_b4", ort_bas=0)
#      ^ SADECE FARK. veri_okul2, dar_alfa=0.5, ood_pay=0.05, wd=0.5,
#        dar_sert=False, dar_sdpa=False hepsi model_b1'den gelir.

fark_bas = M.fark_bas          # kos.py UC sey ister: AYAR, egit, fark_bas


def egit(ayar=None, **kw):
    return M.egit(ayar or AYAR, model_kur=ModelB, **kw)


if __name__ == "__main__":
    M.fark_bas(TABAN, AYAR)
    egit(AYAR)
