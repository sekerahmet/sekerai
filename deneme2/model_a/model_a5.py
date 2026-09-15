# -*- coding: utf-8 -*-
"""model_a5 — GERI BESLEMELI AGIRLIK ORTALAMASI.  TEK FARK: ort_bas 0 -> 10000.

Onceden kayit: belge/onkayit/model_a5.md
Kullanici fikri ve karari, 15 Eylul 2026:

    "Belirli bir adim calistiktan sonra her esikte model bir onceki model
     ile ortalama alacak, oyle devam edecek."

Yani GOLGE ortalama degil, GERI BESLEMELI: esikte ortala ve egitim
ORTALANMIS agirliktan devam etsin.

    model_a4   wd=0.5  veri_okul2  ort_bas=0       ortalama KAPALI
    model_a5   wd=0.5  veri_okul2  ort_bas=10000   10.000'den sonra ACIK
               ^ TEK FARK

TABAN `model_a4` -- kullanici "model_a4 standartlarinda" dedi, yani
2x veri zemininde.

--------------------------------------------------------------------------
BUNUN ADI VAR: LOOKAHEAD

1907.08610 (Zhang, Lucas, Ba, Hinton -- NeurIPS 2019):

    theta, k adim ic optimizatorle (bizde AdamW) egitilir
    phi   <- phi + alfa * (theta - phi)
    theta <- phi                     (hizli agirlik SIFIRLANIR)

`ort_alfa = 0.5` -> tam olarak "bir onceki modelle esit ortalama".

TEK FARK OLCEK VE BU ONEMLI: makale k = 5..10 adim tariyor. Burada
k = `olc_her` = 2000, yani 200-400 KATI. Bu, Lookahead'in taradigi bolge
DEGIL; SWA'ya yakin ama SWA geri besleme yapmaz. Kol, iki yontemin
arasinda olculmemis bir noktada duruyor.

RISK ACIK: alfa=0.5 ile 2000 adimda bir ortalamak ogrenmeyi sonumleyebilir.
Sonumleme cikarsa once `ort_her` sorgulanmali (kucultulur), fikir degil.

--------------------------------------------------------------------------
NEDEN BU PROJEDE ANLAMLI

Agirlik ortalamasinin bu gorevdeki kazanci OLCULDU ve buyuk -- ve `wd`
ile buyuyor:

    en iyi TEK anlik goruntu -> SON pencere (agirlik ortalamasi)
      model_a1  0.2290 -> 0.3473   1,52x
      model_a2  0.2693 -> 0.4220   1,57x
      model_a   0.2767 -> 0.4947   1,79x
      model_a3  0.3830 -> 0.8743   2,28x

Su ana kadar bu kazanc KOSUDAN SONRA alindi. Soru: egitimin ICINE
konursa birikir mi?

BU DOSYA MIMARIYI YENIDEN TANIMLAMAZ: `model_a4`ten import eder, yalniz
`AYAR`in tek alanini degistirir.
"""
from __future__ import annotations

from model_a import egit, fark_bas                           # noqa: F401
from model_a4 import AYAR as TABAN

AYAR = TABAN.degistir(ad="model_a5", ort_bas=10000)
#      ^ SADECE FARK. wd=0.5 ve veri_okul2 model_a4'ten gelir.
#        `ort_her` (0 -> olc_her=2000) ve `ort_alfa` (0.5) varsayilanda;
#        kullanici "sonra neye gore agirlikli ortalama olmasina karar
#        veririz" dedi -- tarama icin hazir duruyorlar.

if __name__ == "__main__":
    fark_bas(TABAN, AYAR)
    egit(AYAR)
