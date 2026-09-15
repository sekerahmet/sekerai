# -*- coding: utf-8 -*-
"""model_a2 — SEKIZ AYRI KATMAN, dongusuz.  l 4 -> 8, dongu 2 -> 1.

Onceden kayit: belge/onkayit/model_a2.md
Kullanici karari, 15 Eylul 2026.

SORU: kazanc DERINLIKTEN mi geliyor, AGIRLIK PAYLASIMINDAN mi?

    model_a    l=4  dongu=2   3.427.840 par.   8 katman-esdegeri
    model_a2   l=8  dongu=1   6.575.616 par.   8 katman-esdegeri
               ^ AYNI HESAP, IKI KATI PARAMETRE

`model_a` ayni masayi iki kez kullaniyor; `model_a2` sekiz ayri masa
kuruyor. Ikisi de 8 katman-esdegeri hesap yapiyor. Fark yalniz
PAYLASIM.

    a2 ~= a    -> dongu sadece PARAMETRE TASARRUFU, baska bir sey degil
    a  >  a2   -> PAYLASIMIN KENDISI ise yariyor  (makalenin iddiasi)
    a2 >  a    -> derinlik/parametre kazandiriyor, paylasim BEDEL

Ucu de bilgi. Su an hangisi oldugunu BILMIYORUZ -- ve `dongu=2`yi tam da
makalenin bu iddiasina dayanarak sectik:

    2604.07822:24  "systematic generalization in the 2-hop task already
                    emerges from WEIGHT SHARING under fixed recurrence"

Yani bu kol, model_a'nin mimari secimini KENDI VERIMIZDE sinayan koldur.

--------------------------------------------------------------------------
IKI ALAN DEGISIYOR -- ve bu bilincli

ISIMLENDIRME.md "varyasyon = tek dugme" diyor; burada `l` ve `dongu`
birlikte degisiyor cunku sabit tutulan sey UCUNCU bir buyukluk:
katman-esdegeri hesap (l*dongu = 8). `model_a1`e gore ise gercekten tek
dugme: l 4 -> 8.

`fark_bas` her kosuda ikisini de EKRANA basar; "tek fark su" bir IDDIA
degil CIKTI olur.

BU DOSYA MIMARIYI YENIDEN TANIMLAMAZ: `model_a`dan import eder, yalniz
`AYAR` alanlarini degistirir. `pencere_a.py` bu kolu da olcer -- mimariyi
`ayar_t<N>.json`dan okuyor, VARSAYMIYOR.
"""
from __future__ import annotations

from model_a import AYAR as TABAN, egit, fark_bas           # noqa: F401

AYAR = TABAN.degistir(ad="model_a2", l=8, dongu=1)
#      ^ SADECE FARK. Veri/tohum/lr/wd/adim/d/dff TEKRAR YAZILMAZ.

if __name__ == "__main__":
    fark_bas(TABAN, AYAR)
    egit(AYAR)
