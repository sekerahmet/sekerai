# -*- coding: utf-8 -*-
"""model_a7 — WANG'IN GERCEK OOD'si.  TEK FARK: kati_pay 0.0 -> 0.05.

Onceden kayit: belge/onkayit/model_a7.md

    model_a5   ort(k=2000)  kati_pay=0.00
    model_a7   ort(k=2000)  kati_pay=0.05
                            ^ TEK FARK

TABAN `model_a5` -- ailenin en iyi TEK anlik goruntusunu o veriyor
(0.8177; ortalamasiz `model_a4` 0.5020). Sinav zorlasirken modelin en
iyi halinde olmasi isteniyor. `model_a6` DEGIL, cunku `k` ekseni kapandi
(a6 - a5 = +0.0024, bant +-0.10).

--------------------------------------------------------------------------
NEDEN

`ent`te 0.86-0.88'e ulastik ve bunu "kompozisyonu cozduk" diye okumaya
yaklastik. Ama `ent` BIZIM tanimimiz, literaturun degil.

Wang, Yue, Su, Sun (arXiv 2405.15071, NeurIPS 2024) ayni kurulumu olcuyor
-- `wang_phi` oradan geliyor -- ve kompozisyonda OOD dogrulugu %0 buluyor,
22 MILYON adimda bile. Makale acilip OOD tanimi DOGRULANDI (15 Eylul):

    Wang OOD   varlik HICBIR egitim zincirinde gecmiyor: ne bas, ne
               kopru, ne cevap. Yalniz ATOMIK olgularda var.
               (atomicID : atomicOOD = 95 : 5)

    bizim ENT  varlik yalniz BAS olmuyor. Kopru ve cevap olarak
               egitimde BOL BOL geciyor.

OLCULDU: veri_okul2'de tipik bir ENT varligi (Gaziantep) egitim
zincirlerinde 52 kez KOPRU, 131 kez CEVAP. Yani `ent` Wang'in OOD'sinden
KESINLIKLE daha kolay ve 0.88'i %0'in karsisina koymak YANLIS olur.

--------------------------------------------------------------------------
UC GRUP, TEK KOSU

    normal       bas+kopru+cevap  hepsi var   -> comp
    ENT   %20    kopru ve cevap var, BAS YOK  -> ent, ent_yok
    KATI  %5     HICBIR ROLDE yok             -> ent_kati   <-- YENI

Ucu de AYNI modelde, AYNI adimda, AYNI phi'de olculuyor. Kiyas KOSU ICI,
bu yuzden phi'nin model_a5 ile eslesmesi GEREKMIYOR ve silinen zincirlerin
yerine dolgu KOYULMUYOR.

    OLCULDU: kati_pay=0.05'te 6.390 zincir silinir (%7,9). Yerine
    konabilecek havuz AYNI/DONUS turunden, yani TRIVIAL zincirler --
    dolgu egitim dagilimini KOLAYLASTIRIRDI.

BEDELI ACIK: phi 5.09 -> 4.45 (Wang tanimiyla 6.36 -> 5.86). Makalede phi
grokking hizini belirliyor. Bu yuzden model_a7'nin `ent` degeri
model_a5'inkiyle DOGRUDAN KIYASLANMAZ.

--------------------------------------------------------------------------
ATOMIK OLGULAR UC GRUPTA DA DURUYOR -- Wang da atomicOOD'yi egitimde
tutuyor. Sinav "olgulari biliyor, zincir kurabiliyor mu".

BU DOSYA MIMARIYI YENIDEN TANIMLAMAZ: `model_a5`ten import eder, yalniz
`AYAR`in tek alanini degistirir.
"""
from __future__ import annotations

from model_a import egit, fark_bas                           # noqa: F401
from model_a5 import AYAR as TABAN

AYAR = TABAN.degistir(ad="model_a7", kati_pay=0.05)
#      ^ SADECE FARK. wd=0.5, veri_okul2, ort_bas=10000, ort_her=0(->2000),
#        ort_alfa=0.5 hepsi model_a5'ten gelir.

if __name__ == "__main__":
    fark_bas(TABAN, AYAR)
    egit(AYAR)
