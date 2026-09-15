# -*- coding: utf-8 -*-
"""model_a8 — WANG'IN test_inferred_OOD'si.  TEK FARK: ood_pay 0.0 -> 0.05.

Onceden kayit: belge/onkayit/model_a8.md

    model_a5   ood_pay=0.00
    model_a8   ood_pay=0.05     <- Wang'in 95:5 orani BIREBIR

TABAN `model_a5`. `model_a7` DEGIL -- `kati_pay` ayri bir soru ve ikisini
ayni kosuda karistirmak tek dugme ilkesini bozar.

--------------------------------------------------------------------------
NEDEN VAR: model_a7'nin DUZELTMESI

`model_a7`yi "Wang'in gercek OOD'si" diye kurdum. YANLISTI, cunku
makaleyi kurmadan ONCE okumamistim. 15 Eylul'de bastan sona okundu
(§1-§7 + Ek A-F):

    §3.1  "The atomic facts are then the EDGES ... which we partition
           disjointly into atomic_ID and atomic_OOD (95%:5%)"
           -> bolme VARLIK degil KENAR uzerinden

    §2    "Our training set includes ALL the atomic facts and a uniformly
           random portion of the inferred facts deduced from atomic_ID"

    §3.3  "the model does not have any incentive to store atomic facts in
           the upper layers THAT DO NOT APPEAR AS THE SECOND HOP during
           training ... the OOD atomic facts are simply not stored in the
           upper layers when queried during the second hop"

OLCULDU (model_a7):
    ent_kati 2. HOP kenari egitimde 2. HOP olarak gecmis: 4094/4370 (%93,7)

Yani model_a7 makalenin %0'ini ureten KOSULU SAGLAMIYORDU. Olctugu sey
gercek ("zincire hic girmemis varlik") ama Wang'in OOD'si DEGIL.

--------------------------------------------------------------------------
BOLMENIN TANIMI -- makaleden birebir

    EGITIM   iki kenari da ID olan zincirler    (train_inferred_ID)
    SINAV    iki kenari da OOD olan zincirler   (test_inferred_OOD)
    KARISIK  bir kenari OOD -> NE EGITIM NE SINAV, DUSER
    ATOMIK   HEPSI egitimde -- OOD kenarlari tek olgu olarak OGRETILIR

`veri_kur` her kosuda DORT assert ile denetliyor:
    OOD kenari egitim zincirinde herhangi bir hop'ta      0
    sinav zincirinin iki kenari da OOD                    226/226
    2. HOP kenari egitimde 2. HOP olarak gecen            0     <<< ASIL
    OOD atomik olgulari egitimde duruyor                  840/840

Ve Wang'in ozelligi: BAS varlik egitimde BASKA zincirlerde bas OLMUS
(225/226, %100). Tutulan sey VARLIK degil KENAR.

    tr2 85.440 -> 76.695   phi 5.09 -> 4.57  (Wang tanimiyla 6.36 -> 5.71)
    OOD SINAV: 226 zincir

EN BUYUK ZAYIFLIK: 226 ornek. Binom SE p=0.5'te 0.033 (%95: +-0.065).
ONKAYIT §4 BAGLAYICI: sonuc 0.40-0.60 bandina duserse HUKUM VERILMEZ,
ikinci tohum kosulur. Havuzu buyutmek icin ood_pay=0.10 yapilabilirdi
(~900 ornek) ama o Wang'in oranindan sapmak demek -- bu kolun TEK AMACI
orani birebir tutmak. Gurultu tohumla cozulur, tanimi bozarak degil.

BU DOSYA MIMARIYI YENIDEN TANIMLAMAZ: `model_a5`ten import eder.
"""
from __future__ import annotations

from model_a import egit, fark_bas                           # noqa: F401
from model_a5 import AYAR as TABAN

AYAR = TABAN.degistir(ad="model_a8", ood_pay=0.05)
#      ^ SADECE FARK. wd=0.5, veri_okul2, ort_bas=10000, ort_her=0(->2000),
#        ort_alfa=0.5 hepsi model_a5'ten gelir.

if __name__ == "__main__":
    fark_bas(TABAN, AYAR)
    egit(AYAR)
