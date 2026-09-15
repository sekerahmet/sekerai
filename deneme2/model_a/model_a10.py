# -*- coding: utf-8 -*-
"""model_a10 — WANG'IN GRAFINDA PAYLASIMSIZ kontrol.  {l, dongu}.

Onceden kayit: belge/onkayit/model_a9_a10.md  (CIFT KOL: a9 + a10)

    model_a9    l=4  dongu=2   PAYLASIMLI    8 katman-esdegeri hesap
    model_a10   l=8  dongu=1   PAYLASIMSIZ   8 katman-esdegeri hesap
                ^ IKI DUGME, ama TEK KAVRAM: PAYLASIM

NEDEN IKI DUGME
---------------
`dongu`yu tek basina 1 yapmak hesabi da YARIYA indirirdi ve sonuc
"paylasim mi yoktu, hesap mi yetmedi" diye AYRILAMAZDI. `l`yi 8'e
cikarmak hesabi esitler: iki kol da 8 katman-esdegeri.

Bu tam olarak `model_a` <-> `model_a2` ciftinin dugme yapisi ve o cift
ailede zaten olculu:
    model_a   dongu=2 paylasimli   ent 0.4947
    model_a2  l=8 dongu=1          ent 0.4220   (2x parametre, yetismedi)

Wang'in Ek E.2'si de ESIT DERINLIKTE kiyasliyor: 8 katmanli vanilla'ya
karsi ilk 4 + son 4 katmani paylasilan 8 katmanli.

BEDELI: parametre sayisi ARTAR (8 ayri blok vs 4 paylasilan blok). Yani
bu kol paylasimsiz AMA daha cok parametreli -- paylasim aleyhine degil,
LEHINE bir avantaj. Sonuc yine de a9 lehineyse iddia GUCLENIR.

Karar kurali onkayit §4'te, BAGLAYICI.
"""
from __future__ import annotations

from model_a import egit, fark_bas                           # noqa: F401
from model_a9 import AYAR as TABAN

AYAR = TABAN.degistir(ad="model_a10", l=8, dongu=1)
#      ^ SADECE FARK. veri_wang, ood_pay=0.05, wd=0.5, ortalama hepsi
#        model_a9'dan gelir.

if __name__ == "__main__":
    fark_bas(TABAN, AYAR)
    egit(AYAR)
