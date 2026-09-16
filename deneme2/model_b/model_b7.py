# -*- coding: utf-8 -*-
"""model_b7 — MAKALENIN GATE'I.  TEK FARK: dar_alfa 0.5 -> 1.0.

Onceden kayit: belge/onkayit/model_b7.md

    model_b6   veri_okul2  jeton_ad="tam"  dar_alfa=0.5
    model_b7   veri_okul2  jeton_ad="tam"  dar_alfa=1.0
                                           ^ TEK FARK

--------------------------------------------------------------------------
BU BIR KESIF DEGIL, MAKALEYE DONUS

DiscoLoop (arXiv 2607.00341) 5.1, BIREBIR:
    "we use a fixed gate alpha* = 1 and temperature tau = 1"
5.3 (uc-hop), BIREBIR:
    "We use a fixed gating alpha = 1 for DiscoLoop"

Biz B ailesinin ALTI kolunda da alpha=0.5 kostuk. Yani model_b1'in
0.8451'i de model_b6'nin 0.0354'u de MAKALENIN KONFIGURASYONUYLA
ALINMADI.

--------------------------------------------------------------------------
0.5 NEREDEN GELDI -- VE NEDEN GECERSIZ

`model_a9`un 60.000 adim EGITILMIS agirligina SONRADAN mudahale edip
alpha tarandi:

    alfa     ood      comp
    0.00   0.0013   0.9987
    0.30   0.0704   1.0000
    0.50   0.0883   1.0000     <- secilen
    1.00   0.0384   0.8361     <- comp BOZULUYOR diye elendi

Kusur: `model_a9` Phi OLMADAN egitilmisti. Hidden state'leri enjeksiyonu
BEKLEMIYORDU, o yuzden alpha=1 onlari bozdu. alpha=1 ile EGITILEN bir
modelde bu gecerli DEGIL -- model enjeksiyona ko-adapte olur, hidden
state'ler o karisimi bekleyecek sekilde sekillenir.

Bir MUDAHALE taramasindan EGITIM karari cikarmak yanlisti. Hatayi ben
yaptim; kullanici sordu ("alfa 1 denedik ama egitimde denemedik,
olgunlasmis yerde mi denedik?") ve kayitlara bakinca gorundu.

--------------------------------------------------------------------------
MIMARI DEGISMIYOR. ModelB bir satir bile degismedi.
alpha(K-1) = 0 sarti korunuyor: dongu=2'de enjeksiyon yalniz 1. loop'tan
sonra, son loop'un ciktisi dogrudan LM head'e gidiyor.
"""
from __future__ import annotations

import os
import sys

_B = os.path.dirname(os.path.abspath(__file__))
_A = os.path.join(os.path.dirname(_B), "model_a")
for _p in (_A, _B):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import model_a as M                                          # noqa: E402
assert hasattr(M, "egit"), (
    f"model_a MODUL degil PAKET olarak yuklendi: {getattr(M,'__file__',None)}")

from model_b import ModelB                                   # noqa: E402
from model_b6 import AYAR as TABAN                           # noqa: E402

AYAR = TABAN.degistir(ad="model_b7", dar_alfa=1.0)
#      ^ SADECE FARK. veri_okul2, jeton_ad="tam", ood_pay=0.05, wd=0.5,
#        ort_bas=10000, dar_tau=1.0, dar_kapi=False hepsi model_b6'dan.

fark_bas = M.fark_bas


def egit(ayar=None, **kw):
    return M.egit(ayar or AYAR, model_kur=ModelB, **kw)


if __name__ == "__main__":
    M.fark_bas(TABAN, AYAR)
    egit(AYAR)
