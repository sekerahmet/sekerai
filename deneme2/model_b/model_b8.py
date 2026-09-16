# -*- coding: utf-8 -*-
"""model_b8 — TESHIS: kopruyu ZORLA hidden state'e koy.

Taban model_b6, TEK FARK: kopru_kayip 0.0 -> 1.0.
Onceden kayit: belge/onkayit/model_b8.md

    model_b6   jeton_ad="tam"  dar_alfa=0.5  kopru_kayip=0.0
    model_b8   jeton_ad="tam"  dar_alfa=0.5  kopru_kayip=1.0
                                             ^ TEK FARK

--------------------------------------------------------------------------
BU BIR MIMARI ONERI DEGIL, TESHIS

Kopru etiketi GRAFTAN geliyor; gercek metinde boyle bir etiket YOK.
Yani bu kol "boyle bir model yapalim" demiyor. Tek sordugu:

    KOPRU hidden state'e KONULURSA bilesim duzelir mi?

    duzelir      -> kopru gercekten darbogaz. model_c / gate / head
                    calismalari ANLAMLI.
    duzelmez     -> ariza kopruden SONRAKI asamada. model_c yanlis yeri
                    onaracakti; kosulmadan kapanir.

SINAVDA HICBIR SEY ENJEKTE EDILMEZ. Model normal kosar, comp/ood her
zamanki gibi olculur. Ek sinyal YALNIZ egitimde, temsili sekillendirmek
icin.

--------------------------------------------------------------------------
NEDEN GEREKLI -- OLCULDU (16 Eylul, lineer sonda)

model_b6'nin hidden state'inden ((e,r1) ciftlerine gore AYRIK bolme):

    slot        aday   SONDA top-1   EN SIK sinif   sans
    0 (ad)       313      0.0162        0.0227      0.0032  <- EN SIK'IN ALTI
    1 (soyad)     22      0.3906        0.0843      0.0455
    2 (dolgu)      2      0.9514        0.9498      0.5000  <- <YOK>

model_b1, ayni olcum:
    0 (entity)  1517      0.2350        0.0065      0.0007  <- 36 KAT

Koprunun AYIRT EDICI parcasi (ad) model_b6'nin temsilinde YOK. Phi'yi
kac head yaparsan yap, olmayan bir seyi okuyamaz. Once "konulsa ise
yarar miydi" sorusu cevaplanmali.

--------------------------------------------------------------------------
NEREYE KONULUYOR

    [S2] e1 e2 e3 r1 r2 ?  a1 a2 a3 EOS
                  ^^ ^^              <- BURAYA. Iki sebep:
                                        1) nedensel olarak (e,r1) GORULDU,
                                           kopru orada BELIRLI
                                        2) ana kayip bu pozisyonlari
                                           KULLANMIYOR (cevap 3+yuva'dan)

Iki pozisyon oldugu icin koprunun ILK IKI token'i. Ucuncu yuva zaten
%94,7 oraninda <YOK> dolgusu -- bilgi tasimiyor.

MIMARI DEGISMIYOR. ModelB bir satir bile degismedi.
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

AYAR = TABAN.degistir(ad="model_b8", kopru_kayip=1.0)
#      ^ SADECE FARK. veri_okul2, jeton_ad="tam", dar_alfa=0.5,
#        ood_pay=0.05, wd=0.5, ort_bas=10000 hepsi model_b6'dan.

fark_bas = M.fark_bas


def egit(ayar=None, **kw):
    return M.egit(ayar or AYAR, model_kur=ModelB, **kw)


if __name__ == "__main__":
    M.fark_bas(TABAN, AYAR)
    egit(AYAR)
