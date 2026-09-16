# -*- coding: utf-8 -*-
"""model_b6 — VARLIKLAR COK JETON, DOGRU KODLAMA.  TEK FARK: jeton_ad="tam".

Onceden kayit: belge/onkayit/model_b6.md
Kullanici karari 16 Eylul: "model_b6 olsun ama veri duzgun olsun bu
sefer, veriyi cok detayli kontrol et. Biz dogru veri kurup dogru
sorulari sormuyorsak zaten bastan yanlis demektir, olctugumuz de yanlis
demektir."

    model_b1   veri_okul2  dar_alfa=0.5  jeton_ad=""      TEK jeton
    model_b5   veri_okul2  dar_alfa=0.5  jeton_ad="ilk"   KUSURLU
    model_b6   veri_okul2  dar_alfa=0.5  jeton_ad="tam"   DOGRU
                                         ^ TEK FARK (model_b1'e gore)

--------------------------------------------------------------------------
NEDEN model_b5 KUSURLUYDU

"Ilk alt cizgiden bol" dedim ve "anlamli olan o" diye savundum --
KONTROL ETMEDEN. Kullanici sordu: "bazi veriler Mehmet Yilmaz bazilari
Mehmet_Yilmaz ise bence veri yanlis." Olculdu, HAKLI:

  1) AYNI DIZGE, IKI AYRI JETON -- 7 tane: Analiz, Aydin, Cebir, Dogan,
     Fizik, Hukuk, Kimya. Aydin(sehir) bir id, Aydin(soyad) BASKA id.
  2) BOLUNMEMIS BILESIKLER: Lisesi | Fen_Lisesi | Anadolu_Lisesi
     -> uc ayri, ilgisiz jeton.
  3) Ayni tip icinde uzunluk tutarsiz.

--------------------------------------------------------------------------
DOGRUSU -- butun alt cizgiler, TEK PAYLASILAN sozluk, 3 yuva

    Ayse_Yilmaz       -> (Ayse,    Yilmaz, <YOK>)
    Adana_Fen_Lisesi  -> (Adana,   Fen,    Lisesi)
    Adana             -> (Adana,   <YOK>,  <YOK>)
    Nukleer_Fizik     -> (Nukleer, Fizik,  <YOK>)

    2120 varlik -> 441 jeton (440 parca + <YOK>), vocab 466
    `Adana` sehirde de okulda da AYNI jeton
    `Lisesi` butun okullarda ORTAK
    `Fizik` tek basina da Nukleer_Fizik'te de AYNI

VERI DENETIMI kosudan ONCE yapildi, 8 kontrol + bolme gecerliligi +
kisayol olcumu: onkayit model_b6.md §2. Ozet:

    comp (UCLU duzeyi)   GECERLI, 0 sizinti
    ood  (KENAR duzeyi)  GECERLI, 0 sizinti   <- BIRINCIL OLCU
    ent  (VARLIK duzeyi) varlik duzeyinde gecerli ama JETON duzeyinde
                         ZAYIF: 293/312'sinde butun jetonlar bas
                         konumunda gecmis -> "gorulmemis VARLIK" degil
                         "gorulmemis BILESIM" olcuyor
    DONUS kisayolu       sinav bolmelerinde 0 (seen'de %4,8)
    jeton kopyalama      baba/anne %32, kardes %16, okul %0 -- ama
                         UC JETON DA dogru olmali, tek basina gecmez

--------------------------------------------------------------------------
MIMARI DEGISMIYOR. ModelB bir satir bile degismedi.
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

AYAR = TABAN.degistir(ad="model_b6", jeton_ad="tam")
#      ^ SADECE FARK. veri_okul2, dar_alfa=0.5, ood_pay=0.05, wd=0.5,
#        ort_bas=10000 hepsi model_b1'den gelir.

fark_bas = M.fark_bas


def egit(ayar=None, **kw):
    return M.egit(ayar or AYAR, model_kur=ModelB, **kw)


if __name__ == "__main__":
    M.fark_bas(TABAN, AYAR)
    egit(AYAR)
