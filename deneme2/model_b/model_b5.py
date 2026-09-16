# -*- coding: utf-8 -*-
"""model_b5 — VARLIKLAR IKI JETON.  TEK FARK: jeton_ad False -> True.

Onceden kayit: belge/onkayit/model_b5.md
Kullanici karari 16 Eylul: "model_b5 olsun, sadece veri degissin ve
bakalim." Ve: "ben kendim soru sorup cevabini alacagim... 'Ayse Yilmaz
baba' diye yazdigimda 'Mehmet Yilmaz' gibi bir cevap verdigini gormek
istiyorum."  -> `sor.py`

    model_b1   veri_okul2  dar_alfa=0.5  jeton_ad=False
    model_b5   veri_okul2  dar_alfa=0.5  jeton_ad=True
                                         ^ TEK FARK

--------------------------------------------------------------------------
NEDEN -- sembolik rejim ile dil arasindaki KOPRU

`model_b1` bu grafta gorevi bitirdi: ent 1.000, comp 0.999, birlestirme
kaybi +0.0000 (model_b1.md §9). Ama BUTUN VARLIKLAR TEK JETON.

Gercek dilde degiller. Ve DiscoLoop makalesi bu boslugu DOLDURMUYOR --
okunarak dogrulandi: "The paper does not address multi-token entities.
The symbolic and synthetic experiments use single-token entities."

    BU KOLUN SORUSU: kopru varlik IKI JETONSA darbogaz hayatta kaliyor mu?

Darbogaz isleci gizli durumu TEK bir gommeye yuvarliyor. Kopru hic
yazilmiyor -- model onu `?` pozisyonundaki gizli durumda tutuyor. Iki
jetonluysa, tek gizli durum iki jetonu birden temsil etmeli ama isleç
birine yuvarlayabilir. MEKANIZMANIN KIRILACAGI YER BURASI.

Ve bu, "kucuk bir Ingilizce model yapilabilir mi" sorusunun UCUZ on
provasi. Dogrudan dil yolu ELENDI: DiscoLoop'un avantaji 440M/20B'de ve
~13B jetondan SONRA cikiyor; bizim butcemiz ~82M. Orada kosmak "ise
yaramadi" degil "butce yetmedi" uretir.

--------------------------------------------------------------------------
VERI -- graf DEGISMIYOR, yalniz KODLAMA

    Ayse_Yilmaz       -> (Ayse, Yilmaz)        ad + soyad
    Ankara_Fen_Lisesi -> (Ankara, Fen_Lisesi)  sehir + tur
    Ankara            -> (Ankara, <YOK>)

ILK alt cizgiden, cunku anlamli olan o. Sonucu:

    SOZLUK  2145 -> ~300   varliklar BILESIK hale geliyor
    700 kisi 100 ad + 7 soyad ile kuruluyor
    hicbir jeton TEK BASINA kisiyi belirlemiyor

Soyadi paylasimi UYDURMA DEGIL, veri_okul'da zaten var: "SOYADI KALITIMI:
cocuk / kardes / anne / baba AYNI soyadi tasir."

    SIMDI  [Q2] e     r1 r2 ?  a      EOS   T_LEN  8
    BU KOL [Q2] e1 e2 r1 r2 ?  a1 a2  EOS   T_LEN 11

--------------------------------------------------------------------------
MIMARI DEGISMIYOR -- bu yuzden model_c DEGIL

`ModelB` bir satir bile degismiyor; dar_alfa=0.5, isleç ayni. Degisen
GOREV TANIMI. `model_c`, isleç bir jeton yerine JETON DIZISI uretecek
hale gelirse sakli -- ki bu kosunun sonucu onu gerektirebilir.

UYARI (onkayit §4): OLCME IZI ZORUNLU OLARAK FARKLI olacak -- sozluk ve
diziler degisti. `57cf60a5e9af` denetimi GECMEYECEK ve bu hata degil.
Kiyas BOLME ORANLARI uzerinden, birebir ornek eslesmesi uzerinden DEGIL.
Ailedeki diger kiyaslardan ZAYIF, ve oyle raporlanacak.
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

AYAR = TABAN.degistir(ad="model_b5", jeton_ad="ilk")   # KUSURLU, bkz. onkayit §8.1
#      ^ SADECE FARK. veri_okul2, dar_alfa=0.5, ood_pay=0.05, wd=0.5,
#        ort_bas=10000 hepsi model_b1'den gelir.

fark_bas = M.fark_bas


def egit(ayar=None, **kw):
    return M.egit(ayar or AYAR, model_kur=ModelB, **kw)


if __name__ == "__main__":
    M.fark_bas(TABAN, AYAR)
    egit(AYAR)
