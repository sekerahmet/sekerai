# -*- coding: utf-8 -*-
"""model_b2 — AYNI MODEL, BASKA HESAP YOLU.  TEK FARK: dar_sdpa False -> True.

Onceden kayit: belge/onkayit/model_b2.md
Kullanici karari 16 Eylul: "model_b2 olarak yazar misin."

    model_b1   veri_okul2  dar_alfa=0.5  dar_sdpa=False
    model_b2   veri_okul2  dar_alfa=0.5  dar_sdpa=True
                                         ^ TEK FARK

--------------------------------------------------------------------------
BU KOL DOGRULUK SORMUYOR, MALIYET SORUYOR

`model_b1` `model_a8`i her olcude gecti ama darbogazin bir bedeli var ve
o bedel KALICI -- cikarimda kapatilinca `one` 0.9520 -> 0.0243 dusuyor
(model_b.md §9.4). Olculen yuk (T4, fp16, sira dondurulerek, 9 tur
ortanca):

    yigin      a8        b1 NAIF     yuk
        1   4.489 ms    4.953 ms   +10.3%
     1000  17.696 ms   22.140 ms   +25.1%

Soru: bu yukun ne kadari GEREKSIZ?

--------------------------------------------------------------------------
TESHIS: `Phi` ZATEN BIR DIKKAT KATMANI

    Phi(h) = softmax( nf(h) Wᵀ / tau ) @ W
             ^Q          ^K    ^olcek    ^V        Q=nf(h),  K=V=W

`head = nn.Linear(d, vocab, bias=False)` ve `head.weight = emb.weight`
(model_a.py, bagli gomme) oldugu icin bu bir BENZETME DEGIL, CEBIRSEL
OZDESLIK. Yani `Phi` dogrudan SDPA'ya verilebilir.

Bugunku kod (N,V) olasilik matrisini HBM'e UC KEZ gidip getiriyor
(B=1000, V=2145 -> 34,3 MB). SDPA onu HIC YAZMAZ: sozlugu dilimler,
kosan softmax ile toplami gunceller. FlashAttention'in asil fikri bu --
carpma sayisi ayni, BELLEK TRAFIGI dusuyor.

    yigin      b1 NAIF    b1 SDPA    yuk NAIF   yuk SDPA   kurtarilan
        1     4.953 ms   4.825 ms     +10.3%      +7.5%       27%
     1000    22.140 ms  20.659 ms     +25.1%     +16.7%       33%

Bu bir ALT SINIR: T4 sm 7.5, gercek Flash cekirdegi sm80 istiyor ve
CALISMIYOR; PyTorch bellek-verimli yedege dusuyor.

NOT: modelin KENDI dikkati zaten SDPA kullaniyor (`model_a.Blok`). Bu kol
yeni bir teknik ya da bagimlilik getirmiyor -- kod tabaninin zaten
kullandigi fonksiyonu, kullanmadigi bir yerde kullaniyor.

--------------------------------------------------------------------------
NE SOYLENEMEZ  (onkayit §5)

"model_b2, model_b1 ile AYNI sonucu verir" DENEMEZ. Ayni FONKSIYON ama
BIT DUZEYINDE ayni degil (fp16 toplama sirasi; olculen bagil fark
~1e-3). 20.000 adimda bu birikir ve iki kosu FARKLI YORUNGELERE ayrilir.
Ayrismanin ne kadar sonuc farki yaratacagi OLCULMEDI -- bu kolda tohum
tekrari yok.

Bu kolun HUKMU HIZDA; dogruluk yalniz bir GUVENLIK KAPISI.

ASIL ACIK SORU: yukaridaki olcumlerin hepsi ILERI GECIS. Egitim `Phi`
uzerinden GERI YAYILIM da yapiyor ve SDPA'nin backward'i ayri bir
cekirdek. Egitimde kazanip kazanmadigi BILINMIYOR.
"""
from __future__ import annotations

import sys, os

_B = os.path.dirname(os.path.abspath(__file__))
_A = os.path.join(os.path.dirname(_B), "model_a")
for _p in (_A, _B):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# `deneme2/model_a` bir KLASOR; yol once eklenmezse `import model_a` onu
# NAMESPACE PAKETI olarak bulur ve hata ancak egitim baslarken cikar.
# 16 Eylul'de model_b1'in ilk Colab kosusu tam boyle coktu.

import model_a as M                                          # noqa: E402
assert hasattr(M, "egit"), (
    f"model_a MODUL degil PAKET olarak yuklendi: {getattr(M,'__file__',None)}")

from model_b import ModelB                                   # noqa: E402
from model_b1 import AYAR as TABAN                           # noqa: E402

AYAR = TABAN.degistir(ad="model_b2", dar_sdpa=True)
#      ^ SADECE FARK. veri_okul2, dar_alfa=0.5, ood_pay=0.05, wd=0.5,
#        ort_bas=10000 hepsi model_b1'den (dolayisiyla model_a8'den) gelir.

fark_bas = M.fark_bas          # kos.py UC sey ister: AYAR, egit, fark_bas


def egit(ayar=None, **kw):
    return M.egit(ayar or AYAR, model_kur=ModelB, **kw)


if __name__ == "__main__":
    M.fark_bas(TABAN, AYAR)
    egit(AYAR)
