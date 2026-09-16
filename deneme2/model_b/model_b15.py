# -*- coding: utf-8 -*-
"""model_b15 — BICIM CESITLILIGI (ek isaretleyicili dil).

Onceden kayit: belge/onkayit/model_b15.md
Taban model_b14.  UC DUGME:

    model_b14   veri_okul3 (2120)   ek_kip=""     bicim=1   t_len 11
    model_b15   veri_okul4 (1060)   ek_kip="tr"   bicim=3   t_len 17

!! ATFETME YAPILAMAZ -- ve artik hedef atfetme DEGIL. Kullanici karari,
16 Eylul (CLAUDE.md "KIYAS ARTIK ARKA PLANDA"): *"artik model
karsilastirma yok, bizim dilimizi ogrenen bir model yapmaya
calisiyoruz; kriterimiz comp ve ent."*

--------------------------------------------------------------------------
NEDEN -- SEMPTOMUMUZU BIREBIR TARIF EDEN TEK BULGU

Physics of LM 3.1 (arXiv 2309.14316), BIREBIR:

  "such encoding is only possible with knowledge augmentations like
   permutation/rewriting of entity-attribute knowledge during
   pretraining. Without these augmentations, the language model can
   still MEMORIZE the training data, but it is NOT LINEARLY ENCODED in
   the entity's hidden embeddings, making knowledge extraction via QAs
   quite hard, if not impossible, even with instruction fine-tuning."

Bizim olcumumuz:

    one 1.0000  seen 1.0000               EZBER TAM
    dogrusal sonda slot 0:  comp/ent/ood/seen hepsi "bilgi YOK"
    ASAMA-1 cevaptan onceki pozisyon:  0.0275

Ezber tam, dogrusal kodlama yok. Onlarin "augmentation olmadan"
dedigi durumun tarifi bu. Ve bizim verimizde SIFIR augmentation var:
her olgu egitimde TEK bir yuzey biciminde geciyor.

--------------------------------------------------------------------------
NEDEN EK ISARETLEYICI GEREKTI (kullanici, 16 Eylul)

Ilk onerim "iliskinin yerini degistirelim" idi. YANLIS:

    Fatma anne Ayse   !=   Ayse anne Fatma

Bugunku dilde rolu POZISYON tasiyor, o yuzden permutasyon ANLAMI
BOZAR. Kullanici duzeltti: Turkce'de sira serbesttir cunku rolu EK
tasir -- "Ayse'nin annesi Fatma'dir" = "Fatma'dir Ayse'nin annesi".
Ve kesme isareti de bir jetondur.

Dort jeton eklendi -- ILISKI DEGIL, DILBILGISI:

    '      ozel adla ek arasina     (Ayse'nin)
    <NIN>  tamlayan  (sahip)
    <SI>   tamlanan  (iliski)
    <DIR>  yuklem    (cevap)

Uc yuzey bicimi (hepsinde CEVAP SONDA -- cevap-basta bir bicimde cevap
pozisyonu hicbir sey olcmezdi):

    0  Ayse Yilmaz <YOK> ' <NIN> anne <SI> ? Fatma Yilmaz <YOK> ' <DIR> <EOS>
    1  anne <SI> Ayse Yilmaz <YOK> ' <NIN> ? Fatma Yilmaz <YOK> ' <DIR> <EOS>
    2  Ayse Yilmaz <YOK> ' <NIN> anne <SI> Fatma Yilmaz <YOK> ' <DIR> <EOS>

2-hop Turkce'de zaten boyle kuruluyor: "Ayse'nin cocugunun kardesi"
     e ' <NIN> cocuk <SI> <NIN> kardes <SI> ?
                          ^^^^^^^^^^^ cocugunun = cocuk <SI> <NIN>

--------------------------------------------------------------------------
BICIM CESITLILIGI OLGU **VE** SORU SATIRLARINDA

Ilk tasariminda yalniz olgu satirlari cogaltiliyordu (maliyet icin).
Kullanici duzeltti, 16 Eylul: *"bu arada sorulari da degistirmek
gerekiyor, mesela Ahmet Yilmaz'in annesinin kardesi gibi."*

Gerekcesi saglam: `seen` 1.0000 + `comp` 0.03 tablosu modelin 2-hop
cevaplarini DUZ ESLEME olarak depoladigini soyluyor, ve ayni zinciri
UC yuzeyde gormek o eslemeyi tam olarak zorlastirir.

Bedeli VARLIK SAYISINI YARIYA INDIREREK odendi (veri_okul4):

    2120 varlik, olgu+soru x3   ->  593.944 satir   (~2,2 saat)
    1060 varlik, olgu+soru x3   ->  297.548 satir   (~65 dk)

SINAV hep bicim 0 -- egitim daha cok yuzey gorur, sinav TEK bicim,
yani egitim/sinav BICIM UYUSMAZLIGI yok.

--------------------------------------------------------------------------
ESKI KIP BIT AYNI -- OLCULDU

`ek_kip` kapaliyken hicbir sey degismiyor. Yeni jetonlar SOZLUGUN
SONUNA eklendi, REL_OFF/ent_off/yuva_ara KAYMADI. model_b6, model_b13
ve model_b14'un egitim havuzlarinin sha256'si yamadan ONCE ve SONRA
BIREBIR AYNI:

    model_b6   93.495 satir   1d3ccd30e25b2b27
    model_b13 116.815 satir   5b9eab1e785570f7
    model_b14 198.688 satir   29569380a3926fd6

test_sabit.py'nin 2-6. bolumleri (graf, gordugu veri, model
baslangici, dugme yapisi, eski kosularin okunabilirligi) TAM GECTI;
yalniz "alan sayisi 40" kilidi dustu -- kasitli, ve gerekcesi bu dosya.

--------------------------------------------------------------------------
NEDEN VARLIKLAR YARIYA INDI (veri_okul4)

phi OLCEK-DEGISMEZ: butun havuzlar orantili kuculunce zincir de olgu
da ayni oranda duser, phi = 9,57 tavani DEGISMEZ. Yani gorev AYNI
ZORLUKTA -- bu bir GOREV dugmesi degil, BUTCE dugmesi.

Kazanc iki katli: (1) ayni adimda iki kati epoch -- model_b14 kapilari
ancak 40.000'de gecebilmisti; (2) uc bicimi odenebilir kildi.

Literatur de bu aralikta: 2505.17923'un calisan kurulumlari |E|=250 ve
|E|=500, Wang da ayni mertebede. veri_okul3 |E|=2120 ile 4-8 kat
ustundeydi. Arsivde ters yonde bir gozlem de var: veri x4 yapilinca
`ent` KOTULESMISTI (model_a4 0.0113) -- ve o kosular wd=0.1 ile, yani
bugun geri dondugumuz rejimde yapilmisti.

--------------------------------------------------------------------------
BEDEL

    varlik       2120 -> 1060
    t_len          11 -> 17          adim basina ~%50 pahali
    vocab         466 -> 301         (1060 varlik, daha kucuk jeton havuzu)
    havuz     198.688 -> 297.548
    ent_yok      3000 -> 1957        gurultulenir, ZATEN hukum vermiyor
    ood           402 ->  210        gurultulenir, ZATEN hukum vermiyor
    |R| 17 KALIR, arama uzayi 289 KALIR

    !! olcme izi  b7a4e954b2a2 -> d6751004648c
       Sinav ZINCIRLERI degisti (varlik kumesi yarilandi). model_b14 ile
       AYNI TABLODA okunamaz. Kol IPTAL DEGIL -- not dusuldu, devam.

MIMARI: bir satir bile degismedi. Ek parametre YOK.
"""
from __future__ import annotations

import os
import sys

_B = os.path.dirname(os.path.abspath(__file__))
_A = os.path.join(os.path.dirname(_B), "model_a")
_K = os.path.dirname(_B)
for _p in (_A, _B, _K):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import model_a as M                                          # noqa: E402
assert hasattr(M, "egit"), (
    f"model_a MODUL degil PAKET olarak yuklendi: {getattr(M,'__file__',None)}")

from model_b import ModelB                                   # noqa: E402
from model_b14 import AYAR as TABAN                          # noqa: E402

AYAR = TABAN.degistir(ad="model_b15", veri_ad="veri_okul4",
                      ek_kip="tr", bicim=3)
#      ^ UC DUGME (kullanici karari, CLAUDE.md "KIYAS ARTIK ARKA PLANDA"):
#        1) veri_okul4  -- ayni YOGUN sema, YARIM havuz (1060 varlik).
#           phi OLCEK-DEGISMEZ: tavan 9,57 KALIYOR, gorev ayni zorlukta.
#           Kazanc BUTCE: ayni adimda iki kati epoch, ve bicim
#           cesitliligini ODENEBILIR kiliyor.
#        2)+3) ek_kip + bicim -- BIR CIFT, ek_kip olmadan bicim
#           ANLAMSIZ (eksiz dilde sirayi degistirmek anlami bozar),
#           egitim_havuzu assert ile reddediyor.
#        ident/wd/cosine model_b14'ten DEVRALINIR (kural 4).
#        ATFETME YAPILAMAZ -- ve artik hedef atfetme degil, `comp`/`ent`.

fark_bas = M.fark_bas


def egit(ayar=None, **kw):
    return M.egit(ayar or AYAR, model_kur=ModelB, **kw)


if __name__ == "__main__":
    M.fark_bas(TABAN, AYAR)
    egit(AYAR)
