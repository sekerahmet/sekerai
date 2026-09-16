# -*- coding: utf-8 -*-
"""veri_okul4 — `veri_okul3`un YOGUN SEMASI, KUCUK havuzda (1x).

Kullanici karari, 16 Eylul 2026: *"varliklari yariya indir"*.

    veri_okul3   2120 varlik   olgu 21.920   phi TAVANI 9,57
    veri_okul4   1060 varlik   olgu 10.960   phi TAVANI 9,57   <- AYNI

--------------------------------------------------------------------------
GOREV AYNI ZORLUKTA -- phi OLCEK-DEGISMEZ

phi = zincir / olgu. Butun havuzlar ORANTILI kuculunce pay da payda da
ayni oranda duser; phi DEGISMEZ. `veri_okul` (1x) ile `veri_okul2`
(2x) arasindaki oranlar da korunuyor (kisi/sehir 8,75, kisi/okul 3,50,
okul/sehir 2,50) -- `veri_okul2.py` bunu bilerek boyle kurmustu.

Yani bu bir GOREV dugmesi DEGIL, BUTCE dugmesi.

--------------------------------------------------------------------------
NE KAZANDIRIYOR

Ayni adim sayisinda IKI KATI epoch. model_b14 kapilari ancak 40.000
adimda gecebilmisti (havuz 198.688); burada havuz yariya iniyor.

Ve asil sebep: bu, BICIM CESITLILIGINI odenebilir kiliyor. Sorular da
3 bicimde uretilince havuz 2x'te 593.944 olurdu; 1x'te ~297.000, yani
bugunku model_b15 tasarisiyla ayni mertebede.

--------------------------------------------------------------------------
LITERATUR DE BU ARALIKTA

    arXiv 2505.17923  2-hop small   |E| = 250
    arXiv 2505.17923  2-hop large   |E| = 500
    Wang ve ark.                    ayni mertebede
    BIZ veri_okul3                  |E| = 2120   <- 4-8 KAT ustunde
    BIZ veri_okul4                  |E| = 1060

Arsivde ters yonde bir veri noktasi da var ve simdi anlamli: veri x4
yapilinca `ent` KOTULESMISTI (model_a4 0.0113). O kosular wd=0.1 ile
yapilmisti -- yani bugun geri dondugumuz rejimin AYNISINDA. Olcek
buyutmek zarar verdiyse, kucultmek ters yonde bir gozlem.

--------------------------------------------------------------------------
BEDELI

Sinav bolmeleri kuculur. `ood` ve `ent_yok` gurultulenir -- ikisi de
ZATEN hukum vermiyor. `comp`/`ent`/`one`/`seen` 3000'de tavanli ve
havuzlari yetiyor.

!! OLCME IZI DEGISIR. Varlik kumesi degisince sinav zincirleri de
degisir; veri_okul3 sonuclariyla AYNI TABLODA okunamaz.
Kullanici karari (CLAUDE.md, "KIYAS ARTIK ARKA PLANDA"): bu kolu
IPTAL ETMEZ, not dusulur ve devam edilir.

--------------------------------------------------------------------------
NE DEGISMIYOR

Sema (17 iliski, sekiz alan acilimi), arama uzayi (17^2 = 289),
yapisal kisitlar, BLOK=100, bolme oranlari, tip oranlari.
"""
from __future__ import annotations

import veri_okul as VO
import veri_okul3 as V3

TIPLER = V3.TIPLER
ILISKI = V3.ILISKI                     # 17 SEMBOL -- DEGISMEDI
BLOK = V3.BLOK
SEMA = V3.SEMA                         # veri_okul3 ile AYNI SEMA
GEREKTIRIR = V3.GEREKTIRIR


def kur(tohum=0):
    """`veri_okul` (1x) grafi + `veri_okul3`un sekiz alan acilimi."""
    return V3.genislet(VO.kur(tohum), tohum)


zincirler = VO.zincirler
yaz = VO.yaz
turetilebilir = V3.turetilebilir


if __name__ == "__main__":
    import collections
    import veri_okul2 as V2          # noqa: F401  (kiyas icin)
    for ad_, M in (("okul  1x", VO), ("okul2 2x", V2),
                   ("okul3 2x+", V3), ("okul4 1x+", __import__("veri_okul4"))):
        G = M.kur(0)
        z = VO.zincirler(G)
        print("%s  varlik %5d  olgu %6d  zincir %7d  phi TAVANI %5.2f  |R|=%d"
              % (ad_, sum(G["n"].values()), len(G["olgu"]), len(z),
                 len(z) / len(G["olgu"]), len(G["iliski"])))
        print("     sinif", dict(collections.Counter(x[6] for x in z)))
