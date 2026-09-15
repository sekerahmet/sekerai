# -*- coding: utf-8 -*-
"""veri_wang — WANG 2405.15071 §3.1'in RASTGELE GRAFI.  Sema YOK, tip YOK.

    "we generate a random knowledge graph G consisting of |E| entities and
     |R| = 200 relations, where each entity (as the subject) has 20 outgoing
     edges that connect through a random relation to another random entity
     (as the object). The atomic facts are then the edges."
                                                    -- makale §3.1, birebir

NEDEN VAR
---------
`veri_okul` OLCULDU ve Wang'in grafindan YAPISAL OLARAK KOLAY cikti:

    OOD sinav zincirinin (r1,r2) ciftini model egitimde kac kez gordu
      veri_okul   ortanca 911   ortalama 879   (120 farkli cift)
      Wang        ~8,6                         (40.000 olasi cift)
      ORAN        103 KAT

Yani `veri_okul`da model kurali YUZLERCE baska varlikta calisiyor ve
"yeni bir varliga uygula" deniyor. Wang'da kurali 9 kez goruyor. Ayni
sinav DEGIL. Ustelik `veri_okul`un ILISKILERI TIPLI (`vali` yalniz
SEHIR'den cikar) -- tip onseli sans seviyesini 0.00047'den 0.0025'e,
BES KAT yukseltiyor. Wang'in grafinda tip yok.

BU MODUL O IKI FARKI KAPATIR.

FARKLAR -- makaleden BILEREK sapilan tek yer
---------------------------------------------
    |E|   1000   (makale 2000)
Sebep makalenin KENDI olcumu, §3.2: "When fixing the ratio phi, the
training data size does not qualitatively affect the model's
generalization ... scaling the data affects neither the relative speed of
ID generalization and training improvement ... nor the systematicity
level (OOD performance stays zero)."
|E|=1000 epoch basina adimi YARIYA indirir; |R|, kenar sayisi ve dolayi-
siyla CIFT BASINA ZINCIR orani DEGISMEZ -- kapatmak istedigimiz fark o.

    |R| = 200, varlik basina 20 kenar   <- makaleyle AYNI
    tip YOK, sema YOK                    <- makaleyle AYNI

TURETILEN
---------
    atomik olgu        20.000
    tum 2-adimli zincir ~400.000
    cift basina zincir  ~7      (makale ~8,6)   <- 103 KAT fark KAPANDI
    sans seviyesi       0,001   (tip onseli YOK)

`ent_yok` BURADA BASKA SEY OLCER. `veri_okul`da "kisayol TIP OLARAK
imkansiz" demekti. Burada tip yok; "kisayol olgusu (e,r2) MEVCUT DEGIL"
demek. Graf %10 yogun oldugu icin zincirlerin cogu bu sinifa duser --
`veri_okul`da tersiydi. Ayni etiket, BASKA anlam; rapor okunurken
karistirilmamali.
"""
from __future__ import annotations

import numpy as np

N_VARLIK = 1000
N_ILISKI = 200
N_KENAR = 20          # varlik basina cikan kenar (makale: 20)

TIPLER = ["VARLIK"]   # TEK TIP = tipsiz. Sozlesme bir liste istiyor.
ILISKI = [f"r{i:03d}" for i in range(N_ILISKI)]
SEMA = {r: {"VARLIK": "VARLIK"} for r in ILISKI}
#      ^ her iliski her seyden her seye gidebilir. Hicbir sey yasak degil.


def kur(tohum=0):
    """Rastgele bilgi grafi. Makale §3.1: her varlik 20 RASTGELE iliskiyle
    20 RASTGELE varliga baglanir."""
    assert N_KENAR < N_ILISKI, "kenar sayisi iliski sayisindan az olmali"
    rng = np.random.RandomState(1000 + tohum)
    ad = [f"e{i:04d}" for i in range(N_VARLIK)]

    olgu = {}
    for i, e in enumerate(ad):
        # ILISKILER TEKRARSIZ: ayni (varlik, iliski) ikilisi iki hedefe
        # gidemez -- yoksa `olgu` sozlugu birini sessizce ezerdi ve varlik
        # basina kenar sayisi 20'nin ALTINA duserdi.
        rs = rng.permutation(N_ILISKI)[:N_KENAR]
        for r in rs:
            # hedef KENDISI olmasin: (e, r, e) zinciri DONUS sinifini
            # sisirir ve makalede de dugumler birbirine baglaniyor.
            h = int(rng.randint(N_VARLIK - 1))
            if h >= i:
                h += 1
            olgu[(e, ILISKI[int(r)])] = ad[h]

    assert len(olgu) == N_VARLIK * N_KENAR, \
        f"olgu {len(olgu)} != {N_VARLIK * N_KENAR} -- kenar cakismasi var"
    sozluk = ["<pad>", "<soru>", "?", "<son>"] + ILISKI + ad
    return dict(ad={"VARLIK": ad}, n={"VARLIK": N_VARLIK}, sozluk=sozluk,
                tip={a: "VARLIK" for a in ad},
                olgu=olgu, sema=SEMA, iliski=ILISKI)


def zincirler(G):
    """2 adimli zincirler, DORT SINIFA ayrilmis -- `veri_okul.zincirler`
    ile AYNI siniflandirma, tek farki tip suzgeci YOK (tip de yok).

    (e, r1, r2, kopru, cevap, kisayol, sinif)
        DONUS  cevap == e            zincir basa donuyor
        YOK    kisayol olgusu YOK    `veri_okul`da "tip olarak imkansiz"
                                     idi; burada "(e,r2) kenari mevcut
                                     degil". Graf %10 yogun -> COGUNLUK.
        AYNI   kisayol == cevap      kisayol tesadufen dogru
        AYIRT  digerleri             SINAVIN YAPILDIGI SINIF
    """
    olgu = G["olgu"]
    out = []
    for (e, r1), b in olgu.items():
        for r2 in G["iliski"]:
            if r2 == r1:
                continue
            cev = olgu.get((b, r2))
            if cev is None:            # ikinci hop olgusu yok -> zincir yok
                continue
            ks = olgu.get((e, r2))
            sinif = ("DONUS" if cev == e else
                     "YOK" if ks is None else
                     "AYNI" if ks == cev else "AYIRT")
            out.append((e, r1, r2, b, cev, ks, sinif))
    return out


def yaz(G, z, f=print):
    S = {k: [x for x in z if x[6] == k]
         for k in ("AYIRT", "YOK", "AYNI", "DONUS")}
    f("=" * 76)
    f("WANG RASTGELE GRAFI -- sema YOK, tip YOK")
    f("=" * 76)
    f(f"  VARLIK           : {N_VARLIK}   ILISKI: {N_ILISKI}   "
      f"varlik basina kenar: {N_KENAR}")
    f(f"  SOZLUK           : {len(G['sozluk'])} token")
    f(f"  ATOMIK OLGU      : {len(G['olgu'])}")
    f(f"  2-ADIMLI ZINCIR  : {len(z)}")
    for k in ("AYIRT", "YOK", "AYNI", "DONUS"):
        f(f"    {k:<6} {len(S[k]):>8}  (%{100*len(S[k])/max(1,len(z)):.1f})")
    _c = len({(x[1], x[2]) for x in z})
    f(f"  farkli (r1,r2) cifti : {_c}   zincir/cift ~{len(z)/max(1,_c):.1f}")
    f(f"  sans seviyesi        : {1/N_VARLIK:.4f}  (tip onseli YOK)")


if __name__ == "__main__":
    G = kur(0)
    yaz(G, zincirler(G))
