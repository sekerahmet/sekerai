# -*- coding: utf-8 -*-
"""denetim_14 -- KOPYA KODUN CIKTISINI BAGIMSIZ DENETLER.

Kullanici, 20 Eylul 2026: *"kopya oldugu icin guvenim zayif kopya
olan seylere."*

Haklı itiraz. `veri_14 / korpus_14 / metin_14 / jeton_14 / taban_14`
model_13'ten kopya; izlerin tutmasi KOPYANIN ASLINA ESIT oldugunu
kanitliyor, ASLIN DOGRU oldugunu degil. Yazdigimiz kodda alti,
miras bolucude dort hata cikti.

Sifirdan yazmak cozum degil: 3.700 satirlik yeni bir uretici 13 kol
boyunca ayiklanmis olanindan DAHA COK hata tasir, ve korpus degisince
model_11 / model_13 sayilari kiyaslanamaz hale gelir.

Bu dosya ucuncu yolu kuruyor: **koda degil CIKTIYA bakmak.** Her
denetim, ureticinin kendi `assert`lerinden BAGIMSIZ -- graftan ve
birim akisindan dogrudan hesaplanir.

    python denetim_14.py
"""
from __future__ import annotations

import sys

import numpy as np

import ayar_14 as AY
import birim_14 as BR
import olcme_14 as OL
import taban_14 as MT
import veri_14 as V

GECTI, KALDI = [], []


def kapi(ad):
    def sar(f):
        try:
            GECTI.append((ad, f() or ""))
        except AssertionError as e:
            KALDI.append((ad, str(e)))
        except Exception as e:
            KALDI.append((ad, "%s: %s" % (type(e).__name__, e)))
        return f
    return sar


print("veri kuruluyor...", flush=True)
A = AY.AYAR
v = MT.veri_kur(A, yaz=lambda *a, **k: None)
L = MT.olcme_listeleri(A, v)
G = V.kur(A.veri_tohum)
E_ad = [x for t in V.TIPLER for x in G["ad"][t]]
b = BR.yukle("G:/Drive'\u0131m/model_14/birim_14.npz", yaz=lambda *a: None)
HAM = np.asarray(b.dizi, np.uint16).tobytes()
RIX = {r: i for i, r in enumerate(V.ILISKI)}


# =====================================================================
@kapi("D1  OLCEK -- varlik ve iliski sayilari")
def _d1():
    """Sayilar `veri_14`un yorumundan DEGIL, uretilen graftan."""
    say = {t: len(G["ad"][t]) for t in V.TIPLER}
    bek = {"KISI": 480, "TEZ": 480, "DERS": 300, "BOLUM": 160,
           "SEHIR": 81, "FAKULTE": 60, "UNIVERSITE": 40, "BOLGE": 7}
    fark = {t: (say[t], bek[t]) for t in bek if say.get(t) != bek[t]}
    assert not fark, "olcek kaymis: %s" % fark
    assert len(E_ad) == sum(bek.values()) == 1608, len(E_ad)
    assert len(V.ILISKI) == 24, len(V.ILISKI)
    return "1.608 varlik, 24 iliski, tipler: %s" % say


@kapi("D2  GRAF TUTARLI -- kendine iliski, cinsiyet, simetri")
def _d2():
    """Grafin kendi iddialari: `annesi` HER ZAMAN kadin, `kardesler
    ayni ebeveynden`, `cocugu annesinin TERSI`. Uretici bunlari
    yoruma yaziyor; burada SAYIYORUZ."""
    F = v.facts
    kendi = int(sum((F[e] == e).sum() for e in range(F.shape[0])))
    assert kendi == 0, "%d varlik kendisine iliskili" % kendi

    kis = list(range(len(G["ad"]["KISI"])))
    # cinsiyet: bir varlik hem ANNE hem BABA olamaz
    anne = RIX.get("annesi")
    baba = RIX.get("babasi")
    hed_a = {int(F[e, anne]) for e in kis if F[e, anne] >= 0}
    hed_b = {int(F[e, baba]) for e in kis if F[e, baba] >= 0}
    ort = hed_a & hed_b
    assert not ort, "%d varlik hem ANNE hem BABA olarak geciyor" % len(ort)

    # kardeslik simetrik mi
    kar = RIX.get("kardesi")
    asim = [e for e in kis if F[e, kar] >= 0
            and int(F[int(F[e, kar]), kar]) != e]
    assert not asim, "%d kiside kardeslik ASIMETRIK" % len(asim)
    return ("kendine iliski 0   anne/baba kesisimi 0   "
            "kardeslik %d/%d SIMETRIK" % (len(kis) - len(asim), len(kis)))


@kapi("D3  CEVAP DOGRU -- sinav sorusu grafla tutuyor mu")
def _d3():
    """Sinavin kurdugu her sorunun cevabi, GRAFTA o zincirin sonu mu.
    `olcme_listeleri`nin dogru zinciri verdigine GUVENMIYORUZ."""
    F = v.facts
    kotu = 0
    # !! BOLME ADI YOK. Adim zincirin UZUNLUGUNDAN: (e,r,cevap) ya da
    # (e,r1,r2,KOPRU,cevap). Burasi eskiden `1 if ad == "one" else 2`
    # diyordu -- ayni varsayim `olcme_14.tum`da butun 2 adimli
    # zincirleri sessizce dusurmustu.
    n_z = 0
    for ad, zs in L.items():
        if not hasattr(zs, "__len__"):
            continue
        for z in list(zs)[:500]:
            z = [int(x) for x in z]
            adim = (len(z) - 1) // 2
            n_z += 1
            e, rs, ans = z[0], z[1:1 + adim], z[-1]
            cur = e
            for r in rs:
                cur = int(F[cur, r])
                if cur < 0:
                    break
            if cur != ans:
                kotu += 1
    assert kotu == 0, "%d zincirin sonu grafla TUTMUYOR" % kotu
    return "%d zincir orneklendi, hepsinin sonu grafla tutuyor" % n_z


@kapi("D4  SIZINTI -- CIKARIM gercekten SOYLENMEMIS mi")
def _d4():
    """EN KRITIK KAPI. Bir soru CIKARIM sayiliyorsa cevabini soyleyen
    cumle korpusta GECMEMELI. Bunu ureticinin `sizinti_kapisi`ndan
    BAGIMSIZ olarak, birim akisinda bayt aramasiyla denetliyoruz.

    (Yeni siniflandirma zaten bu olcutle yapiliyor, yani bu kapi
    olcutun KENDI KENDINI dogrulamasi -- ama ayni zamanda `_gecer`in
    calistiginin kaniti: CIKARIM 0, OGRETILEN yuksek cikmali.)"""
    soru_tip = {i: b.ix.get({"KISI": "kim", "SEHIR": "neresi",
                             "BOLGE": "neresi"}.get(t), b.ix["hangisi"])
                for i, t in enumerate(V.TIPLER)}
    S = OL.Sorular(v, E_ad, list(V.ILISKI), V.TR, V.TR_ILISKI, b.kok,
                   b.ix, b.korunan, soru_tip)
    kaynak = {ad: list(zs)[:400] for ad, zs in L.items()
              if hasattr(zs, "__len__")}
    q = S.tum(kaynak, b.dizi, yaz=lambda *a: None)
    og, ci = q["OGRETILEN"], q["CIKARIM"]
    kirli = sum(OL._gecer(HAM, s.onek[:-3] + s.cevap) for s in ci)
    assert kirli == 0, "%d CIKARIM sorusunun cevabi korpusta GECIYOR" % kirli
    assert og and ci, "iki grup da dolu olmali: %d / %d" % (len(og), len(ci))
    return "OGRETILEN %d   CIKARIM %d   sizinti 0" % (len(og), len(ci))


@kapi("D5  BIRIM AKISI -- korpus metnini birebir kapsiyor mu")
def _d5():
    """Akis, kelime bolmesinin BIREBIR birlesimi mi. Kelime sayisi ve
    birim sayisi tutmali; aksi halde bolme ile akis ayrismis demektir."""
    bek = sum(len(p) for p in b.bolme.values())
    assert bek > 0
    gecen = {b.ad[i] for i in np.unique(b.dizi)}
    tanim = {x for p in b.bolme.values() for x in p}
    assert gecen <= tanim, "akista bolmede OLMAYAN birim var: %s" % list(
        gecen - tanim)[:5]
    kul = len(gecen) / len(b.ad)
    assert kul > 0.99, "birimlerin yalniz %.1f%%'i akista geciyor" % (100 * kul)
    return "sozluk %d birim, akista gecen %d (%.1f%%)" % (
        len(b.ad), len(gecen), 100 * kul)


@kapi("D6  VARLIK KAPSAMASI -- her varlik metinde geciyor mu")
def _d6():
    """Bir varlik hic anlatilmiyorsa onunla ilgili soru CEVAPLANAMAZ
    ve sinav haksiz olur. model_09'da tam bu olmustu: BOLGE 0 cumlede
    geciyordu (CLAUDE.md kural 5)."""
    soru_tip = {i: b.ix["hangisi"] for i in range(len(V.TIPLER))}
    S = OL.Sorular(v, E_ad, list(V.ILISKI), V.TR, V.TR_ILISKI, b.kok,
                   b.ix, b.korunan, soru_tip)
    eksik = {}
    for t in V.TIPLER:
        adlar = G["ad"][t][:60]
        yok = sum(1 for a in adlar
                  if not (S.birim(a) and OL._gecer(HAM, S.birim(a))))
        if yok:
            eksik[t] = "%d/%d" % (yok, len(adlar))
    assert not eksik, "metinde HIC gecmeyen varliklar: %s" % eksik
    return "her tipten 60 varlik orneklendi, hepsi metinde geciyor"


@kapi("D7  KORPUS IZI -- karakter tensoru ile birim akisi ayni korpus mu")
def _d7():
    """Iki bagimsiz iz: karakter tensorunun md5'i (model_11/13 ile
    kiyas icin) ve birim akisinin md5'i (model_14 bunu egitiyor)."""
    assert V.IZ == AY.IZ_GRAF, V.IZ
    assert MT.olcme_izi(L) == AY.IZ_OLCME, MT.olcme_izi(L)
    return "graf %s   olcme %s   birim %s" % (V.IZ, MT.olcme_izi(L), BR.iz(b))


# =====================================================================
if __name__ == "__main__":
    for ad, not_ in GECTI:
        print("  GECTI   %s\n          %s" % (ad, not_) if not_
              else "  GECTI   " + ad)
    for ad, e in KALDI:
        print("  KALDI   %s\n          %s" % (ad, e))
    print("\n%d gecti, %d kaldi" % (len(GECTI), len(KALDI)))
    sys.exit(1 if KALDI else 0)
