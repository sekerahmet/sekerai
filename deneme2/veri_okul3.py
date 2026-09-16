# -*- coding: utf-8 -*-
"""veri_okul3 — `veri_okul2`nin YOGUNLASTIRILMISI.  phi tavani 7,04 -> 9,57.

Kullanici karari, 16 Eylul 2026.

SORU: model iki hop'u da TEK TEK 1.0000 biliyor ama birlestiremiyor.
Literatur bunun sebebi olarak VERI YOGUNLUGUNU gosteriyor. Bizim
yogunlugumuz olculdu ve iki bagimsiz kaynagin altinda.

--------------------------------------------------------------------------
NEDEN -- IKI KAYNAK, AYNI YERI GOSTERIYOR

phi = bir atomik olgu basina kac turetilmis (2-hop) ornek.

    Wang ve ark. (2024)          grokking araligi   phi  7 - 18
    arXiv 2505.17923  2-hop large                   phi 10,0   *
    arXiv 2505.17923  2-hop small                   phi 16,0   *
    BIZ (veri_okul2)                                phi  4,57

    * BEN HESAPLADIM, onlar basmiyor: |E|=500 |R|=20 -> olgu 10.000,
      "all possible reasoning questions"in %50'si -> 100.000 egitim
      sorusu. small: |E|=250 -> olgu 5.000, %80 -> 80.000.

--------------------------------------------------------------------------
NEDEN YENI ILISKI SEMBOLU EKLEMIYORUZ

Ilk tasarim 17 YENI sembol ekliyordu (es, patron, doktor, dogum...);
phi tavani 13,07 cikiyordu. ITIRAZ arXiv 2505.17923'ten geldi -- veri
acliginin asil kaynagini AYRISTIRMISLAR:

  "the search space (i.e. |R|^k relation combinations per entity)...
   Fixing 1-hop and 2-hop relations reduces the required training
   budget to x1, while increasing them leads to rapid budget growth."

Yeni sembol |R|'yi 17 -> 34 yapar, arama uzayini 289 -> 1156'ya, yani
DORT KATINA cikarir. phi yukselirken gorev de zorlasir; net etki
belirsiz, hatta negatif olabilir.

Bu dosya |R|'ye HIC DOKUNMUYOR. Var olan 17 sembolu baska TIPLERDE de
gecerli kiliyor -- `okul`, `sehir`, `ders` zaten boyle (cok alanli),
o kalip yayiliyor.

    veri_okul2   |R|=17  arama uzayi 289  olgu 16.800  phi TAVANI 7,04
    veri_okul3   |R|=17  arama uzayi 289  olgu 21.920  phi TAVANI 9,57
    (ilk tasarim) |R|=34 arama uzayi 1156 olgu 29.920  phi TAVANI 13,07

--------------------------------------------------------------------------
SEKIZ ACILIM -- hepsi Turkcede GERCEK kullanim

    kardes   KISI->KISI      + OKUL->OKUL    "kardes okul"
                             + SEHIR->SEHIR  "kardes sehir"
    komsu    SEHIR->SEHIR    + KISI->KISI    "komsu kisi"
                             + OKUL->OKUL    "komsu okul"
    rakip    OKUL->OKUL      + KISI->KISI    "rakip kisi"
                             + SEHIR->SEHIR  "rakip sehir"
    kurucu   OKUL->KISI      + SEHIR->KISI   "sehrin kurucusu"
                             + DERS->KISI    "dersin kurucusu"
    hoca     DERS->KISI      + OKUL->KISI    "okulun hocasi"
    okul     KISI/SEHIR->OKUL + DERS->OKUL   "dersin verildigi okul"
    sehir    KISI/OKUL->SEHIR + DERS->SEHIR  "dersin verildigi sehir"
    ders     KISI/OKUL->DERS  + SEHIR->DERS  "sehrin dersi"

    tip basina iliski   KISI  OKUL  SEHIR  DERS
    veri_okul2            10     5      3     2
    veri_okul3            12     8      7     5

En buyuk kazanc ZAYIF tiplerde. Bugun kopru bir DERS'e dustugunde
oradan yalniz 2 yol cikiyor -- modelin "kopruden devam edilir"
kuralini cikarmasi icin gereken cesitlilik orada YOK.

--------------------------------------------------------------------------
NE ACMADIM, ve NEDEN

    hoca   KISI->KISI   ACMADIM -- `ogretmen`in AYNISI olurdu
    kurucu KISI->KISI   ACMADIM -- `baba`ya yaklasirdi
    ders   DERS->DERS   ACMADIM -- `onkosul` var
    okul   OKUL->OKUL   ACMADIM -- `rakip`/`kardes` var

Iki iliski AYNI eslemeyse model birini digerinden OKUR ve bilesim
ogrenmeden gecer. Bu goz kararina birakilmadi: `turetilebilir()`
her (r1,r2) cifti icin r2(r1(x)) == r3(x) taramasi yapiyor ve
`test_veri3.py` bunu kilitliyor.

--------------------------------------------------------------------------
KONTROL: veri_okul2 bu grafin ALT KUMESI

Ayni tohumla `veri_okul2.kur()` cagriliyor, o olgularin HICBIRINE
dokunulmuyor, yeni olgular AYRI bir RNG akisindan ekleniyor. Yani

    veri_okul2.kur(0)["olgu"]  ⊆  veri_okul3.kur(0)["olgu"]     BIREBIR

ve bu `test_veri3.py`de siniriyor. Kolun kontrolu `model_b13`tur:
ayni ayar (wd 0.1 + cosine + identity), tek fark GRAF.

--------------------------------------------------------------------------
!! OLCME IZI DEGISIR

Veri degisince `olcme_izi` (bugun 57cf60a5e9af) degisir. O parmak izi
butun defterlerde assert ile kilitli ve model_b6..b13 merdivenini
kiyaslanabilir tutan sey odur. Bu graf YENI BIR MERDIVEN baslatir:
veri_okul3 sonuclari veri_okul2 sonuclariyla AYNI TABLODA okunamaz.

--------------------------------------------------------------------------
NE DEGISMIYOR

Varlik kumesi (2120), varlik adlari, sozluk, jeton kodlamasi, t_len,
iliski SEMBOLLERI (17), yapisal kisitlar (kardesler ayni ebeveyn,
soyadi kalitimi, ters ciftler, simetri), BLOK=100, bolme oranlari.
"""
from __future__ import annotations

import numpy as np

import veri_okul as VO
import veri_okul2 as V2

TIPLER = VO.TIPLER
ILISKI = VO.ILISKI                     # 17 SEMBOL -- DEGISMEDI
BLOK = VO.BLOK

# Genisletilmis sema. Eski girdiler AYNEN; yalniz yeni (tip -> tip)
# satirlari eklendi.
SEMA = {r: dict(m) for r, m in VO.SEMA.items()}
SEMA["kardes"].update({"OKUL": "OKUL", "SEHIR": "SEHIR"})
SEMA["komsu"].update({"KISI": "KISI", "OKUL": "OKUL"})
SEMA["rakip"].update({"KISI": "KISI", "SEHIR": "SEHIR"})
SEMA["kurucu"].update({"SEHIR": "KISI", "DERS": "KISI"})
SEMA["hoca"].update({"OKUL": "KISI"})
SEMA["okul"].update({"DERS": "OKUL"})
SEMA["sehir"].update({"DERS": "SEHIR"})
SEMA["ders"].update({"SEHIR": "DERS"})

assert set(SEMA) == set(VO.SEMA), "YENI SEMBOL EKLENMIS -- arama uzayi bozulur"

# `_esle` cift sayida indeks ister; hepsi cift mi diye kur()'da denetlenir.
GEREKTIRIR = dict(VO.GEREKTIRIR)


def _farkli_esle(rng, idx, yasak, ad, deneme=400):
    """Simetrik esleme kur, ama `yasak` sozluklerinin HICBIRIYLE ayni
    olmasin. `arkadas`in veri_okul.py'deki kurulusunun aynisi -- iki
    iliski ayni eslemeyse model birini digerinden OKUR."""
    idx = list(idx)
    for _ in range(deneme):
        e = VO._esle(rng, idx)
        if all(all(e[i] != y[i] for i in idx) for y in yasak):
            return e
    raise RuntimeError(f"{ad}: yasaklardan farkli esleme kurulamadi")


def genislet(G, tohum=0):
    """Verilen grafa SEKIZ ALAN ACILIMINI ekler. OLCEKTEN BAGIMSIZ --
    `veri_okul2` (2x) ile de `veri_okul` (1x) ile de calisir; phi
    OLCEK-DEGISMEZ oldugu icin tavan iki durumda da 9,57.

    Eski olgular BIREBIR korunur: gelen sozluge yalnizca EKLEME
    yapilir. Yeni olgular ayri bir RNG akisindan (tohum + 7777) gelir
    ki taban grafin akisi kaymasin."""
    ad, olgu = G["ad"], G["olgu"]
    kisi, okul, sehir, ders = ad["KISI"], ad["OKUL"], ad["SEHIR"], ad["DERS"]
    for t, v in (("KISI", kisi), ("OKUL", okul), ("SEHIR", sehir), ("DERS", ders)):
        assert len(v) % 2 == 0, f"{t} sayisi ({len(v)}) CIFT olmali (_esle)"
    eski = dict(olgu)                  # sonunda DEGISMEDIGI sinanir
    rng = np.random.RandomState(tohum + 7777)

    n_k, n_o, n_s, n_d = len(kisi), len(okul), len(sehir), len(ders)
    # mevcut simetrik eslemeler -- yenileri bunlardan FARKLI olmali
    _ind = lambda grup: {a: i for i, a in enumerate(grup)}
    ik, io, isz = _ind(kisi), _ind(okul), _ind(sehir)
    kardes_k = {i: ik[olgu[(c, "kardes")]] for i, c in enumerate(kisi)}
    arkadas_k = {i: ik[olgu[(c, "arkadas")]] for i, c in enumerate(kisi)}
    rakip_o = {i: io[olgu[(o, "rakip")]] for i, o in enumerate(okul)}
    komsu_s = {i: isz[olgu[(s, "komsu")]] for i, s in enumerate(sehir)}

    # --- KISI uzerine: komsu, rakip (kardes/arkadas ile CAKISMASIN) ------
    komsu_kisi = _farkli_esle(rng, range(n_k), [kardes_k, arkadas_k], "komsu/KISI")
    rakip_kisi = _farkli_esle(rng, range(n_k),
                              [kardes_k, arkadas_k, komsu_kisi], "rakip/KISI")
    for i, c in enumerate(kisi):
        olgu[(c, "komsu")] = kisi[komsu_kisi[i]]
        olgu[(c, "rakip")] = kisi[rakip_kisi[i]]

    # --- OKUL uzerine: kardes, komsu (rakip ile CAKISMASIN), hoca --------
    kardes_okul = _farkli_esle(rng, range(n_o), [rakip_o], "kardes/OKUL")
    komsu_okul = _farkli_esle(rng, range(n_o), [rakip_o, kardes_okul], "komsu/OKUL")
    for i, o in enumerate(okul):
        olgu[(o, "kardes")] = okul[kardes_okul[i]]
        olgu[(o, "komsu")] = okul[komsu_okul[i]]
        olgu[(o, "hoca")] = kisi[int(rng.randint(n_k))]

    # --- SEHIR uzerine: kardes, rakip (komsu ile CAKISMASIN), kurucu, ders
    kardes_sehir = _farkli_esle(rng, range(n_s), [komsu_s], "kardes/SEHIR")
    rakip_sehir = _farkli_esle(rng, range(n_s), [komsu_s, kardes_sehir], "rakip/SEHIR")
    for i, s in enumerate(sehir):
        olgu[(s, "kardes")] = sehir[kardes_sehir[i]]
        olgu[(s, "rakip")] = sehir[rakip_sehir[i]]
        olgu[(s, "kurucu")] = kisi[int(rng.randint(n_k))]
        olgu[(s, "ders")] = ders[int(rng.randint(n_d))]

    # --- DERS uzerine: kurucu, okul, sehir -------------------------------
    for i, d in enumerate(ders):
        olgu[(d, "kurucu")] = kisi[int(rng.randint(n_k))]
        olgu[(d, "okul")] = okul[int(rng.randint(n_o))]
        olgu[(d, "sehir")] = sehir[int(rng.randint(n_s))]

    G["sema"] = SEMA

    # --- DENETIMLER -------------------------------------------------------
    for (e, r), h in eski.items():
        assert olgu[(e, r)] == h, f"ESKI OLGU DEGISTI: {e} {r}"
    kendi = [(e, r) for (e, r), h in olgu.items() if h == e]
    assert not kendi, f"kendine giden olgu: {kendi[:5]}"
    for grup, r in ((kisi, "komsu"), (kisi, "rakip"), (okul, "kardes"),
                    (okul, "komsu"), (sehir, "kardes"), (sehir, "rakip")):
        bozuk = [x for x in grup if olgu[(olgu[(x, r)], r)] != x]
        assert not bozuk, f"{r} simetrik degil: {bozuk[:3]}"
    for (e, r) in olgu:
        assert e in G["tip"], f"bilinmeyen varlik: {e}"
        assert G["tip"][e] in SEMA[r], f"SEMA disi olgu: {G['tip'][e]} {r}"
        assert G["tip"][olgu[(e, r)]] == SEMA[r][G["tip"][e]], \
            f"TIP ihlali: {e} {r} {olgu[(e, r)]}"
    for t in TIPLER:
        gec = [r for r in ILISKI if t in SEMA[r]]
        for a in ad[t]:
            eksik = [r for r in gec if (a, r) not in olgu]
            assert not eksik, f"{a} ({t}) icin eksik olgu: {eksik}"
    return G


def kur(tohum=0):
    """`veri_okul2` (2x) grafi + sekiz alan acilimi."""
    return genislet(V2.kur(tohum), tohum)


zincirler = VO.zincirler
yaz = VO.yaz


def turetilebilir(G, esik=0.5):
    """(r1, r2) ciftinin sonucu, TEK bir r3 olgusuyla AYNI mi?

    Boyle bir cift varsa o zincir 2-hop DEGILDIR: model r3'u ezberleyip
    gecer. Var olan tek ornek `kardes`+`baba` -> `baba` (GEREKTIRIR'de
    yazili, `zincirler` bunu AYNI sinifina atip sinav disi birakiyor).
    Bu tarama YENI bir tane dogmadigini sinar.

    Donus: [(r1, r2, r3, oran)] -- oran `esik`i ASANLAR.
    """
    olgu, tip, sema = G["olgu"], G["tip"], G["sema"]
    out = []
    for r1 in ILISKI:
        for r2 in ILISKI:
            pay = {}
            top = 0
            for (e, r), b in olgu.items():
                if r != r1 or tip[b] not in sema[r2]:
                    continue
                c = olgu[(b, r2)]
                top += 1
                for r3 in ILISKI:
                    if tip[e] in sema[r3] and olgu[(e, r3)] == c:
                        pay[r3] = pay.get(r3, 0) + 1
            for r3, k in pay.items():
                if top and k / top > esik:
                    out.append((r1, r2, r3, k / top))
    return sorted(out, key=lambda x: -x[3])


if __name__ == "__main__":
    import collections
    for ad_, M in (("okul2", V2), ("okul3", __import__("veri_okul3"))):
        G = M.kur(0)
        z = zincirler(G)
        print("%s  varlik %5d  olgu %6d  zincir %7d  phi TAVANI %5.2f  |R|=%d"
              % (ad_, sum(G["n"].values()), len(G["olgu"]), len(z),
                 len(z) / len(G["olgu"]), len(G["iliski"])))
        print("     sinif", dict(collections.Counter(x[6] for x in z)))
        print("     tip basina iliski: " + "  ".join(
            "%s=%d" % (t, sum(1 for r in G["iliski"] if t in G["sema"][r]))
            for t in TIPLER))
    print()
    print("TURETILEBILIR CIFTLER (oran > 0.5):")
    for r1, r2, r3, o in turetilebilir(kur(0)):
        print("   %-10s %-10s -> %-10s  %.3f   %s"
              % (r1, r2, r3, o, "BEKLENEN (GEREKTIRIR)"
                 if (r1, r2) in GEREKTIRIR else "!! YENI"))
