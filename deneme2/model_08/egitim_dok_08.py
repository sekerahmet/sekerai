# -*- coding: utf-8 -*-
"""egitim_dok_08 — VERININ KENDISI, OKUNAN metin.

Kullanici karari, 18 Eylul 2026. Onceki surum SILINDI ve bastan
yazildi; istenen duzen:

    00 icindekiler   01 jetonlar   02 varliklar   03 iliskiler
    04 ornekler      (her iliskiden 10 cumle)
    05 egitim hop1   06 egitim hop2
    07 egitim FIM hop1   08 egitim FIM hop2
    09 sorular       10 sinav      11 sinav (ikincil yuzeyler)

    "text en basinda bu egitim nasil yapiliyor anlatan bir kisim
     sonra da aciklamasiz duz ornekler ama her tipten 100 tane
     limiti, aralarinda ---tip 1 aciklama ---- diye ayrilsin."

ONCEKI SURUMDEN IKI YAPISAL FARK
--------------------------------

1) ACIKLAMA DOSYANIN ICINDE. Eski kural *"dosyalarin icinde aciklama
   YOK, ne aradigin 00_ICINDEKILER'de"* idi ve 40+ numarali dosya
   uretiyordu. Kullanici o dosyalari tek tek acip "bu nasil bir soru?"
   diye sormak zorunda kaldi -- yani ayrim ISE YARAMADI: veriyi okuyan
   kisi aciklamayi BASKA BIR DOSYADA aramiyor, orada ariyor.

2) !! SATIRLAR YENIDEN URETILMIYOR, HAVUZDAN DILIMLENIYOR.
   Eskiden dokum `kodla_*`i KENDISI cagirip havuzu taklit ediyordu.
   BOSLUK DOLDURMA (FIM) satirlarinda bu FIILEN KAYDI ve olculdu:

       havuz FIM blogu        221.346 satir
       eski dokumun urettigi  221.346 satir
       KESISIM                 98.054
       dokumde olup havuzda OLMAYAN  123.292   (%56)

   Sebep: `egitim_havuzu` parcalari [1hop_b0, 2hop_b0, 1hop_b1, ...]
   sirasiyla tuketiyor, eski dokum ise [1hop_b0, 1hop_b1, 1hop_b2,
   2hop_b0, ...] sirasiyla. Ayni tohum, BASKA sira -> baska konumlar.
   Eski dokumun kendi yorumu bu tuzagi TARIF EDIYORDU ("ayni sirayla
   ilerlenmezse bosluk KONUMLARI havuzdan kayar") ve yine de yanlis
   siralanmisti -- yorum kapi degildir.

   Artik `Havuz` sinifi `egitim_havuzu`nun DONDURDUGU diziyi diliyor.
   Taklit yok, dolayisiyla kayma da yok. `Havuz.kapi()` her uretimde
   dilimlerin dogru yere dustugunu MEKANIK olarak dogrular (FIM icin
   gidis-donus: bosluk geri kapatilinca kaynak satir CIKMALI).

SINAV bolmeleri havuzda YOK (tutuluyorlar) -- onlar `pencere_08`nin
cagirdigi AYNI kodlayicilarla kuruluyor.

    python egitim_dok_08.py [--klasor <yol>] [--limit N]
"""
from __future__ import annotations

import argparse
import io
import os
import sys

_K = os.path.dirname(os.path.abspath(__file__))
if _K not in sys.path:
    sys.path.insert(0, _K)

import numpy as np                                             # noqa: E402
import taban_08 as M                                           # noqa: E402
import analiz_08 as AZ                                         # noqa: E402

NL = chr(10)
TAB = chr(9)

LIMIT = 100        # her TIPten kac ornek  (kullanici karari)
LIMIT_ORNEK = 10   # 04_ornekler: her ILISKIDEN kac cumle


# ===================== ILISKI NOTLARI ====================================
# Tek satirlik anlam notu. TURETILEMEZ (semantik iddia), o yuzden burada
# duruyor -- ama ANAHTARLARI `veri_08.ILISKI` ile birebir tutmak ZORUNDA:
# asagidaki `assert` bir iliski eklenince/silinince dokumu DUSURUR.
# (Bu kapi olmasaydi yeni bir iliski sessizce notsuz dokulurdu.)
_H = "HIYERARSI — yukari (cok -> bir)"
_G = "GOREV — kurumdan KISI'ye"
_A = "AILE — gercek soy agaci, cinsiyet ve kusak tutarli"
_P = "KISI'nin geri kalani"
_T = "TEZ"
_Y = "YATAY"

ILISKI_NOT = {
    "bolumu":       (_H, "Dersi ACAN bolum / kisinin bagli oldugu bolum."),
    "fakultesi":    (_H, "Bolumun bagli oldugu fakulte."),
    "universitesi": (_H, "Fakultenin bagli oldugu universite."),
    "sehri":        (_H, "Universitenin bulundugu il. Tablodan, gercek "
                         "yerlesim."),
    "bolgesi":      (_H, "Ilin cografi bolgesi. Tablodan, gercek cografya."),
    "rektoru":      (_G, "Universitenin rektoru."),
    "kurucusu":     (_G, "Universitenin kurucusu. En yasli kusaktan secilir."),
    "dekani":       (_G, "Fakultenin dekani."),
    "baskani":      (_G, "Bolum baskani."),
    "hocasi":       (_G, "Dersi veren ogretim uyesi."),
    "valisi":       (_G, "Ilin valisi."),
    "annesi":       (_A, "Hep kadin, tam bir ust kusak."),
    "babasi":       (_A, "Hep erkek, tam bir ust kusak."),
    "kardesi":      (_A, "Simetrik, ters cinsiyet, ayni kusak, ayni anne VE "
                         "ayni baba."),
    "cocugu":       (_A, "annesi/babasi TERSI. Baba oglunu, anne kizini "
                         "gosterir."),
    "danismani":    (_P, "Sabit noktasiz, ayni aileden OLMAYAN."),
    "ogrencisi":    (_P, "danismani TERSI (birebir)."),
    "arkadasi":     (_P, "Simetrik; kardesi/annesi/babasi/cocugu ile "
                         "CAKISMAZ."),
    "memleketi":    (_P, "Dogdugu il."),
    "yasadigi_yer": (_P, "Oturdugu il. Memleketten FARKLI olmak zorunda."),
    "tezi":         (_P, "Kisinin tezi. Her kisinin bir tezi var."),
    "yazari":       (_T, "tezi TERSI (birebir)."),
    "konusu":       (_T, "Tezin konusu olan ders."),
    "onkosulu":     (_Y, "Dersin on kosulu. DONGUSUZ -- her ders kendinden "
                         "onceki birini alir."),
}
# ILISKILERIN OKUMA SIRASI: sema gruplari, `SEMA`nin kendi sirasi degil.
ILISKI_SIRA = (_H, _G, _A, _P, _T, _Y)

# SINAV BOLMESI -> (okunur ad, amac, karar kurali).
# Anahtarlar `olcme_listeleri`nin anahtarlari.
SINAV_NOT = {
    "one": ("one — TEK ADIMLI OLGU",
            "Tek adimlik olgu, EGITIMDE GORULMUS.",
            "AMAC: SAGLIK KAPISI, one >= 0.98. Gecmezse hicbir sey "
            "yorumlanmaz."),
    "seen": ("seen — GORULMUS 2-HOP (ezber)",
             "Egitimde GORULMUS 2-hop zinciri.",
             "AMAC: SAGLIK KAPISI, seen >= 0.95. Modelin ezberleyip "
             "ezberlemedigi."),
    "comp": ("comp — GORULMEMIS UCLU (bilesim)",
             "(varlik, r1, r2) UCLUSU egitimde gorulmemis.",
             "AMAC: OLGUNLUK, comp >= 0.50. Altindaysa ENT YORUMLANMAZ."),
    "ent": ("ent — VARLIK HIC ZINCIR BASI OLMAMIS",
            "Varlik egitimde poz 1'de HIC gorunmemis. Olgularinda, kopru "
            "ve cevap olarak var; soru BASI olarak yok.",
            "AMAC: HUKUM BURADA VERILIR. Kolun asil sayisi."),
    "ent_yok": ("ent_yok — KISAYOL TIP OLARAK IMKANSIZ",
                "`ent` ile AYNI varlik havuzu, AYRIK zincirler (kesisim 0): "
                "r2 ozneye tip olarak uygulanamaz.",
                "AMAC: BIRIM TESTI, ent_yok_kisayol == 0.000 olmali."),
    "ent_kati": ("ent_kati — VARLIK HICBIR ZINCIRDE YOK",
                 "Varlik HICBIR egitim zincirinde gecmemis -- ne bas, ne "
                 "kopru, ne cevap. Wang 2405.15071'in OOD'si.",
                 "AMAC: IKINCIL."),
    "ood": ("ood — IKI KENAR DA DAGITIM DISI",
            "Zincirin IKI kenari da egitimdeki hicbir zincirde gecmiyor "
            "(Wang'in test_inferred_OOD'si). Bas varlik BASKA kenarlariyla "
            "zincir basi OLMUS -- tutulan sey VARLIK degil KENAR.",
            "AMAC: IKINCIL."),
}

# OLCU AILESI -> 11'deki BOLUM BASLIGI. `pencere_08`nin urettigi her
# ikincil olcunun burada bir karsiligi olmali.
#
# !! NEDEN VAR: ayni hata UC KEZ yapildi (kullanici, 17 Eylul).
#   1) `soru_` olculuyordu, dosyasi YOKTU   -> "bu soru degil ki"
#   2) `fim_` olculuyordu, dosyasi YOKTU    -> "bosluk doldurmada ent nerede"
#   3) `ek_` olculuyordu, dosyasi YOKTU     -> "verileri ona gore uretmedin ki"
# Ucunde de olcu vardi, BAKILACAK SEY yoktu. Artik kod SORUYOR.
OLCU_BOLUM = {
    "soru_":      "SORU BICIMI",
    "fim_ozne1_": "BOSLUK DOLDURMA - OZNE",
    "fim_rel_":   "BOSLUK DOLDURMA - ILISKI",
    "ek_":        "BILDIRME EKI",
}


def _sus(*a, **k):
    pass


def _s(n):
    """Turkce binlik ayraci."""
    return f"{int(n):,}".replace(",", ".")


def _y(x):
    """Turkce ondalik ayraci -- yuzde icin."""
    return f"{x:.1f}".replace(".", ",")


def _sec(n, k):
    """n satirin icinden k tanesinin INDEKSI -- BASTAN DEGIL, esit arali.

    Bastan almak siralamanin kendisini kopyalar: `v.one` tipe ve ada
    gore sirali, yani ilk 100 satir hep ayni birkac varligi ve hep ayni
    iliskiyi gosterirdi. Esit aralikli secim hem butun araligi gezer hem
    DETERMINISTIK (tohum gerekmez, dosya iki kosuda ayni cikar)."""
    if k <= 0 or n <= k:
        return list(range(n))
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


# ===================== HAVUZ DILIMLERI ===================================
class Havuz:
    """`egitim_havuzu`nun DONDURDUGU diziyi etiketli dilimlere ayirir.

    !! HICBIR SATIRI YENIDEN URETMEZ. Bu sinifin butun isi, blok
    sinirlarini `egitim_havuzu`nun kurulum sirasindan hesaplamak:

        for b in range(bicim):  1hop  ,  2hop          <- duz bildirim
        FIM: parca SIRASIYLA, kaynak satir basina fim_kat satir
        SORU: soru_kat x (1hop, 2hop)
        IYELIK KISA YOLU: kisayol_kat x kisayol
        KIMLIK: n_ent x tekrar   (havuzun payi ident_frac olana kadar)

    Siralama degisirse `kapi()` duser -- sessizce kaymaz."""

    def __init__(self, ayar, v):
        self.ayar, self.v = ayar, v
        self.X = M.egitim_havuzu(ayar, v, yaz=_sus)[0]
        bic = max(1, ayar.bicim)
        n1, n2 = len(v.one), len(v.tr2)
        self.bic = bic
        self.dilim = {}
        duz = []
        i = 0
        for b in range(bic):
            for ad, n in (("hop1", n1), ("hop2", n2)):
                anahtar = f"{ad}_b{b}"
                self.dilim[anahtar] = (i, i + n)
                duz.append((anahtar, i, i + n))
                i += n
        self.duz_son = i
        fk = getattr(ayar, "fim_kat", 0)
        self.fim_kat = fk
        if fk:
            # FIM blogu duz blogun AYNI SIRASINI takip eder ve her kaynak
            # satirdan fim_kat ARDISIK satir uretir -> dilim carpimla.
            for anahtar, a, b2 in duz:
                self.dilim["fim_" + anahtar] = (i + a * fk, i + b2 * fk)
            i += self.duz_son * fk
        sk = getattr(ayar, "soru_kat", 0)
        if sk:
            # soru_kat tekrari BIREBIR AYNI satirlar (kodla_soru
            # deterministik) -- ilk turu etiketliyoruz, gerisini atliyoruz.
            self.dilim["soru_hop1"] = (i, i + n1)
            self.dilim["soru_hop2"] = (i + n1, i + n1 + n2)
            i += (n1 + n2) * sk
        kk = getattr(ayar, "kisayol_kat", 0)
        nk = len(v.kisayol)
        if kk and nk:
            self.dilim["kisayol"] = (i, i + nk)
            i += nk * kk
        self.kimlik_tekrar = 0
        if ayar.ident_frac > 0:
            self.dilim["kimlik"] = (i, i + v.n_ent)
            self.kimlik_tekrar = (len(self.X) - i) // v.n_ent
            i = len(self.X)
        assert i == len(self.X), (
            f"blok toplami {i} != havuz {len(self.X)} -- `egitim_havuzu`nun "
            "kurulum sirasi DEGISMIS, dilimler artik yanlis yerde")

    def al(self, anahtar):
        a, b = self.dilim[anahtar]
        return self.X[a:b]

    def say(self, anahtar):
        a, b = self.dilim[anahtar]
        return b - a

    # --- DILIMLEME DOGRU MU ---------------------------------------------
    def kapi(self, yaz=print):
        """Dilimler gercekten iddia ettikleri satirlar mi?

        Dogrudan kodlayiciyla KARSILASTIRIR (uretmek icin degil,
        DOGRULAMAK icin). FIM'de karsilastirilacak kodlayici yok --
        orada GIDIS-DONUS: boslugu geri kapatinca kaynak satir cikmali.
        """
        v, ayar = self.v, self.ayar
        n = 0
        for b in range(self.bic):
            assert np.array_equal(self.al(f"hop1_b{b}"),
                                  M.kodla_1hop(v, v.one, b)[0]), f"hop1_b{b}"
            assert np.array_equal(self.al(f"hop2_b{b}"),
                                  M.kodla_2hop(v, v.tr2, b)[0]), f"hop2_b{b}"
            n += 2
        if "soru_hop1" in self.dilim:
            assert np.array_equal(self.al("soru_hop1"),
                                  M.kodla_soru(v, v.one, 1)[0]), "soru_hop1"
            assert np.array_equal(self.al("soru_hop2"),
                                  M.kodla_soru(v, v.tr2, 2)[0]), "soru_hop2"
            n += 2
        if "kisayol" in self.dilim:
            assert np.array_equal(self.al("kisayol"),
                                  M.kodla_soru(v, list(v.kisayol), 1)[0]), \
                "kisayol"
            n += 1
        if "kimlik" in self.dilim:
            assert np.array_equal(
                self.al("kimlik"),
                M.kodla_kimlik_q1(v, range(v.n_ent))[0]), "kimlik"
            n += 1
        # --- FIM GIDIS-DONUS, HER DILIM ICIN AYRI ---
        #
        # !! ILK SURUM BU KONTROLU FIM BLOGUNUN TABANINDAN yapiyordu
        # (`self.X[self.duz_son + f]`) ve `self.dilim`e HIC BAKMIYORDU.
        # Mutasyonla olculdu: `fim_hop1_b0` dilimini BIR SATIR kaydirmak
        # kapiyi DUSURMUYORDU -- oysa dokulen sey tam olarak o dilim.
        # Kapi artik `al()` uzerinden, yani dokumun OKUDUGU satirlari
        # dogruluyor.
        fk = self.fim_kat
        if fk:
            for b in range(self.bic):
                for h in (1, 2):
                    kay_a, kay_b = self.dilim[f"hop{h}_b{b}"]
                    F = self.al(f"fim_hop{h}_b{b}")
                    assert len(F) == (kay_b - kay_a) * fk, (
                        f"fim_hop{h}_b{b}: {len(F)} satir, kaynakta "
                        f"{kay_b - kay_a} x {fk} bekleniyordu")
                    kotu, denendi = 0, 0
                    for i in _sec(len(F), 120):
                        dz = [int(t) for t in F[i] if int(t) != M.PAD]
                        if v.bosluk not in dz or v.ayir not in dz:
                            kotu += 1
                            continue
                        p, j = dz.index(v.bosluk), dz.index(v.ayir)
                        geri = dz[:p] + dz[j + 1:] + dz[p + 1:j]
                        kaynak = self.X[kay_a + i // fk]
                        denendi += 1
                        if geri != [int(t) for t in kaynak
                                    if int(t) != M.PAD]:
                            kotu += 1
                    assert denendi and not kotu, (
                        f"fim_hop{h}_b{b} gidis-donus: {kotu}/{denendi} "
                        "satir kaynagina DONMUYOR -- dilim kaydi")
                    n += 1
        yaz(f"  havuz dilimleme kapisi GECTI: {n} dilim dogrulandi "
            f"(FIM gidis-donus dahil)")


# ===================== SATIR ACIKLAYICI ==================================
def _ayrim(d, v, X, P=None, T=None, i=0, fim=False):
    """Bir satir tipi icin (VERILEN, ISTENEN, OLCULEN).

    Elle yazilmiyor: diziyi ve `kodla_*`in dondurdugu P/T'yi okuyor.

    !! `tam_kayip=True` -- KAYIP HER KONUMDA hesaplaniyor. Burasi
    kaybin sekli DEGIL, satirin NE OGRETTIGI: olculen cevap yuvasi.
    Devrik bicimlerde o yuva YOK (cevap kendi baglamindan once gelir),
    ve bu bir kusur degil, TANIM."""
    r = [int(t) for t in X[i] if int(t) != M.PAD]
    if fim:
        j = r.index(v.ayir)
        return d.oku(r[:j + 1]), d.oku(r[j + 1:]), d.oku(r[j + 1:])
    if T is not None and int((np.asarray(T[i]) >= 0).sum()) > 0:
        p0 = int(P[i][0])
        hedef = [int(t) for t in np.asarray(T[i]) if int(t) >= 0]
        return (d.oku(r[:p0 + 1]), d.oku(r[p0 + 1:]), d.oku(hedef))
    return ("(cumlenin tamami)",
            "HER KONUMDA sonraki jeton",
            "YOK -- cevap yuvasi maskeli "
            "(cevap kendi baglamindan ONCE geliyor)")


def _ayrim_blok(ayrim):
    """(VERILEN, ISTENEN, OLCULEN) -> aciklama satirlari."""
    ver, ist, olc = ayrim
    out = [f"VERILEN : {ver}", f"ISTENEN : {ist}"]
    if olc != ist:
        out.append(f"OLCULEN : {olc}")
    return out


# ===================== DOSYA YAZICI ======================================
class Bolum:
    """Bir TIP: baslik + aciklama + satirlar."""

    def __init__(self, ad, aciklama, satirlar, toplam=None):
        self.ad = ad
        self.aciklama = ([aciklama] if isinstance(aciklama, str)
                         else list(aciklama))
        self.satirlar = list(satirlar)
        self.toplam = len(satirlar) if toplam is None else toplam


def _baslik_cizgisi(no, ad):
    s = f"---- TIP {no}: {ad} "
    return s + "-" * max(4, 72 - len(s))


def dosya_yaz(yol, baslik, giris, bolumler, limit, etiket):
    """Bir dosya: BASLIK + GIRIS + (ayrac + aciklama + duz satirlar)*.

    !! `etiket` BEDAVA DEGIL. Ilk surumde her basliga "havuzda N satir"
    yaziliyordu -- SINAV dosyalarinda da. Sinav zincirleri havuzda
    OLMADIGI icin (tutuluyorlar, kolun butun sorusu bu) o etiket
    dosyanin en ustunde yazan seyle CELISIYORDU. Etiket artik dosya
    basina veriliyor: havuzda / olculen / veride."""
    with io.open(yol, "w", encoding="utf-8", newline="") as f:
        f.write(baslik + NL)
        f.write("=" * 72 + NL + NL)
        for ln in giris:
            f.write(ln + NL)
        f.write(NL)
        n_yaz = 0
        for no, b in enumerate(bolumler, 1):
            gos = b.satirlar if limit <= 0 else [
                b.satirlar[i] for i in _sec(len(b.satirlar), limit)]
            f.write(NL + _baslik_cizgisi(no, b.ad) + NL)
            f.write(f"     {etiket} {_s(b.toplam)} satir"
                    f"   ·   asagida {_s(len(gos))}" + NL)
            for ln in b.aciklama:
                f.write("     " + ln + NL)
                    # (aciklama SATIRLARIN ICINDE degil, BASLIKTA --
                    #  kullanici: "satirlarda bunlara gerek yok")
            f.write("-" * 72 + NL)
            for x in gos:
                f.write(x + NL)
            n_yaz += len(gos)
    return n_yaz


# ===================== DOSYALAR ==========================================
def kur_01_jetonlar(d, v, H):
    def _tr(i):
        # `oku()` CUMLE BASI buyuk harf yapar; tek jetonluk bir listede
        # bu YANLIS okunur ("bolumu" -> "Bölümü", oysa jeton cins isim).
        w = d.jeton_ad(i)
        return d.VM.TR_ILISKI.get(w) or d.VM.TR.get(w, w)

    kume = [("OZEL", 0, M.SPECIAL, "noktalama ve dolgu"),
            ("ILISKI", M.SPECIAL, M.SPECIAL + v.n_rel,
             "cins isim, hep KUCUK harf"),
            ("AD KELIMELERI", v.yuva_ara[0][0], v.ek0,
             "varlik adlarinin kelimeleri, hep BUYUK harf"),
            ("EK ve ISARET", v.ek0, v.vocab,
             "tamlayan/bildirme allomorflari, soru sozcukleri, <BOS>/<AYIR>")]
    bol = []
    for ad, a, b, not_ in kume:
        bol.append(Bolum(ad, [not_, "bicim:  numara" + TAB + "jeton" + TAB
                              + "turkce yazimi"],
                         [f"{i}{TAB}{d.jeton_ad(i)}{TAB}{_tr(i)}"
                          for i in range(a, b)]))
    giris = [
        f"Modelin gordugu BUTUN jetonlar: {_s(v.vocab)} tane.",
        "",
        "Varlik TEK KELIME DEGIL -- jeton dizisi. 'Ahmet Aydin' modelin",
        "gordugu IKI ayri jeton; kimlik CIFTTE. Sozluk PAYLASILAN, yani",
        "ayni kelime hangi yuvada gecerse gecsin ayni jeton.",
        "",
        "Ekler AYRI JETON ve allomorflari da ayri: Turkce'de tamlayan eki",
        "sekiz bicimde (-in/-in/-un/-un, sesliden sonra -nin/...). Hangisinin",
        "gelecegi ONCEKI KELIMEDEN belirli, yani YENI BILGI TASIMAZ -- dilin",
        "gercek yuzeyi oldugu icin var.",
    ]
    return "01 — JETONLAR (SOZLUK)", giris, bol


def kur_02_varliklar(d, v, H):
    bol = []
    for ti, tad in enumerate(v.tip_ad):
        es = [e for e in range(v.n_ent) if int(v.tip[e]) == ti]
        if not es:
            continue
        uz = sorted({d.uzunluk(e) for e in es})
        bol.append(Bolum(
            tad,
            [f"{_s(len(es))} varlik   ·   ad uzunlugu {uz} jeton"],
            [d.ad(e) for e in es]))
    giris = [
        f"{_s(v.n_ent)} varlik, {len(v.tip_ad)} tip.",
        "",
        "Adlar YUVALARDAN kuruluyor (ad + soyad, ya da tek kelime).",
        "Bir tipin butun uyeleri ayni sayida yuva kullanmak ZORUNDA DEGIL:",
        "iller tek kelime, kisiler iki kelime.",
    ]
    return "02 — VARLIKLAR", giris, bol


def kur_03_iliskiler(d, v, H):
    eksik = set(ILISKI_NOT) ^ set(d.R)
    assert not eksik, (
        f"ILISKI_NOT ile veri_08.ILISKI TUTMUYOR: {sorted(eksik)} "
        "-- iliski eklenmis/silinmis, notu da yazilmali")
    sayim = {}
    for _e, r, _a in v.one:
        sayim[int(r)] = sayim.get(int(r), 0) + 1
    bol = []
    for grup in ILISKI_SIRA:
        sat = []
        for ri, r in enumerate(d.R):
            g, not_ = ILISKI_NOT[r]
            if g != grup:
                continue
            imza = ", ".join(f"{k} -> {y}" for k, y in d.VM.SEMA[r].items())
            sat.append(f"{d.VM.TR_ILISKI[r]}{TAB}{r}{TAB}{imza}"
                       f"{TAB}{sayim.get(ri, 0)} olgu{TAB}{not_}")
        if sat:
            bol.append(Bolum(grup, [], sat))
    giris = [
        f"{_s(v.n_rel)} iliski, toplam {_s(len(v.one))} olgu.",
        "",
        "bicim:  turkce" + TAB + "kod adi" + TAB + "tip imzasi" + TAB
        + "olgu sayisi" + TAB + "not",
        "",
        "!! HER (varlik, iliski) CIFTININ TEK HEDEFI VAR. Bu bir eksiklik",
        "degil, SINAVIN SARTI: 'X'in dersinin hocasi kimdir?' sorusunun tek",
        "dogru cevabi yoksa comp/ent dogrulugu TANIMSIZ olur. Dolayisiyla",
        "'derse kayitli ogrenciler' gibi cok-cok baglar bu dilin DISINDA",
        "kaliyor; dil, dunyanin TEK DEGERLI diliminden kuruluyor.",
    ]
    return "03 — ILISKILER (SEMA)", giris, bol


def kur_04_ornekler(d, v, H):
    a0 = H.dilim["hop1_b0"][0]
    gore = {}
    for i, (_e, r, _a) in enumerate(v.one):
        gore.setdefault(int(r), []).append(i)
    bol = []
    for grup in ILISKI_SIRA:
        for ri, r in enumerate(d.R):
            g, not_ = ILISKI_NOT[r]
            idx = gore.get(ri, [])
            if g != grup or not idx:
                continue
            imza = ", ".join(f"{k} -> {y}" for k, y in d.VM.SEMA[r].items())
            bol.append(Bolum(
                f"{d.VM.TR_ILISKI[r]}  ({imza})",
                [grup, not_],
                [d.oku(H.X[a0 + i]) for i in idx]))
    giris = [
        "Her iliskiden ornek cumleler -- kanonik sirada (bicim 0).",
        "",
        "Cumleler HAVUZDAN okunuyor, yeniden uretilmiyor: burada ne",
        "goruyorsan egitimde o var.",
        "",
        f"Her iliskiden {LIMIT_ORNEK} cumle gosteriliyor.",
    ]
    return "04 — HER ILISKIDEN ORNEK CUMLELER", giris, bol


_BICIM_AD = ("KANONIK SIRA", "YUKLEM BASTA", "ILISKI BASTA")
_BICIM_NOT = (
    ["Ozne basta, cevap sonda. OLCUM HEP BU BICIMDE yapilir.",
     "\"Ahmet Aydin'in kardesi Elif Aydin'dir.\""],
    ["Yuklem one alinmis. Cevap cumlenin BASINDA, yani soldan tahmin",
     "EDILEMEZ -- cevap yuvasi olcumu bu bicimde MASKELI. Satir yine",
     "tam kayipla egitiliyor: isi TERS YON, cevaptan ozneyi cikarmak.",
     "\"Elif Aydin'dir Ahmet Aydin'in kardesi.\""],
    ["Iliski one alinmis, cevabin ardindan VIRGUL (sinir isareti).",
     "Cevap yuvasi olcumu burada da MASKELI, ayni sebeple.",
     "\"Kardesidir Elif Aydin, Ahmet Aydin'in.\""],
)


def _duz_giris(d, v, H, ayar, hop):
    n = len(v.one) if hop == 1 else len(v.tr2)
    duz = n * H.bic
    pay = 100.0 * duz / len(H.X)
    ayrim = _ayrim(d, v, *(M.kodla_1hop if hop == 1
                           else M.kodla_2hop)(v, (v.one if hop == 1
                                                  else v.tr2)[:1], 0))
    g = [
        "BU EGITIM NASIL YAPILIYOR",
        "",
    ]
    if hop == 1:
        g += [
            "Bir OLGU: (varlik, iliski) -> varlik. Grafta " + _s(len(v.one))
            + " olgu var ve",
            "hepsi EGITIME giriyor -- tutulan (held-out) OLGU YOK. Tutulan",
            "sey ZINCIR: sinav, tek tek olgulari bilen ama onlari",
            "BIRLESTIREMEYEN modeli yakalamak icin kuruldu.",
        ]
    else:
        g += [
            "Iki adimli bir gercek, TEK CUMLEDE:",
            "",
            "    Ahmet Aydin --bolumu--> Etimesgut Tarih Bolumu   (KOPRU)",
            "    Etimesgut Tarih Bolumu --fakultesi--> Cayirova Fen Fakultesi",
            "    => \"Ahmet Aydin'in bolumunun fakultesi Cayirova Fen "
            "Fakultesi'dir.\"",
            "",
            "!! KOPRU CUMLEDE GECMIYOR -- ne oznede ne cevapta. Model onu",
            "YAZMADAN kullanmak zorunda. Bu projenin BUTUN sorusu bu.",
            "",
            _s(len(v.tr2)) + " zincir egitimde; sinav zincirleri bunlarin"
            " DISINDA (tutuluyor).",
        ]
    g += [
        "",
        "AYNI OLGU UC FARKLI TURKCE SIRADA yaziliyor. Sebep: Turkce'de oge",
        "sirasi serbest, grameri EK tasiyor ('in sahip, -dir yuklem). Tek",
        "sirada egitilen model rolu KONUMDAN okur; sira degisince ekten",
        "okumak ZORUNDA kalir. Olculdu (model_05): bir varliktan sonra gelen",
        "jeton cesidi 2 idi ve `ent` 0.0423'te kaldi.",
        "",
        "MODELE NE VERILIYOR, NE ISTENIYOR  (kanonik bicim icin)",
    ]
    g += ["  " + x for x in _ayrim_blok(ayrim)]
    g += [
        "",
        "Kayip HER KONUMDA hesaplaniyor (tam_kayip=True) -- yani satirin",
        "tamami egitiliyor. 'OLCULEN' yalniz `dogruluk()`un nereye baktigini",
        "soyler: ad jetonlari + sinir isareti. Sondaki bildirme eki (-dir) ve",
        "nokta EGITILIYOR ama birincil olcude maskeli, cunku adin son",
        "unlusu belli olunca -dir zaten belli (olculdu: 73 farkli son jeton,",
        "cakisma 0). `pencere_08`nin `ek_` sutunu onu AYRICA olcer.",
        "",
        f"SATIR SAYISI   {_s(n)} olgu x {H.bic} bicim = {_s(duz)} satir"
        f"   (havuzun %{_y(pay)}'i)",
        "",
        "Asagidaki ornekler havuzdan ESIT ARALIKLI secildi -- bastan degil,",
        "cunku havuz tipe ve ada gore sirali ve ilk N satir hep ayni birkac",
        "varligi gosterirdi.",
    ]
    return g


def _kur_duz(d, v, H, ayar, hop):
    bol = []
    for b in range(H.bic):
        X = H.al(f"hop{hop}_b{b}")
        bol.append(Bolum(_BICIM_AD[b], _BICIM_NOT[b],
                         [d.oku(r) for r in X]))
    return (f"0{4 + hop} — EGITIM: {hop}-HOP BILDIRIM",
            _duz_giris(d, v, H, ayar, hop), bol)


def kur_05_egitim_hop1(d, v, H, ayar):
    return _kur_duz(d, v, H, ayar, 1)


def kur_06_egitim_hop2(d, v, H, ayar):
    return _kur_duz(d, v, H, ayar, 2)


def _fim_giris(d, v, H, ayar, hop):
    n = len(v.one) if hop == 1 else len(v.tr2)
    fim = n * H.bic * H.fim_kat
    pay = 100.0 * fim / len(H.X)
    ornek = _ayrim(d, v, H.al(f"fim_hop{hop}_b0"), fim=True)
    return [
        "BU EGITIM NASIL YAPILIYOR",
        "",
        f"YENI CUMLE YOK. 0{4 + hop}'teki AYNI cumleler, bir jetonu <BOS> ile",
        "bosaltilmis.",
        "",
        "Model NEDENSEL: boslugun SAGINI o konumda goremez. O yuzden",
        "cikarilan jeton cumlenin SONUNA, <AYIR>dan sonra yaziliyor -- model",
        "once cumlenin TAMAMINI gorur, sonra eksik parcayi yazar. Standart",
        "fill-in-the-middle duzeni.",
        "",
        "    Fatma Yilmaz'in annesi Ayse Yilmaz'dir.",
        "    Fatma <BOS>'in annesi Ayse Yilmaz'dir. <AYIR> Yilmaz",
        "",
        f"Her bildirim satirindan {H.fim_kat} varyant uretiliyor. Bosaltilan",
        "konum tohumlu bir uretecle secilir (`default_rng(1000 + veri_tohum)`),",
        "her konum ESIT OLASILIKLA. Bosluk HER ZAMAN TEK JETON.",
        "",
        "NEDEN: olculdu (model_05, 1-hop kanonik satir) -- soldan bakinca",
        "soyadin tabani 1,866 nat, iliskininki 2,016 nat; ikisi de",
        "OGRENILEMEZ. Cumlenin TAMAMI gorulunce ikisi de belirli hale",
        "geliyor. Bu kolun iddiasi o 4,4 nat'i ogrenilebilir ise cevirmek.",
        "",
        "<BOS> girdide GORUNUR ama HEDEF DEGIL (kayipta PAD'e cevriliyor):",
        "olculdu, <BOS>'un YERINI tahmin etmek kaybin %75'i olurdu ve",
        "ogrenilemezdi.",
        "",
        "MODELE NE VERILIYOR, NE ISTENIYOR",
    ] + ["  " + x for x in _ayrim_blok(ornek)] + [
        "",
        f"SATIR SAYISI   {_s(n)} x {H.bic} bicim x {H.fim_kat} varyant"
        f" = {_s(fim)} satir   (havuzun %{_y(pay)}'i)",
        "",
        "!! SORU satirlari (09) ve KISA YOL satirlari BOSLUK DOLDURMAYA",
        "GIRMIYOR -- bilerek: FIM'in isi dil bilgisi (ek, sira, sinir) ve onu",
        "duz bildirimin satirlari zaten ogretiyor.",
    ]


def _kur_fim(d, v, H, ayar, hop):
    bol = []
    for b in range(H.bic):
        X = H.al(f"fim_hop{hop}_b{b}")
        bol.append(Bolum(
            _BICIM_AD[b] + " uzerinde bosluk",
            [f"Kaynak: 0{4 + hop} TIP {b + 1}. Bosluk her satirda TEK jeton,",
             "konumu tohumlu uretecten."],
            [d.oku(r) for r in X]))
    return (f"0{6 + hop} — EGITIM: BOSLUK DOLDURMA, {hop}-HOP CUMLELERI",
            _fim_giris(d, v, H, ayar, hop), bol)


def kur_07_fim_hop1(d, v, H, ayar):
    return _kur_fim(d, v, H, ayar, 1)


def kur_08_fim_hop2(d, v, H, ayar):
    return _kur_fim(d, v, H, ayar, 2)


def kur_09_sorular(d, v, H, ayar):
    bol = []
    n_soru = 0
    if "soru_hop1" in H.dilim:
        a1 = _ayrim(d, v, *M.kodla_soru(v, v.one[:1], 1))
        bol.append(Bolum(
            "SORU, TEK ADIMLI",
            ["\"Ahmet Aydin'in kardesi kimdir?  Elif Aydin'dir.\""]
            + _ayrim_blok(a1),
            [d.oku(r) for r in H.al("soru_hop1")],
            toplam=H.say("soru_hop1") * ayar.soru_kat))
        a2 = _ayrim(d, v, *M.kodla_soru(v, v.tr2[:1], 2))
        bol.append(Bolum(
            "SORU, IKI ADIMLI",
            ["\"Ahmet Aydin'in danismaninin arkadasi kimdir?  "
             "Derya Yilmaz'dir.\"",
             "Kopru yine YAZILMIYOR -- 06 ile ayni gorev, baska yuzey."]
            + _ayrim_blok(a2),
            [d.oku(r) for r in H.al("soru_hop2")],
            toplam=H.say("soru_hop2") * ayar.soru_kat))
        n_soru = H.say("soru_hop1") + H.say("soru_hop2")
    if "kisayol" in H.dilim:
        bol.append(Bolum(
            "IYELIK KISA YOLU  (yuzey 1 adim, cevap COK adimli)",
            ["\"Ayse Sahin'in dekani kimdir?  Melek Aydin'dir.\"",
             "Turkce iyelik aidiyet kenarlarindan SESSIZCE gecer: okudugum",
             "bolumun fakultesinin dekani, kisa yoldan 'benim dekanim' olur.",
             "Dort hedef: fakultesi, konusu, dekani, universitesi.",
             "!! GRAFA KENAR EKLEMEZ -- `olgu`ya girmez, zincirler gormez,",
             "`olcme_izi` DEGISMEZ. Yalniz egitim havuzuna satir ekler."],
            [d.oku(r) for r in H.al("kisayol")],
            toplam=H.say("kisayol") * ayar.kisayol_kat))
    if "kimlik" in H.dilim:
        ak = _ayrim(d, v, *M.kodla_kimlik_q1(v, range(1)))
        bol.append(Bolum(
            "KIMLIK KOPRUSU  (sifir adim)",
            ["\"Ahmet Aydin kimdir?  Ahmet Aydin'dir.\"",
             "arXiv 2509.24653'un identity bridge'i: e -> e ozdeslik",
             "eslemesi. Graf bilgisi KULLANMAZ -- ne olgu, ne cevap, ne",
             "kopru; yalniz varligin kendi jetonlari.",
             f"{_s(v.n_ent)} BENZERSIZ satir, havuzda x{H.kimlik_tekrar} "
             f"tekrarli (pay %{100 * ayar.ident_frac:.0f})."]
            + _ayrim_blok(ak),
            [d.oku(r) for r in H.al("kimlik")],
            # !! TOPLAM = TEKRARLI sayi. Benzersiz satir 1.608, ama havuzda
            # payi `ident_frac` olana kadar TEKRARLANIYOR; "havuzda 1.608"
            # demek havuzun %20'sini 1.608 satir gostermek olurdu.
            toplam=v.n_ent * H.kimlik_tekrar))
    ksy = H.say("kisayol") * ayar.kisayol_kat if "kisayol" in H.dilim else 0
    giris = [
        "BU EGITIM NASIL YAPILIYOR",
        "",
        "Ayni olgular ve ayni zincirler, bu sefer SORU + KISA CEVAP.",
        "",
        "UC TASARIM KARARI:",
        "",
        "1) CEVAP KISA -- soru TEKRAR EDILMIYOR. `model_05` soruyu sorup",
        "   ardindan cumlenin TAMAMINI tekrar yaziyordu (21 jeton). O tekrar",
        "   modele 9 fazladan konum veriyor ve o konumlarin kosullu entropisi",
        "   0,000 -- yani gradyan uretmiyorlar. Kisa cevap `t_len`e DOKUNMAZ.",
        "",
        "2) 'kimdir', 'kim' DEGIL. Olculmus bir arizayi da duzeltiyor:",
        "   `model_06`da soru sozcugu 70.655/70.655 kez KIMLIK satirinda",
        "   geciyordu, yani 'kimdir?' dilde 'adi tekrar yaz' demekti. Artik",
        "   ayni jeton ikiliyi ONCESINDEKI jeton ayiriyor:",
        "       ... Yilmaz kimdir?     -> KIMLIK   (varliktan sonra)",
        "       ... arkadasi kimdir?   -> SORU     (iliskiden sonra)",
        "",
        "3) Soru sozcugu CEVABIN TIPINDEN (kim / neresi / hangisi). Yeni",
        "   bilgi tasimaz -- cevabin tipi zaten iliskiden belli (olculdu:",
        "   24/24 iliskide hedef tip TEK). Dilin dogal parcasi oldugu icin",
        "   var, gorevi kolaylastirmak icin degil.",
        "",
        "!! SINAV BUNU KULLANMIYOR. Hukum DUZ BILDIRIMLE veriliyor",
        "(`kodla_2hop`, bicim 0). Soru bicimi YENI bir yuzey; birincil",
        "yapmak 'hangi olcu iyi ciktiysa onu sectik' durumu olurdu. Ayri",
        "adla (`soru_`) ve ayri tabloda olculuyor -- verisi 11'de.",
        "",
        f"SATIR SAYISI   soru {_s(n_soru)}   ·   iyelik kisa yolu {_s(ksy)}"
        + (f"   ·   kimlik {_s(H.say('kimlik') * H.kimlik_tekrar)}"
           if "kimlik" in H.dilim else ""),
    ]
    if v.kisayol_atilan:
        t = sum(n for _, n in v.kisayol_atilan)
        giris += [
            "",
            f"!! KISA YOLDA {_s(t)} SATIR URETILMEDI: tutulan (held-out)",
            "veriye dokunuyordu. \"X'in fakultesi = Y\" satiri tam olarak",
            "\"X'in bolumunun fakultesi?\" sorusunun cevabidir -- o zincir",
            "sinavdaysa egitim cevabi ELDEN VERIR. Uc kapi ayri ayri bakiyor:",
            "     " + ", ".join(f"{k} {_s(n)}" for k, n in v.kisayol_atilan),
        ]
    return "09 — EGITIM: SORULAR, KISA YOL, KIMLIK", giris, bol


def kur_10_sinav(d, v, H, ayar):
    bol = []
    for bad, lst in d.L.items():
        if not lst:
            continue
        kodla = M.kodla_1hop if bad == "one" else M.kodla_2hop
        X, P, T = kodla(v, list(lst), 0)
        ad, amac, karar = SINAV_NOT[bad]
        bol.append(Bolum(
            ad,
            [amac, karar] + _ayrim_blok(_ayrim(d, v, X, P, T)),
            [d.oku(r) for r in X]))
    giris = [
        "SINAV — HUKUM BURADA VERILIR",
        "",
        "Satirda TAM CUMLE var, cunku sinav dizisi odur. Modele yalniz",
        "cumlenin BASI veriliyor, gerisini O yaziyor:",
        "",
        "    satir : Kaan Arslan'in annesinin tezi Niceliksel Optik'tir.",
        "    SORU  : Kaan Arslan'in annesinin tezi ______",
        "    CEVAP : Niceliksel Optik'tir.",
        "",
        "KOPRU CUMLEDE GECMIYOR:",
        "    Kaan Arslan --annesi--> Tugce Arslan   (bu ad satirda YOK)",
        "    Tugce Arslan --tezi--> Niceliksel Optik",
        "",
        "Olcum HEP bicim 0 (kanonik sira) ile yapilir -- egitim daha cok",
        "yuzey gorur, sinav TEK yuzeydir.",
        "",
        "KAPILAR (gevsetilmez):",
        "    SAGLIK-1HOP    one  >= 0.98",
        "    SAGLIK-EZBER   seen >= 0.95",
        "    OLGUNLUK       comp >= 0.50     altindaysa ENT YORUMLANMAZ",
        "    BIRIM TESTI    ent_yok_kisayol == 0.000",
        "",
        "!! BOLMELER BIRBIRININ ALT KUMESI DEGIL. `ent` ile `ent_yok` ayni",
        "varlik havuzundan gelir ama ZINCIR KUMELERI AYRIK (kesisim 0) --",
        "\"ent_yok dustu ama ent yukseldi\" bir CELISKI DEGILDIR.",
        "",
        "!! `comp` \"bu iliski CIFTINI hic gormedi\" DEMEK DEGIL. Bolme kodu",
        "ZINCIR tutuyor, cift tutmuyor: comp orneklerinin (r1,r2) cifti",
        "egitimde de var. Cok daha zayif bir genelleme sinavi, ve arsivdeki",
        "comp degerleri de bu anlamda okunmali.",
        "",
        "Veride bir de `ent_arama` bolmesi var (maske/esik aramasi icin",
        f"ayrilmis, {_s(len(v.ent_arama))} zincir). HUKUMDE KULLANILMAZ ve",
        "burada DOKULMEZ -- `olcme_listeleri` onu dondurmuyor.",
        "",
        f"Bolme basina en fazla {_s(ayar.n_olcum_max)} zincir olculur",
        "(`n_olcum_max`); asagidaki sayilar o olculen kumenin sayilaridir.",
    ]
    return "10 — SINAV", giris, bol


def kur_11_sinav_ikincil(d, v, H, ayar):
    bol, bolumler = [], set()

    def ekle(baslik, amac, satirlar, ayrim=None):
        bolumler.add(baslik.split("  ")[0])
        bol.append(Bolum(baslik, list(amac)
                         + (_ayrim_blok(ayrim) if ayrim else []), satirlar))

    # --- SORU BICIMI ---
    if getattr(ayar, "soru_kat", 0) > 0:
        for bad in ("one", "seen", "comp", "ent", "ent_yok", "ood"):
            lst = d.L.get(bad)
            if not lst:
                continue
            X, P, T = M.kodla_soru(v, list(lst), 1 if bad == "one" else 2)
            ekle(f"SORU BICIMI  ·  {bad}",
                 [SINAV_NOT[bad][1],
                  "AYNI zincir, soru yuzeyi. `pencere_08` -> soru_" + bad],
                 [d.oku(r) for r in X], _ayrim(d, v, X, P, T))
    # --- BOSLUK DOLDURMA ---
    if getattr(v, "bosluk", 0):
        for yon, baslik, not_ in (
                ("ozne", "BOSLUK DOLDURMA - OZNE",
                 "Oznenin SON jetonu (cok kelimeli adlarda SOYAD) bosaltildi."),
                ("iliski", "BOSLUK DOLDURMA - ILISKI",
                 "Birinci iliskinin jetonu bosaltildi.")):
            for bad in ("comp", "ent", "ent_yok", "ood"):
                lst = d.L.get(bad)
                if not lst:
                    continue
                X = M.kodla_fim_sinav(v, list(lst), yon)[0]
                anahtar = ("fim_ozne1_" if yon == "ozne" else "fim_rel_") + bad
                ekle(f"{baslik}  ·  {bad}",
                     [SINAV_NOT[bad][1], not_,
                      "HER IKI YON DE TEK JETON bosaltir -- egitimdeki "
                      "bosluklarin",
                      "hepsi tek jetonluk, sinav ondan farkli olursa gorev "
                      "DAGITIM DISI",
                      "olur ve dusuk sayinin nereden geldigi AYRILAMAZ.",
                      "`pencere_08` -> " + anahtar],
                     [d.oku(r) for r in X], _ayrim(d, v, X, fim=True))
    # --- BILDIRME EKI ---
    for bad in ("one", "seen", "comp", "ent"):
        lst = d.L.get(bad)
        if not lst:
            continue
        kodla = M.kodla_1hop if bad == "one" else M.kodla_2hop
        X, P, T = kodla(v, list(lst), 0)
        son_j = (T >= 0).sum(1) - 1
        poz = P[np.arange(len(P)), np.maximum(son_j, 0)] + 1
        sat = []
        for i in range(len(X)):
            r = [int(t) for t in X[i] if int(t) != M.PAD]
            p = int(poz[i])
            if p + 1 >= len(r):
                continue
            sat.append(d.oku(r[:p + 1]) + TAB + d.jeton_ad(r[p + 1]))
        if not sat:
            continue
        _o, _e = sat[0].split(TAB)
        ekle(f"BILDIRME EKI  ·  {bad}",
             [SINAV_NOT[bad][1],
              "Iki sutun, TAB ile ayrik:  onek" + TAB + "beklenen ek",
              "Ad yazildiktan sonra DOGRU allomorf geliyor mu (unlu uyumu).",
              "`pencere_08` -> ek_" + bad],
             sat, (_o, _e, _e))
    giris = [
        "IKINCIL YUZEYLER — HUKUM VERMEZ",
        "",
        "10'daki AYNI ZINCIRLER, uc baska yuzeyde. Neden ayri dosya: hukum",
        "10'da veriliyor, cunku kollar ancak duz bildirim yuzeyinde AYNI",
        "TABLODA okunabiliyor. Buradaki sayilar eklentilerin kendi isini",
        "gosterir.",
        "",
        "    10  BILDIRIM   Kaan Arslan'in annesinin tezi Niceliksel "
        "Optik'tir.",
        "    11  SORU       ... tezi hangisidir? Niceliksel Optik'tir.",
        "    11  BOSLUK     Kaan <BOS>'in annesinin tezi ... <AYIR> Arslan",
        "    11  EK         Kaan Arslan'in ... Niceliksel Optik'  ->  tir",
        "",
        "!! BEKLENEN GERILIM, ONCEDEN YAZILIYOR: soru satiri EGITIMDE var,",
        "ama sinavin `ent` kismi icin ZINCIR BASI olarak YOK -- `ent` tanimi",
        "yuzeyden BAGIMSIZ. Yani `soru_ent` ile `ent` arasindaki fark,",
        "yuzeyin TEK BASINA ne getirdigini soyler.",
        "",
        "BILDIRME EKI neden ayri olculuyor: birincil olcu -dir'i maskeliyor,",
        "cunku adin son jetonundan deterministik (73 son jeton, cakisma 0).",
        "Ama o determinizm MATEMATIKTE var; MODELDE olup olmadigi ayri bir",
        "soru, ve bu kolun tezi \"modele DILI ogretecegiz\".",
    ]
    _olcu_kapisi(bolumler)
    return "11 — SINAV, IKINCIL YUZEYLER", giris, bol


# ===================== KAPILAR ===========================================
def _olcu_kapisi(bolumler, yaz=print):
    """`pencere_08` hangi ikincil olculeri uretiyor -- hepsinin BOLUMU var mi?

    Kaynagi OKUYARAK buluyor: `r[f"<aile>_{...}"]` kaliplari. Modeli
    yuklemeye gerek yok, ve yeni bir olcu eklenince BURASI patlar."""
    import re
    kaynak = io.open(os.path.join(_K, "pencere_08.py"),
                     encoding="utf-8").read()
    # !! [A-Za-z] -- yalniz kucuk harf arayan bir desen, sinama sirasinda
    # UYDURMA_ adli sahte olcuyu KACIRDI (17 Eylul).
    aile = set(re.findall(r'r\[f"([A-Za-z0-9_]+_)\{', kaynak))
    aile -= {"fim_ozne_"}          # ESKI ad, artik uretilmiyor
    eksik = []
    for a in sorted(aile):
        baslik = OLCU_BOLUM.get(a)
        if baslik is None:
            eksik.append(f"{a}  -> OLCU_BOLUM tablosunda YOK")
        elif baslik not in bolumler:
            eksik.append(f"{a}  -> '{baslik}' bolumu 11'de URETILMEDI")
    if eksik:
        yaz("!! OLCU KAPISI DUSTU -- olculen ama DOKULMEYEN sey var:")
        for e in eksik:
            yaz("     " + e)
        raise AssertionError("olculen her sey DOKULMELI: " + "; ".join(eksik))
    yaz(f"  olcu kapisi GECTI: {len(aile)} ikincil olcu ailesinin "
        f"{len(aile)}'sinin bolumu var")


def _havuz_kapisi(H, v, ayar, yaz=print):
    """DOKUMUN TOPLADIGI SATIR SAYISI HAVUZUN KENDISIYLE TUTUYOR MU."""
    bic = max(1, ayar.bicim)
    n1, n2 = len(v.one), len(v.tr2)
    bek = {
        "duz":    ((n1 + n2) * bic,
                   sum(H.say(f"hop{h}_b{b}")
                       for h in (1, 2) for b in range(bic))),
        "bosluk": ((n1 + n2) * bic * H.fim_kat,
                   sum(H.say(f"fim_hop{h}_b{b}")
                       for h in (1, 2) for b in range(bic))),
        "soru":   ((n1 + n2) * getattr(ayar, "soru_kat", 0),
                   (H.say("soru_hop1") + H.say("soru_hop2"))
                   * getattr(ayar, "soru_kat", 0)
                   if "soru_hop1" in H.dilim else 0),
    }
    tamam = True
    for ad, (b, g) in bek.items():
        iyi = (b == g)
        tamam &= iyi
        yaz(f"  {'GECTI ' if iyi else '!! BOZUK'} {ad:<8} "
            f"havuz {b:>9,}   dilim {g:>9,}")
    assert tamam, "dilimler havuzdan kaydi -- yukariya bak"


# ===================== MAIN ==============================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--klasor", default=None)
    ap.add_argument("--limit", type=int, default=LIMIT,
                    help=f"her TIPten en fazla N satir (varsayilan {LIMIT}, "
                         "0 = HEPSI)")
    a = ap.parse_args()
    kl = a.klasor or os.path.join(_K, "veri", "model_08")
    os.makedirs(kl, exist_ok=True)

    print("veri kuruluyor ...")
    d = AZ.Dok("model_08")
    v, ayar = d.v, d.ayar
    print(f"  graf izi {d.VM.IZ}   olcme izi {M.olcme_izi(d.L)}")
    print("egitim havuzu kuruluyor ...")
    H = Havuz(ayar, v)
    print(f"  havuz {_s(len(H.X))} satir, t_len {v.t_len}")
    H.kapi()
    _havuz_kapisi(H, v, ayar)

    # (numara, ad, kurucu, limit)  -- TEK TABLO, numaralarda BOSLUK YOK
    # (numara, ad, kurucu, limit, SAYI ETIKETI)
    # !! ETIKET SUTUNU BEDAVA DEGIL -- 10/11 SINAV dosyalari ve sinav
    # zincirleri HAVUZDA YOK. "havuzda N satir" demek, dosyanin kendi
    # girisiyle celisen bir iddia olurdu.
    _HV, _OL, _VR = "havuzda", "olculen", "veride"
    DOSYALAR = [
        ("01", "jetonlar",        lambda: kur_01_jetonlar(d, v, H), 0, _VR),
        ("02", "varliklar",       lambda: kur_02_varliklar(d, v, H), 0, _VR),
        ("03", "iliskiler",       lambda: kur_03_iliskiler(d, v, H), 0, _VR),
        ("04", "ornekler",        lambda: kur_04_ornekler(d, v, H),
         LIMIT_ORNEK, _VR),
        ("05", "egitim_hop1",     lambda: kur_05_egitim_hop1(d, v, H, ayar),
         a.limit, _HV),
        ("06", "egitim_hop2",     lambda: kur_06_egitim_hop2(d, v, H, ayar),
         a.limit, _HV),
        ("07", "egitim_fim_hop1", lambda: kur_07_fim_hop1(d, v, H, ayar),
         a.limit, _HV),
        ("08", "egitim_fim_hop2", lambda: kur_08_fim_hop2(d, v, H, ayar),
         a.limit, _HV),
        ("09", "sorular",         lambda: kur_09_sorular(d, v, H, ayar),
         a.limit, _HV),
        ("10", "sinav",           lambda: kur_10_sinav(d, v, H, ayar),
         a.limit, _OL),
        ("11", "sinav_ikincil",   lambda: kur_11_sinav_ikincil(d, v, H, ayar),
         a.limit, _OL),
    ]
    icerik = []
    for no, ad, kur, lim, etiket in DOSYALAR:
        baslik, giris, bol = kur()
        dosya = f"{no}_{ad}.txt"
        yol = os.path.join(kl, dosya)
        n = dosya_yaz(yol, baslik, giris, bol, lim, etiket)
        tam = sum(b.toplam for b in bol)
        icerik.append((dosya, baslik, len(bol), n, tam,
                       os.path.getsize(yol), etiket))
        print(f"yazildi: {dosya:<24} {len(bol):>2} tip  "
              f"{n:>6,} satir (toplam {tam:>9,})  "
              f"{os.path.getsize(yol) / 1024:>7.0f} KB")

    # --- TAMLIK KAPISI ---------------------------------------------------
    # EGITIM dosyalarinin (05..09) TOPLAMLARI havuzun TAMAMINI kapatiyor mu.
    #
    # !! BU KAPI "DOKUMDE EKSIK YOK" DEMEKTIR ve elle sayilamaz. Havuzda
    # bir satir tipi daha olsaydi (ornegin BELGE satirlari geri gelseydi)
    # hicbir dosyada gorunmezdi ve bunun tek belirtisi burasi olurdu.
    egitim = sum(t for dsy, _b, _n, _s2, t, _by, _e in icerik
                 if dsy[:2] in ("05", "06", "07", "08", "09"))
    assert egitim == len(H.X), (
        f"!! DOKUM HAVUZU KAPATMIYOR: dosyalar {egitim:,} satir anlatiyor, "
        f"havuz {len(H.X):,}. Fark {len(H.X) - egitim:,} satir HICBIR "
        "DOSYADA gorunmuyor.")
    print(f"  tamlik kapisi GECTI: 05..09 toplami {_s(egitim)} = havuz")

    # --- 00_ICINDEKILER --------------------------------------------------
    yol = os.path.join(kl, "00_ICINDEKILER.txt")
    with io.open(yol, "w", encoding="utf-8", newline="") as f:
        f.write("model_08 — VERI DOKUMU" + NL)
        f.write("=" * 72 + NL + NL)
        f.write(f"graf izi    {d.VM.IZ}" + NL)
        f.write(f"olcme izi   {M.olcme_izi(d.L)}" + NL)
        f.write(f"t_len {v.t_len}   sozluk {_s(v.vocab)}   "
                f"varlik {_s(v.n_ent)}   iliski {v.n_rel}   "
                f"olgu {_s(len(v.one))}" + NL)
        f.write(f"egitim havuzu {_s(len(H.X))} satir" + NL + NL)
        f.write("HER DOSYA: once BU EGITIM/SINAV NASIL YAPILIYOR, sonra" + NL)
        f.write("'---- TIP n: ... ----' ayraclariyla ayrilmis DUZ satirlar."
                + NL)
        f.write("Satirlarin KENDISINDE aciklama yok." + NL + NL)
        f.write("-" * 72 + NL)
        for dosya, baslik, n_tip, n_sat, tam, boy, etiket in icerik:
            f.write(f"{dosya:<24} {baslik}" + NL)
            f.write(f"{'':<24} {n_tip} tip  ·  dosyada {_s(n_sat)} satir"
                    f"  ·  {etiket} {_s(tam)}  ·  {boy / 1024:.0f} KB" + NL)
        f.write("-" * 72 + NL + NL)
        f.write("ORNEKLER KIRPIK." + NL)
        f.write(f"Her tipten en fazla {a.limit if a.limit else 'HEPSI'} satir "
                "gosteriliyor (04'te " + str(LIMIT_ORNEK) + ")." + NL)
        f.write("Secim ESIT ARALIKLI, bastan degil -- havuz tipe ve ada gore"
                + NL)
        f.write("sirali, ilk N satir hep ayni birkac varligi gosterirdi."
                + NL)
        f.write("Tamamini dokmek icin:  python egitim_dok_08.py --limit 0"
                + NL + NL)
        f.write("EGITIM = 05..09.  SINAV = 10 (hukum) ve 11 (ikincil)." + NL)
        f.write("Ikisi CAKISMAZ: sinav zincirleri havuzda YOK -- 10/11'deki"
                + NL)
        f.write("sayilar o yuzden 'olculen', 'havuzda' DEGIL." + NL + NL)
        f.write("TAMLIK: 05..09'un toplami " + _s(egitim) + " = egitim "
                "havuzunun TAMAMI." + NL)
        f.write("Yani havuzda bu dosyalarda gorunmeyen satir tipi YOK; "
                "kod her" + NL)
        f.write("uretimde bunu assert ediyor." + NL + NL)
        f.write("!! SATIRLAR HAVUZDAN DILIMLENIYOR, YENIDEN URETILMIYOR."
                + NL)
        f.write("`egitim_havuzu`nun dondurdugu dizinin ta kendisi okunuyor."
                + NL)
        f.write("Onceki surum kendi kopyasini uretiyordu ve BOSLUK DOLDURMA"
                + NL)
        f.write("satirlarinin %56'si havuzda OLMAYAN satirlardi (olculdu)."
                + NL)
        f.write("`Havuz.kapi()` her uretimde dilimleri dogruluyor." + NL)
    print(f"yazildi: 00_ICINDEKILER.txt")
    print(f"{NL}klasor: {kl}")


if __name__ == "__main__":
    main()
