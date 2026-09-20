# -*- coding: utf-8 -*-
"""olcme_13 -- model_13'un SINAVI. Bastan yazildi, kopya DEGIL.

Kullanici karari, 20 Eylul 2026: *"tabiki onceki modelden hicbirsey
kalmamali. biz kendimiz yeni seyler kurmaliyiz."*

IKI TASARIM KARARI, ikisi de model_11'den AYRILIYOR:

1. PUANLAMA KIMLIK UZERINDEN, dizge uzerinden DEGIL.
   model_11 uretilen METNI dizge olarak esliyordu. 19 Eylul'de bu
   14.123 sorunun HEPSINI 0.0000 yapti: cikti bastaki bir bosluk
   yuzunden hicbir dizgeye esitlenmedi, ve kapi bunu yakalayamadi.
   Burada cevap bir VARLIK; varligin parca dizisi ile uretilen parca
   dizisi karsilastiriliyor. Yuzey (unlu uyumu, noktalama, bosluk)
   hukme girmiyor.

2. SORU SOZLUKTEN KURULUYOR, sablon kopyalanmiyor.
   Soru birimleri varlik parcalari + iliski kelimesi + soru sozcugu +
   ekten oradaki gibi diziliyor. Kurulumun DOGRULUGU varsayilmiyor:
   `soru_kapisi` kurulan soru dizisinin KORPUSTA GERCEKTEN GECTIGINI
   sayiyor. Gecmiyorsa model bilmedigi bir dilde sorgulaniyor demektir
   ve sayilar okunmaz.
"""
from __future__ import annotations

import numpy as np

import ek_13 as EK

# Cumle sonu sayilan birimler -- uretim burada durur.
DUR = (".", "?", ",", ":")


def _parcala(kelime, kokler, bx, korunan):
    """Bir kelimeyi BIRIM ID dizisine cevirir. Bilinmeyen -> None."""
    ix = []
    for p in EK.bol(kelime, kokler, korunan=korunan):
        if p not in bx:
            return None
        ix.append(bx[p])
    return ix


def _yuzey(ad, TR):
    """Graf adi -> KORPUSUN yazdigi Turkce kelimeler.

    !! Graf ASCII (`Ibrahim_Yilmaz`), korpus gercek Turkce
    (`Ibrahim Yilmaz` -> "İbrahim Yılmaz"). Ilk surumde soru
    ASCII adla kuruluyordu ve 3.000 sorunun 2.881'i "birim sozlukte
    yok" diye DUSUYORDU. Cevrimi ureticinin KENDI tablosu yapiyor
    (`veri_13.TR`), elle harf esleme YOK."""
    return [TR.get(w, w) for w in ad.split("_")]


def _varlik_birim(ad, TR, kokler, bx, korunan):
    """`Ibrahim_Yilmaz` -> [id(Ibrahim), id(Yilmaz)]. Ek YOK, ciplak ad."""
    ix = []
    for w in _yuzey(ad, TR):
        p = _parcala(w, kokler, bx, korunan)
        if p is None:
            return None
        ix += p
    return ix


class Sinav:
    """Graf bolmesi -> birim duzeyinde soru/cevap.

    `onek`  : soru birimleri (model buradan devam eder)
    `cevap` : dogru varligin parca dizisi
    `ksy`   : r2'yi KOPRUYE degil OZNEYE uygulayinca cikan varlik; None
              ise kisayol TIP OLARAK imkansiz (`ent_yok` bolmesi).
    """

    def __init__(self, v, E_ad, ILISKI, TR, kokler, bx, birim,
                 korunan=frozenset()):
        self.v, self.E_ad, self.ILISKI, self.TR = v, E_ad, ILISKI, TR
        self.kokler, self.bx, self.birim = kokler, bx, birim
        self.korunan = korunan
        self.TAMLAYAN = bx.get("-TAMLAYAN")
        self.IYELIK = bx.get("-IYELIK")
        self.BILDIRME = bx.get("-BILDIRME")
        self.SORU = bx.get("?")
        assert None not in (self.TAMLAYAN, self.IYELIK, self.BILDIRME,
                            self.SORU), "sozlukte temel ekler YOK"
        # tip -> soru sozcugu ("hangisi" / "kim" / "neresi")
        self.soru_tip = [bx.get(v.soru_ad[i]) for i in v.soru_tip]

    def _iliski(self, r):
        """Iliski kelimesinin birimleri. Yuzey TR tablosundan."""
        ix = []
        for w in _yuzey(self.ILISKI[r], self.TR):
            p = _parcala(w, self.kokler, self.bx, self.korunan)
            if p is None:
                return None
            ix += p
        return ix

    def kur(self, zincir, adim):
        """(e, r1[, r2, kopru], cevap) -> (onek, cevap_birim, ksy_birim).

        Yuzey:  <ozne> -TAMLAYAN <r1> [-TAMLAYAN <r2>] <soru> -BILDIRME ?
        Korpusun soru cumlesiyle ayni dizilis; `soru_kapisi` dogruluyor."""
        z = [int(x) for x in zincir]
        e, rs, ans = z[0], z[1:1 + adim], z[-1]
        onek = _varlik_birim(self.E_ad[e], self.TR, self.kokler,
                             self.bx, self.korunan)
        if onek is None:
            return None
        for j, r in enumerate(rs):
            ri = self._iliski(r)
            if ri is None:
                return None
            onek = onek + [self.TAMLAYAN] + ri
        sz = self.soru_tip[int(self.v.tip[ans])]
        if sz is None:
            return None
        onek = onek + [sz, self.BILDIRME, self.SORU]
        cev = _varlik_birim(self.E_ad[ans], self.TR, self.kokler,
                            self.bx, self.korunan)
        ksy = None
        if adim == 2:
            h = int(self.v.facts[e, z[2]])
            if h >= 0 and h != ans:
                ksy = _varlik_birim(self.E_ad[h], self.TR, self.kokler,
                                    self.bx, self.korunan)
        return (onek, cev, ksy) if cev else None

    def bolme(self, zincirler, adim):
        """Bir bolmenin butun sorulari. Kurulamayan ATILIR ve SAYILIR."""
        out, atilan = [], 0
        for z in zincirler:
            s = self.kur(z, adim)
            if s is None:
                atilan += 1
            else:
                out.append(s)
        return out, atilan


def soru_kapisi(sorular, dizi, n=200):
    """KURULAN SORU KORPUSTA GECIYOR MU?

    Bu bir yetenek olcusu DEGIL, olcumun kendi sagligi. Soru dizilisini
    yanlis kurarsak model bilmedigi bir dilde sorgulanir ve butun
    sayilar YALAN olur -- ve hicbir yetenek kapisi bunu gostermez.
    `n` soru orneklenir, her birinin onek dizisi korpus akisinda
    ARANIR."""
    d = dizi.tolist() if hasattr(dizi, "tolist") else list(dizi)
    ilk = {}
    for i, t in enumerate(d):
        ilk.setdefault(t, []).append(i)
    rng = np.random.default_rng(0)
    ix = rng.choice(len(sorular), min(n, len(sorular)), replace=False)
    bulunan = 0
    for i in ix:
        o = sorular[int(i)][0]
        for p in ilk.get(o[0], ()):
            if d[p:p + len(o)] == o:
                bulunan += 1
                break
    return bulunan, len(ix)


def span(u, kapanis):
    """Uretilen birimlerden CEVAP ARALIGI: ilk kapanis birimine kadar.

    Model cumleyi surdurmeye devam eder; cevap, adin bittigi yere kadar
    olan kisim. Kapanis = ek ya da noktalama (-BILDIRME, ., ? ...)."""
    out = []
    for t in u:
        if t in kapanis:
            break
        out.append(t)
    return out


def puanla(sorular, uretilen, kapanis):
    """KIMLIK uzerinden puanlama. Yuzey (unlu uyumu, bosluk) hukme girmez.

        tam      cevap aralik BIREBIR dogru varlik
        aile     SON parca dogru, varlik yanlis   <- model_11'in arizasi
                 480 kisi / 10 soyad: aileyi bulup icinden rastgele
                 secmek 1/48 = %2,08 verir, olculen %1,89 idi.
                 Bu sutun onu DOGRUDAN sayiyor.
        kisayol  r2 kopruye degil OZNEYE uygulanmis
        bos      hic parca uretmemis  -- BOSLUGA KARSI kapi

    SIRA ONEMLI: kisayol adi dogru cevabin oneki olabilir, once tam."""
    t = a = k = b = 0
    for (onek, cev, ksy), u in zip(sorular, uretilen):
        s = span(u, kapanis)
        if not s:
            b += 1
        elif s == cev:
            t += 1
        elif ksy and s == ksy:
            k += 1
        elif cev and s[-1] == cev[-1]:
            a += 1
    n = max(1, len(sorular))
    return dict(tam=t / n, aile=a / n, kisayol=k / n, bos=b / n)


# =====================================================================
# URETIM -- model son K birime bakar, birim birim devam eder
# =====================================================================
def uret(mdl, sorular, K, oncul=(), n_yeni=8, bs=4096, dev="cuda"):
    """Her sorunun onekinden `n_yeni` birim uretir (argmax).

    !! `oncul` = soruyu ONCELEYEN GERCEK korpus parcasi. Dolgu DEGIL.
    Ilk surum pencereyi `.` ile dolduruyordu ve model egitimde arka
    arkaya on nokta HIC gormemisti: cikti dagiliyordu ("gore gore
    gore"). OLCULDU, 20 Eylul -- dolgu 10 nokta cikti dagitiyor, gercek
    baglam verilince ayni model 5/5 dogru TIP uretiyor. Yani dagilma
    modelin degil ONEGIN arizasiydi.

    Model yalniz son K birime bakiyor, onbellek gerekmez: pencereyi
    kaydirip tekrar cagirmak yeter."""
    import torch
    mdl.eval()
    oncul = list(oncul)
    cik = []
    with torch.no_grad():
        for i in range(0, len(sorular), bs):
            pen = [(oncul + s[0])[-K:] for s in sorular[i:i + bs]]
            assert all(len(p) == K for p in pen), (
                "oncul KISA -- pencere dolmuyor, dolgu YAPILMAZ")
            x = torch.as_tensor(pen, dtype=torch.long, device=dev)
            uc = []
            for _ in range(n_yeni):
                lg, _q, _C = mdl(x)
                nx = lg.argmax(-1)
                uc.append(nx)
                x = torch.cat([x[:, 1:], nx[:, None]], 1)
            cik += torch.stack(uc, 1).tolist()
    mdl.train()
    return cik


def oncul_bul(dizi, bx, K, tohum=0):
    """Korpustan GERCEK bir onculu secer: NOKTA ile biten bir parca.

    Soru gercek metinde bosta durmuyor; onunde bir cumle var. Uydurma
    dolgu yerine korpusun kendi akisindan bir dilim aliniyor, boylece
    onek dagilim ICINDE kaliyor."""
    import numpy as np
    d = dizi.tolist() if hasattr(dizi, "tolist") else list(dizi)
    nok = bx.get(".")
    yer = [i for i, t in enumerate(d[K:len(d) // 4], start=K) if t == nok]
    i = yer[np.random.default_rng(tohum).integers(len(yer))]
    return d[i - K + 1:i + 1]
