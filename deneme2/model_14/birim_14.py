# -*- coding: utf-8 -*-
"""birim_14 -- KORPUS METNI -> BIRIM AKISI -> PENCERELER.

model_13'te bu is COLAB HUCRESINDE duruyordu: surumlenmiyordu,
sinanamiyordu, ve bir kere kaybolmustu. Burada modul.

    metin  ->  kelime sayimi  ->  kok havuzu  ->  bolme  ->  BIRIM
           ->  DIZI (tek uzun akis)  ->  PENCERE (B, L)

model_14'e OZGU iki sey var, ikisi de DENKLEM.md'den:

  FREKANSTAN SINIF   En sik `k_tam` birim KAPALI SINIF sayilir ve tam
                     SO(D) alir; kalan TEK DUZLEM. Bolme etiketten ya
                     da sozlukten DEGIL, yalniz sayimdan cikar --
                     mimari genel kalsin diye (§4.3).

  OBEK YOK           Pencere akistan KEYFI yerden baslar. Semantik
                     bolme, varlik capasi, obek siniri -- hicbiri yok.
                     Pencere basindaki "cop" durum ilk capada silinir
                     (§9.5). model_13'un `oncul_bul`una gerek kalmadi.

Bu dosya DISARIYA HICBIR SEY IMPORT ETMEZ -- yalniz kol ici.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os

import numpy as np

import ek_14 as EK
import jeton_14 as J
import taban_14 as MT
import veri_14 as V


def metin_coz(X, S, n=None) -> list[str]:
    """Egitim tensoru -> metin dilimleri. EOS satir sonu olur.

    `kelime_14`ten TASINDI: o modulun 219 satirinin yalniz bu 6'si
    kullaniliyordu; gerisi model_13'un parca-koordinat makinesi ve
    elenmis BPE'siydi (elenme gerekcesi `ek_14` docstring'inde)."""
    geri = dict(S.geri)
    geri[J.EOS] = "\n"
    sat = X if n is None else X[:n]
    return ["".join(geri[int(t)] for t in r if int(t) != J.PAD)
            for r in sat]


class Birim:
    """Bir korpusun birim dunyasi.

    ad     birim adlari, indeks sirasinda        (n,)
    ix     ad -> indeks
    dizi   butun korpus, tek uzun birim akisi    (N,)
    say    birim frekansi                        (n,)
    bolme  kelime -> birim listesi
    kok    kok havuzu
    """

    def __init__(self, ad, dizi, bolme, kok, korunan, serbest=frozenset()):
        self.ad = ad
        self.ix = {b: i for i, b in enumerate(ad)}
        self.dizi = dizi
        self.bolme = bolme
        self.kok = kok
        self.korunan = korunan
        self.serbest = serbest
        self.say = np.bincount(dizi, minlength=len(ad))

    def __len__(self):
        return len(self.ad)

    def coz(self, dizi) -> str:
        return " ".join(self.ad[int(i)] for i in dizi)

    def kodla(self, kelimeler) -> list[int]:
        """Kelime dizisi -> birim indeksleri. Bilinmeyen kelime ATLANIR."""
        return [self.ix[x] for w in kelimeler
                for x in self.bolme.get(w, ()) if x in self.ix]

    # --- model_14'e OZGU -------------------------------------------
    def kapali(self, k_tam: int):
        """En sik `k_tam` birim -> TAM SO(D).  DENKLEM §4.3.

        Dilbilimsel kapali sinif (ek, noktalama, kalip, iliski sozcugu)
        yuksek frekansli olandir. Bu bir VARSAYIM ve SINANACAK (S4):
        cikan listenin gercekten ek/kalip/iliski olup olmadigina
        bakilir."""
        m = np.zeros(len(self.ad), bool)
        m[np.argsort(-self.say)[:k_tam]] = True
        return m

    def pencere(self, L: int, atla: int = 1):
        """Akistan (B, L) pencere.  OBEK YOK, kayan pencere.

        `atla`=1 her konumdan bir pencere demek; buyutmek ortusmeyi
        azaltir ama ayni olguyu daha az kez gosterir."""
        # +1: sliding_window_view len-L+1 pencere verir, sonuncusu
        # de gecerli. Onceki hal son pencereyi DUSURUYORDU.
        n = (len(self.dizi) - L) // atla + 1
        return np.lib.stride_tricks.sliding_window_view(
            self.dizi, L)[:n * atla:atla]


def iz(b: Birim) -> str:
    """Birim dunyasinin parmak izi -- model_14'un GERCEKTEN egitildigi sey.

    model_11/13'un `korpus_izi` KARAKTER tensorunun md5'iydi. model_14
    karakter gormuyor; izi birim akisi uzerinden almak gerekiyor."""
    h = hashlib.md5()
    h.update("\x00".join(b.ad).encode())
    h.update(b.dizi.astype(np.int32).tobytes())
    return h.hexdigest()[:12]


def kaydet(b: Birim, yol: str, yaz=print) -> str:
    """Birim dunyasini TEK dosyaya yaz.

    Colab'da korpus URETILMEZ; bu dosya Drive'dan okunur. Boylece hem
    ~180 sn kurulum, hem KARAKTER YOLU (jeton + 512'lik paketleme +
    164 MB onbellek) Colab'da hic kosmaz -- model_14 karakter gormuyor."""
    os.makedirs(os.path.dirname(os.path.abspath(yol)), exist_ok=True)
    np.savez_compressed(
        yol,
        ad=np.array(b.ad, object),
        dizi=b.dizi.astype(np.int32),
        bolme=np.array(json.dumps(b.bolme, ensure_ascii=False)),
        kok=np.array(json.dumps(sorted(b.kok), ensure_ascii=False)),
        korunan=np.array(json.dumps(sorted(b.korunan), ensure_ascii=False)),
        serbest=np.array(json.dumps(sorted(b.serbest), ensure_ascii=False)),
        iz=np.array(iz(b)))
    mb = os.path.getsize(yol if yol.endswith(".npz") else yol + ".npz") / 1e6
    yaz(f"birim KAYDEDILDI: {yol}  {mb:.0f} MB  iz {iz(b)}")
    return iz(b)


def yukle(yol: str, yaz=print) -> Birim:
    """Kaydedilmis birim dunyasini oku. Iz DOSYADAN degil yeniden
    hesaplanir ve karsilastirilir -- bozuk dosya sessizce gecmesin."""
    z = np.load(yol, allow_pickle=True)
    b = Birim([str(x) for x in z["ad"]], z["dizi"].astype(np.int64),
              json.loads(str(z["bolme"])), set(json.loads(str(z["kok"]))),
              frozenset(json.loads(str(z["korunan"]))),
              frozenset(json.loads(str(z["serbest"]))))
    bek = str(z["iz"])
    assert iz(b) == bek, f"BIRIM DOSYASI BOZUK: {iz(b)} != {bek}"
    yaz(f"birim YUKLENDI: {len(b.ad)} birim  dizi {len(b.dizi):,}  iz {bek}")
    return b


def kur(ayar, v=None, yaz=print, onbellek: str | None = None) -> Birim:
    """Korpustan birim dunyasini kurar.

    `onbellek` verilirse once ORADAN okunur; yoksa kurulup oraya
    yazilir. Colab bu yolu Drive'daki `model_14/` klasorune verir ve
    korpus ORADA URETILMEZ.

    Bolme OGRENILMIYOR, BILINIYOR: Turkce ekler kapali bir kume
    (`ek_14`). BPE denendi ve `Bahcelievler`i `B|ah|c|eli|ev|l|er`
    diye dogradi -- istatistik, dilbilgisi degil."""
    if onbellek and os.path.exists(onbellek):
        return yukle(onbellek, yaz)
    if v is None:
        v = MT.veri_kur(ayar, yaz=yaz)
    X, S = MT.egitim_havuzu(ayar, v, yaz=yaz)
    metin = metin_coz(X, S)

    say = collections.Counter()
    for s in metin:
        say.update(s.split())

    # KORUNAN: ozel adlarin yuzey bicimleri. Ciplak -a/-e dusurulmustu
    # ama "Kaya -> Kay" turu kazalar icin ikinci kemer.
    _yuz = lambda a: [V.TR.get(w, w) for w in a.split("_")]
    tohum = tuple(w for p in v.par_ad[0] for w in _yuz(p))
    # !! `_trb`: buyuk/kucuk KORUNUR. `_tr` kucuge indirdigi icin ozel
    # ad `Bolumu` ortak ad `bolumu`yu da koruyordu ve `bolumunun`
    # bolunemiyordu.
    korunan = frozenset(EK._trb(t) for t in tohum)
    serbest = EK.serbest_kokler(say)     # korpusta TEK BASINA gecenler

    kok = EK.kok_havuzu(say, tohum=tohum)
    bolme = {w: EK.bol(w, kok, korunan=korunan, serbest=serbest) for w in say}
    ad = sorted({x for p in bolme.values() for x in p})
    ix = {b: i for i, b in enumerate(ad)}
    dizi = np.fromiter((ix[x] for s in metin for w in s.split()
                        for x in bolme[w]), np.int64)

    yaz(f"birim: {len(say):,} kelime -> {len(ad)} BIRIM "
        f"({len(say)/len(ad):.2f}x)   dizi {len(dizi):,}")
    b = Birim(ad, dizi, bolme, kok, korunan, serbest)
    if onbellek:
        kaydet(b, onbellek, yaz)
    return b


# =====================================================================
# KAPI -- kurulumun kendi denetimi.  Sayilar model_13 ile TUTMALI.
# =====================================================================
def kapi(b: Birim) -> str:
    """model_13'un Colab kosusunda olculen sayilar:
         1.269 kelime -> 475 BIRIM (2,67x)   dizi 15.198.500
    Bunlar VERI kararina bagli; kayarlarsa korpus degismis demektir."""
    assert len(b.ad) == len(set(b.ad)), "birim adi tekrarliyor"
    assert b.dizi.max() < len(b.ad) and b.dizi.min() >= 0, "indeks tasmasi"
    assert (b.say > 0).all(), (
        f"{int((b.say == 0).sum())} birim akista HIC gecmiyor")
    tekrar = sorted(zip(b.say, b.ad))[-5:]
    return (f"birim {len(b.ad)}   dizi {len(b.dizi):,}\n"
            f"       en sik: " + "  ".join(f"{a}({s:,})" for s, a in
                                           reversed(tekrar)))
