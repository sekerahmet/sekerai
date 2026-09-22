# -*- coding: utf-8 -*-
"""birim_16 -- KORPUS METNI -> BIRIM AKISI -> PENCERELER.

model_13'te bu is COLAB HUCRESINDE duruyordu: surumlenmiyordu,
sinanamiyordu, ve bir kere kaybolmustu. Burada modul.

    metin  ->  kelime sayimi  ->  kok havuzu  ->  bolme  ->  BIRIM
           ->  DIZI (tek uzun akis)  ->  PENCERE (B, L)

model_16'e OZGU iki sey var, ikisi de DENKLEM.md'den:

  FREKANSTAN SINIF   En sik `k_tam` birim KAPALI SINIF sayilir ve tam
                     SO(D) alir; kalan TEK DUZLEM. Bolme etiketten ya
                     da sozlukten DEGIL, yalniz sayimdan cikar --
                     mimari genel kalsin diye (§4.3).

  OBEK YOK           Pencere akistan KEYFI yerden baslar. Semantik
                     bolme, varlik capasi, obek siniri -- hicbiri yok.
                     Pencere basindaki durum COP kalir: "ilk capada
                     silinir" iddiasi olculdu ve yanlis cikti, kayip
                     onu puanliyor (DENKLEM.md §5.1/G, acik A3).

Bu dosya DISARIYA HICBIR SEY IMPORT ETMEZ -- yalniz kol ici.

ZINCIRDEKI YERI.  Kim kimi cagiriyor, bu dosya nerede:
(model BIRIM goruyor -- karakter yalnizca ara adim ve kapi)

  veri_16     graf: 1608 varlik, 24 iliski, olgular
  metin_16    graf -> duz Turkce cumle
  korpus_16   cumle -> belge -> paketlenmis akis
  jeton_16    KARAKTER sozlugu + GIDIS-DONUS KAPISI
  birim_16    metin -> sayim -> kok havuzu -> BIRIM AKISI -> pencere   <-- BU DOSYA
  ek_16       kelime -> kok + ek    (Turkce morfolojisi)

  taban_16    bolmeler (ezber_* / cikarim_*) + Ayar tanimi
  ayar_16     dugmeler
  hazirla_16  veriyi dosyaya yazar, Colab Drive'dan OKUR

  model_16    MIMARI -- model_15'ten
  train_16    egitim dongusu
  olcme_16    olcu: soru soruldu, cevap dogru mu
"""
from __future__ import annotations

import collections
import hashlib
import json
import os

import numpy as np

import ek_16 as EK
import taban_16 as MT
import veri_16 as V


SINIR = "<belge>"     # BELGE SINIRI -- sozlukte AYRI birim.
#   Karakter yolunda <EOS> vardi ama birim cevriminde DUSUYORDU:
#   metin_coz <EOS>'u chr(10) yapiyordu, `s.split()` onu BOSLUK sayip
#   atiyordu.  Akista belge siniri kalmiyor, pencerelerin %12,9'u iki
#   ayri varligin sayfasini birlestiriyordu.  Sozluge girer cunku model
#   "burada sayfa bitti"yi GORMELI.


def belgeler(ayar, v=None, yaz=print):
    """KORPUS -> Belge listesi.  512'lik paketleme YOK, karakter YOK.

    `korpus_16.havuz()` dort grubu (bildirim / soru / kimlik / reddetme)
    AYRI paketleyip arka arkaya ekliyordu.  OLCULDU: akisin ilk %40'inda
    soru orani %0,54, sonrasinda %6,64 -- bir varligin olgulari ile o
    varliga sorulan sorular ~6M birim uzakta kaliyor ve ayni pencerede
    ASLA bulusmuyorlardi.  Burada dort grup TEK havuzda karisiyor."""
    import korpus_16 as KP
    if v is None:
        v = MT.veri_kur(ayar, yaz=yaz)
    G = V.kur(ayar.veri_tohum)
    bb, bs = KP.sayfalar(v, G, ayar.kopya, ayar.tohum, ayar.tetik,
                         ayar.t_len, ayar.zincir_pay, ayar.n3, yaz)
    kim = KP.kimlik_belgeleri(v, G, tohum=ayar.tohum, yaz=yaz)
    _ns = sum(len(b.cumle) for b in bs)
    red = KP.reddetme_belgeleri(
        v, G, int(_ns * ayar.ret_pay), ayar.tohum,
        bolme=KP.reddetme_bolme(G, ayar.ret_tut, ayar.tohum),
        yaz=yaz) if ayar.ret_pay else []
    yaz(f"  belge {len(bb):,} bildirim + {len(bs):,} soru + "
        f"{len(kim):,} kimlik + {len(red):,} reddetme")
    return bb + bs + kim + red, KP.cakisan_ciftler(v)


def sirala(bel, yasak, tohum=0, yaz=print):
    """Belgeleri KARISTIR, cakisan ikisini YAN YANA koyma.

    Eski kural "ayni 512'lik dilime dusemez" idi.  Paketleme kalkinca
    karsiligi "birim penceresi menziline dusemez" oluyor; menzil belge
    cinsinden tutuluyor cunku bir belge zaten bir pencereden uzun."""
    rs = np.random.default_rng(9000 + tohum)
    sira = [int(i) for i in rs.permutation(len(bel))]
    if not yasak:
        return sira
    GERI, ILERI = 4, 64
    son, yer, kalan, cakis = [], [], sira, 0
    while kalan:
        sec = 0
        for j in range(min(ILERI, len(kalan))):
            e = bel[kalan[j]].e
            if e < 0 or not any((e, x) in yasak for x in son):
                sec = j
                break
        else:
            cakis += 1
        i = kalan[sec]
        kalan = kalan[:sec] + kalan[sec + 1:] if sec else kalan[1:]
        yer.append(i)
        son = (son + [bel[i].e])[-GERI:]
    yaz(f"  siralama {len(yer):,} belge   {cakis:,} yerde cakisma "
        f"kacinilamadi (menzil {GERI} belge)")
    return yer


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

    # --- model_16'e OZGU -------------------------------------------
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
    """Birim dunyasinin parmak izi -- model_16'un GERCEKTEN egitildigi sey.

    model_11/13'un `korpus_izi` KARAKTER tensorunun md5'iydi. model_16
    karakter gormuyor; izi birim akisi uzerinden almak gerekiyor."""
    h = hashlib.md5()
    h.update("\x00".join(b.ad).encode())
    h.update(b.dizi.astype(np.int32).tobytes())
    return h.hexdigest()[:12]


def kaydet(b: Birim, yol: str, yaz=print) -> str:
    """Birim dunyasini TEK dosyaya yaz.

    Colab'da korpus URETILMEZ; bu dosya Drive'dan okunur. Boylece hem
    ~180 sn kurulum, hem KARAKTER YOLU (jeton + 512'lik paketleme +
    164 MB onbellek) Colab'da hic kosmaz -- model_16 karakter gormuyor."""
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
    yazilir. Colab bu yolu Drive'daki `model_16/` klasorune verir ve
    korpus ORADA URETILMEZ.

    Bolme OGRENILMIYOR, BILINIYOR: Turkce ekler kapali bir kume
    (`ek_16`). BPE denendi ve `Bahcelievler`i `B|ah|c|eli|ev|l|er`
    diye dogradi -- istatistik, dilbilgisi degil."""
    if onbellek and os.path.exists(onbellek):
        return yukle(onbellek, yaz)
    if v is None:
        v = MT.veri_kur(ayar, yaz=yaz)
    bel, yasak = belgeler(ayar, v, yaz)
    metin = [bel[i].metin for i in sirala(bel, yasak, ayar.tohum, yaz)]

    say = collections.Counter()
    for s in metin:
        say.update(s.split())

    # OZEL AD yuzeyleri -- HAM, cevrilmemis. `ek_16` bunlari ham
    # dizeyle karsilastiriyor; `_trb` ile ASCII'ye indirilirse hicbiri
    # eslesmez ve ozel adlar korunmaz.
    # !! GRAFTAKI BUTUN adlarin butun parcalari. Once yalniz
    # `par_ad[0]` alinıyordu; cok kelimeli adlarin kalan parcalari
    # disarida kaliyordu.
    _yuz = lambda a: [x for w in a.split("_") for x in V.TR.get(w, w).split()]
    ozel = frozenset(w for t in V.TIPLER for a in V.kur(ayar.veri_tohum)["ad"][t]
                     for w in _yuz(a))

    kok = EK.kok_havuzu(say)             # artik ELLE: KOK | BUTUN
    bolme = {w: EK.bol(w, kok, korunan=ozel) for w in say}
    iyi, eksik = EK.denetle(say, ozel=ozel)
    assert not eksik, (
        "%d kelime BILINEN KOK + BILINEN EK'e cozulemiyor -- `ek_16.KOK` "
        "ya da `BUTUN` eksik: %s" % (len(eksik), sorted(eksik)[:10]))
    korunan, serbest = ozel, frozenset()
    ad = sorted({x for p in bolme.values() for x in p} | {SINIR})
    ix = {b: i for i, b in enumerate(ad)}
    # Her belgenin SONUNA sinir birimi -- akista "sayfa bitti" gorunur.
    dizi = np.fromiter(
        (ix[x] for s in metin
         for x in [y for w in s.split() for y in bolme[w]] + [SINIR]),
        np.int64)

    yaz(f"birim: {len(say):,} kelime -> {len(ad)} BIRIM "
        f"({len(say)/len(ad):.2f}x)   dizi {len(dizi):,}")
    b = Birim(ad, dizi, bolme, kok, korunan, serbest)
    if onbellek:
        kaydet(b, onbellek, yaz)
    return b


# =====================================================================
# KAPI -- kurulumun kendi denetimi.  Sayilar model_13 ile TUTMALI.
# =====================================================================
def kapsama(b: Birim, L: int) -> tuple:
    """L uzunlugundaki pencere SORU + CEVABI ayni zincirde tutuyor mu.

    Sinav cevabi sorunun OZNESINDEN uretiyor; ikisi ayni pencereye
    sigmazsa model o baglantiyi HIC gormez. Pencere uzunlugunun
    gerekcesi bu sayidir."""
    bit = [b.ix[c] for c in ".?!" if c in b.ix]
    son = np.flatnonzero(np.isin(b.dizi, bit))
    q = np.flatnonzero(b.dizi[son] == b.ix["?"])
    q = q[(q > 0) & (q < len(son) - 1)]
    boy = son[q + 1] - son[q - 1]          # soru basindan cevap sonuna
    return len(boy), float(boy.mean()), float((boy <= L).mean())


def kapi(b: Birim, L: int | None = None) -> str:
    """model_13'un Colab kosusunda olculen sayilar:
         1.269 kelime -> 475 BIRIM (2,67x)   dizi 15.198.500
    Bunlar VERI kararina bagli; kayarlarsa korpus degismis demektir."""
    assert len(b.ad) == len(set(b.ad)), "birim adi tekrarliyor"
    assert b.dizi.max() < len(b.ad) and b.dizi.min() >= 0, "indeks tasmasi"
    assert (b.say > 0).all(), (
        f"{int((b.say == 0).sum())} birim akista HIC gecmiyor")
    tekrar = sorted(zip(b.say, b.ad))[-5:]
    s = (f"birim {len(b.ad)}   dizi {len(b.dizi):,}\n"
         f"       en sik: " + "  ".join(f"{a}({t:,})" for t, a in
                                        reversed(tekrar)))
    if L:
        n, ort, kap = kapsama(b, L)
        s += (f"\n       soru+cevap {n:,} cift, ort {ort:.1f} birim   "
              f"L={L} KAPSAMA %{100 * kap:.1f}")
        assert kap > 0.99, (
            f"pencere L={L} soru-cevap ciftlerinin yalniz %{100 * kap:.1f}"
            "'ini kapsiyor -- model baglantiyi goremez")
    return s
