# -*- coding: utf-8 -*-
"""ek_13 -- TURKCE EK SOZLUGU ve kural tabanli bolme.

Kullanici karari, 20 Eylul 2026: *"biz bir sozluk kurmaliyiz ana mantik
bu aslinda... ekleri biliyoruz turkce ozelinde bazi kurallar var. ekleri
de ayri kelime yapabiliriz. noktalama isaretleri de birer konumu olur."*

Bolme OGRENILMIYOR, BILINIYOR. BPE denendi ve `Bahcelievler`i
`B|ah|c|eli|ev|l|er` diye dogradi -- istatistik, dilbilgisi degil.
Turkce ekler KAPALI bir kume; tahmin etmeye gerek yok.

Her ek TEK bir birim olarak diziye yaziliyor; unlu uyumu varyantlari
AYNI eke baglaniyor ("-in" ve "-un" ayni tamlayan eki). Boylece
`evlerimdekilerden` alti ek alir ve hicbir yere sigdirma derdi olmaz.
"""
from __future__ import annotations

import collections
import unicodedata

# --- EK SOZLUGU. Sira ONEMLI: uzun bicim once denenir.
#     Her satir: (ek_adi, yuzey bicimleri)  -- yuzeyler AYNI eke baglanir.
EKLER = [
    ("ILGI",     ["ki"]),
    ("AYRILMA",  ["ndan", "nden", "dan", "den", "tan", "ten"]),
    ("BULUNMA",  ["nda", "nde", "da", "de", "ta", "te"]),
    ("TAMLAYAN", ["nin", "nin", "nun", "nun", "in", "in", "un", "un"]),
    ("BILDIRME", ["dir", "dir", "dur", "dur", "tir", "tir", "tur", "tur"]),
    ("YONELME",  ["ya", "ye", "na", "ne"]),
    #  !! CIPLAK "a"/"e" DUSURULDU: ozel adin son unlusunu
    #  yiyordu (Kaya->Kay, Manisa->Manis, baba->bab) ve
    #  korpusta ciplak yonelme zaten gecmiyor.
    ("IYELIK",   ["si", "si", "su", "su", "i", "i", "u", "u"]),
    ("COGUL",    ["lar", "ler"]),
]
NOKTALAMA = ".,?!:;"

# --- KAPALI SINIF: sozlukte BUTUN duran kelimeler, asla bolunmez.
# Kullanicinin ilkesi: bunlar zaten bildigimiz kelimeler. Olcut tabanli
# havuz `hangi`yi `hang`+IYELIK, `musun`u `mus`+TAMLAYAN diye bozmustu.
KAPALI = {
    "hangi", "hangisi", "kim", "ne", "neresi", "nasil", "kac",
    "mu", "mi", "mu", "mu", "musun", "midir",
    "bir", "bu", "su", "o", "boyle", "soyle", "oyle",
    "gibi", "ile", "icin", "gore", "kadar", "olarak", "peki",
    "ve", "veya", "ama", "fakat", "cunku", "yani", "de", "da",
    "biliyor", "bilinir", "bilindigi", "biliyoruz", "kayitlidir",
    "kayitlarda", "kaynaklarda", "hakkinda", "yasadigi", "soyler",
    "misin", "acaba", "diye", "yok", "var",
}
MIN_KOK = 4          # daha kisa govde kok sayilmaz.
#  OLCULDU: 3'te `kardesi` -> kar+BULUNMA+IYELIK diye ucе
#  bolunuyordu; 5'te `annesi` -> annes+IYELIK (kok yanlis).
#  4'te baba/anne/kardes dogru cikiyor.



def _tr(s: str) -> str:
    """Turkce harfleri ASCII'ye indirger -- YALNIZ eslestirme icin."""
    d = {"\u0131": "i", "\u0130": "i", "\u015f": "s", "\u015e": "s",
         "\u011f": "g", "\u011e": "g", "\u00fc": "u", "\u00dc": "u",
         "\u00f6": "o", "\u00d6": "o", "\u00e7": "c", "\u00c7": "c"}
    return "".join(d.get(c, c) for c in s).lower()


# yuzey -> (ek_adi, uzunluk).  Uzun bicimler once denensin diye sirali.
_YUZEY = []
for _ad, _bs in EKLER:
    for _b in _bs:
        _YUZEY.append((_b, _ad))
_YUZEY.sort(key=lambda x: -len(x[0]))


def bol(kelime, kokler, en_cok=4, korunan=frozenset()):
    """kelime -> [kok, -EK, -EK, ..., noktalama].

    Ek ancak GERIDE KALAN da kok sozlugunde varsa soyulur; yoksa
    `anne` -> `ann` + YONELME diye bozulurdu. Kok sozlugu asagidaki
    `kok_havuzu` ile kurulur, elle yazilmaz."""
    nok = []
    k = kelime
    while k and k[-1] in NOKTALAMA:
        nok.insert(0, k[-1])
        k = k[:-1]
    k = k.replace("'", "")          # Turkce yazim ozel ad ekini zaten ayirir
    if _tr(k) in KAPALI or _tr(k) in korunan:   # sozlukte BUTUN duruyor
        return [k] + nok
    ekler = []
    for _ in range(en_cok):
        bulundu = False
        for yuzey, ad in _YUZEY:
            if len(k) > len(yuzey) + 1 and _tr(k).endswith(yuzey):
                kalan = k[:-len(yuzey)]
                if (_tr(kalan) in kokler and _tr(k) not in KAPALI
                        and _tr(k) not in korunan):
                    ekler.insert(0, "-" + ad)
                    k = kalan
                    bulundu = True
                    break
        if not bulundu:
            break
    return [k] + ekler + nok


def _govdeler(w):
    """Bir kelimenin butun olasi govdeleri (ek zinciri soyularak)."""
    while w and w[-1] in NOKTALAMA:
        w = w[:-1]
    w = w.replace("'", "")
    out, yig = set(), [w]
    for _ in range(4):
        yeni = []
        for k in yig:
            for yuzey, _ad in _YUZEY:
                if len(k) > len(yuzey) + 1 and _tr(k).endswith(yuzey):
                    g = k[:-len(yuzey)]
                    out.add(_tr(g))
                    yeni.append(g)
        yig = yeni
    return out


def kok_havuzu(say, tohum=(), en_az=2):
    # `tohum` = BILINEN ozel adlar. Hem kok sayilir hem KORUNUR:
    # uzerinden ek soyulmaz. Yoksa 'Kaya' -> 'Kay'+YONELME olur.
    """KOK SOZLUGU -- elle yazilmaz, OLCUTLE kurulur.

    Bir govde kok sayilir eger:
      (a) `tohum`daysa  -- bilinen ozel adlar (varlik parcalari), ya da
      (b) EN AZ `en_az` FARKLI kelime ona indirgeniyorsa.

    (b) klasik denetimsiz morfoloji olcutu: `bolumu` ve `bolumunun`
    ayni govdeyi paylasiyorsa `bolum` bir koktur. Tek bir kelimeden
    govde uydurmayi engelliyor."""
    kokler = {_tr(t) for t in tohum} | KAPALI
    aday = collections.Counter()
    for w in say:
        for g in _govdeler(w):
            if g not in kokler:      # KORUNAN adin govdesi aday DEGIL
                aday[g] += 1
    kokler |= {g for g, n in aday.items()
               if n >= en_az and len(g) >= MIN_KOK}
    # eksiz gecen kelimeler de koktur
    for w in say:
        s_ = w
        while s_ and s_[-1] in NOKTALAMA:
            s_ = s_[:-1]
        s_ = s_.replace("'", "")
        if s_ and not any(_tr(s_).endswith(y) and _tr(s_[:-len(y)]) in kokler
                          for y, _ in _YUZEY if len(s_) > len(y) + 1):
            kokler.add(_tr(s_))
    return kokler


def dokum(say, kokler, korunan=frozenset()):
    """Bolmenin KAPSAMASI -- kac kelime ayristi, kok/ek dagilimi."""
    kok_say = collections.Counter()
    ek_say = collections.Counter()
    ayrisan = 0
    for w, n in say.items():
        p = bol(w, kokler, korunan=korunan)
        if len(p) > 1:
            ayrisan += 1
        kok_say[p[0]] += n
        for x in p[1:]:
            ek_say[x] += n
    return ayrisan, kok_say, ek_say
