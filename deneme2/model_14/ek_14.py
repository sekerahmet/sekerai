# -*- coding: utf-8 -*-
"""ek_14 -- TURKCE EK SOZLUGU ve kural tabanli bolme.

Kullanici karari, 20 Eylul 2026: *"biz bir sozluk kurmaliyiz ana mantik
bu aslinda... ekleri biliyoruz turkce ozelinde bazi kurallar var. ekleri
de ayri kelime yapabiliriz. noktalama isaretleri de birer konumu olur."*

Bolme OGRENILMIYOR, BILINIYOR. BPE denendi ve `Bahcelievler`i
`B|ah|c|eli|ev|l|er` diye dogradi -- istatistik, dilbilgisi degil.
Turkce ekler KAPALI bir kume; tahmin etmeye gerek yok.

Her ek TEK bir birim olarak diziye yaziliyor; unlu uyumu varyantlari
AYNI eke baglaniyor ("-in" ve "-un" ayni tamlayan eki).

BILINEN SINIRLAR (20 Eylul, olculdu):
  UNSUZ YUMUSAMASI YOK    `cocugunun` bolunmuyor: kok havuzunda `cocuk`
                          var, soyulunca kalan `cocug`. Ikisi AYRI birim
                          oluyor. Duzeltmek kok eslemesine k/g, p/b, t/d,
                          c/c kurali eklemek demek.
  IYELIK 1./2. KISI YOK   EKLER yalniz 3. kisi iyeligi tasiyor; bu yuzden
                          `evlerimdekilerden` de bolunmuyor (`-im` yok).
                          Korpusta gecmiyor, o yuzden simdilik sorun degil.
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

# --- EK SIRASI (morfotaktik).  Turkce'de ekler SABIT bir sirada gelir:
#       kok + COGUL + IYELIK + DURUM + ILGI + BILDIRME
# Soldan saga KESIN ARTAN olmali. Bu bir dilbilgisi kurali, hile degil.
# OLCULDU 20 Eylul: kural olmadan `bolumunun` -> bolum|TAMLAYAN|TAMLAYAN
# diye ayrisiyordu; Turkce'de arka arkaya iki tamlayan YOKTUR. Dogrusu
# bolum|IYELIK|TAMLAYAN.
SIRA = {"COGUL": 1, "IYELIK": 2,
        "AYRILMA": 3, "BULUNMA": 3, "TAMLAYAN": 3, "YONELME": 3,
        "ILGI": 4, "BILDIRME": 5}
#  !! `-ki`den sonra dongu yeniden baslar (`evdekiler`); bu hal
#  desteklenmiyor. Korpusta gecmiyor.

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
# --- BILINEN KOKLER.  `EKLER` ve `KAPALI` ile AYNI KATEGORIDE bilgi:
# sozluk bilgisi. Neden gerekli: `annesi`nin iki ayrismasi da korpus
# icinden AYIRT EDILEMEZ -- `anne` ve `annes` TAM OLARAK ayni
# kelimelerde geciyor (annesi, annesinin), yani frekans, serbestlik,
# kapsama, unlu uyumu, tampon unsuz -- hicbiri ayirmiyor. `kardesi`nin
# dogru cikmasi TESADUFTU (uzun olan dogruydu); `annesi`de uzun olan
# YANLIS. Ayirt eden tek sey hangisinin GERCEK kelime oldugu.
# Kullanici, 20 Eylul: *"Anne ve si kardes i bunlar boyle ayrilir."*
# !! GENEL bir sistemde burasi bir SOZLUK olurdu. Bizim korpusta
# iliski ve akrabalik kokleri; hepsi `veri_14.SEMA`nin anahtarlarindan.
KOK_BILINEN = {
    "anne", "baba", "kardes", "cocuk", "arkadas", "danisman", "ogrenci",
    "hoca", "rektor", "dekan", "baskan", "vali", "kurucu", "yazar",
    "bolum", "fakulte", "universite", "sehir", "bolge", "ders",
    "memleket", "tez", "konu", "onkosul", "yer",
}

MIN_KOK = 4          # daha kisa govde kok sayilmaz.
#  OLCULDU: 3'te `kardesi` -> kar+BULUNMA+IYELIK diye ucе
#  bolunuyordu; 5'te `annesi` -> annes+IYELIK (kok yanlis).
#  4'te baba/anne/kardes dogru cikiyor.



def _trb(s: str) -> str:
    """Turkce -> ASCII, BUYUK/KUCUK KORUNARAK.

    `korunan` (ozel ad) eslemesi buyuk harfe DUYARLI olmali: `_tr`
    kucuk harfe indirdigi icin ozel ad `Bolumu` ortak ad `bolumu`yu da
    koruyordu ve `bolumunun` -> bolumu|TAMLAYAN diye ayrisiyordu."""
    d = {"ı": "i", "İ": "I", "ş": "s", "Ş": "S",
         "ğ": "g", "Ğ": "G", "ü": "u", "Ü": "U",
         "ö": "o", "Ö": "O", "ç": "c", "Ç": "C"}
    return "".join(d.get(c, c) for c in s)


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


def bol(kelime, kokler, en_cok=4, korunan=frozenset(), serbest=frozenset()):
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
    if _tr(k) in KAPALI or _trb(k) in korunan:  # sozlukte BUTUN duruyor
        return [k] + nok
    coz = _coz(k, kokler, korunan, en_cok, serbest)
    return (list(coz) if coz else [k]) + nok


def _puan(coz, serbest):
    """Ayrismalari siralar:
        1  EK SAYISI        cok ekli ayrisma tercih edilir
        2  KOK BILINEN MI   sozluk bilgisi -- `anne` evet, `annes` hayir
        3  KOK SERBEST MI   korpusta tek basina geciyor mu
        4  kok uzunlugu

    !! KOK UZUNLUGU TEK BASINA YANLIS OLCUT. `kardesi` icin uzun kok
    dogru (kardes|i), `annesi` icin YANLIS (annes|i cikiyor, dogrusu
    anne|si). Ayirt eden sey kokun korpusta TEK BASINA gecip
    gecmedigi: `anne` ve `kardes` geciyor, `annes` ve `karde`
    gecmiyor. Olcut korpustan, elle yazilmiyor."""
    k = _tr(coz[0])
    return (len(coz), k in KOK_BILINEN, k in serbest, len(coz[0]))


def _coz(k, kokler, korunan, kalan_hak, serbest=frozenset()):
    """k -> [kok, -EK, ...] ya da None.  ILERI BAKISLI.

    !! ONCEKI HAL TEK EK SOYABILIYORDU. Kosul "kalan KOK olmali" idi;
    `kardesinin`de `nin` soyulunca kalan `kardesi` cikiyor, o da kok
    degil ARA BICIM, ve soyma reddediliyordu. Sonuc: cok ekli her
    kelime BUTUN kaliyordu -- ve `kardesinin` iki adimli sorunun tam
    kelimesi. `en_cok` dongusu pratikte olu idi.

    Simdi kalan KOKE INDIRGENEBILIYORSA soyuluyor; indirgeme ozyineli.

    !! EN DERIN AYRISMA secilir, ilk bulunan DEGIL. Ilk surumde
    "k'nin kendisi kokse hic bolme" diye erken donus vardi ve
    `kok_havuzu` ara bicimleri de kok sayiyor (`annesi`ye iki kelime
    indirgeniyor, yani `annesi` bir kok). Sonuc: `annesi` BUTUN
    kaliyordu, akis 15,2M'den 14,1M'e DUSUYORDU ve -IYELIK ilk bese
    giremiyordu. Kullanicinin verdigi ayrim `anne|si` tam o kayipti.

    SECIM: once EN COK ek, esitse EN UZUN KOK. Uzun kok tie-break'i
    olmadan `kardesi` -> karde|si cikiyordu ('si' iki harf, 'i' bir
    harf; ikisi de derinlik 2). Dogrusu kardes|i.

    Ve MORFOTAKTIK: ek sirasi `SIRA`ya uymali (bkz. yukari)."""
    t = _tr(k)
    if t in KAPALI or _trb(k) in korunan:
        return [k]                       # BUTUN durur, uzerinden soyulmaz
    en_iyi = [k] if t in kokler else None
    if kalan_hak > 0:
        for yuzey, ad in _YUZEY:
            if len(k) <= len(yuzey) + 1 or not t.endswith(yuzey):
                continue
            alt = _coz(k[:-len(yuzey)], kokler, korunan, kalan_hak - 1,
                       serbest)
            if not alt:
                continue
            ic = [x for x in alt[1:]]                   # icteki ekler
            if ic and SIRA[ad] <= SIRA[ic[-1][1:]]:     # MORFOTAKTIK
                continue
            aday = alt + ["-" + ad]
            if en_iyi is None or _puan(aday, serbest) > _puan(en_iyi, serbest):
                en_iyi = aday
    return en_iyi


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


def serbest_kokler(say):
    """Korpusta TEK BASINA (eksiz) gecen kelimeler.

    `_puan`in ikinci olcutu. Kok havuzu ara bicimleri de kabul ediyor
    (`annes`, `karde`); bu kume onlari ayikliyor."""
    cik = set()
    for w in say:
        while w and w[-1] in NOKTALAMA:
            w = w[:-1]
        cik.add(_tr(w.replace("'", "")))
    return cik


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
    kokler = {_tr(t) for t in tohum} | KAPALI | KOK_BILINEN
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
        p = bol(w, kokler, korunan=korunan, serbest=serbest_kokler(say))
        if len(p) > 1:
            ayrisan += 1
        kok_say[p[0]] += n
        for x in p[1:]:
            ek_say[x] += n
    return ayrisan, kok_say, ek_say
