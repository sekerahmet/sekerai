# -*- coding: utf-8 -*-
"""ek_16 -- TURKCE BOLME.  KOK AYRI TOKEN, EK AYRI TOKEN.

Kullanici, 20 Eylul 2026:
  *"kok ayri token ek ayri token"*
  *"ben bunlarin ayri token olmasini net istedim"*      (ek yuzeyleri)
  *"token listesi cok kritik. bu token ureten yere gerekiyorsa
   tek tek yazmak lazim."*

Bolme OGRENILMIYOR, BILINIYOR. BPE denendi ve `Bahcelievler`i
`B|ah|c|eli|ev|l|er` diye dogradi -- istatistik, dilbilgisi degil.

IKI SOZLUK, ELLE yazildi:
    KOK      isim kokleri -- her biri BIR token
    BUTUN    kalip sozcukler -- hic bolunmez
ve EK YUZEYLERI.  Bunlarin disinda kalan kelime BUTUN birakilir ve
`denetle()` onu RAPORLAR -- sessizce uydurma kok uretilmez.

!! EK TOKENI = GERCEK YUZEY.  `-ın` ile `-in` AYRI tokenler; unlu
uyumu modelin GIRDISINDE. Onceki surum sekiz bicimi tek `-TAMLAYAN`e
cokertiyordu. Kullanici ayni cokertmeyi 17 Eylul'de KARAKTER
duzeyinde yakalamisti (tek <NIN> jetonu sekiz bicimi ortuyordu) ve
orada duzeltilmisti; `ek_16` onu birim duzeyinde geri getirmisti.

!! OZEL AD KENDILIGINDEN KORUNUR.  Ayri bir `korunan` listesine
gerek yok: soyma ancak kalan BILINEN bir koke inerse kabul ediliyor.
`Mersin` -> `Mers`+`in` olamaz cunku `Mers` kok degil. Onceki surum
olcut tabanli bir kok havuzu kuruyordu ve o havuza `Mers`, `Gires`,
`Fakul`, `Univers`, `ogrencis`, `kurucus`, `cocug`, `sehr` gibi
UYDURMA kokler giriyordu.

MORFOFONOLOJI -- kok sozlugu ACIK oldugu icin guvenle uygulanabiliyor:
    YUMUSAMA   cocuk + u  -> cocugu     TOKEN `cocuk` kalir
    DUSME      sehir + i  -> sehri      TOKEN `sehir` kalir
    TAMPON     anne + si                (-s-/-n-/-y- yardimci unsuz,
                                         ek yuzeyinin kendisinde)
BUYUK/KUCUK:  ortak ad kucuge iner (`Kisidir` -> `kişi -dir`), ozel ad
buyuk kalir -- cunku ozel ad zaten coz'e girmez.

ZINCIRDEKI YERI.  Kim kimi cagiriyor, bu dosya nerede:
(model BIRIM goruyor -- karakter yalnizca ara adim ve kapi)

  veri_16     graf: 1608 varlik, 24 iliski, olgular
  metin_16    graf -> duz Turkce cumle
  korpus_16   cumle -> belge -> paketlenmis akis
  jeton_16    KARAKTER sozlugu + GIDIS-DONUS KAPISI
  birim_16    metin -> sayim -> kok havuzu -> BIRIM AKISI -> pencere
  ek_16       kelime -> kok + ek    (Turkce morfolojisi)   <-- BU DOSYA

  taban_16    bolmeler (ezber_* / cikarim_*) + Ayar tanimi
  ayar_16     dugmeler
  hazirla_16  veriyi dosyaya yazar, Colab Drive'dan OKUR

  model_16    MIMARI -- model_15'ten
  kos_16      egitim dongusu
  olcme_16    olcu: soru soruldu, cevap dogru mu
"""
from __future__ import annotations

# =====================================================================
# 1  EK YUZEYLERI.  Her yuzey AYRI TOKEN; kategori yalniz SIRA icin.
# =====================================================================
EKLER = [
    ("ILGI",     ["ki"]),
    ("AYRILMA",  ["ndan", "nden", "dan", "den", "tan", "ten"]),
    ("BULUNMA",  ["nda", "nde", "da", "de", "ta", "te"]),
    ("TAMLAYAN", ["nın", "nin", "nun", "nün", "ın", "in", "un", "ün"]),
    ("BILDIRME", ["dır", "dir", "dur", "dür", "tır", "tir", "tur", "tür"]),
    ("YONELME",  ["ya", "ye", "na", "ne", "a", "e"]),
    #  Ciplak -a/-e ONCEDEN KAPALIYDI: ozel adin son unlusunu yiyordu
    #  (Kaya->Kay, Manisa->Manis). Artik guvenli -- kalan BILINEN bir
    #  koke inmezse soyma kabul edilmiyor.
    ("IYELIK",   ["sı", "si", "su", "sü", "ı", "i", "u", "ü"]),
    ("COGUL",    ["lar", "ler"]),
]
NOKTALAMA = ".,?!:;"

# EK SIRASI (morfotaktik):  kok + COGUL + IYELIK + DURUM + ILGI + BILDIRME
# Soldan saga KESIN ARTAN. OLCULDU: kural olmadan `bolumunun` ->
# bolum|TAMLAYAN|TAMLAYAN cikiyordu; Turkce'de iki tamlayan ard arda yok.
SIRA = {"COGUL": 1, "IYELIK": 2,
        "AYRILMA": 3, "BULUNMA": 3, "TAMLAYAN": 3, "YONELME": 3,
        "ILGI": 4, "BILDIRME": 5}

# =====================================================================
# 2  KOK SOZLUGU -- ELLE.  Korpustaki her ortak ad buradan gelir.
# =====================================================================
KOK = {
    # akrabalik ve roller -- veri_16.SEMA iliskilerinin kokleri
    "anne", "baba", "kardeş", "çocuk", "arkadaş", "danışman", "öğrenci",
    "hoca", "rektör", "dekan", "başkan", "vali", "kurucu", "yazar",
    # tipler ve nesneler
    "bölüm", "fakülte", "üniversite", "şehir", "bölge", "ders",
    "memleket", "tez", "konu", "koşul", "önkoşul", "yer", "kişi", "il",
    "kayıt", "kaynak", "şey", "ön", "bilgi",
}

# KALIP SOZCUKLERI -- hic bolunmez. Olcut tabanli havuz `hangi`yi
# `hang`+IYELIK, `musun`u `mus`+TAMLAYAN diye bozmustu.
BUTUN = {
    "hangi", "hangisi", "kim", "ne", "neresi", "nasıl", "kaç",
    "mu", "mi", "mü", "mı", "musun", "misin", "mıdır", "midir",
    "bir", "bu", "şu", "o", "böyle", "şöyle", "öyle", "şunu",
    "gibi", "ile", "için", "göre", "kadar", "olarak", "peki", "merak",
    "ve", "veya", "ama", "fakat", "çünkü", "yani", "de", "da", "ki",
    "biliyor", "bilinir", "bilindiği", "biliyoruz", "bilmiyorum",
    "kayıtlıdır", "hakkında", "yaşadığı", "söyler", "acaba", "diye",
    "yok", "var", "dersem", "ediyorum", "ekleyelim", "sorayım",
    "tanımıyorum", "okutulan", "olduğu", "olmaz", "verdiği", "konulu",
    "okuyanlardan", "olanlardan", "yaşayanlardan",
    "bilinen", "adlı", "bağlı", "başka", "biri",
}

# =====================================================================
# 3  MORFOFONOLOJI.  Kok TOKENI degismez; degisen yalniz ESLESME.
# =====================================================================
YUMUSAMA = {"k": "ğ", "p": "b", "t": "d", "ç": "c"}
#   cocuk + u -> cocugu.  Son unsuz yumusar, TOKEN `cocuk` kalir.

# Unlu dusmesi: ELLE, cunku kural degil SOZLUK bilgisi
# (sehir -> sehr, ama demir -> demir).
DUSME = {"şehir": "şehr", "oğul": "oğl", "burun": "burn", "ağız": "ağz",
         "akıl": "akl", "isim": "ism", "resim": "resm", "beyin": "beyn"}

_KAT = {}
for _ad, _bs in EKLER:
    for _b in _bs:
        assert _KAT.setdefault(_b, _ad) == _ad, "yuzey iki kategoride: " + _b
_YUZEY = sorted(((b, a) for a, bs in EKLER for b in bs),
                key=lambda x: -len(x[0]))


def _kucuk(s: str) -> str:
    """Turkce'ye dogru kuculme:  I -> ı,  İ -> i."""
    return s.replace("I", "ı").replace("İ", "i").lower()


def _govde_bicimleri(kok: str) -> set:
    """Bir kokun ek alirken girebilecegi butun govde bicimleri."""
    g = {kok}
    if kok in DUSME:
        g.add(DUSME[kok])
    if kok and kok[-1] in YUMUSAMA:
        g.add(kok[:-1] + YUMUSAMA[kok[-1]])
    return g


_GOVDE = {}          # govde bicimi -> KOK TOKENI
for _k in KOK:
    for _g in _govde_bicimleri(_k):
        _GOVDE[_g] = _k


def _cek(k: str, ozel, hak: int, ozel_ac: bool):
    """Ek soyarak coz.  `ozel_ac` ise OZEL AD da kok sayilir."""
    kk = _kucuk(k)
    if kk in BUTUN:
        return [kk]
    if kk in _GOVDE:
        return [_GOVDE[kk]]
    if ozel_ac and k in ozel:
        return [k]
    if hak <= 0:
        return None
    en_iyi = None
    for yuzey, ad in _YUZEY:
        if len(kk) <= len(yuzey) or not kk.endswith(yuzey):
            continue
        alt = _cek(k[:len(k) - len(yuzey)], ozel, hak - 1, ozel_ac)
        if not alt:
            continue
        ic = [x[1:] for x in alt[1:]]
        if ic and SIRA[ad] <= SIRA[_KAT[ic[-1]]]:          # MORFOTAKTIK
            continue
        # !! KALIP SOZCUK YALNIZ BILDIRME ALIR.  `kimdir` = kim|dir
        # dogru, ama `Kimya` = kim|ya DEGIL; `Oya` = o|ya DEGIL.
        # Ikisi de ozel ad ve bolucu onlari parcaliyordu (olculdu,
        # token listesinde gorundu).
        if _kucuk(alt[0]) in BUTUN and ad != "BILDIRME":
            continue
        aday = alt + ["-" + yuzey]
        # EN AZ EK, esitse EN UZUN KOK:  `kardeşi` -> kardeş|i.
        if en_iyi is None or (len(aday), -len(aday[0])) < (len(en_iyi),
                                                           -len(en_iyi[0])):
            en_iyi = aday
    return en_iyi


def _coz(k: str, ozel=frozenset(), hak: int = 4):
    """k -> [kok, -ek, ...] ya da None.  UC ASAMA, sirasi ONEMLI.

    1  YALNIZ ORTAK KOK ile ayris.  `Bölgesi` -> bölge|si olmali
       (kullanici, 20 Eylul) -- ozel ad listesinde olsa bile.
    2  Kelimenin KENDISI ozel adsa BUTUN birak.  `Osmaniye` boyle
       korunuyor; yoksa `Osman`+`i`+`ye` diye parcalaniyordu cunku
       `Osman` da bir ozel ad.
    3  Ozel adi KOK sayarak ayris.  `Mersin'in` -> Mersin|in.
       `Mers` kok olmadigi icin daha fazla bolunemez."""
    r = _cek(k, ozel, hak, False)
    if r:
        return r
    if k in ozel:
        return [k]
    return _cek(k, ozel, hak, True)


def bol(kelime, kokler=None, en_cok=4, korunan=frozenset(),
        serbest=frozenset()):
    """kelime -> [kok, -ek, ..., noktalama].

    `kokler` / `serbest` ARTIK KULLANILMIYOR -- kok sozlugu ELLE
    (KOK / BUTUN); imza geriye uyumluluk icin duruyor. `korunan` da
    gerekmiyor: bilinmeyen kok butun kaliyor (bkz. `_coz`)."""
    nok = []
    k = kelime
    while k and k[-1] in NOKTALAMA:
        nok.insert(0, k[-1])
        k = k[:-1]
    k = k.replace("'", "")
    if not k:
        return nok
    coz = _coz(k, korunan)
    return (coz if coz else [k]) + nok


def denetle(kelimeler, ozel=frozenset()):
    """Her kelime BILINEN KOK + EK'e cozuluyor mu.

    `ozel` OZEL AD yuzeyleri -- onlarin butun kalmasi DOGRU.
    Doner: (cozulen, BUTUN kalan ve ozel de OLMAYAN) -- ikincisi
    sozluge eklenmesi gereken koklerin listesi."""
    iyi, eksik = {}, {}
    for w in kelimeler:
        p = bol(w, korunan=ozel)
        c = [x for x in p if x not in NOKTALAMA]
        if c and (c[0] in KOK or c[0] in BUTUN):
            iyi[w] = p
        elif c and c[0] in ozel:
            iyi[w] = p
        else:
            eksik[w] = p
    return iyi, eksik


# --- ESKI ARAYUZ.  `birim_16` ve `olcme_14` bunlari cagiriyor. -------
def kok_havuzu(say, tohum=(), en_az=2):
    """ARTIK OLCUTE BAKMIYOR -- kok sozlugu ELLE yazildi."""
    return set(KOK) | set(BUTUN)


def serbest_kokler(say):
    return frozenset()


def _tr(s: str) -> str:
    d = {"ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
         "ü": "u", "Ü": "u", "ö": "o", "Ö": "o", "ç": "c", "Ç": "c"}
    return "".join(d.get(c, c) for c in s).lower()


def _trb(s: str) -> str:
    """Turkce -> ASCII, BUYUK/KUCUK KORUNARAK."""
    d = {"ı": "i", "İ": "I", "ş": "s", "Ş": "S", "ğ": "g", "Ğ": "G",
         "ü": "u", "Ü": "U", "ö": "o", "Ö": "O", "ç": "c", "Ç": "C"}
    return "".join(d.get(c, c) for c in s)
