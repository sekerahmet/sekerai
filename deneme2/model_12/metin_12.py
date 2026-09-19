# -*- coding: utf-8 -*-
"""metin_12 — GRAFTAN DUZ TURKCE METIN. model_12'un veri katmani.

model_09'da 3 bildirim kalibi vardi. OLCULDU (model_09 SONUC): bilgi
ezberlendi ama VARLIGIN ADINA baglanmadi -- koprü uzerinden sorulunca
0.04-0.16, ve hatalarin %76-95'i "dogru tip, dogru aile, YANLIS kimlik".

Physics 3.1 (2309.14316), metinden:
    "for knowledge to be reliably extracted, it must be sufficiently
     augmented... Without such augmentation, knowledge may be memorized
     but not extractable, leading to 0% accuracy, regardless of model
     size or training duration."
bioS cumle basina ~50 kalip kullaniyor.

!! BIR CUMLE = BIR OLGU, ISTISNASIZ. Bir cumle iki iliskiyi birden
soyleyemez: soylerse kopru baglama yazilmis olur ve sinav coker.

!! SINAV YUZEYI INDEKSLE DEGIL, ADLA. model_09'da sinav sorusu
`bicim 3` idi, yani "listenin SONUNCUSU". Kalip eklenince o indeks
KAYAR ve sinav sessizce bozulur. Burada bildirim ve soru AYRI listeler,
ve sinav kalibi `SINAV_KALIBI = 0` diye SABIT. `test_12` kilitler.

Unlu uyumu `veri_12`un tablolarindan gelir, burada YENIDEN HESAPLANMAZ.
"""
from __future__ import annotations

import veri_12 as V

# --- yardimcilar ---------------------------------------------------------


def _tr(ad: str) -> str:
    """Graf adi (Ibrahim_Yilmaz) -> ekranda gorunen Turkce (İbrahim Yılmaz)."""
    return " ".join(V.TR.get(w, w) for w in ad.split("_"))


ad_tr = _tr          # disari acik ad


def _nin(k: str, ozel: bool) -> str:
    """Tamlayan eki. Ozel isimde kesme isareti VAR, cins isimde YOK."""
    return ("'" if ozel else "") + V.EK_NIN[V._ek_nin(k)]


def _dir(k: str, ozel: bool) -> str:
    """Bildirme eki. Ayni kural."""
    return ("'" if ozel else "") + V.EK_DIR[V._ek_dir(k)]


def _de(k: str, iyelik: bool) -> str:
    """Bulunma: -de/-da/-te/-ta, iyelikte KAYNASTIRMA -nde/-nda."""
    return "'" + V.EK_DE[V._ek_de(k, iyelik)]


def _den(k: str, iyelik: bool) -> str:
    """Ayrilma."""
    return "'" + V.EK_DEN[V._ek_den(k, iyelik)]


def _e(k: str, iyelik: bool) -> str:
    """Yonelme."""
    return "'" + V.EK_E[V._ek_e(k, iyelik)]


# IYELIK EKIYLE BITEN TIPLER -- hal eki once 'n' ister.
IYELIKLI = ("BOLGE", "UNIVERSITE", "FAKULTE", "BOLUM", "TEZ")


# --- BILDIRIM KALIPLARI --------------------------------------------------
# Her kalip (Xn, r, rd, Y, Yd) alir ve TEK cumle dondurur.
#   Xn = ozne + tamlayan eki      "Ahmet Aydın'ın"
#   r  = iliski kelimesi           "kardeşi"
#   rd = iliski + bildirme eki     "kardeşidir"
#   Y  = cevap, ekSIZ              "Elif Aydın"
#   Yd = cevap + bildirme eki      "Elif Aydın'dır"
#
# !! YUKLEM BASTA kalibi VIRGULLU haliyle var (asagida). model_09'da
# "kullanici eledi" diye not dusmustum; kullanici 18 Eylul: *"elediğimi
# hatırlamıyorum, ile ayırabiliriz dedim"*. Kayit duzeltildi.
BILDIRIM = (
    lambda Xn, r, rd, Y, Yd: f"{Xn} {r} {Yd}.",
    lambda Xn, r, rd, Y, Yd: f"{Yd}, {Xn} {r}.",
    lambda Xn, r, rd, Y, Yd: f"{Y}, {Xn} {rd}.",
    lambda Xn, r, rd, Y, Yd: f"{Xn} {r}, {Yd}.",
    lambda Xn, r, rd, Y, Yd: f"{Y} {Xn} {rd}.",
    # !! YUKLEM BASTA, VIRGULLU kalip BURADAYDI ve CIKARILDI.
    # Uretttigi cumle:  "Kardesidir, Ahmet Aydin'in Elif Aydin."
    # Bu Turkce DEGIL: "Ahmet Aydin'in Elif Aydin" bir oge degil,
    # tamlayan ortada ve tamlanan eksiz kaliyor.
    #
    # Ben bu kalibi kullanicinin *"ile ayirabiliriz"* sozune dayanarak
    # koymustum ve virgulun oge sinirini isaretledigini yazmistim.
    # `denetle()` geri getirilip CIKTI OKUNUNCA gorundu ki virgul
    # cumleyi kurtarmiyor. Kullanicinin isareti kalibin DOGRU
    # oldugunu gostermez -- METIN gosterir.
    #
    # Yerine GLOSS bicimi: cevap BASTA (CEVAP_BASTA 5'i kullaniyor),
    # ayri bir kayit/madde register'i, ve dilbilgisi kusursuz.
    lambda Xn, r, rd, Y, Yd: f"{Y}: {Xn} {r}.",
    lambda Xn, r, rd, Y, Yd: f"Bilindiği gibi {Xn} {r} {Yd}.",
    lambda Xn, r, rd, Y, Yd: f"Kayıtlara göre {Xn} {r} {Yd}.",
    lambda Xn, r, rd, Y, Yd: f"Kaynaklarda {Xn} {r} {Yd}.",
    lambda Xn, r, rd, Y, Yd: f"Bilinen o ki {Xn} {r} {Yd}.",
    lambda Xn, r, rd, Y, Yd: f"{Xn} {r} {Y} olarak bilinir.",
    lambda Xn, r, rd, Y, Yd: f"{Xn} {r} {Y} olarak kayıtlıdır.",
    lambda Xn, r, rd, Y, Yd: f"{Y}, {Xn} {r} olarak bilinir.",
    lambda Xn, r, rd, Y, Yd: f"{Y}, {Xn} {r} olarak kayıtlıdır.",
    lambda Xn, r, rd, Y, Yd: f"{Xn} {r} {Yd}, bu böyle bilinir.",
    lambda Xn, r, rd, Y, Yd: f"Şunu da ekleyelim: {Xn} {r} {Yd}.",
    lambda Xn, r, rd, Y, Yd: f"Bir başka kayıt: {Xn} {r} {Yd}.",
)

# --- SORU KALIPLARI ------------------------------------------------------
# (Xn, r, s) alir; `s` soru sozcugu + bildirme eki  ("kimdir")
#
# !! SINAV_KALIBI = 0. Sinav YALNIZ bunu kullanir ve DEGISMEZ --
# model_09'un sinaviyla ayni yuzey, yani iki kol ayni tabloda okunur.
# Digerleri EGITIMDE var, sinavda YOK.
SORU = (
    lambda Xn, r, s: f"{Xn} {r} {s}?",
    lambda Xn, r, s: f"Peki {Xn} {r} {s}?",
    lambda Xn, r, s: f"{Xn} {r} {s} acaba?",
    lambda Xn, r, s: f"Söyler misin, {Xn} {r} {s}?",
    lambda Xn, r, s: f"{Xn} {r} {s}, biliyor musun?",
    lambda Xn, r, s: f"Bir de şunu sorayım: {Xn} {r} {s}?",
    lambda Xn, r, s: f"Merak ediyorum, {Xn} {r} {s}?",
    lambda Xn, r, s: f"{Xn} {r} {s} dersem ne dersin?",
)
SINAV_KALIBI = 0
# CEVAP BASTA kaliplar: varligin adi cumlenin BASINDA. Ters ifadesi
# OLMAYAN iliskilerde (vali/rektor/dekan/baskan/kurucu/anne/baba ve
# tersi zaten grafta olanlar) sayfaya bunlarla girilir -- boylece
# sayfanin sahibi yine BASTA olur ve adi baglama daha cok maruz kalir.
CEVAP_BASTA = (1, 2, 4, 5, 12, 13)
N_BILDIRIM, N_SORU = len(BILDIRIM), len(SORU)


# --- cumle kurucular -----------------------------------------------------
def _parca(ozne: str, iliski: str, cevap: str):
    X, Y = _tr(ozne), _tr(cevap)
    r = V.TR_ILISKI[iliski]
    return (X + _nin(X, True), r, r + _dir(r, False), Y, Y + _dir(Y, True))


def cumle(ozne: str, iliski: str, cevap: str, kalip: int) -> str:
    """TEK olgunun BILDIRIM cumlesi."""
    return BILDIRIM[kalip % N_BILDIRIM](*_parca(ozne, iliski, cevap))


def soru(ozne: str, iliski: str, cevap: str, tip: str, kalip: int = 0) -> str:
    """TEK olgunun SORU-CEVAP cumlesi. `kalip=0` SINAV yuzeyi."""
    Xn, r, _rd, _Y, Yd = _parca(ozne, iliski, cevap)
    sz = V.SORU_SOZ[tip]
    return SORU[kalip % N_SORU](Xn, r, sz + _dir(sz, False)) + f" {Yd}."


def _yol_parca(ozne: str, iliskiler, cevap: str):
    """N ADIMLI tamlama. `iliskiler` = (r1, r2, ...) sirayla.

    Kullanici, 18 Eylul: *"ahmet'in arkadasinin annesi bunlar yogun
    olur yani hatta 3 lu zincir bile olur"*. Turkce eklemeli bir dil;
    tamlama zincirleri gundelik konusmada var. Olculdu: grafta 44.374
    ikili, 262.403 UCLU yol.

    Derinlik arttikca KOPRU sayisi artar ve hicbiri YAZILMAZ:
        2 adim  "X'in kardesinin bolumu"          1 kopru gizli
        3 adim  "X'in kardesinin cocugunun bolumu" 2 kopru gizli
    Uclu, "ezber mi bilgi mi" sorusunu daha keskin sorar: 29 bin ikili
    cumle ezberlenebilir, 262 bin uclu yol ezberlenemez."""
    X, Y = _tr(ozne), _tr(cevap)
    w = [V.TR_ILISKI[r] for r in iliskiler]
    # "kardesinin cocugunun bolumu" -- son harici hepsi tamlayan ekli
    il = " ".join(x + _nin(x, False) for x in w[:-1]) + " " + w[-1]
    return (X + _nin(X, True), il, il + _dir(w[-1], False), Y,
            Y + _dir(Y, True))


def yol(ozne: str, iliskiler, cevap: str, kalip: int) -> str:
    """N adimli BILDIRIM. KOPRULER GECMEZ."""
    return BILDIRIM[kalip % N_BILDIRIM](*_yol_parca(ozne, iliskiler, cevap))


def yol_soru(ozne: str, iliskiler, cevap: str, tip: str,
             kalip: int = 0) -> str:
    """N adimli SORU-CEVAP. `kalip=0` SINAV yuzeyi."""
    Xn, il, _rd, _Y, Yd = _yol_parca(ozne, iliskiler, cevap)
    sz = V.SORU_SOZ[tip]
    return SORU[kalip % N_SORU](Xn, il, sz + _dir(sz, False)) + f" {Yd}."


def _zincir_parca(ozne: str, r1: str, r2: str, cevap: str):
    X, Y = _tr(ozne), _tr(cevap)
    a, b = V.TR_ILISKI[r1], V.TR_ILISKI[r2]
    il = f"{a}{_nin(a, False)} {b}"          # kardeşinin bölümü
    # !! `rd` TAMLAMANIN TAMAMI olmali, yalniz r2 DEGIL.
    # Ilk surumde `b + _dir(b)` yazmistim ("bölümüdür") ve `{Y}, {Xn} {rd}`
    # kalibi "Sevim Aydin, Hulya Dogan'in VALISIDIR" uretti -- r1 DUSTU ve
    # cumle YANLIS bir olgu soyledi (bir insan bir insanin valisi olamaz).
    # Kullanici ornek isteyince gorundu; istatistikte gorunmuyordu.
    # model_09'da dogruydu, ortak kalip listesine gecerken kirilmisti.
    return (X + _nin(X, True), il, il + _dir(b, False), Y, Y + _dir(Y, True))


def zincir(ozne: str, r1: str, r2: str, cevap: str, kalip: int) -> str:
    """IKI adimli BILDIRIM. KOPRU GECMEZ -- bu kolun butun sorusu bu."""
    return BILDIRIM[kalip % N_BILDIRIM](*_zincir_parca(ozne, r1, r2, cevap))


def zincir_soru(ozne: str, r1: str, r2: str, cevap: str, tip: str,
                kalip: int = 0) -> str:
    """IKI adimli SORU-CEVAP. `kalip=0` SINAV yuzeyi."""
    Xn, il, _rd, _Y, Yd = _zincir_parca(ozne, r1, r2, cevap)
    sz = V.SORU_SOZ[tip]
    return SORU[kalip % N_SORU](Xn, il, sz + _dir(sz, False)) + f" {Yd}."


def kimlik(ad: str, tip: str) -> str:
    """Sifir adim ozdeslik (identity bridge, arXiv 2509.24653)."""
    X = _tr(ad)
    sz = V.SORU_SOZ[tip]
    return f"{X} {sz}{_dir(sz, False)}? {X + _dir(X, True)}."


def biyografi_sorusu(ad: str) -> str:
    """BIYOGRAFI TETIKLEYICISI -- sinav ve kimlik yuzeyleriyle CAKISMAZ."""
    return f"{_tr(ad)} hakkında ne biliyoruz?"


# --- TERS IFADE ----------------------------------------------------------
# Ayni kenar, KONU TERS CEVRILMIS. Yeni kenar EKLEMEZ -- graf izi AYNI.
#
# !! HANGISINE YAZILIR: KARDINALITE belirler (kullanici, 18 Eylul --
# *"1->N, N->N, N->1 iliskilerin nasil ifade edilecegi"*).
#
#   COKLUGU GERCEK       -> "...den biri"  yazilir
#     bir bolgede COK il var, bir sehirde COK universite,
#     bir bolumde COK ogrenci, bir hoca COK ders veriyor.
#
#   COKLUGU VERI ARTIFAKTI -> YAZILMAZ, sira degisimi yeter
#     valisi/rektoru/dekani/baskani/kurucusu: fan-in olcumu 2 gosteriyor
#     ama gercek hayatta bir vali BIR ili yonetir. "...illerden biri
#     Adana'dir" demek modele SISTEMATIK bir cokluk ogretir -- YANLIS.
#     Kullanici yakaladi.
#     Yerine: "Adana'nin valisi Hakan Yilmaz'dir." / "Hakan Yilmaz
#     Adana'nin valisidir." / "Hakan Yilmaz'dir, Adana'nin valisi."
#
#   TERSI ZATEN GRAFTA   -> YAZILMAZ, yeni kenar UYDURMUS olurduk
#     danismani<->ogrencisi, tezi<->yazari, kardesi, arkadasi (480/480)
#
#   annesi/babasi        -> YAZILMAZ. Tersi `cocugu` ve o 1-1; annenin
#     iki cocugu var ama graf BIRINI tutuyor. "cocuklarindan biri"
#     yazmak `cocugu` sorusunu belirsizlestirirdi.
#
# (ozne, cevap, cevap_iyelikli) -> cumle.  `A` konu, `B` anilan.
TERS = {
    "bolgesi": lambda A, B, iy:
        f"{B}{_de(B, iy)}ki illerden biri {A}{_dir(A, True)}.",
    "sehri": lambda A, B, iy:
        f"{B}{_de(B, iy)}ki üniversitelerden biri {A}{_dir(A, True)}.",
    "universitesi": lambda A, B, iy:
        f"{B}{_e(B, iy)} bağlı fakültelerden biri {A}{_dir(A, True)}.",
    "fakultesi": lambda A, B, iy:
        f"{B}{_de(B, iy)}ki bölümlerden biri {A}{_dir(A, True)}.",
    # !! `bolumu` IKI TIPTE kullaniliyor: KISI->BOLUM ve DERS->BOLUM.
    # Ders OKUMAZ, OKUTULUR. Ters ifade OZNENIN tipine bakmak zorunda.
    "bolumu": lambda A, B, iy, t:
        f"{B}{_de(B, iy)} "
        + ("okutulan derslerden" if t == "DERS" else "okuyanlardan")
        + f" biri {A}{_dir(A, True)}.",
    "memleketi": lambda A, B, iy:
        f"Memleketi {B} olanlardan biri {A}{_dir(A, True)}.",
    "yasadigi_yer": lambda A, B, iy:
        f"{B}{_de(B, iy)} yaşayanlardan biri {A}{_dir(A, True)}.",
    "hocasi": lambda A, B, iy:
        f"{B}{_nin(B, True)} verdiği derslerden biri {A}{_dir(A, True)}.",
    "konusu": lambda A, B, iy:
        f"{B} konulu tezlerden biri {A}{_dir(A, True)}.",
    "onkosulu": lambda A, B, iy:
        f"{B}{_nin(B, True)} ön koşul olduğu derslerden biri {A}{_dir(A, True)}.",
}


def ters(ozne: str, iliski: str, cevap: str, tip_cevap: str,
         tip_ozne: str = "") -> str | None:
    """Kenari CEVABIN gozunden anlatir. Tablo yoksa None."""
    f = TERS.get(iliski)
    if f is None:
        return None
    A, B, iy = _tr(ozne), _tr(cevap), tip_cevap in IYELIKLI
    # Bir kalip OZNENIN tipine de bakiyor (`bolumu`); digerleri UC argumanli.
    try:
        return f(A, B, iy)
    except TypeError:
        return f(A, B, iy, tip_ozne)


# --- TIP CUMLESI ---------------------------------------------------------
# Ansiklopedi maddesi tipiyle acilir: "Manisa bir sehirdir. Manisa'nin
# valisi ...". Sayfanin ILK cumlesi, karistirilmaz -- gercek maddeler de
# oyle acilir ve modele tutarli bir "bu nedir" isareti verir.
TIP_KALIP = (
    lambda A, t, td: f"{A} bir {td}.",
    lambda A, t, td: f"{A} adlı bir {t} var.",
    lambda A, t, td: f"Bilindiği gibi {A} bir {td}.",
    lambda A, t, td: f"{A}, bir {td}.",
)


def tip_cumlesi(ad: str, tip: str, kalip: int = 0) -> str:
    """"Manisa bir sehirdir." -- varligin NE OLDUGU."""
    A, t = _tr(ad), V.TIP_AD[tip]
    return TIP_KALIP[kalip % len(TIP_KALIP)](A, t, t + _dir(t, False))


# --- TIP SORUSU ----------------------------------------------------------
# Kullanici karari, 18 Eylul: *"Ayse Yilmaz nedir? Cevap kisidir."*
# Tip bilgisinin SORULABILIR hali -- bildirim tek basina yetmiyor,
# `sayfalar`in soru ikizinde de karsiligi olmali.
#
# !! `kimdir` ILE CAKISMAZ: `kimdir` KIMLIK koprusu ("Ayse Kaya kimdir?
# Ayse Kaya'dir."), `nedir` TIP sorusu ("Ayse Kaya nedir? Kisidir.").
# Ayrisma adin hemen ardinda.
TIP_SORU = (
    lambda A, t, td: f"{A} nedir? {td[0].upper()}{td[1:]}.",
    lambda A, t, td: f"{A} nedir? Bir {t}.",
    lambda A, t, td: f"Peki {A} nedir? {td[0].upper()}{td[1:]}.",
    lambda A, t, td: f"{A} nedir, biliyor musun? Bir {td}.",
)


def tip_sorusu(ad: str, tip: str, kalip: int = 0) -> str:
    """"Ayse Yilmaz nedir? Kisidir." -- tipin SORU yuzeyi."""
    A, t = _tr(ad), V.TIP_AD[tip]
    return TIP_SORU[kalip % len(TIP_SORU)](A, t, t + _dir(t, False))


# --- "YOK" CEVAPLARI -----------------------------------------------------
# Kullanici karari, 18 Eylul: *"bilmiyorum yerine bu yok demesi daha dogru,
# bu bilgiye hakim oldugunu gosterir"* -- ve ardindan *"bilmiyorum demek
# bile cok iyi birsey"*. Ikisi de kazanc: uydurmayan her cevap uydurandan
# iyi. Bu yuzden IKI bicim de veride var, aralarinda ayrim ZORLANMIYOR.
#
# IKI TUR, ve ikisi FARKLI sey olcuyor:
#   VARLIK YOK    modelin VARLIK LISTESINE hakim olmasi gerekir
#   ILISKI OLMAZ  modelin SEMAYA hakim olmasi gerekir
#     ("Adana'nin tezi olmaz" demek icin Adana'nin SEHIR oldugunu ve
#      sehirlerin tezi olmadigini bilmek gerek -- tip cumlesi bunun temeli)
YOK_VARLIK = (
    lambda A, t, tn: f"{A} diye bir {t} yok.",
    lambda A, t, tn: f"Öyle bir {t} yok.",
    lambda A, t, tn: f"{A} diye bir {t} tanımıyorum.",
    lambda A, t, tn: f"Bilmiyorum, {A} diye bir {t} yok.",
    lambda A, t, tn: f"{A} diye bir {t} bilmiyorum.",
)
YOK_ILISKI = (
    lambda t, tn, r: f"{tn[0].upper()}{tn[1:]} {r} olmaz.",
    lambda t, tn, r: f"Bir {tn} {r} olmaz.",
    lambda t, tn, r: f"{tn[0].upper()}{tn[1:]} {r} diye bir şey yok.",
    lambda t, tn, r: f"Öyle bir bilgi yok, {tn} {r} olmaz.",
)


def yok_varlik(ad: str, tip: str, iliski: str, cevap_tip: str,
               kalip: int = 0) -> str:
    """Olmayan varlik soruldu. "Zeynep Kayabasi diye bir kisi yok." """
    A, t = _tr(ad), V.TIP_AD[tip]
    r = V.TR_ILISKI[iliski]
    sz = V.SORU_SOZ[cevap_tip]
    soru_ = f"{A}{_nin(A, True)} {r} {sz}{_dir(sz, False)}?"
    return soru_ + " " + YOK_VARLIK[kalip % len(YOK_VARLIK)](
        A, t, t + _nin(t, False))


def yok_iliski(ad: str, tip: str, iliski: str, cevap_tip: str,
               kalip: int = 0) -> str:
    """Tipe uymayan iliski soruldu. "Sehrin tezi olmaz." """
    A, t = _tr(ad), V.TIP_AD[tip]
    # UNLU DUSMESI: sehir -> sehrin. Govde tablodan, kural genellenmez.
    g = V.TIP_EKLI.get(tip, t)
    r = V.TR_ILISKI[iliski]
    sz = V.SORU_SOZ[cevap_tip]
    soru_ = f"{A}{_nin(A, True)} {r} {sz}{_dir(sz, False)}?"
    return soru_ + " " + YOK_ILISKI[kalip % len(YOK_ILISKI)](
        t, g + _nin(g, False), r)


# ======================= DENETIM =========================================
# !! BU FONKSIYON KOPYALAMADA DUSMUSTU. `metin_09.denetle` vardi;
# model_09 -> model_11 kopyasinda tasinmadi ve bunu KOPYA KAPISI
# gosterdi (kullanici: *"diff calistirip bakmadin mi?"*). Dusmesi
# model_12'da model_09'dakinden DAHA pahali olurdu: orada 4 bicim
# vardi, burada 17 bildirim x 8 soru kalibi + tip + zincir + reddetme.
#
# !! SAYI DEGIL METIN SINAR. Bu kolda ALTI veri kusuru yalniz METNI
# OKUYARAK bulundu ve hicbirini sayisal bir kapi gostermedi
# (zincir cumlesi r1'i dusuruyordu, ders "okuyanlardan biri" oluyordu,
# tip cumlesi IKI KEZ basiliyordu, "Sehirin" yaziliyordu...).


def denetle(tohum: int = 0, yaz=print) -> dict:
    """Her iliski x her kalip -- uretilen metin TURKCE mi, TEKIL mi.

    Kapilar:
      1. her (iliski, kalip) bir cumle uretiyor
      2. BILDIRIM cumlesi TEK nokta, TEK olgu (soru isareti YOK)
      3. SORU cumlesi TAM BIR soru isareti tasiyor
      4. cumle BUYUK HARFLE basliyor
      5. zincir cumlesi KOPRUYU YAZMIYOR   <- sinavin sarti
      6. cift bosluk / bosluk-noktalama YOK  <- ek tablolarindan gelir
    """
    import re as _re
    G = V.kur(tohum)
    tipi = {a: t for t in V.TIPLER for a in G["ad"][t]}
    ilk = {}
    for (o, r), c in G["olgu"].items():
        ilk.setdefault(r, (o, c))
    sayim, ornek = {}, {}
    for r in V.ILISKI:
        assert r in ilk, f"{r} icin olgu YOK"
        o, c = ilk[r]
        for k in range(N_BILDIRIM):
            s = cumle(o, r, c, k)
            assert s.count(".") == 1, f"{r} kalip {k}: TEK OLGU ihlali -> {s}"
            assert "?" not in s, f"{r} kalip {k}: bildirimde soru -> {s}"
            assert s[0].isupper(), f"{r} kalip {k}: buyuk harf yok -> {s}"
            assert not _re.search(r"  |\s[,.?]", s), f"{r} kalip {k}: bosluk -> {s}"
            sayim[("b", r, k)] = len(s)
        for k in range(N_SORU):
            s = soru(o, r, c, tipi[c], k)
            assert s.count("?") == 1, f"{r} soru {k}: tek soru degil -> {s}"
            assert s[0].isupper(), f"{r} soru {k}: buyuk harf yok -> {s}"
            assert not _re.search(r"  |\s[,.?]", s), f"{r} soru {k}: bosluk -> {s}"
            sayim[("s", r, k)] = len(s)
        ornek[r] = soru(o, r, c, tipi[c], SINAV_KALIBI)
    # --- ZINCIR: KOPRU GECMEYECEK
    n_z = 0
    for (o, r1), b in G["olgu"].items():
        for (b2, r2), c in G["olgu"].items():
            if b2 != b or c in (o, b):
                continue
            for k in (0, 5, 11, 16):
                s = zincir(o, r1, r2, c, k)
                assert _tr(b) not in s, f"ZINCIR KOPRUYU YAZIYOR: {s}"
                assert s.count(".") == 1, f"zincir TEK OLGU ihlali -> {s}"
            sq = zincir_soru(o, r1, r2, c, tipi[c], SINAV_KALIBI)
            assert _tr(b) not in sq, f"ZINCIR SORUSU KOPRUYU YAZIYOR: {sq}"
            n_z += 1
            break
        if n_z >= 40:
            break
    # --- TIP ve REDDETME
    for t in V.TIPLER:
        a = G["ad"][t][0]
        for k in range(len(TIP_KALIP)):
            s = tip_cumlesi(a, t, k)
            assert s.count(".") == 1 and s[0].isupper(), f"tip: {s}"
        for k in range(len(TIP_SORU)):
            s = tip_sorusu(a, t, k)
            assert s.count("?") == 1 and s[0].isupper(), f"tip sorusu: {s}"
    for r, ts in V.IMKANSIZ.items():
        h = tipi[ilk[r][1]]
        for t in ts:
            for k in range(len(YOK_ILISKI)):
                s = yok_iliski(G["ad"][t][0], t, r, h, k)
                assert s.count("?") == 1 and s[0].isupper(), f"yok_iliski: {s}"
                assert not _re.search(r"  |\s[,.?]", s), f"yok_iliski bosluk: {s}"
    yaz(f"  {len(V.ILISKI)} iliski x ({N_BILDIRIM} bildirim + {N_SORU} soru)"
        f" = {len(sayim):,} cumle GECTI")
    yaz(f"  zincir {n_z} ornek: KOPRU hicbirinde GECMIYOR")
    yaz(f"  uzunluk {min(sayim.values())}..{max(sayim.values())} karakter")
    return ornek


if __name__ == "__main__":
    print("metin_12 DENETIMI")
    orn = denetle()
    G = V.kur(0)
    tipi = {a: t for t in V.TIPLER for a in G["ad"][t]}
    print("\nHER ILISKIDEN BIR SINAV SORUSU")
    for r in V.ILISKI:
        print("   " + orn[r])
    o = "Ahmet_Aydin"
    c = G["olgu"][(o, "kardesi")]
    print(f"\nAYNI OLGU, {N_BILDIRIM} BILDIRIM KALIBI")
    for k in range(N_BILDIRIM):
        print(f"   {k:2d}  {cumle(o, 'kardesi', c, k)}")
    print(f"\nAYNI OLGU, {N_SORU} SORU KALIBI")
    for k in range(N_SORU):
        print(f"   {k:2d}  {soru(o, 'kardesi', c, tipi[c], k)}")
    k1 = G["olgu"][(o, "bolumu")]
    f1 = G["olgu"][(k1, "fakultesi")]
    print(f"\nIKI ADIM -- kopru '{_tr(k1)}' GECMIYOR")
    print("   " + zincir_soru(o, "bolumu", "fakultesi", f1, tipi[f1]))
    u1 = G["olgu"][(f1, "universitesi")]
    print(f"\nUC ADIM -- iki kopru de GECMIYOR")
    print("   " + yol_soru(o, ("bolumu", "fakultesi", "universitesi"),
                           u1, tipi[u1]))
    print("\nTIP")
    for t in V.TIPLER:
        print(f"   {tip_cumlesi(G['ad'][t][0], t, 0)}"
              f"   |   {tip_sorusu(G['ad'][t][0], t, 0)}")
