# -*- coding: utf-8 -*-
"""veri_06 — model_05'in KENDI verisi. TEK BASINA DURUR.

Kullanici karari, 17 Eylul 2026:
    *"ben artik GERCEK bir veri istiyorum ve mantikli. yani sacma
    iliskiler yok, dil bilgisi dogru vs. phi yuksek yapalim tabi ki ama
    burda kritik nokta verimiz gercekten bu ornekte DUZGUN olmali."*

ve yapinin kendisi icin:
    *"okul yerine universite desek, ders yerine fakulte desek, sonrada
    bolum mantikli olur o zaman"*

`veri_04` ile arasindaki fark ICERIK farkidir, kopya farki degil. Bu
dosya hicbir `veri_*` modulunu import ETMEZ.

==========================================================================
NEDEN YENI VERI -- veri_04'te OLCULEN KUSURLAR

`veri_04` (= `veri_okul4`) yuksek phi'yi (9,57) SEMA'yi zorlayarak
aliyordu: `kardes` iliskisi OKUL ve SEHIR'e, `komsu` KISI'ye, `kurucu`
DERS'e de aciliyordu. Olculdu:

    "Matematik dersinin kurucusu kim"     400 / 10.960 olgu
    "Ankara sehrinin dersi ne"            bu tur, KENDI ICINDE celiskili

Ayrica:

    200 / 200 OKUL adi kendi sehrini SIZDIRIYOR   (Ankara_Lisesi -> Ankara)
    "Ali Yildiz'in kardesinin kardesi"            AYNI havuzdan rastgele
    "Fatma Dogan'in annesi Huseyin Dogan"         cinsiyet TUTMUYOR

Son ikisi `veri_04`te gercekten vardi: aile baglari rastgele baglaniyordu,
soy agaci DEGILDI.

==========================================================================
BU VERIDE NE VAR

Yedi tip, GERCEK bir universite hiyerarsisi + GERCEK bir soy agaci:

    BOLGE  <--bolgesi--  SEHIR  <--sehri--  UNIVERSITE
                                                ^
                                          universitesi
                                                |
      DERS  --bolumu-->  BOLUM  --fakultesi-->  FAKULTE

Zincir ustte YUKARI, altta ASAGI ("amiral" kenarlar) gidiyor:
`ana_fakultesi`, `ana_bolumu`, `ana_dersi`, `merkezi`. Bunlar gercek
("fakultenin amiral bolumu") ve phi'yi yukseltiyor -- SEMA'yi zorlamadan.

--------------------------------------------------------------------------
NOTR ADLANDIRMA -- kasitli, ve SIZINTIYI OLCUYORUZ

FAKULTE ve BOLUM onekleri SEMT listesinden geliyor, ustundeki varligin
adiyla ILGISIZ:

    Cerrahpasa Muhendislik Fakultesi  ->  universitesi  Bogazici Uni.
                ^ Bogazici DEGIL -- cevabin yarisi soruda DURMUYOR

Gercek hayatta da boyle: Cerrahpasa Tip Fakultesi Istanbul
Universitesi'ne baglidir ve adinda "Istanbul" GECMEZ.

KISI soyadi ise BILEREK gercekci, yani AILE ICINDE PAYLASILIYOR. Bu bir
sizintidir ve KALIYOR: gercek hayatta da vardir, ve `sizinti()` onu
zincir zincir olcuyor ki bolme kodu `kopyalanabilir` / `kopyalanamaz`
diye AYIRABILSIN.

--------------------------------------------------------------------------
SOY AGACI -- cinsiyet ve kusak TUTARLI

Her soyadi blogunda 3 kusak x 16 kisi (8 erkek + 8 kadin):

    kusak 0   M0_0..M0_7   F0_0..F0_7      ebeveyni KAYITLI DEGIL
    kusak 1   M1_0..M1_7   F1_0..F1_7      ebeveyni kusak 0
    kusak 2   M2_0..M2_7   F2_0..F2_7      ebeveyni kusak 1

    kardes ciftleri   (M_i , F_{i+1})      <- ters cinsiyet, ayni ebeveyn
    ebeveyn ciftleri  (M_i , F_i)          <- kardes ciftinden KAYIK

Kayiklik SART: (M_i, F_i) hem es hem kardes olsaydi soy agaci ensest
cikardi. Kaydirma bunu yapisal olarak imkansiz kiliyor, `_denetle`
ayrica sinar.

Bundan cikanlar:

    annesi  HER ZAMAN kadin        babasi  HER ZAMAN erkek
    kardesler ayni anne VE ayni babadan
    cocugu, annesi/babasi'nin TERSI
    "annesinin annesi" = buyukanne -- ANLAMLI, rastgele DEGIL
    "kardesinin kardesi" = KENDISI -- DONUS sinifi, sinav disi

--------------------------------------------------------------------------
EKSIK OLGU -- kasitli, ve KENDI SINIFI var

Gercek bir soy agacinin KENARI vardir: kusak 0'in ebeveyni, kusak 2'nin
cocugu kayitli DEGIL. Bunlari uydurmuyoruz.

`veri_04`te her varligin semasindaki HER iliski doluydu; burada degil.
O yuzden zincir sinifi SEMAYA bakar, sozluge DEGIL:

    YOK     kisayol TIP OLARAK imkansiz          <- `ent_yok` bolmesi
    EKSIK   tip mumkun ama OLGU kayitli degil    <- SINAV DISI

Ikisi ayrilmazsa `ent_yok` sessizce kirlenirdi: "kisayol imkansiz" diye
sayilan zincirin bir kismi aslinda "kisayol var ama biz yazmadik"
olurdu. (CLAUDE.md, `ent_yok` tanimi.)
"""
from __future__ import annotations

import hashlib

import numpy as np

# ======================================================================
# SOZLUKLER
# (jeton, TURKCE YAZIM) -- model ASCII jetonu gorur, insan dogru yazimi.
# Turkce yazim YALNIZ `yuzey()` icin; grafta ve sozlukte yeri YOK.
# ======================================================================
ERKEK = [
    ("Ahmet", "Ahmet"), ("Mehmet", "Mehmet"), ("Mustafa", "Mustafa"),
    ("Ali", "Ali"), ("Huseyin", "Hüseyin"), ("Hasan", "Hasan"),
    ("Ibrahim", "İbrahim"), ("Osman", "Osman"), ("Yusuf", "Yusuf"),
    ("Murat", "Murat"), ("Omer", "Ömer"), ("Suleyman", "Süleyman"),
    ("Halil", "Halil"), ("Ismail", "İsmail"), ("Fatih", "Fatih"),
    ("Kemal", "Kemal"), ("Salih", "Salih"), ("Emre", "Emre"),
    ("Burak", "Burak"), ("Cem", "Cem"), ("Furkan", "Furkan"),
    ("Gokhan", "Gökhan"), ("Hakan", "Hakan"), ("Kaan", "Kaan"),
    ("Levent", "Levent"), ("Onur", "Onur"), ("Sinan", "Sinan"),
    ("Tolga", "Tolga"), ("Volkan", "Volkan"), ("Yigit", "Yiğit"),
    ("Baris", "Barış"), ("Caner", "Caner"),
]
KADIN = [
    ("Ayse", "Ayşe"), ("Fatma", "Fatma"), ("Emine", "Emine"),
    ("Hatice", "Hatice"), ("Zeynep", "Zeynep"), ("Elif", "Elif"),
    ("Meryem", "Meryem"), ("Zehra", "Zehra"), ("Hulya", "Hülya"),
    ("Melek", "Melek"), ("Ozlem", "Özlem"), ("Yasemin", "Yasemin"),
    ("Sevim", "Sevim"), ("Gulay", "Gülay"), ("Filiz", "Filiz"),
    ("Derya", "Derya"), ("Ebru", "Ebru"), ("Gamze", "Gamze"),
    ("Hande", "Hande"), ("Irem", "İrem"), ("Kubra", "Kübra"),
    ("Leyla", "Leyla"), ("Merve", "Merve"), ("Nazli", "Nazlı"),
    ("Oya", "Oya"), ("Pinar", "Pınar"), ("Rabia", "Rabia"),
    ("Selin", "Selin"), ("Tugce", "Tuğçe"), ("Yagmur", "Yağmur"),
    ("Asli", "Aslı"), ("Ceren", "Ceren"),
]
SOYAD = [
    ("Yilmaz", "Yılmaz"), ("Kaya", "Kaya"), ("Demir", "Demir"),
    ("Sahin", "Şahin"), ("Celik", "Çelik"), ("Yildiz", "Yıldız"),
    ("Aydin", "Aydın"), ("Ozturk", "Öztürk"), ("Arslan", "Arslan"),
    ("Dogan", "Doğan"),
]

SEHIR_AD = [
    ("Adana", "Adana"), ("Ankara", "Ankara"), ("Antalya", "Antalya"),
    ("Aydin", "Aydın"), ("Balikesir", "Balıkesir"), ("Bursa", "Bursa"),
    ("Canakkale", "Çanakkale"), ("Denizli", "Denizli"),
    ("Diyarbakir", "Diyarbakır"), ("Edirne", "Edirne"), ("Elazig", "Elazığ"),
    ("Erzurum", "Erzurum"), ("Eskisehir", "Eskişehir"),
    ("Gaziantep", "Gaziantep"), ("Giresun", "Giresun"), ("Hatay", "Hatay"),
    ("Isparta", "Isparta"), ("Istanbul", "İstanbul"), ("Izmir", "İzmir"),
    ("Kars", "Kars"), ("Kastamonu", "Kastamonu"), ("Kayseri", "Kayseri"),
    ("Kocaeli", "Kocaeli"), ("Konya", "Konya"), ("Kutahya", "Kütahya"),
    ("Malatya", "Malatya"), ("Manisa", "Manisa"), ("Mardin", "Mardin"),
    ("Mersin", "Mersin"), ("Mugla", "Muğla"), ("Ordu", "Ordu"),
    ("Rize", "Rize"), ("Sakarya", "Sakarya"), ("Samsun", "Samsun"),
    ("Sivas", "Sivas"), ("Tekirdag", "Tekirdağ"), ("Trabzon", "Trabzon"),
    ("Urfa", "Urfa"), ("Van", "Van"), ("Zonguldak", "Zonguldak"),
]
# 40 sehir: `komsusu` SIMETRIK eslesme, tek sayi olsaydi bir sehir
# eslesmeden kalir ve komsulugu tek yonlu olurdu.

BOLGE_AD = [
    ("Marmara", "Marmara"), ("Ege", "Ege"), ("Akdeniz", "Akdeniz"),
    ("Karadeniz", "Karadeniz"), ("Icanadolu", "İç Anadolu"),
    ("Doguanadolu", "Doğu Anadolu"), ("Guneydogu", "Güneydoğu Anadolu"),
]

# SEHIR -> BOLGE: GERCEK cografya. Rastgele dagitilmiyordu ve ilk olcum
# "Gaziantep'in bolgesi Marmara Bolgesi'dir" uretti -- cumle Turkce
# olarak kusursuz ama DUNYA hakkinda yanlis, yani tam da kullanicinin
# istemedigi turden. Bu tek kenar TAM OLARAK belirli oldugu icin
# uydurmaya gerek YOK.
#
# !! `komsusu` ve UNIVERSITE -> SEHIR hala RASTGELE, ve bu KASITLI:
#    - gercek komsuluk kullanilsaydi komsular cogunlukla AYNI bolgede
#      olur, "komsusunun bolgesi" = "bolgesi" cikar ve o zincirler
#      turetilebilir olup SINAVDAN DUSERDI.
#    - universite -> sehir BIREBIR (her sehrin bir universitesi olsun
#      diye). Gercek dagilimda universiteler Istanbul/Ankara'da yigilir
#      ve sehirlerin yarisi bos kalirdi.
SEHIR_BOLGE = {
    "Balikesir": "Marmara", "Bursa": "Marmara", "Canakkale": "Marmara",
    "Edirne": "Marmara", "Istanbul": "Marmara", "Kocaeli": "Marmara",
    "Sakarya": "Marmara", "Tekirdag": "Marmara",
    "Aydin": "Ege", "Denizli": "Ege", "Izmir": "Ege", "Kutahya": "Ege",
    "Manisa": "Ege", "Mugla": "Ege",
    "Adana": "Akdeniz", "Antalya": "Akdeniz", "Hatay": "Akdeniz",
    "Isparta": "Akdeniz", "Mersin": "Akdeniz",
    "Giresun": "Karadeniz", "Kastamonu": "Karadeniz", "Ordu": "Karadeniz",
    "Rize": "Karadeniz", "Samsun": "Karadeniz", "Trabzon": "Karadeniz",
    "Zonguldak": "Karadeniz",
    "Ankara": "Icanadolu", "Eskisehir": "Icanadolu", "Kayseri": "Icanadolu",
    "Konya": "Icanadolu", "Sivas": "Icanadolu",
    "Elazig": "Doguanadolu", "Erzurum": "Doguanadolu", "Kars": "Doguanadolu",
    "Malatya": "Doguanadolu", "Van": "Doguanadolu",
    "Diyarbakir": "Guneydogu", "Gaziantep": "Guneydogu",
    "Mardin": "Guneydogu", "Urfa": "Guneydogu",
}
assert set(SEHIR_BOLGE) == {a for a, _ in SEHIR_AD}, (
    "SEHIR_BOLGE eksik/fazla -- her sehrin bolgesi YAZILI olmali")
assert set(SEHIR_BOLGE.values()) == {a for a, _ in BOLGE_AD}, (
    "bos bolge var -- `merkezi` kenari kurulamaz")

UNI_AD = [
    ("Bogazici", "Boğaziçi"), ("Hacettepe", "Hacettepe"), ("Gazi", "Gazi"),
    ("Selcuk", "Selçuk"), ("Dicle", "Dicle"), ("Firat", "Fırat"),
    ("Uludag", "Uludağ"), ("Anadolu", "Anadolu"), ("Cukurova", "Çukurova"),
    ("Ihlara", "Ihlara"), ("Atilim", "Atılım"), ("Bilkent", "Bilkent"),
    ("Kocatepe", "Kocatepe"), ("Pamukkale", "Pamukkale"),
    ("Harran", "Harran"), ("Inonu", "İnönü"), ("Erciyes", "Erciyes"),
    ("Sogut", "Söğüt"), ("Trakya", "Trakya"), ("Mimarsinan", "Mimar Sinan"),
    ("Galatasaray", "Galatasaray"), ("Yeditepe", "Yeditepe"),
    ("Bahcesehir", "Bahçeşehir"), ("Isikli", "Işıklı"), ("Baskent", "Başkent"),
    ("Cankaya", "Çankaya"), ("Ufuk", "Ufuk"), ("Nigde", "Niğde"),
    ("Munzur", "Munzur"), ("Bandirma", "Bandırma"), ("Toros", "Toros"),
    ("Alanya", "Alanya"), ("Bozok", "Bozok"), ("Artuklu", "Artuklu"),
    ("Recepbey", "Recep Bey"), ("Hitit", "Hitit"), ("Kapadokya", "Kapadokya"),
    ("Sanko", "Sanko"), ("Beykoz", "Beykoz"), ("Esenyurt", "Esenyurt"),
]

# NOTR onek havuzu: semt / mahalle adlari. FAKULTE ve BOLUM adlarinin ilk
# yuvasi buradan gelir ve USTUNDEKI varligin adiyla ILGISI YOKTUR.
SEMT = [
    ("Cerrahpasa", "Cerrahpaşa"), ("Kandilli", "Kandilli"),
    ("Beytepe", "Beytepe"), ("Tandogan", "Tandoğan"), ("Goztepe", "Göztepe"),
    ("Bornova", "Bornova"), ("Cayirova", "Çayırova"), ("Alasehir", "Alaşehir"),
    ("Karabaglar", "Karabağlar"), ("Bahcelievler", "Bahçelievler"),
    ("Sariyer", "Sarıyer"), ("Kadikoy", "Kadıköy"),
    ("Etimesgut", "Etimesgut"), ("Kecioren", "Keçiören"),
    ("Balcova", "Balçova"), ("Yenimahalle", "Yenimahalle"),
    ("Maltepe", "Maltepe"), ("Osmangazi", "Osmangazi"),
    ("Sehitkamil", "Şehitkamil"), ("Melikgazi", "Melikgazi"),
]
# Havuz IKIYE BOLUNUYOR: FAKULTE onekleri ile BOLUM onekleri AYRIK.
# Sebep OLCULDU (17 Eylul, `_denetle` yakaladi): ortak havuzda
# `Alasehir_Elektrik_Bolumu` rastgele `Alasehir_..._Fakultesi`ye baglandi
# ve bolumun adi kendi fakultesini SIZDIRDI. Ayrik havuzda bu YAPISAL
# OLARAK imkansiz, reddet-tekrar dene gerekmiyor.
SEMT_FAK = SEMT[:10]
SEMT_BOL = SEMT[10:]

FAK_ALAN = [
    ("Muhendislik", "Mühendislik"), ("Tip", "Tıp"), ("Hukuk", "Hukuk"),
    ("Fen", "Fen"), ("Edebiyat", "Edebiyat"), ("Iktisat", "İktisat"),
    ("Egitim", "Eğitim"), ("Ilahiyat", "İlahiyat"), ("Ziraat", "Ziraat"),
    ("Mimarlik", "Mimarlık"),
]
BOL_ALAN = [
    ("Bilgisayar", "Bilgisayar"), ("Makine", "Makine"),
    ("Elektrik", "Elektrik"), ("Kimya", "Kimya"), ("Fizik", "Fizik"),
    ("Matematik", "Matematik"), ("Tarih", "Tarih"), ("Felsefe", "Felsefe"),
    ("Isletme", "İşletme"), ("Maliye", "Maliye"), ("Biyoloji", "Biyoloji"),
    ("Cografya", "Coğrafya"), ("Psikoloji", "Psikoloji"),
    ("Sosyoloji", "Sosyoloji"), ("Insaat", "İnşaat"),
    ("Endustri", "Endüstri"),
]

DERS_ONEK = [
    ("Genel", "Genel"), ("Ileri", "İleri"), ("Temel", "Temel"),
    ("Uygulamali", "Uygulamalı"), ("Kuramsal", "Kuramsal"),
    ("Modern", "Modern"), ("Klasik", "Klasik"), ("Sayisal", "Sayısal"),
    ("Deneysel", "Deneysel"), ("Karsilastirmali", "Karşılaştırmalı"),
    ("Cagdas", "Çağdaş"), ("Niceliksel", "Niceliksel"),
    ("Bolgesel", "Bölgesel"), ("Evrensel", "Evrensel"), ("Analitik", "Analitik"),
]
DERS_KOK = [
    ("Analiz", "Analiz"), ("Cebir", "Cebir"), ("Istatistik", "İstatistik"),
    ("Mekanik", "Mekanik"), ("Optik", "Optik"), ("Termodinamik", "Termodinamik"),
    ("Anatomi", "Anatomi"), ("Genetik", "Genetik"), ("Ekoloji", "Ekoloji"),
    ("Arkeoloji", "Arkeoloji"), ("Dilbilgisi", "Dilbilgisi"),
    ("Mantik", "Mantık"), ("Ekonometri", "Ekonometri"),
    ("Muhasebe", "Muhasebe"), ("Algoritma", "Algoritma"),
    ("Veritabani", "Veritabanı"), ("Robotik", "Robotik"),
    ("Malzeme", "Malzeme"), ("Akiskanlar", "Akışkanlar"),
    ("Kartografya", "Kartografya"),
]

# TURKCE YAZIM TABLOSU -- yalniz `yuzey()` kullanir.
TR = {a: b for lst in (ERKEK, KADIN, SOYAD, SEHIR_AD, BOLGE_AD, UNI_AD,
                       SEMT, FAK_ALAN, BOL_ALAN, DERS_ONEK, DERS_KOK)
      for a, b in lst}
TR.update({"Universitesi": "Üniversitesi", "Fakultesi": "Fakültesi",
           "Bolumu": "Bölümü", "Bolgesi": "Bölgesi"})

# ILISKI jetonu KOKTUR, iyelik ekini dizideki <SI> verir:
#     cocuk <SI>        -> "cocugu"
#     cocuk <SI> <NIN>  -> "cocugunun"
# (taban_05.py 808 ve 843 bunu boyle kuruyor, veri_04 de boyleydi.)
#
# !! ILK SURUMDE BU KACIRILDI: iliskiler "annesi", "fakultesi" diye
# EK YAPISIK adlandirilmisti, yani dizi fiilen "annesi" + <SI> =
# "annesisi" diyordu. Kullanici yuzeyden yakaladi (17 Eylul).
#
# Yuzey bicimleri TABLODA, uretilmiyor: Turkce'de unsuz yumusamasi var
# ve kural tabanli uretim burada yanlis sonuc verir --
#     rakip -> rakibi   (p -> b)      cocuk -> cocugu  (k -> g)
#     sehir -> sehri    (unlu duser)
TR_ILISKI = {
    "bolumu": "bölümü", "fakultesi": "fakültesi", "universitesi": "üniversitesi",
    "sehri": "şehri", "bolgesi": "bölgesi",
    "dersi": "dersi", "merkezi": "merkezi",
    "rektoru": "rektörü", "kurucusu": "kurucusu", "dekani": "dekanı",
    "baskani": "başkanı", "hocasi": "hocası", "valisi": "valisi",
    "annesi": "annesi", "babasi": "babası", "kardesi": "kardeşi",
    "cocugu": "çocuğu", "danismani": "danışmanı", "ogrencisi": "öğrencisi",
    "arkadasi": "arkadaşı", "memleketi": "memleketi",
    "tezi": "tezi", "komsusu": "komşusu", "rakibi": "rakibi",
    "onkosulu": "ön koşulu",
}

# EK BICIMLERI -- kullanici karari, 17 Eylul:
#   *"bazilarinda 'in bazilarinda 'nin olmasi lazim, bunlar da ayri
#    token degil mi?"*
# Evet. Ilk surumde tamlayan eki TEK jetondu (<NIN>) ve butun
# allomorflari orturuyordu. Turkce'de sekiz bicimi var; hangisinin
# gelecegi ONCEKI KELIMEDEN belirli (unlu uyumu + son harf sesli mi).
# Yani YENI BILGI TASIMIYOR -- dilin gercek yuzeyi oldugu icin var.
EK_NIN = ("in", "ın", "un", "ün", "nin", "nın", "nun", "nün")
EK_DIR = ("dir", "dır", "dur", "dür", "tir", "tır", "tur", "tür")


def _ek_nin(k):
    """`k` (TURKCE yazim) den sonra gelen tamlayan ekinin EK_NIN indeksi.
        Yilmaz -> 'in   Kaya -> 'nin   Demir -> 'in   Ozturk -> 'un
        kardesi -> nin  bolumu -> nun            (cins isim, kesme YOK)"""
    i = "iıuü".index(_dort(k))
    return i + (4 if k[-1].lower() in SESLI else 0)


def _ek_dir(k):
    """`k` den sonra gelen bildirme ekinin EK_DIR indeksi.
        Yilmaz -> 'dir   Celik -> 'tir (sert unsuz)   Kocaeli -> 'dir"""
    i = "iıuü".index(_dort(k))
    return i + (4 if k[-1].lower() in SERT else 0)


def ek_secim(G):
    """Motor icin: hangi varliktan/iliskiden sonra HANGI ek bicimi gelir.

    Motor Turkce bilmez -- unlu uyumunu VERI MODULU hesaplar ve
    indeks olarak verir. `taban_05.veri_kur` bunu okur.

    Ek, ismin SON gercek yuvasina takilir ("Cerrahpasa Muhendislik
    Fakultesi'nin" -> son yuva "Fakultesi").
    """
    son = lambda a: TR.get(a.split("_")[-1], a.split("_")[-1])
    return dict(
        nin_varlik={a: _ek_nin(son(a)) for a in G["tip"]},
        dir_varlik={a: _ek_dir(son(a)) for a in G["tip"]},
        nin_iliski={r: _ek_nin(TR_ILISKI[r]) for r in ILISKI},
        # ILISKI + BILDIRME: "annesi" -> "annesidir".  model_06'nin yeni
        # yuzey bicimi icin (kullanici, 17 Eylul: *"Annesidir Fatma Yilmaz
        # Ayse Yilmaz'in, bu da olur"*). Turkce'de yuklem one alinabilir ve
        # rolu EK tasir; model konumdan degil ekten anlamak zorunda kalsin
        # diye. Tablo BURADA, motor unlu uyumu HESAPLAMAZ.
        dir_iliski={r: _ek_dir(TR_ILISKI[r]) for r in ILISKI},
        # "kimdir" / "neresidir" / "hangisidir" -- kimlik satiri icin.
        dir_soru={w: _ek_dir(w) for w in set(SORU_SOZ.values())},
    )


# SORU SOZCUGU -- cevabin TIPINE gore. Kullanici karari, 17 Eylul:
# dizide `?` bir KELIME degil, cumle siniri; soru sozcugu ONUN ONUNE gelir.
#     "Hatice Yilmaz'in kardesi kim? Sinan Yilmaz'dir."
# `yuzey()` bunu basiyor. DIZIYE eklenmesi AYRI bir karar (A duğmesi) --
# Diziye EKLENDI (ek_kip='tr2'), t_len turetimine dahil.
SORU_SOZ = {"KISI": "kim", "SEHIR": "neresi", "BOLGE": "neresi",
            "UNIVERSITE": "hangisi", "FAKULTE": "hangisi",
            "BOLUM": "hangisi", "DERS": "hangisi"}

# ===================================================================== SEMA
# iliski -> {kaynak tipi: hedef tipi}.
#
# TIP ORTUSMESI KASITLI ve SINIRLI: `bolumu` hem DERS hem KISI'den,
# `universitesi` hem FAKULTE hem SEHIR'den, `baskani` hem BOLUM hem
# BOLGE'den cikiyor. Ortusme olmadan kisayol TIP OLARAK hep imkansiz
# olur, AYIRT sinifi (sinavin yapildigi yer) kalmaz.
#
# !! veri_04'ten FARK: ortusme UYDURULMADI. Orada `kardes` OKUL'a,
# `kurucu` DERS'e aciliyordu ("Matematik dersinin kurucusu") -- 400 olgu
# bu turdendi. Burada her (kaynak, iliski) cifti Turkce'de SOYLENEBILIR.
SEMA = {
    # --- HIYERARSI, YUKARI (cok -> bir)
    # !! ASAGI YONLU KENAR AYRI BIR SEMBOL DEGIL -- kullanici, 17 Eylul:
    # *"ben duzgun bir turkce ile egitim istiyorum."*  Ilk surumde
    # `ana_fakultesi` / `ana_bolumu` / `ana_dersi` diye UC ayri iliski
    # vardi; uclu de Turkce'de IKI KELIME ve tek jetona sikismislardi.
    # Turkce zaten AYNI kelimeyi kullaniyor:
    #     "bolumun fakultesi"        (asagidan yukari)
    #     "universitenin fakultesi"  (yukaridan asagi)
    # Ayni sembol, KAYNAK TIPI farkli. |R| 27 -> 25.
    "bolumu":        {"DERS": "BOLUM", "KISI": "BOLUM", "FAKULTE": "BOLUM"},
    "fakultesi":     {"BOLUM": "FAKULTE", "UNIVERSITE": "FAKULTE"},
    "universitesi":  {"FAKULTE": "UNIVERSITE", "SEHIR": "UNIVERSITE"},
    "sehri":         {"UNIVERSITE": "SEHIR"},
    "bolgesi":       {"SEHIR": "BOLGE"},
    # --- HIYERARSI, ASAGI ("amiral" kenar -- phi'yi buradan aliyoruz)
    "dersi":         {"BOLUM": "DERS"},
    "merkezi":       {"BOLGE": "SEHIR"},
    # --- GOREV (X -> KISI)
    "rektoru":       {"UNIVERSITE": "KISI"},
    "kurucusu":      {"UNIVERSITE": "KISI"},
    "dekani":        {"FAKULTE": "KISI"},
    "baskani":       {"BOLUM": "KISI", "BOLGE": "KISI"},
    "hocasi":        {"DERS": "KISI"},
    "valisi":        {"SEHIR": "KISI"},
    # --- AILE (GERCEK soy agaci -- cinsiyet ve kusak tutarli)
    "annesi":        {"KISI": "KISI"},
    "babasi":        {"KISI": "KISI"},
    "kardesi":       {"KISI": "KISI"},     # simetrik, TERS cinsiyet
    "cocugu":        {"KISI": "KISI"},     # annesi/babasi TERSI
    # --- KISININ GERI KALANI
    "danismani":     {"KISI": "KISI"},
    "ogrencisi":     {"KISI": "KISI"},     # danismani TERSI
    "arkadasi":  {"KISI": "KISI"},     # simetrik
    "memleketi":    {"KISI": "SEHIR"},
    "tezi":    {"KISI": "DERS"},
    # --- YATAY
    "komsusu":       {"SEHIR": "SEHIR"},   # simetrik
    "rakibi":        {"UNIVERSITE": "UNIVERSITE"},   # simetrik
    "onkosulu":      {"DERS": "DERS"},
}
ILISKI = list(SEMA)
TIPLER = ["KISI", "UNIVERSITE", "FAKULTE", "BOLUM", "DERS", "SEHIR", "BOLGE"]

# Anlamca BIRBIRINI GEREKTIREN ciftler: (r1, r2) -> r_esdeger.
# "kardesinin annesi" = "annesi", cunku kardesler ayni ebeveynden.
# Bunlar AYNI sinifina duser ve sinav disi kalir; burada listelenmelerinin
# sebebi BELGEDE SAYILABILMELERI.
GEREKTIRIR = {
    ("kardesi", "annesi"): "annesi",
    ("kardesi", "babasi"): "babasi",
    ("annesi", "cocugu"): "kardesi",   # ya KENDISI (DONUS) ya kardes
    ("babasi", "cocugu"): "kardesi",
}

# TERS CIFT ADAYLARI: "X -r1-> Y, sonra Y -r2-> ?  ... X'e DONER MI".
# `graf_05` bunlarin oranini olcer. Liste BURADA duruyor cunku SEMADAN
# TURETILEMEZ: hangi ciftin geri donmesi BEKLENDIGI anlamsal bir iddia,
# yapisal degil. Araca gomulu kalsaydi (veri_04'te oyleydi) veri
# degisince arac SESSIZCE bos tablo basardi -- fiilen oldu, 17 Eylul:
# graf_05 `kardes` arayip KeyError verdi.
TERS_ADAY = [
    ("annesi", "cocugu"), ("babasi", "cocugu"),
    ("cocugu", "annesi"), ("cocugu", "babasi"),
    ("danismani", "ogrencisi"), ("ogrencisi", "danismani"),
    ("kardesi", "kardesi"), ("arkadasi", "arkadasi"),
    ("komsusu", "komsusu"), ("rakibi", "rakibi"),
    ("universitesi", "fakultesi"), ("fakultesi", "universitesi"),
    ("fakultesi", "bolumu"), ("bolumu", "fakultesi"),
    ("bolumu", "dersi"), ("dersi", "bolumu"),
    ("bolgesi", "merkezi"), ("merkezi", "bolgesi"),
]
assert all(a in SEMA and b in SEMA for a, b in TERS_ADAY), "TERS_ADAY semada YOK"

# Bir kisinin cocuklari BU iliskilerin kayitlarindan turetilir.
EBEVEYN = ("annesi", "babasi")

# --- OLCEK -------------------------------------------------------------
# Kullanici onayi, 17 Eylul: 480 kisi / 300 ders / 160 bolum / 60 fakulte
# / 40 universite / 40 sehir / 7 bolge.
BLOK = 48          # bir SOYADI blogunda kac kisi (3 kusak x 16)
KUSAK_EN = 16      # bir kusakta kac kisi (8 erkek + 8 kadin)
N_BLOK = 10        # kac soyadi blogu   -> 480 kisi
N_DERS, N_BOLUM, N_FAKULTE, N_UNI = 300, 160, 60, 40


def _esle(rng, idx):
    """idx'i rastgele IKISERLI eslestir; simetrik sozluk dondur."""
    p = list(rng.permutation(list(idx)))
    d = {}
    for k in range(0, len(p) - 1, 2):
        d[p[k]], d[p[k + 1]] = p[k + 1], p[k]
    return d


def _devirsiz(rng, idx):
    """Sabit noktasiz birebir esleme (derangement). Tersi de dondurulur."""
    idx = list(idx)
    for _ in range(2000):
        p = list(rng.permutation(idx))
        if all(x != y for x, y in zip(idx, p)):
            return dict(zip(idx, p)), dict(zip(p, idx))
    raise RuntimeError("devirsiz esleme bulunamadi")


def _farkli_esle(rng, idx, yasak, ad, deneme=600):
    """Simetrik esleme kur, ama `yasak` eslemelerinin HICBIRIYLE ayni
    olmasin. Iki iliski ayni eslemeyse model birini digerinden OKUR."""
    idx = list(idx)
    for _ in range(deneme):
        e = _esle(rng, idx)
        if all(all(e.get(i) != y.get(i) for i in idx) for y in yasak):
            return e
    raise RuntimeError(f"{ad}: farkli simetrik esleme bulunamadi")


# ======================================================================
# GRAF
# ======================================================================
def _kur_hiyerarsi(rng, ad, olgu):
    """BOLGE -> SEHIR -> UNIVERSITE -> FAKULTE -> BOLUM -> DERS.

    Yukari kenarlar COK->BIR (her fakultenin bir universitesi), asagi
    kenarlar BIR->BIR ("amiral"): universitenin ana fakultesi, fakultenin
    ana bolumu, bolumun ana dersi, bolgenin merkezi.
    """
    bolge = [f"{a}_Bolgesi" for a, _ in BOLGE_AD]
    sehir = [a for a, _ in SEHIR_AD]
    uni = [f"{a}_Universitesi" for a, _ in UNI_AD][:N_UNI]

    # --- FAKULTE: onek SEMT'ten, universitenin adiyla ILGISIZ (NOTR).
    fak = []
    for i in range(N_FAKULTE):
        on = SEMT_FAK[i % len(SEMT_FAK)][0]
        al = FAK_ALAN[(i // len(SEMT_FAK)) % len(FAK_ALAN)][0]
        fak.append(f"{on}_{al}_Fakultesi")
    bol = []
    for i in range(N_BOLUM):
        on = SEMT_BOL[i % len(SEMT_BOL)][0]
        al = BOL_ALAN[(i // len(SEMT_BOL)) % len(BOL_ALAN)][0]
        bol.append(f"{on}_{al}_Bolumu")
    ders = []
    for i in range(N_DERS):
        on = DERS_ONEK[i % len(DERS_ONEK)][0]
        kk = DERS_KOK[(i // len(DERS_ONEK)) % len(DERS_KOK)][0]
        ders.append(f"{on}_{kk}")
    for nm, lst in (("FAKULTE", fak), ("BOLUM", bol), ("DERS", ders)):
        assert len(set(lst)) == len(lst), f"{nm} adlari TEKRAR ediyor"

    ad.update(BOLGE=bolge, SEHIR=sehir, UNIVERSITE=uni,
              FAKULTE=fak, BOLUM=bol, DERS=ders)

    # --- YUKARI kenarlar. Dengeli dagitim: i % len(ust).
    #     `rng.permutation` ile karistiriliyor ki "i. bolum i%60. fakulte"
    #     gibi ARITMETIK bir kural dogmasin (veri_04'te sabit kaydirmalar
    #     tam bu yuzden kaldirilmisti).
    def bagla(alt, ust, r):
        p = list(rng.permutation(len(alt)))
        for k, i in enumerate(p):
            olgu[(alt[i], r)] = ust[k % len(ust)]

    # GERCEK cografya -- `bagla` ile DAGITILMIYOR (SEHIR_BOLGE'ye bak).
    for s in sehir:
        olgu[(s, "bolgesi")] = f"{SEHIR_BOLGE[s]}_Bolgesi"
    bagla(uni, sehir, "sehri")
    bagla(fak, uni, "universitesi")
    bagla(bol, fak, "fakultesi")
    bagla(ders, bol, "bolumu")
    bagla(sehir, uni, "universitesi")      # sehrin (ana) universitesi

    # --- ASAGI kenarlar: ustteki varligin "amiral" alti. Altindakiler
    #     arasindan secilir -- yani `ana_fakultesi`nin `universitesi`
    #     KENDISIDIR (DONUS sinifi, sinav disi) ve bu DOGRU.
    def amiral(ust, alt, r_yukari, r_asagi):
        alti = {}
        for a in alt:
            alti.setdefault(olgu[(a, r_yukari)], []).append(a)
        for u in ust:
            g = alti.get(u)
            assert g, f"{u} altinda hic {r_asagi} yok -- dagitim bozuk"
            olgu[(u, r_asagi)] = g[int(rng.randint(len(g)))]

    amiral(uni, fak, "universitesi", "fakultesi")
    amiral(fak, bol, "fakultesi", "bolumu")
    amiral(bol, ders, "bolumu", "dersi")
    amiral(bolge, sehir, "bolgesi", "merkezi")

    # --- YATAY
    kom = _esle(rng, range(len(sehir)))
    for i, s in enumerate(sehir):
        olgu[(s, "komsusu")] = sehir[kom[i]]
    rak = _esle(rng, range(len(uni)))
    for i, u in enumerate(uni):
        olgu[(u, "rakibi")] = uni[rak[i]]
    onk, _ = _devirsiz(rng, range(len(ders)))
    for i, d in enumerate(ders):
        olgu[(d, "onkosulu")] = ders[onk[i]]


def _kur_aile(rng, ad, olgu):
    """GERCEK soy agaci. Her soyadi blogu = 3 kusak x 16 kisi.

    Kusak icinde 8 erkek (M) + 8 kadin (F). Iki AYRI eslesme var ve
    KAYIK olmalari SART:

        kardes ciftleri   (M_i , F_{i+1})
        ebeveyn ciftleri  (M_i , F_i)     -> kusak+1'in i. kardes cifti

    Kayik olmasaydi ayni iki kisi hem kardes hem ebeveyn olurdu. `_denetle`
    bunu ayrica sinar.

    KENARDA BIRAKILANLAR (uydurulmuyor):
        kusak 0  ->  annesi / babasi YOK
        kusak 2  ->  cocugu YOK
    """
    kisi, cinsiyet, kusak = [], {}, {}
    n2 = KUSAK_EN // 2
    for b in range(N_BLOK):
        soy = SOYAD[b % len(SOYAD)][0]
        blok = []                    # blok[k][0]=erkekler, blok[k][1]=kadinlar
        # Ad havuzlari blok BASINA karistiriliyor: ayni ad iki blokta
        # farkli kusakta cikabilsin, "Ahmet hep dede" olmasin.
        em = list(rng.permutation(len(ERKEK)))
        km = list(rng.permutation(len(KADIN)))
        s = 0
        for k in range(3):
            e_, k_ = [], []
            for j in range(n2):
                e_.append(f"{ERKEK[em[s + j]][0]}_{soy}")
                k_.append(f"{KADIN[km[s + j]][0]}_{soy}")
            s += n2
            for x in e_:
                cinsiyet[x] = "E"
                kusak[x] = k
            for x in k_:
                cinsiyet[x] = "K"
                kusak[x] = k
            blok.append((e_, k_))
            kisi += e_ + k_
        # --- kardes: (M_i , F_{i+1})  simetrik, TERS cinsiyet
        for k in range(3):
            e_, k_ = blok[k]
            for i in range(n2):
                a, c = e_[i], k_[(i + 1) % n2]
                olgu[(a, "kardesi")] = c
                olgu[(c, "kardesi")] = a
        # --- ebeveyn: (M_i , F_i) kusak k  ->  kusak k+1'in i. kardes cifti
        for k in range(2):
            pe, pk = blok[k]
            ce, ck = blok[k + 1]
            for i in range(n2):
                baba, anne = pe[i], pk[i]
                ogul, kiz = ce[i], ck[(i + 1) % n2]     # AYNI kardes cifti
                for c in (ogul, kiz):
                    olgu[(c, "babasi")] = baba
                    olgu[(c, "annesi")] = anne
                # `cocugu` TERSI. Baba oglu, anne kizi gosteriyor -- ikisi
                # AYRI cocuk. Ayni cocugu gosterselerdi "annesinin cocugu"
                # ile "babasinin cocugu" birebir ayni olur, iki iliski
                # birbirinden OKUNURDU.
                olgu[(baba, "cocugu")] = ogul
                olgu[(anne, "cocugu")] = kiz
    ad["KISI"] = kisi
    return cinsiyet, kusak


def _kur_roller(rng, ad, olgu, kusak):
    """Kisileri gorevlere ve KISI->X baglarina yerlestir."""
    kisi = ad["KISI"]
    n = len(kisi)
    sec = lambda: kisi[int(rng.randint(n))]

    for u in ad["UNIVERSITE"]:
        olgu[(u, "rektoru")] = sec()
        olgu[(u, "kurucusu")] = sec()
    for f in ad["FAKULTE"]:
        olgu[(f, "dekani")] = sec()
    for b in ad["BOLUM"]:
        olgu[(b, "baskani")] = sec()
    for d in ad["DERS"]:
        olgu[(d, "hocasi")] = sec()
    for s in ad["SEHIR"]:
        olgu[(s, "valisi")] = sec()
    for g in ad["BOLGE"]:
        olgu[(g, "baskani")] = sec()

    for k in kisi:
        olgu[(k, "memleketi")] = ad["SEHIR"][int(rng.randint(len(ad["SEHIR"])))]
        olgu[(k, "bolumu")] = ad["BOLUM"][int(rng.randint(len(ad["BOLUM"])))]
        olgu[(k, "tezi")] = ad["DERS"][int(rng.randint(len(ad["DERS"])))]

    # --- danismani / ogrencisi: TERS cift, sabit noktasiz.
    # Kusak KISITI YOK ve bu kasitli: `kusak` AILE ICI bir kavram, iki
    # ayri ailenin kusaklari kiyaslanamaz. Tek kisit AYNI AILE OLMAMASI
    # -- "kendi cocugunun ogrencisi" cikmasin diye.
    #
    # !! REDDET-TEKRAR DENE ILE KURULAMAZ: 480 kisinin HEPSININ birden
    # aile disina dusme olasiligi ~0.9^480, yani sifir. BLOK duzeyinde
    # kuruluyor -- once bloklar sabit noktasiz eslesiyor, sonra her blok
    # kendi hedef blogunun icine RASTGELE dagiliyor.
    # Blok eslemesi de SABIT KAYDIRMA DEGIL (veri_04'te kaydirmalar tam
    # bu yuzden kaldirilmisti: iki kaydirmanin BILESKESI yine kaydirmadir
    # ve model kopruyu kullanmadan cozer).
    sig, _ = _devirsiz(rng, range(N_BLOK))
    dan = {}
    for b in range(N_BLOK):
        hedef = list(rng.permutation(
            np.arange(sig[b] * BLOK, (sig[b] + 1) * BLOK)))
        for j, i in enumerate(range(b * BLOK, (b + 1) * BLOK)):
            dan[i] = int(hedef[j])
    ogr = {v: k for k, v in dan.items()}
    assert len(ogr) == len(kisi), "danismani birebir DEGIL"
    for i, k in enumerate(kisi):
        olgu[(k, "danismani")] = kisi[dan[i]]
        olgu[(k, "ogrencisi")] = kisi[ogr[i]]

    # --- oda_arkadasi: simetrik, `kardesi` ile AYNI OLMAYAN.
    yer = {k: i for i, k in enumerate(kisi)}
    kar = {i: yer[olgu[(k, "kardesi")]] for i, k in enumerate(kisi)}
    oda = _farkli_esle(rng, range(len(kisi)), [kar], "arkadasi")
    for i, k in enumerate(kisi):
        olgu[(k, "arkadasi")] = kisi[oda[i]]


def _denetle(ad, olgu, tip, cinsiyet, kusak):
    """Verinin SOYLEDIGI seyi gercekten yaptigini sinar.

    Bunlarin hepsi `veri_04`te ya YOKTU ya da TUTMUYORDU -- soy agaci
    orada rastgele bagliydi ve "Fatma'nin annesi Huseyin" uretebiliyordu.
    """
    # 1) SEMA disi olgu YOK
    for (e, r), h in olgu.items():
        assert r in SEMA, f"semada olmayan iliski: {r}"
        assert tip[e] in SEMA[r], f"{tip[e]} tipinden {r} CIKAMAZ: {e}"
        assert tip[h] == SEMA[r][tip[e]], (
            f"{e} {r} -> {h}: tip {tip[h]}, beklenen {SEMA[r][tip[e]]}")
    # 2) kendine giden olgu YOK
    kendi = [(e, r) for (e, r), h in olgu.items() if h == e]
    assert not kendi, f"kendine giden olgu: {kendi[:5]}"
    # 3) CINSIYET: annesi hep kadin, babasi hep erkek
    for (e, r), h in olgu.items():
        if r == "annesi":
            assert cinsiyet[h] == "K", f"{e} annesi ERKEK cikti: {h}"
        if r == "babasi":
            assert cinsiyet[h] == "E", f"{e} babasi KADIN cikti: {h}"
    # 4) KUSAK: ebeveyn tam BIR ust kusak
    for (e, r), h in olgu.items():
        if r in ("annesi", "babasi"):
            assert kusak[h] == kusak[e] - 1, (
                f"{e} (kusak {kusak[e]}) {r} -> {h} (kusak {kusak[h]})")
        if r == "cocugu":
            assert kusak[h] == kusak[e] + 1, f"{e} cocugu kusak atladi: {h}"
        if r == "kardesi":
            assert kusak[h] == kusak[e], f"{e} kardesi baska kusakta: {h}"
    # 5) KARDESLER ayni anne VE ayni babadan
    for k in ad["KISI"]:
        ks = olgu[(k, "kardesi")]
        for r in ("annesi", "babasi"):
            a, b = olgu.get((k, r)), olgu.get((ks, r))
            assert a == b, f"kardes {k}/{ks} farkli {r}: {a} / {b}"
    # 6) ENSEST YOK: ebeveyn cifti kardes OLAMAZ
    for k in ad["KISI"]:
        an, ba = olgu.get((k, "annesi")), olgu.get((k, "babasi"))
        if an and ba:
            assert olgu.get((an, "kardesi")) != ba, (
                f"{k}: annesi ve babasi KARDES ({an} / {ba})")
    # 7) cocugu, annesi/babasi'nin TERSI
    for k in ad["KISI"]:
        c = olgu.get((k, "cocugu"))
        if c:
            r = "annesi" if cinsiyet[k] == "K" else "babasi"
            assert olgu[(c, r)] == k, f"{k} cocugu {c}, ama tersi tutmuyor"
    # 8) SIMETRI
    for grup, r in ((ad["KISI"], "kardesi"), (ad["KISI"], "arkadasi"),
                    (ad["SEHIR"], "komsusu"), (ad["UNIVERSITE"], "rakibi")):
        bozuk = [x for x in grup if olgu[(olgu[(x, r)], r)] != x]
        assert not bozuk, f"{r} simetrik degil: {bozuk[:3]}"
    # 9) TERS CIFT: danismani / ogrencisi
    bozuk = [k for k in ad["KISI"]
             if olgu[(olgu[(k, "danismani")], "ogrencisi")] != k]
    assert not bozuk, f"danismani/ogrencisi ters degil: {bozuk[:3]}"
    # 10) AMIRAL kenar gercekten ALTINDAN secilmis
    for u, ra, ry in (("UNIVERSITE", "fakultesi", "universitesi"),
                      ("FAKULTE", "bolumu", "fakultesi"),
                      ("BOLUM", "dersi", "bolumu"),
                      ("BOLGE", "merkezi", "bolgesi")):
        for x in ad[u]:
            assert olgu[(olgu[(x, ra)], ry)] == x, (
                f"{x} {ra} -> {olgu[(x, ra)]}, ama {ry} geri gelmiyor")
    # 11) NOTRLUK: FAKULTE/BOLUM adinin ilk yuvasi ustundekini SIZDIRMASIN
    for a in ad["FAKULTE"]:
        u = olgu[(a, "universitesi")]
        assert a.split("_")[0] != u.split("_")[0], f"fakulte adi sizdiriyor: {a}"
    for a in ad["BOLUM"]:
        f = olgu[(a, "fakultesi")]
        assert a.split("_")[0] != f.split("_")[0], f"bolum adi sizdiriyor: {a}"


def kur(tohum=0):
    """model_05'in grafi."""
    rng = np.random.RandomState(1000 + tohum)
    ad, olgu = {}, {}
    cinsiyet, kusak = _kur_aile(rng, ad, olgu)
    _kur_hiyerarsi(rng, ad, olgu)
    _kur_roller(rng, ad, olgu, kusak)

    tip = {a: t for t in TIPLER for a in ad[t]}
    _denetle(ad, olgu, tip, cinsiyet, kusak)

    hepsi = [a for t in TIPLER for a in ad[t]]
    assert len(set(hepsi)) == len(hepsi), "varlik adi TEKRAR ediyor"
    parca = sorted({p for a in hepsi for p in a.split("_")})
    ozel = ["<pad>", "<soru>", "?", "<son>"]
    sozluk = ozel + ILISKI + parca
    G = dict(ad=ad, n={t: len(ad[t]) for t in TIPLER}, sozluk=sozluk,
             kim={s: i for i, s in enumerate(sozluk)}, tip=tip,
             olgu=olgu, sema=SEMA, iliski=ILISKI,
             cinsiyet=cinsiyet, kusak=kusak)
    if tohum == 0 and IZ:
        _iz = graf_izi(G)
        assert _iz == IZ, (
            f"veri_05: graf DEGISTI  {IZ} -> {_iz}\n"
            "  Kasitliysa IZ yenilenir ve onkayda not duselir; degilse\n"
            "  degisiklik geri alinir. model_05 sessizce baska bir\n"
            "  veriyle KOSMAZ.")
    return G


# ======================================================================
# ZINCIRLER
# ======================================================================
def zincirler(G):
    """Tip olarak gecerli 2 adimli zincirler, BES SINIFA ayrilmis.

    !! veri_04'ten FARK: sinif SEMAYA bakiyor, `olgu` sozluguNE degil.
    Orada her varligin semadaki her iliskisi DOLUYDU, burada degil --
    soy agacinin KENARI var (kusak 0'in annesi, kusak 2'nin cocugu).

        DONUS   cevap = ozne                       (kardesinin kardesi)
        YOK     kisayol TIP OLARAK imkansiz        -> `ent_yok` bolmesi
        EKSIK   tip mumkun, OLGU kayitli degil     -> SINAV DISI
        AYNI    kisayol cevabin AYNISI             (kardesinin annesi)
        AYIRT   kisayol VAR ve FARKLI              -> sinav BURADA

    YOK ile EKSIK ayrilmazsa `ent_yok` kirlenir: "kisayol imkansiz" diye
    sayilan zincirin bir kismi aslinda "kisayol var ama biz yazmadik"
    olurdu (CLAUDE.md, `ent_yok` tanimi).
    """
    olgu, tip, sema = G["olgu"], G["tip"], G["sema"]
    out = []
    for (e, r1), b in olgu.items():
        for r2 in G["iliski"]:
            if r2 == r1 or tip[b] not in sema[r2]:
                continue
            cev = olgu.get((b, r2))
            if cev is None:
                continue                      # KOPRUNUN kendisi eksik
            mumkun = tip[e] in sema[r2]
            ks = olgu.get((e, r2))
            sinif = ("DONUS" if cev == e else
                     "YOK" if not mumkun else
                     "EKSIK" if ks is None else
                     "AYNI" if ks == cev else "AYIRT")
            out.append((e, r1, r2, b, cev, ks, sinif))
    return out


def turetilebilir(G, esik=0.5):
    """(r1, r2) ciftinin sonucu, TEK bir r3 olgusuyla AYNI mi?

    Boyle bir cift varsa o zincir 2-hop DEGILDIR: model r3'u ezberleyip
    gecer. `GEREKTIRIR`de yazili olanlar BEKLENEN; bu tarama YENI bir
    tane dogmadigini sinar.
    """
    olgu, tip, sema = G["olgu"], G["tip"], G["sema"]
    out = []
    for r1 in ILISKI:
        for r2 in ILISKI:
            if r1 == r2:
                continue
            pay, top = {}, 0
            for (e, r), b in olgu.items():
                if r != r1 or tip[b] not in sema[r2]:
                    continue
                c = olgu.get((b, r2))
                if c is None:
                    continue
                top += 1
                for r3 in ILISKI:
                    if tip[e] in sema[r3] and olgu.get((e, r3)) == c:
                        pay[r3] = pay.get(r3, 0) + 1
            for r3, k in pay.items():
                if top and k / top > esik:
                    out.append((r1, r2, r3, k / top))
    return sorted(out, key=lambda x: -x[3])


def sizinti(G, z):
    """Zincir zincir: cevabin kac yuvasi SORUDA zaten duruyor.

    Aile soyadi BILEREK paylasiliyor (gercek hayatta da oyle), yani
    "Ahmet Yilmaz'in kardesi" sorusunda cevabin 2. yuvasi hazir. Bunu
    silmiyoruz -- SAYIYORUZ, ki bolme kodu `kopyalanabilir` /
    `kopyalanamaz` diye AYIRABILSIN.

    Doner: (kopyalanabilir_zincir_sayisi, yuva bazinda sayac).
    """
    say, kop = {}, 0
    for e, r1, r2, b, cev, ks, sinif in z:
        ep = set(e.split("_"))
        cp = cev.split("_")
        ortak = [j for j, p in enumerate(cp) if p in ep]
        for j in ortak:
            say[j] = say.get(j, 0) + 1
        if ortak:
            kop += 1
    return kop, say


# ======================================================================
# PARMAK IZI
# ======================================================================
# tohum 0 grafinin parmak izi. BOS iken `kur()` denetim YAPMAZ ve bu
# dosya "ilk kurulum" kipindedir. Ilk olcumden sonra DOLDURULUR.
IZ = "3431c633b6b1"


def graf_izi(G):
    """Grafin ICERIGINDEN tureyen sabit parmak izi.

    `ad` listeleri SIRASIYLA karisiyor -- varlik sirasi `sozluk`u,
    `sozluk` da JETON ID'lerini belirliyor, yani siralamak korlestirirdi.
    `sema` de dahil: gecerli zincir kumesi (SINAVIN KENDISI) ona bagli.
    """
    h = hashlib.sha256()
    h.update(repr(sorted(map(str, G["olgu"].items()))).encode())
    h.update(repr(list(G["iliski"])).encode())
    h.update(repr([(t, list(G["ad"][t])) for t in sorted(G["ad"])]).encode())
    h.update(repr(list(G["sozluk"])).encode())
    h.update(repr(sorted((r, sorted(m.items()))
                         for r, m in G["sema"].items())).encode())
    return h.hexdigest()[:12]


# ======================================================================
# YUZEY -- INSAN icin dogru Turkce. Model bunu GORMEZ.
# ======================================================================
KALIN_DUZ, KALIN_YUV, INCE_DUZ, INCE_YUV = "aı", "ou", "ei", "öü"
SESLI = KALIN_DUZ + KALIN_YUV + INCE_DUZ + INCE_YUV
SERT = "pçtkfhsş"


def _son_sesli(k):
    for h in reversed(k.lower()):
        if h in SESLI:
            return h
    return "a"


def _dort(k):
    s = _son_sesli(k)
    return ("ı" if s in KALIN_DUZ else "u" if s in KALIN_YUV
            else "i" if s in INCE_DUZ else "ü")


def _tamlayan(k, ozel=True):
    """-in / -ın / -un / -ün. Ozel isimde KESME ISARETI, cins isimde YOK."""
    e = _dort(k)
    ek = ("n" + e + "n") if k[-1].lower() in SESLI else (e + "n")
    return ("'" if ozel else "") + ek


def _bildirme(k):
    """-dir / -dır / -dur / -dür, sert unsuzden sonra -tir."""
    e = _dort(k)
    return ("'t" + e + "r") if k[-1].lower() in SERT else ("'d" + e + "r")


def tr_ad(a):
    """Jeton dizisini dogru Turkce yazima cevir."""
    return " ".join(TR.get(p, p) for p in a.split("_"))


def yuzey(G, e, r1, r2=None, cev=None, soru_soz=False):
    """2-hop (r2 verilirse) ya da 1-hop soru-cevap, dogru Turkce ekle.

    Iliski KOK olarak geliyor; iyelik ekini `TR_ILISKI` veriyor -- dizide
    o eki `<SI>` jetonu tasiyor.

        yuzey(G, e, "memleketi", "valisi", cev)
          -> Omer Demir'in doğum yerinin valisi?  Osman Doğan'dır

    `soru_soz=True` cevabin tipine gore soru sozcugunu de koyar:
          -> Hatice Yılmaz'ın kardeşi kim?  Sinan Yılmaz'dır
    Soru sozcugu DIZIDE de var (ek_kip='tr2').
    """
    s = tr_ad(e) + _tamlayan(tr_ad(e))
    son_r = r2 or r1
    if r2 is not None:
        # ARA iliski: iyelikli bicim + tamlayan.  "bölümü" -> "bölümünün"
        ara = TR_ILISKI[r1]
        s += " " + ara + _tamlayan(ara, ozel=False)
    s += " " + TR_ILISKI[son_r]
    if soru_soz:
        _t = G["tip"][cev] if cev else SEMA[son_r][
            G["tip"][G["olgu"][(e, r1)]] if r2 else G["tip"][e]]
        s += " " + SORU_SOZ[_t]
    s += "?"
    if cev is None:
        return s
    return s + "  " + tr_ad(cev) + _bildirme(tr_ad(cev))


# ======================================================================
if __name__ == "__main__":
    import collections

    G = kur(0)
    z = zincirler(G)
    sn = collections.Counter(x[6] for x in z)
    sinav = len(z) - sn["DONUS"] - sn["AYNI"] - sn["EKSIK"]

    print("=" * 74)
    print("VARLIKLAR")
    for t in TIPLER:
        print(f"   {t:<12}{G['n'][t]:>5}   ornek: "
              + ", ".join(tr_ad(a) for a in G["ad"][t][:2]))
    print(f"   {'TOPLAM':<12}{sum(G['n'].values()):>5}"
          f"   sozluk {len(G['sozluk'])} jeton"
          f"   yuva {max(len(a.split('_')) for a in G['tip'])}")

    print()
    print("=" * 74)
    print("SAYILAR")
    print(f"   olgu {len(G['olgu'])}   zincir {len(z)}   |R| {len(ILISKI)}")
    print(f"   phi TAVANI {len(z)/len(G['olgu']):.2f}"
          f"   (veri_04: 9.57)")
    print("   sinif " + "  ".join(f"{k} {sn[k]}" for k in
                                  ("AYIRT", "YOK", "AYNI", "DONUS", "EKSIK")))
    print(f"   SINAVA GIREN (AYIRT + YOK) {sinav}")
    kop, yv = sizinti(G, z)
    print(f"   SIZINTI  {kop}/{len(z)} zincirde cevabin en az bir yuvasi")
    print(f"            soruda ZATEN var.  yuva bazinda {dict(sorted(yv.items()))}")
    print("            (aile soyadi -- BILEREK, `sizinti()` ile ayrilabilir)")

    print()
    print("=" * 74)
    print("OLGU ORNEKLERI -- dogru Turkce")
    _g = [("Cerrahpasa", "universitesi"), ("Kecioren", "fakultesi")]
    gos = []
    for r in ("universitesi", "fakultesi", "bolumu", "ana_fakultesi",
              "sehri", "bolgesi", "merkezi", "annesi", "babasi", "kardesi",
              "cocugu", "valisi", "dekani", "onkosulu"):
        for (e, rr), h in G["olgu"].items():
            if rr == r:
                gos.append((e, r, h))
                break
    for e, r, h in gos:
        print("   " + yuzey(G, e, r, cev=h))

    print()
    print("=" * 74)
    print("2-HOP ZINCIR ORNEKLERI -- SINAVA GIRENLER")
    rs = np.random.RandomState(7)
    for et in ("AYIRT", "YOK"):
        alt = [x for x in z if x[6] == et]
        print(f"   --- {et}  ({len(alt)} zincir)")
        for i in rs.permutation(len(alt))[:6]:
            e, r1, r2, b, cev, ks, _ = alt[int(i)]
            print("   " + yuzey(G, e, r1, r2, cev))
            print(f"        KOPRU {tr_ad(b):<34}"
                  + (f"KISAYOL {tr_ad(ks)}" if ks else "kisayol TIP OLARAK YOK"))

    print()
    print("=" * 74)
    print("SOY AGACI -- tek bir kisinin cevresi")
    k = G["ad"]["KISI"][0]
    print(f"   {tr_ad(k)}   cinsiyet {G['cinsiyet'][k]}  kusak {G['kusak'][k]}")
    for r in ("annesi", "babasi", "kardesi", "cocugu", "danismani",
              "arkadasi", "bolumu", "memleketi", "tezi"):
        h = G["olgu"].get((k, r))
        print(f"      {r:<14}{tr_ad(h) if h else '(kayitli degil -- KENAR)'}")
    an = G["olgu"].get((k, "annesi"))
    if an:
        print(f"   {tr_ad(k)} -> annesi {tr_ad(an)} -> annesi "
              f"{tr_ad(G['olgu'].get((an,'annesi'))) if G['olgu'].get((an,'annesi')) else '(KENAR)'}"
              "   <- BUYUKANNE, anlamli")

    print()
    print("=" * 74)
    print("TURETILEBILIR CIFTLER (oran > 0.5)")
    t = turetilebilir(G)
    if not t:
        print("   (yok)")
    for r1, r2, r3, o in t:
        print("   %-14s %-14s -> %-14s  %.3f   %s"
              % (r1, r2, r3, o,
                 "BEKLENEN (GEREKTIRIR)" if (r1, r2) in GEREKTIRIR else "!! YENI"))

    print()
    print("   graf izi", graf_izi(G), " (IZ alani:", repr(IZ) + ")")
