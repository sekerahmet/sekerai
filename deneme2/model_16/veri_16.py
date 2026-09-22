# -*- coding: utf-8 -*-
"""veri_16 — BU KOLUN KENDI verisi. TEK BASINA DURUR.

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

Zincir yalniz YUKARI gidiyor. Asagi yonlu "amiral" kenarlar
(`ana_fakultesi`, `ana_bolumu`, `zorunlu_dersi`) 18 Eylul'de SILINDI:
uydurma kavramlardi ve olculdu -- AYIRT sinifini hic degistirmiyorlardi.

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

ZINCIRDEKI YERI.  Kim kimi cagiriyor, bu dosya nerede:
(model BIRIM goruyor -- karakter yalnizca ara adim ve kapi)

  veri_16     graf: 1608 varlik, 24 iliski, olgular   <-- BU DOSYA
  metin_16    graf -> duz Turkce cumle
  korpus_16   cumle -> belge -> paketlenmis akis
  jeton_16    KARAKTER sozlugu + GIDIS-DONUS KAPISI
  birim_16    metin -> sayim -> kok havuzu -> BIRIM AKISI -> pencere
  ek_16       kelime -> kok + ek    (Turkce morfolojisi)

  taban_16    bolmeler (ezber_* / cikarim_*) + Ayar tanimi
  ayar_16     dugmeler
  hazirla_16  veriyi dosyaya yazar, Colab Drive'dan OKUR

  model_16    MIMARI -- model_15'ten
  train_16    egitim dongusu
  olcme_16    olcu: soru soruldu, cevap dogru mu
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
    # TURKIYE'NIN 81 ILI (kullanici, 18 Eylul: "tr de 80 il var").
    # Onceden 40 secilmis il vardi ve sayi CIFT tutuluyordu, cunku
    # `komsusu` simetrik eslesmeydi. O iliski SILINDI (SEMA'ya bak),
    # dolayisiyla il sayisinin cift olmasi da gerekmiyor.
    ("Adana", "Adana"), ("Adiyaman", "Adıyaman"),
    ("Afyonkarahisar", "Afyonkarahisar"), ("Agri", "Ağrı"),
    ("Aksaray", "Aksaray"), ("Amasya", "Amasya"), ("Ankara", "Ankara"),
    ("Antalya", "Antalya"), ("Ardahan", "Ardahan"), ("Artvin", "Artvin"),
    ("Aydin", "Aydın"), ("Balikesir", "Balıkesir"), ("Bartin", "Bartın"),
    ("Batman", "Batman"), ("Bayburt", "Bayburt"), ("Bilecik", "Bilecik"),
    ("Bingol", "Bingöl"), ("Bitlis", "Bitlis"), ("Bolu", "Bolu"),
    ("Burdur", "Burdur"), ("Bursa", "Bursa"), ("Canakkale", "Çanakkale"),
    ("Cankiri", "Çankırı"), ("Corum", "Çorum"), ("Denizli", "Denizli"),
    ("Diyarbakir", "Diyarbakır"), ("Duzce", "Düzce"), ("Edirne", "Edirne"),
    ("Elazig", "Elazığ"), ("Erzincan", "Erzincan"), ("Erzurum", "Erzurum"),
    ("Eskisehir", "Eskişehir"), ("Gaziantep", "Gaziantep"),
    ("Giresun", "Giresun"), ("Gumushane", "Gümüşhane"),
    ("Hakkari", "Hakkari"), ("Hatay", "Hatay"), ("Igdir", "Iğdır"),
    ("Isparta", "Isparta"), ("Istanbul", "İstanbul"), ("Izmir", "İzmir"),
    ("Kahramanmaras", "Kahramanmaraş"), ("Karabuk", "Karabük"),
    ("Karaman", "Karaman"), ("Kars", "Kars"), ("Kastamonu", "Kastamonu"),
    ("Kayseri", "Kayseri"), ("Kilis", "Kilis"), ("Kirikkale", "Kırıkkale"),
    ("Kirklareli", "Kırklareli"), ("Kirsehir", "Kırşehir"),
    ("Kocaeli", "Kocaeli"), ("Konya", "Konya"), ("Kutahya", "Kütahya"),
    ("Malatya", "Malatya"), ("Manisa", "Manisa"), ("Mardin", "Mardin"),
    ("Mersin", "Mersin"), ("Mugla", "Muğla"), ("Mus", "Muş"),
    ("Nevsehir", "Nevşehir"), ("Nigde", "Niğde"), ("Ordu", "Ordu"),
    ("Osmaniye", "Osmaniye"), ("Rize", "Rize"), ("Sakarya", "Sakarya"),
    ("Samsun", "Samsun"), ("Sanliurfa", "Şanlıurfa"), ("Siirt", "Siirt"),
    ("Sinop", "Sinop"), ("Sirnak", "Şırnak"), ("Sivas", "Sivas"),
    ("Tekirdag", "Tekirdağ"), ("Tokat", "Tokat"), ("Trabzon", "Trabzon"),
    ("Tunceli", "Tunceli"), ("Usak", "Uşak"), ("Van", "Van"),
    ("Yalova", "Yalova"), ("Yozgat", "Yozgat"), ("Zonguldak", "Zonguldak"),
]
assert len(SEHIR_AD) == 81, f"81 il olmali, {len(SEHIR_AD)} var"

BOLGE_AD = [
    ("Marmara", "Marmara"), ("Ege", "Ege"), ("Akdeniz", "Akdeniz"),
    ("Karadeniz", "Karadeniz"), ("Icanadolu", "İç Anadolu"),
    ("Doguanadolu", "Doğu Anadolu"), ("Guneydogu", "Güneydoğu Anadolu"),
]

# SEHIR -> BOLGE: GERCEK cografya, 81 il. Rastgele dagitilmiyordu ve
# ilk olcum "Gaziantep'in bolgesi Marmara Bolgesi'dir" uretti -- cumle
# Turkce olarak kusursuz ama DUNYA hakkinda yanlis.
SEHIR_BOLGE = {}
for _b, _iller in (
    ("Marmara", "Balikesir Bilecik Bursa Canakkale Edirne Istanbul "
                "Kirklareli Kocaeli Sakarya Tekirdag Yalova"),
    ("Ege", "Afyonkarahisar Aydin Denizli Izmir Kutahya Manisa Mugla Usak"),
    ("Akdeniz", "Adana Antalya Burdur Hatay Isparta Kahramanmaras Mersin "
                "Osmaniye"),
    ("Icanadolu", "Aksaray Ankara Cankiri Eskisehir Karaman Kayseri "
                  "Kirikkale Kirsehir Konya Nevsehir Nigde Sivas Yozgat"),
    ("Karadeniz", "Amasya Artvin Bartin Bayburt Bolu Corum Duzce Giresun "
                  "Gumushane Karabuk Kastamonu Ordu Rize Samsun Sinop "
                  "Tokat Trabzon Zonguldak"),
    ("Doguanadolu", "Agri Ardahan Bingol Bitlis Elazig Erzincan Erzurum "
                    "Hakkari Igdir Kars Malatya Mus Tunceli Van"),
    ("Guneydogu", "Adiyaman Batman Diyarbakir Gaziantep Kilis Mardin "
                  "Siirt Sanliurfa Sirnak"),
):
    for _i in _iller.split():
        SEHIR_BOLGE[_i] = _b
assert set(SEHIR_BOLGE) == {a for a, _ in SEHIR_AD}, (
    "SEHIR_BOLGE eksik/fazla -- her ilin bolgesi YAZILI olmali")
assert set(SEHIR_BOLGE.values()) == {a for a, _ in BOLGE_AD}, (
    "bos bolge var -- hicbir ili olmayan bolge OLMAMALI")

# !! HICBIRI IL ADI DEGIL. Gercek hayatta universiteler sehirlerinin
# adini tasir (Ankara Universitesi Ankara'dadir) ama burada TASIMAMALI:
# tasisaydi model "X Universitesi'nin sehri?" sorusunu ADDAN okurdu ve
# `sehri` kenari kopyalanabilir olurdu. `_denetle` bunu sinar.
# ("Nigde" 18 Eylul'de "Zirve" oldu -- 81 il listesinde Nigde VAR.)
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
    ("Cankaya", "Çankaya"), ("Ufuk", "Ufuk"), ("Zirve", "Zirve"),
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
# --- TEZ adlari: 24 x 20 = 480, her kisiye bir tez.
# !! AYRI HAVUZ. `DERS_*` ile ortak kelime YOK -- tezin adi `konusu`nu
# SIZDIRMASIN diye (ayni gerekce SEMT_FAK / SEMT_BOL ayrimindaki gibi,
# `_denetle` §11). Tezin adi yazarini da sizdirmaz: ad havuzundan
# gelmiyor, yoksa "Ayse Yilmaz'in tezi?" sorusunun cevabi soruda
# ZATEN yazili olurdu (`sizinti()` bunu sayardi).
TEZ_ONEK = [
    ("Birincil", "Birincil"), ("Ikincil", "İkincil"), ("Ozgun", "Özgün"),
    ("Kapsamli", "Kapsamlı"), ("Elestirel", "Eleştirel"),
    ("Betimsel", "Betimsel"), ("Kesitsel", "Kesitsel"),
    ("Boylamsal", "Boylamsal"), ("Gorgul", "Görgül"),
    ("Butunsel", "Bütünsel"), ("Ayrintili", "Ayrıntılı"),
    ("Sistematik", "Sistematik"), ("Yenilikci", "Yenilikçi"),
    ("Oncu", "Öncü"), ("Tumlesik", "Tümleşik"), ("Bagimsiz", "Bağımsız"),
    ("Ortak", "Ortak"), ("Surekli", "Sürekli"), ("Asamali", "Aşamalı"),
    ("Donemsel", "Dönemsel"), ("Karsit", "Karşıt"), ("Seckin", "Seçkin"),
    ("Yorumsal", "Yorumsal"), ("Islevsel", "İşlevsel"),
]
TEZ_KOK = [
    ("Inceleme", "İnceleme"), ("Arastirma", "Araştırma"),
    ("Deneme", "Deneme"), ("Cozumleme", "Çözümleme"),
    ("Degerlendirme", "Değerlendirme"), ("Karsilastirma", "Karşılaştırma"),
    ("Model", "Model"), ("Yaklasim", "Yaklaşım"), ("Kuram", "Kuram"),
    ("Uygulama", "Uygulama"), ("Tasarim", "Tasarım"), ("Olcum", "Ölçüm"),
    ("Gozlem", "Gözlem"), ("Sinama", "Sınama"), ("Tarama", "Tarama"),
    ("Derleme", "Derleme"), ("Sentez", "Sentez"), ("Yorum", "Yorum"),
    ("Elestiri", "Eleştiri"), ("Onerme", "Önerme"),
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
                       SEMT, FAK_ALAN, BOL_ALAN, DERS_ONEK, DERS_KOK,
                       TEZ_ONEK, TEZ_KOK)
      for a, b in lst}
TR.update({"Universitesi": "Üniversitesi", "Fakultesi": "Fakültesi",
           "Bolumu": "Bölümü", "Bolgesi": "Bölgesi", "Tezi": "Tezi"})

# ILISKI jetonu KOKTUR, iyelik ekini dizideki <SI> verir:
#     cocuk <SI>        -> "cocugu"
#     cocuk <SI> <NIN>  -> "cocugunun"
# (taban_11.py 808 ve 843 bunu boyle kuruyor, veri_04 de boyleydi.)
#
# !! ILK SURUMDE BU KACIRILDI: iliskiler "annesi", "fakultesi" diye
# EK YAPISIK adlandirilmisti, yani dizi fiilen "annesi" + <SI> =
# "annesisi" diyordu. Kullanici yuzeyden yakaladi (17 Eylul).
#
# Yuzey bicimleri TABLODA, uretilmiyor: Turkce'de unsuz yumusamasi var
# ve kural tabanli uretim burada yanlis sonuc verir --
#     kitap -> kitabi   (p -> b)      cocuk -> cocugu  (k -> g)
#     sehir -> sehri    (unlu duser)
TR_ILISKI = {
    "bolumu": "bölümü", "fakultesi": "fakültesi", "universitesi": "üniversitesi",
    "sehri": "şehri", "bolgesi": "bölgesi",
    "rektoru": "rektörü", "kurucusu": "kurucusu", "dekani": "dekanı",
    "baskani": "başkanı", "hocasi": "hocası", "valisi": "valisi",
    "annesi": "annesi", "babasi": "babası", "kardesi": "kardeşi",
    "cocugu": "çocuğu", "danismani": "danışmanı", "ogrencisi": "öğrencisi",
    "arkadasi": "arkadaşı", "memleketi": "memleketi",
    # !! BIR KISININ UC SEHRI VAR (kullanici, 18 Eylul: *"bir kisinin
    # memleketi farkli, universitenin bulundugu sehir farkli, yasadigi
    # sehir farkli"*). Ucu de AYRI ADLA sorulabilsin diye `yasadigi_yer`
    # eklendi; boylece "X'in sehri" gibi BELIRSIZ bir soru hic dogmuyor.
    #   memleketi          dogdugu sehir
    #   yasadigi_yer       oturdugu sehir          YENI
    #   ...universitesinin sehri   okudugu sehir   (yolla, zaten vardi)
    # "yer" secildi, "sehir" DEGIL: Turkce'de "sehir" + tamlayan =
    # "sehrin" (unlu dusmesi) ve `_ek_nin` tablosu duzenli eki uretir,
    # "sehirin" yazardi. "yer" -> "yerin", duzenli.
    "yasadigi_yer": "yaşadığı yer",
    "tezi": "tezi", "yazari": "yazarı", "konusu": "konusu",
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

# --- HAL EKLERI (model_11) -----------------------------------------------
# Ters ifadeler icin: "Marmara Bolgesi'NDEKI illerden biri Bursa'dir",
# "Bogazici Universitesi'NE bagli fakultelerden biri ...".
# model_09'da bunlara ihtiyac yoktu -- her cumle tamlayan + bildirme
# ile kuruluyordu. model_11'da varligin KENDI SAYFASI yaziliyor ve
# ters ifade hal eki istiyor.
#
# !! IKI YONLU unlu uyumu (EK_NIN/EK_DIR DORT yonlu). Ayri fonksiyon.
# !! KAYNASTIRMA -n- (zamir n'si): IYELIK ekiyle biten adlarda hal eki
# arasina 'n' girer.  "Bolgesi" + de  ->  "Bolgesi'NDE",  degil "Bolgesi'de".
# Bizde bes tip boyle biter: BOLGE / UNIVERSITE / FAKULTE / BOLUM / TEZ
# ("Marmara Bolgesi", "Bogazici Universitesi", "... Fakultesi",
#  "... Bolumu", "... Tezi"). Sehir, kisi ve ders adlari BITMEZ.
# Iyelik ekli ad DAIMA sesliyle biter, o yuzden sert-unsuz dali YOK.
EK_DE = ("de", "da", "te", "ta", "nde", "nda")
EK_DEN = ("den", "dan", "ten", "tan", "nden", "ndan")
EK_E = ("e", "a", "ye", "ya", "ne", "na")


def _iki(k):
    """IKI YONLU unlu uyumu: 0 = ince (e), 1 = kalin (a)."""
    return 1 if _son_sesli(k) in (KALIN_DUZ + KALIN_YUV) else 0


def _ek_de(k, iyelik=False):
    """Bulunma eki indeksi. Sert unsuzle bitiyorsa -te/-ta."""
    if iyelik:
        return 4 + _iki(k)
    return _iki(k) + (2 if k[-1].lower() in SERT else 0)


def _ek_den(k, iyelik=False):
    """Ayrilma eki indeksi. Ayni kural."""
    if iyelik:
        return 4 + _iki(k)
    return _iki(k) + (2 if k[-1].lower() in SERT else 0)


def _ek_e(k, iyelik=False):
    """Yonelme eki indeksi. Sesliyle bitiyorsa KAYNASTIRMA 'y'."""
    if iyelik:
        return 4 + _iki(k)
    return _iki(k) + (2 if k[-1].lower() in SESLI else 0)


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
    indeks olarak verir. `taban_11.veri_kur` bunu okur.

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
# TIP ADI -- "Manisa bir SEHIRdir." Kullanici karari, 18 Eylul:
# *"Ayse Yilmaz bir kisidir. Manisa bir sehirdir gibi bilgileri egitime
# ekle."* Bu bilgi model_09'da HIC YOKTU: kimlik cumlesi ("X kimdir?
# X'dir.") dongusel, tipi soylemiyor.
#
# EN DEGERLI UC TIPTE: KISI / SEHIR / DERS -- adlari tiplerini ELE
# VERMIYOR ("Manisa", "Genel Analiz" herhangi bir sey olabilir).
# Digerleri adinda tasiyor ama cumle yine de dogal.
#
# "YOK" cevabinin TEMELI: "Adana'nin tezi olmaz" diyebilmek icin
# modelin Adana'nin SEHIR oldugunu ve sehirlerin tezi olmadigini
# bilmesi gerek.
# !! UNLU DUSMESI. Turkce'de bazi iki heceli adlarda ek gelince ikinci
# hecenin unlusu duser:  sehir -> SEHRIN  (sehirin DEGIL).
# Bizim tip adlarinda yalniz "sehir" boyle. Tablo elle yazilir, kural
# genellenmez -- "bolum" DUSMEZ (bolumun), "kisi" zaten sesliyle biter.
TIP_EKLI = {"SEHIR": "şehr"}          # ek almadan ONCEKI govde

TIP_AD = {"KISI": "kişi", "SEHIR": "şehir", "BOLGE": "bölge",
          "UNIVERSITE": "üniversite", "FAKULTE": "fakülte",
          "BOLUM": "bölüm", "DERS": "ders", "TEZ": "tez"}

SORU_SOZ = {"KISI": "kim", "SEHIR": "neresi", "BOLGE": "neresi",
            "UNIVERSITE": "hangisi", "FAKULTE": "hangisi",
            "BOLUM": "hangisi", "DERS": "hangisi", "TEZ": "hangisi"}

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
    # !! O KARARIN TAM ARKI: 18 Eylul SABAHI geri alindi ("fakultenin
    # bolumu" 60/60 yanlisti), AKSAMI uc sembol de TUMUYLE silindi
    # (uydurma kavramlardi). Bugunku |R| = 24. Asagidaki nota bak.
    # !! `bolumu` = DERSI ACAN BOLUM (kullanici, 18 Eylul: *"bir ders
    # bir kac bolumun dersi olabilir"*). Dogru: bir dersi BASKA
    # bolumlerin ogrencileri de alir. Ama dersi ACAN/YURUTEN bolum
    # TEKTIR -- ders kodu ona aittir (MAT101 Matematik Bolumu'nundur).
    # Tekil iyelik bu okumayla dogru.
    #
    # !! COK DEGERLI ILISKI BU DILDE YAZILAMAZ, ve bu bir eksiklik
    # DEGIL, SINAVIN SARTI. `Veri.facts` bir (n_ent, n_rel) matrisi:
    # her (varlik, iliski) ciftinin TEK hedefi var. 2-hop sinavi da
    # bunu gerektiriyor -- "X'in dersinin hocasi kimdir?" sorusunun tek
    # dogru cevabi yoksa `comp`/`ent` dogrulugu TANIMSIZ olur.
    # Dolayisiyla "derse kayitli ogrenciler" (cok-cok) ve "dersi veren
    # bolumler" (cok-cok) bu dilin DISINDA kalir; dil, dunyanin
    # TEK DEGERLI diliminden kuruluyor.
    "bolumu":        {"DERS": "BOLUM", "KISI": "BOLUM"},
    "fakultesi":     {"BOLUM": "FAKULTE"},
    # !! SEHIR kolu KALDIRILDI (18 Eylul). Kullanici: *"bazi sehirlerde
    # 5 farkli universite olabilir."*  Oyleyse "sehrin universitesi"
    # TANIMSIZ bir sorudur -- Istanbul'un 5 universitesi varken hangisi?
    # Bir gun onceki duzeltme (rastgele `bagla` -> `amiral`) bu kenari
    # TEKIL yapmisti; dogru cozum tekillestirmek degil, KALDIRMAK.
    # "Universitenin sehri" TERS YONDE duruyor ve o TEKIL: her
    # universite bir sehirdedir.
    "universitesi":  {"FAKULTE": "UNIVERSITE"},
    "sehri":         {"UNIVERSITE": "SEHIR"},
    "bolgesi":       {"SEHIR": "BOLGE"},
    # !! ASAGI YONLU UC KENAR KALDIRILDI (18 Eylul, kullanici:
    # *"ana_fakultesi ne?"*). `ana_fakultesi`, `ana_bolumu` ve
    # `zorunlu_dersi` bir ustun ALTINDAKILERDEN RASTGELE BIRINI secip
    # ona tekil bir ad takiyordu:
    #
    #     olgu[(u, r_asagi)] = g[rng.randint(len(g))]
    #
    # Turkce'de "universitenin ana fakultesi" ya da "fakultenin ana
    # bolumu" diye bir sey YOK. ("Anabilim dali" gercek, ama o BOLUMUN
    # ALTINDA, fakultenin degil.) "Bolumun zorunlu dersi" kavram olarak
    # gercek ama bir bolumun COK zorunlu dersi olur.
    #
    # Yani ucu de `kardes_sehri` / `rakibi` ile AYNI SINIF: keyfi bir
    # secime gercek gorunumlu bir ad. Ayni gun sabah "fakultenin bolumu
    # 60/60 yanlis" diye tespit edilip bu adlar TURETILMISTI -- yani
    # DILBILGISI duzeltilmis, KAVRAM uydurma birakilmisti. (Uc sembol
    # 17 Eylul'de kullanici tarafindan zaten silinmisti; 18 Eylul sabahi
    # geri getirilmeleri hataydi.)
    #
    # !! DOSYANIN KENDI GEREKCESI DE OLCULDU VE CURUDU. Burada
    # "phi'yi buradan aliyoruz" yaziyordu. Olculdu:
    #     ucu de VARKEN   olgu 7641  zincir 47000  AYIRT 23854  phi 6.15
    #     ucu de YOKKEN   olgu 7381  zincir 45221  AYIRT 23854  phi 6.13
    # AYIRT (sinavin yapildigi sinif) HIC DEGISMIYOR. Hiyerarsi artik
    # yalniz YUKARI akiyor.
    # --- GOREV (X -> KISI)
    "rektoru":       {"UNIVERSITE": "KISI"},
    "kurucusu":      {"UNIVERSITE": "KISI"},
    "dekani":        {"FAKULTE": "KISI"},
    # !! BOLGE kolu KALDIRILDI (18 Eylul): Turkiye'de bolgenin baskani
    # YOK, uydurma bir kurumdu. Tip ortusmesi `bolumu` / `universitesi`
    # ile zaten saglaniyor.
    "baskani":       {"BOLUM": "KISI"},
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
    "yasadigi_yer": {"KISI": "SEHIR"},
    # !! TEZ ARTIK AYRI BIR TIP (kullanici karari, 18 Eylul). Onceden
    # `tezi` KISI -> DERS idi ve "Ahmet'in tezi Evrensel Istatistik'tir"
    # diyordu -- bir tez DERS OLAMAZ.
    "tezi":     {"KISI": "TEZ"},
    "yazari":   {"TEZ": "KISI"},
    "konusu":   {"TEZ": "DERS"},
    # --- YATAY
    # !! SEHIR -> SEHIR ILISKISI YOK (kullanici, 18 Eylul: *"kardes
    # sehir, komsu sehir gibi seyleri kaldiralim, gercek dil kapsaminda
    # cok anlamli seyler degil"*).
    #
    # Once "komsusu" vardi, esleme RASTGELEYDI -- "Adana'nin komsusu
    # Tekirdag'dir" dunya hakkinda YANLIS. Gercek komsuluk da
    # kullanilamazdi: bir ilin 3-8 komsusu var, tekil iyelik TEKILLIK
    # ima eder. Sonra gercek ve tekil bir bag olan "kardes sehir" diye
    # yeniden adlandirildi -- ama o da keyfi bir eslemeye gercek bir ad
    # aramaktan ibaretti. Iliskinin KENDISI kaldirildi: uydurma bir bag,
    # dogru adlandirilmis uydurma bir bagdir.
    # !! "rakibi" (UNIVERSITE -> UNIVERSITE) KALDIRILDI -- `kardes_sehri`
    # ile AYNI SINIF: rastgele simetrik esleme, ve "universitenin rakibi"
    # resmi ya da tanimli bir bag degil. Olculdu: AYIRT 23994 -> 23838
    # (%0,7). UNIVERSITE'den cikanlar: ana_fakultesi, kurucusu, rektoru,
    # sehri.
    "onkosulu":      {"DERS": "DERS"},
}
# ===================================================== IMKANSIZ CIFTLER
# (iliski -> o iliskinin KATEGORI HATASI oldugu tipler)
#
# !! "SEMADA YOK" ile "GERCEKTE OLMAZ" AYNI SEY DEGIL. Bu tablo ELLE
# yazildi; SEMA'nin tumleyeni ALINMAZ.
#
# Gerekce OLCULDU (18 Eylul, reddetme verisi ilk kuruldugunda): tumleyen
# alinmisti ve uretilen cumlelerin bir kismi YANLIS cikti --
#     "Bolumun ogrencisi olmaz."   bolumde ogrenci VAR; grafta yalniz
#                                  KISI->bolumu yonu tanimli, tersi degil
#     "Fakultenin sehri olmaz."    fakulte bir sehirde -- universitesi
#                                  uzerinden
#     "Tezin danismani olmaz."     tezin danismani en standart bagdir
# Modele DURUSTLUK ogretirken YALAN ogretmis olurduk.
#
# SECIM KURALI: supheliyse DISARIDA. 167 sema-disi ciftin 73'u burada;
# kalan 94'u "dogru ama modellenmemis" diye ELENDI. (Ilk elemede 79
# kalmisti; uretilen 79 cumle GOZLE okunup uc grup daha cikarildi --
# "kardes universite", "tez juri baskani", "sehrin kuzey bolumu".)
IMKANSIZ = {
    # --- AILE: yalniz canlinin. Kurum/cansiz varlikta kategori hatasi.
    "annesi":   ("UNIVERSITE", "FAKULTE", "BOLUM", "DERS", "SEHIR",
                 "BOLGE", "TEZ"),
    "babasi":   ("UNIVERSITE", "FAKULTE", "BOLUM", "DERS", "SEHIR",
                 "BOLGE", "TEZ"),
    "cocugu":   ("UNIVERSITE", "FAKULTE", "BOLUM", "DERS", "SEHIR",
                 "BOLGE", "TEZ"),
    # !! SEHIR, UNIVERSITE, FAKULTE, BOLUM DISARIDA: "kardes sehir",
    # "kardes universite", "kardes okul" DUNYADA gercek kavramlar
    # (isbirligi baglari). Semadan kaldirildilar -- esleme keyfiydi --
    # ama "olmaz" demek YANLIS olurdu.
    "kardesi":  ("DERS", "BOLGE", "TEZ"),
    "arkadasi": ("UNIVERSITE", "FAKULTE", "BOLUM", "DERS", "SEHIR",
                 "BOLGE", "TEZ"),
    # --- SAHSI: dogmak ve yasamak yalniz canliya ait
    "memleketi":    ("UNIVERSITE", "FAKULTE", "BOLUM", "DERS", "SEHIR",
                     "BOLGE", "TEZ"),
    "yasadigi_yer": ("UNIVERSITE", "FAKULTE", "BOLUM", "DERS", "SEHIR",
                     "BOLGE", "TEZ"),
    # --- GOREV: unvani TASIYAN kurum turu bellidir
    # !! KISI/FAKULTE/BOLUM DISARIDA: "fakultemizin rektoru" gundelik
    # dilde soylenir (bagli olunan universitenin rektoru kastedilir).
    "rektoru":  ("DERS", "SEHIR", "BOLGE", "TEZ"),
    "dekani":   ("DERS", "SEHIR", "BOLGE", "TEZ"),
    # !! BOLGE ICERIDE: Turkiye'de bolgenin valisi YOK -- ayni gerekce
    # `baskani`nin BOLGE kolunu semadan dusurmustu.
    "valisi":   ("UNIVERSITE", "FAKULTE", "BOLUM", "DERS", "BOLGE", "TEZ"),
    # !! SEHIR DISARIDA: belediye BASKANI var.
    # !! TEZ DISARIDA: "tez juri BASKANI" var.
    "baskani":  ("DERS",),
    # !! UNIVERSITE/FAKULTE/BOLUM DISARIDA: hepsinin hocasi VAR.
    # TEZ disarida: "tezin hocasi" = danisman, soylenir.
    "hocasi":   ("SEHIR", "BOLGE"),
    # --- OKUL KAVRAMI cansiza/kisiye
    # !! BOLUM/FAKULTE/UNIVERSITE/TEZ DISARIDA: hepsinin giris kosulu
    # olabilir; "onkosul" dar anlamda DERSE ait ama bu belirsiz.
    "onkosulu": ("KISI", "SEHIR", "BOLGE"),
    # --- YAZARLIK yalniz METNIN
    # !! DERS DISARIDA: ders malzemesinin yazari olur.
    "yazari":   ("KISI", "UNIVERSITE", "FAKULTE", "BOLUM", "SEHIR", "BOLGE"),
    # --- TEZ yazmak yalniz KISININ isi
    # !! BOLUM/FAKULTE/UNIVERSITE/DERS DISARIDA: "bolumun tezleri"
    # (o bolumde yazilanlar) soylenir.
    "tezi":     ("SEHIR", "BOLGE"),
    # !! `bolumu` HIC YOK. Kalan tek aday SEHIR/BOLGE idi, ama
    # "sehrin bolumu" ESIT SESLI: kelimenin "kesim, parca" anlami
    # ("sehrin kuzey bolumu") dogru bir ifadedir.
}

ILISKI = list(SEMA)
TIPLER = ["KISI", "UNIVERSITE", "FAKULTE", "BOLUM", "DERS", "SEHIR",
          "BOLGE", "TEZ"]

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
# `graf_09` bunlarin oranini olcer. Liste BURADA duruyor cunku SEMADAN
# TURETILEMEZ: hangi ciftin geri donmesi BEKLENDIGI anlamsal bir iddia,
# yapisal degil. Araca gomulu kalsaydi (veri_04'te oyleydi) veri
# degisince arac SESSIZCE bos tablo basardi -- fiilen oldu, 17 Eylul:
# graf_09 `kardes` arayip KeyError verdi.
TERS_ADAY = [
    ("annesi", "cocugu"), ("babasi", "cocugu"),
    ("cocugu", "annesi"), ("cocugu", "babasi"),
    ("danismani", "ogrencisi"), ("ogrencisi", "danismani"),
    ("kardesi", "kardesi"), ("arkadasi", "arkadasi"),

    ("tezi", "yazari"), ("yazari", "tezi"),
]
assert all(a in SEMA and b in SEMA for a, b in TERS_ADAY), "TERS_ADAY semada YOK"

# IMKANSIZ, SEMA ile CAKISAMAZ -- cakisirsa grafta VAR OLAN bir olguya
# "olmaz" ogretirdik.
for _r, _ts in IMKANSIZ.items():
    assert _r in SEMA, f"IMKANSIZ'da semada olmayan iliski: {_r}"
    for _t in _ts:
        assert _t in TIPLER, f"IMKANSIZ'da olmayan tip: {_t}"
        assert _t not in SEMA[_r], f"{_t} {_r} SEMADA VAR, imkansiz denemez"
N_IMKANSIZ = sum(len(_v) for _v in IMKANSIZ.values())

# Bir kisinin cocuklari BU iliskilerin kayitlarindan turetilir.
EBEVEYN = ("annesi", "babasi")

# --- OLCEK -------------------------------------------------------------
# Kullanici onayi, 17 Eylul: 480 kisi / 300 ders / 160 bolum / 60 fakulte
# / 40 universite / 7 bolge. SEHIR 18 Eylul'de 40 -> 81 oldu
# (Turkiye'nin butun illeri), TEZ tipi eklendi (480).
BLOK = 48          # bir SOYADI blogunda kac kisi (3 kusak x 16)
KUSAK_EN = 16      # bir kusakta kac kisi (8 erkek + 8 kadin)
N_BLOK = 10        # kac soyadi blogu   -> 480 kisi
N_DERS, N_BOLUM, N_FAKULTE, N_UNI = 300, 160, 60, 40

# UNIVERSITE -> SEHIR: GERCEK YERLESIM.
#
# Kullanici, 18 Eylul: *"bazi sehirlerde 5 farkli universite olabilir."*
# Onceden `bagla` 40 universiteyi sehirlere BIREBIR dagitiyordu (her
# sehirde tam 1). Ilk duzeltme carpik ama RASTGELE bir dagilim verdi ve
# "Hakkari'de 5 universite, Istanbul'da hic" cikti -- yani "Gaziantep'in
# bolgesi Marmara" ile AYNI TURDEN bir hata.
#
# Cozum `SEHIR_BOLGE` ile ayni: uydurma, TABLO. Universiteler gercek
# sehirlerine konuyor.
#
# !! SIZINTI YARATMIYOR -- denetlendi. Ad SIZINTISI JETON duzeyindedir
# (`_denetle` §11: "Alasehir_..._Bolumu" ile "Alasehir_..._Fakultesi"
# `Alasehir` jetonunu PAYLASIR). Buradaki 40 adin hicbiri bir il adi
# degil: "Pamukkale" ile "Denizli" ortak jeton tasimiyor, yani model
# universiteyi sehrine baglamak icin OLGUYU ogrenmek zorunda. Gercege
# uymak burada bedava.
UNI_SEHIR = {
    "Bogazici": "Istanbul", "Mimarsinan": "Istanbul",
    "Galatasaray": "Istanbul", "Yeditepe": "Istanbul",
    "Bahcesehir": "Istanbul", "Beykoz": "Istanbul", "Esenyurt": "Istanbul",
    "Hacettepe": "Ankara", "Gazi": "Ankara", "Atilim": "Ankara",
    "Bilkent": "Ankara", "Baskent": "Ankara", "Cankaya": "Ankara",
    "Ufuk": "Ankara",
    "Zirve": "Gaziantep", "Sanko": "Gaziantep",
    "Pamukkale": "Denizli", "Isikli": "Denizli",
    "Selcuk": "Konya", "Dicle": "Diyarbakir", "Firat": "Elazig",
    "Uludag": "Bursa", "Anadolu": "Eskisehir", "Cukurova": "Adana",
    "Ihlara": "Aksaray", "Kocatepe": "Afyonkarahisar",
    "Harran": "Sanliurfa", "Inonu": "Malatya", "Erciyes": "Kayseri",
    "Sogut": "Bilecik", "Trakya": "Edirne", "Munzur": "Tunceli",
    "Bandirma": "Balikesir", "Toros": "Mersin", "Alanya": "Antalya",
    "Bozok": "Yozgat", "Artuklu": "Mardin", "Recepbey": "Rize",
    "Hitit": "Corum", "Kapadokya": "Nevsehir",
}
assert set(UNI_SEHIR) == {a for a, _ in UNI_AD}, (
    "UNI_SEHIR eksik/fazla -- her universitenin sehri YAZILI olmali")
assert set(UNI_SEHIR.values()) <= {a for a, _ in SEHIR_AD}, (
    "UNI_SEHIR'de il listesinde OLMAYAN sehir var")
assert not {a for a, _ in UNI_AD} & {a for a, _ in SEHIR_AD}, (
    "universite adi bir IL ADI -- jeton paylasir, `sehri` kopyalanabilir olur")


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

    Yalniz YUKARI kenarlar var, COK->BIR (her fakultenin bir
    universitesi). Asagi yonlu kenar YOK -- SEMA'daki nota bak.
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
    # --- UNIVERSITE -> SEHIR: TABLODAN, rastgele DEGIL (UNI_SEHIR).
    # Dagilim tabloya gomulu: Istanbul 7, Ankara 7, Gaziantep 2,
    # Denizli 2, 22 ilde 1'er; kalan 55 il universitesiz. O iller
    # `memleketi`, `yasadigi_yer`, `valisi` ve `bolgesi` ile grafta
    # DURUYOR.
    for _u in uni:
        olgu[(_u, "sehri")] = UNI_SEHIR[_u.split("_")[0]]
    bagla(fak, uni, "universitesi")
    bagla(bol, fak, "fakultesi")
    bagla(ders, bol, "bolumu")

    # --- ASAGI KENAR YOK. Bir zamanlar uc "amiral" kenari vardi
    #     (`ana_fakultesi`, `ana_bolumu`, `zorunlu_dersi`); 18 Eylul'de
    #     SILINDILER -- uydurma kavramlardi ve AYIRT sinifini hic
    #     degistirmiyorlardi (SEMA'daki nota bak).
    # !! `sehir -universitesi-> uni` KENARI DA KALDIRILDI (18 Eylul aksami).
    # Ayni gun once `bagla`dan `amiral`e cevrilmisti (40 sehrin 39'unda
    # universite BASKA SEHIRDEYDI). Sonra kullanici "bazi sehirlerde 5
    # farkli universite olabilir" dedi -- ve tekil bir "sehrin
    # universitesi" TANIMSIZ oldugu icin kenar TUMUYLE KALKTI. Ters
    # yon (`universitenin sehri`) duruyor ve o TEKIL.

    # --- YATAY  (SEHIR -> SEHIR ve UNIVERSITE -> UNIVERSITE YOK,
    #     SEMA'daki notlara bak. Geriye yalniz DERS -> DERS kaldi.)
    # --- ON KOSUL: DONGUSUZ (DAG). 18 Eylul duzeltmesi.
    # Onceden `_devirsiz` bir PERMUTASYON uretiyordu; permutasyon
    # dongulere ayrilir ve olculdu: 300 dersin HEPSI bir dongude
    # (uzunluk 274 + 20 + 6). Yani "A'nin on kosulu B, B'nin ... A".
    # 2-dongu yoktu (o deneniyordu) ama uzun dongu vardi ve "on kosul"
    # bir SIRA DUZENI ima eder.
    # Simdi: dersler karistirilir, her ders KENDINDEN ONCEKILERDEN
    # birini on kosul alir. Ilk ders on kosulsuz kalir (EKSIK sinifi --
    # `zincirler` bunu zaten ayiriyor). Sira RASTGELE, yani ders ADI
    # sirayi SIZDIRMIYOR.
    _sira = list(rng.permutation(len(ders)))
    for j in range(1, len(_sira)):
        olgu[(ders[_sira[j]], "onkosulu")] = ders[_sira[int(rng.randint(j))]]


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

    # !! KURUCU EN YASLI KUSAKTAN (18 Eylul duzeltmesi). Onceden
    # `sec()` butun kusaklardan seciyordu ve olculdu: 40 universitenin
    # 16'sinin kurucusu EN GENC kusaktandi (kusak 2, yani torunlar).
    # Bir kurumu kuran kisi o kurumun bugunku gencinden yasli olmali.
    _k0 = [k for k in kisi if kusak[k] == 0]
    assert _k0, "kusak 0 BOS"
    sec0 = lambda: _k0[int(rng.randint(len(_k0)))]

    for u in ad["UNIVERSITE"]:
        olgu[(u, "rektoru")] = sec()
        olgu[(u, "kurucusu")] = sec0()
    for f in ad["FAKULTE"]:
        olgu[(f, "dekani")] = sec()
    for b in ad["BOLUM"]:
        olgu[(b, "baskani")] = sec()
    for d in ad["DERS"]:
        olgu[(d, "hocasi")] = sec()
    for s in ad["SEHIR"]:
        olgu[(s, "valisi")] = sec()
    # (BOLGE baskani KALDIRILDI -- SEMA'daki nota bak.)

    for k in kisi:
        olgu[(k, "memleketi")] = ad["SEHIR"][int(rng.randint(len(ad["SEHIR"])))]
        olgu[(k, "bolumu")] = ad["BOLUM"][int(rng.randint(len(ad["BOLUM"])))]
    # --- YASADIGI YER: memleketten FARKLI olmak ZORUNDA.
    # Ayni olsaydi iki iliski %100 ortusur, model birini digerinden
    # okurdu (`_farkli_esle`nin kendi gerekcesi). Farkli olmasi ayrica
    # kullanicinin sorusunun karsiligi: "bir kisinin memleketi farkli,
    # yasadigi sehir farkli."
    for k in kisi:
        while True:
            y = ad["SEHIR"][int(rng.randint(len(ad["SEHIR"])))]
            if y != olgu[(k, "memleketi")]:
                olgu[(k, "yasadigi_yer")] = y
                break

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

    # --- arkadasi: simetrik; kardesi/annesi/babasi/cocugu ile AYNI OLMAYAN.
    # !! YASAK LISTESI GENISLEDI (18 Eylul). Onceden yalniz `kardesi`
    # yasakliydi ve olculdu: `arkadasi` 480 kisinin 1'inde `annesi`,
    # 1'inde `babasi`, 1'inde `cocugu` ile CAKISIYORDU. `_farkli_esle`nin
    # kendi gerekcesi "iki iliski ayni eslemeyse model birini digerinden
    # OKUR" -- o gerekce bu ucu de kapsiyor.
    yer = {k: i for i, k in enumerate(kisi)}
    _yasak = []
    for r in ("kardesi", "annesi", "babasi", "cocugu"):
        _yasak.append({i: yer[olgu[(k, r)]] for i, k in enumerate(kisi)
                       if (k, r) in olgu})
    oda = _farkli_esle(rng, range(len(kisi)), _yasak, "arkadasi")
    for i, k in enumerate(kisi):
        olgu[(k, "arkadasi")] = kisi[oda[i]]

    # --- TEZ: ayri bir VARLIK TIPI (kullanici karari, 18 Eylul).
    # Her kisinin bir tezi, her tezin bir yazari (tersi) ve bir konusu
    # (DERS) var. Tez adi AYRI havuzdan (TEZ_ONEK x TEZ_KOK = 24x20=480),
    # yani ne konusunu ne yazarini sizdiriyor.
    tez = [f"{TEZ_ONEK[i % len(TEZ_ONEK)][0]}_"
           f"{TEZ_KOK[(i // len(TEZ_ONEK)) % len(TEZ_KOK)][0]}_Tezi"
           for i in range(len(kisi))]
    assert len(set(tez)) == len(tez) == len(kisi), "TEZ adlari TEKRAR ediyor"
    ad["TEZ"] = tez
    _p = list(rng.permutation(len(kisi)))     # kisi i -> tez _p[i]
    for i, k in enumerate(kisi):
        t = tez[_p[i]]
        olgu[(k, "tezi")] = t
        olgu[(t, "yazari")] = k
        olgu[(t, "konusu")] = ad["DERS"][int(rng.randint(len(ad["DERS"])))]


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
    for grup, r in ((ad["KISI"], "kardesi"), (ad["KISI"], "arkadasi")):
        bozuk = [x for x in grup if olgu[(olgu[(x, r)], r)] != x]
        assert not bozuk, f"{r} simetrik degil: {bozuk[:3]}"
    # 9) TERS CIFT: danismani / ogrencisi
    bozuk = [k for k in ad["KISI"]
             if olgu[(olgu[(k, "danismani")], "ogrencisi")] != k]
    assert not bozuk, f"danismani/ogrencisi ters degil: {bozuk[:3]}"
    # 10) (BOSALDI) -- AMIRAL kenar denetimi. Denetledigi uc kenar
    #     18 Eylul'de SILINDI (SEMA'daki nota bak); hiyerarsi artik
    #     yalniz YUKARI akiyor, dolayisiyla "geri geliyor mu" diye
    #     sorulacak asagi kenar KALMADI.
    # 11) NOTRLUK: FAKULTE/BOLUM adinin ilk yuvasi ustundekini SIZDIRMASIN
    for a in ad["FAKULTE"]:
        u = olgu[(a, "universitesi")]
        assert a.split("_")[0] != u.split("_")[0], f"fakulte adi sizdiriyor: {a}"
    for a in ad["BOLUM"]:
        f = olgu[(a, "fakultesi")]
        assert a.split("_")[0] != f.split("_")[0], f"bolum adi sizdiriyor: {a}"
    # 12) KISININ UC SEHRI AYRISIYOR -- memleketi != yasadigi yer
    _c = [k for k in ad["KISI"]
          if olgu[(k, "memleketi")] == olgu[(k, "yasadigi_yer")]]
    assert not _c, f"memleketi == yasadigi_yer: {_c[:3]}"
    # 13) ARKADASI baska hicbir KISI->KISI iliskisiyle CAKISMIYOR
    for r in ("kardesi", "annesi", "babasi", "cocugu"):
        _c = [k for k in ad["KISI"] if olgu.get((k, r)) == olgu[(k, "arkadasi")]]
        assert not _c, f"arkadasi == {r}: {_c[:3]}"
    # 14) ON KOSUL DONGUSUZ -- her ders zincirin SONUNA varmali
    for d in ad["DERS"]:
        _g, c = set(), d
        while c is not None:
            assert c not in _g, f"on kosul DONGUSU: {d} -> ... -> {c}"
            _g.add(c)
            c = olgu.get((c, "onkosulu"))
    # 15a) UNIVERSITE DAGILIMI: hicbir sehirde 5'ten fazla olmasin, ve
    #      universite adi KENDI SEHRINI sizdirmasin.
    import collections as _c2
    _us = _c2.Counter(olgu[(u, "sehri")] for u in ad["UNIVERSITE"])
    assert max(_us.values()) >= 5, (
        f"kullanici: bazi sehirlerde 5 farkli universite olabilir -- "
        f"en cok {_us.most_common(1)}")
    assert len(set(_us.values())) >= 3, (
        f"dagilim CARPIK olmali, duz degil: {sorted(_us.values())}")
    for u in ad["UNIVERSITE"]:
        assert u.split("_")[0] != olgu[(u, "sehri")], (
            f"universite adi SEHRINI sizdiriyor: {u}")
    # 15) KURUCU en yasli kusaktan
    _c = [u for u in ad["UNIVERSITE"] if kusak[olgu[(u, "kurucusu")]] != 0]
    assert not _c, f"kurucusu kusak 0 DEGIL: {_c[:3]}"
    # 16) TEZ: birebir, ve adi ne yazarini ne konusunu SIZDIRIYOR
    _yz = [olgu[(t, "yazari")] for t in ad["TEZ"]]
    assert len(set(_yz)) == len(ad["TEZ"]) == len(ad["KISI"]), "tez birebir DEGIL"
    for k in ad["KISI"]:
        assert olgu[(olgu[(k, "tezi")], "yazari")] == k, f"{k} tezi/yazari ters DEGIL"
    for t in ad["TEZ"]:
        _tp = set(t.split("_"))
        assert not (_tp & set(olgu[(t, "yazari")].split("_"))), \
            f"tez adi YAZARINI sizdiriyor: {t}"
        assert not (_tp & set(olgu[(t, "konusu")].split("_"))), \
            f"tez adi KONUSUNU sizdiriyor: {t}"


def kur(tohum=0):
    """model_09'in grafi."""
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
            f"veri_16: graf DEGISTI  {IZ} -> {_iz}\n"
            "  Kasitliysa IZ yenilenir ve onkayda not duselir; degilse\n"
            "  degisiklik geri alinir. model_09 sessizce baska bir\n"
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
# !! 18 Eylul'de YENILENDI. Graf Turkce ve mantik duzeltmeleriyle
# degisti (SEMA notlarina bak): sehrin universitesi amiral kenar oldu,
# ana_fakultesi / ana_bolumu / zorunlu_dersi ayrildi, yasadigi_yer ve
# TEZ tipi eklendi, bolgenin baskani kaldirildi, on kosul dongusuz
# kuruldu, kurucu en yasli kusaktan secildi.
# ESKI: 3431c633b6b1  (model_09 / model_06 / model_07)
IZ = "3cd9a2575e47"


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
    for r in ("universitesi", "fakultesi", "bolumu", "tezi",
              "sehri", "bolgesi", "yasadigi_yer", "annesi", "babasi", "kardesi",
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
