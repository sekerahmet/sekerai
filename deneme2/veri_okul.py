# -*- coding: utf-8 -*-
"""VERI_OKUL — okul/aile dunyasi. Kullanicinin onerdigi bicim:

    Ahmet kardes Ayse       Ayse  kardes Ahmet       (SIMETRIK)
    Ahmet baba   Galip      Galip cocuk  Ahmet       (TERS CIFT)
    Ahmet ogretmen Elif     Elif  ogrenci Ahmet      (TERS CIFT)
    Ahmet okul   Bilecik_Anadolu_Lisesi

NEDEN BU ALAN (olculdu, `sema_hesap.py`):
  phi tavani = kopru tipinin ORTALAMA CIKIS DERECESI. Alan adi phi'yi
  degistirmez, TIP BASINA ILISKI SAYISI degistirir. Bir insanin dogal
  olarak 10 iliskisi vardir, bir bolgenin 3. Cografya semasi phi 3.01'de
  tavan yapiyordu; burada 7.04 (dosyayi kosunca basilir).
  Ama tek bir tipe (KISI) her seyi yiginca kisayol HER ZAMAN mumkun olur
  ve gomulu kontrol %59'dan %14'e duser. Bu yuzden OKUL / SEHIR / DERS de
  zenginlestirildi ve her iliski DAR kaynakli tutuldu.

UC TASARIM KURALI (her biri olculmus bir arizadan cikti):

  1. TEK DEGERLI. "Mehmet kardes Ahmet, Veli" olmaz. `facts[e,r]` tek hedef
     tutuyor, dogruluk `argmax == hedef`. Cok degerli olursa cevap bir KUME
     olur ve acc/shortcut tanimsizlasir.

  2. ANLAMCA BIRBIRINI GEREKTIREN iliski cifti ZINCIRI OLCULEMEZ YAPAR.
     "kardesinin babasi" = "babasi". Cografya verisinde ayni sey `dili` ile
     olmustu ve zincirlerin %39'unu olculemez hale getirmisti. Burada
     tutarliligi BOZMUYORUZ -- o zincirleri AYNI diye sinavdan cikariyoruz,
     maliyeti sayilir (asagida basilir).

  3. TERS CIFT (baba/cocuk, ogretmen/ogrenci) DONUS uretir: cevap sorulan
     varligin KENDISI. Tek degerli oldugu icin hepsi degil -- OLCULDU:
        baba cocuk      350/700 DONUS, 350 GERCEK kompozisyon (kardesi cikar)
        ogretmen ogrenci 700/700 DONUS (tam esleme, istisnasiz)
     Toplam disarida kalan: AYNI 1.566 + DONUS 2.810 = zincirlerin %8'i.
     (Cografya verisinde bu %29 idi.)

SOYADI KALITIMI: cocuk / kardes / anne / baba AYNI soyadi tasir; bir soyadi
blogu 50 kardes cifti = 100 kisi. Ebeveyn haritasi bu yuzden BLOK ICINDE
kaydirilir -- global kaydirma butun cifleri tek halkaya baglayip kalitimi
imkansiz kiliyordu. Assert ile denetleniyor.

GERCEK olan : kisi adlari, 80 il, ders adlari, okul adlari
URETILMIS   : kim kimin akrabasi/ogretmeni/arkadasi -- HEPSI RASTGELE.
              14 Eylul'e kadar ARITMETIK KAYDIRMA idi (kardes i<->i+1,
              arkadas i<->i+350, ogretmen i->i+211). Tutarliydi, 23
              kontrolun hepsi geciyordu, ama gorevi COZULMUS kiliyordu:
              iki kaydirmanin bileskesi yine kaydirmadir. Kosu iptal
              edildi, uretec rastgeleye cevrildi, veri_kontrol.py'ye iki
              kapi eklendi (24: iliski, 25: zincir).

DOCSTRING'DEKI SAYILAR ELLE YAZILI -> BAYATLAR. veri_gercek.py'de tam olarak
bu oldu (sema daralinca 5.700 yazili kaldi, gercek 4.140 idi) ve ancak
GitHub'da goze carpinca yakalandi. Gecerli sayi her zaman dosyanin CIKTISI.

    python veri_okul.py [--dok cikti.txt]      <- sayilar burada
    python veri_kontrol.py                     <- 23 tutarlilik kontrolu
"""
import argparse
import io
import numpy as np

# ===================================================================== ADLAR
ERKEK = ["Ahmet", "Mehmet", "Mustafa", "Ali", "Huseyin", "Hasan", "Ibrahim",
         "Osman", "Yusuf", "Murat", "Omer", "Ramazan", "Suleyman", "Halil",
         "Ismail", "Riza", "Fatih", "Kemal", "Salih", "Emre", "Serkan",
         "Burak", "Cem", "Deniz", "Ege", "Furkan", "Gokhan", "Hakan",
         "Ilker", "Kaan", "Levent", "Metin", "Nihat", "Onur", "Polat",
         "Rafet", "Sinan", "Tolga", "Ufuk", "Volkan", "Yigit", "Zeki",
         "Baris", "Caner", "Dogan", "Erdem", "Ferhat", "Galip", "Haluk",
         "Ismet"]

KADIN = ["Ayse", "Fatma", "Emine", "Hatice", "Zeynep", "Elif", "Meryem",
         "Sultan", "Zehra", "Hulya", "Melek", "Ozlem", "Yasemin", "Nurten",
         "Sevim", "Gulay", "Filiz", "Derya", "Ebru", "Funda", "Gamze",
         "Hande", "Irem", "Jale", "Kubra", "Leyla", "Merve", "Nazli", "Oya",
         "Pinar", "Rabia", "Selin", "Tugce", "Ulku", "Vildan", "Yagmur",
         "Zuhal", "Asli", "Berna", "Ceren", "Damla", "Esra", "Feride",
         "Gonca", "Hilal", "Ipek", "Kader", "Lale", "Mine", "Nehir"]

SOYAD = ["Yilmaz", "Kaya", "Demir", "Sahin", "Celik", "Yildiz", "Aydin"]

IL = ["Adana", "Adiyaman", "Afyon", "Agri", "Amasya", "Ankara", "Antalya",
      "Artvin", "Aydin", "Balikesir", "Bilecik", "Bingol", "Bitlis", "Bolu",
      "Burdur", "Bursa", "Canakkale", "Cankiri", "Corum", "Denizli",
      "Diyarbakir", "Edirne", "Elazig", "Erzincan", "Erzurum", "Eskisehir",
      "Gaziantep", "Giresun", "Gumushane", "Hakkari", "Hatay", "Isparta",
      "Mersin", "Istanbul", "Izmir", "Kars", "Kastamonu", "Kayseri",
      "Kirklareli", "Kirsehir", "Kocaeli", "Konya", "Kutahya", "Malatya",
      "Manisa", "Maras", "Mardin", "Mugla", "Mus", "Nevsehir", "Nigde",
      "Ordu", "Rize", "Sakarya", "Samsun", "Siirt", "Sinop", "Sivas",
      "Tekirdag", "Tokat", "Trabzon", "Tunceli", "Urfa", "Usak", "Van",
      "Yozgat", "Zonguldak", "Aksaray", "Bayburt", "Karaman", "Kirikkale",
      "Batman", "Sirnak", "Bartin", "Ardahan", "Igdir", "Yalova", "Karabuk",
      "Kilis", "Osmaniye"]
# 80 il: komsuluk SIMETRIK eslesme ile kuruluyor, tek sayi olsaydi bir il
# eslesmeden kalir ve o ilin komsulugu tek yonlu olurdu.

DERS = ["Matematik", "Geometri", "Analiz", "Cebir", "Istatistik", "Olasilik",
        "Fizik", "Mekanik", "Optik", "Termodinamik", "Elektrik", "Manyetizma",
        "Kimya", "Organik_Kimya", "Anorganik_Kimya", "Biyokimya",
        "Biyoloji", "Genetik", "Botanik", "Zooloji", "Ekoloji", "Anatomi",
        "Tarih", "Cografya", "Jeoloji", "Arkeoloji", "Antropoloji",
        "Edebiyat", "Dilbilgisi", "Kompozisyon", "Siir", "Roman",
        "Felsefe", "Mantik", "Sosyoloji", "Psikoloji", "Ekonomi", "Hukuk",
        "Muzik", "Resim", "Heykel", "Tiyatro", "Sinema", "Fotograf",
        "Beden_Egitimi", "Yuzme", "Atletizm", "Basketbol", "Futbol",
        "Ingilizce", "Almanca", "Fransizca", "Arapca", "Rusca", "Ispanyolca",
        "Bilgisayar", "Algoritma", "Veritabani", "Ag_Sistemleri", "Robotik",
        "Muhendislik", "Mimarlik", "Tasarim", "Elektronik", "Makine",
        "Tip", "Eczacilik", "Hemsirelik", "Veterinerlik", "Dishekimligi",
        "Tarim", "Bahcecilik", "Ormancilik", "Balikcilik", "Hayvancilik",
        "Muhasebe", "Isletme", "Pazarlama", "Lojistik", "Turizm"]

OKUL_TUR = ["Lisesi", "Fen_Lisesi", "Anadolu_Lisesi"]

# ===================================================================== SEMA
# iliski -> {kaynak tipi: hedef tipi}.  TIP ORTUSMESI KASITLI:
# `sehir` ve `ders` uc ayri tipten cikiyor -> kisayolun MUMKUN oldugu
# zincirler olusuyor. Ortusme olmazsa gomulu kontrol %100 olur ve AYIRT
# sinifi (sinavin yapildigi yer) kalmaz.
SEMA = {
    "kardes":   {"KISI": "KISI"},                 # simetrik
    "baba":     {"KISI": "KISI"},
    "anne":     {"KISI": "KISI"},
    "cocuk":    {"KISI": "KISI"},                 # baba/anne TERSI
    "ogretmen": {"KISI": "KISI"},
    "ogrenci":  {"KISI": "KISI"},                 # ogretmen'in TERSI
    "arkadas":  {"KISI": "KISI"},                 # simetrik
    "okul":     {"KISI": "OKUL", "SEHIR": "OKUL"},
    "sehir":    {"KISI": "SEHIR", "OKUL": "SEHIR"},
    "ders":     {"KISI": "DERS", "OKUL": "DERS"},
    "mudur":    {"OKUL": "KISI"},
    "kurucu":   {"OKUL": "KISI"},
    "rakip":    {"OKUL": "OKUL"},
    "komsu":    {"SEHIR": "SEHIR"},
    "vali":     {"SEHIR": "KISI"},
    "hoca":     {"DERS": "KISI"},
    "onkosul":  {"DERS": "DERS"},
}
ILISKI = list(SEMA)
TIPLER = ["KISI", "OKUL", "SEHIR", "DERS"]

# Anlamca BIRBIRINI GEREKTIREN cifler: (r1, r2) -> r_esdeger
# "kardesinin babasi" = "babasi".  Bunlar AYNI sinifina duser; burada
# listelenmesinin sebebi belgede SAYILABILMESI.
GEREKTIRIR = {("kardes", "baba"): "baba", ("kardes", "anne"): "anne"}
BLOK = 100         # bir SOYADI blogunda kac kisi (50 erkek + 50 kadin)

# 14 EYLUL -- SABIT KAYDIRMALAR KALDIRILDI.
# Ilk surumde kardes i<->i+1, arkadas i<->i+350, ogretmen i->i+211 idi.
# Tutarliydi ve 23 kontrolun hepsi geciyordu, ama GOREVI COZULMUS KILIYORDU:
# iki kaydirmanin BILESKESI yine bir kaydirmadir, yani "kardes ogretmen"
# = "+212" demek. Model bunu 848 egitim varligindan KURAL olarak ogrenip
# hic gormedigi varliga uyguluyordu -- kopruyu kullanmadan.
#   OLCULDU (kosu iptal, belge/bulgu/egri_G_iptal_20260914.json):
#     ENT-AYIRT  r1,r2 IKISI DE kaydirma %58  ->  ent 0.882
#     ENT-YOK    r1,r2 IKISI DE kaydirma  %0  ->  ent 0.070
#   D3.3'un rastgele grafinda ayni olcu 0.009 idi. Yani ENT-YOK dogru
#   zorlugu olcuyordu, ENT-AYIRT aritmetikle cozulmustu.
# Simdi HER esleme RASTGELE. Yapi kisitlari (kardesler ayni ebeveyn, ters
# ciftler, soyadi kalitimi, simetri) DURUYOR -- onlar aritmetik degil.


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
    for _ in range(1000):
        p = list(rng.permutation(idx))
        if all(x != y for x, y in zip(idx, p)):
            return dict(zip(idx, p)), dict(zip(p, idx))
    raise RuntimeError("devirsiz esleme bulunamadi")


N_KISI, N_OKUL = 700, 200


def kur(tohum=0):
    rng = np.random.RandomState(tohum)

    # cift indeks ERKEK, tek indeks KADIN; her 100 kisi bir SOYADI blogu.
    kisi = [f"{(ERKEK if i % 2 == 0 else KADIN)[(i // 2) % 50]}"
            f"_{SOYAD[(i // 2) // 50]}" for i in range(N_KISI)]
    assert len(set(kisi)) == N_KISI, "kisi adi tekrari"
    sehir, ders = list(IL), list(DERS)
    okul = [f"{sehir[i % len(sehir)]}_{OKUL_TUR[(i // len(sehir)) % 3]}"
            for i in range(N_OKUL)]
    assert len(set(okul)) == N_OKUL, "okul adi tekrari"

    ad = {"KISI": kisi, "OKUL": okul, "SEHIR": sehir, "DERS": ders}
    hepsi = [a for t in TIPLER for a in ad[t]]
    assert len(hepsi) == len(set(hepsi)),         "TEKRAR EDEN AD: " + str([a for a in set(hepsi) if hepsi.count(a) > 1])

    olgu = {}
    n = N_KISI

    # --- AILE: her SOYADI blogu KENDI ICINDE, hepsi RASTGELE --------------
    # Kisitlar (aritmetik degil, YAPISAL):
    #   kardesler ayni anne-babayi paylasir
    #   baba erkek, anne kadin, ikisi de AYNI soyadi blogundan
    #   kimse kendi cocugunun ebeveyni degil; anne ile baba KARDES degil
    #   her erkek TAM 1 ciftin babasi, her kadin TAM 1 ciftin annesi
    #     -> `cocuk` (baba/anne tersi) her kiside TANIMLI
    for b0 in range(n // BLOK):
        blok = list(range(b0 * BLOK, (b0 + 1) * BLOK))
        erk = [i for i in blok if i % 2 == 0]
        kad = [i for i in blok if i % 2 == 1]
        for deneme in range(2000):
            kp = list(rng.permutation(blok))
            cift = [(kp[2 * k], kp[2 * k + 1]) for k in range(len(blok) // 2)]
            uye = {x: j for j, c in enumerate(cift) for x in c}
            ba = list(rng.permutation(erk))      # cift j'nin babasi ba[j]
            an = list(rng.permutation(kad))      # cift j'nin annesi an[j]
            if not all(uye[ba[j]] != j and uye[an[j]] != j       # kendi cocugu degil
                       and uye[ba[j]] != uye[an[j]]             # es ile kardes degil
                       for j in range(len(cift))):
                continue
            # RASTGELE esleme KISA DONGU uretebiliyor; kaydirmali surumde
            # bunlar yapisal olarak imkansizdi, simdi ACIKCA elenmeli.
            # Olculdu: ilk denemede 10 kisinin babasi ayni zamanda cocugu,
            # 6 kisinin dedesi kendisiydi.
            _ba = {x: ba[uye[x]] for x in blok}          # x -> babasi
            _an = {x: an[uye[x]] for x in blok}          # x -> annesi
            _co = {}
            for j2, (x2, y2) in enumerate(cift):
                _co[ba[j2]] = x2
                _co[an[j2]] = x2
            if all(_ba[x] != _co[x] and _an[x] != _co[x]         # baba/anne = cocuk
                   and _ba[_ba[x]] != x and _an[_an[x]] != x     # dede/nine = kendisi
                   and _co[_co[x]] != x                          # torun = kendisi
                   for x in blok):
                break
        else:
            raise RuntimeError(f"blok {b0}: aile kurulamadi")
        for j, (x, y) in enumerate(cift):
            olgu[(kisi[x], "kardes")] = kisi[y]
            olgu[(kisi[y], "kardes")] = kisi[x]
            for c in (x, y):
                olgu[(kisi[c], "baba")] = kisi[ba[j]]
                olgu[(kisi[c], "anne")] = kisi[an[j]]
            olgu[(kisi[ba[j]], "cocuk")] = kisi[x]     # kayitli cocuk: ciftin ILKI
            olgu[(kisi[an[j]], "cocuk")] = kisi[x]
    assert all((c, "cocuk") in olgu for c in kisi), "cocuk eksik"

    # --- ogretmen / ogrenci: RASTGELE devirsiz esleme, tersi KESIN --------
    ogr, ters = _devirsiz(rng, range(n))
    for i, c in enumerate(kisi):
        olgu[(c, "ogretmen")] = kisi[ogr[i]]
        olgu[(c, "ogrenci")] = kisi[ters[i]]

    # --- arkadas: RASTGELE simetrik esleme, kardesten FARKLI --------------
    for _ in range(200):
        ark = _esle(rng, range(n))
        if all(kisi[ark[i]] != olgu[(kisi[i], "kardes")] for i in range(n)):
            break
    else:
        raise RuntimeError("arkadas eslemesi kurulamadi")
    for i, c in enumerate(kisi):
        olgu[(c, "arkadas")] = kisi[ark[i]]

    # --- okul / sehir / ders ---------------------------------------------
    # Kisinin sehri, OKULUNUN sehrinden BAGIMSIZ. Bagimli olsaydi
    # "X okul sehir" = "X sehir" olurdu ve zincir olculemezdi.
    for i, c in enumerate(kisi):
        olgu[(c, "okul")] = okul[int(rng.randint(N_OKUL))]
        olgu[(c, "sehir")] = sehir[int(rng.randint(len(sehir)))]
        olgu[(c, "ders")] = ders[int(rng.randint(len(ders)))]
    rak = _esle(rng, range(N_OKUL))
    for i, o in enumerate(okul):
        olgu[(o, "mudur")] = kisi[int(rng.randint(n))]
        olgu[(o, "kurucu")] = kisi[int(rng.randint(n))]
        olgu[(o, "sehir")] = sehir[i % len(sehir)]      # okul ADINDAKI il
        olgu[(o, "rakip")] = okul[rak[i]]
        olgu[(o, "ders")] = ders[int(rng.randint(len(ders)))]
    kom = _esle(rng, range(len(sehir)))
    for i, s_ in enumerate(sehir):
        olgu[(s_, "komsu")] = sehir[kom[i]]
        olgu[(s_, "vali")] = kisi[int(rng.randint(n))]
        olgu[(s_, "okul")] = okul[int(rng.randint(N_OKUL))]
    onk, _t = _devirsiz(rng, range(len(ders)))
    for i, d in enumerate(ders):
        olgu[(d, "hoca")] = kisi[int(rng.randint(n))]
        olgu[(d, "onkosul")] = ders[onk[i]]

    # --- DENETIMLER -------------------------------------------------------
    _soy = lambda a: a.rsplit("_", 1)[1]
    for c in kisi:
        for r in ("kardes", "baba", "anne", "cocuk"):
            assert _soy(olgu[(c, r)]) == _soy(c),                 f"soyadi kalitimi bozuk: {c} {r} {olgu[(c, r)]}"
        assert olgu[(olgu[(c, "baba")], "kardes")] != olgu[(c, "anne")],             f"anne-baba kardes cikti: {c}"
    for grup, r in ((sehir, "komsu"), (okul, "rakip"),
                    (kisi, "kardes"), (kisi, "arkadas")):
        bozuk = [x for x in grup if olgu[(olgu[(x, r)], r)] != x]
        assert not bozuk, f"{r} simetrik degil: {bozuk[:3]}"
    kendi = [(e, r) for (e, r), h in olgu.items() if h == e]
    assert not kendi, f"kendine giden olgu: {kendi[:5]}"

    ozel = ["<pad>", "<soru>", "?", "<son>"]
    sozluk = ozel + ILISKI + hepsi
    return dict(ad=ad, n={t: len(ad[t]) for t in TIPLER}, sozluk=sozluk,
                kim={s: i for i, s in enumerate(sozluk)},
                tip={a: t for t in TIPLER for a in ad[t]},
                olgu=olgu, sema=SEMA, iliski=ILISKI)



def zincirler(G):
    """Tip olarak gecerli 2 adimli zincirler, DORT SINIFA ayrilmis.
    Siniflarin tanimi `veri_gercek.zincirler` ile AYNI -- iki veri kumesi
    ayni olcum hattindan gecsin diye."""
    out = []
    for (e, r1), b in G["olgu"].items():
        for r2 in G["iliski"]:
            if r2 == r1:
                continue
            if G["tip"][b] not in G["sema"][r2]:
                continue
            cev = G["olgu"][(b, r2)]
            ks = G["olgu"].get((e, r2))
            sinif = ("DONUS" if cev == e else
                     "YOK" if ks is None else
                     "AYNI" if ks == cev else "AYIRT")
            out.append((e, r1, r2, b, cev, ks, sinif))
    return out


def yaz(G, z, f=print):
    S = {k: [x for x in z if x[6] == k]
         for k in ("AYIRT", "YOK", "AYNI", "DONUS")}
    f("=" * 76)
    f("OKUL / AILE DUNYASI")
    f("=" * 76)
    f("  tip buyuklukleri : " + "  ".join(f"{k}={v}" for k, v in G["n"].items()))
    f(f"  SOZLUK           : {len(G['sozluk'])} token  "
      f"({len(G['iliski'])} iliski + {sum(G['n'].values())} varlik + 4 ozel)")
    f(f"  ATOMIK OLGU      : {len(G['olgu'])}")
    f(f"  2-ADIMLI ZINCIR  : {len(z)}")
    f(f"  phi TAVANI       : {len(z)/len(G['olgu']):.2f}")
    for k, ac in (("AYIRT", "kisayol VAR, cevaptan FARKLI -> SINAV burada"),
                  ("YOK",   "kisayol IMKANSIZ           -> gomulu kontrol"),
                  ("AYNI",  "kisayol = cevap            -> sinavda KULLANILMAZ"),
                  ("DONUS", "cevap = sorulan varlik     -> sinavda KULLANILMAZ")):
        f(f"     {k:6s} {len(S[k]):6d}  ({100*len(S[k])/len(z):2.0f}%)   {ac}")

    f("\n--- SEMA ---")
    for r, m in G["sema"].items():
        f(f"   {r:10s} " + ",  ".join(f"{k}->{v}" for k, v in m.items()))

    f("\n--- BIR AILE ---")
    for c in G["ad"]["KISI"][:4]:
        for r in ("kardes", "baba", "anne", "cocuk", "ogretmen", "ogrenci",
                  "arkadas", "okul", "sehir", "ders"):
            f(f"   {c:18s} {r:9s} {G['olgu'][(c, r)]}")
        f("")

    f("--- ANLAMCA GEREKTIREN CIFTLER (AYNI olmak ZORUNDA) ---")
    for (r1, r2), esd in GEREKTIRIR.items():
        ad_ = [x for x in z if x[1] == r1 and x[2] == r2]
        kac = sum(1 for x in ad_ if x[6] == "AYNI")
        f(f"   '{r1} {r2}' = '{esd}'   {kac}/{len(ad_)} zincir AYNI cikti")

    f("\n--- TERS CIFTLER (DONUS uretir, ama hepsi degil) ---")
    for r1, r2 in (("baba", "cocuk"), ("anne", "cocuk"), ("cocuk", "baba"),
                   ("ogretmen", "ogrenci"), ("ogrenci", "ogretmen")):
        ad_ = [x for x in z if x[1] == r1 and x[2] == r2]
        dn = sum(1 for x in ad_ if x[6] == "DONUS")
        f(f"   {r1:9s} {r2:9s}  {dn}/{len(ad_)} DONUS, "
          f"{len(ad_)-dn} gercek kompozisyon")

    f("\n--- ZINCIR TURLERI ---")
    tur = {}
    for e, r1, r2, b, a, ks, s in z:
        d = tur.setdefault((r1, r2),
                           {"AYIRT": 0, "YOK": 0, "AYNI": 0, "DONUS": 0})
        d[s] += 1
    f(f"   {'r1':10s} {'r2':10s} {'AYIRT':>7s} {'YOK':>7s} {'AYNI':>7s}"
      f" {'DONUS':>7s}")
    for (r1, r2), d in sorted(tur.items()):
        f(f"   {r1:10s} {r2:10s} {d['AYIRT']:7d} {d['YOK']:7d} {d['AYNI']:7d}"
          f" {d['DONUS']:7d}")

    ornek = {"AYIRT": "model kisayola saparsa YAKALARIZ",
             "YOK": "kisayol imkansiz (tip izin vermiyor)",
             "AYNI": "kisayol dogru cevabi veriyor -- sinavda kullanilamaz",
             "DONUS": "cevap sorulan varligin kendisi -- kopyalamak yetiyor"}
    for k in ("AYIRT", "YOK", "AYNI", "DONUS"):
        f(f"\n--- ZINCIR / {k}  ({ornek[k]}) ---")
        for e, r1, r2, b, a, ks, _s in S[k][:8]:
            f(f"   {e} {r1} {r2} ?   ->  {a}")
            f(f"       kopru: {b:20s} |  KISAYOL '{e} {r2}' = {ks or 'YOK'}")

    f("\n--- TOKEN DIZILIMI (her varlik TEK token) ---")
    for e, r1, r2, b, a, ks, _s in S["AYIRT"][:3]:
        d = ["<soru>", e, r1, r2, "?", a, "<son>"]
        f(f"   {' '.join(d)}")
        f(f"       -> {[G['kim'][t] for t in d]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dok", default="")
    a = ap.parse_args()
    G = kur()
    z = zincirler(G)
    yaz(G, z)
    if a.dok:
        with io.open(a.dok, "w", encoding="utf-8", newline="") as fh:
            def f(s=""):
                fh.write(str(s) + "\n")
            yaz(G, z, f)
            f("\n\n" + "=" * 76)
            f("TUM ATOMIK OLGULAR")
            f("=" * 76)
            for (e, r), h in G["olgu"].items():
                f(f"{e}\t{r}\t{h}")
            f("\n\n" + "=" * 76)
            f("TUM 2-ADIMLI ZINCIRLER (sinif / soru / cevap / kopru / kisayol)")
            f("=" * 76)
            for e, r1, r2, b, ans, ks, s in z:
                f(f"{s:5s}\t{e} {r1} {r2} ?\t-> {ans}\tkopru={b}"
                  f"\tkisayol={ks or 'YOK'}")
        print(f"\n-> tam dokum: {a.dok}")


if __name__ == "__main__":
    main()
