# -*- coding: utf-8 -*-
"""veri_04 — model_04'in KENDI verisi. TEK BASINA DURUR.

Kullanici karari, 16 Eylul 2026:
    *"bunlarin hepsi model_04 folderi altinda olmali. model_04 diger
    hicbir model ile ayni seyi kullanmamali."*

Bu dosya `veri_okul*` modullerinin HICBIRINI import ETMEZ. Govdesi
onlardan KOPYALANDI (uretici: scratchpad/kur_veri00.py) ve artik
model_04'a aittir: `model_b` ailesi icin yapilan bir degisiklik buraya
GECMEZ.

==========================================================================
ICERIK -- bugun `veri_okul4` ile AYNI, ve bu KASITLI

Bagimsizlik MIMARI icin gecerli, veri icin DEGIL: ayni veriyi gormezse
kol hicbir sey olcmez (sinav bolmeleri ayrisir, `olcme_izi` degisir,
sayilar ayni tabloda okunamaz).

    veri_okul  (1x taban graf)  +  veri_okul3'un SEKIZ ALAN ACILIMI
    1060 varlik   10.960 olgu   104.940 zincir   |R| 17
    phi TAVANI 9,57

Kopyanin DOGRULUGU goz karariyla degil, OLCUMLE tutuluyor:

    IZ            grafin parmak izi, `kur()` her cagrilista denetler
    test_04.py    veri_04.kur(0) ile veri_okul4.kur(0) DERIN
                  KARSILASTIRMA ile birebir ayni mi -- olgu sozlugu,
                  varlik listeleri, sema, sozluk, zincirler

Yani `veri_okul4` bir gun degisirse bu dosya DEGISMEZ (kopya), ama
kilit farki GORUR ve soyler: o gun ya fark bilerek kabul edilir
(onkayda not duselir) ya da kopya guncellenir.

==========================================================================
YAPI

    _taban_kur(tohum)   veri_okul.kur -- 1x graf
    genislet(G, tohum)  veri_okul3.genislet -- sekiz alan acilimi
    kur(tohum)          ikisi arka arkaya + IZ denetimi   <- KULLANILAN

IKI SEMA var, orijinaldeki gibi: `SEMA_TABAN` (taban grafin semasi) ve
`SEMA` (sekiz alan acilimiyla genisletilmis). `_taban_kur` birincisini,
`genislet` ikincisini `G["sema"]`ya koyar. Ilk birlestirmede tek sozluk
vardi ve taban graf kendi basina TUTARSIZ kaliyordu -- ayrinti asagida,
SEMA tanimlarinin yaninda.
"""
from __future__ import annotations

import hashlib

import numpy as np

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
SEMA_TABAN = {
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
ILISKI = list(SEMA_TABAN)
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


def _taban_kur(tohum=0, olcek=None):
    """`olcek=None` -> BUGUNKU graf, BIREBIR. Baska bir sey verilirse
    havuzlar buyur/kucultulur ve graf olceklenir.

    ```
    kur(0)                                  # 1060 varlik,  8400 olgu
    kur(0, veri_okul2.OLCEK2)               # 2120 varlik, 16800 olgu
    ```

    Havuzlar MODUL GLOBALI olarak duruyor; burada YEREL isme baglaniyor.
    `globals()` ile okunmasinin sebebi teknik: ayni ismi yerel olarak
    atayinca Python butun govdeyi yerel sayar, yani asagidaki 130 satirin
    HICBIRINE dokunmadan olcek degistirilebiliyor. Alternatifi 9 ayri
    yerde arama-degistirme yapmakti; biri atlanirsa graf SESSIZCE karisik
    olcekte cikardi.

    SABIT KALAN: iliski semasi, yapisal kisitlar, BLOK=100, tur sayilari.
    Yani bu bir OLCEK dugmesidir, GOREV dugmesi degil.
    """
    o = olcek or {}
    N_KISI = o.get("n_kisi", globals()["N_KISI"])
    N_OKUL = o.get("n_okul", globals()["N_OKUL"])
    SOYAD = o.get("soyad", globals()["SOYAD"])
    IL = o.get("il", globals()["IL"])
    DERS = o.get("ders", globals()["DERS"])
    assert N_KISI % BLOK == 0, f"N_KISI ({N_KISI}) BLOK'a ({BLOK}) bolunmeli"
    assert N_KISI <= 2 * 50 * len(SOYAD), (
        f"N_KISI={N_KISI} isim havuzunu asiyor: 2*50*{len(SOYAD)} soyad "
        f"= {2 * 50 * len(SOYAD)}. SOYAD listesini buyut.")
    assert N_OKUL <= len(IL) * len(OKUL_TUR), (
        f"N_OKUL={N_OKUL} > {len(IL)}*{len(OKUL_TUR)}={len(IL)*len(OKUL_TUR)}. "
        f"IL listesini ya da OKUL_TUR'u buyut.")
    assert len(IL) % 2 == 0, "komsu esitlemesi icin IL sayisi CIFT olmali"

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
                olgu=olgu, sema=SEMA_TABAN, iliski=ILISKI)



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


# --- veri_okul3: SEKIZ ALAN ACILIMININ SEMASI --------------------------
# Eski girdiler AYNEN; yalniz yeni (tip -> tip) satirlari eklendi.
#
# !! KUSUR ve DUZELTMESI (16 Eylul hakemligi). Ilk birlestirmede TEK bir
# `SEMA` vardi: taban semanin USTUNE yaziliyordu. Sonuc grafi dogruydu
# (derin karsilastirma gecti) ama TABAN GRAF kendi basina TUTARSIZ
# kaliyordu -- `_taban_kur()` semasi henuz olgusu OLMAYAN tip ciftleri
# vaat ediyordu:
#
#     ORIJINAL  VO.zincirler(VO.kur(0))       -> 59.140 zincir
#     ILK KOPYA zincirler(_taban_kur(0))      -> KeyError ('...', 'rakip')
#
# Bugun kimse `_taban_kur`u tek basina cagirmiyor, ama bu bir MAYINDI.
# Orijinaldeki gibi IKI AYRI sozluk tutuluyor.
SEMA = {r: dict(m) for r, m in SEMA_TABAN.items()}
SEMA["kardes"].update({"OKUL": "OKUL", "SEHIR": "SEHIR"})
SEMA["komsu"].update({"KISI": "KISI", "OKUL": "OKUL"})
SEMA["rakip"].update({"KISI": "KISI", "SEHIR": "SEHIR"})
SEMA["kurucu"].update({"SEHIR": "KISI", "DERS": "KISI"})
SEMA["hoca"].update({"OKUL": "KISI"})
SEMA["okul"].update({"DERS": "OKUL"})
SEMA["sehir"].update({"DERS": "SEHIR"})
SEMA["ders"].update({"SEHIR": "DERS"})

assert set(SEMA) == set(SEMA_TABAN), "YENI SEMBOL EKLENMIS -- arama uzayi bozulur"
assert len(ILISKI) == 17, f"|R| 17 olmali: {len(ILISKI)}"


# ======================================================================
# veri_okul3 — SEKIZ ALAN ACILIMI
# ======================================================================
def _farkli_esle(rng, idx, yasak, ad, deneme=400):
    """Simetrik esleme kur, ama `yasak` sozluklerinin HICBIRIYLE ayni
    olmasin. `arkadas`in veri_okul.py'deki kurulusunun aynisi -- iki
    iliski ayni eslemeyse model birini digerinden OKUR."""
    idx = list(idx)
    for _ in range(deneme):
        e = _esle(rng, idx)
        if all(all(e[i] != y[i] for i in idx) for y in yasak):
            return e
    raise RuntimeError(f"{ad}: yasaklardan farkli esleme kurulamadi")



def genislet(G, tohum=0):
    """Verilen grafa SEKIZ ALAN ACILIMINI ekler. OLCEKTEN BAGIMSIZ --
    `veri_okul2` (2x) ile de `veri_okul` (1x) ile de calisir; phi
    OLCEK-DEGISMEZ oldugu icin tavan iki durumda da 9,57.

    Eski olgular BIREBIR korunur: gelen sozluge yalnizca EKLEME
    yapilir. Yeni olgular ayri bir RNG akisindan (tohum + 7777) gelir
    ki taban grafin akisi kaymasin."""
    ad, olgu = G["ad"], G["olgu"]
    kisi, okul, sehir, ders = ad["KISI"], ad["OKUL"], ad["SEHIR"], ad["DERS"]
    for t, v in (("KISI", kisi), ("OKUL", okul), ("SEHIR", sehir), ("DERS", ders)):
        assert len(v) % 2 == 0, f"{t} sayisi ({len(v)}) CIFT olmali (_esle)"
    eski = dict(olgu)                  # sonunda DEGISMEDIGI sinanir
    rng = np.random.RandomState(tohum + 7777)

    n_k, n_o, n_s, n_d = len(kisi), len(okul), len(sehir), len(ders)
    # mevcut simetrik eslemeler -- yenileri bunlardan FARKLI olmali
    _ind = lambda grup: {a: i for i, a in enumerate(grup)}
    ik, io, isz = _ind(kisi), _ind(okul), _ind(sehir)
    kardes_k = {i: ik[olgu[(c, "kardes")]] for i, c in enumerate(kisi)}
    arkadas_k = {i: ik[olgu[(c, "arkadas")]] for i, c in enumerate(kisi)}
    rakip_o = {i: io[olgu[(o, "rakip")]] for i, o in enumerate(okul)}
    komsu_s = {i: isz[olgu[(s, "komsu")]] for i, s in enumerate(sehir)}

    # --- KISI uzerine: komsu, rakip (kardes/arkadas ile CAKISMASIN) ------
    komsu_kisi = _farkli_esle(rng, range(n_k), [kardes_k, arkadas_k], "komsu/KISI")
    rakip_kisi = _farkli_esle(rng, range(n_k),
                              [kardes_k, arkadas_k, komsu_kisi], "rakip/KISI")
    for i, c in enumerate(kisi):
        olgu[(c, "komsu")] = kisi[komsu_kisi[i]]
        olgu[(c, "rakip")] = kisi[rakip_kisi[i]]

    # --- OKUL uzerine: kardes, komsu (rakip ile CAKISMASIN), hoca --------
    kardes_okul = _farkli_esle(rng, range(n_o), [rakip_o], "kardes/OKUL")
    komsu_okul = _farkli_esle(rng, range(n_o), [rakip_o, kardes_okul], "komsu/OKUL")
    for i, o in enumerate(okul):
        olgu[(o, "kardes")] = okul[kardes_okul[i]]
        olgu[(o, "komsu")] = okul[komsu_okul[i]]
        olgu[(o, "hoca")] = kisi[int(rng.randint(n_k))]

    # --- SEHIR uzerine: kardes, rakip (komsu ile CAKISMASIN), kurucu, ders
    kardes_sehir = _farkli_esle(rng, range(n_s), [komsu_s], "kardes/SEHIR")
    rakip_sehir = _farkli_esle(rng, range(n_s), [komsu_s, kardes_sehir], "rakip/SEHIR")
    for i, s in enumerate(sehir):
        olgu[(s, "kardes")] = sehir[kardes_sehir[i]]
        olgu[(s, "rakip")] = sehir[rakip_sehir[i]]
        olgu[(s, "kurucu")] = kisi[int(rng.randint(n_k))]
        olgu[(s, "ders")] = ders[int(rng.randint(n_d))]

    # --- DERS uzerine: kurucu, okul, sehir -------------------------------
    for i, d in enumerate(ders):
        olgu[(d, "kurucu")] = kisi[int(rng.randint(n_k))]
        olgu[(d, "okul")] = okul[int(rng.randint(n_o))]
        olgu[(d, "sehir")] = sehir[int(rng.randint(n_s))]

    G["sema"] = SEMA

    # --- DENETIMLER -------------------------------------------------------
    for (e, r), h in eski.items():
        assert olgu[(e, r)] == h, f"ESKI OLGU DEGISTI: {e} {r}"
    kendi = [(e, r) for (e, r), h in olgu.items() if h == e]
    assert not kendi, f"kendine giden olgu: {kendi[:5]}"
    for grup, r in ((kisi, "komsu"), (kisi, "rakip"), (okul, "kardes"),
                    (okul, "komsu"), (sehir, "kardes"), (sehir, "rakip")):
        bozuk = [x for x in grup if olgu[(olgu[(x, r)], r)] != x]
        assert not bozuk, f"{r} simetrik degil: {bozuk[:3]}"
    for (e, r) in olgu:
        assert e in G["tip"], f"bilinmeyen varlik: {e}"
        assert G["tip"][e] in SEMA[r], f"SEMA disi olgu: {G['tip'][e]} {r}"
        assert G["tip"][olgu[(e, r)]] == SEMA[r][G["tip"][e]], \
            f"TIP ihlali: {e} {r} {olgu[(e, r)]}"
    for t in TIPLER:
        gec = [r for r in ILISKI if t in SEMA[r]]
        for a in ad[t]:
            eksik = [r for r in gec if (a, r) not in olgu]
            assert not eksik, f"{a} ({t}) icin eksik olgu: {eksik}"
    return G



def turetilebilir(G, esik=0.5):
    """(r1, r2) ciftinin sonucu, TEK bir r3 olgusuyla AYNI mi?

    Boyle bir cift varsa o zincir 2-hop DEGILDIR: model r3'u ezberleyip
    gecer. Var olan tek ornek `kardes`+`baba` -> `baba` (GEREKTIRIR'de
    yazili, `zincirler` bunu AYNI sinifina atip sinav disi birakiyor).
    Bu tarama YENI bir tane dogmadigini sinar.

    Donus: [(r1, r2, r3, oran)] -- oran `esik`i ASANLAR.
    """
    olgu, tip, sema = G["olgu"], G["tip"], G["sema"]
    out = []
    for r1 in ILISKI:
        for r2 in ILISKI:
            pay = {}
            top = 0
            for (e, r), b in olgu.items():
                if r != r1 or tip[b] not in sema[r2]:
                    continue
                c = olgu[(b, r2)]
                top += 1
                for r3 in ILISKI:
                    if tip[e] in sema[r3] and olgu[(e, r3)] == c:
                        pay[r3] = pay.get(r3, 0) + 1
            for r3, k in pay.items():
                if top and k / top > esik:
                    out.append((r1, r2, r3, k / top))
    return sorted(out, key=lambda x: -x[3])


# ======================================================================
# model_04'IN VERISI — iki adim + PARMAK IZI
# ======================================================================

# tohum 0 grafinin parmak izi. Olculdu 16 Eylul 2026; uc ayri surecte
# ayni cikti (sozluk sirasina bagli DEGIL, her sey siralanip karilir).
IZ = "3f6751c4ccd4"


def graf_izi(G):
    """Grafin ICERIGINDEN tureyen sabit parmak izi.

    !! ILK SURUMUN IKI KOR NOKTASI VARDI (16 Eylul hakemligi, olculdu):

      1. `ad` listeleri SIRALANARAK karilyordu. Oysa varlik SIRASI
         `sozluk`u, `sozluk` da JETON ID'lerini belirliyor. KISI
         listesini ters cevirdim -- IZ KIPIRDAMADI, ama butun kodlama
         kaymis olurdu.
      2. `sema` hic karilmiyordu. SEMA'ya bir tip cifti ekledim --
         IZ yine ayni cikti, oysa gecerli zincir kumesi (SINAVIN
         KENDISI) degisirdi.

    Simdi: `olgu` SIRALI (sozluk sirasi Python surumune gore oynayabilir,
    icerik oynamaz), `ad` / `sozluk` / `iliski` SIRASIYLA, `sema` da
    dahil. Determinizm uc ayri surecte sinandi.
    """
    h = hashlib.sha256()
    h.update(repr(sorted(map(str, G["olgu"].items()))).encode())
    h.update(repr(list(G["iliski"])).encode())
    h.update(repr([(t, list(G["ad"][t])) for t in sorted(G["ad"])]).encode())
    h.update(repr(list(G["sozluk"])).encode())
    h.update(repr(sorted((r, sorted(m.items()))
                         for r, m in G["sema"].items())).encode())
    return h.hexdigest()[:12]


def kur(tohum=0):
    """model_04'in grafi: 1x taban + sekiz alan acilimi."""
    G = genislet(_taban_kur(tohum), tohum)
    if tohum == 0:
        _iz = graf_izi(G)
        assert _iz == IZ, (
            f"veri_04: graf DEGISTI  {IZ} -> {_iz}\n"
            "  Bu dosya TEK BASINA duruyor -- degisiklik BURADA yapildi.\n"
            "  Kasitliysa IZ yenilenir ve onkayda not duselir; degilse\n"
            "  degisiklik geri alinir. model_04 sessizce baska bir\n"
            "  veriyle KOSMAZ.")
    return G


if __name__ == "__main__":
    import collections
    G = kur(0)
    z = zincirler(G)
    print("veri_04  varlik %d  olgu %d  zincir %d  |R| %d  phi TAVANI %.2f"
          % (sum(G["n"].values()), len(G["olgu"]), len(z), len(ILISKI),
             len(z) / len(G["olgu"])))
    print("   sinif", dict(collections.Counter(x[6] for x in z)))
    print("   graf izi", graf_izi(G), " (beklenen", IZ + ")")
    print("\nTURETILEBILIR CIFTLER (oran > 0.5):")
    for r1, r2, r3, o in turetilebilir(G):
        print("   %-10s %-10s -> %-10s  %.3f   %s"
              % (r1, r2, r3, o, "BEKLENEN (GEREKTIRIR)"
                 if (r1, r2) in GEREKTIRIR else "!! YENI"))
