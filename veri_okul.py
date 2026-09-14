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
  tavan yapiyordu; burada ~5.8.
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
     Toplam disarida kalan: AYNI 2.297 + DONUS 2.816 = zincirlerin %9'u.
     (Cografya verisinde bu %29 idi.)

SOYADI KALITIMI: cocuk / kardes / anne / baba AYNI soyadi tasir; bir soyadi
blogu 50 kardes cifti = 100 kisi. Ebeveyn haritasi bu yuzden BLOK ICINDE
kaydirilir -- global kaydirma butun cifleri tek halkaya baglayip kalitimi
imkansiz kiliyordu. Assert ile denetleniyor.

GERCEK olan : kisi adlari, 81 il, ders adlari, okul adlari
URETILMIS   : kim kimin akrabasi/ogretmeni/arkadasi (tohumla, tutarli)

    python veri_okul.py [--dok cikti.txt]
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
OGR_KAY = 211      # ogretmen halkasi: sabit kaydirma -> tam esleme, sabit nokta yok
CIFT_BLOK = 50     # bir SOYADI blogunda kac kardes cifti (50 cift = 100 kisi)
BABA_KAY, ANNE_KAY = 17, 31    # BLOK ICINDE kaydirma; gcd(.,50)=1 -> birebir


def _eb(kc, kay):
    """Kardes cifti kc'nin ebeveyninin geldigi cift. Kaydirma SOYADI
    BLOGUNUN ICINDE kalir; global kaydirma (eski hali: (kc+97) % 350)
    butun cifleri tek halkaya bagliyordu ve soyadi kalitimi IMKANSIZ oluyordu.

    BABA ve ANNE AYRI kaydirmadan gelir. Tek kaydirma kullanilinca baba ile
    anne AYNI ciftin iki uyesi, yani BIRBIRININ KARDESI oluyordu
    (Ahmet_Yilmaz kardes Ayse_Yilmaz, ikisinin de cocuk Onur_Yilmaz)."""
    b = kc // CIFT_BLOK
    return b * CIFT_BLOK + ((kc % CIFT_BLOK + kay) % CIFT_BLOK)

N_KISI, N_OKUL = 700, 200


def kur(tohum=0):
    rng = np.random.RandomState(tohum)

    # cift indeks ERKEK, tek indeks KADIN. Aile kurulumu bunu kullaniyor:
    # baba kisi[2p] (erkek), anne kisi[2p+1] (kadin).
    kisi = [f"{(ERKEK if i % 2 == 0 else KADIN)[(i // 2) % 50]}"
            f"_{SOYAD[(i // 2) // 50]}" for i in range(N_KISI)]
    assert len(set(kisi)) == N_KISI, "kisi adi tekrari"
    sehir = list(IL)
    ders = list(DERS)
    okul = [f"{sehir[i % len(sehir)]}_{OKUL_TUR[(i // len(sehir)) % 3]}"
            for i in range(N_OKUL)]
    assert len(set(okul)) == N_OKUL, "okul adi tekrari"

    ad = {"KISI": kisi, "OKUL": okul, "SEHIR": sehir, "DERS": ders}
    hepsi = [a for t in TIPLER for a in ad[t]]
    assert len(hepsi) == len(set(hepsi)), \
        "TEKRAR EDEN AD: " + str([a for a in set(hepsi) if hepsi.count(a) > 1])

    olgu = {}
    n = N_KISI

    # --- AILE: kisiler IKISERLI kardes. 350 kardes cifti. -----------------
    # Kardesler AYNI anne-babayi paylasir (tutarli). Bunun bedeli
    # ("kardes baba" = "baba") olculur ve sinavdan cikarilir.
    for k in range(0, n, 2):
        olgu[(kisi[k], "kardes")] = kisi[k + 1]
        olgu[(kisi[k + 1], "kardes")] = kisi[k]
        # cift k//2'nin ebeveyni: AYNI SOYADI blogundan, ama baba ile anne
        # FARKLI ciflerden -- yoksa evli cift birbirinin kardesi olur.
        baba = kisi[2 * _eb(k // 2, BABA_KAY)]
        anne = kisi[2 * _eb(k // 2, ANNE_KAY) + 1]
        for c in (kisi[k], kisi[k + 1]):
            olgu[(c, "baba")] = baba
            olgu[(c, "anne")] = anne

    # --- cocuk: baba/anne'nin TERSI, TEK DEGERLI -------------------------
    # Her kisi TAM OLARAK bir kardes ciftinin ebeveyni (harita birebir), yani
    # `cocuk` her kiside tanimli. Kayitli cocuk ciftin ILKI: ikinci cocuk icin
    # "X baba cocuk" -> KARDESI cikar, o zincir GERCEK kompozisyondur.
    for k in range(0, n, 2):
        olgu[(kisi[2 * _eb(k // 2, BABA_KAY)], "cocuk")] = kisi[k]
        olgu[(kisi[2 * _eb(k // 2, ANNE_KAY) + 1], "cocuk")] = kisi[k]
    assert all((c, "cocuk") in olgu for c in kisi), "cocuk eksik"

    # SOYADI KALITIMI: cocuk, kardes, anne, baba AYNI soyadi tasimali.
    _soy = lambda a: a.rsplit("_", 1)[1]
    for c in kisi:
        for r in ("kardes", "baba", "anne", "cocuk"):
            assert _soy(olgu[(c, r)]) == _soy(c),                 f"soyadi kalitimi bozuk: {c} {r} {olgu[(c, r)]}"
        # evli cift birbirinin kardesi OLMAMALI
        assert olgu[(olgu[(c, "baba")], "kardes")] != olgu[(c, "anne")],             f"anne-baba kardes cikti: {c}"

    # --- ogretmen / ogrenci: TAM ESLEME, tersi KESIN ---------------------
    # Rastgele harita denendi: kisilerin ~%37'si kimseye ogretmen olmuyor ve
    # `ogrenci` onlarda UYDURUK dolduruluyordu. Sabit kaydirma birebir ve
    # sabit noktasiz; `ogrenci` tam ters. Bedeli: "ogretmen ogrenci" zinciri
    # her zaman DONUS (700 zincir, sinavdan cikar) -- sayilir ve yazilir.
    for i, c in enumerate(kisi):
        olgu[(c, "ogretmen")] = kisi[(i + OGR_KAY) % n]
        olgu[(c, "ogrenci")] = kisi[(i - OGR_KAY) % n]

    # --- arkadas: SIMETRIK, kardes cifti DISINDA bir eslesme -------------
    # kisi[i] <-> kisi[(i + n//2) % n]  -> karsilikli ve kardesten farkli
    for i, c in enumerate(kisi):
        olgu[(c, "arkadas")] = kisi[(i + n // 2) % n]

    # --- okul / sehir / ders --------------------------------------------
    # DIKKAT: kisinin sehri, OKULUNUN sehrinden BAGIMSIZ secilir. Bagimli
    # olsaydi "X okul sehir" = "X sehir" olur ve zincir olculemezdi
    # (cografya verisinde `dili` boyle bozulmustu).
    for i, c in enumerate(kisi):
        olgu[(c, "okul")] = okul[int(rng.randint(N_OKUL))]
        olgu[(c, "sehir")] = sehir[int(rng.randint(len(sehir)))]
        olgu[(c, "ders")] = ders[int(rng.randint(len(ders)))]
    for i, o in enumerate(okul):
        olgu[(o, "mudur")] = kisi[int(rng.randint(n))]
        olgu[(o, "kurucu")] = kisi[int(rng.randint(n))]
        olgu[(o, "sehir")] = sehir[i % len(sehir)]        # okul ilindedir
        olgu[(o, "rakip")] = okul[i + 1 if i % 2 == 0 else i - 1]  # SIMETRIK
        olgu[(o, "ders")] = ders[int(rng.randint(len(ders)))]
    for i, s_ in enumerate(sehir):
        # SIMETRIK: i<->i+1 ciftleri. Halka kullanilirsa "Adana komsu
        # Adiyaman" olur ama "Adiyaman komsu Adana" OLMAZ -- komsuluk
        # anlamca simetrik, 81/81 il bu kontrolden kaliyordu.
        olgu[(s_, "komsu")] = sehir[i + 1 if i % 2 == 0 else i - 1]
        olgu[(s_, "vali")] = kisi[int(rng.randint(n))]
        olgu[(s_, "okul")] = okul[int(rng.randint(N_OKUL))]
    for i, d in enumerate(ders):
        olgu[(d, "hoca")] = kisi[int(rng.randint(n))]
        olgu[(d, "onkosul")] = ders[(i + 1) % len(ders)]

    # SIMETRIK iliskiler: r(r(x)) == x.  Halka kurulumu bu kontrolden
    # kaliyordu ve "A komsu B ama B komsu A degil" uretiyordu.
    for grup, r in ((sehir, "komsu"), (okul, "rakip"),
                    (kisi, "kardes"), (kisi, "arkadas")):
        bozuk = [x for x in grup if olgu[(olgu[(x, r)], r)] != x]
        assert not bozuk, f"{r} simetrik degil: {bozuk[:3]}"

    # kendine gitmesin
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
