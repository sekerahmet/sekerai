# -*- coding: utf-8 -*-
"""veri_okul2 — okul grafinin IKI KATI.  1060 -> 2120 varlik, 8400 -> 16800 olgu.

Kullanici karari, 15 Eylul 2026.

SORU: `wd=0.5` calisirken OLCEK ne yapiyor?

Arsivde olcek DENENMISTI ve KOTULESMISTI (veri x4 -> A4 ENT 0.0113).
AMA o kosularin hepsi `wd=0.1` ile yapildi -- yani bugun yanlis tarafta
oldugunu OLCTUGUMUZ degerle. `model_a3` (wd=0.5) ent'i 0.4240'tan
0.5980'e cikardi. Olcek sorusu bu zeminde YENIDEN aciktir.

--------------------------------------------------------------------------
NE DEGISIYOR: YALNIZ BOYUT

                     1x        2x
    KISI            700      1400
    OKUL            200       400
    SEHIR            80       160
    DERS             80       160
    ------------------------------
    VARLIK          1060      2120     tam 2x
    OLGU            8400     16800     tam 2x

DORT HAVUZ DA IKIYE KATLANIYOR -- bu KASITLI. Yalniz KISI ve OKUL
buyutulseydi 1400 kisi yine 80 sehre dagilirdi, yani "kisi basina sehir"
yogunlugu degisirdi ve bu bir OLCEK degil GOREV degisikligi olurdu.
Boyle: her oran sabit.

    kisi/sehir      8,75  ->  8,75
    kisi/okul       3,50  ->  3,50
    okul/sehir      2,50  ->  2,50

NE DEGISMIYOR: iliski semasi (17 iliski), yapisal kisitlar (kardesler ayni
ebeveyn, ters ciftler, soyadi kalitimi, simetri), BLOK=100, OKUL_TUR (3),
bolme oranlari (ent_pay 0.20, comp_pay 0.10, arama_pay 0.25).

--------------------------------------------------------------------------
HAVUZ MALZEMESI NEREDEN

`veri_okul.py`nin havuzlari TAVANDAYDI: N_KISI=700 tam olarak
2 x 50 isim x 7 soyad. Yani buyutmek bir parametre degisikligi DEGIL,
yeni isim malzemesi gerektiriyordu.

    SOYAD   7 -> 14     yedi gercek Turk soyadi eklendi
    IL     80 -> 160    seksen gercek Turk ILCE adi eklendi
    DERS   80 -> 160    seksen ders adi eklendi

Adlarin "gercek" olmasi modele hicbir sey ifade etmez -- hepsi sadece
token. Okunabilirlik icin gercek adlar secildi: `tani_a` ciktisina
bakarken "Ahmet_Ozturk" okumak "e1423" okumaktan iyi.

--------------------------------------------------------------------------
BU DOSYA GRAFI YENIDEN TANIMLAMAZ

`veri_okul.kur()` 15 Eylul'de `olcek` parametresi aldi; varsayilan
`None` ve o durumda graf BIREBIR eskisi gibi cikiyor (olgu SHA256
`44a1016bd74f894b`, dogrulandi). Burasi yalniz o parametreyi dolduruyor.
`zincirler`, `TIPLER`, `ILISKI`, `SEMA` -- hepsi `veri_okul`dan.
"""
from __future__ import annotations

import veri_okul as VO

# ---------------------------------------------------------------- HAVUZLAR
SOYAD_EK = ["Ozturk", "Arslan", "Dogan", "Kilic", "Aslan", "Cetin", "Korkmaz"]

# Gercek Turk ILCE adlari. IL listesiyle cakismayacak sekilde secildi.
IL_EK = [
    "Uskudar", "Kadikoy", "Besiktas", "Fatih", "Beyoglu", "Bakirkoy",
    "Maltepe", "Pendik", "Kartal", "Tuzla", "Silivri", "Catalca",
    "Sariyer", "Zeytinburnu", "Bagcilar", "Esenler", "Gungoren", "Avcilar",
    "Kucukcekmece", "Sisli", "Beykoz", "Umraniye", "Atasehir", "Sancaktepe",
    "Cekmekoy", "Sultanbeyli", "Basaksehir", "Arnavutkoy", "Beylikduzu",
    "Esenyurt", "Cankaya", "Kecioren", "Mamak", "Etimesgut", "Sincan",
    "Yenimahalle", "Pursaklar", "Golbasi", "Polatli", "Beypazari",
    "Konak", "Bornova", "Karsiyaka", "Buca", "Bayrakli", "Cigli",
    "Gaziemir", "Narlidere", "Cesme", "Urla", "Nilufer", "Osmangazi",
    "Yildirim", "Gemlik", "Inegol", "Mudanya", "Iznik", "Orhangazi",
    "Selcuklu", "Meram", "Karatay", "Eregli", "Aksehir", "Beysehir",
    "Seyhan", "Cukurova", "Yuregir", "Ceyhan", "Kozan", "Tarsus",
    "Akdeniz", "Toroslar", "Yenisehir", "Silifke", "Anamur", "Alanya",
    "Manavgat", "Kemer", "Kas", "Serik",
]

DERS_EK = [
    "Topoloji", "Kombinatorik", "Kriptografi", "Grafteorisi", "Sayilar_Teorisi",
    "Diferansiyel", "Integral", "Lineer_Cebir", "Numerik_Analiz", "Optimizasyon",
    "Kuantum", "Gorelilik", "Astrofizik", "Nukleer_Fizik", "Plazma_Fizigi",
    "Akiskanlar", "Dalga_Teorisi", "Akustik", "Kristalografi", "Yariiletken",
    "Polimer", "Elektrokimya", "Fizikokimya", "Analitik_Kimya", "Spektroskopi",
    "Mikrobiyoloji", "Immunoloji", "Farmakoloji", "Fizyoloji", "Norobilim",
    "Embriyoloji", "Histoloji", "Patoloji", "Toksikoloji", "Epidemiyoloji",
    "Meteoroloji", "Osinografi", "Sismoloji", "Volkanoloji", "Mineraloji",
    "Paleontoloji", "Kartografya", "Jeodezi", "Hidroloji", "Klimatoloji",
    "Etik", "Estetik", "Retorik", "Epistemoloji", "Metafizik",
    "Dilbilim", "Fonetik", "Semantik", "Cevirbilim", "Halkbilim",
    "Opera", "Bale", "Orkestra", "Koro", "Seramik",
    "Grafik", "Animasyon", "Tipografi", "Moda", "Peyzaj",
    "Finans", "Bankacilik", "Sigortacilik", "Vergi_Hukuku", "Maliye",
    "Ceza_Hukuku", "Medeni_Hukuk", "Ticaret_Hukuku", "Anayasa_Hukuku",
    "Uluslararasi_Hukuk",
    "Yazilim", "Mekatronik", "Otomasyon", "Telekomunikasyon", "Yapay_Zeka",
]

SOYAD2 = list(VO.SOYAD) + SOYAD_EK        #   7 ->  14
IL2 = list(VO.IL) + IL_EK                 #  80 -> 160
DERS2 = list(VO.DERS) + DERS_EK           #  80 -> 160

OLCEK2 = dict(n_kisi=1400, n_okul=400, soyad=SOYAD2, il=IL2, ders=DERS2)

# --- HAVUZ DENETIMLERI -- sessiz gecmesin ---------------------------------
for _ad, _l, _n in (("SOYAD", SOYAD2, 14), ("IL", IL2, 160), ("DERS", DERS2, 160)):
    assert len(_l) == _n, f"{_ad} {len(_l)} olmali {_n}"
    assert len(set(_l)) == _n, (
        f"{_ad} TEKRAR: " + str(sorted({x for x in _l if _l.count(x) > 1})))
# Sehir ve ders adlari AYNI ad uzayinda (ikisi de varlik) -- cakisamazlar.
_ort = set(IL2) & set(DERS2)
assert not _ort, f"IL ve DERS cakisiyor: {sorted(_ort)}"

TIPLER = VO.TIPLER
ILISKI = VO.ILISKI
SEMA = VO.SEMA
zincirler = VO.zincirler


def kur(tohum=0):
    """2x okul grafi. Imza `veri_okul.kur` ile AYNI -- `model_a` ayrimsiz cagirir."""
    return VO.kur(tohum, olcek=OLCEK2)


if __name__ == "__main__":
    import collections
    for ad, M in (("1x", VO), ("2x", __import__("veri_okul2"))):
        G = M.kur(0)
        z = M.zincirler(G)
        n = {t: len(G["ad"][t]) for t in TIPLER}
        print("%s  varlik %5d %s  olgu %6d  zincir %7d  sozluk %5d"
              % (ad, sum(n.values()), n, len(G["olgu"]), len(z), len(G["sozluk"])))
        print("     sinif", dict(collections.Counter(x[6] for x in z)))
