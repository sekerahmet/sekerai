# -*- coding: utf-8 -*-
"""VERI_KONTROL — `veri_okul.py`'nin TUTARLILIGINI mekanik olarak denetler.

Yargiya bagli degil: her kontrol bir kosul ve bir SAYI. Sifir degilse bozuk.
CLAUDE.md 10a: "Olcumu bilinen bir degere karsi dogrula."

Bu dosya iki gercek arizayi yakaladi (14 Eylul):
  komsu simetrik degil   81/81 il   "Adana komsu Adiyaman" ama tersi yok
  rakip simetrik degil  200/200 okul

Ayrica OKUMAYLA (kontrolle degil) uc anlamsiz iliski bulundu ve cikarildi:
  SEHIR->ders ("Ankara ders Fizik"), DERS->sehir ("Matematik sehir Bolu"),
  kitap DERS->DERS ("Matematik kitap Geometri" -> `onkosul` oldu).
  Maliyeti olculdu: zincir -%4, AYIRT -%5.

BILEREK KABUL EDILEN (kapali dunyada kacinilmaz, denetim BILGI olarak basar):
  * ata halkasi 50 kusak -- `baba` 50 adimda basa donuyor
  * `onkosul` halkasi 80 ders -- mantiken hicbir ders alinamaz
  * okul adi ilini iceriyor (Ankara_Lisesi). SIZINTI DEGIL: her varlik TEK
    token, model adin icindeki yapiyi gormuyor.

    python veri_kontrol.py
"""
import collections, sys
import veri_okul as V

G = V.kur(); o = G["olgu"]; tip = G["tip"]; ad = G["ad"]
K, O, S, D = ad["KISI"], ad["OKUL"], ad["SEHIR"], ad["DERS"]
ERK = set(V.ERKEK); KAD = set(V.KADIN)
ilk = lambda a: a.rsplit("_", 1)[0]
soy = lambda a: a.rsplit("_", 1)[1]
bulgu = []
def kontrol(ad_, kosul_bozan, ornek=None):
    n = len(kosul_bozan)
    print(f"{'BOZUK' if n else '  ok ':6s} {ad_:52s} {n}")
    if n:
        bulgu.append((ad_, n, kosul_bozan[:3]))

# --- 1. TAMLIK: her tipin her iliskisi dolu, hedef tipi dogru -----------
eksik, yanlis_tip = [], []
for t in V.TIPLER:
    for r, m in V.SEMA.items():
        if t not in m: continue
        for e in ad[t]:
            if (e, r) not in o: eksik.append((e, r))
            elif tip[o[(e, r)]] != m[t]: yanlis_tip.append((e, r, o[(e, r)]))
kontrol("her (varlik, iliski) dolu", eksik)
kontrol("hedefin TIPI semaya uyuyor", yanlis_tip)

# --- 2. SIMETRI ---------------------------------------------------------
kontrol("kardes simetrik  k(k(x))==x",
        [x for x in K if o[(o[(x, "kardes")], "kardes")] != x])
kontrol("arkadas simetrik a(a(x))==x",
        [x for x in K if o[(o[(x, "arkadas")], "arkadas")] != x])
kontrol("komsu simetrik   (SEHIR)",
        [x for x in S if o[(o[(x, "komsu")], "komsu")] != x])
kontrol("rakip simetrik   (OKUL)",
        [x for x in O if o[(o[(x, "rakip")], "rakip")] != x])

# --- 3. TERS CIFTLER ----------------------------------------------------
kontrol("ogrenci = ogretmen'in tersi",
        [x for x in K if o[(o[(x, "ogretmen")], "ogrenci")] != x])
kontrol("erkegin cocugunun babasi kendisi",
        [x for x in K if ilk(x) in ERK and o[(o[(x, "cocuk")], "baba")] != x])
kontrol("kadinin cocugunun annesi kendisi",
        [x for x in K if ilk(x) in KAD and o[(o[(x, "cocuk")], "anne")] != x])

# --- 4. AILE TUTARLILIGI ------------------------------------------------
kontrol("kardesler ayni babayi paylasiyor",
        [x for x in K if o[(x, "baba")] != o[(o[(x, "kardes")], "baba")]])
kontrol("kardesler ayni anneyi paylasiyor",
        [x for x in K if o[(x, "anne")] != o[(o[(x, "kardes")], "anne")]])
kontrol("baba ERKEK adi", [x for x in K if ilk(o[(x, "baba")]) not in ERK])
kontrol("anne KADIN adi", [x for x in K if ilk(o[(x, "anne")]) not in KAD])
kontrol("soyadi kalitimi (kardes/baba/anne/cocuk)",
        [(x, r) for x in K for r in ("kardes", "baba", "anne", "cocuk")
         if soy(o[(x, r)]) != soy(x)])
kontrol("anne ile baba KARDES degil",
        [x for x in K if o[(o[(x, "baba")], "kardes")] == o[(x, "anne")]])
kontrol("kisi kendi kardesi/babasi/annesi/cocugu degil",
        [(x, r) for x in K for r in ("kardes", "baba", "anne", "cocuk")
         if o[(x, r)] == x])
kontrol("babasi ayni zamanda cocugu DEGIL",
        [x for x in K if o[(x, "baba")] == o[(x, "cocuk")]])
kontrol("kardesi ayni zamanda ebeveyni DEGIL",
        [x for x in K if o[(x, "kardes")] in (o[(x, "baba")], o[(x, "anne")])])
kontrol("dede (baba baba) kisinin kendisi DEGIL",
        [x for x in K if o[(o[(x, "baba")], "baba")] == x])
kontrol("her kisi TAM 1 ciftin babasi/annesi (cocuk ortusmesi yok)",
        [k for k, v in collections.Counter(
            o[(x, "cocuk")] for x in K).items() if v != 2])

# --- 5. SOZLUK ----------------------------------------------------------
kontrol("sozlukte tekrar yok",
        [a for a, c in collections.Counter(G["sozluk"]).items() if c > 1])
kontrol("iliski adi bir varlik adi DEGIL",
        [r for r in V.ILISKI if r in tip])
kontrol("okul adindaki il, okulun sehri ile ayni",
        [x for x in O if not x.startswith(o[(x, "sehir")] + "_")])

# --- 6. ATA HALKASI (kacinilmaz, ama OLCULSUN) --------------------------
x = K[0]; gor = []
for _ in range(400):
    x = o[(x, "baba")]
    if x in gor: break
    gor.append(x)
print(f"  bilgi  ata halkasi uzunlugu (baba baba ...)          {len(gor)}")

# --- 7. CEVAP YOGUNLASMASI ---------------------------------------------
for r in V.ILISKI:
    c = collections.Counter(h for (e, rr), h in o.items() if rr == r)
    top = c.most_common(1)[0]
    print(f"  bilgi  {r:9s} {len(c):4d} farkli cevap, en sik {top[1]:4d} kez")

print()
print(f"BOZUK KONTROL SAYISI: {len(bulgu)}")
for a, n, orn in bulgu:
    print(f"  - {a}: {n} ornek, ilk 3 {orn}")
