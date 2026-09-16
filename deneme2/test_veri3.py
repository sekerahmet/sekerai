# -*- coding: utf-8 -*-
"""veri_okul3 KILIDI.

`veri_okul3` bir OLCEK degil YOGUNLUK dugmesidir ve uzerine kol
kurulacak. Kilit su dordunu tutar:

  1. veri_okul2 bu grafin ALT KUMESI -- BIREBIR (kontrol boyle gecerli)
  2. |R| = 17 DEGISMEDI -- arama uzayi (17^2) sabit
  3. phi TAVANI 9,57 -- kolun sebebi bu sayi
  4. TURETILEBILIR cift yalniz GEREKTIRIR'dekiler

AYRINTI=1 -> gecen kontroller de basilir.
"""
import os
import sys

_B = os.path.dirname(os.path.abspath(__file__))
if _B not in sys.path:
    sys.path.insert(0, _B)

import veri_okul as VO          # noqa: E402
import veri_okul2 as V2         # noqa: E402
import veri_okul3 as V3         # noqa: E402

AYRINTI = os.environ.get("AYRINTI") == "1"
_gecti = _bozuk = 0


def ok(kosul, ad, ek=""):
    global _gecti, _bozuk
    if kosul:
        _gecti += 1
        if AYRINTI:
            print(f"    ok   {ad}" + (f"  {ek}" if ek else ""))
    else:
        _bozuk += 1
        print(f"    !!   BOZUK: {ad}" + (f"  {ek}" if ek else ""))


print("=== veri_okul3 KILIDI ===")
G2 = V2.kur(0)
G3 = V3.kur(0)
z2 = VO.zincirler(G2)
z3 = V3.zincirler(G3)

# --- 1) ALT KUME, BIREBIR --------------------------------------------------
eksik = [k for k in G2["olgu"] if k not in G3["olgu"]]
ok(not eksik, "veri_okul2'nin HER olgusu veri_okul3'te var", f"eksik {len(eksik)}")
degisen = [k for k, v in G2["olgu"].items() if G3["olgu"].get(k) != v]
ok(not degisen, "veri_okul2 olgulari DEGISMEDI (birebir)",
   f"degisen {len(degisen)}")
ok(G2["ad"] == G3["ad"], "varlik kumesi AYNI (2120)")
ok(G2["sozluk"] == G3["sozluk"], "SOZLUK degismedi")
ok(G2["tip"] == G3["tip"], "tip haritasi degismedi")

# --- 2) ARAMA UZAYI SABIT --------------------------------------------------
ok(len(G3["iliski"]) == 17, "|R| = 17 (YENI SEMBOL YOK)", str(len(G3["iliski"])))
ok(list(G3["iliski"]) == list(G2["iliski"]), "iliski LISTESI birebir ayni")
ok(set(V3.SEMA) == set(VO.SEMA), "sema ANAHTARLARI ayni -- yalniz ALAN acildi")

# --- 3) YOGUNLUK -----------------------------------------------------------
phi2 = len(z2) / len(G2["olgu"])
phi3 = len(z3) / len(G3["olgu"])
ok(len(G3["olgu"]) == 21920, "olgu 21.920", str(len(G3["olgu"])))
ok(len(z3) == 209880, "zincir 209.880", str(len(z3)))
ok(abs(phi3 - 9.57) < 0.01, f"phi TAVANI 9,57", f"{phi3:.4f}  (okul2 {phi2:.4f})")
ok(phi3 > phi2, "phi YUKSELDI", f"{phi2:.2f} -> {phi3:.2f}")
bek = {"KISI": 12, "OKUL": 8, "SEHIR": 7, "DERS": 5}
for t, b in bek.items():
    k = sum(1 for r in G3["iliski"] if t in G3["sema"][r])
    ok(k == b, f"{t} tipinde {b} iliski", str(k))
# sinav sinifi BUYUDU mu
import collections                                              # noqa: E402
s2 = collections.Counter(x[6] for x in z2)
s3 = collections.Counter(x[6] for x in z3)
ok(s3["AYIRT"] > s2["AYIRT"], "AYIRT (sinav sinifi) buyudu",
   f"{s2['AYIRT']} -> {s3['AYIRT']}")
ok(s3["DONUS"] / len(z3) < s2["DONUS"] / len(z2),
   "DONUS ORANI artmadi", f"{s2['DONUS']/len(z2):.4f} -> {s3['DONUS']/len(z3):.4f}")

# --- 4) TURETILEBILIRLIK ---------------------------------------------------
tur = V3.turetilebilir(G3)
yeni = [x for x in tur if (x[0], x[1]) not in V3.GEREKTIRIR]
ok(not yeni, "YENI turetilebilir cift YOK", str(yeni[:3]))
ok(len(tur) == 2, "yalniz GEREKTIRIR'in 2 cifti", str([(a, b) for a, b, *_ in tur]))
# iki iliski AYNI eslemeyse model birini digerinden okur
cak = []
for t in V3.TIPLER:
    rs = [r for r in G3["iliski"] if t in G3["sema"][r]]
    for i, r1 in enumerate(rs):
        for r2 in rs[i + 1:]:
            if G3["sema"][r1][t] != G3["sema"][r2][t]:
                continue
            ayni = sum(1 for a in G3["ad"][t]
                       if G3["olgu"][(a, r1)] == G3["olgu"][(a, r2)])
            if ayni / len(G3["ad"][t]) > 0.5:
                cak.append((t, r1, r2, ayni / len(G3["ad"][t])))
ok(not cak, "ayni tipte IKI ILISKI ayni eslemeyi kullanmiyor", str(cak[:3]))

# --- 5) YAPI ---------------------------------------------------------------
kendi = [(e, r) for (e, r), h in G3["olgu"].items() if h == e]
ok(not kendi, "kendine giden olgu YOK", str(kendi[:3]))
for grup, r in (("KISI", "komsu"), ("KISI", "rakip"), ("OKUL", "kardes"),
                ("OKUL", "komsu"), ("SEHIR", "kardes"), ("SEHIR", "rakip"),
                ("KISI", "kardes"), ("KISI", "arkadas"), ("OKUL", "rakip"),
                ("SEHIR", "komsu")):
    bozuk = [x for x in G3["ad"][grup] if G3["olgu"][(G3["olgu"][(x, r)], r)] != x]
    ok(not bozuk, f"{r}/{grup} SIMETRIK", str(bozuk[:2]))
ihl = [(e, r) for (e, r), h in G3["olgu"].items()
       if G3["tip"][h] != G3["sema"][r][G3["tip"][e]]]
ok(not ihl, "TIP ihlali yok", str(ihl[:3]))
eks = [(a, r) for t in V3.TIPLER for a in G3["ad"][t]
       for r in G3["iliski"] if t in G3["sema"][r] and (a, r) not in G3["olgu"]]
ok(not eks, "her varligin her GECERLI iliskisi TANIMLI", str(eks[:3]))
_soy = lambda a: a.rsplit("_", 1)[1]
bozuk = [c for c in G3["ad"]["KISI"] for r in ("kardes", "baba", "anne", "cocuk")
         if _soy(G3["olgu"][(c, r)]) != _soy(c)]
ok(not bozuk, "soyadi kalitimi DURUYOR (aile iliskileri)", str(bozuk[:2]))

# --- 6) BELIRLENIM ---------------------------------------------------------
ok(V3.kur(0)["olgu"] == G3["olgu"], "kur(0) BELIRLENIMLI (iki cagri ayni)")
ok(V3.kur(1)["olgu"] != G3["olgu"], "kur(1) FARKLI graf uretiyor")

# --- 7) MODUL SOZLESMESI (model_a ne bekliyor) -----------------------------
for g in ("kur", "zincirler", "TIPLER", "ILISKI"):
    ok(hasattr(V3, g), f"veri_okul3.{g} var")

print(f"\n{_gecti} gecti, {_bozuk} BOZUK")
sys.exit(1 if _bozuk else 0)
