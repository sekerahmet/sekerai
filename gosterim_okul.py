# -*- coding: utf-8 -*-
"""GOSTERIM — G deneyinde modele NE ogretiyoruz, NE soruyoruz.

Token dizilimi `sifirdan.py`'nin enc_one/enc_two'sundan BIREBIR alindi
(elle kopyalanmadi, satirlar asagida):
    enc_one (sifirdan.py:321)  [S1] e r  ? cevap <son>   hedef POZ 3
    enc_two (sifirdan.py:357)  [S2] e r1 r2 ? cevap <son>  hedef POZ 4
Ozel token'lar sifirdan.py:222 -> PAD 0, S1 1, S2 2, ? 3, <son> 4
REL_OFF = 8, ENT_OFF = 8 + iliski sayisi.

    python gosterim_okul.py
"""
import collections
import numpy as np
import veri_okul as V

G = V.kur(); z = V.zincirler(G)
VARLIK = [a for t in V.TIPLER for a in G["ad"][t]]
EID = {a: i for i, a in enumerate(VARLIK)}
RID = {r: i for i, r in enumerate(V.ILISKI)}
SPECIAL, REL_OFF = 8, 8
ENT_OFF = SPECIAL + len(V.ILISKI)
VOCAB = ENT_OFF + len(VARLIK)
S1, S2, QM, EOS = 1, 2, 3, 4

def d1(e, r, a):                       # 1 adimli
    return ([S1, ENT_OFF+EID[e], REL_OFF+RID[r], QM, ENT_OFF+EID[a], EOS],
            ["[S1]", e, r, "?", a, "<son>"], 3)

def d2(e, r1, r2, a):                  # 2 adimli
    return ([S2, ENT_OFF+EID[e], REL_OFF+RID[r1], REL_OFF+RID[r2], QM,
             ENT_OFF+EID[a], EOS],
            ["[S2]", e, r1, r2, "?", a, "<son>"], 4)

def bas(t):
    ids, kel, hp = t
    print("      " + "  ".join(f"{k}" for k in kel))
    print("      " + str(ids) + f"   <- tahmin POZ {hp}'te yapilir "
          f"(orada dogru cevap {ids[hp+1]} beklenir)")

print("=" * 76)
print("SOZLUK (tokenizer)")
print("=" * 76)
print(f"  0..7      ozel     : <pad> [S1] [S2] ? <son> (+3 kullanilmayan)")
print(f"  8..{ENT_OFF-1}     iliski   : {len(V.ILISKI)} adet -> " +
      ", ".join(f"{r}={REL_OFF+RID[r]}" for r in V.ILISKI[:6]) + ", ...")
print(f"  {ENT_OFF}..{VOCAB-1}  varlik   : {len(VARLIK)} adet -> " +
      ", ".join(f"{a}={ENT_OFF+EID[a]}" for a in VARLIK[:4]) + ", ...")
print(f"  VOCAB = {VOCAB}   (kelime duzeyi: her varlik/iliski TEK token,"
      f" alt-parca YOK)")

# ---------------------------------------------------------------- BOLMELER
bs = collections.defaultdict(list)
for x in z: bs[x[0]].append(x)
rng = np.random.RandomState(0); ENT = set()
for t in V.TIPLER:
    a = [x for x in G["ad"][t] if x in bs]
    ENT |= set(a[i] for i in rng.permutation(len(a))[:int(round(len(a)*0.20))])
ent = [x for x in z if x[0] in ENT]; tr = []; comp = []
for e, lst in bs.items():
    if e in ENT: continue
    p = rng.permutation(len(lst)); k = int(round(len(lst)*0.90))
    for i, j in enumerate(p): (tr if i < k else comp).append(lst[j])

print("\n" + "=" * 76)
print("1) NE OGRETIYORUZ  — egitim kumesi")
print("=" * 76)
print(f"\n  A. BILGI: butun atomik olgular, 1 ADIMLI SORU olarak. {len(G['olgu'])} adet")
for (e, r), a in list(G["olgu"].items())[:3]:
    bas(d1(e, r, a))
print(f"\n  B. GOREV: 2 adimli zincirlerin bir kismi. {len(tr)} adet"
      f"   (phi = {len(tr)}/{len(G['olgu'])} = {len(tr)/len(G['olgu']):.2f})")
for e, r1, r2, b, a, ks, s in tr[:3]:
    bas(d2(e, r1, r2, a))
    print(f"          kopru {b} DIZIDE YOK -- model onu yazmadan kullanmali")

print("\n" + "=" * 76)
print("2) NE SORUYORUZ  — sinav kumeleri (hicbiri egitimde YOK)")
print("=" * 76)
tabl = [("COMP", comp, "varlik zincir basi OLMUS, ama BU (r1,r2) cifti onunla hic gorulmedi"),
        ("ENT-AYIRT", [x for x in ent if x[6] == "AYIRT"],
         "varlik HIC zincir basi olmamis + kisayol VAR ve YANLIS  <- ASIL SORU"),
        ("ENT-YOK", [x for x in ent if x[6] == "YOK"],
         "varlik HIC zincir basi olmamis + kisayol IMKANSIZ       <- KONTROL")]
for ad_, lst, ac in tabl:
    print(f"\n  {ad_}  ({len(lst)} soru)  {ac}")
    for e, r1, r2, b, a, ks, s in lst[:2]:
        bas(d2(e, r1, r2, a))
        print(f"          kopru {b}   |  KISAYOL '{e} {r2}' = {ks or 'YOK'}")

print("\n" + "=" * 76)
print("3) CEVABI NASIL PUANLIYORUZ")
print("=" * 76)
e, r1, r2, b, a, ks, s = [x for x in ent if x[6] == "AYIRT"][0]
print(f"  Soru : [S2] {e} {r1} {r2} ?")
print(f"  Model POZ 4'te TEK bir varlik token'i uretir. Dort ihtimal:")
print(f"     {a:22s} DOGRU    -> `ent` sayaci")
print(f"     {ks:22s} KISAYOL  -> `ent_shortcut` sayaci  ('{e} {r2}' cevabi)")
print(f"     {b:22s} KOPRU    -> ara cevabi yazmis, 2. adimi atlamis")
print(f"     {'(digerleri)':22s} BASKA")
print("  Birincil olcu = `ent` dogrulugu, GM / G orani.")
print("  Mekanizma olcusu = `ent_shortcut`, GM'de G'den DUSUK olmali.")
