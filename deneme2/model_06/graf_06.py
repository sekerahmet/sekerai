# -*- coding: utf-8 -*-
"""graf_06 — model_05'in KENDI GRAF TUTARLILIK DENETIMI. TEK BASINA DURUR.

Kullanici karari, 16 Eylul 2026: *"bunlarin hepsi model_05 folderi
altinda olmali. model_05 diger hicbir model ile ayni seyi kullanmamali.
Analiz icinde analiz_05 kullanalim mesela, digerleri icin de."*

`graf_dok.py`nin KOPYASI (uretici: scratchpad/kur_analiz00.py). Iki fark:
motor `taban_05`, ve `--model` secenegi YOK -- bu arac yalniz model_05'i
tanir. Dokum `model_05/veri/model_05/` altina gider (depoya girmez).

    python graf_05.py
"""
import sys
import collections

import argparse
import importlib
import os

_D = os.path.dirname(os.path.abspath(__file__))
if _D not in sys.path:
    sys.path.insert(0, _D)
import taban_06 as M         # noqa: E402

# `--model` YOK: bu arac yalniz model_05'i tanir.
argparse.ArgumentParser().parse_args()
from ayar_06 import AYAR     # noqa: E402
VO = importlib.import_module(AYAR.veri_ad)

v = M.veri_kur(AYAR, yaz=lambda *a: None)
G = VO.kur(AYAR.veri_tohum)
IL = G["iliski"]
soz = lambda e: " ".join(M.kelimeler(v, e))
R_ = {n: i for i, n in enumerate(IL)}

print("ILISKILER:", ", ".join(IL))
print(f"varlik {v.n_ent}, iliski {v.n_rel}\n")

print("=" * 76)
print("1) TERS CIFT: X -r-> Y ise Y -r'-> X mi?")
print("=" * 76)
# VERIDEN geliyor, burada GOMULU DEGIL. Gomuluyken veri degisince bu
# arac KeyError verdi (17 Eylul) -- liste artik veri_05'in yaninda.
CIFT = list(VO.TERS_ADAY)
print(f"   {'X -r-> Y, Y -r2-> ?':<26}{'n':>7}{'== X':>8}{'oran':>9}   ornek")
for r, r2 in CIFT:
    if r not in R_ or r2 not in R_:
        continue
    i, j = R_[r], R_[r2]
    n = geri = 0
    orn = ""
    for e in range(v.n_ent):
        y = int(v.facts[e, i])
        if y < 0:
            continue
        z = int(v.facts[y, j])
        if z < 0:
            continue
        n += 1
        if z == e:
            geri += 1
        elif not orn:
            orn = f"{soz(e)} -{r}-> {soz(y)} -{r2}-> {soz(z)}"
    if n:
        print(f"   {r + ' / ' + r2:<26}{n:>7}{geri:>8}{geri/n:>9.1%}"
              f"   {'' if geri == n else orn}")

print()
print("=" * 76)
print("2) SIMETRI: X -r-> Y ise Y -r-> X mi?")
print("=" * 76)
print(f"   {'iliski':<14}{'n':>7}{'simetrik':>10}{'oran':>9}   ornek")
for r in IL:
    i = R_[r]
    n = sim = 0
    orn = ""
    for e in range(v.n_ent):
        y = int(v.facts[e, i])
        if y < 0:
            continue
        n += 1
        if int(v.facts[y, i]) == e:
            sim += 1
        elif not orn:
            orn = f"{soz(e)} -{r}-> {soz(y)} -{r}-> {soz(int(v.facts[y, i]))}"
    if n:
        isim = "SIMETRIK" if sim == n else ("kismen" if sim else "-")
        print(f"   {r:<14}{n:>7}{sim:>10}{sim/n:>9.1%}   {isim}"
              + (f"   {orn}" if 0 < sim < n else ""))

print()
print("=" * 76)
print("3) ISLEVSEL Mi: facts[e,r] TEK deger tutuyor -- KAC gercek hedef var")
print("=" * 76)
print("   Ham graf (X,r)->Y sozlugu TEK deger tutuyor. Ama gercekte")
print("   bir ebeveynin BIRDEN COK cocugu olabilir. `cocuk` KIMI gosteriyor?")
cok = collections.Counter()
for (x, r), y in G["olgu"].items():
    cok[(x, r)] += 1
print(f"   ham grafta ayni (X,r) icin birden cok kayit: "
      f"{sum(1 for c in cok.values() if c > 1)}  (sozluk zaten tek tutar)")
ters_say = collections.defaultdict(set)
for (x, r), y in G["olgu"].items():
    if r in VO.EBEVEYN:
        ters_say[y].add(x)          # y'nin COCUKLARI
d = collections.Counter(len(s) for s in ters_say.values())
print(f"   bir kisinin KAC cocugu var (anne/baba kayitlarindan turetilen):")
for k in sorted(d):
    print(f"      {k} cocuk: {d[k]} kisi")
birden = [y for y, s in ters_say.items() if len(s) > 1]
print(f"   -> {len(birden)} kisinin BIRDEN COK cocugu var; `cocuk` iliskisi")
print(f"      onlardan YALNIZ BIRINI gosteriyor. veri_05'te bu SECIM")
print(f"      KASITLI: baba OGLU, anne KIZI gosteriyor. Ikisi de AYNI")
print(f"      cocugu gosterseydi `annesi+cocugu` ile `babasi+cocugu`")
print(f"      birebir ayni olur, iki iliski birbirinden OKUNURDU.")

print()
print("=" * 76)
print("4) MANTIKSAL GEREKTIRME: (r1,r2) zinciri TEK bir r3 kenariyla ayni mi")
print("=" * 76)
# !! r3 ARTIK r2 VARSAYILMIYOR. veri_04'te iki gerektirme de
# (kardes,baba)->baba bicimindeydi, yani r3 == r2 idi ve dongu r2'yi
# r3 yerine kullaniyordu. veri_05'te (annesi,cocugu)->kardesi var:
# r3 != r2. Eski kod bu cifti "0.0%" diye basiyordu, yani GERCEK
# gerektirmeyi GORMUYORDU. r3 artik GEREKTIRIR'in DEGERINDEN geliyor.
for (r1, r2), r3 in VO.GEREKTIRIR.items():
    i, j, k3 = R_[r1], R_[r2], R_[r3]
    n = ayni = 0
    for e in range(v.n_ent):
        b = int(v.facts[e, i])
        if b < 0:
            continue
        a = int(v.facts[b, j])
        d2 = int(v.facts[e, k3])
        if a < 0 or d2 < 0:
            continue
        n += 1
        ayni += int(a == d2)
    print(f"   {r1}+{r2}: {ayni}/{n} = {ayni/max(1,n):.1%} zincirin cevabi "
          f"= facts[e,{r3}]  -> TEK KENARLA ulasilir")
print("   ^ bu bir VERI KUSURU DEGIL, grafin mantigi. Ama o zincirler")
print("     kopru KURMADAN cozulebilir. veri_dok bunu 'TEK KENAR acigi'")
print("     diye olcuyor ve %2 esigi koyuyor.")

print()
print("=" * 76)
print("5) TIP TUTARLILIGI: her iliski beklenen TIPE mi gidiyor")
print("=" * 76)
SEMA = G["sema"]
kotu = 0
for r in IL:
    i = R_[r]
    bek = SEMA.get(r, {})
    yanlis = []
    for e in range(v.n_ent):
        y = int(v.facts[e, i])
        if y < 0:
            continue
        kt, ht = v.tip_ad[v.tip[e]], v.tip_ad[v.tip[y]]
        if kt in bek and bek[kt] != ht:
            yanlis.append((e, y, kt, ht, bek[kt]))
    if yanlis:
        kotu += len(yanlis)
        print(f"   !! {r}: {len(yanlis)} yanlis tip, or. "
              f"{soz(yanlis[0][0])}({yanlis[0][2]}) -> "
              f"{soz(yanlis[0][1])}({yanlis[0][3]}), beklenen {yanlis[0][4]}")
print(f"   tip ihlali: {kotu}" + ("  -> HEPSI TUTUYOR" if not kotu else ""))
