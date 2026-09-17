# -*- coding: utf-8 -*-
"""havuz_03 — model_03'in KENDI EGITIM HAVUZU DENETIMI. TEK BASINA DURUR.

Kullanici karari, 16 Eylul 2026: *"bunlarin hepsi model_03 folderi
altinda olmali. model_03 diger hicbir model ile ayni seyi kullanmamali.
Analiz icinde analiz_03 kullanalim mesela, digerleri icin de."*

`havuz_dok.py`nin KOPYASI (uretici: scratchpad/kur_analiz00.py). Iki fark:
motor `taban_03`, ve `--model` secenegi YOK -- bu arac yalniz model_03'i
tanir. Dokum `model_03/veri/model_03/` altina gider (depoya girmez).

    python havuz_03.py
"""
import sys
import collections
import numpy as np

import argparse
import importlib
import os

_D = os.path.dirname(os.path.abspath(__file__))
if _D not in sys.path:
    sys.path.insert(0, _D)
import taban_03 as M         # noqa: E402

# `--model` YOK: bu arac yalniz model_03'i tanir.
argparse.ArgumentParser().parse_args()
from ayar_03 import AYAR     # noqa: E402

v = M.veri_kur(AYAR, yaz=lambda *a: None)
X, P, T, kim, KP, KT = M.egitim_havuzu(AYAR, v, yaz=lambda *a: None)
print(f"=== havuz_dok  {AYAR.ad} ===")
print(f"havuz {X.shape}   t_len {v.t_len}   vocab {v.vocab}\n")

kotu = [0]


def bak(ad, tamam, ayrinti=""):
    print(f"  {'GECTI ' if tamam else '!! BOZUK'}  {ad}")
    if ayrinti:
        print(f"           {ayrinti}")
    if not tamam:
        kotu[0] += 1


# --- satir tipini AYIR --------------------------------------------------
n_q1 = (X == M.Q1).sum(1)
n_q2 = (X == M.Q2).sum(1)
m_bel = n_q1 == 2
m_1h = (n_q1 == 1) & (n_q2 == 0)
m_2h = n_q2 == 1
print(f"satir tipleri: 1-hop {int(m_1h.sum()):,}  2-hop {int(m_2h.sum()):,}  "
      f"BELGE {int(m_bel.sum()):,}  toplam {len(X):,}")
bak("her satir TAM BIR tipe giriyor",
    int((m_bel | m_1h | m_2h).sum()) == len(X),
    f"siniflandirilamayan: {len(X) - int((m_bel | m_1h | m_2h).sum())}")

print("\n1) JETON GECERLILIGI")
bak("butun jeton id'leri sozlukte", bool((X >= 0).all() and (X < v.vocab).all()),
    f"min {int(X.min())}  max {int(X.max())}  vocab {v.vocab}")
_ort = 0
for r in X:
    nz = np.nonzero(r)[0]
    if len(nz) and nz.max() - nz.min() + 1 != len(nz):
        _ort += 1
bak("PAD yalniz KUYRUKTA (arada yok)", _ort == 0, f"aradaki PAD: {_ort} satir")

print("\n2) SATIRLAR HAM GRAFLA TUTUYOR MU")
lo = v.ent_off
ters = {}
for e in range(v.n_ent):
    ters[tuple(int(v.par[e, j]) + lo for j in range(v.yuva))] = e


def coz(jet):
    return ters.get(tuple(int(t) for t in jet), -1)


# 1-hop: [S1] e(3) r ? a(3) EOS
i1 = np.nonzero(m_1h)[0]
kotu1 = 0
for i in i1[:4000]:
    r = X[i]
    e = coz(r[1:1 + v.yuva])
    rel = int(r[1 + v.yuva]) - M.REL_OFF
    a = coz(r[3 + v.yuva:3 + 2 * v.yuva])
    if e < 0 or a < 0 or int(v.facts[e, rel]) != a:
        kotu1 += 1
bak(f"1-hop satirlari graftaki olguyla AYNI (n={min(4000, len(i1))})",
    kotu1 == 0, f"tutmayan: {kotu1}")

# 2-hop: [S2] e(3) r1 r2 ? a(3) EOS
i2 = np.nonzero(m_2h)[0]
kotu2 = 0
for i in i2[:4000]:
    r = X[i]
    e = coz(r[1:1 + v.yuva])
    r1 = int(r[1 + v.yuva]) - M.REL_OFF
    r2 = int(r[2 + v.yuva]) - M.REL_OFF
    a = coz(r[4 + v.yuva:4 + 2 * v.yuva])
    b = int(v.facts[e, r1]) if e >= 0 else -1
    if e < 0 or b < 0 or a < 0 or int(v.facts[b, r2]) != a:
        kotu2 += 1
bak(f"2-hop cevabi = facts[facts[e,r1],r2] (n={min(4000, len(i2))})",
    kotu2 == 0, f"tutmayan: {kotu2}")

print("\n3) BELGE SATIRLARI -- model_b11'in YENI seyi")
ib = np.nonzero(m_bel)[0]
ko_a = ko_b = ko_z = 0
uzun = collections.Counter()
for i in ib[:6000]:
    r = X[i]
    e1 = coz(r[1:1 + v.yuva])
    ra = int(r[1 + v.yuva]) - M.REL_OFF
    b1 = coz(r[3 + v.yuva:3 + 2 * v.yuva])
    o = 4 + 2 * v.yuva                     # 2. olgunun [S1]'i
    e2 = coz(r[o + 1:o + 1 + v.yuva])
    rb = int(r[o + 1 + v.yuva]) - M.REL_OFF
    b2 = coz(r[o + 3 + v.yuva:o + 3 + 2 * v.yuva])
    if e1 < 0 or b1 < 0 or int(v.facts[e1, ra]) != b1:
        ko_a += 1
    if e2 < 0 or b2 < 0 or int(v.facts[e2, rb]) != b2:
        ko_b += 1
    if b1 != e2:
        ko_z += 1                          # ZINCIR KOPUK
    uzun[int((r != M.PAD).sum())] += 1
n = min(6000, len(ib))
bak(f"belge 1. olgusu graftaki olgu (n={n})", ko_a == 0, f"tutmayan {ko_a}")
bak(f"belge 2. olgusu graftaki olgu (n={n})", ko_b == 0, f"tutmayan {ko_b}")
bak(f"ZINCIR SURUYOR: 1. olgunun cevabi = 2. olgunun oznesi (n={n})",
    ko_z == 0, f"kopuk zincir {ko_z}")
bak("belge uzunlugu t_len'i asmiyor", (max(uzun) if uzun else 0) <= v.t_len,
    ("uzunluk dagilimi: " + "  ".join(f"{k}:{c}" for k, c in sorted(uzun.items())))
    if uzun else "BELGE YOK (belge_pay=0) -- bu bolum bu kolda GECERSIZ")

print("\n4) BELGEDEKI OLGULAR EGITIMDE ZATEN VAR MI (yeni olgu YOK mu)")
one_k = {(e, r) for e, r, _ in v.one}
eks_a = eks_b = 0
for i in ib[:6000]:
    r = X[i]
    e1 = coz(r[1:1 + v.yuva]); ra = int(r[1 + v.yuva]) - M.REL_OFF
    o = 4 + 2 * v.yuva
    e2 = coz(r[o + 1:o + 1 + v.yuva]); rb = int(r[o + 1 + v.yuva]) - M.REL_OFF
    eks_a += int((e1, ra) not in one_k)
    eks_b += int((e2, rb) not in one_k)
bak("belgedeki 1. olgu `one`da ZATEN var", eks_a == 0, f"eksik {eks_a}")
bak("belgedeki 2. olgu `one`da ZATEN var", eks_b == 0, f"eksik {eks_b}")

print("\n5) SIZINTI -- belge bir SINAV zincirini aciga vuruyor mu")
sinav = {(x[0], x[1], x[2]) for lst in
         (v.comp, v.ent, v.ent_yok, v.ent_arama, v.ood) for x in lst}
siz = 0
for i in ib[:6000]:
    r = X[i]
    e1 = coz(r[1:1 + v.yuva]); ra = int(r[1 + v.yuva]) - M.REL_OFF
    o = 4 + 2 * v.yuva
    rb = int(r[o + 1 + v.yuva]) - M.REL_OFF
    siz += int((e1, ra, rb) in sinav)
bak(f"hicbir belge bir SINAV uclusunu aciga vurmuyor (n={n})", siz == 0,
    f"sizan {siz}   sinav uclusu {len(sinav):,}")

print("\n6) CELISKI -- ayni soru, FARKLI cevap")
soz = collections.defaultdict(set)
for i in np.concatenate([i1[:6000], i2[:6000]]):
    soru = tuple(int(t) for t in X[i][:P[i][0]])
    soz[soru].add(tuple(int(t) for t in T[i]))
cel = {k: cv for k, cv in soz.items() if len(cv) > 1}
bak("ayni soruya IKI FARKLI cevap veren satir YOK", not cel,
    f"celiskili soru: {len(cel)}")

print("\n7) KAYIP HEDEFLERI")
bak("butun hedefler VARLIK blogunda",
    bool((T >= v.yuva_ara[0][0]).all() and (T < v.yuva_ara[0][1]).all()),
    f"T min {int(T.min())} max {int(T.max())}, blok {v.yuva_ara[0]}")
bak("P pozisyonlari next-token sozlesmesine uyuyor",
    bool(all((X[i, P[i] + 1] == T[i]).all()
             for i in np.random.RandomState(0).randint(0, len(X), 3000))))
bak("BELGE satirlarinda kopru hedefi -1 (maskeli)",
    bool((KT[m_bel][:, 0] < 0).all()))

print("\n8) TEKRARLAR")
_, say = np.unique(X[:20000], axis=0, return_counts=True)
bak("ayni satirin asiri tekrari YOK (ilk 20.000)", int(say.max()) <= 3,
    f"en cok tekrar {int(say.max())}  farkli satir {len(say):,}/20.000")

# --- 9) DAGILIM: belge EKLEMEK olgu SIKLIGINI de degistiriyor ----------
# Bu bir SAGLIK sorusu degil, bir CONFOUND sorusu: kol ES-KONUM olcmek
# icin kuruldu ama belgeler atomik olguyu TEKRAR TEKRAR tasiyor.
if len(ib):
    print("\n9) OLGU SIKLIGI -- belge IKINCI bir seyi de degistiriyor")
    say9 = collections.Counter((e, r) for e, r, _ in v.one)
    for i in ib:
        r = X[i]
        e1 = coz(r[1:1 + v.yuva]); ra = int(r[1 + v.yuva]) - M.REL_OFF
        o = 4 + 2 * v.yuva
        e2 = coz(r[o + 1:o + 1 + v.yuva])
        rb = int(r[o + 1 + v.yuva]) - M.REL_OFF
        say9[(e1, ra)] += 1
        say9[(e2, rb)] += 1
    d9 = sorted(say9.values())
    print(f"   bir olgu havuzda: en az {d9[0]}  ortanca {d9[len(d9)//2]}  "
          f"ortalama {sum(d9)/len(d9):.1f}  en cok {d9[-1]}")
    print("   (belge_pay=0 iken HEPSI 1'di)")
    _hic = sum(1 for e in range(v.n_ent) for r in range(v.n_rel)
               if v.facts[e, r] >= 0 and (e, r) not in say9)
    bak("havuzda HIC gecmeyen olgu YOK", _hic == 0, f"eksik {_hic}")
    print("   !! CONFOUND: belge yalniz ES-KONUM eklemiyor, atomik olguyu")
    print("      da COGALTIYOR. Onkayit model_b11.md 5b'de yazili.")
    print("      KONTROL: ayni tekrar, RASTGELE eslesmis olgular")
    print("      (zincirlenmemis) -> es-konumu TEKRARDAN ayirir.")

print(f"\n{'=' * 74}")
print(f"HAVUZ DENETIMI: {kotu[0]} BOZUK" if kotu[0] else
      "HAVUZ DENETIMI: HEPSI GECTI")
if kotu[0]:
    raise SystemExit(1)
