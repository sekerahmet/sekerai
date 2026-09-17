# -*- coding: utf-8 -*-
"""havuz_06 — model_05'in KENDI EGITIM HAVUZU DENETIMI. TEK BASINA DURUR.

Kullanici karari, 16 Eylul 2026: *"bunlarin hepsi model_05 folderi
altinda olmali. model_05 diger hicbir model ile ayni seyi kullanmamali.
Analiz icinde analiz_05 kullanalim mesela, digerleri icin de."*

`havuz_dok.py`nin KOPYASI (uretici: scratchpad/kur_analiz00.py). Iki fark:
motor `taban_05`, ve `--model` secenegi YOK -- bu arac yalniz model_05'i
tanir. Dokum `model_05/veri/model_05/` altina gider (depoya girmez).

    python havuz_05.py
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
import taban_06 as M         # noqa: E402

# `--model` YOK: bu arac yalniz model_05'i tanir.
argparse.ArgumentParser().parse_args()
from ayar_06 import AYAR     # noqa: E402

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
# !! YENIDEN YAZILDI, 17 Eylul. Once [S1]/[S2] isaretcilerini sayiyordu:
#       n_q1 = (X == M.Q1).sum(1)
# ek_kip="tr2"de o isaretciler YOK (silindi) -- ve daha oncesinde "tr"
# kipinde de dizilere KONMUYORDU, yani bu arac model_03/model_04'te de
# "2-hop 0 satir" diye BOS tablo basiyordu ve iki denetimi HEP dusuyordu.
# Kimse fark etmemisti cunku arac hukumde kullanilmiyor.
#
# Dogru olcut: satirdaki ILISKI JETONU SAYISI.
#   0 iliski -> KIMLIK   ("Ibrahim Yilmaz kimdir? Ibrahim Yilmaz'dir.")
#   1 iliski -> 1-HOP    ("...'in kardesi kim? ...'dir.")
#   2 iliski -> 2-HOP    ("...'in babasinin memleketi neresi? ...'dir.")
# !! CEVAP CUMLESINE gore sayilir, butun satira gore DEGIL.
# 17 Eylul'den beri cevap soruyu YENIDEN YAZIYOR, yani iliski jetonu
# satirda IKI KEZ geciyor: bir soruda, bir cevapta. Butun satiri
# saymak 1-hop'u 2-hop sanmaya yol aciyordu (olculdu: 12.508 satir
# yanlis sinifa dustu).
def _cevap_kismi(r):
    q = np.nonzero(r == M.QM)[0]
    return r[int(q[0]) + 1:] if len(q) else r
n_rel = np.array([int(((c >= M.REL_OFF) & (c < M.REL_OFF + v.n_rel)).sum())
                  for c in (_cevap_kismi(r) for r in X)])
m_kim = n_rel == 0
m_1h = n_rel == 1
m_2h = n_rel == 2
print(f"satir tipleri: KIMLIK {int(m_kim.sum()):,}  1-hop {int(m_1h.sum()):,}"
      f"  2-hop {int(m_2h.sum()):,}  toplam {len(X):,}")
bak("her satir TAM BIR tipe giriyor",
    int((m_kim | m_1h | m_2h).sum()) == len(X),
    f"siniflandirilamayan: {len(X) - int((m_kim | m_1h | m_2h).sum())}")

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
# !! ADLAR DEGISKEN UZUNLUKTA (<YOK> silindi). Cozumleme POZISYONA gore
# YAPILAMAZ; jeton dizisinden KESME ISARETINE kadar okunur.
_KES = v.ek0
_ters = {tuple(M.kelimeler(v, e)): e for e in range(v.n_ent)}
_soz = {}
for j in range(v.yuva):
    for i, w in enumerate(v.par_ad[j]):
        _soz[v.yuva_ara[j][0] + i] = w


def oku_ad(r, i):
    """r[i]'den baslayarak KESME ISARETINE kadar ad oku.
    Doner: (varlik id ya da -1, kesmeden SONRAKI konum)."""
    kel = []
    while i < len(r) and int(r[i]) in _soz:
        kel.append(_soz[int(r[i])])
        i += 1
    return _ters.get(tuple(kel), -1), i


def iliskiler(r):
    """CEVAP CUMLESINDEKI iliskiler. Soru kismindakiler AYNISININ
    tekrari -- ikisini birden saymak sayiyi IKIYE katliyordu."""
    c = _cevap_kismi(r)
    return [int(t) - M.REL_OFF for t in c
            if M.REL_OFF <= int(t) < M.REL_OFF + v.n_rel]


def ozne(r):
    """CEVAP CUMLESININ oznesi -- cumle ozneyle basliyor."""
    return oku_ad(_cevap_kismi(r), 0)[0]


def cevap(r):
    """Cevap, cumlenin SONUNDAKI addir: ... ad ' DIR .

    !! SORU ISARETINE gore aranmaz -- BILDIRIM satirinda `?` YOK
    (17 Eylul: cevap artik soruyu yeniden yazip oyle cevapliyor, ve
    bildirim bicimi tek basina duruyor)."""
    i = int(np.nonzero(r == _KES)[0][-1])      # SON kesme isareti
    j = i - 1
    while j >= 0 and int(r[j]) in _soz:
        j -= 1
    return oku_ad(r, j + 1)[0]


i1 = np.nonzero(m_1h)[0]
kotu1 = 0
for i in i1[:4000]:
    r = X[i]
    rel, e, a = iliskiler(r), ozne(r), cevap(r)
    if e < 0 or a < 0 or len(rel) != 1 or int(v.facts[e, rel[0]]) != a:
        kotu1 += 1
bak(f"1-hop satirlari graftaki olguyla AYNI (n={min(4000, len(i1))})",
    kotu1 == 0, f"tutmayan: {kotu1}")

i2 = np.nonzero(m_2h)[0]
kotu2 = 0
for i in i2[:4000]:
    r = X[i]
    rel, e, a = iliskiler(r), ozne(r), cevap(r)
    b = int(v.facts[e, rel[0]]) if e >= 0 and len(rel) == 2 else -1
    if e < 0 or b < 0 or a < 0 or int(v.facts[b, rel[1]]) != a:
        kotu2 += 1
bak(f"2-hop cevabi = facts[facts[e,r1],r2] (n={min(4000, len(i2))})",
    kotu2 == 0, f"tutmayan: {kotu2}")

print("\n3) KIMLIK SATIRLARI -- ozdeslik eslemesi (e -> e)")
ik = np.nonzero(m_kim)[0]
kotu_k = 0
for i in ik[:4000]:
    r = X[i]
    e, a = ozne(r), cevap(r)
    if e < 0 or e != a:
        kotu_k += 1
bak(f"kimlik satirinda cevap = SORU (n={min(4000, len(ik))})",
    kotu_k == 0, f"tutmayan: {kotu_k}")
bak("kimlik satirinda ILISKI jetonu YOK", True, "tanim geregi (n_rel == 0)")

# !! 4) ve 5) BOLUMLERI (BELGE) KALDIRILDI: `belge_pay` ve
# `kodla_belge` 17 Eylul silindi -- belge satiri ek isaretleyici
# tasimiyordu, yani ek_kip ile ZATEN birlikte kurulamiyordu.
print("\n6) CELISKI -- ayni soru, FARKLI cevap")
soz = collections.defaultdict(set)
for i in np.concatenate([i1[:6000], i2[:6000]]):
    soru = tuple(int(t) for t in X[i][:P[i][0]])
    soz[soru].add(tuple(int(t) for t in T[i]))
cel = {k: cv for k, cv in soz.items() if len(cv) > 1}
bak("ayni soruya IKI FARKLI cevap veren satir YOK", not cel,
    f"celiskili soru: {len(cel)}")

print("\n7) KAYIP HEDEFLERI")
# !! HEDEFLER ARTIK MASKELI OLABILIR (-1) ve son hedef KESME
# ISARETI. Adlar degisken uzunlukta (<YOK> silindi, 17 Eylul):
#   "Kocaeli'dir."      -> [Kocaeli, ', -1, -1]
#   "Ozlem Yilmaz'dir." -> [Ozlem, Yilmaz, ', -1]
bak("maskesiz hedefler CEVAP blogunda (varlik kelimesi + kesme)",
    bool(((T < 0) | ((T >= v.cevap_ara[0]) & (T < v.cevap_ara[1]))).all()),
    f"T min {int(T[T >= 0].min())} max {int(T.max())}, blok {v.cevap_ara}")
bak("her cevabin SON maskesiz hedefi KESME ISARETI",
    bool(all(int(t[int((t >= 0).sum()) - 1]) == v.ek0 for t in T[:5000])),
    f"kesme jetonu {v.ek0}")
bak("ILK hedef HIC maskeli degil (her adin en az bir kelimesi var)",
    bool((T[:, 0] >= 0).all()))
bak("P pozisyonlari next-token sozlesmesine uyuyor (maskesizlerde)",
    bool(all(((X[i, P[i] + 1] == T[i]) | (T[i] < 0)).all()
             for i in np.random.RandomState(0).randint(0, len(X), 3000))))
print("\n8) TEKRARLAR")
_, say = np.unique(X[:20000], axis=0, return_counts=True)
bak("ayni satirin asiri tekrari YOK (ilk 20.000)", int(say.max()) <= 3,
    f"en cok tekrar {int(say.max())}  farkli satir {len(say):,}/20.000")

# --- 9) DAGILIM: belge EKLEMEK olgu SIKLIGINI de degistiriyor ----------
# Bu bir SAGLIK sorusu degil, bir CONFOUND sorusu: kol ES-KONUM olcmek
# icin kuruldu ama belgeler atomik olguyu TEKRAR TEKRAR tasiyor.
print(f"\n{'=' * 74}")
print(f"HAVUZ DENETIMI: {kotu[0]} BOZUK" if kotu[0] else
      "HAVUZ DENETIMI: HEPSI GECTI")
if kotu[0]:
    raise SystemExit(1)
