# -*- coding: utf-8 -*-
"""comp DOGRULUGU, OZNENIN EGITIMDE KAC ZINCIRI GORULDUGUNE gore.

SORU. `model_00` comp'ta `model_b15`in 4,7 katinda, ama tip/menzil
kisiti IKISINDE DE AYNI (%99,7 vs %99,8). Degisen sey, sabit aday
kumesi icinde dogru cevabin SIRALAMASI (ortanca 37. -> 16.).

Peki bu ne? Iki okuma var:

  A) BILESIM       kopruyu kuruyor (zayif da olsa) -> ozne ne kadar
                   gorulmus olursa olsun kazanc BENZER olmali
  B) OZNE ISTATISTIGI  "bu ozneden r2 ile nereye gidilir" dagilimini
                   ezberliyor -> kazanc, oznenin EGITIMDE GORULEN
                   ZINCIR SAYISIYLA birlikte buyumeli

Olcum: comp ornekleri, oznenin egitim zinciri sayisina gore kovalara
bolunur; her kovada dogruluk ayri raporlanir. `ent` bolmesi (ozne HIC
zincir basi olmamis) dogal "0 kova" noktasidir.

    python ozne_kovasi.py <klasor> --aile model_00|model_b
"""
from __future__ import annotations

import argparse
import collections
import dataclasses as dc
import glob
import io
import json
import os
import sys

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("klasor")
ap.add_argument("--aile", required=True)
ap.add_argument("--genislik", type=int, default=5)
a = ap.parse_args()

KOK = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(KOK, a.aile))
sys.path.insert(0, os.path.join(KOK, "model_a"))
sys.path.insert(0, KOK)

if a.aile == "model_00":
    import taban_00 as M
    from model_00 import ModelSade as SINIF
    import pencere_00 as P
else:
    import model_a as M
    from model_b import ModelB as SINIF
    import pencere_b          # noqa: F401  (MODEL_SINIFI kancasi)
    import pencere_a as P

ham = json.load(io.open(glob.glob(os.path.join(a.klasor, "ayar_t*.json"))[0],
                        encoding="utf-8"))
alan = {f.name for f in dc.fields(M.Ayar)}
d = {k: v for k, v in ham.items() if k in alan}
for k in alan - set(d):
    d[k] = M.ESKI_VARSAYILAN[k]
d["mask_blok"] = tuple(d["mask_blok"])
d["betas"] = tuple(d["betas"])
ayar = M.Ayar(**d)
v = M.veri_kur(ayar, yaz=lambda *x: None)

snaps = sorted(glob.glob(os.path.join(a.klasor, "snap", "snap_*.pt")))
pen = snaps[-a.genislik:]
print(f"model {ayar.ad}   pencere {len(pen)} anlik goruntu: "
      f"{os.path.basename(pen[0])} .. {os.path.basename(pen[-1])}")
net = SINIF(ayar, v.vocab)
net.load_state_dict(P.agirlik_ortalamasi(pen))
net.eval()


@torch.no_grad()
def dogruluk(lst):
    """Her ornek icin BUTUN yuvalar dogru mu (bool dizi)."""
    X, Pp, T = M.kodla_2hop(v, lst)
    ARA = (v.yuva_ara if v.par is not None
           else [(v.ent_off, v.ent_off + v.n_ent)])
    ok = np.ones(len(X), bool)
    for i in range(0, len(X), 256):
        xb = torch.from_numpy(X[i:i + 256])
        lg = net(xb).float()
        ar = torch.arange(xb.shape[0])
        pb = torch.from_numpy(Pp[i:i + 256])
        tb = torch.from_numpy(T[i:i + 256])
        d_ = None
        for j, (lo, hi) in enumerate(ARA):
            z = lg[ar, pb[:, j], lo:hi]
            e_ = z.argmax(-1) == tb[:, j] - lo
            d_ = e_ if d_ is None else (d_ & e_)
        ok[i:i + xb.shape[0]] = d_.numpy()
    return ok


# --- oznenin EGITIMDE gorulen zincir sayisi ----------------------------
bas = collections.Counter(x[0] for x in v.tr2)
print(f"egitim 2-hop zinciri {len(v.tr2):,}   farkli ozne {len(bas):,}")

comp = list(v.comp)
ok = dogruluk(comp)
n_z = np.array([bas.get(x[0], 0) for x in comp])

KOVA = [(0, 0), (1, 40), (41, 70), (71, 90), (91, 100), (101, 10 ** 9)]
print()
print("=== comp DOGRULUGU, OZNENIN EGITIM ZINCIR SAYISINA GORE ===")
print(f"  {'kova':>12}{'n':>8}{'DOGRU':>9}")
for lo, hi in KOVA:
    m = (n_z >= lo) & (n_z <= hi)
    if m.sum() == 0:
        continue
    ad = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+")
    print(f"  {ad:>12}{int(m.sum()):>8}{ok[m].mean():>9.4f}")

# --- KONTROL: OZNENIN KARMASIKLIGI ------------------------------------
# !! Yukaridaki kova KARISTIRICI icerir: az zinciri olan ozne, ayni
# zamanda az CEVABI olan oznedir. Aday kumesi kucukse dogru bilmek
# zaten kolaydir. Asil eksen, oznenin ulastigi FARKLI CEVAP sayisi --
# arXiv 2509.24653'un `Complexity` tanimi (max_{e1} #{farkli e3}).
import importlib
VM = importlib.import_module(ayar.veri_ad)
G = VM.kur(ayar.veri_tohum)
O, tip, sema = G["olgu"], G["tip"], G["sema"]
R = list(VM.ILISKI)
E = [x for t in VM.TIPLER for x in G["ad"][t]]
karm = {}
for e_i, e_ad in enumerate(E):
    ce = set()
    for r1 in R:
        if tip[e_ad] not in sema[r1]:
            continue
        b = O.get((e_ad, r1))
        if b is None:
            continue
        for r2 in R:
            if r2 == r1 or tip[b] not in sema[r2]:
                continue
            c = O.get((b, r2))
            if c is not None and c != e_ad:
                ce.add(c)
    karm[e_i] = len(ce)
n_c = np.array([karm.get(x[0], 0) for x in comp])
print()
print("=== KONTROL: comp DOGRULUGU, OZNENIN KARMASIKLIGINA gore ===")
print("    (karmasiklik = ozneden ulasilan FARKLI cevap sayisi;")
print("     arXiv 2509.24653 ayni ekseni C=1..9 arasinda olcmustu)")
print(f"  {'karmasiklik':>14}{'n':>8}{'DOGRU':>9}{'ort.zincir':>12}")
for lo, hi in [(0, 60), (61, 85), (86, 100), (101, 108), (109, 10**9)]:
    m = (n_c >= lo) & (n_c <= hi)
    if m.sum() == 0:
        continue
    ad = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
    print(f"  {ad:>14}{int(m.sum()):>8}{ok[m].mean():>9.4f}{n_z[m].mean():>12.1f}")
print(f"  Pearson r (karmasiklik, dogruluk) = "
      f"{np.corrcoef(n_c, ok.astype(float))[0, 1]:+.4f}")

print()
r = np.corrcoef(n_z, ok.astype(float))[0, 1]
print(f"  Pearson r (zincir sayisi, dogruluk) = {r:+.4f}")

# `ent` = dogal SIFIR kovasi: ozne HIC zincir basi olmamis.
ent = list(v.ent)
oke = dogruluk(ent)
print(f"  ent  (ozne HIC zincir basi olmamis)  n={len(ent):,}"
      f"  DOGRU {oke.mean():.4f}")
print()
print("  OKUMA: dogruluk zincir sayisiyla BUYUYORSA mekanizma OZNE")
print("  ISTATISTIGI; DUZ ise bilesime daha yakin.")
