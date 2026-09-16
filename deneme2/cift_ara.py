# -*- coding: utf-8 -*-
"""ARA DEGERLEME HIPOTEZINI SINAR.

HIPOTEZ (16-17 Eylul, olcumlerden turedi): model bilesim yapmiyor;
OZNE BASINA ILISKI-CIFTI TABLOSU ogrenip o tabloda ara degerleme
yapiyor. Egitimde 2-hop ornekleri oldugu icin bu, egitim hedefini
TAMAMEN karsiliyor ve bilesim devresine basinc dogmuyor.

KESKIN TAHMINI. `(e, r1, r2)` comp sorusunda basari, modelin
    (e, r1, *)  -- ayni ozne, ayni BIRINCI iliski, baska ikinci
    (e, *, r2)  -- ayni ozne, ayni IKINCI iliski, baska birinci
zincirlerini gorup gormedigine BAGLI olmali. Ikisini de gormusse
"satir ve sutun kesisimi" doldurulabilir; hicbirini gormemisse
doldurulamaz.

Bilesim olsaydi bu BAGLILIK OLMAZDI: kopruyu kuran model, o ozne icin
hangi ciftleri gordugune bakmaksizin cevabi bulurdu.

    python cift_ara.py <klasor> --aile model_00|model_b
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
    import pencere_b          # noqa: F401
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
print(f"model {ayar.ad}   pencere {os.path.basename(pen[0])} .. "
      f"{os.path.basename(pen[-1])}")
net = SINIF(ayar, v.vocab)
net.load_state_dict(P.agirlik_ortalamasi(pen))
net.eval()


@torch.no_grad()
def dogruluk(lst):
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


# --- egitimde hangi (ozne, r1) ve (ozne, r2) gorulmus -------------------
satir = collections.Counter()      # (e, r1) -> kac kez
sutun = collections.Counter()      # (e, r2) -> kac kez
for x in v.tr2:
    satir[(x[0], x[1])] += 1
    sutun[(x[0], x[2])] += 1

comp = list(v.comp)
ok = dogruluk(comp)
s_var = np.array([satir[(x[0], x[1])] > 0 for x in comp])
t_var = np.array([sutun[(x[0], x[2])] > 0 for x in comp])

print(f"\negitim 2-hop {len(v.tr2):,}   comp {len(comp):,}")
print()
print("=== comp DOGRULUGU: (e,r1) ve (e,r2) egitimde GORULDU MU ===")
print(f"  {'(e,r1)':>8}{'(e,r2)':>8}{'n':>8}{'DOGRU':>9}")
for sv in (True, False):
    for tv in (True, False):
        m = (s_var == sv) & (t_var == tv)
        if m.sum() == 0:
            continue
        print(f"  {'VAR' if sv else 'YOK':>8}{'VAR' if tv else 'YOK':>8}"
              f"{int(m.sum()):>8}{ok[m].mean():>9.4f}")
print()
print("  OKUMA: ikisi de VAR olan kova, ikisi de YOK olandan BELIRGIN")
print("  yuksekse ARA DEGERLEME hipotezi destekleniyor. Fark yoksa")
print("  model o ozne icin cifte BAGIMSIZ davraniyor -- bilesime yakin.")

# Sayi olarak da: kac FARKLI r1 / r2 ile gorulmus
n_r1 = collections.Counter()
n_r2 = collections.Counter()
for x in v.tr2:
    n_r1[x[0]] = n_r1[x[0]]
for e in {x[0] for x in v.tr2}:
    n_r1[e] = len({y[1] for y in v.tr2 if y[0] == e})
    n_r2[e] = len({y[2] for y in v.tr2 if y[0] == e})
c1 = np.array([n_r1.get(x[0], 0) for x in comp])
print()
print("=== comp DOGRULUGU: oznenin gordugu FARKLI r1 sayisina gore ===")
print(f"  {'farkli r1':>11}{'n':>8}{'DOGRU':>9}")
for lo, hi in [(0, 0), (1, 8), (9, 12), (13, 15), (16, 99)]:
    m = (c1 >= lo) & (c1 <= hi)
    if m.sum() == 0:
        continue
    ad = f"{lo}" if lo == hi else f"{lo}-{hi}"
    print(f"  {ad:>11}{int(m.sum()):>8}{ok[m].mean():>9.4f}")
