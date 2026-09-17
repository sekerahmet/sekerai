# -*- coding: utf-8 -*-
"""YUVA KESITI -- `comp` dusuklugunun ne kadari BIRLESIM CEZASI?

SORU. Varlik UC YUVA (Ad, Soyad, <YOK>) ve `comp` UCUNUN DE dogru
olmasini istiyor. Yuva basina dogruluk p ise birlesik dogruluk ~p^3
olurdu. Olctugumuz comp 0.1303; eger p ~ 0.5 ise bu SASIRTICI DEGIL ve
"bilesim olmuyor" dedigimiz seyin buyuk kismi temsil sorunu degil,
KESISIM cezasi olur.

Literatur bizden farkli olcuyor: ENTREC (2402.16837) varligin YALNIZ
ILK JETONUNU okuyor. Yani sayilarimiz ayni seyi olcmuyor olabilir.

KESKIN TEST. Yuva hatalari BAGIMSIZ mi?
    ortak ~ p1*p2*p3   -> BAGIMSIZ: model kismen biliyor, kesisim cezasi GERCEK
    ortak ~ min(p_i)   -> BAGIMLI : model ya hepsini bilir ya hicbirini,
                                    kesisim cezasi YOK, teshis saglam

    python yuva_kesit.py <klasor> --aile model_00
"""
from __future__ import annotations

import argparse
import importlib
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
ap.add_argument("--son", type=int, default=None,
                help="pencerenin BITTIGI adim (esit butce kiyasi)")
a = ap.parse_args()

KOK = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(KOK, a.aile))
sys.path.insert(0, KOK)

# AILE ADINDAN TURETILIR, elle listelenmez. Onceki surumde sabit bir
# if-zinciri vardi (model_00/model_02) ve her yeni kolda "bilinmeyen aile"
# diye duruyordu -- 17 Eylul'de model_01 ve model_03'u olcemedi.
# Turetme GUVENLI: modul GERCEKTEN o klasorden mi geldi diye denetleniyor,
# yani yanlislikla baska bir kolun tabanini yuklemek imkansiz.
_AD = os.path.join(KOK, a.aile)
assert os.path.isdir(_AD), f"aile klasoru YOK: {_AD}"
_N = a.aile.split("_")[1]
M = importlib.import_module(f"taban_{_N}")
P = importlib.import_module(f"pencere_{_N}")
SINIF = importlib.import_module(a.aile).ModelSade
for _m in (M, P):
    assert os.path.dirname(os.path.abspath(_m.__file__)) == _AD, (
        f"{_m.__name__} {a.aile} DISINDAN geldi: {_m.__file__} -- "
        "sys.path kirli, olcum BASKA bir kolun tabaniyla yapilirdi")

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
if a.son is not None:
    # ESIT BUTCE KIYASI: model_00 80.000'e kadar kostu, model_01/03 20.000'de
    # durdu. Son pencereyi almak 72-80k ile 12-20k'yi kiyaslardi -- DORT KAT
    # butce farki. --son 20000 penceresi AYNI butceye getirir.
    _hedef = [y for y in snaps if y.endswith(f"{a.son:08d}.pt")]
    assert len(_hedef) == 1, f"{a.son} adiminda anlik goruntu YOK"
    snaps = snaps[:snaps.index(_hedef[0]) + 1]
pen = snaps[-a.genislik:]
net = SINIF(ayar, v.vocab)
net.load_state_dict(P.agirlik_ortalamasi(pen))
net.eval()
print(f"model {ayar.ad}   pencere {os.path.basename(pen[0])} .. "
      f"{os.path.basename(pen[-1])}   yuva {v.yuva}")

ARA = (v.yuva_ara if v.par is not None
       else [(v.ent_off, v.ent_off + v.n_ent)])


@torch.no_grad()
def yuva_dogru(lst, iki_hop=True):
    """(n, yuva) bool -- her ornek, her yuva icin DOGRU MU."""
    kod = M.kodla_2hop if iki_hop else M.kodla_1hop
    X, Pp, T = kod(v, lst)
    ok = np.zeros((len(X), len(ARA)), bool)
    for i in range(0, len(X), 256):
        xb = torch.from_numpy(X[i:i + 256])
        lg = net(xb).float()
        ar = torch.arange(xb.shape[0])
        pb = torch.from_numpy(Pp[i:i + 256])
        tb = torch.from_numpy(T[i:i + 256])
        for j, (lo, hi) in enumerate(ARA):
            z = lg[ar, pb[:, j], lo:hi]
            ok[i:i + xb.shape[0], j] = (z.argmax(-1) == tb[:, j] - lo).numpy()
    return ok


def rapor(ad, ok):
    n = len(ok)
    p = ok.mean(0)
    ortak = ok.all(1).mean()
    carp = float(np.prod(p))
    enk = float(p.min())
    print(f"  {ad:<9}{n:>7}", end="")
    for j in range(ok.shape[1]):
        print(f"{p[j]:>10.4f}", end="")
    print(f"{ortak:>10.4f}{carp:>10.4f}{enk:>9.4f}", end="")
    # ortak, carpim ile min arasinda NEREDE? 0 = bagimsiz, 1 = tam bagimli
    kon = 0.0 if enk - carp < 1e-9 else (ortak - carp) / (enk - carp)
    print(f"{kon:>9.3f}")
    return ortak, carp, enk, kon


rng = np.random.RandomState(0)


def ornekle(lst, n=3000):
    lst = list(lst)
    if len(lst) <= n:
        return lst
    return [lst[i] for i in rng.choice(len(lst), n, replace=False)]


print()
print("=== YUVA KESITI ===")
print(f"  {'bolme':<9}{'n':>7}" + "".join(f"{'yuva'+str(j+1):>10}"
                                          for j in range(v.yuva))
      + f"{'ORTAK':>10}{'carpim':>10}{'min':>9}{'baglilik':>9}")
sat = {}
for ad, lst in (("comp", v.comp), ("ent", v.ent), ("seen", v.tr2),
                ("ood", v.ood)):
    if not len(lst):
        continue
    sat[ad] = rapor(ad, yuva_dogru(ornekle(lst)))
_one = ornekle(v.one)
sat["one"] = rapor("one", yuva_dogru(_one, iki_hop=False))

print()
print("  ORTAK   = butun yuvalar dogru (bizim `comp` tanimimiz)")
print("  carpim  = p1*p2*p3 -- yuvalar BAGIMSIZ olsaydi beklenen")
print("  min     = en zayif yuva -- yuvalar TAM BAGIMLI olsaydi beklenen")
print("  baglilik= 0 -> BAGIMSIZ (kesisim cezasi GERCEK)")
print("            1 -> TAM BAGIMLI (kesisim cezasi YOK, teshis saglam)")
print()
c = sat.get("comp")
if c:
    print(f"  comp: ORTAK {c[0]:.4f}   ilk yuva tek basina {c[0]:.4f} degil -- "
          f"yukaridaki 'yuva1' sutununa bak.")
    print(f"        baglilik {c[3]:.3f}")
