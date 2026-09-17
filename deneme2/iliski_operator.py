# -*- coding: utf-8 -*-
"""ILISKI DOGRUSAL BIR OPERATOR MU -- hic model EGITMEDEN olculur.

SORU. Onerilen mimari (paylasilan hop operatoru) su varsayima dayaniyor:

    h(b) ~ W_r h(a)       her (a --r--> b) olgusu icin

Dogruysa bilesim MATRIS CARPIMI olur (W_r2 W_r1) ve gorulmemis ucluye
kendiliginden genellenir. Yanlissa `A` dogrusal OLMAMALI (MLP gerekir).

OLCUM. Her iliski icin olgular EGITIM/SINAV diye bolunur, W_r yalniz
egitim yarisinda ridge ile kestirilir, R^2 SINAV yarisinda okunur.
Ayni bolmede UC taban:

    ortalama      h(b) ~ ortalama(h_b)        yapisiz taban
    KIMLIK        h(b) ~ h(a)                 !! EN ONEMLI TABAN
    oteleme       h(b) ~ h(a) + t_r           TransE tarzi
    operator      h(b) ~ W_r h(a)             ONERININ VARSAYIMI

!! KIMLIK TABANI NEDEN SART. Varliklarimiz UC jeton (Ad, Soyad, <YOK>)
ve ayni ailedeki kisilerin SOYADI ORTAK. "b'yi a'ya esitle" bu yuzden
tek basina yuksek R^2 verebilir -- ogrenilmis bir iliski yapisi
oldugu icin degil, girdinin bir parcasi cikitida AYNEN durdugu icin.
Ayni kusur 16 Eylul'de dogrusal sondada yakalanmisti: taban
max(en_sik, KOPYA) yapilmadan slot1 yanlislikla "BILGI VAR" diyordu.
Operator ancak KIMLIGI BELIRGIN GECERSE bir sey soyluyor.

!! ILERI GECIS ELLE KURULMAZ: `net.nf` uzerine forward hook (hiza.py'de
16 Eylul'de elle kurulan ileri gecis tam bu yuzden kirilmisti).

    python iliski_operator.py <klasor> --aile model_00
"""
from __future__ import annotations

import argparse
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
ap.add_argument("--lam", type=float, default=1.0, help="ridge duzenlilestirmesi")
a = ap.parse_args()

KOK = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(KOK, a.aile))
sys.path.insert(0, os.path.join(KOK, "model_a"))
sys.path.insert(0, KOK)

if a.aile == "model_00":
    import taban_00 as M
    from model_00 import ModelSade as SINIF
    import pencere_00 as P
elif a.aile == "model_01":
    import taban_01 as M
    from model_01 import ModelSade as SINIF
    import pencere_01 as P
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
assert snaps, f"anlik goruntu YOK: {a.klasor}/snap"
pen = snaps[-a.genislik:]
net = SINIF(ayar, v.vocab)
net.load_state_dict(P.agirlik_ortalamasi(pen))
net.eval()
print(f"model {ayar.ad}   pencere {os.path.basename(pen[0])} .. "
      f"{os.path.basename(pen[-1])}")

tut = {}
net.nf.register_forward_hook(lambda m, i, o: tut.__setitem__("h", o.detach()))


@torch.no_grad()
def varlik_temsili():
    """Her varlik TEK BASINA verilir; son jetonunun nf(h) vektoru."""
    n = v.n_ent
    X = M._bos(n, v)
    poz = np.zeros(n, np.int64)
    for i in range(n):
        jt = M._e(v, i)
        X[i, :len(jt)] = jt
        poz[i] = len(jt) - 1
    out = []
    for i in range(0, n, 256):
        net(torch.from_numpy(X[i:i + 256]))
        h = tut["h"]
        out.append(h[torch.arange(h.shape[0]), poz[i:i + 256]].clone())
    return torch.cat(out).float().numpy()


H = varlik_temsili()
H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9)
print(f"varlik temsili {H.shape}   (nf cikisi, L2 normlu)")


def r2(y, yhat):
    ss = ((y - yhat) ** 2).sum()
    st = ((y - y.mean(0)) ** 2).sum()
    return 1.0 - ss / st


print()
print("=== ILISKI DOGRUSAL OPERATOR MU -- SINAV YARISINDA R^2 ===")
_RAD = None
try:
    import importlib
    _VM = importlib.import_module(ayar.veri_ad)
    _RAD = list(_VM.ILISKI)
except Exception as _ex:                                     # noqa: BLE001
    print(f"  (iliski adlari okunamadi: {_ex})")
print(f"  {'iliski':>14}{'n':>7}{'ortalama':>10}{'KIMLIK':>9}"
      f"{'oteleme':>9}{'operator':>10}{'fark':>8}")
sat = []
rng = np.random.RandomState(0)
for r in range(v.n_rel):
    ix = np.where(v.facts[:, r] >= 0)[0]
    if len(ix) < 60:
        continue
    jx = v.facts[ix, r]
    p = rng.permutation(len(ix))
    k = int(len(ix) * 0.7)
    tr, te = p[:k], p[k:]
    A_tr, B_tr = H[ix[tr]], H[jx[tr]]
    A_te, B_te = H[ix[te]], H[jx[te]]

    ort = r2(B_te, np.repeat(B_tr.mean(0, keepdims=True), len(B_te), 0))
    kim = r2(B_te, A_te)                      # !! b = a  -- KOPYA TABANI
    t_r = (B_tr - A_tr).mean(0, keepdims=True)
    ote = r2(B_te, A_te + t_r)
    G = A_tr.T @ A_tr + a.lam * np.eye(A_tr.shape[1])
    W = np.linalg.solve(G, A_tr.T @ B_tr)
    opr = r2(B_te, A_te @ W)

    sat.append((ort, kim, ote, opr))
    _ad = _RAD[r] if _RAD and r < len(_RAD) else str(r)
    print(f"  {_ad:>14}{len(ix):>7}{ort:>10.4f}{kim:>9.4f}"
          f"{ote:>9.4f}{opr:>10.4f}{opr-max(kim,ote,ort):>8.4f}")

s = np.array(sat)
print("  " + "-" * 58)
print(f"  {'ORTANCA':>14}{'':>7}{np.median(s[:,0]):>10.4f}"
      f"{np.median(s[:,1]):>9.4f}{np.median(s[:,2]):>9.4f}"
      f"{np.median(s[:,3]):>10.4f}")
# !! KIYAS `ortalama` ILE YAPILIR, kimlik/oteleme ILE DEGIL. Onlar
# capraz-tip iliskilerde -36'ya kadar dusuyor; "operator onlari gecti"
# demek ANLAMSIZ olurdu (0, -36'yi zaten geciyor). Yapisiz taban ~0'dir
# ve asil soru operator'un ONU gecip gecmedigi.
_yapi = int((s[:, 3] > s[:, 0] + 0.05).sum())
print(f"  operator YAPISIZ TABANI (ortalama) 0.05'ten fazla GECTIGI "
      f"iliski: {_yapi}/{len(s)}")
print(f"  ve bunlarin kacinda KIMLIGI de geciyor: "
      f"{int(((s[:, 3] > s[:, 0] + 0.05) & (s[:, 3] > s[:, 1] + 0.05)).sum())}")
print()
print("  OKUMA: `fark` sutunu = operator - EN IYI taban. POZITIF ve")
print("  belirgin olmali. KIMLIK yuksek cikan iliskilerde operator'un")
print("  yuksekligi BIR SEY SOYLEMEZ: varlik uc jeton ve ayni ailenin")
print("  SOYADI ORTAK, yani girdinin bir parcasi cikitida AYNEN duruyor.")
print("  `fark` ~0 ise iliski dogrusal operator olarak TEMSIL EDILMIYOR")
print("  ve onerilen `A` DOGRUSAL OLMAMALI (MLP gerekir).")
