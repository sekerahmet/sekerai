# -*- coding: utf-8 -*-
"""h(e1,r1) ile h(e2) HIZALANIYOR MU -- identity bridge makalesinin
mekanizma iddiasinin BIZIM verimizde sinanmasi.

arXiv 2509.24653: "the hidden states of (e1,r1) increasingly approach
those of e2 ... when the representations for one-hop data and the target
bridge collapse into the same subspace, the second hop can reliably
latch onto the correct features and composition succeeds."

OLCUM
  h_A = 1-hop sorusunun ("e1 ' <NIN> r1 <SI> ?") SORU SONU gizli durumu
  h_B = kopru varliginin TEK BASINA verildigi dizinin son gizli durumu
  ->  cos(h_A, h_B_DOGRU)   vs   cos(h_A, h_B_RASTGELE)

Hizalanma VARSA dogru koprunun benzerligi rastgeleden BELIRGIN yuksek
olmali. Ayrica SIRALAMA olcusu: butun varliklar arasinda dogru kopru
kacinci sirada (1.0 = hep birinci).

!! ILERI GECISI ELLE KURMUYORUZ. `net.nf` uzerine forward hook takilir;
boylece olcum modelin IC yapisini bilmek zorunda kalmaz ve ModelSade
(RoPE) ile ModelB (dongu+Phi) AYNI kodla olculur. 16 Eylul hakemliginde
elle kurulan ileri gecis tam bu yuzden kirilmisti.
"""
import argparse
import glob
import io
import json
import os
import sys

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("klasor")
ap.add_argument("--aile", required=True, help="model_00 | model_b")
ap.add_argument("--n", type=int, default=600)
ap.add_argument("--snap", default="", help="tek anlik goruntu; bos ise EN YENI")
a = ap.parse_args()

KOK = "C:/AI_NEW_MODEL/deneme2"
sys.path.insert(0, os.path.join(KOK, a.aile))
sys.path.insert(0, os.path.join(KOK, "model_a"))
sys.path.insert(0, KOK)

if a.aile == "model_00":
    import taban_00 as M
    from model_00 import ModelSade as SINIF
else:
    import model_a as M
    from model_b import ModelB as SINIF

# --- ayar + veri -------------------------------------------------------
import dataclasses as dc
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
yol = a.snap or snaps[-1]
sd = torch.load(yol, map_location="cpu")
net = SINIF(ayar, v.vocab)
net.load_state_dict({k: t.float() for k, t in sd.items()}, strict=False)
net.eval()
print(f"model  {ayar.ad}  anlik goruntu {os.path.basename(yol)}")

# --- nf cikisini HOOK ile yakala (elle ileri gecis YOK) ----------------
tut = {}
net.nf.register_forward_hook(lambda m, i, o: tut.__setitem__("h", o.detach()))


@torch.no_grad()
def gizli(X, poz):
    """X: (B, T) -> secilen pozisyonun nf(h) vektoru."""
    out = []
    for i in range(0, len(X), 256):
        xb = torch.from_numpy(X[i:i + 256])
        net(xb)
        h = tut["h"]
        out.append(h[torch.arange(h.shape[0]), poz[i:i + 256]].clone())
    return torch.cat(out)


# --- ornekler ----------------------------------------------------------
rng = np.random.RandomState(0)
lst = [v.comp[i] for i in rng.choice(len(v.comp), min(a.n, len(v.comp)),
                                     replace=False)]
# h_A: 1-hop sorusu (e1, r1) -- SORU SONU pozisyonu
X1, P1, _ = M.kodla_1hop(v, [(x[0], x[1], x[3]) for x in lst])
pozA = (P1[:, 0] - 1).astype(np.int64)          # ilk cevap yuvasinin ONCESI
hA = gizli(X1, pozA)

# h_B: koprunun KENDISI, TEK BASINA.
#
# !! ILK SURUM YANLISTI (16 Eylul, kendi hakemligim): h_B'yi kopruyu
# OZNE yapan bir 1-hop sorusundan almistim -- yani h_A ve h_B AYNI
# `' <NIN> r1 <SI>` sonekiyle bitiyordu ve benzarligin bir kismi
# PAYLASILAN JETONLARDAN geliyordu, kopruden degil. Capraz-disi taban
# farkli r1'ler icerdigi icin fark SISIYORDU.
#
# Simdi: yalniz varligin jetonlari, gerisi PAD. Ayrica asagida AYNI r1
# tabani da hesaplaniyor.
X2 = M._bos(len(lst), v)
pozB = np.zeros(len(lst), np.int64)
for i, x in enumerate(lst):
    jt = M._e(v, x[3])
    X2[i, :len(jt)] = jt
    pozB[i] = len(jt) - 1
hB = gizli(X2, pozB)
r1ler = np.array([x[1] for x in lst])

An = torch.nn.functional.normalize(hA, dim=-1)
Bn = torch.nn.functional.normalize(hB, dim=-1)
S = An @ Bn.T                                   # (N, N) cos benzerligi
dog = S.diag()
n = len(dog)
kar = S.clone()
kar.fill_diagonal_(float("-inf"))
sira = (S > dog[:, None]).sum(1).float()        # kacinci sirada
print()
print("=== h(e1,r1)  ile  h(e2)  HIZALANMASI ===")
print(f"  ornek {n}   (comp bolmesinden)")
print(f"  cos DOGRU kopru     {dog.mean():.4f}  +/- {dog.std():.4f}")
ras = kar[kar > -1e30].mean()
print(f"  cos RASTGELE kopru  {ras:.4f}")
print(f"  fark                {dog.mean() - ras:+.4f}")
# AYNI r1 TABANI: benzerlik kopruden mi, paylasilan iliski jetonundan mi?
m = torch.from_numpy(r1ler[:, None] == r1ler[None, :])
m.fill_diagonal_(False)
if m.any():
    print(f"  cos AYNI r1, BASKA kopru  {S[m].mean():.4f}"
          f"   -> kopru katkisi {dog.mean() - S[m].mean():+.4f}")
print()
print(f"  DOGRU kopru 1. sirada     %{100.0 * (sira == 0).float().mean():.1f}"
      f"   (sans %{100.0/n:.2f})")
print(f"  ilk %1'de                 %{100.0 * (sira < n * 0.01).float().mean():.1f}")
print(f"  ortanca sira              {sira.median():.0f}. / {n}")
