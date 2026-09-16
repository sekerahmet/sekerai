# -*- coding: utf-8 -*-
"""KAHIN TESTI: Phi MUKEMMEL calissaydi bilesim olur muydu?

Onceki denemem (baglamda kopruyu YAZMAK) GECERSIZDI: [S2]'yi 9.
pozisyona koyunca `seen` de 0.3150 -> 0.0000 dustu, yani olculen sey
bilesim degil BICIM idi.

Bu test POZISYONLARA HIC DOKUNMUYOR. Dizi aynen normal sinav dizisi.
Tek degisiklik: birinci loop sonrasi, DiscoLoop'un enjekte ettigi
Phi(h) yerine GERCEK KOPRUNUN gommesi konuyor -- yalniz kopru
pozisyonlarinda (kopru_hedefi, yani r1/r2 yuvalari; model_b8 tam
oralari denetlemisti).

    normal   h = h + a(h) * RMSNorm(Phi(h))
    kahin    h = h + a(h) * RMSNorm(W[gercek_kopru])    <- kopru pozlarinda

Yani Phi'ye MUKEMMEL cevabi veriyoruz.

  KAHIN >> normal  ->  asagi akis DEVRESI VAR, eksik olan Phi'nin
                       kopruyu URETMESI
  KAHIN ~  normal  ->  devre YOK; kopru elden verilse bile olmuyor

Bu, makalenin kendi mudahalesinin ("an easy training-free realignment
intervention nearly closes the generalization gap") bizim rejimdeki
karsiligi. EGITIM YOK, yalniz ileri gecis.
"""
import os
import sys
import numpy as np
import torch

# YOL DEPODAN TURETILIR, elle YAZILMAZ. Sabit "C:/AI_NEW_MODEL/deneme2"
# yaziliydi: yerelde calisiyor, Colab'da `ModuleNotFoundError: pencere_a`
# ile DUSUYORDU (16 Eylul, model_b13 kahin testi). Betik nerede durursa
# dursun kendi ailesini bulmali.
R = os.path.dirname(os.path.abspath(__file__))
for p in (R + "/model_a", R + "/model_b", R):
    sys.path.insert(0, p)
import model_a as M          # noqa: E402
import pencere_a as P        # noqa: E402
from model_b import ModelB   # noqa: E402

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("klasor", help="<model>/t0 -- ya da IZOLE edilmis bir kopyasi")
_ap.add_argument("--genislik", type=int, default=5)
_ap.add_argument("--n", type=int, default=600)
_ap.add_argument("--zorla", action="store_true",
                 help="gate'i KOPRU pozlarinda 1.0'a zorla")
_a = _ap.parse_args()
KL = _a.klasor
ayar = P.ayar_oku(KL)
snap = P.anlik_goruntuler(KL)
pen = list(snap)[-_a.genislik:]
v = M.veri_kur(ayar, yaz=lambda *a: None)
L = M.olcme_listeleri(ayar, v)
net = ModelB(ayar, v.vocab)
net.load_state_dict(P.agirlik_ortalamasi([snap[x] for x in pen]))
net.eval()
Y = v.yuva
LO, HI = v.yuva_ara[0]
KPOZ, NK = M.kopru_hedefi(v)
print(f"=== {ayar.ad}  pencere {pen[0]}-{pen[-1]} ===")
print(f"   kopru pozisyonlari {KPOZ}  ({NK} jeton)   t_len {ayar.t_len}")


def ileri(X, kahin=None):
    """ModelB.forward'in AYNISI; `kahin` verilirse 1. loop sonrasi
    Phi'nin yerine gercek kopru gommesi konur (yalniz KPOZ'da)."""
    xb = torch.from_numpy(X)
    h = net.emb(xb) + net.pos(torch.arange(xb.shape[1]))[None]
    son = ayar.dongu - 1
    for k in range(ayar.dongu):
        for blk in net.bloklar:
            h = blk(h)
        if net.dar_acik and k < son:
            g = net._phi(h)
            if kahin is not None:
                for j, poz in enumerate(KPOZ):
                    g[:, poz] = net.emb.weight[kahin[:, j]]
            a = (torch.sigmoid((g * net.kapi_w).sum(-1, keepdim=True)
                               + net.kapi_b)
                 if ayar.dar_kapi else
                 torch.full((1, 1, 1), float(ayar.dar_alfa)))
            if _a.zorla and kahin is not None:
                a = a.expand(g.shape[0], g.shape[1], 1).clone()
                for poz in KPOZ:
                    a[:, poz] = 1.0
            h = h + a * net.dar_norm(g)
    return net.head(net.nf(h))


def dogru(lg, poz, hedef):
    t = (lg[:, poz:poz + Y, LO:HI].argmax(-1) + LO).numpy()
    return float((t == hedef).all(1).mean())


print("\n" + "=" * 74)
print("KAHIN: Phi'nin yerine GERCEK KOPRU gommesi (pozisyonlar AYNEN)")
print("=" * 74)
for bol in ("comp", "ent", "ood", "seen"):
    lst = (L.get(bol) or [])[:_a.n]
    if not lst:
        continue
    X, Pp, T = M.kodla_2hop(v, lst)
    p0 = int(Pp[0][0])
    kop = np.array([[int(t) for t in M._e(v, x[3])][:NK] for x in lst])
    # KONTROL: ILGISIZ bir varligin gommesi -- fark "kopru DOGRU oldugu
    # icin mi" yoksa "bir sey enjekte edildigi icin mi"?
    rs = np.random.RandomState(0)
    sah = np.array([[int(t) for t in M._e(v, int(rs.randint(v.n_ent)))][:NK]
                    for _ in lst])
    with torch.no_grad():
        d_n = dogru(ileri(X), p0, T)
        d_k = dogru(ileri(X, kop), p0, T)
        d_s = dogru(ileri(X, sah), p0, T)
    print(f"\n  {bol}  n={len(lst)}")
    print(f"    NORMAL   Phi(h) enjekte              {d_n:.4f}")
    print(f"    KAHIN    GERCEK kopru enjekte        {d_k:.4f}   "
          f"({d_k - d_n:+.4f})")
    print(f"    KONTROL  ILGISIZ varlik enjekte      {d_s:.4f}   "
          f"({d_s - d_n:+.4f})")
    h = ("DEVRE VAR -- eksik olan Phi'nin kopruyu URETMESI"
         if d_k > max(0.30, d_s + 0.10) else
         "DEVRE YOK -- kopru elden verilse BILE olmuyor"
         if d_k < d_s + 0.05 else "kismi / belirsiz")
    print(f"    -> {h}")


# --- MUDAHALE GERCEKTEN ETKILI MI --------------------------------------
# Bu kontrol OLMADAN sonuc okunmaz: mudahale hic etki etmiyorsa
# "devre yok" degil "kod bozuk" demektir. 16 Eylul'de tam bu soruldu.
print()
print("=" * 74)
print("MUDAHALE ETKI ETTI MI? (etmediyse KODDA HATA var, sonuc OKUNMAZ)")
print("=" * 74)
lst = (L.get("comp") or [])[:_a.n]
X, Pp, T = M.kodla_2hop(v, lst)
p0 = int(Pp[0][0])
kop = np.array([[int(t) for t in M._e(v, x[3])][:NK] for x in lst])
with torch.no_grad():
    lg_n, lg_k = ileri(X), ileri(X, kop)
t_n = (lg_n[:, p0:p0 + Y, LO:HI].argmax(-1) + LO).numpy()
t_k = (lg_k[:, p0:p0 + Y, LO:HI].argmax(-1) + LO).numpy()
_deg = int((t_n != t_k).any(1).sum())
print(f"   logit farki ortalama {(lg_k - lg_n).abs().mean():.6f}   "
      f"en buyuk {(lg_k - lg_n).abs().max():.4f}")
print(f"   TAHMINI DEGISEN ornek {_deg}/{len(lst)} ({_deg/len(lst):.1%})")
print("   -> " + ("MUDAHALE ETKILI, sonuc okunabilir" if _deg
                  else "!! HIC DEGISMEDI -- KODDA HATA, sonuc OKUNMAZ"))
