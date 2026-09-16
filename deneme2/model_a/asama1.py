# -*- coding: utf-8 -*-
"""asama1 — DiscoLoop'un KENDI teshisi: ilk donguden sonra KOPRU var mi?

    python asama1.py <klasor> [--genislik 5] [--tara] [--cikti out.json]

Makale (arXiv 2607.00341, 3.2), BIREBIR:

  "the key diagnostic is the intermediate Stage-1 accuracy, obtained by
   reading the next-token prediction off the post-first-loop hidden
   states H(1) via the LM head W, i.e., applying f_theta only once
   (K = 1) at inference time."

Yeni egitim GEREKTIRMEZ, CPU'da kosar -- GPU'daki bir kosuyu bozmaz.

--------------------------------------------------------------------------
NE AYIRIR

    kopru COZULEBILIYOR ama bilesim YOK  -> hizalama/kapasite sorunu
    kopru COZULEMIYOR                    -> kopru zaten TASINMIYOR;
                                            Phi'nin enjekte edecegi
                                            temiz gomme YOK

Ikincisi olcursek `dar_alfa`yi ya da `d`yi buyutmek bir sey degistirmez:
Phi olmayan bir seyi enjekte edemez.

--------------------------------------------------------------------------
POZISYON SECILMEZ, TARANIR

Makalede dizi dort jeton ve kopru "r1'in pozisyonundan" okunuyor. Bizim
dizimiz daha uzun ve cok-jetonlu kodlamada koprunun KAC pozisyona
yayildigi belirsiz. Pozisyonu ELLE secmek olcumu uydurur: ilk denemede
r1'den baslayan pencereyi sectim ve `model_b1`de 0.52 okudum -- oysa
gercek yer poz 3'tu ve orada 1.0000 cikiyordu.

Bu yuzden `--tara` BUTUN pozisyonlari basar; hukum "EN IYI pozisyon"
uzerinden verilir. Kopru hicbir pozisyonda cozulemiyorsa hukum saglamdir.

    [S2] e1 e2 e3 r1 r2 ?  a1 a2 a3 EOS      (yuva=3)
    [S2] e        r1 r2 ?  a        EOS      (yuva=1)

DIKKAT -- dolgu yuvasi aldatir: cok-jetonlu kodlamada son yuva cogunlukla
`<YOK>`tur, yani "kopru j2 %94" bilgi DEGIL, dolgunun sikligidir. Hukum
AYIRT EDICI yuvaya (yuva 0) gore verilir; tablo hepsini basar ki bu
tuzak gorunur kalsin.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

# `asama1_b` bunu doldurur; None -> model_a.Model. `pencere_a` ve
# `tani_a`daki kancanin AYNISI.
MODEL_SINIFI = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_a as M                                           # noqa: E402
import pencere_a as P                                         # noqa: E402


def _tek_dongu(ayar, vocab, sd):
    """K=1 model. AGIRLIK AYNI -- dongulu transformerda bloklar
    PAYLASIMLI, yani "f_theta'yi bir kez uygulamak" tam olarak budur.
    Yeni parametre YOK, yeniden egitim YOK."""
    a1 = ayar.degistir(dongu=1)
    net = (MODEL_SINIFI or M.Model)(a1, vocab).to(M.DEV)
    net.load_state_dict(sd)
    net.eval()
    return net


def _jetonlar(v, e):
    return [int(t) for t in M._e(v, e)]


@torch.no_grad()
def olc(net1, net, v, lst, bs=256):
    """Her pozisyon icin: yuva-kisitli argmax koprunun j. jetonu mu?

    Doner: (poz, yuva) sekilli dogruluk matrisi + asama-2 cevap dogrulugu.
    """
    X, Pz, T = M.kodla_2hop(v, lst)
    kopru = np.array([_jetonlar(v, x[3]) for x in lst])
    ARA = v.yuva_ara[:Pz.shape[1]]
    n, tlen = X.shape
    tut = np.zeros((tlen, len(ARA)))
    as2 = np.zeros(n, bool)
    for i in range(0, n, bs):
        xb = torch.from_numpy(X[i:i + bs]).to(M.DEV)
        lg1 = net1(xb).float()
        lg2 = net(xb).float()
        kb = torch.from_numpy(kopru[i:i + bs]).to(M.DEV)
        for t in range(tlen):
            for j, (lo, hi) in enumerate(ARA):
                z = lg1[:, t, lo:hi]
                tut[t, j] += (z.argmax(-1) == kb[:, j] - lo).sum().item()
        d = None
        ar = torch.arange(xb.shape[0], device=M.DEV)
        tb = torch.from_numpy(T[i:i + bs]).to(M.DEV)
        pb = torch.from_numpy(Pz[i:i + bs]).to(M.DEV)
        for j, (lo, hi) in enumerate(ARA):
            z = lg2[ar, pb[:, j], lo:hi]
            e = z.argmax(-1) == tb[:, j] - lo
            d = e if d is None else (d & e)
        as2[i:i + xb.shape[0]] = d.cpu().numpy()
    return tut / n, as2.mean()


def bas(ad, mat, as2, v, yaz=print):
    """EN IYI pozisyon hukum verir; butun tablo `--tara` ile basilir."""
    eniyi = int(mat[:, 0].argmax())
    yaz(f"\n  {ad}")
    yaz(f"    ASAMA-1 kopru YUVA 0 (ayirt edici):  "
        f"en iyi poz {eniyi} -> {mat[eniyi, 0]:.4f}")
    if v.yuva > 1:
        yaz(f"    (ayni pozda diger yuvalar: "
            + "  ".join(f"j{j}={mat[eniyi, j]:.4f}"
                        for j in range(1, mat.shape[1]))
            + "   <- dolgu yuvasi ALDATIR)")
    yaz(f"    ASAMA-2 cevap dogrulugu:             {as2:.4f}")
    return dict(en_iyi_poz=eniyi, asama1_yuva0=float(mat[eniyi, 0]),
                asama1_hepsi_poz=[[float(x) for x in r] for r in mat],
                asama2=float(as2))


def tara_bas(mat, X0, v, yaz=print):
    yaz(f"    {'poz':>4} {'girdi':<8}"
        + "".join(f"{'kopru j' + str(j):>11}" for j in range(mat.shape[1])))
    for t in range(mat.shape[0]):
        g = int(X0[t])
        ad = ("[S2]" if g == M.Q2 else "[S1]" if g == M.Q1 else
              "?" if g == M.QM else "<SON>" if g == M.EOS else
              "<PAD>" if g == M.PAD else
              "@iliski" if M.SPECIAL <= g < M.SPECIAL + v.n_rel else "jeton")
        yaz(f"    {t:>4} {ad:<8}"
            + "".join(f"{mat[t, j]:>11.4f}" for j in range(mat.shape[1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("klasor")
    ap.add_argument("--genislik", type=int, default=5)
    ap.add_argument("--tara", action="store_true",
                    help="BUTUN pozisyonlarin tablosunu bas")
    ap.add_argument("--bolme", default="ood,ent,comp,seen")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--cikti", default=None)
    a = ap.parse_args()

    ayar = P.ayar_oku(a.klasor)
    kunye = P.kunye_oku(a.klasor)
    snap = P.anlik_goruntuler(a.klasor)
    adimlar = list(snap)
    g = min(a.genislik, len(adimlar))
    pen = adimlar[-g:]
    print(f"=== asama1  {ayar.ad} tohum {ayar.tohum} ===   cihaz {M.DEV}")
    P.kunye_bas(kunye)
    print(f"  pencere {pen[0]}-{pen[-1]} ({g} anlik goruntu)")

    v = M.veri_kur(ayar, yaz=lambda *x: None)
    L = M.olcme_listeleri(ayar, v)
    iz = M.olcme_izi(L)
    _k = (kunye or {}).get("olcme_izi")
    if _k and _k != iz:
        raise SystemExit(f"!! OLCME SETI DEGISMIS: egitim {_k}, asama1 {iz}")
    print(f"  parmak izi {iz}   yuva {v.yuva}   vocab {v.vocab}")

    sd = P.agirlik_ortalamasi([snap[x] for x in pen])
    net = (MODEL_SINIFI or M.Model)(ayar, v.vocab).to(M.DEV)
    net.load_state_dict(sd)
    net.eval()
    net1 = _tek_dongu(ayar, v.vocab, sd)

    sonuc = {}
    for bol in a.bolme.split(","):
        lst = (L.get(bol) or [])[:a.n]
        if not lst:
            continue
        mat, as2 = olc(net1, net, v, lst)
        sonuc[bol] = bas(f"{bol}  n={len(lst)}", mat, as2, v)
        if a.tara:
            tara_bas(mat, M.kodla_2hop(v, lst[:1])[0][0], v)

    yol = a.cikti or os.path.join(
        a.klasor, f"asama1_{ayar.ad}_t{ayar.tohum}_g{g}.json")
    M._yaz_json(yol, dict(ad=ayar.ad, tohum=ayar.tohum, pencere=pen,
                          genislik=g, parmak_izi=iz, yuva=v.yuva,
                          sonuc=sonuc))
    print(f"\n-> {yol}")


if __name__ == "__main__":
    main()
