# -*- coding: utf-8 -*-
"""sor — modele ELLE soru sor, cevabini ve GERCEGINI yan yana gor.

Kullanici istegi, 16 Eylul 2026:
    "ben kendim soru sorup cevabini alacagim kendim kontrol edecegim
     bir sey istiyorum. Yani 'Ayse Yilmaz baba' diye yazdigimda
     'Mehmet Yilmaz' gibi bir cevap verdigini gormek istiyorum."

    python sor.py <klasor> [--genislik 5] [--soru "Ayse_Yilmaz baba"]
    python sor.py <klasor>                      # etkilesimli

!! BU BIR OLCU DEGIL. Elle sorulan sorular SECILMIS sorulardir; hukum
`pencere_b` ile verilir. Bu arac ANLAMAK icin, KANITLAMAK icin degil.
Her cikti bu uyariyi tasiyor.

Neden gercegi de basiyor: model "Mehmet_Yilmaz" dediginde bunun dogru
olup olmadigini bilemezsin -- graf sentetik, gercek dunyayla ilgisi yok.
Arac hem modelin cevabini hem GRAFTAKI dogru cevabi hem de sorunun
hangi BOLMEDEN oldugunu basar.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

_B = os.path.dirname(os.path.abspath(__file__))
_A = os.path.join(os.path.dirname(_B), "model_a")
for _p in (_A, _B, os.path.dirname(_B)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import model_a as M                                          # noqa: E402
import pencere_a as P                                        # noqa: E402
from model_b import ModelB                                   # noqa: E402


def _ad(v, e):
    """Varlik id -> okunabilir ad."""
    if v.par is None:
        return f"e{e}"
    a = v.par_ad[0][int(v.par[e, 0])]
    b = v.par_ad[1][int(v.par[e, 1])]
    return a if b == "<YOK>" else f"{a}_{b}"


def kur(klasor, genislik=5):
    ayar = P.ayar_oku(klasor)
    v = M.veri_kur(ayar, yaz=lambda *a: None)
    # ILISKI ADLARI `Veri`de YOK -- veri modulunden alinir, TEK KAYNAK.
    import importlib
    v.iliski = list(importlib.import_module(ayar.veri_ad).ILISKI)
    snap = P.anlik_goruntuler(klasor)             # {adim: yol}
    adimlar = list(snap)
    pen = adimlar[-min(genislik, len(adimlar)):]
    sd = P.agirlik_ortalamasi([snap[x] for x in pen])
    net = ModelB(ayar, v.vocab).to(M.DEV)
    net.load_state_dict(sd)
    net.eval()
    return ayar, v, net, pen


def _bolme(v, e, r1, r2):
    """Bu soru hangi bolmeden? Bolme listeleri TEK KAYNAK -- burada
    yeniden turetilmiyor, `veri_kur`un urettigi listelerde ARANIYOR."""
    if r2 is None:
        return "one" if any(x[0] == e and x[1] == r1 for x in v.one) else "?"
    for ad in ("ood", "ent_yok", "ent", "ent_kati", "comp", "ent_arama", "tr2"):
        for x in getattr(v, ad, []):
            if x[0] == e and x[1] == r1 and x[2] == r2:
                return {"tr2": "seen"}.get(ad, ad)
    return "?"


BOLME_NOT = {
    "one":      "atomik olgu -- egitimde GORULDU",
    "seen":     "2-hop zincir -- egitimde GORULDU (ezber)",
    "comp":     "ucluyu gormedi (varlik zincir basi OLDU)",
    "ent":      "varlik hic zincir basi olmadi -- ASIL SINAV",
    "ent_yok":  "ent + kisayol TIP OLARAK imkansiz",
    "ent_kati": "varlik hicbir zincirde yok",
    "ood":      "iki kenar da atomic_OOD (Wang'in tanimi)",
    "ent_arama": "arama havuzu -- HUKUMDE KULLANILMAZ",
    "?":        "graf'ta boyle bir zincir YOK (cevabi da yok)",
}


@torch.no_grad()
def sor(v, net, metin, yaz=print):
    p = metin.replace(",", " ").split()
    if len(p) not in (2, 3):
        yaz("  kullanim: <varlik> <iliski> [<iliski2>]")
        return
    ad2id = {_ad(v, e): e for e in range(v.n_ent)}
    rel2id = {r: i for i, r in enumerate(v.iliski)} if hasattr(v, "iliski") \
        else None
    e_ad, rels = p[0], p[1:]
    if e_ad not in ad2id:
        yak = [a for a in ad2id if a.lower().startswith(e_ad.lower()[:4])][:6]
        yaz(f"  '{e_ad}' grafta YOK." + (f"  Benzer: {yak}" if yak else ""))
        return
    e = ad2id[e_ad]
    try:
        rid = [rel2id[r] for r in rels]
    except (KeyError, TypeError):
        yaz(f"  iliski YOK: {rels}.  Gecerliler: {list(rel2id)}")
        return

    if len(rid) == 1:
        X, Pp, _ = M.kodla_1hop(v, [(e, rid[0], 0)])
        gercek = int(v.facts[e, rid[0]])
        kopru = None
    else:
        b = int(v.facts[e, rid[0]])
        gercek = int(v.facts[b, rid[1]]) if b >= 0 else -1
        kopru = b
        X, Pp, _ = M.kodla_2hop(v, [(e, rid[0], rid[1], 0, 0)])

    lg = net(torch.from_numpy(X).to(M.DEV)).float()
    ARA = [(v.p1_off, v.p1_off + v.n1), (v.p2_off, v.p2_off + v.n2)][:Pp.shape[1]]
    tah = []
    for j, (lo, hi) in enumerate(ARA):
        tah.append(int(lg[0, int(Pp[0, j]), lo:hi].argmax()) + lo)
    if v.par is None:
        m_ad = _ad(v, tah[0] - v.ent_off)
    else:
        a = v.par_ad[0][tah[0] - v.p1_off]
        b2 = v.par_ad[1][tah[1] - v.p2_off]
        m_ad = a if b2 == "<YOK>" else f"{a}_{b2}"

    g_ad = _ad(v, gercek) if gercek >= 0 else "(olgu YOK)"
    bol = _bolme(v, e, rid[0], rid[1] if len(rid) > 1 else None)
    yaz(f"  model   {m_ad}")
    yaz(f"  gercek  {g_ad}" + ("        DOGRU" if m_ad == g_ad and gercek >= 0
                               else "        YANLIS" if gercek >= 0 else ""))
    yaz(f"  bolme   {bol}   ({BOLME_NOT.get(bol, '')})")
    if kopru is not None and kopru >= 0:
        yaz(f"  kopru   {_ad(v, kopru)}   (model bunu YAZMADAN kullanmali)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("klasor")
    ap.add_argument("--genislik", type=int, default=5)
    ap.add_argument("--soru", action="append", default=None)
    a = ap.parse_args()

    ayar, v, net, pen = kur(a.klasor, a.genislik)
    print(f"=== sor  {ayar.ad} t{ayar.tohum} ===")
    print(f"  pencere {pen[0]}-{pen[-1]} ({len(pen)} anlik goruntu)")
    print(f"  varlik {v.n_ent}  iliski {v.n_rel}  sozluk {v.vocab}  "
          f"yuva {v.yuva}")
    print("  !! BU BIR OLCU DEGIL -- elle sorulan sorular SECILMIS "
          "sorulardir.")
    print("     Hukum `pencere_b` ile verilir. Bu arac ANLAMAK icin.")
    print()
    if a.soru:
        for q in a.soru:
            print(f"> {q}")
            sor(v, net, q)
            print()
        return 0
    print("  ornek:  Ayse_Yilmaz baba        |  Ayse_Yilmaz cocuk kardes")
    print("  cikis:  bos satir\n")
    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            break
        if not q:
            break
        sor(v, net, q)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
