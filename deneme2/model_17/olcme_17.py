# -*- coding: utf-8 -*-
"""olcme_17 -- TEK OLCUT: zincir verildi, bilesik iliski DOGRU MU.

Uretim yok, uzunluk yok, kismi puan yok.  Cevap tek bir etiket:
    o, _ = m.dikkat(w)          onek -> tek cikti noktasi
    tahmin = en yakin E[c]
    dogru mu = (tahmin == hedef)

VE OLCU TEK SAYI DEGIL, k'YA GORE TABLO.  Sorunun tamami bu:
    k = 2,3     egitimde GORULDU        -> ezber
    k >= 4      HIC gorulmedi           -> bilesimsel genelleme
Tek ortalama ikisini karistirir ve egriyi gizler.

Sans seviyesi 1/18 = 0,056 (18 hedef etiketi).
"""
from __future__ import annotations

import torch


def etiketler(*kumeler) -> list:
    """Hedef olarak GECEN birimler -- cevap uzayi.  Veriden cikar."""
    return sorted(set(int(x) for O in kumeler for v in O.values()
                      for x in v[1]))


def tahmin(m, w, parca: int = 4096, maske=None, etiket=None):
    """w (n, T) -> (n,) tahmin edilen etiket.  maske: dolgu yuvalari.

    etiket verilirse okuma YALNIZ o birimler uzerinde.  Verilmezse butun
    sozluk -- C'de olculdu: sinav cevaplarinin %42,6'si etiket bile
    degildi ("necklace", "exactly"), sinav 0,0454 yerine 0,0829'du.
    A (sozluk 21) ve B2 (41) etkilenmiyor: %0,0 ve %0,8."""
    ET = None if etiket is None else torch.as_tensor(etiket,
                                                     device=m.E.device)
    cik = []
    with torch.no_grad():
        E = m.E if ET is None else m.E[ET]
        for i in range(0, w.shape[0], parca):
            mm = None if maske is None else maske[i:i + parca]
            o, _ = m.dikkat(w[i:i + parca], mm)
            j = torch.cdist(o, E).argmin(-1)
            cik.append(j if ET is None else ET[j])
    return torch.cat(cik)


def _ac(v, aygit):
    """Obek (w, h) ya da (w, h, maske) olabilir -- C'de dolgu var."""
    mk = v[2].to(aygit) if len(v) > 2 else None
    return v[0].to(aygit), v[1].to(aygit), mk


def oran(m, O, aygit="cuda", parca=4096, etiket=None) -> dict:
    """{k: (dogru, toplam)} -- obek obek."""
    d = {}
    for k, v in O.items():
        w, h, mk = _ac(v, aygit)
        t = tahmin(m, w, parca, mk, etiket)
        d[k] = (int((t == h).sum()), int(w.shape[0]))
    return d


def olcut(EG, DG, SI, aygit="cuda", en=None, etiket=None):
    """train_17'nin bekledigi bicim:  olcut(m, "eg"|"dg"|"si") -> oran.

    `en` verilirse her obekten en fazla o kadar ornek -- egitim
    sirasindaki olcum ucuzlasir, SON olcum tam veriyle yapilir."""
    kume = {"eg": EG, "dg": DG, "si": SI}

    def f(m, taraf, tam=False):
        O = kume[taraf]
        if en and not tam:
            O = {k: tuple(t[:en] for t in v) for k, v in O.items()}
        d = oran(m, O, aygit, etiket=etiket)
        dg = sum(a for a, _ in d.values())
        tp = sum(b for _, b in d.values())
        return dg / max(tp, 1)

    return f


def tablo(m, O, aygit="cuda", yaz=print, gorulen=(2, 3), etiket=None):
    """k'ya gore doguruluk tablosu -- HUKUM BURADAN OKUNUR."""
    d = oran(m, O, aygit, etiket=etiket)
    yaz(f"{'k':>4} {'dogru':>7} {'toplam':>7} {'oran':>7}   durum")
    yaz("-" * 46)
    for k in sorted(d):
        a, b = d[k]
        yaz(f"{k:>4} {a:>7} {b:>7} {a/b:>7.4f}   "
            + ("gorulen k" if k in gorulen else "GORULMEMIS"))
    yaz("-" * 46)
    ga = sum(a for k, (a, _) in d.items() if k in gorulen)
    gb = sum(b for k, (_, b) in d.items() if k in gorulen)
    ya = sum(a for k, (a, _) in d.items() if k not in gorulen)
    yb = sum(b for k, (_, b) in d.items() if k not in gorulen)
    if gb:
        yaz(f"{'GORULEN':>12} {ga:>7} {gb:>7} {ga/gb:>7.4f}")
    if yb:
        yaz(f"{'GORULMEMIS':>12} {ya:>7} {yb:>7} {ya/yb:>7.4f}")
    yaz(f"{'sans':>12} {'':>7} {'':>7} {1/18:>7.4f}")
    return d


def kirilim(m, O, ad, aygit="cuda", en=12):
    """Ornek ornek dokum -- sayi degil, MODELIN SECTIGI okunur."""
    cik = []
    for k in sorted(O):
        v = O[k]
        w, h = v[0], v[1]
        mk = None if len(v) < 3 else v[2][:en].to(aygit)
        t = tahmin(m, w[:en].to(aygit), maske=mk)
        for i in range(min(en, w.shape[0])):
            g = w[i] if len(v) < 3 else w[i][v[2][i]]
            zincir = " ".join(ad[int(x)] for x in g[:-1])
            cik.append((k, zincir, ad[int(t[i])], ad[int(h[i])],
                        int(t[i]) == int(h[i])))
    return cik
