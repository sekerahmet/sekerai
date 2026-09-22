# -*- coding: utf-8 -*-
"""olcme_17 -- SONRAKI JETON.  Etiket yok, tek sayi yok: SAYI + METIN.

KAYIP PERPLEXITY DEGIL.  Puan -||o - E||^2, yani olcegi kalibre degil;
baslangicta sozluk 50 iken kayip 20,8 cikiyor, ln(50)=3,91 degil.
Kayip egrisi izlenir ama BIR SEY SOYLEMEZ; hukum dogruluktan okunur.

IKI ALAN, birlikte okunur (CLAUDE.md):
    SAYI   sonraki jeton dogrulugu -- egitim ve DOGRULAMA ayri
    METIN  istemden URETIM, GOZLE.  Makalenin 44 degerlendirme istemi
           (Evaluation prompts.yaml) hazir duruyor.
"""
from __future__ import annotations

import torch


def _ac(X):
    """(W, M) ya da tek W -- ikisini de kabul et."""
    return X if isinstance(X, (tuple, list)) else (X, None)


def dogruluk(m, W, parca: int = 256, aygit=None) -> float:
    """Sonraki jeton dogrulugu.  W (n, T) -> oran.

    W CPU'da olabilir: PARCA PARCA tasinir.  Tam korpusta egitim
    penceresi 8,8 GB eder, tumunu GPU'ya koymak OOM demektir."""
    W, M = _ac(W)
    dg = tp = 0
    aygit = aygit or next(m.parameters()).device
    with torch.no_grad():
        for i in range(0, W.shape[0], parca):
            w = W[i:i + parca].to(aygit).long()
            mk = None if M is None else M[i:i + parca].to(aygit)
            t = m.dizi(w, mk)[:, :-1].argmax(-1)
            d = t == w[:, 1:]
            if mk is None:
                dg += int(d.sum()); tp += d.numel()
            else:                         # DOLGU SAYILMAZ
                a = mk[:, 1:]
                dg += int((d & a).sum()); tp += int(a.sum())
    return dg / max(tp, 1)


def olcut(EG, DG, aygit="cuda", en=2000):
    """train_17'nin bekledigi bicim:  olcut(m, "eg"|"dg") -> oran.

    `en` egitim SIRASINDA ornek sayisini kisar; son olcum TAM veriyle."""
    kume = {"eg": EG, "dg": DG}

    def f(m, taraf, tam=False):
        W, M = _ac(kume[taraf])
        if not tam:
            W, M = W[:en], (None if M is None else M[:en])
        return dogruluk(m, (W, M), aygit=aygit)

    return f


def kirilim(m, W, ad, aygit="cuda", parca=256):
    """Dogrulugu KONUMA gore boler -- onek uzadikca duzeliyor mu?

    Ilk konumlarda onek kisa; model orada zayif olabilir ve tek ortalama
    bunu gizler."""
    W, M = _ac(W)
    T = W.shape[1]
    dg, tp = torch.zeros(T - 1), torch.zeros(T - 1)
    with torch.no_grad():
        for i in range(0, W.shape[0], parca):
            w = W[i:i + parca].to(aygit).long()
            mk = None if M is None else M[i:i + parca].to(aygit)
            t = m.dizi(w, mk)[:, :-1].argmax(-1)
            d = t == w[:, 1:]
            a = torch.ones_like(d) if mk is None else mk[:, 1:]
            dg += (d & a).sum(0).float().cpu()
            tp += a.sum(0).float().cpu()   # konum basina GERCEK hedef
    return (dg / tp.clamp(min=1)).tolist()


def tablo(m, W, aygit="cuda", yaz=print, dilim=8):
    """Konum dilimlerine gore dogruluk -- HUKUM BURADAN OKUNUR."""
    o = kirilim(m, W, None, aygit)
    n = len(o)
    yaz(f"{'konum':>12} {'dogruluk':>10}")
    yaz("-" * 24)
    for k in range(dilim):
        a, b = k * n // dilim, (k + 1) * n // dilim
        yaz(f"{f'{a}-{b}':>12} {sum(o[a:b]) / (b - a):>10.4f}")
    yaz("-" * 24)
    yaz(f"{'TUMU':>12} {sum(o) / n:>10.4f}")
    return o


def devam(m, onek, ad, ix, coz, adim=60, aygit="cuda", tohum=None,
          sicaklik=0.0):
    """Istemin devamini URET.  Sayi degil METIN -- gozle okunur.

    sicaklik 0 -> hep en yakin token (belirlenimci).
    """
    g = None if tohum is None else torch.Generator().manual_seed(tohum)
    w = [ix.get(t, ix["<bilinmeyen>"]) for t in onek]
    with torch.no_grad():
        for _ in range(adim):
            p = m.dizi(torch.tensor([w], device=aygit))[0, -1]
            if sicaklik <= 0:
                c = int(p.argmax())
            else:
                c = int(torch.multinomial((p / sicaklik).softmax(-1), 1,
                                          generator=g))
            w.append(c)
    return coz(torch.tensor(w), ad)
