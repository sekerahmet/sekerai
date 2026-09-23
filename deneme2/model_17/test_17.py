# -*- coding: utf-8 -*-
"""test_17 -- model_17'nin kapilari.  `python test_17.py`

Kapi bir YETENEK olcmez; kodun kendi iddiasini dogrular.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_17 as TR                                           # noqa: E402
import veri_t17 as V                                            # noqa: E402
from model_17 import Yol                                        # noqa: E402

GECTI, KALDI = [], []


def kapi(ad, sart, not_=""):
    (GECTI if sart else KALDI).append(ad)
    print("  %-40s %s  %s" % (ad, "GECTI" if sart else "KALDI", not_))


def _veri():
    """Kucuk rastgele egitim verisi: (N, W, M)."""
    N, T, n = 20, 16, 64
    g = torch.Generator().manual_seed(7)
    return (N, torch.randint(0, N, (n, T), generator=g),
            torch.ones(n, T, dtype=torch.bool))


def _temizle():
    for x in (TR.GUNLUK, TR.SONUC, TR.DURDUR, TR._DISKE):
        x.clear()


def _agirlik(ad):
    return {k: v.clone()
            for k, v in TR.SONUC[ad]["model"].state_dict().items()}


# --- 1.  SURDURME  ==  KESINTISIZ
# Kural 1: uzatma SURDURMEDIR.  Ayni yorunge cikmazsa surdurme bir
# yanilsamadir -- sessizce BASKA bir model uretir.  22 Eylul'de
# kalmisti: yedek o adimin step()'inden SONRA yazildigi icin dongu
# kaydedilen adimi ikinci kez atiyordu (fark 2,061e-03).
def t_surdurme():
    N, W, M = _veri()

    def kos(ad, kok, adim, surdur=None):
        _temizle()
        TR._kos(ad, (W, M), N, lambda *a, **k: 0.0, "cpu", kok, None,
                8, 8, 2e-3, 0.01, adim, 0, 8, 2, 2, surdur, False)
        return _agirlik(ad)

    kok = tempfile.mkdtemp()
    try:
        A = kos("KESINTISIZ", kok, 8)
        kos("BOLUK", kok, 4)
        B = kos("BOLUK", kok, 8, surdur=kok + "/BOLUK/t4.pt")
        en = max(float((A[k] - B[k]).abs().max()) for k in A)
    finally:
        shutil.rmtree(kok, ignore_errors=True)
    kapi("surdurme == kesintisiz", en == 0.0, "fark %.3e" % en)


# --- 2.  DOLGU KAYBA GIRMEZ
# Her hikaye bir pencere; kalan yer <dolgu>.  Maskeli kayip, hikayeleri
# tek tek islemekle AYNI sayiyi vermeli, yoksa dolgu modele ogretiliyor.
def t_dolgu():
    torch.manual_seed(0)
    m = Yol(20, boyut=8, durum=8, tohum=0)
    uz = [12, 7, 16]
    W = torch.zeros(3, 16, dtype=torch.long)
    M = torch.zeros(3, 16, dtype=torch.bool)
    g = torch.Generator().manual_seed(3)
    for i, L in enumerate(uz):
        W[i, :L] = torch.randint(1, 20, (L,), generator=g)
        M[i, :L] = True
    with torch.no_grad():
        a = float(m.kayip(W, M))
        pay = sum(float(m.kayip(W[i:i + 1, :L])) * (L - 1)
                  for i, L in enumerate(uz))
        b = pay / sum(L - 1 for L in uz)
    kapi("maskeli kayip == hikaye hikaye", abs(a - b) < 1e-5,
         "%.6f / %.6f" % (a, b))


# --- 3.  int16 KAYIPSIZ
def t_int16():
    a = np.arange(4003, dtype=np.int32)
    kapi("sozluk int16'ya sigar", a.max() < np.iinfo(np.int16).max
         and (a.astype(np.int16).astype(np.int32) == a).all(),
         "en buyuk %d < %d" % (a.max(), np.iinfo(np.int16).max))


# --- 4.  HER HIKAYE BIR PENCERE
def t_pencere():
    hk, dl = 1, 0
    a = np.array([5, 6, hk, 7, 8, 9, hk, 3, hk], dtype=np.int16)
    P, M = V.pencere(a, 5, None, hk, dl)
    ok = (len(P) == 3 and list(M.sum(1)) == [2, 3, 1]
          and list(P[1][:3]) == [7, 8, 9] and (P[~M] == dl).all())
    kapi("pencere hikaye sinirinda", ok, "%d pencere" % len(P))


# --- 5.  DURDUR + SURDUR  ==  KESINTISIZ
# Durdurulan kosu son TAMAMLANAN adimi yazmali.  Atilmamis adimi yazarsa
# surdurme bir adim eksik atar -- §1'in durdurma yolundaki esi.
def t_durdur():
    N, W, M = _veri()
    kok = tempfile.mkdtemp()

    def kos(ad, adim, surdur=None, dur=None, lr=2e-3):
        _temizle()
        say = [0]

        def olcut(m, taraf, tam=False):
            say[0] += 1
            if dur is not None and say[0] == 2 * (dur + 1):
                TR.DURDUR.add(ad)               # `dur` adiminin olcumu bitti
            return 0.0

        TR._kos(ad, (W, M), N, olcut, "cpu", kok, None, 8, 8, lr, 0.01,
                adim, 0, 8, 1, 3, surdur, False)
        return _agirlik(ad)

    try:
        A = kos("KESINTISIZ", 8)
        kos("DUR", 8, dur=4)
        pt = sorted(f for f in os.listdir(kok + "/DUR") if f.endswith(".pt"))
        B = kos("DUR", 8, surdur=kok + "/DUR/t4.pt")
        en = max(float((A[k] - B[k]).abs().max()) for k in A)
        try:
            kos("DUR", 8, surdur=kok + "/DUR/t4.pt", lr=4e-3)
            ayar = False
        except ValueError as h:
            ayar = "lr" in str(h)
    finally:
        shutil.rmtree(kok, ignore_errors=True)
    kapi("durdur: son TAMAMLANAN adim yazilir",
         pt == ["t0.pt", "t3.pt", "t4.pt"], " ".join(pt))
    kapi("durdur + surdur == kesintisiz", en == 0.0, "fark %.3e" % en)
    kapi("surdurme ayar farkini yakalar", ayar, "lr 2e-3 -> 4e-3")


# --- 6.  coz GIDIS-DONUS
# Gozle bakilan metin jetonlari BIREBIR tasimali; tasimazsa kusur
# ekranda "model sacmaladi" gibi gorunur.
def t_coz():
    cumle = ('Lily\'s mom said, "Don\'t go!" (It was 3 o\'clock.) '
             'The girls\' toys -- café... ok?')
    j = V.JETON.findall(cumle)
    ad = [V.DOLGU, V.HIKAYE, V.BILINMEYEN] + sorted(set(j) | {"'", '"', "s"})
    ix = {a: i for i, a in enumerate(ad)}
    ok = V.JETON.findall(V.coz(np.array([ix[t] for t in j]), ad)) == j
    r = np.random.default_rng(0)
    for _ in range(300):
        d = r.integers(3, len(ad), 25)                  # ozel jetonlar HARIC
        ok = ok and V.JETON.findall(V.coz(d, ad)) == [ad[i] for i in d]
    kapi("coz: jeton -> metin -> jeton", ok,
         "%d birim, 300 rastgele dizi" % (len(ad) - 3))


if __name__ == "__main__":
    print("test_17")
    for f in (t_surdurme, t_dolgu, t_int16, t_pencere, t_durdur, t_coz):
        f()
    print("\n%d GECTI   %d KALDI" % (len(GECTI), len(KALDI)))
    sys.exit(1 if KALDI else 0)
