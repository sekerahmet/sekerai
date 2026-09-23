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
import olcme_17 as OL                                           # noqa: E402
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


# --- 3.  AKIS: int16, hikaye sayisi, okuma siniri
# en_mb eskiden iki katini okuyordu: bolumden artan karakter sayilmiyordu.
def t_akis():
    d = tempfile.mkdtemp()
    try:
        yol = os.path.join(d, "ornek.txt")
        hik = ["Lily had a red ball.", "Tom saw a big dog.", "The cat sat."] * 50
        with open(yol, "w", encoding="utf-8") as f:
            f.write(V.SINIR.join(hik) + V.SINIR)
        sus = lambda *a: None
        _, ix = V.sozluk(yol, 20, yaz=sus)
        a = V.akis(yol, ix, yaz=sus)
        n300 = sum(1 for _ in V._hikayeler_kar(yol, 100, 300))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    say = int((a == ix[V.HIKAYE]).sum())
    kapi("akis int16, hikaye sayisi, okuma siniri",
         a.dtype == np.int16 and say == len(hik) and n300 == 10,
         "%d hikaye; 300 karakterde %d (10 olmali)" % (say, n300))


# --- 4.  HER HIKAYE BIR PENCERE, <hikaye> BASTA ve SONDA
def t_pencere():
    hk, dl = 1, 0
    a = np.array([5, 6, hk, 7, 8, 9, hk, 3, hk], dtype=np.int16)
    P, M = V.pencere(a, 5, None, hk, dl)
    ok = (len(P) == 3 and list(M.sum(1)) == [4, 5, 3]
          and list(P[1]) == [hk, 7, 8, 9, hk] and list(P[2][:3]) == [hk, 3, hk]
          and (P[~M] == dl).all())
    P4, _ = V.pencere(a, 4, None, hk, dl)           # L+2 > T olan ATILIR
    kapi("pencere <hikaye> basta ve sonda", ok and len(P4) == 2,
         "%d pencere; T=4'te %d" % (len(P), len(P4)))


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


# --- 7.  OLCU KENDI ICINDE
# Kullanici: "bizim ölçümlerimiz kendi içinde olmalı".  bas + govde + son
# hepsini vermeli (surumler govdede kiyaslanir); tablo'nun TUMU dogrulukla
# ayni olmali (eskiden konumlar esit sayiliyordu).
def t_olc():
    hk, dl = 1, 0
    g = torch.Generator().manual_seed(9)
    akis = []
    for L in (1, 4, 9, 13):
        akis += torch.randint(3, 20, (L,), generator=g).tolist() + [hk]
    P, M = V.pencere(np.array(akis, dtype=np.int16), 16, None, hk, dl)
    W, M = torch.from_numpy(P.astype(np.int64)), torch.from_numpy(M)
    m = Yol(20, boyut=8, durum=8, tohum=0)
    r = OL.olc(m, (W, M), aygit="cpu", hk=OL.bos_kimligi(W, M))
    n = r["n_bas"] + r["n_govde"] + r["n_son"]
    ort = sum(r["ce_" + k] * r["n_" + k] for k in ("bas", "govde", "son")) / n
    satir = []
    OL.tablo(m, (W, M), aygit="cpu", yaz=satir.append)
    tumu = float(satir[-1].split()[1])
    ok = (r["n_bas"] == 4 and r["n_son"] == 4 and r["n_govde"] == 23
          and abs(ort - r["ce"]) < 1e-6 and abs(tumu - r["dogruluk"]) < 1e-4)
    kapi("olcu: turler toplami, agirlikli tablo", ok,
         "bas %d  govde %d  son %d" % (r["n_bas"], r["n_govde"], r["n_son"]))


if __name__ == "__main__":
    print("test_17")
    for f in (t_surdurme, t_dolgu, t_akis, t_pencere, t_durdur, t_coz, t_olc):
        f()
    print("\n%d GECTI   %d KALDI" % (len(GECTI), len(KALDI)))
    sys.exit(1 if KALDI else 0)
