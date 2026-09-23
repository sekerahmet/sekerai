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
from model_17 import Yol, DT, durum_gecisi                      # noqa: E402

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
def t_dolgu(m=None, ad="maskeli kayip == hikaye hikaye"):
    torch.manual_seed(0)
    m = m if m is not None else Yol(20, boyut=8, durum=8, tohum=0)
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
    kapi(ad, abs(a - b) < 1e-5, "%.6f / %.6f" % (a, b))


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


# ============================================================
# DT -- HEDEF MIMARI (TASARIM.md).  Kagit ustu hesabin iddialari.
# ============================================================
def _dt():
    return DT(20, genislik=16, durum=8, blok=2, bellek=32, tohum=0)


# --- 8.  PARAMETRE == KAGIT USTU HESAP
def t_dt_parametre():
    n, d, s, mb, Y, L = 4003, 256, 64, 1024, 1, 2
    blok = (2 * d + 2 * Y * d * s + Y * d + Y + d * s + 2 * s * s + s * d + 1
            + d * mb + mb + mb * d + d)
    kagit = 2 * n * d + L * blok + d
    kod = sum(p.numel() for p in DT(n).parameters())
    kapi("DT parametre == kagit ustu", kod == kagit,
         "%s / %s" % (f"{kod:,}", f"{kagit:,}"))


# --- 9.  DURUM TAKIBI: bardak oyunu, 1.000 rastgele takas
# beta 2, k = (e_i - e_j)/sqrt2 -> takas.  Top HALA dogru bardakta olmali;
# sira onemli: dogru cevap takaslarin SIRASIYLA hesaplaniyor.
def t_dt_bardak():
    s2 = 2 ** -0.5
    takas = {2: (0, 1), 3: (1, 2), 4: (0, 2)}
    K = {1: [1., 0., 0.], 2: [s2, -s2, 0.], 3: [0., s2, -s2], 4: [s2, 0., -s2]}
    g = torch.Generator().manual_seed(11)
    dizi = [1] + torch.randint(2, 5, (1000,), generator=g).tolist()
    top = 0
    for t in dizi[1:]:
        i, j = takas[t]
        top = j if top == i else i if top == j else top
    T = len(dizi) + 1                                    # sonda soru
    k, v = torch.zeros(3, T, 1, 3), torch.zeros(3, T, 1, 3)
    be, q = torch.zeros(3, T, 1), torch.zeros(3, T, 3)
    for t, tok in enumerate(dizi):
        k[:, t, 0] = torch.tensor(K[tok])
        be[:, t, 0] = 1.0 if tok == 1 else 2.0
    v[:, 0, 0] = torch.tensor([0., 0., 1.])              # top 1. bardakta
    k[:, T - 1, 0, 0] = 1.0                              # soru: beta 0
    q[torch.arange(3), T - 1, torch.arange(3)] = 1.0     # c. dizi c. bardagi sorar
    h = durum_gecisi(k, v, be, q)[:, -1].norm(dim=-1)
    ok = (int(h.argmax()) == top and float(h[top]) > 0.999
          and float(h.sort().values[1]) < 1e-3)
    kapi("DT bardak oyunu (1.000 takas)", ok,
         "top %d. bardakta, okunan %s" % (top + 1, [round(float(x), 4) for x in h]))


# --- 10.  DELTA KURALI: yazilan yuva SULANMAZ
def t_dt_delta():
    g = torch.Generator().manual_seed(12)
    T = 302
    k, v = torch.zeros(1, T, 1, 4), torch.zeros(1, T, 1, 4)
    be, q = torch.ones(1, T, 1), torch.zeros(1, T, 4)
    a = torch.randn(4, generator=g)
    k[0, 0, 0, 0] = 1.0
    v[0, 0, 0] = a                                       # e1 yuvasina a
    yuva = torch.randint(1, 4, (T - 2,), generator=g)
    k[0, 1:T - 1, 0][torch.arange(T - 2), yuva] = 1.0    # 300 kez BASKA yuvaya
    v[0, 1:T - 1, 0] = torch.randn(T - 2, 4, generator=g)
    be[0, T - 1, 0] = 0.0
    k[0, T - 1, 0, 0] = 1.0
    q[0, T - 1, 0] = 1.0                                 # soru: e1 yuvasi
    h = durum_gecisi(k, v, be, q)[0, -1]
    kapi("DT delta kurali: yuva sulanmaz", torch.allclose(h, a, atol=1e-5),
         "300 baska yazmadan sonra fark %.1e" % float((h - a).abs().max()))


# --- 11.  NEDENSELLIK: gelecek gecmisi degistirmez
def t_dt_nedensel():
    m = _dt()
    g = torch.Generator().manual_seed(5)
    w = torch.randint(1, 20, (2, 12), generator=g)
    w2 = w.clone()
    w2[:, 8] = (w[:, 8] % 19) + 1
    with torch.no_grad():
        a, b = m.dizi(w), m.dizi(w2)
    once = float((a[:, :8] - b[:, :8]).abs().max())
    sonra = float((a[:, 8:] - b[:, 8:]).abs().max())
    kapi("DT nedensel: gelecek gecmisi degistirmez", once == 0.0 and sonra > 0,
         "once %.1e  sonra %.1e" % (once, sonra))


# --- 12.  SURDURME == KESINTISIZ, DT ile
def t_dt_surdurme():
    N, W, M = _veri()

    def kos(ad, kok, adim, surdur=None):
        _temizle()
        TR._kos(ad, (W, M), N, lambda *a, **k: 0.0, "cpu", kok, None,
                8, 8, 2e-3, 0.01, adim, 0, 8, 2, 2, surdur, False,
                mimari="dt", genislik=16, blok=2, bellek=32)
        return _agirlik(ad)

    kok = tempfile.mkdtemp()
    try:
        A = kos("KESINTISIZ", kok, 8)
        kos("BOLUK", kok, 4)
        B = kos("BOLUK", kok, 8, surdur=kok + "/BOLUK/t4.pt")
        en = max(float((A[k] - B[k]).abs().max()) for k in A)
    finally:
        shutil.rmtree(kok, ignore_errors=True)
    kapi("DT surdurme == kesintisiz", en == 0.0, "fark %.3e" % en)


if __name__ == "__main__":
    print("test_17")
    for f in (t_surdurme, t_dolgu, t_akis, t_pencere, t_durdur, t_coz, t_olc,
              t_dt_parametre, t_dt_bardak, t_dt_delta, t_dt_nedensel,
              lambda: t_dolgu(_dt(), "DT maskeli kayip == hikaye hikaye"),
              t_dt_surdurme):
        f()
    print("\n%d GECTI   %d KALDI" % (len(GECTI), len(KALDI)))
    sys.exit(1 if KALDI else 0)
