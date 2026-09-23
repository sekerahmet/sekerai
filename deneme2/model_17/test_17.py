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
import veri_mat17 as VM                                         # noqa: E402
import veri_t17 as V                                            # noqa: E402
from model_17 import Yol, DT, durum_gecisi, GeciciBellek        # noqa: E402

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
    say = int((a == ix[V.SON]).sum())
    kapi("akis int16, hikaye sayisi, okuma siniri",
         a.dtype == np.int16 and say == len(hik) and n300 == 10,
         "%d hikaye; 300 karakterde %d (10 olmali)" % (say, n300))


# --- 4.  HER HIKAYE BIR PENCERE, <eos> BASTA ve SONDA
def t_pencere():
    hk, dl = 1, 0
    a = np.array([5, 6, hk, 7, 8, 9, hk, 3, hk], dtype=np.int16)
    P, M = V.pencere(a, 5, None, hk, dl)
    ok = (len(P) == 3 and list(M.sum(1)) == [4, 5, 3]
          and list(P[1]) == [hk, 7, 8, 9, hk] and list(P[2][:3]) == [hk, 3, hk]
          and (P[~M] == dl).all())
    P4, _ = V.pencere(a, 4, None, hk, dl)           # L+2 > T olan ATILIR
    kapi("pencere <eos> basta ve sonda", ok and len(P4) == 2,
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
    ad = [V.DOLGU, V.SON, V.BILINMEYEN] + sorted(set(j) | {"'", '"', "s"})
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


# ============================================================
# GECICI BELLEK (TASARIM.md §GECICI BELLEK) -- iddialari tek tek.
# ============================================================
def _gb(W=6, kafa=4):
    """Gecici bellegi ACIK (Wo rastgele) kucuk DT."""
    m = DT(20, genislik=16, durum=8, blok=2, bellek=32, tohum=0, gb_W=W,
           gb_kafa=kafa)
    g = torch.Generator().manual_seed(41)
    with torch.no_grad():
        for b in m.bloklar:
            b.gecici.Wo.normal_(0.0, 0.3, generator=g)
    return m


# --- G1.  SESSIZ BASLANGIC: Wo = 0 -> gecici belleksiz modelin AYNISI
# Tek dugmeli kiyas bunun ustunde duruyor: fark YALNIZ ogrenilenden gelmeli.
def t_gb_sessiz():
    a = DT(20, genislik=16, durum=8, blok=2, bellek=32, tohum=0)
    b = DT(20, genislik=16, durum=8, blok=2, bellek=32, tohum=0, gb_W=6, gb_kafa=4)
    sa, sb = a.state_dict(), b.state_dict()
    ayni = set(sa) < set(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    w = torch.randint(1, 20, (3, 12), generator=torch.Generator().manual_seed(8))
    with torch.no_grad():
        fark = float((a.dizi(w) - b.dizi(w)).abs().max())
    kapi("gecici bellek: sessiz baslangic", ayni and fark == 0.0,
         "ortak agirliklar ayni, cikti farki %.1e" % fark)


# --- G1b.  ONCUL: kafa h "h+1 geri"den baslar; sessiz baslangic bozulmaz
def t_gb_oncul():
    m = DT(20, genislik=16, durum=8, blok=2, bellek=32, tohum=0, gb_W=6,
           gb_kafa=4, gb_oncul=3.0)
    beklenen = torch.zeros(4, 6)
    for h in range(4):
        beklenen[h, h + 1] = 3.0
    ok = all(torch.equal(b.gecici.b.detach(), beklenen) for b in m.bloklar)
    a = DT(20, genislik=16, durum=8, blok=2, bellek=32, tohum=0)
    w = torch.randint(1, 20, (3, 12), generator=torch.Generator().manual_seed(8))
    with torch.no_grad():
        fark = float((a.dizi(w) - m.dizi(w)).abs().max())
    pay = float(torch.softmax(torch.tensor([3.0] + [0.0] * 15), 0)[0])
    kapi("gecici bellek: oncul h+1 geri, sessiz", ok and fark == 0.0,
         "W=16'da baslangic payi %.2f, cikti farki %.1e" % (pay, fark))


# --- G2.  "k GERI" GERCEKTEN k GERI
def t_gb_k_geri():
    d, W, h, T = 8, 6, 4, 10
    g = torch.Generator().manual_seed(31)
    gb = GeciciBellek(d, W, h, lambda *s: torch.randn(*s, generator=g))
    kk = [1, 2, 3, 0]
    with torch.no_grad():
        gb.Wq.zero_()
        gb.Wk.zero_()
        gb.Wv.copy_(torch.eye(d))
        gb.Wo.copy_(torch.eye(d))
        gb.b.fill_(-50.0)
        for i, k in enumerate(kk):
            gb.b[i, k] = 50.0
        kayit = torch.randn(1, T, d, generator=g)
        cik = gb(torch.randn(1, T, d, generator=g), kayit)
    dk = d // h
    ok = all(torch.allclose(cik[0, t, i * dk:(i + 1) * dk],
                            kayit[0, t - k, i * dk:(i + 1) * dk], atol=1e-6)
             for i, k in enumerate(kk) for t in range(3, T))
    kapi("gecici bellek: k geri == k geri", ok, "kafalar %s geri" % kk)


# --- G3.  PENCERE: W'den geride kalan okunmaz
def t_gb_pencere():
    d, W, h, T, t = 8, 4, 2, 12, 10
    g = torch.Generator().manual_seed(33)
    gb = GeciciBellek(d, W, h, lambda *s: torch.randn(*s, generator=g))
    with torch.no_grad():
        gb.Wo.normal_(0.0, 1.0, generator=g)
        q, kayit = torch.randn(1, T, d, generator=g), torch.randn(1, T, d, generator=g)
        taban = gb(q, kayit)[0, t]
        dis, ic = kayit.clone(), kayit.clone()
        dis[0, t - W] += 5.0                   # pencerenin HEMEN disi
        ic[0, t - W + 1] += 5.0                # pencerenin en eski ici
        a = float((gb(q, dis)[0, t] - taban).abs().max())
        b = float((gb(q, ic)[0, t] - taban).abs().max())
    kapi("gecici bellek: pencere disi okunmaz", a == 0.0 and b > 0,
         "disi %.1e  ici %.1e" % (a, b))


# --- G4.  NEDENSELLIK, gecici bellek acik
def t_gb_nedensel():
    m = _gb()
    w = torch.randint(1, 20, (2, 12), generator=torch.Generator().manual_seed(5))
    w2 = w.clone()
    w2[:, 8] = (w[:, 8] % 19) + 1
    with torch.no_grad():
        a, b = m.dizi(w), m.dizi(w2)
    once = float((a[:, :8] - b[:, :8]).abs().max())
    sonra = float((a[:, 8:] - b[:, 8:]).abs().max())
    kapi("gecici bellek: gelecek gecmisi degistirmez", once == 0.0 and sonra > 0,
         "once %.1e  sonra %.1e" % (once, sonra))


# --- G5.  PARAMETRE == KAGIT: blok basina 4 d^2 + kafa W + d
def t_gb_parametre():
    n, d, s, mb, W, h, L = 13, 16, 16, 64, 16, 4, 2
    taban = sum(p.numel() for p in DT(n, genislik=d, durum=s, bellek=mb).parameters())
    kod = sum(p.numel() for p in DT(n, genislik=d, durum=s, bellek=mb, gb_W=W,
                                     gb_kafa=h).parameters())
    kagit = taban + L * (4 * d * d + h * W + d)
    kapi("gecici bellek: parametre == kagit", kod == kagit,
         "%s / %s" % (f"{kod:,}", f"{kagit:,}"))


# --- G5b.  KONUM YANLILIGI DECAY DISINDA -- aksi halde "k geri" her adim
# sifira cekilir.  Gecici belleksiz modelde ayrim DEGISMEZ.
def t_gb_sonum():
    m = DT(13, genislik=16, durum=16, bellek=64, gb_W=16, gb_kafa=4)
    dis = {a for a, p in m.named_parameters() if not TR.sonumlu(a, p)}
    eski = DT(13, genislik=16, durum=16, bellek=64)
    ayni = all(TR.sonumlu(a, p) == (p.dim() >= 2) for a, p in eski.named_parameters())
    ok = ({"bloklar.0.gecici.b", "bloklar.1.gecici.b"} <= dis
          and not any(a.endswith(("gecici.Wq", "gecici.Wo")) for a in dis) and ayni)
    kapi("gecici bellek: b decay disinda", ok,
         "decay disi: %s" % ", ".join(sorted(a for a in dis if "gecici" in a)))


# --- G6.  SURDURME == KESINTISIZ, gecici bellek acik
def t_gb_surdurme():
    N, W, M = _veri()

    def kos(ad, kok, adim, surdur=None):
        _temizle()
        TR._kos(ad, (W, M), N, lambda *a, **k: 0.0, "cpu", kok, None,
                8, 8, 2e-3, 0.01, adim, 0, 8, 2, 2, surdur, False,
                mimari="dt", genislik=16, blok=2, bellek=32, gb_W=6, gb_kafa=4)
        return _agirlik(ad)

    kok = tempfile.mkdtemp()
    try:
        A = kos("KESINTISIZ", kok, 8)
        kos("BOLUK", kok, 4)
        B = kos("BOLUK", kok, 8, surdur=kok + "/BOLUK/t4.pt")
        en = max(float((A[k] - B[k]).abs().max()) for k in A)
    finally:
        shutil.rmtree(kok, ignore_errors=True)
    kapi("gecici bellek: surdurme == kesintisiz", en == 0.0, "fark %.3e" % en)


# ============================================================
# DUZ MATEMATIK SINAVI (veri_mat17) -- olcutun KENDISI sinaniyor.
# ============================================================
# --- 13.  PENCERE: <eos> soru cevap <eos>, hedef YALNIZ cevapta
def t_mat_pencere():
    W, M, H = VM.pencereler([(1, 1), (21, 23)])
    E_ = VM.EOS
    ok = (W[0].tolist() == [E_, 1, 10, 1, 11, 2, E_, E_, E_, E_]
          and W[1].tolist() == [E_, 2, 1, 10, 2, 3, 11, 4, 4, E_]
          and M.sum(1).tolist() == [7, 10]
          and torch.nonzero(H[0]).squeeze(1).tolist() == [5, 6]
          and torch.nonzero(H[1]).squeeze(1).tolist() == [7, 8, 9])
    kapi("matematik penceresi ve hedef maskesi", ok, "cevap + EOS hedef")


class _Kahin(torch.nn.Module):
    """Cevabi BILEN sahte model; '='den sonraki HER konumu doldurur.  kip:
    dogru / erken (hemen EOS) / fazla (EOS yerine fazladan bir rakam) /
    birler (yalniz birler basamagi yanlis)."""

    def __init__(self, kip):
        super().__init__()
        self.kip = kip

    def dizi(self, w, maske=None):
        cik = torch.zeros(w.shape[0], w.shape[1], VM.N)
        for r, s in enumerate(w.tolist()):
            e = s.index(VM.ESIT)
            ter, say = [], ""
            for t in s[1:e]:
                if t == VM.ARTI:
                    ter, say = ter + [int(say)], ""
                else:
                    say += str(t)
            hedef = VM.rak(sum(ter + [int(say)])) + [VM.EOS]
            for j in range(e, len(s)):
                yazilan = j - e
                if self.kip == "erken":
                    t = VM.EOS
                elif self.kip == "fazla" and yazilan == len(hedef) - 1:
                    t = 7
                elif self.kip == "birler" and yazilan == len(hedef) - 2:
                    t = (hedef[-2] + 1) % 10
                else:
                    t = hedef[min(yazilan, len(hedef) - 1)]
                cik[r, j, t] = 10.0
        return cik


# --- 14.  OLCUT: dogruyu 1, erken durani ve fazla yazani 0 saymali
def t_mat_sor():
    s = [(1, 1), (21, 23), (472, 182), (1, 1, 5), (23, 1, 120), (500, 500, 500)]
    d, e, f = (VM.sor(_Kahin(k), s, aygit="cpu") for k in ("dogru", "erken", "fazla"))
    ok = ((d["sayi"], d["uzunluk"], d["ilk"]) == (1.0, 1.0, 1.0)
          and (e["sayi"], e["uzunluk"], e["ilk"]) == (0.0, 0.0, 0.0)
          and (f["sayi"], f["uzunluk"], f["ilk"]) == (0.0, 0.0, 1.0))
    kapi("sor: dogru 1, erken 0, fazla rakam 0", ok,
         "dogru %s  erken %s  fazla %s" % tuple(
             (r["sayi"], r["uzunluk"], r["ilk"]) for r in (d, e, f)))


# --- 14b.  BASAMAK TANISI: saga hizali, birler AYRI okunur
# Hizalama kayarsa "birler ogrenilmedi" hukmu baska basamagin sayisiyla verilir.
def t_mat_basamak():
    s = [(1, 1), (21, 23), (472, 182), (1, 1, 5), (23, 1, 120),
         (500, 500, 500), (9, 1), (99, 1)]
    d = VM.basamak(_Kahin("dogru"), s)
    b = VM.basamak(_Kahin("birler"), s)
    ok = (all(v["og"] == 1.0 and v["ser"] == 1.0 for v in d.values())
          and all((v["og"], v["ser"]) == ((0.0, 0.0) if k[1] == "birler"
                                          else (1.0, 1.0))
                  for k, v in b.items())
          and sorted(k[0] for k in d if k[1] == "eos") == [1, 2, 3, 4])
    kapi("basamak: saga hizali, birler ayri", ok,
         "%d yuva; birler-yanlis kahinde yalniz birler 0" % len(d))


# --- 15.  CEVAP MASKELI EGITIM: surdurme == kesintisiz
def t_mat_egitim():
    g = torch.Generator().manual_seed(21)
    sorular = [tuple(x) for x in torch.randint(0, 60, (64, 2), generator=g).tolist()]
    W, M, H = VM.pencereler(sorular)

    def kos(ad, kok, adim, surdur=None):
        _temizle()
        TR._kos(ad, (W, M, H), VM.N, lambda *a, **k: 0.0, "cpu", kok, None,
                8, 8, 2e-3, 0.01, adim, 0, 8, 2, 2, surdur, False,
                mimari="dt", genislik=16, blok=2, bellek=32)
        return _agirlik(ad), TR.SONUC[ad]["kayip_iz"]

    kok = tempfile.mkdtemp()
    try:
        A, ka = kos("KESINTISIZ", kok, 8)
        kos("BOLUK", kok, 4)
        B, kb = kos("BOLUK", kok, 8, surdur=kok + "/BOLUK/t4.pt")
        en = max(float((A[k] - B[k]).abs().max()) for k in A)
    finally:
        shutil.rmtree(kok, ignore_errors=True)
    kapi("cevap maskeli egitim, surdurme", en == 0.0, "fark %.3e" % en)
    # Egri her adimda okunur; surdurulen kosuda delik ya da kayma olmamali.
    kapi("her adimin kaybi: surdurmede eksiksiz",
         len(ka) == 9 and not bool(ka.isnan().any()) and torch.equal(ka, kb),
         "%d adim, surdurulen == kesintisiz" % len(ka))


# --- 16.  DEFTER (mat_17.ipynb): kural 2 ve kural 8 MEKANIK
# GPU hucresi GPU'yu kendi icinde sorar; her kosu kendi hucresinde baslar;
# hucrelerde tanimsiz ad yok (Colab'da NameError bir gidis-donus demek).
def t_defter(yol=None):
    import ast
    import builtins
    import json
    yol = yol or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "mat_17.ipynb")
    hucre = [("".join(c["source"])) for c in
             json.load(open(yol, encoding="utf-8"))["cells"]
             if c["cell_type"] == "code"]
    tanimli, kusur = set(dir(builtins)) | {"display", "get_ipython"}, []
    for s in hucre:
        kunye = s.splitlines()[0]
        py = "\n".join(x for x in s.splitlines() if not x.lstrip().startswith("!"))
        try:
            agac = ast.parse(py)
        except SyntaxError as h:
            kusur.append("%s: sozdizimi %s" % (kunye[:30], h))
            continue
        if not kunye.startswith("# ") or kunye.count("|") < 2:
            kusur.append("kunye yok: " + kunye[:30])
        gpu = "|  GPU  |" in kunye
        if gpu and "torch.cuda.is_available()" not in py:
            kusur.append("GPU kapisi yok: " + kunye[:30])
        if py.count(".baslat(") > 1 or (".baslat(" in py and not gpu):
            kusur.append("baslat kurali: " + kunye[:30])
        bu = set()
        for d in ast.walk(agac):
            if isinstance(d, ast.Name) and isinstance(d.ctx, (ast.Store, ast.Del)):
                bu.add(d.id)
            elif isinstance(d, (ast.FunctionDef, ast.ClassDef)):
                bu.add(d.name)
            elif isinstance(d, ast.arg):
                bu.add(d.arg)
            elif isinstance(d, (ast.Import, ast.ImportFrom)):
                bu |= {(a.asname or a.name).split(".")[0] for a in d.names}
            elif isinstance(d, ast.ExceptHandler) and d.name:
                bu.add(d.name)
        eksik = sorted({d.id for d in ast.walk(agac) if isinstance(d, ast.Name)
                        and isinstance(d.ctx, ast.Load)} - tanimli - bu)
        if eksik:
            kusur.append("%s: tanimsiz %s" % (kunye[:30], eksik))
        tanimli |= bu
    kapi("defter: GPU kapisi, tek kosu, tanimsiz ad", not kusur,
         "; ".join(kusur) if kusur else "%d kod hucresi" % len(hucre))


if __name__ == "__main__":
    print("test_17")
    for f in (t_surdurme, t_dolgu, t_akis, t_pencere, t_durdur, t_coz, t_olc,
              t_dt_parametre, t_dt_bardak, t_dt_delta, t_dt_nedensel,
              lambda: t_dolgu(_dt(), "DT maskeli kayip == hikaye hikaye"),
              t_dt_surdurme, t_gb_sessiz, t_gb_oncul, t_gb_k_geri, t_gb_pencere,
              t_gb_nedensel,
              lambda: t_dolgu(_gb(), "gecici bellek: maskeli kayip == hikaye"),
              t_gb_parametre, t_gb_sonum, t_gb_surdurme,
              t_mat_pencere, t_mat_sor, t_mat_basamak, t_mat_egitim, t_defter):
        f()
    print("\n%d GECTI   %d KALDI" % (len(GECTI), len(KALDI)))
    sys.exit(1 if KALDI else 0)
