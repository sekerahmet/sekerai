# -*- coding: utf-8 -*-
"""tests -- model_18'in kapilari.  `python tests.py`

Kapi bir YETENEK olcmez; kodun kendi iddiasini dogrular.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as VM                                               # noqa: E402
import train as TR                                              # noqa: E402
from model_18 import PV as _PV                                  # noqa: E402

GECTI, KALDI = [], []
KUCUK = dict(d_order=16, d_content=0, vectors=8, active=2, layers=2)   # egitim kapilari icin kucuk PV
# Kapilar parcalari TEK TEK sinar: tepedeki tasarimin (model_18) ozellikleri burada KAPALI baslar,
# her kapi sinadigini acar.  Tepenin kendisi t_tepe'de.
SADE = dict(d_order=128, d_content=0, t_max=64, start_norm=None, lam=1.0, chain="absolute",
            c_cache=False, cache_topk=8, query=False, c_content=False, select="distance", active=8, load_balance=0.0)


def PV(n, **ayar):
    return _PV(n, **dict(SADE, **ayar))


def _run(*a, **ayar):
    return TR._run(*a, **dict(SADE, **ayar))


def kapi(ad, sart, not_=""):
    (GECTI if sart else KALDI).append(ad)
    print("  %-42s %s  %s" % (ad, "GECTI" if sart else "KALDI", not_))


def _dizi(s):
    return torch.tensor([VM.AD.index(a) if a != "E" else VM.EOS for a in s])


def _kos(ad, kok, data, n_vocab, adim, metric=None, resume=None, lr=2e-3):
    """_run'i kucuk PV ile, ayni iplikte kosturur; Run'i doner."""
    r = TR.RUNS[ad] = TR.Run(ad, kok)
    _run(r, data, n_vocab, metric or _sifir, "cpu", lr, adim, 0, 8,
            2, 2, resume=resume, **KUCUK)
    return r


def _sifir(m, side, full=False):
    return {"accuracy": 0.0}


def _agirlik(r):
    return {k: v.clone() for k, v in r.result["model"].state_dict().items()}


# ============================================================
# PV -- model_18.py'nin iddialari tek tek.
# ============================================================

# --- 1.  ZINCIR: yeni token yalniz bir terim ekler, eski terimler degismez
def t_zincir():
    m, w = PV(VM.N), _dizi("E31+52=8")
    C = m.C(w[None])[0]
    ok = all(torch.allclose(C[t] - m.RM[t] * m.P[w[t]], C[t - 1], atol=1e-6)
             for t in range(1, len(w)))
    kapi("zincir: C_t - RM_t*P[w_t] == C_(t-1)", ok, "%d halka" % len(w))


# --- 2.  NEDENSEL: sonraki token onceki konumlarin puanini degistirmez
def t_nedensel():
    m = PV(VM.N)
    a, b = m.scoreboard(_dizi("E31+52=8")[None]), m.scoreboard(_dizi("E31+52=9")[None])
    kapi("nedensel", torch.equal(a[0, :-1], b[0, :-1]), "son token degisti, oncekiler ayni")


# --- 3.  SESSIZ BASLANGIC: V_a = 0, C yerinde kalir
def t_sessiz():
    m, w = PV(VM.N), _dizi("E31+52=8")[None]
    kapi("baslangicta C_m == C", torch.equal(m.move(w)[0], m.C(w)))


# --- 4.  CM: geriye yuruyus zincirin halkalarini birebir verir
def t_cm():
    m, w = PV(VM.N), _dizi("E31+52=8")
    C = m.C(w[None])[0]
    yol = m.CM(w)
    ok = (len(yol) == len(w) and all(torch.allclose(c, C[t], atol=1e-5) for t, c in yol)
          and [t for t, _ in m.CM(w, K=3)] == [7, 6, 5])
    kapi("CM == zincir, K ile sinir", ok, "%d adim; K=3 -> 7 6 5" % len(yol))


# --- 5.  PAYDA: W_v toplami 1 -- butun vektorler ayni V_a ise tasima tam V_a
def t_payda():
    m, w = PV(VM.N), _dizi("E31+52=8")[None]
    u = torch.randn(m.d_sum, generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        for L in m.V:
            L.finish.copy_(L.start + u)
        fark = float((m.move(w)[0] - m.C(w) - m.layers * u).abs().max())
    kapi("payda: W_v toplami 1", fark < 1e-4, "fark %.1e" % fark)


# --- 6.  GRADYAN yalniz AKTIF vektorlere (son konumun hedefi yok)
def t_gradyan():
    m, w = PV(VM.N), _dizi("E31+52=8")[None]
    with torch.no_grad():
        for L in m.V:
            L.finish.add_(0.01 * torch.randn(L.finish.shape,
                                             generator=torch.Generator().manual_seed(6)))
    m.loss(w).backward()
    ok = True
    for L, ids in zip(m.V, m.trace(w)):
        g = (L.start.grad.abs().sum(-1) + L.finish.grad.abs().sum(-1)) > 0
        ok &= set(g.nonzero().flatten().tolist()) == set(ids[0, :-1].flatten().tolist())
    kapi("gradyan yalniz aktif vektorlere", ok, "%d layer" % m.layers)


# --- 7.  PARAMETRE == hesap: vectors x 2 x d x layers + layers (S_v)
def t_parametre():
    m = PV(VM.N)
    bek = m.vectors * 2 * m.d_sum * m.layers + m.layers
    par = sum(p.numel() for p in m.parameters())
    sabit = sorted(a for a, _ in m.named_buffers())
    kapi("parametre == hesap; P ve RM sabit", par == bek and sabit == ["P", "RM"],
         "%s / %s" % (f"{par:,}", f"{bek:,}"))


# --- 8.  MASKE: maskeli kayip == soru soru
def t_mask():
    m = PV(20, **KUCUK)
    uz = [12, 7, 16]
    W = torch.zeros(3, 16, dtype=torch.long)
    M = torch.zeros(3, 16, dtype=torch.bool)
    g = torch.Generator().manual_seed(3)
    for i, L in enumerate(uz):
        W[i, :L] = torch.randint(1, 20, (L,), generator=g)
        M[i, :L] = True
    with torch.no_grad():
        a = float(m.loss(W, M))
        b = sum(float(m.loss(W[i:i + 1, :L])) * (L - 1) for i, L in enumerate(uz))
        b /= sum(L - 1 for L in uz)
    kapi("maskeli kayip == soru soru", abs(a - b) < 1e-5, "%.6f / %.6f" % (a, b))


# ============================================================
# TRAIN -- surdurme ve durdurma (model_17'den).
# ============================================================
def _veri():
    """Kucuk rastgele egitim verisi: (N, (W, M, H)); her konum hedef."""
    N, T, n = 20, 16, 64
    g = torch.Generator().manual_seed(7)
    M = torch.ones(n, T, dtype=torch.bool)
    return N, (torch.randint(0, N, (n, T), generator=g), M, M)


# --- 9.  SURDURME == KESINTISIZ (kural 1: uzatma surdurmedir)
def t_surdurme():
    N, data = _veri()
    kok = tempfile.mkdtemp()
    try:
        A = _agirlik(_kos("KESINTISIZ", kok, data, N, 8))
        _kos("BOLUK", kok, data, N, 4)
        B = _agirlik(_kos("BOLUK", kok, data, N, 8, resume=kok + "/BOLUK/t4.pt"))
        en = max(float((A[k] - B[k]).abs().max()) for k in A)
    finally:
        shutil.rmtree(kok, ignore_errors=True)
    kapi("surdurme == kesintisiz", en == 0.0, "fark %.3e" % en)


# --- 10.  DURDUR: son tamamlanan adim yazilir; durdur + surdur == kesintisiz
def t_durdur():
    N, data = _veri()
    kok = tempfile.mkdtemp()

    def kos(ad, adim, surdur=None, dur=None, lr=2e-3):
        r = TR.RUNS[ad] = TR.Run(ad, kok)
        say = [0]

        def metric(m, side, full=False):
            say[0] += 1
            if dur is not None and say[0] == 2 * (dur + 1):
                r.stop()                        # `dur` adiminin olcumu bitti
            return {"accuracy": 0.0}

        _run(r, data, N, metric, "cpu", lr, adim, 0, 8, 1, 3,
                resume=surdur, **KUCUK)
        return _agirlik(r)

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


# --- 11.  SKOR: sozluk tablosu, karesiz == -S_p*D, ogrenilen S_p, surdurme skoru ayirt eder
def t_skor():
    import model_18 as M
    kural = [M.score_rule(n) for n in (13, 49, 50, 99, 100, 499, 500, 1999, 2000, 50000)]
    bek = [(False, 10.0), (False, 10.0), (False, 5.0), (False, 5.0), (False, 2.0),
           (False, 2.0), (True, 2.0), (True, 2.0), (True, M.LEARNED), (True, M.LEARNED)]
    kapi("skor tablosu: sinir ust satira", kural == bek, "13 -> karesiz 10, 2000 -> ogrenilen")

    torch.manual_seed(0)
    kare = PV(13, squared=True, S_p=1.0, **KUCUK)
    karesiz = PV(13, **KUCUK)                                  # tablodan: karesiz, S_p 10
    C_m = torch.randn(4, 16)
    D = torch.cdist(C_m, kare.P)
    fark = float((karesiz.score(C_m) + 10.0 * D).abs().max())
    fark2 = float((kare.score(C_m) + D * D).abs().max())
    sira = torch.equal(kare.score(C_m).argsort(-1), karesiz.score(C_m).argsort(-1))
    kapi("karesiz == -10*D, kare == -D^2", fark < 1e-4 and fark2 < 1e-3 and sira,
         "fark %.1e / %.1e, sira ayni" % (fark, fark2))

    m = PV(2500, **KUCUK)
    m.score(C_m).sum().backward()
    ok = (m.squared and m.S_p_learned and float(m.S_p) == 0.0 and m.S_p.grad is not None
          and sum(p.numel() for p in m.parameters())
          == m.vectors * 2 * m.d_sum * m.layers + m.layers + 1)
    kapi("ogrenilen S_p: e^0 = 1, gradyan alir", ok, "parametre +1")

    N, data = _veri()
    kok = tempfile.mkdtemp()
    try:
        r = TR.RUNS["KARE"] = TR.Run("KARE", kok)
        _run(r, data, N, _sifir, "cpu", 2e-3, 3, 0, 8, 3, 3,
                squared=True, S_p=1.0, **KUCUK)
        yol = kok + "/KARE/t3.pt"
        eski = torch.load(yol, weights_only=False)
        del eski["S_p"], eski["squared"]         # alanlar gelmeden yazilmis paket
        torch.save(eski, yol)
        r = TR.RUNS["KARE"] = TR.Run("KARE", kok)
        _run(r, data, N, _sifir, "cpu", 2e-3, 6, 0, 8, 3, 3,
                resume=yol, squared=True, S_p=1.0, **KUCUK)
        kabul = r.result.get("step") == 6
        try:
            r = TR.RUNS["KARE"] = TR.Run("KARE", kok)
            _run(r, data, N, _sifir, "cpu", 2e-3, 9, 0, 8, 3, 3,
                    resume=yol, **KUCUK)             # tablodan: karesiz 10
            yakaladi = False
        except ValueError as h:
            yakaladi = "S_p" in str(h) and "squared" in str(h)
    finally:
        shutil.rmtree(kok, ignore_errors=True)
    kapi("surdurme skor farkini yakalar", kabul and yakaladi,
         "eski kare paket: kare ile surer, tabloyla durur")


# --- 12.  START BOYU: start_norm ile secimi C belirler; surdurme farki yakalar
def t_start():
    torch.manual_seed(0)
    eski = PV(4003, d_order=256, d_content=0, vectors=64, t_max=256)
    yeni = PV(4003, d_order=256, d_content=0, vectors=64, t_max=256, start_norm=1.0)
    L = yeni.V[0]
    boy = float(L.start.detach().norm(dim=-1).median())
    ayni = torch.equal(eski.V[0].start * (1.0 / 256 ** 0.5), L.start) and torch.equal(L.start, L.finish)
    w = torch.randint(3, 4003, (8, 200))
    C = yeni.C(w).reshape(-1, 256)
    kisa = set(L.start.norm(dim=-1).argsort()[:L.active].tolist())

    def kisa_payi(Ls):
        from model_18 import distance
        sec = distance(C, Ls.start).topk(Ls.active, dim=-1, largest=False).indices
        k = set(Ls.start.norm(dim=-1).argsort()[:Ls.active].tolist())
        return sum(len(k & set(r)) for r in sec.tolist()) / sec.numel()
    a, b = kisa_payi(eski.V[0]), kisa_payi(L)
    kapi("start_norm: boy 1, secimi C belirler", abs(boy - 1) < 0.1 and ayni and b < a / 3,
         "boy %.2f; en kisa 8'in payi randn %%%.0f -> %%%.0f" % (boy, 100 * a, 100 * b))

    N, data = _veri()
    kok = tempfile.mkdtemp()
    try:
        r = TR.RUNS["S"] = TR.Run("S", kok)
        _run(r, data, N, _sifir, "cpu", 2e-3, 3, 0, 8, 3, 3, **KUCUK)
        yol = kok + "/S/t3.pt"
        p = torch.load(yol, weights_only=False)
        del p["start_norm"]                      # alan gelmeden yazilmis paket
        torch.save(p, yol)
        try:
            r = TR.RUNS["S"] = TR.Run("S", kok)
            _run(r, data, N, _sifir, "cpu", 2e-3, 6, 0, 8, 3, 3, resume=yol,
                    start_norm=1.0, **KUCUK)
            yakaladi = False
        except ValueError as h:
            yakaladi = "start_norm" in str(h)
    finally:
        shutil.rmtree(kok, ignore_errors=True)
    kapi("surdurme start_norm farkini yakalar", yakaladi, "eski randn paket -> start_norm 1")


# --- 13.  LAM: C_t = lam*C_(t-1) + RM_t*P[w_t]; CM geri yurur; nedensel; surdurme farki
def t_lam():
    m = PV(VM.N, lam=0.9)
    w = _dizi("E31+52=8")
    C = m.C(w[None])[0]
    zincir = all(torch.allclose(C[t] - m.RM[t] * m.P[w[t]], 0.9 * C[t - 1], atol=1e-5)
                 for t in range(1, len(w)))
    cm = all(torch.allclose(c, C[t], atol=1e-4) for t, c in m.CM(w))
    bir = torch.equal(PV(VM.N, lam=1.0).C(w[None]), PV(VM.N).C(w[None]))
    a, b = m.scoreboard(_dizi("E31+52=8")[None]), m.scoreboard(_dizi("E31+52=9")[None])
    nedensel = torch.equal(a[0, :-1], b[0, :-1])
    uzun = PV(50, d_order=32, d_content=0, t_max=512, lam=0.7).C(torch.randint(0, 50, (2, 512)))
    kapi("lam: zincir, CM, nedensel, lam 1 == cumsum", zincir and cm and bir and nedensel
         and bool(torch.isfinite(uzun).all()), "0,9; T=512'de lam 0,7 sonlu")

    N, data = _veri()
    kok = tempfile.mkdtemp()
    try:
        r = TR.RUNS["L"] = TR.Run("L", kok)
        _run(r, data, N, _sifir, "cpu", 2e-3, 3, 0, 8, 3, 3, **KUCUK)
        yol = kok + "/L/t3.pt"
        p = torch.load(yol, weights_only=False)
        del p["lam"]                             # alan gelmeden yazilmis paket
        torch.save(p, yol)
        try:
            r = TR.RUNS["L"] = TR.Run("L", kok)
            _run(r, data, N, _sifir, "cpu", 2e-3, 6, 0, 8, 3, 3, resume=yol, lam=0.9, **KUCUK)
            yakaladi = False
        except ValueError as h:
            yakaladi = "lam" in str(h)
    finally:
        shutil.rmtree(kok, ignore_errors=True)
    kapi("surdurme lam farkini yakalar", yakaladi, "eski paket (lam 1) -> lam 0,9")


# --- 14.  RELATIVE: C_t = lam*kaydir(C_(t-1)) + P[w_t]; ayni baglam her konumda ayni nokta
def t_relative():
    m = PV(VM.N, lam=0.7, chain="relative")
    w = _dizi("E31+52=8")
    C = m.C(w[None])[0]
    zincir = torch.allclose(C[0], m.P[w[0]], atol=1e-6) and all(
        torch.allclose(C[t], 0.7 * torch.roll(C[t - 1], 1) + m.P[w[t]], atol=1e-5)
        for t in range(1, len(w)))
    cm = all(torch.allclose(c, C[t], atol=1e-4) for t, c in m.CM(w))
    a, b = m.scoreboard(_dizi("E31+52=8")[None]), m.scoreboard(_dizi("E31+52=9")[None])
    nedensel = torch.equal(a[0, :-1], b[0, :-1])
    # ayni son 3 token, farkli konum: solma sayesinde C'ler neredeyse ayni (absolute'ta degil)
    g = torch.Generator().manual_seed(9)
    on = torch.randint(0, 50, (40,), generator=g)
    uc = torch.tensor([7, 8, 9])
    kisa, uzun = torch.cat([on[:5], uc]), torch.cat([on, uc])
    cos = {}
    for zin in ("relative", "absolute"):
        mm = PV(50, d_order=256, d_content=0, t_max=64, lam=0.5, chain=zin)
        cos[zin] = float(torch.cosine_similarity(mm.C(kisa[None])[0, -1], mm.C(uzun[None])[0, -1], dim=0))
    kapi("relative: zincir, CM, nedensel, konumdan bagimsiz",
         zincir and cm and nedensel and cos["relative"] > 0.9 and cos["absolute"] < 0.5,
         "ayni son 3 kelime, 8. ve 43. konum: cos relative %.2f, absolute %.2f"
         % (cos["relative"], cos["absolute"]))

    N, data = _veri()
    kok = tempfile.mkdtemp()
    try:
        r = TR.RUNS["R"] = TR.Run("R", kok)
        _run(r, data, N, _sifir, "cpu", 2e-3, 3, 0, 8, 3, 3, **KUCUK)
        yol = kok + "/R/t3.pt"
        p = torch.load(yol, weights_only=False)
        del p["chain"]                           # alan gelmeden yazilmis paket
        torch.save(p, yol)
        try:
            r = TR.RUNS["R"] = TR.Run("R", kok)
            _run(r, data, N, _sifir, "cpu", 2e-3, 6, 0, 8, 3, 3, resume=yol,
                    chain="relative", **KUCUK)
            yakaladi = False
        except ValueError as h:
            yakaladi = "chain" in str(h)
        r = TR.RUNS["R2"] = TR.Run("R2", kok)
        _run(r, data, N, _sifir, "cpu", 2e-3, 4, 0, 8, 2, 4, chain="relative", lam=0.7,
                start_norm=1.0, **KUCUK)
        egitim = r.result.get("step") == 4
    finally:
        shutil.rmtree(kok, ignore_errors=True)
    kapi("relative egitilir; surdurme chain farkini yakalar", yakaladi and egitim,
         "eski paket (absolute) -> relative")


# --- 15.  C_CACHE + GATE: nedensel, normalize log p, gate 0 == modelin kendisi, kopya calisir
def t_ccache():
    torch.manual_seed(0)
    kw = dict(d_order=256, d_content=0, t_max=64, lam=0.5, chain="relative")
    m = PV(50, c_cache=True, **kw)
    g = torch.Generator().manual_seed(4)
    w = torch.randint(0, 50, (1, 30), generator=g)
    a = m.scoreboard(w)
    w2 = w.clone(); w2[0, -1] = (w2[0, -1] + 1) % 50
    nedensel = torch.allclose(a[0, :-1], m.scoreboard(w2)[0, :-1], atol=1e-6)
    normal = float((a.logsumexp(-1)).abs().max()) < 1e-4
    yalin = PV(50, c_cache=False, **kw)
    with torch.no_grad():
        m.cache.gate_0.fill_(-60.0)
        kapali = torch.allclose(m.scoreboard(w), torch.log_softmax(yalin.scoreboard(w), -1), atol=1e-4)
        m.cache.gate_0.fill_(60.0)
        seri = torch.randperm(50, generator=g)[:12]
        tekrar = torch.cat([seri, seri[:9]])[None]            # ikinci kez: ...x9 -> x10 bekleniyor
        kopya = int(m.scoreboard(tekrar)[0, -1].argmax()) == int(seri[9])
        m.cache.gate_0.fill_(-1.0)
        M = torch.ones(1, 30, dtype=torch.bool); M[0, 20:] = False
        tam = float(torch.nn.functional.cross_entropy(m.scoreboard(w)[0, :-1], w[0, 1:],
                                                      reduction="none")[:19].mean())
        hizli = abs(float(m.loss(w, M)) - tam) < 1e-5
    kapi("c_cache: nedensel, log p, gate 0 == model, kopya", nedensel and normal and kapali and kopya,
         "tekrar eden dizide gate 1 -> sonraki kelime defterden")
    kapi("c_cache hizli kayip == tam tablo", hizli, "egitim yolu (B,T,n) kurmadan ayni kayip")

    N, data = _veri()
    kok = tempfile.mkdtemp()
    try:
        r = TR.RUNS["CC"] = TR.Run("CC", kok)
        _run(r, data, N, _sifir, "cpu", 2e-3, 3, 0, 8, 3, 3, **KUCUK)
        yol = kok + "/CC/t3.pt"
        p = torch.load(yol, weights_only=False)
        del p["c_cache"]                         # alan gelmeden yazilmis paket
        torch.save(p, yol)
        try:
            r = TR.RUNS["CC"] = TR.Run("CC", kok)
            _run(r, data, N, _sifir, "cpu", 2e-3, 6, 0, 8, 3, 3, resume=yol, c_cache=True, **KUCUK)
            yakaladi = False
        except ValueError as h:
            yakaladi = "c_cache" in str(h)
        r = TR.RUNS["CC2"] = TR.Run("CC2", kok)
        _run(r, data, N, _sifir, "cpu", 2e-3, 4, 0, 8, 2, 4, chain="relative", lam=0.7,
                start_norm=1.0, c_cache=True, **KUCUK)
        egitim = r.result.get("step") == 4 and torch.isfinite(r.result["step_losses"]).all()
    finally:
        shutil.rmtree(kok, ignore_errors=True)
    kapi("c_cache egitilir; surdurme farki yakalar", yakaladi and bool(egitim),
         "eski paket (c_cache yok) -> c_cache")


# --- 16.  BASLANGIC AYARLARI: S_v, S_c, gate_0, S_p tepeden ve kosudan; surdurme farki
def t_init():
    import model_18 as M
    m = PV(4003, d_order=32, d_content=0, t_max=16, c_cache=True, s_v_init=0.5, s_c_init=2.0, gate_0_init=-1.0,
           s_p_init=0.25)
    v = [float(m.V[0].S_v), float(m.cache.S_c), float(m.cache.gate_0), float(m.S_p)]
    d0 = PV(4003, d_order=32, d_content=0, t_max=16, c_cache=True)
    v0 = [float(d0.V[0].S_v), float(d0.cache.S_c), float(d0.cache.gate_0), float(d0.S_p)]
    ok = (v == [0.5, 2.0, -1.0, 0.25]
          and v0 == [M.S_V_INIT, M.S_C_INIT, M.GATE_0_INIT, M.S_P_INIT])
    N, data = _veri()
    kok = tempfile.mkdtemp()
    try:
        r = TR.RUNS["I"] = TR.Run("I", kok)
        _run(r, data, N, _sifir, "cpu", 2e-3, 3, 0, 8, 3, 3, **KUCUK)
        try:
            r = TR.RUNS["I"] = TR.Run("I", kok)
            _run(r, data, N, _sifir, "cpu", 2e-3, 6, 0, 8, 3, 3, resume=kok + "/I/t3.pt",
                    s_v_init=0.5, **KUCUK)
            yakaladi = False
        except ValueError as h:
            yakaladi = "s_v_init" in str(h)
    finally:
        shutil.rmtree(kok, ignore_errors=True)
    kapi("baslangic ayarlari: tepeden, kosudan, surdurmede", ok and yakaladi,
         "S_v 0 / S_c 3 / gate_0 -2 / S_p 0; s_v_init farki yakalanir")


# --- 17.  C_CONTENT: parca hesabi == adim adim; C_order == relative; nedensel; CM; gradyan
def t_content():
    import model_18 as M
    torch.manual_seed(0)
    n, d, dc = 60, 64, 32
    m = PV(n, t_max=64, lam=0.7, chain="relative", c_content=True, d_order=d - dc, d_content=dc)
    with torch.no_grad():                       # ogrenilmis gibi: token ve boyuta gore farkli, bazisi ~0
        m.lam_w.normal_(0, 3); m.beta_w.normal_(0, 1)
        m.lam_w[:, :4] = -40.0                  # alt sinir: LAM_W_MIN
    w = torch.randint(0, n, (2, 45))            # 45: parca sinirlari (16) icinden gecer
    C = m.C(w)
    lam, beta = m.lam_beta(w)
    h, adim = torch.zeros(2, dc), []
    for t in range(w.shape[1]):
        h = lam[:, t] * h + beta[:, t] * m.P[w[:, t], d - dc:]
        adim.append(h)
    icerik = torch.allclose(C[..., d - dc:], torch.stack(adim, 1), atol=1e-5)
    rel = PV(n, d_order=d - dc, d_content=0, t_max=64, lam=0.7, chain="relative")
    rel.P = m.P[:, :d - dc]
    sira = torch.allclose(C[..., :d - dc], rel.C(w), atol=1e-5)
    w2 = w.clone(); w2[:, 30] = (w2[:, 30] + 1) % n
    nedensel = torch.equal(C[:, :30], m.C(w2)[:, :30])
    with torch.no_grad():
        m.lam_w.clamp_(0, 3)                    # CM lam'a boler: cok kucuk lam sayisal hatayi buyutur
    cm = all(torch.allclose(c, m.C(w[:1])[0, t], atol=1e-3) for t, c in m.CM(w[0], K=10))
    m.loss(w).backward()
    grad = bool(m.lam_w.grad.abs().sum() > 0 and m.beta_w.grad.abs().sum() > 0)
    kapi("C_content: parca == adim adim, C_order == relative", icerik and sira and nedensel,
         "45 adim, 16'lik parcalar, lam_w alt sinirda (e^-5) dahil")
    kapi("C_content: CM geri yuruyus, lam_w/beta_w gradyan alir", cm and grad)

    yok = PV(n, d_order=d, d_content=0, t_max=64, lam=0.7, chain="relative")
    kapali = PV(n, t_max=64, lam=0.7, chain="relative", d_order=d - dc, d_content=dc)
    ayni = torch.equal(yok.C(w), kapali.C(w)) and not hasattr(kapali, "lam_w")
    bas = _PV(n, t_max=64, c_cache=False)                  # boyutlar tepeden
    l0, b0 = bas.lam_beta(w)
    baslangic = (abs(float(l0.mean()) - M.LAM_W_INIT) < 1e-5 and abs(float(b0.mean()) - M.BETA_W_INIT) < 1e-5
                 and (bas.d_order, bas.d_content, bas.d_sum) == (M.D_ORDER, M.D_CONTENT, M.D_SUM))
    kapi("C_content kapali == bugunku zincir; ayarlar tepeden", ayni and baslangic,
         "D_SUM %d = %d + %d, lam_w %.2f beta_w %.2f"
         % (bas.d_sum, bas.d_order, bas.d_content, float(l0.mean()), float(b0.mean())))


# --- 19.  TEPE = MODELIN SON YAPISI: ayarsiz PV, tepedeki tasarimi kurar
def t_tepe():
    import model_18 as M
    m = _PV(60)
    ok = (m.chain == "relative" and m.lam == M.LAM and m.c_content and m.d_sum == M.D_SUM
          and m.cache is not None and m.cache.query is not None and m.cache.topk is M.CACHE_TOPK
          and m.t_max == M.T_MAX and hasattr(m, "lam_w") and m.select == "direction"
          and all(L.active == len(L.start) for L in m.V) and m.cache.query.active == len(m.cache.query.start))
    kapi("tepe: ayarsiz PV == son yapi", ok,
         "chain %s lam %g, D_SUM %d = %d + %d, defter topk %s + Q %d ok, T_MAX %d; hepsi aktif, secim yon"
         % (m.chain, m.lam, m.d_sum, m.d_order, m.d_content, m.cache.topk,
            len(m.cache.query.start) if m.cache.query is not None else 0, m.t_max))


# --- 18.  Q: oklar bosken Q == C (defter aynen); butun gecmis == k=T; gradyan oklara ulasir
def t_query():
    torch.manual_seed(0)
    kw = dict(d_order=64, d_content=0, t_max=64, lam=0.5, chain="relative", start_norm=1.0, c_cache=True)
    w = torch.randint(0, 50, (2, 40))
    a = PV(50, **kw)
    b = PV(50, query=True, query_vectors=16, query_active=4, **kw)
    bos = torch.allclose(a.scoreboard(w), b.scoreboard(w), atol=1e-5)
    hepsi = PV(50, cache_topk=None, query=True, query_vectors=16, query_active=4, **kw)
    kT = PV(50, cache_topk=40, query=True, query_vectors=16, query_active=4, **kw)
    tum = torch.allclose(hepsi.scoreboard(w), kT.scoreboard(w), atol=1e-5)
    with torch.no_grad():
        hepsi.cache.gate_0.fill_(0.0)
        hepsi.cache.query.finish.add_(0.3 * torch.randn_like(hepsi.cache.query.finish))
        M_ = torch.ones_like(w, dtype=torch.bool); M_[1, 25:] = False
        lp = hepsi.scoreboard(w)[:, :-1]
        ce = torch.nn.functional.cross_entropy(lp.transpose(1, 2), w[:, 1:], reduction="none")
        tam = float(ce[M_[:, 1:]].mean())
    hizli = abs(float(hepsi.loss(w, M_)) - tam) < 1e-5
    w2 = w.clone(); w2[:, -1] = (w2[:, -1] + 1) % 50
    nedensel = torch.allclose(lp[:, :-1], hepsi.scoreboard(w2)[:, :-2], atol=1e-6)
    hepsi.loss(w).backward()
    grad = bool(hepsi.cache.query.finish.grad.abs().sum() > 0)
    Cm = PV(50, query=True, query_vectors=16, query_active=4, query_by="C", **kw)
    secim = (torch.allclose(Cm.scoreboard(w), a.scoreboard(w), atol=1e-5)
             and PV(50, query_vectors=16, **kw).cache.query is None)          # kapaliyken ok yok
    kapi("Q: oklar bos -> bugunku defter; butun gecmis == k=T", bos and tum and secim)
    kapi("Q: hizli kayip == tam tablo, nedensel, oklar gradyan alir", hizli and nedensel and grad)

    N, data = _veri()
    kok = tempfile.mkdtemp()
    try:
        r = TR.RUNS["QC"] = TR.Run("QC", kok)
        _run(r, data, N, _sifir, "cpu", 2e-3, 3, 0, 8, 3, 3, chain="relative", lam=0.7,
                start_norm=1.0, c_cache=True, **KUCUK)
        yol = kok + "/QC/t3.pt"
        p = torch.load(yol, weights_only=False)
        for a in ("query", "query_vectors", "query_active", "query_by", "c_content", "d_order",
                  "d_content", "lam_w_init", "beta_w_init"):
            del p[a]                                # alanlar gelmeden yazilmis paket
        p["d"] = p.pop("d_sum")                     # eski paket boyutu 'd' diye yazardi
        torch.save(p, yol)
        okur = _PV.from_package(p)
        okur_ok = okur.d_sum == 16 and not okur.c_content and okur.cache.query is None
        r = TR.RUNS["QC"] = TR.Run("QC", kok)       # kapaliyken eski paket SURDURULUR
        _run(r, data, N, _sifir, "cpu", 2e-3, 4, 0, 8, 2, 4, resume=yol, chain="relative",
                lam=0.7, start_norm=1.0, c_cache=True, **KUCUK)
        eski = r.result.get("step") == 4
        try:
            r = TR.RUNS["QC"] = TR.Run("QC", kok)
            _run(r, data, N, _sifir, "cpu", 2e-3, 6, 0, 8, 3, 3, resume=yol, chain="relative",
                    lam=0.7, start_norm=1.0, c_cache=True, c_content=True, **dict(KUCUK, d_order=8, d_content=8))
            yakaladi = False
        except ValueError as h:
            yakaladi = "c_content" in str(h)
        r = TR.RUNS["QC2"] = TR.Run("QC2", kok)
        _run(r, data, N, _sifir, "cpu", 2e-3, 4, 0, 8, 2, 4, chain="relative", lam=0.7,
                start_norm=1.0, c_cache=True, cache_topk=None, query=True, query_vectors=8, query_active=2,
                c_content=True, **dict(KUCUK, d_order=8, d_content=8))
        egitim = r.result.get("step") == 4 and torch.isfinite(r.result["step_losses"]).all()
        r = TR.RUNS["QC3"] = TR.Run("QC3", kok)
        _run(r, data, N, _sifir, "cpu", 2e-3, 6, 0, 8, 2, 4, resume=kok + "/QC2/t4.pt",
                chain="relative", lam=0.7, start_norm=1.0, c_cache=True, cache_topk=None,
                query=True, query_vectors=8, query_active=2, c_content=True, **dict(KUCUK, d_order=8, d_content=8))
        surdu = r.result.get("step") == 6
    finally:
        shutil.rmtree(kok, ignore_errors=True)
    kapi("Q + C_content egitilir, surdurulur; eski paket: surer / farki yakalanir",
         eski and okur_ok and yakaladi and bool(egitim) and surdu,
         "kapaliyken eski paket surer; cache_topk None, 8 ok, d_order 8 + d_content 8")


# --- 20.  SELECT: "direction" katman 1'den itibaren boydan bagimsiz secer; katman 0 ve "distance" ayni
def t_select():
    import model_18 as M
    torch.manual_seed(0)
    w = torch.randint(0, 50, (2, 30))
    kw = dict(d_order=64, t_max=64, lam=0.7, chain="relative", start_norm=1.0)
    uz, yon = PV(50, **kw), PV(50, select="direction", **kw)
    with torch.no_grad():
        for m_ in (uz, yon):
            for L in m_.V:
                L.finish.add_(3.0 * torch.randn_like(L.finish))     # hareket buyuk: boy secimi ezer
    ayni0 = torch.equal(uz.V[0](uz.C(w))[1], yon.V[0](yon.C(w))[1])  # katman 0 C'ye uzakliga gore
    x = yon.V[0](yon.C(w))[0]
    L1 = yon.V[1]
    boydan = torch.equal(L1(x)[1], L1(x, by=5.0 * x)[1])               # yone bakar: 5 kat boy ayni secim
    uz_boy = not torch.equal(uz.V[1](x)[1], uz.V[1](x, by=5.0 * x)[1])  # distance: boy secimi degistirir
    q = PV(50, c_cache=True, query=True, query_vectors=16, query_active=4, select="direction", **kw)
    varsayilan = M.SELECT == "direction" and PV(50, **kw).V[1].direction is False   # tepe yon; SADE uzaklik
    kapi("SELECT direction: katman 1+ boydan bagimsiz, katman 0 ayni",
         ayni0 and boydan and uz_boy and q.cache.query.direction and varsayilan,
         "distance'ta 5 kat boy secimi degistiriyor, direction'da degistirmiyor")

    N, data = _veri()
    kok = tempfile.mkdtemp()
    try:
        r = TR.RUNS["S"] = TR.Run("S", kok)
        _run(r, data, N, _sifir, "cpu", 2e-3, 4, 0, 8, 2, 4, chain="relative", lam=0.7, start_norm=1.0,
             c_cache=True, select="direction", **KUCUK)
        egitim = r.result.get("step") == 4 and torch.isfinite(r.result["step_losses"]).all()
        try:
            r = TR.RUNS["S"] = TR.Run("S", kok)
            _run(r, data, N, _sifir, "cpu", 2e-3, 6, 0, 8, 2, 4, resume=kok + "/S/t4.pt", chain="relative",
                 lam=0.7, start_norm=1.0, c_cache=True, select="distance", **KUCUK)
            yakaladi = False
        except ValueError as h:
            yakaladi = "select" in str(h)
    finally:
        shutil.rmtree(kok, ignore_errors=True)
    kapi("SELECT egitilir; surdurme farki yakalar", bool(egitim) and yakaladi, "direction -> distance")


# --- 21.  HEPSI (ACTIVE = VECTORS): C_m = C + sum_butun softmax(-D^2 e^S_v) V; her vektor gradyan alir
def t_hepsi():
    import model_18 as M
    torch.manual_seed(0)
    w = torch.randint(0, 50, (2, 30))
    m = PV(50, d_order=32, vectors=16, active=16, t_max=64, lam=0.7, chain="relative", start_norm=1.0,
           select="direction")
    with torch.no_grad():
        for L in m.V:
            L.finish.add_(0.5 * torch.randn_like(L.finish))
    x = m.C(w)
    L0, L1 = m.V[0], m.V[1]
    elle = x + torch.softmax(-M.distance(x, L0.start) * L0.S_v.exp(), -1) @ (L0.finish - L0.start)
    y = L0(x)[0]
    yon = M.distance(torch.nn.functional.normalize(y, dim=-1), torch.nn.functional.normalize(L1.start, dim=-1))
    elle1 = y + torch.softmax(-yon * L1.S_v.exp(), -1) @ (L1.finish - L1.start)
    formul = torch.allclose(y, elle, atol=1e-5) and torch.allclose(L1(y)[0], elle1, atol=1e-5)
    m.loss(w).backward()
    hepsi = all(bool(((L.start.grad.abs().sum(-1) > 0) & (L.finish.grad.abs().sum(-1) > 0)).all()) for L in m.V)
    kapi("hepsi: C_m = C + sum softmax V (distance / direction); her vektor gradyan alir", formul and hepsi,
         "16 vektorun 16'si, 2 katman")

    N, data = _veri()
    kok = tempfile.mkdtemp()
    try:
        r = TR.RUNS["H"] = TR.Run("H", kok)
        _run(r, data, N, _sifir, "cpu", 2e-3, 4, 0, 8, 2, 4, chain="relative", lam=0.7, start_norm=1.0,
             c_cache=True, select="direction", **dict(KUCUK, active=8))
        egitim = r.result.get("step") == 4 and torch.isfinite(r.result["step_losses"]).all()
    finally:
        shutil.rmtree(kok, ignore_errors=True)
    kapi("hepsi egitilir (8 vektorun 8'i aktif)", bool(egitim))


# --- 22.  LOAD BALANCE: 0 -> kayip ayni; esit kullanimda terim 1; top-k'da secilmeyen start'lar gradyan alir
def t_balance():
    import model_18 as M
    torch.manual_seed(0)
    w = torch.randint(0, 50, (2, 12))                              # az konum, cok vektor: secilmeyen kalir
    kw = dict(d_order=32, vectors=64, active=2, t_max=64, lam=0.7, chain="relative", start_norm=1.0,
              c_cache=True, select="direction")
    a, b = PV(50, **kw), PV(50, load_balance=0.5, **kw)
    with torch.no_grad():
        for m_ in (a, b):
            for L in m_.V:
                L.finish.add_(0.5 * torch.randn(L.finish.shape, generator=torch.Generator().manual_seed(1)))
        C = b.C(w)
        C_m, ids, P = b._layers(C, probs=True)
        say = torch.ones(2, 11, dtype=torch.bool)
        terim = float(b.balance(P, say))
        esit = float(b.balance([torch.full((2, 12, 64), 1 / 64)], say))
        fark = float(b.loss(w)) - float(a.loss(w))
    ayni_0 = torch.equal(a._layers(C)[0], C_m)                      # probs ciktiyi degistirmez
    b.loss(w).backward()
    L = b.V[1]
    secilen = set(ids[1][:, :-1].flatten().tolist())
    olu = [i for i in range(64) if i not in secilen]
    olu_grad = bool(olu) and bool((L.start.grad[olu].abs().sum(-1) > 0).all())
    a.loss(w).backward()
    olu_0 = bool((a.V[1].start.grad[olu].abs().sum(-1) == 0).all()) if olu else True
    kapi("load balance: esitte 1, kayip + katsayi*terim, secilmeyen start gradyan alir",
         abs(esit - 1) < 1e-9 and abs(fark - 0.5 * terim) < 1e-4 and terim >= len(P) - 1e-6
         and ayni_0 and olu_grad and olu_0 and M.LOAD_BALANCE > 0,
         "terim %.2f (4 katman, en az 4); secilmeyen %d start: dengeyle gradyan var, dengesiz yok"
         % (terim, len(olu)))

    N, data = _veri()
    kok = tempfile.mkdtemp()
    try:
        r = TR.RUNS["LB"] = TR.Run("LB", kok)
        _run(r, data, N, _sifir, "cpu", 2e-3, 4, 0, 8, 2, 4, chain="relative", lam=0.7, start_norm=1.0,
             c_cache=True, select="direction", load_balance=0.01, **KUCUK)
        egitim = r.result.get("step") == 4 and torch.isfinite(r.result["step_losses"]).all()
        try:
            r = TR.RUNS["LB"] = TR.Run("LB", kok)
            _run(r, data, N, _sifir, "cpu", 2e-3, 6, 0, 8, 2, 4, resume=kok + "/LB/t4.pt", chain="relative",
                 lam=0.7, start_norm=1.0, c_cache=True, select="direction", **KUCUK)
            yakaladi = False
        except ValueError as h:
            yakaladi = "load_balance" in str(h)
    finally:
        shutil.rmtree(kok, ignore_errors=True)
    kapi("load balance egitilir; surdurme farki yakalar", bool(egitim) and yakaladi, "0,01 -> 0")


# ============================================================
# DATA_STORIES -- TinyStories (model_17 veri_t17 + olcme_17'den).
# ============================================================
def t_stories():
    import numpy as np
    import data_stories as DS
    hk, dl = 1, 0
    a = np.array([5, 6, hk, 7, 8, 9, hk, 3, hk], dtype=np.int16)
    P, M = DS.pencere(a, 5, None, hk, dl)
    ok = (len(P) == 3 and list(M.sum(1)) == [4, 5, 3]
          and list(P[1]) == [hk, 7, 8, 9, hk] and (P[~M] == dl).all())
    P4, _ = DS.pencere(a, 4, None, hk, dl)           # L+2 > T olan ATILIR
    kapi("hikaye penceresi <eos> basta ve sonda", ok and len(P4) == 2,
         "%d pencere; T=4'te %d" % (len(P), len(P4)))

    # olc: bantlar ve eos_ok, elle sayimla ayni
    torch.manual_seed(0)
    m = PV(12, t_max=16, **KUCUK)
    W = torch.randint(3, 12, (6, 16))
    Mk = torch.zeros(6, 16, dtype=torch.bool)
    for i, L in enumerate((16, 9, 12, 5, 16, 7)):
        W[i, 0], W[i, L - 1] = hk, hk
        W[i, L:] = dl
        Mk[i, :L] = True
    DS.BANTLAR, eski = ((0, 4), (4, 10), (10, 16)), DS.BANTLAR
    try:
        r = DS.olc(m, W, Mk, hk, aygit="cpu", parca=4)
    finally:
        DS.BANTLAR = eski
    with torch.no_grad():
        d = (m.scoreboard(W)[:, :-1].argmax(-1) == W[:, 1:]) & Mk[:, 1:]
    k = torch.arange(1, 16)
    bek = [float(d[:, (k >= a_) & (k < b_)].sum() / Mk[:, 1:][:, (k >= a_) & (k < b_)].sum())
           for a_, b_ in ((0, 4), (4, 10), (10, 16))]
    e = Mk[:, 1:] & (W[:, 1:] == hk)
    ok = (abs(r["accuracy"] - float(d.sum() / Mk[:, 1:].sum())) < 1e-9
          and all(abs(r["diag"][n] - b) < 1e-9 for n, b in
                  zip(("acc_0_4", "acc_4_10", "acc_10_16"), bek))
          and abs(r["diag"]["eos_ok"] - float(d[e].sum() / e.sum())) < 1e-9)
    kapi("hikaye olcutu: accuracy, bantlar, eos_ok", ok,
         "accuracy %.3f  eos_ok %.3f" % (r["accuracy"], r["diag"]["eos_ok"]))


# ============================================================
# DATA -- matematik (model_17'den).
# ============================================================

# --- 11.  PENCERE ve HEDEF MASKESI
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

    def scoreboard(self, w, mask=None):
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


# --- 12.  OLCUT: dogruyu 1, erken durani ve fazla yazani 0 saymali
def t_mat_sor():
    s = [(1, 1), (21, 23), (472, 182), (1, 1, 5), (23, 1, 120), (500, 500, 500)]
    d, e, f = (VM.sor(_Kahin(k), s, aygit="cpu") for k in ("dogru", "erken", "fazla"))
    alan = lambda r: (r["accuracy"], r["length_ok"], r["first_digit"])
    ok = (alan(d) == (1.0, 1.0, 1.0) and alan(e) == (0.0, 0.0, 0.0)
          and alan(f) == (0.0, 0.0, 1.0))
    kapi("sor: dogru 1, erken 0, fazla rakam 0", ok,
         "dogru %s  erken %s  fazla %s" % tuple(alan(r) for r in (d, e, f)))


# --- 13.  BASAMAK: saga hizali, birler AYRI okunur
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


# --- 14.  CEVAP MASKELI EGITIM: surdurme == kesintisiz, her adimin kaybi eksiksiz
def t_mat_egitim():
    g = torch.Generator().manual_seed(21)
    sorular = [tuple(x) for x in torch.randint(0, 60, (64, 2), generator=g).tolist()]
    W, M, H = VM.pencereler(sorular)

    def kos(ad, kok, adim, surdur=None):
        r = _kos(ad, kok, (W, M, H), VM.N, adim, resume=surdur)
        return _agirlik(r), r.result["step_losses"]

    kok = tempfile.mkdtemp()
    try:
        A, ka = kos("KESINTISIZ", kok, 8)
        kos("BOLUK", kok, 4)
        B, kb = kos("BOLUK", kok, 8, surdur=kok + "/BOLUK/t4.pt")
        en = max(float((A[k] - B[k]).abs().max()) for k in A)
    finally:
        shutil.rmtree(kok, ignore_errors=True)
    kapi("cevap maskeli egitim, surdurme", en == 0.0, "fark %.3e" % en)
    kapi("her adimin kaybi: surdurmede eksiksiz",
         len(ka) == 9 and not bool(ka.isnan().any()) and torch.equal(ka, kb),
         "%d adim, surdurulen == kesintisiz" % len(ka))


# --- 15.  NOTEBOOK: kural 2 ve kural 8 MEKANIK
# GPU hucresi GPU'yu kendi icinde sorar; her kosu kendi hucresinde baslar;
# hucrelerde tanimsiz ad yok (Colab'da NameError bir gidis-donus demek).
def t_notebook(yol=None):
    import ast
    import builtins
    import json
    yol = yol or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "notebook.ipynb")
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
        if py.count(".start(") > 1 or (".start(" in py and not gpu):
            kusur.append("start kurali: " + kunye[:30])
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
    kapi("notebook: GPU kapisi, tek kosu, tanimsiz ad", not kusur,
         "; ".join(kusur) if kusur else "%d kod hucresi" % len(hucre))


if __name__ == "__main__":
    print("tests (model_18)")
    for f in (t_zincir, t_nedensel, t_sessiz, t_cm, t_payda, t_gradyan,
              t_parametre, t_mask, t_surdurme, t_durdur, t_skor, t_start, t_lam, t_relative, t_ccache, t_init, t_content, t_query, t_tepe, t_select, t_hepsi, t_balance, t_mat_pencere,
              t_mat_sor, t_mat_basamak, t_mat_egitim, t_stories, t_notebook):
        f()
    t_notebook(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "notebook_stories.ipynb"))
    print("\n%d GECTI   %d KALDI" % (len(GECTI), len(KALDI)))
    sys.exit(1 if KALDI else 0)
