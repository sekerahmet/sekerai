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

# BELIRLENIMCI KOSU: bit-esitlik kapilari (surdurme == kesintisiz) icin.  Eskiden model_18'in import yan etkisiydi
# ve onu import eden her betigi tek iplige dusuruyordu (hakem P9).
if not torch.cuda.is_available():
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as VM                                               # noqa: E402
import train as TR                                              # noqa: E402
from model_18 import PV as _PV                                  # noqa: E402

PASSED, FAILED = [], []
SMALL = dict(d_order=16, d_content=0, vectors=8, active=2, layers=2)   # egitim kapilari icin kucuk PV
# Kapilar parcalari TEK TEK sinar: tepedeki tasarimin (model_18) ozellikleri burada KAPALI baslar,
# her kapi sinadigini acar.  Tepenin kendisi t_tepe'de.
PLAIN = dict(d_order=128, d_content=0, t_max=64, start_norm=None, lam=1.0, chain="absolute",
            c_cache=False, cache_topk=8, query=False, c_content=False, select="distance", active=8, load_balance=0.0, c_m_norm=False,
            attention=False)


def PV(n, **cfg):
    return _PV(n, **dict(PLAIN, **cfg))


def _run(*a, **cfg):
    return TR._run(*a, **dict(PLAIN, **cfg))


def check(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print("  %-42s %s  %s" % (name, "GECTI" if cond else "KALDI", detail))


def _seq(s):
    return torch.tensor([VM.VOCAB.index(a) if a != "E" else VM.EOS for a in s])


def _run_small(name, root, data, n_vocab, steps, metric=None, resume=None, lr=2e-3):
    """_run'i kucuk PV ile, ayni iplikte kosturur; Run'i doner."""
    r = TR.RUNS[name] = TR.Run(name, root)
    _run(r, data, n_vocab, metric or _zero_metric, "cpu", lr, steps, 0, 8,
            2, 2, resume=resume, **SMALL)
    return r


def _zero_metric(m, side, full=False):
    return {"accuracy": 0.0}


def _weights_of(r):
    return {k: v.clone() for k, v in r.result["model"].state_dict().items()}


# ============================================================
# PV -- model_18.py'nin iddialari tek tek.
# ============================================================

# --- 1.  ZINCIR: yeni token yalniz bir terim ekler, eski terimler degismez
def t_chain():
    m, w = PV(VM.N), _seq("E31+52=8")
    C = m.C(w[None])[0]
    ok = all(torch.allclose(C[t] - m.RM[t] * m.P[w[t]], C[t - 1], atol=1e-6)
             for t in range(1, len(w)))
    check("zincir: C_t - RM_t*P[w_t] == C_(t-1)", ok, "%d halka" % len(w))


# --- 2.  NEDENSEL: sonraki token onceki konumlarin puanini degistirmez
def t_causal():
    m = PV(VM.N)
    a, b = m.scoreboard(_seq("E31+52=8")[None]), m.scoreboard(_seq("E31+52=9")[None])
    check("nedensel", torch.equal(a[0, :-1], b[0, :-1]), "son token degisti, oncekiler ayni")


# --- 3.  SESSIZ BASLANGIC: V_a = 0, C yerinde kalir
def t_silent():
    m, w = PV(VM.N), _seq("E31+52=8")[None]
    check("baslangicta C_m == C", torch.equal(m.move(w)[0], m.C(w)))


# --- 4.  CM: geriye yuruyus zincirin halkalarini birebir verir
def t_cm():
    m, w = PV(VM.N), _seq("E31+52=8")
    C = m.C(w[None])[0]
    path = m.CM(w)
    ok = (len(path) == len(w) and all(torch.allclose(c, C[t], atol=1e-5) for t, c in path)
          and [t for t, _ in m.CM(w, K=3)] == [7, 6, 5])
    check("CM == zincir, K ile sinir", ok, "%d adim; K=3 -> 7 6 5" % len(path))


# --- 5.  PAYDA: W_v toplami 1 -- butun vektorler ayni V_a ise tasima tam V_a
def t_denominator():
    m, w = PV(VM.N), _seq("E31+52=8")[None]
    u = torch.randn(m.d_sum, generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        for L in m.V:
            L.finish.copy_(L.start + u)
        diff = float((m.move(w)[0] - m.C(w) - m.layers * u).abs().max())
    check("payda: W_v toplami 1", diff < 1e-4, "fark %.1e" % diff)


# --- 6.  GRADYAN yalniz AKTIF vektorlere (son konumun hedefi yok)
def t_gradient():
    m, w = PV(VM.N), _seq("E31+52=8")[None]
    with torch.no_grad():
        for L in m.V:
            L.finish.add_(0.01 * torch.randn(L.finish.shape,
                                             generator=torch.Generator().manual_seed(6)))
    m.loss(w).backward()
    ok = True
    for L, ids in zip(m.V, m.trace(w)):
        g = (L.start.grad.abs().sum(-1) + L.finish.grad.abs().sum(-1)) > 0
        ok &= set(g.nonzero().flatten().tolist()) == set(ids[0, :-1].flatten().tolist())
    check("gradyan yalniz aktif vektorlere", ok, "%d layer" % m.layers)


# --- 7.  PARAMETRE == hesap: vectors x 2 x d x layers + layers (S_v)
def t_params():
    m = PV(VM.N)
    expected = m.vectors * 2 * m.d_sum * m.layers + m.layers
    n_par = sum(p.numel() for p in m.parameters())
    buffers = sorted(a for a, _ in m.named_buffers())
    check("parametre == hesap; P ve RM sabit", n_par == expected and buffers == ["P", "RM"],
         "%s / %s" % (f"{n_par:,}", f"{expected:,}"))


# --- 8.  MASKE: maskeli kayip == soru soru
def t_mask():
    m = PV(20, **SMALL)
    lengths = [12, 7, 16]
    W = torch.zeros(3, 16, dtype=torch.long)
    M = torch.zeros(3, 16, dtype=torch.bool)
    g = torch.Generator().manual_seed(3)
    for i, L in enumerate(lengths):
        W[i, :L] = torch.randint(1, 20, (L,), generator=g)
        M[i, :L] = True
    with torch.no_grad():
        a = float(m.loss(W, M))
        b = sum(float(m.loss(W[i:i + 1, :L])) * (L - 1) for i, L in enumerate(lengths))
        b /= sum(L - 1 for L in lengths)
    check("maskeli kayip == soru soru", abs(a - b) < 1e-5, "%.6f / %.6f" % (a, b))


# ============================================================
# TRAIN -- surdurme ve durdurma (model_17'den).
# ============================================================
def _data():
    """Kucuk rastgele egitim verisi: (N, (W, M, H)); her konum hedef."""
    N, T, n = 20, 16, 64
    g = torch.Generator().manual_seed(7)
    M = torch.ones(n, T, dtype=torch.bool)
    return N, (torch.randint(0, N, (n, T), generator=g), M, M)


# --- 9.  SURDURME == KESINTISIZ (kural 1: uzatma surdurmedir)
def t_resume():
    N, data = _data()
    root = tempfile.mkdtemp()
    try:
        A = _weights_of(_run_small("KESINTISIZ", root, data, N, 8))
        _run_small("BOLUK", root, data, N, 4)
        B = _weights_of(_run_small("BOLUK", root, data, N, 8, resume=root + "/BOLUK/t4.pt"))
        max_diff = max(float((A[k] - B[k]).abs().max()) for k in A)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    check("surdurme == kesintisiz", max_diff == 0.0, "fark %.3e" % max_diff)


# --- 10.  DURDUR: son tamamlanan adim yazilir; durdur + surdur == kesintisiz
def t_stop():
    N, data = _data()
    root = tempfile.mkdtemp()

    def run_(name, steps, resume_from=None, stop_at=None, lr=2e-3):
        r = TR.RUNS[name] = TR.Run(name, root)
        count = [0]

        def metric(m, side, full=False):
            count[0] += 1
            if stop_at is not None and count[0] == 2 * (stop_at + 1):
                r.stop()                        # `dur` adiminin olcumu bitti
            return {"accuracy": 0.0}

        _run(r, data, N, metric, "cpu", lr, steps, 0, 8, 1, 3,
                resume=resume_from, **SMALL)
        return _weights_of(r)

    try:
        A = run_("KESINTISIZ", 8)
        run_("DUR", 8, stop_at=4)
        pt = sorted(f for f in os.listdir(root + "/DUR") if f.endswith(".pt"))
        B = run_("DUR", 8, resume_from=root + "/DUR/t4.pt")
        max_diff = max(float((A[k] - B[k]).abs().max()) for k in A)
        try:
            run_("DUR", 8, resume_from=root + "/DUR/t4.pt", lr=4e-3)
            cfg_caught = False
        except ValueError as h:
            cfg_caught = "lr" in str(h)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    check("durdur: son TAMAMLANAN adim yazilir",
         pt == ["t0.pt", "t3.pt", "t4.pt"], " ".join(pt))
    check("durdur + surdur == kesintisiz", max_diff == 0.0, "fark %.3e" % max_diff)
    check("surdurme ayar farkini yakalar", cfg_caught, "lr 2e-3 -> 4e-3")


# --- 11.  SKOR: sozluk tablosu, karesiz == -S_p*D, ogrenilen S_p, surdurme skoru ayirt eder
def t_score():
    import model_18 as M
    rule_ = [M.score_rule(n) for n in (13, 49, 50, 99, 100, 499, 500, 1999, 2000, 50000)]
    expected = [(False, 10.0), (False, 10.0), (False, 5.0), (False, 5.0), (False, 2.0),
           (False, 2.0), (True, 2.0), (True, 2.0), (True, M.LEARNED), (True, M.LEARNED)]
    check("skor tablosu: sinir ust satira", rule_ == expected, "13 -> karesiz 10, 2000 -> ogrenilen")

    torch.manual_seed(0)
    sq = PV(13, squared=True, S_p=1.0, **SMALL)
    nonsq = PV(13, **SMALL)                                  # tablodan: karesiz, S_p 10
    C_m = torch.randn(4, 16)
    D = torch.cdist(C_m, sq.P)
    diff = float((nonsq.score(C_m) + 10.0 * D).abs().max())
    diff2 = float((sq.score(C_m) + D * D).abs().max())
    order_ok = torch.equal(sq.score(C_m).argsort(-1), nonsq.score(C_m).argsort(-1))
    check("karesiz == -10*D, kare == -D^2", diff < 1e-4 and diff2 < 1e-3 and order_ok,
         "fark %.1e / %.1e, sira ayni" % (diff, diff2))

    m = PV(2500, **SMALL)
    m.score(C_m).sum().backward()
    ok = (m.squared and m.S_p_learned and float(m.S_p) == 0.0 and m.S_p.grad is not None
          and sum(p.numel() for p in m.parameters())
          == m.vectors * 2 * m.d_sum * m.layers + m.layers + 1)
    check("ogrenilen S_p: e^0 = 1, gradyan alir", ok, "parametre +1")

    N, data = _data()
    root = tempfile.mkdtemp()
    try:
        r = TR.RUNS["KARE"] = TR.Run("KARE", root)
        _run(r, data, N, _zero_metric, "cpu", 2e-3, 3, 0, 8, 3, 3,
                squared=True, S_p=1.0, **SMALL)
        path = root + "/KARE/t3.pt"
        previous = torch.load(path, weights_only=False)
        del previous["S_p"], previous["squared"]         # alanlar gelmeden yazilmis paket
        torch.save(previous, path)
        r = TR.RUNS["KARE"] = TR.Run("KARE", root)
        _run(r, data, N, _zero_metric, "cpu", 2e-3, 6, 0, 8, 3, 3,
                resume=path, squared=True, S_p=1.0, **SMALL)
        accepted = r.result.get("step") == 6
        try:
            r = TR.RUNS["KARE"] = TR.Run("KARE", root)
            _run(r, data, N, _zero_metric, "cpu", 2e-3, 9, 0, 8, 3, 3,
                    resume=path, **SMALL)             # tablodan: karesiz 10
            caught = False
        except ValueError as h:
            caught = "S_p" in str(h) and "squared" in str(h)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    check("surdurme skor farkini yakalar", accepted and caught,
         "eski kare paket: kare ile surer, tabloyla durur")


# --- 12.  START BOYU: start_norm ile secimi C belirler; surdurme farki yakalar
def t_start():
    torch.manual_seed(0)
    previous = PV(4003, d_order=256, d_content=0, vectors=64, t_max=256)
    new_ = PV(4003, d_order=256, d_content=0, vectors=64, t_max=256, start_norm=1.0)
    L = new_.V[0]
    median_norm = float(L.start.detach().norm(dim=-1).median())
    same = torch.equal(previous.V[0].start * (1.0 / 256 ** 0.5), L.start) and torch.equal(L.start, L.finish)
    w = torch.randint(3, 4003, (8, 200))
    C = new_.C(w).reshape(-1, 256)
    short_ = set(L.start.norm(dim=-1).argsort()[:L.active].tolist())

    def short_share(Ls):
        from model_18 import distance
        selected_ids = distance(C, Ls.start).topk(Ls.active, dim=-1, largest=False).indices
        k = set(Ls.start.norm(dim=-1).argsort()[:Ls.active].tolist())
        return sum(len(k & set(r)) for r in selected_ids.tolist()) / selected_ids.numel()
    a, b = short_share(previous.V[0]), short_share(L)
    check("start_norm: boy 1, secimi C belirler", abs(median_norm - 1) < 0.1 and same and b < a / 3,
         "boy %.2f; en kisa 8'in payi randn %%%.0f -> %%%.0f" % (median_norm, 100 * a, 100 * b))

    N, data = _data()
    root = tempfile.mkdtemp()
    try:
        r = TR.RUNS["S"] = TR.Run("S", root)
        _run(r, data, N, _zero_metric, "cpu", 2e-3, 3, 0, 8, 3, 3, **SMALL)
        path = root + "/S/t3.pt"
        p = torch.load(path, weights_only=False)
        del p["start_norm"]                      # alan gelmeden yazilmis paket
        torch.save(p, path)
        try:
            r = TR.RUNS["S"] = TR.Run("S", root)
            _run(r, data, N, _zero_metric, "cpu", 2e-3, 6, 0, 8, 3, 3, resume=path,
                    start_norm=1.0, **SMALL)
            caught = False
        except ValueError as h:
            caught = "start_norm" in str(h)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    check("surdurme start_norm farkini yakalar", caught, "eski randn paket -> start_norm 1")


# --- 13.  LAM: C_t = lam*C_(t-1) + RM_t*P[w_t]; CM geri yurur; nedensel; surdurme farki
def t_lam():
    m = PV(VM.N, lam=0.9)
    w = _seq("E31+52=8")
    C = m.C(w[None])[0]
    chain_ok = all(torch.allclose(C[t] - m.RM[t] * m.P[w[t]], 0.9 * C[t - 1], atol=1e-5)
                 for t in range(1, len(w)))
    cm = all(torch.allclose(c, C[t], atol=1e-4) for t, c in m.CM(w))
    one_eq = torch.equal(PV(VM.N, lam=1.0).C(w[None]), PV(VM.N).C(w[None]))
    a, b = m.scoreboard(_seq("E31+52=8")[None]), m.scoreboard(_seq("E31+52=9")[None])
    causal = torch.equal(a[0, :-1], b[0, :-1])
    long_ = PV(50, d_order=32, d_content=0, t_max=512, lam=0.7).C(torch.randint(0, 50, (2, 512)))
    check("lam: zincir, CM, nedensel, lam 1 == cumsum", chain_ok and cm and one_eq and causal
         and bool(torch.isfinite(long_).all()), "0,9; T=512'de lam 0,7 sonlu")

    N, data = _data()
    root = tempfile.mkdtemp()
    try:
        r = TR.RUNS["L"] = TR.Run("L", root)
        _run(r, data, N, _zero_metric, "cpu", 2e-3, 3, 0, 8, 3, 3, **SMALL)
        path = root + "/L/t3.pt"
        p = torch.load(path, weights_only=False)
        del p["lam"]                             # alan gelmeden yazilmis paket
        torch.save(p, path)
        try:
            r = TR.RUNS["L"] = TR.Run("L", root)
            _run(r, data, N, _zero_metric, "cpu", 2e-3, 6, 0, 8, 3, 3, resume=path, lam=0.9, **SMALL)
            caught = False
        except ValueError as h:
            caught = "lam" in str(h)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    check("surdurme lam farkini yakalar", caught, "eski paket (lam 1) -> lam 0,9")


# --- 14.  RELATIVE: C_t = lam*kaydir(C_(t-1)) + P[w_t]; ayni baglam her konumda ayni nokta
def t_relative():
    m = PV(VM.N, lam=0.7, chain="relative")
    w = _seq("E31+52=8")
    C = m.C(w[None])[0]
    chain_ok = torch.allclose(C[0], m.P[w[0]], atol=1e-6) and all(
        torch.allclose(C[t], 0.7 * torch.roll(C[t - 1], 1) + m.P[w[t]], atol=1e-5)
        for t in range(1, len(w)))
    cm = all(torch.allclose(c, C[t], atol=1e-4) for t, c in m.CM(w))
    a, b = m.scoreboard(_seq("E31+52=8")[None]), m.scoreboard(_seq("E31+52=9")[None])
    causal = torch.equal(a[0, :-1], b[0, :-1])
    # ayni son 3 token, farkli konum: solma sayesinde C'ler neredeyse ayni (absolute'ta degil)
    g = torch.Generator().manual_seed(9)
    prefix = torch.randint(0, 50, (40,), generator=g)
    three = torch.tensor([7, 8, 9])
    short_, long_ = torch.cat([prefix[:5], three]), torch.cat([prefix, three])
    cos = {}
    for chain_type in ("relative", "absolute"):
        mm = PV(50, d_order=256, d_content=0, t_max=64, lam=0.5, chain=chain_type)
        cos[chain_type] = float(torch.cosine_similarity(mm.C(short_[None])[0, -1], mm.C(long_[None])[0, -1], dim=0))
    check("relative: zincir, CM, nedensel, konumdan bagimsiz",
         chain_ok and cm and causal and cos["relative"] > 0.9 and cos["absolute"] < 0.5,
         "ayni son 3 kelime, 8. ve 43. konum: cos relative %.2f, absolute %.2f"
         % (cos["relative"], cos["absolute"]))

    N, data = _data()
    root = tempfile.mkdtemp()
    try:
        r = TR.RUNS["R"] = TR.Run("R", root)
        _run(r, data, N, _zero_metric, "cpu", 2e-3, 3, 0, 8, 3, 3, **SMALL)
        path = root + "/R/t3.pt"
        p = torch.load(path, weights_only=False)
        del p["chain"]                           # alan gelmeden yazilmis paket
        torch.save(p, path)
        try:
            r = TR.RUNS["R"] = TR.Run("R", root)
            _run(r, data, N, _zero_metric, "cpu", 2e-3, 6, 0, 8, 3, 3, resume=path,
                    chain="relative", **SMALL)
            caught = False
        except ValueError as h:
            caught = "chain" in str(h)
        r = TR.RUNS["R2"] = TR.Run("R2", root)
        _run(r, data, N, _zero_metric, "cpu", 2e-3, 4, 0, 8, 2, 4, chain="relative", lam=0.7,
                start_norm=1.0, **SMALL)
        trained = r.result.get("step") == 4
    finally:
        shutil.rmtree(root, ignore_errors=True)
    check("relative egitilir; surdurme chain farkini yakalar", caught and trained,
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
    causal = torch.allclose(a[0, :-1], m.scoreboard(w2)[0, :-1], atol=1e-6)
    normal = float((a.logsumexp(-1)).abs().max()) < 1e-4
    plain = PV(50, c_cache=False, **kw)
    with torch.no_grad():
        m.cache.gate_0.fill_(-60.0)
        gate_off_ok = torch.allclose(m.scoreboard(w), torch.log_softmax(plain.scoreboard(w), -1), atol=1e-4)
        m.cache.gate_0.fill_(60.0)
        series = torch.randperm(50, generator=g)[:12]
        repeated = torch.cat([series, series[:9]])[None]            # ikinci kez: ...x9 -> x10 bekleniyor
        copy_ok = int(m.scoreboard(repeated)[0, -1].argmax()) == int(series[9])
        m.cache.gate_0.fill_(-1.0)
        M = torch.ones(1, 30, dtype=torch.bool); M[0, 20:] = False
        full_res = float(torch.nn.functional.cross_entropy(m.scoreboard(w)[0, :-1], w[0, 1:],
                                                      reduction="none")[:19].mean())
        fast = abs(float(m.loss(w, M)) - full_res) < 1e-5
    check("c_cache: nedensel, log p, gate 0 == model, kopya", causal and normal and gate_off_ok and copy_ok,
         "tekrar eden dizide gate 1 -> sonraki kelime defterden")
    check("c_cache hizli kayip == tam tablo", fast, "egitim yolu (B,T,n) kurmadan ayni kayip")

    N, data = _data()
    root = tempfile.mkdtemp()
    try:
        r = TR.RUNS["CC"] = TR.Run("CC", root)
        _run(r, data, N, _zero_metric, "cpu", 2e-3, 3, 0, 8, 3, 3, **SMALL)
        path = root + "/CC/t3.pt"
        p = torch.load(path, weights_only=False)
        del p["c_cache"]                         # alan gelmeden yazilmis paket
        torch.save(p, path)
        try:
            r = TR.RUNS["CC"] = TR.Run("CC", root)
            _run(r, data, N, _zero_metric, "cpu", 2e-3, 6, 0, 8, 3, 3, resume=path, c_cache=True, **SMALL)
            caught = False
        except ValueError as h:
            caught = "c_cache" in str(h)
        r = TR.RUNS["CC2"] = TR.Run("CC2", root)
        _run(r, data, N, _zero_metric, "cpu", 2e-3, 4, 0, 8, 2, 4, chain="relative", lam=0.7,
                start_norm=1.0, c_cache=True, **SMALL)
        trained = r.result.get("step") == 4 and torch.isfinite(r.result["step_losses"]).all()
    finally:
        shutil.rmtree(root, ignore_errors=True)
    check("c_cache egitilir; surdurme farki yakalar", caught and bool(trained),
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
    N, data = _data()
    root = tempfile.mkdtemp()
    try:
        r = TR.RUNS["I"] = TR.Run("I", root)
        _run(r, data, N, _zero_metric, "cpu", 2e-3, 3, 0, 8, 3, 3, **SMALL)
        try:
            r = TR.RUNS["I"] = TR.Run("I", root)
            _run(r, data, N, _zero_metric, "cpu", 2e-3, 6, 0, 8, 3, 3, resume=root + "/I/t3.pt",
                    s_v_init=0.5, **SMALL)
            caught = False
        except ValueError as h:
            caught = "s_v_init" in str(h)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    check("baslangic ayarlari: tepeden, kosudan, surdurmede", ok and caught,
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
    h, steps = torch.zeros(2, dc), []
    for t in range(w.shape[1]):
        h = lam[:, t] * h + beta[:, t] * m.P[w[:, t], d - dc:]
        steps.append(h)
    content_ok = torch.allclose(C[..., d - dc:], torch.stack(steps, 1), atol=1e-5)
    rel = PV(n, d_order=d - dc, d_content=0, t_max=64, lam=0.7, chain="relative")
    rel.P = m.P[:, :d - dc]
    order_ok = torch.allclose(C[..., :d - dc], rel.C(w), atol=1e-5)
    w2 = w.clone(); w2[:, 30] = (w2[:, 30] + 1) % n
    causal = torch.equal(C[:, :30], m.C(w2)[:, :30])
    with torch.no_grad():
        m.lam_w.clamp_(0, 3)                    # CM lam'a boler: cok kucuk lam sayisal hatayi buyutur
    cm = all(torch.allclose(c, m.C(w[:1])[0, t], atol=1e-3) for t, c in m.CM(w[0], K=10))
    m.loss(w).backward()
    grad = bool(m.lam_w.grad.abs().sum() > 0 and m.beta_w.grad.abs().sum() > 0)
    check("C_content: parca == adim adim, C_order == relative", content_ok and order_ok and causal,
         "45 adim, 16'lik parcalar, lam_w alt sinirda (e^-5) dahil")
    check("C_content: CM geri yuruyus, lam_w/beta_w gradyan alir", cm and grad)

    torch.manual_seed(0)
    s = PV(n, t_max=64, lam=0.7, chain="relative", c_content=True, d_order=d - dc, d_content=dc, content_scalar=True)
    with torch.no_grad():
        s.lam_w.normal_(0, 3); s.beta_w.normal_(0, 1)
    Cs, (ls, bs) = s.C(w), s.lam_beta(w)
    h, steps = torch.zeros(2, dc), []
    for t in range(w.shape[1]):
        h = ls[:, t] * h + bs[:, t] * s.P[w[:, t], d - dc:]
        steps.append(h)
    with torch.no_grad():
        s.lam_w.clamp_(0, 3)
    scalar_ok = (s.lam_w.shape == (n, 1) and s.beta_w.shape == (n, 1)
              and torch.allclose(Cs[..., d - dc:], torch.stack(steps, 1), atol=1e-5)
              and all(torch.allclose(c, s.C(w[:1])[0, t], atol=1e-3) for t, c in s.CM(w[0], K=10)))
    check("C_content kelime basina tek deger: parca == adim adim, CM", scalar_ok,
         "lam_w, beta_w (%d, 1): %d parametre" % (n, 2 * n))

    no_content = PV(n, d_order=d, d_content=0, t_max=64, lam=0.7, chain="relative")
    model_off = PV(n, t_max=64, lam=0.7, chain="relative", d_order=d - dc, d_content=dc)
    same = torch.equal(no_content.C(w), model_off.C(w)) and not hasattr(model_off, "lam_w")
    base = _PV(n, t_max=64, c_cache=False)                  # boyutlar tepeden
    l0, b0 = base.lam_beta(w)
    initial = (abs(float(l0.mean()) - M.LAM_W_INIT) < 1e-5 and abs(float(b0.mean()) - M.BETA_W_INIT) < 1e-5
                 and (base.d_order, base.d_content, base.d_sum) == (M.D_ORDER, M.D_CONTENT, M.D_SUM))
    check("C_content kapali == bugunku zincir; ayarlar tepeden", same and initial,
         "D_SUM %d = %d + %d, lam_w %.2f beta_w %.2f"
         % (base.d_sum, base.d_order, base.d_content, float(l0.mean()), float(b0.mean())))


# --- 19.  TEPE = MODELIN SON YAPISI: ayarsiz PV, tepedeki tasarimi kurar
def t_top():
    import model_18 as M
    m = _PV(60)
    ok = (m.chain == "relative" and m.lam == M.LAM and m.c_content and m.d_sum == M.D_SUM
          and m.cache is not None and m.cache.query is None and m.cache.topk == M.CACHE_TOPK == 8
          and m.t_max == M.T_MAX and hasattr(m, "lam_w") and m.select == "direction_all"
          and all(L.active == M.ACTIVE == 8 and L.direction for L in m.V)
          and m.c_m_norm and m.attn is not None)
    check("tepe: ayarsiz PV == en iyi olculen yapi", ok,
         "chain %s lam %g, D_SUM %d = %d + %d, defter topk %s, Q yok, T_MAX %d; aktif %d, secim %s, c_m_norm, attention"
         % (m.chain, m.lam, m.d_sum, m.d_order, m.d_content, m.cache.topk, m.t_max, M.ACTIVE, m.select))


# --- 18.  Q: oklar bosken Q == C (defter aynen); butun gecmis == k=T; gradyan oklara ulasir
def t_query():
    torch.manual_seed(0)
    kw = dict(d_order=64, d_content=0, t_max=64, lam=0.5, chain="relative", start_norm=1.0, c_cache=True)
    w = torch.randint(0, 50, (2, 40))
    a = PV(50, **kw)
    b = PV(50, query=True, query_vectors=16, query_active=4, **kw)
    empty_eq = torch.allclose(a.scoreboard(w), b.scoreboard(w), atol=1e-5)
    full_hist = PV(50, cache_topk=None, query=True, query_vectors=16, query_active=4, **kw)
    kT = PV(50, cache_topk=40, query=True, query_vectors=16, query_active=4, **kw)
    all_ok = torch.allclose(full_hist.scoreboard(w), kT.scoreboard(w), atol=1e-5)
    with torch.no_grad():
        full_hist.cache.gate_0.fill_(0.0)
        full_hist.cache.query.finish.add_(0.3 * torch.randn_like(full_hist.cache.query.finish))
        M_ = torch.ones_like(w, dtype=torch.bool); M_[1, 25:] = False
        lp = full_hist.scoreboard(w)[:, :-1]
        ce = torch.nn.functional.cross_entropy(lp.transpose(1, 2), w[:, 1:], reduction="none")
        full_res = float(ce[M_[:, 1:]].mean())
    fast = abs(float(full_hist.loss(w, M_)) - full_res) < 1e-5
    w2 = w.clone(); w2[:, -1] = (w2[:, -1] + 1) % 50
    causal = torch.allclose(lp[:, :-1], full_hist.scoreboard(w2)[:, :-2], atol=1e-6)
    full_hist.loss(w).backward()
    grad = bool(full_hist.cache.query.finish.grad.abs().sum() > 0)
    Cm = PV(50, query=True, query_vectors=16, query_active=4, query_by="C", **kw)
    selection_ok = (torch.allclose(Cm.scoreboard(w), a.scoreboard(w), atol=1e-5)
             and PV(50, query_vectors=16, **kw).cache.query is None)          # kapaliyken ok yok
    check("Q: oklar bos -> bugunku defter; butun gecmis == k=T", empty_eq and all_ok and selection_ok)
    check("Q: hizli kayip == tam tablo, nedensel, oklar gradyan alir", fast and causal and grad)

    N, data = _data()
    root = tempfile.mkdtemp()
    try:
        r = TR.RUNS["QC"] = TR.Run("QC", root)
        _run(r, data, N, _zero_metric, "cpu", 2e-3, 3, 0, 8, 3, 3, chain="relative", lam=0.7,
                start_norm=1.0, c_cache=True, **SMALL)
        path = root + "/QC/t3.pt"
        p = torch.load(path, weights_only=False)
        for a in ("query", "query_vectors", "query_active", "query_by", "c_content", "d_order",
                  "d_content", "lam_w_init", "beta_w_init"):
            del p[a]                                # alanlar gelmeden yazilmis paket
        p["d"] = p.pop("d_sum")                     # eski paket boyutu 'd' diye yazardi
        torch.save(p, path)
        reader = _PV.from_package(p)
        reader_ok = reader.d_sum == 16 and not reader.c_content and reader.cache.query is None
        r = TR.RUNS["QC"] = TR.Run("QC", root)       # kapaliyken eski paket SURDURULUR
        _run(r, data, N, _zero_metric, "cpu", 2e-3, 4, 0, 8, 2, 4, resume=path, chain="relative",
                lam=0.7, start_norm=1.0, c_cache=True, **SMALL)
        old_resumed = r.result.get("step") == 4
        try:
            r = TR.RUNS["QC"] = TR.Run("QC", root)
            _run(r, data, N, _zero_metric, "cpu", 2e-3, 6, 0, 8, 3, 3, resume=path, chain="relative",
                    lam=0.7, start_norm=1.0, c_cache=True, c_content=True, **dict(SMALL, d_order=8, d_content=8))
            caught = False
        except ValueError as h:
            caught = "c_content" in str(h)
        r = TR.RUNS["QC2"] = TR.Run("QC2", root)
        _run(r, data, N, _zero_metric, "cpu", 2e-3, 4, 0, 8, 2, 4, chain="relative", lam=0.7,
                start_norm=1.0, c_cache=True, cache_topk=None, query=True, query_vectors=8, query_active=2,
                c_content=True, **dict(SMALL, d_order=8, d_content=8))
        trained = r.result.get("step") == 4 and torch.isfinite(r.result["step_losses"]).all()
        r = TR.RUNS["QC3"] = TR.Run("QC3", root)
        _run(r, data, N, _zero_metric, "cpu", 2e-3, 6, 0, 8, 2, 4, resume=root + "/QC2/t4.pt",
                chain="relative", lam=0.7, start_norm=1.0, c_cache=True, cache_topk=None,
                query=True, query_vectors=8, query_active=2, c_content=True, **dict(SMALL, d_order=8, d_content=8))
        resumed = r.result.get("step") == 6
    finally:
        shutil.rmtree(root, ignore_errors=True)
    check("Q + C_content egitilir, surdurulur; eski paket: surer / farki yakalanir",
         old_resumed and reader_ok and caught and bool(trained) and resumed,
         "kapaliyken eski paket surer; cache_topk None, 8 ok, d_order 8 + d_content 8")


# --- 20.  SELECT: "direction" katman 1'den itibaren boydan bagimsiz secer; katman 0 ve "distance" ayni
def t_select():
    import model_18 as M
    torch.manual_seed(0)
    w = torch.randint(0, 50, (2, 30))
    kw = dict(d_order=64, t_max=64, lam=0.7, chain="relative", start_norm=1.0)
    dist_model, dir_model = PV(50, **kw), PV(50, select="direction", **kw)
    with torch.no_grad():
        for m_ in (dist_model, dir_model):
            for L in m_.V:
                L.finish.add_(3.0 * torch.randn_like(L.finish))     # hareket buyuk: boy secimi ezer
    same0 = torch.equal(dist_model.V[0](dist_model.C(w))[1], dir_model.V[0](dir_model.C(w))[1])  # katman 0 C'ye uzakliga gore
    x = dir_model.V[0](dir_model.C(w))[0]
    L1 = dir_model.V[1]
    dir_invariant = torch.equal(L1(x)[1], L1(x, by=5.0 * x)[1])               # yone bakar: 5 kat boy ayni secim
    dist_norm = not torch.equal(dist_model.V[1](x)[1], dist_model.V[1](x, by=5.0 * x)[1])  # distance: boy secimi degistirir
    q = PV(50, c_cache=True, query=True, query_vectors=16, query_active=4, select="direction", **kw)
    default = M.SELECT == "direction_all" and PV(50, **kw).V[1].direction is False   # tepe hep yon; PLAIN uzaklik
    check("SELECT direction: katman 1+ boydan bagimsiz, katman 0 ayni",
         same0 and dir_invariant and dist_norm and q.cache.query.direction and default,
         "distance'ta 5 kat boy secimi degistiriyor, direction'da degistirmiyor")
    all_dir = PV(50, select="direction_all", **kw)
    C = all_dir.C(w)
    scale_free0 = torch.equal(all_dir.V[0](C)[1], all_dir.V[0](C, by=5.0 * C)[1])          # katman 0 da yone bakar
    check("SELECT direction_all: katman 0 da boydan bagimsiz",
         scale_free0 and all(L.direction for L in all_dir.V) and not dir_model.V[0].direction,
         "5 kat boy katman 0'da secimi degistirmiyor; direction'da katman 0 uzaklik")

    N, data = _data()
    root = tempfile.mkdtemp()
    try:
        r = TR.RUNS["S"] = TR.Run("S", root)
        _run(r, data, N, _zero_metric, "cpu", 2e-3, 4, 0, 8, 2, 4, chain="relative", lam=0.7, start_norm=1.0,
             c_cache=True, select="direction", **SMALL)
        trained = r.result.get("step") == 4 and torch.isfinite(r.result["step_losses"]).all()
        try:
            r = TR.RUNS["S"] = TR.Run("S", root)
            _run(r, data, N, _zero_metric, "cpu", 2e-3, 6, 0, 8, 2, 4, resume=root + "/S/t4.pt", chain="relative",
                 lam=0.7, start_norm=1.0, c_cache=True, select="distance", **SMALL)
            caught = False
        except ValueError as h:
            caught = "select" in str(h)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    check("SELECT egitilir; surdurme farki yakalar", bool(trained) and caught, "direction -> distance")


# --- 21.  HEPSI (ACTIVE = VECTORS): C_m = C + sum_butun softmax(-D^2 e^S_v) V; her vektor gradyan alir
def t_all_active():
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
    manual = x + torch.softmax(-M.distance(x, L0.start) * L0.S_v.exp(), -1) @ (L0.finish - L0.start)
    y = L0(x)[0]
    dir_dist = M.distance(torch.nn.functional.normalize(y, dim=-1), torch.nn.functional.normalize(L1.start, dim=-1))
    manual1 = y + torch.softmax(-dir_dist * L1.S_v.exp(), -1) @ (L1.finish - L1.start)
    formula = torch.allclose(y, manual, atol=1e-5) and torch.allclose(L1(y)[0], manual1, atol=1e-5)
    m.loss(w).backward()
    all_grad = all(bool(((L.start.grad.abs().sum(-1) > 0) & (L.finish.grad.abs().sum(-1) > 0)).all()) for L in m.V)
    check("hepsi: C_m = C + sum softmax V (distance / direction); her vektor gradyan alir", formula and all_grad,
         "16 vektorun 16'si, 2 katman")

    N, data = _data()
    root = tempfile.mkdtemp()
    try:
        r = TR.RUNS["H"] = TR.Run("H", root)
        _run(r, data, N, _zero_metric, "cpu", 2e-3, 4, 0, 8, 2, 4, chain="relative", lam=0.7, start_norm=1.0,
             c_cache=True, select="direction", **dict(SMALL, active=8))
        trained = r.result.get("step") == 4 and torch.isfinite(r.result["step_losses"]).all()
    finally:
        shutil.rmtree(root, ignore_errors=True)
    check("hepsi egitilir (8 vektorun 8'i aktif)", bool(trained))


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
        counted = torch.ones(2, 11, dtype=torch.bool)
        term = float(b.balance(P, counted))
        uniform = float(b.balance([torch.full((2, 12, 64), 1 / 64)], counted))
        diff = float(b.loss(w)) - float(a.loss(w))
    same_0 = torch.equal(a._layers(C)[0], C_m)                      # probs ciktiyi degistirmez
    b.loss(w).backward()
    L = b.V[1]
    selected = set(ids[1][:, :-1].flatten().tolist())
    dead = [i for i in range(64) if i not in selected]
    dead_grad = bool(dead) and bool((L.start.grad[dead].abs().sum(-1) > 0).all())
    a.loss(w).backward()
    dead0 = bool((a.V[1].start.grad[dead].abs().sum(-1) == 0).all()) if dead else True
    check("load balance: esitte 1, kayip + katsayi*terim, secilmeyen start gradyan alir",
         abs(uniform - 1) < 1e-9 and abs(diff - 0.5 * term) < 1e-4 and term >= len(P) - 1e-6
         and same_0 and dead_grad and dead0 and M.LOAD_BALANCE > 0,
         "terim %.2f (4 katman, en az 4); secilmeyen %d start: dengeyle gradyan var, dengesiz yok"
         % (term, len(dead)))

    N, data = _data()
    root = tempfile.mkdtemp()
    try:
        r = TR.RUNS["LB"] = TR.Run("LB", root)
        _run(r, data, N, _zero_metric, "cpu", 2e-3, 4, 0, 8, 2, 4, chain="relative", lam=0.7, start_norm=1.0,
             c_cache=True, select="direction", load_balance=0.01, **SMALL)
        trained = r.result.get("step") == 4 and torch.isfinite(r.result["step_losses"]).all()
        try:
            r = TR.RUNS["LB"] = TR.Run("LB", root)
            _run(r, data, N, _zero_metric, "cpu", 2e-3, 6, 0, 8, 2, 4, resume=root + "/LB/t4.pt", chain="relative",
                 lam=0.7, start_norm=1.0, c_cache=True, select="direction", **SMALL)
            caught = False
        except ValueError as h:
            caught = "load_balance" in str(h)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    check("load balance egitilir; surdurme farki yakalar", bool(trained) and caught, "0,01 -> 0")

    # sicaklik SABIT: denge terimi S_v'ye gradyan vermez; Q'nun oklari da dengelenir; parts NLL'yi ayirir
    torch.manual_seed(0)
    q = PV(50, load_balance=0.5, query=True, query_vectors=16, query_active=16, **kw)
    with torch.no_grad():
        q.cache.query.finish.add_(0.3 * torch.randn_like(q.cache.query.finish))
    C = q.C(w)
    C_m, _, P = q._layers(C, probs=True)
    q.balance(P, torch.ones(2, 11, dtype=torch.bool)).backward()
    temperature = all(L.S_v.grad is None or float(L.S_v.grad.abs()) == 0 for L in q.V)
    start = all(bool(L.start.grad.abs().sum() > 0) for L in q.V)
    q.zero_grad()
    total, nll = q.loss(w, parts=True)
    plain = PV(50, query=True, query_vectors=16, query_active=16, **kw)
    plain.load_state_dict(q.state_dict())
    same_nll = abs(float(nll) - float(plain.loss(w))) < 1e-5 and float(total) > float(nll)
    total.backward()
    q_balance = bool(q.cache.query.start.grad.abs().sum() > 0)
    check("load balance: sicaklik sabit, Q oklari dengelenir, parts NLL'yi ayirir",
         temperature and start and same_nll and q_balance,
         "denge terimi S_v'ye 0 gradyan; NLL denge kapaliyla ayni")


# --- 23.  C_M_NORM: puan yalniz C_m'nin YONUNE bakar; = 2 e^S_p cos + sabit; S_p baslangici formulden
def t_cmnorm():
    import math
    import model_18 as M
    torch.manual_seed(0)
    m = PV(4003, d_order=64, t_max=64, lam=0.7, chain="relative", c_cache=True, c_m_norm=True)
    x = torch.randn(2, 5, 64)
    s1, s5 = m.score(x), m.score(5.0 * x)
    scale_free = torch.allclose(s1, s5, atol=1e-4)
    cos = torch.nn.functional.normalize(x, dim=-1) @ m.P.T
    formula = torch.allclose(torch.log_softmax(s1, -1), torch.log_softmax(2 * m.S_p.exp() * cos, -1), atol=1e-4)
    expected = math.log(math.log(M.C_M_NORM_P / (1 - M.C_M_NORM_P) * 4002) / 2)
    initial = abs(float(m.S_p) - expected) < 1e-6 and abs(m.s_p_init - expected) < 1e-6
    p = torch.softmax(2 * m.S_p.exp() * torch.cat([torch.ones(1), torch.zeros(4002)]), 0)[0]
    model_off = PV(4003, d_order=64, t_max=64, lam=0.7, chain="relative", c_cache=True)
    model_off_ok = not model_off.c_m_norm and float(model_off.S_p) == M.S_P_INIT and M.C_M_NORM   # tepe acik, PLAIN kapali
    check("C_M_NORM: boydan bagimsiz, 2e^S_p cos, S_p formulden",
         scale_free and formula and initial and abs(float(p) - M.C_M_NORM_P) < 1e-3 and model_off_ok,
         "n 4003: S_p %.3f, kusursuz eslesmede p %.3f" % (float(m.S_p), float(p)))

    N, data = _data()
    root = tempfile.mkdtemp()
    try:
        r = TR.RUNS["CN"] = TR.Run("CN", root)
        _run(r, data, N, _zero_metric, "cpu", 2e-3, 4, 0, 8, 2, 4, chain="relative", lam=0.7, start_norm=1.0,
             c_cache=True, c_m_norm=True, **SMALL)
        trained = r.result.get("step") == 4 and torch.isfinite(r.result["step_losses"]).all()
        pk = torch.load(root + "/CN/t4.pt", weights_only=False)
        package_ok = pk["c_m_norm"] is True and abs(pk["s_p_init"] - PV(N, c_m_norm=True).s_p_init) < 1e-9
        try:
            r = TR.RUNS["CN"] = TR.Run("CN", root)
            _run(r, data, N, _zero_metric, "cpu", 2e-3, 6, 0, 8, 2, 4, resume=root + "/CN/t4.pt", chain="relative",
                 lam=0.7, start_norm=1.0, c_cache=True, **SMALL)
            caught = False
        except ValueError as h:
            caught = "c_m_norm" in str(h)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    check("C_M_NORM egitilir; pakete yazilir, surdurme farki yakalar", bool(trained) and package_ok and caught)


# --- 24.  SPEED: hizlandirmalar hesabi DEGISTIRMEZ -- top-k seyrek carpim, siralamasiz butun gecmis,
# en uzun hikayeye kirpma ve yalniz hedefli konumlarda puan: kayip ve BUTUN gradyanlar ayni
def t_speed():
    import model_18 as M
    F = torch.nn.functional
    g = torch.Generator().manual_seed(11)
    L = M.VectorLayer(32, 16, 4, lambda *s: torch.randn(*s, generator=g), 1.0, direction=True)
    with torch.no_grad():
        L.finish.add_(0.1 * torch.randn(L.finish.shape, generator=g))
    C = torch.randn(3, 7, 32, generator=g, requires_grad=True)
    new_, ids = L(C)
    D_s2 = M.distance(F.normalize(C, dim=-1), F.normalize(L.start, dim=-1)).gather(-1, ids)
    W_v = torch.softmax(-D_s2 * L.S_v.exp(), -1)
    previous = C + (W_v.unsqueeze(-1) * (L.finish - L.start)[ids]).sum(-2)     # onceki (..., active, d) yolu
    grad_inputs = [C, L.start, L.finish, L.S_v]
    ga, gb = (torch.autograd.grad(x.square().sum(), grad_inputs) for x in (new_, previous))
    topk = torch.allclose(new_, previous, atol=1e-5) and all(torch.allclose(a, b, atol=1e-5) for a, b in zip(ga, gb))
    check("speed: top-k seyrek satir x matris == kopya yolu", topk, "cikti ve gradyanlar")

    kw = dict(d_order=64, d_content=64, t_max=64, lam=0.7, chain="relative", c_content=True, c_cache=True,
              cache_topk=None, query=True, query_vectors=8, select="direction", load_balance=0.01,
              start_norm=1.0, vectors=32)
    lengths = [22, 17, 9]
    w = torch.randint(0, 50, (3, 40), generator=g)            # dolgu RASTGELE: kirpma onu hic gormemeli
    Mk = torch.zeros(3, 40, dtype=torch.bool)
    for i, n in enumerate(lengths):
        Mk[i, :n] = True
    t, mk, rows = TR.trim(w, Mk, Mk, shapes=8)
    trimmed = t.shape[1] == 25 and len(rows) % max(1, 3 * 40 // 64) == 0 and int((rows >= 0).sum()) == sum(lengths) - 3

    def grad(m, *a, **k):
        top, nll = m.loss(*a, parts=True, **k)
        p = list(m.parameters())
        gr = torch.autograd.grad(top, p, allow_unused=True)
        return top, nll, [torch.zeros_like(x) if y is None else y for x, y in zip(p, gr)]

    results = []
    at = dict(kw, cache_topk=8, query=False, attention=True, attn_heads=2, attn_dim=8)
    for name, cfg in (("butun gecmis + Q + denge", kw), ("defter top-8", dict(kw, cache_topk=8, query=False)),
                     ("deftersiz", dict(kw, c_cache=False)), ("attention katman 0'dan sonra", at),
                     ("attention son katmandan sonra", dict(at, attn_after=3))):
        torch.manual_seed(0)
        m = PV(50, **cfg)
        with torch.no_grad():
            for p in m.parameters():
                p.add_(0.05 * torch.randn(p.shape, generator=g))
        a, b = grad(m, w, Mk), grad(m, t, mk, rows=rows)
        same = (abs(float(a[0] - b[0])) < 1e-5 and abs(float(a[1] - b[1])) < 1e-5
                and all(torch.allclose(x, y, atol=1e-5) for x, y in zip(a[2], b[2])))
        if not (cfg["query"] and cfg["c_cache"]):     # attention'dan sonraki katmanlar yalniz rows'ta
            with torch.no_grad():
                same &= m._layers(m.C(t), tokens=t, rows=rows.clamp(min=0))[0].shape == (len(rows), m.d_sum)
        if cfg["c_cache"] and cfg["cache_topk"] is None:      # siralamasiz yol == siralayan (k = T) yol
            with torch.no_grad():
                s1 = m.scoreboard(w)
                m.cache.topk = 10 ** 6
                same &= torch.allclose(s1, m.scoreboard(w), atol=1e-5)
                m.cache.topk = None
        results.append("%s %s" % (name, "ayni" if same else "FARKLI"))
    check("speed: kirpma + yalniz hedefte katman ve puan == tam kayip", trimmed and all(s.endswith("ayni") for s in results),
         "T 40 -> %d, satir %d; " % (t.shape[1], len(rows)) + ", ".join(results))


# --- 25.  ATTENTION: W_o = 0 -> attention'siz ile ayni; nedensel; anahtar C_(j-1); gradyan; point; paket
def t_attention():
    F = torch.nn.functional
    g = torch.Generator().manual_seed(12)
    kw = dict(d_order=32, d_content=32, t_max=64, lam=0.7, chain="relative", c_content=True, c_cache=True,
              select="direction", start_norm=1.0, vectors=16, c_m_norm=True)
    w = torch.randint(0, 50, (2, 30), generator=g)
    torch.manual_seed(0)
    without_attn = PV(50, **kw)
    torch.manual_seed(0)
    with_attn = PV(50, attention=True, attn_heads=2, attn_dim=8, **kw)
    extra_params = sum(p.numel() for p in with_attn.parameters()) - sum(p.numel() for p in without_attn.parameters())
    with torch.no_grad():
        silent = torch.equal(without_attn.scoreboard(w), with_attn.scoreboard(w))
        with_attn.attn.W_o.weight.normal_(0, 0.3, generator=g)
        a = with_attn.scoreboard(w)
        w2 = w.clone(); w2[:, 20] = (w2[:, 20] + 1) % 50
        b = with_attn.scoreboard(w2)
        causal = torch.allclose(a[:, :20], b[:, :20], atol=1e-6) and not torch.allclose(a[:, 20:], b[:, 20:])
        # elle: t, j <= t'ye bakar; k_j = W_k norm(C_(j-1)), k_0 = 0; v_j = W_v norm(C_m,j)
        L, C = with_attn.attn, with_attn.C(w)
        C_m0 = with_attn.V[0](C)[0]
        prev = torch.cat([torch.zeros_like(C[:, :1]), C[:, :-1]], 1)
        q = L.W_q(F.normalize(C_m0, dim=-1)).view(2, 30, 2, 8)
        k = L.W_k(F.normalize(prev, dim=-1)).view(2, 30, 2, 8)
        v = L.W_v(F.normalize(C_m0, dim=-1)).view(2, 30, 2, 8)
        manual_out = torch.stack([torch.stack([C_m0[i, t] + L.W_o(torch.cat([
            torch.softmax(k[i, :t + 1, h] @ q[i, t, h] / 8 ** 0.5, 0) @ v[i, :t + 1, h] for h in range(2)]))
            for t in range(30)]) for i in range(2)])
        shifted = torch.allclose(L(C, C_m0), manual_out, atol=1e-5)
        torch.manual_seed(0)
        point_model = PV(50, attention=True, attn_heads=2, attn_dim=8, attn_value="point", **kw)
        point_model.load_state_dict(with_attn.state_dict())
        p_ = point_model.scoreboard(w)
        point = bool(torch.isfinite(p_).all()) and not torch.allclose(p_, a, atol=1e-5)
    with_attn.loss(w).backward()
    grad = all(bool(x.grad.abs().sum() > 0) for x in (L.W_q.weight, L.W_k.weight, L.W_v.weight, L.W_o.weight))
    check("attention: W_o 0 -> attention'siz ile ayni, nedensel, anahtar C_(j-1)",
         silent and causal and shifted and extra_params == 4 * 64 * 16,
         "elle hesapla ayni; 20. konum degisince oncekiler ayni; +%d parametre" % extra_params)
    check("attention: gradyan W_q/W_k/W_v/W_o'ya ulasir, point calisir", grad and point)

    N, data = _data()
    root = tempfile.mkdtemp()
    try:
        r = TR.RUNS["AT"] = TR.Run("AT", root)
        _run(r, data, N, _zero_metric, "cpu", 2e-3, 4, 0, 8, 2, 4, chain="relative", lam=0.7, start_norm=1.0,
             c_cache=True, attention=True, attn_heads=2, attn_dim=4, **SMALL)
        trained = r.result.get("step") == 4 and torch.isfinite(r.result["step_losses"]).all()
        pk = torch.load(root + "/AT/t4.pt", weights_only=False)
        loaded = _PV.from_package(pk)
        package_ok = (pk["attention"] is True and pk["attn_heads"] == 2 and loaded.attn is not None
                 and loaded.load_balance == pk["load_balance"])
        try:
            r = TR.RUNS["AT"] = TR.Run("AT", root)
            _run(r, data, N, _zero_metric, "cpu", 2e-3, 6, 0, 8, 2, 4, resume=root + "/AT/t4.pt", chain="relative",
                 lam=0.7, start_norm=1.0, c_cache=True, **SMALL)
            caught = False
        except ValueError as h:
            caught = "attention" in str(h)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    check("attention egitilir; pakete yazilir, surdurme farki yakalar", bool(trained) and package_ok and caught,
         "attention acik paket -> kapali cagri")


# --- 26.  SAGLIK: cevabi bilinen durumlar; inspect == move; dikkat nedensel; olcut ve egitim butunlesmesi
def t_health():
    import model_18 as M
    import data_stories as DS
    g = torch.Generator().manual_seed(21)
    uniform = float(M.entropy(torch.full((16,), 1 / 16)).exp())
    single = float(M.entropy(torch.eye(16)[3]).exp())
    three = M.directions90(torch.randn(500, 3, generator=g) @ torch.randn(3, 64, generator=g))
    known = abs(uniform - 16) < 1e-4 and abs(single - 1) < 1e-6 and three <= 3

    kw = dict(d_order=32, d_content=32, t_max=64, lam=0.7, chain="relative", c_content=True, c_cache=True,
              select="direction", start_norm=1.0, vectors=16, active=4, c_m_norm=True)
    w = torch.randint(3, 50, (2, 30), generator=g)
    mask = torch.ones(2, 30, dtype=torch.bool)
    torch.manual_seed(0)
    m = PV(50, attention=True, attn_heads=2, attn_dim=8, **kw)
    with torch.no_grad():
        m.attn.W_o.weight.normal_(0, 0.3, generator=g)
        ins = m.inspect(w)
        same = torch.allclose(ins["C_m"], m.move(w)[0], atol=1e-6)
        selection_ok = all(bool(((W > 0).sum(-1) == 4).all() and torch.allclose(W.sum(-1), torch.ones(2, 30), atol=1e-5))
                    for W in ins["W"])
        A = ins["attn"]
        causal = (torch.allclose(A.sum(-1), torch.ones_like(A[..., 0]), atol=1e-5)
                    and float(A.triu(1).abs().sum()) == 0.0)
        h = m.health(w, mask)
        for L in m.V:
            L.S_v.fill_(-30.0)                                # neredeyse duz: etkin = aktif sayisi
        flat_ = m.health(w, mask)
        for L in m.V:
            L.S_v.fill_(30.0)                                 # keskin: her konumda 1
        sharp = m.health(w, mask)
    keys_ok = (all(("vec_each_%d" % i) in h and ("dead_%d" % i) in h for i in range(4))
               and {"dir_C", "dir_Cm", "norm_C", "gate", "lam_c", "attn_ent", "attn_first", "pred_distinct"} <= set(h)
               and len(h["attn_ent"]) == 2)
    counts_ok = (all(abs(flat_["vec_each_%d" % i] - 4) < 0.05 for i in range(4))
             and all(abs(sharp["vec_each_%d" % i] - 1) < 0.05 for i in range(4)))
    check("saglik: bilinen durumlar, inspect == move, dikkat nedensel", known and same and selection_ok and causal,
         "duz 16 -> %.2f, tek -> %.2f, rank 3 -> %d yon" % (uniform, single, three))
    check("saglik: anahtarlar, etkin vektor duzde aktif / keskinde 1", keys_ok and counts_ok,
         "aktif 4: duz %.2f, keskin %.2f" % (flat_["vec_each_0"], sharp["vec_each_0"]))

    vocab_list = [DS.PAD_TOKEN, DS.EOS_TOKEN, DS.UNK_TOKEN] + ["k" + c for c in "abcdefghi"]      # JETON rakami ayri boler
    W = torch.randint(3, 12, (6, 16), generator=g)
    Mk = torch.zeros(6, 16, dtype=torch.bool)
    for i, n in enumerate((16, 9, 12, 5, 16, 7)):
        W[i, 0], W[i, n - 1], W[i, n:] = 1, 1, 0
        Mk[i, :n] = True
    metric_fn = DS.make_metric((W, Mk), (W, Mk), eos=1, device="cpu", limit=6, vocab=vocab_list, prompts=["ka kb kc", "kd yok"])
    torch.manual_seed(0)
    k = PV(12, t_max=16, c_cache=True, chain="relative", lam=0.7, start_norm=1.0, **SMALL)
    plain_eval, full_res = metric_fn(k, "heldout"), metric_fn(k, "heldout", save=True)
    complete = (metric_fn.health and "health" not in metric_fn(k, "train") and "distinct4" not in plain_eval["health"]
             and len(full_res["health"]["texts"]) == 2 and 0 <= full_res["health"]["distinct4"] <= 1)
    root = tempfile.mkdtemp()
    try:
        r = TR.RUNS["SA"] = TR.Run("SA", root)
        _run(r, (W, Mk, Mk), 12, metric_fn, "cpu", 2e-3, 4, 0, 8, 2, 4, t_max=16, c_cache=True, chain="relative",
             lam=0.7, start_norm=1.0, **SMALL)
        pk = torch.load(root + "/SA/t4.pt", weights_only=False)
        trained = (any("saglik  vektor" in s for s in r.log) and any("saglik ozeti" in s for s in r.log)
                  and "distinct4" in pk["health"] and "vektor" in pk["grads"])
        # olcum 0, 2, 4: uc saglik satiri; BITTI 4'un sagligini yeniden hesaplamaz, satiri tekrar yazmaz
        n_health_lines = sum("saglik  vektor" in s for s in r.log)
        step_lines = [s for s in r.log if len(s.split()) > 2 and s.split()[1].isdigit()]
        timing_ok = (pk["done"] and {"train", "heldout"} <= set(pk["eval_sec"]) and len(step_lines) == 3
                and all("  olcum " in s for s in step_lines) and any("tam olcum" in s for s in r.log)
                and any("adim 0 haric" in s for s in r.log))
    finally:
        shutil.rmtree(root, ignore_errors=True)
    check("saglik: olcut ve egitim -- gunlukte satir, pakette sozluk ve gradyan", complete and trained,
         "sonda: %d istem (bilinmeyenli istem elendi)" % len(full_res["health"]["texts"]))
    check("saglik: kosu sonu sagligi tekrar olcmez; olcum suresi gunlukte ve pakette", n_health_lines == 3 and timing_ok,
         "%d saglik satiri" % n_health_lines)


# --- 27.  LR SOGUTMASI: cosine decay_start -> steps, taban lr x floor; plansiz paketten dal acilir,
# planli paket yalniz AYNI planla surer
def t_decay():
    N, data = _data()
    root = tempfile.mkdtemp()
    try:
        r = TR.RUNS["LR"] = TR.Run("LR", root)
        _run(r, data, N, _zero_metric, "cpu", 2e-3, 4, 0, 8, 2, 4, **SMALL)                       # sabit LR, t4
        constant_ok = r.result["lr_now"] == 2e-3 and r.result["decay_start"] is None
        r = TR.RUNS["LRD"] = TR.Run("LRD", root)
        _run(r, data, N, _zero_metric, "cpu", 2e-3, 8, 0, 8, 2, 4, resume=root + "/LR/t4.pt", decay_start=4,
             decay_floor=0.1, **SMALL)                                                      # dal: 4 -> 8
        lines = {s.split()[1]: s for s in r.log if len(s.split()) > 2 and s.split()[1].isdigit()}
        plan = ("lr 1.10e-03" in lines.get("6", "") and "lr 2.00e-04" in lines.get("8", "")
                and abs(r.result["lr_now"] - 2e-4) < 1e-12)               # adim 6: cos(pi/2); adim 8: taban
        try:
            r = TR.RUNS["LRD"] = TR.Run("LRD", root)
            _run(r, data, N, _zero_metric, "cpu", 2e-3, 10, 0, 8, 2, 4, resume=root + "/LRD/t8.pt", decay_start=4,
                 decay_floor=0.1, **SMALL)                                  # plan (4, 8) -> (4, 10): durmali
            caught = False
        except ValueError as h:
            caught = "sogutma" in str(h)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    check("LR sogutmasi: cosine, taban lr x 0,1; sabitten dal, planli ayni planla", constant_ok and plan and caught,
         "adim 6: 1,10e-3, adim 8: 2,00e-4; plan degisince surdurme durur")


# --- 28.  KOSU SONU: son adim olculmus ama tam yedek adimi degilse olcum satiri IKINCI kez yazilmaz
# (LR_CONST / LR_DECAY gunlugunde 5.000 satiri iki kez vardi)
def t_finish():
    N, data = _data()
    root = tempfile.mkdtemp()
    try:
        r = TR.RUNS["FIN"] = TR.Run("FIN", root)
        _run(r, data, N, _zero_metric, "cpu", 2e-3, 6, 0, 8, 2, 4, **SMALL)          # 6 olculur, tam yedek 4'te
        last = [s for s in r.log if len(s.split()) > 2 and s.split()[1] == "6"]
        saved = os.path.exists(root + "/FIN/t6.pt")
        noted = any("yedek t6" in s for s in r.log)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    check("kosu sonu: son olcum satiri bir kez, yedek notu", len(last) == 1 and saved and noted,
          "adim 6 satiri %d kez; t6.pt %s" % (len(last), "var" if saved else "YOK"))


# ============================================================
# DATA_STORIES -- TinyStories (model_17 veri_t17 + olcme_17'den).
# ============================================================
def t_stories():
    import numpy as np
    import data_stories as DS
    eos_id, pad = 1, 0
    a = np.array([5, 6, eos_id, 7, 8, 9, eos_id, 3, eos_id], dtype=np.int16)
    P, M = DS.make_windows(a, 5, None, eos_id, pad)
    ok = (len(P) == 3 and list(M.sum(1)) == [4, 5, 3]
          and list(P[1]) == [eos_id, 7, 8, 9, eos_id] and (P[~M] == pad).all())
    P4, _ = DS.make_windows(a, 4, None, eos_id, pad)           # L+2 > T olan ATILIR
    check("hikaye penceresi <eos> basta ve sonda", ok and len(P4) == 2,
         "%d pencere; T=4'te %d" % (len(P), len(P4)))

    # v4: normalizasyon, satir sonu TEK <nl>, kusur suzgeci (sebep sirasi: karakter, bas, son), decode'da satir basi
    toks = DS.tokenize(DS.normalize("Tom said, “Hi.”\n\nLily ran to the café… It´s fun."))
    ok_tok = toks == ["Tom", "said", ",", '"', "Hi", ".", '"', DS.NL_TOKEN, "Lily", "ran", "to", "the", "cafe",
                      ".", ".", ".", "It's", "fun", "."]
    why = [DS.defect(DS.normalize(s).strip()) for s in
           ("u don't know.", "Once upon a time, there was a", "Tim ran.\n巴恩.", "Tim saw a tree\U0001F334.",
            "-\nOnce upon a time.", "Moral of the Story: be kind.", "Tim ran!")]
    ok_def = why == ["bas", "son", "karakter", "karakter", "bas", None, None]
    v = [DS.PAD_TOKEN, DS.EOS_TOKEN, DS.UNK_TOKEN, DS.NL_TOKEN, '"', "Hi", ".", "Tom", "said"]
    txt = DS.decode(np.array([4, 5, 6, 4, 3, 7, 8, 6]), v)
    check("v4: normalizasyon, <nl>, kusur suzgeci, decode satir basi", ok_tok and ok_def and txt == '"Hi."\nTom said.',
          "%d token; sebepler %s; %r" % (len(toks), why, txt))

    # v4 uctan uca: kusurlular atilir, <nl> akista, ikinci cagri onbellekten, eski surum URETILMEZ
    root = tempfile.mkdtemp()
    try:
        good = ["Tim ran to the park.\nHe saw a dog.", "Tom saw a cat.\n\n\"Hi,\" said Tom."]
        bad = ["u don't go.", "Tim ran to the"]
        with open(os.path.join(root, "TinyStoriesV2-GPT4-train.txt"), "w", encoding="utf-8") as f:
            f.write(DS.SEPARATOR.join((good + bad) * 3))
        with open(os.path.join(root, "TinyStoriesV2-GPT4-valid.txt"), "w", encoding="utf-8") as f:
            f.write(DS.SEPARATOR.join(["Tim saw a dog.", "Tom ran."]))
        logs = []
        vocab, (P, M), (Q, _) = DS.build(root, T=16, limit=8, log=logs.append)
        nl_id = vocab.index(DS.NL_TOKEN)
        built = (vocab[:4] == list(DS.SPECIAL) and len(P) == 6 and len(Q) == 2 and int((P == nl_id).sum()) == 6
                 and any("'bas': 3" in s and "'son': 3" in s for s in logs))
        logs2 = []
        vocab2, (P2, _), _ = DS.build(root, T=16, limit=8, log=logs2.append)
        cached = vocab2 == vocab and np.array_equal(P2, P) and sum("onbellekten" in s for s in logs2) == 3
        try:
            DS.build(root, T=16, limit=8, version="v2", log=logs.append)
            old_refused = False
        except FileNotFoundError:
            old_refused = True
        # paralel yol == tek surec: kucuk bloklar (cok blok) ve 2 surec; build'in yazdigi akisla da ayni
        tr, quiet = os.path.join(root, "TinyStoriesV2-GPT4-train.txt"), (lambda *a: None)
        v1, ix1 = DS.build_vocab(tr, 8, chunk_mb=0.00005, log=quiet)
        v2, _ = DS.build_vocab(tr, 8, chunk_mb=0.00005, workers=2, log=quiet)
        a1 = DS.build_stream(tr, ix1, chunk_mb=0.00005, log=quiet)
        a2 = DS.build_stream(tr, ix1, chunk_mb=0.00005, workers=2, log=quiet)
        cached_a = np.load(os.path.join(root, "onbellek", "akis_train_tam_n8_%s.npy" % DS.DATA_VERSION))
        parallel_same = v1 == v2 == vocab and np.array_equal(a1, a2) and np.array_equal(a1, cached_a)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    check("v4 build: kusurlular atilir, <nl> akista, onbellek okunur, eski surum uretilmez; paralel == tek surec",
          built and cached and old_refused and parallel_same, "6 + 2 pencere, 6 <nl>; ikinci cagri onbellekten")

    # decode: istemin actigi tirnak devamda kapanis olarak okunur (eskiden devam ters bosluklu basiliyordu)
    vocab = ["<dolgu>", DS.EOS_TOKEN, DS.UNK_TOKEN, '"', "?", "The", "cat", "said", ",", "Hi"]
    cont = [4, 3, 5, 6, 7, 8, 3, 9, 3]                            # ? " The cat said , " Hi "
    inside, plain = DS.decode(np.array(cont), vocab, in_quote=True), DS.decode(np.array(cont), vocab)
    check("decode: istemden devreden acik tirnak", inside == '?" The cat said, "Hi"', "%r  (eski: %r)" % (inside, plain))

    # evaluate: bantlar, eos_ok ve ce, KIRPMASIZ tam tabloyla ayni (evaluate pencereleri siralayip kirpiyor);
    # ikinci model defterli + attention'li: konumlar arasi karisma varken de kirpma sonucu degistirmez.
    # Beklenen oranlar TAM bolmeyle (float32 bolme 1e-9'u asar: 2/55'te 1,2e-9).
    ratio = lambda c, n: int(c.sum()) / int(n.sum())
    W = torch.randint(3, 12, (6, 16), generator=torch.Generator().manual_seed(3))    # iki modelde de acc_ar 0,5
    Mk = torch.zeros(6, 16, dtype=torch.bool)
    for i, L in enumerate((16, 9, 12, 5, 16, 7)):
        W[i, 0], W[i, L - 1] = eos_id, eos_id
        W[i, L:] = pad
        Mk[i, :L] = True
    k = torch.arange(1, 16)
    e = Mk[:, 1:] & (W[:, 1:] == eos_id)
    ok, out = True, []
    for cfg in (dict(), dict(chain="relative", lam=0.7, start_norm=1.0, c_cache=True, attention=True,
                              attn_heads=2, attn_dim=4)):
        torch.manual_seed(0)
        m = PV(12, t_max=16, **dict(SMALL, **cfg))
        if cfg:
            with torch.no_grad():
                m.attn.W_o.weight.normal_(0, 0.3)
        DS.BANDS, previous = ((0, 4), (4, 10), (10, 16)), DS.BANDS
        try:
            r = DS.evaluate(m, W, Mk, eos_id, device="cpu", chunk=4)
        finally:
            DS.BANDS = previous
        with torch.no_grad():
            p = m.scoreboard(W)[:, :-1]
            d = (p.argmax(-1) == W[:, 1:]) & Mk[:, 1:]
            ce = torch.nn.functional.cross_entropy(p.transpose(1, 2), W[:, 1:], reduction="none")[Mk[:, 1:]].mean()
        expected = [ratio(d[:, (k >= a_) & (k < b_)], Mk[:, 1:][:, (k >= a_) & (k < b_)])
               for a_, b_ in ((0, 4), (4, 10), (10, 16))]
        ok &= (abs(r["accuracy"] - ratio(d, Mk[:, 1:])) < 1e-9
               and all(abs(r["diag"][n] - b) < 1e-9 for n, b in zip(("acc_0_4", "acc_4_10", "acc_10_16"), expected))
               and abs(r["diag"]["eos_ok"] - ratio(d[e], e)) < 1e-9
               and abs(r["ce"] - float(ce)) < 1e-5)
        out.append("accuracy %.3f ce %.3f" % (r["accuracy"], r["ce"]))
    check("hikaye olcutu: accuracy, bantlar, eos_ok, ce (kirpilmis)", ok, "; ".join(out))

    # acc_ar: hedef, hikayede daha once tamamlanmis bir ikiliyi tamamliyor; elle sayimla ayni
    example = DS.repeated_bigram(torch.tensor([[1, 5, 6, 5, 6, 2, 5, 6]]), torch.ones(1, 7, dtype=torch.bool))
    manual = example[0].tolist() == [False, False, False, True, False, False, True]
    ar_mask = torch.zeros(6, 15, dtype=torch.bool)
    for i in range(6):
        seen = set()
        for t in range(15):
            if Mk[i, t + 1]:
                pair = (int(W[i, t]), int(W[i, t + 1]))
                ar_mask[i, t] = pair in seen
                seen.add(pair)
    a_ = Mk[:, 1:]
    expected = {name: ratio(d[s], s) for name, s in (("acc_ar", a_ & ar_mask), ("acc_other", a_ & ~ar_mask))}
    check("hikaye olcutu: acc_ar / acc_other (tekrar eden ikili)",
         manual and int(ar_mask.sum()) > 0 and all(abs(r["diag"][name] - b) < 1e-9 for name, b in expected.items()),
         "%d tekrar ikili; acc_ar %.3f  acc_other %.3f" % (int(ar_mask.sum()), r["diag"]["acc_ar"], r["diag"]["acc_other"]))


# ============================================================
# DATA -- matematik (model_17'den).
# ============================================================

# --- 11.  PENCERE ve HEDEF MASKESI
def t_math_windows():
    W, M, H = VM.make_windows([(1, 1), (21, 23)])
    E_ = VM.EOS
    ok = (W[0].tolist() == [E_, 1, 10, 1, 11, 2, E_, E_, E_, E_]
          and W[1].tolist() == [E_, 2, 1, 10, 2, 3, 11, 4, 4, E_]
          and M.sum(1).tolist() == [7, 10]
          and torch.nonzero(H[0]).squeeze(1).tolist() == [5, 6]
          and torch.nonzero(H[1]).squeeze(1).tolist() == [7, 8, 9])
    check("matematik penceresi ve hedef maskesi", ok, "cevap + EOS hedef")


class _Oracle(torch.nn.Module):
    """Cevabi BILEN sahte model; '='den sonraki HER konumu doldurur.  kip:
    dogru / erken (hemen EOS) / fazla (EOS yerine fazladan bir rakam) /
    birler (yalniz birler basamagi yanlis)."""

    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def scoreboard(self, w, mask=None):
        out = torch.zeros(w.shape[0], w.shape[1], VM.N)
        for r, s in enumerate(w.tolist()):
            e = s.index(VM.EQUALS)
            terms_, digits = [], ""
            for t in s[1:e]:
                if t == VM.PLUS:
                    terms_, digits = terms_ + [int(digits)], ""
                else:
                    digits += str(t)
            target = VM.digits(sum(terms_ + [int(digits)])) + [VM.EOS]
            for j in range(e, len(s)):
                written = j - e
                if self.mode == "erken":
                    t = VM.EOS
                elif self.mode == "fazla" and written == len(target) - 1:
                    t = 7
                elif self.mode == "birler" and written == len(target) - 2:
                    t = (target[-2] + 1) % 10
                else:
                    t = target[min(written, len(target) - 1)]
                out[r, j, t] = 10.0
        return out


# --- 12.  OLCUT: dogruyu 1, erken durani ve fazla yazani 0 saymali
def t_math_ask():
    s = [(1, 1), (21, 23), (472, 182), (1, 1, 5), (23, 1, 120), (500, 500, 500)]
    d, e, f = (VM.ask(_Oracle(k), s, device="cpu") for k in ("dogru", "erken", "fazla"))
    fields = lambda r: (r["accuracy"], r["length_ok"], r["first_digit"])
    ok = (fields(d) == (1.0, 1.0, 1.0) and fields(e) == (0.0, 0.0, 0.0)
          and fields(f) == (0.0, 0.0, 1.0))
    check("sor: dogru 1, erken 0, fazla rakam 0", ok,
         "dogru %s  erken %s  fazla %s" % tuple(fields(r) for r in (d, e, f)))


# --- 13.  BASAMAK: saga hizali, birler AYRI okunur
def t_math_digits():
    s = [(1, 1), (21, 23), (472, 182), (1, 1, 5), (23, 1, 120),
         (500, 500, 500), (9, 1), (99, 1)]
    d = VM.digit_check(_Oracle("dogru"), s)
    b = VM.digit_check(_Oracle("birler"), s)
    ok = (all(v["teacher"] == 1.0 and v["free"] == 1.0 for v in d.values())
          and all((v["teacher"], v["free"]) == ((0.0, 0.0) if k[1] == "birler"
                                          else (1.0, 1.0))
                  for k, v in b.items())
          and sorted(k[0] for k in d if k[1] == "eos") == [1, 2, 3, 4])
    check("basamak: saga hizali, birler ayri", ok,
         "%d yuva; birler-yanlis kahinde yalniz birler 0" % len(d))


# --- 14.  CEVAP MASKELI EGITIM: surdurme == kesintisiz, her adimin kaybi eksiksiz
def t_math_training():
    g = torch.Generator().manual_seed(21)
    questions = [tuple(x) for x in torch.randint(0, 60, (64, 2), generator=g).tolist()]
    W, M, H = VM.make_windows(questions)

    def run_(name, root, steps, resume_from=None):
        r = _run_small(name, root, (W, M, H), VM.N, steps, resume=resume_from)
        return _weights_of(r), r.result["step_losses"]

    root = tempfile.mkdtemp()
    try:
        A, losses_a = run_("KESINTISIZ", root, 8)
        run_("BOLUK", root, 4)
        B, losses_b = run_("BOLUK", root, 8, resume_from=root + "/BOLUK/t4.pt")
        max_diff = max(float((A[k] - B[k]).abs().max()) for k in A)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    check("cevap maskeli egitim, surdurme", max_diff == 0.0, "fark %.3e" % max_diff)
    check("her adimin kaybi: surdurmede eksiksiz",
         len(losses_a) == 9 and not bool(losses_a.isnan().any()) and torch.equal(losses_a, losses_b),
         "%d adim, surdurulen == kesintisiz" % len(losses_a))


# --- 15.  NOTEBOOK: kural 2 ve kural 8 MEKANIK
# GPU hucresi GPU'yu kendi icinde sorar; her kosu kendi hucresinde baslar;
# hucrelerde tanimsiz ad yok (Colab'da NameError bir gidis-donus demek).
def t_notebook(path=None):
    import ast
    import builtins
    import json
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "notebook.ipynb")
    cells = [("".join(c["source"])) for c in
             json.load(open(path, encoding="utf-8"))["cells"]
             if c["cell_type"] == "code"]
    defined, defects = set(dir(builtins)) | {"display", "get_ipython"}, []
    for s in cells:
        header = s.splitlines()[0]
        py = "\n".join(x for x in s.splitlines() if not x.lstrip().startswith("!"))
        try:
            tree = ast.parse(py)
        except SyntaxError as h:
            defects.append("%s: sozdizimi %s" % (header[:30], h))
            continue
        if not header.startswith("# ") or header.count("|") < 2:
            defects.append("kunye yok: " + header[:30])
        gpu = "|  GPU  |" in header
        if gpu and "torch.cuda.is_available()" not in py:
            defects.append("GPU kapisi yok: " + header[:30])
        if py.count(".start(") > 1 or (".start(" in py and not gpu):
            defects.append("start kurali: " + header[:30])
        this_cell = set()
        for d in ast.walk(tree):
            if isinstance(d, ast.Name) and isinstance(d.ctx, (ast.Store, ast.Del)):
                this_cell.add(d.id)
            elif isinstance(d, (ast.FunctionDef, ast.ClassDef)):
                this_cell.add(d.name)
            elif isinstance(d, ast.arg):
                this_cell.add(d.arg)
            elif isinstance(d, (ast.Import, ast.ImportFrom)):
                this_cell |= {(a.asname or a.name).split(".")[0] for a in d.names}
            elif isinstance(d, ast.ExceptHandler) and d.name:
                this_cell.add(d.name)
        missing = sorted({d.id for d in ast.walk(tree) if isinstance(d, ast.Name)
                        and isinstance(d.ctx, ast.Load)} - defined - this_cell)
        if missing:
            defects.append("%s: tanimsiz %s" % (header[:30], missing))
        defined |= this_cell
    check("notebook: GPU kapisi, tek kosu, tanimsiz ad", not defects,
         "; ".join(defects) if defects else "%d kod hucresi" % len(cells))


if __name__ == "__main__":
    print("tests (model_18)")
    for f in (t_chain, t_causal, t_silent, t_cm, t_denominator, t_gradient,
              t_params, t_mask, t_resume, t_stop, t_score, t_start, t_lam, t_relative, t_ccache, t_init, t_content, t_query, t_top, t_select, t_all_active, t_balance, t_cmnorm, t_speed, t_attention, t_health, t_decay, t_finish, t_math_windows,
              t_math_ask, t_math_digits, t_math_training, t_stories, t_notebook):
        f()
    t_notebook(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "notebook_stories.ipynb"))
    print("\n%d GECTI   %d KALDI" % (len(PASSED), len(FAILED)))
    sys.exit(1 if FAILED else 0)
