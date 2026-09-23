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
from model_18 import PV                                         # noqa: E402

GECTI, KALDI = [], []
KUCUK = dict(d=16, vectors=8, active=2, layers=2)      # egitim kapilari icin kucuk PV


def kapi(ad, sart, not_=""):
    (GECTI if sart else KALDI).append(ad)
    print("  %-42s %s  %s" % (ad, "GECTI" if sart else "KALDI", not_))


def _dizi(s):
    return torch.tensor([VM.AD.index(a) if a != "E" else VM.EOS for a in s])


def _kos(ad, kok, data, n_vocab, adim, metric=None, resume=None, lr=2e-3):
    """_run'i kucuk PV ile, ayni iplikte kosturur; Run'i doner."""
    r = TR.RUNS[ad] = TR.Run(ad, kok)
    TR._run(r, data, n_vocab, metric or _sifir, "cpu", lr, adim, 0, 8,
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
    u = torch.randn(m.d, generator=torch.Generator().manual_seed(5))
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
    bek = m.vectors * 2 * m.d * m.layers + m.layers
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

        TR._run(r, data, N, metric, "cpu", lr, adim, 0, 8, 1, 3,
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
          == m.vectors * 2 * m.d * m.layers + m.layers + 1)
    kapi("ogrenilen S_p: e^0 = 1, gradyan alir", ok, "parametre +1")

    N, data = _veri()
    kok = tempfile.mkdtemp()
    try:
        r = TR.RUNS["KARE"] = TR.Run("KARE", kok)
        TR._run(r, data, N, _sifir, "cpu", 2e-3, 3, 0, 8, 3, 3,
                squared=True, S_p=1.0, **KUCUK)
        yol = kok + "/KARE/t3.pt"
        eski = torch.load(yol, weights_only=False)
        del eski["S_p"], eski["squared"]         # alanlar gelmeden yazilmis paket
        torch.save(eski, yol)
        r = TR.RUNS["KARE"] = TR.Run("KARE", kok)
        TR._run(r, data, N, _sifir, "cpu", 2e-3, 6, 0, 8, 3, 3,
                resume=yol, squared=True, S_p=1.0, **KUCUK)
        kabul = r.result.get("step") == 6
        try:
            r = TR.RUNS["KARE"] = TR.Run("KARE", kok)
            TR._run(r, data, N, _sifir, "cpu", 2e-3, 9, 0, 8, 3, 3,
                    resume=yol, **KUCUK)             # tablodan: karesiz 10
            yakaladi = False
        except ValueError as h:
            yakaladi = "S_p" in str(h) and "squared" in str(h)
    finally:
        shutil.rmtree(kok, ignore_errors=True)
    kapi("surdurme skor farkini yakalar", kabul and yakaladi,
         "eski kare paket: kare ile surer, tabloyla durur")


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
              t_parametre, t_mask, t_surdurme, t_durdur, t_skor, t_mat_pencere,
              t_mat_sor, t_mat_basamak, t_mat_egitim, t_stories, t_notebook):
        f()
    t_notebook(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "notebook_stories.ipynb"))
    print("\n%d GECTI   %d KALDI" % (len(GECTI), len(KALDI)))
    sys.exit(1 if KALDI else 0)
