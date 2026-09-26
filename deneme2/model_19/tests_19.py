# -*- coding: utf-8 -*-
"""tests_19 -- model_19, decompose_19 ve diagnose_19'un kapilari.  Kucuk modellerde, CPU, saniyeler.

    python tests_19.py
"""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
torch.set_num_threads(1)

import data_stories_19 as DS  # noqa: E402
import decompose_19 as DC  # noqa: E402
import diagnose_19 as DG  # noqa: E402
from model_19 import PointRelation  # noqa: E402

RESULTS = []
BASE = dict(n=30, d_order=16, d_content=16, vectors=8, active=2, layers=4, attn_heads=2, attn_dim=4, t_max=40, seed=3)
OFF = dict(embed=False, readout=False, chain_sim=False, distance=False, attn_after=(0,), norm_inputs=False)
ON = dict(embed=True, readout=True, chain_sim=True, distance=True, attn_after=(0, 2), induction_query=True,
          norm_inputs=True)


def check(name, ok, note=""):
    RESULTS.append(bool(ok))
    print("  %-78s %s  %s" % (name, "GECTI" if ok else "KALDI", note))


def tokens_and_mask(n=30, B=3, T=24, seed=7):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, n, (B, T), generator=g)
    mask = torch.ones_like(tokens, dtype=torch.bool)
    mask[-1, T - 6:] = False
    return tokens, mask


def reference_logp(m, tokens):
    """TASARIM_19'un formulleri, model_19'un kodundan BAGIMSIZ ve en saf haliyle (konum konum donguler, float64):
    tokens (T,) -> log p (T, n), konum t'de SONRAKI token.  Kod tasarimdan saparsa scoreboard'la tutmaz."""
    import model_19 as M
    f = lambda x: x.detach().double()
    unit = lambda x: x / x.norm()
    T, Do, n = len(tokens), m.d_order, m.n
    P = f(m.P)
    E = []                                                   # E[w] = norm(P[w] + dE[w]),  dE[w] = W U[w]
    for w in tokens.tolist():
        x = P[w].clone()
        if m.embed_delta is not None:
            x = x + (f(m.embed_delta[w]) if m.embed_up is None else f(m.embed_up) @ f(m.embed_delta[w]))
        E.append(unit(x))
    C = []                                                   # zincir: [sira | icerik]
    h = torch.zeros(m.d_content, dtype=torch.float64)
    for t in range(T):
        order = sum(m.lam ** (t - i) * torch.roll(E[i][:Do], t - i) for i in range(t + 1))  # yasi kadar kaydir, lam^yas
        w = int(tokens[t])
        lam = M.LAM_W_MIN + (1 - M.LAM_W_MIN) * torch.sigmoid(f(m.lam_w[w]))
        h = lam * h + torch.sigmoid(f(m.beta_w[w])) * E[t][Do:]                            # kelime basina sonme
        C.append(torch.cat([order, h]))
    X = [c.clone() for c in C]
    enter = (lambda x: unit(x)) if getattr(m, "norm_inputs", False) else (lambda x: x)   # Oe15: girdi kureye
    for i, L in enumerate(m.moves):                          # hareket: yone gore en yakin `active` start
        S, F_ = f(L.start), f(L.finish)
        X = [enter(x) for x in X]
        for t in range(T):
            cos = torch.stack([unit(X[t]) @ unit(s) for s in S])
            order_ = sorted(range(len(S)), key=lambda a: -float(cos[a]))[:L.active]
            a = torch.softmax(torch.stack([-(2 - 2 * cos[j]) * math.exp(float(L.S_v)) for j in order_]), 0)
            X[t] = X[t] + sum(a[q] * (F_[j] - S[j]) for q, j in enumerate(order_))
        if i in m.attn_after:                                # attention: butun konumlar ayni onceki durumdan
            att = m.attns[m.attn_after.index(i)]
            Wq, Wk, Wv, Wo = f(att.W_q), f(att.W_k), f(att.W_v), f(att.W_o)
            before = [unit(x) for x in X]
            X = [enter(x) for x in X]
            new = []
            for t in range(T):
                add = torch.zeros(m.d, dtype=torch.float64)
                for hd in range(att.heads):
                    r = slice(hd * att.dim, (hd + 1) * att.dim)
                    q = Wq[r] @ before[t]
                    if att.W_qc is not None:
                        q = q + f(att.W_qc)[r] @ unit(C[t])
                    s = []
                    for j in range(t + 1):
                        if att.first:                         # anahtar bir ONCEKI konumun ham zinciri; konum 0: 0
                            k = Wk[r] @ unit(C[j - 1]) if j > 0 else torch.zeros(att.dim, dtype=torch.float64)
                        else:
                            k = Wk[r] @ before[j]
                        s.append(q @ k / math.sqrt(att.dim) - (float(att.m[hd]) * (t - j) if att.m is not None else 0))
                    a = torch.softmax(torch.stack(s), 0)
                    add = add + Wo[:, r] @ sum(a[j] * (Wv[r] @ before[j]) for j in range(t + 1))
                new.append(X[t] + add)
            X = new
    out = []
    for t in range(T):                                       # puan(w) = P[w] . (2 e^S_p norm(C_m) + R_PC C_t)
        z = 2 * math.exp(float(m.S_p)) * unit(X[t])
        if m.readout is not None:
            R = f(m.readout.M) if m.readout.M is not None else f(m.readout.A) @ f(m.readout.B).T
            z = z + R @ C[t]
        lp = torch.log_softmax(P @ z, 0)
        led = m.ledger
        if led is not None:                                  # defter: j < t - skip, benzer zincirden sonra gelen kelime
            cand = [j for j in range(T) if j < t - led.skip]
            p_cache = torch.zeros(n, dtype=torch.float64)
            sim_max = 0.0
            if cand:
                qn = unit(C[t])
                if led.relation is not None:
                    RC = f(led.relation.M) if led.relation.M is not None else f(led.relation.A) @ f(led.relation.B).T
                    qn = unit(qn + RC @ qn)
                sims = torch.stack([qn @ unit(C[j]) for j in cand])
                wts = torch.softmax(sims * math.exp(float(led.S_c)), 0)
                for q_, j in enumerate(cand):
                    p_cache[int(tokens[j + 1])] += wts[q_]
                sim_max = float(sims.max())
            if cand:
                g_in = unit(X[t]) @ f(led.gate_d) + float(led.gate_0) + (float(led.gate_s) * sim_max if led.gate_s is not None else 0)
                g = torch.sigmoid(g_in)
            else:
                g = torch.tensor(0.0, dtype=torch.float64)
            lp = torch.log((1 - g) * lp.exp() + g * p_cache)
        out.append(lp)
    return torch.stack(out)


def trained(cfg, steps=5, **kw):
    """Birkac adim egitilmis model: yeni parcalar sifirdan ayrilmis olsun."""
    m = PointRelation(**cfg, **kw)
    tokens, mask = tokens_and_mask(cfg["n"])
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(steps):
        opt.zero_grad()
        m.loss(tokens, mask).backward()
        opt.step()
    return m.eval()


# --- 1. MODEL: yeni parcalar sifir etkiyle, ayri ureteçten; nedensel; hizli kayip yolu tam tabloyla ayni
def t_model():
    tokens, mask = tokens_and_mask()
    with torch.no_grad():
        for rank in (8, None):
            d = float((PointRelation(**BASE, **OFF, rank=rank).scoreboard(tokens)
                       - PointRelation(**BASE, **ON, rank=rank).scoreboard(tokens)).abs().max())
            check("model: rank %s, yeni parcalar acik, adim 0'da zeminle ayni puan (Oe4)" % rank, d < 1e-5, "fark %.1e" % d)
    d = PointRelation(**BASE, rank=8)
    zemin = (d.embed_delta is None and d.readout is None and d.ledger.relation is None and len(d.attns) == 1
             and d.attns[0].m is None and d.config() == PointRelation(**BASE, **OFF, rank=8).config())
    with torch.no_grad():
        nl = PointRelation(**BASE, **OFF, rank=8, ledger=False).scoreboard(tokens)
    check("model: varsayilan zemin; scoreboard deftersiz de log p (normalize)",
          zemin and float(nl.logsumexp(-1).abs().max()) < 1e-5)
    a = PointRelation(**BASE, **dict(OFF, embed=True), rank=8)
    b = PointRelation(**BASE, **ON, rank=8)
    check("model: bir parcanin baslangici digerlerinin acik olmasina bagli degil",
          torch.equal(a.embed_delta, b.embed_delta) and torch.equal(a.moves[1].start, b.moves[1].start)
          and torch.equal(a.attns[0].W_q, b.attns[0].W_q))
    m = trained(BASE, rank=8, **ON)
    moved = [m.embed_up, m.readout.B, m.ledger.relation.B, m.attns[1].W_o, m.attns[0].m, m.attns[0].W_qc]
    check("model: 5 adimda yeni parcalar sifirdan ayrildi (gradyan akiyor)", all(float(p.abs().max()) > 0 for p in moved))
    with torch.no_grad():
        r1 = float((PointRelation(**BASE, **OFF, rank=8).scoreboard(tokens)
                    - PointRelation(**BASE, **OFF, rank=8, gate_sim=False).scoreboard(tokens)).abs().max())
    g = trained(BASE, rank=8, gate_sim=False)
    check("model: gate_sim=False (R1) adim 0'da zeminle ayni, benzerlik girdisi yok; egitilir",
          r1 < 1e-6 and g.ledger.gate_s is None and float(g.ledger.gate_d.abs().max()) > 0, "fark %.1e" % r1)
    t2 = tokens.clone()
    t2[:, 15] = (t2[:, 15] + 1) % 30
    with torch.no_grad():
        d = float((m.scoreboard(tokens)[:, :15] - m.scoreboard(t2)[:, :15]).abs().max())
        check("model: nedensel -- 15. token degisince onceki konumlar ayni", d < 1e-5, "fark %.1e" % d)
        total, nll = m.loss(tokens, mask, parts=True)
        lp = m.scoreboard(tokens)[:, :-1].gather(-1, tokens[:, 1:, None])[..., 0]
        w = mask[:, 1:].float()
        ref = float(-(lp * w).sum() / w.sum())
        check("model: hizli kayip yolu == scoreboard'un dogru kelimedeki log p'si", abs(float(nll) - ref) < 1e-5,
              "%.6f / %.6f" % (float(nll), ref))
        rows = torch.cat([torch.nonzero(mask[:, 1:].reshape(-1))[:, 0], torch.full((5,), -1)])
        l1, l2 = float(m.loss(tokens, mask)), float(m.loss(tokens, mask, rows=rows))
        check("model: rows'lu kayip rows'suz ile ayni", abs(l1 - l2) < 1e-5, "%.6f / %.6f" % (l1, l2))
        m2 = PointRelation.from_package({"config": m.config(), "weights": m.state_dict()})
        check("model: config + agirliklar -> ayni model",
              float((m.scoreboard(tokens) - m2.scoreboard(tokens)).abs().max()) == 0.0)
    models = (("butun parcalar acik", m), ("zemin", trained(BASE, rank=8, **OFF)),
              ("tam rank", trained(BASE, rank=None, **ON)))
    with torch.no_grad():
        for label, mm in models:
            ref = reference_logp(mm, tokens[0])
            got = mm.double().scoreboard(tokens[:1])[0]
            mm.float()
            # olasilikta ve log p > -15'te: defter karisimindaki p_defter + 1e-12 tabani log p ~ -36'yi -27,6'da tutar
            dp, dl = float((ref.exp() - got.exp()).abs().max()), float((ref - got).abs()[ref > -15].max())
            check("model: tasarimin formulleri bagimsiz ve saf yazilinca (dongulerle) ayni p -- %s" % label,
                  dp < 1e-6 and dl < 1e-5, "p farki %.1e  log p farki %.1e (icerik yarisi fp32)" % (dp, dl))
    h = m.health(tokens, mask)
    need = {"vec_each_0", "norm_in_3", "gate", "gate_dir", "gate_sim", "ledger_eff", "attn0_m", "attn1_dist",
            "embed_shift", "readout", "chain_sim", "induction_q", "pred_distinct"}
    check("model: saglik anahtarlari (Oe3, Oe6, Oe9, Oe15)", need <= set(h), ", ".join(sorted(need - set(h))))
    # gate 1'e doyunca (logit > ~17, fp32'de sigmoid = 1) gradyan NaN olmamali; S_c cok buyuse de defter agirliklari 1'e toplanmali
    x = trained(BASE, rank=8, **ON)
    with torch.no_grad():
        x.ledger.gate_0.fill_(40.0)
        x.ledger.S_c.fill_(12.0)
    x.zero_grad()
    x.loss(tokens, mask).backward()
    finite = all(torch.isfinite(p.grad).all() for p in x.parameters() if p.grad is not None)
    with torch.no_grad():
        W_c, _, valid = x.ledger.neighbors(x.chain(tokens))
        rows = W_c.sum(-1)
        sums = bool(((rows[valid] - 1).abs() < 1e-5).all()) and bool((rows[~valid] == 0).all())
        lp = x.scoreboard(tokens)
    check("model: gate doyunca gradyan sonlu; S_c buyukken defter agirliklari 1'e toplaniyor; tablo sonlu",
          finite and sums and bool(torch.isfinite(lp).all()))


# --- 2. DECOMPOSE: parcalarin toplami modelin kendisi; puan farki parcalara TAM ayrilir; model degisince durur
def t_decompose():
    for rank in (8, None):
        m = trained(BASE, rank=rank, **ON)
        tokens = tokens_and_mask()[0][0]
        R = DC.forward_parts(m, tokens)
        DC.verify(m, R, tokens)                                   # tutuyor: sessiz
        worst = 0.0
        for t in (3, 11, 22):
            for a, b in ((1, 2), (5, None), (int(tokens[t + 1]) if t + 1 < len(tokens) else 0, 7)):
                ex = DC.explain(m, R, tokens, t, a, b)
                diff = (float(R["score"][t, a] - (R["score"][t].mean() if b is None else R["score"][t, b])))
                worst = max(worst, abs(ex["total"] - diff), abs(sum(ex["parts"].values()) - ex["total"]))
        keys = set(DC.explain(m, R, tokens, 5, 1, 2)["parts"])
        full = {"chain_order", "chain_content", "readout"} | {"layer_%d" % i for i in range(4)} | {
            "attn%d_head_%d" % (k, h) for k in range(2) for h in range(2)}
        check("decompose: rank %s -- puan farki == parcalarin toplami, butun parcalar var" % rank,
              worst < 1e-3 and keys == full, "en buyuk fark %.1e" % worst)
    m = trained(BASE, rank=8, **ON)
    tokens = tokens_and_mask()[0][0]
    R = DC.forward_parts(m, tokens)
    with torch.no_grad():
        m.moves[1].finish.add_(1.0)                                # model degisti, decompose eskidi
    try:
        DC.verify(m, R, tokens)
        fired = False
    except AssertionError:
        fired = True
    check("decompose: modelle tutmazsa durur (verify)", fired)
    z = PointRelation(**BASE, **OFF, rank=8).eval()
    R = DC.forward_parts(z, tokens)
    DC.verify(z, R, tokens)
    check("decompose: zeminde (yeni parcalar kapali) de tutuyor", "readout" not in DC.explain(z, R, tokens, 5, 1, 2)["parts"])


# --- 3. DIAGNOSE: kayda hazir, rapor/ozet/sayfa kayittan, sayfa JS'i gecerli, okuma ayni devam, matematik sozlugu
def t_diagnose():
    words = [a + b for a in "abcdef" for b in "abcde"]
    vocab = list(DS.SPECIAL) + words
    m = trained(dict(BASE, n=len(vocab)), rank=8, **ON)
    prompts = [["ornek 1", "aa ab ac ad ae"], ["sonda 1", "ba bb bc"]]
    stories = DG.compute(m, vocab, "KUCUK t0", prompts, steps=10, log=lambda s: None)
    rows = [r for s in stories for r in s["rows"]]
    native = json.loads(json.dumps(stories)) == stories
    for s in stories:
        s["focus"] = DG.focus(s)
    page = DG.page(stories, "KUCUK Decompose")
    start = page.index('id="data">') + len('id="data">')
    page_ok = json.loads(page[start:page.index("</script>", start)])[0]["label"] == "ornek 1"
    report_ok = "PARCALAR" in "\n".join(DG.report(stories)) or not any(s["focus"] for s in stories)
    summary_ok = any("KUCUK t0" in line for line in DG.summary(stories)) and any(
        "dongu anatomisi" in line for line in DG.summary(stories))
    # a b c d x | a b c d x | a b c d y: ikinci tur k=1 ile bes kez devam eder, ucuncu turda k=2 devam etmez (y);
    # baslangictan (12) onceki token'lar sayilmaz
    rp = DG.repeat_profile([("a b c d x a b c d x a b c d y".split(), 0)], n=4, kmax=3)
    rp_late = DG.repeat_profile([("a b c d x a b c d x a b c d y".split(), 12)], n=4, kmax=3)
    summary_ok = summary_ok and rp == {1: (5, 5), 2: (0, 1), 3: (0, 0)} and rp_late == {1: (2, 2), 2: (0, 1), 3: (0, 0)}
    fields = rows and all({"heads", "vectors", "readout", "cache", "gate_dir", "gate_sim"} <= set(r) for r in rows)
    check("diagnose: JSON'a hazir; rapor, ozet ve sayfa kayittan; yeni alanlar satirda",
          native and page_ok and report_ok and summary_ok and fields, "%d secim" % len(rows))
    js_ok, note = True, "node yok, JS sinamasi atlandi"
    if shutil.which("node"):
        code = page[page.rindex("<script>") + len("<script>"):page.rindex("</script>")]
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
            f.write(code)
        r = subprocess.run(["node", "--check", f.name], capture_output=True, text=True)
        os.remove(f.name)
        js_ok, note = r.returncode == 0, "node --check " + ("temiz" if r.returncode == 0 else r.stderr[:160])
    check("diagnose: gezgin sayfasinin betigi JS olarak gecerli", js_ok, note)
    rd = DG.read(m, vocab, "KUCUK t0", prompts, steps=10, log=lambda s: None)
    same = all(r["tokens"] == s["tokens"] and r["text"] == s["text"] for r, s in zip(rd, stories))
    ev = DG.reading_prompts(["aa ab", "zz aa", "ba bb", "ac ad", "ae ba", "bb bc", "bd be"],
                            {a: i for i, a in enumerate(vocab)}, n=2)
    check("diagnose: okuma decompose'la ayni devami yazar; okuma istemleri sonda disi, sirayla",
          same and ev == [["eval 1", "ae ba"], ["eval 2", "bb bc"]], str(ev))
    math_vocab = [str(i) for i in range(10)] + ["+", "=", "<eos>"]
    mm = trained(dict(BASE, n=len(math_vocab)), rank=8, **ON)
    prompts, tag = DG.package_prompts(None, math_vocab)
    ms = DG.compute(mm, math_vocab, "MAT t0", prompts[:2], steps=5, log=lambda s: None)
    first = ms[0]["tokens"][:ms[0]["n_prompt"]]
    check("diagnose: matematik sozlugunde istemler rakam rakam, decompose calisiyor",
          tag == "_math" and first == ["<eos>", "1", "2", "3", "+", "4", "5", "6", "="] and all(s["rows"] for s in ms),
          " ".join(first))


# --- 4. EGITIM: tam yedekte kayit ve decompose; saglik olcutsuz da yaziliyor; SURDURME == KESINTISIZ (kural 1)
def t_train():
    import train_19 as TR
    vocab = [str(i) for i in range(10)] + ["+", "=", "<eos>"]
    g = torch.Generator().manual_seed(11)
    q = torch.randint(0, 12, (40, 14), generator=g)
    q[:, 0] = 12
    filled = torch.ones_like(q, dtype=torch.bool)
    targets = torch.zeros_like(filled)
    targets[:, 9:] = True
    metric = lambda m, side, full=False: {"accuracy": 0.0, "ce": 1.0, "diag": {}}
    cfg = dict(d_order=16, d_content=16, vectors=8, active=2, layers=4, attn_heads=2, attn_dim=4, t_max=40, rank=8, **ON)
    root = tempfile.mkdtemp()
    try:
        def go(name, steps, resume=None, extra=None, **kw):
            run = TR.RUNS[name] = TR.Run(name, root)
            TR._run(run, (q, filled, targets), len(vocab), metric, "cpu", 2e-3, steps, 0, 8, 4, 4, weights_every=4,
                    resume=resume, vocab=vocab, extra=extra, model_kw=dict(cfg, **kw))
            return torch.load(f"{root}/{name}/t{steps}.pt", weights_only=False)
        full = go("A", 8, extra=dict(trial="egitim", changed={"lr": 0.002}))
        files = set(os.listdir(f"{root}/A/decompose"))
        log = open(f"{root}/A/gunluk.txt", encoding="utf-8").read()
        written = ({"t4_math.json", "t8_math.json", "t8_math.html"} <= files and "decompose t8:" in log
                   and "DENEME EGITIM   degisen {'lr': 0.002}" in log and full["trial"] == "egitim")
        health = "saglik  hareket" in log and "yeni |dE|" in log
        go("B", 4)
        resumed = go("B", 8, resume=f"{root}/B/t4.pt")
        diff = max(float((full["weights"][k].float() - resumed["weights"][k].float()).abs().max()) for k in full["weights"])
        zero = go("Z", 4, **OFF, induction_query=False)
        base_ok = "YOK (zemin)" in open(f"{root}/Z/gunluk.txt", encoding="utf-8").read() and zero["config"]["embed"] is False
        try:
            go("B", 12, resume=f"{root}/B/t8.pt", rank=None)
            refused = False
        except ValueError:
            refused = True
    finally:
        shutil.rmtree(root, ignore_errors=True)
    check("egitim: tam yedekte decompose (matematik istemleri) + gunlukte satir; deneme turu pakette ve gunlukte", written)
    check("egitim: olcut saglik vermese de saglik satiri (yeni parcalarin boyu)", health)
    check("egitim: surdurme (4 + 4) == kesintisiz 8 adim", diff == 0.0, "agirlik farki %.1e" % diff)
    check("egitim: zemin ayariyla kosu; config farkliysa surdurme BASLAMAZ", base_ok and refused)
    mm = PointRelation(**BASE, rank=8, induction_query=True)
    opt = TR._optimizer(mm, 1e-3, 0.1)
    names = {n for n, p in mm.named_parameters() if TR.DECAY.match(n)}
    groups_ok = (type(opt).__name__ == "AdamW" and [g["weight_decay"] for g in opt.param_groups] == [0.1, 0.0]
                 and "lam_w" not in names and "S_p" not in names and "attns.0.W_qc" in names
                 and type(TR._optimizer(mm, 1e-3, 0.0)).__name__ == "Adam")
    check("egitim: weight decay (L1) yalniz hareket ve attention matrislerinde; 0'da Adam (zemin)", groups_ok,
          "%d decay parametresi" % len(names))
    f = lambda s, **k: TR.lr_at(s, 2e-3, 100, **k)
    plan = (abs(f(0, warmup=10) - 2e-4) < 1e-12 and abs(f(9, warmup=10) - 2e-3) < 1e-12 and f(50) == 2e-3
            and abs(f(80, decay_start=80) - 2e-3) < 1e-12 and abs(f(100, decay_start=80) - 2e-4) < 1e-12
            and abs(f(100, decay_start=80, decay_floor=0.05) - 1e-4) < 1e-12)
    check("egitim: LR plani -- isinma dogrusal, sabit, sogutma lr -> lr x taban", plan,
          "adim 0 %.1e, 9 %.1e, 100 %.1e" % (f(0, warmup=10), f(9, warmup=10), f(100, decay_start=80)))


# --- 5. HIKAYE: ayni egitim yolu hikaye olcutuyle -- saglik ve dongu olcutten, tam yedekte sonda istemleriyle decompose
def t_stories():
    import numpy as np
    import train_19 as TR
    words = [a + b for a in "abcdef" for b in "abcde"]
    vocab = list(DS.SPECIAL) + words
    eos = vocab.index(DS.EOS_TOKEN)
    rng = np.random.default_rng(12)
    stream = np.concatenate([np.append(rng.integers(4, len(vocab), rng.integers(4, 18)), eos) for _ in range(60)])
    W, M = DS.make_windows(stream.astype(np.int64), 20, story_id=eos)
    W, M = torch.from_numpy(W), torch.from_numpy(M)
    train, heldout = (W[:40], M[:40]), (W[40:], M[40:])
    metric = DS.make_metric(train, heldout, eos=eos, device="cpu", limit=8, vocab=vocab,
                            prompts=["aa ab ac", "ba bb bc bd"])
    cfg = dict(d_order=16, d_content=16, vectors=8, active=2, layers=4, attn_heads=2, attn_dim=4, t_max=64, rank=8, **ON)
    root = tempfile.mkdtemp()
    try:
        run = TR.RUNS["S"] = TR.Run("S", root)
        TR._run(run, (train[0], train[1], train[1]), len(vocab), metric, "cpu", 2e-3, 8, 0, 8, 4, 4, weights_every=4,
                vocab=vocab, model_kw=cfg)
        k = torch.load(f"{root}/S/t8.pt", weights_only=False)
        files = set(os.listdir(f"{root}/S/decompose"))
        log = open(f"{root}/S/gunluk.txt", encoding="utf-8").read()
    finally:
        shutil.rmtree(root, ignore_errors=True)
    done = k.get("done") and math.isfinite(k["heldout_acc"]) and math.isfinite(k["heldout_ce"]) and "BITTI" in log
    health = "saglik  hareket" in log and "yeni |dE|" in log and "dongu" in log
    backup = {"t4.json", "t8.json", "t8.html"} <= files and not any("_math" in f for f in files) and "decompose t8:" in log
    check("hikaye: olcut data_stories_19 -- tam olcum sonlu, saglik ve dongu olcutten, decompose sonda istemleriyle",
          done and health and backup, "dosyalar %s" % sorted(files))


# --- 6. TURKCE ILISKI VERISI (data_tr_19): pencere = butun parca, cevap kurali, sinav uretimi, BPE ile decompose
def t_tr():
    import data_tr_19 as TR
    parts = ["Ali Kaya'nın annesi Ayşe Kaya'dır.", "Ali Kaya'nın annesi kimdir? Ayşe Kaya'dır.",
             "Kayıtlara göre Ayşe Kaya'nın kardeşi Can Kaya'dır.", "Can Kaya bir kişidir."] * 5
    tok = TR.train_bpe(parts, target=300)
    ids = TR.encode(tok, parts)
    d = {"ids": torch.tensor([i for x in ids for i in x], dtype=torch.int16),
         "offsets": torch.tensor([0] + list(np.cumsum([len(x) for x in ids]))), "vocab": TR.vocab_of(tok)}
    W, M = TR.make_windows(d)
    eos = d["vocab"].index(TR.EOS)
    rows = []
    for w, mk in zip(W.tolist(), M.tolist()):
        n = sum(mk)
        rows.append(w[0] == eos and w[n - 1] == eos and eos not in w[1:n - 1] and not any(mk[n:]))
    whole = all(rows) and [TR.detok(tok, w[1:sum(mk) - 1]) for w, mk in zip(W.tolist(), M.tolist())] == parts
    rule = (TR.correct("Ayşe Kaya'dır.", "Ayşe Kaya") and TR.correct("Ayşe Kaya", "Ayşe Kaya")
            and not TR.correct("Ayşe Kayalar.", "Ayşe Kaya") and not TR.correct("Can Kaya'dır.", "Ayşe Kaya"))
    check("tr: pencere = <eos> BUTUN parca <eos> (yarim cumle yok), gidis-donus; cevap kurali (kuyruktaki ek)",
          whole and rule, "pencere %s" % (tuple(W.shape),))
    m = PointRelation(len(d["vocab"]), d_order=16, d_content=16, vectors=8, active=2, layers=4, attn_heads=2,
                      attn_dim=4, t_max=40, rank=8, seed=5).eval()
    said = TR.ask(m, tok, ["Ali Kaya'nın annesi kimdir?", "Can Kaya'nın annesi kimdir?"], device="cpu", steps=6)
    vocab = d["vocab"]
    toks, n_prompt = DC.greedy(m, TR.encode(tok, ["Ali Kaya'nın annesi kimdir?"])[0], vocab,
                               {a: i for i, a in enumerate(vocab)}, steps=4)
    text_ok = DG.detok([vocab.index("▁Ali"), vocab.index("▁Kaya"), vocab.index("'")], vocab).startswith("Ali Kaya'")
    check("tr: sinav uretimi (ask) ve BPE istemiyle decompose; diagnose BPE metnini cozer",
          len(said) == 2 and all(isinstance(s, str) for s in said) and n_prompt == 1 + len(TR.encode(tok, ["Ali Kaya'nın annesi kimdir?"])[0])
          and DG.is_bpe(vocab) and not DG.is_math(vocab) and text_ok, str(said))
    d["exam"] = {k: [("Ali Kaya'nın annesi kimdir?", "Ayşe Kaya"), ("Can Kaya'nın annesi kimdir?", "Ayşe Kaya")] * 2
                 for k in TR.SHOW}
    lines = TR.exam_text(m, tok, d, per_split=3)
    with tempfile.TemporaryDirectory() as root:
        for f in ("w4.pt", "t8.pt", "w12.pt", "model_x.pt"):
            open(os.path.join(root, f), "w").close()
        pick = (os.path.basename(TR.latest_package(root)), os.path.basename(TR.latest_package(root, 8)))
    check("tr: nitel sinav (exam_text) bolme basina baslik + soru/model/dogru; latest_package en yeni ya da istenen adim",
          len(lines) == 4 * len(TR.SHOW) and all(lines[4 * i].startswith("-- " + k) for i, k in enumerate(TR.SHOW))
          and pick == ("w12.pt", "t8.pt"), "%d satir %s" % (len(lines), pick))
    surface = lambda x, rr, a: ("%s'nın %s kimdir?" % (x, " ".join(rr)), a)
    G = {"olgu": {("Ali Kaya", "kardeşi"): "Can Kaya"}}
    lines, st = TR.hop_report(m, tok, [("Ali Kaya", ("annesi", "kardeşi"), "Ayşe Kaya", "Can Kaya")], surface, G)
    check("tr: 2R teshisi (hop_report) -- iki adim + kisayol sorulur, ayrilan token'da decompose (verify'li)",
          len(st) == 1 and st[0][3] in ("dogru", "kopru", "kisayol", "diger") and any("PARCALAR" in s for s in lines)
          and any(s.strip().startswith("kisayol") for s in lines), str(st))
    tokens = torch.tensor([eos] + TR.encode(tok, ["Ali Kaya'nın annesi kimdir?"])[0])
    R = DC.forward_parts(m, tokens)
    t = len(tokens) - 1
    st = TR.stages(m, R, t)
    ap = TR.answer_parts(m, R, tokens, t, vocab.index("▁Ayşe"), [1, 2])
    n = TR.count_written(TR.possessor_index(parts), parts, "Ali_Kaya", ("annesi",), "Ayse_Kaya")
    check("tr: 2R analizi -- asamalar C_m'de biter, dogru token'in parcalari toplami tutar, egitimde kac kez yazili",
          [s for s, _ in st][-1] == "katman 3" and abs(sum(ap[k] for k in TR.part_rows(m)[:-1]) - ap["toplam"]) < 1e-3
          and 0 <= ap["A1 ad"] <= 1 and n == 10, "asama %s  toplam %.4f  yazili %d" % ([s for s, _ in st], ap["toplam"], n))
    # tasarim modeli (iki attention, R_PC, E, R_CC, mesafe): yeni parcalar sifirdan oynatilir ki etkileri olsun
    md = PointRelation(len(vocab), d_order=16, d_content=16, vectors=8, active=2, layers=4, attn_heads=2, attn_dim=4,
                       t_max=40, rank=8, seed=5, embed=True, readout=True, chain_sim=True, distance=True,
                       attn_after=(0, 2), norm_inputs=True).eval()
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for p_ in md.parameters():
            p_.add_(0.05 * torch.randn(p_.shape, generator=g))
    R = DC.forward_parts(md, tokens)
    DC.verify(md, R, tokens)
    ap = TR.answer_parts(md, R, tokens, t, vocab.index("▁Ayşe"), [1, 2], focus=[5])
    q2 = "Ali Kaya'nın annesinin kardeşi kimdir?"
    t2s = torch.tensor([eos] + TR.encode(tok, [q2])[0])
    top = vocab[int(DC.forward_parts(md, t2s)["score"][-1].argmax())]
    patched = TR.patch_name_head(md, tok, q2, "Ali Kaya'nın annesi kimdir?", [("dogru", "Can Kaya")])
    import tr_graph_19 as V19
    surf = lambda x, rr, a: ("%s'nın %s kimdir?" % (x.replace("_", " "), " ".join(V19.TR_ILISKI[r] for r in rr)),
                             a.replace("_", " "))
    tip = {e: "KISI" for e in ("Ali_Kaya", "Ayse_Kaya", "Can_Kaya")}
    pools = {"KISI": torch.tensor(sorted({TR.encode(tok, [e.replace("_", " ")])[0][0] for e in tip}))}
    lines, table, ranks, type_ranks, lens = TR.hop_analysis(
        md, tok, [("Ali_Kaya", ("annesi", "kardesi"), "Ayse_Kaya", "Can_Kaya")], surf,
        {"olgu": {("Ali_Kaya", "kardesi"): "Can_Kaya"}}, tip, pools, TR.possessor_index(parts), parts, detail=1)
    check("tr: tasarim modelinde analiz -- parcalar toplami (R_PC, iki attention dahil), mudahalesiz ileri hesap modelin "
          "kendi puani, konum x asama okumasi",
          abs(sum(ap[k] for k in TR.part_rows(md)[:-1]) - ap["toplam"]) < 1e-3 and "readout" in TR.part_rows(md)
          and "att2 ad" in ap and 0 <= ap["A2 odak"] <= 1 and patched[1].split("ilk 5: ")[1].split()[0] == top
          and ("2R", "okuma") in ranks and ("kopru", "r1 kelimesi", "attention 2") in lens,
          "toplam %.4f  ust %s  %s" % (ap["toplam"], top, patched[1][:60]))
    R2 = DC.forward_parts(md, t2s)
    t2 = len(t2s) - 1
    same, moved = TR.reroute(md, R2, t2, 1, []), TR.reroute(md, R2, t2, 1, [(t2 - 1, 1)])
    check("tr: yeniden yonlendirme -- mudahalesiz modelin kendi puani, agirlik tasinca puan degisir",
          torch.allclose(same, R2["score"][t2], atol=1e-4) and not torch.allclose(moved, same, atol=1e-4),
          "fark %.2e / %.2e" % (float((same - R2["score"][t2]).abs().max()), float((moved - same).abs().max())))
    Gc = {"olgu": {("X", "r2"): "A"}, "tip": {"X": "KISI", "Y": "SEHIR", "Z": "KISI"}, "sema": {"r2": {"KISI"}}}
    got = [TR.chain_class(Gc, x, "r2", a) for x, a in (("X", "X"), ("X", "A"), ("X", "B"), ("Y", "B"), ("Z", "B"))]
    check("tr: 2R zincir sinifi (DONUS, AYNI, AYIRT, YOK, EKSIK)", got == ["DONUS", "AYNI", "AYIRT", "YOK", "EKSIK"],
          " ".join(got))
    from looped_19 import LoopedRelation
    mb = LoopedRelation(len(vocab), d=16, rank=4, vectors=8, active=2, heads=2, head_dim=4, k_max=3, t_max=40, seed=2).eval()
    fake = {"exam": {"ezber_olgu": [("Ali Kaya'nın annesi kimdir?", "Ayşe Kaya"), ("Ayşe Kaya'nın kardeşi kimdir?", "Can Kaya"),
                                    ("Ali Kaya'nın annesinin kardeşi kimdir?", "Can Kaya")]}}
    rep = TR.passes_report(mb, tok, fake, splits=("ezber_olgu",))
    check("tr: LoopedRelation durma raporu -- bolme basina son konumun gecisi (baslangicta hepsi 1)",
          len(rep) == 2 and rep[1].split()[1] == "3" and rep[1].split()[4] == "1.0" and rep[1].split()[8] == "1.0", rep[1])
    gparts =["Ali Kaya'nın annesinin kardeşi Can Kaya'dır.", "Can Kaya'nın yaşadığı yerin bölgesi Marmara Bölgesi'dir."]
    gtok = TR.train_bpe(parts + gparts * 3, target=300, split_genitive=True)
    gv = TR.vocab_of(gtok)
    g1, g2 = ([gv[i] for i in TR.encode(gtok, [p])[0]] for p in gparts)
    check("tr: tamlayan eki ayri token (kardesi|nin, yer|in), ozel ad eki ve gidis-donus bozulmaz",
          g1[4:6] == ["▁annesi", "nin"] and g2[5:7] == ["▁yer", "in"] and g1[2:4] == ["'", "nın"]
          and all(gtok.decode(TR.encode(gtok, [p])[0]) == p for p in parts + gparts), str(g1))
    q1s, q2s = "Ali Kaya'nın kardeşi kimdir?", "Ali Kaya'nın kardeşinin yaşadığı yerin bölgesi neresidir?"
    mt = TR.MorphTokenizer.build(parts + gparts + [q1s, q2s], frozenset({"Ali", "Kaya", "Can", "Ayşe", "Marmara"}))
    mv = TR.vocab_of(mt)
    q1, q2 = ([mt.ix[TR.EOS]] + TR.encode(mt, [q])[0] for q in (q1s, q2s))
    k1, k2 = TR.relation_keys(q1, mv), TR.relation_keys(q2, mv)
    check("tr: kok ayri ek ayri (MorphTokenizer, tr_morph_19) -- gidis-donus birebir; 2R'de r1'in arama konumuna kadar onek "
          "1R'dekiyle AYNI; iliski arama konumlari (kardes i|nin, yer|in, bolge si)",
          all(mt.decode(TR.encode(mt, [p])[0]) == p for p in parts + gparts + [q1s, q2s])
          and [mv[x] for x in q2[5:8]] == ["▁kardeş", "i", "nin"] and q2[:k2[0] + 1] == q1[:k1[0] + 1]
          and [mv[q2[j]] for j in k2] == ["i", "▁yer", "si"], str([mv[x] for x in q2]))
    d2 = {"exam": {"x": [(str(i), str(i)) for i in range(40)]}}
    s1, s2 = TR.exam_pairs(d2, "x", 8), TR.exam_pairs(d2, "x", 8)
    check("tr: sinirli sinav sabit tohumlu ORNEK (ilk N degil: bolmeler iliskiye gore sirali); sinirsiz = butun bolme",
          s1 == s2 and len(s1) == 8 and s1 != d2["exam"]["x"][:8] and TR.exam_pairs(d2, "x") == d2["exam"]["x"],
          str([q for q, _ in s1]))
    two = ["Ali Kaya'nın annesi Ayşe Kaya'dır.", "Ayşe Kaya, Ali Kaya'nın annesidir."]
    ev = TR.training_evidence(TR.possessor_index(two), two, "Ali_Kaya", ("annesi",), "Ayse_Kaya")
    ps, cont = TR.answer_in_context(m, tok, two[0], "Ayşe Kaya")
    check("tr: hata ayiklama -- egitimdeki yon (cevap ifadeden sonra = ileri), cumlede cevaba kadar ogretmen zorlamasi",
          [f for _, f in ev] == [True, False] and len(ps) == len(TR.encode(tok, ["Ayşe Kaya"])[0])
          and all(0 <= p <= 1 for p in ps) and isinstance(cont, str), str(ev))


def reference_looped(m, tokens, stop):
    """LoopedRelation'in formulleri (TASARIM_19 "Bul-Bak Dongusu"), looped_19'un kodundan BAGIMSIZ, konum konum donguler,
    float64: tokens (T,) -> (log p (T, n), konum basina gecis).  Kod tasarimdan saparsa scoreboard'la tutmaz."""
    f = lambda z: z.detach().double()
    unit = lambda z: z / z.norm() if float(z.norm()) > 0 else z * 0
    T, d, P = len(tokens), m.d, f(m.P)
    dE = lambda w: f(m.embed_delta[w]) if m.embed_up is None else f(m.embed_up) @ f(m.embed_delta[w])
    E = [unit(P[w] + dE(w)) for w in tokens.tolist()]
    C = [sum(m.lam ** (t - i) * torch.roll(E[i], t - i) for i in range(t + 1)) for t in range(T)]   # yasi kadar kaydir
    mat = lambda rel: f(rel.M) if rel.M is not None else f(rel.A) @ f(rel.B).T      # carpim float64'te
    R = [mat(m.readout) @ c for c in C]
    RCC, skip = mat(m.ledger.relation), m.ledger.skip
    Dv = []
    for t in range(T):
        cand = [j for j in range(T) if j < t - skip]
        if not cand:
            Dv.append(torch.zeros(d, dtype=torch.float64))
            continue
        q = unit(unit(C[t]) + RCC @ unit(C[t]))
        wts = torch.softmax(torch.stack([q @ unit(C[j]) for j in cand]) * math.exp(float(m.ledger.S_c)), 0)
        Dv.append(sum(wts[i] * E[j + 1] for i, j in enumerate(cand)))           # benzer zincirden SONRA gelen
    a, b = f(m.source_weights)
    X = [unit(unit(C[t]) + a * unit(R[t]) + b * unit(Dv[t])) for t in range(T)]
    find = m.find
    Wq, Wk, Wv, Wo, dim = f(find.W_q), f(find.W_k), f(find.W_v), f(find.W_o), find.dim

    def move(layer, y):
        S, F_ = f(layer.start), f(layer.finish)
        cos = torch.stack([unit(y) @ unit(s) for s in S])
        best = sorted(range(len(S)), key=lambda j: -float(cos[j]))[:layer.active]
        w = torch.softmax(torch.stack([-(2 - 2 * cos[j]) * math.exp(float(layer.S_v)) for j in best]), 0)
        return sum(w[i] * (F_[j] - S[j]) for i, j in enumerate(best))

    alpha = None if m.alpha is None else f(m.alpha)

    def step(x, part, i):                                                      # 'input' ya da 'bounded' (nGPT)
        return unit(x) + part if alpha is None else unit(unit(x) + alpha[i] * unit(part))

    done, passes = [False] * T, [0] * T
    for _ in range(m.k_max):
        base = [unit(x) for x in X]
        Y = []
        for t in range(T):                                                     # find
            add = torch.zeros(d, dtype=torch.float64)
            for h in range(find.heads):
                r = slice(h * dim, (h + 1) * dim)
                s = []
                for j in range(t + 1):
                    if h < find.trace:                                          # anahtar: bir onceki konumun zinciri
                        k = Wk[r] @ unit(C[j - 1]) if j > 0 else torch.zeros(dim, dtype=torch.float64)
                    else:
                        k = Wk[r] @ base[j]
                    s.append((Wq[r] @ base[t]) @ k / math.sqrt(dim) - float(find.m[h]) * (t - j))
                A = torch.softmax(torch.stack(s), 0)
                add = add + Wo[:, r] @ sum(A[j] * (Wv[r] @ base[j]) for j in range(t + 1))
            Y.append(step(X[t], add, 0))
        for i, layer in enumerate(m.lookups):                                  # lookup 1, lookup 2
            Y = [step(y, move(layer, y), i + 1) for y in Y]
        new = [X[t] if done[t] else Y[t] for t in range(T)]
        for t in range(T):
            passes[t] += 0 if done[t] else 1
        if stop:
            change = [float((unit(new[t]) - unit(X[t])).norm()) for t in range(T)]
            worst = 0.0
            for t in range(T):                                                 # t'nin kendisi ve oncesi
                worst = max(worst, change[t])
                done[t] = done[t] or worst < m.stop_eps
            X = new
            if all(done):
                break
        else:
            X = new
    lp = torch.stack([torch.log_softmax(P @ (2 * math.exp(float(m.S_p)) * unit(x)), 0) for x in X])
    if m.gate_d is not None:                                                   # acik kopya: (1 - g) p + g p_defter
        rows = []
        for t in range(T):
            p = lp[t].exp()
            cand = [j for j in range(T) if j < t - skip]
            if cand:
                q = unit(unit(C[t]) + RCC @ unit(C[t]))
                wts = torch.softmax(torch.stack([q @ unit(C[j]) for j in cand]) * math.exp(float(m.ledger.S_c)), 0)
                pc = torch.zeros(len(P), dtype=torch.float64)
                for i, j in enumerate(cand):
                    pc[int(tokens[j + 1])] += wts[i]
                g = torch.sigmoid(unit(X[t]) @ f(m.gate_d) + float(m.gate_0))
                p = (1 - g) * p + g * pc
            rows.append(torch.log(p))
        lp = torch.stack(rows)
    return lp, passes


LOOP_CFG = dict(n=30, d=32, rank=8, vectors=16, active=4, heads=2, head_dim=4, trace_heads=1, k_max=3, t_max=40)


# --- 8. LOOPED RELATION (Bul-Bak Dongusu): adim 0'da etkisiz, gradyan akar, tasarimla ayni, nedensel (durma dahil), kayip == tablo, ayrisma tam
def t_looped():
    import train_19 as TR
    from looped_19 import LoopedRelation, model_from_package
    tokens, mask = tokens_and_mask()
    with torch.no_grad():
        m0 = LoopedRelation(**LOOP_CFG, seed=3).eval()
        C = m0.sources(tokens)["C"]
        ref = torch.log_softmax(m0.point(C) @ m0.P.T, -1)
        d0 = float((m0.scoreboard(tokens, stop=False) - ref).abs().max())
        p0 = m0.run(tokens, stop=True)[1]["passes"]
    check("looped: adim 0'da etkisiz (kaynak agirliklari, W_o, hareketler 0): puan = norm(C)'nin okumasi; durma 1. geciste",
          d0 < 1e-5 and bool((p0 == 1).all()), "fark %.1e  gecis %s" % (d0, p0.unique().tolist()))
    m = LoopedRelation(**LOOP_CFG, seed=3)
    start = {k: v.detach().clone() for k, v in m.state_dict().items()}
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(5):
        opt.zero_grad()
        m.loss(tokens, mask).backward()
        opt.step()
    moved = {k: float((v - start[k]).abs().max()) for k, v in m.state_dict().items() if k in start}
    need = ("source_weights", "find.W_o", "find.m", "lookups.0.finish", "lookups.1.finish", "ledger.relation.B",
            "embed_up", "readout.B", "ledger.S_c", "S_p")
    check("looped: 5 adimda her yeni parca 0'dan ayrildi (kaynaklar, bul, iki bak, R_CC, E, R_PC)",
          all(moved[k] > 0 for k in need), " ".join("%s %.0e" % (k, moved[k]) for k in need if moved[k] == 0))
    g = torch.Generator().manual_seed(4)
    with torch.no_grad():
        for p_ in m.parameters():
            p_.add_(0.1 * torch.randn(p_.shape, generator=g))
    m.eval()
    with torch.no_grad():
        mixed = None                           # durma yolu gercekten sinansin: bazi konumlar erken dursun, bazilari degil
        for e in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2):
            m.stop_eps = e
            if len(m.run(tokens[:1], stop=True)[1]["passes"][0].unique()) > 1:
                mixed = e
                break
        check("looped: durma kuralinin sinanacagi karisik bir esik var (konumlar farkli gecislerde duruyor)",
              mixed is not None, "esik %s" % mixed)
        mixed = mixed or 0.5
        for stop, eps in ((False, LOOP_CFG.get("stop_eps", 0.01)), (True, mixed)):
            m.stop_eps = eps
            ref, ref_passes = reference_looped(m, tokens[0], stop)
            got = m.double().scoreboard(tokens[:1], stop=stop)[0]
            passes = m.run(tokens[:1], stop=stop)[1]["passes"][0].tolist()
            m.float()
            dp = float((ref.exp() - got.exp()).abs().max())
            check("looped: tasarimin formulleri bagimsiz ve saf yazilinca ayni p, durma kurali %s" % ("ACIK" if stop else "yok"),
                  dp < 1e-10 and passes == ref_passes, "p farki %.1e  gecis %s" % (dp, passes))
        for label, extra in (("tam rank", dict(rank=None)), ("acik kopya kanali", dict(copy_gate=True)),
                             ("bounded adim (nGPT)", dict(step_norm="bounded"))):
            mx = LoopedRelation(**dict(LOOP_CFG, **extra), seed=6).eval()
            for p_ in mx.parameters():
                p_.add_(0.1 * torch.randn(p_.shape, generator=g))
            ref, _ = reference_looped(mx, tokens[0], False)
            got = mx.double().scoreboard(tokens[:1], stop=False)[0]
            dp = float((ref.exp() - got.exp()).abs().max())
            check("looped: tasarimin formulleri bagimsiz ve saf yazilinca ayni p -- %s" % label, dp < 1e-10, "p farki %.1e" % dp)
        m.stop_eps = mixed
        t2 = tokens.clone()
        t2[:, 15] = (t2[:, 15] + 1) % 30
        dc = float((m.scoreboard(tokens)[:, :15] - m.scoreboard(t2)[:, :15]).abs().max())
        check("looped: nedensel -- 15. token degisince onceki konumlar ayni, durma kurali dahil", dc < 1e-5, "fark %.1e" % dc)
        tr = m.trace(tokens[0], stop=True)
        whole = sum(q[2] * q[3][:, None] for q in tr["parts"])
        x_run = m.run(tokens[:1], stop=True)[0][0]
        dx = float((whole - x_run).abs().max()) / (1 + float(x_run.abs().max()))
        check("looped: tam ayrisma -- kaynaklar + her gecisin her adimi (carpanlariyla) son durumu birebir verir",
              dx < 1e-4 and torch.equal(tr["passes"], m.run(tokens[:1], stop=True)[1]["passes"][0]), "fark %.1e" % dx)
    # bounded adim: adim 0'da etkisiz (alpha 0), gradyan akar, durma acikken tasarimla ayni, nedensel, ayrisma tam
    with torch.no_grad():
        mb0 = LoopedRelation(**LOOP_CFG, step_norm="bounded", seed=3).eval()
        Cb = mb0.sources(tokens)["C"]
        db0 = float((mb0.scoreboard(tokens, stop=False) - torch.log_softmax(mb0.point(Cb) @ mb0.P.T, -1)).abs().max())
    mb = LoopedRelation(**LOOP_CFG, step_norm="bounded", seed=3)
    start_b = {k: v.detach().clone() for k, v in mb.state_dict().items()}
    opt = torch.optim.Adam(mb.parameters(), lr=1e-2)
    for _ in range(5):
        opt.zero_grad()
        mb.loss(tokens, mask).backward()
        opt.step()
    moved_b = {k: float((v - start_b[k]).abs().max()) for k, v in mb.state_dict().items() if k in start_b}
    need_b = ("alpha", "find.W_o", "find.W_q", "lookups.0.finish", "lookups.1.start", "source_weights", "embed_up")
    check("looped bounded: adim 0'da etkisiz (alpha 0); 5 adimda alpha ve parcalar degisti",
          db0 < 1e-5 and all(moved_b[k] > 0 for k in need_b), "fark %.1e  %s" % (db0, " ".join(k for k in need_b if moved_b[k] == 0)))
    with torch.no_grad():
        for p_ in mb.parameters():
            p_.add_(0.1 * torch.randn(p_.shape, generator=g))
        mb.eval()
        eps_b = None
        for e in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
            mb.stop_eps = e
            if len(mb.run(tokens[:1], stop=True)[1]["passes"][0].unique()) > 1:
                eps_b = e
                break
        mb.stop_eps = eps_b or 0.3
        ref, ref_passes = reference_looped(mb, tokens[0], True)
        got = mb.double().scoreboard(tokens[:1], stop=True)[0]
        passes_b = mb.run(tokens[:1], stop=True)[1]["passes"][0].tolist()
        mb.float()
        dpb = float((ref.exp() - got.exp()).abs().max())
        t2 = tokens.clone()
        t2[:, 15] = (t2[:, 15] + 1) % 30
        dcb = float((mb.scoreboard(tokens)[:, :15] - mb.scoreboard(t2)[:, :15]).abs().max())
        trb = mb.trace(tokens[0], stop=True)
        xb = mb.run(tokens[:1], stop=True)[0][0]
        dxb = float((sum(q[2] * q[3][:, None] for q in trb["parts"]) - xb).abs().max())
    check("looped bounded: durma acik tasarimla ayni p, nedensel, tam ayrisma (karisik esik %s)" % eps_b,
          eps_b is not None and dpb < 1e-10 and passes_b == ref_passes and dcb < 1e-5 and dxb < 1e-4,
          "p %.1e  gecis %s  nedensel %.1e  ayrisma %.1e" % (dpb, sorted(set(passes_b)), dcb, dxb))
    for copy_gate in (False, True):
        mc = LoopedRelation(**LOOP_CFG, copy_gate=copy_gate, seed=5)
        with torch.no_grad():
            for p_ in mc.parameters():
                p_.add_(0.1 * torch.randn(p_.shape, generator=g))
            total, nll = mc.loss(tokens, mask, parts=True)
            lp = mc.scoreboard(tokens, stop=False)[:, :-1].gather(-1, tokens[:, 1:, None])[..., 0]
            w = mask[:, 1:].float()
            ref_nll = float(-(lp * w).sum() / w.sum())
            rows = torch.cat([torch.nonzero(mask[:, 1:].reshape(-1))[:, 0], torch.full((5,), -1)])
            l_rows = float(mc.loss(tokens, mask, rows=rows))
        check("looped: kayip yolu == scoreboard'un dogru kelimedeki log p'si (K_MAX gecis), rows'lu ayni, acik kopya %s"
              % ("ACIK" if copy_gate else "yok"),
              abs(float(nll) - ref_nll) < 1e-5 and abs(l_rows - float(total)) < 1e-5, "%.6f / %.6f" % (float(nll), ref_nll))
    # denetimin buldugu kenarlar: gecersiz defter satiri, hedefsiz batch, arch'siz eski paket, kendi degisimi olcusu
    from looped_19 import unit_or_zero
    v = torch.zeros(1, 3, 4, requires_grad=True)
    valid = torch.tensor([[False, False, True]])
    out = unit_or_zero(v + torch.tensor([0.0, 0.0, 1.0])[None, :, None], valid)
    out.sum().backward()
    empty = LoopedRelation(**LOOP_CFG, seed=3).loss(tokens, torch.zeros_like(mask))
    old_pk = {"config": PointRelation(**BASE, **OFF, rank=8).config(), "weights": PointRelation(**BASE, **OFF, rank=8).state_dict()}
    settle0 = LoopedRelation(**LOOP_CFG, seed=3).eval().settle_passes(tokens)
    check("looped: gecersiz defter satiri 0 ve gradyani sinirli; hedefsiz batch kaybi sonlu; arch'siz paket PointRelation;"
          " kendi degisimi olcusu baslangicta 1",
          float(out[0, :2].abs().max()) == 0.0 and float(v.grad.abs().max()) < 10 and bool(torch.isfinite(empty))
          and isinstance(model_from_package(old_pk), PointRelation) and bool((settle0 == 1).all()),
          "grad %.1e  bos kayip %.3f" % (float(v.grad.abs().max()), float(empty)))
    m.stop_eps = m.config()["stop_eps"]                # esik yukarida sinama icin degistirildi; paketteki deger
    k = {"arch": m.arch, "config": m.config(), "weights": m.state_dict()}
    m2 = model_from_package(k)
    h = m.health(tokens, mask)
    line = TR._health_line(h, {"find": 0.1}, {"train": 0.1, "heldout": 0.2})
    check("looped: paketten ayni model (arch ile), saglik anahtarlari ve gunluk satiri",
          isinstance(m2, LoopedRelation) and float((m.scoreboard(tokens) - m2.scoreboard(tokens)).abs().max()) == 0.0
          and {"passes", "passes_at_max", "change_0", "vec_all_1", "src_a", "ledger_eff", "attn_m"} <= set(h)
          and line.startswith("saglik  gecis"), line[:70])
    vocab = [str(i) for i in range(10)] + ["+", "=", "<eos>"]
    gq = torch.Generator().manual_seed(11)
    q = torch.randint(0, 12, (40, 14), generator=gq)
    q[:, 0] = 12
    filled = torch.ones_like(q, dtype=torch.bool)
    metric = lambda mm, side, full=False: {"accuracy": 0.0, "ce": 1.0, "diag": {}}
    root = tempfile.mkdtemp()
    try:
        cfg = dict(LOOP_CFG, arch="looped_relation")
        cfg.pop("n")
        run = TR.RUNS["LOOP"] = TR.Run("LOOP", root)
        TR._run(run, (q, filled, filled), len(vocab), metric, "cpu", 2e-3, 4, 0, 8, 2, 4, weights_every=2,
                vocab=vocab, model_kw=cfg)
        pk = torch.load(f"{root}/LOOP/t4.pt", weights_only=False)
        log = open(f"{root}/LOOP/gunluk.txt", encoding="utf-8").read()
        ok = (pk["arch"] == "looped_relation" and isinstance(model_from_package(pk), LoopedRelation) and "saglik  gecis" in log
              and "mimari looped_relation" in log and "decompose t4: looped_relation icin yok" in log)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    check("looped: egitim dongusu arch='looped_relation' ile kurar, kaydeder, saglik satiri yazar", ok)


if __name__ == "__main__":
    print("tests (model_19)")
    for f in (t_model, t_decompose, t_diagnose, t_train, t_stories, t_tr, t_looped):
        f()
    print("\n%d GECTI   %d KALDI" % (sum(RESULTS), len(RESULTS) - sum(RESULTS)))
    sys.exit(0 if all(RESULTS) else 1)
