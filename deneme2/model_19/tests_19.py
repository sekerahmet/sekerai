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
OFF = dict(embed=False, readout=False, chain_sim=False, distance=False, attn_after=(0,))
ON = dict(embed=True, readout=True, chain_sim=True, distance=True, attn_after=(0, 2), induction_query=True)


def check(name, ok, note=""):
    RESULTS.append(bool(ok))
    print("  %-78s %s  %s" % (name, "GECTI" if ok else "KALDI", note))


def tokens_and_mask(n=30, B=3, T=24, seed=7):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, n, (B, T), generator=g)
    mask = torch.ones_like(tokens, dtype=torch.bool)
    mask[-1, T - 6:] = False
    return tokens, mask


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


if __name__ == "__main__":
    print("tests (model_19)")
    for f in (t_model, t_decompose, t_diagnose, t_train, t_stories, t_tr):
        f()
    print("\n%d GECTI   %d KALDI" % (sum(RESULTS), len(RESULTS) - sum(RESULTS)))
    sys.exit(0 if all(RESULTS) else 1)
