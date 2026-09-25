# -*- coding: utf-8 -*-
"""tests_19 -- model_19, decompose_19 ve diagnose_19'un kapilari.  Kucuk modellerde, CPU, saniyeler.

    python tests_19.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

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
ON = dict(embed=True, readout=True, chain_sim=True, distance=True, attn_after=(0, 2))


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
    a = PointRelation(**BASE, **dict(OFF, embed=True), rank=8)
    b = PointRelation(**BASE, **ON, rank=8)
    check("model: bir parcanin baslangici digerlerinin acik olmasina bagli degil",
          torch.equal(a.embed_delta, b.embed_delta) and torch.equal(a.moves[1].start, b.moves[1].start)
          and torch.equal(a.attns[0].W_q, b.attns[0].W_q))
    m = trained(BASE, rank=8, **ON)
    moved = [m.embed_up, m.readout.B, m.ledger.relation.B, m.attns[1].W_o, m.attns[0].m]
    check("model: 5 adimda yeni parcalar sifirdan ayrildi (gradyan akiyor)", all(float(p.abs().max()) > 0 for p in moved))
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
            "embed_shift", "readout", "chain_sim", "pred_distinct"}
    check("model: saglik anahtarlari (Oe3, Oe6, Oe9, Oe15)", need <= set(h), ", ".join(sorted(need - set(h))))


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
    summary_ok = any("KUCUK t0" in line for line in DG.summary(stories))
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
    cfg = dict(d_order=16, d_content=16, vectors=8, active=2, layers=4, attn_heads=2, attn_dim=4, t_max=40, rank=8)
    root = tempfile.mkdtemp()
    try:
        def go(name, steps, resume=None, **kw):
            run = TR.RUNS[name] = TR.Run(name, root)
            TR._run(run, (q, filled, targets), len(vocab), metric, "cpu", 2e-3, steps, 0, 8, 4, 4, weights_every=4,
                    resume=resume, vocab=vocab, model_kw=dict(cfg, **kw))
            return torch.load(f"{root}/{name}/t{steps}.pt", weights_only=False)
        full = go("A", 8)
        files = set(os.listdir(f"{root}/A/decompose"))
        log = open(f"{root}/A/gunluk.txt", encoding="utf-8").read()
        written = {"t4_math.json", "t8_math.json", "t8_math.html"} <= files and "decompose t8:" in log
        health = "saglik  hareket" in log and "yeni |dE|" in log
        go("B", 4)
        resumed = go("B", 8, resume=f"{root}/B/t4.pt")
        diff = max(float((full["weights"][k].float() - resumed["weights"][k].float()).abs().max()) for k in full["weights"])
        zero = go("Z", 4, embed=False, readout=False, chain_sim=False, distance=False, attn_after=(0,))
        base_ok = "YOK (zemin)" in open(f"{root}/Z/gunluk.txt", encoding="utf-8").read() and zero["config"]["embed"] is False
        try:
            go("B", 12, resume=f"{root}/B/t8.pt", rank=None)
            refused = False
        except ValueError:
            refused = True
    finally:
        shutil.rmtree(root, ignore_errors=True)
    check("egitim: tam yedekte decompose (matematik istemleri) + gunlukte satir", written, "")
    check("egitim: olcut saglik vermese de saglik satiri (yeni parcalarin boyu)", health)
    check("egitim: surdurme (4 + 4) == kesintisiz 8 adim", diff == 0.0, "agirlik farki %.1e" % diff)
    check("egitim: zemin ayariyla kosu; config farkliysa surdurme BASLAMAZ", base_ok and refused)


if __name__ == "__main__":
    print("tests (model_19)")
    for f in (t_model, t_decompose, t_diagnose, t_train):
        f()
    print("\n%d GECTI   %d KALDI" % (sum(RESULTS), len(RESULTS) - sum(RESULTS)))
    sys.exit(0 if all(RESULTS) else 1)
