# -*- coding: utf-8 -*-
"""tests_math -- toplama egitiminin kapilari: veri (data_math; model_15'in izi), maske, batch'ler, surdurme, olcut
(exam_math; model_15'in sor'uyla ayni), Colab calistirici (colab_math) ve defter.  CPU; Drive ve ag yok.

    python tests_math.py
"""
import ast
import copy
import json
import os
import re
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))          # model_20: model ve genel egitim
torch.set_num_threads(1)

import colab_math as C  # noqa: E402
import data_math as DM  # noqa: E402
import exam_math as EM  # noqa: E402
import train_20 as TR  # noqa: E402
from model_20 import BlockModel  # noqa: E402

RESULTS = []
TINY = dict(d=16, units=16)
TRAIN = [(i * 37 % 500, i * 91 % 500) for i in range(1, 19)] + [(5, 7, 9), (120, 33, 400), (1, 2, 3), (250, 250, 250), (0, 0, 7)]
HELDOUT = [(i * 53 % 500, i * 29 % 500) for i in range(1, 9)] + [(10, 20, 30), (499, 499, 499)]


def check(name, ok, note=""):
    RESULTS.append(bool(ok))
    print("  %-104s %s  %s" % (name, "GECTI" if ok else "KALDI", note))


def tiny_data():
    width = max(len(DM.question(t)) + len(DM.digits(sum(t))) + 2 for t in TRAIN + HELDOUT)
    return dict(name="fixture", fingerprint="fixture", vocab=DM.VOCAB, train=TRAIN, heldout=HELDOUT, width=width,
                windows=DM.make_windows(TRAIN, width))


def random_model(seed=5):
    m = BlockModel(DM.N, **TINY)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():                        # egitilmemis model birim donusum gibi davranir; test bos gecmesin
        for p in m.parameters():
            if p.requires_grad:
                p.copy_(0.5 * torch.randn(p.shape, generator=g))
    return m


class Scripted(torch.nn.Module):
    """Sahte model: her soruya script(terimler) rakamlarini yazar, bitince <eos>."""

    def __init__(self, script):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.script = script

    def logits(self, ids):
        out = torch.zeros(ids.shape[0], ids.shape[1], DM.N)
        for r, row in enumerate(ids.tolist()):
            eq = row.index(DM.EQUALS)
            terms, cur = [], []
            for t in row[1:eq] + [DM.PLUS]:
                if t == DM.PLUS:
                    terms.append(int("".join(map(str, cur))))
                    cur = []
                else:
                    cur.append(t)
            said, k = self.script(tuple(terms)), len(row) - eq - 1
            out[r, -1, said[k] if k < len(said) else DM.EOS] = 1.0
        return out


def reference_sor(m, questions, en=20000):
    """Bagimsiz referans: model_15'in veri_cok.sor hesabi (vektorel), yalniz model cagrisi Model X'e uyarlandi ve istem
    <eos> ile basliyor (data_mat_19 gibi; model_15'in egitim biciminde bastaki <eos> yoktu)."""
    if en < len(questions):
        g = torch.Generator().manual_seed(12345)
        questions = [questions[j] for j in torch.randperm(len(questions), generator=g)[:en].tolist()]
    buckets = {}
    for ts in questions:
        q = DM.question(ts)
        buckets.setdefault(len(q), []).append((q, DM.digits(sum(ts))))
    K = max(len(DM.digits(sum(ts))) for ts in questions) + 1
    right = length = count = 0
    with torch.no_grad():
        for items in buckets.values():
            path = torch.tensor([[DM.EOS] + q for q, _ in items])
            out = []
            for _ in range(K):
                t = m.logits(path)[:, -1].argmax(-1)
                out.append(t)
                path = torch.cat([path, t[:, None]], 1)
            Cm = torch.stack(out, 1)
            U = torch.tensor([len(c) for _, c in items])
            T = torch.full((len(items), K), -1, dtype=torch.long)
            for r, (_, c) in enumerate(items):
                T[r, :len(c)] = torch.tensor(c)
            has = Cm == DM.EOS
            first = torch.where(has.any(1), has.float().argmax(1), torch.full_like(U, K))
            inside = torch.arange(K)[None] < U[:, None]
            right += int((((Cm == T) | ~inside).all(1) & (first == U)).sum())
            length += int((first == U).sum())
            count += len(items)
    return right / count, length / count


def t_data(tmp):
    check("sozluk 13: rakamlar 0-9 kendi id'leri, + 10, = 11, <eos> 12; question: '472 + 182 =' rakam rakam",
          DM.VOCAB == [str(i) for i in range(10)] + ["+", "=", "<eos>"] and DM.N == 13
          and DM.question((472, 182)) == [4, 7, 2, 10, 1, 8, 2, 11] and DM.digits(0) == [0]
          and DM.question((5, 0, 12)) == [5, 10, 0, 10, 1, 2, 11])

    dur, cok = DM.build("dur"), DM.build("cok")
    count = lambda qs, n: sum(len(t) == n for t in qs)
    check("build: koddan uretilen dur ve cok model_15'in kayitli iziyle AYNI (eb4c73ab05178e18, 8d0f89938c67e0e7); sayilar "
          "kayit.txt ile ayni; genislik 13 / 18; audit (ayrik, cok'un ikili yarisi = dur)",
          (len(dur["train"]), len(dur["heldout"])) == (126776, 124224) and (len(cok["train"]), len(cok["heldout"])) == (252365, 249635)
          and (count(cok["train"], 3), count(cok["heldout"], 3)) == (125589, 125411) and (dur["width"], cok["width"]) == (13, 18)
          and DM.audit(dur)["heldout_2term"] == 124224 and DM.audit(cok)["train_3term"] == 125589
          and tuple(dur["windows"][0].shape) == (126776, 13), "%s %s" % (dur["fingerprint"], cok["fingerprint"]))

    old = DM.MAX_TERM
    try:
        DM.MAX_TERM = 20
        DM.build("dur")
        refused = False
    except ValueError:
        refused = True
    finally:
        DM.MAX_TERM = old
    check("build: kod degisirse (MAX_TERM 20) iz tutmaz, DURUR", refused)

    path = os.path.join(tmp, "veri_fixture.pt")
    d = dict(cift_eg=TRAIN, cift_tu=HELDOUT, eg=DM.examples(TRAIN), tu=DM.examples(HELDOUT))
    DM.FINGERPRINTS["fixture"] = DM.fingerprint(d["eg"], d["tu"])
    torch.save(d, path)
    got = DM.load(path, "fixture")
    via = DM.build("fixture", source="drive", path=path)
    d2 = dict(soru_eg=TRAIN, soru_tu=HELDOUT, eg=DM.examples(TRAIN), tu=DM.examples(HELDOUT))
    torch.save(d2, path + "2")
    got2 = DM.load(path + "2", "fixture")
    d["eg"][0][0][0, 0] += 1                     # tek token
    torch.save(d, path)
    try:
        DM.load(path, "fixture")
        refused = False
    except ValueError:
        refused = True
    check("load (Drive yedegi): model_15 dosya bicimi (cift_* ve soru_*) okunur, build(source='drive') ayni; dosyada tek token "
          "degisirse iz tutmaz, DURUR", got == (TRAIN, HELDOUT) and got2 == got and via["train"] == TRAIN and refused)
    return dur


def t_windows(tmp):
    W, M, H = DM.make_windows([(472, 182), (1, 1), (5, 7, 9)], 18)
    E = DM.EOS
    check("make_windows: <eos> soru cevap <eos>, sagdan <eos> dolgu; M gercek token'lar; H cevap rakamlari + son <eos>",
          W[0].tolist() == [E, 4, 7, 2, 10, 1, 8, 2, 11, 6, 5, 4, E] + [E] * 5 and M[0].sum() == 13 and M[0, 12] and not M[0, 13]
          and H[0].nonzero().flatten().tolist() == [9, 10, 11, 12] and W[1, :6].tolist() == [E, 1, 10, 1, 11, 2]
          and H[1].nonzero().flatten().tolist() == [5, 6] and H[2].nonzero().flatten().tolist() == [7, 8, 9])

    m = random_model()
    data = tiny_data()
    W, M, H = data["windows"]
    with torch.no_grad():
        nll_h = m.loss(W, H)[1]
        logits = m.logits(W[:, :-1])
        manual = torch.nn.functional.cross_entropy(logits[H[:, 1:]], W[:, 1:][H[:, 1:]])
        W2 = W.clone()
        W2[~M] = torch.randint(0, 10, (int((~M).sum()),), generator=torch.Generator().manual_seed(1))
        nll_pad = m.loss(W2, H)[1]
        nll_m = m.loss(W, M)[1]
        ce = EM.answer_ce(m, data["windows"], "cpu")
    check("maske: model.loss(W, H) = cevap konumlarinda elle CE; dolgu degisse de ayni (nedensel attention, mask yalniz hedef "
          "secer); mask M soru token'larini da katar; answer_ce ayni sayi",
          torch.allclose(nll_h, manual, atol=1e-6) and torch.allclose(nll_h, nll_pad, atol=1e-6) and not torch.allclose(nll_h, nll_m)
          and abs(ce - float(nll_h)) < 1e-5, "H %.4f  dolgu %.4f  M %.4f  ce %.4f" % (nll_h, nll_pad, nll_m, ce))

    B = 5
    per = len(TRAIN) // B
    ref = DM.batches(data, B, seed=3)
    ref = {s: ref(s) for s in range(3 * per)}
    f = DM.batches(data, B, seed=3)
    same = all(torch.equal(f(s)[0], ref[s][0]) and torch.equal(f(s)[1], ref[s][1]) for s in (2 * per + 1, 0, per, 1, 2 * per + 1))
    rows0 = torch.cat([ref[s][0] for s in range(per)])
    distinct = len({tuple(r) for r in rows0.tolist()}) == per * B
    full = DM.batches(data, B, seed=3, answer_only=False)
    Wd, Md, Hd = data["windows"]
    idx = [next(i for i in range(len(Wd)) if torch.equal(Wd[i], row)) for row in ref[0][0]]
    check("batches: adimin fonksiyonu (sirasiz = sirali cagri); sekil hep (B, genislik); epok icinde her soru en cok bir kez; "
          "epok sirayi degistirir; answer_only mask = H, degilse M",
          same and distinct and not torch.equal(ref[per][0], ref[0][0]) and all(tuple(ref[s][0].shape) == (B, data["width"]) for s in ref)
          and torch.equal(ref[0][1], Hd[idx]) and torch.equal(full(0)[1], Md[idx]), "%d soru, B %d -> %d adim/epok" % (len(TRAIN), B, per))

    for answer_only in (True, False):
        packs = {}

        def keep(step, model, opt):
            packs[step] = dict(step=step, model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(opt.state_dict()))
        whole, curve = TR.train_seq("shared", None, None, DM.N, steps=10, log_at=(0, 10), save_every=3, save=keep, model_kw=TINY,
                                    batches=DM.batches(data, B, 1, answer_only))
        resumed, _ = TR.train_seq("shared", None, None, DM.N, steps=10, log_at=(), checkpoint=packs[6], model_kw=TINY,
                                  batches=DM.batches(data, B, 1, answer_only))
        bit = all(torch.equal(a, b) for a, b in zip(whole.state_dict().values(), resumed.state_dict().values()))
        check("surdurme, batches: yedek araligi (3) epoku (4 adim) bolmuyor; 6. adim paketinden surdurulen = kesintisiz 10 adim, "
              "bit duzeyinde (answer_only %s)" % answer_only, bit and sorted(packs) == [3, 6, 9],
              "nll %.3f -> %.3f" % (curve[0]["nll"], curve[-1]["nll"]))


def t_exam(dur):
    scripts = {(2, 3): [5], (47, 25): [7, 2], (500, 499): [9, 9, 9], (0, 0): [0], (472, 182): [6, 5, 4, 4],
               (99, 1): [1, 0], (15, 17): [2, 3], (123, 456): [5, 7, 9, 9, 9, 9, 9, 9]}
    fake = Scripted(lambda t: scripts[t])
    r = EM.ask(fake, list(scripts), "cpu", limit=100)
    check("ask, elle kurulmus 8 soru: dogru 4 (2+3, 47+25, 500+499, 0+0); fazla rakam, erken durma, hic durmama, yanlis rakam "
          "YANLIS -> accuracy 4/8, length_ok 5/8, first_digit 7/8",
          r == {"accuracy": 0.5, "length_ok": 0.625, "first_digit": 0.875, "n": 8}, str(r))

    def mixed(terms):
        s = sum(terms)
        c = DM.digits(s)
        if s % 5 == 0:
            return [(c[0] + 1) % 10] + c[1:]         # ilk rakam yanlis
        if s % 3 == 0:
            return c[:-1] if len(c) > 1 else c + [3]  # erken durur / fazla
        if s % 7 == 0:
            return c + [c[-1]]                        # fazla rakam
        return c
    qs = DM.pairs()[::997] + [(i * 7 % 500, i * 13 % 500, i * 29 % 500) for i in range(60)]
    fake = Scripted(mixed)
    mine, sample = EM.ask(fake, qs, "cpu", limit=10**9), EM.ask(fake, qs, "cpu", limit=100)
    ref, ref_sample = reference_sor(fake, qs), reference_sor(fake, qs, en=100)
    check("ask = model_15'in sor hesabi (bagimsiz vektorel referans), 2 ve 3 terim karisik: butun kume ve tohum 12345'lik "
          "100'luk ornek ayni", (mine["accuracy"], mine["length_ok"]) == ref and (sample["accuracy"], sample["length_ok"]) == ref_sample
          and 0 < mine["accuracy"] < 1 and sample["n"] == 100, "acc %.4f len %.4f / ornek %.2f" % (mine["accuracy"], mine["length_ok"],
                                                                                                  sample["accuracy"]))
    m = random_model()
    parts = EM.breakdown(m, qs, "cpu", "terms")
    whole = EM.ask(m, qs, "cpu", limit=10**9)
    total = C._total(parts)
    by_digits = C._total(EM.breakdown(fake, qs, "cpu", "digits"))
    check("breakdown: terim (2/3) ve hane kirilimlarinin n agirlikli toplami = butun kume", sorted(parts) == [2, 3]
          and all(abs(total[k] - whole[k]) < 1e-12 for k in ("accuracy", "length_ok", "first_digit"))
          and abs(by_digits["accuracy"] - mine["accuracy"]) < 1e-12 and total["n"] == len(qs))
    saved = TR.generate.__defaults__
    TR.generate.__defaults__ = (False,)                     # eski yol: her token'da butun dizi yeniden
    try:
        old_ask, old_rows = EM.ask(m, qs, "cpu", limit=10**9), EM.show(m, qs[:40])
    finally:
        TR.generate.__defaults__ = saved
    check("ask, show: onbellekli uretim (AttentionCache) = eski tam yeniden hesap (rastgele Model X, 2 ve 3 terim)",
          old_ask == whole and old_rows == EM.show(m, qs[:40]), str(whole))
    rows = EM.show(Scripted(lambda t: scripts[t]), [(2, 3), (15, 17), (123, 456)])
    check("show (kural 12): (soru, dogrusu, modelin yazdigi, dogru mu); hic durmayan isaretli",
          rows == [("2 + 3", "5", "5", True), ("15 + 17", "32", "23", False), ("123 + 456", "579", "5799 (durmadi)", False)], str(rows))


def t_colab(tmp):
    data = tiny_data()
    per = len(TRAIN) // 4
    out = os.path.join(tmp, "runs", "r")
    kw = dict(every=3, device="cpu", save_every=2, batch_size=4, exam_limit=50, model_kw=TINY)
    run = C.start("TEST_R", data, out, steps=7, **kw)
    run["thread"].join(600)
    files = sorted(os.listdir(out))
    cfg = json.load(open(os.path.join(out, "config.json")))
    fin = json.load(open(os.path.join(out, "final.json"), encoding="utf-8")) if "final.json" in files else {}
    ex = json.load(open(os.path.join(out, "exams.json"))) if "exams.json" in files else []
    check("colab_math: CPU'da start -> config, log, exams, final, model, checkpoint_t000002/4/6; config: veri, iz, batch, "
          "epok, answer_only, model boyu; final: saglik, butun kume, terim/hane kirilimi, tani sorulari, satirlar",
          run["done"] and not run["error"] and files == ["checkpoint_t000002.pt", "checkpoint_t000004.pt", "checkpoint_t000006.pt",
                                                         "config.json", "exams.json", "final.json", "log.txt", "model.pt"]
          and cfg["data"] == "fixture" and cfg["steps_per_epoch"] == per and cfg["answer_only"] is True and cfg["batch_size"] == 4
          and cfg["model_kw"] == dict(d=16, turns=4, layers=2, heads=4, output_skip=False, units=16, t_max=512) and cfg["setting"] == "shared" and cfg["rope"] is True
          and set(fin) == {"health", "breakdown", "train", "heldout", "panel", "rows"} and fin["heldout"]["n"] == len(HELDOUT)
          and set(fin["breakdown"]["heldout"]["terms"]) == {"2", "3"} and len(fin["panel"]) == len(EM.PANEL)
          and len(fin["rows"]) == len(HELDOUT) and [e["step"] for e in ex] == [0, 3, 6] and "heldout_terms" in ex[0],
          str(run["error"] or files))

    os.remove(os.path.join(out, "checkpoint_t000006.pt"))           # 4. adimdan sonra kesilmis gibi (epok 5 adim)
    whole = torch.load(os.path.join(out, "model.pt"))
    run2 = C.start("TEST_R", data, out, steps=7, resume=True, **kw)
    run2["thread"].join(600)
    log = open(os.path.join(out, "log.txt"), encoding="utf-8").read()
    after = torch.load(os.path.join(out, "model.pt"))
    ex2 = json.load(open(os.path.join(out, "exams.json")))
    try:
        C.start("TEST_R2", data, out, steps=9, resume=True, **kw)
        refused = False
    except RuntimeError:
        refused = True
    check("colab_math surdurme: yedek araligi (2) epoku (5) bolmuyor; 4. adim paketinden surdurulen = kesintisiz, bit duzeyinde; "
          "paketten sonraki sinavlar atilip yeniden yazilir; ayar farkliysa reddeder",
          run2["done"] and not run2["error"] and "SURDURULDU adim 4'den" in log and refused
          and all(torch.equal(whole[k], after[k]) for k in whole) and [e["step"] for e in ex2] == [0, 3, 6] and per == 5,
          str(run2["error"]))

    out_t = os.path.join(tmp, "runs", "t")
    run = C.start("TEST_T", data, out_t, steps=2, setting="transformer_novalue",
                  **dict(kw, model_kw=dict(d=16, layers=1, units=16)))
    run["thread"].join(600)
    cfg = json.load(open(os.path.join(out_t, "config.json")))
    check("colab_math: setting='transformer_novalue' (kiyas transformer'i) ayni veri, batch ve sinavla kosar, dosyalarini yazar",
          run["done"] and not run["error"] and cfg["setting"] == "transformer_novalue"
          and cfg["model_kw"] == dict(d=16, layers=1, heads=1, units=16) and {"final.json", "model.pt", "checkpoint_t000002.pt"}
          <= set(os.listdir(out_t)), str(run["error"] or cfg))
    C.pulse(1)
    C.stop()


def t_notebook():
    nb = json.load(open(os.path.join(HERE, "colab_math.ipynb"), encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    head = re.compile(r"^# \S+ .+\|  (CPU|GPU)  \|  tekrar: ")
    parsed = []
    for src in cells:
        try:
            ast.parse("\n".join(l for l in src.split("\n") if not l.lstrip().startswith("!")))
            parsed.append(True)
        except SyntaxError:
            parsed.append(False)
    gpu = [s for s in cells if head.match(s) and head.match(s).group(1) == "GPU"]
    runs = [(re.search(r"^DATA_NAME, D = '(\w+)', (\d+)", s, re.M) or [None] * 3) for s in gpu]
    check("defter: her kod hucresi kunyeyle baslar ve gecerli Python; her GPU hucresinde GPU kapisi C.start'tan once; YALNIZ "
          "cok x D 64 ve 96 (HAZIRLIK yalniz cok uretir, kosu hucresi cok disini reddeder); BATCH_SIZE 256, EPOCHS 100, STEPS "
          "epoktan; sinav ve yedek 5.000; veri koddan; kod train_math'ten",
          all(head.match(s) for s in cells) and all(parsed) and len(gpu) == 2
          and re.search(r"^DATASETS = \('cok',\)", cells[0], re.M) and "'dur'" not in "".join(cells)
          and all("assert DATA_NAME == 'cok'" in s for s in gpu)
          and all(s.index("torch.cuda.is_available()") < s.index("C.start(") and "mem_get_info" in s for s in gpu)
          and sorted((r[1], r[2]) for r in runs) == [("cok", "64"), ("cok", "96")]
          and all(re.search(p, s, re.M) for s in gpu for p in (r"^BATCH_SIZE = 256", r"^EPOCHS = 100", r"^STEPS = EPOCHS \*",
                                                                  r"^EVERY = 5000", r"^SAVE_EVERY = 5000", r"^MODEL_KW = dict\(d=D, units=4 \* D"))
          and re.search(r"^SOURCE = 'code'", cells[0], re.M) and "/train_math" in cells[0] and "kinship" not in "".join(cells),
          "%d kod hucresi" % len(cells))


if __name__ == "__main__":
    print("tests (train_math)")
    tmp = tempfile.mkdtemp()
    dur = t_data(tmp)
    t_windows(tmp)
    t_exam(dur)
    t_colab(tmp)
    t_notebook()
    print("\n%d GECTI   %d KALDI" % (sum(RESULTS), len(RESULTS) - sum(RESULTS)))
    sys.exit(0 if all(RESULTS) else 1)
