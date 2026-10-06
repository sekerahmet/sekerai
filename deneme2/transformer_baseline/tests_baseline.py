"""tests_baseline -- transformer tabaninin kapilari (CPU, kucuk).  Adillik sozlesmesi (belge/model_z_temel/17):
    T1  hedefler Model Z'nin examples() hedefleriyle hikaye hikaye birebir ayni (sira + tur), uc baglamda
    T2  hikaye basina hedef W + S + 1
    T3  causal ve dolgu: logit dolguya ve gelecege bagli degil; 64'luk kova kaybi degistirmez; shuffled tek hikayede = full
    T4  Block Model Z'ninkiyle ayni cikti
    T5  parametre sayisi (d 256, L 4, H 4, SS sozlugu) 17.938.688
    T6  kesilip surdurulen = kesintisiz (agirlik farki 0)
    T7  exam_scores: kayip = forward kaybi, END / EOS / tur sayimlari elle sayilanla, bpb formulu
    T8  uretim: EOS yalniz cumle basinda, cumle ortasinda EOS -> END; batch'te satirlar bagimsiz; belirlenimli
    T9  Drive varsa: sinav alt kumesi (model_y sizintisiz kumesi, sha256), Stories.batch Model Z ile ayni, bayt paydasi =
        model_y valid_bytes (K3), sinav hedef sayisi 280.849 (iki yol), egitim hedef sayisi (ofsetlerden)
    T10 ana kod klasor disindan import etmez
Model Z yalniz burada okunur (kullanici onayi, 6 Ekim); bulunamazsa ATLANDI yazilir.

    python tests_baseline.py
"""
import ast
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile

import torch  # noqa: I001  (Windows: torch numpy'den once)
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MODEL_Z_SENTENCE = os.path.join(os.path.dirname(HERE), "model_z", "core", "sentence")
ROOT = "G:/Drive'ım/model_z/simplestories_full"
MODEL_Y_BYTES = "G:/Drive'ım/simplestories/gpt2/valid_bytes.npy"
EXAM_SET = "G:/Drive'ım/simplestories/exam_stories.npy"
EXAM_SET_SHA256 = "ee0e13faebbf55b21bf9a5fc9d5159d7c33490b08b3c1ee89c3939072e4bf090"
torch.set_num_threads(4)

import baseline as BL  # noqa: E402
import generate_baseline as GB  # noqa: E402
import train_baseline as TB  # noqa: E402

RESULTS = []


def check(name, ok, note=""):
    RESULTS.append(bool(ok))
    print("  %-70s %s  %s" % (name, "GECTI" if ok else "KALDI", note), flush=True)


def skip(name, why):
    print("  %-70s ATLANDI  %s" % (name, why), flush=True)


def model_z():
    """Model Z'nin train_sentence ve sentence modulleri (yalniz test); yoksa None."""
    if not os.path.isdir(MODEL_Z_SENTENCE):
        return None
    if MODEL_Z_SENTENCE not in sys.path:
        sys.path.append(MODEL_Z_SENTENCE)
    import sentence
    import train_sentence
    return train_sentence, sentence


def synthetic(n=40, vocab=30, seed=0):
    """Rastgele hikayeler: 1 kelimelik cumle, tek cumlelik hikaye ve <unk> (0) dahil."""
    rng = np.random.default_rng(seed)
    out = [[[int(rng.integers(0, vocab)) for _ in range(rng.integers(1, 8))] for _ in range(rng.integers(1, 7))]
           for _ in range(n)]
    out[0] = [[5]]
    out[1] = [[0, 3], [7], [0]]
    return out


def labels(ex):
    """Hedefi olan konumlar, sirayla: (hedef, tur) listesi, hikaye basina."""
    kind = ex["first"] * 0 + ex["mid"] * 1 + ex["end"] * 2 + ex["eos"] * 3
    out = {}
    for r in range(len(ex["target"])):
        keep = ex["target"][r] >= 0
        out.setdefault(int(ex["story"][r]), []).extend(zip(ex["target"][r][keep].tolist(), kind[r][keep].tolist()))
    return out


def t1_targets(mz):
    st = TB.Stories.from_lists(synthetic(), "cpu")
    ids, mask = st.batch(torch.arange(st.n))
    END, EOS = 30, 31
    if mz is None:
        skip("T1 hedefler Model Z examples() ile ayni", "Model Z yok: " + MODEL_Z_SENTENCE)
        return
    T, _ = mz
    zex = T.examples(ids, mask, torch.zeros(ids.shape[0], ids.shape[1], 1), END, EOS)
    want = labels(zex)
    for c in TB.CONTEXTS:
        got = labels(TB.story_sequences(ids, mask, END, EOS, c))
        check("T1 hedefler Model Z examples() ile ayni (sentetik, %s)" % c, got == want,
              "%d hikaye, %d hedef" % (len(want), sum(map(len, want.values()))))


def t2_counts():
    stories = synthetic()
    st = TB.Stories.from_lists(stories, "cpu")
    ex = TB.story_sequences(*st.batch(torch.arange(st.n)), 30, 31)
    per = (ex["target"] >= 0).sum(1).tolist()
    want = [sum(map(len, s)) + len(s) + 1 for s in stories]
    check("T2 hikaye basina hedef = W + S + 1", per == want, "toplam %d" % sum(per))
    tok = ex["tok"][1]
    check("T2 dizi bicimi [BOS] w.. END w.. END", tok[:8].tolist() == [0, 0, 3, 30, 7, 30, 0, 30],
          str(tok[:8].tolist()))


def tiny(vocab=30, d=16, layers=2, heads=2, seed=0):
    torch.manual_seed(seed)
    return BL.BaselineTransformer([str(i) for i in range(vocab)], d, layers, heads).eval()


@torch.no_grad()
def t3_causal():
    model = tiny()
    st = TB.Stories.from_lists(synthetic(), "cpu")
    ids, mask = st.batch(torch.arange(st.n))
    ex = TB.story_sequences(ids, mask, model.END, model.EOS)
    h = model.hidden(ex["tok"])
    diff = 0.0
    for r in (0, 1, 5):
        e1 = TB.story_sequences(*st.batch(torch.tensor([r])), model.END, model.EOS)
        L = e1["tok"].shape[1]
        diff = max(diff, float((model.hidden(e1["tok"])[0] - h[r, :L]).abs().max()))
    check("T3 dolgu: tek basina = batch icinde (logit)", diff < 1e-5, "fark %.1e" % diff)
    widths, same = [], True
    for c in TB.CONTEXTS:
        e = TB.story_sequences(ids, mask, model.END, model.EOS, c)
        real = int((e["target"] >= 0).nonzero()[:, 1].max()) + 1
        a = float(model(e["tok"], e["target"]))
        b = float(model(e["tok"][:, :real], e["target"][:, :real]))
        widths.append((e["tok"].shape[1], real))
        same &= abs(a - b) < 1e-6 and e["tok"].shape[1] % 64 == 0 and e["tok"].shape[1] >= real
    check("T3 64'luk kova dolgusu kaybi degistirmez (uc baglam)", same, "boy / gercek %s" % widths)
    tok = ex["tok"][3:4].clone()
    a = model.hidden(tok)
    tok[0, 6:] = 9
    b = model.hidden(tok)
    d0 = float((a[0, :6] - b[0, :6]).abs().max())
    check("T3 causal: gelecek degisince gecmis logit ayni", d0 == 0.0, "fark %.1e" % d0)
    for r in (1, 4):
        ids1, mask1 = st.batch(torch.tensor([r]))
        full = TB.story_sequences(ids1, mask1, model.END, model.EOS)
        shuf = TB.story_sequences(ids1, mask1, model.END, model.EOS, "shuffled")
        hf, hs = model.hidden(full["tok"]), model.hidden(shuf["tok"])
        lf = hf[full["target"] >= 0] @ model.E.weight.T
        ls = hs[shuf["target"] >= 0] @ model.E.weight.T
        diff = float((lf - ls).abs().max())
        check("T3 shuffled tek hikayede (bagisci kendisi) = full, hikaye %d" % r, diff < 1e-5, "fark %.1e" % diff)


@torch.no_grad()
def t4_block(mz):
    if mz is None:
        skip("T4 Block Model Z ile ayni", "Model Z yok")
        return
    _, S = mz
    torch.manual_seed(1)
    ours, theirs = BL.Block(64, 4, 176), S.Block(64, 4, 176)
    for p in ours.parameters():
        torch.nn.init.normal_(p, std=0.1)
    theirs.load_state_dict(ours.state_dict())
    x = torch.randn(3, 20, 64)
    pos = torch.arange(20).expand(3, -1)
    d = float((ours(x, pos, None) - theirs(x, pos, None)).abs().max())
    check("T4 Block ciktisi Model Z'ninkiyle ayni", d == 0.0, "fark %.1e" % d)
    ok = all(math.isclose(a, b) for a, b in [(BL.rope(x.view(3, 4, 5, 64), pos[:, :5]).sum().item(),
                                               S.rope(x.view(3, 4, 5, 64), pos[:, :5]).sum().item())])
    check("T4 rope Model Z'ninkiyle ayni", ok)


def t5_params():
    m = BL.BaselineTransformer(["w"] * 57707, 256, 4, 4)
    n = sum(p.numel() for p in m.parameters())
    check("T5 parametre (d 256, L 4, H 4, sozluk 57.707) = 17.938.688", n == 17938688, "%d" % n)


def write_root(top, train, exam, vocab):
    """Sentetik Drive: <top>/model_z/simplestories_full (make_ss_sentences --stories bicimi + bayt paydasi) ve
    <top>/simplestories/exam_stories.npy + .json (her 7. hikaye 'sizan' sayilip disarida).  -> (root, sinav kumesi)"""
    folder = os.path.join(top, "model_z", "simplestories_full")
    os.makedirs(folder)
    os.makedirs(os.path.join(top, "simplestories"))
    exam_set = np.array([i for i in range(len(exam)) if i % 7], dtype=np.int64)
    np.save(os.path.join(top, "simplestories", "exam_stories.npy"), exam_set)
    json.dump(dict(sha256=hashlib.sha256(np.ascontiguousarray(exam_set)).hexdigest(), stories=len(exam_set)),
              open(os.path.join(top, "simplestories", "exam_stories.json"), "w"))
    json.dump(vocab, open(os.path.join(folder, "ss_vocab.json"), "w"))
    for prefix, stories in (("ss_story", train), ("ss_exam_story", exam)):
        sents = [s for st in stories for s in st]
        np.save(os.path.join(folder, prefix + "_ids.npy"), np.array([w for s in sents for w in s], dtype=np.int32))
        np.save(os.path.join(folder, prefix + "_sentence_offsets.npy"), np.r_[0, np.cumsum([len(s) for s in sents])])
        np.save(os.path.join(folder, prefix + "_offsets.npy"), np.r_[0, np.cumsum([len(st) for st in stories])])
    np.save(os.path.join(folder, "ss_exam_story_bytes.npy"), np.array([5 * sum(map(len, s)) for s in exam]))
    return folder, exam_set


def t6_resume():
    tmp = tempfile.mkdtemp()
    try:
        exam_n = max(json.load(open(os.path.join(HERE, "ss_prompts.json")))["stories"]) + 1
        rng = np.random.default_rng(3)
        exam = [[[int(rng.integers(0, 30)) for _ in range(3)] for _ in range(4)] for _ in range(exam_n)]
        root, exam_set = write_root(tmp, synthetic(70, seed=1), exam, [str(i) for i in range(30)])
        base = ["--root", root, "--d", "16", "--layers", "1", "--heads", "2", "--batch", "8", "--epochs", "2"]
        out_a, out_b = os.path.join(tmp, "a"), os.path.join(tmp, "b")
        real, saved, real_gen = TB.story_sequences, TB.CHECKPOINT_SECS, GB.generate_story
        GB.generate_story = lambda m, p, max_words=20, decode="greedy": real_gen(m, p, max_words, decode)   # kisa
        calls = [0]

        def stop_after(*a, **k):
            calls[0] += 1
            if calls[0] == 6:                                      # 6. adimda kesilir (epok 1 = 9 adim)
                raise KeyboardInterrupt
            return real(*a, **k)
        TB.main(base + ["--out", out_a])
        try:
            TB.CHECKPOINT_SECS = -1
            TB.story_sequences = stop_after
            try:
                TB.main(base + ["--out", out_b])
            except KeyboardInterrupt:
                pass
        finally:
            TB.story_sequences, TB.CHECKPOINT_SECS = real, saved
        pack = torch.load(os.path.join(out_b, "checkpoint.pt"), weights_only=False)
        TB.main(base + ["--out", out_b, "--resume", "1"])
        a = torch.load(os.path.join(out_a, "agent.pt"), weights_only=False)["state"]
        b = torch.load(os.path.join(out_b, "agent.pt"), weights_only=False)["state"]
        d = max(float((a[k] - b[k]).abs().max()) for k in a)
        check("T6 kesilip surdurulen = kesintisiz (epok %d hikaye %d'den)" % (pack["epoch"], pack["next"]), d == 0.0,
              "agirlik farki %.1e" % d)
        r = json.load(open(os.path.join(out_a, "results.json")))
        cfg = json.load(open(os.path.join(out_a, "config.json")))
        want = np.sort(exam_set[np.random.default_rng(0).permutation(len(exam_set))[:TB.EXAM_STORIES]]).tolist()
        check("T6 sinav: sizintisiz kumeden secim, yol ve sha256 kayitli", cfg["exam_rows"] == want
              and r["args"]["exam_set_sha256"] == hashlib.sha256(np.ascontiguousarray(exam_set)).hexdigest()
              and r["args"]["exam_set"].endswith("exam_stories.npy"), "%d hikaye" % len(want))
        check("T6 results.json, samples.txt, config.json yazildi; generation: greedy + sample sentence_repeat",
              len(r["history"]) == 2 and os.path.exists(os.path.join(out_a, "samples.txt"))
              and all("sentence_repeat" in r["generation"][d] for d in ("greedy", "sample"))
              and os.path.exists(os.path.join(out_a, "config.json")))
        try:
            TB.main(base + ["--out", os.path.join(tmp, "yok"), "--resume", "1"])
            check("T6 surdurme paketi yoksa durur", False)
        except AssertionError as e:
            check("T6 surdurme paketi yoksa durur", "surdurme paketi yok" in str(e))
    finally:
        GB.generate_story = real_gen
        shutil.rmtree(tmp, ignore_errors=True)


@torch.no_grad()
def t7_exam():
    model = tiny()
    st = TB.Stories.from_lists(synthetic(), "cpu")
    rows = torch.arange(st.n)
    nbytes = 1234
    e = TB.exam_scores(model, st, rows, 1000, "full", nbytes)
    ex = TB.story_sequences(*st.batch(rows), model.END, model.EOS)
    loss = float(model(ex["tok"], ex["target"]))
    check("T7 exam kaybi = forward kaybi", abs(e["loss"] - loss) < 1e-4, "%.4f / %.4f" % (e["loss"], loss))
    keep = ex["target"] >= 0
    pred = (model.hidden(ex["tok"]) @ model.E.weight.T).argmax(-1)
    tg = ex["target"]
    hand = dict(end_ok=float((pred[ex["end"]] == model.END).float().mean()),
                eos_ok=float((pred[ex["eos"]] == model.EOS).float().mean()),
                acc=float((pred[keep] == tg[keep]).float().mean()),
                false_eos=float((pred[ex["first"]] == model.EOS).float().mean()))
    ok = all(abs(e[k] - v) < 1e-4 for k, v in hand.items())
    check("T7 END / EOS / acc / yanlis EOS elle sayilanla ayni", ok, str({k: e[k] for k in hand}))
    bt = e["by_type"]
    n = sum(v["n"] for v in bt.values())
    total = sum(v["loss"] * v["n"] for v in bt.values())
    bpb = (total - bt["eos"]["loss"] * bt["eos"]["n"]) / math.log(2) / nbytes
    check("T7 turler hedefleri boluyor, bpb = EOS haric nll / ln2 / bayt",
          n == e["targets"] and abs(total / n - e["loss"]) < 1e-3 and abs(bpb - e["bits_per_byte"]) < 1e-3,
          "bpb %.4f / %.4f" % (bpb, e["bits_per_byte"]))
    others = [TB.exam_scores(model, st, rows, 7, c)["targets"] for c in TB.CONTEXTS]
    check("T7 uc baglamda ayni hedef sayisi (parcali batch)", len(set(others + [e["targets"]])) == 1, str(others))


class FakeModel(torch.nn.Module):
    """Uretim kurali icin: END'den sonra (3. END'e kadar) kelime 1, kelime 1'den sonra EOS (cumle ortasi -> END olmali)."""

    def __init__(self):
        super().__init__()
        self.END, self.EOS = 5, 6
        self.E = torch.nn.Embedding(7, 7)
        self.E.weight.data = torch.eye(7)

    def hidden(self, tok):
        out = torch.zeros(*tok.shape, 7)
        for b in range(tok.shape[0]):
            for p in range(tok.shape[1]):
                t, ends = int(tok[b, p]), int((tok[b, :p + 1] == self.END).sum())
                nxt = (self.EOS if ends >= 3 else 1) if (t == self.END and p > 0) else (self.EOS if t == 1 else 1)
                out[b, p, nxt] = 100                         # tek nokta dagilim (sample da ayni secer)
        return out


def t8_generate():
    res = GB.generate_story(FakeModel(), [[[2]], [[2, 3], [4]]], max_words=20)
    check("T8 cumle ortasinda EOS -> END, EOS yalniz cumle basinda", res == [([[1], [1]], [], True), ([[1]], [], True)],
          str(res))
    res = GB.generate_story(FakeModel(), [[[2]]], max_words=20, decode="sample")
    check("T8 sample: ayni kural (tek nokta dagilim)", res == [([[1], [1]], [], True)], str(res))
    model = tiny()
    with torch.no_grad():
        model.E.weight[model.END:] = 0                            # END / EOS hic secilmesin: 30 kelime uretir
    a = GB.generate_story(model, [[[1, 2, 3]], [[4], [5, 6]]], max_words=30)
    b = GB.generate_story(model, [[[1, 2, 3]], [[4], [5, 6]]], max_words=30)
    one = GB.generate_story(model, [[[4], [5, 6]]], max_words=30)
    check("T8 greedy belirlenimli; batch'te satir tek basina ile ayni", a == b and a[1] == one[0],
          "%d / %d kelime" % (len(a[0][1]), len(a[1][1])))
    sa = GB.generate_story(model, [[[1, 2, 3]], [[4], [5, 6]]], max_words=30, decode="sample")
    sb = GB.generate_story(model, [[[1, 2, 3]], [[4], [5, 6]]], max_words=30, decode="sample")
    check("T8 sample sabit tohumla tekrar uretilebilir, greedy'den farkli", sa == sb and sa != a)


@torch.no_grad()
def t9_drive(mz):
    if not os.path.isdir(ROOT):
        skip("T9 Drive", "yok: " + ROOT)
        return
    exam = TB.Stories.from_files(ROOT, "ss_exam_story", "cpu")
    exam_set = np.load(EXAM_SET)
    sha = hashlib.sha256(np.ascontiguousarray(exam_set)).hexdigest()
    check("T9 sinav kumesi model_y exam_stories.npy, sha256 sabit", sha == EXAM_SET_SHA256 and len(exam_set) == 19807,
          "%s, %d hikaye" % (sha[:12], len(exam_set)))
    pick = np.sort(exam_set[np.random.default_rng(0).permutation(len(exam_set))[:TB.EXAM_STORIES]])
    leaked = json.load(open(EXAM_SET[:-4] + ".json"))["leaked"]
    check("T9 sinav alt kumesi ilk indeksler; sizan hikaye yok", pick[:5].tolist() == [27, 54, 97, 131, 141]
          and not set(leaked) & set(pick.tolist()), "%s, sizan %s disarida" % (pick[:5].tolist(), leaked))
    mz_src = open(os.path.join(MODEL_Z_SENTENCE, "train_sentence.py"), encoding="utf-8").read() if mz else ""
    z_expr = "np.sort(allowed[np.random.default_rng(0).permutation(len(allowed))[:EXAM_STORIES]])"
    z_has = z_expr in mz_src and "EXAM_STORIES = 1000" in mz_src and '"exam_stories.npy"' in mz_src
    mz_pick = np.sort(exam_set[np.random.default_rng(0).permutation(len(exam_set))[:1000]])     # Model Z ifadesi
    check("T9 Model Z ile ayni 1.000 hikaye (%s)" % ("Model Z train_sentence'teki ifade" if z_has
                                                     else "Model Z kodunda YOK: ayni formulle hesaplandi"),
          np.array_equal(pick, mz_pick))
    vocab = json.load(open(os.path.join(ROOT, "ss_vocab.json"), encoding="utf-8"))
    END, EOS = len(vocab), len(vocab) + 1
    rows = torch.as_tensor(pick)
    count, kinds = 0, {}
    for c in range(0, len(rows), 64):
        ex = TB.story_sequences(*exam.batch(rows[c:c + 64]), END, EOS)
        count += int((ex["target"] >= 0).sum())
    check("T9 sinav hedef sayisi = 280.849", count == 280849, str(count))
    if mz is not None:
        T, _ = mz
        zst = T.Stories.from_files(ROOT, "ss_exam_story", "cpu")
        same, zcount, eq = True, 0, True
        for c in range(0, len(rows), 64):
            a, b = exam.batch(rows[c:c + 64]), zst.batch(rows[c:c + 64])
            same &= bool(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))
            zex = T.examples(b[0], b[1], torch.zeros(b[0].shape[0], b[0].shape[1], 1), END, EOS)
            zcount += int((zex["target"] >= 0).sum())
            if c < 128:
                eq &= labels(zex) == labels(TB.story_sequences(*a, END, EOS))
        check("T9 Stories.batch Model Z ile ayni (1.000 sinav hikayesi)", same)
        check("T9 Model Z examples() sinav hedef sayisi ayni", zcount == count, str(zcount))
        check("T9 gercek sinav hikayelerinde hedefler Model Z ile birebir (128 hikaye)", eq)
    else:
        skip("T9 Model Z kiyasi", "Model Z yok")
    if os.path.exists(MODEL_Y_BYTES):
        a, b = np.load(os.path.join(ROOT, "ss_exam_story_bytes.npy")), np.load(MODEL_Y_BYTES)
        check("T9 bayt paydasi = model_y valid_bytes (K3)", len(a) == len(b) == exam.n and np.array_equal(a, b),
              "%d hikaye, alt kume %d bayt" % (len(a), int(a[pick].sum())))
        check("T9 alt kume baytlari = 1.166.410", int(a[pick].sum()) == 1166410)
    else:
        skip("T9 bayt paydasi", "yok: " + MODEL_Y_BYTES)
    so = np.load(os.path.join(ROOT, "ss_story_sentence_offsets.npy"), mmap_mode="r")
    sto = np.load(os.path.join(ROOT, "ss_story_offsets.npy"))
    total = int(so[-1]) + (len(so) - 1) + (len(sto) - 1)
    check("T9 egitim hedef sayisi (W + S + hikaye, ofsetlerden) = 629.896.885", total == 629896885, str(total))


def t10_imports():
    own = {"baseline", "train_baseline", "generate_baseline"}
    allowed = {"argparse", "hashlib", "json", "math", "os", "time", "ctypes", "torch", "numpy"} | own
    bad = []
    for f in ("baseline.py", "train_baseline.py", "generate_baseline.py"):
        tree = ast.parse(open(os.path.join(HERE, f), encoding="utf-8").read())
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) else []
            bad += ["%s: %s" % (f, n) for n in names if n.split(".")[0] not in allowed]
            if isinstance(node, ast.Attribute) and node.attr == "path" and getattr(node.value, "id", "") == "sys":
                bad.append("%s: sys.path" % f)
    check("T10 ana kod klasor disindan import etmez", not bad, ", ".join(bad))


def main():
    if os.name == "nt":
        print("guc kisitlamasi (EcoQoS) kapali:", TB._no_power_throttling(), flush=True)   # arka planda ~10 kat yavas
    mz = model_z()
    print("Model Z:", "var" if mz else "YOK", flush=True)
    t1_targets(mz)
    t2_counts()
    t3_causal()
    t4_block(mz)
    t5_params()
    t7_exam()
    t8_generate()
    t10_imports()
    t9_drive(mz)
    t6_resume()
    print("\n%d GECTI   %d KALDI" % (sum(RESULTS), len(RESULTS) - sum(RESULTS)), flush=True)
    return all(RESULTS)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
