# -*- coding: utf-8 -*-
"""tests_tinystories -- TinyStories egitiminin kapilari: tokenizer, Drive onbellegi okuma ve iz (data_tinystories),
batch'lerin adima bagliligi, surdurme, sinav (exam_tinystories) ve Colab calistirici (colab_tinystories).
CPU, saniyeler; Drive ve ag YOK: onbellek dosyalari asagidaki kucuk metinden ayni bicimde (v4) yazilir.

    python tests_tinystories.py
"""
import collections
import copy
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))          # model_20: model ve genel egitim
torch.set_num_threads(1)

import data_tinystories as DT  # noqa: E402
import exam_tinystories as ET  # noqa: E402
import train_20 as TR  # noqa: E402

RESULTS = []
SEQ = 128                                         # fixture penceresi; istemlerin en uzunu ~80 token
TINY = dict(d=16, turns=2, units=16, t_max=SEQ)   # kucuk Model X

# Kanonik yazim (tek bosluk, paragraflar tek satir sonuyla): decode(encode(s)) == s beklenir.
TRAIN = [
    'Once upon a time, there was a girl named Lily. She had a red ball.\nOne day, Lily went to the park with her ball.',
    'Tom saw a big dog. The dog was happy. Tom said, "Can we play?" The dog wagged its tail.',
    "Lily's mom made a cake. It was sweet and good. Lily said thank you to her mom.",
    'One day, Ben found 3 red balls in the garden. He gave one to Sue. Sue was very happy!',
    'The cat did not like the rain. It sat under the table. When the sun came out, the cat went to play.',
    'Anna and Ben were friends. They liked to play in the park.\n"Let\'s go to the slide!" said Anna. Ben ran to the slide.',
    "Tim was sad. He lost his toy car. His dad helped him look for it. They found it under the bed. Tim was happy again.",
    'A little bird sat on a tree. It sang a happy song. The girl heard the song and smiled.',
    "Max did not want to sleep. He wanted to play. His mom said, \"It's time for bed.\" Max went to bed.",
    'Sue had a blue hat. The wind took the hat away. Sue ran and ran. A boy found the hat and gave it back.',
    "Once upon a time, there was a big tree. The tree was old. Many birds lived in the tree. They were happy.",
    'Ben wanted to help his mom. He put the toys in the box. Mom said, "Good job, Ben!" Ben smiled.',
    'The dog found a bone. He dug a hole in the garden. Then he put the bone in the hole. Wait... where is the bone?',
    'Lily and Tom went to the sea. They saw a big fish. "Look!" said Tom. The fish went back into the water.',
    "Tom ran. " * 70,                               # 140+ token: pencereye sigmaz, atilir
]
VALID = [
    'A zebra came to the park. Lily and Tom saw the zebra. They were very happy.',     # "zebra" sozlukte yok
    TRAIN[3],                                                                          # train'de birebir: sinava girmez
    "Sue ran. " * 70,                                                                  # sigmaz
    'The cat sat under the tree. A little bird sang a song. The cat smiled.',
    'Max had a red car. He gave the car to Ben. Ben said, "Thank you!"',
    'One day, the girl found a hat in the garden. It was blue. She was happy.',
]


def check(name, ok, note=""):
    RESULTS.append(bool(ok))
    print("  %-78s %s  %s" % (name, "GECTI" if ok else "KALDI", note))


def write_cache(root, train=TRAIN, valid=VALID, tag=DT.TAG, vocab=None):
    """Fixture'i Drive onbellegi bicimiyle yazar: sozluk (object dizisi, SPECIAL + train kelimeleri, siklik sirasiyla;
    verilirse o), akislar int16, her hikayeden sonra <eos>."""
    if vocab is None:
        counts = collections.Counter(t for s in train for t in DT.tokenize(DT.normalize(s)))
        counts.pop(DT.NL_TOKEN, None)
        vocab = list(DT.SPECIAL) + [w for w, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    cache = os.path.join(root, "onbellek")
    os.makedirs(cache, exist_ok=True)
    np.save(os.path.join(cache, "sozluk_%s.npy" % tag), np.array(vocab, dtype=object))
    eos = vocab.index(DT.EOS_TOKEN)
    for name, stories in (("train", train), ("valid", valid)):
        ids = [i for s in stories for i in DT.encode(s, vocab) + [eos]]
        np.save(os.path.join(cache, "akis_%s_%s.npy" % (name, tag)), np.array(ids, dtype=np.int16))
    return vocab


def fixture():
    root = tempfile.mkdtemp()
    write_cache(root)
    return root, DT.build(root, seq_len=SEQ, log=lambda s: None)


def t_tokens():
    check("tokenizer: kivrik tirnak duz, aksan duser, uc nokta '...', kisaltma tek token, bos satirlar tek <nl>",
          DT.normalize("“Hi,” café…") == '"Hi," cafe...'
          and DT.tokenize("Lily's ball.\n\n\nIt's red!") == ["Lily's", "ball", ".", "<nl>", "It's", "red", "!"])
    root, d = fixture()
    vocab = d["vocab"]
    same = [DT.decode(DT.encode(s, vocab), vocab) == s for s in TRAIN[:-1]]
    ids_back = [DT.encode(DT.decode(DT.encode(s, vocab), vocab), vocab) == DT.encode(s, vocab) for s in TRAIN]
    unk = DT.encode(VALID[0], vocab).count(vocab.index(DT.UNK_TOKEN))
    check("tokenizer: decode(encode(s)) == s (kanonik yazim, tirnak ve <nl> dahil); encode(decode(ids)) == ids; "
          "sozlukte olmayan kelime <bilinmeyen>", all(same) and all(ids_back) and unk == 2,
          "%d/%d metin, zebra -> %d <bilinmeyen>" % (sum(same), len(same), unk))
    shutil.rmtree(root)


def t_data():
    root, d = fixture()
    d2 = DT.build(root, seq_len=SEQ, log=lambda s: None)
    counts = DT.audit(d)
    ok = (counts["train_stories"] == 15 and counts["train_kept"] == 14 and counts["valid_stories"] == 6
          and counts["valid_kept"] == 5 and counts["valid_in_train"] == 1 and d["valid_in_train"].tolist() == [1]
          and d["fingerprint"] == d2["fingerprint"] and len(d["fingerprint"]) == 12)
    check("veri: sayimlar (sigmayan hikaye atilir), train'de birebir gecen valid bulunur, iki okuma ayni iz, audit gecer",
          ok, str(counts))

    ids, mask = DT.sequences(d, "train", [0, 5])
    eos, pad = d["vocab"].index(DT.EOS_TOKEN), d["vocab"].index(DT.PAD_TOKEN)
    L = int(d["train_length"][0])
    story = DT.encode(TRAIN[0], d["vocab"])
    check("veri: pencere <eos> hikaye <eos> + sagdan <dolgu>, sekil (n, SEQ), mask = L + 2; metin geri okunur",
          ids.shape == (2, SEQ) and ids[0, 0] == eos and ids[0, 1:L + 1].tolist() == story and ids[0, L + 1] == eos
          and (ids[0, L + 2:] == pad).all() and int(mask[0].sum()) == L + 2 and ids.dtype == torch.long
          and DT.decode(ids[0, 1:L + 1], d["vocab"]) == TRAIN[0], "L %d" % L)

    # iz: train'de tek token degisirse birlesik iz ve train parcasi degisir, sozluk ve valid parcasi degismez
    other = tempfile.mkdtemp()
    changed = TRAIN[:]
    changed[2] = changed[2].replace("sweet", "good")
    write_cache(other, train=changed, vocab=d["vocab"])
    d3 =DT.build(other, seq_len=SEQ, log=lambda s: None)
    f, f3 = d["fingerprints"], d3["fingerprints"]
    check("veri: train'de tek kelime degisince iz degisir; sozluk ve valid parcalari ayni kalir",
          d3["fingerprint"] != d["fingerprint"] and f3["train"] != f["train"] and f3["vocab"] == f["vocab"]
          and f3["valid"] == f["valid"], "%s -> %s" % (d["fingerprint"], d3["fingerprint"]))

    caught = []
    bad = dict(d, train=d["train"].copy())
    bad["train"][3] = pad                                             # akista <dolgu>
    for broken in (bad, dict(d, vocab=[DT.EOS_TOKEN, DT.PAD_TOKEN] + d["vocab"][2:])):
        try:
            DT.audit(broken)
            caught.append(False)
        except AssertionError:
            caught.append(True)
    try:
        DT.build(tempfile.mkdtemp(), log=lambda s: None)
        caught.append(False)
    except AssertionError as e:
        caught.append("kural 9" in str(e))
    check("veri: audit akista <dolgu>yu ve bozuk sozluk basini yakalar; onbellek yoksa build durur (uretmez)",
          all(caught) and len(caught) == 3, str(caught))
    shutil.rmtree(root)
    shutil.rmtree(other)


def t_batches():
    root, d = fixture()
    per_epoch = len(d["train_start"]) // 4
    rows = {DT.sequences(d, "train", [i])[0].numpy().tobytes(): i for i in range(len(d["train_start"]))}

    def taken(fn, step):
        ids, mask = fn(step)
        assert ids.shape == (4, SEQ) and mask.shape == (4, SEQ)
        return [rows[r.numpy().tobytes()] for r in ids]

    a, b = DT.batches(d, 4, seed=0), DT.batches(d, 4, seed=0)
    order = [7, 0, 3, 7, 1, 2]                                       # karisik sirayla cagrilir (surdurme gibi)
    same = all(taken(a, s) == taken(b, s) for s in order) and all(taken(a, s) == taken(b, s) for s in range(9))
    epoch0 = [r for s in range(per_epoch) for r in taken(a, s)]
    epoch1 = [r for s in range(per_epoch, 2 * per_epoch) for r in taken(a, s)]
    other = [r for s in range(per_epoch) for r in taken(DT.batches(d, 4, seed=1), s)]
    check("batches: ayni step her zaman ayni parca (cagri sirasindan bagimsiz), sekil sabit; epok icinde tekrar yok; "
          "epok ve tohum sirayi degistirir",
          same and len(set(epoch0)) == len(epoch0) == 4 * per_epoch and epoch0 != epoch1 and epoch0 != other,
          "epok %d adim, %s | %s" % (per_epoch, epoch0, epoch1))

    packs = {}
    keep = lambda step, model, opt: packs.setdefault(step, dict(step=step, model=copy.deepcopy(model.state_dict()),
                                                                   optimizer=copy.deepcopy(opt.state_dict())))
    n = len(d["vocab"])
    full, curve = TR.train_seq("shared", None, None, n, steps=8, log_at=(0, 8), batches=DT.batches(d, 4, 0),
                               save_every=2, save=keep, model_kw=TINY)
    resumed, _ = TR.train_seq("shared", None, None, n, steps=8, log_at=(), batches=DT.batches(d, 4, 0),
                              checkpoint=packs[4], model_kw=TINY)
    same = all(torch.equal(x, y) for x, y in zip(full.state_dict().values(), resumed.state_dict().values()))
    check("surdurme: batches ile 8 adim (%d epok), 4. adim paketinden surdurulen = kesintisiz, bit duzeyinde; kayip duser"
          % (8 // per_epoch), same and curve[-1]["nll"] < curve[0]["nll"] and full.blocks[0].attention.W_query.shape[0] == 16,
          "%.3f -> %.3f" % (curve[0]["nll"], curve[-1]["nll"]))
    shutil.rmtree(root)


def t_bucket():
    """bucket: epokta ayni hikayeler ve ayni adim sayisi, batch'ler boya gore; genislik WIDTH_MULTIPLE'in kati ve en uzun
    pencere sigar; adimin fonksiyonu (surdurme bit duzeyinde)."""
    root, d = fixture()
    n_rows, bs = len(d["train_start"]), 4
    per_epoch = n_rows // bs
    rows = {DT.sequences(d, "train", [i])[0][0, :int(d["train_length"][i]) + 2].numpy().tobytes(): i
            for i in range(n_rows)}
    taken = lambda ids, mask: [rows[r[:int(m_.sum())].numpy().tobytes()] for r, m_ in zip(ids, mask)]
    full = DT.batches(d, bs, 0)
    b1, b2 = DT.batches(d, bs, 0, bucket=2), DT.batches(d, bs, 0, bucket=2)
    order = [5, 0, 3, 5, 1, 2, 4]                                         # karisik sira (surdurme gibi)
    determ = all(torch.equal(b1(s)[0], b2(s)[0]) for s in order)
    ep = lambda fn, e: sorted(r for s in range(e * per_epoch, (e + 1) * per_epoch) for r in taken(*fn(s)))
    lens = d["train_length"]
    spread = lambda fn: np.mean([np.ptp(lens[taken(*fn(s))]) for s in range(per_epoch)])
    widths = [b1(s)[0].shape[1] for s in range(per_epoch)]
    fits = all(w % DT.WIDTH_MULTIPLE == 0 or w == SEQ for w in widths) and all(
        int(b1(s)[1].sum(1).max()) <= b1(s)[0].shape[1] < int(b1(s)[1].sum(1).max()) + DT.WIDTH_MULTIPLE
        or b1(s)[0].shape[1] == SEQ for s in range(per_epoch))
    check("batches bucket: adimin fonksiyonu (cagri sirasindan bagimsiz); her epok bucket=None ile AYNI hikayeler, ayni "
          "adim sayisi; batch ici boy farki daha kucuk; genislik WIDTH_MULTIPLE'in kati, en uzun pencere sigar",
          determ and ep(b1, 0) == ep(full, 0) and ep(b1, 1) == ep(full, 1) and len(set(ep(b1, 0))) == per_epoch * bs
          and spread(b1) < spread(full) and fits,
          "boy farki bucket %.1f, rastgele %.1f; genislik %s" % (spread(b1), spread(full), widths))

    packs = {}
    keep = lambda step, model, opt: packs.setdefault(step, dict(step=step, model=copy.deepcopy(model.state_dict()),
                                                                   optimizer=copy.deepcopy(opt.state_dict())))
    n = len(d["vocab"])
    whole, _ = TR.train_seq("shared", None, None, n, steps=8, log_at=(), batches=DT.batches(d, bs, 0, bucket=2),
                            save_every=2, save=keep, model_kw=TINY)
    resumed, _ = TR.train_seq("shared", None, None, n, steps=8, log_at=(), batches=DT.batches(d, bs, 0, bucket=2),
                              checkpoint=packs[4], model_kw=TINY)
    check("surdurme, bucket: 4. adim paketinden surdurulen = kesintisiz, bit duzeyinde",
          all(torch.equal(x, y) for x, y in zip(whole.state_dict().values(), resumed.state_dict().values())))
    shutil.rmtree(root)


def t_exam():
    root, d = fixture()
    rows = ET.exam_rows(d)
    check("exam_rows: train'de birebir gecen valid disarida, sirali, tekrarlanabilir; count kirpar",
          rows.tolist() == [0, 2, 3, 4] and ET.exam_rows(d, None).tolist() == rows.tolist()
          and ET.exam_rows(d, 2).tolist() == ET.exam_rows(d, 2).tolist() and len(ET.exam_rows(d, 2)) == 2,
          str(rows.tolist()))

    # kisa egitilmis model: accuracy 0'dan buyuk olsun, uretim tek token'a takilmasin
    m, _ = TR.train_seq("shared", None, None, len(d["vocab"]), steps=60, log_at=(), batches=DT.batches(d, 4, 0),
                        model_kw=TINY)
    e1, e64 = ET.exam(m, d, rows, batch_size=1), ET.exam(m, d, rows)
    ids, mask = DT.sequences(d, "valid", rows)
    with torch.no_grad():
        _, nll = m.loss(ids, mask)
        logits = m.logits(ids[:, :-1])
    valid = mask[:, 1:]
    acc = int((logits.argmax(-1) == ids[:, 1:])[valid].sum()) / int(valid.sum())
    check("exam: nll = model.loss'un nll'i, accuracy = elle hesap (kirpmasiz pencere); batch 1 ile 64 ayni",
          abs(e64["nll"] - float(nll)) < 1e-5 and abs(e64["accuracy"] - acc) < 1e-9 and e64["n"] == int(valid.sum())
          and e64["accuracy"] > 0
          and all(np.isclose(e1[k], e64[k], rtol=1e-5, atol=1e-6, equal_nan=True) for k in e64)
          and abs(e64["ppl"] - np.exp(e64["nll"])) < 1e-6,
          "nll %.4f acc %.4f eos %.2f" % (e64["nll"], e64["accuracy"], e64["eos_ok"]))

    eos = d["vocab"].index(DT.EOS_TOKEN)
    prompts = [[eos] + DT.encode(p, d["vocab"]) for p in ET.PROMPTS[:3]] + [[eos] + DT.encode(TRAIN[1][:18], d["vocab"])]
    rows_t = ET.texts(m, d, prompts, 12)
    manual = []
    with torch.no_grad():
        for p in prompts:
            w = list(p)
            for _ in range(12):
                w.append(int(m.logits(torch.tensor([w]))[0, -1].argmax()))
            g = w[len(p):]
            manual.append(g[:g.index(eos)] if eos in g else g)
    check("texts: acgozlu devam = tek tek elle uretim, ilk <eos>'ta kesilir; istem metni geri okunur",
          [r["ids"] for r in rows_t] == manual and rows_t[3]["prompt"] == TRAIN[1][:18]
          and len({tuple(g) for g in manual}) > 1, str(manual[:2]))

    halves, reals = ET.story_prompts(d, rows[:2])
    st = ET.texts(m, d, halves, 5, reals=reals)
    whole = [DT.encode(VALID[i], d["vocab"]) for i in (0, 3)]              # valid pencere 0 ve 2 = VALID[0], VALID[3]
    check("story_prompts: istem <eos> + hikayenin ilk yarisi, gercek devam ikinci yarisi; texts 'real'i yazar",
          [h[1:] + r for h, r in zip(halves, reals)] == whole and all(h[0] == eos for h in halves)
          and "real" in st[0] and st[0]["real"] == DT.decode(reals[0], d["vocab"]))

    lc = ET.loop_check([list(range(8)) * 2, list(range(20)), []])
    check("loop_check: tekrar eden 8'li sayilir, farkli 4'lu orani", lc["loop"] == 1 and 0.8 < lc["distinct4"] < 1,
          str(lc))
    shutil.rmtree(root)


def t_colab():
    import colab_tinystories as C
    root, d = fixture()
    tmp = tempfile.mkdtemp()
    out = tmp + "/r"
    run = C.start("TEST", d, out, steps=3, every=1, device="cpu", batch_size=4, model_kw=TINY, save_every=1)
    run["thread"].join(600)
    files = sorted(os.listdir(out))
    cfg = json.load(open(out + "/config.json"))
    ex = json.load(open(out + "/exams.json", encoding="utf-8"))
    fin = json.load(open(out + "/final.json", encoding="utf-8")) if "final.json" in files else {}
    check("colab_tinystories: CPU'da start -> checkpoint her adim, config (iz, epok, model_kw + anchor, rope), sinav "
          "(4 istem metni), son olcum (butun valid, alt kume, 12 istem, 4 valid hikayesi)",
          run["done"] and not run["error"]
          and files == ["checkpoint_t000001.pt", "checkpoint_t000002.pt", "checkpoint_t000003.pt", "config.json",
                        "exams.json", "final.json", "log.txt", "model.pt"]
          and cfg["fingerprint"] == d["fingerprint"] and cfg["steps_per_epoch"] == 3 and cfg["epochs"] == 1.0
          and cfg["model_kw"]["d"] == 16 and "anchor" in cfg["model_kw"] and cfg["rope"] is True
          and [e["step"] for e in ex] == [0, 1, 2, 3] and len(ex[0]["texts"]) == ET.PROBE_PROMPTS
          and fin.get("valid", {}).get("n", 0) > 0 and len(fin.get("prompts", [])) == len(ET.PROMPTS)
          and len(fin.get("stories", [])) == 4 and "real" in fin["stories"][0],
          str(run["error"] or files))

    first = torch.load(out + "/model.pt")
    os.remove(out + "/checkpoint_t000003.pt")                        # 2. adimdan sonra kesilmis gibi
    old_cfg = {k: v for k, v in cfg.items() if k != "bucket"}        # bucket'tan onceki config: rastgele batch'le kosmustu
    json.dump(old_cfg, open(out + "/config.json", "w"), indent=1)
    try:
        C.start("TEST", d, out, steps=3, every=1, device="cpu", batch_size=4, model_kw=TINY, save_every=1, resume=True,
                bucket=2)
        refused_bucket = False
    except RuntimeError:
        refused_bucket = True
    run2 = C.start("TEST", d, out, steps=3, every=1, device="cpu", batch_size=4, model_kw=TINY, save_every=1, resume=True)
    run2["thread"].join(600)
    log = open(out + "/log.txt", encoding="utf-8").read()
    ex2 = json.load(open(out + "/exams.json", encoding="utf-8"))
    second = torch.load(out + "/model.pt")
    try:
        C.start("TEST2", d, out, steps=5, every=1, device="cpu", batch_size=4, model_kw=TINY, save_every=1, resume=True)
        refused = False
    except RuntimeError:
        refused = True
    check("colab_tinystories, surdurme: son paketten devam (bucket anahtarsiz eski config dahil), model = kesintisiz "
          "kosununki (bit duzeyinde), sinav satiri cift degil; ayar (steps, bucket) farkliysa reddeder",
          run2["done"] and not run2["error"] and "SURDURULDU adim 2'den" in log
          and [e["step"] for e in ex2] == [0, 1, 2, 3]
          and all(torch.equal(first[k], second[k]) for k in first) and refused and refused_bucket,
          str(run2["error"] or [e["step"] for e in ex2]))

    run3 = C.start("TEST3", d, tmp + "/s", steps=500, every=1, device="cpu", batch_size=4, model_kw=TINY)
    C.stop()
    run3["thread"].join(600)
    check("colab_tinystories: stop -> bir sonraki sinavda model.pt yazilir, iplik cikar; pulse hatasiz",
          not run3["thread"].is_alive() and not run3["done"] and "DURDURULDU" in run3["lines"][-1]
          and os.path.exists(run3["out"] + "/model.pt"), run3["lines"][-1])
    C.pulse(1)
    shutil.rmtree(root)
    shutil.rmtree(tmp)


if __name__ == "__main__":
    print("tests (train_tinystories)")
    for f in (t_tokens, t_data, t_batches, t_bucket, t_exam, t_colab):
        f()
    print("\n%d GECTI   %d KALDI" % (sum(RESULTS), len(RESULTS) - sum(RESULTS)))
    sys.exit(0 if all(RESULTS) else 1)
