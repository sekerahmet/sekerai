# -*- coding: utf-8 -*-
"""tests_simplestories -- SimpleStories egitiminin kapilari: encode / decode gidis-donus, tek seferlik uretim (build_cache)
ve okuma (build), iz dogrulamasi (bozulunca durur), bolme sizintisi, uzun hikayenin pencerelere bolunmesi, sinav kumesi,
bits_per_byte, batch sekilleri (data_simplestories); sinav (exam_simplestories) ve Colab calistirici (colab_simplestories).
CPU; Drive ve ag YOK: parquet'ler asagidaki kucuk metinden yazilir, iki tokenizer ayni ayarlarla burada egitilir
(ss4096 = SimpleStories WordPiece ayarlari, gpt2 = byte-level BPE).  GPU yolu (compile, bf16, gpu_peak_gb) sinanamaz.

    python tests_simplestories.py
"""
import json
import math
import os
import shutil
import sys
import tempfile

import torch  # pyarrow ve tokenizers'tan ONCE (Windows DLL sirasi)
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))          # model_y: model ve genel egitim
torch.set_num_threads(1)

import data_simplestories as DS  # noqa: E402
import exam_simplestories as ES  # noqa: E402
import train_y as TR  # noqa: E402

RESULTS = []
SEQ = 128                                         # fixture penceresi
TINY = dict(d=16, turns=2, units=16, t_max=SEQ, output_link=False)   # kucuk Model X (phi'siz; phi TEST4'te)
TRAIN = [
    'Once upon a time, there was a girl named Lily. She had a red ball.\n\nOne day, Lily went to the park with her ball.',
    'Tom saw a big dog. The dog was happy. Tom said, "Can we play?" The dog wagged its tail.',
    "Lily's mom made a cake. It was sweet and good. Lily said thank you to her mom.",
    'One day, Ben found 12 red balls in the garden. He gave one to Sue. Sue was very happy!',
    'The cat did not like the rain. It sat under the table. When the sun came out, the cat went to play.',
    'Anna and Ben were friends. They liked to play in the park.\n\n"Let\'s go to the slide!" said Anna. Ben ran to it.',
    "Tim was sad. He lost his toy car. His dad helped him look for it. They found it under the bed. Tim was happy again.",
    'A little bird sat on a tree. It sang a happy song. The girl heard the song and smiled.',
    "Max did not want to sleep. He wanted to play. His mom said, \"It's time for bed.\" Max went to bed.",
    'Sue had a blue hat. The wind took the hat away. Sue ran and ran. A well-known boy found the hat and gave it back.',
]
TRAIN.append(" ".join(TRAIN) + " " + " ".join(TRAIN))    # SEQ'e sigmaz: pencerelere BOLUNUR (>= 3 pencere)
VALID = [
    'A zebra came to the park. Lily and Tom saw the zebra @ noon. They were very happy.',   # '@' ss sozlugunde yok
    TRAIN[3],                                                                             # birebir: iki tokenizer'da
    TRAIN[5].upper(),                      # yalniz buyuk harf farki: ss4096'da (kucuk harf) train'de, gpt2'de degil
    "Sue ran. " * 70,                                                                     # sigmaz
    'The cat sat under the tree. A little bird sang a song. The cat smiled.',
    'Max had a red car. He gave the car to Ben. Ben said, "Thank you!"',
]
NAME = "data/test-00000-of-00001.parquet"
TRAIN_FILES = ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"]


def check(name, ok, note=""):
    RESULTS.append(bool(ok))
    print("  %-78s %s  %s" % (name, "GECTI" if ok else "KALDI", note))


def make_tokenizers(folder):
    """ss4096: SimpleStories'in create_tokenizer ayarlari (kucuk harf, Whitespace + Punctuation + tek rakam, WordPiece,
    [UNK] 0, [EOS] 1); gpt2: byte-level BPE, eos <|endoftext|>.  Kucuk sozluk: kelimeler '##' parcalara bolunsun."""
    ss = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    ss.normalizer = normalizers.Sequence([normalizers.Lowercase(), normalizers.Replace("``", '"'),
                                          normalizers.Replace("''", '"')])
    ss.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Whitespace(), pre_tokenizers.Punctuation(),
                                                pre_tokenizers.Digits(individual_digits=True)])
    ss.post_processor = processors.TemplateProcessing(single="$A [EOS]", special_tokens=[("[EOS]", 1)])
    ss.decoder = decoders.WordPiece(prefix="##")
    ss.train_from_iterator(TRAIN, trainers.WordPieceTrainer(vocab_size=120, special_tokens=["[UNK]", "[EOS]"]))
    bpe = Tokenizer(models.BPE())
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.decoder = decoders.ByteLevel()
    bpe.train_from_iterator(TRAIN, trainers.BpeTrainer(vocab_size=400, special_tokens=["<|endoftext|>"],
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    paths = {}
    for tag, tok in (("ss4096", ss), ("gpt2", bpe)):
        paths[tag] = os.path.join(folder, tag + "_tokenizer.json")
        tok.save(paths[tag])
    return dict(ss4096=dict(path=paths["ss4096"], eos="[EOS]"), gpt2=dict(path=paths["gpt2"], eos="<|endoftext|>"))


def fixture():
    """Gecici kok: raw/ parquet'ler (train iki dosya), tokenizer'lar; build_cache yerel kaynakla (ag yok)."""
    root, src = tempfile.mkdtemp(), tempfile.mkdtemp()
    for name, stories in ((NAME, VALID), (TRAIN_FILES[0], TRAIN[:6]), (TRAIN_FILES[1], TRAIN[6:])):
        path = os.path.join(root, "raw", *name.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pq.write_table(pa.table({"story": stories, "topic": ["x"] * len(stories)}), path)
    dataset = dict(repo=None, column="story", files=dict(valid=[NAME], train=TRAIN_FILES), sha256={})
    tokenizers = make_tokenizers(src)
    meta = DS.build_cache(root, dataset=dataset, tokenizers=tokenizers, log=lambda s: None, chunk=4, seq_len=SEQ)
    return root, src, dataset, tokenizers, meta


def load(root, tag):
    return DS.build(root, tag, seq_len=SEQ, log=lambda s: None)


def t_tokens(root):
    ss, g = load(root, "ss4096")["vocab"], load(root, "gpt2")["vocab"]
    texts = TRAIN + VALID
    lossless = [DS.decode(DS.encode(s, g), g) == s for s in texts]
    check("gpt2: decode(encode(s)) == s (buyuk harf, satir sonu, '@' dahil); eos'un sozlukteki yazimi <eos>",
          all(lossless) and g.count(DS.EOS_TOKEN) == 1 and "<|endoftext|>" not in g,
          "%d/%d metin" % (sum(lossless), len(lossless)))

    ids_back = [DS.encode(DS.decode(DS.encode(s, ss), ss), ss) == DS.encode(s, ss) for s in texts]
    pieces = sum(ss[i].startswith("##") for s in texts for i in DS.encode(s, ss))
    unk = DS.encode(VALID[0], ss).count(ss.index(DS.UNK_TOKEN))
    shown = DS.decode(DS.encode(TRAIN[5], ss), ss)
    check("ss4096: encode(decode(ids)) == ids ('##' parcalar dahil); kucuk harf; kisaltma, tirnak, rakam, tire yapisik; "
          "sozlukte olmayan karakter iceren kelime [UNK] (fixture'da 'z' ve '@' yok)",
          all(ids_back) and pieces > 0 and unk == 3
          and DS.decode(DS.encode(TRAIN[1], ss), ss) == 'tom saw a big dog. the dog was happy. tom said, "can we play?" '
                                                          'the dog wagged its tail.'
          and "12 red balls" in DS.decode(DS.encode(TRAIN[3], ss), ss)
          and shown.startswith("anna and ben") and "let's" in shown and "well-known" in DS.decode(DS.encode(TRAIN[9], ss), ss),
          "%d/%d metin, %d '##' parca, zebra x 2 + '@' -> %d [UNK]; %r" % (sum(ids_back), len(ids_back), pieces, unk, shown[:60]))

    e = ss.index(DS.EOS_TOKEN)
    a, b = DS.encode("the cat", ss), DS.encode("the dog", ss)
    copy_ok = DS.encode("the cat", list(ss)) == a
    try:
        DS.encode("the cat", ["x", "y"])
        refused = False
    except AssertionError as err:
        refused = "yuklu degil" in str(err)
    check("decode: eos hikaye siniri '---'; sozlugun kopyasi da tokenizer'ini bulur; bilinmeyen sozluk reddedilir",
          DS.decode(a + [e] + b, ss) == "the cat\n\n---\n\nthe dog" and DS.decode(a + [e], ss) == "the cat\n\n---"
          and copy_ok and refused)


def t_data(root, dataset, tokenizers, meta):
    d = {tag: load(root, tag) for tag in DS.TAGS}
    counts = {tag: DS.audit(d[tag]) for tag in DS.TAGS}
    c = counts["ss4096"]
    same = all(counts[t][k] == c[k] for t in DS.TAGS for k in ("train_stories", "valid_stories", "valid_kept", "exam"))
    windows = {t: sum(-(-(len(DS.encode(s, d[t]["vocab"])) + 1) // (SEQ - 1)) for s in TRAIN) for t in DS.TAGS}
    check("veri: sayimlar (valid'de sigmayan atilir; train'de pencere = sum ceil((L + 1) / (SEQ - 1))), audit gecer, iki "
          "okuma ayni iz; kaynak parcalari fingerprint.json'dakilerle ayni, birlesik iz windows ve exam'i da katar",
          same and c["train_stories"] == 11 and c["valid_stories"] == 6 and c["valid_kept"] == 5 and c["exam"] == 3
          and all(counts[t]["train_kept"] == windows[t] for t in DS.TAGS)
          and all(load(root, t)["fingerprint"] == d[t]["fingerprint"] != meta[t]["fingerprint"] for t in DS.TAGS)
          and all(d[t]["fingerprints"][k] == meta[t]["fingerprints"][k] for t in DS.TAGS for k in meta[t]["fingerprints"])
          and all({"windows", "exam"} <= set(d[t]["fingerprints"]) for t in DS.TAGS),
          str({t: {k: counts[t][k] for k in ("train_stories", "train_kept", "valid_kept", "exam")} for t in DS.TAGS}))

    vit = {t: d[t]["valid_in_train"].tolist() for t in DS.TAGS}
    text = meta["gpt2"]["counts"]
    check("sizinti: birebir valid iki tokenizer'da bulunur; yalniz buyuk harf farki ss4096'da (kucuk harf) train'de, gpt2'de "
          "degil; metin duzeyi sayim fingerprint.json'da",
          vit == {"ss4096": [1, 2], "gpt2": [1]} and text["valid_text_in_train"] == 1
          and text["train_text_duplicates"] == 0 and text["valid_text_duplicates"] == 0, str(vit))

    kept = [0, 1, 2, 4, 5]
    ok_bytes = all(d[t]["valid_bytes"].tolist() == [len(VALID[i].encode("utf-8")) for i in kept] for t in DS.TAGS)
    fp_bytes = d["ss4096"]["fingerprints"]["valid_bytes"] == d["gpt2"]["fingerprints"]["valid_bytes"]
    roundtrip = all(meta[t]["counts"]["valid_roundtrip_fails"] == 0 and meta[t]["counts"]["valid_fast_encode_differs"] == 0
                    for t in DS.TAGS)
    check("bayt: hikaye basina UTF-8 bayt pencerelerle hizali, tokenizer'dan bagimsiz (iki tag'de ayni iz); uretimde "
          "valid gidis-donusu ve toplu encode = tek tek encode",
          ok_bytes and fp_bytes and roundtrip and d["ss4096"]["counts"]["valid_unk"] == 3,
          str({t: meta[t]["counts"]["valid_roundtrip"] for t in DS.TAGS}))

    for t in DS.TAGS:
        ids, mask = DS.sequences(d[t], "valid", [0, 3])
        v = d[t]["vocab"]
        eos = v.index(DS.EOS_TOKEN)
        L = int(d[t]["valid_length"][0])
        ok = (ids.shape == (2, SEQ) and ids.dtype == torch.long and ids[0, 0] == eos
              and ids[0, 1:L + 1].tolist() == DS.encode(VALID[0], v) and ids[0, L + 1] == eos
              and (ids[0, L + 2:] == v.index(DS.PAD_TOKEN)).all() and int(mask[0].sum()) == L + 2)
        check("%s: valid penceresi eos hikaye eos + sagdan dolgu (eos id'si, mask disi), sekil (n, SEQ)" % t, ok,
              "L %d" % L)

    again = DS.build_cache(root, dataset=dataset, tokenizers=tokenizers, log=lambda s: None)
    check("build_cache: var olan tag klasoru atlanir (uretmez, dokunmaz); iz ayni kalir",
          again == {} and all(load(root, t)["fingerprint"] == d[t]["fingerprint"] for t in DS.TAGS))


def t_fingerprint(root):
    caught = {}
    for what in ("token", "bytes", "count", "missing", "exam"):
        other = tempfile.mkdtemp()
        shutil.copytree(os.path.join(root, "gpt2"), os.path.join(other, "gpt2"))
        for n in ("exam_stories.npy", "exam_stories.json"):
            shutil.copyfile(os.path.join(root, n), os.path.join(other, n))
        f = os.path.join(other, "gpt2", "train.npy" if what == "token" else "train_bytes.npy")
        if what == "exam":
            f = os.path.join(other, "exam_stories.npy")
        a = np.load(f)
        if what == "exam":
            a[0] += 1                                           # sinav kumesinde tek hikaye
        elif what == "token":
            a[3] = a[3] + 1                                     # tek token
        elif what == "bytes":
            a[0] += 1                                           # tek hikayenin bayti
        elif what == "count":
            a = a[:-1]                                          # bir hikaye eksik
        np.save(f, a)
        if what == "missing":
            os.remove(f)
        try:
            load(other, "gpt2")
            caught[what] = ""
        except AssertionError as e:
            caught[what] = str(e)
        shutil.rmtree(other)
    check("iz: train'de tek token ya da tek bayt degisince build DURUR ve farkli parcayi soyler; hikaye sayisi bayt "
          "satiriyla tutmazsa, dosya yoksa ve sinav kumesi degisince durur (uretmez)",
          "iz tutmuyor" in caught["token"] and "'train'" in caught["token"]
          and "iz tutmuyor" in caught["bytes"] and "'train_bytes'" in caught["bytes"] and "'train'" not in caught["bytes"]
          and "hikaye siniri" in caught["count"] and "kural 9" in caught["missing"]
          and "exam_stories.npy izi" in caught["exam"],
          " | ".join("%s: %s" % (k, v[-40:]) for k, v in caught.items()))

    d = load(root, "ss4096")
    bad = []
    broken = dict(d, train=d["train"].copy())
    broken["train"][2] = len(d["vocab"])                        # sozluk disi id
    for x in (broken, dict(d, vocab=d["vocab"][:5] + d["vocab"][4:5] + d["vocab"][6:]),
              dict(d, valid_bytes=d["valid_bytes"][:-1]), dict(d, train_length=d["train_length"] - 1),
              dict(d, exam=np.concatenate([d["exam"], d["valid_in_train"]]))):
        try:
            DS.audit(x)
            bad.append(False)
        except AssertionError:
            bad.append(True)
    check("audit: sozluk disi id, sozlukte tekrar, hizasiz bayt dizisi, eksik hedef (pencere kisaltilmis) ve sizintili "
          "sinav kumesini yakalar", all(bad) and len(bad) == 5, str(bad))


def t_bits_per_byte(root):
    d = load(root, "gpt2")
    rows = np.arange(len(d["valid_start"]))
    ids, mask = DS.sequences(d, "valid", rows)
    V = len(d["vocab"])
    logits = torch.zeros(ids.shape[0], ids.shape[1] - 1, V, dtype=torch.float64)                 # tekduze model: her hedef ln V
    target, valid = ids[:, 1:], mask[:, 1:]
    ce = logits.logsumexp(-1) - logits.gather(-1, target[..., None])[..., 0]
    nll_sum, n = float(ce[valid].sum()), int(valid.sum())
    n_bytes = int(d["valid_bytes"].sum())
    bpb = DS.bits_per_byte(nll_sum, n_bytes)
    check("bits_per_byte: toplam nll / ln 2 / toplam bayt; tekduze modelde = hedef sayisi x log2 V / bayt; hedef = "
          "hikaye + eos (L + 1)",
          abs(bpb - n * math.log2(V) / n_bytes) < 1e-9 and n == int((d["valid_length"] + 1).sum())
          and abs(DS.bits_per_byte(math.log(2) * 8, 8) - 1.0) < 1e-12,
          "n %d bayt %d bpb %.4f" % (n, n_bytes, bpb))


def t_batches(root):
    d = load(root, "ss4096")
    bs = 2
    per_epoch = len(d["train_start"]) // bs
    rows = {DS.sequences(d, "train", [i])[0].numpy().tobytes(): i for i in range(len(d["train_start"]))}

    def taken(fn, step):
        ids, mask = fn(step)
        assert ids.shape == (bs, SEQ) and mask.shape == (bs, SEQ)
        return [rows[r.numpy().tobytes()] for r in ids]

    a, b = DS.batches(d, bs, seed=0), DS.batches(d, bs, seed=0)
    order = [7, 0, 3, 7, 1, 2]
    same = all(taken(a, s) == taken(b, s) for s in order)
    epoch0 = [r for s in range(per_epoch) for r in taken(a, s)]
    epoch1 = [r for s in range(per_epoch, 2 * per_epoch) for r in taken(a, s)]
    other = [r for s in range(per_epoch) for r in taken(DS.batches(d, bs, seed=1), s)]
    check("batches: sekil (batch, SEQ); ayni step ayni parca (cagri sirasindan bagimsiz); epok icinde tekrar yok; epok ve "
          "tohum sirayi degistirir",
          same and len(set(epoch0)) == len(epoch0) == bs * per_epoch and epoch0 != epoch1 and epoch0 != other,
          "epok %d adim" % per_epoch)

    b1, b2 = DS.batches(d, bs, 0, bucket=2), DS.batches(d, bs, 0, bucket=2)
    widths = [b1(s)[0].shape[1] for s in range(per_epoch)]
    fits = all(int(b1(s)[1].sum(1).max()) <= w < int(b1(s)[1].sum(1).max()) + DS.WIDTH_MULTIPLE and w % DS.WIDTH_MULTIPLE == 0
               for s, w in zip(range(per_epoch), widths))
    short = {DS.sequences(d, "train", [i])[0][0, :int(d["train_length"][i]) + 2].numpy().tobytes(): i
             for i in range(len(d["train_start"]))}
    ep_b = sorted(short[r[:int(m.sum())].numpy().tobytes()] for s in range(per_epoch) for r, m in zip(*b1(s)))
    check("batches bucket: adimin fonksiyonu; epokta bucket=None ile ayni hikayeler; genislik WIDTH_MULTIPLE'in kati ve "
          "en uzun pencere sigar", all(torch.equal(b1(s)[0], b2(s)[0]) for s in order) and fits
          and ep_b == sorted(epoch0), "genislik %s" % widths)


def t_split(root):
    """Uzun train hikayesi pencerelere bolunur: her token tam bir kez hedef, devam penceresi bir oncekinin son token'iyla
    baslar, pencereler <= SEQ, yalniz son pencere eos ile biter, yalniz ilk pencere eos ile baslar."""
    for t in DS.TAGS:
        d = load(root, t)
        v, eos = d["vocab"], d["vocab"].index(DS.EOS_TOKEN)
        heads = np.flatnonzero(d["train_head"]).tolist() + [len(d["train_start"])]
        ok, most = len(heads) - 1 == len(TRAIN), 0
        for i in range(len(TRAIN)):
            rows = list(range(heads[i], heads[i + 1]))
            ids, mask = DS.sequences(d, "train", rows)
            n = mask.sum(1).tolist()
            got = [x for r, m in zip(ids.tolist(), n) for x in r[1:m]]
            ok &= got == DS.encode(TRAIN[i], v) + [eos]                      # her token tam bir kez hedef
            ok &= ids[0, 0].item() == eos and all(r[0] != eos for r in ids[1:].tolist())
            ok &= ids[-1, n[-1] - 1].item() == eos and all(m == SEQ and r[m - 1] != eos
                                                            for r, m in zip(ids[:-1].tolist(), n[:-1]))
            ok &= all(ids[k + 1, 0].item() == ids[k, n[k] - 1].item() for k in range(len(rows) - 1))
            ok &= max(n) <= SEQ
            most = max(most, len(rows))
        c = d["counts"]
        check("%s bolme: uzun hikaye %d pencere; hedefler birlestirilince hikaye + eos (her token bir kez); devam penceresi "
              "oncekinin son token'iyla baslar; yalniz ilk eos ile baslar, yalniz son eos ile biter; pencere <= SEQ" % (t, most),
              ok and most >= 3 and c["train_split"] == 1 and c["train_targets"] == c["train_tokens"]
              and c["train_kept"] == len(TRAIN) - 1 + most, str({k: c[k] for k in ("train_kept", "train_split",
                                                                                  "train_targets", "train_tokens")}))


def t_exam(root):
    d = {t: load(root, t) for t in DS.TAGS}
    rows = ES.exam_rows(d["ss4096"])
    check("sinav kumesi: iki tag'de ayni hikayeler (sigan, hicbir tag'de train'de birebir degil); exam_rows sirali, "
          "count kirpar, tekrarlanabilir",
          rows.tolist() == d["gpt2"]["exam"].tolist() == [0, 3, 4] and ES.exam_rows(d["gpt2"], None).tolist() == [0, 3, 4]
          and len(ES.exam_rows(d["ss4096"], 2)) == 2 and ES.exam_rows(d["ss4096"], 2).tolist() == ES.exam_rows(
              d["ss4096"], 2).tolist(), str(rows.tolist()))
    for t in DS.TAGS:
        m, _ = TR.train_seq("shared", None, None, len(d[t]["vocab"]), steps=60, log_at=(),
                            batches=DS.batches(d[t], 4, 0), model_kw=TINY)
        e1, e64, auto = (ES.exam(m, d[t], rows, batch_size=b) for b in (1, 64, None))
        ids, mask = DS.sequences(d[t], "valid", rows)
        eos = d[t]["vocab"].index(DS.EOS_TOKEN)
        with torch.no_grad():
            _, nll = m.loss(ids, mask)
            logits = m.logits(ids[:, :-1]).double()
        target, valid = ids[:, 1:], mask[:, 1:]
        ce = logits.logsumexp(-1) - logits.gather(-1, target[..., None])[..., 0]
        acc = int((logits.argmax(-1) == target)[valid].sum()) / int(valid.sum())
        b = int(d[t]["valid_bytes"][rows].sum())
        bpb = float(ce[valid & (target != eos)].sum()) / math.log(2) / b
        bpb_eos = float(ce[valid].sum()) / math.log(2) / (b + len(rows))
        check("%s exam: nll = model.loss'un nll'i, accuracy elle; bits_per_byte (eos haric) ve _eos (eos dahil, +1 "
              "bayt/hikaye) elle; batch 1 = 64 = varsayilan" % t,
              abs(e64["nll"] - float(nll)) < 1e-5 and abs(e64["accuracy"] - acc) < 1e-9 and e64["n"] == int(valid.sum())
              and abs(e64["bits_per_byte"] - bpb) < 1e-5 and abs(e64["bits_per_byte_eos"] - bpb_eos) < 1e-5
              and e64["bytes"] == b and e64["stories"] == len(rows)
              and all(np.isclose(x[k], e64[k], rtol=1e-5, atol=1e-6, equal_nan=True) for x in (e1, auto) for k in e64),
              "nll %.4f acc %.4f bpb %.4f / %.4f" % (e64["nll"], e64["accuracy"], e64["bits_per_byte"],
                                                    e64["bits_per_byte_eos"]))

    ds, v = d["ss4096"], d["ss4096"]["vocab"]
    eos = v.index(DS.EOS_TOKEN)
    prompts = [[eos] + DS.encode(p, v) for p in ES.PROMPTS[:3]] + [[eos] + DS.encode(TRAIN[1][:18], v)]
    rows_t = ES.texts(m_ss(ds), ds, prompts, 12)
    manual = []
    model = m_ss(ds)
    with torch.no_grad():
        for p in prompts:
            w = list(p)
            for _ in range(12):
                w.append(int(model.logits(torch.tensor([w]))[0, -1].argmax()))
            g = w[len(p):]
            manual.append(g[:g.index(eos)] if eos in g else g)
    check("texts: acgozlu devam = tek tek elle uretim, ilk <eos>'ta kesilir; istem metni geri okunur (kucuk harf)",
          [r["ids"] for r in rows_t] == manual and rows_t[3]["prompt"] == TRAIN[1][:18].lower(), str(manual[:2]))

    halves, reals = ES.story_prompts(ds, rows[:2])
    st = ES.texts(model, ds, halves, 5, reals=reals)
    whole = [DS.encode(VALID[i], v) for i in (0, 4)]                      # valid pencere 0 ve 3 = VALID[0], VALID[4]
    lc = ES.loop_check([list(range(8)) * 2, list(range(20)), []])
    check("story_prompts: istem <eos> + ilk yari, gercek devam ikinci yari; loop_check tekrar eden 8'li",
          [h[1:] + r for h, r in zip(halves, reals)] == whole and all(h[0] == eos for h in halves)
          and st[0]["real"] == DS.decode(reals[0], v) and lc["loop"] == 1 and 0.8 < lc["distinct4"] < 1, str(lc))


_MODELS = {}


def m_ss(d):
    """ss4096 fixture'inda kisa egitilmis kucuk model (bir kez)."""
    if "ss" not in _MODELS:
        _MODELS["ss"], _ = TR.train_seq("shared", None, None, len(d["vocab"]), steps=60, log_at=(),
                                        batches=DS.batches(d, 4, 0), model_kw=TINY)
    return _MODELS["ss"]


def t_colab(root):
    import colab_simplestories as C
    d = load(root, "ss4096")
    tmp = tempfile.mkdtemp()
    out = tmp + "/r"
    run = C.start("TEST", d, out, steps=3, every=1, device="cpu", batch_size=4, model_kw=TINY, save_every=1)
    run["thread"].join(600)
    files = sorted(os.listdir(out))
    cfg = json.load(open(out + "/config.json"))
    ex = json.load(open(out + "/exams.json", encoding="utf-8"))
    fin = json.load(open(out + "/final.json", encoding="utf-8")) if "final.json" in files else {}
    check("colab_simplestories: CPU'da start -> checkpoint her adim, config (tag, iz, sozluk, pencere), sinav (bpb, "
          "step_ms; CPU'da gpu_peak_gb yok), son olcum (butun sinav kumesi, 12 istem, valid hikayeleri, perf)",
          run["done"] and not run["error"]
          and files == ["checkpoint_t000001.pt", "checkpoint_t000002.pt", "checkpoint_t000003.pt", "config.json",
                        "exams.json", "final.json", "log.txt", "model.pt"]
          and cfg["tag"] == "ss4096" and cfg["fingerprint"] == d["fingerprint"] and cfg["vocab"] == len(d["vocab"])
          and cfg["train_windows"] == len(d["train_start"]) and cfg["exam_stories"] == 3
          and [e["step"] for e in ex] == [0, 1, 2, 3] and "step_ms" not in ex[0] and all("step_ms" in e for e in ex[1:])
          and not any("gpu_peak_gb" in e for e in ex) and "bits_per_byte" in ex[0] and len(ex[0]["texts"]) == ES.PROBE_PROMPTS
          and fin.get("valid", {}).get("stories") == 3 and len(fin.get("prompts", [])) == len(ES.PROMPTS)
          and len(fin.get("stories", [])) == 3 and fin["perf"]["step_ms"] is not None and fin["perf"]["gpu_peak_gb"] is None,
          str(run["error"] or files))

    first = torch.load(out + "/model.pt")
    os.remove(out + "/checkpoint_t000003.pt")                        # 2. adimdan sonra kesilmis gibi
    run2 = C.start("TEST", d, out, steps=3, every=1, device="cpu", batch_size=4, model_kw=TINY, save_every=1, resume=True)
    run2["thread"].join(600)
    log = open(out + "/log.txt", encoding="utf-8").read()
    second = torch.load(out + "/model.pt")
    try:
        C.start("TEST2", d, out, steps=5, every=1, device="cpu", batch_size=4, model_kw=TINY, save_every=1, resume=True)
        refused = False
    except RuntimeError:
        refused = True
    check("colab_simplestories, surdurme: son paketten devam, model = kesintisiz kosununki (bit duzeyinde); ayar farkliysa "
          "reddeder", run2["done"] and not run2["error"] and "SURDURULDU adim 2'den" in log
          and all(torch.equal(first[k], second[k]) for k in first) and refused, str(run2["error"]))

    run4 = C.start("TEST4", d, tmp + "/p", steps=2, every=1, device="cpu", batch_size=4,
                   model_kw=dict(TINY, output_link=True), save_every=1)
    run4["thread"].join(600)
    ex4 = json.load(open(tmp + "/p/exams.json", encoding="utf-8"))
    fin4 = json.load(open(tmp + "/p/final.json", encoding="utf-8")) if os.path.exists(tmp + "/p/final.json") else {}
    old = dict(cfg, model_kw={k: v for k, v in cfg["model_kw"].items() if k != "output_link"})
    json.dump(old, open(out + "/config.json", "w"), indent=1)   # 29 Eylul oncesi config: output_link yazilmamis
    if os.path.exists(out + "/checkpoint_t000003.pt"):
        os.remove(out + "/checkpoint_t000003.pt")
    run5 = C.start("TEST", d, out, steps=3, every=1, device="cpu", batch_size=4, model_kw=TINY, save_every=1, resume=True)
    run5["thread"].join(600)
    check("colab_simplestories, output_link: config'te acik (TINY'de False); True'da q, u sinavda, gunlukte ve "
          "final.json'da; output_link'siz eski config surdurulur",
          cfg["model_kw"]["output_link"] is False and run4["done"] and not run4["error"]
          and all("link_q" in e and "link_u" in e for e in ex4) and any("bag q" in l for l in run4["lines"])
          and "link_q" in fin4 and fin4["link_u"] >= 0 and run5["done"] and not run5["error"],
          str(run4["error"] or run5["error"] or run4["lines"][-2:]))

    run3 = C.start("TEST3", d, tmp + "/s", steps=500, every=1, device="cpu", batch_size=4, model_kw=TINY)
    C.stop()
    run3["thread"].join(600)
    check("colab_simplestories: stop -> model.pt yazilir, iplik cikar; pulse hatasiz (GPU yolu -- compile, bf16, "
          "gpu_peak_gb -- CPU'da sinanamaz)",
          not run3["thread"].is_alive() and not run3["done"] and "DURDURULDU" in run3["lines"][-1]
          and os.path.exists(run3["out"] + "/model.pt"), run3["lines"][-1])
    C.pulse(1)
    shutil.rmtree(tmp)


if __name__ == "__main__":
    print("tests (train_simplestories)")
    root, src, dataset, tokenizers, meta = fixture()
    try:
        t_tokens(root)
        t_data(root, dataset, tokenizers, meta)
        t_fingerprint(root)
        t_bits_per_byte(root)
        t_batches(root)
        t_split(root)
        t_exam(root)
        t_colab(root)
    finally:
        shutil.rmtree(root)
        shutil.rmtree(src)
    print("\n%d GECTI   %d KALDI" % (sum(RESULTS), len(RESULTS) - sum(RESULTS)))
    sys.exit(0 if all(RESULTS) else 1)
