# -*- coding: utf-8 -*-
"""tests_simplestories -- SimpleStories egitiminin kapilari: encode / decode gidis-donus, tek seferlik uretim (build_cache)
ve okuma (build), iz dogrulamasi (bozulunca durur), bolme sizintisi, uzun hikayenin pencerelere bolunmesi, sinav kumesi,
bits_per_byte, batch sekilleri, text_reference.json (data_simplestories); sinav (exam_simplestories: exam, nll_by_frequency, exam_train,
alpha_summary, count_text_errors) ve Colab calistirici (colab_simplestories: tokens_per_sec, mfu).
CPU; Drive ve ag YOK: parquet'ler asagidaki kucuk metinden yazilir, iki tokenizer ayni ayarlarla burada egitilir
(ss4096 = SimpleStories WordPiece ayarlari, gpt2 = byte-level BPE).  GPU yolu (compile, bf16, gpu_peak_gb, mfu) sinanamaz.

    python tests_simplestories.py
"""
import collections
import hashlib
import json
import math
import os
import re
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
import model_y as M  # noqa: E402
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


def t_text_reference(root):
    ok = {}
    for t in DS.TAGS:
        d = load(root, t)
        ref = json.load(open(os.path.join(root, t, "text_reference.json"), encoding="utf-8"))
        words, xax = DS._text_reference(d["vocab"], d["train"])
        body = dict(words=sorted(words), xax={"%s|%s" % k: c for k, c in xax.items()})
        sha = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        source = json.load(open(os.path.join(root, t, "fingerprint.json"), encoding="utf-8"))["fingerprint"]
        ok[t] = (d["text_reference"]["words"] == words and d["text_reference"]["xax"] == xax and len(xax) > 0
                 and ref["source"] == source and ref["sha256"] == sha and d["text_reference"]["sha256"] == sha[:12]
                 and ref["counts"]["words"] == len(words) and ref["counts"]["train_tokens"] == len(d["train"]))
    check("text_reference.json: build_cache iki tag'de uretir; build okur; kelime kumesi ve 'X and X' sayilari train'den "
          "yeniden hesapla ayni; kaynak izi = fingerprint.json, icerik izi = govdenin sha256'si", all(ok.values()), str(ok))

    caught, regen = {}, False
    for what in ("missing", "content", "source", "rule"):
        other = tempfile.mkdtemp()
        shutil.copytree(os.path.join(root, "gpt2"), os.path.join(other, "gpt2"))
        for n in ("exam_stories.npy", "exam_stories.json"):
            shutil.copyfile(os.path.join(root, n), os.path.join(other, n))
        f = os.path.join(other, "gpt2", "text_reference.json")
        ref = json.load(open(f, encoding="utf-8"))
        if what == "missing":
            os.remove(f)
        else:
            if what == "content":
                ref["xax"][next(iter(ref["xax"]))] += 1                   # tek sayim
            elif what == "source":
                ref["source"] = "0" * 12
            else:
                ref["rule"] += " (eski)"
            json.dump(ref, open(f, "w", encoding="utf-8"), ensure_ascii=False)
        try:
            load(other, "gpt2")
            caught[what] = ""
        except AssertionError as e:
            caught[what] = str(e)
        if what == "missing":
            meta = DS.build_text_reference(other, "gpt2", log=lambda s: None)
            again = DS.build_text_reference(other, "gpt2", log=lambda s: None)
            regen = (meta is not None and again is None and meta["sha256"] == ref["sha256"]
                     and load(other, "gpt2")["text_reference"]["sha256"] == ref["sha256"][:12])
        shutil.rmtree(other)
    check("text_reference.json yoksa build DURUR (build_text_reference'i soyler, bellekte uretmez); icerik izi, kaynak izi "
          "ya da kural tutmazsa DURUR; build_text_reference ayni dosyayi yeniden uretir, varsa dokunmaz",
          "build_text_reference" in caught["missing"] and "icerik izi" in caught["content"]
          and "baska bir veriden" in caught["source"] and "kural degismis" in caught["rule"] and regen,
          " | ".join("%s: %s" % (k, v[-50:]) for k, v in caught.items()))


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
              and all(np.isclose(x[k], e64[k], rtol=1e-5, atol=1e-6, equal_nan=True) for x in (e1, auto) for k in e64
                      if k != "nll_by_frequency"),
              "nll %.4f acc %.4f bpb %.4f / %.4f" % (e64["nll"], e64["accuracy"], e64["bits_per_byte"],
                                                    e64["bits_per_byte_eos"]))

        # nll_by_frequency: fixture sozlugu kucuk (120 / 400) -> sinirlar (8, 32, 64), dort bant dolu
        V, fp = len(d[t]["vocab"]), d[t]["fingerprint"]
        saved, ES.FREQUENCY_BANDS = ES.FREQUENCY_BANDS, (8, 32, 64)
        ES._CACHE.pop(("band", fp), None)
        try:
            f64, f1 = (ES.exam(m, d[t], rows, batch_size=b_)["nll_by_frequency"] for b_ in (64, 1))
        finally:
            ES.FREQUENCY_BANDS = saved
            ES._CACHE.pop(("band", fp), None)
        counts = np.bincount(d[t]["train"].astype(np.int64), minlength=V)
        rank = np.empty(V, dtype=np.int64)
        rank[np.argsort(-counts, kind="stable")] = np.arange(V)
        band = torch.as_tensor((rank >= 8).astype(int) + (rank >= 32) + (rank >= 64))
        probs = logits.softmax(-1)[valid]
        want = []
        for k in range(4):
            sel = valid & (band[target] == k)
            nk = int(sel.sum())
            want.append((nk, float(ce[sel].sum()) / nk if nk else None,
                         float(probs[:, band == k].sum()) / nk if nk else None))
        got = [(x["targets"], x["nll"], x["mass_ratio"]) for x in f64]
        close = lambda a, b_: (a is None and b_ is None) or (a is not None and b_ is not None and abs(a - b_) < 1e-5)
        check("%s nll_by_frequency: bant = train'in bincount siklik sirasi; bantta hedef sayisi, nll, softmax kutlesi / "
              "hedef sayisi elle; batch 1 = 64" % t,
              all(g[0] == w[0] and close(g[1], w[1]) and close(g[2], w[2]) for g, w in zip(got, want))
              and all(a["targets"] == c["targets"] and close(a["nll"], c["nll"]) and close(a["mass_ratio"], c["mass_ratio"])
                      for a, c in zip(f1, f64))
              and [x["ranks"] for x in f64] == [[0, 8], [8, 32], [32, 64], [64, V]] and sum(w[0] for w in want) == len(ce[valid])
              and all(w[0] for w in want), str([(g[0], round(g[2], 3)) for g in got if g[0]]))

        # exam_train: B ajaninin b6_gap secimi, hikaye hikaye elle
        tr = ES.exam_train(m, d[t])
        a = d[t]["train"]
        ends = np.flatnonzero(a == eos)
        st, ln = np.concatenate([[0], ends[:-1] + 1]), ends - np.concatenate([[0], ends[:-1] + 1])
        fits = np.flatnonzero((ln > 0) & (ln + 2 <= SEQ))
        order = fits[np.argsort(ln[fits], kind="stable")]
        rng, picks = np.random.default_rng(7), []
        for L in d[t]["valid_length"][ES.exam_rows(d[t], 128)]:
            picks.append(order[int(np.clip(np.searchsorted(ln[order], L) + rng.integers(-200, 200), 0, len(order) - 1))])
        nll_s = acc_s = n_s = 0
        with torch.no_grad():
            for r in picks:
                s = [eos] + a[st[r]:st[r] + ln[r]].tolist() + [eos]
                z, y = m.logits(torch.tensor([s[:-1]])).double()[0], torch.tensor(s[1:])
                nll_s += float((z.logsumexp(-1) - z.gather(-1, y[:, None])[:, 0]).sum())
                acc_s += int((z.argmax(-1) == y).sum())
                n_s += len(y)
        check("%s exam_train: b6_gap secimi (tohum 7, boy sirasinda +-200, pencereye sigan) ve hikaye hikaye elle nll, acc, "
              "bayt (train_bytes)" % t,
              tr["n"] == n_s and abs(tr["nll"] - nll_s / n_s) < 1e-5 and abs(tr["accuracy"] - acc_s / n_s) < 1e-9
              and tr["stories"] == len(picks) == 3 and tr["bytes"] == int(d[t]["train_bytes"][picks].sum()),
              "nll %.4f (valid %.4f)" % (tr["nll"], e64["nll"]))

        al = ES.alpha_summary(m)
        A, F_ = (x.detach().numpy() for x in (m.alpha_attention, m.alpha_facts))
        f0 = 0 if m.first_turn_facts else 1                 # FIRST_TURN_FACTS=False: tur 1'in alpha_F'si None
        check("%s alpha_summary: tur basina medyan ve |alpha|'nin en buyugu (numpy)" % t,
              np.allclose(al["attention"]["median"], np.median(A, -1), atol=1e-4)
              and np.allclose(al["facts"]["median"][f0:], np.median(F_, -1)[f0:], atol=1e-4)
              and np.allclose(al["attention"]["max_abs"], np.abs(A).max(-1), atol=1e-4)
              and np.allclose(al["facts"]["max_abs"][f0:], np.abs(F_).max(-1)[f0:], atol=1e-4)
              and al["facts"]["median"][:f0] == [None] * f0 and len(al["facts"]["median"]) == 2, json.dumps(al))

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
    """fixture'da (tag basina) kisa egitilmis kucuk model (bir kez)."""
    if d["tag"] not in _MODELS:
        _MODELS[d["tag"]], _ = TR.train_seq("shared", None, None, len(d["vocab"]), steps=60, log_at=(),
                                            batches=DS.batches(d, 4, 0), model_kw=TINY)
    return _MODELS[d["tag"]]


UNIT = re.compile(r"\d|[^\W\d_]+|[^\w\s]")                          # birim: kelime, tek rakam, tek noktalama
XWORD = re.compile(r"[a-z']+")
XAX_A = re.compile(r"\b((?:[a-z']+ ){0,6}[a-z']+),? (and|or) \1\b")   # A ajaninin ifadesi (run_review/A_errors/common.py)
XAX_STORIES = ["the big dog and the big dog ran and ran and ran. we went higher and higher.",
               "a cat or a cat sat by the sun and the sunflower. red, and red.",
               "one two three four five six seven eight and two three four five six seven eight.",
               "we went higher and higher and the big dog and the big dog sat."]


def units_ref(ids, vocab, tag):
    """Bagimsiz birimler: ss4096 A'nin words'u ('##' onceki kelimeye yapisir), gpt2 cozulmus metin kucuk harf + UNIT."""
    if tag == "gpt2":
        return UNIT.findall(DS.decode(ids, vocab).lower())
    w = []
    for t in ids:
        s = vocab[t]
        if s.startswith("##") and w:
            w[-1] += s[2:]
        else:
            w.append(s)
    return w


def sentences_ref(ws):
    """A'nin sentences'i: tirnak atilir, . ! ? ile biter."""
    out, cur = [], []
    for w in ws:
        if w == '"':
            continue
        cur.append(w)
        if w in {".", "!", "?"}:
            out.append(tuple(cur))
            cur = []
    if cur:
        out.append(tuple(cur))
    return out


def xax_per_conj(units):
    """Her 'and' / 'or' icin en uzun X (<= 7 birim, [a-z']+, hemen once ve hemen sonra ayni), ortusmeli."""
    out = collections.Counter()
    for i, u in enumerate(units):
        for m in (range(7, 0, -1) if u in ("and", "or") else ()):
            x = units[i - m:i] if i >= m else None
            if x and x == units[i + 1:i + 1 + m] and all(XWORD.fullmatch(w) for w in x):
                out[(" ".join(x), u)] += 1
                break
    return out


def sample_by_hand(model, prompts, n, flags, seed, eos):
    """Tek tek tam yeniden hesap; her adimda butun satirlar tek multinomial cagrisi (ayni uretec tuketimi)."""
    gen = torch.Generator().manual_seed(seed)
    seqs, outs = [list(p) for p in prompts], [[] for _ in prompts]
    with torch.no_grad():
        for _ in range(n):
            z = torch.stack([model.logits(torch.tensor([s]))[0, -1] for s in seqs]).float()
            tok = torch.where(torch.tensor(flags), torch.multinomial(z.softmax(-1), 1, generator=gen)[:, 0], z.argmax(-1))
            for s, o, x in zip(seqs, outs, tok.tolist()):
                s.append(x)
                o.append(x)
    return [o[:o.index(eos)] if eos in o else o for o in outs]


def t_text_errors(root):
    d = {t: load(root, t) for t in DS.TAGS}
    same, cross = {}, True
    for t in DS.TAGS:
        v = d[t]["vocab"]
        tab = DS._token_table(v)
        same[t] = sum(DS._units(DS.encode(s, v), tab) == units_ref(DS.encode(s, v), v, t) for s in TRAIN + VALID)
    ss, g = (d[t]["vocab"] for t in DS.TAGS)
    cross = [DS._units(DS.encode(s, ss), DS._token_table(ss)) == DS._units(DS.encode(s, g), DS._token_table(g))
             for s in TRAIN]
    check("birimler: ss4096 = A'nin words'u ('##' birlesir), gpt2 = cozulmus metin kucuk harf + tek rakam / noktalama; "
          "TRAIN'de iki tokenizer ayni birimleri verir",
          all(k == len(TRAIN + VALID) for k in same.values()) and all(cross), "%s, ortak %d/%d" % (same, sum(cross), len(cross)))

    for t in DS.TAGS:
        v = d[t]["vocab"]
        eos = v.index(DS.EOS_TOKEN)
        ids = [DS.encode(s, v) for s in XAX_STORIES]
        stream = np.array([x for s in ids for x in s + [eos]], dtype=np.uint16)
        want_words = {w for s in ids for w in units_ref(s, v, t) if w.isalpha()}
        want_xax = sum((xax_per_conj(units_ref(s, v, t)) for s in ids), collections.Counter())
        got = []
        for chunk in (DS._CHUNK, 5):
            saved, DS._CHUNK = DS._CHUNK, chunk
            try:
                got.append(DS._text_reference(v, stream))
            finally:
                DS._CHUNK = saved
        check("%s train referansi: alfabetik kelime kumesi ve her 'and' / 'or' icin en uzun 'X and X' (<= 7 birim, "
              "ortusmeli) elle; parca boyu (5 token, eos sinirinda) sonucu degistirmez" % t,
              all(w == want_words and dict(x) == dict(want_xax) for w, x in got)
              and dict(want_xax) == {("the big dog", "and"): 2, ("ran", "and"): 2, ("higher", "and"): 2, ("a cat", "or"): 1,
                                     ("two three four five six seven eight", "and"): 1}, str(dict(got[0][1])))

    cases = [u.split() for u in ("dug and dug and dug", "the big dog and the big dog ran", "a b c d e f g h and b c d e f g h",
                                 "the sun and the sunflower", "red , and red", "x and y", "i can do this and i can do this !",
                                 "cats and dogs and cats and dogs .", "and and and", "or or")]
    cases += [units_ref(DS.encode(s, g), g, "gpt2") for s in XAX_STORIES]
    check("_xax (uretilen metin) = A'nin ifadesinin finditer'i ' '.join(birimler) uzerinde: soldan, ortusmeden, en uzun X",
          all(ES._xax(u) == [(x.group(1), x.group(2)) for x in XAX_A.finditer(" ".join(u))] for u in cases),
          str([ES._xax(u) for u in cases[:3]]))

    ok = []
    for t in DS.TAGS:
        v, dt = d[t]["vocab"], d[t]
        eos = v.index(DS.EOS_TOKEN)
        a = dt["valid"]
        ends = np.flatnonzero(a == eos)
        skip = set(dt["valid_start"][ES.exam_rows(dt)[:ES.STORY_CONTINUATIONS]].tolist())
        want = {" ".join(s) for b, e in zip(np.concatenate([[0], ends[:-1] + 1]).tolist(), ends.tolist()) if b not in skip
                for s in sentences_ref(units_ref(a[b:e].tolist(), v, t)) if len(s) >= 5}
        ok.append(ES._stock_reference(dt, ES.STORY_CONTINUATIONS) == want and len(want) > 0)
    check("kalip cumle referansi: valid'in (devami uretilen sinav hikayeleri haric) >= 5 birimlik cumleleri elle",
          all(ok), str(ok))

    v = g
    tab, eos = DS._token_table(v), v.index(DS.EOS_TOKEN)
    p1 = [eos] + DS.encode('Tom saw a big dog. "Can we play now?" said Tom.', v)
    g1 = DS.encode(' The dog ran to the park. Tom saw a big dog. "Can we play now?" asked Sue. The zorblax and zorblax ran '
                   'and ran. The dog ran to the park.', v)
    p2, g2 = [eos] + DS.encode("Sue had a hat.", v), DS.encode(" Sue smiled.", v)
    alpha = [w for w in units_ref(g1, v, "gpt2") + units_ref(g2, v, "gpt2") if w.isalpha()]
    r = ES._count([p1, p2], [g1, g2], [True, False], tab, set(alpha) - {"zorblax"}, {("ran", "and"): ES.XAX_LEGIT_MIN},
                  {"the dog ran to the park ."})
    T = len(g1) + len(g2)
    g8 = [tuple(g1[i:i + 8]) for i in range(len(g1) - 7)]
    want = dict(stories=2, tokens=T, ended=0.5, loop=(len(set(g8)) < len(g8)) / 2, sentence_repeat=0.5,
                sentence_repeat_per1k=round(3000 / T, 4), quote_repeat=0.5, quote_repeat_per1k=round(1000 / T, 4), xax=0.5,
                xax_per1k=round(1000 / T, 4), xax_legit_per1k=round(1000 / T, 4), stock=0.4, sentences=5, nonword=0.5,
                nonword_per1k=round(2000 / len(alpha), 4), words=len(alpha),
                examples=dict(xax=["zorblax and zorblax"], nonword=["zorblax", "zorblax"]))
    check("_count elle: cumle tekrari (istemden 2, devamdan 1), soz tekrari, legit olmayan 'zorblax and zorblax' (legit "
          "'ran and ran' ayri), kalip cumle 2/5, uydurma kelime, 8'li dongu, hikaye paylari ve 1000 token basina",
          r == want, str({k: (r[k], want[k]) for k in want if r.get(k) != want[k]}))

    ok = []
    for t in DS.TAGS:                                                  # istem kelime ortasinda kesilir (story_prompts L // 2)
        v = d[t]["vocab"]
        tab, eos = DS._token_table(v), v.index(DS.EOS_TOKEN)
        ids = DS.encode("the cat sat on the mountain. it was", v)
        piece = lambda i: v[ids[i]].startswith("##") if t == "ss4096" else (
            DS.decode(ids[i:i + 1], v)[:1].isalpha() and DS.decode(ids[i - 1:i], v)[-1:].isalpha())
        k = max(i for i in range(1, len(DS.encode("the cat sat on the mountain", v))) if piece(i))   # 'mountain' icinde
        r3 = ES._count([[eos] + ids[:k]], [ids[k:]], [True], tab, {"mountain", "it", "was"}, {}, set())
        ok.append(r3["nonword"] == 0 and r3["words"] == 3)
    check("_count: istem kelime ortasinda biterse o kelime devamin (parca 'tain' uydurma sayilmaz), iki tokenizer'da",
          all(ok), str(ok))

    ds = d["ss4096"]
    v = ds["vocab"]
    eos = v.index(DS.EOS_TOKEN)
    prompts = [[eos] + DS.encode(TRAIN[i][:30], v) for i in (0, 1, 7)]
    flags = [False] * 3 + [True] * 3
    model, n, V = m_ss(ds), 12, len(v)
    prompts.append([eos])                                             # tek token'li istem (onbellekte istem gecisi yok)
    flags = [False] * 4 + [True] * 4
    got, ended = ES._continue(model, prompts + prompts, n, flags, 3, eos, V)
    again, _ = ES._continue(model, prompts + prompts, n, flags, 3, eos, V)
    alone, _ = ES._continue(model, [[eos]], n, [False], 3, eos, V)
    check("_continue: acgozlu satirlar = texts (onbellekli; tek token'li istem dahil); ornekleme satirlari = tek tek tam "
          "hesapla ayni uretec; ayni tohum ayni devam",
          got[:4] == [r_["ids"] for r_ in ES.texts(model, ds, prompts, n)] and got == again and alone[0] == got[3]
          and got == sample_by_hand(model, prompts + prompts, n, flags, 3, eos) and got[4:] != got[:4]
          and ended == [len(x) < n for x in got], str(got[4][:6]))

    budget, ES.CACHE_BUDGET = ES.CACHE_BUDGET, 1                      # onbellekli parca satir basina bir
    try:
        split, _ = ES._continue(model, prompts, n, [False] * 4, 3, eos, V)
    finally:
        ES.CACHE_BUDGET = budget
    check("_continue: onbellekli uretim CACHE_BUDGET'a gore satir parcalarina bolunur; acgozlu devam tek batch'le ayni",
          split == got[:4], str(split[0][:6]))

    ok = []
    for t in DS.TAGS:
        dt = d[t]
        model = m_ss(dt)
        eos = dt["vocab"].index(DS.EOS_TOKEN)
        r = ES.count_text_errors(model, dt, 300, real=True)
        prompts, reals = ES.story_prompts(dt, ES.exam_rows(dt)[:ES.STORY_CONTINUATIONS])
        k, n = len(prompts), min(300, SEQ - max(len(p) for p in prompts))
        gens, ended = ES._continue(model, prompts + prompts, n, [False] * k + [True] * k, 0, eos, len(dt["vocab"]))
        ref = (DS._token_table(dt["vocab"]), dt["text_reference"]["words"], dt["text_reference"]["xax"],
               ES._stock_reference(dt, ES.STORY_CONTINUATIONS))
        ok.append(r == dict(greedy=ES._count(prompts, gens[:k], ended[:k], *ref),
                            sampled=ES._count(prompts, gens[k:], ended[k:], *ref),
                            real=ES._count(prompts, reals, [True] * k, *ref))
                  and r["greedy"]["stories"] == 3 and r["real"]["loop"] == 0 and r["real"]["sentence_repeat"] == 0
                  and max(len(x) for x in gens) <= n and ES.count_text_errors(None, dt, 300, real=True) == dict(real=r["real"]))
    check("count_text_errors: sabit alt kumenin ilk STORY_CONTINUATIONS hikayesi, istem + devam pencereye sigar; greedy / "
          "sampled (tohum 0) / real = _count(_continue); model None yalniz real", all(ok), str(ok))


def t_correction_break(root):
    """Duzeltme molasi: _repeats birebir ve 8'li tekrari yakalar, dogal cumleye dokunmaz; correction_break satirlari istem
    + metin + hedef, mask yalniz hedefte; train_seq molayla kosar, kaydi tutar ve mola satirlari guncellemeyi degistirir."""
    d = load(root, "gpt2")
    v = d["vocab"]
    tab = DS._token_table(v)
    enc = lambda s: DS.encode(s, v)
    ctx = enc("Tom saw a big red dog in the park. The dog ran away.")
    check("correction_break: _repeats birebir cumle ve 8'li ortak oneki yakalar, yeni cumleye dokunmaz",
          ES._repeats(ctx, enc(" Tom saw a big red dog in the park."), tab)
          and ES._repeats(ctx, enc(" Tom saw a big red dog in the yard."), tab)
          and not ES._repeats(ctx, enc(" Sue came home and ate a cake."), tab))
    model, _ = TR.train_seq("shared", None, None, len(v), steps=5, log_at=(), batches=DS.batches(d, 4, 0), model_kw=TINY)
    ids, mask, rec = ES.correction_break(d, prompts=8)(2, model)
    eos = v.index(DS.EOS_TOKEN)
    ok = all(m.any() and ids[i, 0] == eos and not m[:2].any() for i, m in enumerate(mask))
    check("correction_break: satir <eos> + istem + metin + hedef, mask yalniz hedef cumlede; kayit tutarli",
          ok and rec["stories"] == 8 and rec["corrected"] == len(ids) <= rec["entered"] <= 8
          and rec["corrected"] + rec["forced"] == rec["entered"], str(rec))
    fixed = (torch.tensor([[eos] + enc("Tom ran. Sue came home.")]), None)
    fixed = (fixed[0], torch.zeros_like(fixed[0], dtype=torch.bool))
    fixed[1][0, -4:] = True
    calls = []
    stub = lambda step, m: (calls.append(step) or (fixed[0], fixed[1], dict(step=step)))
    runs = [TR.train_seq("shared", None, None, len(v), steps=6, log_at=(), batches=DS.batches(d, 4, 0), model_kw=TINY,
                         **kw)[0] for kw in ({}, dict(correction_break=stub, break_every=2, break_weight=1.0))]
    differ = any(not torch.equal(a, b) for a, b in zip(runs[0].state_dict().values(), runs[1].state_dict().values()))
    check("correction_break: train_seq her break_every adimda cagirir (adim > 0), kaydi tutar, guncelleme degisir",
          calls == [2, 4] and [r["step"] for r in runs[1].correction_break] == [2, 4] and differ, "cagri %s" % calls)


def t_speed(root):
    import colab_simplestories as C
    d = load(root, "ss4096")
    work = dict(targets=0, keys=0)
    f = C._counting(DS.batches(d, 4, 0, bucket=2), work)
    targets = keys = 0
    for s in (0, 3, 5):
        for row in f(s)[1].tolist():
            n = sum(row) - 1                                   # hedef: pencere - 1
            targets += n
            keys += sum(range(1, n + 1))                       # hedef t (1..n) nedensel attention'da t konum gorur
    check("_counting: cekilen batch'lerin hedef token'i ve attention anahtari elle (bucket'li genislik dolgu sayilmaz)",
          work == dict(targets=targets, keys=keys) and targets > 0, str(work))

    V, D_, units, turns = 4096, 384, 1024, 6
    bm = M.BlockModel(V, d=D_, turns=turns, layers=3, units=units, heads=4, fact_activation="swiglu", first_turn_facts=True)
    N, Ld = C._model_flops(bm)
    bm2 = M.BlockModel(V, d=D_, turns=turns, layers=3, units=units, heads=4, fact_activation="swiglu", shared_facts=False, first_turn_facts=True)
    check("_model_flops: 3x2 (d 384, FactUnits 1024, 6 tur, 4 head) N = 6 (4 d^2 + 3 d units) + V d; ileri FLOP/token "
          "2 N + 4 L d x 169,9 anahtar = 25,95 M; ayri FactUnits ayni",
          (N, Ld) == (turns * (4 * D_ * D_ + 3 * D_ * units) + V * D_, turns * D_) and round((2 * N + 4 * Ld * 169.9) / 1e6, 2)
          == 25.95 and 2 * N == 24379392 and C._model_flops(bm2) == (N, Ld), "N %d L d %d" % (N, Ld))

    check("mfu: (6 N hedef + 12 L d anahtar) / sure / tepe; tepe tablosu cihaz adindan (L40S, T4'te bf16 yok -> None)",
          C._mfu((10, 3), dict(targets=100, keys=50), 2.0, 1e3) == 3.9 and C._mfu(None, work, 1.0, 1e3) is None
          and C._mfu((10, 3), work, 1.0, None) is None and C._peak_for("NVIDIA L4", "bf16") == 121e12
          and C._peak_for("NVIDIA L40S", "bf16") is None and C._peak_for("NVIDIA A100-SXM4-40GB", "tf32") == 156e12
          and C._peak_for("NVIDIA H100 PCIe", "bf16") == 756e12 and C._peak_for("NVIDIA H100 80GB HBM3", "bf16") == 989e12
          and C._peak_for("Tesla T4", "bf16") is None and C._peak_for("Tesla T4", "fp32") == 8.1e12)

    al = ES.alpha_summary(M.BlockModel(64, d=16, turns=4, layers=2, units=8, first_turn_facts=False, last_facts_alpha_init=1.0))
    line = C._alpha_text(al)
    check("alpha_summary: FIRST_TURN_FACTS=False'ta tur 1'in alpha_F'si None (kullanilmiyor), satirda '-'; en buyuk onu saymaz",
          al["facts"]["median"][0] is None and al["facts"]["max_abs"][0] is None and al["attention"]["median"][0] is not None
          and "aF -/" in line and "(1.00)" in line, line)


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

    draw = DS.batches(d, 4, 0)
    implied = [int(draw(e["step"])[1][:, 1:].sum()) / e["tokens_per_sec"] for e in ex[1:]]    # every 1: aralikta tek batch
    log = open(out + "/log.txt", encoding="utf-8").read()
    check("colab_simplestories, yeni olculer: tokens_per_sec (step_ms olan her sinavda; aralikta cekilen batch'in hedefi / "
          "egitim suresi), save_secs (aralikta yedek yazimi; adim 0'da yedek yok) ve egitim + yedek = aralik; CPU'da mfu None; "
          "her sinavda exam_train, alpha_summary (2 tur), nll_by_frequency; text_errors her 4. sinavda (adim 0) ve sonda "
          "gercekle; gunlukte referans, token/sn, yedek ve metin",
          "tokens_per_sec" not in ex[0] and "save_secs" not in ex[0] and all(e["mfu"] is None for e in ex[1:])
          and all(x > 0 and abs(x + e["save_secs"] - e["step_ms"] / 1000) < 2e-3 for x, e in zip(implied, ex[1:]))
          and ex[1]["save_secs"] == 0 and all(e["save_secs"] > 0 for e in ex[2:]) and " yedek " in log
          and all("exam_train" in e and len(e["alpha_summary"]["facts"]["median"]) == 2 and len(e["nll_by_frequency"]) == 4
                  for e in ex)
          and [e["step"] for e in ex if "text_errors" in e] == [0] and set(ex[0]["text_errors"]) == {"greedy", "sampled"}
          and set(fin["text_errors"]) == {"greedy", "sampled", "real"} and fin["exam_train"]["stories"] == 3
          and len(fin["alpha_summary"]["attention"]["max_abs"]) == 2 and fin["perf"]["tokens_per_sec"] > 0
          and fin["perf"]["mfu"] is None and "referans" in log and "token/sn" in log and "       metin (3 hikaye)" in log
          and "SON metin (3 hikaye)" in log and "SON train" in log,
          "%s | %s" % ([round(x, 4) for x in implied], [e.get("step_ms") for e in ex]))

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
    new_fields = ("tokens_per_sec", "mfu", "save_secs", "exam_train", "alpha_summary", "nll_by_frequency", "text_errors")
    json.dump([{k: v for k, v in e.items() if k not in new_fields}
               for e in json.load(open(out + "/exams.json", encoding="utf-8"))],
              open(out + "/exams.json", "w", encoding="utf-8"), indent=1)   # 29 Eylul oncesi exams.json: yeni olculer yok
    if os.path.exists(out + "/checkpoint_t000003.pt"):
        os.remove(out + "/checkpoint_t000003.pt")
    run5 = C.start("TEST", d, out, steps=3, every=1, device="cpu", batch_size=4, model_kw=TINY, save_every=1, resume=True)
    run5["thread"].join(600)
    check("colab_simplestories, output_link: config'te acik (TINY'de False); True'da q, u sinavda, gunlukte ve "
          "final.json'da; output_link'siz eski config ve yeni olcusuz eski exams.json surdurulur",
          cfg["model_kw"]["output_link"] is False and run4["done"] and not run4["error"]
          and all("link_q" in e and "link_u" in e for e in ex4) and any("bag q" in l for l in run4["lines"])
          and "link_q" in fin4 and fin4["link_u"] >= 0 and run5["done"] and not run5["error"],
          str(run4["error"] or run5["error"] or run4["lines"][-2:]))

    # tarif ayarlari config'e ve train_seq'e; ogrenilen olcek sinava ve gunluge
    seen, real = [], TR.train_seq
    TR.train_seq = lambda *a, **k: (seen.append(k.get("matmul_precision")), real(*a, **k))[1]
    try:
        run6 = C.start("TEST6", d, tmp + "/m", steps=1, every=1, device="cpu", batch_size=4, model_kw=TINY, save_every=1,
                       matmul_precision="fp32")
        run6["thread"].join(600)
        cfg6 = json.load(open(tmp + "/m/config.json"))
        try:
            C.start("TEST6", d, tmp + "/m", steps=1, every=1, device="cpu", batch_size=4, model_kw=TINY, save_every=1,
                    resume=True)                                  # varsayilan bf16: ayar farkli
            refused6 = False
        except RuntimeError:
            refused6 = True
    finally:
        TR.train_seq = real
    log = open(out + "/log.txt", encoding="utf-8").read()
    check("colab_simplestories, tarif: matmul_precision config'e yazilir (varsayilan bf16, verilen fp32) ve train_seq'e "
          "iletilir, farkliysa surdurme reddedilir; muon_tangent / coherence_power config'te sabit (True / 1,0); loss_chunk model_kw'de "
          "(varsayilan 4096) ve ilk satirda; ogrenilen olcek sinavda (output_scale, adim 0'da scale_for(V)), gunlukte, sonda",
          cfg["matmul_precision"] == "bf16" and cfg["muon_tangent"] is True and cfg["coherence_power"] == 1.0
          and cfg6["matmul_precision"] == "fp32" and cfg6["muon_tangent"] is True and cfg6["coherence_power"] == 1.0
          and seen == ["fp32"] and run6["done"] and not run6["error"] and refused6
          and cfg["model_kw"]["loss_chunk"] == M.LOSS_CHUNK == 4096 and "loss_chunk 4096" in log
          and abs(ex[0]["output_scale"] - M.scale_for(len(d["vocab"]))) < 1e-4 and " | olcek " in log and "output_scale" in fin,
          "%s %s" % (seen, run6["error"] or ""))

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
        t_text_reference(root)
        t_bits_per_byte(root)
        t_batches(root)
        t_split(root)
        t_exam(root)
        t_text_errors(root)
        t_correction_break(root)
        t_speed(root)
        t_colab(root)
    finally:
        shutil.rmtree(root)
        shutil.rmtree(src)
    print("\n%d GECTI   %d KALDI" % (sum(RESULTS), len(RESULTS) - sum(RESULTS)))
    sys.exit(0 if all(RESULTS) else 1)
