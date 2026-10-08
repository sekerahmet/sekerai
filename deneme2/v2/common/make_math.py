"""make_math -- DeepMind Mathematics Dataset (Saxton ve ark. 2019; belge 96) -> V2 disk bicimi, karakter duzeyi (ASCII
sozlugu).  Kullanici, 9 Ekim: "matematik bizim için ilginç bir test olabilir"; "ama o kadar büyük sözlüğe gerek var mı ?";
"bir ajana ver hazırlasın bence bizim yapıya göre".

Girdi (indirilmis ve acilmis arsiv, mathematics_dataset-v1.0/): train-easy/, train-medium/, train-hard/, interpolate/,
extrapolate/ altinda <modul>.txt; satir sirasiyla soru, cevap (TFDS okuyucusu gibi: bos satir atilir, lines[::2] soru,
lines[1::2] cevap).  Karakter: yazdirilabilir ASCII (kod - 32 -> 0..94), EOS 95, END 96 (data.ASCII).

Belge = soru + cevap.  Cumleler (--question one | split): one -> soru tek cumle (Z_1), cevap tek cumle; split -> soru ". " /
"? " sonra buyuk harfte cumlelere (bosluk onceki cumlede), cevap tek cumle.  Ham akis belge basina [soru | cevap | EOS].
Cikti <out>/ (stream_root = data_dir):
    vocab.json                       {"name": "ascii"} (data.data_vocab)
    ascii/{train,valid,extrapolate}.npy   uint8 akis;  ascii/{split}_bytes.npy (belge bayti; valid_bytes = sinav),
                                     ascii/{split}_answer_bytes.npy (cevap bayti; --answer_only sinavinin paydasi)
    {split}_sentence_offsets.npy, _story_offsets.npy, _boundaries.json, _doc_module.npy (int16, math_meta.json modules)
    train_pack_plan_e1.npz, exam_pack_plan.npz (valid'den modul basina --exam_per_module, tohum 0), reading_prompts.json
    math_meta.json   moduller, belge sayilari, sinav temizligi (soru / soru + cevap tam eslesme, train'de), en sik cevap tabani
valid = interpolate, extrapolate ayri bolum (yalniz sinav: diag/math_exam.py).  train = train-easy + medium + hard (README:
"Mixing these uniformly reproduces the paper's results"), --train_per_file N ile dosya basina ilk N cift (0: hepsi).

    python make_math.py --src <mathematics_dataset-v1.0> --out <klasor> [--question one|split] [--train_per_file N]
                        [--modules a,b] [--exam_per_module 100]
"""
import torch  # noqa: F401,I001  (Windows: torch once)

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D  # noqa: E402

TRAIN_DIRS = ("train-easy", "train-medium", "train-hard")
TEST_DIRS = dict(valid="interpolate", extrapolate="extrapolate")
SPLIT_RE = re.compile(r"(?<=[.?!] )(?=[A-Z])")      # --question split: ". X" / "? X" (bosluk onceki cumlede; 0.5 kesilmez)
READINGS = 10                                       # okuma istemi: valid'den ilk 10 modulun ilk belgesi


def read_pairs(path, limit=0):
    """<modul>.txt -> [(soru, cevap)] (bos satir atilir; satir sonu atilir, bosluk korunur).  limit > 0: ilk limit cift."""
    lines = [x for x in open(path, encoding="ascii", newline="").read().split("\n") if x]
    lines = [x[:-1] if x.endswith("\r") else x for x in lines]
    assert len(lines) % 2 == 0, "%s: tek sayida satir" % path
    pairs = list(zip(lines[::2], lines[1::2]))
    return pairs[:limit] if limit else pairs


def sentences(q, a, mode):
    """Belge -> cumle metinleri (bitistirilmesi = q + a)."""
    qs = [q] if mode == "one" else [s for s in SPLIT_RE.split(q) if s]
    assert "".join(qs) == q
    return qs + [a]


def encode(text):
    """Metin -> ASCII sozlugu (uint8; kod - 32).  Yazdirilabilir disi karakter DURUR."""
    x = np.frombuffer(text.encode("ascii"), np.uint8).astype(np.int16) - 32
    assert len(x) == 0 or (x.min() >= 0 and x.max() < 95), "yazdirilabilir ASCII disi: %r" % text
    return x.astype(np.uint8)


def write_split(out, split, docs, mode):
    """docs [(modul_no, soru, cevap)] -> akis + sinir dosyalari.  -> boundaries meta."""
    sd = os.path.join(out, D.ASCII.stream_dir)
    os.makedirs(sd, exist_ok=True)
    parts, sents, story, nbytes, abytes = [], [], [0], [], []
    at = 0
    for _, q, a in docs:
        for s in sentences(q, a, mode):
            sents.append((at, at + len(s)))
            at += len(s)
        parts += [encode(q + a), np.array([D.ASCII.eos], np.uint8)]
        at += 1
        story.append(len(sents))
        nbytes.append(len(q) + len(a))
        abytes.append(len(a))
    x = np.concatenate(parts) if parts else np.zeros(0, np.uint8)
    path = os.path.join(sd, split + ".npy")
    np.save(path, x)
    np.save(os.path.join(sd, split + "_bytes.npy"), np.array(nbytes, np.int64))
    np.save(os.path.join(sd, split + "_answer_bytes.npy"), np.array(abytes, np.int64))   # --answer_only sinavi
    sents = np.array(sents, np.int64).reshape(-1, 2)
    np.save(os.path.join(out, split + "_sentence_offsets.npy"), sents)
    np.save(os.path.join(out, split + "_story_offsets.npy"), np.array(story, np.int64))
    np.save(os.path.join(out, split + "_doc_module.npy"), np.array([m for m, _, _ in docs], np.int16))
    L = sents[:, 1] - sents[:, 0]
    meta = dict(split=split, stream=path, stream_sha256=hashlib.sha256(open(path, "rb").read()).hexdigest(),
                rule="make_math: belge = soru + cevap, soru %s (belge 96)" % mode, profile="math", question=mode,
                stories=len(docs), sentences=len(sents), tokens_in_sentences=int(L.sum()),
                max_sentence_tokens=int(L.max()) if len(L) else 0, created=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(meta, open(os.path.join(out, split + "_boundaries.json"), "w", encoding="utf-8"), indent=1)
    return meta


def _key(s):
    """Metin -> 64 bit ozet (tam eslesme sayimi; carpisma olasiligi ~N^2 / 2^65, HESAP)."""
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little", signed=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, help="mathematics_dataset-v1.0 klasoru")
    ap.add_argument("--out", required=True)
    ap.add_argument("--question", default="one", choices=("one", "split"))
    ap.add_argument("--train_per_file", type=int, default=0, help="train dosyasi basina ilk N cift (0: hepsi)")
    ap.add_argument("--modules", default="", help="virgulle modul listesi (bos: train-easy'deki hepsi)")
    ap.add_argument("--exam_per_module", type=int, default=100, help="exam_pack_plan: valid'den modul basina belge")
    args = ap.parse_args(argv)
    t0 = time.time()
    log = lambda m: print("[%6.1f sn] %s" % (time.time() - t0, m), flush=True)  # noqa: E731
    mods = args.modules.split(",") if args.modules else sorted(
        f[:-4] for f in os.listdir(os.path.join(args.src, TRAIN_DIRS[0])) if f.endswith(".txt"))
    mid = {m: i for i, m in enumerate(mods)}
    os.makedirs(args.out, exist_ok=True)
    json.dump(dict(name=D.ASCII.name), open(os.path.join(args.out, "vocab.json"), "w"))
    train, tq, tqa, freq = [], {}, {}, {}
    for m in mods:
        keys_q, keys_qa, c = [], [], Counter()
        for d in TRAIN_DIRS:
            p = os.path.join(args.src, d, m + ".txt")
            for q, a in read_pairs(p, args.train_per_file) if os.path.exists(p) else []:
                train.append((mid[m], q, a))
                keys_q.append(_key(q))
                keys_qa.append(_key(q + "\n" + a))
                c[a] += 1
        tq[m], tqa[m] = np.unique(np.array(keys_q, np.int64)), np.unique(np.array(keys_qa, np.int64))
        freq[m] = c.most_common(1)[0] if c else ("", 0)
    log("train: %d belge, %d modul" % (len(train), len(mods)))
    metas = {"train": write_split(args.out, "train", train, args.question)}
    clean = {}
    for split, d in TEST_DIRS.items():
        docs, per = [], {}
        for m in sorted(f[:-4] for f in os.listdir(os.path.join(args.src, d)) if f.endswith(".txt")) \
                if os.path.isdir(os.path.join(args.src, d)) else []:
            base = max((t for t in tq if m == t or m.startswith(t + "_")), key=len, default=m)   # extrapolate: _big vb.
            if m not in mid:
                mid[m] = len(mods)
                mods.append(m)
            pairs = read_pairs(os.path.join(args.src, d, m + ".txt"))
            docs += [(mid[m], q, a) for q, a in pairs]
            hit_q = int(np.isin([_key(q) for q, _ in pairs], tq.get(base, np.zeros(0, np.int64))).sum())
            hit_qa = int(np.isin([_key(q + "\n" + a) for q, a in pairs], tqa.get(base, np.zeros(0, np.int64))).sum())
            top = freq.get(base, ("", 0))[0]
            per[m] = dict(train_module=base if base in tq else None, docs=len(pairs), question_in_train=hit_q,
                          pair_in_train=hit_qa, most_frequent_answer=top,
                          most_frequent_answer_acc=round(sum(a == top for _, a in pairs) / max(1, len(pairs)), 4))
        metas[split] = write_split(args.out, split, docs, args.question)
        clean[split] = per
        log("%s: %d belge, %d modul; soru train'de %d, cift train'de %d" % (
            split, len(docs), len(per), sum(v["question_in_train"] for v in per.values()),
            sum(v["pair_in_train"] for v in per.values())))
    longest = max(m["max_sentence_tokens"] for m in metas.values())
    for split, m in metas.items():
        json.dump(dict(m, max_sentence_tokens_all=longest), open(os.path.join(args.out, split + "_boundaries.json"), "w",
                                                                 encoding="utf-8"), indent=1)
    tr = D.TokenStories(args.out, args.out, "train")
    ro, rs = D.pack_plan(tr.lengths(), D.ROW_LEN, 0, 1)
    np.savez(os.path.join(args.out, "train_pack_plan_e1.npz"), row_offsets=ro, row_stories=rs, seed=0, epoch=1,
             row_len=D.ROW_LEN)
    va = D.TokenStories(args.out, args.out, "valid")
    vm = np.load(os.path.join(args.out, "valid_doc_module.npy"))
    rng = np.random.default_rng(0)
    pick = np.sort(np.concatenate([rng.permutation(np.flatnonzero(vm == k))[:args.exam_per_module]
                                   for k in np.unique(vm)])) if len(vm) else np.zeros(0, np.int64)
    ro, rs = D.pack_plan(va.lengths()[pick], D.ROW_LEN, None)
    np.savez(os.path.join(args.out, "exam_pack_plan.npz"), row_offsets=ro, row_stories=pick[rs].astype(np.int32), seed=-1,
             epoch=0, row_len=D.ROW_LEN, exam_set_sha256=hashlib.sha256(np.ascontiguousarray(pick)).hexdigest())
    first = [int(np.flatnonzero(vm == k)[0]) for k in np.unique(vm)[:READINGS]]
    json.dump(dict(prompts=[dict(label="%s %d" % (mods[int(vm[i])], i), story=i,
                                 sentences=len(va.sentences(i)) - 1, decode="greedy") for i in first]),
              open(os.path.join(args.out, "reading_prompts.json"), "w"), indent=1)
    json.dump(dict(source=os.path.abspath(args.src), question=args.question, train_per_file=args.train_per_file,
                   modules=mods, train_docs=len(train),
                   train_most_frequent_answer={m: dict(answer=a, count=n) for m, (a, n) in freq.items()},
                   cleanliness=clean, exam_per_module=args.exam_per_module, exam_docs=int(len(pick)),
                   note="question_in_train / pair_in_train: test sorusu (sorusu + cevabi) train'de birebir (64 bit ozet); "
                        "most_frequent_answer_acc: modulun train'deki en sik cevabi her soruya verilince tam eslesme"),
              open(os.path.join(args.out, "math_meta.json"), "w", encoding="utf-8"), indent=1)
    log("BITTI: %s (train %d satir plan, sinav %d belge)" % (args.out, len(tr.lengths()), len(pick)))


if __name__ == "__main__":
    main()
