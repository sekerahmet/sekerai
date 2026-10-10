"""make_chunks -- deneme (kullanici, 10 Ekim: "32 hazırla ama riskler neler onu ölç"): cumle siniri yerine SABIT parca.  Ayni ham
akis (gpt2/<split>.npy), ayni hikayeler; her hikaye C token'lik parcalara bolunur (son parca kisa), Model Z'de her parcanin sonunda
Z (C + 1 konum).  Satira sigmayan train parcasi (1 + T + parca sayisi > row_len) tam parca sinirinda bolunur, onceki parca(lar)
devam ediyor (story_continues; son parcaya EOS hedefi konmaz, data.build_batch).  valid bolunmez (sinav hikayeleri sigar; hikaye
sirasi = valid_bytes).  Cikti <out>/: <split>_sentence_offsets.npy, _story_offsets.npy, train_story_continues.npy,
<split>_boundaries.json, train_pack_plan_e1.npz, exam_pack_plan.npz (ayni 1.000 sinav hikayesi), train_token_counts.npy ve
reading_prompts.json (kopya).  Kiyas: sinavda END / EOS haric gercek token nll'i (END yeri ve sayisi degisir).

    python make_chunks.py <kaynak veri klasoru (fineweb_edu_s000)> <cikti klasoru> [--chunk 31]
"""
import torch  # noqa: F401,I001  (Windows: torch once)

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as D  # noqa: E402


def chunk_offsets(sent, story, continues, C, row_len, split_parts):
    """(cumle offsets, hikaye offsets, continues) -> ayni hikayelerin C'lik parca offsets'i, yeni hikaye offsets, continues."""
    a, b = sent[story[:-1], 0], sent[story[1:] - 1, 1]
    per_row = (row_len - 1) // (C + 1)                                   # tam parcayla satira sigan parca sayisi
    offs, counts, cont = [], [], []
    for i, (s, t) in enumerate(zip(a.tolist(), b.tolist())):
        st = np.arange(s, t, C, dtype=np.int64)
        piece = np.stack([st, np.minimum(st + C, t)], 1)
        n = len(st)
        if split_parts and 1 + (t - s) + n > row_len:
            cuts = list(range(0, n, per_row)) + [n]
        else:
            cuts = [0, n]
        for j in range(len(cuts) - 1):
            offs.append(piece[cuts[j]:cuts[j + 1]])
            counts.append(cuts[j + 1] - cuts[j])
            last = j == len(cuts) - 2
            cont.append(bool(continues[i]) if (last and continues is not None) else not last)
    return np.concatenate(offs), np.r_[0, np.cumsum(counts)].astype(np.int64), np.array(cont, dtype=bool)


def main(src, out, C):
    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    for split in ("valid", "train"):
        sent = np.load(os.path.join(src, split + "_sentence_offsets.npy"))
        story = np.load(os.path.join(src, split + "_story_offsets.npy"))
        cpath = os.path.join(src, split + "_story_continues.npy")
        cont_in = np.load(cpath) if os.path.exists(cpath) else None
        offs, so, cont = chunk_offsets(sent, story, cont_in, C, D.ROW_LEN, split == "train")
        tok_in = int((sent[:, 1] - sent[:, 0]).sum())
        assert int((offs[:, 1] - offs[:, 0]).sum()) == tok_in, "token sayisi degisti"
        inner = np.ones(len(offs) - 1, dtype=bool)
        inner[so[1:-1] - 1] = False                                      # hikaye-parca icinde parcalar bitisik
        assert (offs[1:, 0][inner] == offs[:-1, 1][inner]).all() and (offs[:, 1] - offs[:, 0]).max() <= C, "parca bicimi"
        np.save(os.path.join(out, split + "_sentence_offsets.npy"), offs)
        np.save(os.path.join(out, split + "_story_offsets.npy"), so)
        if split == "train":
            np.save(os.path.join(out, split + "_story_continues.npy"), cont)
        meta = json.load(open(os.path.join(src, split + "_boundaries.json"), encoding="utf-8"))
        meta.update(rule="deneme make_chunks: sabit %d token'lik parca (cumle siniri yok); kaynak %s" % (C, src),
                    chunk=C, stories=len(so) - 1, sentences=len(offs), tokens_in_sentences=tok_in, max_sentence_tokens=C,
                    max_sentence_tokens_all=C, split_parts=int(split == "train"), created=time.strftime("%Y-%m-%d %H:%M:%S"))
        json.dump(meta, open(os.path.join(out, split + "_boundaries.json"), "w", encoding="utf-8"), indent=1)
        print("%s: %d hikaye -> %d parca-hikaye, %d parca, token %d (ayni) | %.0f sn" % (
            split, len(story) - 1, len(so) - 1, len(offs), tok_in, time.time() - t0), flush=True)
    train = D.TokenStories(src, out, "train")
    L = train.lengths()
    assert L.max() <= D.ROW_LEN, "satira sigmayan parca kaldi: %d" % L.max()
    ro, rs = D.pack_plan(L, D.ROW_LEN, 0, 1)
    np.savez(os.path.join(out, "train_pack_plan_e1.npz"), row_offsets=ro, row_stories=rs, seed=0, epoch=1, row_len=D.ROW_LEN)
    print("train paket plani: %d satir, %d adim, doluluk %.3f | %.0f sn" % (
        len(ro) - 1, -(-(len(ro) - 1) // D.BATCH_ROWS), L.sum() / ((len(ro) - 1) * D.ROW_LEN), time.time() - t0), flush=True)
    valid = D.TokenStories(src, out, "valid")
    E = np.load(os.path.join(src, "exam_stories.npy"))
    ex_src = np.load(os.path.join(src, "exam_pack_plan.npz"))
    pick = np.sort(E[np.random.default_rng(0).permutation(len(E))[:D.EXAM_STORIES]])
    assert np.array_equal(np.sort(ex_src["row_stories"]), pick), "sinav hikayeleri kaynaktakiyle ayni degil"
    assert valid.n == len(np.load(os.path.join(src, "gpt2", "valid_bytes.npy"))), "valid hikaye sirasi"
    fit = valid.lengths()[pick] <= D.ROW_LEN                         # kucuk C'de satira sigmayan sinav hikayesi disarida
    dropped, pick = pick[~fit], pick[fit]
    ro, rs = D.pack_plan(valid.lengths()[pick], D.ROW_LEN, None)
    np.savez(os.path.join(out, "exam_pack_plan.npz"), row_offsets=ro, row_stories=pick[rs].astype(np.int32), seed=-1,
             epoch=0, row_len=D.ROW_LEN, exam_set_sha256=str(ex_src["exam_set_sha256"]), dropped=dropped.astype(np.int32))
    if len(dropped):
        print("UYARI: %d sinav hikayesi satira sigmadi, sinav disi: %s" % (len(dropped), dropped.tolist()), flush=True)
    for f in ("train_token_counts.npy", "reading_prompts.json"):
        if os.path.exists(os.path.join(src, f)):
            shutil.copy2(os.path.join(src, f), os.path.join(out, f))
    print("sinav paket plani: %d hikaye, %d satir | BITTI %.0f sn" % (len(pick), len(ro) - 1, time.time() - t0), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--chunk", type=int, default=31, help="parca token sayisi (Z / END haric); 31 -> 32 konum")
    a = ap.parse_args()
    main(a.src, a.out, a.chunk)
