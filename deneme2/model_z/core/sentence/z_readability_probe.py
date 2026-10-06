"""z_readability_probe -- bir z ayarindan modelin dogrusal olarak ne okuyabildigi (CPU teshis araci, egitim yok; ad
onayli, kullanici 6 Ekim).  Geri acma (decode_z) dis cozucunun olcusudur; SentenceTransformer z'yi tek dogrusal adimla
d boyuta indirerek gorur (z_in).  Bu arac ayni kisiti tasir: ridge + rank d siniri (indirgenmis rank regresyonu).

    hedefler (SS sinav hikayeleri, hikayeye gore bolunmus):
        kelime   ayni cumlede kelime var mi: en sik --top kelime, cumle basina recall@n (hepsi; icerik: ilk 50 disi)
        ilk      ayni cumlenin ilk kelimesi: en sik --classes sinif, dogruluk
    taban: siklik sirasi (z'siz).  --model verilirse egitilmis SentenceTransformer'in z_in'inden sonraki x de olculur.
Olculdu (1.500 hikaye, rank 256, kelime icerik / ilk kelime): rol 0,417 / 0,818 (z_in sonrasi 0,371 / 0,769); rolsuz
0,297 / 0,853; rolsuz + torba kanali 0,624 / 0,937 (belge/model_z_temel/15).

    python z_readability_probe.py [--roles 0|1] [--bag_channel 0|1] [--z 512] [--stories 1500] [--rank 256]
                                  [--model agent.pt] [--root SS klasoru] [--meaning ...] [--grammar ...]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "grammar"))
from train_grammar import _no_power_throttling  # noqa: E402
from sentence_z import build_keys, encode_z  # noqa: E402

RUNS = "G:/Drive'ım/model_z/runs/"
MEANING = RUNS + "meaning_ss_full_w5_d256_b2048_k0_t3e-05_e3_20261005_160420/agent.pt"
GRAMMAR = RUNS + "grammar_ssfull_d256_cosine_lr0.001_b1024_20261003_211559/agent.pt"
ROOT = "G:/Drive'ım/model_z/simplestories_full"
LONGEST = 226           # SS egitim en uzun cumlesi: egitimdeki build_keys ile ayni anahtarlar (P longest'e bagli)
FUNCTION_WORDS = 50     # icerik = en sik 50 kelime disi
LAMBDAS = (1e-2, 1e-1, 1.0)


def exam_sentences(root, stories, seed):
    """SS sinav hikayelerinden rastgele `stories` hikaye -> cumleler (kelime kimlikleri), hikaye sirasi."""
    ids = np.load(os.path.join(root, "ss_exam_story_ids.npy"))
    so = np.load(os.path.join(root, "ss_exam_story_sentence_offsets.npy"))
    st = np.load(os.path.join(root, "ss_exam_story_offsets.npy"))
    pick = np.random.default_rng(seed).choice(len(st) - 1, stories, replace=False)
    sents, story = [], []
    for k, s in enumerate(pick):
        for j in range(st[s], st[s + 1]):
            sents.append(ids[so[j]:so[j + 1]].astype(np.int64))
            story.append(k)
    return sents, np.array(story)


@torch.no_grad()
def encode_all(keys, sents, batch=512):
    out = []
    for c in range(0, len(sents), batch):
        part = sents[c:c + batch]
        L = max(len(s) for s in part)
        x = torch.zeros(len(part), L, dtype=torch.long)
        m = torch.zeros(len(part), L, dtype=torch.bool)
        for i, s in enumerate(part):
            x[i, :len(s)] = torch.from_numpy(s)
            m[i, :len(s)] = True
        out.append(encode_z(keys, x, m))
    return torch.cat(out)


def _ridge(X, Y, lam):
    A = X.T @ X
    return torch.linalg.solve(A + lam * torch.trace(A) / len(A) * torch.eye(len(A)), X.T @ Y)


def _reduce_rank(X, B, r):
    """Indirgenmis rank: tahminlerin ilk r ozyonune izdusum (rank <= r, modelin z_in'i gibi)."""
    if min(B.shape) <= r:
        return B
    Yh = X @ B
    _, U = torch.linalg.eigh(Yh.T @ Yh)
    return B @ U[:, -r:] @ U[:, -r:].T


def recall_at_n(S, Y, cols=None):
    """Satir basina: en yuksek n skorun gercek kelimeleri kapsama orani, n = cumlenin (kolonlardaki) kelime sayisi."""
    if cols is not None:
        S, Y = S[:, cols], Y[:, cols]
    k = Y.sum(1).astype(int)
    ok = k > 0
    S, Y, k = S[ok], Y[ok], k[ok]
    order = np.argsort(-S, 1)
    return float(np.mean([Y[i, order[i, :k[i]]].sum() / k[i] for i in range(len(k))]))


def targets(sents, V, top, classes):
    cnt = np.bincount(np.concatenate(sents), minlength=V)
    col = -np.ones(V, np.int64)
    col[np.argsort(-cnt)[:top]] = np.arange(top)
    Y = np.zeros((len(sents), top), np.float32)
    for i, s in enumerate(sents):
        c = col[s]
        Y[i, c[c >= 0]] = 1
    first = np.array([s[0] for s in sents])
    fcol = -np.ones(V, np.int64)
    fcol[np.argsort(-np.bincount(first, minlength=V))[:classes]] = np.arange(classes)
    return Y, fcol[first]


def probe(X, Y, first, story, rank, classes):
    """X (N, D) ozellik; hikayeye gore bolme: ilk %57 uydurma, %10 dogrulama (lambda), kalan %33 sinav.
    -> dict(word, word_content, first)."""
    S = story.max() + 1
    fit, val, te = story < 0.57 * S, (story >= 0.57 * S) & (story < 0.67 * S), story >= 0.67 * S
    X = X.float()
    mu, sd = X[fit].mean(0), X[fit].std(0).clamp_min(1e-6)
    X = torch.cat([(X - mu) / sd, torch.ones(len(X), 1)], 1)
    content = np.arange(Y.shape[1]) >= FUNCTION_WORDS
    Yt = torch.from_numpy(Y)
    ybar = Yt[fit].mean(0)
    scored = []
    for lam in LAMBDAS:
        B = _reduce_rank(X[fit], _ridge(X[fit], Yt[fit] - ybar, lam), rank)
        scored.append((recall_at_n((X[val] @ B + ybar).numpy(), Y[val]), lam, B))
    best = max(scored, key=lambda x: x[0])[2]
    S_te = (X[te] @ best + ybar).numpy()
    yk = first >= 0
    oh = torch.zeros(len(first), classes)
    oh[np.nonzero(yk)[0], first[yk]] = 1
    f, v, t = fit & yk, val & yk, te & yk
    scored = []
    for lam in LAMBDAS:
        B = _reduce_rank(X[f], _ridge(X[f], oh[f], lam), rank)
        scored.append((float(((X[v] @ B).argmax(1).numpy() == first[v]).mean()), lam, B))
    bestf = max(scored, key=lambda x: x[0])[2]
    return dict(word=round(recall_at_n(S_te, Y[te]), 3), word_content=round(recall_at_n(S_te, Y[te], content), 3),
                first=round(float(((X[t] @ bestf).argmax(1).numpy() == first[t]).mean()), 3),
                baseline_word=round(recall_at_n(np.tile(ybar.numpy(), (te.sum(), 1)), Y[te]), 3),
                baseline_word_content=round(recall_at_n(np.tile(ybar.numpy(), (te.sum(), 1)), Y[te], content), 3))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--meaning", default=MEANING)
    ap.add_argument("--grammar", default=GRAMMAR)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--z", type=int, default=512, help="konum kanali boyutu")
    ap.add_argument("--roles", type=int, default=0)
    ap.add_argument("--bag_channel", type=int, default=1)
    ap.add_argument("--stories", type=int, default=1500)
    ap.add_argument("--rank", type=int, default=256, help="modelin d'si (z_in rank'i)")
    ap.add_argument("--top", type=int, default=2000)
    ap.add_argument("--classes", type=int, default=300)
    ap.add_argument("--model", default=None, help="egitilmis SentenceTransformer agent.pt: z_in(z_norm(z)) de olculur")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)
    if os.name == "nt":
        print("guc kisitlamasi (EcoQoS) kapali:", _no_power_throttling(), flush=True)
    torch.set_num_threads(args.threads)
    t0 = time.time()
    keys = build_keys(args.meaning, args.grammar, args.z, LONGEST, roles=bool(args.roles),
                      bag_channel=bool(args.bag_channel))
    sents, story = exam_sentences(args.root, args.stories, args.seed)
    Z = encode_all(keys, sents)
    Y, first = targets(sents, len(keys["vocab"]), args.top, args.classes)
    name = "z %d (konum %d%s%s)" % (keys["z"], keys["z_pos"], " + rol" if args.roles else "",
                                     " ; torba" if args.bag_channel else "")
    print("%d hikaye, %d cumle, %s, rank %d (%.0f sn hazirlik)" % (args.stories, len(sents), name, args.rank,
                                                                     time.time() - t0), flush=True)
    feats = {name: Z}
    if args.model:
        st = torch.load(args.model, map_location="cpu", weights_only=False)["state"]
        assert st["z_in.weight"].shape[1] == keys["z"], "model z_dim %d, z %d" % (st["z_in.weight"].shape[1], keys["z"])
        zn = Z * torch.rsqrt((Z ** 2).mean(-1, keepdim=True) + 1e-6) * st["z_norm.weight"].float()
        feats["egitilmis z_in sonrasi (%d)" % st["z_in.weight"].shape[0]] = zn @ st["z_in.weight"].float().T
    out = {}
    for k, X in feats.items():
        r = probe(X, Y, first, story, args.rank, args.classes)
        out[k] = r
        print("%-40s kelime recall@n %.3f (icerik %.3f) | ilk kelime %.3f | taban %.3f / %.3f   (%.0f sn)" % (
            k, r["word"], r["word_content"], r["first"], r["baseline_word"], r["baseline_word_content"],
            time.time() - t0), flush=True)
    return out


if __name__ == "__main__":
    main()
