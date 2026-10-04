"""train_meaning -- meaning agent'in egitimi ve olcumu (ajan: meaning.py).

Veri: hikayelerden ardisik WINDOW cumlelik pencereler; pencerenin butun cumlelerinin farkli kelimeleri tek ortak torba
(sira ve cumle siniri yok).  Egitim: torbadaki icerik kelimelerinin her biri MASK_RATE olasilikla (en az biri) mask ile
degistirilir; model gizli kelimeleri kalanlara bakarak tahmin eder; kayip gizli yuvalarda -log P(kelime).  Bicim
kelimeleri (cumlelerin %2'sinden fazlasinda gecen) gizlenmez, kisa listeye her zaman girer.
Olcu (sinav hikayeleri, her cumle gecisi): shortlist_recall -- sonraki cumlenin kelimeleri, son WINDOW cumlenin
kelimelerinin shortlist'i (ilk N) + hikayede gecmis kelimeler + bicim kelimeleri listesinde mi; icerik kelimeleri ayri.
Taban: baglamsiz en sik 200 icerik kelimesi.  Goz: birkac kelime tek basina verilince ilk 10 tahmin.
Sonunda agent.pt ve neighbors.pt (her kelimenin ilk 50 tahmini).

    python train_meaning.py [--window 1] [--epochs 4] [--device cpu|cuda] [--out klasor]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

from meaning import D, MeaningAgent, build_neighbor_table, shortlist

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_Z = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "grammar"))
from train_grammar import _no_power_throttling  # noqa: E402

WINDOW = 1              # pencere: kac cumle (kullanici, 4 Ekim: "önce 1 bakarız ... performansa göre bakarız")
MASK_RATE = 0.25        # icerik kelimesinin gizlenme olasiligi, torbada en az bir (olculmedi)
BATCH = 256             # pencere
LR = 3e-3
SHORTLIST_N = (10, 20, 50)
EYE = ("Turkey", "Ankara", "baklava", "Peru")


def windows_of(agent, stories, W):
    """Hikayeler -> pencereler (N, L) kimlik, maske: her hikayede ardisik W cumlenin farkli kelimeleri."""
    rows = []
    for st in stories:
        for t in range(max(1, len(st) - W + 1)):
            rows.append(sorted(set(agent.ids([w for s in st[t:t + W] for w in s])) - {0}))
    L = max(len(r) for r in rows)
    ids = torch.zeros(len(rows), L, dtype=torch.long)
    present = torch.zeros(len(rows), L, dtype=torch.bool)
    for i, r in enumerate(rows):
        ids[i, :len(r)] = torch.tensor(r)
        present[i, :len(r)] = True
    return ids, present


def loss_of(agent, ids, present, content, rate, gen):
    """Icerik kelimelerinin her biri rate olasilikla (torbada en az biri) gizlenir -> gizli yuvalarda ort -log P."""
    cand = present & content[ids]
    r = torch.rand(ids.shape, generator=gen).to(ids.device)
    hidden = cand & (r < rate)
    first = r.masked_fill(~cand, 2.0).argmin(1)                  # en az bir gizli (icerik kelimesi varsa)
    hidden[torch.arange(len(ids)), first] |= cand[torch.arange(len(ids)), first]
    logp = agent(ids, present, hidden)
    return -(logp.gather(2, ids[..., None])[..., 0] * hidden).sum() / hidden.sum().clamp(min=1)


def shortlist_recall(agent, exam, function, W, freq_top):
    """-> {N: (liste ort, butun kelimeler listede, icerik kelimeleri listede)}; N = 'taban' siklik listesi."""
    fn = set(np.flatnonzero(function).tolist())
    out = {}
    for N in SHORTLIST_N + ("taban",):
        hit = tot = hit_c = tot_c = size = n = 0
        for st in exam:
            seen = set()
            for t in range(len(st) - 1):
                seen |= set(st[t]) - {0}
                cand = seen | fn
                if N == "taban":
                    cand |= freq_top
                else:
                    cand |= shortlist(agent, sorted({x for s in st[max(0, t - W + 1):t + 1] for x in s} - {0}), N)
                target = set(st[t + 1]) - {0}
                content = [x for x in target if not function[x]]
                hit += sum(x in cand for x in target)
                tot += len(target)
                hit_c += sum(x in cand for x in content)
                tot_c += len(content)
                size += len(cand)
                n += 1
        out[N] = (round(size / n, 1), round(hit / tot, 4), round(hit_c / max(tot_c, 1), 4))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", type=int, default=WINDOW, help="pencere: kac cumle (kelimeleri ortak torba)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=D)
    ap.add_argument("--mask_rate", type=float, default=MASK_RATE)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--out", default=None, help="kosu klasoru: agent.pt, neighbors.pt, results.json")
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(4)
        if os.name == "nt":
            print("guc kisitlamasi (EcoQoS) kapali:", _no_power_throttling(), flush=True)
    folder = os.path.join(MODEL_Z, "data", "countries")
    vocab = json.load(open(os.path.join(folder, "country_vocab.json"), encoding="utf-8"))
    stories = [json.loads(line) for line in open(os.path.join(folder, "country_stories.jsonl"), encoding="utf-8")]
    agent = MeaningAgent(vocab, d=args.d).to(args.device)
    train = [s["sentences"] for s in stories if s["split"] == "train"]
    exam = [[agent.ids(x) for x in s["sentences"]] for s in stories if s["split"] == "exam"]
    ids, present = windows_of(agent, train, args.window)
    df = np.zeros(len(vocab))
    n_sent = 0
    for st in train:
        for s in st:
            n_sent += 1
            df[list(set(agent.ids(s)))] += 1
    function = df / n_sent > 0.02                       # bicim kelimesi: gizlenmez, her listeye girer
    content = torch.tensor(~function, device=args.device)
    content[0] = False
    freq_top = set([i for i in np.argsort(-df) if not function[i] and i != 0][:200])
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr)
    gen = torch.Generator().manual_seed(args.seed)
    n = len(ids)
    print("veri: %d pencere (%d cumle, ortak torba), sozluk %d, bicim kelimesi %d | d %d, mask_rate %g, lr %g cosine, "
          "batch %d, %d parametre" % (n, args.window, len(vocab), function.sum(), args.d, args.mask_rate, args.lr,
                                      args.batch, sum(p.numel() for p in agent.parameters())), flush=True)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n, generator=gen)
        total = 0.0
        for b in range(0, n, args.batch):
            done = ((epoch - 1) * n + b) / (args.epochs * n)
            for group in opt.param_groups:
                group["lr"] = args.lr * 0.5 * (1 + math.cos(math.pi * done))
            rows = perm[b:b + args.batch]
            loss = loss_of(agent, ids[rows].to(args.device), present[rows].to(args.device), content, args.mask_rate, gen)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(rows)
        print("epok %d  kayip %.4f  (%.0f sn)" % (epoch, total / n, time.time() - t0), flush=True)
    agent.eval()
    res = shortlist_recall(agent, exam, function, args.window, freq_top)
    for N, (size, all_, cont) in res.items():
        print("   shortlist_recall %-5s liste ort %5.1f kelime | sonraki cumlenin kelimeleri listede %.3f (icerik %.3f)" % (
            N, size, all_, cont), flush=True)
    table = build_neighbor_table(agent, 50)
    for w in EYE:
        if w in agent.index:
            i = agent.index[w]
            print("   %-8s %s" % (w, " ".join("%s %.2f" % (vocab[j], p) for j, p in zip(
                table["ids"][i, :10].tolist(), table["scores"][i, :10].tolist()))), flush=True)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        torch.save(dict(vocab=vocab, state=agent.state_dict(), args=vars(args)), os.path.join(args.out, "agent.pt"))
        torch.save(table, os.path.join(args.out, "neighbors.pt"))
        json.dump(dict(shortlist_recall={str(k): v for k, v in res.items()}), open(os.path.join(args.out, "results.json"), "w"),
                  indent=1)
        print("kaydedildi:", args.out, flush=True)
    return agent


if __name__ == "__main__":
    main()
