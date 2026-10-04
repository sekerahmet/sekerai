"""train_meaning -- meaning agent'in egitimi ve olcumu (ajan: meaning.py).

Veri: hikayelerden ardisik WINDOW cumlelik pencereler; pencerenin farkli kelimeleri sirasiz.  Kayip (SGNS): her kelime
icin penceredeki obur kelimeler olumlu (log sigmoid bag), sikligin 0,75 kuvvetiyle cekilen NEGATIVES kelime olumsuz.
Olcu (sinav hikayeleri, her cumle gecisi): shortlist_recall -- sonraki cumlenin kelimeleri, son WINDOW cumlenin
kelimelerinin shortlist'i (ilk N bag) + hikayede gecmis kelimeler + bicim kelimeleri (cumlelerin %2'sinden fazlasinda
gecen) listesinde mi; icerik kelimeleri (bicim disi) ayri.  Taban: baglamsiz en sik 200 icerik kelimesi.
Goz: birkac kelimenin tablo satiri.  Sonunda agent.pt ve neighbors.pt (her kelimenin ilk 50 bagi).

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
import torch.nn.functional as F

from meaning import D, MeaningAgent, build_neighbor_table, shortlist

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_Z = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "grammar"))
from train_grammar import _no_power_throttling  # noqa: E402

WINDOW = 1              # pencere: kac cumle (kullanici, 4 Ekim: "önce 1 bakarız ... performansa göre bakarız")
NEGATIVES = 5           # kelime basina olumsuz ornek (word2vec degeri; bu modelde olculmedi)
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
    mask = torch.zeros(len(rows), L, dtype=torch.bool)
    for i, r in enumerate(rows):
        ids[i, :len(r)] = torch.tensor(r)
        mask[i, :len(r)] = True
    return ids, mask


def loss_of(agent, ids, mask, neg_dist, negatives, gen):
    """SGNS: -[ort log sigmoid(bag(i, j)) olumlu ciftler + ort log sigmoid(-bag(i, n)) olumsuzlar]."""
    h = agent(ids, mask)
    pos = mask[:, :, None] & mask[:, None, :] & (ids[:, :, None] != ids[:, None, :])
    s = h @ agent.E(ids).transpose(1, 2) / h.shape[-1] ** 0.5
    neg = torch.multinomial(neg_dist, ids.numel() * negatives, replacement=True, generator=gen).view(*ids.shape, negatives)
    sn = agent.bond(h, neg.to(ids.device))
    lp = (F.logsigmoid(s) * pos).sum() / pos.sum().clamp(min=1)
    ln = (F.logsigmoid(-sn) * mask[..., None]).sum() / (mask.sum() * negatives)
    return -(lp + ln)


def shortlist_recall(table, exam, function, W, freq_top):
    """-> {N: (liste ort, butun kelimeler listede, icerik kelimeleri listede)}; N = 'taban' siklik listesi."""
    fn = set(np.flatnonzero(function).tolist())
    out = {}
    for N in SHORTLIST_N + ("taban",):
        hit = tot = hit_c = tot_c = size = n = 0
        for st in exam:
            seen = set()
            for t in range(len(st) - 1):
                seen |= set(st[t])
                cand = seen | fn
                if N == "taban":
                    cand |= freq_top
                else:
                    cand |= shortlist(table, {x for s in st[max(0, t - W + 1):t + 1] for x in s}, N)
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
    ap.add_argument("--window", type=int, default=WINDOW, help="pencere: kac cumle")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=D)
    ap.add_argument("--negatives", type=int, default=NEGATIVES)
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
    ids, mask = windows_of(agent, train, args.window)
    count = torch.zeros(len(vocab))
    df = np.zeros(len(vocab))
    n_sent = 0
    for st in train:
        for s in st:
            n_sent += 1
            k = agent.ids(s)
            count.index_add_(0, torch.tensor(k), torch.ones(len(k)))
            df[list(set(k))] += 1
    function = df / n_sent > 0.02                       # bicim kelimesi: her listeye girer
    neg_dist = count ** 0.75
    neg_dist[0] = 0
    freq_top = set([i for i in np.argsort(-df) if not function[i] and i != 0][:200])
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr)
    gen = torch.Generator().manual_seed(args.seed)
    n = len(ids)
    print("veri: %d pencere (%d cumle), sozluk %d, bicim kelimesi %d | d %d, negatives %d, lr %g cosine, batch %d, "
          "%d parametre" % (n, args.window, len(vocab), function.sum(), args.d, args.negatives, args.lr, args.batch,
                            sum(p.numel() for p in agent.parameters())), flush=True)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n, generator=gen)
        total = 0.0
        for b in range(0, n, args.batch):
            done = ((epoch - 1) * n + b) / (args.epochs * n)
            for group in opt.param_groups:
                group["lr"] = args.lr * 0.5 * (1 + math.cos(math.pi * done))
            rows = perm[b:b + args.batch]
            loss = loss_of(agent, ids[rows].to(args.device), mask[rows].to(args.device), neg_dist, args.negatives, gen)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(rows)
        print("epok %d  kayip %.4f  (%.0f sn)" % (epoch, total / n, time.time() - t0), flush=True)
    agent.eval()
    table = build_neighbor_table(agent, 50)
    res = shortlist_recall(table, exam, function, args.window, freq_top)
    for N, (size, all_, content) in res.items():
        print("   shortlist_recall %-5s liste ort %5.1f kelime | sonraki cumlenin kelimeleri listede %.3f (icerik %.3f)" % (
            N, size, all_, content), flush=True)
    for w in EYE:
        if w in agent.index:
            print("   %-8s %s" % (w, " ".join(vocab[j] for j in table["ids"][agent.index[w], :10].tolist())), flush=True)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        torch.save(dict(vocab=vocab, state=agent.state_dict(), args=vars(args)), os.path.join(args.out, "agent.pt"))
        torch.save(table, os.path.join(args.out, "neighbors.pt"))
        json.dump(dict(shortlist_recall={str(k): v for k, v in res.items()}), open(os.path.join(args.out, "results.json"), "w"),
                  indent=1)
        print("kaydedildi:", args.out, flush=True)
    return agent, table


if __name__ == "__main__":
    main()
