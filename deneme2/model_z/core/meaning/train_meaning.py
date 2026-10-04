"""train_meaning -- meaning agent'in egitimi (ajan: meaning.py).

Veri: hikayelerden ardisik WINDOW cumlelik pencereler; pencerenin butun cumlelerinin farkli kelimeleri tek ortak torba
(sira ve cumle siniri yok).  Egitim (kullanici, 4 Ekim: "1 cümlenin tüm kelimeleri sırayla gizlenmezse model nasıl
öğrenecek ? Ben rastgele demedim hiç"): torbanin her kelimesi sirayla birer kez mask ile gizlenir; model kalanlara bakip
gizliyi tahmin eder; kayip -log P(gizli kelime).
Olcu goz ile (kullanici: "Sınav görülmemiş mantıklı liste değil biz gözle bakıp Türkiye için ne yapmış ona bakmak"):
EYE ulkelerinin cumlelerinde her kelime sirayla gizlenir, ilk 5 tahmin olasiligiyla yazilir.
Sonunda agent.pt ve neighbors.pt (her kelime tek basina verilince ilk 50 tahmin).

    python train_meaning.py [--window 1] [--epochs 4] [--device cpu|cuda] [--out klasor]
"""
import argparse
import json
import math
import os
import sys
import time

import torch

from meaning import D, MeaningAgent, build_neighbor_table

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_Z = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "grammar"))
from train_grammar import _no_power_throttling  # noqa: E402

WINDOW = 1              # pencere: kac cumle (kullanici, 4 Ekim: "önce 1 bakarız ... performansa göre bakarız")
BATCH = 256             # ornek (pencere x gizlenen kelime)
LR = 3e-3
EYE = ("Turkey", "Peru")


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


def loss_of(agent, ids, present, slot):
    """Ornek: pencere + gizlenen yuva (slot) -> ort -log P(gizli kelime)."""
    hidden = torch.zeros_like(present)
    hidden[torch.arange(len(ids)), slot] = True
    logp = agent(ids, present, hidden)
    return -logp[torch.arange(len(ids)), slot, ids[torch.arange(len(ids)), slot]].mean()


@torch.no_grad()
def eye(agent, stories, countries, W=1, top=5, windows=8):
    """Ulkenin pencerelerinde her kelime sirayla gizli -> ilk top tahmin.  W 1: ulkenin butun farkli cumleleri; W > 1:
    ulkenin ilk sinav hikayesinin ilk `windows` penceresi (W cumlenin kelimeleri ortak torba)."""
    dev = agent.E.weight.device
    for c in countries:
        sents = []
        if W == 1:
            for s in stories:
                if s["country"] == c:
                    for x in s["sentences"]:
                        if x not in sents:
                            sents.append(x)
        else:
            st = next(s["sentences"] for s in stories if s["country"] == c and s["split"] == "exam")
            for t in range(min(windows, len(st) - W + 1)):
                bag = []
                for x in st[t:t + W]:
                    bag += [w for w in x if w not in bag]
                sents.append(bag)
        print("\n   === %s (%d pencere, %d cumle)" % (c, len(sents), W), flush=True)
        for x in sents:
            ids = torch.tensor([agent.ids(x)], device=dev)
            present = torch.ones_like(ids, dtype=torch.bool)
            print("   " + " ".join(x), flush=True)
            for j, w in enumerate(x):
                hidden = torch.zeros_like(present)
                hidden[0, j] = True
                p = agent(ids, present, hidden)[0, j].exp()
                best = p.topk(top)
                guess = " ".join("%s %.2f" % (agent.vocab[k], v) for k, v in zip(best.indices.tolist(), best.values.tolist()))
                mark = "+" if agent.vocab[best.indices[0]] == w else " "
                print("      %s %-12s -> %s" % (mark, w, guess), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", type=int, default=WINDOW, help="pencere: kac cumle (kelimeleri ortak torba)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=D)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--out", default=None, help="kosu klasoru: agent.pt, neighbors.pt")
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
    ids, present = windows_of(agent, [s["sentences"] for s in stories if s["split"] == "train"], args.window)
    win, slot = present.nonzero(as_tuple=True)          # her pencerenin her kelimesi bir ornek
    n = len(win)
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr)
    gen = torch.Generator().manual_seed(args.seed)
    print("veri: %d pencere (%d cumle, ortak torba), %d ornek (her kelime sirayla gizli), sozluk %d | d %d, lr %g cosine, "
          "batch %d, %d parametre" % (len(ids), args.window, n, len(vocab), args.d, args.lr, args.batch,
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
            w = win[rows]
            loss = loss_of(agent, ids[w].to(args.device), present[w].to(args.device), slot[rows].to(args.device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(rows)
        print("epok %d  kayip %.4f  (%.0f sn)" % (epoch, total / n, time.time() - t0), flush=True)
    agent.eval()
    eye(agent, stories, EYE, args.window)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        torch.save(dict(vocab=vocab, state=agent.state_dict(), args=vars(args)), os.path.join(args.out, "agent.pt"))
        torch.save(build_neighbor_table(agent, 50), os.path.join(args.out, "neighbors.pt"))
        print("kaydedildi:", args.out, flush=True)
    return agent


if __name__ == "__main__":
    main()
