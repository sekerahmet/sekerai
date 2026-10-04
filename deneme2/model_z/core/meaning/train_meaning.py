"""train_meaning -- meaning agent'in egitimi (ajan: meaning.py).

Veri: her cumlede biten buyuyen pencere (hikaye basinda 1, 1-2, 1-3 ... en cok WINDOW cumle; kullanici, 4 Ekim: "sıralı
ilk cümle sonra ilk cümle ve ikinci cümle sonra ilk üç cümle gibi"); pencerenin farkli kelimeleri tek ortak torba.
Egitim (kullanici: "1 cümlenin tüm kelimeleri sırayla gizlenmezse model nasıl öğrenecek ?"): torbanin her kelimesi sirayla
birer kez gizlenir; gizli kelime kalanlarin bag tablosu oylariyla tahmin edilir; kayip -log P(gizli kelime).
Olcu goz ile (kullanici: "biz gözle bakıp Türkiye için ne yapmış ona bakmak"): EYE kelimelerinin tablo satiri; EYE
ulkelerinin ilk sinav hikayesinde gizli kelime tahminleri ve en cok oy veren kelimeler.
Sonunda agent.pt ve neighbors.pt (bag tablosunun her satirinin ilk 50 kelimesi).

    python train_meaning.py [--window 5] [--epochs 4] [--device cpu|cuda] [--out klasor]
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

WINDOW = 5              # pencere: en cok kac cumle (kullanici, 4 Ekim: "Meaning agent window 5")
BATCH = 256             # ornek (pencere x gizlenen kelime)
LR = 3e-3
EYE_WORDS = ("Turkey", "Ankara", "baklava", "Peru", "Lima", "Japan", "Spanish", "South", "the", "is", ".")
EYE_COUNTRIES = ("Turkey", "Peru")


def windows_of(agent, stories, W):
    """Hikayeler -> pencereler (N, L) kimlik, maske: her cumlede biten, geriye en cok W cumlenin farkli kelimeleri."""
    rows = []
    for st in stories:
        for t in range(len(st)):
            rows.append(sorted(set(agent.ids([w for s in st[max(0, t - W + 1):t + 1] for w in s])) - {0}))
    L = max(len(r) for r in rows)
    ids = torch.zeros(len(rows), L, dtype=torch.long)
    present = torch.zeros(len(rows), L, dtype=torch.bool)
    for i, r in enumerate(rows):
        ids[i, :len(r)] = torch.tensor(r)
        present[i, :len(r)] = True
    return ids, present


def loss_of(agent, ids, present, slot):
    """Ornek: pencere + gizlenen yuva (slot) -> ort -log P(gizli kelime)."""
    rows = torch.arange(len(ids), device=ids.device)
    seen = present.clone()
    seen[rows, slot] = False
    logp, _ = agent(ids, seen)
    return -logp[rows, ids[rows, slot]].mean()


@torch.no_grad()
def eye(agent, stories, W, windows=6, top=5):
    """Tablo satirlari; ulkelerin ilk sinav hikayesinde (ilk `windows` pencere) her kelime gizli -> ilk top tahmin ve en
    cok oy veren 3 kelime."""
    vocab = agent.vocab
    table = build_neighbor_table(agent, 15)
    print("\n   BAG TABLOSU (R satiri, ilk 15):", flush=True)
    for w in EYE_WORDS:
        if w in agent.index:
            i = agent.index[w]
            print("   %-8s %s" % (w, ", ".join("%s %.1f" % (vocab[j], v) for j, v in zip(
                table["ids"][i].tolist(), table["scores"][i].tolist()))), flush=True)
    hit = total = 0
    for c in EYE_COUNTRIES:
        st = next(s["sentences"] for s in stories if s["country"] == c and s["split"] == "exam")
        print("\n   === %s (ilk %d pencere, en cok %d cumle)" % (c, windows, W), flush=True)
        for t in range(min(windows, len(st))):
            bag = []
            for x in st[max(0, t - W + 1):t + 1]:
                bag += [w for w in x if w not in bag]
            ids = torch.tensor([agent.ids(bag)], device=agent.bias.device)
            print("   " + " ".join(bag), flush=True)
            for j, w in enumerate(bag):
                present = torch.ones_like(ids, dtype=torch.bool)
                present[0, j] = False
                logp, alpha = agent(ids, present)
                best = logp[0].exp().topk(top)
                voters = alpha[0].topk(min(3, len(bag) - 1))
                ok = vocab[best.indices[0]] == w
                hit += ok
                total += 1
                print("      %s %-12s -> %-60s oy: %s" % (
                    "+" if ok else " ", w, " ".join("%s %.2f" % (vocab[k], v) for k, v in zip(
                        best.indices.tolist(), best.values.tolist())),
                    " ".join("%s %.2f" % (bag[k], v) for k, v in zip(voters.indices.tolist(), voters.values.tolist()))),
                    flush=True)
    print("\n   gizli kelime ilk tahminde dogru: %d / %d" % (hit, total), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", type=int, default=WINDOW, help="pencere: en cok kac cumle (kelimeleri ortak torba)")
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
    print("veri: %d pencere (en cok %d cumle, ortak torba), %d ornek (her kelime sirayla gizli), sozluk %d | d %d, lr %g "
          "cosine, batch %d, %d parametre" % (len(ids), args.window, n, len(vocab), args.d, args.lr, args.batch,
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
    eye(agent, stories, args.window)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        torch.save(dict(vocab=vocab, state=agent.state_dict(), args=vars(args)), os.path.join(args.out, "agent.pt"))
        torch.save(build_neighbor_table(agent, 50), os.path.join(args.out, "neighbors.pt"))
        print("kaydedildi:", args.out, flush=True)
    return agent


if __name__ == "__main__":
    main()
