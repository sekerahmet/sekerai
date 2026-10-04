"""train_meaning -- meaning agent'in egitimi (ajan: meaning.py).

Veri: her cumlede biten buyuyen pencere (hikaye basinda 1, 1-2, 1-3 ... en cok WINDOW cumle; kullanici, 4 Ekim: "sıralı
ilk cümle sonra ilk cümle ve ikinci cümle sonra ilk üç cümle gibi"); pencerenin farkli kelimeleri tek ortak torba.  Ulke:
data/countries hikayeleri; SS: make_ss_sentences --stories dosyalari (ss_story_*.npy; kullanici: "eğitim hikaye hikaye").
Egitim (kullanici: "1 cümlenin tüm kelimeleri sırayla gizlenmezse model nasıl öğrenecek ?"): penceredeki her kelime
sirayla gizli sayilir ve obur kelimelerden tahmin edilir (toplu hesap); kayip -log P(gizli kelime) ortalamasi.  Sik
kelime seyreltmesi (word2vec): kelime pencerede sqrt(t / f) + t / f olasilikla kalir.
Olcu goz ile (kullanici: "biz gözle bakıp Türkiye için ne yapmış ona bakmak"; "Sınav yok bunda göz ile kontrol var"):
EYE kelimelerinin tablo satiri.  Sonunda agent.pt ve neighbors.pt (her kelimenin ilk 50 komsusu, kosinus).

    python train_meaning.py [--data countries|simplestories] [--root klasor] [--window 5] [--epochs 4]
                            [--device cpu|cuda] [--out klasor]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

from meaning import D, MeaningAgent, build_neighbor_table

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_Z = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "grammar"))
from train_grammar import _no_power_throttling  # noqa: E402

WINDOW = 5              # pencere: en cok kac cumle (kullanici, 4 Ekim: "Meaning agent window 5")
BATCH = 256             # pencere; tam SS'te Colab hucresi buyuk verir
LR = 3e-3
MAX_WORDS = 256         # pencerede en cok kelime (uzun pencerede son kelimeler)
PROGRESS_SECS = 60      # epok icinde ara satir araligi (kullanici, 4 Ekim: "ekle bunları")
SUBSAMPLE = 1e-3        # sik kelime seyreltmesi (word2vec): kelime p = min(1, sqrt(t / f) + t / f) olasilikla kalir,
                        # f kelimenin sikligi; hem gizlenen hem oy veren (kullanici, 4 Ekim: "Evet")
EYE = {"countries": ("Turkey", "Ankara", "baklava", "Peru", "Lima", "Japan", "Spanish", "South", "the", "is", "."),
       "simplestories": ("dragon", "forest", "rain", "cookie", "school", "ocean", "Lily", "happy", "said", "the", ".")}


class Windows:
    """Hikaye verisi (kelimeler uc uca, cumle baslari, hikaye baslari) -> pencere batch'leri (ids, present) cihazda.  Pencere
    s. cumlede biter, geriye en cok W cumle (hikaye basini gecmez); farkli kelimeler, <unk> (0) haric."""

    def __init__(self, flat, sent_off, story_off, W, device):
        self.flat = torch.as_tensor(flat, dtype=torch.long, device=device)
        self.sent_off = torch.as_tensor(sent_off, dtype=torch.long, device=device)
        counts = np.diff(story_off)
        first = np.repeat(np.asarray(story_off[:-1]), counts)                 # cumlenin hikayesinin ilk cumlesi
        s = np.arange(len(sent_off) - 1)
        self.begin = torch.as_tensor(np.maximum(first, s - W + 1), dtype=torch.long, device=device)
        self.n = len(s)

    def batch(self, rows):
        a = self.sent_off[self.begin[rows]]
        b = self.sent_off[rows + 1]
        a = torch.maximum(a, b - MAX_WORDS)
        L = int((b - a).max())
        pos = a[:, None] + torch.arange(L, device=a.device)[None]
        ok = pos < b[:, None]
        ids = torch.where(ok, self.flat[pos.clamp(max=len(self.flat) - 1)], torch.zeros_like(pos))
        ids = ids.sort(1).values                                               # ayni kelimeler yan yana
        present = (ids != 0) & torch.cat([torch.ones_like(ids[:, :1], dtype=torch.bool), ids[:, 1:] != ids[:, :-1]], 1)
        return ids, present


def load(args, vocab_out):
    """-> (sozluk, egitim Windows, kelime sayilari)."""
    if args.data == "countries":
        folder = os.path.join(MODEL_Z, "data", "countries")
        vocab = json.load(open(os.path.join(folder, "country_vocab.json"), encoding="utf-8"))
        index = {w: i for i, w in enumerate(vocab)}
        stories = [json.loads(line)["sentences"] for line in open(os.path.join(folder, "country_stories.jsonl"),
                                                                    encoding="utf-8") if '"split": "train"' in line]
        sents = [[index.get(w, 0) for w in x] for st in stories for x in st]
        flat = np.array([w for x in sents for w in x])
        sent_off = np.r_[0, np.cumsum([len(x) for x in sents])]
        story_off = np.r_[0, np.cumsum([len(st) for st in stories])]
    else:
        root = args.root
        vocab = json.load(open(os.path.join(root, "ss_vocab.json"), encoding="utf-8"))
        flat = np.load(os.path.join(root, "ss_story_ids.npy"))
        sent_off = np.load(os.path.join(root, "ss_story_sentence_offsets.npy"))
        story_off = np.load(os.path.join(root, "ss_story_offsets.npy"))
    count = np.bincount(flat, minlength=len(vocab))
    return vocab, Windows(flat, sent_off, story_off, args.window, args.device), torch.as_tensor(count, dtype=torch.float)


def loss_of(agent, ids, present, keep=None, gen=None):
    """Pencereler -> her kelime gizliyken -log P ortalamasi.  keep (V,): sik kelime seyreltmesi (atilan kelime ne gizlenir
    ne oy verir)."""
    if keep is not None:
        present = present & (torch.rand(ids.shape, generator=gen, device=ids.device) < keep[ids])
    logp, ok = agent(ids, present)
    use = present & ok
    return -(logp * use).sum() / use.sum().clamp(min=1)


@torch.no_grad()
def eye(agent, words):
    """Kelimelerin tablo satirlari (ilk 15, kosinus)."""
    table = build_neighbor_table(agent, 15)
    print("\n   BAG TABLOSU (ilk 15):", flush=True)
    for w in words:
        if w in agent.index:
            i = agent.index[w]
            print("   %-8s %s" % (w, ", ".join("%s %.1f" % (agent.vocab[j], v) for j, v in zip(
                table["ids"][i].tolist(), table["scores"][i].tolist()))), flush=True)
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="countries", choices=("countries", "simplestories"))
    ap.add_argument("--root", default=None, help="SS: ss_vocab.json ve ss_story_*.npy klasoru")
    ap.add_argument("--window", type=int, default=WINDOW, help="pencere: en cok kac cumle (kelimeleri ortak torba)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=D)
    ap.add_argument("--batch", type=int, default=BATCH, help="pencere")
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--subsample", type=float, default=SUBSAMPLE, help="sik kelime seyreltme esigi t (0: kapali)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--out", default=None, help="kosu klasoru: agent.pt, neighbors.pt")
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(4)
        if os.name == "nt":
            print("guc kisitlamasi (EcoQoS) kapali:", _no_power_throttling(), flush=True)
    vocab, data, count = load(args, args.out)
    agent = MeaningAgent(vocab, d=args.d).to(args.device)
    keep = None
    if args.subsample > 0:
        f = count / count.sum()
        keep = ((args.subsample / f.clamp_min(1e-12)).sqrt() + args.subsample / f.clamp_min(1e-12)).clamp(max=1).to(args.device)
        print("sik kelime seyreltmesi t %g: kalma olasiligi %s" % (args.subsample, ", ".join(
            "%s %.2f" % (w, keep[agent.index[w]].item()) for w in EYE[args.data][-4:])), flush=True)
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    n = data.n
    print("veri %s: %d pencere (en cok %d cumle, ortak torba), sozluk %d | d %d, lr %g cosine, batch %d pencere, %d parametre"
          % (args.data, n, args.window, len(vocab), args.d, args.lr, args.batch, sum(p.numel() for p in agent.parameters())),
          flush=True)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed + epoch)).to(args.device)
        total, t_epoch = torch.zeros((), device=args.device), time.time()
        t_shown, steps = time.time(), -(-n // args.batch)
        for step, b in enumerate(range(0, n, args.batch), 1):
            if time.time() - t_shown > PROGRESS_SECS:          # ara satir: kayip yalniz burada okunur
                t_shown, el = time.time(), time.time() - t_epoch
                print("  epok %d adim %d / %d (%%%.0f)  kayip %.3f  %.0f pencere/sn  kalan ~%.0f dk (butun egitim)" % (
                    epoch, step, steps, 100 * step / steps, total.item() / step, b / el,
                    ((steps - step) + (args.epochs - epoch) * steps) * el / step / 60), flush=True)
            done = ((epoch - 1) * n + b) / (args.epochs * n)
            for group in opt.param_groups:
                group["lr"] = args.lr * 0.5 * (1 + math.cos(math.pi * done))
            ids, present = data.batch(perm[b:b + args.batch])
            loss = loss_of(agent, ids, present, keep, gen)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.detach()                             # .item() yok: her adimda GPU beklenmez
        print("epok %d  kayip %.4f  (%.0f sn)" % (epoch, total.item() / steps, time.time() - t0), flush=True)
    agent.eval()
    table = eye(agent, EYE[args.data])
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        torch.save(dict(vocab=vocab, state=agent.state_dict(), args=vars(args)), os.path.join(args.out, "agent.pt"))
        torch.save(build_neighbor_table(agent, 50), os.path.join(args.out, "neighbors.pt"))
        print("kaydedildi:", args.out, flush=True)
    return agent


if __name__ == "__main__":
    main()
