"""train_context -- baglam ajaninin egitimi ve olcumu (ajan: context.py).

Kayip: her cumleden sonra durumun verdigi dagilim, sonraki cumlenin kelimelerine (sayilariyla) capraz entropi.  Etiket yok.
Olculer (sinav hikayeleri, her cumle gecisi): recall_at_10 / 50 / 100 -- sonraki cumlenin kelimelerinin (tekrarsiz)
ilk k aday icindeki payi; iki taban cizgisi ayni olcuyle: siklik (baglami hic okumaz) ve hikayede gecenler (once
hikayede gecen kelimeler, sonra siklik).  Dagilim farki: gecislerin dagilimi ortalama dagilimdan ne kadar ayri (0: her
baglamda ayni dagilim = cokus).  Goz: birkac sinav hikayesinde her cumleden sonra en olasi ve sikliga gore en cok
yukselen kelimeler.

    python train_context.py [--data countries] [--epochs 30] [--device cpu|cuda] [--slots 8] [--out klasor] [--resume 1]
"""
import argparse
import json
import math
import os
import time
from collections import Counter

import torch

from context import D, SLOTS, ContextAgent

MODEL_Z = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILES = {"countries": "country_"}   # data/<ad>/<onek>{stories.jsonl, vocab.json}
BATCH = 64              # hikaye
LR = 1e-3
SCHEDULE = "cosine"     # constant | cosine (adim adim --epochs sonunda 0)
KS = (10, 50, 100)
SHOW = 2                # goz: kac sinav hikayesi yazilir


def _cached(path, sources, build):
    """path varsa ve kaynaklarin hepsinden yeniyse onu oku; yoksa build() -> path'e yaz (her koşuda yeniden kodlanmasın)."""
    if os.path.exists(path) and all(os.path.getmtime(path) > os.path.getmtime(s) for s in sources):
        return torch.load(path, weights_only=False)
    out = build()
    torch.save(out, path + ".part")
    os.replace(path + ".part", path)
    return out


def _encode(agent, stories):
    """Hikayeler -> ids (N, T, W) dolgulu, mask (N, T, W)."""
    T = max(len(s["sentences"]) for s in stories)
    W = max(len(x) for s in stories for x in s["sentences"])
    ids = torch.zeros(len(stories), T, W, dtype=torch.long)
    for n, s in enumerate(stories):
        for t, x in enumerate(s["sentences"]):
            ids[n, t, :len(x)] = torch.tensor(agent.ids(x))
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for n, s in enumerate(stories):
        for t, x in enumerate(s["sentences"]):
            mask[n, t, :len(x)] = True
    return ids, mask


def run(agent, ids, mask):
    """Hikaye batch'i -> her gecisin log dagilimi (B, T-1, V): t. cumleye kadar okunmus durum, t+1'i tahmin eder."""
    state = agent.initial_state(len(ids))
    out = []
    for t in range(ids.shape[1] - 1):
        state = agent.read(state, ids[:, t], mask[:, t])
        out.append(agent.next_words(state))
    return torch.stack(out, 1)


def loss_of(agent, ids, mask):
    logp = run(agent, ids, mask)
    target, tmask = ids[:, 1:], mask[:, 1:]
    hit = logp.gather(2, target) * tmask
    return -hit.sum() / tmask.sum()


def evaluate(agent, ids, mask, unigram, stories, show=SHOW):
    """-> recall_at_k (ajan, siklik, hikayede gecenler), dagilim farki; goz satirlari yazilir."""
    with torch.no_grad():
        logp = torch.cat([run(agent, ids[c:c + 256], mask[c:c + 256]) for c in range(0, len(ids), 256)])
    V = logp.shape[-1]
    k_max = max(KS)
    freq_rank = unigram.argsort(descending=True)[:k_max].tolist()
    hits = {name: Counter() for name in ("ajan", "siklik", "hikaye")}
    total, probs = 0, []
    for n in range(len(ids)):
        sents = [ids[n, t][mask[n, t]].tolist() for t in range(ids.shape[1]) if mask[n, t].any()]
        story = Counter()
        for t in range(len(sents) - 1):
            story.update(sents[t])
            target = set(sents[t + 1]) - {0}
            top = logp[n, t].topk(k_max).indices.tolist()
            seen = [w for w, _ in sorted(story.items(), key=lambda x: (-x[1], -unigram[x[0]]))]
            ranked = {"ajan": top, "siklik": freq_rank, "hikaye": (seen + [w for w in freq_rank if w not in story])[:k_max]}
            for name, r in ranked.items():
                for k in KS:
                    hits[name][k] += len(target & set(r[:k]))
            total += len(target)
            probs.append(logp[n, t].exp())
    p = torch.stack(probs)
    spread = 0.5 * (p - p.mean(0)).abs().sum(1).mean().item()
    out = {name: {"recall_at_%d" % k: round(h[k] / total, 4) for k in KS} for name, h in hits.items()}
    out["dagilim_farki"] = round(spread, 4)
    lift = logp - unigram.clamp_min(1e-12).log()
    for n in range(min(show, len(ids))):
        T = int(mask[n].any(1).sum())
        print("   --- %s" % stories[n].get("country", n))
        for t in range(T - 1):
            words = lambda r: " ".join(agent.vocab[i] for i in r)
            print("   okunan: %s" % words(ids[n, t][mask[n, t]].tolist()))
            print("      en olasi : %s" % words(logp[n, t].topk(8).indices.tolist()))
            print("      yukselen : %s" % words(lift[n, t].topk(8).indices.tolist()))
            print("      gercek   : %s" % words(ids[n, t + 1][mask[n, t + 1]].tolist()))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="countries", choices=sorted(FILES))
    ap.add_argument("--root", default=None, help="veri klasoru (varsayilan data/<data>)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=D)
    ap.add_argument("--slots", type=int, default=SLOTS, help="durumun yuva sayisi")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--schedule", default=SCHEDULE, choices=("constant", "cosine"))
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--every", type=int, default=10, help="kac epokta bir olcum")
    ap.add_argument("--out", default=None, help="kosu klasoru: her epok checkpoint.pt, sonda agent.pt ve results.json")
    ap.add_argument("--resume", type=int, default=0, help="1: --out'taki checkpoint.pt'den kaldigi epoktan surdur")
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(4)
    folder, prefix = args.root or os.path.join(MODEL_Z, "data", args.data), FILES[args.data]
    vocab_path, stories_path = os.path.join(folder, prefix + "vocab.json"), os.path.join(folder, prefix + "stories.jsonl")
    vocab = json.load(open(vocab_path, encoding="utf-8"))
    agent = ContextAgent(vocab, d=args.d, slots=args.slots).to(args.device)
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr)

    def build():
        stories = [json.loads(line) for line in open(stories_path, encoding="utf-8")]
        train, exam = [s for s in stories if s["split"] == "train"], [s for s in stories if s["split"] == "exam"]
        return dict(train=_encode(agent, train), exam=_encode(agent, exam), n_train=len(train),
                    exam_names=[dict(country=s.get("country")) for s in exam])
    data = _cached(os.path.join(folder, prefix + "stories.pt"), (vocab_path, stories_path), build)
    train, exam = range(data["n_train"]), data["exam_names"]
    ids, mask = (t.to(args.device) for t in data["train"])
    exam_ids, exam_mask = (t.to(args.device) for t in data["exam"])
    count = torch.bincount(ids[mask], minlength=len(vocab)).float()
    unigram = ((count + 1) / (count + 1).sum()).cpu()
    gen = torch.Generator().manual_seed(args.seed)
    print("veri %s: egitim %d hikaye, sinav %d, sozluk %d, en cok %d cumle / %d kelime | d %d, slots %d, lr %g %s, "
          "batch %d, %d parametre | cihaz %s" % (
              args.data, len(train), len(exam), len(vocab), ids.shape[1], ids.shape[2], args.d, args.slots, args.lr,
              args.schedule, args.batch, sum(p.numel() for p in agent.parameters()), args.device), flush=True)
    t0, history, first = time.time(), [], 1
    ckpt = os.path.join(args.out, "checkpoint.pt") if args.out else None
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    if args.resume:
        pack = torch.load(ckpt, map_location=args.device, weights_only=False)
        assert pack["vocab"] == vocab, "sozluk checkpoint'tekinden farkli"
        keys = ("data", "seed", "d", "slots", "batch", "lr", "schedule") + (("epochs",) if args.schedule == "cosine" else ())
        diff = {k: (pack["args"].get(k), vars(args)[k]) for k in keys if pack["args"].get(k) != vars(args)[k]}
        assert not diff, "surdurme ayari checkpoint'ten farkli (checkpoint, simdi): %s" % diff
        agent.load_state_dict(pack["state"])
        opt.load_state_dict(pack["opt"])
        gen.set_state(pack["gen"].cpu())
        first, history = pack["epoch"] + 1, pack["history"]
        print("SURDURULDU: epok %d'den" % pack["epoch"], flush=True)
    for epoch in range(first, args.epochs + 1):
        perm = torch.randperm(len(train), generator=gen)
        total, t_epoch = 0.0, time.time()
        for b in range(0, len(train), args.batch):
            if args.schedule == "cosine":
                done = ((epoch - 1) * len(train) + b) / (args.epochs * len(train))
                for group in opt.param_groups:
                    group["lr"] = args.lr * 0.5 * (1 + math.cos(math.pi * done))
            rows = perm[b:b + args.batch].to(args.device)
            loss = loss_of(agent, ids[rows], mask[rows])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(rows)
        if epoch % args.every == 0 or epoch == args.epochs:
            agent.eval()
            last = epoch == args.epochs
            res = evaluate(agent, exam_ids, exam_mask, unigram, exam, show=SHOW if last else 0)
            agent.train()
            print("epok %3d  kayip %.3f  (%.0f sn, epok %.1f sn) | dagilim farki %.3f" % (
                epoch, total / len(train), time.time() - t0, time.time() - t_epoch, res["dagilim_farki"]), flush=True)
            for name in ("ajan", "siklik", "hikaye"):
                print("   %-7s %s" % (name, "  ".join("ilk%d %.3f" % (k, res[name]["recall_at_%d" % k]) for k in KS)),
                      flush=True)
            history.append(dict(epoch=epoch, loss=total / len(train), result=res))
        if ckpt:
            torch.save(dict(vocab=vocab, state=agent.state_dict(), opt=opt.state_dict(), gen=gen.get_state(),
                            epoch=epoch, history=history, args=vars(args)), ckpt + ".part")
            os.replace(ckpt + ".part", ckpt)
    if args.out:
        torch.save(dict(vocab=vocab, state=agent.state_dict(), args=vars(args)), os.path.join(args.out, "agent.pt"))
        json.dump(history, open(os.path.join(args.out, "results.json"), "w"), indent=1)
        print("kaydedildi:", args.out, flush=True)
    return agent


if __name__ == "__main__":
    main()
