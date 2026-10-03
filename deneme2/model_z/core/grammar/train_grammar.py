"""train_grammar -- gramer ajaninin egitimi ve olcumu (ajan: grammar.py).

Kayip: ardil (satir) + oncel (sutun) CE; ayni kelimenin kopyalari esdeger hedef.  Gorev ve yuva etiketi yok.
Olculer (sinav bolmeleri ve cumle boyu ayri): tam dogru (ayni torbadan kurulabilen herhangi gecerli cumle), dogru yuva,
dogru komsu, deneme sayisi (Gumbel gurultulu G ile dogru bulunana kadar, en cok TRIES), cumle basina dizme suresi.

    python train_grammar.py [--data countries|simplestories] [--epochs 30] [--device cpu|cuda] [--d 64] [--lr 3e-3]
                            [--schedule constant|cosine] [--compile 1] [--precision bf16] [--out klasor] [--resume 1]
"""
import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict

import numpy as np
import torch

from grammar import D, NEG, UNK, GrammarAgent, order_by_relation

MODEL_Z = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILES = {"countries": "country_", "simplestories": "ss_"}   # data/<ad>/<onek>{train,exam,sentences}.jsonl
LR = 3e-3               # Adam; d=256'da bu degerle 6. epoktan sonra dagildi
SCHEDULE = "constant"   # constant | cosine (adim adim --epochs sonunda 0)
UNK_RATE = 0.1          # egitimde kelimenin <unk> yapilma olasiligi: bilinmeyen kelimeye yer bulmayi da ogrensin
TRIES = 100             # deneme sayisi olcusunun ust siniri
LENGTH_BANDS = ((1, 10), (11, 20), (21, 1000))   # olculer cumle boyuna gore de


def _load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")] if os.path.exists(path) else []


def _encode(agent, rows):
    """Cumleler -> bir kez: ids (N, L) dolgulu, boy (N,), kelimeler."""
    L = max(len(r["words"]) for r in rows)
    ids = torch.zeros(len(rows), L, dtype=torch.long)
    for b, r in enumerate(rows):
        ids[b, :len(r["words"])] = torch.tensor(agent.ids(r["words"]))
    return dict(ids=ids, length=torch.tensor([len(r["words"]) for r in rows]), words=[r["words"] for r in rows])


def _batch(enc, rows, gen):
    """Satirlar -> karisik torbalar (sinav).  order[b, i] = torbadaki i. kelimenin cumledeki konumu."""
    L = enc["ids"].shape[1]
    mask = torch.arange(L)[None] < enc["length"][rows][:, None]
    order = torch.rand(len(rows), L, generator=gen).masked_fill(~mask, 2.0).argsort(1)
    return enc["ids"][rows].gather(1, order), mask, order


def loss_of(agent, enc, rows, gen, unk_rate, forward=None, bf16=False):
    """Egitim torbasi cumlenin kendisi: okuyucu konumsuz, G kelimelerin veriliş sirasiyla birlikte permute olur, kayip
    sirasizdir; karistirmak bir sey degistirmez.  Hedefler sabit: ardil i+1, oncel i-1, boundary (indeks L) ilk / son
    kelimeye.  enc, rows ve gen ajanin cihazinda; dolgu batch'in en uzun cumlesine (8'e yuvarli) kirpilir.
    forward: agent'in derlenmis hali (torch.compile) ya da agent; bf16: ileri hesap autocast (GPU)."""
    dev = next(agent.parameters()).device
    length = enc["length"][rows]
    L = min(-(-int(length.max()) // 8) * 8, enc["ids"].shape[1])   # 8'in kati: torch.compile her boyda yeniden derlemesin
    orig = enc["ids"][rows, :L]
    pos = torch.arange(L, device=dev)[None]
    mask = pos < length[:, None]
    ids = orig.masked_fill((torch.rand(orig.shape, generator=gen, device=dev) < unk_rate) & mask, agent.index[UNK])
    succ = torch.cat([torch.where(pos + 1 < length[:, None], pos + 1, L), torch.zeros_like(length)[:, None]], 1)
    pred = torch.cat([torch.where(pos > 0, pos - 1, L).expand(len(rows), -1), (length - 1)[:, None]], 1)
    key = torch.cat([orig.masked_fill(~mask, -5), torch.full_like(length, -7)[:, None]], 1)
    same = key[:, :, None] == key[:, None, :]           # ayni kelimenin kopyalari esdeger hedef
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bf16):
        G = (agent if forward is None else forward)(ids, mask)
    G = G.float()
    L1 = G.shape[1]
    node = torch.cat([mask, torch.ones(len(ids), 1, dtype=torch.bool, device=dev)], 1).float()
    hit_s = same.gather(1, succ[:, :, None].expand(-1, -1, L1))     # j, gercek ardille ayni kelime
    hit_p = same.gather(1, pred[:, :, None].expand(-1, -1, L1))
    ls = G.log_softmax(2).masked_fill(~hit_s, NEG).logsumexp(2)                    # ardil (satir)
    lp = G.log_softmax(1).transpose(1, 2).masked_fill(~hit_p, NEG).logsumexp(2)   # oncel (sutun)
    return -((ls + lp) * node).sum() / node.sum()


def evaluate(agent, enc, valid, seed=0):
    """-> butun ve cumle boyu bantlari: tam dogru, dogru yuva, dogru komsu, deneme (en cok TRIES; bulunamayan TRIES + 1),
    dizme ms.  Butun sinav parca parca tek ileri hesapla."""
    gen = torch.Generator().manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    dev = next(agent.parameters()).device
    rows = torch.arange(len(enc["words"]))
    stats = defaultdict(Counter)
    t_order = 0.0
    for c in range(0, len(rows), 512):
        part = rows[c:c + 512]
        ids, mask, order = _batch(enc, part, gen)
        with torch.no_grad():
            G = agent(ids.to(dev), mask.to(dev)).cpu().double().numpy()
        L = ids.shape[1]
        for b, r in enumerate(part.tolist()):
            words = enc["words"][r]
            n = len(words)
            bag = [words[o] for o in order[b, :n].tolist()]
            idx = list(range(n)) + [L]
            g = G[b][np.ix_(idx, idx)]
            t = time.perf_counter()
            built = tuple(bag[i] for i in order_by_relation(g))
            t_order += time.perf_counter() - t
            options = valid.get(tuple(sorted(words)), {tuple(words)})
            best = max(options, key=lambda o: sum(a == x for a, x in zip(built, o)))
            k, found = 1, built in options
            noise = np_rng.gumbel(size=(TRIES,) + g.shape) if not found else None
            while not found and k < TRIES:
                found = tuple(bag[i] for i in order_by_relation(g + noise[k])) in options
                k += 1
            band = next(f"{lo}-{hi}" if hi < 1000 else f"{lo}+" for lo, hi in LENGTH_BANDS if lo <= n <= hi)
            for key in ("all", band):
                s = stats[key]
                s["n"] += 1
                s["exact"] += built in options
                s["slot"] += sum(a == x for a, x in zip(built, best))
                s["slots"] += n
                s["pairs"] += sum((Counter(zip(built, built[1:])) & Counter(zip(best, best[1:]))).values())
                s["n_pairs"] += n - 1
                s["tries"] += k if found else TRIES + 1
    out = {}
    for key, s in stats.items():
        out[key] = dict(n=s["n"], exact=round(s["exact"] / s["n"], 4), slot=round(s["slot"] / s["slots"], 4),
                        neighbor=round(s["pairs"] / max(s["n_pairs"], 1), 4), tries=round(s["tries"] / s["n"], 2))
    out["all"]["ms"] = round(1000 * t_order / max(len(rows), 1), 3)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="countries", choices=sorted(FILES))
    ap.add_argument("--root", default=None, help="veri klasoru (varsayilan data/<data>; SS: Drive'daki hazir dosyalar)")
    ap.add_argument("--epochs", type=int, default=30)          # ulke verisinde 60 ile ayni sonuc (olculdu)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=D, help="kelime temsili boyu")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--schedule", default=SCHEDULE, choices=("constant", "cosine"))
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--compile", type=int, default=1, help="1: torch.compile (yalniz cuda; CPU'da kendiliginden kapali)")
    ap.add_argument("--precision", default="bf16", choices=("bf16", "fp32"), help="egitim ileri hesabi (yalniz cuda); "
                    "sinav hep fp32")
    ap.add_argument("--every", type=int, default=10, help="kac epokta bir olcum")
    ap.add_argument("--out", default=None, help="kosu klasoru: her epok checkpoint.pt (ajan, optimizer, epok, rastgelelik), "
                    "sonda agent.pt; olcumler results.json")
    ap.add_argument("--resume", type=int, default=0, help="1: --out'taki checkpoint.pt'den kaldigi epoktan surdur")
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(4)                        # kucuk model: 16 is parcacigi 4'ten yavas (olculdu)
    folder, prefix = args.root or os.path.join(MODEL_Z, "data", args.data), FILES[args.data]
    train = _load(os.path.join(folder, prefix + "train.jsonl"))
    exam = _load(os.path.join(folder, prefix + "exam.jsonl"))
    valid = defaultdict(set)                            # ayni torbadan kurulabilen gecerli cumleler
    for r in _load(os.path.join(folder, prefix + "sentences.jsonl")) or train + exam:
        valid[tuple(sorted(r["words"]))].add(tuple(r["words"]))
    vocab = [UNK] + sorted({w for r in train for w in r["words"]})
    agent = GrammarAgent(vocab, d=args.d).to(args.device)
    cuda = torch.device(args.device).type == "cuda"
    forward = torch.compile(agent, dynamic=True) if (cuda and args.compile) else agent
    bf16 = cuda and args.precision == "bf16"
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr)
    splits = {s: _encode(agent, [r for r in exam if r["split"] == s]) for s in sorted({r["split"] for r in exam})}
    enc = {k: v.to(args.device) for k, v in _encode(agent, train).items() if k != "words"}   # egitim verisi bir kez cihazda
    gen = torch.Generator().manual_seed(args.seed)                     # epok sirasi (CPU)
    gen_unk = torch.Generator(device=args.device).manual_seed(args.seed)   # <unk> secimi (cihazda)
    print("veri %s: egitim %d cumle, sozluk %d, sinav %s, en uzun %d kelime | d %d, lr %g %s, %d parametre | cihaz %s "
          "compile %s %s" % (args.data, len(train), len(vocab), {s: len(e["words"]) for s, e in splits.items()},
                             enc["ids"].shape[1], args.d, args.lr, args.schedule, sum(p.numel() for p in agent.parameters()), args.device, forward is not agent,
             "bf16" if bf16 else "fp32"), flush=True)
    t0 = time.time()
    res, history, first = None, [], 1
    ckpt = os.path.join(args.out, "checkpoint.pt") if args.out else None
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    if args.resume:
        pack = torch.load(ckpt, map_location=args.device, weights_only=False)
        assert pack["vocab"] == vocab, "sozluk checkpoint'tekinden farkli: ayni veriyle surdurulur"
        agent.load_state_dict(pack["state"])
        opt.load_state_dict(pack["opt"])
        gen.set_state(pack["gen"])
        gen_unk.set_state(pack["gen_unk"].cpu())
        first, history = pack["epoch"] + 1, pack["history"]
        print("SURDURULDU: epok %d'den (%s)" % (pack["epoch"], ckpt), flush=True)
    for epoch in range(first, args.epochs + 1):
        perm = torch.randperm(len(train), generator=gen).to(args.device)
        total, t_epoch = torch.zeros((), device=args.device), time.time()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        for b in range(0, len(train), args.batch):
            if args.schedule == "cosine":                     # adim epok ve batch'ten: surdurmede ayni lr
                done = ((epoch - 1) * len(train) + b) / (args.epochs * len(train))
                for group in opt.param_groups:
                    group["lr"] = args.lr * 0.5 * (1 + math.cos(math.pi * done))
            rows = perm[b:b + args.batch]
            loss = loss_of(agent, enc, rows, gen_unk, UNK_RATE, forward, bf16)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.detach() * len(rows)                # .item() yok: her adimda GPU beklenmez
        total = total.item()
        if cuda:
            torch.cuda.synchronize()
        perf = dict(train_secs=round(time.time() - t_epoch, 2), sentences_per_sec=round(len(train) / (time.time() - t_epoch)),
                    gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else None)
        if epoch % args.every == 0 or epoch == args.epochs:
            agent.eval()
            t_eval = time.time()
            res = {s: evaluate(agent, e, valid) for s, e in splits.items()}
            perf["eval_secs"] = round(time.time() - t_eval, 1)
            agent.train()
            print("epok %3d  kayip %.3f  (%.0f sn) | egitim %.1f sn/epok, %d cumle/sn%s | sinav %.1f sn" % (
                epoch, total / len(train), time.time() - t0, perf["train_secs"], perf["sentences_per_sec"],
                (", gpu tepe %.2f GB" % perf["gpu_peak_gb"]) if cuda else "", perf["eval_secs"]), flush=True)
            for s, v in res.items():
                print("   %-8s %s" % (s, "  ".join(
                    "%s: n %d tam %.3f yuva %.3f komsu %.3f deneme %.1f%s" % (
                        k, x["n"], x["exact"], x["slot"], x["neighbor"], x["tries"],
                        (" %.2f ms" % x["ms"]) if "ms" in x else "") for k, x in v.items())), flush=True)
            history.append(dict(epoch=epoch, loss=total / len(train), result=res, perf=perf))
        if ckpt:                                         # her epok: once .part, sonra yerine (yarim dosya kalmaz)
            torch.save(dict(vocab=vocab, state=agent.state_dict(), opt=opt.state_dict(), gen=gen.get_state(),
                            gen_unk=gen_unk.get_state(), epoch=epoch, history=history, args=vars(args)), ckpt + ".part")
            os.replace(ckpt + ".part", ckpt)
    if args.out:
        torch.save(dict(vocab=vocab, state=agent.state_dict(), args=vars(args)), os.path.join(args.out, "agent.pt"))
        json.dump(history, open(os.path.join(args.out, "results.json"), "w"), indent=1)
        print("kaydedildi:", args.out, flush=True)
    return agent, res


if __name__ == "__main__":
    main()
