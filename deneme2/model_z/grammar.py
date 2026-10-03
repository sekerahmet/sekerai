"""grammar -- Model Z gramer ajani (kullanici, 3 Ekim: "Gramer ajanının temel görevi verilen tüm kelimelerden anlamlı bir
cümle kurabilmesi"; "Next token mantığında değil tek seferde"; "ilişki matrisi gramerin öğrenerek hem değiştirdiği hem de
kullandığı birşey"; "Emin olduklarımızı yazalım"; adlar onayli).  Matematik: belge/model_z_temel/03_gramer_matematik.md.

Girdi: cumlenin kelimeleri, karisik (torba).  Cikti: butun cumle bir anda.
    E               kelime temsili (sozluk x d); egitimde gecmeyen kelime <unk>
    boundary        sinir dugumu: cumlenin basi ve sonu, torbaya eklenir
    torba okuyucu   konumsuz attention (TransformerEncoder): h_i, kelimenin torbadaki baglami; girdi sirasindan bagimsiz
    relation_matrix G[i, j] = h_i^T W h_j + e_i^T U e_j: "j, i'nin hemen ardindan gelir"; ilk terim baglamli (kalip),
                    ikincisi sozcuksel (kelimenin kendisi: ad ici sira).  boundary'nin satiri cumlenin ilk kelimesi, sutunu
                    son kelimesi
    order_by_relation  cumle = G uzerinde boundary'den gecen tek cevrim: her dugume bir ardil (Macar atamasi, tek seferde);
                    atama birden fazla cevrim verirse patch_cycles birlestirir (Karp yamasi).  Maliyet n^3
Kayip: ardil (satir) + oncel (sutun) CE; ayni kelimenin kopyalari esdeger hedef.  Gorev ve yuva etiketi yok.
Olculer (sinav bolmeleri ve cumle boyu ayri): tam dogru (ayni torbadan kurulabilen herhangi gecerli cumle), dogru yuva,
dogru komsu, deneme sayisi (Gumbel gurultulu G ile dogru bulunana kadar, en cok TRIES), cumle basina dizme suresi.

    python grammar.py [--data countries|simplestories] [--epochs 30] [--device cpu|cuda] [--compile 1] [--precision bf16]
"""
import argparse
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {"countries": "country_", "simplestories": "ss_"}   # data/<ad>/<onek>{train,exam,sentences}.jsonl
UNK = "<unk>"
D = 64                  # kelime temsili boyu
HEADS = 4
LAYERS = 3
UNK_RATE = 0.1          # egitimde kelimenin <unk> yapilma olasiligi: bilinmeyen kelimeye yer bulmayi da ogrensin
TRIES = 100             # deneme sayisi olcusunun ust siniri
NEG = -1e9
LENGTH_BANDS = ((1, 10), (11, 20), (21, 1000))   # olculer cumle boyuna gore de


def _load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")] if os.path.exists(path) else []


class GrammarAgent(torch.nn.Module):
    def __init__(self, vocab, d=D, heads=HEADS, layers=LAYERS):
        super().__init__()
        self.vocab = vocab
        self.index = {w: i for i, w in enumerate(vocab)}
        self.E = torch.nn.Embedding(len(vocab), d)
        self.boundary = torch.nn.Parameter(torch.randn(d) / d ** 0.5)
        layer = torch.nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True)
        self.reader = torch.nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.W = torch.nn.Parameter(torch.randn(d, d) / d ** 0.5)      # baglamli bag (kalip)
        self.U = torch.nn.Parameter(torch.randn(d, d) / d ** 0.5)      # sozcuksel bag (ad ici sira: Phnom -> Penh)

    def ids(self, words):
        return [self.index.get(w, self.index[UNK]) for w in words]

    def forward(self, ids, mask):
        """ids (B, L) karisik torba, mask (B, L) gercek kelime -> G (B, L+1, L+1); son dugum boundary.  Kendine bag ve
        dolgu NEG."""
        B, L = ids.shape
        e = torch.cat([self.E(ids), self.boundary.expand(B, 1, -1)], 1)
        m = torch.cat([mask, torch.ones(B, 1, dtype=torch.bool, device=ids.device)], 1)
        h = self.reader(e, src_key_padding_mask=~m)
        G = (torch.einsum("bid,de,bje->bij", h, self.W, h) + torch.einsum("bid,de,bje->bij", e, self.U, e)) / e.shape[-1] ** 0.5
        ok = m[:, :, None] & m[:, None, :] & ~torch.eye(L + 1, dtype=torch.bool, device=ids.device)[None]
        return G.masked_fill(~ok, NEG)


def relation_matrix(agent, words):
    """Tek torba -> G ((n+1) x (n+1), son satir / sutun boundary) (okuma icin)."""
    dev = next(agent.parameters()).device
    with torch.no_grad():
        return agent(torch.tensor([agent.ids(words)], device=dev), torch.ones(1, len(words), dtype=torch.bool,
                                                                                device=dev))[0].cpu()


def order_by_relation(G):
    """G ((n+1) x (n+1), son dugum boundary; numpy) -> kelime sirasi (torba indeksleri).  Her dugume bir ardil: Macar
    atamasi toplam bag puani en buyuk eslemeyi tek seferde verir; birden fazla cevrim cikarsa patch_cycles."""
    n = G.shape[0] - 1
    rows, cols = linear_sum_assignment(-G)
    succ = dict(zip(rows.tolist(), cols.tolist()))
    succ = patch_cycles(G, succ)
    seq, x = [], succ[n]
    while x != n:
        seq.append(x)
        x = succ[x]
    return seq


def patch_cycles(G, succ):
    """Atamanin cevrimleri tek cevrime (Karp yamasi): her adimda iki cevrimden birer bag (a -> b, c -> d) a -> d, c -> b
    olur; toplam puandaki kaybi en kucuk degisim secilir."""
    seen, cycles = set(), []
    for s in range(G.shape[0]):
        if s in seen:
            continue
        cyc, x = [], s
        while x not in seen:
            seen.add(x)
            cyc.append(x)
            x = succ[x]
        cycles.append(cyc)
    while len(cycles) > 1:
        best = None
        for ai in range(len(cycles)):
            for bi in range(ai + 1, len(cycles)):
                for a in cycles[ai]:
                    for c in cycles[bi]:
                        b, d = succ[a], succ[c]
                        delta = G[a, d] + G[c, b] - G[a, b] - G[c, d]
                        if best is None or delta > best[0]:
                            best = (delta, a, c, ai, bi)
        _, a, c, ai, bi = best
        succ[a], succ[c] = succ[c], succ[a]
        merged, x = [], a
        while True:
            merged.append(x)
            x = succ[x]
            if x == a:
                break
        cycles = [cy for i, cy in enumerate(cycles) if i not in (ai, bi)] + [merged]
    return succ


def _encode(agent, rows):
    """Cumleler -> bir kez: ids (N, L) dolgulu, boy (N,), kelimeler."""
    L = max(len(r["words"]) for r in rows)
    ids = torch.zeros(len(rows), L, dtype=torch.long)
    for b, r in enumerate(rows):
        ids[b, :len(r["words"])] = torch.tensor(agent.ids(r["words"]))
    return dict(ids=ids, length=torch.tensor([len(r["words"]) for r in rows]), words=[r["words"] for r in rows])


def _batch(enc, rows, gen, unk_rate=0.0, unk_id=0):
    """Satirlar -> karisik torbalar ve hedefler (dongusuz).  order[b, i] = torbadaki i. kelimenin cumledeki konumu.
    succ / pred: her dugumun (boundary dahil, indeks L) gercek ardili / oncelisi; same[b, i, j]: i ve j ayni kelime."""
    L = enc["ids"].shape[1]
    length = enc["length"][rows]
    mask = torch.arange(L)[None] < length[:, None]
    order = torch.rand(len(rows), L, generator=gen).masked_fill(~mask, 2.0).argsort(1)
    orig = enc["ids"][rows].gather(1, order)
    ids = orig.masked_fill((torch.rand(orig.shape, generator=gen) < unk_rate) & mask, unk_id)
    inv = order.argsort(1)                              # cumledeki konum -> torbadaki yer
    nxt = order + 1
    succ = torch.where((nxt < length[:, None]) & mask, inv.gather(1, nxt.clamp(max=L - 1)), torch.full_like(nxt, L))
    prv = order - 1
    pred = torch.where((order > 0) & mask, inv.gather(1, prv.clamp(min=0)), torch.full_like(prv, L))
    first, last = inv[:, 0], inv.gather(1, (length - 1)[:, None])[:, 0]
    succ = torch.cat([succ, first[:, None]], 1)          # boundary'nin ardili ilk kelime
    pred = torch.cat([pred, last[:, None]], 1)           # boundary'nin oncelisi son kelime
    key = torch.cat([orig.masked_fill(~mask, -5), torch.full((len(rows), 1), -7)], 1)
    same = key[:, :, None] == key[:, None, :]
    return ids, mask, succ, pred, same, order


def loss_of(agent, enc, rows, gen, unk_rate, forward=None, bf16=False):
    """forward: agent'in derlenmis hali (torch.compile) ya da agent; bf16: ileri hesap autocast (GPU)."""
    ids, mask, succ, pred, same, _ = _batch(enc, rows, gen, unk_rate, agent.index[UNK])
    dev = next(agent.parameters()).device
    ids, mask, succ, pred, same = (t.to(dev, non_blocking=True) for t in (ids, mask, succ, pred, same))
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
        ids, mask, _, _, _, order = _batch(enc, part, gen)
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
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
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
    folder, prefix = args.root or os.path.join(HERE, "data", args.data), FILES[args.data]
    train = _load(os.path.join(folder, prefix + "train.jsonl"))
    exam = _load(os.path.join(folder, prefix + "exam.jsonl"))
    valid = defaultdict(set)                            # ayni torbadan kurulabilen gecerli cumleler
    for r in _load(os.path.join(folder, prefix + "sentences.jsonl")) or train + exam:
        valid[tuple(sorted(r["words"]))].add(tuple(r["words"]))
    vocab = [UNK] + sorted({w for r in train for w in r["words"]})
    agent = GrammarAgent(vocab).to(args.device)
    cuda = torch.device(args.device).type == "cuda"
    forward = torch.compile(agent, dynamic=True) if (cuda and args.compile) else agent
    bf16 = cuda and args.precision == "bf16"
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr)
    splits = {s: _encode(agent, [r for r in exam if r["split"] == s]) for s in sorted({r["split"] for r in exam})}
    enc = _encode(agent, train)
    gen = torch.Generator().manual_seed(args.seed)
    print("veri %s: egitim %d cumle, sozluk %d, sinav %s, en uzun %d kelime | cihaz %s compile %s %s" % (
        args.data, len(train), len(vocab), {s: len(e["words"]) for s, e in splits.items()}, enc["ids"].shape[1],
        args.device, forward is not agent, "bf16" if bf16 else "fp32"), flush=True)
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
        first, history = pack["epoch"] + 1, pack["history"]
        print("SURDURULDU: epok %d'den (%s)" % (pack["epoch"], ckpt), flush=True)
    for epoch in range(first, args.epochs + 1):
        perm = torch.randperm(len(train), generator=gen)
        total, t_epoch = 0.0, time.time()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        for b in range(0, len(train), args.batch):
            rows = perm[b:b + args.batch]
            loss = loss_of(agent, enc, rows, gen, UNK_RATE, forward, bf16)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(rows)
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
                            epoch=epoch, history=history, args=vars(args)), ckpt + ".part")
            os.replace(ckpt + ".part", ckpt)
    if args.out:
        torch.save(dict(vocab=vocab, state=agent.state_dict(), args=vars(args)), os.path.join(args.out, "agent.pt"))
        json.dump(history, open(os.path.join(args.out, "results.json"), "w"), indent=1)
        print("kaydedildi:", args.out, flush=True)
    return agent, res


if __name__ == "__main__":
    main()
