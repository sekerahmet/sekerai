"""grammar -- Model Z gramer ajani (kullanici, 3 Ekim: "Gramer ajanının temel görevi verilen tüm kelimelerden anlamlı bir
cümle kurabilmesi"; "Next token mantığında değil tek seferde ... Yuvalar doğru olacak"; adlar onayli).

Girdi: cumlenin kelimeleri, karisik (torba).  Cikti: her kelimeye bir yuva (konum), butun cumle bir anda.
    E            kelime temsili (sozluk x d); egitimde gecmeyen kelime <unk>
    torba okuyucu   konumsuz attention (TransformerEncoder): h_i, kelimenin torbadaki baglami; girdi sirasindan bagimsiz
    yuva matrisi    P[i, k] = <h_i, q_k>, q_k = bastan k. yuva + sondan (n - 1 - k). yuva (cumle boyu degisir)
    relation_matrix G[i, j] = h_i^T W h_j: "j, i'nin hemen ardindan gelir"
    assign_slots    satir ve sutun log olasiliklarinin toplami uzerinde Macar algoritmasi: her kelime bir yuvaya, her
                    yuvaya bir kelime, tek seferde
Kayip: yuva (satir: kelimenin yuvasi, sutun: yuvanin kelimesi; ayni kelime birden fazlaysa yuvalari esdeger) + komsu
(G'de gercek ardil).  Gorev etiketi yok.
Olculer (sinav bolmeleri ayri): tam dogru (ayni torbadan kurulabilen herhangi bir gecerli cumle), dogru yuva, dogru komsu,
deneme sayisi (Gumbel gurultulu atamalarla dogru bulunana kadar, en cok TRIES).

    python grammar.py [--epochs 60] [--seed 0]
"""
import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "countries")
UNK = "<unk>"
D = 64                  # kelime temsili boyu
HEADS = 4
LAYERS = 3
MAX_SLOTS = 16          # en uzun cumle 12 kelime
UNK_RATE = 0.1          # egitimde kelimenin <unk> yapilma olasiligi: bilinmeyen kelimeye yer bulmayi da ogrensin
TRIES = 100             # deneme sayisi olcusunun ust siniri


def _load(name):
    return [json.loads(line) for line in open(os.path.join(DATA, name), encoding="utf-8")]


class GrammarAgent(torch.nn.Module):
    def __init__(self, vocab, d=D, heads=HEADS, layers=LAYERS):
        super().__init__()
        self.vocab = vocab
        self.index = {w: i for i, w in enumerate(vocab)}
        self.E = torch.nn.Embedding(len(vocab), d)
        layer = torch.nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True)
        self.reader = torch.nn.TransformerEncoder(layer, layers)
        self.slot_from_start = torch.nn.Parameter(torch.randn(MAX_SLOTS, d) / d ** 0.5)
        self.slot_from_end = torch.nn.Parameter(torch.randn(MAX_SLOTS, d) / d ** 0.5)
        self.W = torch.nn.Parameter(torch.randn(d, d) / d ** 0.5)

    def ids(self, words):
        return [self.index.get(w, self.index[UNK]) for w in words]

    def forward(self, ids, mask):
        """ids (B, n) karisik torba, mask (B, n) gercek kelime -> yuva log puanlari (B, n kelime, n yuva) ve
        relation_matrix (B, n, n)."""
        h = self.reader(self.E(ids), src_key_padding_mask=~mask)
        n = mask.sum(1)
        k = torch.arange(ids.shape[1])
        from_end = (n[:, None] - 1 - k[None]).clamp(min=0, max=MAX_SLOTS - 1)
        q = self.slot_from_start[k][None] + self.slot_from_end[from_end]                    # (B, n yuva, d)
        P = torch.einsum("bid,bkd->bik", h, q) / h.shape[-1] ** 0.5
        G = torch.einsum("bid,de,bje->bij", h, self.W, h) / h.shape[-1] ** 0.5
        both = mask[:, :, None] & mask[:, None, :]
        return P.masked_fill(~both, -1e9), G.masked_fill(~both, -1e9)


def relation_matrix(agent, words):
    """Tek torba -> G (n x n): G[i, j] "j, i'nin hemen ardindan gelir" puani (okuma icin)."""
    with torch.no_grad():
        _, G = agent(torch.tensor([agent.ids(words)]), torch.ones(1, len(words), dtype=torch.bool))
    return G[0]


def assign_slots(P, noise=None):
    """P (n kelime, n yuva) -> her kelimenin yuvasi.  Satir (kelime -> yuva) ve sutun (yuva -> kelime) log olasiliklari
    toplanir, Macar algoritmasi toplam puani en buyuk tek atamayi verir.  noise: Gumbel gurultusu (deneme sayisi)."""
    score = P.log_softmax(1) + P.log_softmax(0)
    if noise is not None:
        score = score + noise
    rows, cols = linear_sum_assignment(-score.detach().numpy())
    slot = [0] * len(rows)
    for r, c in zip(rows, cols):
        slot[r] = int(c)
    return slot


def _batch(agent, rows, rng, unk_rate=0.0):
    """Cumleler -> karisik torbalar (ids, mask), yuva hedefi (B, n, n; ayni kelimenin yuvalari esdeger), ardil hedefi."""
    n = max(len(r["words"]) for r in rows)
    ids = torch.zeros(len(rows), n, dtype=torch.long)
    mask = torch.zeros(len(rows), n, dtype=torch.bool)
    slot_t = torch.zeros(len(rows), n, n)
    next_t = torch.full((len(rows), n), -1, dtype=torch.long)
    for b, r in enumerate(rows):
        words = r["words"]
        order = list(range(len(words)))
        rng.shuffle(order)                              # torbadaki i. kelime cumlenin order[i]. kelimesi
        where = {o: i for i, o in enumerate(order)}
        w_ids = agent.ids(words)
        for i, o in enumerate(order):
            ids[b, i] = agent.index[UNK] if rng.random() < unk_rate else w_ids[o]
        mask[b, :len(words)] = True
        same = defaultdict(list)
        for pos, w in enumerate(words):
            same[w].append(pos)
        for i, o in enumerate(order):
            slots = same[words[o]]
            slot_t[b, i, slots] = 1.0 / len(slots)
            if o + 1 < len(words):
                next_t[b, i] = where[o + 1]
    return ids, mask, slot_t, next_t


def loss_of(agent, rows, rng, unk_rate):
    ids, mask, slot_t, next_t = _batch(agent, rows, rng, unk_rate)
    P, G = agent(ids, mask)
    rowm = mask.float()
    slot_loss = -((slot_t * P.log_softmax(2)).sum(2) * rowm).sum() / rowm.sum()                 # kelime -> yuva
    col_t = slot_t / slot_t.sum(1, keepdim=True).clamp(min=1e-9)
    slot_loss = slot_loss - ((col_t * P.log_softmax(1)).sum(1) * rowm).sum() / rowm.sum()       # yuva -> kelime
    has = next_t >= 0
    next_loss = F.cross_entropy(G[has], next_t[has])
    return slot_loss + next_loss


def evaluate(agent, rows, valid, seed=0):
    """-> tam dogru, dogru yuva, dogru komsu, ortalama deneme (en cok TRIES; bulunamayan TRIES + 1 sayilir)."""
    rng = random.Random(seed)
    g = torch.Generator().manual_seed(seed)
    exact = slots = pairs = tries = 0
    n_slots = n_pairs = 0
    for r in rows:
        words = list(r["words"])
        rng.shuffle(words)
        with torch.no_grad():
            P, _ = agent(torch.tensor([agent.ids(words)]), torch.ones(1, len(words), dtype=torch.bool))
        P = P[0]
        options = valid[tuple(sorted(r["words"]))]

        def sentence(slot):
            out = [None] * len(words)
            for i, s in enumerate(slot):
                out[s] = words[i]
            return tuple(out)

        built = sentence(assign_slots(P))
        best = max(options, key=lambda o: sum(a == b for a, b in zip(built, o)))
        exact += built in options
        slots += sum(a == b for a, b in zip(built, best))
        n_slots += len(best)
        gold = set(zip(best, best[1:]))
        pairs += sum(p in gold for p in zip(built, built[1:]))
        n_pairs += len(best) - 1
        k, found = 1, built in options
        while not found and k < TRIES:
            u = torch.rand(P.shape, generator=g).clamp(1e-9, 1 - 1e-9)
            k += 1
            found = sentence(assign_slots(P, -torch.log(-torch.log(u)))) in options
        tries += k if found else TRIES + 1
    n = max(len(rows), 1)
    return dict(n=len(rows), exact=round(exact / n, 4), slot=round(slots / max(n_slots, 1), 4),
                neighbor=round(pairs / max(n_pairs, 1), 4), tries=round(tries / n, 2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    train, exam = _load("country_train.jsonl"), _load("country_exam.jsonl")
    valid = defaultdict(set)                            # ayni torbadan kurulabilen gecerli cumleler (butun veri)
    for r in _load("country_sentences.jsonl"):
        valid[tuple(sorted(r["words"]))].add(tuple(r["words"]))
    vocab = [UNK] + sorted({w for r in train for w in r["words"]})
    agent = GrammarAgent(vocab)
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr)
    splits = {s: [r for r in exam if r["split"] == s] for s in ("seen", "unseen_fact")}
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train)
        total = 0.0
        for b in range(0, len(train), args.batch):
            loss = loss_of(agent, train[b:b + args.batch], rng, UNK_RATE)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(train[b:b + args.batch])
        if epoch % 10 == 0 or epoch == args.epochs:
            agent.eval()
            res = {s: evaluate(agent, rows, valid) for s, rows in splits.items()}
            agent.train()
            print("epok %3d  kayip %.3f  |  %s  (%.0f sn)" % (epoch, total / len(train), "  ".join(
                "%s: tam %.3f yuva %.3f komsu %.3f deneme %.1f" % (s, v["exact"], v["slot"], v["neighbor"], v["tries"])
                for s, v in res.items()), time.time() - t0), flush=True)
    return agent, res


if __name__ == "__main__":
    main()
