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

    python grammar.py [--epochs 30] [--seed 0]
"""
import argparse
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np
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
    toplanir, Macar algoritmasi toplam puani en buyuk tek atamayi verir.  noise: Gumbel gurultusu (deneme sayisi);
    P numpy ise hazir puan sayilir (denemelerde log_softmax bir kez)."""
    score = P if isinstance(P, np.ndarray) else (P.log_softmax(1) + P.log_softmax(0)).detach().numpy()
    if noise is not None:
        score = score + noise
    rows, cols = linear_sum_assignment(-score)
    slot = [0] * len(rows)
    for r, c in zip(rows, cols):
        slot[r] = int(c)
    return slot


def _encode(agent, rows):
    """Cumleler -> bir kez: ids (N, L) dolgulu, boy (N,), ayni (N, L, L): ayni kelimenin yuvalari esdeger
    (ayni[p, q] = 1/m, p ve q konumlarinda ayni kelime, m o kelimenin sayisi)."""
    L = max(len(r["words"]) for r in rows)
    ids = torch.zeros(len(rows), L, dtype=torch.long)
    same = torch.zeros(len(rows), L, L)
    for b, r in enumerate(rows):
        w = r["words"]
        ids[b, :len(w)] = torch.tensor(agent.ids(w))
        for p, a in enumerate(w):
            hits = [q for q, c in enumerate(w) if c == a]
            same[b, p, hits] = 1.0 / len(hits)
    return dict(ids=ids, length=torch.tensor([len(r["words"]) for r in rows]), same=same,
                words=[r["words"] for r in rows])


def _shuffle(enc, rows, gen):
    """Satir basina rastgele siralama (dolgu sonda): order[b, i] = torbadaki i. kelimenin cumledeki konumu."""
    L = enc["ids"].shape[1]
    valid = torch.arange(L)[None] < enc["length"][rows, None]
    keys = torch.rand(len(rows), L, generator=gen).masked_fill(~valid, 2.0)
    return keys.argsort(1), valid


def _batch(enc, rows, gen, unk_rate=0.0, unk_id=0):
    """Satirlar -> karisik torbalar (ids, mask), yuva hedefi (B, L, L), ardil hedefi (B, L; yok: -1).  Dongusuz."""
    order, mask = _shuffle(enc, rows, gen)
    L = order.shape[1]
    ids = enc["ids"][rows].gather(1, order)
    ids = ids.masked_fill((torch.rand(ids.shape, generator=gen) < unk_rate) & mask, unk_id)
    slot_t = enc["same"][rows].gather(1, order[..., None].expand(-1, -1, L)) * mask[..., None]
    inv = order.argsort(1)                              # cumledeki konum -> torbadaki yer
    succ = order + 1
    has = (succ < enc["length"][rows, None]) & mask
    next_t = torch.where(has, inv.gather(1, succ.clamp(max=L - 1)), torch.full_like(succ, -1))
    return ids, mask, slot_t, next_t, order


def loss_of(agent, enc, rows, gen, unk_rate):
    ids, mask, slot_t, next_t, _ = _batch(enc, rows, gen, unk_rate, agent.index[UNK])
    P, G = agent(ids, mask)
    rowm = mask.float()
    slot_loss = -((slot_t * P.log_softmax(2)).sum(2) * rowm).sum() / rowm.sum()                 # kelime -> yuva
    col_t = slot_t / slot_t.sum(1, keepdim=True).clamp(min=1e-9)
    slot_loss = slot_loss - ((col_t * P.log_softmax(1)).sum(1) * rowm).sum() / rowm.sum()       # yuva -> kelime
    has = next_t >= 0
    next_loss = F.cross_entropy(G[has], next_t[has])
    return slot_loss + next_loss


def evaluate(agent, enc, valid, seed=0):
    """-> tam dogru, dogru yuva, dogru komsu, ortalama deneme (en cok TRIES; bulunamayan TRIES + 1 sayilir).  Butun
    sinav tek ileri hesapta; deneme yalniz ilk atamasi yanlis cumlelerde."""
    gen = torch.Generator().manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    rows = torch.arange(len(enc["words"]))
    ids, mask, _, _, order = _batch(enc, rows, gen)
    with torch.no_grad():
        P, _ = agent(ids, mask)
    exact = slots = pairs = tries = n_slots = n_pairs = 0
    for b, words in enumerate(enc["words"]):
        n = len(words)
        bag = [words[o] for o in order[b, :n].tolist()]
        Pb = P[b, :n, :n]
        options = valid[tuple(sorted(words))]

        def sentence(slot):
            out = [None] * n
            for i, s in enumerate(slot):
                out[s] = bag[i]
            return tuple(out)

        built = sentence(assign_slots(Pb))
        best = max(options, key=lambda o: sum(a == c for a, c in zip(built, o)))
        exact += built in options
        slots += sum(a == c for a, c in zip(built, best))
        n_slots += n
        gold = set(zip(best, best[1:]))
        pairs += sum(p in gold for p in zip(built, built[1:]))
        n_pairs += n - 1
        k, found = 1, built in options
        if not found:
            score = (Pb.log_softmax(1) + Pb.log_softmax(0)).numpy()
            noise = np_rng.gumbel(size=(TRIES,) + score.shape)
        while not found and k < TRIES:
            found = sentence(assign_slots(score, noise[k])) in options
            k += 1
        tries += k if found else TRIES + 1
    m = max(len(enc["words"]), 1)
    return dict(n=len(enc["words"]), exact=round(exact / m, 4), slot=round(slots / max(n_slots, 1), 4),
                neighbor=round(pairs / max(n_pairs, 1), 4), tries=round(tries / m, 2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=30)          # 60 ile ayni sonuc (olculdu)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)                            # kucuk model: 16 iplik 4'ten yavas (olculdu)
    train, exam = _load("country_train.jsonl"), _load("country_exam.jsonl")
    valid = defaultdict(set)                            # ayni torbadan kurulabilen gecerli cumleler (butun veri)
    for r in _load("country_sentences.jsonl"):
        valid[tuple(sorted(r["words"]))].add(tuple(r["words"]))
    vocab = [UNK] + sorted({w for r in train for w in r["words"]})
    agent = GrammarAgent(vocab)
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr)
    splits = {s: _encode(agent, [r for r in exam if r["split"] == s]) for s in ("seen", "unseen")}
    enc = _encode(agent, train)                         # bir kez: kelime -> sayi, esdeger yuvalar
    gen = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(len(train), generator=gen)
        total = 0.0
        for b in range(0, len(train), args.batch):
            rows = perm[b:b + args.batch]
            loss = loss_of(agent, enc, rows, gen, UNK_RATE)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(rows)
        if epoch % 10 == 0 or epoch == args.epochs:
            agent.eval()
            res = {s: evaluate(agent, e, valid) for s, e in splits.items()}
            agent.train()
            print("epok %3d  kayip %.3f  |  %s  (%.0f sn)" % (epoch, total / len(train), "  ".join(
                "%s: tam %.3f yuva %.3f komsu %.3f deneme %.1f" % (s, v["exact"], v["slot"], v["neighbor"], v["tries"])
                for s, v in res.items()), time.time() - t0), flush=True)
    return agent, res


if __name__ == "__main__":
    main()
