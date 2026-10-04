"""grammar -- Model Z gramer ajani (kullanici, 3 Ekim: "Gramer ajanının temel görevi verilen tüm kelimelerden anlamlı bir
cümle kurabilmesi"; "Next token mantığında değil tek seferde"; "ilişki matrisi gramerin öğrenerek hem değiştirdiği hem de
kullandığı birşey"; "Emin olduklarımızı yazalım"; adlar onayli).  Matematik: belge/model_z_temel/03_gramer_matematik.md.
Egitim ve olcum: train_grammar.py.

Girdi: cumlenin kelimeleri, karisik (torba).  Cikti: butun cumle bir anda.
    E               kelime temsili (sozluk x d); egitimde gecmeyen kelime <unk>
    boundary        sinir dugumu: cumlenin basi ve sonu, torbaya eklenir
    torba okuyucu   konumsuz attention (TransformerEncoder): h_i, kelimenin torbadaki baglami; girdi sirasindan bagimsiz
    relation_matrix G[i, j] = (h_i^T W h_j + e_i^T U e_j) / sqrt(d), torba ortalamasi cikarilmis, fp32: "j, i'nin
                    hemen ardindan gelir"; ilk terim baglamli (kalip), ikincisi sozcuksel (kelimenin kendisi: ad ici sira).
                    boundary'nin satiri cumlenin ilk kelimesi, sutunu son kelimesi
    order_by_relation  cumle = G uzerinde boundary'den gecen tek cevrim: her dugume bir ardil (Macar atamasi, tek seferde);
                    atama birden fazla cevrim verirse patch_cycles birlestirir (Karp yamasi).  Maliyet n^3
"""
import heapq

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

UNK = "<unk>"
D = 64                  # kelime temsili boyu
HEADS = 4
LAYERS = 3
NEG = -1e9


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
        ok = m[:, :, None] & m[:, None, :] & ~torch.eye(L + 1, dtype=torch.bool, device=ids.device)[None]
        with torch.autocast(ids.device.type, enabled=False):          # G fp32: bf16'da buyuk G'nin farklari silinir
            h, e = h.float(), e.float()
            G = (torch.einsum("bid,de,bje->bij", h, self.W, h) + torch.einsum("bid,de,bje->bij", e, self.U, e)) / e.shape[-1] ** 0.5
            # kayip ve dizme G'ye eklenen sabite duyarsiz: ortalama cikarilir ki G serbestce kaymasin
            G = G - (G * ok).sum((1, 2), keepdim=True) / ok.sum((1, 2), keepdim=True)
        return G.masked_fill(~ok, NEG)


def relation_matrix(agent, words):
    """Tek torba -> G (torch, (n+1) x (n+1), son satir / sutun boundary) (okuma icin)."""
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


def quality_index(G, order):
    """Kalite endeksi (kullanici, 4 Ekim: "judge değil quality index"; "Olur uygun"): dizilisteki her bagin ardil (satir) ve
    oncel (sutun) log-olasiliklarinin ortalamasi, bag basina.  Egitimin olctugu olasiliklar; her torbada normalize, torbalar
    arasi kiyaslanabilir.  G numpy (n+1) x (n+1), son dugum boundary; order: torba indeksleri (order_by_relation)."""
    n = G.shape[0] - 1
    path = [n] + list(order) + [n]
    row = G - np.logaddexp.reduce(G, axis=1, keepdims=True)        # ardil: satir log-softmax
    col = G - np.logaddexp.reduce(G, axis=0, keepdims=True)        # oncel: sutun log-softmax
    return float(np.mean([(row[a, b] + col[a, b]) / 2 for a, b in zip(path, path[1:])]))


def order_alternatives(G, k, labels=None):
    """G -> puana gore en iyi k farkli cumle (torba indeksleri).  Murty: en iyi atamadan baslayip bir bagi yasaklayarak /
    oncekileri sabitleyerek sonraki en iyi atamalar; her atama patch_cycles ile tek cevrime.  labels (kelimeler): ayni
    kelimenin kopyalarini degistiren sira ayni cumle sayilir.  Deterministik."""
    big = 1e12                                          # yasak bag (sonsuz yerine: scipy sonsuzda hata verir)
    cost = np.where(G <= NEG / 2, big, -G)
    n = G.shape[0] - 1

    def solve(c):
        r, s = linear_sum_assignment(c)
        return (c[r, s].sum(), s) if c[r, s].max() < big / 2 else None

    first = solve(cost)
    heap, tick, out, seen = [(first[0], 0, cost, first[1], 0)], 1, [], set()
    while heap and len(out) < k and tick < 50 * k:
        score, _, c, s, fixed = heapq.heappop(heap)
        succ = patch_cycles(G, dict(enumerate(s.tolist())))
        seq, x = [], succ[n]
        while x != n:
            seq.append(x)
            x = succ[x]
        key = tuple(labels[i] for i in seq) if labels is not None else tuple(seq)
        if key not in seen:
            seen.add(key)
            out.append(seq)
        child = c.copy()
        for i in range(fixed, n + 1):                   # i. bag yasak, 0..i-1 sabit
            ban = child.copy()
            ban[i, s[i]] = big
            got = solve(ban)
            if got is not None:
                heapq.heappush(heap, (got[0], tick, ban, got[1], i))
                tick += 1
            keep = child[i, s[i]]
            child[i, :], child[:, s[i]] = big, big
            child[i, s[i]] = keep
    return out


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

