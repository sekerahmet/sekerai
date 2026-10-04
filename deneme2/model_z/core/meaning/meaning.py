"""meaning -- Model Z meaning agent (kullanici, 4 Ekim: "kelimeler arası anlam bağını oluştıran bir adım ... Türkiye Ankara
Asia lira gibi kelimeleri yakınlaştıran"; "Cümleler içinde attention ile"; "saatlerdir bu tabloyu öğrenen modelle kurmanı
istedim"; adlar onayli).  Egitim: train_meaning.py.

Bag tablosu modelin kendi parametresi; gizli kelime yalniz tablo uzerinden tahmin edilir (kullanici, 4 Ekim: "Tamam sen
öneri mimari kur"):
    R[i, j] = source_i . target_j / sqrt(d)     bag tablosu (V x V, dusuk boyutlu)
    puan(j) = bias_j + sum_i R[i, j]            torbadaki her kelime i gizli kelimeye bagi kadar oy verir (esit, toplamsal)
    bias    kelimenin genel sikligi: "the", "is" tabloyu doldurmasin
(Attention'li surum denendi: sorgu gizli yeri bilmedigi icin oylari kalip kelimelerine verdi, Turkey'nin satiri
egitilmedi.)
Komsu tablosu (build_neighbor_table): R'nin her satirinin en buyuk m degeri.  Kisa liste (shortlist): okunan cumlenin
kelimeleri icin tablodan NEIGHBORS komsu, DEPTH adim (komsularin komsulari).
"""
import torch

UNK = "<unk>"
D = 128                 # kelime vektoru boyu: ulkede ~110 ulke kumesi ayrilabilsin (olculmedi)
NEIGHBORS = 5           # kisa liste: kelime basina komsu (kullanici, 4 Ekim: "Komşu 5 derinlik 5 yap")
DEPTH = 5               # kisa liste: komsularin komsulari kac adim (kullanici: "benim n dediğim derinlikti")


class MeaningAgent(torch.nn.Module):
    def __init__(self, vocab, d=D):
        super().__init__()
        self.vocab = vocab
        self.index = {w: i for i, w in enumerate(vocab)}
        self.source = torch.nn.Embedding(len(vocab), d)          # oy veren
        self.target = torch.nn.Embedding(len(vocab), d)          # oy alan
        self.bias = torch.nn.Parameter(torch.zeros(len(vocab)))
        for e in (self.source, self.target):                     # kucuk baslangic: 30 oyun toplami da kucuk kalsin
            torch.nn.init.normal_(e.weight, std=0.1)

    def ids(self, words):
        return [self.index.get(w, self.index[UNK]) for w in words]

    def forward(self, ids, present):
        """ids (B, L) torbadaki kelimeler (sirasiz), present (B, L) gercek ve gorunen yuva -> gizli kelime icin log
        olasilik (B, V)."""
        u = self.source(ids) * present[..., None]
        vote = u.sum(1)                                          # sum_i source_i
        return (vote @ self.target.weight.T / u.shape[-1] ** 0.5 + self.bias).log_softmax(-1)

    @torch.no_grad()
    def relation(self, rows):
        """Kelime kimlikleri (n,) -> R satirlari (n, V)."""
        return self.source(rows) @ self.target.weight.T / self.source.weight.shape[1] ** 0.5


@torch.no_grad()
def build_neighbor_table(agent, m):
    """-> {"ids": (V, m), "scores": (V, m)}: R'nin her satirinin en buyuk m degeri (kelimenin kendisi ve <unk> haric)."""
    V = len(agent.vocab)
    ids, scores = [], []
    for c in range(0, V, 4096):
        rows = torch.arange(c, min(c + 4096, V), device=agent.bias.device)
        r = agent.relation(rows)
        r[torch.arange(len(rows)), rows] = -1e9
        r[:, agent.index[UNK]] = -1e9
        top = r.topk(m, dim=1)
        ids.append(top.indices.cpu())
        scores.append(top.values.cpu())
    return dict(vocab=agent.vocab, ids=torch.cat(ids), scores=torch.cat(scores))


def shortlist(table, words, n=NEIGHBORS, depth=DEPTH):
    """Kelime kimlikleri -> komsular depth adim (her adimda yeni gelen kelimelerin ilk n komsusu) + kendileri (kimlik
    kumesi, <unk> haric)."""
    found = set(words)
    frontier = set(words)
    for _ in range(depth):
        if not frontier:
            break
        nxt = set(table["ids"][sorted(frontier), :n].flatten().tolist()) - found - {0}
        found |= nxt
        frontier = nxt
    found.discard(0)
    return found
