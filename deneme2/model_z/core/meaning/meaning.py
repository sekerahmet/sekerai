"""meaning -- Model Z meaning agent (kullanici, 4 Ekim: "kelimeler arası anlam bağını oluştıran bir adım ... Türkiye Ankara
Asia lira gibi kelimeleri yakınlaştıran"; "Cümleler içinde attention ile"; "saatlerdir bu tabloyu öğrenen modelle kurmanı
istedim"; adlar onayli).  Egitim: train_meaning.py.

Bag tablosu modelin kendi parametresi; gizli kelime yalniz tablo uzerinden tahmin edilir (kullanici, 4 Ekim: "Tamam sen
öneri mimari kur"):
    R[i, j] = source_i . target_j / sqrt(d)     bag tablosu (V x V, dusuk boyutlu)
    P(j | i) = softmax_j(bias_j + R[i, j])      torbadaki her kelime tek basina tahmin eder
    P(j gizli | torba) = ortalama_i P(j | i)    (kullanici, 4 Ekim: "Kur"): bag kelime ciftine yazilir
    bias    kelimenin genel sikligi: "the", "is" tabloyu doldurmasin
Komsu tablosu (build_neighbor_table): kos(source_i, target_j) -- IN-OUT kosinusu birlikte gelen kelimeleri verir, ham
carpimda vektor boyu buyuk sik kelimeler one cikar (Mitra ve ark. 2016, belge/makaleler/2016/mitra2016_desm.txt: "the
IN-OUT cosine similarities are high between words that often co-occur in the same query or document").  Kisa liste (shortlist): okunan cumlenin
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

    def forward(self, ids, present, chunk=8192):
        """ids (B, L) penceredeki farkli kelimeler (sirasiz), present (B, L) gercek yuva -> (log P(yuvadaki kelime |
        penceredeki obur kelimeler) (B, L), oy veren var mi (B, L)): her kelime sirayla gizli sayilir, toplu hesap.  P(j | i) paydasi yalniz i'ye bagli;
        batch'teki farkli kelimeler icin bir kez (chunk satir parcalarla)."""
        u, v = self.source(ids), self.target(ids)
        scale = u.shape[-1] ** 0.5
        uq, inv = ids.unique(return_inverse=True)
        log_z = torch.cat([(self.source(uq[c:c + chunk]) @ self.target.weight.T / scale + self.bias).logsumexp(1)
                           for c in range(0, len(uq), chunk)])
        s = u @ v.transpose(1, 2) / scale + self.bias[ids][:, None, :] - log_z[inv][:, :, None]   # s[b, i, j] = log P(j | i)
        L = ids.shape[1]
        voters = present[:, :, None] & present[:, None, :] & ~torch.eye(L, dtype=torch.bool, device=ids.device)
        n = voters.sum(1)                                                                        # (B, L) oy veren sayisi
        return s.masked_fill(~voters, -1e9).logsumexp(1) - n.clamp(min=1).log(), n > 0


@torch.no_grad()
def build_neighbor_table(agent, m):
    """-> {"ids": (V, m), "scores": (V, m)}: her kelime icin kos(source_i, target_j) en buyuk m kelime (kendisi ve <unk>
    haric)."""
    V = len(agent.vocab)
    ids, scores = [], []
    src = torch.nn.functional.normalize(agent.source.weight, dim=1)
    tgt = torch.nn.functional.normalize(agent.target.weight, dim=1)
    for c in range(0, V, 4096):
        rows = torch.arange(c, min(c + 4096, V), device=agent.bias.device)
        r = src[rows] @ tgt.T
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
