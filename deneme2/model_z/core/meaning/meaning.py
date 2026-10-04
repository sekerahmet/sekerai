"""meaning -- Model Z meaning agent (kullanici, 4 Ekim: "kelimeler arası anlam bağını oluştıran bir adım ... Türkiye Ankara
Asia lira gibi kelimeleri yakınlaştıran"; "Cümleler içinde attention ile"; "Meaning agent ok"; adlar onayli).  Egitim ve
olcum: train_meaning.py.

Amac: context agent 57.000 kelime yerine onceki cumlelerin kelimelerine anlamca yakin kelimelerden kisa bir liste
(vocabulary shortlist) uzerinde calissin.
    E               kelime temsili (sozluk x d)
    okuyucu         konumsuz attention (TransformerEncoder): h_i, kelimenin penceredeki (WINDOW cumle) anlami
    bag(i, j)       h_i . e_j / sqrt(d); egitim skip-gram + negative sampling (SGNS): pencerede birlikte gecen kelime
                    olumlu, sikligin 0,75 kuvvetiyle cekilen kelime olumsuz
    neighbors       her kelimenin baglamsiz okunusu (tek basina h_w) ile butun sozluge bag; en guclu m kelime tabloya
"""
import torch

UNK = "<unk>"
D = 64                  # kelime temsili boyu (gramer ile ayni; olculmedi)
HEADS = 4
LAYERS = 2              # okuyucu katmani (olculmedi)


class MeaningAgent(torch.nn.Module):
    def __init__(self, vocab, d=D, heads=HEADS, layers=LAYERS):
        super().__init__()
        self.vocab = vocab
        self.index = {w: i for i, w in enumerate(vocab)}
        self.E = torch.nn.Embedding(len(vocab), d)
        layer = torch.nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True)
        self.reader = torch.nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def ids(self, words):
        return [self.index.get(w, self.index[UNK]) for w in words]

    def forward(self, ids, mask):
        """ids (B, L) pencerenin kelimeleri (sirasiz), mask (B, L) -> h (B, L, d)."""
        return self.reader(self.E(ids), src_key_padding_mask=~mask)

    def bond(self, h, ids):
        """h (..., d), ids (..., k) -> bag puani (..., k)."""
        return (h.unsqueeze(-2) * self.E(ids)).sum(-1) / h.shape[-1] ** 0.5


@torch.no_grad()
def build_neighbor_table(agent, m, chunk=4096):
    """-> {"ids": (V, m) en guclu bagli kelimeler, "scores": (V, m)}; kelimenin kendisi ve <unk> haric."""
    dev = agent.E.weight.device
    V = len(agent.vocab)
    ids_out, scores_out = [], []
    for c in range(0, V, chunk):
        w = torch.arange(c, min(c + chunk, V), device=dev)
        h = agent(w[:, None], torch.ones(len(w), 1, dtype=torch.bool, device=dev))[:, 0]
        s = h @ agent.E.weight.T / h.shape[-1] ** 0.5
        s[torch.arange(len(w)), w] = -1e9
        s[:, agent.index[UNK]] = -1e9
        top = s.topk(m, dim=1)
        ids_out.append(top.indices.cpu())
        scores_out.append(top.values.cpu())
    return dict(vocab=agent.vocab, ids=torch.cat(ids_out), scores=torch.cat(scores_out))


def shortlist(table, words, n):
    """Kelime kimlikleri -> her birinin en guclu n bagli kelimesinin birlesimi (kimlik kumesi)."""
    if not len(words):
        return set()
    return set(table["ids"][list(words), :n].flatten().tolist())
