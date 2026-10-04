"""meaning -- Model Z meaning agent (kullanici, 4 Ekim: "kelimeler arası anlam bağını oluştıran bir adım ... Türkiye Ankara
Asia lira gibi kelimeleri yakınlaştıran"; "Cümleler içinde attention ile"; "benim 2 ve ya 3 cümle dediğim cümlenin tüm
kelimeleri ortak. Yoksa cümle tahmini değil"; adlar onayli).  Egitim ve olcum: train_meaning.py.

Amac: context agent 57.000 kelime yerine elindeki kelimelerle ayni metin parcasinda bulunan kelimelerden kisa bir liste
(vocabulary shortlist) uzerinde calissin.  Sira ve cumle tahmini yok: WINDOW cumlenin kelimeleri tek ortak torba.
    E               kelime temsili (sozluk x d); cikis da ayni temsil
    mask            gizli yerin temsili: torbadaki bir kelime yerine konur
    okuyucu         konumsuz attention (TransformerEncoder); mask Q ile hangi kelimeye bakacagini secer
    tahmin          P(w | torba) = softmax(h_mask . e_w / sqrt(d) + bias_w)
    shortlist       verilen kelimeler + mask -> olasiligi en yuksek n kelime (verilenler haric)
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
        self.mask = torch.nn.Parameter(torch.randn(d) / d ** 0.5)
        layer = torch.nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True)
        self.reader = torch.nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.bias = torch.nn.Parameter(torch.zeros(len(vocab)))

    def ids(self, words):
        return [self.index.get(w, self.index[UNK]) for w in words]

    def forward(self, ids, present, hidden):
        """ids (B, L) torba (sirasiz), present (B, L) gercek yuva, hidden (B, L) mask konan yuva -> gizli yuvalar icin
        log olasilik (B, L, V) (gizli olmayan yuvalarda anlamsiz)."""
        e = torch.where(hidden[..., None], self.mask.expand_as(self.E(ids)), self.E(ids))
        h = self.reader(e, src_key_padding_mask=~present)
        return (h @ self.E.weight.T / h.shape[-1] ** 0.5 + self.bias).log_softmax(-1)


@torch.no_grad()
def predict(agent, words):
    """Kelime kimlikleri (liste) -> ayni parcada bulunacak kelimelerin olasiligi (V,): torba + bir mask."""
    dev = agent.E.weight.device
    ids = torch.tensor([list(words) + [0]], device=dev)
    present = torch.ones_like(ids, dtype=torch.bool)
    hidden = torch.zeros_like(present)
    hidden[0, -1] = True
    p = agent(ids, present, hidden)[0, -1].exp()
    p[list(words)] = 0
    p[agent.index[UNK]] = 0
    return p


def shortlist(agent, words, n):
    """Kelime kimlikleri -> ayni parcada bulunma olasiligi en yuksek n kelime (kimlik kumesi; verilenler haric)."""
    if not len(words):
        return set()
    return set(predict(agent, words).topk(n).indices.tolist())


@torch.no_grad()
def build_neighbor_table(agent, m):
    """-> {"ids": (V, m), "scores": (V, m)}: her kelime tek basina verilince ilk m tahmin (okuma ve inceleme icin)."""
    ids, scores = [], []
    for w in range(len(agent.vocab)):
        top = predict(agent, [w]).topk(m)
        ids.append(top.indices.cpu())
        scores.append(top.values.cpu())
    return dict(vocab=agent.vocab, ids=torch.stack(ids), scores=torch.stack(scores))
