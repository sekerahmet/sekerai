"""meaning -- Model Z meaning agent (kullanici, 4 Ekim: "kelimeler arası anlam bağını oluştıran bir adım ... Türkiye Ankara
Asia lira gibi kelimeleri yakınlaştıran"; "Cümleler içinde attention ile"; "benim 2 ve ya 3 cümle dediğim cümlenin tüm
kelimeleri ortak. Yoksa cümle tahmini değil"; adlar onayli).  Egitim ve olcum: train_meaning.py.

Amac: context agent 57.000 kelime yerine elindeki kelimelerle ayni metin parcasinda bulunan kelimelerden kisa bir liste
(vocabulary shortlist) uzerinde calissin.  Sira ve cumle tahmini yok: WINDOW cumlenin kelimeleri tek ortak torba.
    E               kelime temsili (sozluk x d); cikis da ayni temsil
    mask            gizli yerin temsili: torbadaki bir kelime yerine konur
    okuyucu         konumsuz attention (TransformerEncoder); mask Q ile hangi kelimeye bakacagini secer
    tahmin          P(w | torba) = softmax(h_mask . e_w / sqrt(d) + bias_w)
    komsu tablosu   egitimden sonra meaning agent'in tahminlerinden (her kelime icin en yakin kelimeler; henuz yok)
    shortlist       komsu tablosundan: kelime basina NEIGHBORS komsu, DEPTH adim
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


NEIGHBORS = 5           # kisa liste: kelime basina komsu (kullanici, 4 Ekim: "Komşu 5 derinlik 5 yap")
DEPTH = 5               # kisa liste: komsularin komsulari kac adim (kullanici: "benim n dediğim derinlikti")


def shortlist(table, words, n=NEIGHBORS, depth=DEPTH):
    """Kelime kimlikleri -> komsular depth adim (her adimda yeni gelen kelimelerin ilk n komsusu) + tablonun her zaman
    giren kelimeleri ("frequent", varsa) + kendileri (kimlik kumesi, <unk> haric).  table["ids"] (V, m): meaning agent'in
    komsu tablosu."""
    found = set(words)
    frontier = set(words)
    for _ in range(depth):
        if not frontier:
            break
        nxt = set(table["ids"][sorted(frontier), :n].flatten().tolist()) - found - {0}
        found |= nxt
        frontier = nxt
    if "frequent" in table:
        found |= set(table["frequent"].tolist())
    found.discard(0)
    return found
