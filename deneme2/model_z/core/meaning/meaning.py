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


WINDOW = 2              # komsu tablosu: kac cumlelik pencerede birlikte gecme (kullanici, 4 Ekim: "önce 1 bakarız")
NEIGHBORS = 10          # kelime basina komsu (kullanici: "n=10 olacak sekilde")
STRONG_LIFT = 3.0       # guclu bag: P(j | i) / P(j) en az (elle; olculmedi)
STRONG_SHARE = 0.05     # guclu bag: i'nin pencerelerinin en az bu payinda j de var (elle; olculmedi)
FREQUENT_SHARE = 0.02   # cumlelerin bu payindan fazlasinda gecen kelime listeye hep girer (siklik tablosu; elle)


def build_neighbor_table(stories, V, window=WINDOW, n=NEIGHBORS):
    """Hikayeler (cumle -> kelime kimlikleri) -> komsu tablosu: {"ids": (V, n) (dolgu 0), "frequent": kimlikler}.
    Komsu: ayni pencerede (window cumlenin kelimeleri) guclu bagli kelimeler (kat >= STRONG_LIFT, birlikte >= STRONG_SHARE),
    birlikte gecme payina gore ilk n.  Bicim kelimelerinin guclu bagi olmaz: komsu getirmezler."""
    import numpy as np
    import scipy.sparse as sp
    rows, cols, k = [], [], 0
    df, n_sent = np.zeros(V), 0
    for st in stories:
        for s in st:
            n_sent += 1
            df[list(set(s))] += 1
        for t in range(max(1, len(st) - window + 1)):
            ws = {x for s in st[t:t + window] for x in s}
            rows += [k] * len(ws)
            cols += list(ws)
            k += 1
    X = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(k, V))
    C = (X.T @ X).tocsr()
    win = C.diagonal()
    ids = torch.zeros(V, n, dtype=torch.long)
    for i in range(1, V):
        a, b = C.indptr[i], C.indptr[i + 1]
        j, c = C.indices[a:b], C.data[a:b]
        share = c / max(win[i], 1)
        keep = (j != i) & (j != 0) & (share >= STRONG_SHARE) & (share / np.maximum(win[j] / k, 1e-12) >= STRONG_LIFT)
        top = j[keep][np.argsort(-share[keep])][:n]
        ids[i, :len(top)] = torch.from_numpy(top.astype(np.int64))
    frequent = torch.from_numpy(np.flatnonzero(df / max(n_sent, 1) > FREQUENT_SHARE))
    return dict(ids=ids, frequent=frequent, window=window, n=n)


def shortlist(table, words, n=None):
    """Kelime kimlikleri -> her birinin ilk n komsusu + sik kelimeler + kendileri (kimlik kumesi, <unk> haric)."""
    n = n or table["n"]
    out = set(table["frequent"].tolist()) | set(words)
    if len(words):
        out |= set(table["ids"][list(words), :n].flatten().tolist())
    out.discard(0)
    return out
