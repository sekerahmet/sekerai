"""codec -- Model Z codec agent (kullanici, 5 Ekim: "encoder ve decoder gibi iki ajan olmalı. Birisi cümleyi vektöre
çevirirken diğer vektörü cümleye çevirmeyi öğrenmeli"; "Tek ajan"; "Codec mantıklı"; decoder kayipsizlik denetimi:
"o bir kontrol veri kaybının olmadığını görmek").  Egitim: train_codec.py.  Parcalar okunan calismalardan
(belge/makaleler: SONAR 2023, AUTOBOT 2021, TSDAE 2021):

    encode  [CLS] + cumle (sirali kelimeler) -> TransformerEncoder (LAYERS, sinus konum) -> CLS ciktisi sorgu olan
            attention havuzu (AUTOBOT denklem 2) -> Linear -> LayerNorm = z (Z)
    decode  z -> cumle, kelime kelime cumlenin kendi sirasiyla.  Her katman: nedensel self-attention, z'ye kapili
            cross-attention (AUTOBOT denklem 4: g = sigmoid(G q + G' z), o = g * W_V z), FFN.  Sonraki kelime =
            h . E^T (kelime vektorleri encoder ile ortak); cumle END ile biter
Uzunluk siniri yok: konum sinus ile, cumle kesilmez.
"""

import torch
import torch.nn.functional as F

UNK = "<unk>"
D = 256                 # kelime ve katman boyu
Z = 256                 # cumle vektoru boyu
LAYERS = 3              # encoder ve decoder katman sayisi (her biri)
HEADS = 4


def sinusoid(T, d, device):
    """(T, d) sinus konum kodlamasi (Vaswani ve ark. 2017); uzunluk siniri yok."""
    pos = torch.arange(T, device=device, dtype=torch.float)[:, None]
    i = torch.arange(0, d, 2, device=device, dtype=torch.float)
    ang = pos / 10000 ** (i / d)
    out = torch.zeros(T, d, device=device)
    out[:, 0::2], out[:, 1::2] = ang.sin(), ang.cos()
    return out


class GatedZLayer(torch.nn.Module):
    """Decoder katmani: nedensel self-attention + z'ye kapili cross-attention (AUTOBOT) + FFN; pre-norm."""

    def __init__(self, d, z, heads):
        super().__init__()
        self.n1, self.n2, self.n3 = torch.nn.LayerNorm(d), torch.nn.LayerNorm(d), torch.nn.LayerNorm(d)
        self.self_attn = torch.nn.MultiheadAttention(d, heads, batch_first=True)
        self.value = torch.nn.Linear(z, d)                  # W_V z
        self.gate_q = torch.nn.Linear(d, d)                 # G q
        self.gate_z = torch.nn.Linear(z, d, bias=False)     # G' z
        self.ffn = torch.nn.Sequential(torch.nn.Linear(d, 4 * d), torch.nn.GELU(), torch.nn.Linear(4 * d, d))

    def forward(self, x, z, causal):
        h = self.n1(x)
        x = x + self.self_attn(h, h, h, attn_mask=causal, need_weights=False)[0]
        q = self.n2(x)
        x = x + torch.sigmoid(self.gate_q(q) + self.gate_z(z)[:, None]) * self.value(z)[:, None]
        return x + self.ffn(self.n3(x))


class CodecAgent(torch.nn.Module):
    def __init__(self, vocab, d=D, z=Z, layers=LAYERS, heads=HEADS):
        super().__init__()
        self.vocab = vocab
        self.index = {w: i for i, w in enumerate(vocab)}
        V = len(vocab)
        self.END, self.BOS, self.CLS = V, V + 1, V + 2                  # ozel kelimeler sozlugun sonunda
        self.d = d
        self.E = torch.nn.Embedding(V + 3, d)
        torch.nn.init.normal_(self.E.weight, std=d ** -0.5)             # cikis logit'i h . E: N(0, 1) ile logit ~ +-sqrt(d)
        layer = torch.nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = torch.nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.pool = torch.nn.MultiheadAttention(d, heads, batch_first=True)
        self.to_z = torch.nn.Linear(d, z)
        self.z_norm = torch.nn.LayerNorm(z, elementwise_affine=False)
        self.decoder = torch.nn.ModuleList(GatedZLayer(d, z, heads) for _ in range(layers))
        self.out_norm = torch.nn.LayerNorm(d)

    def ids(self, words):
        return [self.index.get(w, self.index[UNK]) for w in words]

    def encode(self, ids, mask):
        """ids, mask (B, L) sirali cumle (dolgu 0, mask False) -> z (B, Z)."""
        B, L = ids.shape
        x = torch.cat([torch.full((B, 1), self.CLS, device=ids.device), ids], 1)
        m = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=ids.device), mask], 1)
        h = self.encoder(self.E(x) + sinusoid(L + 1, self.d, ids.device)[None], src_key_padding_mask=~m)
        pooled, _ = self.pool(h[:, :1], h, h, key_padding_mask=~m, need_weights=False)   # sorgu: CLS ciktisi
        return self.z_norm(self.to_z(pooled[:, 0]))

    def decode_logits(self, z, prev):
        """z (B, Z), prev (B, T) girdi kelimeleri [BOS, w1 ..] -> sonraki kelime logit'leri (B, T, V + 3)."""
        T = prev.shape[1]
        x = self.E(prev) + sinusoid(T, self.d, prev.device)[None]
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=prev.device), 1)
        for layer in self.decoder:
            x = layer(x, z, causal)
        return self.out_norm(x) @ self.E.weight.T

    def forward(self, ids, mask, enc_ids=None, enc_mask=None):
        """Egitim -> (logit'ler (B, L + 1, V + 3), hedefler (B, L + 1); dolgu -100).  enc_ids / enc_mask: encoder'a
        giden (gurultulu) cumle; verilmezse ids / mask (gurultusuz)."""
        B, L = ids.shape
        z = self.encode(ids if enc_ids is None else enc_ids, mask if enc_mask is None else enc_mask)
        n = mask.sum(1)
        prev = torch.cat([torch.full((B, 1), self.BOS, device=ids.device), ids], 1)
        tgt = torch.cat([ids, torch.zeros(B, 1, dtype=ids.dtype, device=ids.device)], 1)
        tgt = tgt.scatter(1, n[:, None], self.END)                               # son kelimeden sonra END
        tgt = tgt.masked_fill(torch.arange(L + 1, device=ids.device)[None] > n[:, None], -100)
        return self.decode_logits(z, prev), tgt

    @torch.no_grad()
    def decode(self, z, max_words):
        """z (B, Z) -> kelime kimlik listeleri (acgozlu, END'e kadar ya da max_words; END / BOS / CLS uretilmez)."""
        B = len(z)
        prev = torch.full((B, 1), self.BOS, device=z.device)
        done = torch.zeros(B, dtype=torch.bool, device=z.device)
        for _ in range(max_words + 1):
            logits = self.decode_logits(z, prev)[:, -1]
            logits[:, self.BOS] = logits[:, self.CLS] = -1e9
            w = torch.where(done, torch.full_like(done, self.END, dtype=torch.long), logits.argmax(-1))
            done |= w == self.END
            prev = torch.cat([prev, w[:, None]], 1)
            if done.all():
                break
        out = []
        for row in prev[:, 1:].tolist():
            out.append(row[:row.index(self.END)] if self.END in row else row)
        return out


def delete_words(ids, mask, ratio, gen):
    """TSDAE gurultusu: her kelime ratio olasilikla silinir (en az bir kelime kalir); kalanlar sola kaydirilir."""
    if ratio <= 0:
        return ids, mask
    keep = mask & (torch.rand(ids.shape, generator=gen, device=ids.device) >= ratio)
    first = mask & (mask.cumsum(1) == 1)                                     # hic kelime kalmazsa ilki kalir
    keep = torch.where(keep.any(1, keepdim=True), keep, first)
    order = (~keep).to(torch.int8).argsort(dim=1, stable=True)
    return ids.gather(1, order) * keep.gather(1, order), keep.gather(1, order)


def loss_of(forward, ids, mask, noise=0.0, gen=None):
    """-> ortalama -log P(dogru sonraki kelime) (kelime basina, END dahil); encoder'a gurultulu cumle (noise > 0)."""
    enc_ids, enc_mask = delete_words(ids, mask, noise, gen)
    logits, tgt = forward(ids, mask, enc_ids, enc_mask)
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)
