"""context -- Model Z baglam ajani (kullanici, 4 Ekim: "Context agent mantıklı geldi"; "Bende b ye sıcağım"; "Geri okuma
bence faz 2 olsun E okuması da faz iki olsun. Önce yalın halini kuralım").  Egitim ve olcum: train_context.py.

Hikaye cumle cumle okunur; durum her cumleden sonra guncellenir, sonraki cumlenin kelime dagilimini verir.
    E               kelime temsili (sozluk x d); cikti katmani da ayni temsil (bagli agirlik)
    durum           SLOTS yuva (SLOTS x d); baslangic ogrenilen
    read            yuvalar cumlenin kelimelerine attention ile bakar (cumle torba: kelime sirasi yok), sonra
                    birbirlerine bakar, sonra MLP; her adim artik baglantili
    next_words      her yuva sozluk uzerinde bir dagilim onerir; agirlikli karisim (birden fazla devam yonu)
"""
import torch

D = 128                 # kelime temsili ve yuva boyu
SLOTS = 10              # durumun yuva sayisi; ulke hikayesinde ~10 olgu turu = sonraki cumlenin ~10 olasi yonu (olculmedi)
HEADS = 4


class ContextAgent(torch.nn.Module):
    def __init__(self, vocab, d=D, slots=SLOTS, heads=HEADS):
        super().__init__()
        self.vocab = vocab
        self.index = {w: i for i, w in enumerate(vocab)}
        self.E = torch.nn.Embedding(len(vocab), d)
        self.start = torch.nn.Parameter(torch.randn(slots, d) / d ** 0.5)
        self.norm_read, self.norm_words = torch.nn.LayerNorm(d), torch.nn.LayerNorm(d)
        self.look = torch.nn.MultiheadAttention(d, heads, batch_first=True)          # yuva -> cumle kelimeleri
        self.norm_self = torch.nn.LayerNorm(d)
        self.mix = torch.nn.MultiheadAttention(d, heads, batch_first=True)           # yuva <-> yuva
        self.norm_mlp = torch.nn.LayerNorm(d)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(d, 4 * d), torch.nn.GELU(), torch.nn.Linear(4 * d, d))
        self.norm_out = torch.nn.LayerNorm(d)
        self.out = torch.nn.Linear(d, d, bias=False)                                 # yuva -> kelime uzayi
        self.weight = torch.nn.Linear(d, 1)                                           # yuvanin karisimdaki payi

    def ids(self, words):
        return [self.index.get(w, 0) for w in words]

    def initial_state(self, batch):
        return self.start.expand(batch, -1, -1)

    def read(self, state, ids, mask):
        """state (B, K, d), ids / mask (B, W) bir cumle -> yeni durum.  mask'i bos satir (cumlesi bitmis hikaye) durumu
        degistirmez."""
        words = self.norm_words(self.E(ids))
        empty = ~mask.any(1)
        key_mask = ~mask | empty[:, None] & (torch.arange(mask.shape[1], device=ids.device) == 0)[None]
        seen, _ = self.look(self.norm_read(state), words, words, key_padding_mask=key_mask, need_weights=False)
        new = state + seen
        q = self.norm_self(new)
        new = new + self.mix(q, q, q, need_weights=False)[0]
        new = new + self.mlp(self.norm_mlp(new))
        return torch.where(empty[:, None, None], state, new)

    def next_words(self, state):
        """state (B, K, d) -> log olasilik (B, sozluk): yuvalarin dagilimlarinin agirlikli karisimi."""
        s = self.norm_out(state)
        logits = self.out(s) @ self.E.weight.T / s.shape[-1] ** 0.5                 # (B, K, V)
        share = self.weight(s).squeeze(-1).log_softmax(-1)                           # (B, K)
        return (share[:, :, None] + logits.log_softmax(-1)).logsumexp(1)
