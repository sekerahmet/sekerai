"""context -- Model Z baglam ajani (kullanici, 4 Ekim: "Context agent mantıklı geldi"; "Bende b ye sıcağım"; "Ya kapsamı
genişlet bu cümle için kelime torbası da üretsin"; "Sen matematikçinin önerisini kodla"; "İsimler ok büyük harf olsun").
Matematik: belge/model_z_temel/07 §4, 09 (sayi basligi hurdle).  Egitim ve olcum: train_context.py.

Hikaye cumle cumle okunur; durumdan sonraki cumle icin K aday kelime torbasi ve olasiliklari cikar; torbalar gramere gider.
    E               kelime temsili (sozluk x d); torba basligi da ayni temsili kullanir
    durum           SLOTS yuva; read: yuvalar cumlenin kelimelerine (torba, sira yok), sonra birbirine bakar, sonra MLP
    yon             h = yuvalarin ortalamasi; u_k = W_k h + q_k (K = DIRECTIONS), pi = softmax(a . u_k + b)
    sayi basligi    z_kw = u_k . e_w / sqrt(d) + beta_w + seen_w [w hikayede gecti]; kelime var mi: sigmoid(z_kw);
                    varsa kac kez: softmax(gamma_w) uzerinden 1 / 2 / 3+ (hurdle; 09: sirali baslikta P(c=1) <= 0,348)
    torba olasiligi log P(B | k) = sum_w log P(c_w(B) | k) (kesin); karisim P(B) = sum_k pi_k P(B | k)
    next_bags       her yonun torbasi = her kelimenin en olasi sayisi (arama yok), olasiligi pi_k P(B_k | k)
<unk> (indeks 0) hic uretilmez; buyuk / kucuk harf korunur.
"""
import torch
import torch.nn.functional as F

D = 64                  # kelime temsili ve yuva boyu; son surum B (kullanici, 4 Ekim: "Son sürüm b"): d 64, K 30
SLOTS = 10              # durumun yuva sayisi (olculmedi)
DIRECTIONS = 30         # aday torba (yon) sayisi; ulke verisinde dogru torbanin K adayda olma tavani K10 0,849, K20 0,998
                        # (07 §2, sayim)
LEVELS = 4              # sayi: 0, 1, 2, 3+
HEADS = 4


class ContextAgent(torch.nn.Module):
    def __init__(self, vocab, d=D, slots=SLOTS, directions=DIRECTIONS, heads=HEADS):
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
        self.directions = directions
        self.W = torch.nn.Linear(d, directions * d)                                   # h -> u_k (q_k: bias)
        self.share = torch.nn.Linear(d, 1)                                            # pi = softmax(a . u_k + b)
        self.beta = torch.nn.Parameter(torch.full((len(vocab),), -5.0))              # kelimenin genel varlik egilimi
        self.seen_bias = torch.nn.Parameter(torch.zeros(len(vocab)))                 # hikayede gecmis kelimeye etki
        g = torch.zeros(len(vocab), 3); g[:, 0] = 4.0                                  # hurdle: P(c=1|c>=1) ~0,96 baslangic
        self.gamma = torch.nn.Parameter(g)                                            # kelime basina sayi dagilimi (baglamsiz)

    def ids(self, words):
        return [self.index.get(w, 0) for w in words]

    def set_prior(self, presence):
        """beta_w = logit(kelimenin bir cumlede bulunma sikligi): baslangicta torba sozlugun genel egilimi."""
        p = presence.clamp(1e-6, 1 - 1e-6)
        with torch.no_grad():
            self.beta.copy_((p / (1 - p)).log())

    def initial_state(self, batch):
        """-> (yuvalar (B, slots, d), hikayede gecen kelimeler (B, V))."""
        return self.start.expand(batch, -1, -1), torch.zeros(batch, len(self.vocab), device=self.start.device)

    def read(self, state, ids, mask):
        """state, ids / mask (B, W) bir cumle -> yeni durum.  Bos satir (cumlesi bitmis hikaye) durumu degistirmez;
        attention'da bir anahtar acik tutulur ki tamamen maskeli satir olmasin."""
        slots, seen = state
        words = self.norm_words(self.E(ids))
        empty = ~mask.any(1)
        key_mask = ~mask
        key_mask[:, 0] = key_mask[:, 0] & ~empty
        look, _ = self.look(self.norm_read(slots), words, words, key_padding_mask=key_mask, need_weights=False)
        new = slots + look
        q = self.norm_self(new)
        new = new + self.mix(q, q, q, need_weights=False)[0]
        new = new + self.mlp(self.norm_mlp(new))
        new_seen = seen.scatter(1, ids, mask.float()).maximum(seen)
        return torch.where(empty[:, None, None], slots, new), torch.where(empty[:, None], seen, new_seen)

    def _heads(self, state):
        """durum -> log pi (B, K), z (B, K, V)."""
        slots, seen = state
        h = self.norm_out(slots).mean(1)
        u = self.W(h).view(len(h), self.directions, -1)
        log_pi = self.share(u).squeeze(-1).log_softmax(-1)
        z = u @ self.E.weight.T / u.shape[-1] ** 0.5 + self.beta + (self.seen_bias * seen)[:, None, :]
        z = z.clone()
        z[..., 0] = -1e4                                                              # <unk> uretilmez
        return log_pi, z

    def _level_log_probs(self, z, words=None):
        """z (..., n) -> log P(c = 0, 1, 2, 3+) (..., n, 4): var mi sigmoid(z), varsa kac kez softmax(gamma_w)."""
        lc = (self.gamma if words is None else self.gamma[words]).log_softmax(-1)       # (n, 3) ya da (B, P, 3)
        if words is not None:
            lc = lc[:, None]                                                          # (B, 1, P, 3)
        return torch.cat([F.logsigmoid(-z)[..., None], F.logsigmoid(z)[..., None] + lc], -1)

    def bag_log_prob(self, state, words, counts):
        """Torba -> (log pi (B, K), log P(torba | k) (B, K)).  words (B, P) torbadaki farkli kelimeler (dolgu 0), counts
        (B, P) sayilari (0 = dolgu; 3'ten buyuk 3 sayilir)."""
        log_pi, z = self._heads(state)
        absent = F.logsigmoid(-z).sum(-1)                                             # her kelime 0: (B, K)
        zp = z.gather(2, words[:, None, :].expand(-1, z.shape[1], -1))                # (B, K, P)
        lp = self._level_log_probs(zp, words)                                         # (B, K, P, 4)
        c = counts.clamp(max=LEVELS - 1)[:, None, :, None].expand(-1, z.shape[1], -1, 1)
        fix = (lp.gather(3, c).squeeze(3) - lp[..., 0]) * (counts > 0)[:, None, :]
        return log_pi, absent + fix.sum(-1)

    @torch.no_grad()
    def next_bags(self, state):
        """-> sayilar (B, K, V) (0..3, 3 = 3+), log olasilik (B, K) = log pi_k + log P(B_k | k)."""
        log_pi, z = self._heads(state)
        best, counts = self._level_log_probs(z).max(-1)
        return counts, log_pi + best.sum(-1)
