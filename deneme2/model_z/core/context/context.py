"""context -- Model Z baglam ajani (kullanici, 4 Ekim: "Context agent mantıklı geldi"; "Ya kapsamı genişlet bu cümle için
kelime torbası da üretsin"; "10 kelime verdim. sonraki torbadan da 7 kelime daha veriyorum ama 3 tanesi ni de sen bul";
"Tamam kodla ve gpu da koş").  Matematik: belge/model_z_temel/05 (Cerceve), 07, 09.  Egitim ve olcum: train_context.py.

Gorev: hikayeden sonraki cumle icin aday kelime torbalari (gramere gidecek).  En yalin hal (attention yok, yuva yok):
    E               kelime temsili (sozluk x d); cikti da ayni temsil (bagli)
    durum           h = gamma . h + sum_{cumle} e_w (eskiler sonumlenir; gamma ogrenilen, boyut basina) ve gorulen kelimeler
    havuz           sonraki torbanin bilinen kismi: p = sum c_w e_w
    tahmin          u = MLP([LN h ; LN p]); z_w = u . e_w / sqrt(d) + beta_w + seen_w [w hikayede gecti]
    sayi (hurdle)   eksik kelime var mi: sigmoid(z_w); varsa kac kez: softmax(gamma_w) uzerinden 1 / 2 / 3+ (09)
Egitim: sonraki torbadan rastgele sayida kelime saklanir, ajan saklananlari bulur (maskeli torba tamamlama).
Uretim (next_bags): havuz bos -> en olasi K kelime tohum; her tohumla torbanin kalani tek seferde (her kelimenin en olasi
sayisi).  <unk> (indeks 0) uretilmez; buyuk / kucuk harf korunur.
"""
import torch
import torch.nn.functional as F

D = 32                  # matematikcinin 09 kosulari d 32 ile (olculdu, ulke verisi); SS icin ayrica secilecek
DIRECTIONS = 20         # aday torba (yon = tohum kelime) sayisi; ulke verisinde dogru torbanin K adayda olma tavani K10 0,849, K20 0,998 (07 §2)
COUNTS = 3              # var olan kelimenin sayisi: 1, 2, 3+


class ContextAgent(torch.nn.Module):
    def __init__(self, vocab, d=D):
        super().__init__()
        self.vocab = vocab
        self.index = {w: i for i, w in enumerate(vocab)}
        V = len(vocab)
        self.E = torch.nn.Embedding(V, d)
        self.decay = torch.nn.Parameter(torch.ones(d))                               # gamma = sigmoid(decay)
        self.norm_story, self.norm_pool = torch.nn.LayerNorm(d), torch.nn.LayerNorm(d)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(2 * d, 4 * d), torch.nn.GELU(), torch.nn.Linear(4 * d, d))
        self.beta = torch.nn.Parameter(torch.full((V,), -5.0))                       # kelimenin genel varlik egilimi
        self.seen_bias = torch.nn.Parameter(torch.zeros(V))                          # hikayede gecmis kelimeye etki
        self.count_logits = torch.nn.Parameter(torch.zeros(V, COUNTS))               # varsa kac kez (baglamdan bagimsiz)

    def ids(self, words):
        return [self.index.get(w, 0) for w in words]

    def set_prior(self, presence):
        """beta_w = logit(kelimenin bir cumlede bulunma sikligi)."""
        p = presence.clamp(1e-6, 1 - 1e-6)
        with torch.no_grad():
            self.beta.copy_((p / (1 - p)).log())

    def initial_state(self, batch):
        dev = self.E.weight.device
        return torch.zeros(batch, self.E.weight.shape[1], device=dev), torch.zeros(batch, len(self.vocab), device=dev)

    def read(self, state, ids, mask):
        """state (h, gorulen), ids / mask (B, W) bir cumle -> yeni durum; bos satir durumu degistirmez."""
        h, seen = state
        empty = ~mask.any(1, keepdim=True)
        new_h = torch.sigmoid(self.decay) * h + (self.E(ids) * mask[..., None]).sum(1)
        new_seen = seen.scatter(1, ids, mask.float()).maximum(seen)
        return torch.where(empty, h, new_h), torch.where(empty, seen, new_seen)

    def _scores(self, state, pool):
        """durum, havuz sayilari (B, V) -> z (B, V): eksik kelime var mi logiti."""
        h, seen = state
        p = pool @ self.E.weight
        u = self.mlp(torch.cat([self.norm_story(h), self.norm_pool(p)], -1))
        z = u @ self.E.weight.T / u.shape[-1] ** 0.5 + self.beta + self.seen_bias * seen
        z = z.clone()
        z[:, 0] = -1e4                                                                # <unk> uretilmez
        return z

    def _log_counts(self, z):
        """z (B, V) -> log P(sayi = 0, 1, 2, 3+) (B, V, 4)."""
        present = F.logsigmoid(z)[..., None] + self.count_logits.log_softmax(-1)
        return torch.cat([F.logsigmoid(-z)[..., None], present], -1)

    def bag_log_prob(self, state, pool, missing):
        """Havuz (bilinen kisim) ve eksik kisim sayilari (B, V) -> log P(eksik | baglam, havuz) (B,)."""
        lc = self._log_counts(self._scores(state, pool))
        c = missing.clamp(max=COUNTS).long()[..., None]
        return lc.gather(2, c).squeeze(2).sum(1)

    @torch.no_grad()
    def _complete(self, state, pool):
        """Havuz -> eksik kisim (her kelimenin en olasi sayisi) ve log olasiligi."""
        best, counts = self._log_counts(self._scores(state, pool)).max(-1)
        return counts, best.sum(1)

    @torch.no_grad()
    def next_bags(self, state, k=DIRECTIONS):
        """-> aday torbalar (B, K, V) sayilar (3 = 3+) ve log olasilik (B, K): tohum (havuz bos iken en olasi K kelime),
        sonra tohumla tamamlama.  log P = log P(tohum var | bos havuz) + log P(kalan | tohum)."""
        B, V = len(state[0]), len(self.vocab)
        z0 = self._scores(state, torch.zeros(B, V, device=state[0].device))
        seed_lp, seeds = F.logsigmoid(z0).topk(k, dim=1)                             # (B, K)
        pool = torch.zeros(B * k, V, device=z0.device).scatter_(1, seeds.reshape(-1, 1), 1.0)
        rep = tuple(s.repeat_interleave(k, 0) for s in state)
        rest, rest_lp = self._complete(rep, pool)
        bags = (rest + pool.long()).clamp(max=COUNTS).view(B, k, V)
        return bags, seed_lp + rest_lp.view(B, k)
