# -*- coding: utf-8 -*-
"""model_19 -- sabit cevap noktalari uzerinde ogrenilen iliskiler.  Tasarim: TASARIM_19.md (taslak).

    P[w]      cevap noktasi, SABIT (birim kure, rastgele)
    E[w]      giris noktasi = norm(P[w] + dE[w])                              R_PP'nin yeri (embed)
    C_t       zincir = [sira yarisi | icerik yarisi](E[w_0..t])
    C_m       V0 -> attention -> V1 -> V2 -> attention -> V3 (C_t)          hareketler + iki attention
    puan(w)   P[w] . z,  z = 2 e^S_p norm(C_m) + R_PC(C_t)                    cevap: z'ye en yakin P (readout)
    defter    sim(t,j) = cos(norm(Ĉ_t + R_CC(Ĉ_t)), Ĉ_j),  j < t - skip       zincir benzerligi (chain_sim)
    p         (1 - gate) softmax(puan) + gate p_defter

Yeni parcalar (embed, readout, chain_sim, distance, ikinci attention) sifir etkiyle ve AYRI bir ureteçten
baslar: hepsi kapaliyken model_19'un kendi zemini, acikken adim 0'da zeminle ayni puan (TASARIM_19 Oe4).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

D_ORDER = 512          # sira yarisi: kaydirmali, lam^yas
D_CONTENT = 512        # icerik yarisi: kaydirmasiz, kelime basina ogrenilen sonme
VECTORS = 256          # katman basina hareket
ACTIVE = 8             # her noktada aktif hareket (yone gore top-k)
LAYERS = 4
ATTN_AFTER = (0, 2)    # attention bu katmanlardan sonra; ikincisi yeni parca (Oe13)
ATTN_HEADS = 4
ATTN_DIM = 64
T_MAX = 512
LAM = 0.7              # sira yarisinin sonmesi
LAM_W_INIT = 0.9       # icerik yarisinin sonme baslangici (kelime x boyut)
BETA_W_INIT = 0.5
LAM_W_MIN = math.exp(-5)   # parca hesabi fp32'de tasmasin
CONTENT_CHUNK = 16
START_NORM = 1.0       # hareket start'larinin baslangic boyu
S_V_INIT = 0.0
LOAD_BALANCE = 0.01
S_P_P = 0.9            # S_p baslangici: kusursuz eslesmede dogru kelimeye bu olasilik (formul, __init__)
CACHE_SKIP = 3         # defter son 3 konumu aramaz
S_C_INIT = 3.0
GATE_0_INIT = -2.0
RANK = 256             # iliski matrislerinin ranki; None ya da >= D: tam matris
NEW_SEED = 1_000_003   # yeni parcalarin ureteci: seed + NEW_SEED (Oe4)


def entropy(p):
    """Son eksende entropi (nat); 0 log 0 = 0."""
    return -(p * p.clamp_min(1e-30).log()).sum(-1)


def directions90(X, rows=4096):
    """X (N,d): ortalanmis X'in varyansinin %90'ini tasiyan yon sayisi."""
    X = X[:rows].float()
    s2 = torch.linalg.svdvals(X - X.mean(0)) ** 2
    return int((s2.cumsum(0) / s2.sum() < 0.9).sum()) + 1


class Relation(nn.Module):
    """Ogrenilen dogrusal iliski x -> R x, sifir etkiyle baslar.
    rank None ya da >= d: tam d x d, sifir.  Degilse A B^T: A standart rastgele, B sifir (carpim 0, gradyan akar)."""

    def __init__(self, d, rank, gen):
        super().__init__()
        if rank is None or rank >= d:
            self.M = nn.Parameter(torch.zeros(d, d))
            self.A = self.B = None
        else:
            self.M = None
            self.A = nn.Parameter(torch.randn(d, rank, generator=gen) / math.sqrt(d))
            self.B = nn.Parameter(torch.zeros(d, rank))

    def forward(self, x):
        return x @ self.M.T if self.M is not None else (x @ self.B) @ self.A.T

    def size(self):
        """||R|| (Frobenius): buyume gunluge yazilir (Oe6)."""
        return float((self.M if self.M is not None else self.A @ self.B.T).norm())


class MoveLayer(nn.Module):
    """Hareket sozlugu: `vectors` tane (start, finish).  Noktaya YONCE en yakin `active` start aktif; nokta
    aktif hareketlerin agirlikli ortalamasi kadar tasinir: C_m + sum_a W_a (finish_a - start_a)."""

    def __init__(self, d, vectors, active, gen, start_norm, s_v_init):
        super().__init__()
        start = torch.randn(vectors, d, generator=gen) * (start_norm / math.sqrt(d))
        self.start = nn.Parameter(start)
        self.finish = nn.Parameter(start.clone())              # hareket 0: nokta yerinde baslar
        self.S_v = nn.Parameter(torch.tensor(float(s_v_init)))
        self.active = active

    def forward(self, C_m, probs=False):
        """-> C_m tasinmis; probs: + BUTUN hareketler uzerinde softmax (denge terimi, S_v sabit)."""
        D2 = self._distance(C_m)
        out = C_m + self._weights(D2) @ (self.finish - self.start)
        return (out, torch.softmax(-D2 * self.S_v.exp().detach(), -1)) if probs else (out, None)

    def weights(self, C_m):
        """(..., vectors): secim agirliklari, secilmeyen 0 -- saglik icin."""
        return self._weights(self._distance(C_m))

    def _distance(self, C_m):
        return 2 - 2 * F.normalize(C_m, dim=-1) @ F.normalize(self.start, dim=-1).T

    def _weights(self, D2):
        if self.active >= D2.shape[-1]:
            return torch.softmax(-D2 * self.S_v.exp(), -1)
        d, ids = D2.topk(self.active, dim=-1, largest=False)
        return torch.zeros_like(D2).scatter(-1, ids, torch.softmax(-d * self.S_v.exp(), -1))


class Attention(nn.Module):
    """Gecmisten okuma: C_m += W_o softmax(q.k / sqrt(dim) - m_h (t - j)) v, nedensel.  W_o = 0 baslar.
    first: anahtar bir ONCEKI konumun ham zinciri (induction'in ilk basi gomulu).  Degilse q, k, v islenmis
    durumdan: onceki katmanlarin yazdigini okur (Oe13).  distance: bas basina mesafe egilimi m_h, 0'dan (Oe14)."""

    def __init__(self, d, heads, dim, gen, first, distance):
        super().__init__()
        self.heads, self.dim, self.first = heads, dim, first
        init = lambda: nn.Parameter(torch.randn(heads * dim, d, generator=gen) / math.sqrt(d))
        self.W_q, self.W_k, self.W_v = init(), init(), init()
        self.W_o = nn.Parameter(torch.zeros(d, heads * dim))
        self.m = nn.Parameter(torch.zeros(heads)) if distance else None

    def forward(self, C, C_m):
        B, T, _ = C_m.shape
        q, k, v = self._qkv(C, C_m)
        bias = self._bias(T, q.dtype, q.device)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, is_causal=bias is None)
        return C_m + out.transpose(1, 2).reshape(B, T, -1) @ self.W_o.T

    def weights(self, C, C_m):
        """(B, heads, T, T): forward'in agirliklari -- saglik icin."""
        q, k, _ = self._qkv(C, C_m)
        T = q.shape[2]
        s = q @ k.transpose(-1, -2) / self.dim ** 0.5
        bias = self._bias(T, s.dtype, s.device)
        if bias is None:
            bias = torch.zeros(T, T, dtype=s.dtype, device=s.device).masked_fill(
                torch.ones(T, T, dtype=torch.bool, device=s.device).triu(1), float("-inf"))
        return (s + bias).softmax(-1)

    def _qkv(self, C, C_m):
        C_mn = F.normalize(C_m, dim=-1)
        if self.first:                                        # k_j = W_k norm(C_(j-1)); konum 0'in anahtari 0
            k = F.normalize(C, dim=-1) @ self.W_k.T
            k = torch.cat([torch.zeros_like(k[:, :1]), k[:, :-1]], 1)
        else:
            k = C_mn @ self.W_k.T
        return self._split(C_mn @ self.W_q.T), self._split(k), self._split(C_mn @ self.W_v.T)

    def _bias(self, T, dtype, device):
        """(heads, T, T): -m_h (t - j), gelecek -inf.  Mesafe egilimi yoksa None (yalniz nedensel)."""
        if self.m is None:
            return None
        pos = torch.arange(T, device=device)
        age = (pos[:, None] - pos[None, :]).to(dtype)
        return (-self.m.to(dtype)[:, None, None] * age).masked_fill(age < 0, float("-inf"))

    def _split(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.heads, self.dim).transpose(1, 2)


class Ledger(nn.Module):
    """Defter: hikayenin kendi gecmisi.  t, j < t - skip zincirlerine benzerlikle bakar; benzer C_j'den sonra
    gelen kelime (nxt_j = w_(j+1)) oy alir.  Butun gecmis (Oe3: R_CC her j'den gradyan alir).
    sim(t,j) = cos(norm(Ĉ_t + R_CC(Ĉ_t)), Ĉ_j): sinirli (Oe2).  gate = sigmoid(Ĉ_m.g_d + g_s max_j sim + g_0)."""

    def __init__(self, d, skip, s_c_init, gate_0_init, relation):
        super().__init__()
        self.skip = skip
        self.S_c = nn.Parameter(torch.tensor(float(s_c_init)))        # benzerlik keskinligi
        self.gate_d = nn.Parameter(torch.zeros(d))                      # gate'in 1. girdisi: durumun yonu
        self.gate_s = nn.Parameter(torch.zeros(()))                     # 2. girdisi: en yuksek benzerlik
        self.gate_0 = nn.Parameter(torch.tensor(float(gate_0_init)))
        self.relation = relation                                        # R_CC ya da None

    def neighbors(self, C):
        """C (B,T,d) -> W_c (B,T,T) j'ye agirlik (gecersiz 0), sim_max (B,T), valid (B,T): gecmisi olan konum."""
        T = C.shape[1]
        Cn = F.normalize(C, dim=-1)
        q = Cn if self.relation is None else F.normalize(Cn + self.relation(Cn), dim=-1)
        sim = q @ Cn.transpose(1, 2)
        pos = torch.arange(T, device=C.device)
        allowed = pos[None, :] < pos[:, None] - self.skip                # j < t - skip
        W_c = torch.softmax((sim * self.S_c.exp()).masked_fill(~allowed, -1e4), -1) * allowed
        valid = allowed.any(-1).expand(C.shape[0], T)
        sim_max = torch.where(valid, sim.masked_fill(~allowed, -1.0).max(-1).values, torch.zeros(()))
        return W_c, sim_max, valid

    def gate_inputs(self, C_m, sim_max):
        """-> (yon payi, benzerlik payi): gate = sigmoid(ikisinin toplami + g_0) (Oe9: ayri okunur)."""
        return F.normalize(C_m, dim=-1) @ self.gate_d, self.gate_s * sim_max

    def gate(self, C_m, sim_max, valid):
        a, b = self.gate_inputs(C_m, sim_max)
        return torch.sigmoid(a + b + self.gate_0) * valid

    def mix(self, tokens, C, C_m, scores):
        """Tam tablo: log p (B,T,n) = log[(1 - gate) softmax(scores) + gate p_defter]."""
        W_c, sim_max, valid = self.neighbors(C)
        T = tokens.shape[1]
        nxt = torch.roll(tokens, -1, 1)[:, None, :].expand(-1, T, -1)
        p_cache = torch.zeros_like(scores).scatter_add_(-1, nxt, W_c)
        gate = self.gate(C_m, sim_max, valid).unsqueeze(-1)
        return torch.logaddexp(torch.log1p(-gate) + torch.log_softmax(scores, -1),
                               torch.log(gate + 1e-12) + torch.log(p_cache + 1e-12))


class PointRelation(nn.Module):
    """model_19.  Kurucu ayarlari config() ile pakete yazilir, from_package() ile geri kurulur."""
    arch = "point_relation"

    def __init__(self, n, d_order=D_ORDER, d_content=D_CONTENT, vectors=VECTORS, active=ACTIVE, layers=LAYERS,
                 attn_after=ATTN_AFTER, attn_heads=ATTN_HEADS, attn_dim=ATTN_DIM, t_max=T_MAX, lam=LAM,
                 lam_w_init=LAM_W_INIT, beta_w_init=BETA_W_INIT, start_norm=START_NORM, s_v_init=S_V_INIT,
                 load_balance=LOAD_BALANCE, ledger=True, cache_skip=CACHE_SKIP, s_c_init=S_C_INIT,
                 gate_0_init=GATE_0_INIT, rank=RANK, embed=True, readout=True, chain_sim=True, distance=True,
                 seed=0):
        """Yeni parcalar: embed (E), readout (R_PC), chain_sim (R_CC), distance (mesafe egilimi), attn_after'in
        ilkinden sonrakiler (ikinci attention).  Hepsi kapali: zemin."""
        super().__init__()
        attn_after = tuple(sorted(int(i) for i in attn_after))
        assert all(0 <= i < layers for i in attn_after), (attn_after, layers)
        assert not chain_sim or ledger, "chain_sim defterde yasar"
        self._config = dict(n=n, d_order=d_order, d_content=d_content, vectors=vectors, active=active,
                            layers=layers, attn_after=list(attn_after), attn_heads=attn_heads, attn_dim=attn_dim,
                            t_max=t_max, lam=lam, lam_w_init=lam_w_init, beta_w_init=beta_w_init,
                            start_norm=start_norm, s_v_init=s_v_init, load_balance=load_balance, ledger=ledger,
                            cache_skip=cache_skip, s_c_init=s_c_init, gate_0_init=gate_0_init, rank=rank,
                            embed=embed, readout=readout, chain_sim=chain_sim, distance=distance, seed=seed)
        self.n, self.d_order, self.d_content, self.t_max = n, d_order, d_content, t_max
        self.d = d = d_order + d_content
        self.lam, self.load_balance, self.attn_after = float(lam), float(load_balance), attn_after
        gen = torch.Generator().manual_seed(seed)                        # zeminin parcalari
        # yeni parca basina AYRI uretec: bir parcanin baslangici digerlerinin acik olup olmamasina bagli degil (Oe4)
        part = lambda k: torch.Generator().manual_seed(seed + NEW_SEED * k)

        P = torch.randn(n, d, generator=gen)
        self.register_buffer("P", P / P.norm(dim=-1, keepdim=True))
        self.moves = nn.ModuleList(MoveLayer(d, vectors, active, gen, start_norm, s_v_init) for _ in range(layers))
        self.attns = nn.ModuleList(
            Attention(d, attn_heads, attn_dim, gen if k == 0 else part(3 + k), first=k == 0, distance=distance)
            for k in range(len(attn_after)))
        lam0 = (lam_w_init - LAM_W_MIN) / (1 - LAM_W_MIN)
        self.lam_w = nn.Parameter(torch.full((n, d_content), math.log(lam0 / (1 - lam0))))
        self.beta_w = nn.Parameter(torch.full((n, d_content), math.log(beta_w_init / (1 - beta_w_init))))
        # S_p: kusursuz eslesmede (cos 1, digerleri ~0) dogru kelimeye S_P_P olasilik -> e^S_p = ln(p/(1-p)(n-1))/2
        self.s_p_init = math.log(math.log(S_P_P / (1 - S_P_P) * (n - 1)) / 2)
        self.S_p = nn.Parameter(torch.tensor(self.s_p_init))

        # yeni parcalar: sifir etki
        self.embed_delta = self.embed_up = None
        if embed:                                                        # Oe8: kelime basina tablo
            if rank is None or rank >= d:
                self.embed_delta = nn.Parameter(torch.zeros(n, d))
            else:
                self.embed_delta = nn.Parameter(torch.randn(n, rank, generator=part(1)) / math.sqrt(rank))
                self.embed_up = nn.Parameter(torch.zeros(d, rank))
        self.readout = Relation(d, rank, part(2)) if readout else None   # R_PC
        self.ledger = (Ledger(d, cache_skip, s_c_init, gate_0_init, Relation(d, rank, part(3)) if chain_sim else None)
                       if ledger else None)

    def config(self):
        return dict(self._config)

    @classmethod
    def from_package(cls, k):
        m = cls(**k["config"])
        m.load_state_dict(k["weights"])
        return m.eval()

    # --- giris ve zincir
    def embed_shift(self, tokens):
        """dE[w] (..., d); embed kapaliyken None."""
        if self.embed_delta is None:
            return None
        u = self.embed_delta[tokens]
        return u if self.embed_up is None else u @ self.embed_up.T

    def embed(self, tokens):
        """E[w] = norm(P[w] + dE[w]) (Oe1: kurede); embed kapaliyken P[w]."""
        x, dE = self.P[tokens], self.embed_shift(tokens)
        return x if dE is None else F.normalize(x + dE, dim=-1)

    def chain(self, tokens):
        """tokens (B,T) -> C (B,T,d) = [sira | icerik].  Nedensel: C_t yalniz <= t'yi toplar."""
        assert tokens.shape[1] <= self.t_max, "dizi t_max'tan uzun"
        x = self.embed(tokens)
        return torch.cat([self._order(x[..., :self.d_order]), self._content(tokens, x[..., self.d_order:])], -1)

    def _order(self, x):
        """C_t = sum_i lam^(t-i) kaydir^(t-i) x_i = kaydir^t(sum_i lam^(t-i) kaydir^-i x_i); kaydir^k x[j] = x[j-k]."""
        T, D = x.shape[1], x.shape[-1]
        pos = torch.arange(T, device=x.device)[:, None]
        dim = torch.arange(D, device=x.device)[None, :]
        z = self._decayed_sum(x.gather(-1, ((dim + pos) % D).expand(x.shape)))
        return z.gather(-1, ((dim - pos) % D).expand(x.shape))

    def _decayed_sum(self, terms):
        """sum_(i<=t) lam^(t-i) terms_i, (T,T) alt ucgen carpanla (lam^-t'li cumsum 512'de tasar)."""
        age = torch.arange(terms.shape[1], device=terms.device)
        age = age[:, None] - age[None, :]
        decay = torch.where(age >= 0, self.lam ** age.clamp(min=0).float(), torch.zeros(()))
        return torch.einsum("ti,bid->btd", decay.to(terms.dtype), terms)

    def lam_beta(self, tokens):
        """-> (lam_w[w], beta_w[w]) (..., d_content): lam (LAM_W_MIN, 1), beta (0, 1)."""
        return LAM_W_MIN + (1 - LAM_W_MIN) * torch.sigmoid(self.lam_w[tokens]), torch.sigmoid(self.beta_w[tokens])

    def _content(self, tokens, x):
        """C_content_t = lam_w[w_t] C_content_(t-1) + beta_w[w_t] x_t, CONTENT_CHUNK'lik parcalarla, fp32."""
        lam, beta = self.lam_beta(tokens)
        u, log_lam = (beta * x).float(), lam.float().log()
        h, out = u.new_zeros(u.shape[0], u.shape[-1]), []
        for s in range(0, u.shape[1], CONTENT_CHUNK):
            L = log_lam[:, s:s + CONTENT_CHUNK].cumsum(1)
            y = L.exp() * (h[:, None] + ((-L).exp() * u[:, s:s + CONTENT_CHUNK]).cumsum(1))
            out.append(y)
            h = y[:, -1]
        return torch.cat(out, 1).to(x.dtype)

    # --- katmanlar ve cikis
    def _layers(self, C, probs=False, pick=None):
        """-> (C_m, denge icin katman basina P).  pick: son attention'dan sonraki katmanlar yalniz sayilan
        konumlarda calisir, C_m (N,d) doner (kayip yolu; hesap ayni)."""
        first = self.attn_after[-1] + 1 if self.attns else 0
        C_m, P = C, []
        for i, layer in enumerate(self.moves):
            if pick is not None and i == first:
                C_m = pick(C_m)
            C_m, p = layer(C_m, probs)
            if probs:
                P.append(p)
            if i in self.attn_after:
                C_m = self.attns[self.attn_after.index(i)](C, C_m)
        if pick is not None and first == len(self.moves):
            C_m = pick(C_m)
        return C_m, P

    def point(self, C_m, C):
        """z = 2 e^S_p norm(C_m) + R_PC(C): puan(w) = P[w] . z."""
        z = 2 * self.S_p.exp() * F.normalize(C_m, dim=-1)
        return z if self.readout is None else z + self.readout(C)

    def scoreboard(self, tokens, targets_mask=None):
        """tokens (B,T) -> (B,T,n): her konumda SONRAKI token.  Defter varsa log p (normalize)."""
        C = self.chain(tokens)
        C_m = self._layers(C)[0]
        scores = self.point(C_m, C) @ self.P.T
        return scores if self.ledger is None else self.ledger.mix(tokens, C, C_m, scores)

    def loss(self, tokens, targets_mask=None, parts=False, rows=None):
        """SONRAKI TOKEN: konum j, j+1'i tahmin eder; targets_mask verilirse yalniz isaretli hedefler.
        rows: (R,) duz konum (B*(T-1)), -1 dolgu -- puan tablosu yalniz orada (sonuc rows'suz ile ayni).
        parts: (toplam, NLL)."""
        counted = (torch.ones_like(tokens[:, 1:], dtype=torch.bool) if targets_mask is None
                   else targets_mask[:, 1:].bool())
        flat = lambda x: x[:, :-1].reshape(-1, *x.shape[2:])
        target, w = tokens[:, 1:].reshape(-1), counted.reshape(-1)
        if rows is None:
            pick = flat
        else:
            keep, rows = rows >= 0, rows.clamp(min=0)
            pick = lambda x: flat(x)[rows]
            target, w = target[rows], w[rows] & keep
        C = self.chain(tokens)
        balanced = self.load_balance > 0
        C_m, P = self._layers(C, probs=balanced, pick=pick if rows is not None else None)
        if C_m.dim() == 3:
            C_m = pick(C_m)
        s = self.point(C_m, pick(C)) @ self.P.T                              # (N, n)
        if self.ledger is None:
            nll = F.cross_entropy(s, target, reduction="none")
        else:
            W_c, sim_max, valid = self.ledger.neighbors(C)
            nxt = torch.roll(tokens, -1, 1)                                  # nxt_j = w_(j+1); t'nin hedefi nxt_t
            p_cache = pick((W_c * (nxt[:, None, :] == nxt[:, :, None])).sum(-1))
            gate = self.ledger.gate(C_m, pick(sim_max), pick(valid))
            log_model = s.gather(-1, target[:, None])[:, 0] - s.logsumexp(-1)
            nll = -torch.logaddexp(torch.log1p(-gate) + log_model,
                                   torch.log(gate + 1e-12) + torch.log(p_cache + 1e-12))
        w = w.to(nll.dtype)
        nll = (nll * w).sum() / w.sum()
        loss = nll + self.load_balance * self._balance(P, counted, w) if balanced else nll
        return (loss, nll) if parts else loss

    def _balance(self, P, counted, w):
        """Denge: katman basina vectors * sum_a Pbar_a^2 (en kucuk 1: kullanim esit), sayilan konumlarda."""
        def mean_p(p):
            a, x = (counted, p[:, :-1]) if p.dim() == 3 else (w, p)
            a = a.to(p.dtype).unsqueeze(-1)
            return (x * a).sum(tuple(range(x.dim() - 1))) / a.sum()
        return sum(p.shape[-1] * (mean_p(p) ** 2).sum() for p in P)

    # --- okuma araclari (hesaba girmez)
    @torch.no_grad()
    def inspect(self, tokens):
        """Ara degerler: C, C_m, katman basina secim W ve girdi boyu, attention basina agirlik, defter."""
        C = self.chain(tokens)
        C_m, W, norm_in, A = C, [], [], []
        for i, layer in enumerate(self.moves):
            norm_in.append(C_m.norm(dim=-1))
            W.append(layer.weights(C_m))
            C_m = layer(C_m)[0]
            if i in self.attn_after:
                att = self.attns[self.attn_after.index(i)]
                A.append(att.weights(C, C_m))
                C_m = att(C, C_m)
        r = {"C": C, "C_m": C_m, "W": W, "norm_in": norm_in, "attn": A}
        if self.ledger is not None:
            W_c, sim_max, valid = self.ledger.neighbors(C)
            r["gate_dir"], r["gate_sim"] = self.ledger.gate_inputs(C_m, sim_max)
            r.update(gate=self.ledger.gate(C_m, sim_max, valid), W_c=W_c, valid=valid)
        return r

    @torch.no_grad()
    def health(self, tokens, mask):
        """Saglik: sabit bir sondada modelin ici.  tokens (B,T), mask (B,T) gercek konumlar -> sozluk.
          vec_each_l / vec_all_l / dead_l   katman l'de etkin hareket (konum basina / butun konumlarda) ve olu
          norm_in_l                         katman l'nin girdisinin boyu (Oe15)
          norm_C, dir_C, norm_Cm, dir_Cm    boy ve varyansin %90'ini tasiyan yon sayisi
          exp_S_v, exp_S_p, lam_c           ogrenilen olcekler, icerik sonmesi
          gate, gate_dir, gate_sim, exp_S_c, ledger_eff   defter: gate ve iki girdisi, etkin komsu sayisi (Oe3)
          attn<k>_ent / _first / _dist / _m bas basina: entropi, konum 0 payi, bakis mesafesi, mesafe egilimi
          embed_shift, readout, chain_sim   yeni parcalarin boyu (Oe6)
          pred_distinct, pred_ent           en yuksek puani alan farkli kelime, tahmin entropisi"""
        a = mask.bool()
        ins, h = self.inspect(tokens), {}
        for i, W in enumerate(ins["W"]):
            W_bar = W[a].mean(0)
            h["vec_each_%d" % i] = float(entropy(W[a]).exp().mean())
            h["vec_all_%d" % i] = float(entropy(W_bar).exp())
            h["dead_%d" % i] = int((W_bar < 0.01 / W_bar.numel()).sum())
            h["norm_in_%d" % i] = float(ins["norm_in"][i][a].mean())
        for name, X in (("C", ins["C"][a]), ("Cm", ins["C_m"][a])):
            h["norm_" + name], h["dir_" + name] = float(X.norm(dim=-1).mean()), directions90(X)
        h["exp_S_v"] = [float(L.S_v.exp()) for L in self.moves]
        h["exp_S_p"] = float(self.S_p.exp())
        h["lam_c"] = float(self.lam_beta(tokens)[0][a].mean())
        if self.ledger is not None:
            v = ins["valid"] & a
            h["gate"] = float(ins["gate"][a].mean())
            h["gate_dir"], h["gate_sim"] = float(ins["gate_dir"][a].mean()), float(ins["gate_sim"][a].mean())
            h["exp_S_c"] = float(self.ledger.S_c.exp())
            h["ledger_eff"] = float(entropy(ins["W_c"][v]).exp().mean()) if v.any() else 0.0
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        age = (pos[:, None] - pos[None, :]).clamp(min=0).float()
        for k, A in enumerate(ins["attn"]):
            for name, x in (("ent", entropy(A)), ("first", A[..., 0]), ("dist", (A * age).sum(-1))):
                h["attn%d_%s" % (k, name)] = [float(x[:, j][a].mean()) for j in range(A.shape[1])]
            if self.attns[k].m is not None:
                h["attn%d_m" % k] = [float(x) for x in self.attns[k].m]
        if self.embed_delta is not None:
            h["embed_shift"] = float(self.embed_shift(torch.arange(self.n, device=self.P.device)).norm(dim=-1).mean())
        if self.readout is not None:
            h["readout"] = self.readout.size()
        if self.ledger is not None and self.ledger.relation is not None:
            h["chain_sim"] = self.ledger.relation.size()
        lp = torch.log_softmax(self.scoreboard(tokens), -1)[a]
        h["pred_distinct"] = int(lp.argmax(-1).unique().numel())
        h["pred_ent"] = float(-(lp.exp() * lp).sum(-1).mean())
        return h
