# -*- coding: utf-8 -*-
"""looped_19 -- LoopedRelation (tasarim adi "Bul-Bak Dongusu"): uc iliski giriste kaynak, tek blok ayni
agirliklarla en cok K_MAX kez.
Tasarim: TASARIM_19.md "Bul-Bak Dongusu" (26 Eylul); artifact "Bul–Bak Döngüsü".

    E[w]   = norm(P[w] + U[w] W^T)                                                    R_PP
    C_t    = sum_(i<=t) LAM^(t-i) kaydir^(t-i) E_i                                     sira zinciri
    R_t    = R_PC C_t                                                                  zincirden kelimeye
    W(t,j) = softmax_(j < t-skip)(e^S_c cos(norm(C^_t + R_CC C^_t), C^_j))             defter, R_CC
    D_t    = sum_j W(t,j) E[w_(j+1)]                                                   defterin onerisi
    x0     = norm(norm(C) + a norm(R) + b norm(D))                                     a = b = 0 baslar
    blok   adim adim: find, lookup1, lookup2; agirliklar paylasilir.  step_norm:
             'input'    x <- norm(x) + parca                      (Oe15: yalniz girdi kureye)
             'bounded'  x <- norm(norm(x) + alpha_i . norm(parca))  (nGPT gibi cikti da; alpha = 0 baslar)
    dur    konum t, gecis k: max_(j<=t) |norm(x^k_j) - norm(x^(k-1)_j)| < eps  ya da  k = K_MAX
    p(w)   = softmax(P[w] . 2 e^S_p norm(x))       (secenek: (1 - g) p + g p_defter, acik kopya)

'input'te parcanin boyu sinirsiz: CPU'da (26 Eylul) 2. adimdan itibaren gecis 4'te butun konumlar ayni durumda
(cos 1,00), lookup'lar 8 yuvaya coktu.  'bounded' her adimin etkisini alpha ile sinirlar.
Kayip hep K_MAX gecisle (cevabi korumayi ogrensin); durma kurali yalniz scoreboard'da.  Durma t'nin yalniz kendisine
ve oncesine bakar: bir konumun gecis sayisi sonraki token'lara baglanmaz (nedensel).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_19 import MoveLayer, entropy

D = 1024               # kullanici, 25 Eylul: "D 1024, hikayedekiyle ayni"
RANK = 256             # iliski matrisleri ve E'nin ranki; kullanici: "R 256"
VECTORS = 1024         # lookup katmani basina yuva (tasarim, 26 Eylul; olculmedi)
ACTIVE = 8             # yone gore en yakin k (model_18'den; olculmedi)
HEADS, HEAD_DIM = 4, 64    # find (model_18'den; olculmedi)
TRACE_HEADS = 1        # anahtari ifade izi (bir onceki konumun zinciri) olan bas (tasarim; olculmedi)
K_MAX = 4              # en cok gecis; kullanici, 26 Eylul: "K_4 şimdilik max koyalım"
STOP_EPS = 0.01        # durma esigi |norm fark| (olculmedi; olcumle secilecek)
LAM = 0.7              # sira zincirinin sonmesi (model_18'den; olculmedi)
CACHE_SKIP = 3         # defter son 3 konumu aramaz (model_18'den; olculmedi)
S_C_INIT = 3.0         # defter benzerlik keskinligi baslangici (model_18'den)
S_P_P = 0.9            # S_p baslangici: kusursuz eslesmede dogru kelimeye bu olasilik (model_19)
START_NORM = 1.0       # hareket start'larinin baslangic boyu (model_18'den)
S_V_INIT = 0.0         # hareket secim keskinligi baslangici (model_18'den)
LOAD_BALANCE = 0.01    # top-k cokusune karsi denge (model_18'de olculdu; burada dogrulanmadi)
GATE_0_INIT = -2.0     # acik kopya kanali (secenek) gate'i ~0,12'den
T_MAX = 512            # bellek korumasi, konum siniri degil; kullanici: "512 limiti"
STEP_NORM = "input"    # 'input' (tasarim, Oe15) ya da 'bounded' (nGPT; oneri, kullanici karari bekliyor)
NEW_SEED = 1_000_003   # parca basina ayri uretec: seed + NEW_SEED * k


def unit_or_zero(v, valid):
    """norm(v), gecersiz satirda 0.  norm(0)'in tersine gradyani sinirli (fp32'de 1e12 gradyan sifirla carpiliyordu,
    fp16'da ileri hesap NaN): gecersiz satir once sifirdan farkli yapilir, sonra maskelenir."""
    return torch.where(valid[..., None], F.normalize(v + (~valid)[..., None].to(v.dtype), dim=-1), v.new_zeros(()))


def order_chain(x, lam):
    """C_t = sum_(i<=t) lam^(t-i) kaydir^(t-i) x_i;  kaydir^k x[j] = x[j-k].  (T,T) alt ucgen carpanla."""
    T, Dm = x.shape[1], x.shape[-1]
    pos = torch.arange(T, device=x.device)[:, None]
    dim = torch.arange(Dm, device=x.device)[None, :]
    age = pos - pos.T
    decay = torch.where(age >= 0, lam ** age.clamp(min=0).to(x.dtype), x.new_zeros(()))   # x.dtype: float64 denetim tam
    z = torch.einsum("ti,bid->btd", decay, x.gather(-1, ((dim + pos) % Dm).expand(x.shape)))
    return z.gather(-1, ((dim - pos) % Dm).expand(x.shape))


class LowRank(nn.Module):
    """x -> A B^T x.  zero: B = 0 (carpim 0, gradyan A uzerinden akar); degilse ikisi kucuk rastgele -- norm(R x)
    baslangicta tanimli olsun (R_PC).  rank None ya da >= d: tam matris."""

    def __init__(self, d, rank, gen, zero):
        super().__init__()
        if rank is None or rank >= d:
            self.M = nn.Parameter(torch.zeros(d, d) if zero else torch.randn(d, d, generator=gen) / d)
            self.A = self.B = None
        else:
            self.M = None
            self.A = nn.Parameter(torch.randn(d, rank, generator=gen) / math.sqrt(d))
            self.B = nn.Parameter(torch.zeros(d, rank) if zero else torch.randn(d, rank, generator=gen) / math.sqrt(d))

    def forward(self, x):
        return x @ self.M.T if self.M is not None else (x @ self.B) @ self.A.T

    def matrix(self):
        return self.M if self.M is not None else self.A @ self.B.T

    def size(self):
        return float(self.matrix().norm())


class FindAttention(nn.Module):
    """Find (bul): attention.  q, k, v onceki gecisin durumundan; ilk `trace` basin anahtari bir ONCEKI konumun zinciri
    (ifade izi), konum 0'inki 0.  Mesafe egilimi -m_h (t - j), 0'dan.  x <- norm(x) + W_o softmax(.) v; W_o = 0 baslar."""

    def __init__(self, d, heads, dim, trace, gen):
        super().__init__()
        self.heads, self.dim, self.trace = heads, dim, trace
        init = lambda: nn.Parameter(torch.randn(heads * dim, d, generator=gen) / math.sqrt(d))
        self.W_q, self.W_k, self.W_v = init(), init(), init()
        self.W_o = nn.Parameter(torch.zeros(d, heads * dim))
        self.m = nn.Parameter(torch.zeros(heads))

    def _split(self, z):
        B, T, _ = z.shape
        return z.view(B, T, self.heads, self.dim).transpose(1, 2)

    def logits(self, C, x):
        """(B, heads, T, T): q.k / sqrt(dim) - m_h (t - j), gelecek -inf."""
        T = x.shape[1]
        xn = F.normalize(x, dim=-1)
        cut = self.trace * self.dim
        k = xn @ self.W_k[cut:].T
        if cut:
            Cn = F.normalize(C, dim=-1)
            prev = torch.cat([torch.zeros_like(Cn[:, :1]), Cn[:, :-1]], 1)
            k = torch.cat([prev @ self.W_k[:cut].T, k], -1)
        s = self._split(xn @ self.W_q.T) @ self._split(k).transpose(-1, -2) / math.sqrt(self.dim)
        pos = torch.arange(T, device=x.device)
        age = (pos[:, None] - pos[None, :]).to(s.dtype)
        return (s - self.m.to(s.dtype)[:, None, None] * age).masked_fill(age < 0, float("-inf"))

    def values(self, x):
        """(B, heads, T, d): bas h'nin j konumundan getirecegi, W_o dahil."""
        v = self._split(F.normalize(x, dim=-1) @ self.W_v.T)                       # (B, H, T, dim)
        return torch.einsum("dhk,bhjk->bhjd", self.W_o.view(-1, self.heads, self.dim), v)

    def output(self, C, x):
        """W_o softmax(.) v: attention'in ekledigi (artik yok)."""
        A = torch.softmax(self.logits(C, x), -1)
        v = self._split(F.normalize(x, dim=-1) @ self.W_v.T)
        return (A @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], -1) @ self.W_o.T


class LedgerSource(nn.Module):
    """Ledger (defter), girdi kaynagi: t, j < t - skip zincirlerine benzerlik (R_CC ogrenir, 0'dan); agirliklar W (B,T,T), gecersiz 0."""

    def __init__(self, d, rank, skip, s_c_init, gen):
        super().__init__()
        self.skip = skip
        self.relation = LowRank(d, rank, gen, zero=True)
        self.S_c = nn.Parameter(torch.tensor(float(s_c_init)))

    def weights(self, C):
        """-> W (B,T,T), sim (B,T,T), valid (B,T): gecmisi olan konum."""
        T = C.shape[1]
        Cn = F.normalize(C, dim=-1)
        sim = F.normalize(Cn + self.relation(Cn), dim=-1) @ Cn.transpose(1, 2)
        pos = torch.arange(T, device=C.device)
        allowed = pos[None, :] < pos[:, None] - self.skip
        logits = (sim * self.S_c.exp()).masked_fill(~allowed, torch.finfo(sim.dtype).min)
        return torch.softmax(logits, -1) * allowed, sim, allowed.any(-1).expand(C.shape[0], T)


class LoopedRelation(nn.Module):
    """LoopedRelation (Bul-Bak Dongusu).  Kurucu ayarlari config() ile pakete yazilir, from_package() ile geri kurulur."""
    arch = "looped_relation"

    def __init__(self, n, d=D, rank=RANK, vectors=VECTORS, active=ACTIVE, heads=HEADS, head_dim=HEAD_DIM,
                 trace_heads=TRACE_HEADS, k_max=K_MAX, stop_eps=STOP_EPS, lam=LAM, cache_skip=CACHE_SKIP,
                 s_c_init=S_C_INIT, start_norm=START_NORM, s_v_init=S_V_INIT, load_balance=LOAD_BALANCE,
                 copy_gate=False, gate_0_init=GATE_0_INIT, t_max=T_MAX, step_norm=STEP_NORM, seed=0):
        super().__init__()
        assert 0 <= trace_heads <= heads and k_max >= 1 and step_norm in ("input", "bounded")
        self._config = dict(n=n, d=d, rank=rank, vectors=vectors, active=active, heads=heads, head_dim=head_dim,
                            trace_heads=trace_heads, k_max=k_max, stop_eps=stop_eps, lam=lam, cache_skip=cache_skip,
                            s_c_init=s_c_init, start_norm=start_norm, s_v_init=s_v_init, load_balance=load_balance,
                            copy_gate=copy_gate, gate_0_init=gate_0_init, t_max=t_max, step_norm=step_norm,
                            seed=seed)
        self.n, self.d, self.t_max, self.k_max = n, d, t_max, k_max
        self.stop_eps, self.lam, self.load_balance = float(stop_eps), float(lam), float(load_balance)
        gen = torch.Generator().manual_seed(seed)
        part = lambda k: torch.Generator().manual_seed(seed + NEW_SEED * k)

        P = torch.randn(n, d, generator=gen)
        self.register_buffer("P", P / P.norm(dim=-1, keepdim=True))
        self.find = FindAttention(d, heads, head_dim, trace_heads, gen)
        self.lookups = nn.ModuleList(MoveLayer(d, vectors, active, gen, start_norm, s_v_init) for _ in range(2))
        self.s_p_init = math.log(math.log(S_P_P / (1 - S_P_P) * (n - 1)) / 2)
        self.S_p = nn.Parameter(torch.tensor(self.s_p_init))
        # E = norm(P + U W^T): W = 0 (E = P baslar); rank None ya da >= d: tam tablo, 0'dan
        if rank is None or rank >= d:
            self.embed_delta, self.embed_up = nn.Parameter(torch.zeros(n, d)), None
        else:
            self.embed_delta = nn.Parameter(torch.randn(n, rank, generator=part(1)) / math.sqrt(rank))
            self.embed_up = nn.Parameter(torch.zeros(d, rank))
        self.readout = LowRank(d, rank, part(2), zero=False)            # R_PC
        self.ledger = LedgerSource(d, rank, cache_skip, s_c_init, part(3))    # R_CC
        self.source_weights = nn.Parameter(torch.zeros(2))              # a (R_PC), b (defter): 0'dan
        self.gate_d = nn.Parameter(torch.zeros(d)) if copy_gate else None
        self.gate_0 = nn.Parameter(torch.tensor(float(gate_0_init))) if copy_gate else None
        # bounded: adim basina (find, lookup1, lookup2) boyut boyut adim boyu, 0'dan (adim 0'da etkisiz).  norm(0)
        # tanimsiz oldugu icin parcalar sifirdan baslayamaz: W_o ve bitis - baslangic kucuk rastgele.
        self.alpha = None
        if step_norm == "bounded":
            self.alpha = nn.Parameter(torch.zeros(3, d))
            g4 = part(4)
            with torch.no_grad():
                self.find.W_o.copy_(torch.randn(self.find.W_o.shape, generator=g4) / math.sqrt(heads * head_dim))
                for layer in self.lookups:
                    layer.finish.add_(torch.randn(layer.finish.shape, generator=g4) * (start_norm / math.sqrt(d)))

    def config(self):
        return dict(self._config)

    @classmethod
    def from_package(cls, k):
        m = cls(**k["config"])
        m.load_state_dict(k["weights"])
        return m.eval()

    # --- giris: uc kaynak
    def embed_shift(self, tokens):
        u = self.embed_delta[tokens]
        return u if self.embed_up is None else u @ self.embed_up.T

    def embed(self, tokens):
        return F.normalize(self.P[tokens] + self.embed_shift(tokens), dim=-1)

    def sources(self, tokens):
        """tokens (B,T) -> E, C, R (R_PC C), W (defter), D (defterin onerisi), valid, x0."""
        assert tokens.shape[1] <= self.t_max, "dizi t_max'tan uzun (bellek korumasi; konum siniri degil)"
        E = self.embed(tokens)
        C = order_chain(E, self.lam)
        R = self.readout(C)
        W, sim, valid = self.ledger.weights(C)
        Dv = W @ torch.roll(E, -1, 1)                                  # E[w_(j+1)]; j = T-1 hic gecerli degil
        a, b = self.source_weights[0], self.source_weights[1]
        x0 = F.normalize(F.normalize(C, dim=-1) + a * F.normalize(R, dim=-1) + b * unit_or_zero(Dv, valid), dim=-1)
        return dict(E=E, C=C, R=R, W=W, sim=sim, D=Dv, valid=valid, x0=x0)

    # --- blok ve dongu
    def _step(self, x, part, i):
        """Bir adim (i: 0 find, 1-2 lookup).  'input': norm(x) + parca;  'bounded': norm(norm(x) + alpha_i . norm(parca))."""
        xn = F.normalize(x, dim=-1)
        if self.alpha is None:
            return xn + part
        return F.normalize(xn + self.alpha[i] * F.normalize(part, dim=-1), dim=-1)

    @staticmethod
    def _move(layer, x, probs=False):
        """Lookup katmaninin itisi: yone en yakin `active` yuvanin (bitis - baslangic) agirlikli toplami; probs: denge icin
        butun yuvalar uzerinde softmax (S_v sabit)."""
        D2 = layer._distance(x)
        delta = layer._weights(D2) @ (layer.finish - layer.start)
        return delta, (torch.softmax(-D2 * layer.S_v.exp().detach(), -1) if probs else None)

    def block(self, C, x, probs=False):
        """Bir gecis: find, lookup 1, lookup 2.  -> (x, denge icin katman basina P)."""
        x = self._step(x, self.find.output(C, x), 0)
        P = []
        for i, layer in enumerate(self.lookups):
            delta, p = self._move(layer, x, probs)
            x = self._step(x, delta, i + 1)
            if probs:
                P.append(p)
        return x, P

    def run(self, tokens, stop=False, probs=False):
        """-> (son durum (B,T,d), bilgi: kaynaklar + passes (B,T) konum basina gecis + probs).
        stop: konum t, prefiksindeki (j <= t) en buyuk degisim eps'ten kucukse durur; durmus konum degismez."""
        src = self.sources(tokens)
        C, x = src["C"], src["x0"]
        done = torch.zeros(tokens.shape, dtype=torch.bool, device=tokens.device)
        passes = torch.zeros(tokens.shape, dtype=torch.long, device=tokens.device)
        P = []
        for _ in range(self.k_max):
            new, p = self.block(C, x, probs)
            P += p
            passes = passes + (~done).long()
            if stop:
                new = torch.where(done[..., None], x, new)
                change = (F.normalize(new, dim=-1) - F.normalize(x, dim=-1)).norm(dim=-1)
                done = done | (torch.cummax(change, 1).values < self.stop_eps)
                x = new
                if bool(done.all()):
                    break
            else:
                x = new
        return x, dict(src, passes=passes, probs=P)

    # --- cikis
    @torch.no_grad()
    def settle_passes(self, tokens):
        """Konum basina, K_MAX gecisle kosulunca KENDI degisiminin esigin altina ilk dustugu gecis (hic dusmezse
        K_MAX).  Durma kuralinin gecisi onek yuzunden istem boyuyla buyur; bu olcu boydan bagimsiz (yine nedensel)."""
        src = self.sources(tokens)
        C, x = src["C"], src["x0"]
        settled = torch.full(tokens.shape, self.k_max, dtype=torch.long, device=tokens.device)
        for k in range(1, self.k_max + 1):
            new = self.block(C, x)[0]
            change = (F.normalize(new, dim=-1) - F.normalize(x, dim=-1)).norm(dim=-1)
            settled = torch.where((change < self.stop_eps) & (settled == self.k_max), torch.full_like(settled, k), settled)
            x = new
        return settled

    def point(self, x):
        return 2 * self.S_p.exp() * F.normalize(x, dim=-1)

    def _gate_logits(self, x):
        return F.normalize(x, dim=-1) @ self.gate_d + self.gate_0

    def scoreboard(self, tokens, stop=True):
        """tokens (B,T) -> (B,T,n): her konumda SONRAKI token'in log p'si.  stop: durma kurali (degerlendirme)."""
        x, info = self.run(tokens, stop=stop)
        scores = self.point(x) @ self.P.T
        if self.gate_d is None:
            return torch.log_softmax(scores, -1)
        T = tokens.shape[1]
        nxt = torch.roll(tokens, -1, 1)[:, None, :].expand(-1, T, -1)
        p_cache = torch.zeros_like(scores).scatter_add_(-1, nxt, info["W"])
        g, valid = self._gate_logits(x), info["valid"]
        log_1mg = torch.where(valid, F.logsigmoid(-g), g.new_zeros(()))
        log_g = torch.where(valid, F.logsigmoid(g), g.new_full((), -float("inf")))
        return torch.logaddexp(log_1mg[..., None] + torch.log_softmax(scores, -1), log_g[..., None] + torch.log(p_cache + 1e-12))

    def loss(self, tokens, targets_mask=None, parts=False, rows=None):
        """SONRAKI TOKEN, hep K_MAX gecis.  rows: (R,) duz konum (B*(T-1)), -1 dolgu.  parts: (toplam, NLL)."""
        counted = (torch.ones_like(tokens[:, 1:], dtype=torch.bool) if targets_mask is None
                   else targets_mask[:, 1:].bool())
        flat = lambda z: z[:, :-1].reshape(-1, *z.shape[2:])
        target, w = tokens[:, 1:].reshape(-1), counted.reshape(-1)
        if rows is None:
            pick = flat
        else:
            keep, rows = rows >= 0, rows.clamp(min=0)
            pick = lambda z: flat(z)[rows]
            target, w = target[rows], w[rows] & keep
        balanced = self.load_balance > 0
        x, info = self.run(tokens, stop=False, probs=balanced)
        s = self.point(pick(x)) @ self.P.T
        if self.gate_d is None:
            nll = F.cross_entropy(s, target, reduction="none")
        else:
            nxt = torch.roll(tokens, -1, 1)
            p_cache = pick((info["W"] * (nxt[:, None, :] == nxt[:, :, None])).sum(-1))
            g, valid = pick(self._gate_logits(x)), pick(info["valid"])
            log_1mg = torch.where(valid, F.logsigmoid(-g), g.new_zeros(()))
            log_g = torch.where(valid, F.logsigmoid(g), g.new_full((), -float("inf")))
            log_model = s.gather(-1, target[:, None])[:, 0] - s.logsumexp(-1)
            nll = -torch.logaddexp(log_1mg + log_model, log_g + torch.log(p_cache + 1e-12))
        w = w.to(nll.dtype)
        nll = (nll * w).sum() / w.sum().clamp_min(1)                 # hedefsiz batch: 0, NaN degil
        loss = nll + self.load_balance * self._balance(info["probs"], counted) if balanced else nll
        return (loss, nll) if parts else loss

    def _balance(self, P, counted):
        """Denge: katman basina vectors * sum_a Pbar_a^2 (en kucuk 1), gecisler uzerinde ortalama, sayilan konumlarda."""
        a = counted.to(P[0].dtype).unsqueeze(-1)
        total = sum(p.shape[-1] * (((p[:, :-1] * a).sum((0, 1)) / a.sum().clamp_min(1)) ** 2).sum() for p in P)
        return total * len(self.lookups) / len(P)

    # --- okuma araclari (hesaba girmez)
    @torch.no_grad()
    def trace(self, tokens, stop=True):
        """tokens (T,) -> tam ayrisma.  parts: [ad, gecis, parca (T,d), carpan (T,)]; son durum = sum carpan * parca
        (birebir; kureye inisler yalniz sayiyla carpar).  Ayrica passes (T,), A[gecis] (H,T,T),
        ov[gecis] (H,T,d) bas h'nin j'den getirdigi, moves[gecis] = [(ids, w)] katman basina, x (T,d)."""
        src = self.sources(tokens[None])
        C, x = src["C"], src["x0"]
        T = tokens.shape[0]
        a, b = self.source_weights[0], self.source_weights[1]
        raw = [("chain", F.normalize(C, dim=-1)[0]), ("R_PC", a * F.normalize(src["R"], dim=-1)[0]),
               ("ledger", b * unit_or_zero(src["D"], src["valid"])[0])]
        first = 1 / sum(p for _, p in raw).norm(dim=-1).clamp_min(1e-12)
        parts = [[name, 0, p, first.clone()] for name, p in raw]
        done = torch.zeros(T, dtype=torch.bool, device=tokens.device)
        passes = torch.zeros(T, dtype=torch.long, device=tokens.device)
        A_all, ov_all, moves = [], [], []

        def enter(y, active):
            s = torch.where(active, 1 / y[0].norm(dim=-1).clamp_min(1e-12), torch.ones_like(first))
            for q in parts:
                q[3] = q[3] * s

        def add(y, part, name, k, i, active):
            """_step'in ayrismasi: girdi kureye (carpan), parcanin katkisi eklenir; bounded'da katki alpha . norm(parca)
            ve toplam yine kureye (ikinci carpan).  -> yeni durum (1,T,d)."""
            enter(y, active)
            c = part[0] if self.alpha is None else self.alpha[i] * F.normalize(part[0], dim=-1)
            parts.append([name, k, c * active[:, None], torch.ones_like(first)])
            new = self._step(y, part, i)
            if self.alpha is not None:
                enter(F.normalize(y, dim=-1) + c[None], active)
            return new

        for k in range(1, self.k_max + 1):
            active = ~done
            A = torch.softmax(self.find.logits(C, x), -1)[0]
            ov = self.find.values(x)[0]
            y = add(x, torch.einsum("htj,hjd->td", A, ov)[None], "find", k, 0, active)
            step = []
            for i, layer in enumerate(self.lookups):
                D2 = layer._distance(y)[0]
                dist, ids = D2.topk(min(layer.active, D2.shape[-1]), dim=-1, largest=False)
                wts = torch.softmax(-dist * layer.S_v.exp(), -1)
                step.append((ids, wts))
                delta = torch.einsum("tk,tkd->td", wts, (layer.finish - layer.start)[ids])
                y = add(y, delta[None], "lookup%d" % (i + 1), k, i + 1, active)
            A_all.append(A)
            ov_all.append(ov)
            moves.append(step)
            passes += active.long()
            new = torch.where(done[None, :, None], x, y)
            if stop:
                change = (F.normalize(new, dim=-1) - F.normalize(x, dim=-1)).norm(dim=-1)[0]
                done = done | (torch.cummax(change, 0).values < self.stop_eps)
                x = new
                if bool(done.all()):
                    break
            else:
                x = new
        return dict(parts=parts, passes=passes, A=A_all, ov=ov_all, moves=moves, x=x[0], sources=src)

    @torch.no_grad()
    def health(self, tokens, mask):
        """Saglik: tokens (B,T), mask (B,T) gercek konumlar -> sozluk.
          passes, passes_at_max            durma kuraliyla konum basina gecis, K_MAX'a varan pay
          change_k, norm_x_k               K_MAX gecisle: gecis k'de durumun degisimi ve boyu (kureye inmeden)
          vec_each_l / vec_all_l / dead_l  son geciste lookup katmani l: etkin hareket (konum basina / butun) ve olu
          attn_ent_k / first_k / dist_k    gecis k'de find: entropi, konum 0 payi, bakis mesafesi; attn_m
          src_a, src_b, ledger_eff         kaynak agirliklari, defterin etkin komsu sayisi
          same_dir                         son gecis, lookup 1 girdisi: konumlar arasi ortalama cos (1: hepsi ayni)
          alpha                            bounded: adim basina ortalama |alpha| (find, lookup1, lookup2)
          exp_S_p, exp_S_c, norm_C, embed_shift, readout, chain_sim, gate, pred_distinct, pred_ent"""
        a = mask.bool()
        src = self.sources(tokens)
        C, x = src["C"], src["x0"]
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        age = (pos[:, None] - pos[None, :]).clamp(min=0).float()
        h = {}
        for k in range(self.k_max):
            A = torch.softmax(self.find.logits(C, x), -1)
            h["attn_ent_%d" % k] = float(entropy(A).mean(1)[a].mean())
            h["attn_first_%d" % k] = float(A[..., 0].mean(1)[a].mean())
            h["attn_dist_%d" % k] = float((A * age).sum(-1).mean(1)[a].mean())
            y = self._step(x, self.find.output(C, x), 0)
            if k == self.k_max - 1:                                     # konumlar ayrisiyor mu (cos 1: cokus)
                yn = F.normalize(y, dim=-1)[a]
                yn = yn[torch.randperm(len(yn), generator=torch.Generator().manual_seed(0))[:512].to(yn.device)]
                h["same_dir"] = float((yn @ yn.T)[~torch.eye(len(yn), dtype=torch.bool, device=yn.device)].mean())
            for i, layer in enumerate(self.lookups):
                if k == self.k_max - 1:
                    W = layer.weights(y)
                    W_bar = W[a].mean(0)
                    h["vec_each_%d" % i] = float(entropy(W[a]).exp().mean())
                    h["vec_all_%d" % i] = float(entropy(W_bar).exp())
                    h["dead_%d" % i] = int((W_bar < 0.01 / W_bar.numel()).sum())
                y = self._step(y, self._move(layer, y)[0], i + 1)
            h["change_%d" % k] = float((F.normalize(y, dim=-1) - F.normalize(x, dim=-1)).norm(dim=-1)[a].mean())
            h["norm_x_%d" % k] = float(y.norm(dim=-1)[a].mean())
            x = y
        _, info = self.run(tokens, stop=True)
        h["passes"] = float(info["passes"][a].float().mean())
        h["passes_at_max"] = float((info["passes"][a] == self.k_max).float().mean())
        h["attn_m"] = [float(v) for v in self.find.m]
        if self.alpha is not None:
            h["alpha"] = [float(v) for v in self.alpha.abs().mean(-1)]
        h["src_a"], h["src_b"] = float(self.source_weights[0]), float(self.source_weights[1])
        v = src["valid"] & a
        h["ledger_eff"] = float(entropy(src["W"][v]).exp().mean()) if v.any() else 0.0
        h["exp_S_p"], h["exp_S_c"] = float(self.S_p.exp()), float(self.ledger.S_c.exp())
        h["norm_C"] = float(C.norm(dim=-1)[a].mean())
        h["embed_shift"] = float(self.embed_shift(torch.arange(self.n, device=tokens.device)).norm(dim=-1).mean())
        h["readout"], h["chain_sim"] = self.readout.size(), self.ledger.relation.size()
        if self.gate_d is not None:
            h["gate"] = float((torch.sigmoid(self._gate_logits(x)) * src["valid"])[a].mean())
        lp = self.scoreboard(tokens)[a]
        h["pred_distinct"] = int(lp.argmax(-1).unique().numel())
        h["pred_ent"] = float(-(lp.exp() * lp).sum(-1).mean())
        return h


def model_from_package(k):
    """Paketteki mimariye gore model: looped_relation ya da point_relation (arch yazmayan eski paket); tanimadigi
    mimaride DURUR."""
    from model_19 import PointRelation
    archs = {LoopedRelation.arch: LoopedRelation, PointRelation.arch: PointRelation}
    arch = k.get("arch", PointRelation.arch)
    if arch not in archs:
        raise ValueError("paketin mimarisi tanimsiz: %r (bilinen: %s)" % (arch, sorted(archs)))
    return archs[arch].from_package(k)
