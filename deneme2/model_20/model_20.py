# -*- coding: utf-8 -*-
"""model_20 -- adim adim kurulan model (CLAUDE.md kural 13).

Adim 1: TokenPoints + NextRelation.  Bir token girer, sonraki token'in olasiligi cikar; geriye bakmaz.
    PL = norm(PF + shift)          PF sabit (tohumdan), shift 0'dan ogrenilir; model YALNIZ PL'yi kullanir
    q  = norm(W_next . PL_i)       NextRelation: YALNIZ yon -- sonraki token'in beklenen noktasi (kurede)
    skor_j = scale . <q, PL_j>     olasilik softmax, kayip -log p(hedef) + anchor . sum|shift|^2
    scale  = ln(CONFIDENCE (n-1) / (1-CONFIDENCE))   guven; elle girilmez, nokta sayisindan

Adim 2: CausalAttention.  Her konum kendisi ve onceki konumlarin PL noktalarina bakar (nedensel, tam attention, SDPA).
    a_tj = softmax_{j<=t}( att_scale . <norm(W_query PL_t), norm(W_key PL_j)> )     att_scale = scale_for(T_MAX)
    c_t  = sum_j a_tj PL_j            getirilen sey noktalarin KENDISI: "getirilen Alice mi" dogrudan okunur
    q_t  = norm(W_next PL_t + W_context c_t)       W_context 0'dan: baslangicta model Adim 1'in aynisi
"""
import math

import torch
import torch.nn.functional as F

D = 64               # nokta boyutu; olculmedi (sozluk 74 > 64)
CONFIDENCE = 0.99    # hedef tam q yonundeyken, rakipler dikken verilebilecek olasilik -> scale
POINTS_SEED = 0
LEARN_POINTS = True  # False: PL = PF (sabit noktalar)
ANCHOR = 1e-3        # PF'den uzaklasmanin bedeli (0 = serbest); data_20 (iz 90024739fb8f) icin olculdu:
                     # kayip <= tavan + 0,01; veri degisirse yeniden olculur
ATTENTION = True     # Adim 2: attention; False = Adim 1 (yalniz son token)
T_MAX = 512          # baglam siniri (hedef); attention olcegi bundan: 512 konum arasindan 0,99 guvenle secebilsin


def scale_for(n, confidence=CONFIDENCE):
    """Hedefin skoru rakiplerden scale kadar yuksekken p(hedef) = confidence."""
    return math.log(confidence * (n - 1) / (1 - confidence))


class TokenPoints(torch.nn.Module):
    """Her token'in kuredeki noktasi.  learn=False: PL = PF (sabit noktalar)."""

    def __init__(self, n, d=D, learn=LEARN_POINTS, anchor=ANCHOR, seed=POINTS_SEED):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        # PF: n x d, rastgele, her satir norm(v) = v / |v| ile boyu 1'e indirilir; buffer = ogrenilmez, hic degismez
        self.register_buffer("fixed_points", F.normalize(torch.randn(n, d, generator=g), dim=-1))
        # shift (Δ): n x d, 0'dan; Parameter = egitimle degisir (learn=False ise dondurulur)
        self.shift = torch.nn.Parameter(torch.zeros(n, d), requires_grad=learn)
        self.learn, self.anchor = learn, anchor

    def points(self):
        # PL_i = (PF_i + Δ_i) / |PF_i + Δ_i|      her token icin
        return F.normalize(self.fixed_points + self.shift, dim=-1)

    def anchor_loss(self):
        """PF'den uzaklasmanin bedeli."""
        if not (self.learn and self.anchor):
            return self.shift.new_zeros(())
        # λ · Σ_i Σ_k Δ_ik²
        return self.anchor * (self.shift ** 2).sum()


class NextRelation(torch.nn.Module):
    """W_next (d x d), kucuk rastgele: cikti kureye indirildigi icin W_next'in boyu bir sey degistirmez, yalniz yonu.
    (0'dan baslayamaz: norm(0)'in yonu yok, gradyani 1/eps.)"""

    def __init__(self, d=D, seed=POINTS_SEED + 1):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        # W_next: d x d, degerler rastgele / √d (baslangic), sonra egitimle degisir
        self.W_next = torch.nn.Parameter(torch.randn(d, d, generator=g) / d ** 0.5)

    def forward(self, p):
        # p @ W_next.T: her satir p icin W_next · p (matris carpi vektor): (W_next·p)_a = Σ_b W_next_ab · p_b
        # q = W_next·p / |W_next·p|
        return F.normalize(p @ self.W_next.T, dim=-1)


class BigramModel(torch.nn.Module):
    def __init__(self, n, d=D, learn_points=LEARN_POINTS, anchor=ANCHOR, confidence=CONFIDENCE, seed=POINTS_SEED):
        super().__init__()
        self.tokens = TokenPoints(n, d, learn_points, anchor, seed)
        self.next = NextRelation(d, seed + 1)
        self.scale = scale_for(n, confidence)

    def logits(self, ids):
        P = self.tokens.points()
        # skor_j = scale · <q, PL_j> = scale · Σ_a q_a · PL_ja      her token j icin (74 tane)
        return self.scale * self.next(P[ids]) @ P.T

    def loss(self, inputs, targets):
        """(toplam, nll): toplam = nll + capa."""
        # cross_entropy: p_j = e^skor_j / Σ_i e^skor_i (softmax);  nll = ortalama( -log p_hedef )
        nll = F.cross_entropy(self.logits(inputs), targets)
        return nll + self.tokens.anchor_loss(), nll


class CausalAttention(torch.nn.Module):
    """Adim 2: nedensel tam attention; getirdigi sey onceki konumlarin PL noktalari.  W_context 0'dan."""

    def __init__(self, d=D, t_max=T_MAX, confidence=CONFIDENCE, seed=POINTS_SEED + 2):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        # W_query, W_key: d x d, rastgele / √d baslar, egitimle degisir.  PL'yi baska bir yone ceviren dogrusal donusum
        # (dondurur, gerer, sikistirir); ardindan norm geldigi icin yalniz yon kalir.
        self.W_query = torch.nn.Parameter(torch.randn(d, d, generator=g) / d ** 0.5)   # "ne ariyorum"
        self.W_key = torch.nn.Parameter(torch.randn(d, d, generator=g) / d ** 0.5)   # "bende ne var"
        self.W_context = torch.nn.Parameter(torch.zeros(d, d))                      # getirileni tahmine katar, 0'dan
        self.scale = scale_for(t_max, confidence)                                  # ln(0,99 · 511 / 0,01) = 10,83

    def queries_keys(self, x):
        # q_t = W_query·PL_t / |W_query·PL_t|      k_j = W_key·PL_j / |W_key·PL_j|      (ara sonuc, saklanmaz)
        return F.normalize(x @ self.W_query.T, dim=-1), F.normalize(x @ self.W_key.T, dim=-1)

    def forward(self, x):
        """x (B, T, d) PL dizisi -> c (B, T, d)."""
        q, k = self.queries_keys(x)
        # scaled_dot_product_attention (SDPA) su hesabi yapar, weights() ile ayni:
        #   s_tj = scale · <q_t, k_j>                 her konum t, her konum j icin
        #   j > t ise s_tj = -∞                       is_causal: sonrakilere bakilmaz
        #   a_tj = e^s_tj / Σ_{i<=t} e^s_ti            softmax, satir toplami 1
        #   c_t  = Σ_{j<=t} a_tj · PL_j               getirilen: noktalarin agirlikli karisimi
        return F.scaled_dot_product_attention(q, k, x, is_causal=True, scale=self.scale)

    def weights(self, x):
        """Okuma icin acik hesap: a (B, T, T), satir t yalniz j <= t."""
        q, k = self.queries_keys(x)
        s = self.scale * q @ k.transpose(-1, -2)                  # s_tj = scale · <q_t, k_j>
        T = x.shape[-2]
        s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1), float("-inf"))   # j > t: -∞
        return torch.softmax(s, -1)                               # a_tj = e^s_tj / Σ_i e^s_ti


class SequenceModel(torch.nn.Module):
    """Dizi uzerinde model: attention=False iken her konumda Adim 1 (BigramModel) ile ayni."""

    def __init__(self, n, d=D, attention=ATTENTION, learn_points=LEARN_POINTS, anchor=ANCHOR, confidence=CONFIDENCE,
                 t_max=T_MAX, seed=POINTS_SEED):
        super().__init__()
        self.tokens = TokenPoints(n, d, learn_points, anchor, seed)
        self.next = NextRelation(d, seed + 1)
        self.attention = CausalAttention(d, t_max, confidence, seed + 2) if attention else None
        self.scale = scale_for(n, confidence)

    def logits(self, ids):
        """ids (B, T) -> (B, T, n): konum t'de t+1'inci token."""
        P = self.tokens.points()                          # PL, 74 x d
        x = P[ids]                                        # her konumun token'inin noktasi
        raw = x @ self.next.W_next.T                      # raw_t = W_next · PL_t
        if self.attention is not None:
            raw = raw + self.attention(x) @ self.attention.W_context.T   # raw_t = W_next · PL_t + W_context · c_t
        # q_t = raw_t / |raw_t|;   skor_tj = scale · <q_t, PL_j>   (74 token)
        return self.scale * F.normalize(raw, dim=-1) @ P.T

    def loss(self, ids, mask):
        """mask (B, T) gercek token; hedef t+1 gercekse konum t sayilir.  (toplam, nll)."""
        logits = self.logits(ids[:, :-1])                 # konum t'nin skorlari ...
        valid = mask[:, 1:]                               # ... hedefi t+1'deki token; <pad> hedefler sayilmaz
        # cross_entropy: p = softmax(skor);  nll = ortalama( -log p(hedef) )  butun gercek konumlarda
        nll = F.cross_entropy(logits[valid], ids[:, 1:][valid])
        return nll + self.tokens.anchor_loss(), nll       # + λ · Σ Δ²


# ---- okumalar

@torch.no_grad()
def deviation(model):
    """Token basina PF ile PL arasindaki aci (derece).  2 asin(|a-b|/2): acos(cos) float32'de 1 yakininda
    gurultulu (kipirdamamis nokta 0,03 derece okunuyordu)."""
    t = model.tokens
    chord = (t.points() - t.fixed_points).norm(dim=-1)
    return torch.rad2deg(2 * torch.asin((chord / 2).clamp(max=1)))


@torch.no_grad()
def neighbors(P, k=3):
    """Her token icin kosinusu en yuksek k baska token: (indeksler, acilar)."""
    cos = P @ P.T
    cos.fill_diagonal_(-2)
    c, i = cos.topk(k, dim=-1)
    return i, torch.rad2deg(torch.acos(c.clamp(-1, 1)))


@torch.no_grad()
def transitions(model, k=5):
    """Her token icin sonraki token olasiligi en yuksek k: (indeksler, olasiliklar)."""
    n = model.tokens.fixed_points.shape[0]
    p = torch.softmax(model.logits(torch.arange(n)), -1)
    v, i = p.topk(k, dim=-1)
    return i, v
