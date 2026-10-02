# -*- coding: utf-8 -*-
"""model_y -- Model Y (CLAUDE.md kural 13: adim adim kuruldu).

TokenPoints: her token'in kuredeki noktasi.
    PL = norm(PF + shift)          PF sabit (tohumdan), shift 0'dan ogrenilir; kayba anchor . sum|shift|^2 eklenir
    scale  = ln(CONFIDENCE (n-1) / (1-CONFIDENCE))   guven; elle girilmez, nokta sayisindan
CausalAttention: nedensel tam attention, skor att_scale . <norm(W_query x_t), norm(W_key x_j)>, att_scale = scale_for(T_MAX).

BlockModel.  Her konumun bir durumu var (hidden, h); TURNS tur boyunca Block uygulanir.
    h = PL (INPUT_EMBEDDING'de norm(E[token]))           durum konumun noktasiyla baslar
    her turda (Block):
      x_t = h_t + sum_{k=0..3} w_k h_(t-k)              Canon-A: attention'in girdisi (w 0'dan)
      c_t = sum_j a_tj x_j                              attention DURUMLARA bakar, head basina W_value dilimi
      h_t = norm(h_t + a_A (norm(W_context c_t) - h_t))                 normalized update: alpha kadar don
      h_t = norm(h_t + a_F (norm(W_fact_out (SiLU(W_fact_in x) * (W_fact_up x))) - h_t)),  x = sqrt(d) h_t   FactUnits
    girdisi durum olan matrislerin satirlari, duruma yazanlarin sutunlari kurede (her optimizer adimindan sonra)
    cikis: skor = e^tau phi(<h, PL>), tau = log_output_scale ln(scale)'dan ogrenilir; phi (OUTPUT_LINK) yoksa dogrusal
    LOSS_CHUNK: egitim kaybi sozluk parcalariyla (tam logits tablosu yok, gradyan ileri hesapta); sinav logits'le
    LAYERS: turlar LAYERS farkli Block'u sirayla kullanir (2 katman x 2 tur: A B A B)

ROPE=True: attention'da q ve k konuma gore dondurulur (RoPE); iki konumun skoru aradaki mesafeye de bagli olur.  Ogrenilen
sayi yok; donme boyu degistirmez.

Paketli pencere (FineWeb, maskeli paketleme; Llama 3 yolu): document_positions (B, T) = token'in kendi belgesindeki konumu,
belge basinda 0.  Attention (GPU'da flex_attention) ve Canon belge sinirini gecmez, RoPE konumu belge basinda 0'dan.
Verilmezse SDPA is_causal.
"""
import functools
import math

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

# GPU'da derlenmis flex: derlenmemisi yavas ve T x T gecici bellek acar.  CPU'da (testler, float64) derlenmemisi: derlenmis
# flex CPU'da float64 kabul etmiyor.  dynamic=False: sinav pencereleri sabit boyda
_flex_compiled = torch.compile(flex_attention, dynamic=False)

D = 64               # nokta boyutu
CONFIDENCE = 0.99    # hedef tam q yonundeyken, rakipler dikken verilebilecek olasilik -> scale
POINTS_SEED = 0
ANCHOR = 1e-3        # PF'den uzaklasmanin bedeli (0 = serbest)
T_MAX = 512          # baglam siniri (hedef); attention olcegi bundan: 512 konum arasindan 0,99 guvenle secebilsin
TURNS = 4            # blok tur sayisi
LAYERS = 2           # farkli Block (katman) sayisi, turlar sirayla doner: tur i -> Block i mod LAYERS (A B A B)
SHARED_FACTS = False # True: FactUnits de turlar arasinda paylasilir.  False: her turun kendi FactUnits'i, attention paylasimli
FACT_UNITS = 170     # FactUnits (SwiGLU) birim sayisi: SwiGLU'nun yerlesik genisligi 8/3 x D
ALPHA_INIT = 0.1            # normalized update h <- norm(h + alpha (norm(u) - h)): alpha tur ve alt blok basina d sayi
                            # (nGPT), baslangici nGPT 2026 tarifi (derinlikten bagimsiz 0,1)
LAST_FACTS_ALPHA_INIT = 0.1  # yalniz son turun FactUnits alpha'si (alpha_facts[turns - 1]) bundan baslar, gerisi ALPHA_INIT
INPUT_EMBEDDING = True   # True: girdi ayri, ogrenilen tablo (V x d, PF'den baslar, capa yok; nGPT'deki E_input) -- cikis
                         # PL'de kalir.  False: girdi = cikis = PL
INPUT_EMBEDDING_SPHERE = True  # INPUT_EMBEDDING'de girdi tablosunun satirlari basta ve her optimizer adimindan sonra
                               # birim boya (nGPT); serbest tabloda gradyan satira dik, boy buyur, etkin lr duser
FIRST_TURN_FACTS = False  # False: tur 1'in FactUnits alt adimi yok
HEADS = 4            # attention head sayisi (> 1): d H'ye bolunur, W_value (d x d, birim baslar) her head'in tasiyacagini
                     # secer
ROPE = True          # attention'in q ve k'sina RoPE (konum bilgisi)
LOSS_CHUNK = 4096    # egitim kaybi sozluk parcalariyla: tam logits tablosu (B x T x V) olusmaz, gradyan ileri hesapta
                     # biriktirilir.  0 = tek parca.  Sinav ve uretim logits'le
OUTPUT_LINK = True   # cikis bagi phi: skor = s phi(c), phi(c) = c (1 + c (q + c (q^2/3 + u))), c = <h, PL>, q = link_q,
                     # u = link_u >= 0 (her adimdan sonra kirpilir) -> phi' = (1 + q c)^2 + 3 u c^2 >= 0: phi hep artan, en
                     # olasi token degismez; q = u = 0: dogrusal.  False: dogrusal skor
ATTENTION_LOG_SCALE = True   # True: attention olcegi sabit scale_for(T_MAX) yerine sorgu basina scale_for(n), n = gordugu anahtar
                             # sayisi (belge ici konum + 1); T_MAX = 512'de n = 512 konumu sabit olcek (SSMax'in s log n'i)
ROPE_BASE = "auto"   # RoPE tabani: sayi ya da "auto" = rope_base_for(head boyu, t_max); eski config'ler sabit 10.000


@functools.lru_cache(maxsize=None)
def rope_base_for(head_dim, context):
    """RoPE tabani, Men 2024 (arXiv 2405.14591, Eq. 15-17): B(m) = sum_{i < d/2} cos(m taban^(-2i/d)) >= 0 her m <= 2 x
    context icin (benzer token rastgeleden fazla dikkat alabilsin; 2 kat pay: sinir tirtikli, uretim baglami asabilir).
    Adaylar 1, 2, 5 x 10^k, en az 10.000; hicbiri tutmazsa (kucuk head, test modelleri) en uzun mesafeyi tasiyan."""
    m = torch.arange(2 * context + 1, dtype=torch.float64)
    best = None
    for base in (c * 10.0 ** k for k in range(4, 13) for c in (1, 2, 5)):
        theta = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
        negative = (torch.cos(m[:, None] * theta).sum(1) < 0).nonzero()
        reach = int(negative[0]) - 1 if len(negative) else 2 * context        # B(m) >= 0 kalan en uzun mesafe
        if reach >= 2 * context:
            return base
        best = max(best or (reach, base), (reach, base), key=lambda r: r[0])
    return best[1]


def apply_rope(x, positions=None, base=10000.0):
    """RoPE, x (..., T, d): konum t'de her boyut cifti t · base^(-2i/d) acisiyla dondurulur; iki konumun
    <q, k> skoru yalniz aradaki mesafeye bagli kalir.  Parametresi yok.  positions (B, T): satir basina konum
    (onbellekli uretim); None = 0..T-1."""
    T, dh = x.shape[-2], x.shape[-1]
    # hesap en az fp32: bf16'da konum 256'dan sonra tam sayi degil, aci kayar.  fp32 / fp64'te sonuc aynen
    work = torch.promote_types(x.dtype, torch.float32)
    freq = float(base) ** (-torch.arange(0, dh, 2, device=x.device, dtype=work) / dh)      # (d/2,)
    positions = torch.arange(T, device=x.device, dtype=work) if positions is None else positions.to(work)
    angle = positions[..., None] * freq                                                     # (..., T, d/2)
    cos, sin = angle.cos(), angle.sin()
    x1, x2 = x[..., 0::2].to(work), x[..., 1::2].to(work)
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2).to(x.dtype)


def same_document_causal(first):
    """mask_mod (flex_attention imzasi): sorgu q, anahtar kv'ye bakar <=> kv <= q ve kv ayni belgede (kv >= first[b, q]);
    first (B, T) = t - konum_t, t'nin belgesinin penceredeki ilk konumu."""
    def mask_mod(b, h, q, kv):
        return (kv <= q) & (kv >= first[b, q])
    return mask_mod


def build_document_mask(document_positions, flex=False):
    """Paketli pencere: document_positions (B, T) her token'in kendi belgesindeki konumu (belge basinda 0) -> belge ici
    nedensel maske.  flex: flex_attention'in blok-seyrek BlockMask'i; degilse AYNI mask_mod'dan yogun (B, T, T) bool."""
    B, T = document_positions.shape
    t = torch.arange(T, device=document_positions.device)
    mask_mod = same_document_causal(t - document_positions)
    if flex:
        return create_block_mask(mask_mod, B, None, T, T, device=document_positions.device)
    return mask_mod(torch.arange(B, device=t.device)[:, None, None], None, t[None, :, None], t[None, None, :])


def masked_nll(logits, targets, valid):
    """Ortalama -log p(hedef), yalniz valid konumlarda.  logits[valid] ile ayni hesap ama sekil sabit (boole indeks yok):
    torch.compile tek grafik kurar, adimda GPU->CPU senkronu yok.  Gradyan ayni; deger son bitte farkli olabilir."""
    per_token = F.cross_entropy(logits.flatten(0, -2), targets.flatten(), reduction="none")
    weight = valid.flatten().to(per_token.dtype)
    return (per_token * weight).sum() / weight.sum()


def scale_for(n, confidence=CONFIDENCE):
    """Hedefin skoru rakiplerden scale kadar yuksekken p(hedef) = confidence."""
    return math.log(confidence * (n - 1) / (1 - confidence))


class TokenPoints(torch.nn.Module):
    """Her token'in kuredeki noktasi."""

    def __init__(self, n, d=D, anchor=ANCHOR, seed=POINTS_SEED):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        # PF: n x d, rastgele, her satir norm(v) = v / |v| ile boyu 1'e indirilir; buffer = ogrenilmez, hic degismez
        self.register_buffer("fixed_points", F.normalize(torch.randn(n, d, generator=g), dim=-1))
        # shift (Δ): n x d, 0'dan; Parameter = egitimle degisir
        self.shift = torch.nn.Parameter(torch.zeros(n, d))
        self.anchor = anchor

    def points(self):
        # PL_i = (PF_i + Δ_i) / |PF_i + Δ_i|      her token icin
        return F.normalize(self.fixed_points + self.shift, dim=-1)

    def anchor_loss(self):
        """PF'den uzaklasmanin bedeli."""
        if not self.anchor:
            return self.shift.new_zeros(())
        # λ · Σ_i Σ_k Δ_ik²
        return self.anchor * (self.shift ** 2).sum()


class CausalAttention(torch.nn.Module):
    """Nedensel tam attention, head basina: q, k birim (+ RoPE), getirilen W_value x'in head dilimi, head'ler yan yana.
    W_context 0'dan (BlockModel rastgele baslatir)."""

    def __init__(self, d=D, t_max=T_MAX, confidence=CONFIDENCE, seed=POINTS_SEED + 2, rope=False, heads=HEADS,
                 attention_log_scale=False, rope_base=ROPE_BASE):
        super().__init__()
        assert heads > 1 and d % heads == 0, "head sayisi > 1 olmali ve d'yi bolmeli"
        assert not rope or (d // heads) % 2 == 0, "RoPE icin head boyu cift olmali"
        g = torch.Generator().manual_seed(seed)
        self.rope, self.heads = rope, heads
        self.rope_base = rope_base_for(d // heads, t_max) if rope_base == "auto" else float(rope_base)
        self.attention_log_scale, self.confidence = attention_log_scale, confidence
        # W_query, W_key: d x d, rastgele / √d baslar, egitimle degisir.  Girdiyi baska bir yone ceviren dogrusal donusum
        # (dondurur, gerer, sikistirir); ardindan norm geldigi icin yalniz yon kalir.
        self.W_query = torch.nn.Parameter(torch.randn(d, d, generator=g) / d ** 0.5)   # "ne ariyorum"
        self.W_key = torch.nn.Parameter(torch.randn(d, d, generator=g) / d ** 0.5)   # "bende ne var"
        self.W_context = torch.nn.Parameter(torch.zeros(d, d))                      # getirileni duruma yazar
        self.scale = scale_for(t_max, confidence)                                  # ln(0,99 · 511 / 0,01) = 10,83
        self.W_value = torch.nn.Parameter(torch.eye(d))   # birim baslar: ilk adimda head h durumun h. dilimini tasir

    def queries_keys(self, x, positions=None):
        # q_t = W_query·x_t / |W_query·x_t|      k_j = W_key·x_j / |W_key·x_j|      (ara sonuc, saklanmaz)
        # (.., T, d) -> (.., H, T, d/H): her head kendi dilimini normlar
        q, k = (z.unflatten(-1, (self.heads, -1)).transpose(-3, -2) for z in (x @ self.W_query.T, x @ self.W_key.T))
        positions = None if positions is None else positions[:, None]   # (B, 1, T): head ekseniyle hizali
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        if self.rope:                                     # konuma gore dondur; boy 1 kalir
            q, k = apply_rope(q, positions, self.rope_base), apply_rope(k, positions, self.rope_base)
        if self.attention_log_scale:                      # q x scale_for(n) / scale: cagiranin scale'iyle skor scale_for(n) <q, k>
            q = q * self.query_scale(q.shape[-2], positions, q.device).to(q.dtype)
        return q, k

    def query_scale(self, T, positions=None, device=None):
        """ATTENTION_LOG_SCALE: sorgu basina scale_for(n) / scale, n = sorgunun gordugu anahtar sayisi = belge ici konum + 1
        (positions None: t + 1).  n >= 2: n = 1'de softmax tek anahtarda 1, olcek sonucu degistirmez (scale_for(1) = -inf).
        -> (..., T, 1), float64."""
        n = (torch.arange(T, device=device) if positions is None else positions).to(torch.float64) + 1
        s = torch.log(self.confidence * (n.clamp(min=2) - 1) / (1 - self.confidence))
        return (s / self.scale)[..., None]

    def forward(self, x, cache=None, document_positions=None, document_mask=None):
        """x (B, T, d) durum dizisi -> c (B, T, d) = [a_1 (V_1 x) ; ... ; a_H (V_H x)].  cache (AttentionCache): onbellekli
        uretim; x yalniz yeni konumlar.  document_positions (B, T): paketli pencere -- RoPE konumu belge basinda 0, attention
        belge icinde; document_mask: build_document_mask'in ciktisi (BlockMask: flex_attention, bool tensor: SDPA), None
        ise yogun.  SDPA: s_tj = scale <q_t, k_j>, j > t'de -inf, a = softmax(s), c_t = sum_j a_tj v_j (weights() ile ayni)."""
        if cache is not None:
            assert document_positions is None, "onbellekli uretim tek belgeli"
            return cache.attend(self, x)
        v = (x @ self.W_value.T).unflatten(-1, (self.heads, -1)).transpose(-3, -2)   # head dilimleri (.., H, T, d/H)
        if document_positions is not None:
            mask = build_document_mask(document_positions) if document_mask is None else document_mask
            q, k = self.queries_keys(x, document_positions)
            if torch.is_tensor(mask):
                c = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None], scale=self.scale)
            elif q.is_cuda:                               # autocast flex'i kapsamaz: tipler elle.  Egitimde (autocast) q, k fp32
                # geliyor, v bf16: q, k v'nin tipine.  fp32 (sinav, autocast yok): kucuk blok, varsayilan blok A100'un
                # paylasimli bellegini asiyor (head 96)
                if v.dtype != q.dtype:
                    q, k = q.to(v.dtype), k.to(v.dtype)
                    options = dict(BLOCK_M1=32, BLOCK_N1=64, BLOCK_M2=64, BLOCK_N2=32)   # geri yayilim bloklari
                else:
                    options = dict(BLOCK_M=32, BLOCK_N=32, num_stages=1) if q.dtype == torch.float32 else None
                c = _flex_compiled(q, k.to(q.dtype), v.to(q.dtype), block_mask=mask, scale=self.scale,
                                   kernel_options=options)
            else:
                c = flex_attention(q, k.to(q.dtype), v.to(q.dtype), block_mask=mask, scale=self.scale)
            return c.transpose(-3, -2).flatten(-2)
        q, k = self.queries_keys(x)
        c = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        return c.transpose(-3, -2).flatten(-2)

    def weights(self, x):
        """Okuma icin acik hesap: a (B, H, T, T), satir t yalniz j <= t.  x attention'in GERCEK girdisi olmali: Block'ta
        Canon karisimi (Block.forward'daki x), h degil."""
        q, k = self.queries_keys(x)
        s = self.scale * q @ k.transpose(-1, -2)                  # s_tj = scale · <q_t, k_j>
        T = x.shape[-2]
        s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1), float("-inf"))   # j > t: -∞
        return torch.softmax(s, -1)                               # a_tj = e^s_tj / Σ_i e^s_ti


class AttentionCache:
    """Onbellekli uretim icin BIR turun attention bellegi: yazilmis konumlarin key'leri (RoPE'li) ve value'lari.
    Attention nedensel: onceki konumlarin durumu yeni token'la degismez, yeni konum yalniz kendi q, k, v'sini hesaplar.
    Ilk cagri istem (B, L) -- normal ileri hesabin aynisi; sonraki her cagri satir basina TEK yeni token.
    lengths (B,): istem uzunluklari (sagdan dolgulu; satir r'nin ilk yeni token'i konum lengths[r]'ye yazilir, dolgunun
    key'leri o konuma gelinceye kadar maskeli kalir).  capacity: istem + uretilecek token.
    Canon: turun Canon oncesi girdilerinden yalniz son 3 konum saklanir (canon_cache; 3 yuvali halka, yuva = konum mod 3);
    yeni konum onceki 3 konumu buradan okur.  Key ve value head basina."""

    def __init__(self, lengths, capacity):
        self.next_position, self.capacity = lengths.clone(), capacity   # satir basina siradaki konum
        self.keys = self.values = self.canon_inputs = None
        self.span = 0                                            # yazilmis en uzun satirin boyu (Python sayisi: senkron yok)

    def canon_cache(self, x):
        """x (B, t, d): yeni konumlarin Canon oncesi girdisi -> (B, 3 + t, d), basta ayni turun onceki 3 konumu
        (istemden once 0).  x'i saklar; attend'den ONCE cagrilir (next_position henuz ilerlemedi)."""
        B, t, d = x.shape
        pos = self.next_position[:, None]                       # (B, 1): istemde istemin boyu, sonra yeni konum
        back = pos - torch.arange(3, 0, -1, device=x.device)    # (B, 3): t-3, t-2, t-1; < 0 ise istemden once (0)
        ring = (back % 3)[..., None].expand(-1, -1, d)          # halka yuvasi = konum mod 3
        if self.canon_inputs is None:                           # istem: oncesi yok, tam hesaptaki dolgu; son 3 konum halkaya
            self.canon_inputs = x.new_zeros(B, 3, d)
            self.canon_inputs.scatter_(1, ring, x.gather(1, back.clamp(min=0)[..., None].expand(-1, -1, d))
                                       * (back >= 0)[..., None])
            return F.pad(x, (0, 0, 3, 0))
        assert t == 1, "istemden sonra satir basina tek token"
        before = self.canon_inputs.gather(1, ring) * (back >= 0)[..., None]
        self.canon_inputs.scatter_(1, (pos % 3)[..., None].expand(-1, -1, d), x)   # t'nin yuvasi t-3'unku: once okundu
        return torch.cat([before, x], 1)

    def attend(self, at, x):
        """Key ve value (B, H, S, d/H); value W_value x'in head dilimi."""
        B, t = x.shape[:2]
        H = at.heads
        v = (x @ at.W_value.T).unflatten(-1, (H, -1)).transpose(1, 2)
        if self.keys is None:                                    # istem: CausalAttention.forward ile ayni hesap
            q, k = at.queries_keys(x)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=at.scale)
            self.keys = k.new_zeros(*k.shape[:-2], self.capacity, k.shape[-1])
            self.values = v.new_zeros(*v.shape[:-2], self.capacity, v.shape[-1])
            self.keys[..., :t, :], self.values[..., :t, :] = k, v
            self.span = t
        else:
            assert t == 1, "istemden sonra satir basina tek token"
            pos = self.next_position[:, None]                    # (B, 1)
            q, k = at.queries_keys(x, pos)
            at_pos = pos[:, None, :, None]                       # yazilacak konum: (B, 1, 1, 1)
            self.keys.scatter_(-2, at_pos.expand(*k.shape), k)
            self.values.scatter_(-2, at_pos.expand(*v.shape), v)
            self.next_position = self.next_position + 1          # yeni tensor: pos (gorunum) degismesin
            self.span += 1                                       # butun satirlar birer ilerler
            S = self.span
            allowed = (torch.arange(S, device=x.device)[None, None, :] <= pos[..., None])[:, None]   # (B, 1, 1, S): j <= konum
            out = F.scaled_dot_product_attention(q, self.keys[..., :S, :], self.values[..., :S, :], attn_mask=allowed,
                                                 scale=at.scale)
        return out.transpose(1, 2).flatten(-2)                   # head'ler yan yana


class FactUnits(torch.nn.Module):
    """Durum donusturme: her konumda ayri.  u = SiLU(W_fact_in · h) ⊙ (W_fact_up · h): kapi (W_fact_in) icerigi (W_fact_up)
    acar.  Cikti W_fact_out · u.  W_fact_out 0'dan (BlockModel rastgele baslatir)."""

    def __init__(self, d=D, units=FACT_UNITS, seed=POINTS_SEED + 3):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W_fact_in = torch.nn.Parameter(torch.randn(units, d, generator=g) / d ** 0.5)   # durumdan birimlere (kapi)
        self.W_fact_up = torch.nn.Parameter(torch.randn(units, d, generator=g) / d ** 0.5)   # tasinan icerik
        self.W_fact_out = torch.nn.Parameter(torch.zeros(d, units))                          # birimlerden duruma

    def forward(self, h):
        # u_i = SiLU(W_fact_in[i] · h) · (W_fact_up[i] · h);   cikti_a = Σ_i W_fact_out[a,i] · u_i
        return (F.silu(h @ self.W_fact_in.T) * (h @ self.W_fact_up.T)) @ self.W_fact_out.T


class Block(torch.nn.Module):
    """Bir tur: Canon, attention durumlara bakar ve getirdigini duruma yazar; FactUnits durumu donusturur.  Her alt adim
    normalized update: h <- norm(h + alpha (norm(u) - h))."""

    def __init__(self, d=D, t_max=T_MAX, confidence=CONFIDENCE, units=FACT_UNITS, seed=POINTS_SEED + 2, rope=False,
                 heads=HEADS, attention_log_scale=False, rope_base=ROPE_BASE):
        super().__init__()
        self.attention = CausalAttention(d, t_max, confidence, seed, rope=rope, heads=heads,
                                         attention_log_scale=attention_log_scale, rope_base=rope_base)
        self.facts = FactUnits(d, units, seed + 1)
        self.canon = True                                 # analysis betikleri okur (Canon artik hep acik)
        # w_k: k konum oncesinden, boyut basina; 0'dan (adim 0 = Canon'suz)
        self.canon_weights = torch.nn.Parameter(torch.zeros(4, d))

    def forward(self, h, cache=None, alpha_attention=None, alpha_facts=None, turn_facts=None, skip_facts=False,
                document_positions=None, document_mask=None):
        """alpha_attention, alpha_facts (d,): bu turun alpha'lari.  turn_facts: bu turun FactUnits'i (SHARED_FACTS=False'ta
        ikinci gelis); None: Block'un kendi FactUnits'i.  skip_facts: FactUnits alt adimi yok (FIRST_TURN_FACTS=False'ta
        tur 1).  document_positions (B, T), document_mask: paketli pencere (Canon ve attention belge icinde;
        CausalAttention.forward)."""
        at = self.attention
        unit = lambda v: F.normalize(v, dim=-1)
        T = h.shape[-2]                                      # Canon-A: x_t = h_t + sum_k w_k * h_(t-k), baslangictan once 0
        full = F.pad(h, (0, 0, 3, 0)) if cache is None else cache.canon_cache(h)   # onceki 3 konum + h
        if document_positions is None:
            x = h + sum(self.canon_weights[k] * full[..., 3 - k:3 - k + T, :] for k in range(4))
        else:                                                # h_(t-k) yalniz ayni belgede (k <= konum_t), degilse 0
            x = h + sum(self.canon_weights[k] * full[..., 3 - k:3 - k + T, :]
                        * (document_positions >= k)[..., None].to(h.dtype) for k in range(4))
        added = at(x, cache=cache, document_positions=document_positions, document_mask=document_mask) @ at.W_context.T
        h = unit(h + alpha_attention * (unit(added) - h))                         # h = norm(h + α_A ⊙ (norm(W_context c) - h))
        if skip_facts:
            return h
        # W_fact_in satirlari birim, h birim -> girdi kosinus (tipik ±1/√d); √d ile O(1) olcekte
        units = self.facts if turn_facts is None else turn_facts
        return unit(h + alpha_facts * (unit(units(h * h.shape[-1] ** 0.5)) - h))   # h = norm(h + α_F ⊙ (norm(olgu) - h))


class BlockModel(torch.nn.Module):
    """Durum (hidden) PL ile baslar, TURNS tur Block; cikis son durumun kendisi (ayri cikis matrisi yok)."""

    def __init__(self, n, d=D, turns=TURNS, layers=LAYERS, anchor=ANCHOR, confidence=CONFIDENCE, t_max=T_MAX, units=FACT_UNITS,
                 seed=POINTS_SEED, rope=ROPE, heads=HEADS, loss_chunk=LOSS_CHUNK,
                 last_facts_alpha_init=LAST_FACTS_ALPHA_INIT, output_link=OUTPUT_LINK, shared_facts=SHARED_FACTS,
                 input_embedding=INPUT_EMBEDDING, first_turn_facts=FIRST_TURN_FACTS, input_embedding_sphere=INPUT_EMBEDDING_SPHERE,
                 attention_log_scale=ATTENTION_LOG_SCALE, rope_base=ROPE_BASE,
                 stream_norm=True, normalized_update=True, sphere_weights=True, canon=True, fact_activation="swiglu"):
        """stream_norm, normalized_update, sphere_weights, canon, fact_activation: kaldirilan seceneklerin tek degeri
        (analysis cagrilari)."""
        super().__init__()
        assert stream_norm and normalized_update and sphere_weights and canon and fact_activation == "swiglu", \
            "kaldirilan secenek (normsuz akis, normalized_update / sphere_weights / canon kapali, relu / reglu)"
        self.tokens = TokenPoints(n, d, anchor, 100 * seed)   # 100 * seed: tohumlar rastgele sayi paylasmaz
        assert turns % layers == 0, "turns (%d) layers'in (%d) kati olmali: her Block esit sayida tur" % (turns, layers)
        self.blocks = torch.nn.ModuleList(Block(d, t_max, confidence, units, 100 * seed + 10 + 2 * i, rope=rope, heads=heads,
                                                attention_log_scale=attention_log_scale, rope_base=rope_base)
                                          for i in range(layers))
        self.turns, self.layers = turns, layers
        # analysis betikleri okur (kaldirilan secenekler, tek degerleri)
        self.stream_norm, self.layer_norm, self.learn_output_scale = True, False, True
        self.normalized_update, self.sphere_weights, self.canon = True, True, True
        self.shared_facts = shared_facts or turns == layers
        if not self.shared_facts:                         # ikinci ve sonraki gelisler: tur i >= layers -> extra_facts[i - layers]
            self.extra_facts = torch.nn.ModuleList(
                FactUnits(d, units, 100 * seed + 10 + 2 * i + 1) for i in range(layers, turns))
        self.rope, self.heads = rope, heads
        g = torch.Generator().manual_seed(100 * seed + 9)   # sifir yon normalize edilemez: W_context, W_fact_out rastgele baslar
        with torch.no_grad():
            for b in self.blocks:
                b.attention.W_context.copy_(torch.randn(d, d, generator=g) / d ** 0.5)
                b.facts.W_fact_out.copy_(torch.randn(d, units, generator=g) / units ** 0.5)
            for f in getattr(self, "extra_facts", ()):
                f.W_fact_out.copy_(torch.randn(d, units, generator=g) / units ** 0.5)
        # alpha: tur basina (paylasilan blokta da), (tur, d); son turun FactUnits'i LAST_FACTS_ALPHA_INIT'ten
        self.alpha_attention = torch.nn.Parameter(torch.full((turns, d), ALPHA_INIT))
        self.alpha_facts = torch.nn.Parameter(torch.full((turns, d), ALPHA_INIT))
        with torch.no_grad():
            self.alpha_facts[-1] = last_facts_alpha_init
        self.first_turn_facts = first_turn_facts
        if not first_turn_facts and not (self.shared_facts and turns > self.layers):
            self.blocks[0].facts = None                   # tur 1'in takimini baska tur kullanmiyor: parametresi de yok
        if input_embedding:                               # PF'den: ilk adimda girdi PL ile ayni
            self.input_embedding = torch.nn.Parameter(self.tokens.fixed_points.detach().clone())
        self.input_embedding_sphere = bool(input_embedding and input_embedding_sphere)
        self.normalize_weights()
        self.scale = scale_for(n, confidence)
        # cikis olcegi e^tau, tau = ln(scale)'dan ogrenilir: baslangicta sabit scale ile bit duzeyinde ayni
        self.log_output_scale = torch.nn.Parameter(torch.tensor(math.log(self.scale)))
        self.output_link = output_link
        if self.output_link:                              # q = u = 0: dogrusal skor
            self.link_q = torch.nn.Parameter(torch.zeros(()))
            self.link_u = torch.nn.Parameter(torch.zeros(()))
        assert loss_chunk >= 0, "loss_chunk: 0 (tek parca) ya da parca boyu"
        self.loss_chunk = loss_chunk

    def turn_blocks(self):
        return [self.blocks[i % len(self.blocks)] for i in range(self.turns)]     # A B A B

    def input_states(self, ids):
        """h0 (B, T, d): PL[ids]; INPUT_EMBEDDING'de norm(E[ids])."""
        E = getattr(self, "input_embedding", None)
        return self.tokens.points()[ids] if E is None else F.normalize(E[ids], dim=-1)

    def hidden(self, ids, caches=None, document_positions=None):
        """Her turdan sonraki durumlar: [h0, h1, ..., h_TURNS], her biri (B, T, d); h0 = input_states.  caches: tur basina
        bir AttentionCache (onbellekli uretim; ids yalniz yeni token'lar).  document_positions (B, T): paketli pencere,
        token'in kendi belgesindeki konumu (belge basinda 0); None = tek belge (bugunku hesap)."""
        assert document_positions is None or caches is None, "paketli pencere: onbellekli uretim desteklenmiyor"
        h = self.input_states(ids)
        out = [h]
        if document_positions is not None:                # maske bir kez, butun turlar icin
            mask = self.document_mask(document_positions)
        for i, block in enumerate(self.turn_blocks()):
            alphas = dict(alpha_attention=self.alpha_attention[i], alpha_facts=self.alpha_facts[i])
            if not self.shared_facts and i >= self.layers:
                alphas["turn_facts"] = self.extra_facts[i - self.layers]
            if i == 0 and not self.first_turn_facts:
                alphas["skip_facts"] = True
            if document_positions is not None:
                alphas.update(document_positions=document_positions, document_mask=mask)
            h = block(h, None if caches is None else caches[i], **alphas)
            out.append(h)
        return out

    def document_mask(self, document_positions):
        """GPU'da flex_attention'in BlockMask'i (yogun T x T maske 8.192'de sigmaz); CPU'da (flex'in geri yayilimi yok) ayni
        mask_mod'dan yogun bool maske."""
        return build_document_mask(document_positions, flex=document_positions.is_cuda)

    @torch.no_grad()
    def normalize_weights(self):
        """Kure agirliklari (nGPT): girdisi durum olan matrislerin satirlari (W_query, W_key, W_value, W_fact_in, W_fact_up),
        duruma yazanlarin sutunlari (W_context, W_fact_out) ve input_embedding_sphere'de girdi tablosunun satirlari birim boya;
        baslangicta ve her optimizer adimindan sonra (train_seq)."""
        if getattr(self, "input_embedding_sphere", False):
            self.input_embedding.copy_(F.normalize(self.input_embedding, dim=1))
        for b in self.blocks:
            at, f = b.attention, b.facts                  # f None: FIRST_TURN_FACTS=False'ta tur 1'in takimi yok
            for w in (at.W_query, at.W_key, at.W_value) + ((f.W_fact_in, f.W_fact_up) if f is not None else ()):
                w.copy_(F.normalize(w, dim=1))
            for w in (at.W_context,) + ((f.W_fact_out,) if f is not None else ()):
                w.copy_(F.normalize(w, dim=0))
        for f in getattr(self, "extra_facts", ()):
            for w in (f.W_fact_in, f.W_fact_up):
                w.copy_(F.normalize(w, dim=1))
            f.W_fact_out.copy_(F.normalize(f.W_fact_out, dim=0))

    def logits(self, ids, caches=None, document_positions=None):
        P = self.tokens.points()
        h = self.hidden(ids, caches, document_positions)[-1]   # son durum, kurede
        # e^tau = scale · e^(tau - ln scale): baslangicta us tam 0
        scale = self.scale * torch.exp(self.log_output_scale - math.log(self.scale))
        if self.output_link:                                   # skor_tj = scale · phi(<h_t, PL_j>)
            c = h @ P.T
            q, u = self.link_q, self.link_u
            return scale * (c * (1 + c * (q + c * (q * q / 3 + u))))
        return scale * h @ P.T                                 # skor_tj = scale · <h_t, PL_j>

    def _link_parts(self, c):
        """phi(c) = c + q c^2 + (q^2/3 + u) c^3 ve turevleri: phi'(c) = (1 + q c)^2 + 3 u c^2, dphi/dq = c^2 + 2q/3 c^3,
        dphi/du = c^3 (parcali kayip icin; q, u sabit).  q, u tensor kalir: float() compile'da grafigi kirar (274 ms/adim)."""
        q, u = self.link_q.detach().to(c.dtype), self.link_u.detach().to(c.dtype)
        c2 = c * c
        c3 = c2 * c
        return c + q * c2 + (q * q / 3 + u) * c3, (1 + q * c) ** 2 + 3 * u * c2, c2 + (2 * q / 3) * c3, c3

    def loss(self, ids, mask, document_positions=None):
        """document_positions (B, T), ids ile hizali: paketli pencere (hidden); hedef t+1, konum t'nin belgesinden
        tahmin edilir."""
        inner = None if document_positions is None else document_positions[:, :-1]
        if self.loss_chunk:                                    # parcali: (N x V) tablosu yok, gradyan ileri hesapta
            P = self.tokens.points()
            h = self.hidden(ids[:, :-1], None, inner)[-1].flatten(0, -2)   # (N, d), N = B (T - 1); cikis logits'teki gibi
            scale = self.scale * torch.exp(self.log_output_scale - math.log(self.scale))
            # z_ij = s <h_i, P_j>; once butun parcalardan lse_i, sonra p_ij = e^(z_ij - lse_i) ile gradyanlar:
            #   dL/dh_i = s w_i (sum_j p_ij P_j - P_y)   dL/dP_j = s sum_i w_i (p_ij - [y_i = j]) h_i
            #   dL/ds = sum_i w_i (sum_j p_ij <h_i, P_j> - <h_i, P_y>)        w_i = maske / gecerli hedef sayisi
            with torch.no_grad():
                work = torch.promote_types(h.dtype, torch.float32)   # lse ve birikimler en az fp32 (autocast'te carpim bf16)
                hd, Pd = h.detach(), P.detach()
                s = scale.detach()
                y = ids[:, 1:].flatten()
                w = mask[:, 1:].flatten().to(work)
                w = w / w.sum()                                # masked_nll gibi: dolgu dahil sabit sekil, agirlik 0
                chunks = range(0, Pd.shape[0], self.loss_chunk)
                lse = None
                for c in chunks:
                    z = (hd @ Pd[c:c + self.loss_chunk].T).to(work)
                    z = (self._link_parts(z)[0] if self.output_link else z).mul_(s)
                    part = torch.logsumexp(z, -1)
                    lse = part if lse is None else torch.logaddexp(lse, part)
                    del z
                hw = hd.to(work) * w[:, None]
                grad_h = torch.zeros_like(hd, dtype=work)      # sum_j p_ij P_j
                grad_P = torch.zeros_like(Pd, dtype=work)      # sum_i w_i p_ij h_i
                mean_dot = torch.zeros_like(lse)               # sum_j p_ij <h_i, P_j>  (phi'de sum_j p_ij phi_ij)
                if self.output_link:                           # phi: z = s phi(c); dz/dc = s phi'(c), dphi/dq, dphi/du
                    mean_q, mean_u = torch.zeros_like(lse), torch.zeros_like(lse)
                for c in chunks:
                    Pc = Pd[c:c + self.loss_chunk].to(work)
                    dot = (hd @ Pc.T).to(work)
                    if self.output_link:
                        ph, dph, dq, du = self._link_parts(dot)
                        p = (ph * s).sub_(lse[:, None]).exp_()
                        g = p * dph
                        grad_h += g @ Pc
                        grad_P[c:c + self.loss_chunk] = g.T @ hw
                        mean_dot += (p * ph).sum(-1)
                        mean_q += (p * dq).sum(-1)
                        mean_u += (p * du).sum(-1)
                        del dot, p, g, ph, dph, dq, du
                        continue
                    p = (dot * s).sub_(lse[:, None]).exp_()
                    grad_h += p @ Pc
                    grad_P[c:c + self.loss_chunk] = p.T @ hw
                    mean_dot += p.mul_(dot).sum(-1)
                    del dot, p
                dot_y = (hd * Pd[y]).sum(-1).to(work)
                if self.output_link:                           # hedefte: phi, phi', dphi/dq, dphi/du
                    ph_y, dph_y, dq_y, du_y = self._link_parts(dot_y)
                    value = (w * (lse - s * ph_y)).sum()
                    grad_h = (s * w)[:, None] * (grad_h - dph_y[:, None] * Pd[y].to(work))
                    grad_P = s * grad_P.index_add_(0, y, -hw * dph_y[:, None])
                    grad_s = (w * (mean_dot - ph_y)).sum()
                    grad_q = s * (w * (mean_q - dq_y)).sum()
                    grad_u = s * (w * (mean_u - du_y)).sum()
                else:
                    value = (w * (lse - s * dot_y)).sum()
                    grad_h = (s * w)[:, None] * (grad_h - Pd[y].to(work))
                    grad_P = s * grad_P.index_add_(0, y, -hw)
                    grad_s = (w * (mean_dot - dot_y)).sum()
            # deger = value; eklenen terimlerin degeri 0, gradyanlari grad_* (geri yayilimda yeniden hesap yok)
            nll = value + (grad_h * (h - h.detach())).sum() + (grad_P * (P - P.detach())).sum()
            if scale.requires_grad:
                nll = nll + grad_s * (scale - scale.detach())
            if self.output_link:
                nll = nll + grad_q * (self.link_q - self.link_q.detach()) + grad_u * (self.link_u - self.link_u.detach())
            return nll + self.tokens.anchor_loss(), nll
        nll = masked_nll(self.logits(ids[:, :-1], None, inner), ids[:, 1:], mask[:, 1:])
        return nll + self.tokens.anchor_loss(), nll


# ---- okumalar

@torch.no_grad()
def deviation(model):
    """Token basina PF ile PL arasindaki aci (derece).  2 asin(|a-b|/2): acos(cos) float32'de 1 yakininda
    gurultulu (kipirdamamis nokta 0,03 derece okunuyordu)."""
    t = model.tokens
    chord = (t.points() - t.fixed_points).norm(dim=-1)
    return torch.rad2deg(2 * torch.asin((chord / 2).clamp(max=1)))
