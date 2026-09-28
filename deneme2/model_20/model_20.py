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

Adim 3: BlockModel.  Her konumun bir durumu var (hidden, h); TURNS tur boyunca Block uygulanir.
    h = PL                                              durum konumun noktasiyla baslar
    her turda (Block):
      c_t = sum_j a_tj h_j                              attention DURUMLARA bakar, durumlarin kendisini getirir
      h_t = norm(h_t + W_context c_t)                   okunan duruma yazilir (W_context 0'dan)
      h_t = norm(h_t + W_fact_out ReLU(W_fact_in h_t - fact_threshold))   FactUnits: iki bilgi birlikteyse yanar
                                                        (W_fact_out 0'dan)
    cikis: skor = scale <h, PL>                         son durum dogrudan noktalarla karsilastirilir (W_next yok)
    SHARED_BLOCK: ayni Block her turda (True) ya da her tura ayri Block (False)
    LAYERS: paylasilan blokta farkli Block sayisi; tur i Block i mod LAYERS'i kullanir (2 katman x 2 tur: A B A B)

Oneri A, COPY_PATH: attention ayni agirliklarla o konumlardaki kelimelerin kendisini de getirir, bir kapi yazilip
yazilmayacagina karar verir.
      c'_t = sum_j a_tj PL_j                            kelime noktalari: turlarda degismez
      g_t  = sigmoid(W_copy_gate h_t + copy_gate_bias)  kapi: bu konumda kopyala mi (0-1)
      h_t  = norm(h_t + W_context c_t + g_t W_copy c'_t) (W_copy 0'dan)

STREAM_NORM=False: akis normalize edilmez, okunan kopya normalize edilir (transformer'daki gibi; L2 norm, ogrenilen sayi yok).
      h_t = h_t + W_context sum_j a_tj norm(h_j)        (q, k da norm(h)'dan)
      h_t = h_t + W_fact_out ReLU(W_fact_in norm(h_t) - fact_threshold)
    cikis: skor = scale <norm(h_son), PL>

ROPE=True: attention'da q ve k konuma gore dondurulur (RoPE, transformer'daki apply_rope); iki konumun skoru aradaki mesafeye
de bagli olur.  Ogrenilen sayi yok; donme boyu degistirmez, cosine ve sabit att_scale aynen.
"""
import math

import torch
import torch.nn.functional as F

D = 64               # nokta boyutu; olculmedi (sozluk 74 > 64)
CONFIDENCE = 0.99    # hedef tam q yonundeyken, rakipler dikken verilebilecek olasilik -> scale
POINTS_SEED = 0
LEARN_POINTS = True  # False: PL = PF (sabit noktalar)
ANCHOR = 1e-3        # PF'den uzaklasmanin bedeli (0 = serbest); 8 aileli data_20 (iz 90024739fb8f) icin olculdu:
                     # kayip <= tavan + 0,01.  32 ailede (27 Eylul) yeniden olculmedi
ATTENTION = True     # Adim 2: attention; False = Adim 1 (yalniz son token)
T_MAX = 512          # baglam siniri (hedef); attention olcegi bundan: 512 konum arasindan 0,99 guvenle secebilsin
TURNS = 4            # Adim 3: blok tur sayisi (kullanici karari: 2; 28 Eylul: 2 x 2 varsayilan -> 4)
SHARED_BLOCK = True  # True: ayni Block her turda; False: her tura ayri Block (ayri katmanlar)
LAYERS = 2           # SHARED_BLOCK'ta farkli Block (katman) sayisi, turlar sirayla doner: tur i -> Block i mod LAYERS
                     # (kullanici, 28 Eylul: "2 tane paylaşımlı katman", sira ABAB, ad LAYERS).  Varsayilan 2 x 2 (kullanici,
                     # 28 Eylul: "2X2 şu an varsayılan olsun"; TinyStories 1 epok ppl 7,95 / X2 9,73).  1 = tek Block (X2)
FACT_UNITS = 256     # FactUnits birim sayisi; 4 x D (transformer aliskanligi), olculmedi
COPY_PATH = False    # Oneri A: kopya yolu ve kapisi; False = bugunku model birebir (kullanici onayi, 27 Eylul)
STREAM_NORM = True   # True: durum her eklemeden sonra kureye (bugunku).  False: akis normalize edilmez, yalniz attention'in
                     # ve FactUnits'in okudugu kopya normalize edilir (transformer gibi); cikis <norm(h), PL> (27 Eylul)
LAYER_NORM = False   # True: L2 norm yerine LayerNorm (norm_attention, norm_facts, norm_final; ogrenilen kazanc ve kayma);
                     # cikis <norm_final(h), PL>, sabit scale yok (kullanici, 27 Eylul: "Layernorm yapalım")
NORMALIZED_UPDATE = True    # h <- norm(h + alpha (norm(u) - h)), u blok ciktisi; alpha ogrenilen, tur ve alt blok basina
                            # d sayi (nGPT).  False: h <- norm(h + u) (28 Eylul'e kadar).  Kusur 1: FactUnits durumu eziyordu
                            # (|u| ~ 10-178, |h| = 1).  Varsayilan (kullanici, 28 Eylul: "evet ikisi de varsayılan olsun";
                            # akrabalik paket 188,3 / taban 178,5)
ALPHA_INIT = 0.1            # alpha'nin baslangici: nGPT 2026 tarifi (derinlikten bagimsiz 0,1)
SPHERE_WEIGHTS = True       # W_query, W_key, W_fact_in satirlari ve W_context, W_fact_out sutunlari baslangicta ve her
                            # optimizer adimindan sonra birim boya (nGPT); FactUnits girdisi sqrt(d) x kosinus.  Kusur 2:
                            # agirliklar ~20 kat buyuyor, adim sonuyordu.  Varsayilan (kullanici, 28 Eylul)
CANON = True         # Canon-A (Allen-Zhu 2025): attention girdisi x_t + sum_k w_k * x_(t-k), k = 0..3, w 0'dan.  Varsayilan:
                     # Model X2 = X1 + Canon (kullanici, 28 Eylul: "evet model X2 hayırlı olsun. Canon=True."; TinyStories
                     # "X1+C çok daha iyi görünüyor açık ara", akrabalik 191,0 / X1 188,3 ve cokussuz)
HEADS = 1            # attention head sayisi (kullanici, 28 Eylul: "heads = 4 onaylıyorum"; olcum: uzak bakan tek head
                     # sorgularin %76'sinda agirligini 2+ ayri bolgeye boluyor, ad hedeflerinin %31'inde iki adi birlikte
                     # getiriyor).  1 = tek head, V yok (bugunku model); H > 1: d H'ye bolunur, W_value (d x d, birim
                     # baslar) her head'in tasiyacagini secer
ROPE = True          # attention'in q ve k'sina RoPE (konum bilgisi).  Varsayilanlar = Model X (C' + RoPE; kullanici, 27 Eylul:
                     # "Model X varsayilan model olsun" onayi); False = RoPE'suz C'


def apply_rope(x, positions=None):
    """RoPE, x (..., T, d): konum t'de her boyut cifti t · 10000^(-2i/d) acisiyla dondurulur; iki konumun
    <q, k> skoru yalniz aradaki mesafeye bagli kalir.  Parametresi yok.  positions (B, T): satir basina konum
    (onbellekli uretim); None = 0..T-1."""
    T, dh = x.shape[-2], x.shape[-1]
    # hesap en az fp32: bf16'da konum 256'dan sonra tam sayi degil, aci kayar.  fp32 / fp64'te sonuc aynen
    work = torch.promote_types(x.dtype, torch.float32)
    freq = 10000.0 ** (-torch.arange(0, dh, 2, device=x.device, dtype=work) / dh)          # (d/2,)
    positions = torch.arange(T, device=x.device, dtype=work) if positions is None else positions.to(work)
    angle = positions[..., None] * freq                                                     # (..., T, d/2)
    cos, sin = angle.cos(), angle.sin()
    x1, x2 = x[..., 0::2].to(work), x[..., 1::2].to(work)
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2).to(x.dtype)


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
        # 100 * seed: her tohum kendi 100'luk araliginda -- tohumlar rastgele sayi paylasmaz (denetim, 27 Eylul); tohum 0 aynen
        self.tokens = TokenPoints(n, d, learn_points, anchor, 100 * seed)
        self.next = NextRelation(d, 100 * seed + 1)
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
    """Nedensel tam attention; getirdigi sey girdinin kendisi (Adim 2: PL noktalari, Adim 3: durumlar).  W_context 0'dan."""

    def __init__(self, d=D, t_max=T_MAX, confidence=CONFIDENCE, seed=POINTS_SEED + 2, copy_path=False, rope=False,
                 heads=1):
        super().__init__()
        assert d % heads == 0 and (heads == 1 or not copy_path), "d head sayisina bolunmeli; kopya yolu yalniz tek head'de"
        g = torch.Generator().manual_seed(seed)
        self.rope, self.heads = rope, heads
        # W_query, W_key: d x d, rastgele / √d baslar, egitimle degisir.  PL'yi baska bir yone ceviren dogrusal donusum
        # (dondurur, gerer, sikistirir); ardindan norm geldigi icin yalniz yon kalir.
        self.W_query = torch.nn.Parameter(torch.randn(d, d, generator=g) / d ** 0.5)   # "ne ariyorum"
        self.W_key = torch.nn.Parameter(torch.randn(d, d, generator=g) / d ** 0.5)   # "bende ne var"
        self.W_context = torch.nn.Parameter(torch.zeros(d, d))                      # getirileni tahmine katar, 0'dan
        self.scale = scale_for(t_max, confidence)                                  # ln(0,99 · 511 / 0,01) = 10,83
        # V (value) matrisi tek head'de yok: W_context'ten once ikinci bir matris, arada dogrusal olmayan adim olmadigi
        # icin W_context ile carpimi tek bir d x d matrise esit olurdu.  Cok head'de her head'in tasiyacagini V secer
        if heads > 1:                                     # birim baslar: ilk adimda head h durumun h. dilimini tasir
            self.W_value = torch.nn.Parameter(torch.eye(d))
        self.copy_path = copy_path
        if copy_path:                                     # Oneri A; sifirlar uretecten sayi cekmez, diger baslangiclar ayni
            self.W_copy = torch.nn.Parameter(torch.zeros(d, d))                  # getirilen kelimeleri duruma yazar, 0'dan
            self.W_copy_gate = torch.nn.Parameter(torch.zeros(1, d))             # kapi: bu konumda kopyala mi
            self.copy_gate_bias = torch.nn.Parameter(torch.zeros(1))             # baslangicta kapi yari acik (0,5)

    def queries_keys(self, x, positions=None):
        # q_t = W_query·PL_t / |W_query·PL_t|      k_j = W_key·PL_j / |W_key·PL_j|      (ara sonuc, saklanmaz)
        q, k = x @ self.W_query.T, x @ self.W_key.T
        if self.heads > 1:                                # (.., T, d) -> (.., H, T, d/H): her head kendi dilimini normlar
            q, k = (z.unflatten(-1, (self.heads, -1)).transpose(-3, -2) for z in (q, k))
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        if self.rope:                                     # konuma gore dondur; boy 1 kalir
            q, k = apply_rope(q, positions), apply_rope(k, positions)
        return q, k

    def forward(self, x, points=None, cache=None):
        """x (B, T, d) PL ya da durum dizisi -> c (B, T, d).  points verilirse (Oneri A) ayni agirliklarla (c, c').
        cache (AttentionCache): onbellekli uretim; x yalniz yeni konumlar."""
        if cache is not None:
            assert self.heads == 1, "onbellekli uretim yalniz tek head'de"
            return cache.attend(self, x, points)
        q, k = self.queries_keys(x)
        if self.heads > 1:                                # c = [a_1 (V_1 x) ; ... ; a_H (V_H x)], head'ler yan yana
            v = (x @ self.W_value.T).unflatten(-1, (self.heads, -1)).transpose(-3, -2)
            c = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
            return c.transpose(-3, -2).flatten(-2)
        # scaled_dot_product_attention (SDPA) su hesabi yapar, weights() ile ayni:
        #   s_tj = scale · <q_t, k_j>                 her konum t, her konum j icin
        #   j > t ise s_tj = -∞                       is_causal: sonrakilere bakilmaz
        #   a_tj = e^s_tj / Σ_{i<=t} e^s_ti            softmax, satir toplami 1
        #   c_t  = Σ_{j<=t} a_tj · x_j                getirilen: agirlikli karisim (Adim 2: x = PL, Adim 3: x = h)
        if points is None:
            return F.scaled_dot_product_attention(q, k, x, is_causal=True, scale=self.scale)
        # c'_t = Σ a_tj · PL_j: iki deger yan yana tek SDPA'dan, ayni a_tj ile
        both = F.scaled_dot_product_attention(q, k, torch.cat([x, points], -1), is_causal=True, scale=self.scale)
        return both[..., :x.shape[-1]], both[..., x.shape[-1]:]

    def weights(self, x):
        """Okuma icin acik hesap: a (B, T, T) -- cok head'de (B, H, T, T) --, satir t yalniz j <= t."""
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
    key'leri o konuma gelinceye kadar maskeli kalir).  capacity: istem + uretilecek token."""

    def __init__(self, lengths, capacity):
        self.next_position, self.capacity = lengths.clone(), capacity   # satir basina siradaki konum
        self.keys = self.values = None
        self.span = 0                                            # yazilmis en uzun satirin boyu (Python sayisi: senkron yok)

    def attend(self, at, x, points=None):
        v = x if points is None else torch.cat([x, points], -1)
        B, t, d = x.shape
        if self.keys is None:                                    # istem: CausalAttention.forward ile ayni hesap
            q, k = at.queries_keys(x)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=at.scale)
            self.keys = k.new_zeros(B, self.capacity, k.shape[-1])
            self.values = v.new_zeros(B, self.capacity, v.shape[-1])
            self.keys[:, :t], self.values[:, :t] = k, v
            self.span = t
        else:
            assert t == 1, "istemden sonra satir basina tek token"
            pos = self.next_position[:, None]                    # (B, 1)
            q, k = at.queries_keys(x, pos)
            self.keys.scatter_(1, pos[..., None].expand(-1, -1, k.shape[-1]), k)
            self.values.scatter_(1, pos[..., None].expand(-1, -1, v.shape[-1]), v)
            self.next_position = self.next_position + 1          # yeni tensor: pos (gorunum) degismesin
            self.span += 1                                       # butun satirlar birer ilerler
            S = self.span
            allowed = torch.arange(S, device=x.device)[None, None, :] <= pos[..., None]   # j <= konum; ilerisi dolgu
            out = F.scaled_dot_product_attention(q, self.keys[:, :S], self.values[:, :S], attn_mask=allowed,
                                                 scale=at.scale)
        if points is None:
            return out
        return out[..., :d], out[..., d:]


class SequenceModel(torch.nn.Module):
    """Dizi uzerinde model: attention=False iken her konumda Adim 1 (BigramModel) ile ayni."""

    def __init__(self, n, d=D, attention=ATTENTION, learn_points=LEARN_POINTS, anchor=ANCHOR, confidence=CONFIDENCE,
                 t_max=T_MAX, seed=POINTS_SEED):
        super().__init__()
        self.tokens = TokenPoints(n, d, learn_points, anchor, 100 * seed)          # 100 * seed: BigramModel'deki gibi
        self.next = NextRelation(d, 100 * seed + 1)
        self.attention = CausalAttention(d, t_max, confidence, 100 * seed + 2) if attention else None
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
        nll = masked_nll(logits, ids[:, 1:], valid)
        return nll + self.tokens.anchor_loss(), nll       # + λ · Σ Δ²


class FactUnits(torch.nn.Module):
    """Adim 3 donusturme: her konumda ayri.  u = ReLU(W_fact_in · h - fact_threshold): birim, girdileri birlikte yeterince
    guclu ise yanar (VE gibi); cikti W_fact_out · u duruma eklenir.  W_fact_out 0'dan: baslangicta etkisiz."""

    def __init__(self, d=D, units=FACT_UNITS, seed=POINTS_SEED + 3):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W_fact_in = torch.nn.Parameter(torch.randn(units, d, generator=g) / d ** 0.5)   # durumdan birimlere
        self.fact_threshold = torch.nn.Parameter(torch.zeros(units))                          # ornekteki "- 1"
        self.W_fact_out = torch.nn.Parameter(torch.zeros(d, units))                          # birimlerden duruma, 0'dan

    def forward(self, h):
        # u_i = max(0, Σ_b W_fact_in[i,b] · h_b - fact_threshold_i);   cikti_a = Σ_i W_fact_out[a,i] · u_i
        return torch.relu(h @ self.W_fact_in.T - self.fact_threshold) @ self.W_fact_out.T


class Block(torch.nn.Module):
    """Adim 3 bir tur: attention durumlara bakar ve getirdigini duruma yazar; FactUnits durumu donusturur."""

    def __init__(self, d=D, t_max=T_MAX, confidence=CONFIDENCE, units=FACT_UNITS, seed=POINTS_SEED + 2, copy_path=False,
                 stream_norm=True, layer_norm=False, rope=False, normalized_update=False, sphere_weights=False, canon=False,
                 heads=1):
        super().__init__()
        self.attention = CausalAttention(d, t_max, confidence, seed, copy_path=copy_path, rope=rope, heads=heads)
        self.facts = FactUnits(d, units, seed + 1)
        self.stream_norm, self.layer_norm = stream_norm, layer_norm
        self.normalized_update, self.sphere_weights, self.canon = normalized_update, sphere_weights, canon
        if canon:                                         # w_k: k konum oncesinden, boyut basina; 0'dan (adim 0 = Canon'suz)
            self.canon_weights = torch.nn.Parameter(torch.zeros(4, d))
        if layer_norm:                                    # L2 normun yerine: ortalama cikar, ogrenilen kazanc ve kayma
            self.norm_attention = torch.nn.LayerNorm(d)
            self.norm_facts = torch.nn.LayerNorm(d)

    def forward(self, h, points=None, cache=None, alpha_attention=None, alpha_facts=None):
        """alpha_attention, alpha_facts (d,): normalized_update'te bu turun alpha'lari."""
        at = self.attention
        unit = lambda v: F.normalize(v, dim=-1)
        norm_a, norm_f = (self.norm_attention, self.norm_facts) if self.layer_norm else (unit, unit)
        x = h if self.stream_norm else norm_a(h)             # attention'in okudugu; stream_norm'da h zaten normlu
        if self.canon:                                       # Canon-A: x_t + sum_k w_k * x_(t-k), baslangictan once 0
            assert cache is None, "Canon onbellekli uretimi desteklemiyor (onceki konumlar gerekir)"
            T = x.shape[-2]
            x = x + sum(self.canon_weights[k] * F.pad(x, (0, 0, k, 0))[..., :T, :] for k in range(4))
        if at.copy_path:
            assert points is not None, "kopya yolu acik: kelime noktalari (points) verilmeli"   # yoksa c, c' batch'ten bolunur
            c, c_copy = at(x, points, cache)
            gate = torch.sigmoid(x @ at.W_copy_gate.T + at.copy_gate_bias)          # g_t, (B, T, 1): konumun kendi durumundan
            added = c @ at.W_context.T + gate * (c_copy @ at.W_copy.T)             # W_context · c_t + g_t · W_copy · c'_t
        else:
            added = at(x, cache=cache) @ at.W_context.T                            # W_context · c_t
        # sphere_weights: W_fact_in satirlari birim, h birim -> girdi kosinus (tipik ±1/√d); √d ile esik O(1) olcekte
        facts = (lambda v: self.facts(v * v.shape[-1] ** 0.5)) if self.sphere_weights else self.facts
        if self.normalized_update:                                                 # u'nun boyu silinir, adimi alpha belirler
            h = unit(h + alpha_attention * (unit(added) - h))                     # h = norm(h + α_A ⊙ (norm(W_context c) - h))
            return unit(h + alpha_facts * (unit(facts(h)) - h))                   # h = norm(h + α_F ⊙ (norm(olgu(h)) - h))
        if self.stream_norm:
            h = norm_a(h + added)                                                  # h_t = norm(h_t + W_context · c_t)
            return norm_f(h + facts(h))                                            # h_t = norm(h_t + olgu(h_t))
        h = h + added                                                              # akis normalize edilmez:
        return h + facts(norm_f(h))                                                # h_t = h_t + olgu(norm(h_t))


class BlockModel(torch.nn.Module):
    """Adim 3: durum (hidden) PL ile baslar, TURNS tur Block; cikis son durumun kendisi (W_next yok).
    normalized_update ve sphere_weights kapaliyken (28 Eylul'e kadarki model) W_context = W_fact_out = 0 baslar, durum PL'de
    kalir: skor = scale <PL_t, PL>."""

    def __init__(self, n, d=D, turns=TURNS, shared=SHARED_BLOCK, layers=LAYERS, learn_points=LEARN_POINTS,
                 anchor=ANCHOR, confidence=CONFIDENCE, t_max=T_MAX, units=FACT_UNITS, seed=POINTS_SEED,
                 copy_path=COPY_PATH, stream_norm=STREAM_NORM, layer_norm=LAYER_NORM, rope=ROPE,
                 normalized_update=NORMALIZED_UPDATE, sphere_weights=SPHERE_WEIGHTS, canon=CANON, heads=HEADS):
        super().__init__()
        assert not normalized_update or (stream_norm and not layer_norm), "normalized_update akis normuyla (L2) calisir"
        self.tokens = TokenPoints(n, d, learn_points, anchor, 100 * seed)          # 100 * seed: BigramModel'deki gibi
        assert not shared or turns % layers == 0, "turns (%d) layers'in (%d) kati olmali: her Block esit sayida tur" % (
            turns, layers)                                # ayri blokta (shared=False) layers yok sayilir: her tura bir Block
        count = layers if shared else turns
        self.blocks = torch.nn.ModuleList(Block(d, t_max, confidence, units, 100 * seed + 10 + 2 * i, copy_path=copy_path,
                                                stream_norm=stream_norm, layer_norm=layer_norm, rope=rope,
                                                normalized_update=normalized_update, sphere_weights=sphere_weights,
                                                canon=canon, heads=heads)
                                          for i in range(count))
        self.turns, self.shared, self.copy_path, self.stream_norm = turns, shared, copy_path, stream_norm
        self.layers, self.layer_norm, self.rope, self.heads = layers, layer_norm, rope, heads
        self.normalized_update, self.sphere_weights, self.canon = normalized_update, sphere_weights, canon
        if normalized_update or sphere_weights:           # sifir yon normalize edilemez: W_context, W_fact_out rastgele baslar
            g = torch.Generator().manual_seed(100 * seed + 9)
            with torch.no_grad():
                for b in self.blocks:
                    b.attention.W_context.copy_(torch.randn(d, d, generator=g) / d ** 0.5)
                    b.facts.W_fact_out.copy_(torch.randn(d, units, generator=g) / units ** 0.5)
        if normalized_update:                             # tur basina (paylasilan blokta da): (tur, d)
            self.alpha_attention = torch.nn.Parameter(torch.full((turns, d), ALPHA_INIT))
            self.alpha_facts = torch.nn.Parameter(torch.full((turns, d), ALPHA_INIT))
        if sphere_weights:
            self.normalize_weights()
        if layer_norm:                                    # cikista: keskinligi kazanc ogrenir, sabit scale kullanilmaz
            self.norm_final = torch.nn.LayerNorm(d)
        self.scale = scale_for(n, confidence)

    def turn_blocks(self):
        return [self.blocks[i % len(self.blocks)] for i in range(self.turns)]     # paylasilan: A B A B; ayri: her tura biri

    def hidden(self, ids, caches=None):
        """Her turdan sonraki durumlar: [h0 = PL, h1, ..., h_TURNS], her biri (B, T, d).  caches: tur basina bir
        AttentionCache (onbellekli uretim; ids yalniz yeni token'lar)."""
        h = self.tokens.points()[ids]
        points = h if self.copy_path else None                # Oneri A: kelimelerin kendi noktalari, turlarda degismez
        out = [h]
        for i, block in enumerate(self.turn_blocks()):
            alphas = (dict(alpha_attention=self.alpha_attention[i], alpha_facts=self.alpha_facts[i])
                      if self.normalized_update else {})
            h = block(h, points, None if caches is None else caches[i], **alphas)
            out.append(h)
        return out

    @torch.no_grad()
    def normalize_weights(self):
        """sphere_weights: girdisi durum olan matrislerin satirlari (W_query, W_key, W_fact_in), duruma yazanlarin sutunlari
        (W_context, W_copy, W_fact_out) birim boya; baslangicta ve her optimizer adimindan sonra (train_seq)."""
        for b in self.blocks:
            at, f = b.attention, b.facts
            for w in (at.W_query, at.W_key, f.W_fact_in) + ((at.W_value,) if at.heads > 1 else ()):
                w.copy_(F.normalize(w, dim=1))
            for w in (at.W_context, f.W_fact_out) + ((at.W_copy,) if at.copy_path else ()):
                w.copy_(F.normalize(w, dim=0))

    def logits(self, ids, caches=None):
        P = self.tokens.points()
        h = self.hidden(ids, caches)[-1]                       # son durum; stream_norm'da zaten kurede
        if self.layer_norm:
            return self.norm_final(h) @ P.T                    # skor_tj = <LN(h_t), PL_j>
        if not self.stream_norm:
            h = F.normalize(h, dim=-1)                         # akis normalize edilmediyse cikista bir kez
        return self.scale * h @ P.T                            # skor_tj = scale · <h_t, PL_j>

    def loss(self, ids, mask):
        nll = masked_nll(self.logits(ids[:, :-1]), ids[:, 1:], mask[:, 1:])
        return nll + self.tokens.anchor_loss(), nll


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
