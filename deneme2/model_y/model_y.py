# -*- coding: utf-8 -*-
"""model_y -- adim adim kurulan model (CLAUDE.md kural 13).

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
    her turda (Block), varsayilan (Model X2, 2 x 2):
      x_t = h_t + sum_{k=0..3} w_k h_(t-k)              Canon-A: attention'in girdisi (w 0'dan)
      c_t = sum_j a_tj x_j                              attention DURUMLARA bakar (HEADS > 1: head basina W_value dilimi)
      h_t = norm(h_t + a_A (norm(W_context c_t) - h_t))                 normalized_update: alpha kadar don
      h_t = norm(h_t + a_F (norm(W_fact_out (SiLU(W_fact_in x) * (W_fact_up x))) - h_t)),  x = sqrt(d) h_t   FactUnits
      anahtarlar kapaliyken (28 Eylul oncesi): h_t = norm(h_t + W_context c_t), h_t = norm(h_t + FactUnits(h_t));
      W_context ve W_fact_out 0'dan (paket acikken rastgele, birim sutun)
    cikis: skor = scale <h, PL>                         son durum dogrudan noktalarla karsilastirilir (W_next yok)
    LEARN_OUTPUT_SCALE: skor = e^tau <h, PL>, tau = log_output_scale ogrenilir, ln(scale)'dan baslar
    LOSS_CHUNK: egitim kaybi sozluk parcalariyla (tam logits tablosu yok, gradyan ileri hesapta); sinav logits'le
    SHARED_BLOCK / LAYERS: turlar LAYERS farkli Block'u sirayla kullanir (2 katman x 2 tur: A B A B); SHARED_BLOCK=False:
    her tura ayri Block

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
SHARED_BLOCK = True  # True: turlar LAYERS Block'u sirayla paylasir; False: her tura ayri Block (ayri katmanlar)
LAYERS = 2           # SHARED_BLOCK'ta farkli Block (katman) sayisi, turlar sirayla doner: tur i -> Block i mod LAYERS
                     # (kullanici, 28 Eylul: "2 tane paylaşımlı katman", sira ABAB, ad LAYERS).  Varsayilan 2 x 2 (kullanici,
                     # 28 Eylul: "2X2 şu an varsayılan olsun"; TinyStories 1 epok ppl 7,95 / X2 9,73).  1 = tek Block x TURNS tur
SHARED_FACTS = False # True: SHARED_BLOCK'ta FactUnits de turlar arasinda paylasilir (29 Eylul'e kadar).  False: her turun
                     # kendi FactUnits'i, attention paylasimli kalir (kullanici, 29 Eylul: "bu öneri mantıklı geldi bana";
                     # varsayilan: "Facts fazla iken iyi sonuç aldık değil mi ? Tabiki olsun"): SimpleStories 1 epok EMA acc
                     # 0,5547 -> 0,5659, bpb 0,5932 -> 0,5694; parametre 6,9M -> 10,4M, hesap ayni
FACT_UNITS = 170     # FactUnits birim sayisi; SwiGLU'nun yerlesik genisligi 8/3 x D (Shazeer 2020 "2/3", LLaMA "2/3 4d",
                     # MobileLLM 576 -> 1536); TinyStories'te 384 -> 1024 olculdu (ppl 6,94).  relu'da 4 x D = 256 aliskanligi
FACT_ACTIVATION = "swiglu"   # FactUnits: "relu" u = ReLU(W_fact_in x - fact_threshold) | "swiglu" u = SiLU(W_fact_in x) *
                           # (W_fact_up x), esik yok (kapili; nGPT gibi x = sqrt(d) h) | "reglu" u = ReLU(W_fact_in x -
                           # fact_threshold) * (W_fact_up x) (kapi tam sifir, okunur).  Kullanici, 28 Eylul: "önce sadece S
                           # bakalım", sonra "1 ve 2 ok": varsayilan swiglu (TinyStories 10k ppl 6,94 / relu 7,22) ve G kolu
STREAM_NORM = True   # True: durum her eklemeden sonra kureye (bugunku).  False: akis normalize edilmez, yalniz attention'in
                     # ve FactUnits'in okudugu kopya normalize edilir (transformer gibi); cikis <norm(h), PL> (27 Eylul)
LAYER_NORM = False   # True: L2 norm yerine LayerNorm (norm_attention, norm_facts, norm_final; ogrenilen kazanc ve kayma);
                     # cikis <norm_final(h), PL>, sabit scale yok (kullanici, 27 Eylul: "Layernorm yapalım")
NORMALIZED_UPDATE = True    # h <- norm(h + alpha (norm(u) - h)), u blok ciktisi; alpha ogrenilen, tur ve alt blok basina
                            # d sayi (nGPT).  False: h <- norm(h + u) (28 Eylul'e kadar).  Kusur 1: FactUnits durumu eziyordu
                            # (|u| ~ 10-178, |h| = 1).  Varsayilan (kullanici, 28 Eylul: "evet ikisi de varsayılan olsun";
                            # akrabalik paket 188,3 / taban 178,5)
ALPHA_INIT = 0.1            # alpha'nin baslangici: nGPT 2026 tarifi (derinlikten bagimsiz 0,1)
LAST_FACTS_ALPHA_INIT = 1.0  # yalniz son turun FactUnits alpha'si (alpha_facts[turns - 1]) bundan baslar, gerisi ALPHA_INIT
                             # (kullanici, 28 Eylul: "Başlangıç hatası: model ilk adımda kendi girdisini tahmin ediyor önerin
                             # kabul").  0,1'de son durum girdi noktasinda kaliyor (cos 0,914): adim-0 kaybi 12,41 > ln n;
                             # 1,0'da ln n + ln E e^{s<h,p>} ~ 9,23.  Egitilmis modelde son alpha_F medyan 0,956
SPHERE_WEIGHTS = True       # W_query, W_key, W_fact_in, W_value satirlari ve W_context, W_fact_out sutunlari basta ve her
                            # optimizer adimindan sonra birim boya (nGPT); FactUnits girdisi sqrt(d) x kosinus.  Kusur 2:
                            # agirliklar ~20 kat buyuyor, adim sonuyordu.  Varsayilan (kullanici, 28 Eylul)
CANON = True         # Canon-A (Allen-Zhu 2025): attention girdisi x_t + sum_k w_k * x_(t-k), k = 0..3, w 0'dan.  Varsayilan:
                     # Model X2 = X1 + Canon (kullanici, 28 Eylul: "evet model X2 hayırlı olsun. Canon=True."; TinyStories
                     # "X1+C çok daha iyi görünüyor açık ara", akrabalik 191,0 / X1 188,3 ve cokussuz)
INPUT_EMBEDDING = False  # True: girdi ayri, ogrenilen tablo (V x d, PF'den baslar, capa yok; nGPT'deki E_input) -- cikis
                         # PL'de kalir.  False: girdi = cikis = PL (29 Eylul'e kadarki model).  Kullanici, 29 Eylul: "Bunu
                         # yapalım bence ihtiyaç net zaten ngpt yapmış ama c mantıklı gibi"; adlar "Önerilerin kabul"
INPUT_BIGRAMS = 0        # > 0: girdiye (onceki token, token) ikilisinin satiri eklenir (Over-Tokenized); satir sayisi = en
                         # sik K ikili (liste veriden, bigram_keys), listede olmayan ikilide yalniz token.  0 = yok
INPUT_EMBEDDING_SPHERE = True  # INPUT_EMBEDDING'de girdi tablosunun satirlari basta ve her optimizer adimindan sonra
                               # birim boya (nGPT).  False (63338af): gradyan satira dik, boy buyuyor, etkin lr 1/|E| ile
                               # dusuyordu (t4500'de |E| medyan 4,9).  Kullanici, 29 Eylul: adlar "Onaylıyorum"
FIRST_TURN_FACTS = True  # False: tur 1'in FactUnits alt adimi yok (C ajani: tur 1 FactUnits fiilen token tablosu)
HEADS = 4            # attention head sayisi (kullanici, 28 Eylul: "Evet, varsayılan 4"; TinyStories 10k ppl 7,22 / tek head
                     # 7,63).  H > 1: d H'ye bolunur, W_value (d x d, birim baslar) her head'in tasiyacagini secer;
                     # 1 = tek head, V yok (28 Eylul'e kadarki model)
ROPE = True          # attention'in q ve k'sina RoPE (konum bilgisi).  Varsayilanlar = Model X (C' + RoPE; kullanici, 27 Eylul:
                     # "Model X varsayilan model olsun" onayi); False = RoPE'suz C'
LEARN_OUTPUT_SCALE = True   # cikis olcegi ogrenilir: e^tau, tau = log_output_scale (kullanici, 28 Eylul: "evet öğrenilen çıkış
                            # ölçeği olsun!").  3 ajan: donuk modelde olcek ~15'e cekilince -0,053..-0,067 nat; nGPT de
                            # logit olcegini ogreniyor.  False: sabit scale (28 Eylul'e kadarki model)
LOSS_CHUNK = 4096    # egitim kaybi sozluk parcalariyla: tam logits tablosu (B x T x V) olusmaz, gradyan ileri hesapta biriktirilir
                     # (kullanici, 28 Eylul: sozluk ileride 50k; "4096 ilk önerdiğin olsun").  0 = tek parca (28 Eylul'e kadarki
                     # yol).  Sinav ve uretim logits'le, degismez
OUTPUT_LINK = True   # cikis bagi phi (kullanici, 28 Eylul: "o zaman bu koşuyu da başlat"; adlar onayli): skor = s phi(c),
                     # phi(c) = c (1 + c (q + c (q^2/3 + u))), c = <h, PL>, q = link_q, u = link_u >= 0 (her adimdan sonra
                     # kirpilir) -> phi' = (1 + q c)^2 + 3 u c^2 >= 0: phi hep artan, en olasi token degismez; q = u = 0:
                     # dogrusal.  Varsayilan (kullanici, 29 Eylul: "1 evet varsayılan olsun"): SimpleStories 1 epok EMA bpb
                     # 0,6413 -> 0,6090, acc +1,1 puan (lr'yi de ~yariya indirdi, ayristirilamadi).  False: dogrusal skor


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
        # skor_j = scale · <q, PL_j> = scale · Σ_a q_a · PL_ja      her token j icin (n tane)
        return self.scale * self.next(P[ids]) @ P.T

    def loss(self, inputs, targets):
        """(toplam, nll): toplam = nll + capa."""
        # cross_entropy: p_j = e^skor_j / Σ_i e^skor_i (softmax);  nll = ortalama( -log p_hedef )
        nll = F.cross_entropy(self.logits(inputs), targets)
        return nll + self.tokens.anchor_loss(), nll


class CausalAttention(torch.nn.Module):
    """Nedensel tam attention.  Tek head'de getirdigi sey girdinin kendisi (Adim 2: PL noktalari, Adim 3: durumlar);
    HEADS > 1: head basina W_value x'in dilimi.  W_context 0'dan (paket acikken BlockModel rastgele baslatir)."""

    def __init__(self, d=D, t_max=T_MAX, confidence=CONFIDENCE, seed=POINTS_SEED + 2, rope=False, heads=1):
        super().__init__()
        assert d % heads == 0, "d head sayisina bolunmeli"
        assert not rope or (d // heads) % 2 == 0, "RoPE icin head boyu cift olmali"
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

    def queries_keys(self, x, positions=None):
        # q_t = W_query·PL_t / |W_query·PL_t|      k_j = W_key·PL_j / |W_key·PL_j|      (ara sonuc, saklanmaz)
        q, k = x @ self.W_query.T, x @ self.W_key.T
        if self.heads > 1:                                # (.., T, d) -> (.., H, T, d/H): her head kendi dilimini normlar
            q, k = (z.unflatten(-1, (self.heads, -1)).transpose(-3, -2) for z in (q, k))
            positions = None if positions is None else positions[:, None]   # (B, 1, T): head ekseniyle hizali
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        if self.rope:                                     # konuma gore dondur; boy 1 kalir
            q, k = apply_rope(q, positions), apply_rope(k, positions)
        return q, k

    def forward(self, x, cache=None):
        """x (B, T, d) PL ya da durum dizisi -> c (B, T, d).  cache (AttentionCache): onbellekli uretim; x yalniz yeni
        konumlar."""
        if cache is not None:
            return cache.attend(self, x)
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
        return F.scaled_dot_product_attention(q, k, x, is_causal=True, scale=self.scale)

    def weights(self, x):
        """Okuma icin acik hesap: a (B, T, T) -- cok head'de (B, H, T, T) --, satir t yalniz j <= t.  x attention'in
        GERCEK girdisi olmali: Block'ta Canon karisimi (Block.forward'daki x), h degil."""
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
    yeni konum onceki 3 konumu buradan okur.  Cok head'de key ve value head basina."""

    def __init__(self, lengths, capacity):
        self.next_position, self.capacity = lengths.clone(), capacity   # satir basina siradaki konum
        self.keys = self.values = self.canon_inputs = None
        self.last_token = None                                   # INPUT_BIGRAMS: satir basina onceki token (istemden sonra)
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
        """Tek head: key (B, S, d), value x.  Cok head: key ve value (B, H, S, d/H); value W_value x'in head dilimi."""
        B, t = x.shape[:2]
        H = at.heads
        v = x if H == 1 else (x @ at.W_value.T).unflatten(-1, (H, -1)).transpose(1, 2)
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
            at_pos = pos[..., None] if H == 1 else pos[:, None, :, None]   # yazilacak konum: (B, 1, 1) / (B, 1, 1, 1)
            self.keys.scatter_(-2, at_pos.expand(*k.shape), k)
            self.values.scatter_(-2, at_pos.expand(*v.shape), v)
            self.next_position = self.next_position + 1          # yeni tensor: pos (gorunum) degismesin
            self.span += 1                                       # butun satirlar birer ilerler
            S = self.span
            allowed = torch.arange(S, device=x.device)[None, None, :] <= pos[..., None]   # (B, 1, S): j <= konum
            if H > 1:
                allowed = allowed[:, None]                       # (B, 1, 1, S): butun head'ler
            out = F.scaled_dot_product_attention(q, self.keys[..., :S, :], self.values[..., :S, :], attn_mask=allowed,
                                                 scale=at.scale)
        return out if H == 1 else out.transpose(1, 2).flatten(-2)   # head'ler yan yana


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
        P = self.tokens.points()                          # PL, n x d
        x = P[ids]                                        # her konumun token'inin noktasi
        raw = x @ self.next.W_next.T                      # raw_t = W_next · PL_t
        if self.attention is not None:
            raw = raw + self.attention(x) @ self.attention.W_context.T   # raw_t = W_next · PL_t + W_context · c_t
        # q_t = raw_t / |raw_t|;   skor_tj = scale · <q_t, PL_j>   (n token)
        return self.scale * F.normalize(raw, dim=-1) @ P.T

    def loss(self, ids, mask):
        """mask (B, T) gercek token; hedef t+1 gercekse konum t sayilir.  (toplam, nll)."""
        logits = self.logits(ids[:, :-1])                 # konum t'nin skorlari ...
        valid = mask[:, 1:]                               # ... hedefi t+1'deki token; <pad> hedefler sayilmaz
        # cross_entropy: p = softmax(skor);  nll = ortalama( -log p(hedef) )  butun gercek konumlarda
        nll = masked_nll(logits, ids[:, 1:], valid)
        return nll + self.tokens.anchor_loss(), nll       # + λ · Σ Δ²


class FactUnits(torch.nn.Module):
    """Adim 3 donusturme: her konumda ayri.  relu: u = ReLU(W_fact_in · h - fact_threshold), birim girdileri birlikte
    yeterince guclu ise yanar.  swiglu: u = SiLU(W_fact_in · h) ⊙ (W_fact_up · h), kapi (W_fact_in) icerigi (W_fact_up)
    acar; esik yok.  Cikti W_fact_out · u.  W_fact_out 0'dan: baslangicta etkisiz (paket acikken BlockModel rastgele
    baslatir)."""

    def __init__(self, d=D, units=FACT_UNITS, seed=POINTS_SEED + 3, activation="relu"):
        super().__init__()
        assert activation in ("relu", "swiglu", "reglu"), activation
        g = torch.Generator().manual_seed(seed)
        self.activation = activation
        self.W_fact_in = torch.nn.Parameter(torch.randn(units, d, generator=g) / d ** 0.5)   # durumdan birimlere (kapi)
        if activation != "swiglu":
            self.fact_threshold = torch.nn.Parameter(torch.zeros(units))                      # ornekteki "- 1"
        if activation != "relu":
            self.W_fact_up = torch.nn.Parameter(torch.randn(units, d, generator=g) / d ** 0.5)   # tasinan icerik
        self.W_fact_out = torch.nn.Parameter(torch.zeros(d, units))                          # birimlerden duruma, 0'dan

    def forward(self, h):
        if self.activation == "swiglu":                  # u_i = SiLU(W_fact_in[i] · h) · (W_fact_up[i] · h)
            return (F.silu(h @ self.W_fact_in.T) * (h @ self.W_fact_up.T)) @ self.W_fact_out.T
        # u_i = max(0, Σ_b W_fact_in[i,b] · h_b - fact_threshold_i);   cikti_a = Σ_i W_fact_out[a,i] · u_i
        u = torch.relu(h @ self.W_fact_in.T - self.fact_threshold)
        if self.activation == "reglu":                   # kapi tam sifir, icerik W_fact_up · h
            u = u * (h @ self.W_fact_up.T)
        return u @ self.W_fact_out.T


class Block(torch.nn.Module):
    """Adim 3 bir tur: attention durumlara bakar ve getirdigini duruma yazar; FactUnits durumu donusturur."""

    def __init__(self, d=D, t_max=T_MAX, confidence=CONFIDENCE, units=FACT_UNITS, seed=POINTS_SEED + 2,
                 stream_norm=True, layer_norm=False, rope=False, normalized_update=False, sphere_weights=False, canon=False,
                 heads=1, fact_activation="relu"):
        super().__init__()
        self.attention = CausalAttention(d, t_max, confidence, seed, rope=rope, heads=heads)
        self.facts = FactUnits(d, units, seed + 1, activation=fact_activation)
        self.stream_norm, self.layer_norm = stream_norm, layer_norm
        self.normalized_update, self.sphere_weights, self.canon = normalized_update, sphere_weights, canon
        if canon:                                         # w_k: k konum oncesinden, boyut basina; 0'dan (adim 0 = Canon'suz)
            self.canon_weights = torch.nn.Parameter(torch.zeros(4, d))
        if layer_norm:                                    # L2 normun yerine: ortalama cikar, ogrenilen kazanc ve kayma
            self.norm_attention = torch.nn.LayerNorm(d)
            self.norm_facts = torch.nn.LayerNorm(d)

    def forward(self, h, cache=None, alpha_attention=None, alpha_facts=None, turn_facts=None, skip_facts=False):
        """alpha_attention, alpha_facts (d,): normalized_update'te bu turun alpha'lari.  turn_facts: bu turun FactUnits'i
        (SHARED_FACTS=False'ta ikinci gelis); None: Block'un kendi FactUnits'i.  skip_facts: FactUnits alt adimi yok
        (FIRST_TURN_FACTS=False'ta tur 1)."""
        at = self.attention
        unit = lambda v: F.normalize(v, dim=-1)
        norm_a, norm_f = (self.norm_attention, self.norm_facts) if self.layer_norm else (unit, unit)
        x = h if self.stream_norm else norm_a(h)             # attention'in okudugu; stream_norm'da h zaten normlu
        if self.canon:                                       # Canon-A: x_t + sum_k w_k * x_(t-k), baslangictan once 0
            T = x.shape[-2]
            full = F.pad(x, (0, 0, 3, 0)) if cache is None else cache.canon_cache(x)   # onceki 3 konum + x
            x = x + sum(self.canon_weights[k] * full[..., 3 - k:3 - k + T, :] for k in range(4))
        added = at(x, cache=cache) @ at.W_context.T                                # W_context · c_t
        # sphere_weights: W_fact_in satirlari birim, h birim -> girdi kosinus (tipik ±1/√d); √d ile esik O(1) olcekte
        units = self.facts if turn_facts is None else turn_facts
        facts = (lambda v: units(v * v.shape[-1] ** 0.5)) if self.sphere_weights else units
        if self.normalized_update:                                                 # u'nun boyu silinir, adimi alpha belirler
            h = unit(h + alpha_attention * (unit(added) - h))                     # h = norm(h + α_A ⊙ (norm(W_context c) - h))
            return h if skip_facts else unit(h + alpha_facts * (unit(facts(h)) - h))   # h = norm(h + α_F ⊙ (norm(olgu(h)) - h))
        if self.stream_norm:
            h = norm_a(h + added)                                                  # h_t = norm(h_t + W_context · c_t)
            return h if skip_facts else norm_f(h + facts(h))                       # h_t = norm(h_t + olgu(h_t))
        h = h + added                                                              # akis normalize edilmez:
        return h if skip_facts else h + facts(norm_f(h))                           # h_t = h_t + olgu(norm(h_t))


class BlockModel(torch.nn.Module):
    """Adim 3: durum (hidden) PL ile baslar, TURNS tur Block; cikis son durumun kendisi (W_next yok).
    normalized_update ve sphere_weights kapaliyken (28 Eylul'e kadarki model) W_context = W_fact_out = 0 baslar, durum PL'de
    kalir: skor = scale <PL_t, PL>."""

    def __init__(self, n, d=D, turns=TURNS, shared=SHARED_BLOCK, layers=LAYERS, learn_points=LEARN_POINTS,
                 anchor=ANCHOR, confidence=CONFIDENCE, t_max=T_MAX, units=FACT_UNITS, seed=POINTS_SEED,
                 stream_norm=STREAM_NORM, layer_norm=LAYER_NORM, rope=ROPE,
                 normalized_update=NORMALIZED_UPDATE, sphere_weights=SPHERE_WEIGHTS, canon=CANON, heads=HEADS,
                 fact_activation=FACT_ACTIVATION, learn_output_scale=LEARN_OUTPUT_SCALE, loss_chunk=LOSS_CHUNK,
                 last_facts_alpha_init=LAST_FACTS_ALPHA_INIT, output_link=OUTPUT_LINK, shared_facts=SHARED_FACTS,
                 input_embedding=INPUT_EMBEDDING, input_bigrams=INPUT_BIGRAMS, first_turn_facts=FIRST_TURN_FACTS,
                 bigram_keys=None, input_embedding_sphere=INPUT_EMBEDDING_SPHERE):
        """bigram_keys (INPUT_BIGRAMS > 0): en sik ikililerin anahtarlari (onceki * n + token), artan sirali, uzunluk
        input_bigrams; None ya da str (config'teki iz): tampon 0'larla kurulur, state_dict'ten dolar."""
        super().__init__()
        assert not normalized_update or (stream_norm and not layer_norm), "normalized_update akis normuyla (L2) calisir"
        assert not (sphere_weights and layer_norm), "sphere_weights LayerNorm'la denenmedi: FactUnits girdisi sqrt(d) kat buyuk"
        self.tokens = TokenPoints(n, d, learn_points, anchor, 100 * seed)          # 100 * seed: BigramModel'deki gibi
        assert not shared or turns % layers == 0, "turns (%d) layers'in (%d) kati olmali: her Block esit sayida tur" % (
            turns, layers)                                # ayri blokta (shared=False) layers yok sayilir: her tura bir Block
        count = layers if shared else turns
        self.blocks = torch.nn.ModuleList(Block(d, t_max, confidence, units, 100 * seed + 10 + 2 * i,
                                                stream_norm=stream_norm, layer_norm=layer_norm, rope=rope,
                                                normalized_update=normalized_update, sphere_weights=sphere_weights,
                                                canon=canon, heads=heads, fact_activation=fact_activation)
                                          for i in range(count))
        self.turns, self.shared, self.stream_norm = turns, shared, stream_norm
        self.layers = layers if shared else len(self.blocks)   # farkli Block sayisi (ayri blokta her tura bir)
        self.shared_facts = shared_facts or not shared or turns == layers   # ayri blokta zaten tur basina
        if not self.shared_facts:                         # ikinci ve sonraki gelisler: tur i >= layers -> extra_facts[i - layers]
            self.extra_facts = torch.nn.ModuleList(       # tohum: ayri blokta tur i'nin FactUnits'ininki
                FactUnits(d, units, 100 * seed + 10 + 2 * i + 1, activation=fact_activation) for i in range(layers, turns))
        self.layer_norm, self.rope, self.heads, self.fact_activation = layer_norm, rope, heads, fact_activation
        self.normalized_update, self.sphere_weights, self.canon = normalized_update, sphere_weights, canon
        if normalized_update or sphere_weights:           # sifir yon normalize edilemez: W_context, W_fact_out rastgele baslar
            g = torch.Generator().manual_seed(100 * seed + 9)
            with torch.no_grad():
                for b in self.blocks:
                    b.attention.W_context.copy_(torch.randn(d, d, generator=g) / d ** 0.5)
                    b.facts.W_fact_out.copy_(torch.randn(d, units, generator=g) / units ** 0.5)
                for f in getattr(self, "extra_facts", ()):
                    f.W_fact_out.copy_(torch.randn(d, units, generator=g) / units ** 0.5)
        if normalized_update:                             # tur basina (paylasilan blokta da): (tur, d)
            self.alpha_attention = torch.nn.Parameter(torch.full((turns, d), ALPHA_INIT))
            self.alpha_facts = torch.nn.Parameter(torch.full((turns, d), ALPHA_INIT))
            with torch.no_grad():                         # son tur: model girdisini tekrar etmesin (LAST_FACTS_ALPHA_INIT)
                self.alpha_facts[-1] = last_facts_alpha_init
        self.first_turn_facts = first_turn_facts
        if not first_turn_facts and not (shared and self.shared_facts and turns > self.layers):
            self.blocks[0].facts = None                   # tur 1'in takimini baska tur kullanmiyor: parametresi de yok
        if input_embedding:                               # PF'den: ilk adimda girdi PL ile ayni
            self.input_embedding = torch.nn.Parameter(self.tokens.fixed_points.detach().clone())
        self.input_embedding_sphere = bool(input_embedding and input_embedding_sphere)
        assert not self.input_embedding_sphere or sphere_weights, "input_embedding_sphere normalize_weights'le (sphere_weights)"
        if input_bigrams:                                 # 0'dan: ilk adimda ikili katkisi yok
            self.input_bigrams = torch.nn.Parameter(torch.zeros(input_bigrams, d))
            keys = (torch.zeros(input_bigrams, dtype=torch.long) if bigram_keys is None or isinstance(bigram_keys, str)
                    else torch.as_tensor(bigram_keys, dtype=torch.long).clone())
            assert keys.shape == (input_bigrams,) and bool((keys[1:] >= keys[:-1]).all()), "bigram_keys: artan, input_bigrams uzun"
            self.register_buffer("bigram_keys", keys)
        if sphere_weights:
            self.normalize_weights()
        if layer_norm:                                    # cikista: keskinligi kazanc ogrenir, sabit scale kullanilmaz
            self.norm_final = torch.nn.LayerNorm(d)
        self.scale = scale_for(n, confidence)
        self.learn_output_scale = learn_output_scale and not layer_norm   # LayerNorm'da cikis olcegi yok
        if self.learn_output_scale:                       # tau = ln(scale): baslangicta sabit scale ile bit duzeyinde ayni
            self.log_output_scale = torch.nn.Parameter(torch.tensor(math.log(self.scale)))
        self.output_link = output_link and not layer_norm       # LayerNorm'da cikis olcegi ve bag yok
        if self.output_link:                              # q = u = 0: dogrusal skor
            self.link_q = torch.nn.Parameter(torch.zeros(()))
            self.link_u = torch.nn.Parameter(torch.zeros(()))
        assert loss_chunk >= 0, "loss_chunk: 0 (tek parca) ya da parca boyu"
        self.loss_chunk = loss_chunk

    def turn_blocks(self):
        return [self.blocks[i % len(self.blocks)] for i in range(self.turns)]     # paylasilan: A B A B; ayri: her tura biri

    def input_states(self, ids, last=None):
        """h0 (B, T, d): PL[ids]; INPUT_EMBEDDING'de norm(E[ids] + ikili satiri).  Ikili (onceki, token): onceki konum 0'da
        last (B,) ya da yok (-1); bigram_keys'te yoksa katki 0."""
        E = getattr(self, "input_embedding", None)
        B2 = getattr(self, "input_bigrams", None)
        if E is None and B2 is None:
            return self.tokens.points()[ids]
        x = E[ids] if E is not None else self.tokens.points()[ids]
        if B2 is not None:
            first = (last if last is not None else torch.full_like(ids[:, 0], -1))[:, None]
            prev = torch.cat([first, ids[:, :-1]], 1)
            key = prev * self.tokens.fixed_points.shape[0] + ids              # onceki yoksa negatif: listede yok
            pos = torch.searchsorted(self.bigram_keys, key).clamp(max=self.bigram_keys.shape[0] - 1)
            hit = (self.bigram_keys[pos] == key) & (prev >= 0)
            x = x + B2[pos] * hit[..., None].to(B2.dtype)
        return F.normalize(x, dim=-1)

    def hidden(self, ids, caches=None):
        """Her turdan sonraki durumlar: [h0, h1, ..., h_TURNS], her biri (B, T, d); h0 = input_states.  caches: tur basina
        bir AttentionCache (onbellekli uretim; ids yalniz yeni token'lar)."""
        last = None
        if caches is not None and getattr(self, "input_bigrams", None) is not None:
            c0 = caches[0]
            last = c0.last_token                              # istemde None: istemin icindeki onceki token'lar
            if last is None:                                  # istem sagdan dolgulu: satirin son gercek token'i
                n = c0.next_position
                c0.last_token = torch.where(n > 0, ids.gather(1, (n - 1).clamp(min=0)[:, None])[:, 0], -1)
            else:
                c0.last_token = ids[:, -1]
        h = self.input_states(ids, last)
        out = [h]
        for i, block in enumerate(self.turn_blocks()):
            alphas = (dict(alpha_attention=self.alpha_attention[i], alpha_facts=self.alpha_facts[i])
                      if self.normalized_update else {})
            if not self.shared_facts and i >= self.layers:
                alphas["turn_facts"] = self.extra_facts[i - self.layers]
            if i == 0 and not self.first_turn_facts:
                alphas["skip_facts"] = True
            h = block(h, None if caches is None else caches[i], **alphas)
            out.append(h)
        return out

    @torch.no_grad()
    def normalize_weights(self):
        """sphere_weights: girdisi durum olan matrislerin satirlari (W_query, W_key, W_fact_in, W_value), duruma yazanlarin
        sutunlari (W_context, W_fact_out) ve input_embedding_sphere'de girdi tablosunun satirlari birim boya; baslangicta ve
        her optimizer adimindan sonra (train_seq)."""
        if getattr(self, "input_embedding_sphere", False):
            self.input_embedding.copy_(F.normalize(self.input_embedding, dim=1))
        for b in self.blocks:
            at, f = b.attention, b.facts                  # f None: FIRST_TURN_FACTS=False'ta tur 1'in takimi yok
            for w in ((at.W_query, at.W_key) + ((f.W_fact_in,) if f is not None else ()) + ((at.W_value,) if at.heads > 1 else ())
                      + ((f.W_fact_up,) if f is not None and f.activation != "relu" else ())):
                w.copy_(F.normalize(w, dim=1))
            for w in (at.W_context,) + ((f.W_fact_out,) if f is not None else ()):
                w.copy_(F.normalize(w, dim=0))
        for f in getattr(self, "extra_facts", ()):
            for w in (f.W_fact_in,) + ((f.W_fact_up,) if f.activation != "relu" else ()):
                w.copy_(F.normalize(w, dim=1))
            f.W_fact_out.copy_(F.normalize(f.W_fact_out, dim=0))

    def logits(self, ids, caches=None):
        P = self.tokens.points()
        h = self.hidden(ids, caches)[-1]                       # son durum; stream_norm'da zaten kurede
        if self.layer_norm:
            return self.norm_final(h) @ P.T                    # skor_tj = <LN(h_t), PL_j>
        if not self.stream_norm:
            h = F.normalize(h, dim=-1)                         # akis normalize edilmediyse cikista bir kez
        # e^tau = scale · e^(tau - ln scale): baslangicta us tam 0
        scale = self.scale * torch.exp(self.log_output_scale - math.log(self.scale)) if self.learn_output_scale else self.scale
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

    def loss(self, ids, mask):
        if self.loss_chunk:                                    # parcali: (N x V) tablosu yok, gradyan ileri hesapta
            P = self.tokens.points()
            h = self.hidden(ids[:, :-1])[-1].flatten(0, -2)    # (N, d), N = B (T - 1); cikis logits'teki gibi
            if self.layer_norm:
                h, scale = self.norm_final(h), 1.0
            else:
                if not self.stream_norm:
                    h = F.normalize(h, dim=-1)
                scale = (self.scale * torch.exp(self.log_output_scale - math.log(self.scale)) if self.learn_output_scale
                         else self.scale)
            # z_ij = s <h_i, P_j>; once butun parcalardan lse_i, sonra p_ij = e^(z_ij - lse_i) ile gradyanlar:
            #   dL/dh_i = s w_i (sum_j p_ij P_j - P_y)   dL/dP_j = s sum_i w_i (p_ij - [y_i = j]) h_i
            #   dL/ds = sum_i w_i (sum_j p_ij <h_i, P_j> - <h_i, P_y>)        w_i = maske / gecerli hedef sayisi
            with torch.no_grad():
                work = torch.promote_types(h.dtype, torch.float32)   # lse ve birikimler en az fp32 (autocast'te carpim bf16)
                hd, Pd = h.detach(), P.detach()
                s = scale.detach() if torch.is_tensor(scale) else scale
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
            if torch.is_tensor(scale) and scale.requires_grad:
                nll = nll + grad_s * (scale - scale.detach())
            if self.output_link:
                nll = nll + grad_q * (self.link_q - self.link_q.detach()) + grad_u * (self.link_u - self.link_u.detach())
            return nll + self.tokens.anchor_loss(), nll
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
