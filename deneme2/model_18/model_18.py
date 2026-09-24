# -*- coding: utf-8 -*-
"""model_18 -- PV: NOKTA ve VEKTOR.

Kullanici, 23 Eylul: "kelimeler ya da sayılar koordinatta bir nokta,
vektörler ise mimarinin parametresi görevi ise yeni çıkan noktayı cevaba en
yakın noktaya taşımak yani herşey vektör değil. Seçilim ise iki nokta
arasında ki uzaklık" ve "her vektör her an aktif olmamalı bunu model
öğrenmeli yani sözlük gibi bir de vektör sözlüğü hatta vektör katmanı olmalı".

    P        token noktalari.  SABIT.
    RM       rank multiplier, sira carpani (+-1).  SABIT.
    C        zincir, chain "absolute": C_t = lam*C_(t-1) + RM_t*P[w_t]    (lam 1: hepsinin toplami)
                     chain "relative": C_t = lam*kaydir(C_(t-1)) + P[w_t]  (sira = kelimenin YASI)
             relative'de ayni baglam hikayenin her yerinde ayni noktaya duser.
             yeni token gelince yalniz bir terim eklenir; C_m zincire GIRMEZ.
             lam < 1: eski terimler solar, |C| sinirli kalir, ardisik C'ler ayrisir.
    V        layer basina vektor sozlugu; her vektor (start, finish), ogrenilir.
    layer    C'ye en yakin ACTIVE tane start AKTIF, gerisi PASIF:
               D_s,a = |C - start_a|                            a. aktif vektorun start'ina uzaklik
               W_v,a = e^(-D_s,a^2 S_v) / sum_B e^(-D_s,B^2 S_v)  a. vektorun agirligi
               V_a   = finish_a - start_a                        a. vektor
               C_m   = C + sum_a W_v,a V_a                       tasinmis C
    score    en yakin P en yuksek puan, D = |C_m - P|:  -S_p*D^2 (squared) ya da -S_p*D.
               kare     |C_m|^2 softmax'ta sadelesir, geriye C_m.P kalir: C_m noktadan
                        UZAKLASARAK emin olur (MAT_COK_PV).
               karesiz  kaybin en iyi yeri C_m = P (ucgen esitsizligi).
             squared ve S_p sozluk sayisindan: SCORE_BY_VOCAB.
    C_cache  hikayenin KENDI gecmisi: her konumda (C_j, nxt_j = w_(j+1)).  Simdiki C_t'ye
             en yakin CACHE_TOPK C_j (son CACHE_SKIP haric), agirliklari ile p_cache.
               sim_j = cos(C_t, C_j)    W_c = softmax(e^S_c * sim)    p_cache(k) = sum W_c [nxt_j = k]
             gate = sigmoid(gate_d . C_m/|C_m| + gate_s * en_yakin_sim + gate_0), ogrenilir:
               p = (1 - gate) * softmax(score) + gate * p_cache      (pointer-generator)
             Kagit ustu (REL07 t2000, sabit g 0,1): ppl 30,3 -> 25,9; gecmis ismi getirme %3,9 -> %16.
    CM       countermarch, geriye yuruyus: C_(t-1) = (C_t - RM_t*P[w_t]) / lam
             (relative: C_(t-1) = geri_kaydir((C_t - P[w_t]) / lam)).
             Izleme araci; modelin hesabina girmez.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# BELIRLENIMCI KOSU (model_17'den; olculmus gerekce orada).
if not torch.cuda.is_available():
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

d = 128        # dimension, nokta uzayi: sabit kodda 13 token'in %89'u okunur (belge/analiz/model_17_MAT/sabit_kod_cok.py)
VECTORS = 256  # layer basina vektor.  Kullanici: "V sayısı da 256 şimdilik"
ACTIVE = 8     # her C'de aktif vektor; gerisi pasif.  OLCULMEDI.
LAYERS = 4     # Kullanici: "4 katman olsun"
T_MAX = 64     # RM sayisi = en uzun dizi
# start'in baslangic boyu.  None: randn, boy ~sqrt(d) (MAT_* ve TS_PV_D1024 boyle kostu).
# Olculdu (TS_PV_D1024, 24 Eylul): boy ~32 iken |C| ~10; secimi start'larin KENDI boyu
# belirliyordu -- secilenlerin %62'si (egitimde %82) en kisa 8 start, baglamdan bagimsiz.
# Kullanici: "Startları küçük başlat".
START_NORM = None
# Zincirin solma carpani.  1: hic solmaz (lam gelmeden once kosan her sey).
# Hesap (egitimsiz, 48 hikaye, 24 Eylul): lam 1'de 100. kelimede cos(C_t, C_t+1) 0,995 ve
# ardisik secimde 8 vektorun 7,2'si ayni; lam 0,9'da cos 0,90 ve 4,7 -- her yerde ayni.
# Kullanici: "λ da bizim için alsında Lr gibi birşey ... şu an 0.9 ile başlayabiliriz".
LAM = 1.0
# Sira nasil kodlanir.  "absolute": RM_t mutlak konuma bagli (onceki butun kosular).
# "relative": her adimda eskiler bir kaydirilir; kaydirma sayisi kelimenin yasi.
# Hesap (egitimsiz kNN, 1.500 hikaye depo, d 256, 24 Eylul): absolute lam 0,9 %13,3
# (egitilmis LAM09 %14,8), relative lam 0,7 %34,8; hikayenin ortasinda %8,0 -> %31,0.
CHAIN = "absolute"
C_CACHE = False     # hikayenin kendi gecmisinden kopya (CCache) + gate
CACHE_TOPK = 8      # defterden kac komsu (kagit ustu testteki gibi); vektorlerdeki ACTIVE'in karsiligi
CACHE_SKIP = 3      # son kac konum aranmaz: en yakin C_j hep bir onceki adim olurdu
# Ogrenilen parametrelerin BASLANGIC degerleri.  Kullanici, 24 Eylul: "tepeye al, koşu ayarı olsunlar".
S_V_INIT = 0.0      # vektor keskinligi S_v: kullanilan e^S_v, 1'den baslar
S_C_INIT = 3.0      # defter keskinligi S_c: e^3 = 20, kagit ustu testteki carpan
GATE_0_INIT = -2.0  # gate'in baslangici: sigmoid(-2) = 0,12; kagit ustu en iyi sabit gate 0,1
S_P_INIT = 0.0      # ogrenilen S_p (SCORE_BY_VOCAB LEARNED): e^0 = 1'den baslar
EPS = 1e-6     # karekok D=0'da turevlenmez
LEARNED = "learned"

# Skor sozluk sayisina (n) gore.  Satir: (n BUNDAN KUCUKSE, squared, S_p).
# Sinirdaki n UST satira girer (tam 50 -> S_p 5).  LEARNED: e^s, s = 0 (1'den) baslar.
# Kullanici, 24 Eylul: "sözlük sayısına göre karar veceğiz ... bu değerleri ben
# değiştirebilirim aralıkları sen ona göre model içinde yaz".
SCORE_BY_VOCAB = (
    (50,           False, 10.0),
    (100,          False, 5.0),
    (500,          False, 2.0),
    (2000,         True,  2.0),
    (float("inf"), True,  LEARNED),
)


def score_rule(n):
    """n token'lik sozluk -> (squared, S_p)."""
    for upper, squared, S_p in SCORE_BY_VOCAB:
        if n < upper:
            return squared, S_p


def distance(points, anchors):
    """Uzakligin KARESI |points - anchors|^2.  points (..., d), anchors (m, d) -> (..., m).
    anchors: karsilastirilan noktalar (start'lar ya da P'ler)."""
    return ((points * points).sum(-1, keepdim=True) - 2 * points @ anchors.T
            + (anchors * anchors).sum(-1))


class VectorLayer(nn.Module):
    """Bir layer: `vectors` tane (start, finish).  C'ye en yakin `active`
    tane start AKTIF; C, aktif vektorlerin agirlikli ortalamasiyla tasinir."""

    def __init__(self, d, vectors, active, randn, start_norm=None, s_v_init=S_V_INIT):
        super().__init__()
        start = randn(vectors, d)
        if start_norm is not None:
            start = start * (start_norm / d ** 0.5)               # boy ~start_norm: secimi C belirler
        self.start = nn.Parameter(start)
        self.finish = nn.Parameter(self.start.detach().clone())   # V_a = 0: baslangicta C yerinde kalir
        self.S_v = nn.Parameter(torch.tensor(float(s_v_init)))     # log olcek: kullanilan e^S_v > 0
        self.active = active

    def forward(self, C):
        """C (..., d) -> (C_m, active_ids (..., active))."""
        D_s2, active_ids = distance(C, self.start).topk(self.active, dim=-1, largest=False)
        W_v = torch.softmax(-D_s2 * self.S_v.exp(), -1)          # payda: aktiflerin toplami 1
        V_a = (self.finish - self.start)[active_ids]             # (..., active, d)
        return C + (W_v.unsqueeze(-1) * V_a).sum(-2), active_ids


class CCache(nn.Module):
    """C_cache ve gate.  Kullanici, 24 Eylul: "c_cche ve gate olsun"."""

    def __init__(self, d, topk=CACHE_TOPK, skip=CACHE_SKIP, s_c_init=S_C_INIT,
                 gate_0_init=GATE_0_INIT):
        super().__init__()
        self.topk, self.skip = topk, skip
        self.S_c = nn.Parameter(torch.tensor(float(s_c_init)))     # benzerlik keskinligi
        self.gate_d = nn.Parameter(torch.zeros(d))          # gate'i o anki durumun YONU soyler
        self.gate_s = nn.Parameter(torch.zeros(()))         # ... ve en yakin C_j'nin sim'i
        self.gate_0 = nn.Parameter(torch.tensor(float(gate_0_init)))  # gate'in baslangici

    def forward(self, tokens, C, C_m, scores):
        """-> log p (B,T,n).  p = (1-gate) softmax(scores) + gate p_cache.  Nedensel: t, yalniz
        j < t - skip C_j'lerini ve onlardan sonra gelen nxt_j = w_(j+1) <= w_t'yi gorur."""
        B, T = tokens.shape
        Cn = F.normalize(C, dim=-1)
        sim_all = Cn @ Cn.transpose(1, 2)                            # (B, t, j): cos(C_t, C_j)
        pos = torch.arange(T, device=C.device)
        allowed = pos[None, :] < pos[:, None] - self.skip            # j < t - skip
        k = max(1, min(self.topk, T))
        sim, j = sim_all.masked_fill(~allowed, -2.0).topk(k, dim=-1)
        valid = sim > -1.5
        W_c = torch.softmax((sim * self.S_c.exp()).masked_fill(~valid, -1e4), -1) * valid
        nxt = torch.roll(tokens, -1, dims=1)                         # nxt_j = w_(j+1): C_j'den sonra gelen kelime
        ids = nxt.gather(1, j.reshape(B, -1)).reshape(B, T, k)
        p_cache = torch.zeros_like(scores).scatter_add_(-1, ids, W_c)
        any_valid = valid.any(-1)
        sim_max = torch.where(any_valid, sim[..., 0], torch.zeros_like(sim[..., 0]))
        gate = torch.sigmoid(F.normalize(C_m, dim=-1) @ self.gate_d
                             + self.gate_s * sim_max + self.gate_0) * any_valid
        gate = gate.unsqueeze(-1)
        return torch.logaddexp(torch.log1p(-gate) + torch.log_softmax(scores, -1),
                               torch.log(gate + 1e-12) + torch.log(p_cache + 1e-12))


class PV(nn.Module):
    """NOKTA ve VEKTOR.  Ogrenilen YALNIZ vektorler (start, finish) ve S_v'ler."""
    arch = "pv"

    def __init__(self, n, d=d, vectors=VECTORS, active=ACTIVE, layers=LAYERS,
                 t_max=T_MAX, seed=0, squared=None, S_p=None, start_norm=START_NORM,
                 lam=LAM, chain=CHAIN, c_cache=C_CACHE, cache_topk=CACHE_TOPK,
                 cache_skip=CACHE_SKIP, s_v_init=S_V_INIT, s_c_init=S_C_INIT,
                 gate_0_init=GATE_0_INIT, s_p_init=S_P_INIT):
        """squared, S_p None: SCORE_BY_VOCAB'dan.  Verilirse tabloyu ezer.
        start_norm: start'larin baslangic boyu (None: randn).  lam: zincirin solma carpani.
        chain: "absolute" (RM_t) ya da "relative" (kaydirma).  c_cache: hikayenin gecmisi + gate."""
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        randn = lambda *shape: torch.randn(*shape, generator=generator)
        self.n, self.d, self.vectors = n, d, vectors
        self.active, self.layers, self.t_max = active, layers, t_max
        self.start_norm, self.lam = start_norm, float(lam)
        assert chain in ("absolute", "relative"), chain
        self.chain = chain
        rule = score_rule(n)
        self.squared = rule[0] if squared is None else squared
        S_p = rule[1] if S_p is None else S_p
        self.S_p_learned = S_p == LEARNED
        self.S_p = (nn.Parameter(torch.tensor(float(s_p_init))) if self.S_p_learned
                    else float(S_p))
        P = randn(n, d)
        self.register_buffer("P", P / P.norm(dim=-1, keepdim=True))
        RM = torch.randint(0, 2, (t_max, d), generator=generator).float() * 2 - 1
        self.register_buffer("RM", RM)
        self.V = nn.ModuleList(VectorLayer(d, vectors, active, randn, start_norm, s_v_init)
                               for _ in range(layers))
        self.cache_topk, self.cache_skip = cache_topk, cache_skip
        self.cache = (CCache(d, cache_topk, cache_skip, s_c_init, gate_0_init) if c_cache
                      else None)

    def C(self, tokens):
        """tokens (B,T) -> zincir (B,T,d).  Nedensel: C_t yalniz <= t'yi toplar."""
        length = tokens.shape[1]
        assert length <= self.t_max, "dizi RM sayisindan uzun"
        if self.chain == "absolute":
            return self._decayed_sum(self.RM[:length] * self.P[tokens])
        # relative: C_t = sum_i lam^(t-i) kaydir^(t-i) P[w_i] = kaydir^t( sum_i lam^(t-i) kaydir^-i P[w_i] ).
        # kaydir^k x [j] = x[j - k]  (torch.roll(x, k)).
        x = self.P[tokens]
        pos = torch.arange(length, device=x.device)[:, None]
        dim = torch.arange(self.d, device=x.device)[None, :]
        back = ((dim + pos) % self.d).expand(x.shape)            # kaydir^-i: y_i[j] = x_i[j + i]
        z = self._decayed_sum(x.gather(-1, back))
        forward = ((dim - pos) % self.d).expand(x.shape)         # kaydir^t:  C_t[j] = z_t[j - t]
        return z.gather(-1, forward)

    def _decayed_sum(self, terms):
        """sum_(i<=t) lam^(t-i) terms_i.  lam^-t ile olcekli cumsum 512'de tasar (0,7^-512);
        (T,T) alt ucgen carpan matrisi kesin ve kararli."""
        if self.lam == 1.0:
            return terms.cumsum(1)
        age = torch.arange(terms.shape[1], device=terms.device)
        age = age[:, None] - age[None, :]
        decay = torch.where(age >= 0, self.lam ** age.clamp(min=0).float(), torch.zeros(()))
        return torch.einsum("ti,bid->btd", decay.to(terms.dtype), terms)

    def move(self, tokens):
        """(C_m, layer basina active_ids) -- trace() ve scoreboard() ayni hesabi okur."""
        return self._layers(self.C(tokens))

    def _layers(self, C):
        C_m, active_ids = C, []
        for layer in self.V:
            C_m, layer_ids = layer(C_m)
            active_ids.append(layer_ids)
        return C_m, active_ids

    def score(self, C_m):
        """-S_p*D^2 (squared) ya da -S_p*D.  Buyuk = yakin."""
        D2 = distance(C_m, self.P)
        D = D2 if self.squared else (D2.clamp_min(0) + EPS).sqrt()
        return -(self.S_p.exp() if self.S_p_learned else self.S_p) * D

    def scoreboard(self, tokens, targets_mask=None):
        """tokens (B,T) -> (B,T,n): her konumda SONRAKI token'in puani.  Konumlar
        arasi karisma yok, dolgu yalniz sagda: targets_mask arayuz icin, hesabi degistirmez.
        C_cache varsa donen deger log p (normalize); argmax ve cross_entropy ayni calisir."""
        C = self.C(tokens)
        C_m = self._layers(C)[0]
        scores = self.score(C_m)
        return scores if self.cache is None else self.cache(tokens, C, C_m, scores)

    def loss(self, tokens, targets_mask=None):
        """SONRAKI TOKEN: konum j, j+1'i tahmin eder; targets_mask verilirse
        yalniz orada isaretli hedefler sayilir."""
        scores = self.scoreboard(tokens, targets_mask)[:, :-1].reshape(-1, self.n)   # son konumun hedefi yok
        targets = tokens[:, 1:].reshape(-1)                                            # bir kaydirilmis: gercek sonraki token
        if targets_mask is None:
            return F.cross_entropy(scores, targets)
        counted = targets_mask[:, 1:].reshape(-1)
        losses = F.cross_entropy(scores, targets, reduction="none")
        return (losses * counted).sum() / counted.sum()

    def trace(self, tokens):
        """Izlenebilirlik: her konumda her layer'da hangi vektorler aktifti."""
        return self.move(tokens)[1]

    def CM(self, tokens, K=None):
        """Countermarch: tek dizi tokens (T,), son halkadan K adim geri (None: basa kadar).
        Her adim: C_(t-1) = (C_t - RM_t*P[tokens_t]) / lam  (relative: geri kaydirilir).
        Doner: path = [(position, C_position)],
        yeniden eskiye.  lam < 1'de uzun geriye yuruyuste sayisal hata 1/lam kadar buyur."""
        length = tokens.shape[0]
        link, path = self.C(tokens[None])[0, -1], []
        for position in range(length - 1, -1 if K is None else max(-1, length - 1 - K), -1):
            path.append((position, link))
            if self.chain == "absolute":
                link = (link - self.RM[position] * self.P[tokens[position]]) / self.lam
            else:
                link = torch.roll((link - self.P[tokens[position]]) / self.lam, -1)
        return path
