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
               C_M_NORM C_m once kureye iner: puan = 2 e^S_p cos(C_m, P) + sabit, guveni yalniz S_p tasir.
               karesiz  kaybin en iyi yeri C_m = P (ucgen esitsizligi).
             squared ve S_p sozluk sayisindan: SCORE_BY_VOCAB.
    C_cache  hikayenin KENDI gecmisi: her konumda (C_j, nxt_j = w_(j+1)).  Simdiki C_t'ye
             en yakin CACHE_TOPK C_j (son CACHE_SKIP haric), agirliklari ile p_cache.
               sim_j = cos(C_t, C_j)    W_c = softmax(e^S_c * sim)    p_cache(k) = sum W_c [nxt_j = k]
             gate = sigmoid(gate_d . C_m/|C_m| + gate_s * en_yakin_sim + gate_0), ogrenilir:
               p = (1 - gate) * softmax(score) + gate * p_cache      (pointer-generator)
             Kagit ustu (REL07 t2000, sabit g 0,1): ppl 30,3 -> 25,9; gecmis ismi getirme %3,9 -> %16.
    Q        sorgu: defter C_t ile degil Q_t ile aranir.  Q_t = C_t + sum W_q (finish - start),
             kendi ok sozlugu (VectorLayer); oklar C_m'ye (QUERY_BY) en yakin start'lardan.
             finish = start baslar: Q_t = C_t, bugunku defter.  gate'in sim'i de Q'nun aramasindan.
    C_content  C'nin son d_content boyutu KAYDIRMASIZ, kelimeye ve boyuta gore solar:
               C_content_t = lam_w[w_t] * C_content_(t-1) + beta_w[w_t] * P_content[w_t]
             ilk d_order boyut C_order: bugunku relative zincir (sira), kendi icinde kaydirilir.
             Nokta uzayi her zaman D_SUM = d_order + d_content; C_CONTENT kapaliyken hepsi relative.
             Kaydirmali zincirde boyutlu solma icerigi izleyemez (icerik her adim boyut degistirir).
    Attention  Gecmisten, sozluk katmani ATTN_AFTER'den sonra:
               q_t = W_q norm(C_m,t)   k_j = W_k norm(C_(j-1))   v_j = W_v norm(C_m,j)  ("point": P[w_j])
               C_m += W_o softmax_nedensel(q.k / sqrt(ATTN_DIM)) v.  Anahtar bir ONCEKI konumun zinciri:
               eslesme baglam-baglam, getirilen j'deki kelime (tek katmanda induction).  Konum zincirde (NoPE).
               W_o = 0 baslar: adim 0'da model attention'siz ile ayni.
    CM       countermarch, geriye yuruyus: C_(t-1) = (C_t - RM_t*P[w_t]) / lam
             (relative: C_(t-1) = geri_kaydir((C_t - P[w_t]) / lam)).
             Izleme araci; modelin hesabina girmez.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# TEPE = EN IYI OLCULEN YAPI (grup 3'ten sonra, 25 Eylul).  Kullanici, 25 Eylul: "grup 4 için ben
# belirledim ama en iyi ayarlara getirilmesi gerekiyor".  Kullanici, 24 Eylul: "farklı değerler
# kullandığımızda bu ayrı bir yere yazarız" -- her kosunun tepeden farkli degeri defterdeki KOSU satirinda.
# Onceki tepe (hepsi aktif + Q + butun gecmis; hic kosulmadi): belge/bulgu/model_18_hakem.md §6.
#
# Nokta uzayinin boyutu (dimension) D_SUM = D_ORDER + D_CONTENT; P, C, vektorler ve Q hep D_SUM.
# Kullanici, 24 Eylul: "order 512 ve content 512 diye değerlere karar verdik ama sabit değil
# değişken. D_sum da bunların toplamı".
C_CONTENT = True    # C = [C_order | C_content].  False: D_SUM'in tamami relative (sira)
D_ORDER = 512       # C_order'in boyutu: kaydirmali, sira
D_CONTENT = 512     # C_content'in boyutu: kaydirmasiz, lam_w/beta_w ile solar (C_CONTENT acikken)
D_SUM = D_ORDER + D_CONTENT
VECTORS = 256  # layer basina vektor.  Kullanici: "V sayısı da 256 şimdilik"
# Her C'de aktif vektor (top-k); ACTIVE = VECTORS: hepsi, softmax(-D^2 e^S_v).  Olculen hep 8; hepsi aktif
# yalniz uzaklik seciminde ve dengesiz kostu (DENSE).  Hepsi aktif + yon + denge KOSULMADI -- kullanici,
# 24 Eylul: "bence hepsinden geçsin".
ACTIVE = 8
# Aktif vektorler nasil secilir.  "distance": en yakin start.  "direction": katman 1'den itibaren birim boyda
# en yakin ACI (katman 0 uzaklikla).  "direction_all": katman 0 da yone gore.
# Olculdu (DIR_ALL 4.000, ATTN'e karsi): +0,4 puan, olu vektor 21 -> 0 (hakem notu 1.3'un kurali tuttu).
SELECT = "direction_all"
LAYERS = 4     # Kullanici: "4 katman olsun"
T_MAX = 512    # RM sayisi = en uzun dizi (TinyStories hikayelerinin %98,6'si sigar)
# start'in baslangic boyu.  None: randn, boy ~sqrt(D_SUM) (MAT_* ve TS_PV_D1024 boyle kostu).
# Olculdu (TS_PV_D1024, 24 Eylul): boy ~32 iken |C| ~10; secimi start'larin KENDI boyu
# belirliyordu -- secilenlerin %62'si (egitimde %82) en kisa 8 start, baglamdan bagimsiz.
# Kullanici: "Startları küçük başlat".
START_NORM = 1.0
# Zincirin solma carpani.  1: hic solmaz (lam gelmeden once kosan her sey).
# Hesap (egitimsiz, 48 hikaye, 24 Eylul): lam 1'de 100. kelimede cos(C_t, C_t+1) 0,995 ve
# ardisik secimde 8 vektorun 7,2'si ayni -- solma sart.  0,7'nin gerekcesi relative zincirin kNN hesabi
# (CHAIN); relative'de baska lam egitilmedi.  Kullanici: "λ da bizim için alsında Lr gibi birşey".
LAM = 0.7
# Sira nasil kodlanir.  "absolute": RM_t mutlak konuma bagli (onceki butun kosular).
# "relative": her adimda eskiler bir kaydirilir; kaydirma sayisi kelimenin yasi.
# Hesap (egitimsiz kNN, 1.500 hikaye depo, d 256, 24 Eylul): absolute lam 0,9 %13,3
# (egitilmis LAM09 %14,8), relative lam 0,7 %34,8; hikayenin ortasinda %8,0 -> %31,0.
CHAIN = "relative"
# Hikayenin kendi gecmisinden kopya (CCache) + gate.  Olculdu (NOCACHE 4.000): kapaliyken -0,25 puan,
# geri cagirmada (acc_ar) -2,6 puan.
C_CACHE = True
CACHE_TOPK = 8      # defterden kac komsu.  None (butun gecmis) KOSULMADI
# Son kac konum aranmaz.  Gerekce lam 1 zincirinden (ardisik C'ler cos 0,995); relative'de ardisik
# C_order'lar neredeyse dik, C_content yakinligi geri getiriyor.  Deger taranmadi (yalniz 20: CCSKIP20).
CACHE_SKIP = 3
# Ogrenilen parametrelerin BASLANGIC degerleri.  Kullanici, 24 Eylul: "tepeye al, koşu ayarı olsunlar".
S_V_INIT = 0.0      # vektor keskinligi S_v: kullanilan e^S_v, 1'den baslar
S_C_INIT = 3.0      # defter keskinligi S_c: e^3 = 20, kagit ustu testteki carpan
GATE_0_INIT = -2.0  # gate'in baslangici: sigmoid(-2) = 0,12; kagit ustu en iyi sabit gate 0,1
S_P_INIT = 0.0      # ogrenilen S_p (SCORE_BY_VOCAB LEARNED): e^0 = 1'den baslar
# C_m'yi kelimelerin KURESINE indir: puan = -e^S_p |C_m/|C_m| - P|^2 = 2 e^S_p cos(C_m, P) + sabit.
# Guven |C_m| buyutulerek degil yalniz S_p ile ifade edilir; cevap kurede en yakin sabit nokta.
# Olculdu (CCACHE t10000): |C| 1,4 -> |C_m| 17, e^S_p ~2,8 -- model noktalardan UZAKLASARAK emin oluyor;
# bu boy ara katmanlarda secimi eziyordu ve son katman ortak "guven hareketi" ogreniyordu.
# Hoffer, Hubara, Soudry 2018 (1801.04540): sabit sinif noktalari + birim kure + ogrenilen tek olcek.
# Kullanici, 24 Eylul: "C_M_NORM".  Olculdu (CMNORM 4.000): +2,0 puan (0,4450 -> 0,4649).
C_M_NORM = True
# C_M_NORM acikken S_p'nin baslangici FORMULDEN (sozluk degisince kendisi ayarlanir): kusursuz eslesmede
# (cos 1, digerleri ~0) dogru kelimeye C_M_NORM_P olasilik:  e^S_p = ln(p/(1-p) * (n-1)) / 2.
# n 4003 -> S_p 1,66.  Olcek sabit 2,8 kalsaydi kayip tabani ppl 15,8 olurdu (NormFace sinirindan hesap).
# Kullanici, 24 Eylul: "bu belirlediğimiz sayı sonra başımıza bela olmasın" ve "evet formülle uygula".
C_M_NORM_P = 0.9
# Load balancing (Shazeer ve ark. 2017 "importance"; Switch Transformer'in denge kaybinin akrabasi):
# kayba LOAD_BALANCE * sum_katman vectors * sum_a Pbar_a^2.  Farkli C'lerin farkli vektor kullanmasini
# ister; butun vektorler uzerinden P oldugu icin top-k'da hic secilmeyen start'lar da gradyan alir.
# Olculdu (t2000, secim sagligi): top-8 derin katmanlarda butun C'ler 4-26 vektor; hepsi aktif +
# uzaklik 2-21 (start'lar merkeze yapisiyor).  0,01 MoE'nin standart degeri, bizde OLCULMEDI.
# Kullanici, 24 Eylul: "load balancing ekle, select ile birlikte koş".
LOAD_BALANCE = 0.01
# Q: defteri arayan sorgu.  Kullanici, 24 Eylul: "query (Q_t)", "64 ok, 8 aktif".  Q ile KOSULMADI.
QUERY = False       # defteri Q ile ara.  False: Q = C
QUERY_VECTORS = 64  # QUERY acikken: Q'yu tasiyan ok sayisi
QUERY_ACTIVE = QUERY_VECTORS   # QUERY acikken oklarin hepsi aktif
QUERY_BY = "C_m"    # oklar neye en yakin secilir: "C_m" (modelin dusuncesi) ya da "C".  OLCULMEDI.
# C_content.  Kullanici, 24 Eylul: "lam_w mantıklı beta_w mantıklı tam boyut olsun".
LAM_W_INIT = 0.9    # lam_w'nin baslangici, her token her boyut.  OLCULMEDI.
BETA_W_INIT = 0.5   # beta_w'nin baslangici.  OLCULMEDI.
# lam_w / beta_w kelime x boyut (4.099.072 parametre, modelin %57'si) ya da kelime basina TEK deger (True).
# Incelik hic kiyaslanmadi (hakem notu 1.8).
CONTENT_SCALAR = False
LAM_W_MIN = math.exp(-5)  # lam_w'nin alt siniri: parca hesabi fp32'de tasmasin (16 adim x 5 -> e^80)
CONTENT_CHUNK = 16
# Attention (Gecmisten): defterin C'leri icerikle secilir, getirilen C_m'ye EKLENIR (cikista karismaz).
# Zoology (2312.04927): attention'siz modellerin ppl farkinin %82'si baglamda gecmis ikiliyi tamamlayan
# token'larda.  Kullanici, 25 Eylul: "önceliğimiz attention tasarımı", "state ve point ok".
ATTENTION = True       # olculdu (ATTN 4.000, CMNORM'a karsi): +2,7 puan (0,4649 -> 0,4920)
ATTN_HEADS = 4         # bas sayisi.  OLCULMEDI.
ATTN_DIM = 64          # bas basina boyut; skor olcegi 1/sqrt(ATTN_DIM)
ATTN_AFTER = 0         # sozluk katmani ATTN_AFTER'den sonra: sorgu bir katmandan gecmis, getirileni sonrakiler isler
ATTN_VALUE = "state"   # getirilen: "state" C_m,j (kelime + baglam) | "point" P[w_j] (kelimenin sabit noktasi)
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


def entropy(p):
    """Son eksende entropi (nat); 0 log 0 = 0."""
    return -(p * p.clamp_min(1e-30).log()).sum(-1)


def directions90(X, rows=4096):
    """X (N,d): ortalanmis X'in varyansinin %90'ini tasiyan yon sayisi (ilk `rows` satir)."""
    X = X[:rows].float()
    s2 = torch.linalg.svdvals(X - X.mean(0)) ** 2
    return int((s2.cumsum(0) / s2.sum() < 0.9).sum()) + 1


class VectorLayer(nn.Module):
    """Bir layer: `vectors` tane (start, finish).  C'ye en yakin `active`
    tane start AKTIF; C, aktif vektorlerin agirlikli ortalamasiyla tasinir."""

    def __init__(self, d, vectors, active, randn, start_norm=None, s_v_init=S_V_INIT, direction=False):
        super().__init__()
        self.direction = direction                                # secim yalniz yone bakar (SELECT)
        start = randn(vectors, d)
        if start_norm is not None:
            start = start * (start_norm / d ** 0.5)               # boy ~start_norm: secimi C belirler
        self.start = nn.Parameter(start)
        self.finish = nn.Parameter(self.start.detach().clone())   # V_a = 0: baslangicta C yerinde kalir
        self.S_v = nn.Parameter(torch.tensor(float(s_v_init)))     # log olcek: kullanilan e^S_v > 0
        self.active = active

    def forward(self, C, by=None, probs=False):
        """C (..., d) -> (C_m, active_ids (..., active); hepsi aktifse None).  by: aktifleri secen nokta (None: C).
        probs: ucuncu cikti P (..., vectors) = softmax(-D^2 e^S_v) BUTUN vektorler uzerinde (LOAD_BALANCE).
        P'de S_v SABIT (detach): denge terimi sicakligi dusurup (P'yi duzlestirip) kucultulemez, yalniz
        start'lari oynatarak.  Olculdu (t4000): SELECT_LB'de derin katmanlarin e^S_v'si SELECT'ten dusuktu."""
        D2 = self._distance(C, by)
        P = (torch.softmax(-D2 * self.S_v.exp().detach(), -1),) if probs else ()
        W, active_ids = self._weights(D2)
        return (C + W @ (self.finish - self.start), active_ids) + P

    def weights(self, C, by=None):
        """(..., vectors): forward'in kullandigi secim agirliklari, secilmeyen 0 -- saglik icin."""
        return self._weights(self._distance(C, by))[0]

    def _distance(self, C, by):
        by = C if by is None else by
        if self.direction:                                        # 2 - 2 cos: boy secimi ezmez
            return distance(F.normalize(by, dim=-1), F.normalize(self.start, dim=-1))
        return distance(by, self.start)

    def _weights(self, D2):
        """-> (W (..., vectors), active_ids): secilmeyen 0; hepsi aktifse ids None."""
        if self.active >= self.start.shape[0]:                    # hepsi: tek matris carpimi
            return torch.softmax(-D2 * self.S_v.exp(), -1), None
        D_s2, active_ids = D2.topk(self.active, dim=-1, largest=False)
        W_v = torch.softmax(-D_s2 * self.S_v.exp(), -1)          # payda: aktiflerin toplami 1
        # Aktif agirliklar vectors'luk satira (digerleri 0), tek matris carpimi: (..., active, d) kopyasi yok.
        return torch.zeros_like(D2).scatter(-1, active_ids, W_v), active_ids


class CCache(nn.Module):
    """C_cache ve gate.  Kullanici, 24 Eylul: "c_cche ve gate olsun"."""

    def __init__(self, d, topk=CACHE_TOPK, skip=CACHE_SKIP, s_c_init=S_C_INIT,
                 gate_0_init=GATE_0_INIT, query=QUERY, query_vectors=QUERY_VECTORS,
                 query_active=QUERY_ACTIVE, query_by=QUERY_BY, randn=None, start_norm=None,
                 select=SELECT):
        super().__init__()
        self.topk, self.skip = topk, skip
        self.S_c = nn.Parameter(torch.tensor(float(s_c_init)))     # benzerlik keskinligi
        self.gate_d = nn.Parameter(torch.zeros(d))          # gate'i o anki durumun YONU soyler
        self.gate_s = nn.Parameter(torch.zeros(()))         # ... ve Q'ya en yakin C_j'nin sim'i
        self.gate_0 = nn.Parameter(torch.tensor(float(gate_0_init)))  # gate'in baslangici
        assert query_by in ("C", "C_m"), query_by
        self.query_by = query_by
        self.query = (VectorLayer(d, query_vectors, min(query_active, query_vectors), randn, start_norm,
                                  direction=select != "distance") if query else None)

    def Q(self, C, C_m, probs=False):
        """Sorgu Q_t (B,T,d): oklar yokken C_t.  probs: + oklarin P'si (LOAD_BALANCE; ok yoksa None)."""
        if self.query is None:
            return (C, None) if probs else C
        Q, _, *P = self.query(C, by=C_m if self.query_by == "C_m" else C, probs=probs)
        return (Q, P[0]) if probs else Q

    def parts(self, tokens, C, C_m, probs=False, rows=None):
        """-> (gate (B,T), W_c (B,T,k), ids (B,T,k)): defterin komsulari ve agirliklari; probs: + Q oklarinin P.
        rows: C_m rows'a indirilmis (R, d) gelir (PV._layers); gate (R,) yalniz orada hesaplanir.
        Nedensel: t, yalniz j < t - skip C_j'lerini ve nxt_j = w_(j+1) <= w_t'yi gorur."""
        B, T = tokens.shape
        Cn = F.normalize(C, dim=-1)
        Q, P_q = self.Q(C, C_m, probs=True) if probs else (self.Q(C, C_m), None)
        Qn = Cn if self.query is None else F.normalize(Q, dim=-1)       # ok yoksa Q = C
        sim_all = Qn @ Cn.transpose(1, 2)                            # (B, t, j): cos(Q_t, C_j)
        pos = torch.arange(T, device=C.device)
        allowed = pos[None, :] < pos[:, None] - self.skip            # j < t - skip
        nxt = torch.roll(tokens, -1, dims=1)                         # nxt_j = w_(j+1): C_j'den sonra gelen kelime
        sim = sim_all.masked_fill(~allowed, -2.0)
        if self.topk is None:                                        # butun gecmis: siralamaya gerek yok
            ids = nxt[:, None, :].expand(B, T, T)
            sim_max = sim.max(-1).values
        else:
            k = max(1, min(self.topk, T))
            sim, j = sim.topk(k, dim=-1)
            ids = nxt.gather(1, j.reshape(B, -1)).reshape(B, T, k)
            sim_max = sim[..., 0]
        valid = sim > -1.5
        W_c = torch.softmax((sim * self.S_c.exp()).masked_fill(~valid, -1e4), -1) * valid
        any_valid = valid.any(-1)
        sim_max = torch.where(any_valid, sim_max, torch.zeros_like(sim_max))
        if rows is not None:
            flat = lambda x: x[:, :-1].reshape(-1)[rows]
            sim_max, any_valid = flat(sim_max), flat(any_valid)
        gate = torch.sigmoid(F.normalize(C_m, dim=-1) @ self.gate_d
                             + self.gate_s * sim_max + self.gate_0) * any_valid
        return (gate, W_c, ids, P_q) if probs else (gate, W_c, ids)

    def forward(self, tokens, C, C_m, scores):
        """-> log p (B,T,n), TAM tablo: olcum ve uretim icin.  p = (1-gate) softmax(scores) + gate p_cache."""
        gate, W_c, ids = self.parts(tokens, C, C_m)
        p_cache = torch.zeros_like(scores).scatter_add_(-1, ids, W_c)
        gate = gate.unsqueeze(-1)
        return torch.logaddexp(torch.log1p(-gate) + torch.log_softmax(scores, -1),
                               torch.log(gate + 1e-12) + torch.log(p_cache + 1e-12))

    def target_logp(self, tokens, C, C_m, score, rows=None, probs=False):
        """-> log p(w_(t+1)) (B,T-1): EGITIM yolu.  Kayip yalniz dogru kelimeyi ister: defter
        tarafi k komsudan hesaplanir, (B,T,n) p_cache ve karisim tablosu KURULMAZ.
        score: C_m -> puan.  rows: (B*(T-1)) duz konumlardan secilenler -> (R,); puan tablosu YALNIZ orada.
        C_m (R, d) gelirse zaten rows'a indirilmistir (PV._layers).
        forward()'un dogru kelimedeki degeriyle ayni (tests: t_ccache, t_speed).  probs: + Q oklarinin P'si."""
        compacted = C_m.dim() == 2
        gate, W_c, ids, *P_q = self.parts(tokens, C, C_m, probs=probs, rows=rows if compacted else None)
        target = tokens[:, 1:]
        p_cache = (W_c[:, :-1] * (ids[:, :-1] == target[..., None])).sum(-1)
        if not compacted:
            gate, C_m = gate[:, :-1], C_m[:, :-1]
        if rows is not None:
            flat = lambda x: x.reshape(-1, *x.shape[2:])[rows]
            target, p_cache = flat(target), flat(p_cache)
            if not compacted:
                gate, C_m = flat(gate), flat(C_m)
        s = score(C_m)
        log_model = s.gather(-1, target[..., None])[..., 0] - s.logsumexp(-1)
        logp = torch.logaddexp(torch.log1p(-gate) + log_model,
                               torch.log(gate + 1e-12) + torch.log(p_cache + 1e-12))
        return (logp, P_q[0]) if probs else logp


class Attention(nn.Module):
    """Gecmisten.  q_t = W_q norm(C_m,t), k_j = W_k norm(C_(j-1)), v_j = W_v norm(C_m,j ya da P[w_j]);
    C_m += W_o softmax_nedensel(q.k / sqrt(dim)) v.  W_o = 0 baslar."""

    def __init__(self, d, heads=ATTN_HEADS, dim=ATTN_DIM, value=ATTN_VALUE):
        super().__init__()
        assert value in ("state", "point"), value
        self.heads, self.dim, self.value = heads, dim, value
        self.W_q = nn.Linear(d, heads * dim, bias=False)
        self.W_k = nn.Linear(d, heads * dim, bias=False)
        self.W_v = nn.Linear(d, heads * dim, bias=False)
        self.W_o = nn.Linear(heads * dim, d, bias=False)
        nn.init.zeros_(self.W_o.weight)

    def forward(self, C, C_m, P_w=None):
        """C (B,T,d) ham zincir, C_m (B,T,d) islenmis nokta, P_w (B,T,d) sabit noktalar ("point") -> yeni C_m.
        Nedensel: t yalniz j <= t'yi gorur.  Konum 0'in anahtari sifir (oncesi yok): bos yuva."""
        B, T, _ = C.shape
        C_mn = F.normalize(C_m, dim=-1)
        q, k = self._qk(C, C_mn)
        v = self._split(self.W_v(C_mn if self.value == "state" else F.normalize(P_w, dim=-1)))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return C_m + self.W_o(out.transpose(1, 2).reshape(B, T, -1))

    def weights(self, C, C_m):
        """(B, heads, T, T): forward'in nedensel dikkat agirliklari -- saglik icin."""
        q, k = self._qk(C, F.normalize(C_m, dim=-1))
        T = q.shape[2]
        upper_mask = torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1)
        return (q @ k.transpose(-1, -2) / self.dim ** 0.5).masked_fill(upper_mask, float("-inf")).softmax(-1)

    def _qk(self, C, C_mn):
        """C_mn = norm(C_m).  k_j = W_k norm(C_(j-1)): izdusumden SONRA kaydirilir (W_k biassiz, konum 0 sifir)."""
        k = self.W_k(F.normalize(C, dim=-1))
        k = torch.cat([torch.zeros_like(k[:, :1]), k[:, :-1]], 1)
        return self._split(self.W_q(C_mn)), self._split(k)

    def _split(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.heads, self.dim).transpose(1, 2)


class PV(nn.Module):
    """NOKTA ve VEKTOR.  P sabit; ogrenilen vektorler, S_v, S_p (LEARNED), defter, lam_w/beta_w ve attention."""
    arch = "pv"

    def __init__(self, n, d_order=D_ORDER, d_content=D_CONTENT, vectors=VECTORS, active=ACTIVE, layers=LAYERS,
                 t_max=T_MAX, seed=0, squared=None, S_p=None, start_norm=START_NORM,
                 lam=LAM, chain=CHAIN, c_cache=C_CACHE, cache_topk=CACHE_TOPK,
                 cache_skip=CACHE_SKIP, s_v_init=S_V_INIT, s_c_init=S_C_INIT,
                 gate_0_init=GATE_0_INIT, s_p_init=None, query=QUERY, query_vectors=QUERY_VECTORS,
                 query_active=QUERY_ACTIVE, query_by=QUERY_BY, c_content=C_CONTENT, select=SELECT,
                 load_balance=LOAD_BALANCE, c_m_norm=C_M_NORM,
                 lam_w_init=LAM_W_INIT, beta_w_init=BETA_W_INIT, content_scalar=CONTENT_SCALAR,
                 attention=ATTENTION, attn_heads=ATTN_HEADS,
                 attn_dim=ATTN_DIM, attn_after=ATTN_AFTER, attn_value=ATTN_VALUE):
        """d_order + d_content = d_sum, nokta uzayinin boyutu.  squared, S_p None: SCORE_BY_VOCAB'dan.  Verilirse tabloyu ezer.
        start_norm: start'larin baslangic boyu (None: randn).  lam: zincirin solma carpani.
        chain: "absolute" (RM_t) ya da "relative" (kaydirma).  c_cache: hikayenin gecmisi + gate.
        query: defteri Q ile ara (query_*).  c_content: C = [C_order | C_content]; C_content
        kaydirmasiz, lam_w/beta_w ile solar (content_scalar: kelime basina tek deger).  Kapaliyken d_sum'in
        tamami relative.
        select: aktif vektor secimi, "distance", "direction" (katman 1'den itibaren) ya da "direction_all".
        c_m_norm: puan C_m/|C_m| ile (kure); s_p_init None: C_M_NORM_P formulunden, kapaliyken S_P_INIT.
        attention: sozluk katmani attn_after'den sonra Gecmisten (attn_heads x attn_dim, attn_value)."""
        super().__init__()
        d_sum = int(d_order) + int(d_content)
        if c_content:
            assert chain == "relative" and d_order > 0 and d_content > 0, (chain, d_order, d_content)
        self.c_content, self.d_order, self.d_content, self.d_sum = bool(c_content), int(d_order), int(d_content), d_sum
        generator = torch.Generator().manual_seed(seed)
        randn = lambda *shape: torch.randn(*shape, generator=generator)
        self.n, self.vectors = n, vectors
        self.active, self.layers, self.t_max = active, layers, t_max
        self.start_norm, self.lam = start_norm, float(lam)
        assert chain in ("absolute", "relative"), chain
        assert select in ("distance", "direction", "direction_all"), select
        self.select, self.load_balance, self.c_m_norm = select, float(load_balance), bool(c_m_norm)
        if s_p_init is None:
            s_p_init = (math.log(math.log(C_M_NORM_P / (1 - C_M_NORM_P) * (n - 1)) / 2) if self.c_m_norm
                        else S_P_INIT)
        self.s_p_init = float(s_p_init)
        self.chain = chain
        rule = score_rule(n)
        self.squared = rule[0] if squared is None else squared
        S_p = rule[1] if S_p is None else S_p
        self.S_p_learned = S_p == LEARNED
        self.S_p = (nn.Parameter(torch.tensor(float(s_p_init))) if self.S_p_learned
                    else float(S_p))
        P = randn(n, d_sum)
        self.register_buffer("P", P / P.norm(dim=-1, keepdim=True))
        RM = torch.randint(0, 2, (t_max, d_sum), generator=generator).float() * 2 - 1
        self.register_buffer("RM", RM)
        self.V = nn.ModuleList(VectorLayer(d_sum, vectors, active, randn, start_norm, s_v_init,
                                           direction=select == "direction_all" or (select == "direction" and i > 0))
                               for i in range(layers))
        self.content_scalar = bool(content_scalar)
        if self.c_content:
            # lam_w, beta_w: token x boyut (ya da token basina tek deger), sigmoid'den once (logit).  Sabit
            # baslangic, randn cekmez.
            lam0 = (lam_w_init - LAM_W_MIN) / (1 - LAM_W_MIN)
            width = 1 if self.content_scalar else self.d_content
            self.lam_w = nn.Parameter(torch.full((n, width), math.log(lam0 / (1 - lam0))))
            self.beta_w = nn.Parameter(torch.full((n, width), math.log(beta_w_init / (1 - beta_w_init))))
        self.cache_topk, self.cache_skip = cache_topk, cache_skip
        self.cache = (CCache(d_sum, cache_topk, cache_skip, s_c_init, gate_0_init, query, query_vectors,
                             query_active, query_by, randn, start_norm, select) if c_cache else None)
        assert 0 <= attn_after < layers, (attn_after, layers)
        self.attn_after = int(attn_after)
        self.attn = Attention(d_sum, attn_heads, attn_dim, attn_value) if attention else None

    def C(self, tokens):
        """tokens (B,T) -> zincir (B,T,d).  Nedensel: C_t yalniz <= t'yi toplar."""
        length = tokens.shape[1]
        assert length <= self.t_max, "dizi RM sayisindan uzun"
        if self.chain == "absolute":
            return self._decayed_sum(self.RM[:length] * self.P[tokens])
        x = self.P[tokens]
        if not self.c_content:
            return self._relative(x)
        d_o = self.d_order
        return torch.cat([self._relative(x[..., :d_o]), self._content(tokens, x[..., d_o:])], -1)

    def _relative(self, x):
        """C_order: C_t = sum_i lam^(t-i) kaydir^(t-i) P[w_i] = kaydir^t( sum_i lam^(t-i) kaydir^-i P[w_i] ).
        kaydir^k x [j] = x[j - k]  (torch.roll(x, k)), x'in kendi boyutu icinde."""
        length, D = x.shape[1], x.shape[-1]
        pos = torch.arange(length, device=x.device)[:, None]
        dim = torch.arange(D, device=x.device)[None, :]
        back = ((dim + pos) % D).expand(x.shape)                 # kaydir^-i: y_i[j] = x_i[j + i]
        z = self._decayed_sum(x.gather(-1, back))
        forward = ((dim - pos) % D).expand(x.shape)              # kaydir^t:  C_t[j] = z_t[j - t]
        return z.gather(-1, forward)

    def lam_beta(self, tokens):
        """-> (lam_w[w], beta_w[w]) (..., d_content): lam (LAM_W_MIN, 1), beta (0, 1)."""
        lam = LAM_W_MIN + (1 - LAM_W_MIN) * torch.sigmoid(self.lam_w[tokens])
        return lam, torch.sigmoid(self.beta_w[tokens])

    def _content(self, tokens, x):
        """C_content_t = lam_w[w_t] * C_content_(t-1) + beta_w[w_t] * x_t, CONTENT_CHUNK'lik parcalarla:
        parca icinde L_t = sum log lam, h_t = e^L_t (h_bas + sum_(i<=t) e^-L_i u_i).  fp32."""
        lam, beta = self.lam_beta(tokens)
        u, log_lam = (beta * x).float(), lam.float().log()
        h, out = u.new_zeros(u.shape[0], u.shape[-1]), []
        for s in range(0, u.shape[1], CONTENT_CHUNK):
            L = log_lam[:, s:s + CONTENT_CHUNK].cumsum(1)
            y = L.exp() * (h[:, None] + ((-L).exp() * u[:, s:s + CONTENT_CHUNK]).cumsum(1))
            out.append(y)
            h = y[:, -1]
        return torch.cat(out, 1).to(x.dtype)

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
        """(C_m, layer basina active_ids) -- trace() bunu okur; scoreboard() ayni hesabi _layers ile yapar."""
        return self._layers(self.C(tokens), tokens=tokens)

    def _layers(self, C, probs=False, tokens=None, rows=None):
        """-> (C_m, layer basina active_ids) ; probs: + layer basina P (LOAD_BALANCE).
        attention: sozluk katmani attn_after'den sonra; tokens yalniz "point" icin.
        rows: (R,) duz konum (B*(T-1)) -- konumlar arasi karisma (attention) bittikten sonraki katmanlar YALNIZ
        orada calisir, dolgu hesaplanmaz: C_m (R, d), o katmanlarin P'si (R, vectors).  Kayip yolu."""
        first = self.attn_after + 1 if self.attn is not None else 0     # buradan sonra her katman konum basina
        compact = lambda x: x[:, :-1].reshape(-1, x.shape[-1])[rows]
        C_m, active_ids, P = C, [], []
        for i, layer in enumerate(self.V):
            if rows is not None and i == first:
                C_m = compact(C_m)
            C_m, layer_ids, *p = layer(C_m, probs=probs)
            active_ids.append(layer_ids)
            P += p
            if self.attn is not None and i == self.attn_after:
                C_m = self.attn(C, C_m, self.P[tokens] if self.attn.value == "point" else None)
        if rows is not None and first == len(self.V):
            C_m = compact(C_m)
        return (C_m, active_ids, P) if probs else (C_m, active_ids)

    def balance(self, P, counted, w=None):
        """Load balancing: katman (ve Q) basina vectors * sum_a Pbar_a^2, Pbar = sayilan konumlarda ortalama P.
        P (B,T,vectors) ya da rows'a indirilmis (R,vectors); w (R,) o satirlardan sayilanlar.
        En kucuk 1: kullanim esit.  Her konum keskin kalabilir; yalniz ORTALAMA dengelenir."""
        def mean_p(p):
            a, x = (counted, p[:, :-1]) if p.dim() == 3 else (w, p)
            a = a.to(p.dtype).unsqueeze(-1)
            return (x * a).sum(tuple(range(x.dim() - 1))) / a.sum()
        return sum(p.shape[-1] * (mean_p(p) ** 2).sum() for p in P)

    def score(self, C_m):
        """-S_p*D^2 (squared) ya da -S_p*D.  Buyuk = yakin.  c_m_norm: C_m once kureye iner."""
        if self.c_m_norm:
            C_m = F.normalize(C_m, dim=-1)
        D2 = distance(C_m, self.P)
        D = D2 if self.squared else (D2.clamp_min(0) + EPS).sqrt()
        return -(self.S_p.exp() if self.S_p_learned else self.S_p) * D

    def scoreboard(self, tokens, targets_mask=None):
        """tokens (B,T) -> (B,T,n): her konumda SONRAKI token'in puani.  Konumlar arasi
        karisma yalniz geriye (nedensel), dolgu yalniz sagda: targets_mask arayuz icin, hesabi degistirmez.
        C_cache varsa donen deger log p (normalize); argmax ve cross_entropy ayni calisir."""
        C = self.C(tokens)
        C_m = self._layers(C, tokens=tokens)[0]
        scores = self.score(C_m)
        return scores if self.cache is None else self.cache(tokens, C, C_m, scores)

    def loss(self, tokens, targets_mask=None, parts=False, rows=None):
        """SONRAKI TOKEN: konum j, j+1'i tahmin eder; targets_mask verilirse
        yalniz orada isaretli hedefler sayilir.  load_balance > 0: + load_balance * balance
        (vektor katmanlari ve Q'nun oklari).  parts: (toplam, NLL) -- gunluk NLL'yi yazar, kosular
        denge teriminden bagimsiz kiyaslanir.
        rows: (R,) duz konum (B*(T-1)) listesi, -1 dolgu -- puan tablosu ve attention'dan sonraki katmanlar
        yalniz orada; sayilan her hedef rows'ta olmali (train.trim).  Kayip rows'suz ile AYNI (tests: t_speed)."""
        counted = (torch.ones_like(tokens[:, 1:], dtype=torch.bool) if targets_mask is None
                   else targets_mask[:, 1:].bool())
        C = self.C(tokens)
        balanced = self.load_balance > 0
        w = counted
        if rows is not None:
            keep, rows = rows >= 0, rows.clamp(min=0)
            w = counted.reshape(-1)[rows] & keep
        # Q'nun oklari C_m'ye gore seciliyorsa defter BUTUN konumlarda C_m ister: o zaman katmanlar indirilmez.
        use_rows = rows is not None and (self.cache is None or self.cache.query is None or self.cache.query_by == "C")
        C_m, _, *P = self._layers(C, probs=balanced, tokens=tokens, rows=rows if use_rows else None)
        P = P[0] if balanced else []
        if self.cache is not None:                        # hizli yol: yalniz dogru kelimenin log p'si
            if balanced:
                logp, P_q = self.cache.target_logp(tokens, C, C_m, self.score, rows=rows, probs=True)
                P = P + ([P_q] if P_q is not None else [])
                nll = -logp
            else:
                nll = -self.cache.target_logp(tokens, C, C_m, self.score, rows=rows)
        else:                                             # son konumun hedefi yok; hedef bir kaydirilmis
            target = tokens[:, 1:].reshape(-1)
            if rows is None:
                C_m = C_m[:, :-1].reshape(-1, C_m.shape[-1])
            else:
                target = target[rows]                     # C_m zaten rows'a indirildi (_layers)
            nll = F.cross_entropy(self.score(C_m), target, reduction="none").reshape(w.shape)
        w = w.to(nll.dtype)
        nll = (nll * w).sum() / w.sum()
        loss = nll + self.load_balance * self.balance(P, counted, w) if balanced else nll
        return (loss, nll) if parts else loss

    def trace(self, tokens):
        """Izlenebilirlik: her konumda her layer'da hangi vektorler aktifti."""
        return self.move(tokens)[1]

    @torch.no_grad()
    def inspect(self, tokens):
        """Yalniz okur -- ara degerler.  -> C (B,T,d), C_m (B,T,d) son nokta (move() ile ayni),
        W [katman] (B,T,vectors) secim agirliklari (secilmeyen 0), attn (B,heads,T,T) ya da None, gate (B,T) ya da None."""
        C = self.C(tokens)
        C_m, W, A = C, [], None
        for i, layer in enumerate(self.V):
            W.append(layer.weights(C_m))
            C_m = layer(C_m)[0]
            if self.attn is not None and i == self.attn_after:
                A = self.attn.weights(C, C_m)
                C_m = self.attn(C, C_m, self.P[tokens] if self.attn.value == "point" else None)
        gate = self.cache.parts(tokens, C, C_m)[0] if self.cache is not None else None
        return {"C": C, "C_m": C_m, "W": W, "attn": A, "gate": gate}

    @torch.no_grad()
    def health(self, tokens, mask):
        """Saglik: sabit bir sondada modelin ici.  tokens (B,T), mask (B,T) gercek konumlar -> sozluk.
          vec_each_l / vec_all_l   katman l'de etkin vektor: her konumda e^H(W_t) ortalamasi / butun konumlarda
                                   e^H(ortalama W).  vec_all kucukse farkli C'ler ayni vektorlere gidiyor (cokus)
          dead_l                   ortalama payi esit payin %1'inden az vektor sayisi (olu)
          dir_C / dir_Cm           varyansin %90'ini tasiyan yon sayisi;  norm_C / norm_Cm  ortalama boy
          exp_S_v, exp_S_p, exp_S_c, gate, lam_c   ogrenilen olcekler, defterin gate'i, icerik solmasi (lam_w)
          attn_ent / attn_first / attn_dist   bas basina: entropi (nat), konum 0'a dusen pay, ortalama bakis mesafesi
          pred_distinct / pred_ent  en yuksek puani alan farkli kelime sayisi, tahminin entropisi (nat)"""
        a = mask.bool()
        ins, h = self.inspect(tokens), {}
        for i, W in enumerate(ins["W"]):
            W_bar = W[a].mean(0)
            h["vec_each_%d" % i] = float(entropy(W[a]).exp().mean())
            h["vec_all_%d" % i] = float(entropy(W_bar).exp())
            h["dead_%d" % i] = int((W_bar < 0.01 / W_bar.numel()).sum())
        for name, X in (("C", ins["C"][a]), ("Cm", ins["C_m"][a])):
            h["norm_" + name], h["dir_" + name] = float(X.norm(dim=-1).mean()), directions90(X)
        h["exp_S_v"] = [float(L.S_v.exp()) for L in self.V]
        if self.S_p_learned:
            h["exp_S_p"] = float(self.S_p.exp())
        if self.c_content:
            h["lam_c"] = float(self.lam_beta(tokens)[0][a].mean())
        if ins["gate"] is not None:
            h["gate"], h["exp_S_c"] = float(ins["gate"][a].mean()), float(self.cache.S_c.exp())
        if ins["attn"] is not None:
            A = ins["attn"]
            pos = torch.arange(A.shape[-1], device=A.device)
            age = (pos[:, None] - pos[None, :]).clamp(min=0).to(A.dtype)
            for name, x in (("attn_ent", entropy(A)), ("attn_first", A[..., 0]), ("attn_dist", (A * age).sum(-1))):
                h[name] = [float(x[:, k][a].mean()) for k in range(A.shape[1])]
        lp = torch.log_softmax(self.scoreboard(tokens), -1)[a]
        h["pred_distinct"] = int(lp.argmax(-1).unique().numel())
        h["pred_ent"] = float(-(lp.exp() * lp).sum(-1).mean())
        return h

    @classmethod
    def from_package(cls, k):
        """Kayitli paketten (t/w) model, agirliklari yuklu.  Alan gelmeden yazilmis pakette o ozellik
        kapali; eski paketin 'd' alani d_sum'dir (hepsi relative)."""
        m = cls(k["n"], d_order=k.get("d_order", k.get("d")), d_content=k.get("d_content", 0),
                vectors=k["vectors"], active=k["active"], layers=k["layers"], t_max=k["t_max"],
                seed=k["seed"], squared=k.get("squared", True), S_p=k.get("S_p", 1.0),
                lam=k.get("lam", 1.0), chain=k.get("chain", "absolute"), c_cache=k.get("c_cache", False),
                cache_topk=k.get("cache_topk", 8), cache_skip=k.get("cache_skip", 3),
                query=k.get("query", False), query_vectors=k.get("query_vectors", QUERY_VECTORS),
                query_active=k.get("query_active", QUERY_ACTIVE), query_by=k.get("query_by", QUERY_BY),
                c_content=k.get("c_content", False), content_scalar=k.get("content_scalar", False),
                select=k.get("select", "distance"),
                load_balance=k.get("load_balance", 0.0), c_m_norm=k.get("c_m_norm", False),
                attention=k.get("attention", False),
                attn_heads=k.get("attn_heads", ATTN_HEADS), attn_dim=k.get("attn_dim", ATTN_DIM),
                attn_after=k.get("attn_after", ATTN_AFTER), attn_value=k.get("attn_value", ATTN_VALUE))
        m.load_state_dict(k["weights"])
        return m.eval()

    def CM(self, tokens, K=None):
        """Countermarch: tek dizi tokens (T,), son halkadan K adim geri (None: basa kadar).
        Her adim: C_(t-1) = (C_t - RM_t*P[tokens_t]) / lam  (relative: geri kaydirilir).
        Doner: path = [(position, C_position)],
        yeniden eskiye.  lam < 1'de uzun geriye yuruyuste sayisal hata 1/lam kadar buyur."""
        length, d_o = tokens.shape[0], self.d_order
        link, path = self.C(tokens[None])[0, -1], []
        for position in range(length - 1, -1 if K is None else max(-1, length - 1 - K), -1):
            path.append((position, link))
            w, p = tokens[position], self.P[tokens[position]]
            if self.chain == "absolute":
                link = (link - self.RM[position] * p) / self.lam
            elif not self.c_content:
                link = torch.roll((link - p) / self.lam, -1)
            else:                                   # C_order geri kaydirilir, C_content lam_w'ye bolunur
                lam, beta = self.lam_beta(w)
                link = torch.cat([torch.roll((link[:d_o] - p[:d_o]) / self.lam, -1),
                                  (link[d_o:] - beta * p[d_o:]) / lam])
        return path
