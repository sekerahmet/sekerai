# -*- coding: utf-8 -*-
"""model_18 -- PV: NOKTA ve VEKTOR.

Kullanici, 23 Eylul: "kelimeler ya da sayılar koordinatta bir nokta,
vektörler ise mimarinin parametresi görevi ise yeni çıkan noktayı cevaba en
yakın noktaya taşımak yani herşey vektör değil. Seçilim ise iki nokta
arasında ki uzaklık" ve "her vektör her an aktif olmamalı bunu model
öğrenmeli yani sözlük gibi bir de vektör sözlüğü hatta vektör katmanı olmalı".

    P        token noktalari.  SABIT.
    RM       rank multiplier, sira carpani (+-1).  SABIT.
    C        zincir: C_t = RM_1*P[w_1] + ... + RM_t*P[w_t]
             yeni token gelince yalniz bir terim eklenir; C_m zincire GIRMEZ.
    V        layer basina vektor sozlugu; her vektor (start, finish), ogrenilir.
    layer    C'ye en yakin ACTIVE tane start AKTIF, gerisi PASIF:
               D_s,a = |C - start_a|                            a. aktif vektorun start'ina uzaklik
               W_v,a = e^(-D_s,a^2 S_v) / sum_B e^(-D_s,B^2 S_v)  a. vektorun agirligi
               V_a   = finish_a - start_a                        a. vektor
               C_m   = C + sum_a W_v,a V_a                       tasinmis C
    score    -D^2 S_p, D = |C_m - P|: en yakin P en yuksek puan.
    CM       countermarch, geriye yuruyus: C_(t-1) = C_t - RM_t*P[w_t].
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


def distance(x, Q):
    """Uzakligin KARESI |x - Q|^2.  x (..., d), Q (m, d) -> (..., m)."""
    return (x * x).sum(-1, keepdim=True) - 2 * x @ Q.T + (Q * Q).sum(-1)


class VectorLayer(nn.Module):
    """Bir layer: `vectors` tane (start, finish).  C'ye en yakin `active`
    tane start AKTIF; C, aktif vektorlerin agirlikli ortalamasiyla tasinir."""

    def __init__(self, d, vectors, active, r):
        super().__init__()
        self.start = nn.Parameter(r(vectors, d))
        self.finish = nn.Parameter(self.start.detach().clone())   # V_a = 0: baslangicta C yerinde kalir
        self.S_v = nn.Parameter(torch.zeros(()))                   # log olcek: kullanilan e^S_v > 0
        self.active = active

    def forward(self, C):
        """C (..., d) -> (C_m, active_ids (..., active))."""
        D_s2, active_ids = distance(C, self.start).topk(self.active, dim=-1, largest=False)
        W_v = torch.softmax(-D_s2 * self.S_v.exp(), -1)          # payda: aktiflerin toplami 1
        V_a = (self.finish - self.start)[active_ids]             # (..., active, d)
        return C + (W_v.unsqueeze(-1) * V_a).sum(-2), active_ids


class PV(nn.Module):
    """NOKTA ve VEKTOR.  Ogrenilen YALNIZ vektorler (start, finish), S_v'ler ve S_p."""
    mimari = "pv"

    def __init__(self, n, d=d, vectors=VECTORS, active=ACTIVE, layers=LAYERS,
                 t_max=T_MAX, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        r = lambda *s: torch.randn(*s, generator=g)
        self.n, self.d, self.vectors = n, d, vectors
        self.active, self.layers, self.t_max = active, layers, t_max
        P = r(n, d)
        self.register_buffer("P", P / P.norm(dim=-1, keepdim=True))
        RM = torch.randint(0, 2, (t_max, d), generator=g).float() * 2 - 1
        self.register_buffer("RM", RM)
        self.V = nn.ModuleList(VectorLayer(d, vectors, active, r) for _ in range(layers))
        self.S_p = nn.Parameter(torch.zeros(()))   # log olcek; en yakin P ondan BAGIMSIZ

    def C(self, w):
        """w (B,T) -> zincir (B,T,d).  Nedensel: C_t yalniz <= t'yi toplar."""
        assert w.shape[1] <= self.t_max, "dizi RM sayisindan uzun"
        return (self.RM[:w.shape[1]] * self.P[w]).cumsum(1)

    def move(self, w):
        """(C_m, layer basina active_ids) -- trace() ve scoreboard() ayni hesabi okur."""
        C_m, ids = self.C(w), []
        for layer in self.V:
            C_m, a = layer(C_m)
            ids.append(a)
        return C_m, ids

    def score(self, C_m):
        """-D^2 S_p, D = |C_m - P|.  Buyuk = yakin."""
        return -distance(C_m, self.P) * self.S_p.exp()

    def scoreboard(self, w, mask=None):
        """w (B,T) -> (B,T,n): her konumda SONRAKI token'in puani.  Konumlar
        arasi karisma yok, dolgu yalniz sagda: mask arayuz icin, hesabi degistirmez."""
        return self.score(self.move(w)[0])

    def loss(self, w, mask=None):
        """SONRAKI TOKEN: konum j, j+1'i tahmin eder; mask verilirse yalniz
        mask'teki hedefler sayilir."""
        p = self.scoreboard(w, mask)[:, :-1].reshape(-1, self.n)
        h = w[:, 1:].reshape(-1)
        if mask is None:
            return F.cross_entropy(p, h)
        m = mask[:, 1:].reshape(-1)
        k = F.cross_entropy(p, h, reduction="none")
        return (k * m).sum() / m.sum()

    def trace(self, w):
        """Izlenebilirlik: her konumda her layer'da hangi vektorler aktifti."""
        return self.move(w)[1]

    def CM(self, w, K=None):
        """Countermarch: tek dizi w (T,), son halkadan K adim geri (None: basa kadar).
        Her adim: C_(t-1) = C_t - RM_t*P[w_t].  Doner: [(t, C_t)], yeniden eskiye."""
        T = w.shape[0]
        c, yol = self.C(w[None])[0, -1], []
        for t in range(T - 1, -1 if K is None else max(-1, T - 1 - K), -1):
            yol.append((t, c))
            c = c - self.RM[t] * self.P[w[t]]
        return yol
