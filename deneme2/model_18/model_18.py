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
    score    -D^2, D = |C_m - P|: en yakin P en yuksek puan.  Carpan YOK:
             Kullanici, 24 Eylul: "S_p kaldir" -- secimi degistirmiyordu.
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


def distance(points, anchors):
    """Uzakligin KARESI |points - anchors|^2.  points (..., d), anchors (m, d) -> (..., m).
    anchors: karsilastirilan noktalar (start'lar ya da P'ler)."""
    return ((points * points).sum(-1, keepdim=True) - 2 * points @ anchors.T
            + (anchors * anchors).sum(-1))


class VectorLayer(nn.Module):
    """Bir layer: `vectors` tane (start, finish).  C'ye en yakin `active`
    tane start AKTIF; C, aktif vektorlerin agirlikli ortalamasiyla tasinir."""

    def __init__(self, d, vectors, active, randn):
        super().__init__()
        self.start = nn.Parameter(randn(vectors, d))
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
    """NOKTA ve VEKTOR.  Ogrenilen YALNIZ vektorler (start, finish) ve S_v'ler."""
    arch = "pv"

    def __init__(self, n, d=d, vectors=VECTORS, active=ACTIVE, layers=LAYERS,
                 t_max=T_MAX, seed=0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        randn = lambda *shape: torch.randn(*shape, generator=generator)
        self.n, self.d, self.vectors = n, d, vectors
        self.active, self.layers, self.t_max = active, layers, t_max
        P = randn(n, d)
        self.register_buffer("P", P / P.norm(dim=-1, keepdim=True))
        RM = torch.randint(0, 2, (t_max, d), generator=generator).float() * 2 - 1
        self.register_buffer("RM", RM)
        self.V = nn.ModuleList(VectorLayer(d, vectors, active, randn) for _ in range(layers))

    def C(self, tokens):
        """tokens (B,T) -> zincir (B,T,d).  Nedensel: C_t yalniz <= t'yi toplar."""
        length = tokens.shape[1]
        assert length <= self.t_max, "dizi RM sayisindan uzun"
        return (self.RM[:length] * self.P[tokens]).cumsum(1)

    def move(self, tokens):
        """(C_m, layer basina active_ids) -- trace() ve scoreboard() ayni hesabi okur."""
        C_m, active_ids = self.C(tokens), []
        for layer in self.V:
            C_m, layer_ids = layer(C_m)
            active_ids.append(layer_ids)
        return C_m, active_ids

    def score(self, C_m):
        """-D^2, D = |C_m - P|.  Buyuk = yakin."""
        return -distance(C_m, self.P)

    def scoreboard(self, tokens, targets_mask=None):
        """tokens (B,T) -> (B,T,n): her konumda SONRAKI token'in puani.  Konumlar
        arasi karisma yok, dolgu yalniz sagda: targets_mask arayuz icin, hesabi degistirmez."""
        return self.score(self.move(tokens)[0])

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
        Her adim: C_(t-1) = C_t - RM_t*P[tokens_t].  Doner: path = [(position, C_position)], yeniden eskiye."""
        length = tokens.shape[0]
        link, path = self.C(tokens[None])[0, -1], []
        for position in range(length - 1, -1 if K is None else max(-1, length - 1 - K), -1):
            path.append((position, link))
            link = link - self.RM[position] * self.P[tokens[position]]
        return path
