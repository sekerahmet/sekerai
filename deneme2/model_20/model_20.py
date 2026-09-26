# -*- coding: utf-8 -*-
"""model_20 -- adim adim kurulan model (CLAUDE.md kural 13).

Adim 1: TokenPoints + NextRelation.  Bir token girer, sonraki token'in olasiligi cikar; geriye bakmaz.
    PL = norm(PF + shift)          PF sabit (tohumdan), shift 0'dan ogrenilir; model YALNIZ PL'yi kullanir
    q  = norm(W . PL_i)            NextRelation: YALNIZ yon -- sonraki token'in beklenen noktasi (kurede)
    skor_j = scale . <q, PL_j>     olasilik softmax, kayip -log p(hedef) + anchor . sum|shift|^2
    scale  = ln(CONFIDENCE (n-1) / (1-CONFIDENCE))   guven; elle girilmez, nokta sayisindan
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


def scale_for(n, confidence=CONFIDENCE):
    """Hedefin skoru rakiplerden scale kadar yuksekken p(hedef) = confidence."""
    return math.log(confidence * (n - 1) / (1 - confidence))


class TokenPoints(torch.nn.Module):
    """Her token'in kuredeki noktasi.  learn=False: PL = PF (sabit noktalar)."""

    def __init__(self, n, d=D, learn=LEARN_POINTS, anchor=ANCHOR, seed=POINTS_SEED):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("fixed_points", F.normalize(torch.randn(n, d, generator=g), dim=-1))
        self.shift = torch.nn.Parameter(torch.zeros(n, d), requires_grad=learn)
        self.learn, self.anchor = learn, anchor

    def points(self):
        return F.normalize(self.fixed_points + self.shift, dim=-1)

    def anchor_loss(self):
        """PF'den uzaklasmanin bedeli."""
        if not (self.learn and self.anchor):
            return self.shift.new_zeros(())
        return self.anchor * (self.shift ** 2).sum()


class NextRelation(torch.nn.Module):
    """W (d x d), kucuk rastgele: cikti kureye indirildigi icin W'nin boyu bir sey degistirmez, yalniz yonu.
    (0'dan baslayamaz: norm(0)'in yonu yok, gradyani 1/eps.)"""

    def __init__(self, d=D, seed=POINTS_SEED + 1):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = torch.nn.Parameter(torch.randn(d, d, generator=g) / d ** 0.5)

    def forward(self, p):
        return F.normalize(p @ self.W.T, dim=-1)


class NextTokenModel(torch.nn.Module):
    def __init__(self, n, d=D, learn_points=LEARN_POINTS, anchor=ANCHOR, confidence=CONFIDENCE, seed=POINTS_SEED):
        super().__init__()
        self.tokens = TokenPoints(n, d, learn_points, anchor, seed)
        self.next = NextRelation(d, seed + 1)
        self.scale = scale_for(n, confidence)

    def logits(self, ids):
        P = self.tokens.points()
        return self.scale * self.next(P[ids]) @ P.T

    def loss(self, inputs, targets):
        """(toplam, nll): toplam = nll + capa."""
        nll = F.cross_entropy(self.logits(inputs), targets)
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
