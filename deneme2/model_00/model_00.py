# -*- coding: utf-8 -*-
"""model_00 — KLASIK decoder-only transformer. Bu projenin HICBIR ozelligi YOK.

Kullanici istegi, 16 Eylul 2026: *"model 00 kuralim. Bu model her seyden
bagimsiz, senden istedigim standart kendi bilginle kuracagin 8 katmanli,
loop vs olmayan. Klasik LLM transformer yapisi olsun."*

--------------------------------------------------------------------------
NE YOK

    DONGU YOK          l=8 AYRI katman, agirlik paylasimi YOK (dongu=1)
    Phi DARBOGAZI YOK  dar_alfa=0, dar_kapi=False -> hic kurulmuyor
    MASKE YOK          mask_poz=None, mask_blok=()
    YARDIMCI KAYIP YOK kopru_kayip=0
    model_a'DAN MIRAS YOK -- `M.Model`i genisletmiyor, kendi nn.Module'u

model_b `M.Model`i miras aliyordu; bu dosya ALMIYOR. Amac, projenin
birikmis tercihlerinden (ogrenilmis pozisyon gommesi, GELU+4d MLP)
BAGIMSIZ bir referans kurmak.

--------------------------------------------------------------------------
NE VAR -- 2026'nin standart kucuk-LLM tarifi

    decoder-only + causal maske
    pre-norm + RMSNorm                LayerNorm'a gore kararli, bias yok
    RoPE                              ogrenilmis pozisyon gommesi YERINE
    SwiGLU, d_ff = 8/3 * d            GELU + 4d YERINE (ayni parametre)
    bagli gomme (head.weight = emb.weight)
    hicbir lineer katmanda BIAS YOK
    MHA, head_dim 64 (d=256, nh=4)    bu olcekte GQA/MQA anlamsiz
    dropout 0                         sentetik veride zararli

    l = 8 katman, dongu = 1  ->  8 katman-esdegeri
    model_b15: l=4, dongu=2  ->  8 katman-esdegeri   (AYNI hesap derinligi,
                                 ama 4 AYRI agirlik seti, 2 kez donuyor)

Referanslar: nanoGPT (GPT-2 tarifi), Pythia-70m/160m, Llama tarzi blok.
`wd=0.1` + cosine (min_lr = lr/10) CLAUDE.md kural 4'ten geliyor ve
zaten bu tarifin kendisi.

--------------------------------------------------------------------------
VERI AYARLARI model_b15 ILE BIREBIR AYNI -- bu KASITLI

"Her seyden bagimsiz" MIMARI icin gecerli; veri icin degil. Ayni veriyi
gormezse kol hicbir sey olcmez. Degisen YALNIZ mimari:

    veri_okul4 (1060 varlik)   ek_kip="tr"   bicim=3   t_len 17
    ident_frac=0.2  wd=0.1  cosine  tam_kayip=True

`test_00.py` bunu her kosuda sinar: model_b15 ile AYNI olmasi gereken
butun VERI alanlari birebir tutmali, FARKLI olmasi gerekenler de
yalnizca mimari alanlari olmali.

--------------------------------------------------------------------------
PARAMETRE

    dikkat   4 d^2            = 262.144
    SwiGLU   3 d d_ff         = 540.672      (d_ff 704 = 8/3*256, 64'e yuvarli)
    katman basina             = 802.816
    8 katman                  = 6.422.528
    gomme    vocab x d (bagli, TEK sayilir)

model_b15 3.229.953 -- yani bu model KABACA IKI KATI. Ayni katman
esdegerinde daha cok parametre: dongu agirlik PAYLASIYOR, bu paylasmiyor.
Bu bir kusur degil, kolun tanimi: "standart bir transformer ne yapar".
"""
from __future__ import annotations

import math
import os
import sys

_B = os.path.dirname(os.path.abspath(__file__))
_A = os.path.join(os.path.dirname(_B), "model_a")
_K = os.path.dirname(_B)
for _p in (_A, _B, _K, os.path.join(os.path.dirname(_B), "model_b")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn
import torch.nn.functional as F

import model_a as M                                          # noqa: E402
assert hasattr(M, "egit"), (
    f"model_a MODUL degil PAKET olarak yuklendi: {getattr(M,'__file__',None)}")


# ======================= ROPE ============================================
def _rope_tablo(t_len: int, kafa_d: int, taban: float = 10000.0):
    """(cos, sin) -- (t_len, kafa_d/2). Egitimde SABIT, ogrenilmez."""
    assert kafa_d % 2 == 0, f"head_dim CIFT olmali: {kafa_d}"
    frek = 1.0 / (taban ** (torch.arange(0, kafa_d, 2).float() / kafa_d))
    poz = torch.arange(t_len).float()
    aci = poz[:, None] * frek[None, :]                 # (T, kafa_d/2)
    return torch.cos(aci), torch.sin(aci)


def _rope(x, cos, sin):
    """x: (B, nh, T, kafa_d).  Cift/tek kanallari ikili dondurur."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, None, : x.shape[2]], sin[None, None, : x.shape[2]]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


# ======================= BLOK ============================================
class Blok(nn.Module):
    """pre-norm + MHA(RoPE, causal) + pre-norm + SwiGLU. BIAS YOK."""

    def __init__(self, d: int, nh: int, dff: int):
        super().__init__()
        assert d % nh == 0, f"d ({d}) nh'ye ({nh}) bolunmeli"
        self.nh, self.kafa_d = nh, d // nh
        self.n1, self.n2 = M.RMSNorm(d), M.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.po = nn.Linear(d, d, bias=False)
        # SwiGLU: w2( silu(w1 x) * w3 x ).  UC matris, ikisi "kapi".
        self.w1 = nn.Linear(d, dff, bias=False)
        self.w3 = nn.Linear(d, dff, bias=False)
        self.w2 = nn.Linear(dff, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        h = self.n1(x)
        q, k, v = self.qkv(h).split(D, dim=2)
        q = q.view(B, T, self.nh, self.kafa_d).transpose(1, 2)
        k = k.view(B, T, self.nh, self.kafa_d).transpose(1, 2)
        v = v.view(B, T, self.nh, self.kafa_d).transpose(1, 2)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.po(a.transpose(1, 2).contiguous().view(B, T, D))
        h = self.n2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


# ======================= MODEL ===========================================
class ModelSade(nn.Module):
    """Klasik decoder-only LM. `model_a.Model` ile HICBIR miras bagi YOK."""

    def __init__(self, ayar: M.Ayar, vocab: int):
        super().__init__()
        assert ayar.dongu == 1, (
            f"model_00 DONGUSUZ: dongu={ayar.dongu}. Dongu istiyorsan "
            f"model_b ailesini kullan.")
        assert ayar.dar_alfa == 0 and not ayar.dar_kapi, (
            "model_00'da Phi DARBOGAZI YOK -- dar_alfa=0, dar_kapi=False")
        assert ayar.kopru_kayip == 0, "model_00 YARDIMCI KAYIP KULLANMAZ"
        assert ayar.mask_poz is None and not ayar.mask_blok, (
            "model_00'da MASKE YOK")
        self.ayar = ayar
        self.emb = nn.Embedding(vocab, ayar.d)
        self.bloklar = nn.ModuleList(
            [Blok(ayar.d, ayar.nh, ayar.dff) for _ in range(ayar.l)])
        self.nf = M.RMSNorm(ayar.d)
        self.head = nn.Linear(ayar.d, vocab, bias=False)
        self.head.weight = self.emb.weight            # BAGLI GOMME
        c, s = _rope_tablo(ayar.t_len, ayar.d // ayar.nh)
        self.register_buffer("rope_cos", c, persistent=False)
        self.register_buffer("rope_sin", s, persistent=False)
        self.apply(self._ilk)
        # GPT-2 tarifi: ARTIK yoluna yazan izdusumler 1/sqrt(2L) ile
        # olceklenir, yoksa derinlikle birlikte varyans buyur.
        for ad, p in self.named_parameters():
            if ad.endswith("po.weight") or ad.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0,
                                std=0.02 / math.sqrt(2 * ayar.l))

    @staticmethod
    def _ilk(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def n_param(self):
        """Bagli gomme TEK sayilir (head.weight is emb.weight)."""
        return sum(p.numel() for p in {id(p): p
                                       for p in self.parameters()}.values())

    def forward(self, x):
        h = self.emb(x)
        cos = self.rope_cos.to(h.dtype)
        sin = self.rope_sin.to(h.dtype)
        for blk in self.bloklar:
            h = blk(h, cos, sin)
        return self.head(self.nf(h))


# ======================= AYAR ============================================
from model_b15 import AYAR as TABAN                          # noqa: E402

# VERI ayarlari model_b15'ten AYNEN devralinir (yukaridaki nota bak).
# Degisen YALNIZ mimari:
AYAR = TABAN.degistir(
    ad="model_00",
    l=8, dongu=1,            # 8 AYRI katman, paylasim YOK
    dff=704,                 # 8/3 * 256 = 682,7 -> 64'un kati
    dar_alfa=0.0,            # Phi darbogazi YOK
    dar_kapi=False,          # ogrenilen gecit YOK
)

fark_bas = M.fark_bas


def egit(ayar=None, **kw):
    return M.egit(ayar or AYAR, model_kur=ModelSade, **kw)


if __name__ == "__main__":
    M.fark_bas(TABAN, AYAR)
    egit(AYAR)
