# -*- coding: utf-8 -*-
"""model_05 — KLASIK decoder-only transformer. Bu projenin HICBIR ozelligi YOK.

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
AYARLAR `ayar_05.py`DE -- paylasilan tercihlere ESIR DEGIL

    VERI alanlari         model_b15'ten AYNEN (sinav ve havuz BIT AYNI)
    MIMARI + OPTIMIZASYON STANDART TARIF, referanslariyla

Iki ayar BILEREK devralinmadi (gerekce `ayar_05.py`de):

    ort_bas 10000 -> 0        LOOKAHEAD ORTALAMASI KAPATILDI. Paylasilan
                              ayar 10.000. adimdan sonra yavas agirlik
                              tutup karistiriyor; bu bir optimizer
                              SARMALAYICISI ve hicbir standart tarifte
                              YOK. Standart modelin ne yaptigini olcecek
                              bir kol, standart olmayan bir numarayla
                              kosamaz.
    betas (0.9,0.999) -> (0.9,0.95)   0.999 PyTorch varsayilani; nanoGPT,
                              GPT-3, Llama, Pythia hepsi 0.95.

`test_05.py` her kosuda sinar: VERI alanlari model_b15 ile birebir
tutmali, farklilar da YALNIZ mimari + bu iki optimizasyon alani olmali.

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

# YALNIZ KENDI KLASORU yola eklenir. `model_a` / `model_b` / ust klasor
# EKLENMEZ -- kullanici karari, 16 Eylul: "model_05 diger hicbir model
# ile ayni seyi kullanmamali."
_B = os.path.dirname(os.path.abspath(__file__))
if _B not in sys.path:
    sys.path.insert(0, _B)

import torch
import torch.nn as nn
import torch.nn.functional as F

import taban_05 as M                                         # noqa: E402
assert hasattr(M, "egit"), (
    f"taban_05 MODUL degil PAKET olarak yuklendi: {getattr(M,'__file__',None)}")


# ======================= ROPE ============================================
def _rope_tablo(t_len: int, kafa_d: int, taban: float = 10000.0):
    """(cos, sin) -- (t_len, kafa_d/2). Egitimde SABIT, ogrenilmez."""
    assert kafa_d % 2 == 0, f"head_dim CIFT olmali: {kafa_d}"
    frek = 1.0 / (taban ** (torch.arange(0, kafa_d, 2).float() / kafa_d))
    poz = torch.arange(t_len).float()
    aci = poz[:, None] * frek[None, :]                 # (T, kafa_d/2)
    return torch.cos(aci), torch.sin(aci)


def _rope(x, cos, sin):
    """x: (B, nh, T, kafa_d).  Cift/tek kanallari ikili dondurur.

    GORELILIK SINANDI (16 Eylul, hakemlik): q_m . k_n yalniz (m-n)'ye
    bagli -- m-n=3 icin bes farkli m'de yayilim 1,9e-06; m=n'de skor
    donmemis hale ESIT. Yani donme birimsel ve goreli.

    `.to(x.dtype)`: autocast'ta `emb` fp32, `qkv` fp16 doner; cos/sin
    `h.dtype`den (fp32) geldigi icin q/k fp32'ye YUKSELIRDI. SDPA
    autocast listesinde oldugu icin bu FIILEN sorun CIKARMIYOR (CPU
    bf16 autocast ile sinandi, gecti) -- ama o bir PyTorch politikasi.
    Tek satirla bagimsiz kaliyoruz; sayisal fark YOK."""
    c = cos[None, None, : x.shape[2]].to(x.dtype)
    s = sin[None, None, : x.shape[2]].to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
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
            f"model_05 DONGUSUZ: dongu={ayar.dongu}. Dongu istiyorsan "
            f"model_b ailesini kullan.")
        assert ayar.dar_alfa == 0 and not ayar.dar_kapi, (
            "model_05'da Phi DARBOGAZI YOK -- dar_alfa=0, dar_kapi=False")
        assert ayar.kopru_kayip == 0, "model_05 YARDIMCI KAYIP KULLANMAZ"
        assert ayar.mask_poz is None and not ayar.mask_blok, (
            "model_05'da MASKE YOK")
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

    def govde(self, x):
        """SON katman + son norm -- head ONCESI gizli durum (B, T, d).

        `asama1_05.gizli()` (DOGRUSAL SONDA) bunu cagirir. KUSUR ve
        DUZELTMESI (16 Eylul hakemligi): orada ileri gecis ELLE yeniden
        kuruluyordu --

            h = net1.emb(xb) + net1.pos(...)      # ogrenilmis pozisyon
            for blk in net1.bloklar: h = blk(h)   # tek argumanli blok

        ikisi de `model_a.Model`e ozgu; `ModelSade`de `.pos` YOK (RoPE
        var) ve `Blok.forward` UC argumanli. Sonda 20.000 adimlik
        kosudan SONRA AttributeError ile duserdi -- ve o sonda bu kolun
        onkayitta yazili EK OKUMASI. Artik tek kaynak burasi: sonda
        modelin GERCEKTEN hesapladigini goruyor."""
        # RoPE tablosu t_len'e gore kuruldu. Daha uzun dizi gelirse
        # dilimleme SESSIZCE kisa tablo dondururdu; burada patlasin.
        assert x.shape[1] <= self.rope_cos.shape[0], (
            f"dizi {x.shape[1]} > t_len {self.rope_cos.shape[0]}")
        h = self.emb(x)
        cos = self.rope_cos.to(h.dtype)
        sin = self.rope_sin.to(h.dtype)
        for blk in self.bloklar:
            h = blk(h, cos, sin)
        return self.nf(h)

    def forward(self, x):
        return self.head(self.govde(x))


# ======================= AYAR ============================================
# KENDI ayar dosyasindan okunur -- paylasilan tercihlere ESIR DEGIL.
# Kullanici, 16 Eylul: "ayar dosyasi ise onu ayar00 diye bir dosya yap,
# ordan okusun." Hangi alanin nereden geldigi ve NEDEN o degerde oldugu
# `ayar_05.py`de tek tek yazili -- ve artik `model_b15`ten DEVRALINMIYOR,
# `Ayar()` varsayilaninin uzerine acikca yaziliyor.
from ayar_05 import AYAR, GOREV_ALAN                         # noqa: E402,F401

# Kiyas tabani: MOTORUN VARSAYILANI. (Onceden `model_b15.AYAR` idi; kol
# artik o zincire bagli degil.)
TABAN = M.Ayar()

fark_bas = M.fark_bas


def egit(ayar=None, **kw):
    return M.egit(ayar or AYAR, model_kur=ModelSade, **kw)


if __name__ == "__main__":
    M.fark_bas(TABAN, AYAR)
    egit(AYAR)
