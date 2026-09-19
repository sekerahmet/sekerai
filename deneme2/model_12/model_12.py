# -*- coding: utf-8 -*-
"""model_12 — HIBRIT: Gated DeltaNet-2 x6 + tam dikkat x2, 8 blok.

Soru: 2026'nin guncel mimarisi, BIZIM DILIMIZDE model_11 ile AYNI
SONUCU veriyor mu?  Artimli kol DEGIL, dogru kurulmus IKINCI REFERANS
-- karistirici + optimizer + precision BIRLIKTE degisiyor, ATFETME
YAPILAMAZ. Veri ve sinav model_11 ile BIT AYNI (olcme izi 44e6262e37f3).

    karisim   3:1, dikkat 4. ve 8. blokta   2507.06457 (recall 3:1'de
                                            doyuyor) + 2406.07887
                                            ("evenly dispersed")
    mixer     Gated DeltaNet-2              2605.22791, 1.3B'de Mamba-2
                                            / Mamba-3 / GDN / KDA'yi gecti
    FFN       DEGISMEDI (SwiGLU dff=704)    model_11'de aramanin 8. FFN'de
                                            yapildigi olculdu; sabit tutuldu

!! Bu olcekte HIZ KAZANCI YOK, beklenmiyor da: t_len=512'de dikkatin
karesel kismi hesabin ~%25'i. Kazanc uzun baglamda ve cikarimda.

Gerekce ve karar kurali: `belge/onkayit/model_12.md`.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import taban_12 as M

# ======================= RoPE ===========================================
# model_11'den AYNEN. Tam dikkat bloklari degismedi.


def _rope_tablo(t_len: int, kafa_d: int, taban: float = 10000.0):
    """(cos, sin) -- (t_len, kafa_d/2). Egitimde SABIT, ogrenilmez."""
    assert kafa_d % 2 == 0, f"head_dim CIFT olmali: {kafa_d}"
    frek = 1.0 / (taban ** (torch.arange(0, kafa_d, 2).float() / kafa_d))
    poz = torch.arange(t_len).float()
    aci = poz[:, None] * frek[None, :]                 # (T, kafa_d/2)
    return torch.cos(aci), torch.sin(aci)


def _rope(x, cos, sin):
    """x: (B, nh, T, kafa_d). Cift/tek kanallari ikili dondurur."""
    c = cos[None, None, : x.shape[2]].to(x.dtype)
    s = sin[None, None, : x.shape[2]].to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


# ======================= GATED DELTANET-2 ===============================
# arXiv 2605.22791, Eq. 9/10 (ozyineli) ve Ek A (parcali).
#
#   e_t = b_t (*) k_t        b_t = sigma(W_b x_t) in [0,1]^dk   SILME kapisi
#   z_t = w_t (*) v_t        w_t = sigma(W_w x_t) in [0,1]^dv   YAZMA kapisi
#   g_t = -exp(a) (*) softplus(W_f x_t + delta)   alpha_t = exp(g_t)
#
#   S_t = (I - k_t e_t^T) Diag(alpha_t) S_{t-1} + k_t z_t^T      (Eq. 10)
#   o_t = S_t^T q_t
#
# GDN-2'nin GDN'den farki: tek skaler `beta` IKIYE AYRILIYOR. Makale:
#   "the scalar beta still carries two decisions that need not agree.
#    One decision lives on the key side ... The other lives on the value
#    side"
#   "Ablations show that both gates contribute, with the erase gate
#    accounting for most of the gain."
# b = beta*1 ve w = beta*1 konursa KDA'ya, ayrica alpha skalerse
# Gated DeltaNet'e GERI DONER -- yani eski kurallar alt uzay olarak duruyor.


@torch.no_grad()
def _gdn2_ozyineli(q, k, v, b, w, g):
    """Eq. 9 referansi, adim adim. EGITIMDE KULLANILMAZ.

    Tek isi `_gdn2_parcali`yi sinamak (test_12): parcali form yanlis
    yazilirsa SESSIZCE calisir -- kayip yine duser ama ogrenilen sey
    GDN-2 degildir. Korunma: iki bagimsiz uygulama, aralarinda kapi.

    q,k,b,g: (B,H,T,dk)   v,w: (B,H,T,dv)  ->  (B,H,T,dv)
    """
    B, H, T, dk = q.shape
    dv = v.shape[-1]
    S = torch.zeros(B, H, dk, dv, device=q.device, dtype=torch.float32)
    cik = []
    for t in range(T):
        al = g[:, :, t].float().exp()                     # (B,H,dk)
        kt = k[:, :, t].float()
        Sb = al.unsqueeze(-1) * S                         # Diag(alpha) S
        et = (b[:, :, t].float() * kt)                    # (B,H,dk)
        rt = (Sb * et.unsqueeze(-1)).sum(-2)              # Sb^T e  (B,H,dv)
        zt = (w[:, :, t].float() * v[:, :, t].float())    # (B,H,dv)
        S = Sb + kt.unsqueeze(-1) * (zt - rt).unsqueeze(-2)
        cik.append((S * q[:, :, t].float().unsqueeze(-1)).sum(-2))
    return torch.stack(cik, dim=2)


def _gdn2_parcali(q, k, v, b, w, g, C: int = 16):
    """Parcali form -- arXiv 2605.22791 Ek A, ORAN bicimi (Eq. 41/43).

        D[r,s] = G_r - G_s          s <= r icin <= 0, yani exp(D) <= 1
        T      = tril(sum_j (b*k)_rj exp(D)_rsj k_sj, -1) + I    Eq. 34
        Aqk    = tril(sum_j  q_rj   exp(D)_rsj k_sj,  0)         Eq. 43
        [Y|U]  = T^-1 [gamma*(b*k) | w*v]     R = U - Y S0       Eq. 34/35
        O      = gamma*q S0 + Aqk R                              Eq. 44
        S_C    = Diag(gamma_C) S0 + K_tail^T R                   Eq. 40
                 K_tail = exp(G_C - G) (*) k

    `gamma^-1` HIC KURULMAZ -- butun usteller <= 1, tasma imkansiz.

    !! ONCEKI HALI `G = cumsum(g).clamp(min=-30)` ile gamma^-1 kuruyordu
    ve Eq. 9'u DEGISTIRIYORDU: modelin kendi ilk degerinde kanal x chunk
    ciftlerinin %24-57'sinde kirpma devredeydi, cikti gercek Eq. 9'dan
    bagil L2 0.55-1.17 sapiyordu (19 Eylul hakemligi + olculdu). Kirpmayi
    silmek de cozum degildi: ayni cebir fp32'de NaN veriyor, fp64'te
    4.5e-16. Oran bicimi ikisini birden cozuyor.

    BEDEL: (B,H,n,C,C,dk) ara tensor -- bellek C ile DOGRUSAL. Sonuc
    C'den BAGIMSIZ oldugu icin C=16 (B=32'de 268 MB; C=64'te 1,07 GB).

    SAGDAN DOLGU: T her zaman C'nin kati degil (sinav dizileri 86/113).
    Nedensel oldugu icin guvenli; dolgu sifir, durumu kirletmez.
    """
    Bs, H, T0, dk = q.shape
    T = T0
    if T % C:
        _p = C - T % C
        ped = lambda t: F.pad(t, (0, 0, 0, _p))
        q, k, v, b, w, g = (ped(q), ped(k), ped(v), ped(b), ped(w), ped(g))
        T += _p
    n = T // C
    # !! autocast ALTINDA `.float()` YETMEZ: matmul/einsum girdileri
    # yeniden bf16'ya iner. Makale D.3 durumu ve biriktiricileri fp32
    # istiyor -- blok bunu ZORLUYOR.
    with torch.autocast(device_type=q.device.type, enabled=False):
        rs = lambda t: t.float().reshape(Bs, H, n, C, t.shape[-1])
        q, k, b, g, v, w = rs(q), rs(k), rs(b), rs(g), rs(v), rs(w)
        G = g.cumsum(-2)                         # chunk-yerel, KIRPMA YOK
        # Ust ucgen (s > r) clamp ile 1'e sabitlenir ve asagida tril ile
        # ATILIR; iki islemin de turevi orada 0.
        D = (G.unsqueeze(-2) - G.unsqueeze(-3)).clamp(max=0.0)
        KW = D.exp() * k.unsqueeze(-3)           # (gamma_r/gamma_s) (*) k_s
        Tm = torch.einsum("...rj,...rsj->...rs", b * k, KW).tril(-1)
        Aqk = torch.einsum("...rj,...rsj->...rs", q, KW).tril(0)
        Tm = Tm + torch.eye(C, device=q.device, dtype=q.dtype)
        gam = G.exp()                                            # <= 1
        Kt = (G[..., -1:, :] - G).exp() * k                      # <= 1
        # Eb ve Z yan yana, TEK ucgen cozum, sonra ikiye ayrilir.
        YU = torch.linalg.solve_triangular(
            Tm, torch.cat([gam * (b * k), w * v], dim=-1), upper=False)
        Y, U = YU[..., :dk], YU[..., dk:]
        Qg = gam * q
        S = torch.zeros(Bs, H, dk, v.shape[-1], device=q.device,
                        dtype=torch.float32)
        cik = []
        for c in range(n):
            R = U[:, :, c] - Y[:, :, c] @ S
            cik.append(Qg[:, :, c] @ S + Aqk[:, :, c] @ R)
            S = (gam[:, :, c, -1].unsqueeze(-1) * S
                 + Kt[:, :, c].transpose(-1, -2) @ R)
        o = torch.cat(cik, dim=2)
    return o[:, :, :T0]                          # dolgu KIRPILIR


class GDN2(nn.Module):
    """Gated DeltaNet-2 karistirici. FFN ICERMEZ -- o `Blok`ta.

    Katman parametrelendirmesi makale C.1: q/k/v kisa evrisimli
    izdusumler + SiLU, sonra kafa basina L2 (D.2). b = sigma(Proj_b x)
    SILME (H*dk), w = sigma(Proj_w x) YAZMA (Hv*dv),
    g = -exp(a) (*) softplus(Proj_f x + delta). Cikis: RMSNorm + SiLU
    kapisi, sonra izdusum (D.5).

    SONUM ILK DEGERI Mamba-2 / resmi GDN-2 tarifi: exp(a) ~ U(1,16),
    dt LOG-uniform(1e-3, 0.1), delta = ters-softplus(dt).

    BEYANLI SAPMALAR (hepsi 19 Eylul hakemliginde gorusuldu):
      ilk deger   makale D.5 Xavier uniform + kazanc 2^-2.5 diyor; burada
                  projenin GPT-2 tarifi (normal std=0.02). Gomme/FFN
                  model_11 ile AYNI kalmali. Hakem: zararsiz -- q/k L2
                  normlu, v'nin olcegini cikis RMSNorm'u yutuyor.
      dt tabani   resmi kod 1e-4; burada 1e-3 (Megatron uyarlamasi ve
                  Mamba-2 varsayilani).
      dusuk rank  resmi uygulamada sonum ve cikis kapisi izdusumleri
                  dusuk ranklı; makale bunu SOYLEMIYOR. Burada tam rank
                  -- yalniz parametre butcesi farki.
    """

    def __init__(self, d: int, H: int, dk: int, dv: int, Hv: int,
                 evrisim: int = 4):
        super().__init__()
        assert Hv % H == 0, f"Hv ({Hv}) H'ye ({H}) bolunmeli"
        self.H, self.dk, self.dv, self.Hv = H, dk, dv, Hv
        self.grup = Hv // H
        self.q = nn.Linear(d, H * dk, bias=False)
        self.k = nn.Linear(d, H * dk, bias=False)
        self.v = nn.Linear(d, Hv * dv, bias=False)
        self.pb = nn.Linear(d, H * dk, bias=False)         # silme kapisi
        self.pw = nn.Linear(d, Hv * dv, bias=False)        # yazma kapisi
        self.pf = nn.Linear(d, H * dk, bias=False)         # log-sonum
        self.kapi = nn.Linear(d, Hv * dv, bias=False)      # cikis kapisi
        self.o = nn.Linear(Hv * dv, d, bias=False)
        # cikis normu DEGER KAFASI BASINA (resmi kod); makale yalniz
        # "RMS-normalize" diyor. Birlesik Hv*dv kafada yapilirsa gruplar
        # birbirinin olcegini tasir.
        self.nf = M.RMSNorm(dv)
        # kafa basina `a`, kanal basina `delta` (C.1, Eq. 86)
        self.a = nn.Parameter(torch.zeros(H, 1))
        self.delta = nn.Parameter(torch.zeros(H * dk))
        # !! SONUM ILK DEGERI. Sifir birakilinca alpha = 0.50: durum HER
        # ADIMDA yarilaniyor. Mamba-2 / GDN tarifi: alpha 1'E YAKIN
        # baslar, unutmayi model OGRENIR.
        #   exp(a) ~ U(1,16)     dt ~ LOG-uniform(1e-3, 0.1)
        # !! dt DUZ uniform olunca yari-omur medyani 2,1 KARAKTERE
        # dusuyordu (log-uniform'da 8,6) -- 19 Eylul hakemligi, olculdu.
        with torch.no_grad():
            self.a.copy_(torch.empty(H, 1).uniform_(1.0, 16.0).log())
            _dt = torch.empty(H * dk).uniform_(
                math.log(1e-3), math.log(0.1)).exp()
            self.delta.copy_(torch.expm1(_dt).log())
        # kisa nedensel evrisim (derinlemesine), Nemotron-H: pencere 4
        self.ev = nn.Conv1d(2 * H * dk + Hv * dv, 2 * H * dk + Hv * dv,
                            evrisim, groups=2 * H * dk + Hv * dv,
                            padding=evrisim - 1, bias=False)

    def _tarama_girdisi(self, x):
        """Taramanin girdisi (q, k, v, b, w, g) -- hepsi fp32, (B,H,T,*).

        forward DA test_12 kapisi DA burayi cagirir. Ayri yazilsaydi kapi
        modelin URETTIGINDEN BASKA bir dagilimi sinardi -- 19 Eylul
        hakemliginin bulgusu tam buydu."""
        Bq, T, _ = x.shape
        qkv = torch.cat([self.q(x), self.k(x), self.v(x)], -1)
        # kisa evrisim + SiLU (makale 3.5 ve Sekil 1), SONRA L2
        qkv = F.silu(self.ev(qkv.transpose(1, 2))[..., :T]).transpose(1, 2)
        nq = self.H * self.dk
        q, k, v = qkv.split([nq, nq, self.Hv * self.dv], dim=-1)
        kafa = lambda t, h, dd: t.view(Bq, T, h, dd).transpose(1, 2)
        q = F.normalize(kafa(q, self.H, self.dk).float(), dim=-1)     # D.2
        k = F.normalize(kafa(k, self.H, self.dk).float(), dim=-1)
        v = kafa(v, self.Hv, self.dv).float()
        b = torch.sigmoid(kafa(self.pb(x), self.H, self.dk).float())
        w = torch.sigmoid(kafa(self.pw(x), self.Hv, self.dv).float())
        f = kafa(self.pf(x), self.H, self.dk).float()
        g = -self.a.exp()[None, :, None, :] * F.softplus(
            f + self.delta.view(1, self.H, 1, self.dk))
        # GRUPLANMIS DEGER KAFALARI: anahtar tarafi KOPYALANMIYOR. Ayni
        # (k, e, alpha) altinda iki deger kafasi, dv IKI KATI TEK kafayla
        # OZDES -- durum [S_a | S_b] ayni sol carpanla ilerliyor. Hakem
        # dogruladi: sayisal fark 0.0, tile sirasi resmi kodla uyumlu.
        gr, Hk = self.grup, self.H
        kat = lambda t: (t.view(Bq, Hk, gr, T, -1).permute(0, 1, 3, 2, 4)
                         .reshape(Bq, Hk, T, gr * t.shape[-1]).contiguous())
        return q, k, kat(v), b, kat(w), g

    def forward(self, x, cos=None, sin=None):
        Bq, T, _ = x.shape
        o = _gdn2_parcali(*self._tarama_girdisi(x))
        o = (o.view(Bq, self.H, T, self.grup, self.dv)
             .permute(0, 2, 1, 3, 4).reshape(Bq, T, self.Hv, self.dv))
        o = self.nf(o).reshape(Bq, T, self.Hv * self.dv).to(x.dtype)
        return self.o(o * F.silu(self.kapi(x)))


# ======================= TAM DIKKAT =====================================
class Dikkat(nn.Module):
    """MHA + RoPE + nedensel maske. model_11 ile AYNI, tek fark QK-NORM.

    QK-norm 2026'da standart (Qwen3, Gemma): q ve k kafa basina RMSNorm'dan
    gecer, logit patlamasini engeller. model_11'de YOKTU."""

    def __init__(self, d: int, nh: int):
        super().__init__()
        assert d % nh == 0, f"d ({d}) nh'ye ({nh}) bolunmeli"
        self.nh, self.kafa_d = nh, d // nh
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.po = nn.Linear(d, d, bias=False)
        self.nq = M.RMSNorm(self.kafa_d)
        self.nk = M.RMSNorm(self.kafa_d)

    def forward(self, x, cos, sin):
        Bq, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        sek = lambda t: t.view(Bq, T, self.nh, self.kafa_d).transpose(1, 2)
        q, k, v = self.nq(sek(q)), self.nk(sek(k)), sek(v)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.po(a.transpose(1, 2).contiguous().view(Bq, T, D))


# ======================= BLOK ===========================================
class Blok(nn.Module):
    """pre-norm + KARISTIRICI + pre-norm + SwiGLU. BIAS YOK.

    FFN model_11'den AYNEN alindi ve BILEREK dokunulmadi: model_11'de
    aramanin 8. FFN'de yapildigi olculdu (aile kutlesi 0.000 -> 0.972 tek
    alt katmanda). Karistiriciyi degistirip FFN'i sabit tutmak, o olcumu
    yeni mimaride TEKRARLANABILIR kiliyor."""

    def __init__(self, d: int, dff: int, karistirici: nn.Module):
        super().__init__()
        self.n1, self.n2 = M.RMSNorm(d), M.RMSNorm(d)
        self.mix = karistirici
        self.w1 = nn.Linear(d, dff, bias=False)
        self.w3 = nn.Linear(d, dff, bias=False)
        self.w2 = nn.Linear(dff, d, bias=False)

    def forward(self, x, cos, sin):
        x = x + self.mix(self.n1(x), cos, sin)
        h = self.n2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


# ======================= MODEL ==========================================
DESEN = "GGGAGGGA"        # 3:1, dikkat blok 4 ve 8'de (evenly dispersed)


class ModelHibrit(nn.Module):
    """8 blok, 6 GDN-2 + 2 tam dikkat. `taban_12.Model` ile MIRAS BAGI YOK."""

    def __init__(self, ayar: M.Ayar, vocab: int, desen: str = None):
        super().__init__()
        # DESEN AYARDAN GELIR. Onceki hali varsayilani modul sabitine
        # baglyordu: `ayar.karisim` degisse model IZLEMEZ, sessizce
        # "GGGAGGGA" kurardi ve `ayar_t<N>.json` YALAN yazardi.
        desen = desen or ayar.karisim or DESEN
        assert ayar.dongu == 1, "model_12 DONGUSUZ"
        assert ayar.dar_alfa == 0 and not ayar.dar_kapi, "Phi DARBOGAZI YOK"
        assert ayar.kopru_kayip == 0, "YARDIMCI KAYIP YOK"
        assert ayar.mask_poz is None and not ayar.mask_blok, "MASKE YOK"
        assert len(desen) == ayar.l, f"desen {desen} != l {ayar.l}"
        assert set(desen) <= {"G", "A"}, f"desen yalniz G/A: {desen}"
        self.ayar, self.desen = ayar, desen
        self.emb = nn.Embedding(vocab, ayar.d)
        kd = ayar.d // ayar.nh
        self.bloklar = nn.ModuleList([
            Blok(ayar.d, ayar.dff,
                 Dikkat(ayar.d, ayar.nh) if c == "A" else
                 GDN2(ayar.d, ayar.nh, kd, kd, ayar.nh * ayar.gdn_v_kat))
            for c in desen])
        self.nf = M.RMSNorm(ayar.d)
        self.head = nn.Linear(ayar.d, vocab, bias=False)
        self.head.weight = self.emb.weight            # BAGLI GOMME
        c, s = _rope_tablo(ayar.t_len, kd)
        self.register_buffer("rope_cos", c, persistent=False)
        self.register_buffer("rope_sin", s, persistent=False)
        self.apply(self._ilk)
        # GPT-2 tarifi: ARTIK yoluna yazan izdusumler 1/sqrt(2L) ile
        for ad, p in self.named_parameters():
            if ad.endswith("po.weight") or ad.endswith("w2.weight") \
                    or ad.endswith("mix.o.weight"):
                nn.init.normal_(p, mean=0.0,
                                std=0.02 / math.sqrt(2 * ayar.l))

    @staticmethod
    def _ilk(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def n_param(self, bagli_ayri=False):
        n = sum(p.numel() for p in self.parameters())
        return n + self.emb.weight.numel() if bagli_ayri else n

    def govde(self, x):
        """SON katman + son norm -- head ONCESI gizli durum (B, T, d)."""
        assert x.shape[1] <= self.rope_cos.shape[0], (
            f"dizi {x.shape[1]} > t_len {self.rope_cos.shape[0]}")
        h = self.emb(x)
        cos, sin = self.rope_cos.to(h.dtype), self.rope_sin.to(h.dtype)
        for blk in self.bloklar:
            h = blk(h, cos, sin)
        return self.nf(h)

    def forward(self, x):
        return self.head(self.govde(x))


# ======================= AYAR ===========================================
# KENDI ayar dosyasindan okunur -- paylasilan tercihlere ESIR DEGIL.
from ayar_12 import AYAR, GOREV_ALAN                         # noqa: E402,F401

# Kiyas tabani: MOTORUN VARSAYILANI.
TABAN = M.Ayar()

fark_bas = M.fark_bas


def egit(ayar=None, **kw):
    return M.egit(ayar or AYAR, model_kur=ModelHibrit, **kw)


if __name__ == "__main__":
    M.fark_bas(TABAN, AYAR)
    egit(AYAR)
