# -*- coding: utf-8 -*-
"""dil_13 -- PARCA KOORDINATLI DIL MODELI. Belge Deney 1-3, torch/GPU.

Kelimenin vektoru SAKLANMAZ, parcalarindan HESAPLANIR. Parametre sayisi
sozlukten degil PARCA sayisindan duser -- modelin butun iddiasi bu.

!! MODELE ILISKI VERILMEZ. `forward`a giren tek sey son K kelimenin
id'si. Iliski indeksi yok, "bu kelime iliskidir" isareti yok, ayri
`Rel` tablosu yok -- `annesi` kelimesi `Kok` tablosunda `Cem` ile ayni
raftadir. Belgenin Deney 4'u `zincir_skor(ozne_id, rel_ids, C)` ile
iliskiyi KAHIN gibi aliyordu; burada model onu metinden cikarmak
zorunda.
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# BELGENIN MODELI
# =====================================================================
class ParcaDil(nn.Module):
    """Semanin tamami: koordinat katmani -> MLP -> hedef nokta -> uzaklik."""

    def __init__(self, wp, D=3, K=3, M=4, H=96, tohum=0):
        """`wp` (V, YUVA) -- birimin her yuvadaki parca indeksi.

        YUVA=3  kelime birim,  koordinat concat(kok, ek, nok)  <- bizim korpus
        YUVA=1  parca birim,   koordinat dogrudan tablodan     <- GERCEK DIL
        Ikisi AYNI kodla kosuyor; degisen tek sey yuva sayisi."""
        super().__init__()
        g = torch.Generator().manual_seed(tohum)
        rn = lambda *s: torch.randn(*s, generator=g)
        wp = torch.as_tensor(wp, dtype=torch.long)
        if wp.dim() == 1:
            wp = wp[:, None]
        self.K, self.M, self.D = K, M, D
        self.yuva = wp.shape[1]
        self.DD = self.yuva * D

        # --- KOORDINAT KATMANI: yuva basina bir parca tablosu.
        # Birim basina ogrenilen satir YOK -- koordinat BURADAN kurulur.
        self.register_buffer("wp", wp)            # sabit, ogrenilmez
        self.tablo = nn.ParameterList(
            [nn.Parameter(rn(int(wp[:, s].max()) + 1, D) * 0.5)
             for s in range(self.yuva)])

        # --- YONLENDIRME AGI: son K kelimenin koordinati -> M hedef nokta
        # Girdi/cikti boyutu SOZLUKTEN BAGIMSIZ; sozluk buyuyunce degismez.
        din, dout = K * self.DD, M * (self.DD + 1)   # M nokta + M agirlik
        self.W1 = nn.Parameter(rn(din, H) * math.sqrt(2 / din))
        self.b1 = nn.Parameter(torch.zeros(H))
        self.W2 = nn.Parameter(rn(H, dout) * math.sqrt(1 / H))
        self.b2 = nn.Parameter(torch.zeros(dout))

        # --- PUANLAMA sicakligi. tau ogrenilir, log(1/tau) tutulur.
        self.lt = nn.Parameter(torch.zeros(1))

    def coord(self):
        """KOORDINAT KATMANI. c_w = yuvalarin concat'i (tek yuvada duz lookup).

        Modelin bir kelimeyi BASTIRAMAMASININ sebebi burasi: o kelimeye
        ait bir parametre yok. Kokunu bozarsa o koku paylasan butun
        kelimeler zarar gorur."""
        return torch.cat([t[self.wp[:, s]]
                          for s, t in enumerate(self.tablo)], -1)  # (V, DD)

    def forward(self, X):
        """X: (B, K) son K kelimenin id'si -> (B, V) skor."""
        C = self.coord()                                       # (V, DD)
        x = C[X].reshape(len(X), -1)                           # (B, K*DD)

        # YONLENDIRME: "siradaki kelime hangisi" degil, "UZAYDA NEREYE"
        h = torch.tanh(x @ self.W1 + self.b1)
        o = h @ self.W2 + self.b2
        q = o[:, :self.M * self.DD].view(-1, self.M, self.DD)  # (B, M, DD)
        a = o[:, self.M * self.DD:]                            # (B, M)
        # M > 1 sebebi: baglam bazen iki ayri yone acik. Tek hedef nokta
        # ikisinin ORTASINA duser ve ikisini de iskalar.

        # PUANLAMA: hedefe UZAKLIK. ||q-c||^2 genisletilmis bicimde --
        # (B,M,V,DD) ara tensoru kurulmaz, sadece (B,M,V).
        d2 = ((q * q).sum(-1)[..., None] - 2 * q @ C.T
              + (C * C).sum(-1))                               # (B, M, V)
        sc = a[..., None] - d2 * self.lt.exp()
        return torch.logsumexp(sc, dim=1), q, C                # M uzerinden

    def kayip(self, X, Y, lam=0.5):
        """CE + KAPANMA CEZASI.

        Softmax tek basina yalniz SIRALAMA ister: hedef nokta dogru
        kelimeye uzak kalsa bile tahmin dogrudur, gradyan susar. Ceza
        noktayi kelimenin UZERINE oturtuyor.

        !! EN YAKIN BILESENE, ORTALAMAYA DEGIL. Ilk surum `q.mean(1)`
        kullaniyordu ve M bilesenin ORTALAMASINI cevaba cekiyordu.
        Bilesenler uzmanlasmisti (q3 tez sifatlari, q1 yer adlari...) ve
        ortalamayi tek noktaya cekmek o uzmanlasmayi yok ediyordu.
        OLCULDU 20 Eylul: lam 0.5 -> 20 yapinca `one` 0.0030'dan
        0.0006'ya DUSTU ve tip uyumu bozuldu. Belgede sorun yok cunku
        orada ceza ZINCIR yolunda ve orada karisim YOK.

        `min_m`: bu cevabi tutacak olan bilesen tam ustune otursun,
        digerleri kendi bolgelerinde kalsin."""
        lg, q, C = self(X)
        ce = F.cross_entropy(lg, Y)
        kap = (q - C[Y][:, None]).pow(2).sum(-1).min(1).values.mean()
        return ce + lam * kap, ce


# =====================================================================
# KONTROL -- onsuz "daha az parametreyle ayni is" cumlesi olculemez
# =====================================================================
class GommeDil(nn.Module):
    """Klasik yol. MLP AYNI; tek fark koordinatin HESAPLANMAYIP
    SAKLANMASI, ve puanin uzaklik yerine IC CARPIM olmasi."""

    def __init__(self, V, d=6, K=3, H=96, tohum=0):
        super().__init__()
        g = torch.Generator().manual_seed(tohum)
        rn = lambda *s: torch.randn(*s, generator=g)
        self.K, self.d = K, d
        self.E = nn.Parameter(rn(V, d) * 0.5)   # KELIME BASINA satir
        din = K * d
        self.W1 = nn.Parameter(rn(din, H) * math.sqrt(2 / din))
        self.b1 = nn.Parameter(torch.zeros(H))
        self.W2 = nn.Parameter(rn(H, d) * math.sqrt(1 / H))
        self.b2 = nn.Parameter(torch.zeros(d))

    def forward(self, X):
        x = self.E[X].reshape(len(X), -1)
        h = torch.tanh(x @ self.W1 + self.b1)
        return (h @ self.W2 + self.b2) @ self.E.T, None, self.E

    def kayip(self, X, Y, lam=0.0):
        # Kapanma cezasi BURAYA YAZILAMAZ: "gitmek istedigin nokta" diye
        # bir sey yok, sadece logit var. Geometri yoksa geometrik kisit
        # da yok -- belgenin kendi cumlesi.
        lg, _, _ = self(X)
        ce = F.cross_entropy(lg, Y)
        return ce, ce


# =====================================================================
# EGITIM VE OLCUM
# =====================================================================
def egit(mdl, X, Y, epok=30, bs=4096, lr=3e-3, lam=0.5, dev="cuda",
         Xd=None, Yd=None, yaz=print):
    """Belge: Adam, lr 3e-3. Batch GPU icin buyutuldu (belgede 256).

    `Xd/Yd` verilirse her olcum noktasinda TUTULAN metin de basilir --
    egitim bitini tek basina okumak EZBER olcer."""
    mdl = mdl.to(dev)
    X = torch.as_tensor(X, device=dev)
    Y = torch.as_tensor(Y, device=dev)
    op = torch.optim.Adam(mdl.parameters(), lr=lr)
    N = len(X)
    g = torch.Generator(device=dev).manual_seed(0)
    t0 = time.time()
    for t in range(epok):
        perm = torch.randperm(N, device=dev, generator=g)
        tot = 0.0
        for i in range(0, N, bs):
            b = perm[i:i + bs]
            k, ce = mdl.kayip(X[b], Y[b], lam)
            op.zero_grad(set_to_none=True)
            k.backward()
            op.step()
            tot += ce.item() * len(b)       # CE, ceza HARIC -- bit okunabilsin
        if (t + 1) % max(1, epok // 6) == 0 or t == 0:
            ek = f"   tutulan {bit(mdl, Xd, Yd, dev):.3f}" if Xd is not None else ""
            yaz(f"    epok {t+1:>4}   bit/kelime {tot/N/math.log(2):.4f}"
                f"{ek}   ({time.time()-t0:.0f} sn)")
    return mdl


@torch.no_grad()
def bit(mdl, X, Y, dev="cuda", bs=8192):
    """bit/kelime -- belgedeki `bits`. Iki modeli kiyaslayan TEK sayi."""
    X = torch.as_tensor(X, device=dev)
    Y = torch.as_tensor(Y, device=dev)
    tot = 0.0
    for i in range(0, len(X), bs):
        lg, _, _ = mdl(X[i:i + bs])
        tot += float(F.cross_entropy(lg, Y[i:i + bs], reduction="sum"))
    return tot / len(X) / math.log(2)


def n_par(mdl):
    return sum(p.numel() for p in mdl.parameters())
