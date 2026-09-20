# -*- coding: utf-8 -*-
"""model_13 -- PARCA KOORDINATLI zincir. numpy, elle gradyan.

Belge: `Parca koordinatli dil modeli`, 20 Eylul 2026.
Kurulum birebir: YALNIZ TEK ADIM ogretilir, IKI ADIM hic gosterilmez.
    q = C[ozne] + Rel[r1] + Rel[r2]        skor = -||q - C||^2 * it
    L = -log P(cevap) + lam*||q - C[cevap]||^2      <- kapanma cezasi

!! MLP YOK. Belgede de yok -- `zincir_skor` yonlendirme agina hic
ugramiyor. Dil modeli (Deney 1-3) AYRI bir sey, asama 2'de gelir.

Veri yolu KOLUN KENDISINDE: `taban_13` / `veri_13` / `korpus_13`,
`taban_11`den kopyalandi (kullanici karari, 20 Eylul). Disariya import
YOK -- kapisi iki iz: graf 3cd9a2575e47, olcme 44e6262e37f3.
"""
import time, argparse
import numpy as np

import taban_13 as MT, ayar_13 as AY, veri_13 as V13         # noqa: E402


# ----------------------------------------------------------------- model
class Zincir:
    """Varlik koordinati + iliski YONU. Iki kip:

    kip='serbest'  C ogrenilen (n_ent, DD) tablo       -- TABAN
    kip='parca'    C = concat(Parca[par[e,yuva]])      -- BELGENIN IDDIASI

    Ikisi ayni sinavi verir; fark yalniz koordinatin NEREDEN geldigi.
    """

    def __init__(self, v, D=3, kip="serbest", tohum=0):
        self.rng = np.random.default_rng(tohum)
        self.v, self.kip, self.D = v, kip, D
        n_ent, n_rel = v.facts.shape
        self.n_ent, self.n_rel = n_ent, n_rel
        if kip == "serbest":
            self.DD = 2 * D                      # parca kipiyle esit boyut
            self.C_ = self.rng.normal(size=(n_ent, self.DD)) * 0.5
            self.ps = [self.C_]
        else:
            self.par = v.par                     # (n_ent, yuva)
            self.yuva = v.par.shape[1]
            self.n_par = len(v.par_ad[0])        # PAYLASILAN sozluk
            self.DD = self.yuva * D
            self.Parca = self.rng.normal(size=(self.n_par, D)) * 0.5
            self.ps = [self.Parca]
        self.Rel = self.rng.normal(size=(n_rel, self.DD)) * 0.3
        self.lt = np.array([0.0])                # log(1/tau)
        self.ps += [self.Rel, self.lt]

    def n_params(self):
        return sum(p.size for p in self.ps)

    def coord(self):
        """Varlik koordinatlari. 'parca' kipinde HESAPLANIR, saklanmaz."""
        if self.kip == "serbest":
            return self.C_
        C = np.zeros((self.n_ent, self.DD))
        for s in range(self.yuva):
            ix = self.par[:, s]
            m = ix >= 0                          # -1 = BOS yuva, sifir kalir
            C[m, s * self.D:(s + 1) * self.D] = self.Parca[ix[m]]
        return C

    def _gC_dagit(self, gC):
        """Koordinat gradyanini PARAMETRELERE dagitir."""
        if self.kip == "serbest":
            return [gC]
        gP = np.zeros_like(self.Parca)
        for s in range(self.yuva):
            ix = self.par[:, s]
            m = ix >= 0
            np.add.at(gP, ix[m], gC[m, s * self.D:(s + 1) * self.D])
        return [gP]

    # --- ileri: ozne + iliski TOPLAMI -> butun varliklara karsi skor
    def fwd(self, E, R):
        """E (B,) ozne;  R (B, adim) iliski indeksleri."""
        C = self.coord()
        q = C[E] + self.Rel[R].sum(1)            # (B, DD)
        it = np.exp(self.lt)[0]
        df = q[:, None, :] - C[None, :, :]       # (B, n_ent, DD)
        d2 = (df ** 2).sum(-1)
        return -d2 * it, (C, q, df, it, E, R)

    def grad(self, g, c, Y, lam):
        C, q, df, it, E, R = c
        B = len(q)
        gq = (g[..., None] * (-2 * it * df)).sum(1)          # (B, DD)
        gC = (g[..., None] * (2 * it * df)).sum(0)           # (n_ent, DD)
        git = np.array([(g * (df ** 2).sum(-1)).sum()]) * -it
        # --- kapanma cezasi: hedef, dogru varligin UZERINE otursun.
        # Belgede olmadan iki adim %43, oldugunda %100.
        e = (q - C[Y]) * 2 * lam / B
        gq += e
        np.add.at(gC, Y, -e)
        np.add.at(gC, E, gq)                     # q = C[E] + sum Rel
        gR = np.zeros_like(self.Rel)
        for j in range(R.shape[1]):
            np.add.at(gR, R[:, j], gq)
        return self._gC_dagit(gC) + [gR, git]


class Adam:
    def __init__(self, ps, lr=3e-3):
        self.ps, self.lr, self.t = ps, lr, 0
        self.m = [np.zeros_like(p) for p in ps]
        self.v = [np.zeros_like(p) for p in ps]

    def step(self, gs):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.ps, gs)):
            self.m[i] = .9 * self.m[i] + .1 * g
            self.v[i] = .999 * self.v[i] + .001 * g * g
            mh = self.m[i] / (1 - .9 ** self.t)
            vh = self.v[i] / (1 - .999 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def sm(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def egit(mdl, E, R, Y, ep=60, bs=256, lr=3e-3, lam=0.5, yaz=print):
    op = Adam(mdl.ps, lr)
    rng = np.random.default_rng(0)
    N = len(E)
    for t in range(ep):
        perm = rng.permutation(N)
        tot = 0.0
        for i in range(0, N, bs):
            b = perm[i:i + bs]
            lg, c = mdl.fwd(E[b], R[b])
            P = sm(lg)
            g = P.copy()
            g[np.arange(len(b)), Y[b]] -= 1
            g /= len(b)
            tot += -np.log(P[np.arange(len(b)), Y[b]] + 1e-12).sum()
            op.step(mdl.grad(g, c, Y[b], lam))
        if (t + 1) % 10 == 0 or t == 0:
            yaz(f"    epok {t+1:>3}  kayip {tot/N:.4f}")
    return mdl


# ------------------------------------------------------------- olcum
def sina(mdl, zin, adim, kisayol=None):
    """(tam, kisayol, kapanma). `zin`: (e, r1[, r2, kopru], cevap)."""
    if not len(zin):
        return None
    Z = np.array([list(z) for z in zin], np.int64)
    E = Z[:, 0]
    R = Z[:, 1:1 + adim]
    Y = Z[:, -1]
    lg, (C, q, *_) = mdl.fwd(E, R)
    tah = lg.argmax(1)
    tam = float((tah == Y).mean())
    kap = float(np.sqrt(((q - C[Y]) ** 2).sum(-1)).mean())
    ksy = float('nan')
    if kisayol is not None:
        gec = kisayol >= 0
        ksy = float((tah[gec] == kisayol[gec]).mean()) if gec.any() else 0.0
    return tam, ksy, kap


def kisayol_hedefi(v, zin):
    """r2'yi KOPRUYE degil OZNEYE uygulayinca cikan varlik. -1 = yok."""
    out = []
    for z in zin:
        e, r2, a = int(z[0]), int(z[2]), int(z[-1])
        h = int(v.facts[e, r2])
        out.append(h if h >= 0 and h != a else -1)
    return np.array(out, np.int64)
