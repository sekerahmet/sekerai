# -*- coding: utf-8 -*-
"""KELIME katmani -- korpus metni -> kelime dizisi + parca ayrismasi.

Kullanici, 20 Eylul 2026: *"Artik klasik manada token kavrami yok
sozluk yani kelime kavrami var"*.

Birim KELIME. Sozluk kelimelerden olusur, PARAMETRE parcalardan --
belgenin butun iddiasi bu farkta. Kelimenin koordinati saklanmaz,
parcalarindan HESAPLANIR.

    kelime          kok           ek        nok
    Yildiz'in       Yildiz        'in       -
    Yildiz'dir,     Yildiz        'dir      ,
    Tezi'nin        Tezi          'nin      -
    bilinir.        bilinir       -         .
    Cem             Cem           -         -
    annesi          annesi        -         -

SINIR TAHMIN DEGIL. Korpus `ek_kip="tr2"` ile uretiliyor ve tamlama /
bildirme eklerini KESME ISARETIYLE yaziyor; ayrisma o isareti okuyor.
Belgenin "gercek metinde kok/ek sinirini kim koyacak" sinirini bu
korpus kendi uretimiyle kapatiyor.
"""
from __future__ import annotations

import collections

import numpy as np

import jeton_13 as J

NOKTALAMA = ".,?"


def ayir(k: str):
    """kelime -> (kok, ek, nok). Sondaki noktalama AYRI yuvaya."""
    nok = ""
    while k and k[-1] in NOKTALAMA:
        nok = k[-1] + nok
        k = k[:-1]
    if "'" in k:
        kok, ek = k.split("'", 1)
        return kok, "'" + ek, nok
    return k, "", nok


def metin_coz(X, S, n=None):
    """Egitim tensoru -> metin dilimleri. EOS satir sonu olur."""
    geri = dict(S.geri)
    geri[J.EOS] = "\n"
    sat = X if n is None else X[:n]
    return ["".join(geri[int(t)] for t in r if int(t) != J.PAD) for r in sat]


class Sozluk:
    """Kelime sozlugu + (kok, ek, nok) tablosu.

    `wp[w] = (kok_ix, ek_ix, nok_ix)` -- modelin koordinat katmani
    kelimeyi BURADAN kuruyor, kelime basina ogrenilen vektor YOK."""

    def __init__(self, diziler, en_az=1):
        say = collections.Counter()
        for s in diziler:
            say.update(s.replace("\n", " ").split())
        self.kelime = sorted(k for k, n in say.items() if n >= en_az)
        self.kx = {k: i for i, k in enumerate(self.kelime)}
        par = [ayir(k) for k in self.kelime]
        self.kok_ad = sorted({p[0] for p in par})
        self.ek_ad = sorted({p[1] for p in par})
        self.nok_ad = sorted({p[2] for p in par})
        ki = {a: i for i, a in enumerate(self.kok_ad)}
        ei = {a: i for i, a in enumerate(self.ek_ad)}
        ni = {a: i for i, a in enumerate(self.nok_ad)}
        self.wp = np.array([[ki[a], ei[b], ni[c]] for a, b, c in par],
                           np.int64)
        self.say = say
        # BIREBIRLIK: iki kelime AYNI parca ucluune dusemez, yoksa model
        # onlari ayirt edemez ve tavan sessizce duser.
        assert len({tuple(r) for r in self.wp}) == len(self.kelime), (
            "BIREBIR DEGIL -- ayni (kok,ek,nok) birden cok kelimeye denk")

    @property
    def V(self):
        return len(self.kelime)

    def kodla(self, diziler):
        """Metin dilimleri -> tek kelime-id dizisi. Bilinmeyen ATILIR."""
        out = []
        for s in diziler:
            for k in s.replace("\n", " ").split():
                i = self.kx.get(k)
                if i is not None:
                    out.append(i)
        return np.array(out, np.int64)

    def ozet(self):
        n = len(self.kok_ad) + len(self.ek_ad) + len(self.nok_ad)
        return (f"kelime {self.V:,}   <-   kok {len(self.kok_ad)} + "
                f"ek {len(self.ek_ad)} + nok {len(self.nok_ad)} = {n} parca"
                f"   ({self.V / n:.2f}x)")


def pencereler(dizi, K=3):
    """Kelime dizisi -> (X, Y): son K kelime -> sonraki kelime."""
    n = len(dizi) - K
    X = np.lib.stride_tricks.sliding_window_view(dizi[:-1], K)[:n]
    return np.ascontiguousarray(X), dizi[K:K + n]

# =====================================================================
# GERCEK DIL YOLU -- sinir OGRENILIR, isaretlenmez
# =====================================================================
# Kullanici, 20 Eylul: *"gercek dil yapisina uygun yapalim"*.
#
# Yukaridaki `Sozluk` siniri KESME ISARETINDEN okuyor. Gercek Turkcede
# oyle bir isaret yok: `evlerimdekilerden` yedi parca ve hicbiri
# isaretli degil. Ustelik parca sayisi SABIT degil -- uc yuvali
# concat'e sigmaz.
#
# Burada parcalama KORPUSTAN OGRENILIYOR (BPE) ve parcalar diziye AYRI
# AYRI yaziliyor. "Kelime" diye bir birim kalmiyor; her parca kendi
# basina bir birim, kendi koordinatiyla.


class BPE:
    """Alt-kelime parcalama, korpustan ogrenilir.

    Egitim TIP sozlugu uzerinde, siklikla agirliklandirilmis -- bizde
    ~1.300 tip var, yani saniyeler. Gercek korpusta da BPE boyle egitilir."""

    SON = "</w>"

    def __init__(self, say, n_birlesme=400):
        kel = {w: list(w) + [self.SON] for w in say}
        frek = dict(say)
        self.birlesme = []
        for _ in range(n_birlesme):
            cift = collections.Counter()
            for w, p in kel.items():
                f = frek[w]
                for a, b in zip(p[:-1], p[1:]):
                    cift[(a, b)] += f
            if not cift:
                break
            en, n = cift.most_common(1)[0]
            if n < 2:
                break
            self.birlesme.append(en)
            yeni = en[0] + en[1]
            for w, p in kel.items():
                if len(p) < 2:
                    continue
                o, i = [], 0
                while i < len(p):
                    if i + 1 < len(p) and (p[i], p[i + 1]) == en:
                        o.append(yeni)
                        i += 2
                    else:
                        o.append(p[i])
                        i += 1
                kel[w] = o
        self.sira = {c: i for i, c in enumerate(self.birlesme)}
        self._on = {}
        self.parca_ad = sorted({x for p in kel.values() for x in p})
        self.px = {p: i for i, p in enumerate(self.parca_ad)}

    def parcala(self, w):
        """kelime -> parca listesi. Ogrenilen birlesmeler SIRAYLA."""
        if w in self._on:
            return self._on[w]
        p = list(w) + [self.SON]
        while len(p) > 1:
            c = [(self.sira.get((a, b), 1 << 30), i)
                 for i, (a, b) in enumerate(zip(p[:-1], p[1:]))]
            r, i = min(c)
            if r == 1 << 30:
                break
            p = p[:i] + [p[i] + p[i + 1]] + p[i + 2:]
        self._on[w] = p
        return p


class ParcaSozluk:
    """GERCEK DIL sozlugu: birim PARCA, kelime degil.

    `wp` (V, 1) -- TEK yuva, yani koordinat dogrudan tablodan okunur,
    concat yok. Model tarafinda ayni kod calisir; degisen tek sey yuva
    sayisi. Iki kurulum boylece AYNI kodla kiyaslanabiliyor."""

    def __init__(self, diziler, n_birlesme=400):
        say = collections.Counter()
        for s in diziler:
            say.update(s.split())
        self.bpe = BPE(say, n_birlesme)
        self.kelime = self.bpe.parca_ad
        self.kx = self.bpe.px
        self.wp = np.arange(len(self.kelime), dtype=np.int64)[:, None]
        self.kok_ad = self.kelime
        self.ek_ad = self.nok_ad = ()
        self.say = say

    @property
    def V(self):
        return len(self.kelime)

    def kodla(self, diziler):
        out = []
        for s in diziler:
            for w in s.split():
                for p in self.bpe.parcala(w):
                    out.append(self.kx[p])
        return np.array(out, np.int64)

    def ozet(self):
        n_kel = len(self.say)
        top = sum(self.say.values())
        ort = sum(len(self.bpe.parcala(w)) * c for w, c in self.say.items())
        return (f"PARCA sozlugu {self.V:,} birim   (kelime tipi {n_kel:,})"
                f"   kelime basina {ort/top:.2f} parca")
