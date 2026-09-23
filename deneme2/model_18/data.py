# -*- coding: utf-8 -*-
"""data -- DUZ MATEMATIK: toplama, model_15'in verisiyle (model_17 veri_mat17'den).

Kullanici, 24 Eylul: "bunları kopyala ama matematik kopyala veri olarak ve
adını data yap".

    <eos> 4 7 2 + 1 8 2 = 6 5 4 <eos>
    Kayip YALNIZ cevabin rakamlarinda ve EOS'ta; soru rakamlari rastgele, hedef degil.

Veri model_15'in Drive'daki dosyalarindan (kural 9); bolme AYNEN onlarinki.
Iz yuklerken YENIDEN hesaplanip model_15'in kaydiyla karsilastirilir.

OLCUT model_15'inkiyle AYNI (sor): soru sorulur, EOS'a kadar serbest
uretim, cevap BIREBIR -- erken durmak, fazla rakam, yanlis rakam YANLIS.
Ek olarak ILK rakam: en buyuk basamak butun eldelere bagli, en zor secim.

Kayitli taban (eski mimari, 4.529 parametre, Drive model_15/*/kayit.txt):
    DUR  2 terim   tutulan 0,7770   uzunluk 0,9916
    COK  2+3 terim tutulan 0,4403   3 terimde 0,3693
"""
from __future__ import annotations

import hashlib

import torch
import torch.nn.functional as F

SON = "<eos>"
ARTI, ESIT, EOS = 10, 11, 12
N = 13
AD = [str(i) for i in range(10)] + ["+", "=", SON]

# model_15'in kayit.txt'sindeki izler -- dosya kayarsa yukle() durur.
IZ = {"dur": "eb4c73ab05178e18", "cok": "8d0f89938c67e0e7"}


def rak(x):
    """Sayinin KENDI rakamlari.  Dolgu YOK."""
    return [int(c) for c in str(x)]


def soru(ts):
    """ts: terimler -> d..d + d..d [+ d..d] ="""
    w = []
    for i, x in enumerate(ts):
        if i:
            w.append(ARTI)
        w += rak(x)
    return w + [ESIT]


def yukle(yol, ad):
    """model_15'in veri dosyasi -> (egitim sorulari, tutulan sorular).
    ad: "dur" (2 terim) ya da "cok" (2+3 terim)."""
    d = torch.load(yol, weights_only=False)
    hh = hashlib.sha256()
    for w, _, u in d["eg"] + d["tu"]:
        hh.update(w.numpy().tobytes())
        hh.update(u.numpy().tobytes())
    iz = hh.hexdigest()[:16]
    if iz != IZ[ad]:
        raise ValueError(f"veri izi tutmuyor: {iz} != {IZ[ad]} ({yol})")
    eg = d["soru_eg"] if "soru_eg" in d else d["cift_eg"]
    tu = d["soru_tu"] if "soru_tu" in d else d["cift_tu"]
    return [tuple(x) for x in eg], [tuple(x) for x in tu]


def pencereler(sorular):
    """(W, M, H), n x T.  W: <eos> soru cevap <eos>, sagdan dolgu.
    M: gercek token'lar.  H: HEDEF konumlar -- cevabin rakamlari ve EOS."""
    diz = [[EOS] + soru(ts) + rak(sum(ts)) + [EOS] for ts in sorular]
    T = max(len(d) for d in diz)
    W = torch.full((len(diz), T), EOS, dtype=torch.long)
    M = torch.zeros(len(diz), T, dtype=torch.bool)
    H = torch.zeros(len(diz), T, dtype=torch.bool)
    for i, (d, ts) in enumerate(zip(diz, sorular)):
        W[i, :len(d)] = torch.tensor(d)
        M[i, :len(d)] = True
        H[i, len(d) - len(rak(sum(ts))) - 1:len(d)] = True
    return W, M, H


def _ornekle(sorular, en, tohum=12345):
    """Olcum alt kumesi RASTGELE, tohum sabit -- her kosu ayni kumeyi olcer."""
    if en >= len(sorular):
        return list(sorular)
    g = torch.Generator().manual_seed(tohum)
    return [sorular[j] for j in torch.randperm(len(sorular), generator=g)[:en]
            .tolist()]


def _uret(m, sorular, aygit, parca=4096):
    """Serbest uretim: (soru, uretilen K token) ciftleri.  EOS'ta kesmez,
    K = en uzun cevap + 1 adim yurur; nerede durdugunu sor() okur."""
    kova = {}
    for ts in sorular:
        kova.setdefault(len(soru(ts)), []).append(ts)
    K = max(len(rak(sum(ts))) for ts in sorular) + 1
    cik = []
    with torch.no_grad():
        for grup in kova.values():
            for i in range(0, len(grup), parca):
                oh = grup[i:i + parca]
                yol = torch.tensor([[EOS] + soru(ts) for ts in oh],
                                   device=aygit)
                for _ in range(K):
                    t = m.scoreboard(yol)[:, -1].argmax(-1)
                    yol = torch.cat([yol, t[:, None]], 1)
                cik += [(ts, yol[r, -K:].tolist()) for r, ts in enumerate(oh)]
    return cik, K


def sor(m, sorular, aygit="cuda", en=20000):
    """TEK OLCUT: soru soruldu, cevap DOGRU MU.  Doner: accuracy, length_ok, first_digit, n.

      accuracy     cevap BIREBIR ve EOS dogru yerde
      length_ok    rakamlardan bagimsiz, DOGRU YERDE durdu mu
      first_digit  cevabin ILK rakami dogru mu (en buyuk basamak)"""
    sorular = _ornekle(sorular, en)
    cik, K = _uret(m, sorular, aygit)
    dog = uzn = ilk = 0
    for ts, c in cik:
        hedef = rak(sum(ts))
        dur = c.index(EOS) if EOS in c else K
        uzn += dur == len(hedef)
        dog += dur == len(hedef) and c[:len(hedef)] == hedef
        ilk += c[0] == hedef[0]
    n = len(cik)
    return {"accuracy": dog / n, "length_ok": uzn / n, "first_digit": ilk / n, "n": n}


def cevap_ce(m, WMH, aygit="cuda", parca=4096):
    """Cevap token'larinda (rakamlar + EOS) ortalama CE."""
    W, M, H = WMH
    top = say = 0.0
    with torch.no_grad():
        for i in range(0, len(W), parca):
            w, mk = W[i:i + parca].to(aygit), M[i:i + parca].to(aygit)
            a = H[i:i + parca, 1:].to(aygit)
            p = m.scoreboard(w, mk)[:, :-1]
            ce = F.cross_entropy(p.transpose(1, 2), w[:, 1:], reduction="none")
            top += float(ce[a].sum())
            say += int(a.sum())
    return top / max(say, 1)


def olcut(eg, tu, aygit="cuda", en=2000):
    """train'in bekledigi metric(m, "train"|"heldout", full) -> dict.
      accuracy  ANA OLCUT: cevap birebir dogru (exact match)
      ce        cevap token'larinda kayip (ayni alt kume)
      diag      ANALIZ icin, matematige ozgu: first_digit, length_ok"""
    kume = {"train": eg, "heldout": tu}

    def f(m, side, full=False):
        s = kume[side]
        alt = _ornekle(s, len(s) if full else en)
        r = sor(m, alt, aygit=aygit, en=len(alt))
        return {"accuracy": r["accuracy"], "ce": cevap_ce(m, pencereler(alt), aygit),
                "diag": {"first_digit": r["first_digit"], "length_ok": r["length_ok"]}}

    return f


def kirilim(m, sorular, aygit="cuda", olcu="terim"):
    """{anahtar: sor sonucu}.  olcu "terim" (2/3) ya da "hane" (cevap hanesi)."""
    f = (lambda ts: len(ts)) if olcu == "terim" else (lambda ts: len(rak(sum(ts))))
    g = {}
    for ts in sorular:
        g.setdefault(f(ts), []).append(ts)
    return {a: sor(m, c, aygit=aygit, en=len(c)) for a, c in sorted(g.items())}


KONUM = ("birler", "onlar", "yuzler", "binler")


def basamak(m, sorular, aygit="cpu", parca=4096):
    """TANI, olcut degil: HANGI BASAMAK ogrenildi.  Anahtar (cevap hanesi,
    konum); konum SAGDAN (birler, onlar, yuzler, binler) ya da "eos".
      og    dogru onek verilince o token dogru mu (ogretmen zorlamali)
      ser   serbest uretim, SAGA hizali: modelin yazdigi o basamak dogru mu
      cog   taban: o yuvada hep EN SIK token'i soyleyen
      n     soru sayisi"""
    W, M, _ = pencereler(sorular)
    L = torch.tensor([len(rak(sum(ts))) for ts in sorular])
    a0 = M.sum(1) - L - 1                     # cevabin ilk rakaminin yeri
    tah = []
    with torch.no_grad():
        for i in range(0, len(W), parca):
            tah.append(m.scoreboard(W[i:i + parca].to(aygit), M[i:i + parca]
                                    .to(aygit)).argmax(-1).cpu())
    tah = torch.cat(tah).tolist()
    yaz = {ts: c[:c.index(EOS)] if EOS in c else c
           for ts, c in _uret(m, sorular, aygit)[0]}
    say = {}

    def ekle(k, og, ser, h):
        s = say.setdefault(k, {"og": 0, "ser": 0, "n": 0, "sik": {}})
        s["og"] += og
        s["ser"] += ser
        s["n"] += 1
        s["sik"][h] = s["sik"].get(h, 0) + 1

    for i, ts in enumerate(sorular):
        c, b, g, t = rak(sum(ts)), int(a0[i]), yaz[ts], tah[i]
        for q, h in enumerate(c):
            r = len(c) - 1 - q
            ekle((len(c), KONUM[r]), t[b + q - 1] == h,
                 r < len(g) and g[len(g) - 1 - r] == h, h)
        ekle((len(c), "eos"), t[b + len(c) - 1] == EOS, len(g) == len(c), EOS)
    return {k: {"og": s["og"] / s["n"], "ser": s["ser"] / s["n"],
                "cog": max(s["sik"].values()) / s["n"], "n": s["n"]}
            for k, s in say.items()}


def basamak_tablo(r, yaz=print):
    yaz("  hane  konum         n       og      ser      cog")
    for L in sorted({k[0] for k in r}):
        for ad in KONUM[:L][::-1] + ("eos",):
            v = r[(L, ad)]
            yaz("  %4d  %-7s %8d   %6.4f   %6.4f   %6.4f"
                % (L, ad, v["n"], v["og"], v["ser"], v["cog"]))


def goster(m, sorular, aygit="cpu"):
    """GOZLE: soru, dogru cevap, modelin yazdigi."""
    cik, _ = _uret(m, list(sorular), aygit)
    for ts, c in cik:
        dur = c.index(EOS) if EOS in c else len(c)
        yaz = "".join(AD[t] for t in c[:dur]) + ("" if dur < len(c) else " (durmadi)")
        print(f"  {' + '.join(map(str, ts))} = {sum(ts):<5}  model: {yaz}"
              f"  {'DOGRU' if yaz == str(sum(ts)) else 'yanlis'}")
