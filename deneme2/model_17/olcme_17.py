# -*- coding: utf-8 -*-
"""olcme_17 -- SONRAKI JETON.  Etiket yok: SAYI + TANI + METIN.

SAYI   dogrulama PERPLEXITY (exp CE) -- BIRINCIL.  softmax(-|o-E|^2) duzgun
       bir dagilim, CE onun NLL'si.  Dogruluk yalniz en ustteki tahmini
       gorur; TAM1'de o duz gorunurken perplexity dusuyordu (onkayit §A1).
       Hedefler TURUNE gore ayrilir ki surumler ayni hedefte kiyaslansin
       (kullanici: "bizim ölçümlerimiz kendi içinde olmalı"):
         bas    BOS'tan sonraki ilk kelime        BOS'suz veride yok
         govde  hikayenin 2..L. kelimeleri        TAM1'le AYNI hedefler
         son    EOS: hikaye bitti                 BOS'suz veride yok
TANI   mimarinin her parcasina ayri geri bildirim -- tani().
METIN  istemden URETIM, GOZLE (konus.py).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _ac(X):
    """(W, M) ya da tek W -- ikisini de kabul et."""
    return X if isinstance(X, (tuple, list)) else (X, None)


def bos_kimligi(W, M):
    """Pencereler <hikaye> ile basliyor VE bitiyorsa onun kimligi, yoksa None."""
    if M is None or len(W) == 0:
        return None
    t0 = W[0, 0]
    if not bool((W[:, 0] == t0).all()):
        return None
    son = M.sum(1).long() - 1
    return int(t0) if bool((W[torch.arange(len(W)), son] == t0).all()) else None


def _turler(a, h, hk):
    """Hedef maskeleri (B,T-1): bas, govde, son.  hk None ise hepsi govde."""
    if hk is None:
        z = torch.zeros_like(a)
        return {"bas": z, "govde": a, "son": z}
    bas = torch.zeros_like(a)
    bas[:, 0] = a[:, 0]
    son = a & (h == hk)
    return {"bas": bas, "govde": a & ~bas & ~son, "son": son}


def olc(m, W, parca: int = 256, aygit=None, hk=None) -> dict:
    """Dogruluk ve CE, hedef turune gore de.  W CPU'da olabilir: parca parca
    tasinir (tam korpusta egitim penceresi GPU'ya sigmaz)."""
    W, M = _ac(W)
    aygit = aygit or next(m.parameters()).device
    dg = tp = 0
    top = {k: [0.0, 0] for k in ("bas", "govde", "son")}
    with torch.no_grad():
        for i in range(0, W.shape[0], parca):
            w = W[i:i + parca].to(aygit).long()
            mk = None if M is None else M[i:i + parca].to(aygit)
            p = m.dizi(w, mk)[:, :-1]
            h = w[:, 1:]
            a = (torch.ones_like(h, dtype=torch.bool) if mk is None
                 else mk[:, 1:])
            ce = F.cross_entropy(p.transpose(1, 2), h, reduction="none")
            dg += int(((p.argmax(-1) == h) & a).sum())
            tp += int(a.sum())
            for k, s in _turler(a, h, hk).items():
                top[k][0] += float(ce[s].sum())
                top[k][1] += int(s.sum())
    r = {"dogruluk": dg / max(tp, 1),
         "ce": sum(v[0] for v in top.values()) / max(tp, 1)}
    for k, (c, n) in top.items():
        r["n_" + k] = n
        r["ce_" + k] = c / n if n else None
    return r


def dogruluk(m, W, parca: int = 256, aygit=None) -> float:
    """Sonraki jeton dogrulugu (eski arayuz)."""
    return olc(m, W, parca, aygit)["dogruluk"]


def olcut(EG, DG, aygit="cuda", en=2000):
    """train_17'nin bekledigi bicim:  olcut(m, "eg"|"dg", tam=False) -> dict.

    `en` egitim SIRASINDA ornek sayisini kisar; son olcum TAM veriyle.
    BOS/EOS bir kez, butun pencerelere bakilarak anlasilir."""
    kume = {"eg": _ac(EG), "dg": _ac(DG)}
    hk = {t: bos_kimligi(*kume[t]) for t in kume}

    def f(m, taraf, tam=False):
        W, M = kume[taraf]
        if not tam:
            W, M = W[:en], (None if M is None else M[:en])
        return olc(m, (W, M), aygit=aygit, hk=hk[taraf])

    return f


def _konum_say(m, W, aygit="cuda", parca=256):
    """Konum basina (dogru, hedef) sayilari."""
    W, M = _ac(W)
    T = W.shape[1]
    dg, tp = torch.zeros(T - 1), torch.zeros(T - 1)
    with torch.no_grad():
        for i in range(0, W.shape[0], parca):
            w = W[i:i + parca].to(aygit).long()
            mk = None if M is None else M[i:i + parca].to(aygit)
            t = m.dizi(w, mk)[:, :-1].argmax(-1)
            d = t == w[:, 1:]
            a = torch.ones_like(d) if mk is None else mk[:, 1:]
            dg += (d & a).sum(0).float().cpu()
            tp += a.sum(0).float().cpu()
    return dg, tp


def kirilim(m, W, ad=None, aygit="cuda", parca=256):
    """Dogrulugu KONUMA gore boler (hedefi olmayan konum 0)."""
    dg, tp = _konum_say(m, W, aygit, parca)
    return (dg / tp.clamp(min=1)).tolist()


def tablo(m, W, aygit="cuda", yaz=print, dilim=8):
    """Konum dilimlerine gore dogruluk.  Dilim ortalamasi HEDEF SAYISIYLA
    agirlikli; eskiden konumlar esit sayiliyor, bos konum 0 giriyordu."""
    dg, tp = _konum_say(m, W, aygit)
    n = len(dg)
    yaz(f"{'konum':>12} {'dogruluk':>10} {'hedef':>9}")
    yaz("-" * 33)
    for k in range(dilim):
        a, b = k * n // dilim, (k + 1) * n // dilim
        s = float(tp[a:b].sum())
        yaz(f"{f'{a}-{b}':>12} {float(dg[a:b].sum()) / max(s, 1):>10.4f} "
            f"{int(s):>9,}")
    yaz("-" * 33)
    yaz(f"{'TUMU':>12} {float(dg.sum() / tp.sum().clamp(min=1)):>10.4f} "
        f"{int(tp.sum()):>9,}")
    return (dg / tp.clamp(min=1)).tolist()


def tani(m, DG, n=256, aygit="cpu", yaz=print, tohum=0):
    """Mimariye GERI BILDIRIM -- her parca kendi isini yapiyor mu.

    kayip        hedef turune ve konuma gore
    durum        konum 10'daki kelime degisince durumdaki izi kac token surer
    attention    kac konum ateslenir, kutle ne kadar uzakta
    uzak baglam  son k token ayni, oncesi BASKA hikaye: CE ne kadar artar
    bitis        EOS gercek sonda ne olasilik aliyor, govdede kac kez one cikiyor
    Olcumdur: kullanici onayiyla kosulur (CLAUDE.md kural 0).  Ayni n ve
    tohumla her kosu AYNI orneklemde olculur."""
    W, M = _ac(DG)
    W, M = W.cpu(), (torch.ones_like(W, dtype=torch.bool) if M is None
                     else M.cpu())
    hk = bos_kimligi(W, M)
    L = M.sum(1)
    r = {"n": n}
    w, mk = W[:n].long(), M[:n]
    T = w.shape[1]

    r["olc"] = olc(m, (w, mk), aygit=aygit, hk=hk)
    wa, ma = w.to(aygit), mk.to(aygit)
    with torch.no_grad():
        ic = m.ic(wa, ma)                         # blok basina (durum, P, A, izin)
        lp = m.dizi(wa, ma).log_softmax(-1)
    ara = torch.arange(T, device=aygit)
    uzak = (ara[:, None] - ara[None, :]).clamp(min=0).float()
    hedef = torch.zeros_like(ma)
    hedef[:, :-1] = ma[:, 1:]
    ce = torch.zeros(w.shape, device=aygit)
    ce[:, :-1] = -lp[:, :-1].gather(-1, wa[:, 1:].unsqueeze(-1)).squeeze(-1)
    dilim = {"0-15": (0, 16), "16-63": (16, 64), "64-127": (64, 128),
             "128+": (128, T)}
    r["konum"], r["blok"] = {}, []
    for ad_, (a, b) in dilim.items():
        s = hedef[:, a:b]
        if bool(s.any()):
            r["konum"][ad_] = {"hedef": int(s.sum()),
                               "ppl": float(ce[:, a:b][s].mean().exp())}
    for _, P, A, izin in ic:
        bl = {}
        for ad_, (a, b) in dilim.items():
            s = hedef[:, a:b]
            if not bool(s.any()):
                continue
            kut = A[:, a:b].sum(-1).clamp(min=1e-12)
            bl[ad_] = {
                "ateslenen": float(((P[:, a:b] > 0) & izin[:, a:b]).sum(-1)[s]
                                   .float().mean()),
                "uzaklik": float(((A[:, a:b] * uzak[a:b]).sum(-1) / kut)[s]
                                 .mean())}
        r["blok"].append(bl)

    # durum: konum 10'daki kelime degisince, blok basina
    g = torch.Generator().manual_seed(tohum)
    sec = torch.nonzero(L >= 70).squeeze(1)[:128]
    if len(sec):
        w1 = W[sec].long()
        w2 = w1.clone()
        yeni = torch.randint(3, m.n, (len(sec),), generator=g)
        w2[:, 10] = torch.where(yeni == w1[:, 10], (yeni % (m.n - 3)) + 3, yeni)
        with torch.no_grad():
            H1 = [x[0] for x in m.ic(w1.to(aygit))]
            H2 = [x[0] for x in m.ic(w2.to(aygit))]
        r["durum_aci"] = []
        for a1, a2 in zip(H1, H2):
            aci = torch.rad2deg(torch.acos(
                F.cosine_similarity(a1, a2, dim=-1).clamp(-1, 1))).mean(0)
            r["durum_aci"].append({o: float(aci[10 + o])
                                   for o in (0, 1, 5, 10, 20, 50)})

    # uzak baglam: 181..220 arasi 40 hedef, oncesi baska hikaye
    uzn = torch.nonzero(L >= 221).squeeze(1)[:128]
    ver = torch.nonzero(L >= 181).squeeze(1)
    ver = ver[~torch.isin(ver, uzn)][:len(uzn)]
    if len(uzn) and len(ver) == len(uzn):
        Aw, Bw = W[uzn, :221].long(), W[ver, :221].long()

        def _ce(x):
            with torch.no_grad():
                q = m.dizi(x.to(aygit)).log_softmax(-1)[:, 180:220]
            return float(-q.gather(-1, Aw[:, 181:221].to(aygit).unsqueeze(-1))
                         .mean())
        r["uzak"] = {"hepsi": _ce(Aw)}
        for k in (16, 64, 128):
            r["uzak"][k] = _ce(torch.cat([Bw[:, :181 - k], Aw[:, 181 - k:]], 1))

    # bitis: gercek sonda EOS olasiligi, govdede erken EOS
    if hk is not None:
        t = _turler(mk[:, 1:], w[:, 1:], hk)
        p_hk = lp[:, :-1, hk].exp()
        r["bitis"] = {
            "p_eos_sonda": float(p_hk[t["son"].to(aygit)].mean()),
            "erken_eos": float((lp[:, :-1].argmax(-1) == hk)[t["govde"].to(aygit)]
                               .float().mean())}

    o = r["olc"]
    yaz(f"TANI  {getattr(m, 'mimari', '?')}   {n} pencere   "
        f"BOS/EOS {'VAR' if hk is not None else 'YOK'}")
    yaz(f"  perplexity  hepsi {math.exp(o['ce']):.3f}"
        + "".join(f"   {k} {math.exp(o['ce_' + k]):.3f} ({o['n_' + k]:,})"
                  for k in ("bas", "govde", "son") if o["ce_" + k] is not None))
    yaz(f"  {'konum':>8} {'hedef':>7} {'ppl':>8}" + "".join(
        f"   b{i + 1} ates  uzak" for i in range(len(r["blok"]))))
    for k, v in r["konum"].items():
        yaz(f"  {k:>8} {v['hedef']:>7,} {v['ppl']:>8.3f}" + "".join(
            f"   {bl[k]['ateslenen']:>7.1f} {bl[k]['uzaklik']:>5.1f}"
            for bl in r["blok"] if k in bl))
    for i, d_ in enumerate(r.get("durum_aci", [])):
        yaz(f"  durum izi b{i + 1} (derece)  " + "  ".join(
            f"+{o_}:{v:.1f}" for o_, v in d_.items()))
    if "uzak" in r:
        yaz("  uzak baglam CE  hepsi {:.3f}  ".format(r["uzak"]["hepsi"])
            + "  ".join(f"son {k}: {r['uzak'][k]:.3f}" for k in (16, 64, 128)))
    if "bitis" in r:
        yaz(f"  bitis  p(EOS) gercek sonda {r['bitis']['p_eos_sonda']:.3f}   "
            f"govdede erken EOS {r['bitis']['erken_eos']:.4f}")
    return r


def devam(m, onek, ad, ix, coz, adim=60, aygit="cuda", tohum=None,
          sicaklik=0.0, yasak=(), bos=None):
    """Istemin devamini URET.  Sayi degil METIN -- gozle okunur.

    sicaklik 0 -> hep en yakin token (belirlenimci).
    yasak       okunmayacak jeton kimlikleri.  <bilinmeyen> uretimde
                korpustaki oranin 4,8 katina cikiyor (olculdu) ve
                sicaklik 0'da ust uste kilitleniyor.
    bos         <hikaye> kimligi: BOS/EOS'la egitilmis modelde istemin basina
                konur, model onu URETINCE durulur.  None -> eski davranis.
    """
    # Uretec aygitla ayni yerde olmali; CPU ureteci CUDA'da multinomial'i dusurur.
    g = (None if tohum is None
         else torch.Generator(device=aygit).manual_seed(tohum))
    w = (([bos] if bos is not None else [])
         + [ix.get(t, ix["<bilinmeyen>"]) for t in onek])
    y = torch.tensor(list(yasak), dtype=torch.long, device=aygit)
    with torch.no_grad():
        for _ in range(adim):
            p = m.dizi(torch.tensor([w], device=aygit))[0, -1]
            if len(y):
                p = p.index_fill(0, y, -float("inf"))
            if sicaklik <= 0:
                c = int(p.argmax())
            else:
                c = int(torch.multinomial((p / sicaklik).softmax(-1), 1,
                                          generator=g))
            w.append(c)
            if bos is not None and c == bos:
                break
    return coz(torch.tensor(w[1:] if bos is not None else w), ad)
