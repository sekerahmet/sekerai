# -*- coding: utf-8 -*-
"""ASAMA 1 kosucu + TANI. CPU, numpy. GPU gerekmiyor (kural 2 disi).

Tani kismi kolun VARLIK SEBEBI: `model_11`de arizayi adlandirmak icin
soyad sondasi ve 1/48 aritmetigi gerekmisti. Burada koordinat 6 sayi --
BASILIR. Uc sey basiliyor:

  CAKISMA    ayni (iliski, hedef) ciftini paylasan varliklar UST USTE
             mi dustu?  q = c_e + Rel[r] TERCUMESI, cok-tan-bire
             iliskilerde c_e1 = c_e2 ZORLAR. Olcup gormek icin.
  TIP KIRILIMI  model_11'in KISI/TEZ/BOLUM tablosunun karsiligi.
  YURUYUS    belgedeki gibi GERCEK sayilarla tek bir zincir.
"""
import argparse, time
import numpy as np
from model_13 import Zincir, egit, sina, kisayol_hedefi
import taban_13 as MT, ayar_13 as AY


def tani(mdl, v, E_ad, REL, yaz=print):
    C = mdl.coord()
    n = len(C)
    # genel olcek: rastgele cift uzakligi
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, n, 4000), rng.integers(0, n, 4000)
    genel = float(np.sqrt(((C[i] - C[j]) ** 2).sum(-1)).mean())
    yaz(f"\n  genel varlik-cifti uzakligi (rastgele 4000)   {genel:.3f}")

    yaz("\n  CAKISMA -- ayni (iliski, hedef) ciftini paylasanlar")
    yaz(f"    {'iliski':<22}{'fan-in':>8}{'grup-ici':>10}{'oran':>8}")
    sat = []
    for r in range(v.n_rel):
        hed = v.facts[:, r]
        var = np.where(hed >= 0)[0]
        if len(var) < 4:
            continue
        gr = {}
        for e in var:
            gr.setdefault(int(hed[e]), []).append(e)
        cok = [g for g in gr.values() if len(g) > 1]
        if not cok:
            continue
        d = [float(np.sqrt(((C[a] - C[b]) ** 2).sum()))
             for g in cok for a, b in zip(g[:-1], g[1:])]
        fan = max(len(g) for g in cok)
        sat.append((np.mean(d), r, fan))
    for ic, r, fan in sorted(sat)[:8]:
        ad = v.__dict__.get("rel_ad", [f"r{r}"] * v.n_rel)[r] \
            if "rel_ad" in v.__dict__ else f"r{r}"
        yaz(f"    {ad:<22}{fan:>8}{ic:>10.3f}{ic/genel:>8.2f}")
    if sat:
        yaz("    ^ oran << 1 ise TERCUME cakisma uretmis (TransE'nin "
            "cok-tan-bire kirilmasi)")

    if v.tip is not None:
        yaz("\n  TIP KIRILIMI -- ayni tip icinde uzaklik")
        for t, ad in enumerate(v.tip_ad):
            ix = np.where(v.tip == t)[0]
            if len(ix) < 4:
                continue
            a, b = rng.choice(ix, 600), rng.choice(ix, 600)
            ic = float(np.sqrt(((C[a] - C[b]) ** 2).sum(-1)).mean())
            yaz(f"    {ad:<14}{len(ix):>6} varlik   ic-uzaklik {ic:>7.3f}"
                f"   oran {ic/genel:.2f}")


def yuruyus(mdl, v, zin, E_ad, yaz=print):
    """Belgedeki tablo: adim adim vektor ve uzaklik. GERCEK sayilar."""
    if not len(zin):
        return
    C = mdl.coord()
    e, r1, r2, kop, ans = (int(x) for x in zin[0][:5])
    f = lambda vec: " ".join(f"{x:6.2f}" for x in vec)
    q = C[e].copy()
    yaz(f"\n  YURUYUS  {E_ad[e]}")
    yaz(f"    {'ozne':<26}{f(q)}")
    q = q + mdl.Rel[r1]
    d = float(np.sqrt(((q - C[kop]) ** 2).sum()))
    yaz(f"    {'+ r1 = ara hedef':<26}{f(q)}   {E_ad[kop]}'a {d:.2f}")
    q = q + mdl.Rel[r2]
    d = float(np.sqrt(((q - C[ans]) ** 2).sum()))
    en = int(((q - C) ** 2).sum(-1).argmin())
    yaz(f"    {'+ r2 = son hedef':<26}{f(q)}   {E_ad[ans]}'a {d:.2f}")
    yaz(f"    {'-> en yakin':<26}{E_ad[en]}"
        f"   {'DOGRU' if en == ans else 'YANLIS'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kip", default="serbest", choices=("serbest", "parca"))
    p.add_argument("--D", type=int, default=3)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--epok", type=int, default=60)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--kaydet", default="")
    a = p.parse_args()

    A = AY.AYAR
    t0 = time.time()
    v = MT.veri_kur(A, yaz=lambda *x, **k: None)
    L = MT.olcme_listeleri(A, v)
    iz = MT.olcme_izi(L)
    assert iz == "44e6262e37f3", f"OLCME IZI KAYDI: {iz}"
    import veri_13 as V13
    G = V13.kur(A.veri_tohum)
    E_ad = [x for t in V13.TIPLER for x in G["ad"][t]]
    print(f"veri {time.time()-t0:.0f} sn   varlik {v.n_ent}  iliski {v.n_rel}"
          f"  olgu {len(v.one)}   olcme izi {iz}")

    mdl = Zincir(v, D=a.D, kip=a.kip)
    print(f"\nkip {a.kip}   D {a.D}   DD {mdl.DD}   "
          f"parametre {mdl.n_params():,}   lambda {a.lam}")
    if a.kip == "parca":
        print(f"  {v.n_ent} varlik  <-  {mdl.n_par} PAYLASILAN parca x "
              f"{mdl.yuva} yuva")

    # --- EGITIM: YALNIZ TEK ADIM. Iki adim HIC gosterilmiyor.
    O = np.array(v.one, np.int64)
    print(f"\negitim: {len(O):,} TEK ADIMLIK olgu  (iki adim HIC yok)")
    egit(mdl, O[:, 0], O[:, 1:2], O[:, 2], ep=a.epok, lam=a.lam,
         lr=a.lr, yaz=print)

    # --- SINAV: sabit olcme listeleri
    print(f"\n{'bolme':<10}{'n':>6}{'adim':>6}{'tam':>9}{'kisayol':>9}"
          f"{'kapanma':>9}")
    print("-" * 50)
    r = {}
    for bol, adim in (("one", 1), ("seen", 2), ("comp", 2), ("ent", 2),
                      ("ent_yok", 2), ("ood", 2)):
        z = L.get(bol) or []
        if not len(z):
            continue
        ksy = kisayol_hedefi(v, z) if adim == 2 else None
        tam, k, kap = sina(mdl, z, adim, ksy)
        r[bol] = (tam, k, kap)
        uy = int((ksy >= 0).sum()) if ksy is not None else 0
        print(f"{bol:<10}{len(z):>6}{adim:>6}{tam:>9.4f}"
              + (f"{k:>9.4f}" if uy else f"{'  -  ':>9}")
              + f"{kap:>9.3f}"
              + ("" if adim == 1 or uy else "   (kisayol UYGUN 0)"))

    if "ent_yok" in r:
        k = r["ent_yok"][1]
        ok = (not np.isfinite(k)) or abs(k) < 1e-9
        print(f"\nBIRIM TESTI  ent_yok kisayol = "
              + (f"{k:.4f}" if np.isfinite(k) else "n/a, 0 uygun ornek")
              + ("  GECTI" if ok else "  !! KALDI"))
    print(f"ON KOSUL     one {r.get('one',(0,))[0]:.4f}"
          f"   -> dusukse gerisi okunmaz")
    print(f"KONTROL      seen {r.get('seen',(0,))[0]:.4f} ile "
          f"comp {r.get('comp',(0,))[0]:.4f} YAKIN olmali -- iki adim "
          f"HICBIRINDE ogretilmedi")

    tani(mdl, v, E_ad, V13.ILISKI)
    if a.kaydet:
        # GORSEL icin koordinat + iliski + etiket. Egitim tekrarlanmasin.
        np.savez_compressed(
            a.kaydet, C=mdl.coord(), Rel=mdl.Rel, tip=v.tip,
            tip_ad=np.array(v.tip_ad, object), ad=np.array(E_ad, object),
            rel_ad=np.array(list(V13.ILISKI), object), facts=v.facts,
            comp=np.array([list(z) for z in (L.get("comp") or [])][:400]),
            kip=a.kip, epok=a.epok)
        print("kaydedildi -> " + a.kaydet)
    yuruyus(mdl, v, L.get("comp") or [], E_ad)
    print(f"\ntoplam {time.time()-t0:.0f} sn")


main()
