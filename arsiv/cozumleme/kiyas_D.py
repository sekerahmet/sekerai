# -*- coding: utf-8 -*-
"""
Deney 6 — D kollarinin A+ ile ESLESTIRILMIS kiyasi.

DENEY6_ONKAYIT_D_vs_A.md sec. 3-4'te onceden kayitli protokolun kodu.
KOSUDAN ONCE yazildi: analiz secimi yok.

Kullanim:
    python kiyas_D.py --artiA out/artiA --d1 out/D1 --d2 out/D2 [--a out/A]

Neden eslestirilmis test: uc kol da AYNI agirliklardan basliyor ve AYNI
degerlendirme ornekleriyle olculuyor. "Iki orani karsilastirmak" bu bilgiyi
cope atar. Ornek bazinda "A+'da yanlis, D'de dogru" sayilari cok daha guclu.

Neden VARLIK duzeyinde: 3000 ENT sorusu yalnizca 100 varliktan geliyor.
Bagimsiz birim 100. Ornek duzeyinde McNemar p'yi ~5 kat kucuk gosterir;
raporlanir ama KARAR ONA DAYANMAZ.
"""
import argparse, json, math, os
import numpy as np

BIRINCIL_ESIK = 0.10      # ENT(D) - ENT(A+) >= 0.10
BIRINCIL_P    = 0.01
KAPILAR = dict(one=0.99, comp=0.80, ident=0.95, ident_ent=0.95)


def tablo_oku(d, arm="A", seed=0):
    """kayit_{arm}_s{seed}.parquet (ya da .npz) -> sutun sozlugu."""
    kok = os.path.join(d, f"kayit_{arm}_s{seed}")
    if os.path.exists(kok + ".parquet"):
        import pandas as pd
        return {k: df.to_numpy() for k, df in pd.read_parquet(kok + ".parquet").items()}
    if os.path.exists(kok + ".npz"):
        z = np.load(kok + ".npz", allow_pickle=True)
        return {k: z[k] for k in z.files}
    raise FileNotFoundError(kok + ".parquet|.npz")


def son_adim_kume(cols, kume):
    """Son adimdaki `kume` satirlari -> (anahtar, ok) ve varlik dizisi."""
    ks = np.asarray([str(x) for x in cols["kume"]])
    st = np.asarray(cols["step"], np.int64)
    m = ks == kume
    if not m.any():
        return None
    son = st[m].max()
    m &= (st == son)
    e  = np.asarray(cols["e"],  np.int64)[m]
    r1 = np.asarray(cols["r1"], np.int64)[m]
    r2 = np.asarray(cols["r2"], np.int64)[m]
    ok = np.asarray(cols["ok"], bool)[m]
    anahtar = e.astype(np.int64) * 1_000_000 + r1 * 1000 + r2
    return dict(adim=int(son), anahtar=anahtar, e=e, ok=ok)


def hizala(A, B, ad):
    """Iki kolun ayni ornekleri -> ayni sirada. Eslesmezse durur."""
    assert len(A["anahtar"]) == len(B["anahtar"]), \
        f"{ad}: ornek sayisi farkli {len(A['anahtar'])} vs {len(B['anahtar'])}"
    ia, ib = np.argsort(A["anahtar"]), np.argsort(B["anahtar"])
    assert np.array_equal(A["anahtar"][ia], B["anahtar"][ib]), \
        f"{ad}: degerlendirme kumeleri AYNI DEGIL -> eslestirilmis test gecersiz"
    return A["ok"][ia], B["ok"][ib], A["e"][ia]


def isaret_permutasyon(d, n=20000, rng=None):
    """Varlik basina farklarin isaretini rastgele cevir -> iki yonlu p.
    Eslestirilmis tasarimin dogal testi; dagilim varsayimi yok."""
    rng = rng or np.random.RandomState(0)
    d = d[d != 0]
    if len(d) == 0:
        return 1.0, 0
    goz = abs(d.mean())
    s = rng.choice([-1.0, 1.0], size=(n, len(d)))
    bos = np.abs((s * d).mean(1))
    return float((np.sum(bos >= goz - 1e-12) + 1) / (n + 1)), len(d)


def onyukleme(d, n=20000, rng=None):
    rng = rng or np.random.RandomState(1)
    idx = rng.randint(0, len(d), size=(n, len(d)))
    o = d[idx].mean(1)
    return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def mcnemar_p(b, c):
    """Ornek duzeyi, ANTI-MUHAFAZAKAR (kumelenmeyi yok sayar)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    kuy = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return float(min(1.0, 2 * kuy))


def kume_kiyas(kontrol, deney, ad, kume):
    z = son_adim_kume(kontrol, kume)
    y = son_adim_kume(deney, kume)
    if z is None or y is None:
        print(f"  {kume}: kume bulunamadi, atlandi")
        return None
    okA, okD, ev = hizala(z, y, f"{ad}/{kume}")

    accA, accD = okA.mean(), okD.mean()
    fark = accD - accA

    # --- VARLIK duzeyi (KARAR BURADAN) ---
    vs = np.unique(ev)
    dv = np.array([okD[ev == v].mean() - okA[ev == v].mean() for v in vs])
    p_perm, n_nz = isaret_permutasyon(dv)
    lo, hi = onyukleme(dv)

    # --- ornek duzeyi McNemar (yalnizca referans) ---
    b = int(np.sum(~okA & okD)); c = int(np.sum(okA & ~okD))
    p_mc = mcnemar_p(b, c)

    print(f"  {kume:5s} @adim {z['adim']}   A+ {accA:.4f} -> {ad} {accD:.4f}   "
          f"fark {fark:+.4f}")
    print(f"        varlik duzeyi (n={len(vs)}, sifir-olmayan {n_nz}): "
          f"ort fark {dv.mean():+.4f}  %95 GA [{lo:+.4f}, {hi:+.4f}]  p={p_perm:.4g}")
    print(f"        ornek duzeyi: A+yanlis/D dogru {b}, A+dogru/D yanlis {c}, "
          f"McNemar p={p_mc:.3g}  [ANTI-MUHAFAZAKAR, karar buna dayanmaz]")
    return dict(kume=kume, adim=z["adim"], accA=float(accA), accD=float(accD),
                fark=float(fark), varlik_ort=float(dv.mean()), ga=[lo, hi],
                p_varlik=p_perm, n_varlik=int(len(vs)), b=b, c=c, p_mcnemar=p_mc)


def egri_son(d, arm="A", seed=0):
    y = os.path.join(d, f"egri_{arm}_s{seed}.json")
    if not os.path.exists(y):
        return None
    k = json.load(open(y, encoding="utf-8"))
    return k[-1] if k else None


def saglik(rec, ad):
    if rec is None:
        print(f"  {ad}: egri yok, saglik kapilari OLCULEMEDI")
        return False
    dus = []
    for k, esik in KAPILAR.items():
        if k not in rec:
            if k.startswith("ident"):
                continue           # A+ kolunda kimlik gorevi yok, dogal
            dus.append(f"{k} YOK")
            continue
        if float(rec[k]) < esik:
            dus.append(f"{k}={rec[k]:.4f} < {esik}")
    if dus:
        print(f"  SAGLIK KAPISI DUSTU ({ad}): " + "; ".join(dus))
        print(f"  -> sonuc YORUMLANAMAZ ('olumsuz' DEGIL)")
        return False
    var = [f"{k}={rec[k]:.4f}" for k in KAPILAR if k in rec]
    print(f"  saglik kapilari TAMAM ({ad}): " + "  ".join(var))
    return True


def kol_degerlendir(kontrol, ad, dizin, artiA_dizin):
    print(f"\n{'='*72}\n{ad}  vs  A+\n{'='*72}")
    deney = tablo_oku(dizin)
    r_saglik = saglik(egri_son(dizin), ad)
    sonuc = {k: kume_kiyas(kontrol, deney, ad, k) for k in ("ent", "ent2", "comp")}

    e = sonuc.get("ent")
    if e is None:
        print(f"\n  HUKUM ({ad}): ENT olculemedi")
        return dict(kol=ad, hukum="olculemedi", **sonuc)

    gecti = (e["fark"] >= BIRINCIL_ESIK) and (e["p_varlik"] < BIRINCIL_P)
    if not r_saglik:
        hukum = "GECERSIZ (saglik kapisi)"
    elif gecti:
        hukum = "GECTI"
    else:
        hukum = "GECMEDI"

    print(f"\n  BIRINCIL: ENT farki {e['fark']:+.4f} (esik +{BIRINCIL_ESIK:.2f})  "
          f"p={e['p_varlik']:.4g} (esik {BIRINCIL_P})")
    print(f"  HUKUM ({ad}): {hukum}")
    print(f"  [ikincil] ENT ham degeri {e['accD']:.4f}  "
          f"(A 0.029 | v1 citasi 0.20 | iliski ekseni 0.732)")
    if ad == "D2":
        print("  [kayit] D2 ENT bolmesinin tanimini gevsetiyor: her (e,r1)")
        print("          artik ikinci hop'u onemsiz bir zincirin birinci hop'u.")
        print("          Bkz. DENEY6_ONKAYIT_D_vs_A.md sec. 6.2")
    return dict(kol=ad, hukum=hukum, saglik=r_saglik, **sonuc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artiA", required=True, help="kontrol kolu cikti dizini")
    ap.add_argument("--d1", default="", help="IDENT_MODE=q1 cikti dizini")
    ap.add_argument("--d2", default="", help="IDENT_MODE=q2son cikti dizini")
    ap.add_argument("--a", default="", help="ozgun A kosusu (referans, istege bagli)")
    ap.add_argument("--json", default="", help="sonucu bu dosyaya yaz")
    a = ap.parse_args()

    print("DENEY 6 — D vs A+  (onceden kayitli protokol)")
    print("Karar: ENT(D) - ENT(A+) >= +0.10 VE varlik-kumeli p < 0.01\n")

    kontrol = tablo_oku(a.artiA)
    ka = egri_son(a.artiA)
    if ka:
        print(f"A+ (kontrol) @adim {ka['step']}: " + "  ".join(
            f"{k}={ka[k]:.4f}" for k in ("one", "comp", "ent", "ent2") if k in ka))
    if a.a:
        kb = egri_son(a.a)
        if kb:
            print(f"A  (ozgun)  @adim {kb['step']}: " + "  ".join(
                f"{k}={kb[k]:.4f}" for k in ("one", "comp", "ent", "ent2") if k in kb))
            print("  -> A+ ile A arasindaki fark, 30.000 ek adimin TEK BASINA etkisi")

    cikti = []
    for ad, d in (("D1", a.d1), ("D2", a.d2)):
        if d:
            cikti.append(kol_degerlendir(kontrol, ad, d, a.artiA))

    print(f"\n{'='*72}\nOZET\n{'='*72}")
    for c in cikti:
        e = c.get("ent") or {}
        print(f"  {c['kol']:3s}  ENT {e.get('accD', float('nan')):.4f}  "
              f"fark {e.get('fark', float('nan')):+.4f}  "
              f"p {e.get('p_varlik', float('nan')):.4g}   -> {c['hukum']}")
    if a.json:
        json.dump(cikti, open(a.json, "w", encoding="utf-8"), indent=1, default=float)
        print(f"\n  -> {a.json}")


if __name__ == "__main__":
    main()
