# -*- coding: utf-8 -*-
"""
ASAMA B2 — bolmeyi duzeltip prob'u yeniden calistirir. Model YUKLENMEZ:
kaydedilmis states.npy / eb.npy / ea.npy kullanilir (~1-2 dk, GPU sart degil).

NEDEN:
  kopru_cc.py bolmeyi ULKEYE gore yapiyordu: prob 49 ulkeyle egitilip hic gormedigi
  17 ulkeye genellemek zorundaydi. Tasarimin buna ihtiyaci yok — dongu icinde bellek
  tablosundaki ulkeler zaten bilinir. Gereken "bilinen tablodan dogru ulkeyi sec".
  Dogru kurulum KISIYE gore bolme: test kisileri egitimde gorulmez (ezberleme yine
  imkansiz) ama tum ulkeler egitimde temsil edilir.

  Ayrica: onceki permutasyon testi soru bazliydi, bootstrap ise kisi-kumeli.
  Tutarsizdi ve p'yi iyimser gosteriyordu. Burada ikisi de KISI-kumeli.

CIKTI: iki bolme yan yana, ayni istatistikle. Fark varsa sebebi budur.

Colab:
    %run asamaB2.py
"""
import os, io, re, json, glob, random
import numpy as np

WORK   = "/content/drive/MyDrive/kopru_cc"
SEED   = 0
N_BOOT = 4000
N_PERM = 4000
BAND   = (5, 25)
PCA_DIM = 256
LAMBDAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)

# ---------------------------------------------------------------- veri
def one(pat):
    g = sorted(glob.glob(os.path.join(WORK, pat)))
    if not g: raise SystemExit(f"bulunamadi: {pat} @ {WORK}")
    return g[-1]

kept = json.load(io.open(one("*kept.json"), encoding="utf-8"))
States = np.load(one("*states.npy")); EB = np.load(one("*eb.npy")); EA = np.load(one("*ea.npy"))
print("states:", States.shape, "| eb:", EB.shape, "| ea:", EA.shape)

DATEY = re.compile(r"^[\d\s\-/.,+]+$")
def bad_entity(e):
    e = str(e).strip()
    if len(e) < 2: return True
    return bool(DATEY.match(e)) or sum(c.isdigit() for c in e) > len(e) * 0.4

kb = [k for k in kept if not bad_entity(k["e3"]) and not bad_entity(k["bridge"])]
assert len(kb) == States.shape[0], f"kb {len(kb)} != states {States.shape[0]}"
LAYERS = list(range(States.shape[1]))
H = States.shape[2]

BR_LIST = sorted({k["bridge"] for k in kb}); BR_IX = {e: i for i, e in enumerate(BR_LIST)}
AN_LIST = sorted({k["e3"] for k in kb});     AN_IX = {e: i for i, e in enumerate(AN_LIST)}
def norml(E): return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
ENT = {"bridge": [norml(EB[:, t, :].astype(np.float32)) for t in range(EB.shape[1])],
       "answer": [norml(EA[:, t, :].astype(np.float32)) for t in range(EA.shape[1])]}
Y    = {"bridge": np.array([BR_IX[k["bridge"]] for k in kb]),
        "answer": np.array([AN_IX[k["e3"]]     for k in kb])}
CAND = {"bridge": np.arange(len(BR_LIST)), "answer": np.arange(len(AN_LIST))}
PERSON = np.array([k["person"] for k in kb])
print(f"{len(kb)} soru | {len(set(PERSON))} kisi | {len(BR_LIST)} ulke | {len(AN_LIST)} cevap")

# ---------------------------------------------------------------- ortak fonksiyonlar
def pca_basis(Xtr, k):
    mu = Xtr.mean(0); _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    return mu, Vt[:int(min(k, Vt.shape[0], Xtr.shape[0]-1))].T
def ridge_solve(G, B, lam): return np.linalg.solve(G + lam*np.eye(G.shape[0]), B)
def nrank(E, P, y, cand):
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    S = P @ E[cand].T
    out = np.empty(len(y))
    for i, yt in enumerate(y):
        g = S[i, yt]
        out[i] = ((S[i] > g).sum() + ((S[i] == g).sum()-1)/2) / max(1, len(cand)-1)
    return out
def med(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan
def bci_cluster(v, clu, salt=0, nb=N_BOOT):
    v = np.asarray(v, float); ok = np.isfinite(v)
    if ok.sum() < 5: return (np.nan, np.nan)
    uq = np.unique(clu[ok]); by = {c: np.where(ok & (clu == c))[0] for c in uq}
    r = np.random.RandomState((SEED+salt) % (2**31-1)); m = []
    for _ in range(nb):
        pick = uq[r.randint(0, len(uq), len(uq))]
        m.append(np.median(v[np.concatenate([by[c] for c in pick])]))
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
def perm_cluster(a, b, clu, salt=0, nperm=N_PERM):
    """KISI bazli isaret permutasyonu (onceki surum soru bazliydi -> iyimser p)."""
    uq = np.unique(clu); idx = {c: np.where(clu == c)[0] for c in uq}
    obs = med(b) - med(a)
    r = np.random.RandomState((SEED+salt) % (2**31-1)); cnt = 0
    for _ in range(nperm):
        flip = {c: r.rand() < 0.5 for c in uq}
        aa, bb = a.copy(), b.copy()
        for c in uq:
            if flip[c]:
                ii = idx[c]; aa[ii], bb[ii] = b[ii], a[ii]
        if abs(med(bb) - med(aa)) >= abs(obs) - 1e-12: cnt += 1
    return obs, (cnt + 1) / (nperm + 1)

def split_by(keyfn, salt):
    uniq = sorted({keyfn(k) for k in kb})
    rs = random.Random(SEED + salt); rs.shuffle(uniq)
    n1, n2 = int(.6*len(uniq)), int(.8*len(uniq))
    grp = {e: ("tr" if i < n1 else "va" if i < n2 else "te") for i, e in enumerate(uniq)}
    sp = np.array([grp[keyfn(k)] for k in kb])
    return sp == "tr", sp == "va", sp == "te", len(uniq)

def run_probe(tgt, TR, VA, TE):
    y, cand = Y[tgt], CAND[tgt]
    pte = np.random.RandomState(SEED + 77).permutation(int(TE.sum()))
    P_, N_ = [], []
    for j in LAYERS:
        X = States[:, j, :].astype(np.float32)
        mu, V = pca_basis(X[TR], PCA_DIM)
        Ztr, Zva, Zte = (X[TR]-mu)@V, (X[VA]-mu)@V, (X[TE]-mu)@V
        G = Ztr.T @ Ztr; best = None
        for ti, E in enumerate(ENT[tgt]):
            B = Ztr.T @ E[y[TR]]
            for lam in LAMBDAS:
                A_ = ridge_solve(G, B, lam)
                mv = float(np.median(nrank(E, Zva @ A_, y[VA], cand)))
                if best is None or mv < best[0]: best = (mv, A_, ti)
        _, A_, ti = best
        Pr = Zte @ A_
        P_.append(nrank(ENT[tgt][ti], Pr, y[TE], cand))
        N_.append(nrank(ENT[tgt][ti], Pr[pte], y[TE], cand))
    return np.array(P_), np.array(N_)

# ---------------------------------------------------------------- iki bolme
i0, i1 = BAND[0], BAND[1]
OUT = {}
for mode, keyfn, salt in (("ULKEYE gore (eski)", lambda k: k["bridge"], 5),
                          ("KISIYE gore (yeni)", lambda k: k["person"], 21)):
    TR, VA, TE, nu = split_by(keyfn, salt)
    PB = PERSON[TE]
    print("\n" + "=" * 88)
    print(f"BOLME: {mode}   benzersiz hedef {nu} -> train {TR.sum()} val {VA.sum()} test {TE.sum()}")
    print(f"  test: {TE.sum()} soru / {len(set(PB))} kisi | egitimde gorulen ulke: "
          f"{len({kb[i]['bridge'] for i in np.where(TR)[0]})}/{len(BR_LIST)}")
    Pb, Nb = run_probe("bridge", TR, VA, TE)
    bandB = np.median(Pb[i0:i1+1], axis=0); bandN = np.median(Nb[i0:i1+1], axis=0)
    cB, cN = bci_cluster(bandB, PB, 11), bci_cluster(bandN, PB, 12)
    d = bandN - bandB; cD = bci_cluster(d, PB, 13)
    obs, p = perm_cluster(bandB, bandN, PB, 14)
    print("-" * 88)
    print(f"  KOPRU band medyani : {med(bandB):.3f}  GA [{cB[0]:.3f}, {cB[1]:.3f}]")
    print(f"  NULL  band medyani : {med(bandN):.3f}  GA [{cN[0]:.3f}, {cN[1]:.3f}]")
    print(f"  ESLI FARK          : {med(d):+.3f}  GA [{cD[0]:+.3f}, {cD[1]:+.3f}]")
    print(f"  KISI-kumeli permutasyon: fark {obs:+.3f}, p = {p:.4f}")
    sig = (np.isfinite(cD[0]) and cD[0] > 0.005) and p < 0.05
    print(f"  -> {'KOPRU NULDEN IYI' if sig else 'ayirt edilemiyor'}")
    OUT[mode] = dict(n_test=int(TE.sum()), n_person=len(set(PB)),
                     band=med(bandB), ci=list(cB), null=med(bandN),
                     diff=med(d), ci_diff=list(cD), p=p, sig=bool(sig),
                     per_layer=[float(np.median(Pb[j])) for j in LAYERS])

print("\n" + "=" * 88)
print("KARSILASTIRMA")
print("=" * 88)
for m, o in OUT.items():
    print(f"  {m:22s}: band {o['band']:.3f}  fark {o['diff']:+.3f} "
          f"GA[{o['ci_diff'][0]:+.3f},{o['ci_diff'][1]:+.3f}]  p={o['p']:.4f}  "
          f"{'ANLAMLI' if o['sig'] else '-'}")
print("""
YORUM:
  Kisiye gore bolmede sinyal belirgin sekilde guclenirse: onceki negatif, prob'a
  gereksiz zor bir gorev (gorulmemis ulkeye genelleme) verilmesinden kaynaklaniyordu.
  Iki bolmede de zayifsa: kopru sinyali gercekten yok ya da olcum gucu yetmiyor.

  HER IKI DURUMDA DA acik kalan soru: sinyal katman 2'den itibaren duz. Model iki
  katmanda ulkeyi hesaplayamaz -> sinyal kisi adinin YAZIMINDAN geliyor olabilir.
  Ayrim icin: modelin hop-1'i YANLIS bildigi kisilerde prob gercek ulkeyi mi,
  modelin soyledigi yanlis ulkeyi mi buluyor?
""")
json.dump(OUT, open(os.path.join(WORK, "asamaB2.json"), "w"), indent=1)
print("kaydedildi ->", os.path.join(WORK, "asamaB2.json"))
