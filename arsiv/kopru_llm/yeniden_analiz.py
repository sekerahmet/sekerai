# -*- coding: utf-8 -*-
"""
YENIDEN ANALIZ — GPU gerekmez, kaydedilmis probe.npz uzerinden calisir.

NEDEN:
  kopru_cc.py katmani VAL'da secti ve katman 1'i buldu — kirmizi egrinin en kotu
  oldugu nokta. Val seti 94 soru / ~16 ulke; ustelik her katmanda 15 hiperparametre
  kombinasyonunun minimumu aliniyor. Secim yapacak kadar kararli degil.
  Grafik ise kopru sinyalinin katman 2-28 boyunca 0.21-0.33 bandinda oldugunu
  gosteriyor. Dogru cevap tek katman secmek degil, BANDI degerlendirmek.

NE YAPAR:
  1. Her katman icin medyan + KISI-kumeli bootstrap GA (kopru, cevap, null, ham)
  2. Kac katmanda kopru GA'si null GA'sindan AYRIK -> band-duzeyi kanit
  3. SECIMSIZ ozet istatistik: soru basina orta-band (varsayilan L5-L25) medyani,
     tek bir GA. Hicbir katman secimi yok -> kazananin laneti yok.
  4. Permutasyon band testi: null etiketleriyle ayni istatistigi hesaplayip
     gercek degerin nerede durdugunu soyler.

Colab:
    %run yeniden_analiz.py
"""
import os, io, json, re, glob, random
import numpy as np

WORK   = "/content/drive/MyDrive/kopru_cc"
SEED   = 0
N_BOOT = 4000
BAND   = (5, 25)          # secimsiz ozet icin orta band (dahil)

# ---------------------------------------------------------------- dosyalar
npz = sorted(glob.glob(os.path.join(WORK, "*probe.npz")))
kj  = sorted(glob.glob(os.path.join(WORK, "*kept.json")))
if not npz or not kj:
    raise SystemExit(f"probe.npz / kept.json bulunamadi: {WORK}")
print("probe:", os.path.basename(npz[-1]))
print("kept :", os.path.basename(kj[-1]))
Z = np.load(npz[-1]); kept = json.load(io.open(kj[-1], encoding="utf-8"))
LAYERS = Z["layers"].tolist()

# ---------------------------------------------------------------- kb ve bolmeyi YENIDEN TURET
# (kopru_cc.py ile birebir ayni mantik; SEED sabit oldugu icin deterministik)
DATEY = re.compile(r"^[\d\s\-/.,+]+$")
def bad_entity(e):
    e = str(e).strip()
    if len(e) < 2: return True
    return bool(DATEY.match(e)) or sum(c.isdigit() for c in e) > len(e) * 0.4

kb = [k for k in kept if not bad_entity(k["e3"]) and not bad_entity(k["bridge"])]
print(f"kept {len(kept)} -> Asama B alt kumesi {len(kb)}")

def make_split(keyfn, salt):
    uniq = sorted({keyfn(k) for k in kb})
    rs = random.Random(SEED + salt); rs.shuffle(uniq)
    n1, n2 = int(.6*len(uniq)), int(.8*len(uniq))
    grp = {e: ("tr" if i < n1 else "va" if i < n2 else "te") for i, e in enumerate(uniq)}
    sp = np.array([grp[keyfn(k)] for k in kb])
    return sp == "tr", sp == "va", sp == "te", len(uniq)

TEb = make_split(lambda k: k["bridge"], 5)[2]
TEa = make_split(lambda k: k["e3"], 6)[2]
PERSON = np.array([k["person"] for k in kb])
PB, PA = PERSON[TEb], PERSON[TEa]

for nm, arr, P in (("bridge", Z["bridge_probe"], PB), ("answer", Z["answer_probe"], PA)):
    assert arr.shape[1] == len(P), f"{nm}: {arr.shape[1]} != {len(P)} — bolme yeniden turetilemedi"
print(f"dogrulama OK | kopru test {len(PB)} soru / {len(set(PB))} kisi"
      f" | cevap test {len(PA)} soru / {len(set(PA))} kisi")

# ---------------------------------------------------------------- istatistik
def bci_cluster(vals, clusters, salt=0, nb=N_BOOT):
    vals = np.asarray(vals, float); ok = np.isfinite(vals)
    if ok.sum() < 5: return (np.nan, np.nan)
    uq = np.unique(clusters[ok]); by = {c: np.where(ok & (clusters == c))[0] for c in uq}
    r = np.random.RandomState((SEED + salt) % (2**31 - 1)); m = []
    for _ in range(nb):
        pick = uq[r.randint(0, len(uq), len(uq))]
        m.append(np.median(vals[np.concatenate([by[c] for c in pick])]))
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def med(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan

# ---------------------------------------------------------------- 1) katman katman
print("\n" + "=" * 96)
print("KATMAN KATMAN  (medyan [%95 GA], kisi-kumeli bootstrap; sans 0.500, dusuk=iyi)")
print("=" * 96)
print(f"{'L':>3s}  {'KOPRU':>22s}  {'NULL':>22s}  {'ayrik?':>7s}   {'CEVAP':>22s}")
print("-" * 96)
sep = []
for j, L in enumerate(LAYERS):
    b, n_, a = Z["bridge_probe"][j], Z["bridge_null"][j], Z["answer_probe"][j]
    cb, cn = bci_cluster(b, PB, 100+j), bci_cluster(n_, PB, 200+j)
    ca = bci_cluster(a, PA, 300+j)
    d = np.isfinite(cb[1]) and np.isfinite(cn[0]) and cb[1] < cn[0]
    sep.append(d)
    print(f"{L:3d}  {med(b):.3f} [{cb[0]:.3f},{cb[1]:.3f}]  {med(n_):.3f} [{cn[0]:.3f},{cn[1]:.3f}]"
          f"  {'EVET' if d else '  -':>7s}   {med(a):.3f} [{ca[0]:.3f},{ca[1]:.3f}]")
print("-" * 96)
print(f"kopru GA'si null'dan AYRIK olan katman: {sum(sep)}/{len(LAYERS)}")

# ---------------------------------------------------------------- 2) SECIMSIZ band ozeti
i0 = LAYERS.index(max(BAND[0], LAYERS[0])); i1 = LAYERS.index(min(BAND[1], LAYERS[-1]))
bandB = np.median(Z["bridge_probe"][i0:i1+1], axis=0)   # soru basina band medyani
bandN = np.median(Z["bridge_null"][i0:i1+1],  axis=0)
bandR = np.median(Z["bridge_raw"][i0:i1+1],   axis=0)
cB, cN, cR = bci_cluster(bandB, PB, 11), bci_cluster(bandN, PB, 12), bci_cluster(bandR, PB, 13)

print("\n" + "=" * 96)
print(f"SECIMSIZ BAND OZETI  (L{BAND[0]}-L{BAND[1]}; hicbir katman secilmedi -> kazananin laneti yok)")
print("=" * 96)
print(f"  KOPRU  band medyani : {med(bandB):.3f}   GA [{cB[0]:.3f}, {cB[1]:.3f}]")
print(f"  NULL   band medyani : {med(bandN):.3f}   GA [{cN[0]:.3f}, {cN[1]:.3f}]")
print(f"  HAM    band medyani : {med(bandR):.3f}   GA [{cR[0]:.3f}, {cR[1]:.3f}]")
print(f"  sans                : 0.500")

# esli fark (ayni sorularda), kisi-kumeli
diff = bandN - bandB
cD = bci_cluster(diff, PB, 14)
print(f"\n  ESLI FARK (null - kopru): {med(diff):+.3f}   GA [{cD[0]:+.3f}, {cD[1]:+.3f}]")
print(f"  -> GA tamami sifirin ustundeyse kopru nulden ISTATISTIKSEL OLARAK iyi")

# ---------------------------------------------------------------- 3) permutasyon band testi
r = np.random.RandomState(SEED + 99)
obs = med(bandN) - med(bandB)
perm = []
for _ in range(2000):
    flip = r.rand(len(bandB)) < 0.5          # esli isaret permutasyonu
    a_ = np.where(flip, bandN, bandB); b_ = np.where(flip, bandB, bandN)
    perm.append(np.median(a_) - np.median(b_))
perm = np.array(perm); p = float((np.abs(perm) >= abs(obs)).mean())
print(f"\n  esli isaret permutasyon testi: gozlenen fark {obs:+.3f}, p = {p:.4f}")

# ---------------------------------------------------------------- karar
print("\n" + "=" * 96)
strong = np.isfinite(cD[0]) and cD[0] > 0 and p < 0.05
frac = sum(sep) / len(LAYERS)
if strong and med(bandB) < 0.35:
    V = "KOPRU VAR (band duzeyinde)"
elif strong:
    V = "KOPRU ZAYIF AMA GERCEK"
else:
    V = "AYIRT EDILEMIYOR"
print("YENIDEN ANALIZ SONUCU:", V)
print(f"  band medyani {med(bandB):.3f} | nulden fark {med(diff):+.3f} (p={p:.4f})"
      f" | ayrik katman orani %{100*frac:.0f}")
print("=" * 96)
print("""
UYARI — bu sonuc 'model kopruyu HESAPLIYOR' demek DEGIL:
  Kirmizi egri katman 2'den itibaren duz. Model iki katmanda ulkeyi hesaplamis
  olamaz. Sinyal buyuk ihtimalle KISI ADININ YAZIMINDAN geliyor olabilir
  (Arapca/Slav/Ispanyol adi -> bolge). Bunu ayirt etmek icin siradaki adim:
  modelin hop-1'i YANLIS bildigi kisilerde prob GERCEK ulkeyi mi, yoksa MODELIN
  SOYLEDIGI yanlis ulkeyi mi buluyor?  Gercegi buluyorsa sinyal yuzeysel.
""")
json.dump(dict(verdict=V, band=list(BAND), band_bridge=med(bandB), ci_bridge=list(cB),
               band_null=med(bandN), ci_null=list(cN), band_raw=med(bandR),
               paired_diff=med(diff), ci_diff=list(cD), perm_p=p,
               layers_separated=int(sum(sep)), n_layers=len(LAYERS),
               n_test=len(PB), n_person=len(set(PB))),
          open(os.path.join(WORK, "yeniden_analiz.json"), "w"), indent=1)
print("kaydedildi ->", os.path.join(WORK, "yeniden_analiz.json"))
