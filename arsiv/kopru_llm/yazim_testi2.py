# -*- coding: utf-8 -*-
"""
YAZIM TESTI 2 — 'ad yazimi' taban cizgisi.

ONCEKI TESTIN SORUNU:
  gercek-vs-tahmin karsilastirmasi frekansla karisti: modelin yanlis tahminleri
  populer ulkeler (BAE, Cin, Fransa, ABD) ve bunlar egitimde medyan 12 kez geciyor,
  gercek ulkeler ise medyan 1 kez. Prob sik gordugunu yukari siralar -> karsilastirma
  gecersiz.

BU TEST:
  Ayni ulkeler, ayni aday kume, ayni istatistik. Degisen tek sey GIRDI TEMSILI:

    A) SORU DURUMU  : 2-hop sorusunun son tokeni, band L5-L25   (modelin hesabi dahil)
    B) HAM AD       : kisi adinin KATMAN 0 token gommesi        (saf yazim, hesap yok)
    C) SIG AD       : kisi adinin orta katman gommesi           (ara referans)

  Frekans yanliligi UC KOLDA DA ayni oldugundan karsilastirmayi bozmaz.

  B ~ A  ->  modelin hesabi bir sey eklemiyor        -> YAZIMSAL
  B << A ->  hesap gercek bilgi ekliyor              -> OLGUSAL BILESEN VAR

Colab:
    %run yazim_testi2.py
"""
import os, io, re, json, glob, random, string, urllib.request
import numpy as np, torch

WORK = "/content/drive/MyDrive/kopru_cc"
MODEL_ID = "Qwen/Qwen3-1.7B"
CC_PATH, CC_URL = "/content/compositional_celebrities.json", \
    "https://raw.githubusercontent.com/ofirpress/self-ask/main/datasets/compositional_celebrities.json"
SEED, N_BOOT, N_PERM = 0, 4000, 4000
BAND, PCA_DIM = (5, 25), 256
LAMBDAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
ENC_BATCH = 64

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
def one(p):
    g = sorted(glob.glob(os.path.join(WORK, p)))
    if not g: raise SystemExit(f"bulunamadi: {p}")
    return g[-1]

kept = json.load(io.open(one("*kept.json"), encoding="utf-8"))
States = np.load(one("*states.npy")); EB = np.load(one("*eb.npy"))
DATEY = re.compile(r"^[\d\s\-/.,+]+$")
def bad_entity(e):
    e = str(e).strip()
    return len(e) < 2 or bool(DATEY.match(e)) or sum(c.isdigit() for c in e) > len(e)*0.4
kb = [k for k in kept if not bad_entity(k["e3"]) and not bad_entity(k["bridge"])]
assert len(kb) == States.shape[0]
NL, H = States.shape[1]-1, States.shape[2]
BR = sorted({k["bridge"] for k in kb}); BIX = {c: i for i, c in enumerate(BR)}
PERSON = np.array([k["person"] for k in kb])
print(f"{len(kb)} soru | {len(set(PERSON))} kisi | {len(BR)} ulke | {NL+1} katman")

from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float16, trust_remote_code=True).to(dev).eval()

@torch.no_grad()
def enc(texts, lids, mx=32):
    out = np.zeros((len(texts), len(lids), H), dtype=np.float16)
    for i in tqdm(range(0, len(texts), ENC_BATCH), desc="kodlama"):
        ch = texts[i:i+ENC_BATCH]
        e = tok(ch, return_tensors="pt", padding=True, truncation=True, max_length=mx).to(dev)
        hs = model(**e, output_hidden_states=True).hidden_states
        m = e["attention_mask"].unsqueeze(-1).half(); den = m.sum(1).clamp(min=1)
        for j, l in enumerate(lids):
            out[i:i+len(ch), j, :] = ((hs[l]*m).sum(1)/den).float().cpu().numpy().astype(np.float16)
    return out

# ---- ad gommeleri: katman 0 (saf yazim) ve orta katman
NAMES = sorted(set(PERSON))
NIX = {p: i for i, p in enumerate(NAMES)}
MID = int(round(0.5*NL))
NE = enc(NAMES, [0, MID])
print(f"ad gommeleri: {NE.shape}  (katman 0 ve {MID})")

TGT = sorted({int(round(f*NL)) for f in (0.35, 0.65, 1.0)})
def norml(E): return E/(np.linalg.norm(E, axis=1, keepdims=True)+1e-8)
ENT = [norml(EB[:, t, :].astype(np.float32)) for t in range(EB.shape[1])]
y = np.array([BIX[k["bridge"]] for k in kb])

# ---- kisiye gore bolme (asamaB2 ile ayni)
up = sorted(set(PERSON)); rs = random.Random(SEED+21); rs.shuffle(up)
n1, n2 = int(.6*len(up)), int(.8*len(up))
grp = {p: ("tr" if i < n1 else "va" if i < n2 else "te") for i, p in enumerate(up)}
sp = np.array([grp[p] for p in PERSON]); TR, VA, TE = sp == "tr", sp == "va", sp == "te"
print(f"train {TR.sum()} / val {VA.sum()} / test {TE.sum()} soru "
      f"({len(set(PERSON[TE]))} test kisisi)")

def pca_basis(X, k):
    mu = X.mean(0); _, _, Vt = np.linalg.svd(X-mu, full_matrices=False)
    return mu, Vt[:int(min(k, Vt.shape[0], X.shape[0]-1))].T
def ridge(G, B, lam): return np.linalg.solve(G+lam*np.eye(G.shape[0]), B)
def nrank(E, P, idx):
    P = P/(np.linalg.norm(P, axis=1, keepdims=True)+1e-8); S = P @ E.T
    o = np.empty(len(idx))
    for i, t in enumerate(idx):
        g = S[i, t]; o[i] = ((S[i] > g).sum()+((S[i] == g).sum()-1)/2)/max(1, S.shape[1]-1)
    return o

def fit_eval(Xall):
    """Xall: [n_soru, H] tek temsil. Egit(TR)/sec(VA)/rapor(TE)."""
    mu, V = pca_basis(Xall[TR], PCA_DIM)
    Ztr, Zva, Zte = (Xall[TR]-mu)@V, (Xall[VA]-mu)@V, (Xall[TE]-mu)@V
    G = Ztr.T@Ztr; best = None
    for ti, E in enumerate(ENT):
        B = Ztr.T@E[y[TR]]
        for lam in LAMBDAS:
            A_ = ridge(G, B, lam)
            mv = float(np.median(nrank(E, Zva@A_, y[VA])))
            if best is None or mv < best[0]: best = (mv, A_, ti)
    _, A_, ti = best
    return nrank(ENT[ti], Zte@A_, y[TE])

# ---- A: soru durumu, band medyani
RA = np.median(np.array([fit_eval(States[:, j, :].astype(np.float32))
                         for j in range(BAND[0], BAND[1]+1)]), axis=0)
# ---- B / C: ad gommesi (her soru icin o sorunun kisisinin adi)
pidx = np.array([NIX[p] for p in PERSON])
RB = fit_eval(NE[pidx, 0, :].astype(np.float32))
RC = fit_eval(NE[pidx, 1, :].astype(np.float32))

# ---- istatistik (kisi-kumeli)
PT = PERSON[TE]
def med(x): return float(np.median(np.asarray(x, float)))
def bci(v, salt, nb=N_BOOT):
    v = np.asarray(v, float); uq = np.unique(PT)
    by = {c: np.where(PT == c)[0] for c in uq}
    r = np.random.RandomState(SEED+salt); m = []
    for _ in range(nb):
        pk = uq[r.randint(0, len(uq), len(uq))]
        m.append(np.median(v[np.concatenate([by[c] for c in pk])]))
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
def perm(a, b, salt):
    uq = np.unique(PT); idx = {c: np.where(PT == c)[0] for c in uq}
    obs = med(b)-med(a); r = np.random.RandomState(SEED+salt); c = 0
    for _ in range(N_PERM):
        aa, bb = a.copy(), b.copy()
        for k in uq:
            if r.rand() < .5:
                ii = idx[k]; aa[ii], bb[ii] = b[ii], a[ii]
        c += abs(med(bb)-med(aa)) >= abs(obs)-1e-12
    return obs, (c+1)/(N_PERM+1)

print("\n" + "="*80)
print(f"AYNI test kisileri ({len(set(PT))} kisi, {len(PT)} soru), {len(BR)} aday ulke, sans 0.500")
print("="*80)
for nm, R, s in (("A) SORU DURUMU (band L5-25)", RA, 1),
                 ("B) HAM AD (katman 0, saf yazim)", RB, 2),
                 (f"C) AD (katman {MID})", RC, 3)):
    lo, hi = bci(R, 10+s)
    print(f"  {nm:34s}: {med(R):.3f}  GA [{lo:.3f}, {hi:.3f}]")
print("-"*80)
oBA, pBA = perm(RA, RB, 41)
oCA, pCA = perm(RA, RC, 42)
print(f"  B - A (ham ad vs soru durumu): {oBA:+.3f}  p={pBA:.4f}")
print(f"  C - A (orta ad vs soru durumu): {oCA:+.3f}  p={pCA:.4f}")
print("="*80)

gap = med(RB) - med(RA)
if pBA < 0.05 and gap > 0.08:
    V = "OLGUSAL BILESEN VAR"
    msg = ("Ham addan ulke cok daha kotu okunuyor; modelin hesabi gercek bilgi ekliyor.\n"
           "  -> Adres modelin bilgi durumundan geliyor. Deney 2 gerekcelendi.")
elif pBA >= 0.05 and abs(gap) < 0.08:
    V = "YAZIMSAL"
    msg = ("Ham ad gommesi soru durumu kadar iyi. Modelin hesabi bir sey EKLEMIYOR.\n"
           "  -> Sinyal adin yazimindan geliyor; bu bir adres degil.\n"
           "  -> Tasarimin tasiyici varsayimi DUSTU.")
else:
    V = "KISMI / BELIRSIZ"
    msg = "  -> Fark var ama net degil; daha buyuk orneklem ya da model gerekir."
print("SONUC:", V); print(" ", msg)
json.dump(dict(verdict=V, n_test=int(TE.sum()), n_person=len(set(PT)), n_cand=len(BR),
               A_state=med(RA), B_name_L0=med(RB), C_name_mid=med(RC),
               diff_BA=oBA, p_BA=pBA, diff_CA=oCA, p_CA=pCA),
          open(os.path.join(WORK, "yazim_testi2.json"), "w"), indent=1)
print("\nkaydedildi ->", os.path.join(WORK, "yazim_testi2.json"))
