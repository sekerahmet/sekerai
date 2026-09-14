# -*- coding: utf-8 -*-
"""
YAZIM mi OLGU mu? — koprü sinyalinin kaynagini ayirir.

DURUM:
  Kisiye gore bolmede kopru (ulke) son-token durumundan okunabiliyor:
  band 0.210 vs null 0.519, esli fark +0.160 [+0.099,+0.284], p=0.0002.
  AMA sinyal katman 2'den itibaren duz -> model iki katmanda "Rumi -> Afganistan"
  aramasi yapamaz. Alternatif aciklama: sinyal kisinin ADININ YAZIMINDAN geliyor.

AYRIM:
  Modelin hop-1'i YANLIS bildigi kisilerde gercek ulke ile modelin dedigi ulke farkli.
  Prob hangisini buluyor?
     modelin dedigi  -> modelin INANCINI okuyoruz  -> OLGUSAL
     gercek ulke     -> modelden bagimsiz sinyal   -> YAZIMSAL

YONTEM:
  1. 468 kisinin tamaminda hop-1'i yeniden uret, tahminleri KAYDET
  2. Prob'u yalnizca DOGRU bilinen kisilerle egit (mevcut states.npy)
  3. Yanlis bilinen kisilerin 2-hop sorularini kodla, prob'u uygula
  4. Her kisi icin: gercek ulkenin sirasi  vs  modelin dedigi ulkenin sirasi

Colab:
    %run yazim_vs_olgu.py
"""
import os, io, re, json, glob, random, string, urllib.request
import numpy as np
import torch

WORK    = "/content/drive/MyDrive/kopru_cc"
MODEL_ID = "Qwen/Qwen3-1.7B"
CC_PATH = "/content/compositional_celebrities.json"
CC_URL  = "https://raw.githubusercontent.com/ofirpress/self-ask/main/datasets/compositional_celebrities.json"
CATEGORIES = ["birthplace_capital", "birthplace_callingcode", "birthplace_currency",
              "birthplace_currency_short", "birthplace_tld"]
PROBE_CAT = "birthplace_capital"     # yanlis-kisiler icin tek kategori yeter
SEED, N_BOOT, N_PERM = 0, 4000, 4000
BAND, PCA_DIM = (5, 25), 256
LAMBDAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
GEN_BATCH, ENC_BATCH = 32, 64

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
def one(p):
    g = sorted(glob.glob(os.path.join(WORK, p)))
    if not g: raise SystemExit(f"bulunamadi: {p} @ {WORK}")
    return g[-1]

# ---------------------------------------------------------------- mevcut durumlar
kept   = json.load(io.open(one("*kept.json"), encoding="utf-8"))
States = np.load(one("*states.npy")); EB = np.load(one("*eb.npy"))
DATEY = re.compile(r"^[\d\s\-/.,+]+$")
def bad_entity(e):
    e = str(e).strip()
    if len(e) < 2: return True
    return bool(DATEY.match(e)) or sum(c.isdigit() for c in e) > len(e)*0.4
kb = [k for k in kept if not bad_entity(k["e3"]) and not bad_entity(k["bridge"])]
assert len(kb) == States.shape[0]
LAYERS = list(range(States.shape[1])); H = States.shape[2]
TGT_N = EB.shape[1]
print(f"egitim havuzu: {len(kb)} soru / {len(set(k['person'] for k in kb))} kisi")

# ---------------------------------------------------------------- CC + model
if not os.path.exists(CC_PATH): urllib.request.urlretrieve(CC_URL, CC_PATH)
raw = json.load(io.open(CC_PATH, encoding="utf-8"))["data"]
sub = [x for x in raw if x["category"] in CATEGORIES]
P2Q1 = {}; P2GOLD = {}
for x in sub: P2Q1.setdefault(x["person"], x["Q1"]); P2GOLD.setdefault(x["person"], str(x["A1"][0]))
Q2HOP = {x["person"]: x["Question"] for x in raw if x["category"] == PROBE_CAT}
persons_all = sorted(P2Q1)
print(f"CC: {len(persons_all)} kisi")

from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float16, trust_remote_code=True).to(dev).eval()

FEW = ("Answer each question with a short factual answer.\n\n"
       "Q: What is the capital of France?\nA: Paris\n\n"
       "Q: What is the birthplace (country only) of Albert Einstein?\nA: Germany\n\n")
def prompt(q): return FEW + "Q: " + q + "\nA:"
def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s).translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())
def eq(a, b): return norm(a) == norm(b)

@torch.no_grad()
def gen(ps, d, mx=12):
    o = []
    for i in tqdm(range(0, len(ps), GEN_BATCH), desc=d):
        e = tok(ps[i:i+GEN_BATCH], return_tensors="pt", padding=True,
                truncation=True, max_length=384).to(dev)
        g = model.generate(**e, max_new_tokens=mx, do_sample=False, pad_token_id=tok.pad_token_id)
        for r in g[:, e["input_ids"].shape[1]:]:
            o.append(tok.decode(r, skip_special_tokens=True).split("\n")[0].strip())
    return o

@torch.no_grad()
def enc(texts, mode, mx, lids):
    out = np.zeros((len(texts), len(lids), H), dtype=np.float16)
    for i in tqdm(range(0, len(texts), ENC_BATCH), desc=f"kodlama[{mode}]"):
        ch = texts[i:i+ENC_BATCH]
        e = tok(ch, return_tensors="pt", padding=True, truncation=True, max_length=mx).to(dev)
        hs = model(**e, output_hidden_states=True).hidden_states
        m = e["attention_mask"].unsqueeze(-1).half(); den = m.sum(1).clamp(min=1)
        for j, l in enumerate(lids):
            v = hs[l][:, -1, :] if mode == "last" else (hs[l]*m).sum(1)/den
            out[i:i+len(ch), j, :] = v.float().cpu().numpy().astype(np.float16)
    return out

# ---------------------------------------------------------------- 1) hop-1 tahminleri
CACHE = os.path.join(WORK, "hop1_preds.json")
if os.path.exists(CACHE):
    preds = json.load(io.open(CACHE, encoding="utf-8")); print("hop-1 onbellekten")
else:
    o = gen([prompt(P2Q1[p]) for p in persons_all], "hop-1 (tum kisiler)")
    preds = dict(zip(persons_all, o))
    json.dump(preds, io.open(CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

ALLC = {norm(P2GOLD[p]): P2GOLD[p] for p in persons_all}      # gecerli ulke sozlugu
wrong = []
for p in persons_all:
    g, pr = P2GOLD[p], preds[p]
    if eq(g, pr): continue
    if norm(pr) not in ALLC: continue                          # tahmin bir ulke degil
    if p not in Q2HOP: continue
    if bad_entity(g) or bad_entity(ALLC[norm(pr)]): continue
    wrong.append(dict(person=p, gold=g, pred=ALLC[norm(pr)], q=Q2HOP[p]))
print(f"\nhop-1 YANLIS + tahmin gecerli bir ulke: {len(wrong)} kisi")
if len(wrong) < 40:
    print("!! cok az; sonuc zayif olacak.")
from collections import Counter
print("  en sik karistirmalar:", Counter((w["gold"], w["pred"]) for w in wrong).most_common(5))

# ---------------------------------------------------------------- 2) aday kume + kodlama
CANDS = sorted({w["gold"] for w in wrong} | {w["pred"] for w in wrong} |
               {k["bridge"] for k in kb})
CIX = {c: i for i, c in enumerate(CANDS)}
print(f"aday ulke kumesi: {len(CANDS)}")
NL = States.shape[1] - 1
TGT_LAYERS = sorted({int(round(f*NL)) for f in (0.35, 0.65, 1.0)})
EC = enc(CANDS, "mean", 32, TGT_LAYERS)
SW = enc([prompt(w["q"]) for w in wrong], "last", 320, LAYERS)
def norml(E): return E/(np.linalg.norm(E, axis=1, keepdims=True)+1e-8)
ENTC = [norml(EC[:, t, :].astype(np.float32)) for t in range(EC.shape[1])]

# ---------------------------------------------------------------- 3) prob: dogru kisilerde egit
y_tr_all = np.array([CIX[k["bridge"]] for k in kb])
PER = np.array([k["person"] for k in kb])
up = sorted(set(PER)); rs = random.Random(SEED+21); rs.shuffle(up)
vaset = set(up[:int(.25*len(up))])
TR = np.array([p not in vaset for p in PER]); VA = ~TR
print(f"prob egitimi: train {TR.sum()} / val {VA.sum()} (kisiye gore)")

def pca_basis(X, k):
    mu = X.mean(0); _, _, Vt = np.linalg.svd(X-mu, full_matrices=False)
    return mu, Vt[:int(min(k, Vt.shape[0], X.shape[0]-1))].T
def ridge(G, B, lam): return np.linalg.solve(G+lam*np.eye(G.shape[0]), B)
def ranks(E, P, idx):
    P = P/(np.linalg.norm(P, axis=1, keepdims=True)+1e-8)
    S = P @ E.T
    out = np.empty(len(idx))
    for i, t in enumerate(idx):
        g = S[i, t]; out[i] = ((S[i] > g).sum()+((S[i] == g).sum()-1)/2)/max(1, S.shape[1]-1)
    return out

gi = np.array([CIX[w["gold"]] for w in wrong]); pi = np.array([CIX[w["pred"]] for w in wrong])
# KONTROL: prob egitimde sik gorulen ulkelere yanli olabilir. Ucuncu bir referans
# olarak, ne gercek ne tahmin olan RASTGELE bir ulkenin sirasini da olcuyoruz.
_r = np.random.RandomState(SEED+5)
ri = np.array([_r.choice([c for c in range(len(CANDS)) if c != g and c != q])
               for g, q in zip(gi, pi)])
from collections import Counter as _C
_freq = _C(k["bridge"] for k in kb)
print(f"  egitim frekansi  — gercek ulkeler medyan {np.median([_freq.get(w['gold'],0) for w in wrong]):.0f}"
      f" | tahmin ulkeler medyan {np.median([_freq.get(w['pred'],0) for w in wrong]):.0f}")
RG, RP, RR = [], [], []
for j in range(BAND[0], BAND[1]+1):
    X = States[:, j, :].astype(np.float32)
    mu, V = pca_basis(X[TR], PCA_DIM)
    Ztr, Zva = (X[TR]-mu)@V, (X[VA]-mu)@V
    G = Ztr.T@Ztr; best = None
    for ti, E in enumerate(ENTC):
        B = Ztr.T@E[y_tr_all[TR]]
        for lam in LAMBDAS:
            A_ = ridge(G, B, lam)
            mv = float(np.median(ranks(E, Zva@A_, y_tr_all[VA])))
            if best is None or mv < best[0]: best = (mv, A_, ti)
    _, A_, ti = best
    Pw = ((SW[:, j, :].astype(np.float32)-mu)@V)@A_
    RG.append(ranks(ENTC[ti], Pw, gi)); RP.append(ranks(ENTC[ti], Pw, pi))
    RR.append(ranks(ENTC[ti], Pw, ri))
rg, rp = np.median(np.array(RG), 0), np.median(np.array(RP), 0)   # band medyani
rr = np.median(np.array(RR), 0)

# ---------------------------------------------------------------- 4) karar
def med(x): return float(np.median(np.asarray(x, float)))
def bci(v, salt, nb=N_BOOT):
    r = np.random.RandomState(SEED+salt); v = np.asarray(v, float)
    m = np.median(v[r.randint(0, len(v), size=(nb, len(v)))], axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
d = rg - rp                                   # >0 ise MODELIN dedigi daha iyi siralaniyor
cd = bci(d, 31)
r = np.random.RandomState(SEED+99); obs = med(d); cnt = 0
for _ in range(N_PERM):
    f = r.rand(len(d)) < .5
    cnt += abs(med(np.where(f, -d, d))) >= abs(obs)-1e-12
p = (cnt+1)/(N_PERM+1)

print("\n" + "="*84)
print(f"MODELIN hop-1'i YANLIS bildigi {len(wrong)} kisi | {len(CANDS)} aday ulke")
print("="*84)
print(f"  GERCEK ulkenin sirasi        : {med(rg):.3f}")
print(f"  MODELIN DEDIGI ulkenin sirasi: {med(rp):.3f}")
print(f"  RASTGELE ulkenin sirasi      : {med(rr):.3f}   [kontrol; ~0.5 olmali]")
print(f"  fark (gercek - model)        : {obs:+.3f}  GA [{cd[0]:+.3f}, {cd[1]:+.3f}]  p={p:.4f}")
print("="*84)
if cd[0] > 0 and p < 0.05:
    V = "OLGUSAL"; msg = ("Prob MODELIN INANDIGI ulkeyi buluyor, gercek ulkeyi degil.\n"
        "  Sinyal modelin kendi bilgisinden geliyor -> adres GERCEK.\n"
        "  EYLEM: Deney 2 gerekcelendi.")
elif cd[1] < 0 and p < 0.05:
    V = "YAZIMSAL"; msg = ("Prob GERCEK ulkeyi buluyor, modelin dedigini degil.\n"
        "  Sinyal modelden bagimsiz - kisi adinin yazimindan geliyor.\n"
        "  EYLEM: bu bir adres degil. Tasarimin tasiyici varsayimi dusuyor.")
else:
    V = "AYIRT EDILEMIYOR"; msg = ("Iki aciklama ayrilamiyor (orneklem ya da etki kucuk).\n"
        "  EYLEM: daha buyuk model ya da kalan 9 kategoriyle orneklemi buyut.")
print("SONUC:", V, "\n " + msg)
json.dump(dict(verdict=V, n=len(wrong), n_cand=len(CANDS), rank_gold=med(rg),
               rank_pred=med(rp), rank_rand=med(rr), diff=obs, ci=list(cd), p=p),
          open(os.path.join(WORK, "yazim_vs_olgu.json"), "w"), indent=1)
print("\nkaydedildi ->", os.path.join(WORK, "yazim_vs_olgu.json"))
