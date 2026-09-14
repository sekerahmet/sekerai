# -*- coding: utf-8 -*-
"""
SON TEST — okunan kopru GERI BESLENINCE dogruluk artiyor mu?

BILDIKLERIMIZ:
  ipucusuz 2-hop            : %18.9
  GERCEK kopru token olarak : %62.9        (tavan)
  YANLIS kopru              : %0           (yanlis adres yikici)
  kopru latent'te okunabilir: 0.210 vs null 0.519, p=0.0002
  ama adres KABA            : 82 ulke icinde medyan sira ~18

SORU: prob'un okudugu ulkeyi modele ipucu olarak versek ne olur?
  Bu, "kopru okunabiliyor"u "kopruyu okumak ISE YARIYOR"a cevirir ve
  sira-18 hassasiyetinin yeterli olup olmadigini dogrudan olcer.

KOLLAR (hepsi ayni test kisilerinde, kisiye gore tutulmus):
  none        ipucu yok                         -> taban
  gold        gercek ulke                       -> tavan
  probe@1/3/5 prob'un top-k tahmini             -> ASIL OLCUM
  rand@1/3/5  rastgele ulke(ler), ayni bicimde  -> BICIM KONTROLU
              (liste vermek tek basina yardim ediyor mu?)

Colab:
    %run son_test.py
"""
import os, io, re, json, glob, random, string
import numpy as np, torch

WORK = "/content/drive/MyDrive/kopru_cc"
MODEL_ID = "Qwen/Qwen3-1.7B"
SEED, N_BOOT = 0, 4000
BAND, PCA_DIM = (5, 25), 256
LAMBDAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
GEN_BATCH, TOPKS = 32, (1, 3, 5)

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
BR = sorted({k["bridge"] for k in kb}); BIX = {c: i for i, c in enumerate(BR)}
PERSON = np.array([k["person"] for k in kb]); y = np.array([BIX[k["bridge"]] for k in kb])
print(f"{len(kb)} soru | {len(set(PERSON))} kisi | {len(BR)} ulke")

# ---- kisiye gore bolme (asamaB2 / yazim_testi2 ile AYNI)
up = sorted(set(PERSON)); rs = random.Random(SEED+21); rs.shuffle(up)
n1, n2 = int(.6*len(up)), int(.8*len(up))
grp = {p: ("tr" if i < n1 else "va" if i < n2 else "te") for i, p in enumerate(up)}
sp = np.array([grp[p] for p in PERSON]); TR, VA, TE = sp=="tr", sp=="va", sp=="te"
print(f"train {TR.sum()} / val {VA.sum()} / test {TE.sum()} soru ({len(set(PERSON[TE]))} kisi)")

def norml(E): return E/(np.linalg.norm(E, axis=1, keepdims=True)+1e-8)
ENT = [norml(EB[:, t, :].astype(np.float32)) for t in range(EB.shape[1])]
def pca_basis(X, k):
    mu = X.mean(0); _, _, Vt = np.linalg.svd(X-mu, full_matrices=False)
    return mu, Vt[:int(min(k, Vt.shape[0], X.shape[0]-1))].T
def ridge(G, B, lam): return np.linalg.solve(G+lam*np.eye(G.shape[0]), B)
def scores(E, P): return norml(P) @ E.T

# ---- prob'u TRAIN'de egit, TEST icin ulke siralamasi uret (band uzeri ortalama SIRA)
rank_sum = np.zeros((int(TE.sum()), len(BR)))
for j in range(BAND[0], BAND[1]+1):
    X = States[:, j, :].astype(np.float32)
    mu, V = pca_basis(X[TR], PCA_DIM)
    Ztr, Zva, Zte = (X[TR]-mu)@V, (X[VA]-mu)@V, (X[TE]-mu)@V
    G = Ztr.T@Ztr; best = None
    for ti, E in enumerate(ENT):
        B = Ztr.T@E[y[TR]]
        for lam in LAMBDAS:
            A_ = ridge(G, B, lam)
            S = scores(E, Zva@A_)
            mv = float(np.median([(S[i] > S[i, t]).sum() for i, t in enumerate(y[VA])]))
            if best is None or mv < best[0]: best = (mv, A_, ti)
    _, A_, ti = best
    S = scores(ENT[ti], Zte@A_)
    rank_sum += np.argsort(np.argsort(-S, axis=1), axis=1)   # 0 = en iyi
order = np.argsort(rank_sum, axis=1)                          # band ortalamasina gore siralama

kbT = [k for k, t in zip(kb, TE) if t]
gold_i = y[TE]
hit = {k: float(np.mean([gold_i[i] in order[i, :k] for i in range(len(kbT))])) for k in TOPKS}
print("\nprob isabeti (test):", "  ".join(f"top-{k} %{100*v:.1f}" for k, v in hit.items()))

# ---- model
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
def match_any(pred, golds):
    p = norm(pred)
    for g in golds:
        g = norm(g)
        if len(g) < 3:
            if p == g: return True
        elif p == g or (" "+g+" ") in (" "+p+" "): return True
    return False
def hint(q, names):
    return q + (f" (Hint: {names[0]})" if len(names) == 1
                else f" (Hint: one of {', '.join(names)})")

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

# ---- kollar
rr = random.Random(SEED+7)
def rand_names(i, k):
    out = []
    while len(out) < k:
        c = BR[rr.randrange(len(BR))]
        if c != kbT[i]["bridge"] and c not in out: out.append(c)
    return out

arms = {"none": [prompt(k["question"]) for k in kbT],
        "gold": [prompt(hint(k["question"], [k["bridge"]])) for k in kbT]}
for k in TOPKS:
    arms[f"probe@{k}"] = [prompt(hint(kbT[i]["question"], [BR[c] for c in order[i, :k]]))
                          for i in range(len(kbT))]
    arms[f"rand@{k}"]  = [prompt(hint(kbT[i]["question"], rand_names(i, k)))
                          for i in range(len(kbT))]

ACC, OKS = {}, {}
for nm, ps in arms.items():
    o = gen(ps, nm)
    ok = np.array([match_any(a, it["answers"]) for a, it in zip(o, kbT)])
    OKS[nm] = ok; ACC[nm] = float(ok.mean())

# ---- istatistik: kisi-kumeli bootstrap (esli farklar)
PT = PERSON[TE]; uq = np.unique(PT); by = {c: np.where(PT == c)[0] for c in uq}
def bci_diff(a, b, salt, nb=N_BOOT):
    r = np.random.RandomState(SEED+salt); d = a.astype(float)-b.astype(float); m = []
    for _ in range(nb):
        pk = uq[r.randint(0, len(uq), len(uq))]
        m.append(d[np.concatenate([by[c] for c in pk])].mean())
    return float(np.mean(d)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

print("\n" + "="*84)
print(f"SON TEST — {len(kbT)} soru / {len(uq)} kisi (hepsi egitimde GORULMEDI)")
print("="*84)
for nm in ("none", "gold", "probe@1", "probe@3", "probe@5", "rand@1", "rand@3", "rand@5"):
    print(f"  {nm:9s}: %{100*ACC[nm]:5.1f}")
print("-"*84)
print("  ESLI FARKLAR (kisi-kumeli %95 GA):")
rows = []
for k in TOPKS:
    dn, lo, hi = bci_diff(OKS[f"probe@{k}"], OKS["none"], 10+k)
    dr, rlo, rhi = bci_diff(OKS[f"probe@{k}"], OKS[f"rand@{k}"], 20+k)
    rows.append((k, dn, lo, hi, dr, rlo, rhi))
    print(f"    probe@{k} - none   : {100*dn:+5.1f} puan  GA [{100*lo:+.1f}, {100*hi:+.1f}]")
    print(f"    probe@{k} - rand@{k} : {100*dr:+5.1f} puan  GA [{100*rlo:+.1f}, {100*rhi:+.1f}]"
          f"   <- bicim kontrolu")
print("="*84)

best = max(rows, key=lambda r: r[1])
k, dn, lo, hi, dr, rlo, rhi = best
head = ACC["gold"] - ACC["none"]
recov = (dn / head * 100) if head > 0 else float("nan")
if lo > 0 and rlo > 0:
    V = "ISE YARIYOR"
    msg = (f"probe@{k} tabani {100*dn:+.1f} puan gectiyor ve bicim kontrolunu de geciyor.\n"
           f"  Tavanin (%{100*head:.1f} puanlik bosluk) %{recov:.0f}'ini geri kazaniyor.\n"
           "  -> Latent adresi okuyup geri beslemek ucu uca ISE YARIYOR. Dar tasarim canli.")
elif lo > 0:
    V = "BICIM ETKISI SUPHELI"
    msg = ("Taban gecildi ama rastgele kol da benzer sekilde iyi -> kazanc icerikten\n"
           "  degil, ipucu BICIMINDEN geliyor olabilir. Icerik kaniti yok.")
else:
    V = "ISE YARAMIYOR"
    msg = (f"En iyi kol (probe@{k}) tabani gecmiyor (GA sifiri iceriyor).\n"
           f"  prob isabeti top-{k} %{100*hit[k]:.1f}; yanlis adres yikici oldugundan\n"
           "  bu keskinlik yetmiyor.\n"
           "  -> Dogrusal cikarim yeterli degil. Sonraki soru: uctan uca EGITILMIS bir\n"
           "     cikarici yeterince keskin olabilir mi? Bu ayri ve daha buyuk bir proje.")
print("SONUC:", V); print(" ", msg)
json.dump(dict(verdict=V, n=len(kbT), n_person=int(len(uq)), acc=ACC, hit=hit,
               best_k=int(k), diff_none=dn, ci_none=[lo, hi], diff_rand=dr,
               ci_rand=[rlo, rhi], ceiling=head, recovered_pct=recov),
          open(os.path.join(WORK, "son_test.json"), "w"), indent=1)
print("\nkaydedildi ->", os.path.join(WORK, "son_test.json"))
