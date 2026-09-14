# -*- coding: utf-8 -*-
"""
Kopru Adreslenebilirligi — Compositional Celebrities surumu

Colab:
    !pip -q install -U transformers datasets accelerate
    %run kopru_cc.py

NEDEN CC:
  2WikiMultihopQA'da bilgi filtresi 4000 -> 4 verdi; model obskur Wikidata olgularini
  bilmiyordu (hop-1 %1.5, hop-2 %5.5). Teshis (teshis.py) bunun FORMAT degil BILGI
  sorunu oldugunu gosterdi. CC farkli: kisiler unlu, ikinci hop kapali kume.
  Olculen verim: hop-1 %50, en iyi 5 kategoride ortak %29-48 -> ~855 soru.

SORU: "Rumi'nin dogdugu ulkenin baskenti nedir?"
  hop-1  Rumi -> Afganistan      (KOPRU - soruda GECMIYOR)
  hop-2  Afganistan -> Kabul     (CEVAP)

ASAMA A  Kopruyu bedava versek model toparliyor mu? (darbogaz adresleme mi?)
ASAMA B  Kopru latent durumda var mi, cevaptan yeterince once mi?

SONUCLAR:
  ONCUL YANLIS | IPUCU KULLANILAMIYOR | ADRES YOK | SADECE TIP SINYALI |
  KISAYOL | PENCERE DAR | GEC

Istatistik: katman VAL'da secilir, sayilar TEST'te raporlanir (kazananin laneti).
Permutasyon nulu tip-duzeyi sinyali kimlik sanmayi engeller.
Guven araliklari KISI kumelerinde bootstrap edilir (ayni kisi birden cok kategoride).
"""

# =====================================================================
# AYARLAR
# =====================================================================
MODEL_ID = "Qwen/Qwen3-1.7B"

# teshis_cc.py'nin olctugu en verimli kategoriler (>= %25 ortak verim)
CATEGORIES = ["birthplace_capital", "birthplace_callingcode", "birthplace_currency",
              "birthplace_currency_short", "birthplace_tld"]

PCA_DIM   = 256
LAMBDAS   = (1.0, 10.0, 100.0, 1000.0, 10000.0)
TGT_FRACS = (0.35, 0.65, 1.0)

GATE_GAIN    = 0.15   # ASAMA A kapisi: (dogru ipucu) - (ipucusuz)
HINT_CEILING = 0.60   # dogru-ipucu dogrulugu bunun altindaysa: IPUCU KULLANILAMIYOR

BRIDGE_CI_MAX   = 0.45
MIN_WINDOW_FRAC = 0.15

MIN_KEPT, MIN_TRAIN = 150, 80
GEN_BATCH, ENC_BATCH, N_BOOT, SEED = 32, 64, 2000, 0

USE_DRIVE = True
DRIVE_DIR = "/content/drive/MyDrive/kopru_cc"
CC_URL  = "https://raw.githubusercontent.com/ofirpress/self-ask/main/datasets/compositional_celebrities.json"
CC_PATH = "/content/compositional_celebrities.json"

# =====================================================================
import os, sys, io, json, re, string, random, time, hashlib, urllib.request
import numpy as np
import torch
from collections import defaultdict

T0 = time.time()
def banner(s): print("\n" + "=" * 72); print(s); print("=" * 72, flush=True)
def tick(s):  print(f"[{time.time()-T0:6.0f}s] {s}", flush=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

banner("0/6  ORTAM")
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    _p = torch.cuda.get_device_properties(0)
    print("gpu:", _p.name, f"{_p.total_memory/1e9:.1f} GB")
else:
    print("!! GPU YOK -> Runtime > Change runtime type > T4 GPU")

if USE_DRIVE:
    try:
        from google.colab import drive; drive.mount("/content/drive"); WORK = DRIVE_DIR
    except Exception as e:
        print("Drive baglanamadi (%s) -> /content" % str(e)[:70]); WORK = "/content/kopru_cc"
else:
    WORK = "/content/kopru_cc"
os.makedirs(WORK, exist_ok=True)
MSLUG = MODEL_ID.split("/")[-1].replace(".", "-")
print("calisma dizini:", WORK)
print("(kopan oturumda ayni komutu tekrar calistir - tamamlanan adimlar atlanir)")

# =====================================================================
banner("1/6  VERI — Compositional Celebrities")
if not os.path.exists(CC_PATH):
    print("indiriliyor:", CC_URL)
    urllib.request.urlretrieve(CC_URL, CC_PATH)
raw = json.load(io.open(CC_PATH, encoding="utf-8"))["data"]
print(f"toplam {len(raw)} soru, {len(set(x['category'] for x in raw))} kategori")

bad_cat = [c for c in CATEGORIES if c not in {x["category"] for x in raw}]
if bad_cat:
    print("BILINMEYEN KATEGORI:", bad_cat); sys.exit(1)

items = []
for x in raw:
    if x["category"] not in CATEGORIES: continue
    items.append(dict(question=x["Question"],
                      q1=x["Q1"], bridge=str(x["A1"][0]),
                      q2=x["Q2"], answers=[str(a) for a in x["A2"]],
                      e3=str(x["A2"][0]), person=x["person"], cat=x["category"]))
print(f"secilen {len(CATEGORIES)} kategori -> {len(items)} soru, "
      f"{len({i['person'] for i in items})} kisi, {len({i['bridge'] for i in items})} ulke")
print("ornek:", items[0]["question"])
print("  hop1:", items[0]["q1"], "->", items[0]["bridge"])
print("  hop2:", items[0]["q2"], "->", items[0]["e3"])

_fp = hashlib.md5(json.dumps([(i["person"], i["cat"]) for i in items]).encode()).hexdigest()[:8]
STAMP = f"{MSLUG}__cc{len(CATEGORIES)}__{_fp}"
def cpath(n): return os.path.join(WORK, f"{STAMP}__{n}")
def cached(n): return os.path.exists(cpath(n))
print("onbellek damgasi:", STAMP)
tick("veri hazir")

# =====================================================================
banner("2/6  MODEL")
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float16, trust_remote_code=True).to(device).eval()

N_LAYERS, H = model.config.num_hidden_layers, model.config.hidden_size
LAYERS = list(range(N_LAYERS + 1))
TGT_LAYERS = sorted({int(round(f * N_LAYERS)) for f in TGT_FRACS})
print(f"{MODEL_ID}: {N_LAYERS} katman, hidden={H}")
print("varlik gommesi icin aday hedef katmanlar (val'da secilecek):", TGT_LAYERS)
with torch.no_grad():
    _h = model(**tok(["The capital of France is Paris."], return_tensors="pt").to(device),
               output_hidden_states=True).hidden_states
print("fp16 saglik:", "NaN/Inf VAR -> dtype=torch.float32 dene"
      if any(torch.isnan(x).any() or torch.isinf(x).any() for x in _h) else "temiz")
tick("model hazir")

# =====================================================================
banner("3/6  BILGI FILTRESI")
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
        elif p == g or (" " + g + " ") in (" " + p + " "): return True
    return False

@torch.no_grad()
def gen(prompts, desc, mx=12):
    out = []
    for i in tqdm(range(0, len(prompts), GEN_BATCH), desc=desc):
        enc = tok(prompts[i:i+GEN_BATCH], return_tensors="pt", padding=True,
                  truncation=True, max_length=384).to(device)
        g = model.generate(**enc, max_new_tokens=mx, do_sample=False,
                           pad_token_id=tok.pad_token_id)
        for row in g[:, enc["input_ids"].shape[1]:]:
            out.append(tok.decode(row, skip_special_tokens=True).split("\n")[0].strip())
    return out

if cached("kept.json"):
    kept = json.load(open(cpath("kept.json"), encoding="utf-8")); print("onbellekten:", len(kept))
else:
    # hop-1 KISI basina, hop-2 (ULKE,KATEGORI) basina -> tekrarli uretim yok
    p1 = {}
    for it in items: p1.setdefault(it["person"], (it["q1"], it["bridge"]))
    persons = sorted(p1)
    o1 = gen([prompt(p1[p][0]) for p in persons], "hop-1 (kisi)")
    ok1 = {p: match_any(o, [p1[p][1]]) for p, o in zip(persons, o1)}

    p2 = {}
    for it in items: p2.setdefault((it["bridge"], it["cat"]), (it["q2"], it["answers"]))
    keys = sorted(p2)
    o2 = gen([prompt(p2[k][0]) for k in keys], "hop-2 (ulke x kategori)")
    ok2 = {k: match_any(o, p2[k][1]) for k, o in zip(keys, o2)}

    print(f"\n  hop-1: %{100*np.mean(list(ok1.values())):.1f} ({sum(ok1.values())}/{len(persons)} kisi)")
    print(f"  hop-2: %{100*np.mean(list(ok2.values())):.1f} ({sum(ok2.values())}/{len(keys)} ulke-kategori)")
    kept = [dict(it) for it in items
            if ok1.get(it["person"]) and ok2.get((it["bridge"], it["cat"]))]
    json.dump(kept, open(cpath("kept.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print(f"\nfiltre: {len(items)} -> {len(kept)}")
if len(kept) == 0:
    print("\nFILTREDEN HIC SORU GECMEDI. Daha buyuk MODEL_ID dene."); sys.exit(1)
print(f"  benzersiz kisi: {len({k['person'] for k in kept})} | "
      f"ulke: {len({k['bridge'] for k in kept})} | "
      f"kategori dagilimi: { {c: sum(1 for k in kept if k['cat']==c) for c in CATEGORIES} }")
if len(kept) < MIN_KEPT:
    print(f"!! UYARI: {MIN_KEPT} altinda -> GA'lar genis cikacak.")
tick("filtre bitti")

# =====================================================================
banner("4/6  ASAMA A — ONCUL TESTI")
def hint_paren(it, b):   return it["question"] + f" (Hint: {b})"
def hint_premise(it, b): return f"{it['q1']} {b}. {it['question']}"

if cached("stageA.json"):
    A = json.load(open(cpath("stageA.json"))); print("onbellekten")
else:
    rs = random.Random(SEED)
    pool = sorted({k["bridge"] for k in kept})
    rand_b = []
    for it in kept:                                   # rastgele ipucu != gercek kopru
        c = pool[rs.randrange(len(pool))]; t = 0
        while norm(c) == norm(it["bridge"]) and t < 50:
            c = pool[rs.randrange(len(pool))]; t += 1
        rand_b.append(c)
    runs = {
        "none":         [prompt(it["question"])                 for it in kept],
        "paren":        [prompt(hint_paren(it, it["bridge"]))   for it in kept],
        "premise":      [prompt(hint_premise(it, it["bridge"])) for it in kept],
        "rand_paren":   [prompt(hint_paren(it, b))              for it, b in zip(kept, rand_b)],
        "rand_premise": [prompt(hint_premise(it, b))            for it, b in zip(kept, rand_b)],
    }
    A = {}
    for k, ps in runs.items():
        o = gen(ps, k)
        A[k] = float(np.mean([match_any(a, it["answers"]) for a, it in zip(o, kept)]))
    json.dump(A, open(cpath("stageA.json"), "w"), indent=1)

def bprop(p, n, salt=0, nb=N_BOOT):
    r = np.random.RandomState((SEED + salt) % (2**31 - 1)); k = int(round(p * n))
    x = np.array([1]*k + [0]*(n-k))
    m = x[r.randint(0, n, size=(nb, n))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

n = len(kept)
fmt = "premise" if A["premise"] >= A["paren"] else "paren"
acc_hint, acc_rand, acc_none = A[fmt], A["rand_" + fmt], A["none"]
gain, mis = acc_hint - acc_none, acc_rand - acc_none

print("\nher iki tek-hop olgu BILINIYOR kosuluyla, 2-hop dogrulugu:")
print("-" * 72)
for si, k in enumerate(("none", "paren", "premise", "rand_paren", "rand_premise")):
    lo, hi = bprop(A[k], n, salt=si)
    print(f"  {k:14s}: %{100*A[k]:5.1f}   GA [%{100*lo:.1f}, %{100*hi:.1f}]"
          + ("   <- secilen format" if k == fmt else ""))
print("-" * 72)
print(f"  ADRESLEME KAZANCI (dogru - ipucusuz) : {100*gain:+.1f} puan   [KAPI: >= {100*GATE_GAIN:+.1f}]")
print(f"  rastgele ipucunun etkisi             : {100*mis:+.1f} puan   [saglamlik: <= 0]")
print(f"  dogru-ipucu dogrulugu                : %{100*acc_hint:.1f}      [tavan esigi: %{100*HINT_CEILING:.0f}]")

HINT_USABLE = acc_hint >= HINT_CEILING
READS_HINT  = mis <= 0.02
STAGE_A = ("IPUCU KULLANILAMIYOR" if not HINT_USABLE
           else "GECTI" if gain >= GATE_GAIN else "ONCUL YANLIS")
print("\nASAMA A:", STAGE_A)
if STAGE_A == "GECTI":
    print("  -> Kopruyu vermek belirgin yardim ediyor: darbogaz ADRESLEME.")
    if not READS_HINT:
        print("  !! rastgele ipucu zarar vermiyor -> model ipucunu okumuyor olabilir.")
elif STAGE_A == "IPUCU KULLANILAMIYOR":
    print("  -> Model hop-2'yi dogrudan biliyor ama ipucu olarak verilince kullanamiyor.")
    print("     ADRESLEME hakkinda bilgi VERMEZ - talimat izleme sorunu.")
    print("  EYLEM: ipucu formatini coz (daha buyuk model / chat template).")
else:
    print("  -> Kopruyu bedava vermek yardim etmiyor: darbogaz adresleme DEGIL.")
    print("  EYLEM: DUR. Tasarimi bastan kur.")
tick("Asama A bitti")

if STAGE_A != "GECTI":
    json.dump(dict(model=MODEL_ID, dataset="CC", cats=CATEGORIES, stage_a=STAGE_A,
                   result=STAGE_A, A=A, hint_fmt=fmt, gain=float(gain), mis=float(mis),
                   n_kept=len(kept)), open(cpath("verdict.json"), "w"), indent=1)
    banner(f"SONUC: {STAGE_A}  — Asama B calistirilmadi")
    print("kaydedildi ->", cpath("verdict.json")); sys.exit(0)

# =====================================================================
banner("5/6  ASAMA B — CIKARILABILIRLIK VE ZAMANLAMA")

# Sayisal cevaplar (or. '+93') gomme uzayinda ayirt edilemez -> Asama B alt kumesi
DATEY = re.compile(r"^[\d\s\-/.,+]+$")
def bad_entity(e):
    e = str(e).strip()
    if len(e) < 2: return True
    return bool(DATEY.match(e)) or sum(c.isdigit() for c in e) > len(e) * 0.4

kb = [k for k in kept if not bad_entity(k["e3"]) and not bad_entity(k["bridge"])]
print(f"Asama B alt kumesi (sayisal cevaplar elendi): {len(kept)} -> {len(kb)}")
print(f"  kategori: { {c: sum(1 for k in kb if k['cat']==c) for c in CATEGORIES} }")
if len(kb) < 60:
    print("!! Asama B icin cok az soru kaldi."); sys.exit(1)

@torch.no_grad()
def enc_layers(texts, mode, mx, layer_ids):
    out = np.zeros((len(texts), len(layer_ids), H), dtype=np.float16)
    for i in tqdm(range(0, len(texts), ENC_BATCH), desc=f"kodlama[{mode}]"):
        ch = texts[i:i+ENC_BATCH]
        e = tok(ch, return_tensors="pt", padding=True, truncation=True, max_length=mx).to(device)
        hs = model(**e, output_hidden_states=True).hidden_states
        m = e["attention_mask"].unsqueeze(-1).half(); den = m.sum(1).clamp(min=1)
        for j, l in enumerate(layer_ids):
            v = hs[l][:, -1, :] if mode == "last" else (hs[l]*m).sum(1)/den
            out[i:i+len(ch), j, :] = v.float().cpu().numpy().astype(np.float16)
    return out

# AYRI ADAY KUMELERI:
#   kopru probu  -> yalnizca ULKELER  ("dogru ulkeyi sec", "ulke oldugunu anla" degil)
#   cevap probu  -> yalnizca CEVAPLAR
BR_LIST = sorted({k["bridge"] for k in kb}); BR_IX = {e: i for i, e in enumerate(BR_LIST)}
AN_LIST = sorted({k["e3"] for k in kb});     AN_IX = {e: i for i, e in enumerate(AN_LIST)}
print(f"aday kumeleri -> kopru: {len(BR_LIST)} ulke | cevap: {len(AN_LIST)} cevap")

if cached("states.npy"):
    States = np.load(cpath("states.npy")); EB = np.load(cpath("eb.npy")); EA = np.load(cpath("ea.npy"))
    print("onbellekten:", States.shape)
else:
    States = enc_layers([prompt(k["question"]) for k in kb], "last", 320, LAYERS)
    EB = enc_layers(BR_LIST, "mean", 32, TGT_LAYERS)
    EA = enc_layers(AN_LIST, "mean", 32, TGT_LAYERS)
    np.save(cpath("states.npy"), States); np.save(cpath("eb.npy"), EB); np.save(cpath("ea.npy"), EA)
    print("kaydedildi:", States.shape)

def norml(E): return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
ENT = {"bridge": [norml(EB[:, t, :].astype(np.float32)) for t in range(len(TGT_LAYERS))],
       "answer": [norml(EA[:, t, :].astype(np.float32)) for t in range(len(TGT_LAYERS))]}
Y    = {"bridge": np.array([BR_IX[k["bridge"]] for k in kb]),
        "answer": np.array([AN_IX[k["e3"]]     for k in kb])}
CAND = {"bridge": np.arange(len(BR_LIST)), "answer": np.arange(len(AN_LIST))}
PERSON = np.array([k["person"] for k in kb])

def make_split(keyfn, salt):
    uniq = sorted({keyfn(k) for k in kb})
    rs = random.Random(SEED + salt); rs.shuffle(uniq)
    n1, n2 = int(.6*len(uniq)), int(.8*len(uniq))
    grp = {e: ("tr" if i < n1 else "va" if i < n2 else "te") for i, e in enumerate(uniq)}
    sp = np.array([grp[keyfn(k)] for k in kb])
    TR, VA, TE = sp == "tr", sp == "va", sp == "te"
    assert not ({keyfn(kb[i]) for i in np.where(TR)[0]} &
                {keyfn(kb[i]) for i in np.where(TE)[0]}), "SIZINTI"
    return TR, VA, TE, len(uniq)

SPL = {"bridge": make_split(lambda k: k["bridge"], 5),
       "answer": make_split(lambda k: k["e3"], 6)}
for k, (TR, VA, TE, nu) in SPL.items():
    print(f"{k:7s}: benzersiz hedef {nu:4d} -> train {TR.sum():4d} val {VA.sum():4d} test {TE.sum():4d}")
    if TR.sum() < MIN_TRAIN: print(f"  !! train {TR.sum()} < {MIN_TRAIN}; PCA_DIM dusur.")

def pca_basis(Xtr, k):
    mu = Xtr.mean(0); _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    return mu, Vt[:int(min(k, Vt.shape[0], Xtr.shape[0]-1))].T

def ridge_solve(G, B, lam): return np.linalg.solve(G + lam*np.eye(G.shape[0]), B)

def nrank(E, P, y_true, cand):
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    S = P @ E[cand].T
    out = np.empty(len(y_true))
    for i, yt in enumerate(y_true):
        g = S[i, yt]
        out[i] = ((S[i] > g).sum() + ((S[i] == g).sum() - 1)/2) / max(1, len(cand)-1)
    return out

def bci_cluster(vals, clusters, salt=0, nb=N_BOOT):
    """Kume (kisi) bootstrap: ayni kisi birden cok kategoride -> ornekler bagimsiz degil."""
    vals = np.asarray(vals, float)
    ok = np.isfinite(vals)
    if ok.sum() < 5: return (np.nan, np.nan)
    uq = np.unique(clusters[ok]); by = {c: np.where(ok & (clusters == c))[0] for c in uq}
    r = np.random.RandomState((SEED + salt) % (2**31 - 1)); meds = []
    for _ in range(nb):
        pick = uq[r.randint(0, len(uq), len(uq))]
        idx = np.concatenate([by[c] for c in pick])
        meds.append(np.median(vals[idx]))
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))

res, raw, null, valmed, hyper = {}, {}, {}, {}, {}
for tgt in ("bridge", "answer"):
    TR, VA, TE, _ = SPL[tgt]; y = Y[tgt]; cand = CAND[tgt]
    res[tgt], raw[tgt], null[tgt], valmed[tgt] = {}, {}, {}, {}
    pte = np.random.RandomState(SEED + 77).permutation(int(TE.sum()))
    for j, L in enumerate(tqdm(LAYERS, desc=f"katman[{tgt}]")):
        X = States[:, j, :].astype(np.float32)
        mu, V = pca_basis(X[TR], PCA_DIM)
        Ztr, Zva, Zte = (X[TR]-mu)@V, (X[VA]-mu)@V, (X[TE]-mu)@V
        G = Ztr.T @ Ztr
        best = None
        for ti, E in enumerate(ENT[tgt]):
            B = Ztr.T @ E[y[TR]]
            for lam in LAMBDAS:
                A_ = ridge_solve(G, B, lam)
                mv = float(np.median(nrank(E, Zva @ A_, y[VA], cand)))
                if best is None or mv < best[0]: best = (mv, A_, ti, lam)
        _, A_, ti, lam = best
        hyper[(tgt, L)] = (TGT_LAYERS[ti], lam); valmed[tgt][L] = best[0]
        P = Zte @ A_
        res[tgt][L]  = nrank(ENT[tgt][ti], P, y[TE], cand)
        null[tgt][L] = nrank(ENT[tgt][ti], P[pte], y[TE], cand)
        raw[tgt][L]  = nrank(ENT[tgt][ti], X[TE], y[TE], cand)

np.savez_compressed(cpath("probe.npz"), layers=np.array(LAYERS),
                    **{f"{t}_{k}": np.array([d[t][L] for L in LAYERS])
                       for t in ("bridge", "answer") for k, d in
                       (("probe", res), ("raw", raw), ("null", null))})
for tgt in ("bridge", "answer"):
    tl = [hyper[(tgt, L)][0] for L in LAYERS]; lm = [hyper[(tgt, L)][1] for L in LAYERS]
    print(f"{tgt:7s}: hedef katman { {x: tl.count(x) for x in sorted(set(tl))} }"
          f" | lambda { {x: lm.count(x) for x in sorted(set(lm))} }")
tick("Asama B bitti")

# =====================================================================
banner("6/6  KARAR")
Lb = LAYERS[int(np.argmin([valmed["bridge"][L] for L in LAYERS]))]
La = LAYERS[int(np.argmin([valmed["answer"][L] for L in LAYERS]))]
PB, PA = PERSON[SPL["bridge"][2]], PERSON[SPL["answer"][2]]
mb = np.array([np.median(res["bridge"][L]) for L in LAYERS])
ma = np.array([np.median(res["answer"][L]) for L in LAYERS])
cb = bci_cluster(res["bridge"][Lb], PB, 11)
ca = bci_cluster(res["answer"][La], PA, 12)
nb_ci = bci_cluster(null["bridge"][Lb], PB, 13)
med_b, med_a = float(np.median(res["bridge"][Lb])), float(np.median(res["answer"][La]))
med_nb = float(np.median(null["bridge"][Lb]))
rb = float(np.median(raw["bridge"][Lb]))
win = (La - Lb) / max(1, N_LAYERS)

print(f"model {MODEL_ID} | veri CC | {len(kb)} soru, {len(set(PERSON))} kisi")
print(f"aday: kopru {len(BR_LIST)} ulke, cevap {len(AN_LIST)} | sans 0.500 (asagi = iyi)")
print(f"test: kopru {SPL['bridge'][2].sum()} soru / {len(set(PB))} kisi | "
      f"cevap {SPL['answer'][2].sum()} soru / {len(set(PA))} kisi")
print("katman VAL'da secildi; sayilar TEST'te, GA'lar KISI kumelerinde bootstrap.")
print("-" * 72)
print(f"  KOPRU (ulke)  katman {Lb:3d}  medyan {med_b:.3f}  GA [{cb[0]:.3f}, {cb[1]:.3f}]")
print(f"  permutasyon nulu       medyan {med_nb:.3f}  GA [{nb_ci[0]:.3f}, {nb_ci[1]:.3f}]")
print(f"  CEVAP         katman {La:3d}  medyan {med_a:.3f}  GA [{ca[0]:.3f}, {ca[1]:.3f}]")
print(f"  ham kontrol (projeksiyonsuz, katman {Lb}): {rb:.3f}")
print(f"  PENCERE: L{Lb} -> L{La} = {La-Lb} katman (%{100*win:.1f})  [esik: %{100*MIN_WINDOW_FRAC:.0f}]")
print("-" * 72)

ci_ok      = np.isfinite(cb[1]) and cb[1] < BRIDGE_CI_MAX
beats_null = np.isfinite(cb[1]) and np.isfinite(nb_ci[0]) and cb[1] < nb_ci[0]
R = ("ADRES YOK" if not ci_ok else
     "SADECE TIP SINYALI" if not beats_null else
     "KISAYOL" if not (Lb < La) else
     "PENCERE DAR" if win < MIN_WINDOW_FRAC else "GEC")
print("SONUC:", R, "\n")

if R == "ADRES YOK":
    print("Ezberleyemeyen, tam denetimli bir prob bile ulkeyi cikaramiyor.")
    print("EYLEM: kapsam degisir - lookup projesi degil. (Dogrusal olmayan baslik")
    print("       bulabilir; iptal degil, kapsam degisikligi.)")
elif R == "SADECE TIP SINYALI":
    print(f"Prob sansin ustunde ({med_b:.3f}) ama nulden ({med_nb:.3f}) ayirt edilemiyor.")
    print("EYLEM: adres olarak kullanilamaz.")
elif R == "KISAYOL":
    print(f"Ulke L{Lb}, cevap L{La} - ulke cevaptan ONCE cozulmuyor.")
    print("Model muhtemelen kisi->cevap kisayolunu kullaniyor.")
    print("EYLEM: ara adim yok; kisayol kullanmayan orneklerle tekrarla.")
elif R == "PENCERE DAR":
    print(f"Ulke once cozuluyor ama arada {La-Lb} katman var (%{100*win:.1f}).")
    print("EYLEM: daha derin / looped backbone'da tekrarla.")
    print("NOT: esik gurultu olceginde - argmin'ler oynayabilir.")
else:
    print(f"Ulke L{Lb}'de, cevap L{La}'de - arada {La-Lb} katman (%{100*win:.1f}).")
    print("Hem adres var hem kullanilabilecek pencere var.")
    print("EYLEM: Deney 2 - projeksiyonu donguye bagla, merdiven testi (R_max=1..4).")

if rb > 0.45 and ci_ok:
    print(f"\nNOT: ham kontrol {rb:.3f} (sansta) iken prob {med_b:.3f}.")
    print("     Projeksiyonsuz bir tasarim burada YANLIS NEGATIF verirdi.")

json.dump(dict(model=MODEL_ID, dataset="CC", cats=CATEGORIES, stage_a=STAGE_A, result=R,
               A=A, hint_fmt=fmt, gain=float(gain), mis=float(mis), reads_hint=bool(READS_HINT),
               n_kept=len(kept), n_stageB=len(kb), n_person=len(set(PERSON)),
               n_bridge_cand=len(BR_LIST), n_answer_cand=len(AN_LIST),
               L_bridge=int(Lb), L_answer=int(La), window_frac=float(win),
               layer_selected_on="val", ci_method="cluster_bootstrap_by_person",
               med_bridge=med_b, ci_bridge=list(cb), med_null=med_nb, ci_null=list(nb_ci),
               beats_null=bool(beats_null), med_answer=med_a, ci_answer=list(ca),
               raw_bridge=rb), open(cpath("verdict.json"), "w"), indent=1)

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(LAYERS, mb, "-o", color="tab:red",  lw=2, ms=4, label="KOPRU (ulke) — egitilmis prob")
    ax.plot(LAYERS, ma, "-o", color="tab:blue", lw=2, ms=4, label="CEVAP — egitilmis prob")
    ax.plot(LAYERS, [np.median(null["bridge"][L]) for L in LAYERS], "-",
            color="tab:gray", lw=1.5, label="permutasyon nulu (kopru)")
    ax.plot(LAYERS, [np.median(raw["bridge"][L]) for L in LAYERS], "--",
            color="tab:gray", lw=1.2, alpha=.7, label="kopru, HAM (projeksiyonsuz)")
    ax.axhline(0.5, ls="--", color="k", lw=1, label="sans")
    ax.axvline(Lb, color="tab:red", ls=":", lw=1.5); ax.axvline(La, color="tab:blue", ls=":", lw=1.5)
    if La > Lb:
        ax.axvspan(Lb, La, color="tab:green", alpha=.10)
        ax.text((Lb+La)/2, .06, f"pencere %{100*win:.0f}", ha="center", color="tab:green")
    ax.set_ylim(0, 1); ax.invert_yaxis(); ax.grid(alpha=.3)
    ax.set_xlabel("katman"); ax.set_ylabel("medyan normalize sira (asagi = iyi)")
    ax.set_title(f"{MODEL_ID} — Compositional Celebrities; test hedefleri egitimde GORULMEDI")
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout(); plt.savefig(cpath("kopru.png"), dpi=150)
    print("\ngrafik ->", cpath("kopru.png"))
    try:
        from IPython.display import Image, display; display(Image(filename=cpath("kopru.png")))
    except Exception: pass
except Exception as e:
    print("grafik cizilemedi:", str(e)[:100])

print("\nkarar ozeti ->", cpath("verdict.json"))
tick("bitti")
