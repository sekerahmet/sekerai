# -*- coding: utf-8 -*-
"""
KAPI DENEYI — Uretim dinamikleri, sorunun kendisinin otesinde bilgi tasiyor mu?

TEK SORU:
  Bir LLM'nin cevabinin dogru olup olmayacagini, SORUYU okuduktan sonra (P0) ne
  kadar iyi tahmin edebiliyoruz? Uretim ilerledikce (P10..Pfull) bu tahmin
  ANLAMLI sekilde iyilesiyor mu?

  P0 ~ Pfull  ->  dinamikler bilgi tasimiyor; elimizdeki "soru zorlugu tahmincisi".
                  Erken cekimserlik projesi DURUR.
  Pk >> P0    ->  dinamikler bilgi tasiyor; egri en erken kullanisli k'yi verir.

TASARIM KARARLARI (onceki projelerde ogrenilen hatalar):
  * P0 birinci sinif kol — sonradan eklenen kontrol degil.
  * BIRINCIL karsilastirma ONCEDEN SABIT: Pfull vs P0. k egrisi KESIFSEL.
    ("en iyi k"yi secip onun GA'sini raporlamak kazananin lanetidir.)
  * Esli bootstrap: her iki kol AYNI yeniden orneklenen sorularda degerlendirilir.
  * Katman ONCEDEN SABIT (0.65 derinlik). Diger katmanlar kesifsel.
  * Hiperparametre ic katlamada secilir; dis katlama yalnizca raporlar.
  * "YETERSIZ VERI" ayri bir sonuctur, basarisizlik degildir.
  * Maliyet/tasarruf iddiasi BU ASAMADA YOK. Once bilgi egrisi.

Colab:
    !pip -q install -U transformers datasets accelerate scikit-learn
    %run kapi_deneyi.py
"""

# =====================================================================
# AYARLAR
# =====================================================================
MODEL_ID  = "Qwen/Qwen3-1.7B"
N_Q       = 500          # soru sayisi
MAX_NEW   = 320          # uretim uzunlugu
KPOINTS   = [0, 10, 25, 50, 100]   # + "full" otomatik eklenir
MIN_LEN   = 110          # tum k noktalari tanimli olsun diye asgari uretim uzunlugu
LAYER_FRACS = (0.35, 0.65, 1.0)    # 0.65 = BIRINCIL, digerleri kesifsel
PRIMARY_FRAC = 0.65
PCA_DIM   = 32
CS        = (0.01, 0.1, 1.0)       # logistic regularizasyon izgarasi (ic katlama)
N_FOLD, N_BOOT, SEED = 5, 4000, 0

# --- onceden yazilmis karar kurali ---
DELTA_MIN   = 0.03       # AUROC(Pfull) - AUROC(P0) alt siniri bunu gecmeli
CI_TOO_WIDE = 0.15       # GA bundan genisse: YETERSIZ VERI

GEN_BATCH, FWD_BATCH = 16, 4
USE_DRIVE, DRIVE_DIR = True, "/content/drive/MyDrive/kapi"

# =====================================================================
import os, io, re, json, time, random
import numpy as np, torch

T0 = time.time()
def banner(s): print("\n" + "=" * 74); print(s); print("=" * 74, flush=True)
def tick(s):  print(f"[{time.time()-T0:6.0f}s] {s}", flush=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

banner("0/5  ORTAM")
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0); print("gpu:", p.name, f"{p.total_memory/1e9:.1f} GB")
else:
    print("!! GPU YOK -> Runtime > Change runtime type > T4 GPU")
if USE_DRIVE:
    try:
        from google.colab import drive; drive.mount("/content/drive"); WORK = DRIVE_DIR
    except Exception as e:
        print("Drive yok (%s)" % str(e)[:60]); WORK = "/content/kapi"
else:
    WORK = "/content/kapi"
os.makedirs(WORK, exist_ok=True)
SLUG = MODEL_ID.split("/")[-1].replace(".", "-")
STAMP = f"{SLUG}__n{N_Q}__m{MAX_NEW}"
def cp(n): return os.path.join(WORK, f"{STAMP}__{n}")
print("dizin:", WORK, "| damga:", STAMP)

# =====================================================================
banner("1/5  VERI — GSM8K")
from datasets import load_dataset
ds = None
for cid, cfg in (("openai/gsm8k", "main"), ("gsm8k", "main")):
    try:
        ds = load_dataset(cid, cfg, split="test"); print("YUKLENDI:", cid, "|", len(ds)); break
    except Exception as e:
        print("olmadi:", cid, "->", str(e)[:120])
if ds is None: raise SystemExit("GSM8K yuklenemedi.")

def gold_of(a):
    m = re.search(r"####\s*([-+]?[\d,]*\.?\d+)", a)
    return m.group(1).replace(",", "") if m else None

items = []
for ex in ds:
    g = gold_of(ex["answer"])
    if g is not None: items.append(dict(q=ex["question"], gold=g))
    if len(items) >= N_Q: break
print(f"{len(items)} soru | ornek altin cevap: {items[0]['gold']}")

# =====================================================================
banner("2/5  MODEL + URETIM")
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
NL, H = model.config.num_hidden_layers, model.config.hidden_size
LAYERS = sorted({int(round(f*NL)) for f in LAYER_FRACS})
PRIMARY_L = int(round(PRIMARY_FRAC*NL))
print(f"{MODEL_ID}: {NL} katman, hidden={H} | kaydedilecek katmanlar {LAYERS} "
      f"| BIRINCIL L{PRIMARY_L}")

INSTR = ("Solve the problem step by step. "
         "Finish with the final answer on its own line as: #### <number>")
def build_prompt(q):
    msgs = [{"role": "user", "content": INSTR + "\n\n" + q}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def pred_of(text):
    m = re.search(r"####\s*([-+]?[\d,]*\.?\d+)", text)
    if m: return m.group(1).replace(",", "")
    nums = re.findall(r"[-+]?[\d,]*\.?\d+", text)
    return nums[-1].replace(",", "") if nums else None

def same_num(a, b):
    try: return abs(float(a) - float(b)) < 1e-4
    except Exception: return False

GEN_CACHE = cp("gen.json")
if os.path.exists(GEN_CACHE):
    gens = json.load(io.open(GEN_CACHE, encoding="utf-8")); print("uretim onbellekten")
else:
    tok.padding_side = "left"
    gens = []
    prompts = [build_prompt(it["q"]) for it in items]
    for i in tqdm(range(0, len(prompts), GEN_BATCH), desc="uretim"):
        enc = tok(prompts[i:i+GEN_BATCH], return_tensors="pt", padding=True,
                  truncation=True, max_length=768).to(dev)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        for row in new:
            ids = [int(t) for t in row if int(t) != tok.pad_token_id]
            gens.append(dict(text=tok.decode(ids, skip_special_tokens=True), n_tok=len(ids)))
    json.dump(gens, io.open(GEN_CACHE, "w", encoding="utf-8"), ensure_ascii=False)

for it, g in zip(items, gens):
    it["gen"] = g["text"]; it["n_tok"] = g["n_tok"]
    it["pred"] = pred_of(g["text"])
    it["ok"] = bool(it["pred"] is not None and same_num(it["pred"], it["gold"]))

acc = float(np.mean([it["ok"] for it in items]))
lens = np.array([it["n_tok"] for it in items])
print(f"\ndogruluk: %{100*acc:.1f}  ({sum(it['ok'] for it in items)}/{len(items)})")
print(f"uretim uzunlugu: medyan {np.median(lens):.0f}, %10 {np.percentile(lens,10):.0f}, "
      f"max {lens.max()}")
if acc < 0.15 or acc > 0.85:
    print(f"!! UYARI: taban orani ucta (%{100*acc:.0f}); AUROC kararsiz olabilir.")

use = [it for it in items if it["n_tok"] >= MIN_LEN]
print(f"tum k noktalari tanimli olan soru: {len(use)}/{len(items)} "
      f"(>= {MIN_LEN} token)  -- esli karsilastirma bu altkumede")
if len(use) < 150:
    print("!! UYARI: 150 altinda; GA'lar genis cikacak.")
tick("uretim bitti")

# =====================================================================
banner("3/5  OZELLIK CIKARIMI (ogretmen-zorlamali tek gecis)")
# generate(output_hidden_states=True) adim adim devasa bellek ister.
# Bunun yerine prompt+uretim dizisini TEK seferde ileri besliyoruz: hem tum
# pozisyonlarin gizli durumlari hem de her adimin logit istatistikleri cikiyor.
KS = list(KPOINTS)
FEAT_CACHE = cp("feats.npz")
if os.path.exists(FEAT_CACHE):
    Z = np.load(FEAT_CACHE)
    HS, SC, Y = Z["HS"], Z["SC"], Z["Y"]
    print("ozellikler onbellekten:", HS.shape, SC.shape)
else:
    tok.padding_side = "right"
    nK = len(KS) + 1                       # + full
    HS = np.zeros((len(use), nK, len(LAYERS), H), dtype=np.float16)
    SC = np.zeros((len(use), nK, 6), dtype=np.float32)   # ucuz skaler ozellikler
    for i in tqdm(range(0, len(use), FWD_BATCH), desc="ileri gecis"):
        chunk = use[i:i+FWD_BATCH]
        pr = [build_prompt(it["q"]) for it in chunk]
        pid = [tok(p, add_special_tokens=False)["input_ids"] for p in pr]
        gid = [tok(it["gen"], add_special_tokens=False)["input_ids"] for it in chunk]
        full = [a + b for a, b in zip(pid, gid)]
        mx = max(len(f) for f in full)
        ids = torch.full((len(full), mx), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(full), mx), dtype=torch.long)
        for b, f in enumerate(full):
            ids[b, :len(f)] = torch.tensor(f); att[b, :len(f)] = 1
        ids, att = ids.to(dev), att.to(dev)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=att, output_hidden_states=True)
        logits = out.logits.float()
        for b, it in enumerate(chunk):
            np_, ng = len(pid[b]), len(gid[b])
            # uretim adimi t'nin dagilimi, pozisyon (np_-1+t) logitlerinden gelir
            st = np_ - 1
            lg = logits[b, st:st+ng, :]                      # [ng, V]
            pr_ = torch.softmax(lg, -1)
            top1 = pr_.max(-1).values.cpu().numpy()
            ent = (-(pr_ * torch.log(pr_ + 1e-9)).sum(-1)).cpu().numpy()
            for ki, k in enumerate(KS + ["full"]):
                kk = ng if k == "full" else min(int(k), ng)
                pos = np_ - 1 + kk                            # k token uretilmis durum
                pos = min(pos, len(full[b]) - 1)
                for li, L in enumerate(LAYERS):
                    HS[i+b, ki, li, :] = out.hidden_states[L][b, pos, :].float().cpu().numpy()
                w = top1[:max(kk, 1)]; e = ent[:max(kk, 1)]
                SC[i+b, ki] = [w[-1] if kk > 0 else 1.0,     # maxprob_k
                               e[-1] if kk > 0 else 0.0,     # entropy_k
                               w.min() if kk > 0 else 1.0,   # running-min top1
                               e.mean() if kk > 0 else 0.0,  # running-mean entropy
                               float(kk),                    # uretilen token
                               float(np_)]                   # prompt uzunlugu
        del out, logits
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    Y = np.array([1 if it["ok"] else 0 for it in use], dtype=np.int64)
    np.savez_compressed(FEAT_CACHE, HS=HS, SC=SC, Y=Y)
    print("kaydedildi:", HS.shape, SC.shape)
print(f"etiket dengesi: dogru %{100*Y.mean():.1f}")
tick("ozellikler hazir")

# =====================================================================
banner("4/5  PROBLAR — 5 katlamali capraz dogrulama, kat-disi tahminler")
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

N = len(Y); KNAMES = [f"P{k}" for k in KS] + ["Pfull"]
PL = LAYERS.index(PRIMARY_L)
skf = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)
FOLDS = list(skf.split(np.zeros(N), Y))

def oof_hidden(X):
    """Kat-disi olasilik tahminleri. C ic katlamada secilir, dis katlama yalniz raporlar."""
    oof = np.zeros(N)
    for tr, te in FOLDS:
        best, bc = -1, CS[0]
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED+1)
        for C in CS:
            sc = []
            for itr, ite in inner.split(np.zeros(len(tr)), Y[tr]):
                pipe = make_pipeline(StandardScaler(), PCA(n_components=min(PCA_DIM, len(itr)-1),
                                                           random_state=SEED),
                                     LogisticRegression(C=C, max_iter=2000))
                pipe.fit(X[tr][itr], Y[tr][itr])
                sc.append(roc_auc_score(Y[tr][ite], pipe.predict_proba(X[tr][ite])[:, 1]))
            if np.mean(sc) > best: best, bc = np.mean(sc), C
        pipe = make_pipeline(StandardScaler(), PCA(n_components=min(PCA_DIM, len(tr)-1),
                                                   random_state=SEED),
                             LogisticRegression(C=bc, max_iter=2000))
        pipe.fit(X[tr], Y[tr]); oof[te] = pipe.predict_proba(X[te])[:, 1]
    return oof

def oof_cheap(X):
    oof = np.zeros(N)
    for tr, te in FOLDS:
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        pipe.fit(X[tr], Y[tr]); oof[te] = pipe.predict_proba(X[te])[:, 1]
    return oof

ARMS = {}
for ki, nm in enumerate(KNAMES):
    ARMS[f"hidden_{nm}"] = oof_hidden(HS[:, ki, PL, :].astype(np.float32))
# ucuz baseline'lar (Pfull noktasindaki degerlerle)
fi = len(KNAMES) - 1
ARMS["maxprob"]  = SC[:, fi, 0]
ARMS["entropy"]  = -SC[:, fi, 1]
ARMS["runmin"]   = SC[:, fi, 2]
ARMS["runent"]   = -SC[:, fi, 3]
ARMS["length"]   = -SC[:, fi, 4]
ARMS["cheapcombo"] = oof_cheap(SC[:, fi, :])

AUC = {k: float(roc_auc_score(Y, v)) for k, v in ARMS.items()}
tick("problar bitti")

# =====================================================================
banner("5/5  KARAR")
def boot_delta(a, b, salt=0, nb=N_BOOT):
    """Esli bootstrap: iki kol AYNI yeniden orneklenen sorularda degerlendirilir."""
    r = np.random.RandomState(SEED + salt); d = []
    for _ in range(nb):
        idx = r.randint(0, N, N)
        if len(np.unique(Y[idx])) < 2: continue
        d.append(roc_auc_score(Y[idx], a[idx]) - roc_auc_score(Y[idx], b[idx]))
    d = np.array(d)
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

print(f"n = {N} soru | taban dogruluk %{100*Y.mean():.1f} | birincil katman L{PRIMARY_L}")
print("-" * 74)
print("  KOL                      AUROC")
for nm in KNAMES: print(f"  hidden_{nm:8s}          {AUC['hidden_'+nm]:.3f}")
print("  " + "-" * 40)
for nm in ("maxprob", "entropy", "runmin", "runent", "length", "cheapcombo"):
    print(f"  {nm:22s}   {AUC[nm]:.3f}")
print("-" * 74)

d, lo, hi = boot_delta(ARMS["hidden_Pfull"], ARMS["hidden_P0"], 1)
print("BIRINCIL (onceden sabit):  AUROC(Pfull) - AUROC(P0)")
print(f"   delta = {d:+.3f}   %95 GA [{lo:+.3f}, {hi:+.3f}]   [esik: alt sinir > {DELTA_MIN:+.2f}]")

if lo > DELTA_MIN:                      V = "DINAMIK BILGI VAR"
elif (hi - lo) > CI_TOO_WIDE:           V = "YETERSIZ VERI"
else:                                   V = "DINAMIK BILGI YOK"
print("\nSONUC:", V)
if V == "DINAMIK BILGI VAR":
    print("  Uretim dinamikleri, sorunun otesinde bilgi tasiyor.")
    print("  -> Asagidaki k egrisine bak: en erken kullanisli nokta neresi?")
    print("  -> SONRAKI ADIM: zaman serisi ozellikleri + erken cekimserlik politikasi,")
    print("     ve ANCAK O ZAMAN token tasarrufu olcumu.")
elif V == "YETERSIZ VERI":
    print(f"  GA genisligi {hi-lo:.3f} > {CI_TOO_WIDE}. Karar verilemiyor - BASARISIZLIK DEGIL.")
    print("  -> N_Q artir (1000-2000) ya da taban dogrulugu %50'ye yakin bir model/veri sec.")
else:
    print("  Uretim dinamikleri, soruyu okumanin otesinde anlamli bilgi EKLEMIYOR.")
    print("  Elimizdeki sey bir 'soru zorlugu tahmincisi'; erken cekimserlik tezi dusuyor.")
    print("  -> DUR. Zaman serisi modeli / davranis bloklari bunu kurtarmaz.")

print("\n--- KESIFSEL (karar bunlara dayanmaz) ---")
for nm in KNAMES[1:]:
    dd, ll, hh = boot_delta(ARMS["hidden_"+nm], ARMS["hidden_P0"], 10+KNAMES.index(nm))
    print(f"  {nm:6s} - P0 : {dd:+.3f}  GA [{ll:+.3f}, {hh:+.3f}]")
dc, lc, hc = boot_delta(ARMS["hidden_Pfull"], ARMS["cheapcombo"], 50)
print(f"  hidden_Pfull - cheapcombo : {dc:+.3f}  GA [{lc:+.3f}, {hc:+.3f}]")
print("  (gizli durum, ucuz skalerlerin otesinde bir sey katiyor mu?)")

json.dump(dict(model=MODEL_ID, verdict=V, n=int(N), base_acc=float(Y.mean()),
               primary_layer=int(PRIMARY_L), auroc=AUC,
               delta_full_p0=d, ci=[lo, hi], gen_acc=acc,
               n_total=len(items), n_used=len(use)),
          open(cp("verdict.json"), "w"), indent=1)

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    xs = KS + [int(np.median([it["n_tok"] for it in use]))]
    ys = [AUC["hidden_"+nm] for nm in KNAMES]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.plot(xs, ys, "-o", lw=2, color="tab:red", label="gizli durum probu")
    for nm, c in (("cheapcombo", "tab:blue"), ("runmin", "tab:green"), ("maxprob", "tab:gray")):
        ax.axhline(AUC[nm], ls="--", lw=1.2, color=c, label=f"{nm} ({AUC[nm]:.3f})")
    ax.axhline(0.5, ls=":", color="k", lw=1, label="sans")
    ax.axhline(AUC["hidden_P0"], ls="-", lw=1.2, color="tab:orange",
               label=f"P0 — soru tek basina ({AUC['hidden_P0']:.3f})")
    ax.set_xlabel("uretilen token sayisi"); ax.set_ylabel("AUROC")
    ax.set_title(f"{MODEL_ID} — GSM8K, n={N}  |  SONUC: {V}")
    ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout(); plt.savefig(cp("kapi.png"), dpi=150)
    print("\ngrafik ->", cp("kapi.png"))
    try:
        from IPython.display import Image, display; display(Image(filename=cp("kapi.png")))
    except Exception: pass
except Exception as e:
    print("grafik cizilemedi:", str(e)[:90])
print("karar ozeti ->", cp("verdict.json")); tick("bitti")
