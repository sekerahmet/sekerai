# -*- coding: utf-8 -*-
"""
Kopru Adreslenebilirligi — Asama A + B  (tek dosya)

Colab'da calistirma:
    !pip -q install -U transformers datasets accelerate
    %run kopru.py

NOT: '%run' kullan, '!python' DEGIL. Drive baglama ayni cekirdekte calismak zorunda.

Ne sinar:
  ASAMA A  Cok-hop basarisizligi bir ADRESLEME problemi mi?
           (kopruyu bedava verince model toparliyor mu)
  ASAMA B  Kopru latent durumda var mi, ve cevaptan yeterince once mi?
           (egitilmis ridge prob, ezberleyemeyecek sekilde bolunmus)

Sonuclar ve eylemleri:
  ONCUL YANLIS          -> DUR. Darbogaz adresleme degil.
  IPUCU KULLANILAMIYOR  -> Belirsiz. Once talimat-izleme sorununu coz.
  ADRES YOK             -> Kapsam degisir: lookup degil, egitim problemi.
  SADECE TIP SINYALI    -> Prob kimligi degil kategoriyi buluyor; adres olarak kullanilamaz.
  KISAYOL               -> Model e1->e3 kisayolu kullaniyor, ara adim yok.
  PENCERE DAR           -> Adres var ama kullanilacak hesap zamani yok.
  GEC                   -> Deney 2: donguye bagla, merdiven testi.

Istatistik notu: katman VAL setinde secilir, tum sayilar TEST setinde raporlanir
(kazananin laneti). Permutasyon nulu, tip-duzeyi sinyali kimlik sanmayi engeller.
"""

# =====================================================================
# AYARLAR  — degistirmek isteyebilecegin tek yer burasi
# =====================================================================
MODEL_ID = "Qwen/Qwen3-1.7B"
# Alternatifler: "Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-1.5B", "meta-llama/Llama-3.2-1B"

MAX_ITEMS = 4000      # bilgi filtresine sokulacak aday soru
PCA_DIM   = 256
LAMBDAS   = (1.0, 10.0, 100.0, 1000.0, 10000.0)
TGT_FRACS = (0.35, 0.65, 1.0)   # varlik gommesi icin aday katmanlar (derinlik orani)

GATE_GAIN    = 0.15   # ASAMA A kapisi: (dogru ipucu) - (ipucusuz) >= bu
HINT_CEILING = 0.60   # dogru-ipucu dogrulugu bunun altindaysa: IPUCU KULLANILAMIYOR

BRIDGE_CI_MAX   = 0.45  # kopru GA ust siniri bunun altinda olmali
MIN_WINDOW_FRAC = 0.15  # (L_cevap - L_kopru) / katman sayisi

MIN_KEPT, MIN_TRAIN = 150, 120
GEN_BATCH, ENC_BATCH, N_BOOT, SEED = 32, 64, 2000, 0

USE_DRIVE = True                              # False -> /content (oturum kopunca kaybolur)
DRIVE_DIR = "/content/drive/MyDrive/kopru"

# =====================================================================
import os, sys, json, re, string, random, time
import numpy as np
import torch

T0 = time.time()
def banner(s):
    print("\n" + "=" * 72); print(s); print("=" * 72, flush=True)
def tick(s):
    print(f"[{time.time()-T0:6.0f}s] {s}", flush=True)

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
        from google.colab import drive
        drive.mount("/content/drive")
        WORK = DRIVE_DIR
    except Exception as e:
        print("Drive baglanamadi (%s) -> /content kullanilacak" % str(e)[:80])
        WORK = "/content/kopru"
else:
    WORK = "/content/kopru"
os.makedirs(WORK, exist_ok=True)

MSLUG = MODEL_ID.split("/")[-1].replace(".", "-")
print("calisma dizini:", WORK)
print("(kopan oturumda ayni komutu tekrar calistir - tamamlanan adimlar atlanir)")
# NOT: onbellek damgasi (STAMP) veri hatti kurulduktan SONRA tanimlanir; icinde
# cikarilan zincirlerin parmak izi vardir. Boylece to_triples() duzenlenirse
# eski onbellek sessizce yeniden kullanilmaz.

# =====================================================================
banner("1/6  VERI SEMASI  (kirilgan tek nokta - burada duruyorsa bana ciktiyi gonder)")
from datasets import load_dataset

CANDIDATES = [("xanhho/2WikiMultihopQA", "validation"),
              ("voidful/2WikiMultihopQA", "validation"),
              ("hotpotqa/hotpot_qa", "validation")]
ds = None
for cid, sp in CANDIDATES:
    try:
        ds = load_dataset(cid, split=sp); print("YUKLENDI:", cid, "|", len(ds), "ornek"); break
    except Exception as e:
        print("olmadi:", cid, "->", str(e)[:140])
if ds is None:
    print("\nHICBIR VERI SETI YUKLENEMEDI.")
    print("HuggingFace'te '2WikiMultihopQA' arayip dogru id'yi CANDIDATES'e ekle.")
    sys.exit(1)

print("\n--- alanlar ---")
for k, v in ds[0].items():
    print(f"{k:20s}: {str(v)[:180]}")


def to_triples(ex):
    """evidences -> (ozne, iliski, nesne) ucluleri. Sema farkliysa SADECE burayi duzelt."""
    ev = ex.get("evidences"); out = []
    if ev is None: return out
    if isinstance(ev, list):
        for e in ev:
            if isinstance(e, (list, tuple)) and len(e) == 3:
                out.append(tuple(str(x).strip() for x in e))
            elif isinstance(e, dict):
                ks = {k.lower(): k for k in e}
                if {"subject", "relation", "object"} <= set(ks):
                    out.append((str(e[ks["subject"]]).strip(),
                                str(e[ks["relation"]]).strip(),
                                str(e[ks["object"]]).strip()))
    elif isinstance(ev, dict):
        ks = {k.lower(): k for k in ev}
        if {"subject", "relation", "object"} <= set(ks):
            for s, r, o in zip(ev[ks["subject"]], ev[ks["relation"]], ev[ks["object"]]):
                out.append((str(s).strip(), str(r).strip(), str(o).strip()))
    return out


def find_chain(t):
    for i, (s1, r1, o1) in enumerate(t):
        for j, (s2, r2, o2) in enumerate(t):
            if i != j and o1 and s2 and o1.lower() == s2.lower() and o2 and o2.lower() != s1.lower():
                return (s1, r1, o1, r2, o2)
    return None


# tarih / sayi agirlikli adlar gomme uzayinda ayirt edilemez -> elenir
DATEY = re.compile(r"^[\d\s\-/.,]+$|\b(januar|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\b\s*\d",
                   re.IGNORECASE)
def bad_entity(e):
    e = str(e).strip()
    if len(e) < 2: return True
    return bool(DATEY.match(e)) or sum(c.isdigit() for c in e) > len(e) * 0.4


items, drop_date = [], 0
for ex in ds:
    ty = str(ex.get("type", "")).lower()
    if ty and ty not in ("compositional", "inference", "bridge"): continue
    ch = find_chain(to_triples(ex))
    if not (ch and all(ch)): continue
    e1, r1, br, r2, e3 = ch
    if bad_entity(br) or bad_entity(e3):
        drop_date += 1; continue
    items.append(dict(question=str(ex["question"]), e1=e1, r1=r1, bridge=br, r2=r2, e3=e3))
    if len(items) >= MAX_ITEMS: break

print(f"\nzincirli soru: {len(items)}   (tarih/sayi varligi nedeniyle elenen: {drop_date})")
if not items:
    print("\nHIC ZINCIR CIKMADI -> to_triples() bu semayi tutmuyor.")
    print("Yukaridaki '--- alanlar ---' dokumunu bana gonder, duzeltilmis dosyayi vereyim.")
    sys.exit(1)
print("ornek:", json.dumps(items[0], ensure_ascii=False)[:300])

# Onbellek damgasi: model + ayar + CIKARILAN ZINCIRLERIN parmak izi.
# to_triples() / bad_entity() degisirse parmak izi degisir, onbellek otomatik tazelenir.
import hashlib
_fp = hashlib.md5(json.dumps(
    [(it["e1"], it["r1"], it["bridge"], it["r2"], it["e3"]) for it in items],
    ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
STAMP = f"{MSLUG}__i{MAX_ITEMS}__{_fp}"
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
tok.padding_side = "left"                      # son gercek token daima -1 indisinde
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, trust_remote_code=True).to(device)
model.eval()

N_LAYERS, H = model.config.num_hidden_layers, model.config.hidden_size
LAYERS = list(range(N_LAYERS + 1))
TGT_LAYERS = sorted({int(round(f * N_LAYERS)) for f in TGT_FRACS})
print(f"{MODEL_ID}: {N_LAYERS} katman, hidden={H}")
print("varlik gommesi icin aday hedef katmanlar (val'da secilecek):", TGT_LAYERS)

with torch.no_grad():
    _h = model(**tok(["The capital of France is Paris."], return_tensors="pt").to(device),
               output_hidden_states=True).hidden_states
print("fp16 saglik:", "NaN/Inf VAR -> torch_dtype=torch.float32 dene"
      if any(torch.isnan(x).any() or torch.isinf(x).any() for x in _h) else "temiz")
tick("model hazir")

# =====================================================================
banner("3/6  BILGI FILTRESI")
FEW = ("Answer each question with a short factual answer.\n\n"
       "Q: What is the director of Inception?\nA: Christopher Nolan\n\n"
       "Q: What is the capital of France?\nA: Paris\n\n")
def prompt(q): return FEW + "Q: " + q + "\nA:"

def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s).translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())

def match(pred, gold):
    p, g = norm(pred), norm(gold)
    if len(g) < 3: return p == g
    return p == g or (" " + g + " ") in (" " + p + " ")

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
    a1 = gen([prompt(f"What is the {it['r1']} of {it['e1']}?") for it in items], "hop-1")
    a2 = gen([prompt(f"What is the {it['r2']} of {it['bridge']}?") for it in items], "hop-2")
    kept = [dict(it) for it, x, y in zip(items, a1, a2)
            if match(x, it["bridge"]) and match(y, it["e3"])]
    json.dump(kept, open(cpath("kept.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print(f"filtre: {len(items)} -> {len(kept)}  (her iki tek-hop olguyu da bilenler)")
if len(kept) == 0:
    print("\nFILTREDEN HIC SORU GECMEDI -> model bu olgulari bilmiyor.")
    print("Daha buyuk bir MODEL_ID dene ya da MAX_ITEMS'i artir."); sys.exit(1)
if len(kept) < MIN_KEPT:
    print(f"!! UYARI: {MIN_KEPT} altinda -> istatistik zayif, GA'lar genis cikacak.")
tick("filtre bitti")

# =====================================================================
banner("4/6  ASAMA A — ONCUL TESTI")
def hint_paren(it, b):   return it["question"] + f" (Hint: {b})"
def hint_premise(it, b): return f"Given that the {it['r1']} of {it['e1']} is {b}, {it['question']}"

if cached("stageA.json"):
    A = json.load(open(cpath("stageA.json"))); print("onbellekten")
else:
    rs = random.Random(SEED)
    pool = [it["bridge"] for it in kept]
    rand_b = []
    for it in kept:                                    # rastgele ipucu != gercek kopru
        c = pool[rs.randrange(len(pool))]; tries = 0
        while norm(c) == norm(it["bridge"]) and tries < 50:
            c = pool[rs.randrange(len(pool))]; tries += 1
        rand_b.append(c)
    runs = {
        "none":         [prompt(it["question"])                  for it in kept],
        "paren":        [prompt(hint_paren(it, it["bridge"]))    for it in kept],
        "premise":      [prompt(hint_premise(it, it["bridge"]))  for it in kept],
        "rand_paren":   [prompt(hint_paren(it, b))               for it, b in zip(kept, rand_b)],
        "rand_premise": [prompt(hint_premise(it, b))             for it, b in zip(kept, rand_b)],
    }
    A = {}
    for k, ps in runs.items():
        o = gen(ps, k)
        A[k] = float(np.mean([match(a, it["e3"]) for a, it in zip(o, kept)]))
    json.dump(A, open(cpath("stageA.json"), "w"), indent=1)

def bprop(p, n, salt=0, nb=N_BOOT):
    r = np.random.RandomState((SEED + salt) % (2**31 - 1)); k = int(round(p * n))
    x = np.array([1]*k + [0]*(n-k))
    m = x[r.randint(0, n, size=(nb, n))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

n = len(kept)
fmt = "premise" if A["premise"] >= A["paren"] else "paren"
acc_hint, acc_rand, acc_none = A[fmt], A["rand_" + fmt], A["none"]
gain, mis = acc_hint - acc_none, A["rand_" + fmt] - acc_none

print("\nher iki tek-hop olgu BILINIYOR kosuluyla, 2-hop dogrulugu:")
print("(dogrudan sorulunca hop-2 dogrulugu %100 — filtrenin garantisi)")
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
    print("  -> Kopruyu vermek belirgin sekilde yardim ediyor: darbogaz ADRESLEME.")
    if not READS_HINT:
        print("  !! rastgele ipucu zarar vermiyor -> model ipucunu okumuyor olabilir; dikkatli yorumla.")
elif STAGE_A == "IPUCU KULLANILAMIYOR":
    print("  -> Model hop-2'yi dogrudan sorulunca biliyor ama ipucu olarak verilince kullanamiyor.")
    print("     Bu ADRESLEME hakkinda bilgi VERMEZ - talimat izleme sorunu.")
    print("  EYLEM: once ipucu formatini coz (daha buyuk model / chat template / few-shot).")
else:
    print("  -> Kopruyu bedava vermek yardim etmiyor: darbogaz adresleme DEGIL.")
    print("  EYLEM: DUR. Latent adresli bellek bu problemi cozmez; tasarimi bastan kur.")
tick("Asama A bitti")

if STAGE_A != "GECTI":
    json.dump(dict(model=MODEL_ID, stage_a=STAGE_A, result=STAGE_A, A=A, hint_fmt=fmt,
                   gain=float(gain), mis=float(mis), n_kept=len(kept)),
              open(cpath("verdict.json"), "w"), indent=1)
    banner(f"SONUC: {STAGE_A}  — Asama B calistirilmadi")
    print("kaydedildi ->", cpath("verdict.json"))
    sys.exit(0)

# =====================================================================
banner("5/6  ASAMA B — CIKARILABILIRLIK VE ZAMANLAMA")

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

ent_list = sorted({it["bridge"] for it in kept} | {it["e3"] for it in kept})
ent_ix = {e: i for i, e in enumerate(ent_list)}
print("benzersiz varlik (aday kume):", len(ent_list))

if cached("states.npy"):
    States = np.load(cpath("states.npy")); EntRaw = np.load(cpath("entemb.npy"))
    print("onbellekten:", States.shape, EntRaw.shape)
else:
    States = enc_layers([prompt(it["question"]) for it in kept], "last", 320, LAYERS)
    EntRaw = enc_layers(ent_list, "mean", 32, TGT_LAYERS)
    np.save(cpath("states.npy"), States); np.save(cpath("entemb.npy"), EntRaw)
    print("kaydedildi:", States.shape, EntRaw.shape)

ENT = []
for ti in range(len(TGT_LAYERS)):
    E = EntRaw[:, ti, :].astype(np.float32)
    ENT.append(E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8))

y_bridge = np.array([ent_ix[it["bridge"]] for it in kept])
y_answer = np.array([ent_ix[it["e3"]] for it in kept])
CAND = np.arange(len(ent_list))


def make_split(keyfn, salt):
    """Hedef varliga gore bolme -> test hedefleri egitimde HIC gorulmez (ezberleme imkansiz)."""
    uniq = sorted({keyfn(it) for it in kept})
    rs = random.Random(SEED + salt); rs.shuffle(uniq)
    n1, n2 = int(.6*len(uniq)), int(.8*len(uniq))
    grp = {e: ("tr" if i < n1 else "va" if i < n2 else "te") for i, e in enumerate(uniq)}
    sp = np.array([grp[keyfn(it)] for it in kept])
    TR, VA, TE = sp == "tr", sp == "va", sp == "te"
    assert not ({keyfn(kept[i]) for i in np.where(TR)[0]} &
                {keyfn(kept[i]) for i in np.where(TE)[0]}), "SIZINTI"
    return TR, VA, TE, len(uniq)

SPL = {"bridge": make_split(lambda it: it["bridge"], 5),
       "answer": make_split(lambda it: it["e3"], 6)}
for k, (TR, VA, TE, nu) in SPL.items():
    print(f"{k:7s}: benzersiz hedef {nu:4d} -> train {TR.sum():4d} val {VA.sum():4d} test {TE.sum():4d}")
    if TR.sum() < MIN_TRAIN:
        print(f"  !! UYARI: train {TR.sum()} < {MIN_TRAIN}; PCA_DIM dusur (or. 128).")


def pca_basis(Xtr, k):
    mu = Xtr.mean(0)
    _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    return mu, Vt[:int(min(k, Vt.shape[0], Xtr.shape[0]-1))].T

def ridge_solve(G, B, lam):
    return np.linalg.solve(G + lam*np.eye(G.shape[0]), B)

def nrank(E, P, y_true, cand):
    """Hedefin aday kume icindeki normalize sirasi. 0 = en iyi, 0.5 = sans, 1 = en kotu."""
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    S = P @ E[cand].T
    pos = {c: i for i, c in enumerate(cand)}
    out = np.empty(len(y_true))
    for i, yt in enumerate(y_true):
        g = S[i, pos[yt]]
        out[i] = ((S[i] > g).sum() + ((S[i] == g).sum() - 1)/2) / max(1, len(cand)-1)
    return out

def bci(x, salt=0, nb=N_BOOT):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 5: return (np.nan, np.nan)
    r = np.random.RandomState((SEED + salt) % (2**31 - 1))
    m = np.median(x[r.randint(0, len(x), size=(nb, len(x)))], axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


res, raw, null, valmed, hyper = {}, {}, {}, {}, {}
for tgt, y in (("bridge", y_bridge), ("answer", y_answer)):
    TR, VA, TE, _ = SPL[tgt]
    res[tgt], raw[tgt], null[tgt], valmed[tgt] = {}, {}, {}, {}
    # permutasyon nulu: BASKA bir sorunun durumundan bu sorunun hedefini tahmin et.
    # Prob yalnizca "tip" ogrenmisse (kimlik degil), null da ayni skoru verir.
    pte = np.random.RandomState(SEED + 77).permutation(int(TE.sum()))
    for j, L in enumerate(tqdm(LAYERS, desc=f"katman[{tgt}]")):
        X = States[:, j, :].astype(np.float32)
        mu, V = pca_basis(X[TR], PCA_DIM)                 # katman basina TEK SVD
        Ztr, Zva, Zte = (X[TR]-mu)@V, (X[VA]-mu)@V, (X[TE]-mu)@V
        G = Ztr.T @ Ztr
        best = None
        for ti, E in enumerate(ENT):                      # hedef uzayi da hiperparametre
            B = Ztr.T @ E[y[TR]]
            for lam in LAMBDAS:
                A_ = ridge_solve(G, B, lam)
                mv = float(np.median(nrank(E, Zva @ A_, y[VA], CAND)))   # VAL'da secim
                if best is None or mv < best[0]: best = (mv, A_, ti, lam)
        _, A_, ti, lam = best
        hyper[(tgt, L)] = (TGT_LAYERS[ti], lam)
        valmed[tgt][L] = best[0]        # KATMAN SECIMI bu val medyani uzerinden yapilir
        P = Zte @ A_
        res[tgt][L]  = nrank(ENT[ti], P, y[TE], CAND)          # TEST'te rapor
        null[tgt][L] = nrank(ENT[ti], P[pte], y[TE], CAND)     # permutasyon nulu
        raw[tgt][L]  = nrank(ENT[ti], X[TE], y[TE], CAND)      # projeksiyonsuz kontrol

np.savez_compressed(cpath("probe.npz"), layers=np.array(LAYERS),
                    **{f"{t}_{k}": np.array([d[t][L] for L in LAYERS])
                       for t in ("bridge", "answer") for k, d in (("probe", res), ("raw", raw))})
for tgt in ("bridge", "answer"):
    tl = [hyper[(tgt, L)][0] for L in LAYERS]; lm = [hyper[(tgt, L)][1] for L in LAYERS]
    print(f"{tgt:7s}: hedef katman {dict((x, tl.count(x)) for x in sorted(set(tl)))}"
          f" | lambda {dict((x, lm.count(x)) for x in sorted(set(lm)))}")
if all(hyper[(t, L)][1] == max(LAMBDAS) for t in ("bridge", "answer") for L in LAYERS):
    print("!! Lambda hep en buyuk -> prob asiri uyum sinirinda; n_train kucuk olabilir.")
tick("Asama B bitti")

# =====================================================================
banner("6/6  KARAR")
# KATMAN SECIMI VAL'DA (kazananin laneti). Test yalnizca raporlama icin kullanilir.
Lb = LAYERS[int(np.argmin([valmed["bridge"][L] for L in LAYERS]))]
La = LAYERS[int(np.argmin([valmed["answer"][L] for L in LAYERS]))]
mb = np.array([np.median(res["bridge"][L]) for L in LAYERS])
ma = np.array([np.median(res["answer"][L]) for L in LAYERS])
cb, ca = bci(res["bridge"][Lb], 11), bci(res["answer"][La], 12)
nb_ci   = bci(null["bridge"][Lb], 13)
med_b   = float(np.median(res["bridge"][Lb]))
med_a   = float(np.median(res["answer"][La]))
med_nb  = float(np.median(null["bridge"][Lb]))
rb = float(np.median(raw["bridge"][Lb]))
win = (La - Lb) / max(1, N_LAYERS)

print(f"model {MODEL_ID} | aday varlik {len(ent_list)} | sans 0.500 (asagi = iyi)")
print(f"test sorusu: kopru {SPL['bridge'][2].sum()}  cevap {SPL['answer'][2].sum()}")
print("katman VAL'da secildi; asagidaki tum sayilar TEST setinde.")
print("-" * 72)
print(f"  KOPRU (e2)  katman {Lb:3d}  medyan {med_b:.3f}  GA [{cb[0]:.3f}, {cb[1]:.3f}]")
print(f"  permutasyon nulu      medyan {med_nb:.3f}  GA [{nb_ci[0]:.3f}, {nb_ci[1]:.3f}]")
print(f"  CEVAP (e3)  katman {La:3d}  medyan {med_a:.3f}  GA [{ca[0]:.3f}, {ca[1]:.3f}]")
print(f"  ham kontrol (projeksiyonsuz, katman {Lb}): {rb:.3f}")
print(f"  PENCERE: L{Lb} -> L{La} = {La-Lb} katman (%{100*win:.1f})   [esik: %{100*MIN_WINDOW_FRAC:.0f}]")
print("-" * 72)

ci_ok      = np.isfinite(cb[1]) and cb[1] < BRIDGE_CI_MAX
beats_null = np.isfinite(cb[1]) and np.isfinite(nb_ci[0]) and cb[1] < nb_ci[0]
R = ("ADRES YOK" if not ci_ok else
     "SADECE TIP SINYALI" if not beats_null else
     "KISAYOL" if not (Lb < La) else
     "PENCERE DAR" if win < MIN_WINDOW_FRAC else "GEC")
bridge_found = ci_ok and beats_null
print("SONUC:", R, "\n")

if R == "ADRES YOK":
    print("Ezberleyemeyen, tam denetimli bir prob bile kopruyu cikaramiyor.")
    print("Kopru kimligi latent durumda DOGRUSAL olarak kullanilabilir bicimde yok.")
    print("EYLEM: kapsam degisir - bu bir lookup-tablosu projesi degil. Dogrusal olmayan")
    print("       bir baslik bulabilir; bu yuzden iptal degil, kapsam degisikligi.")
elif R == "SADECE TIP SINYALI":
    print(f"Prob sansin ustunde ({med_b:.3f}) ama permutasyon nulundan ({med_nb:.3f})")
    print("ayirt edilemiyor. Yani ogrendigi sey KIMLIK degil, TIP/kategori:")
    print("'kopru-benzeri bir sey' tahmin ediyor, 'BU kopruyu' degil.")
    print("EYLEM: adres olarak kullanilamaz. Aday kumesini tip-esli hale getirip")
    print("       (ayni kategoriden adaylar) tekrarla; sinyal kaliyorsa kimlik vardir.")
elif R == "KISAYOL":
    print(f"Kopru L{Lb}, cevap L{La} - kopru cevaptan ONCE cozulmuyor.")
    print("Model muhtemelen e1->e3 kisayolunu kullaniyor; ayrik bir ara adim yok.")
    print("EYLEM: kisayol kullanmayan orneklerle (nadir cevap / sik kopru) tekrarla.")
elif R == "PENCERE DAR":
    print(f"Kopru once cozuluyor ama arada yalnizca {La-Lb} katman var (%{100*win:.1f}).")
    print("Adres mevcut, ama onu kullanip bellekten bir sey getirecek hesap zamani yok.")
    print("EYLEM: daha derin / looped bir backbone'da tekrarla (pencere derinlikle acilabilir).")
    print("NOT: bu esik gurultu olceginde - argmin'ler +-3-5 katman oynayabilir.")
else:
    print(f"Kopru L{Lb}'de cozuluyor, cevap L{La}'de - arada {La-Lb} katman (%{100*win:.1f}).")
    print("Hem adres var hem de kullanilabilecek genislikte bir pencere var.")
    print("EYLEM: Deney 2 - projeksiyonu donguye bagla, merdiven testi (R_max=1..4).")

if rb > 0.45 and bridge_found:
    print(f"\nNOT: ham kontrol {rb:.3f} (sansta) iken egitilmis prob {mb.min():.3f}.")
    print("     Projeksiyonsuz bir tasarim burada YANLIS NEGATIF verirdi.")

json.dump(dict(model=MODEL_ID, stage_a=STAGE_A, result=R, A=A, hint_fmt=fmt,
               gain=float(gain), mis=float(mis), reads_hint=bool(READS_HINT),
               n_kept=len(kept), n_ent=len(ent_list),
               L_bridge=int(Lb), L_answer=int(La), window_frac=float(win),
               layer_selected_on="val",
               med_bridge=med_b, ci_bridge=list(cb),
               med_null=med_nb, ci_null=list(nb_ci), beats_null=bool(beats_null),
               med_answer=med_a, ci_answer=list(ca), raw_bridge=rb),
          open(cpath("verdict.json"), "w"), indent=1)

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(LAYERS, mb, "-o", color="tab:red", lw=2, ms=4, label="KOPRU (e2) — egitilmis prob")
    ax.plot(LAYERS, ma, "-o", color="tab:blue", lw=2, ms=4, label="CEVAP (e3) — egitilmis prob")
    ax.plot(LAYERS, [np.median(null["bridge"][L]) for L in LAYERS], "-",
            color="tab:gray", lw=1.5, label="permutasyon nulu (kopru)")
    ax.plot(LAYERS, [np.median(raw["bridge"][L]) for L in LAYERS], "--",
            color="tab:gray", lw=1.2, alpha=0.7, label="kopru, HAM (projeksiyonsuz)")
    ax.axhline(0.5, ls="--", color="k", lw=1, label="sans")
    ax.axvline(Lb, color="tab:red", ls=":", lw=1.5); ax.axvline(La, color="tab:blue", ls=":", lw=1.5)
    if La > Lb:
        ax.axvspan(Lb, La, color="tab:green", alpha=0.10)
        ax.text((Lb+La)/2, 0.06, f"pencere %{100*win:.0f}", ha="center", color="tab:green")
    ax.set_ylim(0, 1); ax.invert_yaxis(); ax.grid(alpha=0.3)
    ax.set_xlabel("katman"); ax.set_ylabel("medyan normalize sira (asagi = iyi)")
    ax.set_title(f"{MODEL_ID} — test seti; test varliklari egitimde GORULMEDI")
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout(); plt.savefig(cpath("kopru.png"), dpi=150)
    print("\ngrafik ->", cpath("kopru.png"))
    try:
        from IPython.display import Image, display
        display(Image(filename=cpath("kopru.png")))
    except Exception:
        pass
except Exception as e:
    print("grafik cizilemedi:", str(e)[:100])

print("\nkarar ozeti ->", cpath("verdict.json"))
tick("bitti")
