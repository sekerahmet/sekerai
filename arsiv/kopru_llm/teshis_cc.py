# -*- coding: utf-8 -*-
"""
TESHIS-2: Compositional Celebrities, Qwen3-1.7B ile kullanilabilir mi?

2WikiMultihopQA'da filtre 4000 -> 4 verdi (model obskür olgulari bilmiyordu).
CC farkli: kisiler unlu, ikinci hop kapali kume (ulke -> baskent/para/alan adi...).

Bu script HICBIR SEY YENIDEN YAZMADAN sunu olcer:
  - hop-1 (kisi -> dogum ulkesi)  : 468 benzersiz kisi
  - hop-2 (ulke -> X)             : her kategori icin 118 benzersiz ulke
  - her kategori icin BEKLENEN ortak verim = hop1 x hop2

Cikti: hangi kategorilerle devam edilecegini soyler.

Colab:
    %run teshis_cc.py
"""
import json, io, re, string, urllib.request, collections
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-1.7B"
CC_URL   = "https://raw.githubusercontent.com/ofirpress/self-ask/main/datasets/compositional_celebrities.json"
CC_PATH  = "/content/compositional_celebrities.json"
BATCH    = 32
MIN_YIELD = 0.25          # bu orani gecen kategoriler onerilir

# ---------------------------------------------------------------- veri
try:
    data = json.load(io.open(CC_PATH, encoding="utf-8"))["data"]
    print("onbellekten:", CC_PATH)
except Exception:
    print("indiriliyor:", CC_URL)
    urllib.request.urlretrieve(CC_URL, CC_PATH)
    data = json.load(io.open(CC_PATH, encoding="utf-8"))["data"]
print(f"toplam {len(data)} soru, {len(set(x['category'] for x in data))} kategori")

# yalnizca kopru = ULKE olan kategoriler (birthyear/birthdate sayisal, elenir)
CATS = sorted({x["category"] for x in data if x["category"].startswith("birthplace_")})
sub  = [x for x in data if x["category"] in CATS]
print(f"birthplace_* : {len(sub)} soru, {len(CATS)} kategori, "
      f"{len({x['A1'][0] for x in sub})} ulke, {len({x['person'] for x in sub})} kisi\n")

# ---------------------------------------------------------------- model
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float16, trust_remote_code=True).to("cuda").eval()

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
    for i in range(0, len(prompts), BATCH):
        enc = tok(prompts[i:i+BATCH], return_tensors="pt", padding=True,
                  truncation=True, max_length=320).to("cuda")
        g = model.generate(**enc, max_new_tokens=mx, do_sample=False,
                           pad_token_id=tok.pad_token_id)
        for row in g[:, enc["input_ids"].shape[1]:]:
            out.append(tok.decode(row, skip_special_tokens=True).split("\n")[0].strip())
        print(f"\r  {desc}: {min(i+BATCH, len(prompts))}/{len(prompts)}", end="", flush=True)
    print()
    return out

# ---------------------------------------------------------------- hop-1: kisi -> ulke
p1 = {}
for x in sub: p1.setdefault(x["person"], (x["Q1"], [str(a) for a in x["A1"]]))
persons = sorted(p1)
outs = gen([prompt(p1[p][0]) for p in persons], "hop-1 (kisi -> ulke)")
hop1_ok = {p: match_any(o, p1[p][1]) for p, o in zip(persons, outs)}
h1 = sum(hop1_ok.values()) / len(persons)
print(f"\nHOP-1  kisi -> dogum ulkesi : %{100*h1:.1f}   ({sum(hop1_ok.values())}/{len(persons)} kisi)\n")

print("  ornekler:")
for p in persons[:6]:
    o = outs[persons.index(p)]
    print(f"    {'OK' if hop1_ok[p] else '  '} {p:28s} -> {o[:30]!r}  (beklenen {p1[p][1][0]})")

# ---------------------------------------------------------------- hop-2: ulke -> X (kategori basina)
print("\n" + "=" * 92)
print(f"{'kategori':30s} {'hop-2':>7s} {'hop-1':>7s} {'ORTAK':>7s} {'~soru':>7s}   ornek")
print("=" * 92)
rows = []
for cat in CATS:
    cs = [x for x in sub if x["category"] == cat]
    q2 = {}
    for x in cs: q2.setdefault(str(x["A1"][0]), (x["Q2"], [str(a) for a in x["A2"]]))
    ks = sorted(q2)
    o2 = gen([prompt(q2[k][0]) for k in ks], f"hop-2 {cat[:24]}")
    ok2 = {k: match_any(o, q2[k][1]) for k, o in zip(ks, o2)}
    h2 = sum(ok2.values()) / len(ks)
    joint = sum(1 for x in cs if hop1_ok.get(x["person"]) and ok2.get(str(x["A1"][0]))) / len(cs)
    rows.append((cat, h2, joint, int(joint * len(cs)), o2[0], q2[ks[0]][1][0]))

print("\n" + "=" * 92)
print(f"{'kategori':30s} {'hop-2':>7s} {'hop-1':>7s} {'ORTAK':>7s} {'~soru':>7s}   ornek cikti")
print("=" * 92)
for cat, h2, joint, n, ex, gold in sorted(rows, key=lambda r: -r[2]):
    star = " <=" if joint >= MIN_YIELD else ""
    print(f"{cat:30s} %{100*h2:5.1f} %{100*h1:5.1f} %{100*joint:5.1f} {n:7d}   "
          f"{ex[:18]!r} (bkl {gold[:14]!r}){star}")
print("=" * 92)

good = [(c, j, n) for c, _, j, n, _, _ in rows if j >= MIN_YIELD]
tot = sum(n for _, _, n in good)
print(f"\nONERI: {len(good)} kategori >= %{100*MIN_YIELD:.0f} verim  ->  toplam ~{tot} kullanilabilir soru")
if good:
    print("  " + ", ".join(c for c, _, _ in good))
    print(f"\n  Benzersiz kisi (hop-1 bilinen): {sum(hop1_ok.values())}")
    print(f"  Benzersiz ulke (kopru adayi)  : {len({str(x['A1'][0]) for x in sub})}")
    print("\nSONUC: CC kullanilabilir. kopru.py'yi bu kategorilere gore uyarlayacagim.")
else:
    print("\nSONUC: CC de yetmiyor. Daha buyuk model (Qwen3-4B) gerekir.")

json.dump(dict(model=MODEL_ID, hop1=h1, n_person=len(persons),
               n_person_ok=sum(hop1_ok.values()),
               cats={c: dict(hop2=h2, joint=j, n=n) for c, h2, j, n, _, _ in rows}),
          open("/content/teshis_cc.json", "w"), indent=1)
print("\nkaydedildi -> /content/teshis_cc.json")
