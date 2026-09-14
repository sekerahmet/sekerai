# -*- coding: utf-8 -*-
"""
TESHIS: filtreden neden sadece 4 soru gecti?

Iki hipotezi ayirir:
  (1) Model bu olgulari BILMIYOR        -> daha buyuk model / farkli veri gerekir
  (2) PROMPT FORMATI tutmuyor           -> prompt duzeltilir, model yeterli

Ayni soruyu iki formatta sorar ve ham ciktiyi gosterir:
  A) ham few-shot tamamlama   (kopru.py'nin kullandigi)
  B) sohbet sablonu           (Qwen3'un dogal formati, dusunme kapali)

Colab:
    %run teshis.py
"""
import json, re, string, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

MODEL_ID = "Qwen/Qwen3-1.7B"
N_SHOW   = 20          # kac soru gosterilsin
N_SCORE  = 200         # kac soru uzerinden oran hesaplansin

# ---------------------------------------------------------------- veri
ds = load_dataset("voidful/2WikiMultihopQA", split="validation")

def to_triples(ex):
    ev = ex.get("evidences"); out = []
    if isinstance(ev, list):
        for e in ev:
            if isinstance(e, (list, tuple)) and len(e) == 3:
                out.append(tuple(str(x).strip() for x in e))
    return out

def find_chain(t):
    for i, (s1, r1, o1) in enumerate(t):
        for j, (s2, r2, o2) in enumerate(t):
            if i != j and o1 and s2 and o1.lower() == s2.lower() and o2 and o2.lower() != s1.lower():
                return (s1, r1, o1, r2, o2)
    return None

items = []
for ex in ds:
    ch = find_chain(to_triples(ex))
    if ch and all(ch):
        e1, r1, br, r2, e3 = ch
        items.append(dict(e1=e1, r1=r1, bridge=br, r2=r2, e3=e3))
    if len(items) >= N_SCORE: break
print(f"teshis icin {len(items)} soru\n")

# ---------------------------------------------------------------- model
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float16, trust_remote_code=True).to("cuda").eval()

def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s).translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())

def match(pred, gold):
    p, g = norm(pred), norm(gold)
    if len(g) < 3: return p == g
    return p == g or (" " + g + " ") in (" " + p + " ")

FEW = ("Answer each question with a short factual answer.\n\n"
       "Q: What is the director of Inception?\nA: Christopher Nolan\n\n"
       "Q: What is the capital of France?\nA: Paris\n\n")

def p_raw(q):
    return FEW + "Q: " + q + "\nA:"

def p_chat(q):
    msgs = [{"role": "user",
             "content": q + "\nAnswer with only the name. No explanation."}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def run(prompts, mx, batch=16):
    out = []
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i+batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to("cuda")
        g = model.generate(**enc, max_new_tokens=mx, do_sample=False,
                           pad_token_id=tok.pad_token_id)
        for row in g[:, enc["input_ids"].shape[1]:]:
            out.append(tok.decode(row, skip_special_tokens=True).strip())
    return out

qs1 = [f"What is the {it['r1']} of {it['e1']}?" for it in items]
qs2 = [f"What is the {it['r2']} of {it['bridge']}?" for it in items]
g1  = [it["bridge"] for it in items]
g2  = [it["e3"] for it in items]

print("uretiliyor (4 gecis, ~2-4 dk)...\n")
A1 = run([p_raw(q)  for q in qs1], 12)
B1 = run([p_chat(q) for q in qs1], 48)
A2 = run([p_raw(q)  for q in qs2], 12)
B2 = run([p_chat(q) for q in qs2], 48)

def first_line(s): return s.split("\n")[0].strip()

# ---------------------------------------------------------------- ham ciktilar
print("=" * 96)
print("HAM CIKTILAR — hop-1  (soru -> beklenen | A=few-shot | B=sohbet sablonu)")
print("=" * 96)
for i in range(min(N_SHOW, len(items))):
    ok_a = "OK" if match(first_line(A1[i]), g1[i]) else "  "
    ok_b = "OK" if match(first_line(B1[i]), g1[i]) else "  "
    print(f"\n{i+1}. {qs1[i][:80]}")
    print(f"   beklenen : {g1[i]}")
    print(f"   A {ok_a}     : {first_line(A1[i])[:90]!r}")
    print(f"   B {ok_b}     : {first_line(B1[i])[:90]!r}")

# ---------------------------------------------------------------- oranlar
def rate(preds, golds): return sum(match(first_line(p), g) for p, g in zip(preds, golds)) / len(golds)
def empty(preds): return sum(1 for p in preds if not first_line(p)) / len(preds)

a1, b1, a2, b2 = rate(A1, g1), rate(B1, g1), rate(A2, g2), rate(B2, g2)
print("\n" + "=" * 96)
print(f"ORANLAR  ({len(items)} soru)")
print("=" * 96)
print(f"  hop-1   A (few-shot) : %{100*a1:5.1f}     B (sohbet) : %{100*b1:5.1f}")
print(f"  hop-2   A (few-shot) : %{100*a2:5.1f}     B (sohbet) : %{100*b2:5.1f}")
print(f"  ikisi birden (tahmini): A %{100*a1*a2:.2f}   B %{100*b1*b2:.2f}")
print(f"  bos cikti orani: A %{100*empty(A1):.1f}   B %{100*empty(B1):.1f}")

print("\n" + "=" * 96)
best_a, best_b = (a1 + a2) / 2, (b1 + b2) / 2
if best_b > best_a * 1.8 and best_b > 0.10:
    print("TESHIS: FORMAT SORUNU.  Sohbet sablonu belirgin sekilde daha iyi.")
    print("  -> kopru.py'deki prompt() sohbet sablonuna cevrilecek. Model yeterli.")
elif max(best_a, best_b) < 0.10:
    print("TESHIS: BILGI SORUNU.  Iki format da cok dusuk.")
    print("  -> Model bu olgulari bilmiyor. Daha buyuk model (Qwen3-4B) ve/veya")
    print("     tum 12576 soru gerekir. Format degistirmek kurtarmaz.")
else:
    print("TESHIS: KARISIK.  Format bir miktar yardim ediyor ama tek basina yetmiyor.")
    print("  -> Hem sohbet sablonuna gec HEM daha buyuk model dene.")
print("=" * 96)

json.dump(dict(model=MODEL_ID, n=len(items),
               hop1_raw=a1, hop1_chat=b1, hop2_raw=a2, hop2_chat=b2,
               joint_raw=a1*a2, joint_chat=b1*b2,
               samples=[dict(q=qs1[i], gold=g1[i], raw=first_line(A1[i]), chat=first_line(B1[i]))
                        for i in range(min(30, len(items)))]),
          open("/content/teshis.json", "w"), indent=1, ensure_ascii=False)
print("\nkaydedildi -> /content/teshis.json")
