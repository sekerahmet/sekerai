# -*- coding: utf-8 -*-
"""diagnose_19 -- model_19 paketinin TEK TOKEN TESHISI.  Okuma araci; modelin hesabina girmez.

Acgozlu uretir, her secimi decompose_19 ile parcalara ayirir (her hikayede verify: parcalarin toplami modelle
tutmazsa durur), kopya kosularini ve dar secimleri bulur, ozetler, gezgin sayfasi yazar.
Paket basina hesap <kosu>/decompose/<paket>[_eval|_math].json'a yazilir ve bir daha YAPILMAZ: istemler, adim
sayisi ve hesabi belirleyen kod ayniysa kayit okunur.

    python diagnose_19.py <paket.pt> [<paket.pt> ...] [--out KLASOR] [--steps 120] [--focus "Thank you , Spike"] [--force]
    python diagnose_19.py <paket.pt> [...] --read [--eval]      okuma: yalniz acgozlu devam, decompose yok
    python diagnose_19.py <paket.pt> [...] --prompts FILE       istemler dosyadan (satir basina bir istem)

Istem seti sozluge gore: hikaye sozlugunde EXAMPLES + sonda (--eval: sonda olmayan ilk READ degerlendirme istemi);
matematik sozlugunde MATH_EXAMPLES.  Cikti (--out; yoksa ilk paketin <kosu>/decompose/): <ad>.txt rapor,
<ad>_ozet.txt ozet, <ad>.html gezgin (agac + decompose).
"""
import argparse
import hashlib
import html
import json
import os
import re
import sys
import time

import numpy as np
import torch

import data_stories_19 as DS
import decompose_19 as DC

HERE = os.path.dirname(os.path.abspath(__file__))
TS_DIR = "G:/Drive'ım/tinystories"      # sonda istemleri buradaki "Evaluation prompts.yaml"dan
STEPS = 120
# train'den 5 hikayenin ilk iki cumlesi (model_18'den; kullanici: "eğitim de gördüğü başlangıç olabilir").
EXAMPLES = (
    "Once upon a time, there was a little fish named Fin. Fin loved to swim all day in the big pond.",
    "One day, a little dog named Spike went for a walk. He saw a weak bird on the ground.",
    "Once, there was a young zebra who wanted to explore the world. He had never been anywhere, but he wanted "
    "to change that.",
    "Molly was so excited for her first field trip. She was going with her classmates to the popular farm, and she "
    "could hardly wait to see all the animals she had read about.",
    "Tom and Sam are friends. They like to play with a ball in the park.",
)
# Toplamada elde durumlari: eldesiz, tek elde, zincirleme elde, uzunluk artisi, uc terim.  Teshis icin (egitimde
# gecmis olabilir; olcum degil).
MATH_EXAMPLES = (
    "1 2 3 + 4 5 6 =",
    "4 7 2 + 1 8 2 =",
    "9 9 9 + 1 =",
    "5 0 5 + 4 9 5 =",
    "8 7 + 6 5 =",
    "2 5 + 3 4 + 1 2 =",
    "9 0 9 + 9 9 + 2 =",
    "1 + 1 =",
)
READ = 10        # okuma istemi sayisi (kullanici, 25 Eylul: "farklı tipler de 10 farklı hikaye")
NARROW = 0.5     # dar secim: ikinci adaya log p farki bundan kucuk
COPY = 4         # kopya: onceki >= COPY token'lik bir parcanin tekrari
TIGHT = 30       # sik dongu: kopyanin kaynagi en cok TIGHT token geride
FORMAT = 1       # kaydin bicimi; compute/detail degisirse artirilir
FOLDER = "decompose"
NOT_NAMES = {"The", "He", "She", "They", "It", "One", "When", "But", "Then", "So", "Once", "Suddenly", "After", "At",
             "In", "His", "Her", "Their", "Mom", "Dad", "I", "Yes", "No", "What", "Let's", "Thank", "Wow", "Look", "Can",
             "We", "You", "This", "That", "There", "From", "Every", "Just", "As", "On", "Finally", "Soon", "Today",
             "Now", "Oh", "Hello", "Hi", "Please", "Don't", "Why", "How", "Where"}


def is_bpe(vocab):
    """BPE sozlugu (data_tr_19): kelime basi token'lari ▁ ile baslar."""
    return any(t.startswith("▁") for t in vocab[:200] if t)


def is_math(vocab):
    """Matematik sozlugu: '=' var, bilinmeyen token yok, BPE degil."""
    return "=" in vocab and DS.UNK_TOKEN not in vocab and not is_bpe(vocab)


def detok(ids, vocab):
    """Token kimlikleri -> okunur metin.  BPE'de ▁ bosluktur; <eos> hikaye ayiracina doner."""
    if not is_bpe(vocab):
        return DS.decode(np.array(ids), vocab)
    text = "".join("\n---\n" if vocab[i] == DS.EOS_TOKEN else "" if vocab[i] == DS.PAD_TOKEN else vocab[i]
                   for i in ids)
    return text.replace("▁", " ").strip()


def cache_key(prompts, steps):
    """Kaydin anahtari: bicim, hesabi belirleyen kodun izi (satir sonundan bagimsiz), adim sayisi, istemler."""
    h = hashlib.sha1()
    for f in ("decompose_19.py", "model_19.py", "data_stories_19.py"):
        with open(os.path.join(HERE, f), "rb") as fh:
            h.update(fh.read().replace(b"\r\n", b"\n"))
    return json.loads(json.dumps({"format": FORMAT, "code": h.hexdigest()[:12], "steps": steps, "prompts": prompts}))


def source(tokens, i):
    """copy_length'in kaynagi: en uzun eslesmenin onceki bitis konumu (yoksa None)."""
    best, where = 0, None
    for e in range(i):
        n = 0
        while n <= e and tokens[e - n] == tokens[i - n]:
            n += 1
        if n > best:
            best, where = n, e
    return where


def copy_runs(rows, n_prompt):
    """Ardisik kopya >= COPY satirlari -> [[giris, bitis, kaynak farki]]."""
    out, cur = [], None
    for r in rows:
        if r["copy"] >= COPY:
            if cur is None:
                cur = [max(n_prompt, r["t"] + 2 - r["copy"]), r["t"] + 1, r["period"]]
            cur[1] = r["t"] + 1
        elif cur is not None:
            out.append(cur)
            cur = None
    return out + ([cur] if cur is not None else [])


def detail(m, R, tokens, vocab, t, a, b):
    """Bir secimin ayrintisi (JSON'a hazir): zincirin, R_PC'nin, baslarin, hareketlerin ve defterin en buyuk paylari."""
    ex = DC.explain(m, R, tokens, t, a, b)
    chain = ex["chain"].sum(1)
    word = lambda i: vocab[int(tokens[i])]
    d = {"chain": [[i, word(i), t - i, round(float(chain[i]), 3), round(float(ex["chain"][i, 0]), 3),
                    round(float(ex["chain"][i, 1]), 3)] for i in chain.abs().argsort(descending=True)[:5].tolist()],
         "heads": [], "vectors": [],
         "mean_parts": {k: round(v, 3) for k, v in DC.explain(m, R, tokens, t, a)["parts"].items()}}
    for k, heads in enumerate(ex["heads"]):
        A = R["attns"][k]["A"]
        for h in range(heads.shape[0]):
            hc = heads[h]
            d["heads"].append([round(float(hc.sum()), 3), [[j, word(j), round(float(A[h, t, j]), 3), round(float(hc[j]), 3)]
                                                            for j in hc.abs().argsort(descending=True)[:3].tolist()],
                               "attention %d · baş %d" % (k + 1, h)])
    for i, v in enumerate(ex["vectors"]):
        d["vectors"].append([round(float(v[:, 2].sum()), 3), [
            [int(v[r, 0]), round(float(v[r, 1]), 3), round(float(v[r, 2]), 3), DC.pushes(m, i, int(v[r, 0]), vocab)]
            for r in v[:, 2].abs().argsort(descending=True)[:4].tolist()]])
    if "readout_words" in ex:
        rw = ex["readout_words"]
        d["readout"] = [[i, word(i), t - i, round(float(rw[i]), 3)] for i in rw.abs().argsort(descending=True)[:5].tolist()]
    if "gate" in R:
        w = R["cache_w"][t]
        d["cache"] = [[j, vocab[int(R["cache_next"][j])], round(float(w[j]), 3)]
                      for j in w.argsort(descending=True)[:8].tolist() if w[j] > 0.01]
    same = [i for i in range(t + 1) if int(tokens[i]) == a]              # secilen kelimenin gectigi konumlar
    d["chain_same"] = round(float(chain[same].sum()), 3) if same else 0.0
    d["attn_same"] = round(float(sum(hs[:, same].sum() for hs in ex["heads"])), 3) if same and ex["heads"] else 0.0
    if "readout_words" in ex:
        d["readout_same"] = round(float(ex["readout_words"][same].sum()), 3) if same else 0.0
    if m.embed_delta is not None:
        d["embed_shift"] = round(float(m.embed_shift(torch.tensor([a], device=m.P.device)).norm()), 3)
    return d


@torch.no_grad()
def compute(m, vocab, name, prompts, steps=STEPS, log=print):
    """-> hikayeler: istem basina acgozlu uretim, her secimin satiri ve ayrintisi (JSON'a hazir)."""
    ix = {a: i for i, a in enumerate(vocab)}
    banned = DC.banned_ids(ix, m.P.device)
    stories = []
    for label, prompt in prompts:
        t0 = time.time()
        tokens, n_prompt = DC.greedy(m, prompt, vocab, ix, steps)
        R, rows = DC.trajectory(m, tokens, n_prompt, banned)
        tl = tokens.tolist()
        for r in rows:
            e = source(tl, r["t"] + 1)
            r["period"] = None if e is None else r["t"] + 1 - e
            r.update(detail(m, R, tokens, vocab, r["t"], r["a"], r["b"]))
            r.update(a=vocab[r["a"]], b=vocab[r["b"]], top=[[vocab[i], round(p, 4)] for i, p in r["top"]],
                     parts={k: round(v, 4) for k, v in r["parts"].items()})
            for k in ("p", "p_b", "margin", "entropy", "model_margin", "gate", "gate_dir", "gate_sim", "p_cache",
                      "cache_shift"):
                if k in r:
                    r[k] = round(r[k], 4)
        stories.append({"model": name, "label": label, "prompt": prompt, "tokens": [vocab[x] for x in tl],
                        "n_prompt": n_prompt, "text": detok(tl[n_prompt:], vocab), "rows": rows,
                        "runs": copy_runs(rows, n_prompt)})
        log("  %s | %s: %d secim, %.0f sn" % (name, label, len(rows), time.time() - t0))
    return stories


def prompt_set(probes=()):
    """Hikaye istem seti [[etiket, istem]]: EXAMPLES + sonda istemleri."""
    return ([["ornek %d" % (i + 1), p] for i, p in enumerate(EXAMPLES)]
            + [["sonda %d" % (i + 1), p] for i, p in enumerate(probes)])


def reading_prompts(all_prompts, token_index, n=READ):
    """Okuma istemleri: degerlendirme istemlerinden kelimelerinin hepsi sozlukte olan ve sonda olmayan ilk n'i,
    dosya sirasiyla -- secim uretilen metne bakmadan."""
    probes = set(DS.probe_prompts(all_prompts, token_index))
    clean = [s for s in all_prompts if s not in probes
             and all(t in token_index for t in DS.tokenize(DS.normalize(s)))]
    return [["eval %d" % (i + 1), p] for i, p in enumerate(clean[:n])]


def package_prompts(ts_dir, vocab, use_eval=False, prompts_file=None):
    """-> (istem seti, kaydin ek adi).  Dosya > matematik > okuma > standart."""
    ix = {a: i for i, a in enumerate(vocab)}
    if prompts_file:
        with open(prompts_file, encoding="utf-8") as f:
            lines = [s.strip() for s in f if s.strip()]
        tag = os.path.splitext(os.path.basename(prompts_file))[0]
        return [["istem %d" % (i + 1), p] for i, p in enumerate(lines)], "_" + tag
    if is_math(vocab):
        return [["toplama %d" % (i + 1), p] for i, p in enumerate(MATH_EXAMPLES)], "_math"
    all_prompts = DS.prompts(ts_dir)
    if use_eval:
        return reading_prompts(all_prompts, ix), "_eval"
    return prompt_set(DS.probe_prompts(all_prompts, ix)), ""


def load(path):
    """Paket -> (model CPU'da, sozluk, ad "<kosu> <paket>", paket)."""
    from model_19 import PointRelation
    k = torch.load(path, weights_only=False, map_location="cpu")
    name = "%s %s" % (os.path.basename(os.path.dirname(os.path.abspath(path))), os.path.splitext(os.path.basename(path))[0])
    return PointRelation.from_package(k), list(k["vocab"]), name, k


def save_record(path, key, step, stories):
    """Kayit: {key, step, stories}; yarim yazilmis dosya kalmaz."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump({"key": key, "step": step, "stories": stories}, f, ensure_ascii=False)
    os.replace(path + ".tmp", path)


def diagnose_package(path, steps=STEPS, ts_dir=TS_DIR, force=False, log=print, use_eval=False, prompts_file=None):
    """Paketin hikayeleri: <kosu>/decompose/<paket><ek>.json varsa ve anahtari tutuyorsa OKUNUR, yoksa hesaplanir."""
    m, vocab, name, k = load(path)
    prompts, tag = package_prompts(ts_dir, vocab, use_eval, prompts_file)
    key = cache_key(prompts, steps)
    stem = os.path.splitext(os.path.basename(path))[0]
    jf = os.path.join(os.path.dirname(os.path.abspath(path)), FOLDER, stem + tag + ".json")
    if not force and os.path.exists(jf):
        with open(jf, encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("key") == key:
            log("kayitli decompose okundu: %s" % jf)
            return saved["stories"], tag
        log("kayitli decompose eski (kod, istem ya da adim degismis), yeniden hesaplaniyor: %s" % jf)
    log("%s: adim %s, %d istem, %d adim, CPU" % (name, k.get("step"), len(prompts), steps))
    stories = compute(m, vocab, name, prompts, steps, log)
    save_record(jf, key, k.get("step"), stories)
    return stories, tag


@torch.no_grad()
def read(m, vocab, name, prompts, steps=STEPS, log=print):
    """Okuma: istem basina acgozlu devam, decompose YOK.  compute ile ayni token'lar (ikisi de DC.greedy)."""
    ix = {a: i for i, a in enumerate(vocab)}
    out = []
    for label, prompt in prompts:
        tokens, n_prompt = DC.greedy(m, prompt, vocab, ix, steps)
        tl = tokens.tolist()
        out.append({"model": name, "label": label, "prompt": prompt, "tokens": [vocab[x] for x in tl], "n_prompt": n_prompt,
                    "text": detok(tl[n_prompt:], vocab),
                    "eos": len(tl) > n_prompt and tl[-1] == ix[DS.EOS_TOKEN]})
    log("  %s: %d istem okundu" % (name, len(prompts)))
    return out


def reading_lines(stories):
    """Okuma dosyasi: istem istem, paketlerin devamlari alt alta."""
    L = []
    for label in dict.fromkeys(s["label"] for s in stories):
        S = [s for s in stories if s["label"] == label]
        L += ["=" * 100, "%s | ISTEM: %s" % (label, S[0]["prompt"]), ""]
        for s in S:
            L += ["-- %s  [%d token%s]" % (s["model"], len(s["tokens"]) - s["n_prompt"], ", <eos> ile bitti" if s["eos"] else ""),
                  s["text"].strip(), ""]
    return L


def focus(s, patterns=()):
    """Acilacak secimler [[konum t, neden]]: verilen dizilerin ilk gecisi, ilk 3 kopya girisi, ilk kopyadan onceki
    en dar 3 secim."""
    tl, n = s["tokens"], s["n_prompt"]
    out = {}
    for pattern in patterns:
        k = len(pattern)
        hit = next((i for i in range(max(n, k - 1), len(tl)) if tuple(tl[i - k + 1:i + 1]) == tuple(pattern)), None)
        if hit is not None:
            out.setdefault(hit - 1, []).append("HEDEF: " + " ".join(pattern))
    for a, _, p in s["runs"][:3]:
        out.setdefault(a - 1, []).append("KOPYA GIRISI (kaynak %d geride)" % p)
    first = s["runs"][0][0] if s["runs"] else len(tl)
    for r in sorted((r for r in s["rows"] if r["t"] + 1 < first), key=lambda r: r["margin"])[:3]:
        out.setdefault(r["t"], []).append("DAR SECIM (fark %.2f), ilk kopyadan once" % r["margin"])
    return [[t, " + ".join(why)] for t, why in out.items()]


def dissection(r):
    """Kayitli ayrintidan bir secimin decompose'u, okunur satirlar."""
    L = ["   ilk 5: " + "  ".join("%s %.3f" % (w, p) for w, p in r["top"]),
         "   PARCALAR (%s - %s, logit): toplam %+.2f = %s" % (r["a"], r["b"], r["model_margin"], "  ".join(
             "%s %+.2f" % (k, v) for k, v in r["parts"].items()))]
    for i, w, age, v, o, c in r["chain"][:4]:
        L.append("    zincir   konum %3d %-12s yas %3d  %+.2f  (sira %+.2f  icerik %+.2f)" % (i, w, age, v, o, c))
    for i, w, age, v in r.get("readout", [])[:4]:
        L.append("    R_PC     konum %3d %-12s yas %3d  %+.2f" % (i, w, age, v))
    for tot, best, lab in r["heads"]:
        L.append("    %-22s %+.2f   %s" % (lab, tot, "   ".join("konum %d %s (A %.2f) %+.2f" % tuple(x) for x in best)))
    for i, (tot, best) in enumerate(r["vectors"]):
        L.append("    katman %d %+.2f   %s" % (i, tot, "   ".join(
            "v%d (w %.2f) %+.2f [%s]" % (vid, w, v, " ".join(pu)) for vid, w, v, pu in best[:3])))
    if "gate" in r:
        L.append("    defter  gate %.3f (yon %+.2f, benzerlik %+.2f)  p_defter(%s) %.3f  farka etkisi %+.2f   %s" % (
            r["gate"], r["gate_dir"], r["gate_sim"], r["a"], r["p_cache"], r["cache_shift"], "   ".join(
                "konum %d -> %s %.2f" % tuple(x) for x in r["cache"] if x[2] > 0.05)))
    L.append("    secilen kelimenin gectigi konumlardan: zincir %+.2f  attention %+.2f%s" % (
        r["chain_same"], r["attn_same"], "  R_PC %+.2f" % r["readout_same"] if "readout_same" in r else ""))
    return L


def report(stories):
    """Hikaye hikaye: uretilen metin, kopya kosulari, secim tablosu, odak secimlerin decompose'u."""
    L = []
    for s in stories:
        L += ["", "=" * 110, "%s | %s | istem: %s" % (s["model"], s["label"], s["prompt"]),
              "URETILEN: " + s["text"].replace("\n", " / "),
              "kopya kosulari (>= %d token): %s" % (COPY, "; ".join(
                  'konum %d-%d (kaynak %d geride): "%s"' % (a, e, p, " ".join(s["tokens"][a:min(e + 1, a + 10)]))
                  for a, e, p in s["runs"]) or "yok")]
        if not s["rows"]:
            continue
        L.append("%5s %-11s %6s %6s %-11s %5s %6s %6s | %s" % (
            "konum", "secilen", "p", "fark", "ikinci", "kopya", "gate", "defter", " ".join(
                "%7s" % k.replace("chain_", "z_").replace("layer_", "k").replace("_head_", "b") for k in s["rows"][0]["parts"])))
        for r in s["rows"]:
            L.append("%5d %-11s %6.3f %6.2f %-11s %5d %6.3f %+6.2f | %s" % (
                r["t"] + 1, r["a"][:11], r["p"], r["margin"], r["b"][:11], r["copy"], r.get("gate", 0),
                r.get("cache_shift", 0), " ".join("%+7.2f" % v for v in r["parts"].values())))
        by_t = {r["t"]: r for r in s["rows"]}
        for t, why in s.get("focus", []):
            r = by_t[t]
            L += ["", "-- %s | konum %d: ...%s  ->  %s (p %.3f)   ikinci %s (p %.3f)   fark %.2f   kopya %d" % (
                why, t + 1, " ".join(s["tokens"][max(0, t - 10):t + 1]), r["a"], r["p"], r["b"], r["p_b"], r["margin"],
                r["copy"])] + dissection(r)
    return L


def loop_measures(S):
    """Hikayeler -> dongu olculeri (uretilen kisim): anlik tekrar, sik dongu payi (kopya >= COPY, kaynak <= TIGHT),
    uzak kalip payi (kopya >= 8, kaynak > TIGHT), dongu60 (ilk 60'ta tekrar eden 8'li), dar secim payi."""
    rows = [r for s in S for r in s["rows"]]
    n = max(len(rows), 1)
    rep = sum(1 for s in S for r in s["rows"] if r["a"] == s["tokens"][r["t"]] and r["a"][:1].isalpha())
    tight = sum(1 for r in rows if r["copy"] >= COPY and r["period"] and r["period"] <= TIGHT)
    far = sum(1 for r in rows if r["copy"] >= 8 and r["period"] and r["period"] > TIGHT)
    loops = 0
    for s in S:
        g = [w for w in s["tokens"][s["n_prompt"]:s["n_prompt"] + 60] if w != "<nl>"]
        g8 = [tuple(g[i:i + 8]) for i in range(len(g) - 7)]
        loops += len(set(g8)) < len(g8)
    return {"rep": rep, "tight": tight / n, "far": far / n, "loops60": loops, "stories": len(S),
            "narrow": sum(r["margin"] < NARROW for r in rows) / n}


def headline(S):
    """Egitim gunlugu icin tek satir."""
    x = loop_measures(S)
    return "dar %%%.0f  sik dongu %%%.1f  uzak kalip %%%.1f  anlik tekrar %d  dongu60 %d/%d" % (
        100 * x["narrow"], 100 * x["tight"], 100 * x["far"], x["rep"], x["loops60"], x["stories"])


GROUPS = (("zincir", lambda k: k.startswith("chain_")), ("katman 0", lambda k: k == "layer_0"),
          ("attention", lambda k: "_head_" in k), ("katman 1-3", lambda k: k.startswith("layer_") and k != "layer_0"),
          ("R_PC", lambda k: k == "readout"))


def part_groups(r):
    """Satirin parcalari gruplara toplanir (secilen - ikinci, log p); defter (cache_shift) ayri grup."""
    g = {name: sum(v for k, v in r["parts"].items() if f(k)) for name, f in GROUPS if any(f(k) for k in r["parts"])}
    g["defter"] = r.get("cache_shift", 0.0)
    return g


def loop_anatomy(S):
    """Dongu anatomisi (betimleyici): kopya icinde / disinda gruplarin ortalama payi ve kopya kaniti (chain_same,
    attn_same); her hikayede ilk kosunun GIRISI -- tekrari baslatan secim -- ve onu en cok iten uc grup."""
    rows_in = [r for s in S for r in s["rows"] if r["copy"] >= COPY]
    rows_out = [r for s in S for r in s["rows"] if r["copy"] < COPY]
    if not rows_in or not rows_out:
        return ["  dongu anatomisi: kopya icinde %d, disinda %d secim -- karsilastirma yok" % (len(rows_in), len(rows_out))]
    names = list(part_groups(rows_in[0]))
    mean = lambda rows, k: np.mean([part_groups(r)[k] for r in rows])
    n_gen = sum(len(s["rows"]) for s in S)
    L = ["  dongu anatomisi (secilen - ikinci, log p; kopya >= %d; uretimin %%%.0f'i kopya icinde):"
         % (COPY, 100 * len(rows_in) / max(n_gen, 1)),
         "    %-13s%s   chain_same  attn_same      p" % ("", "".join("%11s" % k for k in names))]
    for lab, rows in (("kopya icinde", rows_in), ("disinda", rows_out)):
        L.append("    %-13s%s   %+9.2f  %+9.2f   %.2f   n %d" % (
            lab, "".join("%+11.2f" % mean(rows, k) for k in names), np.mean([r["chain_same"] for r in rows]),
            np.mean([r["attn_same"] for r in rows]), np.mean([r["p"] for r in rows]), len(rows)))
    for s in S:
        if not s["runs"]:
            continue
        a, e, period = s["runs"][0]
        r = next((x for x in s["rows"] if x["t"] == a - 1), None)
        if r is None:
            continue
        top = sorted(part_groups(r).items(), key=lambda kv: -abs(kv[1]))[:3]
        L.append("    %-9s giris uretimin %3d. token'i  donem %-4s kosu %3d token   '%s' > '%s' fark %.2f   itenler %s" % (
            s["label"], a - s["n_prompt"] + 1, period, e - a, r["a"], r["b"], r["margin"],
            "  ".join("%s %+.2f" % kv for kv in top)))
    return L


def repeat_profile(seqs, n=4, kmax=4):
    """Tekrar profili.  seqs: [(token'lar, baslangic)] -- yalniz baslangic ve sonrasindaki token'lar sayilir.  Son n
    token'lik parca daha once k kez gectiyse (k = kmax: kmax ve fazlasi), sonraki token son gecisin devamini tekrarladi
    mi.  -> {k: (tekrarladi, firsat)}.  Altin metin ve uretilen metin ayni olcuyle: veri k buyudukce tekrari azaltiyor
    mu, model artiriyor mu."""
    out = {k: [0, 0] for k in range(1, kmax + 1)}
    for s, start in seqs:
        seen = {}
        for t in range(n - 1, len(s) - 1):
            g = tuple(s[t - n + 1:t + 1])
            prev = seen.setdefault(g, [])
            if prev and t + 1 >= start:
                k = min(len(prev), kmax)
                out[k][1] += 1
                out[k][0] += s[t + 1] == s[prev[-1] + 1]
            prev.append(t)
    return {k: tuple(v) for k, v in out.items()}


def is_name(w):
    return bool(re.fullmatch(r"[A-Z][a-z]+", w)) and w not in NOT_NAMES


def summary(stories):
    """Betimleyici ozet (hipotez yok), model basina."""
    L = []
    for model in dict.fromkeys(s["model"] for s in stories):
        S = [s for s in stories if s["model"] == model]
        rows = [r for s in S for r in s["rows"]]
        if not rows:
            continue
        m = np.array([r["margin"] for r in rows])
        keys = list(rows[0]["parts"])
        v = [np.array([r["parts"][k] for k in keys]) for r in rows]
        ratio = np.array([np.abs(x).sum() / max(abs(x.sum()), 1e-9) for x in v])
        top = [max(r["parts"], key=lambda k: abs(r["parts"][k])) for r in rows]
        narrow = m < NARROW
        L += ["=" * 100, "%s: %d hikaye, %d secim" % (model, len(S), len(rows)), "  dongu: " + headline(S),
              "  fark (log p, ikinciye karsi): < 0,25 %%%.0f   < %.1f %%%.0f   < 1 %%%.0f   medyan %.2f" % (
                  100 * (m < 0.25).mean(), NARROW, 100 * narrow.mean(), 100 * (m < 1).mean(), np.median(m)),
              "  karsit kuvvet orani sum|parca| / |fark|: medyan %.1f   dar secimlerde %.1f" % (
                  np.median(ratio), np.median(ratio[narrow]) if narrow.any() else float("nan")),
              "  en buyuk parca kim: " + "  ".join("%s %%%.0f" % (k, 100 * top.count(k) / len(top)) for k in keys if top.count(k)),
              "  mutlak pay ortalamasi: " + "  ".join("%s %.2f" % (k, np.mean([abs(r["parts"][k]) for r in rows])) for k in keys)
              + "   defter %.2f" % np.mean([abs(r.get("cache_shift", 0)) for r in rows])]
        heads = [k for k in keys if "_head_" in k]
        layers = [k for k in keys if k.startswith("layer_")]
        for lab, sel in (("ad secildi", [r for r in rows if is_name(r["a"])]),
                         ("ad ikinci kaldi", [r for r in rows if is_name(r["b"]) and not is_name(r["a"])])):
            if sel:
                L.append("  %-15s n %4d  baslar %s   katmanlar %s" % (lab, len(sel), " ".join(
                    "%+.2f" % np.mean([r["parts"][h] for r in sel]) for h in heads), " ".join(
                    "%+.2f" % np.mean([r["parts"][k] for r in sel]) for k in layers)))
        cp, nc = [r for r in rows if r["copy"] >= COPY], [r for r in rows if r["copy"] < COPY]
        if cp and nc and "gate" in rows[0]:
            L.append("  defter gate: kopya icinde medyan %.3f (> 0,3: %%%.0f)   disinda medyan %.3f (> 0,3: %%%.0f)" % (
                np.median([r["gate"] for r in cp]), 100 * np.mean([r["gate"] > 0.3 for r in cp]),
                np.median([r["gate"] for r in nc]), 100 * np.mean([r["gate"] > 0.3 for r in nc])))
            L.append("  gate'in girdileri, kopya icinde / disinda: yon %+.2f / %+.2f   benzerlik %+.2f / %+.2f" % (
                np.mean([r["gate_dir"] for r in cp]), np.mean([r["gate_dir"] for r in nc]),
                np.mean([r["gate_sim"] for r in cp]), np.mean([r["gate_sim"] for r in nc])))
        L += loop_anatomy(S)
        for s in S:
            L.append("  %-10s secim %3d  dar %2d  kopya icinde %3d  kosu %d" % (
                s["label"], len(s["rows"]), sum(r["margin"] < NARROW for r in s["rows"]),
                sum(r["copy"] >= COPY for r in s["rows"]), len(s["runs"])))
    return L


def page(stories, title):
    """Gezgin: diagnose_viewer_19.html + gomulu veri; tek dosya, tarayicida acilir."""
    keep = lambda r: {k: v for k, v in r.items() if k not in ("model_margin", "entropy", "argmax")}
    data = [dict(s, rows=[keep(r) for r in s["rows"]]) for s in stories]
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    with open(os.path.join(HERE, "diagnose_viewer_19.html"), encoding="utf-8") as f:
        return f.read().replace("__TITLE__", html.escape(title)).replace("__DATA__", blob)


def write_outputs(base, stories, title, patterns=()):
    """<base>.txt (rapor), <base>_ozet.txt, <base>.html (gezgin).  Odaklar burada hesaplanir (kayda girmez)."""
    for s in stories:
        s["focus"] = focus(s, patterns)
    for suffix, lines in ((".txt", report(stories)), ("_ozet.txt", summary(stories))):
        with open(base + suffix, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    with open(base + ".html", "w", encoding="utf-8") as f:
        f.write(page(stories, title))


def main():
    ap = argparse.ArgumentParser(description="model_19 paketinin tek token teshisi (decompose).")
    ap.add_argument("packages", nargs="+", help="t<N>.pt ya da w<N>.pt paketleri")
    ap.add_argument("--out", help="rapor, ozet ve sayfanin klasoru (yoksa ilk paketin <kosu>/decompose/)")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--focus", action="append", default=[], help='acilacak dizi, token\'lar bosluklu: "Thank you , Spike"')
    ap.add_argument("--force", action="store_true", help="kayit olsa da yeniden hesapla")
    ap.add_argument("--ts-dir", default=TS_DIR)
    ap.add_argument("--eval", action="store_true", help="hikaye: sonda olmayan ilk %d degerlendirme istemi" % READ)
    ap.add_argument("--prompts", help="istem dosyasi, satir basina bir istem (matematikte rakamlar boslukla)")
    ap.add_argument("--read", action="store_true", help="yalniz acgozlu devam, decompose yok: okuma_<ad>.txt ve .json")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.packages[0])), FOLDER)
    os.makedirs(out, exist_ok=True)
    names = "_".join(os.path.splitext(os.path.basename(p))[0] for p in a.packages)
    if a.read:
        stories, tag = [], ""
        for p in a.packages:
            m, vocab, name, _ = load(p)
            prompts, tag = package_prompts(a.ts_dir, vocab, a.eval, a.prompts)
            stories += read(m, vocab, name, prompts, a.steps)
        base = os.path.join(out, "okuma" + tag + "_" + names)
        with open(base + ".txt", "w", encoding="utf-8") as f:
            f.write("\n".join(reading_lines(stories)) + "\n")
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump(stories, f, ensure_ascii=False)
        print("yazildi: %s.txt  %s.json" % (base, base))
        return
    stories, tag = [], ""
    for p in a.packages:
        s, tag = diagnose_package(p, a.steps, a.ts_dir, a.force, use_eval=a.eval, prompts_file=a.prompts)
        stories += s
    base = os.path.join(out, names + tag)
    title = "%s Decompose" % " · ".join(dict.fromkeys(os.path.basename(os.path.dirname(os.path.abspath(p)))
                                                      for p in a.packages))
    write_outputs(base, stories, title, [tuple(f.split()) for f in a.focus])
    print("\n".join(summary(stories)))
    print("\nyazildi: %s.txt  %s_ozet.txt  %s.html" % (base, base, base))


if __name__ == "__main__":
    main()
