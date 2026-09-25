# -*- coding: utf-8 -*-
"""diagnose -- kayitli bir paketin TEK TOKEN TESHISI.  Okuma araci; modelin hesabina girmez.

Kullanici, 25 Eylul: "bu araç kalıcı bir diagnostic aracı oldu mu ? her seferinde baştan yapmaz gerekmez bu sayede."

Acgozlu uretir (sicaklik 0, DS.generate ile ayni), her secimi decompose ile parcalara ayirir (her hikayede verify:
parcalarin toplami modelle tutmazsa durur), kopya kosularini ve dar secimleri bulur, ozetler, gezgin sayfasi yazar.

Paket basina hesap paketin yanina yazilir, <kosu>/dokum/<paket>.json, ve bir daha YAPILMAZ: istemler, adim sayisi ve
hesabi belirleyen kod ayniysa kayit okunur.  Rapor, ozet ve sayfa kayittan uretilir; model gerekmez.

    python diagnose.py <paket.pt> [<paket.pt> ...] [--out KLASOR] [--steps 120] [--focus "Thank you , Spike"] [--force]

Cikti (--out; yoksa ilk paketin <kosu>/dokum/ klasoru), <ad> = paket adlari:
    <ad>.txt       hikaye hikaye secim tablosu + odak secimlerin dokumu
    <ad>_ozet.txt  secimlerin darligi, karsit kuvvetler, adlar, hemen tekrar, sik dongu tur tur
    <ad>.html      gezgin, tarayicida acilir.  Agac: istem tepede, her adimda ilk uc aday ve olasiliklari, secilen
                   yesil (kullanici, 25 Eylul: "ağaç gibi seçimleri gösterme").  Dokum: token'a tikla, parcalarini gor
Istemler her pakette ayni -- EXAMPLES + kosunun sabit sonda istemleri -- ki paketler yan yana okunsun.
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

import data_stories as DS
import decompose as DC

HERE = os.path.dirname(os.path.abspath(__file__))
TS_DIR = "G:/Drive'ım/tinystories"      # sonda istemleri buradaki "Evaluation prompts.yaml"dan
STEPS = 120
# train'den 5 hikayenin ilk iki cumlesi (25 Eylul ornek5: rng 7, farkli acilislar).  Kullanici: "eğitim de gördüğü
# başlangıç olabilir".
EXAMPLES = (
    "Once upon a time, there was a little fish named Fin. Fin loved to swim all day in the big pond.",
    "One day, a little dog named Spike went for a walk. He saw a weak bird on the ground.",
    "Once, there was a young zebra who wanted to explore the world. He had never been anywhere, but he wanted "
    "to change that.",
    "Molly was so excited for her first field trip. She was going with her classmates to the popular farm, and she "
    "could hardly wait to see all the animals she had read about.",
    "Tom and Sam are friends. They like to play with a ball in the park.",
)
NARROW = 0.5     # dar secim: ikinci adaya log p farki bundan kucuk
COPY = 4         # kopya: onceki >= COPY token'lik bir parcanin tekrari
TIGHT = 30       # sik dongu: kopyanin kaynagi en cok TIGHT token geride
FORMAT = 1       # kayitli dokumun bicimi; compute/detail degisirse artirilir (eski kayitlar yeniden hesaplanir)
NOT_NAMES = {"The", "He", "She", "They", "It", "One", "When", "But", "Then", "So", "Once", "Suddenly", "After", "At",
             "In", "His", "Her", "Their", "Mom", "Dad", "I", "Yes", "No", "What", "Let's", "Thank", "Wow", "Look", "Can",
             "We", "You", "This", "That", "There", "From", "Every", "Just", "As", "On", "Finally", "Soon", "Today",
             "Now", "Oh", "Hello", "Hi", "Please", "Don't", "Why", "How", "Where"}


def cache_key(prompts, steps):
    """Kayitli dokumun anahtari: bicim, hesabi belirleyen kodun izi, adim sayisi, istemler."""
    h = hashlib.sha1()
    for f in ("decompose.py", "model_18.py", "data_stories.py"):
        with open(os.path.join(HERE, f), "rb") as fh:
            h.update(fh.read())
    return json.loads(json.dumps({"format": FORMAT, "code": h.hexdigest()[:12], "steps": steps, "prompts": prompts}))


def source(tokens, i):
    """DC.copy_length'in kaynagi: en uzun eslesmenin onceki bitis konumu (yoksa None)."""
    best, where = 0, None
    for e in range(i):
        n = 0
        while n <= e and tokens[e - n] == tokens[i - n]:
            n += 1
        if n > best:
            best, where = n, e
    return where


def copy_runs(rows, n_prompt):
    """Ardisik kopya >= COPY satirlari -> [[giris, bitis, kaynak farki]]; giris = kopyalanan parcanin ILK token'i."""
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
    """Bir secimin ayrintisi (JSON'a hazir): zincirin, baslarin, vektorlerin ve defterin en buyuk paylari."""
    ex = DC.explain(m, R, tokens, t, a, b)
    chain = ex["chain"].sum(1)
    word = lambda i: vocab[int(tokens[i])]
    d = {"chain": [[i, word(i), t - i, round(float(chain[i]), 3), round(float(ex["chain"][i, 0]), 3),
                    round(float(ex["chain"][i, 1]), 3)] for i in chain.abs().argsort(descending=True)[:5].tolist()],
         "heads": [], "vectors": [],
         "mean_parts": {k: round(v, 3) for k, v in DC.explain(m, R, tokens, t, a)["parts"].items()}}
    for h in range(ex["heads"].shape[0]):
        hc = ex["heads"][h]
        d["heads"].append([round(float(hc.sum()), 3), [[j, word(j), round(float(R["A"][h, t, j]), 3), round(float(hc[j]), 3)]
                                                        for j in hc.abs().argsort(descending=True)[:3].tolist()]])
    for i, v in enumerate(ex["vectors"]):
        d["vectors"].append([round(float(v[:, 2].sum()), 3), [
            [int(v[r, 0]), round(float(v[r, 1]), 3), round(float(v[r, 2]), 3), DC.pushes(m, i, int(v[r, 0]), vocab)]
            for r in v[:, 2].abs().argsort(descending=True)[:4].tolist()]])
    if "gate" in R:
        d["cache"] = [[j, vocab[n], round(w, 3)] for j, n, w in zip(R["cache_j"][t].tolist(), R["cache_next"][t].tolist(),
                                                                   R["cache_w"][t].tolist()) if w > 0.01]
    same = [i for i in range(t + 1) if int(tokens[i]) == a]              # secilen kelimenin gectigi konumlar
    d["chain_same"] = round(float(chain[same].sum()), 3) if same else 0.0
    d["attn_same"] = round(float(ex["heads"][:, same].sum()), 3) if same else 0.0
    return d


@torch.no_grad()
def compute(m, vocab, name, prompts, steps=STEPS, probe_texts=None, log=print):
    """-> hikayeler: istem basina acgozlu uretim, her secimin satiri ve ayrintisi (JSON'a hazir).
    probe_texts: paketin kendi sonda metinleri (Colab'daki uretim); "sonda i" ilk 60 token'da onunla karsilastirilir."""
    ix = {a: i for i, a in enumerate(vocab)}
    banned = torch.tensor([ix[x] for x in DC.BANNED])
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
            for k in ("p", "p_b", "margin", "entropy", "model_margin", "gate", "p_cache", "cache_shift"):
                if k in r:
                    r[k] = round(r[k], 4)
        s = {"model": name, "label": label, "prompt": prompt, "tokens": [vocab[x] for x in tl], "n_prompt": n_prompt,
             "text": DS.decode(np.array(tl[n_prompt:]), vocab), "rows": rows, "runs": copy_runs(rows, n_prompt)}
        if probe_texts and label.startswith("sonda "):
            i = int(label.split()[1]) - 1
            if i < len(probe_texts):
                s["gpu_same"] = DS.decode(np.array(tl[n_prompt:n_prompt + 60]), vocab) == probe_texts[i]
        stories.append(s)
        log("  %s | %s: %d secim, %.0f sn" % (name, label, len(rows), time.time() - t0))
    return stories


def diagnose_package(path, steps=STEPS, ts_dir=TS_DIR, force=False, log=print):
    """Paketin hikayeleri: <kosu>/dokum/<paket>.json varsa ve anahtari tutuyorsa OKUNUR, yoksa hesaplanip yazilir."""
    from model_18 import PV
    run_dir = os.path.dirname(os.path.abspath(path))
    stem = os.path.splitext(os.path.basename(path))[0]
    name = "%s %s" % (os.path.basename(run_dir), stem)
    k = torch.load(path, weights_only=False, map_location="cpu")
    vocab = list(k["vocab"])
    ix = {a: i for i, a in enumerate(vocab)}
    prompts = ([["ornek %d" % (i + 1), p] for i, p in enumerate(EXAMPLES)]
               + [["sonda %d" % (i + 1), p] for i, p in enumerate(DS.probe_prompts(DS.prompts(ts_dir), ix))])
    key = cache_key(prompts, steps)
    jf = os.path.join(run_dir, "dokum", stem + ".json")
    if not force and os.path.exists(jf):
        with open(jf, encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("key") == key:
            log("kayitli dokum okundu: %s" % jf)
            return saved["stories"]
        log("kayitli dokum eski (kod, istem ya da adim degismis), yeniden hesaplaniyor: %s" % jf)
    m = PV.from_package(k)
    log("%s: adim %s, %d istem, %d adim, CPU" % (name, k.get("step"), len(prompts), steps))
    stories = compute(m, vocab, name, prompts, steps, (k.get("health") or {}).get("texts"), log)
    os.makedirs(os.path.dirname(jf), exist_ok=True)
    with open(jf + ".tmp", "w", encoding="utf-8") as f:
        json.dump({"key": key, "step": k.get("step"), "stories": stories}, f, ensure_ascii=False)
    os.replace(jf + ".tmp", jf)
    return stories


def focus(s, patterns=()):
    """Acilacak secimler [[konum t, neden]]: verilen dizilerin uretilen kisimdaki ilk gecisi, ilk 3 kopya girisi,
    ilk kopyadan onceki en dar 3 secim.  Ayni konum bir kez, nedenleri birlikte."""
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
    """Kayitli ayrintidan bir secimin dokumu, okunur satirlar."""
    L = ["   ilk 5: " + "  ".join("%s %.3f" % (w, p) for w, p in r["top"]),
         "   PARCALAR (%s - %s, logit): toplam %+.2f = %s" % (r["a"], r["b"], r["model_margin"], "  ".join(
             "%s %+.2f" % (k, v) for k, v in r["parts"].items()))]
    for i, w, age, v, o, c in r["chain"][:4]:
        L.append("    zincir   konum %3d %-12s yas %3d  %+.2f  (sira %+.2f  icerik %+.2f)" % (i, w, age, v, o, c))
    for h, (tot, best) in enumerate(r["heads"]):
        L.append("    bas %d  %+.2f   %s" % (h, tot, "   ".join("konum %d %s (A %.2f) %+.2f" % tuple(x) for x in best)))
    for i, (tot, best) in enumerate(r["vectors"]):
        L.append("    katman %d %+.2f   %s" % (i, tot, "   ".join(
            "v%d (w %.2f) %+.2f [%s]" % (vid, w, v, " ".join(pu)) for vid, w, v, pu in best[:3])))
    if "gate" in r:
        L.append("    defter  gate %.3f  p_defter(%s) %.3f  farka etkisi %+.2f   %s" % (
            r["gate"], r["a"], r["p_cache"], r["cache_shift"], "   ".join(
                "konum %d -> %s %.2f" % tuple(x) for x in r["cache"] if x[2] > 0.05)))
    L.append("    secilen kelimenin gectigi konumlardan: zincir %+.2f  attention %+.2f" % (r["chain_same"], r["attn_same"]))
    return L


def report(stories):
    """Hikaye hikaye: uretilen metin, kopya kosulari, secim tablosu, odak secimlerin dokumu."""
    L = []
    for s in stories:
        L += ["", "=" * 110, "%s | %s | istem: %s" % (s["model"], s["label"], s["prompt"]),
              "URETILEN: " + s["text"].replace("\n", " / ")]
        if "gpu_same" in s:
            L.append("Colab'daki kayitli metinle (ilk 60 token): " + ("AYNI" if s["gpu_same"] else "FARKLI"))
        L.append("kopya kosulari (>= %d token): %s" % (COPY, "; ".join(
            'konum %d-%d (kaynak %d geride): "%s"' % (a, e, p, " ".join(s["tokens"][a:min(e + 1, a + 10)]))
            for a, e, p in s["runs"]) or "yok"))
        if not s["rows"]:
            continue
        L.append("%5s %-11s %6s %6s %-11s %5s %6s %6s | %s" % (
            "konum", "secilen", "p", "fark", "ikinci", "kopya", "gate", "defter", " ".join(
                "%6s" % k.replace("chain_", "z_").replace("layer_", "k").replace("head_", "b") for k in s["rows"][0]["parts"])))
        for r in s["rows"]:
            L.append("%5d %-11s %6.3f %6.2f %-11s %5d %6.3f %+6.2f | %s" % (
                r["t"] + 1, r["a"][:11], r["p"], r["margin"], r["b"][:11], r["copy"], r.get("gate", 0),
                r.get("cache_shift", 0), " ".join("%+6.2f" % v for v in r["parts"].values())))
        by_t = {r["t"]: r for r in s["rows"]}
        for t, why in s.get("focus", []):
            r = by_t[t]
            L += ["", "-- %s | konum %d: ...%s  ->  %s (p %.3f)   ikinci %s (p %.3f)   fark %.2f   kopya %d" % (
                why, t + 1, " ".join(s["tokens"][max(0, t - 10):t + 1]), r["a"], r["p"], r["b"], r["p_b"], r["margin"],
                r["copy"])] + dissection(r)
    return L


def is_name(w):
    return bool(re.fullmatch(r"[A-Z][a-z]+", w)) and w not in NOT_NAMES


def summary(stories):
    """Betimleyici ozet (hipotez yok), model basina."""
    L = []
    heads = lambda r: [k for k in r["parts"] if k.startswith("head_")]
    layers = lambda r: [k for k in r["parts"] if k.startswith("layer_")]
    for model in dict.fromkeys(s["model"] for s in stories):
        S = [s for s in stories if s["model"] == model]
        rows = [r for s in S for r in s["rows"]]
        if not rows:
            continue
        m = np.array([r["margin"] for r in rows])
        v = [np.array(list(r["parts"].values())) for r in rows]
        ratio = np.array([np.abs(x).sum() / max(abs(x.sum()), 1e-9) for x in v])
        big = np.array([np.abs(x).max() for x in v])
        keys = list(rows[0]["parts"])
        top = [max(r["parts"], key=lambda k: abs(r["parts"][k])) for r in rows]
        L += ["=" * 100, "%s: %d hikaye, %d secim" % (model, len(S), len(rows)),
              "  fark (log p, ikinciye karsi): < 0,25 %%%.0f   < %.1f %%%.0f   < 1 %%%.0f   medyan %.2f" % (
                  100 * (m < 0.25).mean(), NARROW, 100 * (m < NARROW).mean(), 100 * (m < 1).mean(), np.median(m)),
              "  karsit kuvvet orani sum|parca| / |fark|: medyan %.1f   dar secimlerde %.1f" % (
                  np.median(ratio), np.median(ratio[m < NARROW]) if (m < NARROW).any() else float("nan")),
              "  en buyuk tek parca: medyan %.2f   dar secimlerde %.2f (fark medyani %.2f)" % (
                  np.median(big), np.median(big[m < NARROW]) if (m < NARROW).any() else float("nan"),
                  np.median(m[m < NARROW]) if (m < NARROW).any() else float("nan")),
              "  en buyuk parca kim: " + "  ".join("%s %%%.0f" % (k, 100 * top.count(k) / len(top)) for k in keys if top.count(k)),
              "  mutlak pay ortalamasi: " + "  ".join("%s %.2f" % (k, np.mean([abs(r["parts"][k]) for r in rows])) for k in keys)
              + "   defter %.2f" % np.mean([abs(r.get("cache_shift", 0)) for r in rows])]
        for lab, sel in (("ad secildi", [r for r in rows if is_name(r["a"])]),
                         ("ad ikinci kaldi", [r for r in rows if is_name(r["b"]) and not is_name(r["a"])])):
            if sel:
                L.append("  %-15s n %4d  baslar %s   katmanlar %s" % (lab, len(sel), " ".join(
                    "%+.2f" % np.mean([r["parts"][h] for r in sel]) for h in heads(sel[0])), " ".join(
                    "%+.2f" % np.mean([r["parts"][k] for r in sel]) for k in layers(sel[0]))))
        rep = [(s, r) for s in S for r in s["rows"] if r["a"] == s["tokens"][r["t"]] and s["tokens"][r["t"]][:1].isalpha()]
        avoid = [(s, r) for s in S for r in s["rows"] if r["b"] == s["tokens"][r["t"]] and s["tokens"][r["t"]][:1].isalpha()]
        for lab, sel in (("hemen tekrar secildi", rep), ("hemen tekrar ikinci kaldi", avoid)):
            if sel:
                L.append("  %-26s n %3d  zincir_sira %+.2f  zincir_icerik %+.2f  katmanlar %+.2f  fark %.2f" % (
                    lab, len(sel), np.mean([r["parts"]["chain_order"] for _, r in sel]),
                    np.mean([r["parts"]["chain_content"] for _, r in sel]),
                    np.mean([sum(r["parts"][k] for k in layers(r)) for _, r in sel]), np.mean([r["margin"] for _, r in sel])))
        cp, nc = [r for r in rows if r["copy"] >= COPY], [r for r in rows if r["copy"] < COPY]
        if cp and nc and "gate" in rows[0]:
            L.append("  defter gate: kopya icinde medyan %.3f (> 0,3: %%%.0f)   disinda medyan %.3f (> 0,3: %%%.0f)" % (
                np.median([r["gate"] for r in cp]), 100 * np.mean([r["gate"] > 0.3 for r in cp]),
                np.median([r["gate"] for r in nc]), 100 * np.mean([r["gate"] > 0.3 for r in nc])))
        tight = [r for r in cp if r["period"] and r["period"] <= TIGHT]
        if tight:
            L.append("  sik dongu (kaynak <= %d geride), tur basina:  n | fark | p | gate | defter | zincir | attention | katmanlar"
                     % TIGHT)
            for k in range(1, 7):
                sel = [r for r in tight if (r["copy"] - 1) // r["period"] + 1 == k]
                if sel:
                    f = lambda g: np.mean([g(r) for r in sel])
                    L.append("    tur %d  %3d | %.2f | %.3f | %.3f | %+.2f | %+.2f | %+.2f | %+.2f" % (
                        k, len(sel), f(lambda r: r["margin"]), f(lambda r: r["p"]), f(lambda r: r.get("gate", 0)),
                        f(lambda r: r.get("cache_shift", 0)), f(lambda r: r["parts"]["chain_order"] + r["parts"]["chain_content"]),
                        f(lambda r: sum(r["parts"][h] for h in heads(r))), f(lambda r: sum(r["parts"][k] for k in layers(r)))))
        for s in S:
            entry = {r["t"] + 1: r["margin"] for r in s["rows"]}
            L.append("  %-8s secim %3d  dar %2d  kopya icinde %3d  kosu %d  giris farklari: %s" % (
                s["label"], len(s["rows"]), sum(r["margin"] < NARROW for r in s["rows"]),
                sum(r["copy"] >= COPY for r in s["rows"]), len(s["runs"]),
                " ".join("%.2f" % entry[a] for a, _, _ in s["runs"] if a in entry)))
    return L


def page(stories, title):
    """Gezgin: diagnose_viewer.html + gomulu veri; tek dosya, tarayicida acilir."""
    keep = lambda r: {k: v for k, v in r.items() if k not in ("model_margin", "entropy", "argmax")}
    data = [dict(s, rows=[keep(r) for r in s["rows"]]) for s in stories]
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    with open(os.path.join(HERE, "diagnose_viewer.html"), encoding="utf-8") as f:
        return f.read().replace("__TITLE__", html.escape(title)).replace("__DATA__", blob)


def main():
    ap = argparse.ArgumentParser(description="Kayitli paketin tek token teshisi (decompose).")
    ap.add_argument("packages", nargs="+", help="t<N>.pt ya da w<N>.pt paketleri")
    ap.add_argument("--out", help="rapor, ozet ve sayfanin klasoru (yoksa ilk paketin <kosu>/dokum/)")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--focus", action="append", default=[], help='acilacak dizi, token\'lar bosluklu: "Thank you , Spike"')
    ap.add_argument("--force", action="store_true", help="kayit olsa da yeniden hesapla")
    ap.add_argument("--ts-dir", default=TS_DIR)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    stories = []
    for p in a.packages:
        stories += diagnose_package(p, a.steps, a.ts_dir, a.force)
    patterns = [tuple(f.split()) for f in a.focus]
    for s in stories:
        s["focus"] = focus(s, patterns)
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.packages[0])), "dokum")
    os.makedirs(out, exist_ok=True)
    base = os.path.join(out, "_".join(os.path.splitext(os.path.basename(p))[0] for p in a.packages))
    title = "%s Token Dökümü" % " · ".join(dict.fromkeys(os.path.basename(os.path.dirname(os.path.abspath(p)))
                                                         for p in a.packages))
    for suffix, lines in ((".txt", report(stories)), ("_ozet.txt", summary(stories))):
        with open(base + suffix, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    with open(base + ".html", "w", encoding="utf-8") as f:
        f.write(page(stories, title))
    print("\n".join(summary(stories)))
    print("\nyazildi: %s.txt  %s_ozet.txt  %s.html" % (base, base, base))


if __name__ == "__main__":
    main()
