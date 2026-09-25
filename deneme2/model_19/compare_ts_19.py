"""compare_ts_19 -- model_19 ile TinyStories-8M'in kiyasi (onkayit: belge/onkayit/model_19.md, KIYAS).

    python compare_ts_19.py uret  [--paket KOSU/t40000] [--hikaye 500]    bpc + uretim + kor okuma dosyasi
    python compare_ts_19.py ac                                               puanlar + anahtar -> ozet

Uc olcu:
  bpc      ayni tutulan hikayelerde karakter basina bit (sozlukler farkli: token accuracy'si kiyaslanmaz)
  uretim   ayni istemler, acgozlu ve ornekleme; ayni dongu olculeri (kelime duzeyinde, TOKEN_RE)
  kor      her istem ve kipte iki cikti A/B rastgele sirada; anahtar ayri dosyada, puanlamadan sonra acilir

Model CPU'da; model_19 onbelleksiz uretir (her token butun diziyi yeniden hesaplar).
"""
import glob
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data_stories_19 as DS  # noqa: E402
import diagnose_19 as DG  # noqa: E402
from model_19 import PointRelation  # noqa: E402

ROOT = "G:/Drive'ım/model_19"
TS_DIR = "G:/Drive'ım/tinystories"
OUT = ROOT + "/COMPARE_TS8M"
HF = os.path.expanduser("~/.cache/huggingface/hub")
REF_NAME = "TinyStories-8M"
WORDS = 120                 # devam bu kadar kelimede kesilir (iki model esit)
TS_NEW_TOKENS = 200         # 8M'in BPE token siniri; kesme yine WORDS kelimede
SAMPLING = dict(temperature=0.8, top_p=0.9)
OWN_PROMPTS = [
    "Anna had a red kite and a blue ball. She gave the kite to her little brother Ben, and",
    "The cat was very hungry, so it",
    "Max and his sister Mia went to the beach. Mia wanted to build a big sand castle, but Max",
    "A little bird fell out of its nest. A kind girl saw the bird and",
    "Sam lost his favorite hat. He looked under the bed, but",
    'Tom said to his mom, "Can I have a cookie?" His mom said,',
    "It was a cold and rainy day. Lucy could not go outside, so she",
    "There was an old box in the attic. When Jack opened the box, he found",
    "The frog and the rabbit had a race. The rabbit was fast, but the frog",
    "One morning, Sue woke up and saw that her dog was sick. She",
]


def load_ours(pattern):
    path = "%s/%s.pt" % (ROOT, pattern)
    k = torch.load(path, weights_only=False, map_location="cpu")
    return PointRelation.from_package(k).eval(), DS.to_general_eos(list(k["vocab"])), k


def load_reference():
    """TinyStories-8M yerel onbellekten; tokenizer (GPT-Neo) 3M'in klasorunden -- ikisi ayni tokenizer'i kullanir."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    snap = lambda m, f: next(os.path.dirname(p) for p in glob.glob("%s/models--roneneldan--%s/snapshots/*/%s" % (HF, m, f)))
    model = AutoModelForCausalLM.from_pretrained(snap(REF_NAME, "config.json"), local_files_only=True).eval()
    tok = AutoTokenizer.from_pretrained(snap("TinyStories-3M", "merges.txt"), local_files_only=True)
    return model, tok


def held_stories(vocab, n, max_len=500):
    """Tutulan akistan dosya sirasindaki ilk n hikaye: butun kelimeleri sozlukte, en cok max_len token."""
    a = np.load("%s/onbellek/akis_valid_tam_n8000_v4.npy" % TS_DIR)
    eos, unk = vocab.index(DS.EOS_TOKEN), vocab.index(DS.UNK_TOKEN)
    ends = np.flatnonzero(a == eos)
    out = []
    for b, e in zip(np.concatenate([[0], ends + 1])[:len(ends)], ends):
        s = a[b:e]
        if 0 < len(s) <= max_len and not (s == unk).any():
            out.append(s)
            if len(out) == n:
                break
    return out, eos


@torch.no_grad()
def bpc(ours, vocab, ref, tok, stories, eos):
    """-> satirlar: (karakter, bizim bit, 8M bit).  Ayni metin: bizim decode'umuzun ciktisi; sonu token'i dahil."""
    rows = []
    for s in stories:
        text = DS.decode(s, vocab)
        w = torch.tensor([[eos, *s.tolist(), eos]])
        lp = ours.scoreboard(w)[0, :-1]
        ours_bits = -float(lp.gather(-1, w[0, 1:, None]).sum()) / math.log(2)
        ids = [tok.eos_token_id] + tok(text)["input_ids"] + [tok.eos_token_id]
        x = torch.tensor([ids])
        lq = torch.log_softmax(ref(x).logits[0, :-1].float(), -1)
        ref_bits = -float(lq.gather(-1, x[0, 1:, None]).sum()) / math.log(2)
        rows.append((len(text), ours_bits, ref_bits))
    return rows


def cut(text, words=WORDS):
    """Devami ilk `words` kelimede keser; (metin, kelimeler).  Bos satirlar teke iner: 8M paragraflari bos satirla
    ayiriyor, model_19 tek satir sonuyla -- kor okumada bicim hangi modelin yazdigini ele veriyordu (sinamada goruldu)."""
    ms = list(DS.TOKEN_RE.finditer(text))
    if len(ms) > words:
        text = text[:ms[words - 1].end()]
    return re.sub(r"\n\s*\n+", "\n", text).strip(), [m.group(0) for m in ms[:words]]


def generate_ours(m, vocab, ix, prompt, sample, seed):
    kw = dict(temperature=SAMPLING["temperature"], top_p=SAMPLING["top_p"], seed=seed) if sample else {}
    _, g = DS.generate(m, prompt, vocab, ix, steps=WORDS + 30, device="cpu", penalty=1.0, **kw)   # <nl> kelime sayilmaz
    ended = "---" in g
    return g.split("---")[0].strip(), ended


@torch.no_grad()
def generate_ref(ref, tok, prompt, sample, seed):
    x = tok(prompt, return_tensors="pt")
    kw = dict(do_sample=True, **SAMPLING) if sample else dict(do_sample=False)
    if sample:
        torch.manual_seed(seed)
    y = ref.generate(**x, max_new_tokens=TS_NEW_TOKENS, eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id, **kw)
    new = y[0, x["input_ids"].shape[1]:].tolist()
    ended = tok.eos_token_id in new
    return tok.decode(new[:new.index(tok.eos_token_id)] if ended else new).strip(), ended


def loop_numbers(items):
    """items: [(istem kelimeleri, devam kelimeleri)] -> tekrar profili, dongu60, farkli-4."""
    rp = DG.repeat_profile([(p + g, len(p)) for p, g in items])
    loops = sum(1 for _, g in items if len({tuple(g[i:i + 8]) for i in range(len(g[:60]) - 7)}) < max(len(g[:60]) - 7, 0))
    d4 = [len({tuple(g[i:i + 4]) for i in range(len(g) - 3)}) / (len(g) - 3) for _, g in items if len(g) > 3]
    return {"repeat": rp, "loops60": loops, "n": len(items), "distinct4": float(np.mean(d4)) if d4 else float("nan")}


def produce(pattern, n_stories, n_prompts=None, out=None):
    global OUT
    OUT = out or OUT
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    ours, vocab, k = load_ours(pattern)
    ix = {a: i for i, a in enumerate(vocab)}
    ref, tok = load_reference()
    log = open(OUT + "/gunluk.txt", "a", encoding="utf-8")
    note = lambda s: (print(s), log.write(s + "\n"), log.flush())
    note("KIYAS  model_19 %s (adim %s, heldout %.4f)  <->  %s (%s parametre)" % (
        pattern, "{:,}".format(k["step"]), k["heldout_acc"], REF_NAME, "{:,}".format(sum(p.numel() for p in ref.parameters()))))

    stories, eos = held_stories(vocab, n_stories)
    rows = bpc(ours, vocab, ref, tok, stories, eos)
    chars = sum(r[0] for r in rows)
    b_ours, b_ref = sum(r[1] for r in rows) / chars, sum(r[2] for r in rows) / chars
    better = sum(1 for r in rows if r[1] < r[2])
    note("bpc  %d hikaye, %s karakter:  model_19 %.4f   %s %.4f   hikaye basina model_19 daha iyi %d / %d  (%.0f sn)" % (
        len(rows), "{:,}".format(chars), b_ours, REF_NAME, b_ref, better, len(rows), time.time() - t0))

    prompts = [("makale %d" % (i + 1), p) for i, p in enumerate(DS.prompts(TS_DIR))] + \
              [("kendi %d" % (i + 1), p) for i, p in enumerate(OWN_PROMPTS)]
    prompts = prompts[:n_prompts] if n_prompts else prompts
    raw, blind, key, viewer = [], [], {}, []
    for i, (label, p) in enumerate(prompts):
        pw = DS.TOKEN_RE.findall(p)
        for mode in ("acgozlu", "ornekleme"):
            sample, seed = mode == "ornekleme", 1000 + i
            out = {}
            for name, fn in (("model_19", lambda: generate_ours(ours, vocab, ix, p, sample, seed)),
                             (REF_NAME, lambda: generate_ref(ref, tok, p, sample, seed))):
                g, ended = fn()
                text, words = cut(g)
                ended = ended and len(DS.TOKEN_RE.findall(g)) <= WORDS
                out[name] = dict(text=text, words=words, ended=ended)
            raw.append(dict(label=label, prompt=p, prompt_words=pw, mode=mode, out=out))
            order = ["model_19", REF_NAME]
            random.Random(104729 * i + 2 * (mode == "ornekleme") + 1).shuffle(order)
            item = "%s|%s" % (label, mode)
            key[item] = {"A": order[0], "B": order[1]}
            blind.append("### %s  (%s)\nISTEM  %s\n%s" % (label, mode, p, "\n".join(
                "%s   %s   %s" % (ab, out[nm]["text"], "[bitti]" if out[nm]["ended"] else "[kesildi]")
                for ab, nm in zip("AB", order))))
            viewer.append(dict(id=item, run=pattern, label=label, mode=mode, prompt=p,
                               **{ab: dict(text=out[nm]["text"], ended=out[nm]["ended"]) for ab, nm in zip("AB", order)}))
        note("  %s  %d/%d istem  (%.0f sn)" % (label, i + 1, len(prompts), time.time() - t0))

    for mode in ("acgozlu", "ornekleme"):
        for name in ("model_19", REF_NAME):
            x = loop_numbers([(r["prompt_words"], r["out"][name]["words"]) for r in raw if r["mode"] == mode])
            ended = sum(r["out"][name]["ended"] for r in raw if r["mode"] == mode)
            note("dongu  %-9s %-15s dongu60 %d/%d   farkli4 %.3f   bitti %d/%d   tekrar k=1..4+ %s" % (
                mode, name, x["loops60"], x["n"], x["distinct4"], ended, x["n"], "  ".join(
                    "%.0f%%(%d)" % (100 * c / n, n) if n else "-" for c, n in x["repeat"].values())))
    json.dump(dict(pattern=pattern, bpc=rows, raw=raw), open(OUT + "/ham.json", "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(key, open(OUT + "/anahtar.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(OUT + "/kor.txt", "w", encoding="utf-8").write(
        "KOR OKUMA -- her cikti icin dilbilgisi, yaraticilik, tutarlilik (istemle), olay orgusu; 10 uzerinden.\n"
        "Puanlar puan.json'a: {\"<etiket>|<kip>\": {\"A\": [d, y, t, o], \"B\": [d, y, t, o]}}\n\n" + "\n\n".join(blind) + "\n")
    page = open(os.path.join(HERE, "compare_viewer_19.html"), encoding="utf-8").read()
    blob = json.dumps(viewer, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    open(OUT + "/kor.html", "w", encoding="utf-8").write(page.replace("__DATA__", blob))
    note("yazildi: %s/kor.txt ve kor.html (okunacak, anahtar YOK), anahtar.json (puanlamadan sonra), ham.json  (%.0f sn)"
         % (OUT, time.time() - t0))


def unblind():
    """puan.json + anahtar.json -> boyut ve kip basina ortalama, istem basina eslestirilmis karsilastirma."""
    grades = json.load(open(OUT + "/puan.json", encoding="utf-8"))
    key = json.load(open(OUT + "/anahtar.json", encoding="utf-8"))
    dims = ("dilbilgisi", "yaraticilik", "tutarlilik", "olay orgusu")
    lines = ["KOR OKUMA ACILDI (%d cikti cifti)" % len(grades)]
    for mode in ("acgozlu", "ornekleme"):
        by = {"model_19": [], REF_NAME: []}
        for item, g in grades.items():
            if item.endswith("|" + mode):
                for ab in "AB":
                    by[key[item][ab]].append(g[ab])
        pairs = [(np.array(grades[i]["A" if key[i]["A"] == "model_19" else "B"]),
                  np.array(grades[i]["B" if key[i]["A"] == "model_19" else "A"])) for i in grades if i.endswith("|" + mode)]
        lines.append("\n%s  (%d istem)" % (mode, len(pairs)))
        for name, v in by.items():
            v = np.array(v)
            lines.append("  %-15s %s   toplam %.2f" % (name, "   ".join("%s %.2f" % (d, v[:, j].mean()) for j, d in enumerate(dims)),
                                                     v.sum(1).mean()))
        for j, d in enumerate(dims + ("toplam",)):
            a = np.array([p[0][j] if j < 4 else p[0].sum() for p in pairs])
            b = np.array([p[1][j] if j < 4 else p[1].sum() for p in pairs])
            lines.append("  %-12s model_19 ustun %d   esit %d   %s ustun %d" % (d, (a > b).sum(), (a == b).sum(), REF_NAME, (a < b).sum()))
    if os.path.exists(OUT + "/oy.json"):                 # kullanicinin kor.html'den kopyaladigi oylar
        votes = json.load(open(OUT + "/oy.json", encoding="utf-8"))
        lines.append("\nKULLANICININ GOZU (kor.html)")
        for mode in ("acgozlu", "ornekleme"):
            pick = [key[i][v["choice"]] if v.get("choice") in "AB" else "esit"
                    for i, v in votes.items() if i.endswith("|" + mode) and v.get("choice")]
            mine = [i for i, v in votes.items() if i.endswith("|" + mode) and v.get("choice") and i in grades]
            same = sum(1 for i in mine if (votes[i]["choice"] == "=") == (sum(grades[i]["A"]) == sum(grades[i]["B"]))
                       and (votes[i]["choice"] == "=" or (sum(grades[i]["A"]) > sum(grades[i]["B"])) == (votes[i]["choice"] == "A")))
            lines.append("  %-9s %d oy:  model_19 %d   esit %d   %s %d   |  Claude'un toplam puaniyla ayni yon %d / %d" % (
                mode, len(pick), pick.count("model_19"), pick.count("esit"), REF_NAME, pick.count(REF_NAME), same, len(mine)))
    open(OUT + "/acilis.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["uret"]:
        opt = dict(zip(args[1::2], args[2::2]))
        produce(opt.get("--paket", "PR_TS_ARCH_BASE_S0_40K/t40000"), int(opt.get("--hikaye", 500)),
                int(opt["--istem"]) if "--istem" in opt else None, opt.get("--cikti"))
    elif args[:1] == ["ac"]:
        unblind()
    else:
        print(__doc__)
