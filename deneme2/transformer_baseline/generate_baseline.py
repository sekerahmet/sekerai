"""generate_baseline -- sabit istemlerden kapali dongu hikaye (nitel olcu, CLAUDE.md kural 12).  Istemler ss_prompts.json:
sinav hikayesi indeksleri (ss_exam_story_*, Model Z ile ayni numara) ve verilecek cumle sayilari; Model Z de ayni dosyayi
okuyabilir.  --decode (kullanici onayi, 6 Ekim; Model Z story_generation ile ayni tanim):
    greedy  argmax
    sample  modelin kendi dagilimindan, sicaklik 1, sabit tohum (0); sicaklik / top-k / top-p / tekrar cezasi YOK
EOS yalniz cumle basinda gecerli, cumle ortasinda EOS -> END (Model Z SentenceTransformer.generate kurali).
sentence_repeat: uretilen (END ile biten) cumlelerden, o hikayede (istem + once uretilenler) birebir daha once gecmis
olanlarin orani; decode basina.  Ana saglik sayisi greedy sentence_repeat.
Cikti: <out>/samples.txt (okunur metin: istem, greedy, sample, gercek devam), samples.json, results.json'a "generation".

    python generate_baseline.py --out <kosu klasoru (agent.pt)> [--root SS klasoru] [--device cpu|cuda]
                                [--decode greedy sample]
"""
import argparse
import json
import os

import torch  # noqa: I001  (Windows: torch numpy'den once)

from baseline import BaselineTransformer

HERE = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def generate_story(model, prompts, max_words=1000, decode="greedy"):
    """prompts: istem basina cumle listesi (kelime kimlikleri) -> istem basina (END ile biten cumleler, yarim kalan son
    kelimeler, EOS'a ulasti mi).  Batch'te birlikte; her satir kendi boyunda ilerler (causal: sagdaki dolgu okunmaz)."""
    dev = next(model.parameters()).device
    seqs = [[0] + [w for s in p for w in s + [model.END]] for p in prompts]
    B = len(seqs)
    lens = torch.tensor([len(s) for s in seqs], device=dev)
    tok = torch.zeros(B, int(lens.max()) + max_words, dtype=torch.long, device=dev)
    for i, s in enumerate(seqs):
        tok[i, :len(s)] = torch.tensor(s)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    ended = torch.zeros(B, dtype=torch.bool, device=dev)
    rows = torch.arange(B, device=dev)
    gen = torch.Generator(device=dev).manual_seed(0)
    for _ in range(max_words):
        h = model.hidden(tok[:, :int(lens.max())])
        logits = (h[rows, lens - 1] @ model.E.weight.T).float()
        if decode == "greedy":
            w = logits.argmax(-1)
        else:
            w = torch.multinomial(logits.softmax(-1), 1, generator=gen)[:, 0]
        start = tok[rows, lens - 1] == model.END                   # cumle basi (istem hep END ile biter)
        ended |= ~done & start & (w == model.EOS)
        w = torch.where(~start & (w == model.EOS), model.END, w)
        done |= ended
        tok[rows[~done], lens[~done]] = w[~done]
        lens = lens + (~done).long()
        if done.all():
            break
    out = []
    for i, s in enumerate(seqs):
        sents, cur = [], []
        for t in tok[i, len(s):int(lens[i])].tolist():
            if t == model.END:
                sents.append(cur)
                cur = []
            else:
                cur.append(t)
        out.append((sents, cur, bool(ended[i])))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="kosu klasoru: agent.pt okunur, samples.* yazilir")
    ap.add_argument("--root", default="G:/Drive'ım/model_z/simplestories_full")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--decode", nargs="+", default=["greedy", "sample"], choices=("greedy", "sample"))
    args = ap.parse_args(argv)
    from train_baseline import Stories, _no_power_throttling
    if os.name == "nt" and not args.device.startswith("cuda"):
        print("guc kisitlamasi (EcoQoS) kapali:", _no_power_throttling(), flush=True)   # arka planda ~10 kat yavas
    pack = torch.load(os.path.join(args.out, "agent.pt"), map_location=args.device, weights_only=False)
    a = pack["args"]
    model = BaselineTransformer(pack["vocab"], a["d"], a["layers"], a["heads"]).to(args.device)
    model.load_state_dict(pack["state"])
    model.eval()
    vocab = pack["vocab"]
    spec = json.load(open(os.path.join(HERE, "ss_prompts.json"), encoding="utf-8"))
    exam = Stories.from_files(args.root, "ss_exam_story", "cpu")
    stories = []
    for si in spec["stories"]:
        a0, a1 = int(exam.story_off[si]), int(exam.story_off[si + 1])
        stories.append([exam.flat[exam.sent_off[k]:exam.sent_off[k + 1]].tolist() for k in range(a0, a1)])
    jobs = [(si, st, k) for si, st in zip(spec["stories"], stories) for k in spec["sentences"]]
    text = lambda s: " ".join(vocab[w] for w in s) if s else "(bos cumle)"  # noqa: E731
    res = {d: generate_story(model, [st[:k] for _, st, k in jobs], decode=d) for d in args.decode}
    rows, lines = [], []
    summary = {d: dict(sentences=0, repeats=0, ended=0, loop=0) for d in args.decode}
    for j, (si, st, k) in enumerate(jobs):
        row = dict(story=si, given=k, prompt=[text(s) for s in st[:k]], real=[text(s) for s in st[k:]])
        lines += ["=== hikaye %d, ilk %d cumle verildi" % (si, k), "--- istem"] + row["prompt"]
        for d in args.decode:
            sents, tail, ended = res[d][j]
            seen = {tuple(s) for s in st[:k]}
            repeat = 0
            for s in sents:
                repeat += tuple(s) in seen
                seen.add(tuple(s))
            flat = [w for s in sents for w in s] + tail
            loop = any(flat[i:i + 3] == flat[i + 3:i + 6] for i in range(max(0, len(flat) - 5)))
            row[d] = dict(generated=[text(s) for s in sents], tail=text(tail), ended=ended, sentences=len(sents),
                          sentence_repeat=round(repeat / max(len(sents), 1), 4), loop=loop, generated_ids=sents)
            for key, v in (("sentences", len(sents)), ("repeats", repeat), ("ended", ended), ("loop", loop)):
                summary[d][key] += v
            lines += ["--- model (%s) | EOS %s, sentence_repeat %d/%d, dongu %s" % (
                d, "var" if ended else "YOK", repeat, len(sents), loop)] + [text(s) for s in sents] + \
                (["... " + text(tail)] if tail else [])
        lines += ["--- gercek devam"] + row["real"] + [""]
        rows.append(row)
    for d, s in summary.items():
        s["sentence_repeat"] = round(s["repeats"] / max(s["sentences"], 1), 4)
        s["prompts"] = len(jobs)
    open(os.path.join(args.out, "samples.txt"), "w", encoding="utf-8").write("\n".join(lines))
    json.dump(dict(summary=summary, rows=rows), open(os.path.join(args.out, "samples.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    rpath = os.path.join(args.out, "results.json")
    if os.path.exists(rpath):
        r = json.load(open(rpath, encoding="utf-8"))
        r["generation"] = summary
        json.dump(r, open(rpath, "w", encoding="utf-8"), indent=1)
    print("\n".join(lines), flush=True)
    for d, s in summary.items():
        print("uretim %s: %d istem, %d cumle, sentence_repeat %.4f, EOS'a ulasan %d, dongu %d -> %s" % (
            d, len(jobs), s["sentences"], s["sentence_repeat"], s["ended"], s["loop"],
            os.path.join(args.out, "samples.txt")), flush=True)
    return summary


if __name__ == "__main__":
    main()
