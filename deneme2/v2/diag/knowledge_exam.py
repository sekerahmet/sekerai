"""knowledge_exam -- FineWeb-Edu bilgi sinavi (belge 47 s5, 48; kullanici, 7 Ekim: "istem bilgi istesin ve bilgi dogru
gelsin"; "kısa olması değil tutarlı gelmesi önemli").

    count  ham parquet metninde (kaynak SALT OKUNUR), yalniz TRAIN belgelerinde (gpt2/train_doc_ids.npy), kucuk harf:
           olgu basina (a) konu ve anahtar ayni belgede, (b) ayni +-WINDOW karakter penceresinde belge sayisi; bant
           (>= 100 / 10-100 / 1-10 / 0, (a)'ya gore); istemin birebir (buyuk / kucuk harf duyarli) gectigi belge sayisi;
           arama tablosu tabani: istemin train'de gecen en uzun son eki (kelime, geriye cekilerek) ve ardindan en sik 5
           kelime.  Row group'lar surec havuzunda.  -> <out>/facts.json (kopya), counts.json, lookup.json.
    run    kayitli kosudan (generate_readings.load_run) acgozlu uretim: istem tek acik cumle (generate open_last=True),
           max_sentences 5 / max_tokens 64 (cevap) ve 80 (durma: EOS orani, cumle / token sayisi); puan: cevap penceresi
           (istemi tamamlayan cumle + sonraki cumle) icinde anahtar (yazim cesitleri) varsa ve ondan once celdirici yoksa
           DOGRU, once / yalniz celdirici YANLIS, ikisi de yoksa BOS.  Bant bant ve arama tablosu tabaniyla ayni puanlama.
           -> <kosu>/knowledge_exam.json, knowledge_exam.txt.
Otomatik puanin sinirlari: olumsuzlama, anahtarin baska baglamda gecmesi, sayi bicimleri, eksik celdirici listesi; bir
ornek human-reader ile kor okunur.

    python knowledge_exam.py count --src <fineweb koku> --data <v2/fineweb_edu_s000> --out <.../knowledge> [--workers N]
    python knowledge_exam.py run --run <kosu> --data <v2/fineweb_edu_s000> --knowledge <.../knowledge> [--device cuda]
"""
import torch  # noqa: I001  (Windows: torch once)

import argparse
import collections
import json
import os
import re
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
sys.path.insert(0, HERE)

FACTS = os.path.join(HERE, "facts.json")
WINDOW = 200                # (b) konu ile anahtar arasi en cok karakter
BANDS = ((100, ">= 100"), (10, "10-100"), (1, "1-10"), (0, "0"))
SNIPPETS = 200              # son ek basina toplanan devam ornegi (surec basina)
LOOKUP_WORDS = 5
ANSWER = dict(max_sentences=5, max_tokens=64)
STOP = dict(max_sentences=80, max_tokens=64)


def band(n):
    return next(name for lo, name in BANDS if n >= lo)


def _pattern(words):
    """Yazim cesitleri -> kucuk harf, kelime sinirli regex."""
    return re.compile("|".join(r"(?<!\w)%s(?!\w)" % re.escape(w.lower()) for w in words))


def score(text, fact):
    """-> (DOGRU / YANLIS / BOS, anahtar konumu, celdirici konumu)."""
    low = text.lower()
    k = _pattern(fact["key"]).search(low)
    d = _pattern(fact["distractors"]).search(low) if fact["distractors"] else None
    kp, dp = (k.start() if k else None), (d.start() if d else None)
    if kp is not None and (dp is None or kp < dp):
        return "DOGRU", kp, dp
    if dp is not None:
        return "YANLIS", kp, dp
    return "BOS", kp, dp


def _suffixes(prompt):
    w = prompt.split()
    return [" ".join(w[i:]) for i in range(len(w) - 1)]                 # en uzundan 2 kelimeye


def _count_job(args):
    """Row group -> olgu basina (a, b), istem birebir, son ek sayilari ve devam ornekleri."""
    import pyarrow.parquet as pq
    path, g, ids, facts = args
    if not len(ids):
        return None
    texts = pq.ParquetFile(path).read_row_group(g, columns=["text"]).column("text").to_pylist()
    out = dict(a=np.zeros(len(facts), np.int64), b=np.zeros(len(facts), np.int64), exact=np.zeros(len(facts), np.int64),
               suf=[collections.Counter() for _ in facts], snip=[collections.defaultdict(list) for _ in facts])
    tops = [[t.lower() for t in f["topic"]] for f in facts]
    keys = [[k.lower() for k in f["key"]] for f in facts]
    sufs = [[s.lower() for s in _suffixes(f["prompt"])] for f in facts]
    for j in ids:
        text = texts[j]
        low = text.lower()
        for i, f in enumerate(facts):
            if f["prompt"] in text:
                out["exact"][i] += 1
            for s in sufs[i]:
                p = low.find(s)
                if p >= 0:
                    out["suf"][i][s] += 1
                    lst = out["snip"][i][s]
                    if len(lst) < SNIPPETS:
                        lst.append(text[p + len(s):p + len(s) + 80])
                    break                                                   # en uzun tutan son ek
            if not any(t in low for t in tops[i]) or not any(k in low for k in keys[i]):
                continue
            out["a"][i] += 1
            kpos = [m.start() for k in keys[i] for m in re.finditer(re.escape(k), low)]
            tpos = [m.start() for t in tops[i] for m in re.finditer(re.escape(t), low)]
            kp = np.sort(np.array(kpos))
            near = any(abs(kp[np.clip(np.searchsorted(kp, t), 0, len(kp) - 1)] - t) <= WINDOW or
                       abs(kp[np.clip(np.searchsorted(kp, t) - 1, 0, len(kp) - 1)] - t) <= WINDOW for t in tpos)
            out["b"][i] += near
    return out


def count(args):
    import multiprocessing as mp
    import pyarrow.parquet as pq
    t0 = time.time()
    facts = json.load(open(args.facts, encoding="utf-8"))["facts"]
    meta = json.load(open(os.path.join(args.data, "source.json"), encoding="utf-8"))["source"]
    path = os.path.join(args.src, "raw", os.path.basename(meta["parquet"]))
    pf = pq.ParquetFile(path)
    size = pf.metadata.row_group(0).num_rows
    ids = np.load(os.path.join(args.data, "gpt2", "train_doc_ids.npy"))
    if args.docs:
        ids = ids[:args.docs]
    jobs = [(path, int(g), (ids[ids // size == g] - g * size).tolist(), facts) for g in np.unique(ids // size)]
    if args.workers > 1:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            parts = pool.map(_count_job, jobs, chunksize=1)
    else:
        parts = [_count_job(j) for j in jobs]
    parts = [p for p in parts if p is not None]
    a = sum(p["a"] for p in parts)
    b = sum(p["b"] for p in parts)
    exact = sum(p["exact"] for p in parts)
    counts, lookup = [], []
    for i, f in enumerate(facts):
        suf = collections.Counter()
        for p in parts:
            suf.update(p["suf"][i])
        hit = next((s for s in _suffixes(f["prompt"]) if suf[s.lower()] > 0), None)
        cont = collections.Counter()
        if hit:
            for p in parts:
                for sn in p["snip"][i][hit.lower()]:
                    cont[" ".join(sn.split()[:LOOKUP_WORDS])] += 1
        ans = cont.most_common(1)[0][0] if cont else ""
        counts.append(dict(id=f["id"], docs_same=int(a[i]), docs_window=int(b[i]), band=band(int(a[i])),
                           prompt_exact_docs=int(exact[i])))
        lookup.append(dict(id=f["id"], suffix=hit, suffix_docs=int(suf[hit.lower()]) if hit else 0, answer=ans,
                           score=score(ans, f)[0]))
    os.makedirs(args.out, exist_ok=True)
    if os.path.abspath(args.facts) != os.path.abspath(os.path.join(args.out, "facts.json")):
        shutil.copyfile(args.facts, os.path.join(args.out, "facts.json"))
    info = dict(train_docs=int(len(ids)), window=WINDOW, rule="kucuk harf; (a) konu ve anahtar ayni belgede, (b) +-%d "
                "karakter; bant (a)'ya gore; istem birebir buyuk / kucuk harf duyarli" % WINDOW, seconds=round(time.time() - t0, 1))
    json.dump(dict(info, facts=counts), open(os.path.join(args.out, "counts.json"), "w", encoding="utf-8"), indent=1)
    json.dump(dict(rule="istemin train'de gecen en uzun son eki (kelime, en az 2) ve ardindan en sik ilk %d kelime" %
                   LOOKUP_WORDS, facts=lookup), open(os.path.join(args.out, "lookup.json"), "w", encoding="utf-8"), indent=1)
    bands = collections.Counter(c["band"] for c in counts)
    print("sayim: %d olgu, %d train belgesi, bant %s, arama tablosu DOGRU %d | %.0f sn" % (
        len(facts), len(ids), dict(bands), sum(x["score"] == "DOGRU" for x in lookup), time.time() - t0), flush=True)
    return counts, lookup


def run(args):
    import generate_readings as GR
    from tokenizers import Tokenizer
    t0 = time.time()
    dev = torch.device(args.device)
    model, idt = GR.load_run(args.run, args.data, dev)
    model.eval()
    tok = Tokenizer.from_file(os.path.join(args.data, "gpt2", "tokenizer.json"))
    facts = json.load(open(os.path.join(args.knowledge, "facts.json"), encoding="utf-8"))["facts"]
    counts = {c["id"]: c for c in json.load(open(os.path.join(args.knowledge, "counts.json"), encoding="utf-8"))["facts"]}
    lookup = {c["id"]: c for c in json.load(open(os.path.join(args.knowledge, "lookup.json"), encoding="utf-8"))["facts"]}
    dec = lambda ids: tok.decode(list(ids))  # noqa: E731
    prompts = [[tok.encode(f["prompt"]).ids] for f in facts]
    rows = []
    with torch.no_grad():
        ans = model.generate(prompts, open_last=True, **ANSWER)
        stop = model.generate(prompts, open_last=True, **STOP)
    for f, p, (g, e, eos), (g2, e2, eos2) in zip(facts, prompts, ans, stop):
        window = "".join(dec(s) for s in g[:2])
        res, kp, dp = score(window, f)
        rows.append(dict(id=f["id"], category=f["category"], band=counts[f["id"]]["band"],
                         docs_same=counts[f["id"]]["docs_same"], prompt=f["prompt"], answer=window, score=res,
                         lookup=lookup[f["id"]]["answer"], lookup_score=lookup[f["id"]]["score"],
                         stop_eos=bool(eos2), stop_sentences=len(g2), stop_tokens=int(sum(len(s) for s in g2)),
                         stop_text=dec(p[0]) + "".join(dec(s) for s in g2)))
    summ = {}
    for bname in [n for _, n in BANDS] + ["hepsi"]:
        r = [x for x in rows if bname == "hepsi" or x["band"] == bname]
        if r:
            summ[bname] = dict(n=len(r), **{k: round(sum(x["score"] == k for x in r) / len(r), 3) for k in ("DOGRU", "YANLIS", "BOS")},
                               lookup_dogru=round(sum(x["lookup_score"] == "DOGRU" for x in r) / len(r), 3),
                               eos=round(float(np.mean([x["stop_eos"] for x in r])), 3),
                               stop_sentences=round(float(np.mean([x["stop_sentences"] for x in r])), 1))
    out = dict(run=os.path.basename(os.path.normpath(args.run)), identity=idt, answer=ANSWER, stop=STOP, summary=summ,
               rows=rows, seconds=round(time.time() - t0, 1))
    json.dump(out, open(os.path.join(args.run, "knowledge_exam.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    txt = []
    for x in rows:
        txt += ["=== %s | %s | bant %s (%d belge) | %s | arama tablosu: %s (%s)" % (
            x["id"], x["category"], x["band"], x["docs_same"], x["score"], x["lookup"], x["lookup_score"]),
            x["prompt"] + " >>> " + x["answer"].replace("\n", "\\n"),
            "--- durma: EOS %s, %d cumle, %d token" % (x["stop_eos"], x["stop_sentences"], x["stop_tokens"]), ""]
    open(os.path.join(args.run, "knowledge_exam.txt"), "w", encoding="utf-8").write("\n".join(txt))
    print(json.dumps(summ, indent=1), flush=True)
    return out


def _args(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("count")
    c.add_argument("--src", required=True, help="fineweb koku (raw parquet; SALT OKUNUR)")
    c.add_argument("--data", required=True, help="v2/fineweb_edu_s000 (source.json, gpt2/train_doc_ids.npy)")
    c.add_argument("--out", required=True, help="knowledge klasoru")
    c.add_argument("--facts", default=FACTS)
    c.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    c.add_argument("--docs", type=int, default=0, help="yalniz ilk N train belgesi (sinama)")
    r = sub.add_parser("run")
    r.add_argument("--run", required=True)
    r.add_argument("--data", required=True)
    r.add_argument("--knowledge", required=True)
    r.add_argument("--device", default="cuda")
    return ap.parse_args(argv)


def main(argv=None):
    args = _args(argv)
    if os.name == "nt":
        import train as TR
        TR._no_power_throttling()
    return count(args) if args.cmd == "count" else run(args)


if __name__ == "__main__":
    main()
