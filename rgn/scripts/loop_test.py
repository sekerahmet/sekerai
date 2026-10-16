"""Döngü testi: greedy (argmax) üretimde tekrar.

    python scripts/loop_test.py --runs ss_d512_tf_yg_epok ss_d512_tf_yg_ozel
    python scripts/loop_test.py --runs a b --stories 256 --prompt 32 --new 384

İstem: SS doğrulama hikâyelerinin ilk --prompt token'ı (hikâyeler eşit aralıkla seçilir, her model aynı istemleri görür).
Üretim: greedy, --new token, KV önbelleksiz toplu üretim; ilk EOS'ta (hikâye bitti) o satır durur.
Ölçütler (istemden sonraki metin, EOS'a kadar):
    bitirdi      hikâyeyi EOS ile kendisi bitiren üretim oranı (döngüdeki model bitiremez)
    cümle tekr.  en az bir cümleyi birebir ikinci kez söyleyen üretim oranı
    tekr. cümle  bütün cümleler içinde öncekinin birebir tekrarı olanların oranı
    sıkışma      son 3 cümlesi aynı olan üretim oranı (kalıcı döngü)
    rep-4        tekrarlanan 4-gram oranı (1 - farklı-4)
    uzunluk      ortalama üretilen token (EOS'a kadar)
Model ağırlıkları rgn.train.checkpoint.load_run ile.  Sonuç runs/_loop_test.json.
"""
import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rgn.system import setup_process  # noqa: E402
from rgn.data.loader import valid_tokens  # noqa: E402
from rgn.data.registry import EOS_ID  # noqa: E402
from rgn.paths import runs_dir  # noqa: E402
from rgn.train.checkpoint import load_run  # noqa: E402

_SENT = re.compile(r"[^.!?]+[.!?]+[\"”']?")


def prompts(n, plen, seed=0):
    """SS doğrulama hikâyelerinin ilk plen token'ı: (n, plen) — hikâyeler eşit aralıkla, plen'den kısa olanlar atlanır."""
    a = valid_tokens("ss", 6_000_000)
    starts = np.flatnonzero(a == EOS_ID) + 1
    ends = np.append(starts[1:] - 1, len(a))
    ok = [s for s, e in zip(starts, ends) if e - s > plen + 64]
    pick = np.linspace(0, len(ok) - 1, n).astype(int)
    return torch.tensor(np.stack([a[ok[i]:ok[i] + plen].astype(np.int64) for i in pick]))


def load_model(run, device):
    m, _ = load_run(run, device)
    return m, m.cfg


@torch.no_grad()
def generate(model, idx, new, batch, device):
    """Greedy, KV önbelleksiz; satır EOS üretince durur (sonrası EOS ile dolar).  -> (n, new) üretilen token'lar."""
    out = []
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
    for i in range(0, len(idx), batch):
        x = idx[i:i + batch].to(device)
        done = torch.zeros(len(x), dtype=torch.bool, device=device)
        gen = []
        for _ in range(new):
            with ctx:
                h = model.hidden(x)[:, -1]
            nxt = (h.float() @ model.lm_head.weight.float().T).argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, EOS_ID), nxt)
            done |= nxt == EOS_ID
            gen.append(nxt)
            x = torch.cat([x, nxt[:, None]], 1)
            if bool(done.all()):
                break
        g = torch.stack(gen, 1)
        if g.shape[1] < new:
            g = torch.cat([g, torch.full((len(g), new - g.shape[1]), EOS_ID, device=device)], 1)
        out.append(g.cpu())
    return torch.cat(out)


def analyse(gen, enc):
    rows = []
    for g in gen.tolist():
        n = g.index(EOS_ID) if EOS_ID in g else len(g)
        toks = g[:n]
        text = enc.decode(toks)
        sents = [s.strip().lower() for s in _SENT.findall(text) if len(s.split()) >= 3]
        seen, rep = set(), 0
        for s in sents:
            rep += s in seen
            seen.add(s)
        grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
        c = Counter(sents)
        rows.append(dict(ended=EOS_ID in g, n=n, sents=len(sents), rep_sents=rep,
                         stuck=len(sents) >= 3 and len(set(sents[-3:])) == 1,
                         rep4=1 - len(set(grams)) / len(grams) if grams else 0.0,
                         top=c.most_common(1)[0] if c else ("", 0), text=text))
    k = len(rows)
    total_sents = sum(r["sents"] for r in rows)
    return dict(n=k,
                ended=sum(r["ended"] for r in rows) / k,
                any_rep=sum(r["rep_sents"] > 0 for r in rows) / k,
                rep_sent_frac=sum(r["rep_sents"] for r in rows) / max(1, total_sents),
                stuck=sum(r["stuck"] for r in rows) / k,
                rep4=sum(r["rep4"] for r in rows) / k,
                length=sum(r["n"] for r in rows) / k), rows


def main():
    setup_process()
    ap = argparse.ArgumentParser(description="greedy döngü testi")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--stories", type=int, default=256)
    ap.add_argument("--prompt", type=int, default=32)
    ap.add_argument("--new", type=int, default=384)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--examples", type=int, default=3)
    a = ap.parse_args()
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx = prompts(a.stories, a.prompt)
    print("istem: %d hikâye × ilk %d token, greedy %d token" % (len(idx), a.prompt, a.new), flush=True)
    res = {}
    for run in a.runs:
        t0 = time.time()
        m, cfg = load_model(run, device)
        gen = generate(m, idx, a.new, a.batch, device)
        s, rows = analyse(gen, enc)
        s["sec"] = time.time() - t0
        res[run] = dict(summary=s, examples=sorted(rows, key=lambda r: -r["top"][1])[:a.examples])
        print("  %-32s %.0f sn" % (run, s["sec"]), flush=True)
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print()
    print("%-32s %8s %11s %11s %9s %7s %8s" % ("koşu", "bitirdi", "cümle tekr.", "tekr. cümle", "sıkışma", "rep-4", "uzunluk"))
    for run, r in res.items():
        s = r["summary"]
        print("%-32s %7.1f%% %10.1f%% %10.1f%% %8.1f%% %7.3f %8.0f" % (
            run[:32], 100 * s["ended"], 100 * s["any_rep"], 100 * s["rep_sent_frac"], 100 * s["stuck"], s["rep4"],
            s["length"]))
    for run, r in res.items():
        print("\n== %s — en çok tekrar eden üretimler" % run)
        for ex in r["examples"]:
            sent, cnt = ex["top"]
            print("  [%d×] \"%s\"" % (cnt, sent[:120]))
            print("     » %s" % ex["text"][:300].replace("\n", " "))
    out = runs_dir() / "_loop_test.json"
    out.write_text(json.dumps(dict(args=vars(a), results=res), ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nkaydedildi:", out)


if __name__ == "__main__":
    main()
