"""Toplu değerlendirme: her model BİR KEZ yüklenir, sonra olgu testi + döngü testi + üretim hızı arka arkaya.

    python scripts/eval_all.py --runs fw_s000_d1024L9 fw_s000_d1024L9_yari --dataset fineweb_s000

Checkpoint'ten yüklenen koşu için hafif model.pt yazılır (sonraki yüklemeler hızlı).  Sonda tek kıyas tablosu.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from rgn.paths import ROOT  # noqa: E402
from rgn.train.checkpoint import load_run  # noqa: E402
from rgn.system import setup_process  # noqa: E402
from fact_probe import BUCKETS, from_file, score  # noqa: E402
from gen_speed import gen_rate  # noqa: E402
from loop_test import analyse, generate as loop_generate, prompts  # noqa: E402


def main():
    setup_process()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--dataset", default="fineweb_s000")
    ap.add_argument("--facts", default="assets/olgu_s000.tsv")
    ap.add_argument("--stories", type=int, default=256)
    ap.add_argument("--prompt", type=int, default=64)
    ap.add_argument("--new", type=int, default=256)
    a = ap.parse_args()
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qs = from_file(ROOT / a.facts)
    idx = prompts(a.stories, a.prompt, a.dataset)
    rows = {}
    for run in a.runs:
        t0 = time.time()
        m, step = load_run(run, dev, write_weights=True)
        t_load = time.time() - t0
        ok = [int(max(range(4), key=lambda i: sc[i][0]) == 0)
              for sc in (score(m, enc, q[1], [q[2]] + list(q[3]), dev) for q in qs)]
        by = {"%d–%s" % (lo, hi if hi < 10 ** 9 else "∞"): [o for o, q in zip(ok, qs) if lo <= q[5] <= hi]
              for lo, hi in BUCKETS}
        loop, _ = analyse(loop_generate(m, idx, a.new, 128, dev), enc)
        _, ms1 = gen_rate(m, 1, 512, 64, dev, torch.bfloat16)
        tps64, ms64 = gen_rate(m, 64, 512, 64, dev, torch.bfloat16)
        rows[run] = dict(step=step, fact=100 * sum(ok) / len(ok), by={k: 100 * sum(v) / max(1, len(v)) for k, v in by.items()},
                         loop=loop, ms1=ms1, tps64=tps64, load=t_load, sec=time.time() - t0)
        r = rows[run]
        print("%-30s adım %s  yükleme %.0f sn, toplam %.0f sn | olgu %%%.1f  tekrar %%%.1f  sıkışma %%%.1f  rep-4 %.3f  "
              "%.2f ms/tok (b1)  %.0f tok/s (b64)" % (run, step, t_load, r["sec"], r["fact"], 100 * loop["any_rep"],
                                                     100 * loop["stuck"], loop["rep4"], ms1, tps64), flush=True)
        del m
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    keys = list(next(iter(rows.values()))["by"])
    print("\n%-30s %6s %6s %s %8s %8s %8s %10s %10s" % ("koşu", "adım", "olgu%", " ".join("%7s" % k for k in keys),
                                                       "tekrar%", "sıkışma%", "rep-4", "ms/tok b1", "tok/s b64"))
    for run, r in rows.items():
        l_ = r["loop"]
        print("%-30s %6s %6.1f %s %8.1f %8.1f %8.3f %10.2f %10.0f" % (
            run[:30], r["step"], r["fact"], " ".join("%7.1f" % r["by"][k] for k in keys), 100 * l_["any_rep"],
            100 * l_["stuck"], l_["rep4"], r["ms1"], r["tps64"]))
    print("\n(olgu: %d soru, şans %%25; döngü: %d belge × %d token istem, greedy %d token; hız: önbellekli, bağlam 512)" % (
        len(qs), len(idx), a.prompt, a.new))


if __name__ == "__main__":
    main()
