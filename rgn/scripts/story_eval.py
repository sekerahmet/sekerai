"""Aynı hikâye satırlarında kıyas: rastgele pencereyle ve hikâye düzeniyle eğitilmiş modeller aynı token'larda.

    python scripts/story_eval.py --runs ss_d512_tf_yg_epok ss_d512_tf_yg_cumle

Doğrulama hikâyeleri hikâye başından, cümle sonunda kesilen 512'lik satırlara dizilir (rgn/data/stories.py); her model
aynı token'larda ölçülür.  Rastgele pencere tabanı: eğitildiği gibi maskesiz ve hikâye maskeli (Y pencere 128);
cümle yerel model: kendi maskesiyle.  Ek sütun: modelin kendi düzenindeki (rastgele pencere) doğrulama kaybı.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from rgn.data.loader import valid_tokens  # noqa: E402
from rgn.data.registry import get_spec  # noqa: E402
from rgn.data.stories import load_story_rows  # noqa: E402
from rgn.paths import runs_dir  # noqa: E402
from rgn.system import setup_process  # noqa: E402
from rgn.train.checkpoint import load_run  # noqa: E402
from rgn.train.evaluate import evaluate_rows  # noqa: E402


def _final(run):
    p = runs_dir() / run / "results.json"
    return json.loads(p.read_text(encoding="utf-8"))["final"]["loss"] if p.exists() else None


def main():
    setup_process()
    ap = argparse.ArgumentParser(description="aynı hikâye satırlarında kıyas")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tokens", type=int, default=0, help="0: bütün doğrulama")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
    spec = get_spec("ss")
    stream = valid_tokens("ss")
    rows = load_story_rows(stream, "v2/simplestories_gpt2", "valid").fixed_rows(512, a.tokens)
    n_tok = int((rows[1] != -100).sum())
    print("doğrulama: %d satır, %d token (dolgu %%%.1f)" % (len(rows[0]), n_tok, 100 * float((rows[1] == -100).float().mean())))
    print("%-26s %-28s %10s %8s %8s %18s" % ("koşu", "ölçüm", "kayıp", "bpb", "doğruluk", "kendi düzeni (SON)"))
    for run in a.runs:
        m, step = load_run(run, device)
        c = m.cfg
        modes = [("kendi maskesi (cümle)", True, 0)] if c.local_sentence else \
            [("maskesiz (eğitildiği gibi)", False, 0), ("hikâye maskeli", True, c.local_window)]
        for name, use, win in modes:
            r = evaluate_rows(m, rows, a.batch, device, ctx, spec.bytes_per_token, use_masks=use, window=win)
            f = _final(run)
            print("%-26s %-28s %10.4f %8.4f %8.4f %18s" % (run[:26], name, r["loss"], r["bpb"], r["acc"],
                                                            "%.4f" % f if f else "-"))
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
