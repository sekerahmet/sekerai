"""z_ablate -- Model Z'de gecmis kanalini degerlendirme aninda kapatma (belge 33 s4, belge 35 s5.2; ad onayli, 6 Ekim).

Sinavda (exam_pack_plan.npz; gap_v2 ile ayni batch'leme ve maske yolu) hedef basina nll, kosul kosul: none,
read_off (Z_k kendi cumlesini okumaz: yerel bloklar model_z_mask; G'siz modelde gecmis bilgisi tamamen kesilir, global
bloklar aynen kalir).  Formullu z kosullari (pos0, bag0, half*_0, z0) kaldirildi (belge 44; etiket
v2-before-formula-cleanup-20261007).
Uyari: kapatma okumasi yon icindir; egitilmis model o kosulu gormedi (kapatma != parcasiz egitim).

    python z_ablate.py --runs <kosu klasoru> [...] --data <v2/simplestories_gpt2> --stream <simplestories>
                       [--stories N] [--device cuda] [--out <json>]
"""
import torch  # noqa: I001  (Windows: torch once)

import argparse
import contextlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
sys.path.insert(0, HERE)
import data as D  # noqa: E402
import gap_v2 as G  # noqa: E402

CONDS = ("none", "read_off")


@contextlib.contextmanager
def ablated(model, cond):
    """-> mask_fn: none modelin kendi kurali; read_off yerel bloklarda model_z_mask (global bloklar aynen)."""
    S = sys.modules[type(model).__module__]
    if cond == "none":
        yield model.mask_fn
        return
    assert cond == "read_off", cond
    yield (S.model_z_mask, model.mask_fn[1]) if model.global_layers else S.model_z_mask


def ablate(model, layout, valid, rows, dev, conds):
    """-> {kosul: dict(hepsi, first, mid, end, eos, fark (- none))}, ayrica hedef basina nll (kosul -> array)."""
    out, per = {}, {}
    for cond in conds:
        with ablated(model, cond) as mask_fn:
            nll, _, tk = G.per_target(model, mask_fn, layout, valid, rows, dev)
        per[cond] = nll
        r = dict(hepsi=round(float(nll.mean()), 4))
        for k, name in G.KINDS.items():
            r[name] = round(float(nll[tk == k].mean()), 4)
        out[cond] = r
    for cond in conds:
        out[cond]["fark"] = {k: round(out[cond][k] - out["none"][k], 4) for k in ("hepsi", "first", "mid", "end", "eos")}
    return out, per


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", required=True, help="Model Z kosu klasorleri (agent.pt)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--stories", type=int, default=None, help="yalniz ilk N sinav satirinin hikayeleri")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    log = lambda m: print("[%6.1f sn] %s" % (time.time() - t0, m), flush=True)  # noqa: E731
    dev = G.setup(args.device, log)
    valid = D.TokenStories(args.stream, args.data, "valid")
    rows = G.exam_rows(args.data, args.stories)
    res = {}
    for run in args.runs:
        model, _, layout, idt = G.load(run, args.data, dev)
        assert layout == "model_z", "%s: Model Z degil" % run
        if dev.type == "cuda":
            for block in model.blocks:
                block.compile(dynamic=False)
        r, _ = ablate(model, layout, valid, rows, dev, CONDS)
        name = os.path.basename(os.path.normpath(run))
        res[name] = dict(kosullar=r, global_layers=model.global_layers, stories=args.stories)
        for c in CONDS:
            log("%-44s %-9s nll %.4f  fark %+.4f (first %+.4f, mid %+.4f)" % (name, c, r[c]["hepsi"], r[c]["fark"]["hepsi"],
                                                                            r[c]["fark"]["first"], r[c]["fark"]["mid"]))
        del model
    if args.out:
        json.dump(res, open(args.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return res


if __name__ == "__main__":
    main()
