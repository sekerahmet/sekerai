"""run_summary -- kosu basina tek satir: sinav + performans (belge 48; kullanici, 7 Ekim: "Bu lr koşularında performans da
bakalım").  results.json (+ ayni klasordeki *.log'da 'Recompiling' sayisi) -> sinav kayip / bpb / acc, ms / adim ve
token / sn ortancasi (ilk pencere haric), tepe GPU bellegi (allocated, reserved; pencerelerin en buyugu), ilk pencere
(derleme dahil) suresi, MFU (HESAP).

MFU = token / sn x FLOP / token / PEAK.  FLOP / token = 6 N (N: butun parametre; bagli E cikis carpiminda bir kez) + 12 d
sum_katman k (attention, ileri + geri; k: sorgu basina gorulen ortalama anahtar, token agirlikli).  k veriden (HESAP):
global / transformer katmani: hikaye (parca) icinde konum, sum L^2 / (2 sum L); Model Z yerel katmani: onceki ozet (BOS +
Z) + kendi cumlesindeki konum.  PEAK: RTX PRO 6000 Blackwell bf16 yogun ~500 TFLOP/s (NVIDIA urun sayfasi "1 PFLOP BF16"
seyreklikle; yogun = yarisi; jarvislabs olcumu 504 teorik; HESAP).

    python run_summary.py <kosu klasoru> [...] [--data <veri klasoru>] [--peak 5e14]
"""
import argparse
import glob
import json
import os

import numpy as np

PEAK = 5.0e14


def attention_keys(data):
    """-> (k_global, k_local): token agirlikli ortalama gorulen anahtar (train parcalari)."""
    sent = np.load(os.path.join(data, "train_sentence_offsets.npy"), mmap_mode="r")
    story = np.load(os.path.join(data, "train_story_offsets.npy"))
    L1 = (sent[:, 1] - sent[:, 0] + 1).astype(np.float64)                 # cumle + END / Z
    n = np.diff(story)
    T = np.add.reduceat(L1, story[:-1]) + 1                               # parca boyu (BOS dahil)
    k_global = float((T * (T + 1) / 2).sum() / T.sum())
    k_idx = np.arange(len(L1)) - np.repeat(story[:-1], n)                 # hikaye ici cumle no
    k_local = float(((k_idx + 1) * L1 + L1 * (L1 + 1) / 2).sum() / L1.sum())   # ozet (BOS + onceki Z) + cumle ici
    return k_global, k_local


def summarize(run, data=None, peak=PEAK):
    r = json.load(open(os.path.join(run, "results.json"), encoding="utf-8"))
    idt, log = r["identity"], r["log"]
    win = [w for w in log if not w["first"]]
    first = [w for w in log if w["first"]]
    rec = sum(open(p, encoding="utf-8", errors="replace").read().count("Recompiling") for p in glob.glob(os.path.join(run, "*.log")))
    out = dict(run=os.path.basename(os.path.normpath(run)), exam_loss=r["exam"]["loss"], bpb=r["exam"]["bits_per_byte"],
               acc=r["exam"]["acc"], ms_per_step=r["speed"]["ms_per_step_median"], tokens_per_sec=r["speed"]["tokens_per_sec_median"],
               peak_alloc_gb=max((w.get("peak_gb") or 0) for w in log), peak_reserved_gb=max((w.get("peak_reserved_gb") or 0) for w in log),
               first_window_sec=first[0]["seconds"] if first else None, recompiling=rec, params=r["params"])
    if data and out["tokens_per_sec"]:
        kg, kl = attention_keys(data)
        G = idt.get("global_layers", 0) if idt["model"] == "model_z" else idt["layers"]
        flop = 6 * r["params"] + 12 * idt["d"] * (G * kg + (idt["layers"] - G) * kl)
        out.update(flop_per_token=round(flop / 1e9, 3), k_global=round(kg, 1), k_local=round(kl, 1),
                   mfu=round(out["tokens_per_sec"] * flop / peak, 4))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--data", default=None)
    ap.add_argument("--peak", type=float, default=PEAK)
    a = ap.parse_args(argv)
    rows = []
    for run in a.runs:
        if not os.path.exists(os.path.join(run, "results.json")):
            print("%s | SONUC YOK" % os.path.basename(os.path.normpath(run)))
            continue
        s = summarize(run, a.data, a.peak)
        rows.append(s)
        print("%s | sinav %.4f bpb %.4f acc %.4f | %.1f ms/adim, %.0f tok/sn | tepe %.1f / %.1f GB (allocated / reserved) | ilk "
              "pencere %.0f sn | Recompiling %d%s" % (
                  s["run"], s["exam_loss"], s["bpb"], s["acc"], s["ms_per_step"] or 0, s["tokens_per_sec"] or 0,
                  s["peak_alloc_gb"], s["peak_reserved_gb"], s["first_window_sec"] or 0, s["recompiling"],
                  " | MFU %.3f (HESAP: %.3f GFLOP/token, k global %.0f / yerel %.0f, tepe %.0f TFLOP/s)" % (
                      s["mfu"], s["flop_per_token"], s["k_global"], s["k_local"], a.peak / 1e12) if "mfu" in s else ""),
              flush=True)
    return rows


if __name__ == "__main__":
    main()
