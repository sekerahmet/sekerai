"""Kayıp nerede?  Bir modelin (ve isteğe bağlı kıyas modelinin) aynı doğrulama pencerelerindeki token kaybı, kelime
düzeyinde dökülür: hedef / girdi kelimesinin eğitim sıklığına göre kovalar, kayba en çok katkı veren kelimeler, özel
nöronun tabana göre en çok kaybettirdiği / kazandırdığı kelimeler, kelime başı / kelime parçası token'ları.

    python scripts/word_loss.py --run fw_s000_d512_tf_yg_ozel_rms --base fw_s000_d512_tf_yg --dataset fineweb_s000 \
        --counts assets/fw_s000_vocab_used.json --seq 1024 --windows 1024

"Katkı" = o kelimedeki toplam kayıp / bütün konumlar (ortalama kayba eklediği pay).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rgn.data.loader import valid_tokens  # noqa: E402
from rgn.paths import ROOT  # noqa: E402
from rgn.train.checkpoint import load_run  # noqa: E402
from freq_gain import BUCKETS, _h, token_losses  # noqa: E402


def _dec(enc, t):
    return repr(enc.decode([int(t)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--counts", default="assets/fw_s000_vocab_used.json")
    ap.add_argument("--dataset", default="fineweb_s000")
    ap.add_argument("--windows", type=int, default=1024)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--min_n", type=int, default=50, help="kelime listelerinde en az konum")
    a = ap.parse_args()
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c = json.loads((ROOT / a.counts).read_text(encoding="utf-8"))
    cnt = np.zeros(50304, np.int64)
    cnt[np.array(c["tokens"])] = np.array(c["counts"])
    s = valid_tokens(a.dataset, 16_000_000)
    starts = np.linspace(0, len(s) - a.seq - 2, a.windows).astype(int)
    W = torch.tensor(np.stack([s[p:p + a.seq + 1] for p in starts]).astype(np.int64)).to(dev)
    L = {}
    for name in [a.run] + ([a.base] if a.base else []):
        m, step = load_run(name, dev)
        m.float()
        L[name] = token_losses(m, W, a.batch).cpu().numpy().ravel()
        print("%s (adım %s): ortalama kayıp %.4f" % (name, step, L[name].mean()))
        del m
        torch.cuda.empty_cache()
    lm = L[a.run]
    lb = L.get(a.base)
    inp = W[:, :-1].cpu().numpy().ravel()
    tgt = W[:, 1:].cpu().numpy().ravel()
    N = len(lm)

    print("\n== HEDEF kelimenin eğitim sıklığına göre (kaybın nerede olduğu)")
    print("%-14s %8s %10s %10s %10s" % ("sıklık", "konum %", "kayıp", "kayıp payı", "taban farkı" if lb is not None else ""))
    for lo, hi in BUCKETS:
        mk = (cnt[tgt] >= lo) & (cnt[tgt] < hi)
        if mk.sum() == 0:
            continue
        print("%-14s %7.1f%% %10.4f %9.1f%% %10s" % ("%s–%s" % (_h(lo), _h(hi) if hi < 1 << 60 else "∞"), 100 * mk.mean(),
              lm[mk].mean(), 100 * lm[mk].sum() / lm.sum(), "%+.4f" % (lm[mk] - lb[mk]).mean() if lb is not None else ""))

    word = np.array([enc.decode([int(t)]).startswith(" ") for t in range(50257)] + [False] * 47)
    print("\n== HEDEF token kelime başı mı, kelime parçası mı")
    for nm, mk in (("kelime başı (boşlukla)", word[tgt]), ("parça / noktalama", ~word[tgt])):
        print("%-24s konum %5.1f%%  kayıp %.4f  kayıp payı %5.1f%%%s" % (
            nm, 100 * mk.mean(), lm[mk].mean(), 100 * lm[mk].sum() / lm.sum(),
            "  taban farkı %+.4f" % (lm[mk] - lb[mk]).mean() if lb is not None else ""))

    def per_token(key, vals):
        n = np.bincount(key, minlength=50304)
        tot = np.bincount(key, weights=vals, minlength=50304)
        return n, tot

    n_t, tot_t = per_token(tgt, lm)
    print("\n== Kayba en çok katkı veren HEDEF kelimeler (kayıp × sıklık)")
    print("%-18s %8s %8s %9s %9s" % ("kelime", "konum", "kayıp", "katkı", "taban farkı" if lb is not None else ""))
    d_t = per_token(tgt, lm - lb)[1] if lb is not None else None
    for t in np.argsort(-tot_t)[:a.top]:
        print("%-18s %8d %8.3f %9.4f %9s" % (_dec(enc, t)[:18], n_t[t], tot_t[t] / n_t[t], tot_t[t] / N,
              "%+.3f" % (d_t[t] / n_t[t]) if d_t is not None else ""))

    if lb is not None:
        n_i, _ = per_token(inp, lm)
        d_i = per_token(inp, lm - lb)[1]
        ok = n_i >= a.min_n
        mean_d = np.where(ok, d_i / np.maximum(n_i, 1), 0)
        for title, order in (("özel nöronun EN ÇOK KAZANDIRDIĞI girdi kelimeleri (toplam katkı)", np.argsort(d_i)),
                             ("özel nöronun KAYBETTİRDİĞİ girdi kelimeleri (toplam katkı; hedefli eğitim adayı)",
                              np.argsort(-d_i))):
            print("\n== %s" % title)
            print("%-18s %8s %10s %10s %10s %10s" % ("girdi kelimesi", "konum", "eğitimde", "ort. fark", "katkı",
                                                     ""))
            k = 0
            for t in order:
                if not ok[t]:
                    continue
                print("%-18s %8d %10s %+10.3f %+10.5f" % (_dec(enc, t)[:18], n_i[t], _h(int(cnt[t])), mean_d[t], d_i[t] / N))
                k += 1
                if k >= a.top // 2:
                    break
        worse = ok & (mean_d > 0)
        print("\nözel nöronun ortalama kaybettirdiği girdi kelimesi: %d / %d (en az %d konum); bunların toplam katkısı %+.4f"
              % (worse.sum(), ok.sum(), a.min_n, d_i[worse].sum() / N))


if __name__ == "__main__":
    main()
