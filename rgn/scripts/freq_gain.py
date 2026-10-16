"""Kazanç sıklığa göre nereden geliyor?  İki modelin aynı doğrulama pencerelerindeki token kaybı, (a) girdi token'ının
(b) hedef token'ın eğitimde görülme sayısına göre kovalara ayrılır.  Kelimeye özel nöron girdi token'ına bağlı olduğu için
(a) doğrudan etkiyi gösterir.

    python scripts/freq_gain.py --runs ss_d512_tf_yg_epok ss_d512_tf_yg_ozel --counts assets/ss_vocab_used.json

Kova başına: konum payı, iki modelin ortalama kaybı, fark, toplam farka katkı (fark × pay).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from rgn.data.loader import valid_tokens  # noqa: E402
from rgn.paths import ROOT  # noqa: E402
from rgn.train.checkpoint import load_run  # noqa: E402

BUCKETS = [(0, 100), (100, 1_000), (1_000, 10_000), (10_000, 100_000), (100_000, 1_000_000), (1_000_000, 1 << 62)]


@torch.no_grad()
def token_losses(m, W, batch):
    out = []
    for i in range(0, len(W), batch):
        x = W[i:i + batch]
        logits = m(x[:, :-1]).float()
        out.append(F.cross_entropy(logits.transpose(1, 2), x[:, 1:], reduction="none"))
    return torch.cat(out)


def _h(n):
    return "%dM" % (n // 1_000_000) if n >= 1_000_000 else "%dk" % (n // 1000) if n >= 1000 else str(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs=2, required=True)
    ap.add_argument("--counts", default="assets/ss_vocab_used.json")
    ap.add_argument("--dataset", default="ss")
    ap.add_argument("--windows", type=int, default=192)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c = json.loads((ROOT / a.counts).read_text(encoding="utf-8"))
    cnt = np.zeros(50304, np.int64)
    cnt[np.array(c["tokens"])] = np.array(c["counts"])
    s = valid_tokens(a.dataset, 4_000_000)
    starts = np.linspace(0, len(s) - a.seq - 2, a.windows).astype(int)
    W = torch.tensor(np.stack([s[p:p + a.seq + 1] for p in starts]).astype(np.int64)).to(dev)
    L = []
    for run in a.runs:
        m, step = load_run(run, dev)
        m.float()
        L.append(token_losses(m, W, a.batch).cpu().numpy().ravel())
        print("%s (adım %s): ortalama kayıp %.4f" % (run, step, L[-1].mean()))
        del m
    inp = cnt[W[:, :-1].cpu().numpy().ravel()]
    tgt = cnt[W[:, 1:].cpu().numpy().ravel()]
    d = L[1] - L[0]
    print("toplam fark %+.4f (%d konum)" % (d.mean(), len(d)))
    for name, key in (("GİRDİ token'ının sıklığı (özel satır bu token'ın)", inp), ("HEDEF token'ın sıklığı", tgt)):
        print("\n%s" % name)
        print("%-18s %8s %10s %10s %10s %14s" % ("eğitimde görülme", "pay", a.runs[0][:10], a.runs[1][:10], "fark",
                                                "toplam katkı"))
        for lo, hi in BUCKETS:
            mk = (key >= lo) & (key < hi)
            if mk.sum() == 0:
                continue
            share = mk.mean()
            print("%-18s %7.1f%% %10.4f %10.4f %+10.4f %+14.4f" % (
                "%s–%s" % (_h(lo), _h(hi) if hi < 1 << 60 else "∞"), 100 * share, L[0][mk].mean(), L[1][mk].mean(),
                d[mk].mean(), d[mk].mean() * share))


if __name__ == "__main__":
    main()
