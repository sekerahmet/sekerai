"""Y katmanları pencerenin uzak yarısında neye bakıyor, ve iki Y aynı şeye mi bakıyor?

Uzak bant: [far_lo, pencere) uzaklık (varsayılan pencere/2 .. pencere; D'de 64–127).  Sorgular --qmin'den sonra.
1. Uzak paya giden anahtarların türü (sırayla ilk eşleşen):
     tümevarım   : anahtardan önceki token = sorgu token'ı ("A B … A → B"; Olsson vd. 2022)
     aynı token  : anahtar = sorgu token'ı (önceki geçiş)
     sık token   : en sık --top token (noktalama, "the", ...) ya da belge sonu — dikkat yatağı (Xiao vd. 2023)
     içerik      : geri kalan
2. Katmanlar arası örtüşme: her sorgu için katmanın uzak banttaki dağılımı (head ortalaması, bant içinde 1'e
   normalize); iki katman için Σ min(P1, P2) (0: farklı konumlar, 1: aynı).  Ortalama, iki katmanın uzak payıyla
   ağırlıklı.  G katmanları da aynı bantta karşılaştırılır.

    python scripts/attn_far.py --run fw_s000_d1024L9_yari_epok
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rgn.data.loader import valid_tokens  # noqa: E402
from rgn.data.registry import EOS_ID  # noqa: E402
from rgn.paths import ROOT  # noqa: E402
from rgn.train.checkpoint import load_run  # noqa: E402

KINDS = ("tümevarım", "aynı token", "sık token", "içerik")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", default="fineweb_s000")
    ap.add_argument("--counts", default="assets/fw_s000_vocab_used.json")
    ap.add_argument("--top", type=int, default=30, help="'sık token' sayılan en sık token sayısı")
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--qmin", type=int, default=256)
    ap.add_argument("--far_lo", type=int, default=0, help="0: yerel pencerenin yarısı")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, step = load_run(a.run, dev)
    m.float()
    N, W = a.seq, m.cfg.local_window
    lo, hi = (a.far_lo or W // 2), W
    c = json.loads((ROOT / a.counts).read_text(encoding="utf-8"))
    order = sorted(zip(c.get("counts") or [1] * len(c["tokens"]), c["tokens"]), key=lambda ct: -ct[0])
    frequent = torch.tensor(sorted({t for _, t in order[:a.top]} | {EOS_ID}), device=dev)
    toks = valid_tokens(a.dataset, a.windows * N)
    idx = torch.from_numpy(np.asarray(toks, dtype=np.int64)).view(a.windows, N)
    dist = torch.arange(N, device=dev)[:, None] - torch.arange(N, device=dev)[None, :]
    band = (dist >= lo) & (dist < hi)
    qsel = torch.arange(N, device=dev) >= a.qmin
    L, kinds = len(m.blocks), [b.kind for b in m.blocks]
    far = torch.zeros(L, device=dev)                                   # uzak bant payı (head ve sorgu ortalaması)
    by_kind = torch.zeros(L, len(KINDS), device=dev)
    ov_num = torch.zeros(L, L, device=dev)
    ov_den = torch.zeros(L, L, device=dev)
    un_num = torch.zeros(L, device=dev)                                # referans: düzgün dağılımla örtüşme
    uni = band[qsel].float()
    uni = uni / uni.sum(-1, keepdim=True)
    for w in range(a.windows):
        t = idx[w].to(dev)
        # anahtar türü, sorgu × anahtar (N, N): sırayla ilk eşleşen
        prev = torch.cat([t.new_full((1,), -1), t[:-1]])
        ind = prev[None, :] == t[:, None]
        same = (t[None, :] == t[:, None]) & ~ind
        freq = torch.isin(t, frequent)[None, :].expand(N, N) & ~ind & ~same
        cont = ~(ind | same | freq)
        masks = [k & band for k in (ind, same, freq, cont)]
        x = m.embed(t[None])
        rope = m.rope(N)
        rows = m.priv_map[t[None]] if m.priv_map is not None else None
        ple = m.ple_context(x)
        P = []
        for l, b in enumerate(m.blocks):
            p = b.attn.weights(b.norm1(x), rope)[0].mean(0)              # (N, N) head ortalaması
            pb = (p * band)[qsel]                                         # uzak bant, seçili sorgular
            fl = pb.sum(-1)                                               # sorgu başına uzak pay
            far[l] += fl.mean()
            for k, mk in enumerate(masks):
                by_kind[l, k] += (p * mk)[qsel].sum(-1).mean()
            P.append((pb / fl.clamp_min(1e-9)[:, None], fl))
            x = b(x, rope, None, rows, None, None, None if ple is None else ple[:, :, l])
        for i in range(L):
            un_num[i] += (torch.minimum(P[i][0], uni).sum(-1) * P[i][1] ** 2).sum()
            for j in range(L):
                wgt = P[i][1] * P[j][1]
                ov_num[i, j] += (torch.minimum(P[i][0], P[j][0]).sum(-1) * wgt).sum()
                ov_den[i, j] += wgt.sum()
    far /= a.windows
    by_kind /= a.windows
    ov = ov_num / ov_den.clamp_min(1e-12)
    un = un_num / ov_den.diagonal().clamp_min(1e-12)
    print("%s (adım %s) — %d pencere × %d, sorgu konumu ≥ %d; uzak bant %d–%d; sık token: en sık %d + belge sonu" % (
        a.run, step, a.windows, N, a.qmin, lo, hi - 1, a.top))
    print("\n1. uzak banda giden dikkat (%) ve türlere bölünüşü (uzak payın %'si)")
    print("%-7s %-4s %9s  %s" % ("katman", "tür", "uzak pay", " ".join("%11s" % k for k in KINDS)))
    for l in range(L):
        sh = (100 * by_kind[l] / far[l].clamp_min(1e-12)).tolist()
        print("%-7d %-4s %8.1f%%  %s" % (l, kinds[l], 100 * far[l], " ".join("%10.1f%%" % v for v in sh)))
    print("\n2. katmanlar arası örtüşme, uzak bantta Σ min(P1, P2) (1: aynı konumlar); son sütun referans: düzgün "
          "dağılımla örtüşme (yüksekse dağılım yayvan, örtüşme bilgisiz)")
    print("      " + " ".join("%5s" % ("%d%s" % (j, kinds[j])) for j in range(L)) + "   düzgün")
    for i in range(L):
        print("%-5s " % ("%d%s" % (i, kinds[i])) + " ".join("%5.2f" % ov[i, j] for j in range(L)) + "   %5.2f" % un[i])


if __name__ == "__main__":
    main()
