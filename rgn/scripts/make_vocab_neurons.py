"""Sözlüğü nöronlara PMI komşuluğuyla bölüştürür (kullanıcı, 16 Ekim: "PMI ile atama yap, k 1").

    python scripts/make_vocab_neurons.py --pmi "v2/pmi/fineweb_edu_s000_ppmi2_k5.npz" --neurons 1344

PMI-SVD vektörleri (v2 make_pmi.py: cümle içi PPMI, 64 boyut; FineWeb-Edu s000 — SS için ayrı tablo yok, kelime
ilişkileri genel İngilizce) küresel k-ortalamayla `neurons` kümeye ayrılır; atama dengeli: her nöron ⌈V / N⌉ token'ı
geçmez (sık token önce, en yakın boş kümeye).  Vektörü olmayan token'lar (PMI'de az görülen / sözlük dolgusu) kalan
yerlere rastgele.  Çıktı assets/vocab_neurons_pmi_<N>.json: {"table": V tamsayı (token -> nöron), ...}.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rgn.paths import ROOT, drive_root  # noqa: E402

LOOK = [" cat", " dog", " king", " river", " red", " she", " happy", " ran", " the", " Tom"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmi", default="v2/pmi/fineweb_edu_s000_ppmi2_k5.npz")
    ap.add_argument("--neurons", type=int, default=1344)
    ap.add_argument("--vocab", type=int, default=50304)
    ap.add_argument("--iters", type=int, default=30)
    a = ap.parse_args()
    z = np.load(drive_root() / a.pmi, allow_pickle=True)
    S, counts = torch.from_numpy(z["S"]).float(), np.asarray(z["counts"])
    V, N = a.vocab, a.neurons
    X = torch.zeros(V, S.shape[1])
    X[:len(S)] = S
    norm = X.norm(dim=1)
    valid = (norm > 0).numpy()
    X = X / norm.clamp_min(1e-12)[:, None]
    g = torch.Generator().manual_seed(0)
    vid = torch.from_numpy(np.flatnonzero(valid))
    C = X[vid[torch.randperm(len(vid), generator=g)[:N]]].clone()
    for _ in range(a.iters):                                            # küresel k-ortalama (geçerli token'lar)
        lab = (X[vid] @ C.T).argmax(1)
        C = torch.zeros_like(C).index_add_(0, lab, X[vid])
        empty = C.norm(dim=1) == 0
        C[empty] = X[vid[torch.randint(len(vid), (int(empty.sum()),), generator=g)]]
        C = C / C.norm(dim=1, keepdim=True)
    cap = -(-V // N)
    load = np.zeros(N, np.int64)
    table = np.full(V, -1, np.int64)
    cnt = np.zeros(V, np.int64)
    cnt[:len(counts)] = counts
    order = vid.numpy()[np.argsort(-cnt[vid.numpy()], kind="stable")]  # sık token önce
    cand = torch.topk(X[torch.from_numpy(order)] @ C.T, 64, dim=1).indices.numpy()
    for t, cs in zip(order, cand):
        for c in cs:
            if load[c] < cap:
                table[t], load[c] = c, load[c] + 1
                break
        else:
            c = int(np.argmin(load))
            table[t], load[c] = c, load[c] + 1
    rest = np.flatnonzero(table < 0)
    rng = np.random.default_rng(0)
    slots = np.concatenate([np.full(cap - l, i) for i, l in enumerate(load)])
    table[rest] = rng.permutation(slots)[:len(rest)]
    load = np.bincount(table, minlength=N)
    print("geçerli (PMI vektörlü) token %d / %d | nöron başına token %d..%d (ort %.1f)" % (
        valid.sum(), V, load.min(), load.max(), V / N))
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        for w in LOOK:
            ids = enc.encode_ordinary(w)
            if len(ids) == 1:
                grp = np.flatnonzero(table == table[ids[0]])
                grp = grp[np.argsort(-cnt[grp])][:12]
                print("%-8r nöron %4d: %s" % (w, table[ids[0]], " | ".join(repr(enc.decode([int(t)])) for t in grp)))
    except ImportError:
        pass
    out = ROOT / "assets" / ("vocab_neurons_pmi_%d.json" % N)
    out.write_text(json.dumps({"neurons": N, "vocab": V, "source": a.pmi, "rule": "küresel k-ortalama, dengeli (sık önce)",
                               "table": table.tolist()}), encoding="utf-8")
    print("yazıldı:", out)


if __name__ == "__main__":
    main()
