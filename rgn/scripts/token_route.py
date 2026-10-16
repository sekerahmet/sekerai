"""Bir kelimenin rotasını belirle ve izle (kullanıcı, 16 Ekim: "cat için YYG düzeninde route budur diye ben belirlemek ve
izlemek istiyorum").

    python scripts/token_route.py --word " cat" --make --neurons 64 --heads 4    # taslak rota: routes/cat.json
    python scripts/token_route.py --route routes/cat.json                        # rotayı zorla, ölç

Rota dosyası (elle düzenlenir): katman başına "heads" (o kelime konumunda açık kalan dikkat head'leri) ve "neurons" (açık
kalan MLP nöronları).  Zorlamada kelimenin geçtiği her konumda rota dışındaki head çıktıları ve nöronlar sıfırlanır;
diğer token'lar ve model değişmez.  Ölçüm (eğitilmiş model, doğrulama metni, kelimenin geçtiği N yer):
    kelimeden sonraki token'ın kaybı (kelime ne söylüyor) ve sonraki 10 token'ın kaybı (sonrakiler onu nasıl okuyor)
    rotasız / senin rotan / aynı boyda rastgele rota (kontrol)
    katman başına: rotasız modelde kelimenin MLP etkinliğinin ve dikkat çıktısının ne kadarı rotanın içinde
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from rgn.data.loader import _open  # noqa: E402
from rgn.data.registry import get_spec  # noqa: E402
from rgn.paths import ROOT, drive_root  # noqa: E402


@torch.no_grad()
def forward(m, W, where, route=None):
    """Elle ileri geçiş (softmax tabanı).  where (B, T) bool: kelimenin konumları.  route: katman başına (açık head
    maskesi (H,), açık nöron maskesi (ff,)) ya da None.  -> logit'ler (B, T, V), kayıt: katman başına
    (|nöron etkinliği| (B, T, ff), head çıktı büyüklüğü (B, H, T))."""
    B, T = W.shape
    rope = m.rope(T)
    x = m.embed(W)
    rec = []
    for li, blk in enumerate(m.blocks):
        at = blk.attn
        assert at.w_aux is None, "kapı / önem / odak taşıyan model desteklenmiyor"
        q, k, v = at.project(blk.norm1(x), rope)
        if at.Hkv != at.H:
            k, v = (t.repeat_interleave(at.H // at.Hkv, 1) for t in (k, v))
        s = (q @ k.transpose(-2, -1)) * at.dh ** -0.5
        keep = at._mask(T, x.device) if at.window else torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        o = torch.softmax(s.masked_fill(~keep, float("-inf")), -1) @ v              # (B, H, T, dh)
        hn = o.norm(dim=-1)
        if route is not None:
            off = ~route[li][0]
            o = o.masked_fill((off[None, :, None] & where[:, None, :]).unsqueeze(-1), 0.0)
        y = x + at.w_o(o.transpose(1, 2).reshape(B, T, -1))
        g, up = blk.mlp.a(blk.norm2(y)).chunk(2, dim=-1)
        h = F.silu(g) * up
        rec.append((h.abs(), hn))
        if route is not None:
            h = h.masked_fill(where.unsqueeze(-1) & ~route[li][1][None, None, :], 0.0)
        x = y + blk.mlp.c(h)
    return m.lm_head(m.norm_f(x)), rec


def losses(logits, W, c):
    lp = F.cross_entropy(logits[:, :-1].transpose(1, 2), W[:, 1:], reduction="none")    # (B, T-1)
    return lp[:, c].mean().item(), lp[:, c + 1:c + 11].mean().item()


def main():
    ap = argparse.ArgumentParser(description="kelime rotası: belirle ve izle")
    ap.add_argument("--run", default="ss_d512_tf_yg_epok")
    ap.add_argument("--word", default=" cat")
    ap.add_argument("--route", default="")
    ap.add_argument("--make", action="store_true", help="izleme verisinden taslak rota yaz")
    ap.add_argument("--neurons", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--n", type=int, default=48)
    a = ap.parse_args()
    if "RGN_RUNS" not in os.environ and not Path("/content").is_dir():
        os.environ["RGN_RUNS"] = str(drive_root() / "RGN" / "runs")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from attn_anatomy import load
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    if a.route:
        R = json.loads((ROOT / a.route).read_text(encoding="utf-8"))
        a.word = R["word"]
    tid = enc.encode_ordinary(a.word)
    assert len(tid) == 1, "kelime tek token olmalı"
    tid = tid[0]
    m, step = load(a.run, torch.device("cpu"))
    m.float()
    spec = get_spec("ss")
    stream = np.asarray(_open(drive_root() / spec.valid[0], spec.fmt)[:6_000_000])
    pos = np.flatnonzero(stream == tid)
    pos = pos[(pos >= 200) & (pos < len(stream) - 57)]
    pos = pos[np.linspace(0, len(pos) - 1, min(a.n, len(pos))).astype(int)]
    W = torch.tensor(np.stack([stream[p - 200:p + 56] for p in pos]).astype(np.int64))
    c, where = 200, W == tid
    L = len(m.blocks)
    H, ff = m.blocks[0].attn.H, m.blocks[0].mlp.c.in_features
    base_logits, rec = forward(m, W, where)
    if a.make:
        layers = []
        for li, (hab, hn) in enumerate(rec):
            neu = hab[:, c].mean(0).topk(a.neurons).indices.sort().values.tolist()
            hd = hn[:, :, c].mean(0).topk(a.heads).indices.sort().values.tolist()
            layers.append({"layer": li + 1, "kind": m.blocks[li].kind, "heads": hd, "neurons": neu})
        out = ROOT / "routes" / ("%s.json" % a.word.strip().lower())
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps({"word": a.word, "token": tid, "run": a.run, "rule": "taslak: rotasız modelde en çok "
                                   "çalışan %d nöron ve en büyük çıktılı %d head (katman başına); elle düzenlenebilir" % (
                                       a.neurons, a.heads), "layers": layers}, indent=1), encoding="utf-8")
        print("taslak rota yazıldı:", out)
        R = json.loads(out.read_text(encoding="utf-8"))

    def masks(spec_layers):
        out = []
        for lay in spec_layers:
            hm = torch.zeros(H, dtype=torch.bool)
            hm[lay["heads"]] = True
            nm = torch.zeros(ff, dtype=torch.bool)
            nm[lay["neurons"]] = True
            out.append((hm, nm))
        return out
    rt = masks(R["layers"])
    g = torch.Generator().manual_seed(0)
    rnd = [(torch.zeros(H, dtype=torch.bool).index_fill_(0, torch.randperm(H, generator=g)[:int(h.sum())], True),
            torch.zeros(ff, dtype=torch.bool).index_fill_(0, torch.randperm(ff, generator=g)[:int(n.sum())], True))
           for h, n in rt]
    r_logits, _ = forward(m, W, where, rt)
    x_logits, _ = forward(m, W, where, rnd)
    b0, b1 = losses(base_logits, W, c)
    r0, r1 = losses(r_logits, W, c)
    x0, x1 = losses(x_logits, W, c)
    print("== %s (adım %s) | %r, %d yer | rota: %s" % (a.run, step, a.word, len(W), a.route or "taslak"))
    print("%-24s %22s %26s" % ("", "kelimeden sonraki token", "sonraki 10 token (okuyanlar)"))
    print("%-24s %22.4f %26.4f" % ("rotasız (model kendisi)", b0, b1))
    print("%-24s %15.4f (%+.4f) %19.4f (%+.4f)" % ("senin rotan", r0, r0 - b0, r1, r1 - b1))
    print("%-24s %15.4f (%+.4f) %19.4f (%+.4f)" % ("rastgele rota (kontrol)", x0, x0 - b0, x1, x1 - b1))
    print("\nkatman  head (açık)   nöron   rotasız modelde rotanın payı: MLP etkinliği   dikkat çıktısı")
    for li, ((hab, hn), (hm, nm)) in enumerate(zip(rec, rt)):
        ms = hab[:, c][:, nm].sum() / hab[:, c].sum()
        hs = hn[:, hm, c].pow(2).sum() / hn[:, :, c].pow(2).sum()
        print("K%d %s   %-12s %5d   %38.2f %16.2f" % (li + 1, m.blocks[li].kind, ",".join(map(str, hm.nonzero().flatten().tolist())),
                                                int(nm.sum()), ms.item(), hs.item()))


if __name__ == "__main__":
    main()
