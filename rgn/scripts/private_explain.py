"""Kısa bir metinde kelimeye özel nöronlar ne yapıyor?  (teşhis; CPU'da da çalışır)

Her token t (t+1'i tahmin ederken) için:
  Δ        : özel nöronlar bütün katmanlarda kapatılınca t+1'in kaybındaki artış (nat; + = özel nöron yardım ediyor).
             Toplam etki: kapatılan nöron sonraki katmanlara ve dikkat yoluyla sonraki konumlara da yansır.
  katman   : doğru sonraki token'ın logit'ini en çok iten katmanın nöronu (doğrudan logit katkısı, DLA:
             nöron çıktısı a·w, son RMSNorm'un ölçeğiyle, lm_head'e izdüşürülür; logit lens / nostalgebraist 2020).
  ittikleri: bu konumdaki özel nöron çıktılarının toplamının (bütün katmanlar) en çok yükselttiği 3 token.
Sonra katman başına özet (|özel| / |MLP| boyu, yalnız o katmanın özel nöronu kapatılınca Δ) ve en çok yardım edilen
birkaç konum için katman katman: a (nöronun bağlamdaki etkinliği, kelime başına tek nöron), doğru token'a DLA,
o katmanın ittiği token'lar.

    python scripts/private_explain.py --run fw_s000_d512_ozel_3epok
    python scripts/private_explain.py --run fw_s000_d512_ozel_3epok --text "The capital of Utah is Salt Lake City."

Bağlam kıyası (--focus kelime=hedef, birden çok --text): kelimenin her geçtiği yerde katman başına a ve hedef token'a
DLA; kapı bağlama bakıyorsa a bağlama göre değişir, bakmıyorsa (düz bigram) her bağlamda aynı itiş.
    python scripts/private_explain.py --run fw_s000_d512_ozel_3epok --focus "Brigham= Young"         --text "It was founded by Brigham Young in 1847." "She was treated at Brigham and Women's Hospital."
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from rgn.train.checkpoint import load_run  # noqa: E402

TEXT = ("Salt Lake City is the capital of Utah. It was founded by Brigham Young in 1847. Today the city is known "
        "for its mountains, its universities and its winter sports.")


def _forward(m, idx, off=()):
    """(son normalize öncesi artık akış (N, d), katman başına bileşenler, logit'ler (N, V)); off: özel nöronu
    kapatılan katmanlar."""
    for l in off:
        b = m.blocks[l]
        b.private_out = lambda xn, rows=None, priv=None, name="private", _b=b: \
            type(_b).private_out(_b, xn, rows, priv, name) * 0
    try:
        parts = []
        h = m.hidden(idx, parts=parts)
    finally:
        for l in off:
            del m.blocks[l].private_out
    x = parts[0]["embed"] + sum(p["attn"] + (p["mlp"] if p["mlp"] is not None else 0)
                                + (p["private"] if p["private"] is not None else 0) for p in parts[1:])
    return x[0], parts[1:], m.lm_head(h)[0].float()


def _nll(logits, idx):
    return F.cross_entropy(logits[:-1], idx[0, 1:], reduction="none")


def _act(m, l, xn, rows):
    """Kelime başına tek nöronun bağlamdaki etkinliği a = SiLU(g·x)(u·x) (ilk tablo)."""
    w = F.embedding(rows, getattr(m.blocks[l], m.blocks[l].priv_names[0])).view(*rows.shape, 3, -1)
    return F.silu((w[..., 0, :] * xn).sum(-1)) * (w[..., 1, :] * xn).sum(-1)


def _acts(m, idx, parts):
    """(L, N): her katmanda kelimenin tek nöronunun etkinliği; girdi xn = norm2(x + attn), bileşenlerden kurulur."""
    rows = m.priv_map[idx]
    cur, acts = m.embed(idx), []
    for l, b in enumerate(m.blocks):
        s = cur + parts[l]["attn"]
        acts.append(_act(m, l, b.norm2(s), rows)[0])
        cur = s + (parts[l]["mlp"] if parts[l]["mlp"] is not None else 0) + parts[l]["private"]
    return torch.stack(acts)


def _dla(m, x, parts):
    """(L, N, V): katman başına özel nöron çıktısının logit'lere doğrudan katkısı (son norm ölçeğiyle)."""
    scale = m.norm_f.weight.float() / x.pow(2).mean(-1, keepdim=True).add(m.norm_f.eps).sqrt()
    priv = torch.stack([p["private"][0] for p in parts])
    return torch.einsum("lnd,nd,vd->lnv", priv, scale, m.lm_head.weight.float())


def focus(m, enc, texts, pairs):
    L = len(m.blocks)
    print("bağlam kıyası: a = kelimenin o katmandaki nöronunun etkinliği; DLA = hedef token'a doğrudan itiş (logit)")
    for word, target in pairs:
        tid = enc.encode(target)
        assert len(tid) == 1, "hedef tek token olmalı: %r" % target
        tid = tid[0]
        print("\n== %r -> hedef %r" % (word, target))
        print("%-46s %7s %9s %8s  %s  %s" % ("bağlam", "p(hedef)", "özelsiz", "DLA top.", " ".join("%6s" % ("a%d" % l)
                                                                                     for l in range(L)), "ittikleri"))
        for text in texts:
            idx = torch.tensor([enc.encode(text)])
            tok = [enc.decode([t]) for t in idx[0].tolist()]
            pos = [t for t, w in enumerate(tok) if w.strip() == word.strip()]
            if not pos:
                continue
            x, parts, logits = _forward(m, idx)
            off = _forward(m, idx, off=range(L))[2]
            dla, acts = _dla(m, x, parts), _acts(m, idx, parts)
            for t in pos:
                p, p0 = (logits[t].softmax(-1)[tid], off[t].softmax(-1)[tid])
                top = [enc.decode([int(v)]) for v in dla[:, t].sum(0).topk(3).indices]
                ctx = ("…" + "".join(tok[max(0, t - 6):t + 2]))[-46:]
                print("%-46s %7.3f %9.3f %+8.2f  %s  %s" % (ctx, p, p0, dla[:, t, tid].sum(), " ".join(
                    "%+6.1f" % acts[l, t] for l in range(L)), " ".join(map(repr, top))))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--text", nargs="+", default=[TEXT])
    ap.add_argument("--focus", nargs="*", default=[], help='"kelime=hedef" (hedef boşlukla: "Brigham= Young")')
    ap.add_argument("--detail", type=int, default=4, help="katman katman dökülecek konum sayısı")
    a = ap.parse_args()
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    m, step = load_run(a.run, "cpu", write_weights=True)
    m.float()
    assert m.cfg.private_neurons, "özel nöronsuz model"
    if a.focus:
        return focus(m, enc, a.text, [f.split("=", 1) for f in a.focus])
    L = len(m.blocks)
    idx = torch.tensor([enc.encode(a.text[0])])
    N = idx.shape[1]
    tok = [enc.decode([t]) for t in idx[0].tolist()]
    x, parts, logits = _forward(m, idx)
    priv = torch.stack([p["private"][0] for p in parts])                            # (L, N, d)
    dla = _dla(m, x, parts)                                                          # katman, konum, token
    nll = _nll(logits, idx)
    nll_off = _nll(_forward(m, idx, off=range(L))[2], idx)
    delta = nll_off - nll
    tgt = idx[0, 1:]
    dla_t = dla[:, :-1].gather(2, tgt.view(1, -1, 1).expand(L, -1, 1))[..., 0]      # (L, N-1) doğru token'a

    print("%s (adım %s) — %d token.  Δ: özel nöronlar kapatılınca sonraki token'ın kaybı (nat)" % (a.run, step, N))
    print("\n%-14s %-14s %6s %6s %7s %7s  %s" % ("token", "sonraki", "p", "p(özelsiz)", "Δ", "katman", "özel nöronların ittikleri"))
    tot = dla.sum(0)
    for t in range(N - 1):
        top = [enc.decode([int(v)]) for v in tot[t].topk(3).indices]
        lbest = int(dla_t[:, t].argmax())
        print("%-14r %-14r %6.3f %10.3f %+7.2f %4d %+.1f  %s" % (
            tok[t], tok[t + 1], (-nll[t]).exp(), (-nll_off[t]).exp(), delta[t], lbest, dla_t[lbest, t],
            " ".join(repr(s) for s in top)))
    print("\nortalama kayıp: tam %.3f, özelsiz %.3f (Δ %+.3f); Δ>0 konum %d / %d" % (
        nll.mean(), nll_off.mean(), delta.mean(), int((delta > 0).sum()), N - 1))

    print("\n%-7s %13s %14s %16s" % ("katman", "|özel|/|MLP|", "Δ (yalnız bu)", "doğru token DLA"))
    for l in range(L):
        mlp = parts[l]["mlp"]
        ratio = (priv[l].norm(dim=-1) / mlp[0].norm(dim=-1)).mean() if mlp is not None else float("nan")
        d_l = (_nll(_forward(m, idx, off=[l])[2], idx) - nll).mean()
        print("%-7d %13.2f %+14.3f %+16.2f" % (l, ratio, d_l, dla_t[l].mean()))

    acts = _acts(m, idx, parts)                                                      # (L, N)
    print("\nen çok yardım edilen %d konum, katman katman (a: kelimenin o katmandaki tek nöronunun etkinliği)" % a.detail)
    for t in delta.topk(min(a.detail, N - 1)).indices.tolist():
        print("\n  %r -> %r   Δ %+.2f" % (tok[t], tok[t + 1], delta[t]))
        print("  %-7s %7s %9s  %s" % ("katman", "a", "DLA doğru", "ittikleri (+) / bastırdıkları (−)"))
        for l in range(L):
            up = [enc.decode([int(v)]) for v in dla[l, t].topk(3).indices]
            dn = [enc.decode([int(v)]) for v in (-dla[l, t]).topk(2).indices]
            print("  %-7d %+7.2f %+9.2f  %s / %s" % (l, acts[l, t], dla_t[l, t], " ".join(map(repr, up)),
                                                      " ".join(map(repr, dn))))


if __name__ == "__main__":
    main()
