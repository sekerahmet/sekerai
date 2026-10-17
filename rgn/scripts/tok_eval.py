"""Tokenizer eğitimi (çeşitli derlem) ve son kıyas: bütün ölçüm setlerinde token / kelime, kayıpsızlık, bayt başına işlem,
zor durum testleri.  Derlem: scripts/tok_corpus.py (<drive>/tokenizer/derlem).

  eğit  : python scripts/tok_eval.py --train --sizes 32000 40000 48000
  kıyas : python scripts/tok_eval.py --own tr_bpe_32k_v2.json ... --hazir ytu-ce-cosmos/turkish-gpt2 moganai/MoganBERT-TR

Kayıpsız = belge kodla → çöz → aynı metin (sohbet modeli için şart: satır sonu, boşluk, emoji, büyük harf).  İşlem: hedef
model (d1536 L15) için bayt başına eğitim işlemi ∝ token/bayt × (gövde + PLE bağlamı + V·d); GPT-2 = 1.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tok_compare import SPECIAL, drive_root, encoders, read, train_bpe  # noqa: E402

EDGE = ["İSTANBUL'DA BÜYÜK HARFLE YAZILMIŞ BİR CÜMLE.", "ıIiİ ılık İlk ışık ISPARTA", "Ankara'nın, İzmir’in ve Muğla'ya",
        "Toplantı 14:30'da, 3.250,75 TL ödendi; %18 KDV.", "https://www.ornek.com.tr/sayfa?id=42 ve e-posta: ad@alan.com",
        "Harika!!! 😀🎉 Görüşürüz 👋", "1. madde\n2. madde\n\n- liste öğesi\n  - alt öğe",
        "yaptırabileceklerimizdenmişsinizcesine", "COVID-19, 5G, iPhone 15 Pro ve ChatGPT", "Ünlü uyumu: kitaplık, gözlük, ağaçlık"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--sizes", type=int, nargs="+", default=[32000, 40000, 48000])
    ap.add_argument("--cap_mb", type=float, default=0, help="kaynak başına eğitim metni sınırı (0: hepsi)")
    ap.add_argument("--own", nargs="*", default=[], help="egitilen/ altındaki tokenizer dosyaları")
    ap.add_argument("--hazir", nargs="*", default=[])
    ap.add_argument("--body", type=float, default=354e6)
    ap.add_argument("--d", type=int, default=1536)
    ap.add_argument("--max_eval_mb", type=float, default=5, help="ölçüm seti başına (hız için)")
    ap.add_argument("--derlem", default="", help="derlemin yerel kopyası (boş: <drive>/tokenizer/derlem)")
    a = ap.parse_args()
    root = drive_root() / "tokenizer"
    der = Path(a.derlem) if a.derlem else root / "derlem"
    from tokenizers import Tokenizer
    own = {}
    if a.train:
        texts = []
        for f in sorted(der.glob("train_*.jsonl")):
            t = read(f, a.cap_mb)
            print("  %-26s %7d belge %6.0f MB" % (f.name, len(t), sum(len(x.encode("utf-8")) for x in t) / 1e6), flush=True)
            texts += t
        print("eğitim metni: %d belge, %.0f MB" % (len(texts), sum(len(t.encode("utf-8")) for t in texts) / 1e6), flush=True)
        for s in a.sizes:
            t0 = time.time()
            name = "tr_bpe_%dk_v2.json" % (s // 1000)
            own["bizim %dk v2" % (s // 1000)] = train_bpe(texts, s, root / "egitilen" / name, "tr", SPECIAL)
            print("  BPE %d: %.0f sn -> %s" % (s, time.time() - t0, name), flush=True)
        del texts
    for f in a.own:
        own[Path(f).stem.replace("tr_bpe_", "bizim ")] = Tokenizer.from_file(str(root / "egitilen" / f))
    enc = encoders(root, own, a.hazir)
    evals = {f.stem.replace("eval_", ""): read(f, a.max_eval_mb) for f in sorted(der.glob("eval_*.jsonl"))}
    names = list(evals)
    print("\nölçüm setleri: " + ", ".join("%s (%d belge)" % (n, len(d)) for n, d in evals.items()))
    print("\n== token / kelime (az = iyi)")
    print("%-36s %7s %s %8s %10s" % ("tokenizer", "V", " ".join("%9s" % n[:9] for n in names), "ortalama", "kayıpsız %"))
    res = {}
    for name, (f, V, _) in enc.items():
        tw, tb, ok, n = [], [], 0, 0
        dec = getattr(f, "decode", None)
        for docs in evals.values():
            nb = sum(len(d.encode("utf-8")) for d in docs)
            nw = sum(len(d.split()) for d in docs)
            ids = [f(d) for d in docs]
            nt = sum(len(x) for x in ids)
            tw.append(nt / nw)
            tb.append(nt / nb)
        res[name] = (V, tw, tb)
    los = lossless(enc, evals)
    base = res["gpt2 (50k)"]
    cost = lambda V, x: x * (a.body + V * a.d)
    for name, (V, tw, tb) in res.items():
        print("%-36s %7d %s %8.3f %10.1f" % (name[:36], V, " ".join("%9.3f" % x for x in tw), sum(tw) / len(tw), los[name]))
    print("\n== bayt başına işlem (GPT-2 = 1; düşük = iyi)")
    for name, (V, tw, tb) in res.items():
        rel = [cost(V, x) / cost(base[0], b) for x, b in zip(tb, base[2])]
        print("%-36s %7d %s %8.3f" % (name[:36], V, " ".join("%9.3f" % x for x in rel), sum(rel) / len(rel)))
    print("\n== zor durumlar (token sayısı; ✗ = geri çözülen metin farklı)")
    for s in EDGE:
        print("  %r" % s)
        for name, (f, V, parts) in enc.items():
            p = parts(s)
            okk = "✓" if roundtrip(enc, name, s) else "✗"
            print("    %-34s %s %3d  %s" % (name[:34], okk, len(p), " | ".join(x.replace("\n", "↵") for x in p)[:150]))


_TOK = {}


def roundtrip(enc, name, s):
    f = _TOK.get(name)
    return True if f is None else f(s) == s


def lossless(enc, evals):
    """Her tokenizer için belgelerin % kaçı kodla → çöz sonrası aynı."""
    import tiktoken
    from tokenizers import Tokenizer  # noqa: F401
    g = tiktoken.get_encoding("gpt2")
    out = {}
    for name, (f, V, parts) in enc.items():
        if name.startswith("gpt2"):
            dec = lambda s: g.decode(g.encode_ordinary(s))
        else:
            tok = f.__defaults__[0] if f.__defaults__ else None
            if tok is None:
                out[name] = float("nan")
                continue
            dec = lambda s, t=tok: t.decode(t.encode(s, add_special_tokens=False).ids, skip_special_tokens=False)
        _TOK[name] = dec
        docs = [d for ds in evals.values() for d in ds[:300]]
        out[name] = 100 * sum(dec(d) == d for d in docs) / len(docs)
    return out


if __name__ == "__main__":
    main()
