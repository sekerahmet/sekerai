"""Tokenizer kıyası (Türkçe): kendi byte-level BPE'lerimizi eğitir, hazırlarla aynı ölçüm metninde karşılaştırır.

Adaylar: GPT-2 (tiktoken), ytu-ce-cosmos/turkish-gpt2 (byte-level BPE), dbmdz/bert-base-turkish-cased (WordPiece 32k;
boşluğu düşürür, kayıplı), kendi BPE'lerimiz (--sizes; rakamlar tek tek, GPT-2 tarzı ön bölme, 256 bayt alfabe).
Ölçüler (her ölçüm dosyası için): token / bayt, token / kelime (boşlukla ayrılan), ve hedef model için bayt başına
eğitim işlemi ∝ token/bayt × (gövde + PLE bağlamı + V·d) (--body, --d; varsayılan d1536 L15).  Ayrıca örnek kelimelerin
nasıl bölündüğü.

    python scripts/tok_compare.py --sizes 16000 32000 48000
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("rgn_paths", Path(__file__).resolve().parents[1] / "rgn" / "paths.py")
drive_root = importlib.util.module_from_spec(_spec)                       # rgn paketini (torch) yüklemeden: pyarrow
_spec.loader.exec_module(drive_root)                                       # sonrası torch Windows'ta DLL hatası verir
drive_root = drive_root.drive_root

WORDS = ["İstanbul’un", "Ankara'nın", "Atatürk'ün", "don't", "it's", "evlerimizden", "gidemeyeceklermiş", "öğretmenlerimize", "değerlendirilmesi", "İstanbul'da", "kitaplıktaki",
         "yapılandırılmış", "bilgisayarlarımızdaki", "Çekoslovakyalılaştıramadıklarımızdan", "1923 yılında",
         "Türkiye Cumhuriyeti", "çalışıyorum"]


def read(path, limit_mb=0):
    out, b = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)["text"]
            out.append(t)
            b += len(t.encode("utf-8"))
            if limit_mb and b >= limit_mb * 1e6:
                break
    return out


# Ön bölme kalıbı "tr": GPT-2'ninkinden İngilizce kısaltma kuralı ('s|'t|'re|'ve|'m|'ll|'d) çıkarıldı (Türkçe ekleri
# bozuyordu: İstanbul'da → 'd | a); yerine kesme işareti (' ya da ’) + harf dizisi tek parça ('da, 'nın, 's, 't).
# Rakamlar tek tek (önünde boşluk olabilir).
PRE_TR = r"""[’']\p{L}+| ?\p{L}+| ?\p{N}| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pre_tokenizer(kind):
    from tokenizers import Regex, pre_tokenizers
    if kind == "gpt2":                                                 # GPT-2 kalıbı + tek tek rakam
        return pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True),
                                        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)])
    return pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(PRE_TR), behavior="isolated"),
                                    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])


def train_bpe(texts, size, out, pre="tr"):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizer(pre)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=size, min_frequency=2, special_tokens=["<|endoftext|>"],
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(texts, tr, length=len(texts))
    tok.save(str(out))
    return tok


def encoders(root, own):
    """ad -> (kodlayıcı: str -> token listesi, kelime dağarcığı boyu, parçalar: str -> parça dizgileri)."""
    import tiktoken
    from tokenizers import Tokenizer
    from tokenizers.implementations import BertWordPieceTokenizer
    g = tiktoken.get_encoding("gpt2")
    out = {"gpt2 (50k)": (lambda s: g.encode_ordinary(s), g.n_vocab,
                          lambda s: [g.decode([t]) for t in g.encode_ordinary(s)])}
    h = root / "hazir"
    yt = Tokenizer.from_file(str(h / "ytu-ce-cosmos__turkish-gpt2" / "tokenizer.json"))
    out["turkish-gpt2 (%dk)" % round(yt.get_vocab_size() / 1000)] = (
        lambda s: yt.encode(s, add_special_tokens=False).ids, yt.get_vocab_size(),
        lambda s: [yt.decode([t]) for t in yt.encode(s, add_special_tokens=False).ids])
    bt = BertWordPieceTokenizer(str(h / "dbmdz__bert-base-turkish-cased" / "vocab.txt"), lowercase=False,
                                strip_accents=False, clean_text=True)
    out["bert-turkish (%dk, WordPiece)" % round(bt.get_vocab_size() / 1000)] = (
        lambda s: bt.encode(s, add_special_tokens=False).ids, bt.get_vocab_size(),
        lambda s: bt.encode(s, add_special_tokens=False).tokens)
    for name, tok in own.items():
        out[name] = (lambda s, t=tok: t.encode(s, add_special_tokens=False).ids, tok.get_vocab_size(),
                     lambda s, t=tok: [t.decode([i]) for i in t.encode(s, add_special_tokens=False).ids])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[16000, 32000, 48000])
    ap.add_argument("--train_mb", type=float, default=0, help="eğitim metni sınırı (0: hepsi)")
    ap.add_argument("--body", type=float, default=354e6, help="gövde + PLE bağlamı (d1536 L15: 319M + 35M)")
    ap.add_argument("--d", type=int, default=1536)
    ap.add_argument("--no_train", action="store_true", help="kendi BPE'leri diskten yükle")
    ap.add_argument("--pre", default="tr", choices=["tr", "gpt2"], help="ön bölme kalıbı (dosya adına eklenir)")
    ap.add_argument("--also", nargs="*", default=[], help="kıyasa eklenecek eğitilmiş dosyalar (egitilen/ altında)")
    a = ap.parse_args()
    root = drive_root() / "tokenizer"
    ex = root / "ornek"
    (root / "egitilen").mkdir(parents=True, exist_ok=True)
    from tokenizers import Tokenizer
    own = {}
    fname = lambda s: "tr_bpe_%dk%s.json" % (s // 1000, "" if a.pre == "gpt2" else "_" + a.pre)
    for f in a.also:
        own["eski " + Path(f).stem] = Tokenizer.from_file(str(root / "egitilen" / f))
    if a.no_train:
        for s in a.sizes:
            own["bizim TR %dk (%s)" % (s // 1000, a.pre)] = Tokenizer.from_file(str(root / "egitilen" / fname(s)))
    else:
        texts = read(ex / "train_tr_web.jsonl", a.train_mb) + read(ex / "train_tr_wiki.jsonl", a.train_mb)
        print("eğitim metni: %d belge, %.0f MB" % (len(texts), sum(len(t.encode("utf-8")) for t in texts) / 1e6), flush=True)
        for s in a.sizes:
            t0 = time.time()
            own["bizim TR %dk (%s)" % (s // 1000, a.pre)] = train_bpe(texts, s, root / "egitilen" / fname(s), a.pre)
            print("  BPE %d: %.0f sn" % (s, time.time() - t0), flush=True)
        del texts
    enc = encoders(root, own)
    evals = {n: read(ex / ("eval_tr_%s.jsonl" % n)) for n in ("web", "wiki")}
    print("\n%-30s %7s | %s | %s" % ("tokenizer", "V", " ".join("%9s %9s" % ("tok/bayt", "tok/kel") for _ in evals),
                                    "bayt başına işlem (GPT-2 = 1)"))
    print("%-38s | %s |" % ("", " ".join("%-19s" % ("— %s —" % n) for n in evals)))
    rows = {}
    for name, (f, V, _) in enc.items():
        r = []
        for docs in evals.values():
            nb = sum(len(d.encode("utf-8")) for d in docs)
            nw = sum(len(d.split()) for d in docs)
            nt = sum(len(f(d)) for d in docs)
            r.append((nt / nb, nt / nw))
        rows[name] = (V, r)
    base = rows["gpt2 (50k)"]
    cost = lambda V, tpb: tpb * (a.body + V * a.d)
    for name, (V, r) in rows.items():
        rel = [cost(V, x[0]) / cost(base[0], b[0]) for x, b in zip(r, base[1])]
        print("%-30s %7d | %s | %s" % (name, V, " ".join("%9.4f %9.3f" % x for x in r), "  ".join("%.3f" % x for x in rel)))
    print("\nörnek kelimeler:")
    for w in WORDS:
        print("  %s" % w)
        for name, (_, _, parts) in enc.items():
            print("    %-30s %s" % (name, " | ".join(parts(w))))


if __name__ == "__main__":
    main()
