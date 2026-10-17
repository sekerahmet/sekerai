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


def train_unigram(texts, size, out, pre="tr", max_piece=32):
    """Kayıpsız Unigram: aynı byte-level ön bölme (256 bayt alfabede; bilinmeyen karakter yok); max_piece byte-level
    harf (Türkçe harf 1–2 bayt)."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.Unigram())
    tok.pre_tokenizer = pre_tokenizer(pre)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.UnigramTrainer(vocab_size=size, special_tokens=["<|endoftext|>", "<unk>"], unk_token="<unk>",
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), max_piece_length=max_piece,
                                 show_progress=False)
    tok.train_from_iterator(texts, tr, length=len(texts))
    tok.save(str(out))
    return tok


# Sohbet ve araç token'ları + ileride eklenecekler için boş yer (toplam 128; kelime dağarcığı boyuna dahil)
SPECIAL = (["<|endoftext|>", "<|pad|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
            "<|tool_call|>", "<|/tool_call|>", "<|tool_result|>", "<|/tool_result|>"]
           + ["<|reserved_%d|>" % i for i in range(118)])


def train_bpe(texts, size, out, pre="tr", special=("<|endoftext|>",)):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizer(pre)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=size, min_frequency=2, special_tokens=list(special),
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(texts, tr, length=len(texts))
    tok.save(str(out))
    return tok


HAZIR = ["ytu-ce-cosmos/turkish-gpt2", "dbmdz/bert-base-turkish-cased"]


def hazir_tok(h, repo):
    """Hazır tokenizer (yalnız tokenizer.json ya da vocab.txt indirilir) -> (Tokenizer, etiket)."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    from tokenizers.implementations import BertWordPieceTokenizer
    d = h / repo.replace("/", "__")
    try:
        p = Path(hf_hub_download(repo, "tokenizer.json", local_dir=d)) if not (d / "tokenizer.json").exists() else d / "tokenizer.json"
        tok = Tokenizer.from_file(str(p))
        kind = json.loads(p.read_text(encoding="utf-8"))["model"]["type"]
    except Exception:                                                  # yalnız vocab.txt (BERT)
        p = Path(hf_hub_download(repo, "vocab.txt", local_dir=d)) if not (d / "vocab.txt").exists() else d / "vocab.txt"
        tok, kind = BertWordPieceTokenizer(str(p), lowercase=False, strip_accents=False, clean_text=True), "WordPiece"
    V = tok.get_vocab_size()
    return tok, "%s (%dk%s)" % (repo.split("/")[1][:22], round(V / 1000), "" if kind == "BPE" else ", " + kind)


def encoders(root, own, hazir=HAZIR):
    """ad -> (kodlayıcı: str -> token listesi, kelime dağarcığı boyu, parçalar: str -> parça dizgileri)."""
    import tiktoken
    g = tiktoken.get_encoding("gpt2")
    out = {"gpt2 (50k)": (lambda s: g.encode_ordinary(s), g.n_vocab,
                          lambda s: [g.decode([t]) for t in g.encode_ordinary(s)])}
    for repo in hazir:
        tok, name = hazir_tok(root / "hazir", repo)
        wp = name.endswith("WordPiece)")
        out[name] = (lambda s, t=tok: t.encode(s, add_special_tokens=False).ids, tok.get_vocab_size(),
                     (lambda s, t=tok: t.encode(s, add_special_tokens=False).tokens) if wp else
                     (lambda s, t=tok: [t.decode([i]) for i in t.encode(s, add_special_tokens=False).ids]))
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
    ap.add_argument("--model", default="bpe", choices=["bpe", "unigram"])
    ap.add_argument("--hazir", nargs="*", default=HAZIR, help="hazır tokenizer'lar (Hugging Face; yalnız tokenizer dosyası)")
    a = ap.parse_args()
    root = drive_root() / "tokenizer"
    ex = root / "ornek"
    (root / "egitilen").mkdir(parents=True, exist_ok=True)
    from tokenizers import Tokenizer
    own = {}
    kind = {"bpe": "BPE", "unigram": "Unigram"}[a.model]
    fname = lambda s: "tr_%s_%dk%s.json" % ("bpe" if a.model == "bpe" else "uni", s // 1000, "" if a.pre == "gpt2" else "_" + a.pre)
    for f in a.also:
        own["eski " + Path(f).stem] = Tokenizer.from_file(str(root / "egitilen" / f))
    if a.no_train:
        for s in a.sizes:
            own["bizim TR %s %dk (%s)" % (kind, s // 1000, a.pre)] = Tokenizer.from_file(str(root / "egitilen" / fname(s)))
    else:
        texts = read(ex / "train_tr_web.jsonl", a.train_mb) + read(ex / "train_tr_wiki.jsonl", a.train_mb)
        print("eğitim metni: %d belge, %.0f MB" % (len(texts), sum(len(t.encode("utf-8")) for t in texts) / 1e6), flush=True)
        for s in a.sizes:
            t0 = time.time()
            fn = train_bpe if a.model == "bpe" else train_unigram
            own["bizim TR %s %dk (%s)" % (kind, s // 1000, a.pre)] = fn(texts, s, root / "egitilen" / fname(s), a.pre)
            print("  %s %d: %.0f sn" % (kind, s, time.time() - t0), flush=True)
        del texts
    enc = encoders(root, own, a.hazir)
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
