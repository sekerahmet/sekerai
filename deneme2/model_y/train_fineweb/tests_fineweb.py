# -*- coding: utf-8 -*-
"""tests_fineweb -- FineWeb egitiminin kapilari: tek seferlik uretim (build_shards; fw_tokenize.py V2 bicimi) ve okuma
(load), iz dogrulamasi (bozulunca durur), valid ayirmasi, paketli pencerelerin belge sinirlari ve her hedefin tam bir kez
sayilmasi, sinavin belge belge olcumu (exam = belge tek basina), bpb paydasi, tekrar olculeri ve Colab calistirici (CPU'da
uctan uca, surdurme).  CPU; Drive ve ag YOK: parquet'ler asagidaki uretilmis metinden, tokenizer burada egitilir (bayt
duzeyi BPE, eot <|endoftext|>).  Istege bagli: yereldeki gercek gpt2 tokenizer.json ve sample.parquet varsa bayt tablosu
gercek sozlukte de sinanir.  GPU yolu (compile + flex_attention, bf16, mfu) sinanamaz.

    python tests_fineweb.py
"""
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile

import torch  # pyarrow ve tokenizers'tan ONCE (Windows DLL sirasi)
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
torch.set_num_threads(1)

import colab_fineweb as C  # noqa: E402
import data_fineweb as DF  # noqa: E402
import exam_fineweb as EF  # noqa: E402
import model_y as M  # noqa: E402

RESULTS = []
SEQ = 64                                          # fixture baglami
STRIDE, EXAM = 4, 4                               # fixture valid adimi ve sinav alt kumesi
TINY = dict(d=16, turns=2, layers=1, heads=2, units=16, output_link=False, attention_log_scale=True)
WORDS = ("water cycle energy cell river mountain history empire trade science number theory planet light heat "
         "student teacher lesson example process system change growth plant animal ocean climate data value").split()
EXTRA = ("\u00e9t\u00e9", "M\u00fcller", "\u4e2d\u6587", "na\u00efve", "\U0001f600")   # cok baytli karakterler
REAL_TOKENIZER = os.path.expanduser("~/.cache/huggingface/hub/models--openai-community--gpt2/snapshots/"
                                    "607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json")
REAL_SAMPLE = ("C:/Users/Lenovo/AppData/Local/Temp/claude/c--AI-NEW-MODEL/c8909f31-4ce6-4f6c-9134-2c20db5b6a95/scratchpad/"
               "fineweb/sample.parquet")


def check(name, ok, note=""):
    RESULTS.append(bool(ok))
    print("  %-78s %s  %s" % (name, "GECTI" if ok else "KALDI", note))


def make_texts(n, seed):
    """Belgeler: 1-40 cumle (kisa ve SEQ'ten uzun belgeler), bazisinda cok baytli karakter."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        sentences = []
        for _ in range(rng.choice((1, 2, 3, 5, 8, 20, 40))):
            words = [rng.choice(WORDS + list(EXTRA) if rng.random() < 0.1 else WORDS) for _ in range(rng.randint(3, 9))]
            sentences.append(" ".join(words).capitalize() + rng.choice((".", "!", "?", ",")))
        out.append(" ".join(sentences) + ("\n\n1789: " + rng.choice(WORDS) if rng.random() < 0.3 else ""))
    return out


def make_tokenizer(texts):
    bpe = Tokenizer(models.BPE())
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.decoder = decoders.ByteLevel()
    bpe.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=500, special_tokens=[DF.EOT],
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return bpe


def fixture():
    """Gecici kok: raw/ 14 parquet + manifest.json (parca 13: 22 belge), tokenizer; build_shards yerel (ag yok)."""
    root = tempfile.mkdtemp()
    files, all_texts = [], {}
    os.makedirs(os.path.join(root, "raw"))
    for i in range(14):
        texts = make_texts(22 if i == DF.VALID_SHARD else 6, i)
        if i == 2:
            texts[3] = "A page that quotes <|endoftext|> literally."           # metin icinde eot: sayilir
        name = "sample/10BT/%03d_00000.parquet" % i
        path = os.path.join(root, "raw", os.path.basename(name))
        pq.write_table(pa.table({"text": texts, "id": ["<urn:%d:%d>" % (i, k) for k in range(len(texts))]}), path,
                       row_group_size=4)
        files.append(dict(path=name, size=os.path.getsize(path), sha256=DF._sha256(path)))
        all_texts[i] = texts
    json.dump(dict(repo="HuggingFaceFW/fineweb-edu", revision="test", files=files),
              open(os.path.join(root, "raw", "manifest.json"), "w"), indent=1)
    tok = make_tokenizer([t for v in all_texts.values() for t in v])
    DF.save_tokenizer(root, tok)
    return root, tok, all_texts


def load(root, **kw):
    return DF.load(root, seq_len=kw.pop("seq_len", SEQ), valid_stride=STRIDE, exam_docs=EXAM, log=lambda s: None, **kw)


def t_build(root, tok, texts):
    logs = []
    meta = DF.build_shards(root, tokenizer=tok, log=logs.append, batch=4)
    d = os.path.join(root, "gpt2")
    eot = tok.token_to_id(DF.EOT)
    ok = True
    for i, m in meta.items():
        p = DF._shard_paths(d, i)
        raw = open(p["bin"], "rb").read()
        a = np.frombuffer(raw, dtype=np.uint16)
        off, nb = np.load(p["offsets"]), np.load(p["bytes"])
        ends = np.append(off[1:], len(a))
        docs = [a[s + 1:e].tolist() for s, e in zip(off.tolist(), ends.tolist())]
        ok &= (m["sha256"] == hashlib.sha256(raw).hexdigest() and m["docs"] == len(texts[i]) == len(off)
               and m["tokens"] == len(a) and m["eot"] == eot and m["doc_format"] == "[eot] + text"
               and m["source_sha256"] == DF._sha256(os.path.join(root, "raw", "%03d_00000.parquet" % i))
               and (a[off] == eot).all() and nb.dtype == np.int32 and off.dtype == np.int64
               and nb.tolist() == [len(t.encode("utf-8")) for t in texts[i]]
               and [tok.decode(x, skip_special_tokens=False) for x in docs] == texts[i]
               and docs == [e.ids for e in tok.encode_batch(texts[i], add_special_tokens=False)]
               and sorted(m) == sorted(["shard", "source", "source_sha256", "docs", "tokens", "bytes", "tokenizer", "eot",
                                        "doc_format", "sha256", "secs"]))
    check("build_shards: 14 parca, fw_tokenize V2 bicimi (bin uint16 [eot] + metin, ofset int64, bayt int32, json en son; "
          "json'da akisin sha256'si); decode(belge) == metin, toplu = tek tek kodlama", ok and len(meta) == 14, logs[-1])

    before = {f: os.path.getmtime(os.path.join(d, f)) for f in os.listdir(d)}
    os.remove(os.path.join(d, "shard_005.json"))
    logs = []
    again = DF.build_shards(root, tokenizer=tok, log=logs.append, batch=4)
    same = open(os.path.join(d, "shard_005.bin"), "rb").read()
    check("build_shards: json'u olan parca ATLANIR (kaldigi yerden); json'u olmayan yeniden uretilir, ayni akis",
          sum("ATLANDI" in l for l in logs) == 13 and again[5]["sha256"] == meta[5]["sha256"]
          and hashlib.sha256(same).hexdigest() == meta[5]["sha256"]
          and all(os.path.getmtime(os.path.join(d, f)) == t for f, t in before.items() if not f.startswith("shard_005")))

    bad = tempfile.mkdtemp()
    shutil.copytree(os.path.join(root, "raw"), os.path.join(bad, "raw"))
    man = json.load(open(os.path.join(bad, "raw", "manifest.json")))
    man["files"][0]["sha256"] = "0" * 64
    json.dump(man, open(os.path.join(bad, "raw", "manifest.json"), "w"))
    try:
        DF.build_shards(bad, shards=[0], tokenizer=tok, log=lambda s: None)
        refused = False
    except AssertionError:
        refused = True
    shutil.rmtree(bad)
    check("build_shards: ham dosyanin sha256'si manifest'le tutmazsa DURUR", refused)


def t_load(root, tok, texts):
    data = load(root)
    eot = data["eot"]
    valid_texts = texts[DF.VALID_SHARD][::STRIDE]
    train_texts = [t for i in range(13) for t in texts[i]] + [t for k, t in enumerate(texts[DF.VALID_SHARD]) if k % STRIDE]
    want = []
    for t in train_texts:
        want += [eot] + tok.encode(t, add_special_tokens=False).ids
    starts = np.concatenate([[0], np.cumsum([len(tok.encode(t, add_special_tokens=False).ids) + 1
                                             for t in train_texts])[:-1]])
    vdocs = [DF.valid_doc(data, k) for k in range(len(data["valid_starts"]))]
    check("load: train = parca 000-012 + 013'un valid disi belgeleri, sirayla ([eot] + metin); belge baslangiclari; valid "
          "= 013'un her %d. belgesi, egitimde yok; sinav alt kumesi sirali; metin icindeki eot sayilir" % STRIDE,
          data["train"].tolist() == want and data["train_starts"].tolist() == starts.tolist()
          and [tok.decode(v[1:], skip_special_tokens=False) for v in vdocs] == valid_texts
          and data["valid_bytes"].tolist() == [len(t.encode("utf-8")) for t in valid_texts]
          and not any(t in train_texts for t in valid_texts) and len(data["exam"]) == EXAM
          and (np.diff(data["exam"]) > 0).all() and data["shards"] == list(range(14))
          and data["counts"]["eot_in_text"] == 1 and data["counts"]["split_docs"] > 0,
          "%d token, valid %d, bolunen belge %d" % (len(want), len(vdocs), data["counts"]["split_docs"]))

    part = load(root, train_tokens=len(data["train"]) // 3)
    again = load(root)
    check("load: train_tokens'e yetene kadar parca (sirayla); iz tekrar okumada ayni, SEQ_LEN ve valid adimiyla degisir",
          part["shards"] == list(range(len(part["shards"]))) and len(part["train"]) >= len(data["train"]) // 3
          and len(part["shards"]) < 14 and part["fingerprint"] != data["fingerprint"]
          and again["fingerprint"] == data["fingerprint"]
          and load(root, seq_len=32)["fingerprint"] != data["fingerprint"]
          and DF.load(root, seq_len=SEQ, valid_stride=5, exam_docs=EXAM, log=lambda s: None)["fingerprint"]
          != data["fingerprint"], "iz %s" % data["fingerprint"])

    early = load(root, train_tokens=10, valid_shard=1)
    vt = texts[1][::STRIDE]
    check("load: valid_shard (ON KOSU, tokenize bitmeden): valid o parcanin her %d. belgesi; egitim shard 000'dan, valid "
          "parcasi en sonda ve valid'siz; iz farkli" % STRIDE,
          early["shards"] == [0] and [tok.decode(DF.valid_doc(early, k)[1:], skip_special_tokens=False)
                                      for k in range(len(early["valid_starts"]))] == vt
          and early["fingerprint"] != part["fingerprint"]
          and load(root, valid_shard=1)["shards"][-1] == 1)
    return data


def t_fingerprint(root, tok):
    d = os.path.join(root, "gpt2")
    refused = []
    for name, spoil in (
            ("akis", lambda: _poke(os.path.join(d, "shard_003.bin"), 7)),
            ("ofset", lambda: _save_changed(os.path.join(d, "shard_004_offsets.npy"), lambda a: a + (np.arange(len(a)) == 2))),
            ("bayt", lambda: _save_changed(os.path.join(d, "shard_001_bytes.npy"), lambda a: a + (np.arange(len(a)) == 0))),
            ("tokenizer", lambda: make_tokenizer(["other words"] * 3).save(os.path.join(d, "tokenizer.json")))):
        keep = {f: open(os.path.join(d, f), "rb").read() for f in os.listdir(d)}
        spoil()
        try:
            load(root)
            refused.append((name, False))
        except AssertionError as e:
            refused.append((name, "DURDU" in str(e) or "tutmuyor" in str(e) or "degil" in str(e)))
        for f, b in keep.items():
            open(os.path.join(d, f), "wb").write(b)
    os.remove(os.path.join(d, "shard_013.json"))
    try:
        load(root)
        refused.append(("valid parcasi", False))
    except AssertionError as e:
        refused.append(("valid parcasi", "013" in str(e)))
    open(os.path.join(d, "shard_013.json"), "wb").write(keep["shard_013.json"])
    check("iz: akista tek token, ofset, bayt dizisi ya da tokenizer.json degisince load DURUR; valid parcasi yoksa DURUR",
          all(ok for _, ok in refused), str(refused))


def _poke(path, at):
    b = bytearray(open(path, "rb").read())
    b[2 * at] ^= 1
    open(path, "wb").write(bytes(b))


def _save_changed(path, f):
    a = np.load(path)
    np.save(path, f(a).astype(a.dtype))


def t_windows(data):
    T, a, eot = data["seq_len"], data["train"], data["eot"]
    plan, it = DF.pack(data, 0, 0), data["items"]
    ids, mask, pos = DF.windows(data, plan, np.arange(plan["windows"]))
    starts = data["train_starts"]
    L = np.append(starts[1:], len(a)) - starts
    hits = np.zeros(len(a), dtype=np.int64)                # akis indeksi basina hedef olma sayisi
    closes = np.zeros(len(starts), dtype=np.int64)          # belge basina kapanis eot'u hedefi
    ok_rows, fill_ok = True, True
    for b in range(plan["windows"]):
        at = 0
        for i in plan["item"][plan["ptr"][b]:plan["ptr"][b + 1]].tolist():
            s0, n, size = int(it["start"][i]), int(it["length"][i]), int(it["size"][i])
            want = a[s0:s0 + n].tolist() + ([eot] if it["ends"][i] else [])
            ok_rows &= (ids[b, at:at + size].tolist() == want and pos[b, at:at + size].tolist() == list(range(size))
                        and not bool(mask[b, at]) and bool(mask[b, at + 1:at + size].all()))
            hits[s0 + 1:s0 + n] += 1
            closes[it["doc"][i]] += int(it["ends"][i])
            at += size
        fill_ok &= at <= T + 1 and not bool(mask[b, at:].any())
    nontarget = np.zeros(len(a), dtype=bool)
    nontarget[starts] = True
    whole = L <= T
    split_whole = [d for d in np.flatnonzero(whole).tolist() if (it["doc"] == d).sum() != 1]
    check("best-fit: pencere = parcalarin token'lari (+ kapanis eot'u), konum parca basinda 0, parcanin ilk token'i hedef "
          "degil; her metin token'i tam bir kez hedef, her belgenin kapanis eot'u bir kez; kalan yer dolgu (mask disi)",
          ok_rows and fill_ok and (hits[~nontarget] == 1).all() and (hits[nontarget] == 0).all() and (closes == 1).all(),
          "%d pencere, %d parca, dolgu %%%.1f" % (plan["windows"], len(it["size"]), 100 * plan["padding"]))
    check("best-fit: baglama sigan ([eot] + metin <= T) hicbir belge bolunmez; uzun belge T + 1'lik parcalar (1 ortusme); "
          "kutu sayisi >= toplam / (T + 1), dolgu payi = 1 - toplam / (kutu x (T + 1))",
          not split_whole and whole.sum() < len(L) and it["size"].max() <= T + 1
          and plan["windows"] >= -(-int(it["size"].sum()) // (T + 1))
          and abs(plan["padding"] - (1 - it["size"].sum() / (plan["windows"] * (T + 1)))) < 1e-12,
          "sigan %d / %d belge" % (whole.sum(), len(L)))

    p1 = DF.pack(data, 1, 0)
    boxes1 = [sorted(p1["item"][p1["ptr"][b]:p1["ptr"][b + 1]].tolist()) for b in range(p1["windows"])]
    again = DF.pack(data, 0, 0)
    check("pack: tohumlu ve deterministik (ayni tohum ayni plan); baska tohumda kutu sayisi ayni, yerlesim / sira farkli",
          p1["windows"] == plan["windows"] and np.array_equal(again["item"], plan["item"])
          and (not np.array_equal(p1["order"], plan["order"])
               or boxes1 != [sorted(plan["item"][plan["ptr"][b]:plan["ptr"][b + 1]].tolist()) for b in range(p1["windows"])]))

    draw, draw2 = DF.batches(data, 4, seed=3), DF.batches(data, 4, seed=3)
    per = DF.pack(data, 3, 0)["windows"] // 4
    seen = torch.cat([draw(s)[0] for s in range(per)])
    b5 = [t.clone() for t in draw(per + 5)]
    same = all(torch.equal(x, y) for x, y in zip(draw2(per + 5), b5)) and all(
        torch.equal(x, y) for x, y in zip(draw(2), draw2(2)))
    check("batches: (ids, mask, document_positions), (batch, T + 1); ayni step ayni parca (cagri sirasindan bagimsiz, "
          "surdurme); epok icinde pencere tekrari yok; epoklar farkli sira",
          same and len({tuple(r.tolist()) for r in seen}) == per * 4 and b5[0].shape == (4, SEQ + 1)
          and not torch.equal(draw(0)[0], draw(per)[0]) and b5[2].dtype == torch.long)


def _uniform():
    """Butun skorlari esit model (bpb paydasi)."""
    class U(torch.nn.Module):
        def __init__(self, V):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
            self.V = V

        def logits(self, ids, document_positions=None):
            return torch.zeros(*ids.shape, self.V, dtype=torch.float64) + self.w
    return U


def t_exam(data, tok):
    V, eot = len(data["vocab"]), data["eot"]
    docs = np.arange(len(data["valid_starts"]))
    w = DF.exam_windows(data, docs)
    units = [DF.valid_doc(data, k) for k in docs]
    got = {}
    for r in range(len(w["ids"])):
        for t in range(1, SEQ + 1):
            if w["mask"][r, t]:
                got.setdefault(int(w["doc"][r, t - 1]), []).append(int(w["ids"][r, t]))
    long_docs = sum(len(u) + 1 > SEQ + 1 for u in units)
    check("exam_windows: belge belge -- her belgenin hedefleri metni + kapanis eot'u, tam bir kez; sigan belge bolunmez "
          "(konum 0'dan), sigmayan kendi pencerelerinde; konum = gorulen anahtar - 1",
          all(got[k] == units[k][1:] + [eot] for k in docs) and long_docs > 0
          and all(int(w["document_positions"][r, 0]) == 0 for r in range(len(w["ids"]))),
          "%d belge (%d uzun), %d pencere" % (len(docs), long_docs, len(w["ids"])))

    torch.manual_seed(0)
    model = M.BlockModel(V, **dict(TINY, t_max=SEQ)).double()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.2 * torch.randn(p.shape, generator=torch.Generator().manual_seed(p.numel()), dtype=torch.float64))
    e = EF.exam(model, data, docs)
    nll_sum, hits, n = 0.0, 0, 0
    with torch.no_grad():                                        # belge tek basina, pencere boyunda parcalarla
        for u in units:
            ext = u + [eot]
            for j in range(0, len(ext) - 1, SEQ):
                piece = torch.tensor([ext[j:j + SEQ + 1]])
                z = model.logits(piece[:, :-1])[0]
                y = piece[0, 1:]
                nll_sum += float(torch.nn.functional.cross_entropy(z, y, reduction="sum"))
                hits += int((z.argmax(-1) == y).sum())
                n += len(y)
    check("exam: paketli sinav = her belge tek basina (float64; nll toplami, accuracy, hedef sayisi); bantlar toplami = hepsi",
          abs(e["nll"] * e["n"] - nll_sum) < 1e-8 and abs(e["accuracy"] - hits / n) < 1e-12 and e["n"] == n
          and sum(b["targets"] for b in e["bands"]) == n and len(e["nll_by_frequency"]) == len(EF.FREQUENCY_BANDS) + 1
          and 0 <= e["eos_ok"] <= 1 and e["docs"] == len(docs),
          "fark %.1e  n %d" % (abs(e["nll"] * e["n"] - nll_sum), n))

    U = _uniform()(V)
    u = EF.exam(U, data, docs)
    tb = DF.token_bytes(data["vocab"])
    story = sum(len(x) - 1 for x in units)
    n_bytes = int(data["valid_bytes"][docs].sum())
    check("bpb: payda belgelerin UTF-8 bayti (= hedef token'larin bayti, token_bytes); tekduze modelde = metin hedefi x log2 V "
          "/ bayt; eot dahil: + belge sayisi; bantlarin bayti toplami = hepsi",
          abs(u["bits_per_byte"] - story * math.log2(V) / n_bytes) < 1e-9
          and abs(u["bits_per_byte_eos"] - (story + len(docs)) * math.log2(V) / (n_bytes + len(docs))) < 1e-9
          and sum(int(tb[x[1:]].sum()) for x in units) == n_bytes and sum(b["bytes"] for b in u["bands"]) == n_bytes
          and abs(u["bands"][0]["bits_per_byte"] - u["bits_per_byte"]) < 1e-9 and tb[eot] == 0,
          "bpb %.4f, bayt %d" % (u["bits_per_byte"], n_bytes))

    lg = EF.exam_long(model, data, max_tokens=100)
    lengths = [len(x) for x in units]
    want = sum(min(L, 101) - 1 for L in lengths if L > SEQ) + sum(1 for L in lengths if SEQ < L <= 100)
    check("exam_long: SEQ_LEN'den uzun valid belgeleri tek basina, en cok max_tokens girdi; belge sigarsa kapanis eot'u",
          lg["docs"] == sum(L > SEQ for L in lengths) and lg["n"] == want and lg["max_tokens"] == 100
          and len(lg["bands"]) == len(EF.LONG_BANDS), "%d belge, %d hedef" % (lg["docs"], lg["n"]))

    rep = EF._repeats([[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 3, 4, 5, 6, 7, 8, 9, 9]], [False])
    loop = EF._repeats([[0]], [[5, 6, 7, 8, 9, 10, 11, 12] * 2], [True])
    check("_repeats: tekrar8 = 8'lisi (bu token'da biten) istemde ya da devamda once gecmis token payi; dongu devamin kendi "
          "icinde tekrar eden 8'lisi; farkli4",
          rep["repeat8"] == round(1 / 10, 4) and rep["loop"] == 0 and loop["loop"] == 1 and loop["repeat8"] == round(1 / 16, 4)
          and loop["ended"] == 1 and loop["distinct4"] == round(8 / 13, 4), "%s %s" % (rep, loop))

    real = EF.continuation_repeats(None, data, real=True)
    fit = EF.fitting_docs(data, docs, 8)
    halves, reals = EF.doc_prompts(data, fit)
    written = EF.texts(model, data, halves, reals=reals)
    check("sinav istemleri: baglama sigan belgeler (fitting_docs), istem ilk yari, uretim belgenin kalani + 1 (kapanis) -- "
          "metnin tamami okunur; continuation_repeats gercek devamla (model None: yalniz gercek)",
          set(real) == {"real"} and real["real"]["docs"] == len(EF.fitting_docs(data, data["exam"])[:EF.REPEAT_DOCS])
          and 0 < len(fit) < len(docs) and all(len(units[k]) <= SEQ for k in fit)
          and all(h + r == units[k] for h, r, k in zip(halves, reals, fit))
          and all(len(w["ids"]) <= len(r) + 1 and w["real"] for w, r in zip(written, reals)),
          "%d / %d belge sigar" % (len(fit), len(docs)))


def t_real_gpt2():
    """Istege bagli: gercek gpt2 sozlugunde bayt tablosu ve ornek belgelerde bicim."""
    if not (os.path.exists(REAL_TOKENIZER) and os.path.exists(REAL_SAMPLE)):
        print("  gercek gpt2 / sample.parquet yok: ATLANDI")
        return
    tok = Tokenizer.from_file(REAL_TOKENIZER)
    vocab = [tok.id_to_token(i) for i in range(tok.get_vocab_size(with_added_tokens=True))]
    vocab[tok.token_to_id(DF.EOT)] = DF.EOS_TOKEN
    tb = DF.token_bytes(vocab)
    texts = pq.ParquetFile(REAL_SAMPLE).read_row_group(0, columns=["text"]).column(0).to_pylist()[:200]
    ids = [e.ids for e in tok.encode_batch(texts, add_special_tokens=False)]
    check("gercek gpt2 (50.257): token_bytes toplami = UTF-8 bayt (200 FineWeb-Edu belgesi); decode kayipsiz; eot 50256",
          len(vocab) == 50257 and tok.token_to_id(DF.EOT) == 50256
          and all(int(tb[x].sum()) == len(t.encode("utf-8")) for x, t in zip(ids, texts))
          and all(tok.decode(x) == t for x, t in zip(ids, texts)))


def t_colab(data, root):
    tmp = tempfile.mkdtemp()
    out = tmp + "/r"
    work = dict(targets=0, keys=0)
    f = C._counting(DF.batches(data, 2), work)
    ids, mask, pos = f(0)
    check("_counting: hedef token ve belge ici attention anahtari (hedef t, konum + 1 anahtar gorur) elle",
          work == dict(targets=int(mask[:, 1:].sum()), keys=int(sum(int(pos[r, t]) + 1 for r in range(2) for t in range(SEQ)
                                                                     if mask[r, t + 1]))))
    p = C.plan(data, token_budget=6 * 4 * SEQ, tokens_per_step=4 * SEQ, batch_size=2, schedule="coherence")
    p3 = C.plan(data, token_budget=10, tokens_per_step=3 * SEQ, batch_size=1, schedule="coherence")
    p1 = C.plan(data, tokens_per_step=4 * SEQ, batch_size=4)
    check("plan: parca = token hedefi / (batch x SEQ_LEN), coherence'ta cifte yuvarlanir; adim = butce / adimin token'i; "
          "butcesiz 1 epok", p["micro_batches"] == 2 and p["rows_per_step"] == 4 and p["steps"] == 6
          and p3["micro_batches"] == 4 and p3["steps"] == 1 and p1["steps"] == DF.pack(data, 0, 0)["windows"] // 4 and p1["epochs"] == 1.0
          and p1["windows"] == DF.pack(data, 0, 0)["windows"] and 0 <= p1["padding"] < 1,
          "%s %s" % (p, p3))

    V, d, units = 50257, 80, 32
    bm = M.BlockModel(V, **dict(C.MODEL_KW, d=d, units=units, t_max=SEQ))
    N, Ld = C.CS._model_flops(bm)
    check("_model_flops, varsayilan kosu bicimi (6 blok x 2 = 12 tur, 20 head, tur 1 FactUnits'siz, tur basina FactUnits): "
          "N = 12 x 4 d^2 + 11 x 3 d units + V d, L d = 12 d; RUN_NAME d1280; peak_lr = 0,01 sqrt(384 / d)",
          (N, Ld) == (12 * 4 * d * d + 11 * 3 * d * units + V * d, 12 * d) and C.RUN_NAME == "fineweb_modely_6x2_d1280_gpt2_s0"
          and C.MODEL_KW["d"] == 1280 and C.peak_lr(1280) == 0.0055 and C.peak_lr(1024) == 0.0061 and C.peak_lr(384) == 0.01
          and "lr" not in C.RECIPE and C.TOKENS_PER_STEP == 64 * 8192, "N %d" % N)

    kw = dict(token_budget=3 * 4 * SEQ, tokens_per_step=4 * SEQ, batch_size=2, every=1, device="cpu", save_every=1,
              model_kw=TINY, schedule="coherence", final_cooldown_shape="log", weight_ema=0.9)
    run = C.start("TEST", data, out, **kw)
    run["thread"].join(900)
    files = sorted(os.listdir(out))
    cfg = json.load(open(out + "/config.json"))
    ex = json.load(open(out + "/exams.json", encoding="utf-8"))
    fin = json.load(open(out + "/final.json", encoding="utf-8")) if "final.json" in files else {}
    log = open(out + "/log.txt", encoding="utf-8").read()
    check("colab_fineweb: CPU'da start (3 adim, 2 parca, coherence + log inis, EMA) -> checkpoint her adim, config (veri, "
          "paketleme, iz, baglam, adim ve token hesabi, model_kw acik), sinav bantlari, istem metni, sonda valid / alt kume / "
          "uzun / istemler / belge devamlari / tekrar, ortalama model",
          run["done"] and not run["error"]
          and files == ["checkpoint_t000001.pt", "checkpoint_t000002.pt", "checkpoint_t000003.pt", "config.json",
                        "exams.json", "final.json", "log.txt", "model.pt", "model_weight_ema.pt"]
          and cfg["dataset"] == "fineweb-edu" and cfg["packing"] == "best_fit" and "padding" in cfg and cfg["fingerprint"] == data["fingerprint"]
          and cfg["seq_len"] == SEQ and cfg["micro_batches"] == 2 and cfg["steps"] == 3 and cfg["step_tokens"] == 4 * SEQ
          and cfg["lr"] == C.peak_lr(16) and cfg["schedule"] == "coherence" and cfg["model_kw"]["units"] == 16
          and cfg["model_kw"]["t_max"] == SEQ and cfg["model_kw"]["attention_log_scale"] is True
          and cfg["model_kw"]["packed_attention"] == M.PACKED_ATTENTION
          and cfg["model_kw"]["rope_base"] == M.rope_base_for(16 // 2, SEQ)       # "auto" config'e sayi olarak
          and [e["step"] for e in ex] == [0, 1, 2, 3] and len(ex[0]["bands"]) == len(EF.BANDS)
          and len(ex[0]["texts"]) == EF.PROBE_PROMPTS and "repeats" in ex[0] and "weight_ema" in ex[1]
          and all("tokens_per_sec" in e for e in ex[1:]) and fin.get("valid", {}).get("docs") == len(data["valid_starts"])
          and fin["subset"]["docs"] == EXAM and len(fin["prompts"]) == len(EF.PROMPTS)
          and len(fin["docs"]) == len(EF.fitting_docs(data, data["exam"])[:C.FINAL_DOCS]) > 0
          and set(fin["repeats"]) == {"greedy", "sampled", "real"} and "long" in fin and "weight_ema" in fin
          and "SON valid" in log and "SON uzun" in log and "tekrar" in log,
          str(run["error"] or files) + " docs %d repeats %s" % (len(fin.get("docs", [])), sorted(fin.get("repeats", {}))))

    first = torch.load(out + "/model.pt")
    os.remove(out + "/checkpoint_t000003.pt")
    run2 = C.start("TEST", data, out, resume=True, **kw)
    run2["thread"].join(900)
    second = torch.load(out + "/model.pt")
    try:
        C.start("TEST2", data, out, resume=True, **dict(kw, batch_size=1))
        refused = False
    except RuntimeError:
        refused = True
    check("colab_fineweb, surdurme: son paketten devam, model = kesintisiz kosununki (bit duzeyinde); ayar farkliysa reddeder",
          run2["done"] and not run2["error"] and all(torch.equal(first[k], second[k]) for k in first) and refused,
          str(run2["error"]))

    import internals_y as I
    done, errors = [], []
    for measure, extra in (("trace", ["--stories", "2"]), ("attention_stats", ["--stories", "3"]),
                           ("unit_usage", ["--stories", "3"]), ("ablate", ["--stories", "3", "--cases", "none", "A2", "F2", "C"]),
                           ("point_drift", [])):             # valid: fixture'da varsayilan adimla (50) tek belge
        try:
            I._main([out, measure, "--data", root] + extra)
            done.append(measure)
        except BaseException as e:                         # SystemExit dahil
            errors.append("%s: %r" % (measure, e))
    torch.set_num_threads(1)
    written = sorted(f.split("_")[0] for f in os.listdir(out + "/internals") if f.endswith(".json"))
    stories = I._fineweb(cfg, root)["stories"](3)[1]
    check("internals_y FineWeb: --data FineWeb klasoru -> valid belgeleri tek tek ([eot] + metin + [eot], en cok 2.048), "
          "sinav alt kumesi once; trace / attention_stats / unit_usage / ablate / point_drift (sayim parca .bin'lerinden) "
          "kosar (tuned_lens ayri belge ister: fixture'da tek valid belge)",
          not errors and len(done) == 5 and {"trace", "attention", "unit", "ablate", "point"} <= set(written)
          and all(x[0] == data["eot"] and x[-1] == data["eot"] for x in stories),
          "; ".join(errors)[:300] or str(written))

    run3 = C.start("TEST3", data, tmp + "/s", **dict(kw, token_budget=None, steps=500))
    C.stop()
    run3["thread"].join(900)
    run4 = C.start("TEST4", data, tmp + "/t", stop_at=2, **dict(kw, token_budget=None, save_every=None))
    run4["thread"].join(900)
    ex4 = json.load(open(tmp + "/t/exams.json", encoding="utf-8"))
    check("colab_fineweb: stop -> model.pt yazilir, iplik cikar; stop_at 2 -> 1 epokluk ayarlarla 2. adimin sinavindan sonra "
          "durur (A100 TEST); pulse hatasiz",
          not run3["thread"].is_alive() and "DURDURULDU" in run3["lines"][-1] and os.path.exists(run3["out"] + "/model.pt")
          and "DURDURULDU" in run4["lines"][-1] and [e["step"] for e in ex4] == [0, 1, 2]
          and json.load(open(tmp + "/t/config.json"))["steps"] == DF.pack(data, 0, 0)["windows"] // 4, run3["lines"][-1])
    prof = C.profile_sizes(data, tmp + "/p", candidates=(("a", TINY), ("b", dict(TINY, d=24))), stop_at=2, every=1,
                           batch_size=2, token_budget=1e6, max_hours=1e9, min_ratio=0, device="cpu",
                           tokens_per_step=4 * SEQ)
    prof["thread"].join(900)
    res = prof["results"]
    check("profile_sizes (A100 TEST): adaylar sirayla, gercek ayarlarla stop_at'e kadar; satir basina token/sn, ms/adim, "
          "govde parametresi, token / govde, saat, CU; SECIM kurali gecen en buyuk govde",
          prof["done"] and not prof["error"] and [r["label"] for r in res] == ["a", "b"] and all(r["passes"] for r in res)
          and res[1]["body_params"] > res[0]["body_params"] and abs(res[0]["cu"] - 5.4 * res[0]["hours"]) < 1e-9
          and abs(res[0]["hours"] - 1e6 / res[0]["tokens_per_sec"] / 3600) < 1e-9 and prof["lines"][-1] == "SECIM: b"
          and os.path.exists(tmp + "/p/profile_sizes.json") and os.path.exists(tmp + "/p/a/config.json"),
          str(prof["error"] or prof["lines"][-3:]))
    C.pulse(1)
    shutil.rmtree(tmp)


if __name__ == "__main__":
    print("tests (train_fineweb)")
    root, tok, texts = fixture()
    try:
        t_build(root, tok, texts)
        data = t_load(root, tok, texts)
        t_fingerprint(root, tok)
        t_windows(data)
        t_exam(data, tok)
        t_real_gpt2()
        t_colab(data, root)
    finally:
        shutil.rmtree(root)
    print("\n%d GECTI   %d KALDI" % (sum(RESULTS), len(RESULTS) - sum(RESULTS)))
    sys.exit(0 if all(RESULTS) else 1)
