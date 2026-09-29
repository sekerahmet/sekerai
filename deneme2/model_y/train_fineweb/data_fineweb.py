# -*- coding: utf-8 -*-
"""data_fineweb -- FineWeb-Edu sample-10BT, gpt2 tokenizer, best-fit paketleme + belge maskesi.  Drive'daki parcalar OKUNUR; iz yeniden
hesaplanip parca json'lariyla karsilastirilir, tutmazsa DURUR (kural 9).  Uretim tek seferlik: build_shards (Colab'daki
fw_tokenize.py V2 ile ayni bicim).

Drive <root>/ (Colab: /content/drive/MyDrive/fineweb):
    raw/manifest.json, raw/*.parquet      kaynak (repo, revision, files: path / size / sha256)
    gpt2/tokenizer.json                   tokenizer (save_tokenizer bir kez; iz ve gidis-donus denetimi bununla)
    gpt2/shard_NNN.bin                    uint16 akis: her belge [eot] + encode(metin), art arda
    gpt2/shard_NNN_offsets.npy            int64, belge baslangiclari (eot'un indeksi)
    gpt2/shard_NNN_bytes.npy              int32, belge basina UTF-8 bayt (bpb paydasi; eot'un bayti yok)
    gpt2/shard_NNN.json                   shard, source, source_sha256, docs, tokens, bytes, tokenizer, eot, doc_format,
                                          sha256 (akisin), secs -- EN SON yazilir: varligi parcanin tamam oldugunu gosterir
Ayirma (kural: VALID_RULE): valid = shard_013'un indeks % VALID_STRIDE == 0 belgeleri (row group'lar tek dump'tan; ilk N
belge 1-2 dump olurdu, adimli secim 183 row group'a yayilir); egitimde YOK.  exam: valid'in sabit alt kumesi (EXAM_DOCS).
train: shard 000'dan sirayla (013 en sonda, valid'siz).  Paketleme (PACKING_RULE; kullanici, 30 Eylul: "ya modeli
değiştirelim ya da daha küçük makele seçelim yarım kalmasın"): baglama sigan belge BOLUNMEZ; pencereye (T + 1 token) butun
yerlesir (best-fit decreasing, epok basina tohumla); uzun belge T + 1'lik parcalara (1 ortusme); kalan yer dolgu (mask
disi).  Her belgenin metni ve kapanis eot'u tam bir kez hedef.  document_positions: token'in penceredeki belge parcasindaki
konumu (parca basinda 0) -- model bununla attention'i, Canon'u ve log-n olcegini belge icinde tutar.
Sinav belge belge (exam_windows): butun belgeler pencereye bolunmeden, sigmayan belge kendi pencerelerinde.
Iz (fingerprint): tokenizer, kullanilan parcalarin json sha256'si (akis yeniden hesaplanip karsilastirilir) + ofset / bayt
izleri, ayirma kurali, SEQ_LEN, paketleme kurali.

    python data_fineweb.py build_shards <root> [parca ...]     tek seferlik; Colab, arka planda
"""
import hashlib
import json
import os
import sys
import time

import torch  # pyarrow'dan ONCE (Windows DLL sirasi)
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train_simplestories"))

import data_simplestories as DS  # noqa: E402  (tokenizer yukleme, encode / decode, bits_per_byte)

SEQ_LEN = 8192          # baglam: model girdisi; pencere SEQ_LEN + 1 token (not.md §9)
TAG = "gpt2"
EOT = "<|endoftext|>"
EOS_TOKEN = DS.EOS_TOKEN # sozlukte eot'un yazimi (sinav ve uretim kodu eos'u bu adla bulur)
VALID_SHARD = 13
VALID_STRIDE = 50       # shard_013: 182.101 belge -> 3.643 valid belge (~3,7M token)
EXAM_DOCS = 256         # her sinavin alt kumesi (~260k hedef token)
CHECK_DOCS = 16         # parca basina tokenizer gidis-donus denetimi (ilk belgeler)
VALID_RULE = "valid: shard_%03d belge indeksi %% %d == 0; exam: valid'in tohum 0 permutasyonundan ilk %d, sirali"
PACKING = "best_fit"
PACKING_RULE = ("best-fit decreasing (Ding 2024): belge [eot] + metin + kapanis eot; L <= T bolunmez, L > T parcalar T + 1, "
                "adim T; kutu T + 1; boy buyukten kucuge, esitlik (seed, epok) permutasyonu; en az yer kalan sigan kutu; "
                "kutu sirasi (seed, epok, 1) permutasyonu")


def _sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _short(a):
    return hashlib.sha256(np.ascontiguousarray(a)).hexdigest()[:12]


def _shard_paths(d, i):
    base = os.path.join(d, "shard_%03d" % i)
    return dict(bin=base + ".bin", offsets=base + "_offsets.npy", bytes=base + "_bytes.npy", json=base + ".json")


def fingerprint(parts):
    """Parca izleri -> birlesik iz (sirali 'ad:iz' dizisinin sha256'si, ilk 12 hane)."""
    return hashlib.sha256("|".join("%s:%s" % kv for kv in sorted(parts.items())).encode()).hexdigest()[:12]


# --- tek seferlik uretim (fw_tokenize.py V2 ile ayni bicim)

def save_tokenizer(root, tokenizer=None):
    """Bir kez: <root>/gpt2/tokenizer.json (Tokenizer.from_pretrained("gpt2"), parcalarin tokenizer'i).  Varsa dokunmaz.
    -> sha256."""
    path = os.path.join(root, TAG, "tokenizer.json")
    if not os.path.exists(path):
        from tokenizers import Tokenizer
        os.makedirs(os.path.dirname(path), exist_ok=True)
        (tokenizer or Tokenizer.from_pretrained("gpt2")).save(path + ".part")
        os.replace(path + ".part", path)
    return _sha256(path)


def build_shards(root, shards=None, tokenizer=None, log=print, batch=10000):
    """Tek seferlik (kural 9), Colab: raw/manifest.json'daki parquet i -> gpt2/shard_iii.* (bicim bu dosyanin basinda).
    shard_iii.json varsa ATLANIR (kaldigi parcadan devam); ham dosyanin sha256'si manifest'le dogrulanir.  Her belge
    [eot] + tokenizer.encode(metin, add_special_tokens=False).ids (toplu kodlama, butun cekirdekler).  tokenizer: None ->
    Tokenizer.from_pretrained("gpt2") (ag).  -> {parca: json}"""
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    tok = tokenizer or Tokenizer.from_pretrained("gpt2")
    eot = tok.token_to_id(EOT)
    assert eot is not None and tok.get_vocab_size(with_added_tokens=True) <= 65536, "eot yok ya da sozluk uint16'ya sigmiyor"
    manifest = json.load(open(os.path.join(root, "raw", "manifest.json"), encoding="utf-8"))
    out_dir = os.path.join(root, TAG)
    os.makedirs(out_dir, exist_ok=True)
    done = {}
    for i, f in enumerate(manifest["files"]):
        if shards is not None and i not in shards:
            continue
        paths = _shard_paths(out_dir, i)
        if os.path.exists(paths["json"]):
            log("shard_%03d: var, ATLANDI" % i)
            done[i] = json.load(open(paths["json"], encoding="utf-8"))
            continue
        t0 = time.time()
        src = os.path.join(root, "raw", os.path.basename(f["path"]))
        sha = _sha256(src)
        assert sha == f["sha256"], "%s: sha256 %s, manifest %s -- DURDU" % (src, sha, f["sha256"])
        pieces, lengths, nbytes = [], [], []
        for rb in pq.ParquetFile(src).iter_batches(batch_size=batch, columns=["text"]):
            texts = rb.column(0).to_pylist()
            ids = [e.ids for e in tok.encode_batch_fast(texts, add_special_tokens=False)]
            for x in ids:
                pieces.append(np.array([eot] + x, dtype=np.uint16))
            lengths += [len(x) + 1 for x in ids]
            nbytes += [len(s.encode("utf-8")) for s in texts]
        stream = np.concatenate(pieces)
        offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
        for key, arr in (("bin", stream), ("offsets", offsets), ("bytes", np.array(nbytes, dtype=np.int32))):
            with open(paths[key] + ".part", "wb") as fo:             # dosya nesnesi: np.save ada '.npy' eklemesin
                arr.tofile(fo) if key == "bin" else np.save(fo, arr)
            os.replace(paths[key] + ".part", paths[key])
        meta = dict(shard=i, source=f["path"], source_sha256=f["sha256"], docs=len(lengths), tokens=int(len(stream)),
                    bytes=int(sum(nbytes)), tokenizer="gpt2", eot=eot, doc_format="[eot] + text",
                    sha256=hashlib.sha256(stream.tobytes()).hexdigest(), secs=round(time.time() - t0, 1))
        with open(paths["json"] + ".part", "w", encoding="utf-8") as fo:
            json.dump(meta, fo, indent=1)
        os.replace(paths["json"] + ".part", paths["json"])
        log("shard_%03d: %d belge, %d token, %.0f sn" % (i, meta["docs"], meta["tokens"], meta["secs"]))
        done[i] = meta
    return done


# --- okuma

def _read_shard(d, i, tok, eot, check_docs=CHECK_DOCS):
    """Parca -> (akis, ofsetler, baytlar, json).  Akisin sha256'si json'la; sayimlar, ofsetlerdeki eot, bayt toplami ve
    ilk belgelerde tokenizer gidis-donusu (decode -> bayt sayisi, encode(decode) == id'ler) denetlenir; tutmazsa DURUR."""
    p = _shard_paths(d, i)
    meta = json.load(open(p["json"], encoding="utf-8"))
    raw = open(p["bin"], "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == meta["sha256"], "shard_%03d: akis sha256 %s, json %s -- DURDU (kural 9)" % (i, sha[:12], meta["sha256"][:12])
    stream = np.frombuffer(raw, dtype=np.uint16)
    offsets, nbytes = np.load(p["offsets"]), np.load(p["bytes"])
    assert meta["eot"] == eot and meta["doc_format"] == "[eot] + text", "shard_%03d: eot / bicim farkli" % i
    assert len(stream) == meta["tokens"] and len(offsets) == len(nbytes) == meta["docs"], "shard_%03d: sayimlar" % i
    assert offsets[0] == 0 and (np.diff(offsets) > 0).all() and (stream[offsets] == eot).all() \
        and int(nbytes.sum()) == meta["bytes"], "shard_%03d: belge siniri ya da bayt tutmuyor" % i
    ends = np.append(offsets[1:], len(stream))
    for s, e, b in zip(offsets[:check_docs].tolist(), ends[:check_docs].tolist(), nbytes[:check_docs].tolist()):
        ids = stream[s + 1:e].tolist()
        text = tok.decode(ids, skip_special_tokens=False)
        assert len(text.encode("utf-8")) == b and tok.encode(text, add_special_tokens=False).ids == ids, \
            "shard_%03d: tokenizer.json bu parcanin tokenizer'i degil (belge %d)" % (i, s)
    return stream, offsets, nbytes, meta


def _docs_stream(stream, offsets, keep):
    """Parcadan keep belgeleri: (akis, ofsetler) -- belgeler sirayla, bicim ayni."""
    lengths = np.append(offsets[1:], len(stream)) - offsets
    return stream[np.repeat(keep, lengths)], np.concatenate([[0], np.cumsum(lengths[keep])[:-1]]).astype(np.int64)


def load_valid(root, valid_stride=VALID_STRIDE, exam_docs=EXAM_DOCS, log=print, valid_shard=VALID_SHARD):
    """Drive -> yalniz valid (egitim akisi okunmaz; internals_y ve load'un ilk adimi).  -> vocab, eot, tag, valid (akis),
    valid_starts, valid_bytes, exam, parts (iz parcalari), counts, complete (tamamlanmis parcalar), tokenizer."""
    d = os.path.join(root, TAG)
    tpath = os.path.join(d, "tokenizer.json")
    assert os.path.exists(tpath), "%s yok -- save_tokenizer(root) bir kez (kural 9)" % tpath
    tok, vocab = DS._load(tpath, EOT)
    eot = vocab.index(EOS_TOKEN)
    complete = sorted(int(f[6:9]) for f in os.listdir(d) if f.startswith("shard_") and f.endswith(".json"))
    assert valid_shard in complete, "valid parcasi shard_%03d henuz yok (tokenize bitmedi)" % valid_shard
    parts = dict(tokenizer=_sha256(tpath)[:12], rule=hashlib.sha256((VALID_RULE % (valid_shard, valid_stride, exam_docs))
                                                                     .encode()).hexdigest()[:12])
    stream, offsets, nbytes, meta = _read_shard(d, valid_shard, tok, eot)
    parts["shard_%03d" % valid_shard] = "%s.%s.%s" % (meta["sha256"][:12], _short(offsets), _short(nbytes))
    is_valid = np.arange(len(offsets)) % valid_stride == 0
    valid, valid_starts = _docs_stream(stream, offsets, is_valid)
    exam = np.sort(np.random.default_rng(0).permutation(len(valid_starts))[:exam_docs])
    counts = dict(eot_in_text=int((stream == eot).sum()) - len(offsets), valid_docs=len(valid_starts),
                  valid_tokens=len(valid), valid_bytes=int(nbytes[is_valid].sum()), exam_docs=len(exam),
                  complete_shards=len(complete))
    return dict(vocab=vocab, eot=eot, tag=TAG, valid=valid, valid_starts=valid_starts, valid_bytes=nbytes[is_valid],
                exam=exam, parts=parts, counts=counts, complete=complete, tokenizer=tok,
                valid_shard=(stream, offsets, is_valid))


def load(root, train_tokens=None, seq_len=SEQ_LEN, valid_stride=VALID_STRIDE, exam_docs=EXAM_DOCS, log=print,
         valid_shard=VALID_SHARD):
    """Drive -> veri; uretmez (tokenizer.json ya da tamamlanmis parca yoksa DURUR).  train_tokens: egitim akisina bu kadar
    token yetene kadar parca 000'dan sirayla (None: butun tamamlanmis parcalar; valid parcasi en sonda, valid'siz).  valid
    parcasi sart: valid_shard (varsayilan 013; ON KOSU tokenize bitmeden erken bir parcayla).  -> load_valid'inkiler + seq_len, train (uint16 akis: belgeler [eot] + metin, art arda), train_starts,
    items (best-fit paketlemenin parcalari, _items), shards, counts, fingerprint, fingerprints."""
    data = load_valid(root, valid_stride, exam_docs, log, valid_shard)
    d, eot, tok, parts, counts = os.path.join(root, TAG), data["eot"], data["tokenizer"], data["parts"], data["counts"]
    vs, vo, is_valid = data.pop("valid_shard")
    used, total = [], 0                                    # parcalar ve boylar json'dan: akis bir kez, yerinde yazilir
    for i in [i for i in data["complete"] if i != valid_shard] + [valid_shard]:
        if train_tokens is not None and total >= train_tokens:
            break
        used.append(i)
        total += (len(vs) - len(data["valid"]) if i == valid_shard else
                  json.load(open(_shard_paths(d, i)["json"], encoding="utf-8"))["tokens"])
    assert train_tokens is None or total >= train_tokens, "egitim akisi %d token < train_tokens %d (tamamlanmis parca: %s)" % (
        total, train_tokens, data["complete"])
    train = np.empty(total, dtype=np.uint16)
    starts, at = [], 0
    for i in used:
        if i == valid_shard:
            s, o = _docs_stream(vs, vo, ~is_valid)
        else:
            s, o, nb, meta = _read_shard(d, i, tok, eot)
            parts["shard_%03d" % i] = "%s.%s.%s" % (meta["sha256"][:12], _short(o), _short(nb))
            counts["eot_in_text"] += int((s == eot).sum()) - len(o)
        train[at:at + len(s)] = s
        starts.append(o + at)
        at += len(s)
        del s
    parts.update(seq_len=str(seq_len), packing=hashlib.sha256(PACKING_RULE.encode()).hexdigest()[:12])
    data.update(seq_len=seq_len, train=train, train_starts=np.concatenate(starts), shards=used)
    data["items"] = _items(data)
    counts.update(train_tokens=len(train), train_docs=len(data["train_starts"]), train_shards=len(used),
                  items=len(data["items"]["size"]), split_docs=int((~data["items"]["whole"]).sum()))
    data["fingerprint"], data["fingerprints"] = fingerprint(parts), parts
    log("fineweb %s: parca %s | train %d token, %d belge (%d'i T %d'e sigmiyor: parcalanir), %d parca | valid %d belge, "
        "%d token | sinav %d belge | metin icinde eot %d | iz %s" % (
            TAG, used, counts["train_tokens"], counts["train_docs"], counts["split_docs"], seq_len, counts["items"],
            counts["valid_docs"], counts["valid_tokens"], len(data["exam"]), counts["eot_in_text"], data["fingerprint"]))
    return data


# --- best-fit paketleme (Ding 2024, belge/makaleler/2024/ding2024_fewertruncations.txt)

def _items(data):
    """Egitim belgeleri -> parcalar.  Belge [eot] + metin (akista, L token) + kapanis eot'u (hedef).  L <= T: tek parca,
    BOLUNMEZ.  L > T: T + 1'lik parcalar, T adimla (1 ortusme: parca sinirindaki token bir kez hedef), son parca kalan +
    kapanis.  -> start (akista), length (akistan token), ends (kapanis eot'u eklenir), size = length + ends (<= T + 1),
    whole (belgenin tek parcasi), doc (belge sirasi)."""
    T, starts = data["seq_len"], data["train_starts"]
    L = np.append(starts[1:], len(data["train"])) - starts
    m = np.where(L > T, -(-(L - T) // T), 0)               # belge basina kapanissiz (T + 1'lik) parca sayisi
    doc = np.repeat(np.arange(len(L)), m + 1)
    j = np.arange(len(doc)) - np.repeat(np.cumsum(m + 1) - (m + 1), m + 1)
    ends = j == m[doc]
    length = np.where(ends, L[doc] - j * T, T + 1)
    return dict(start=starts[doc] + j * T, length=length, ends=ends, size=length + ends, whole=(m == 0)[doc], doc=doc)


def _best_fit(sizes, capacity, tie):
    """Best-fit decreasing: parcalar buyukten kucuge (esitlikte tie sirasi), her biri kalan yeri en az olan sigan kutuya,
    yoksa yeni kutu.  -> (kutu, yerlestirme sirasi) parca basina, kutu sayisi."""
    import bisect
    order = np.lexsort((tie, -sizes))
    caps, open_bins = [], {}                               # bos yer -> o kadar yeri olan kutular (yigin)
    box = [0] * len(sizes)
    nb = 0
    size = sizes.tolist()
    for i in order.tolist():
        s = size[i]
        k = bisect.bisect_left(caps, s)
        if k < len(caps):
            c = caps[k]
            stack = open_bins[c]
            b = stack.pop()
            if not stack:
                del open_bins[c]
                caps.pop(k)
        else:
            b, c = nb, capacity
            nb += 1
        box[i] = b
        r = c - s
        if r:
            if r not in open_bins:
                open_bins[r] = []
                bisect.insort(caps, r)
            open_bins[r].append(b)
    rank = np.empty(len(sizes), dtype=np.int64)
    rank[order] = np.arange(len(sizes))
    return np.array(box, dtype=np.int64), rank, nb


_PLANS = {}


def pack(data, seed=0, epoch=0):
    """Epok e'nin paketleme plani (tohum (seed, e); ayni girdi ayni plan): parcalar kutulara (best-fit decreasing, kutu T +
    1), kutularin sirasi karisik.  -> dict(item (kutu kutu parca sirasi), ptr (CSR), order (kutularin epoktaki sirasi),
    windows, padding (dolgu payi))."""
    key = (data["fingerprint"], seed, epoch)
    if key not in _PLANS:
        it, T = data["items"], data["seq_len"]
        box, rank, nb = _best_fit(it["size"], T + 1, np.random.default_rng([seed, epoch]).permutation(len(it["size"])))
        item = np.lexsort((rank, box))
        ptr = np.concatenate([[0], np.cumsum(np.bincount(box, minlength=nb))])
        _PLANS.clear()                                     # yalniz son plan (bellek)
        _PLANS[key] = dict(item=item, ptr=ptr, order=np.random.default_rng([seed, epoch, 1]).permutation(nb), windows=nb,
                           padding=1 - float(it["size"].sum()) / (nb * (T + 1)))
    return _PLANS[key]


def windows(data, plan, boxes):
    """Kutular -> (ids, mask, document_positions), (len(boxes), T + 1).  Parca sirayla: token'lari (+ kapanis eot'u), konum
    0'dan; parcanin ilk token'i hedef degil (onceki parcadan tahmin edilmez), gerisi hedef; kalan yer dolgu (eot, mask disi,
    konum artarak)."""
    T, a, it, eot = data["seq_len"], data["train"], data["items"], data["eot"]
    ids = np.full((len(boxes), T + 1), eot, dtype=np.int64)
    mask = np.zeros((len(boxes), T + 1), dtype=bool)
    pos = np.zeros((len(boxes), T + 1), dtype=np.int64)
    for r, b in enumerate(boxes):
        at = 0
        for i in plan["item"][plan["ptr"][b]:plan["ptr"][b + 1]].tolist():
            s, n, size = int(it["start"][i]), int(it["length"][i]), int(it["size"][i])
            ids[r, at:at + n] = a[s:s + n]
            mask[r, at + 1:at + size] = True
            pos[r, at:at + size] = np.arange(size)
            at += size
        pos[r, at:] = np.arange(1, T + 2 - at) + (pos[r, at - 1] if at else -1)
    return torch.from_numpy(ids), torch.from_numpy(mask), torch.from_numpy(pos)


def batches(data, batch_size, seed=0):
    """Adimin fonksiyonu step -> (ids, mask, document_positions).  Epok e: pack(data, seed, e)'nin kutulari karisik sirayla
    bolunur, artan son parca o epokta kullanilmaz.  Ayni step her zaman ayni parca (surdurme)."""
    n = pack(data, seed, 0)["windows"]                 # kutu sayisi tohumdan bagimsiz (esit boylu parcalar yer degistirir)
    per_epoch = n // batch_size
    assert per_epoch > 0, "batch_size (%d) > pencere (%d)" % (batch_size, n)

    def batch(step):
        epoch, i = divmod(step, per_epoch)
        plan = pack(data, seed, epoch)
        return windows(data, plan, plan["order"][i * batch_size:(i + 1) * batch_size])
    return batch


def valid_doc(data, i):
    """valid belgesi i -> [eot] + metin (id listesi)."""
    s = data["valid_starts"]
    end = s[i + 1] if i + 1 < len(s) else len(data["valid"])
    return data["valid"][s[i]:end].tolist()


def exam_windows(data, docs, seq_len=None):
    """Belge belge sinav pencereleri: docs (valid siralari) butun belgeler olarak, sirayla, bir pencereye sigdikca yan
    yana; her belgenin kapanis hedefi sonraki belgenin eot'u ya da pencere sonuna eklenen eot.  T + 1'e sigmayan belge kendi
    pencerelerinde (1 token ortusme; devam parcasi konum 0'dan: konum = gorulen anahtar - 1).  -> ids, mask,
    document_positions, doc (belge sirasi, dolgu -1), numpy (pencere, T + 1)."""
    T = data["seq_len"] if seq_len is None else seq_len
    eot = data["eot"]
    rows, cur = [], []

    def flush():
        if cur:
            rows.append(([t for _, u in cur for t in u] + [eot], [k for k, u in cur for _ in u] + [cur[-1][0]],
                         [j for _, u in cur for j in range(len(u))] + [len(cur[-1][1])]))
            cur.clear()
    for k in docs:
        unit = valid_doc(data, int(k))
        if len(unit) + 1 > T + 1:                              # kendi pencereleri: [eot] metin [eot], 1 ortusme
            flush()
            ext = unit + [eot]
            for j in range(0, len(ext) - 1, T):
                piece = ext[j:j + T + 1]
                rows.append((piece, [int(k)] * len(piece), list(range(len(piece)))))
            continue
        if sum(len(u) for _, u in cur) + len(unit) + 1 > T + 1:
            flush()
        cur.append((int(k), unit))
    flush()
    n = len(rows)
    out = dict(ids=np.full((n, T + 1), eot, dtype=np.int64), mask=np.zeros((n, T + 1), dtype=bool),
               document_positions=np.zeros((n, T + 1), dtype=np.int64), doc=np.full((n, T + 1), -1, dtype=np.int64))
    for r, (ids, doc, pos) in enumerate(rows):
        L = len(ids)
        out["ids"][r, :L], out["mask"][r, :L], out["doc"][r, :L], out["document_positions"][r, :L] = ids, True, doc, pos
        out["document_positions"][r, L:] = pos[-1] + 1 + np.arange(T + 1 - L)
    return out


def token_bytes(vocab):
    """gpt2 bayt duzeyi BPE: token -> UTF-8 bayt sayisi (sozluk yazimindaki her karakter bir bayt, bytes_to_unicode);
    eot 0.  Belgenin token'larinin toplami metnin baytina esit (bpb bantlari)."""
    table = [c for c in range(ord("!"), ord("~") + 1)] + list(range(ord("\xa1"), ord("\xac") + 1)) + \
        list(range(ord("\xae"), ord("\xff") + 1))
    alphabet = set(chr(c) for c in table) | set(chr(256 + n) for n in range(256 - len(table)))
    out = np.zeros(len(vocab), dtype=np.int64)
    for i, t in enumerate(vocab):
        if t == EOS_TOKEN:
            continue
        assert all(ch in alphabet for ch in t), "bayt duzeyi olmayan token %r" % t
        out[i] = len(t)
    return out


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "build_shards":
        build_shards(sys.argv[2], [int(x) for x in sys.argv[3:]] or None)
    else:
        print(__doc__)
