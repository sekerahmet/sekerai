"""train_sentence -- SentenceTransformer egitimi (ajan: sentence.py; z: sentence_z.py).  Hikaye hikaye: her cumle, onceki
cumlelerinin z'leri verilerek kelime kelime tahmin edilir (teacher forcing); son cumleden sonra hikaye sonu (EOS).
Ulke (data/countries) ve SS (make_ss_sentences --stories dosyalari) ayni yoldan: batch = hikayeler; her cumlenin z'si
batch'te bir kez cihazda hesaplanir (meaning + grammar + konum, formul); ornekler [BOS][z_1..z_k] kelimeler cihazda
vektorel kurulur.  Olcu (sinav hikayeleri, yalniz sonraki 1 cumle; kullanici: "Safece 1 cümle sonrasına bakacak";
"normal transformer ölçüsü ne ise ona bakalım"):
    kayip, ppl, dogruluk (acc butun hedefler, acc_word yalniz kelime), bits per byte (SS: model_y exam_simplestories ile
                    ayni olcu ve ayni 1000 hikaye: model_y'nin sinav kumesi exam_stories.npy, sha256 dogrulanir)
    cumle bitirme   son kelimeden sonra END dogru mu; cumle icinde yanlis END
    hikaye bitirme  son cumleden sonra EOS dogru mu; hikaye icinde yanlis EOS
    z kullanimi     z'ler baska hikayeden (karisik) ve sifir z ile kayip
    ulke            ayrica uretim: gecerli sonraki cumle (valid_next), dogru olgu, tekrar, ayni olgu (repeat_fact), END,
                    dongu + goz icin ornekler
    kapali dongu    (story_generation, son epok, --prompts) model kendi cumlesinin z'siyle hikayeyi surdurur; metin + olcu
checkpoint.pt her CHECKPOINT_SECS'te ve epok sonunda (kaldigi hikayeden surdurur, olcu gecmisi dahil); sonda agent.pt ve
results.json.  Bitmis kosu --resume ile cagrilirsa durur, dosyalara dokunmaz.

    python train_sentence.py [--data countries|simplestories] [--root SS klasoru] [--epochs 4] [--d 64] [--layers 4]
                             [--heads 4] [--batch 16 (hikaye)] [--lr LR_D64*64/d] [--z 512] [--z_roles 1|0]
                             [--bag_channel 0|1] [--train_z true|zero] [--prompts istem.json] [--device cpu|cuda] [--out klasor] [--resume 1]
    python train_sentence.py --eval_only agent.pt [--data ...] [--root ...] [--out klasor]    (egitim yok: sinav + kapali
                             dongu; model boyu, z_roles, train_z ajandan; sonuc --out/eval_<zaman>.json)
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from sentence import BOS, PAD, WORD, ZTOK, SentenceTransformer, output_loss
from sentence_z import build_keys, decode_z, encode_z, keys_to

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_Z = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "grammar"))
from train_grammar import _no_power_throttling  # noqa: E402
RUNS = "G:/Drive'ım/model_z/runs/"
AGENTS = {"countries": (RUNS + "meaning_table_countries_w5_d128_mix_sub_e4_20261004_203221/agent.pt",
                        RUNS + "grammar_countries_d64_cosine_lr0.003_b64_20261004_133746/agent.pt"),
          "simplestories": (RUNS + "meaning_ss_full_w5_d256_b2048_k0_t3e-05_e3_20261005_160420/agent.pt",
                            RUNS + "grammar_ssfull_d256_cosine_lr0.001_b1024_20261003_211559/agent.pt")}
# lr olcutu: Adam'da gizli katman lr'si ~ 1 / genislik (muP mantigi; GPT-3 d 768 -> 6e-4).  Olculdu (ulke, d 64, batch 160
# cumle, 4 epok, 5 Ekim): lr 1e-3 / 3e-3 / 7e-3 -> sinav kaybi 1,006 / 0,444 / 0,402, gecerli sonraki cumle 0,386 / 0,777 /
# 0,872, z kullanimi 3. / 2. / 1. epokta basliyor; kararsizlik yok.  --lr verilmezse LR_D64 * 64 / d (d 256 -> ~1,75e-3,
# orada yeniden olculur).
LR_D64 = 7e-3
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
CLIP = 1.0
WARMUP = 0.05           # adimlarin bu kadari dogrusal isinma, sonra cosine
PROGRESS_SECS = 60      # epok icinde ara satir araligi
CHECKPOINT_SECS = 600
EXAM_STORIES = 1000     # SS: sinavin sabit alt kumesi (model_y exam_simplestories.exam_rows ile ayni kural, ayni hikayeler)
SHOW = 8


class Stories:
    """Hikayeler (kelimeler uc uca + cumle ve hikaye sinirlari) -> hikaye batch'leri (ids (B, T, W), mask) cihazda.  SS:
    make_ss_sentences --stories dosyalari (<prefix>_ids, _sentence_offsets, _offsets); ulke: from_lists.  Arsivdeki context
    egitiminden (arsiv/model_z_20261005); kesme yok (kullanici: "maks kelime diye birşey yok"), <unk> da kelimedir."""

    def __init__(self, flat, sent_off, story_off, device):
        self.flat = torch.as_tensor(np.asarray(flat), dtype=torch.int32, device=device)    # SS 580M kelime: int32 2,3 GB
        self.sent_off = torch.as_tensor(np.asarray(sent_off), dtype=torch.long, device=device)
        self.story_off = torch.as_tensor(np.asarray(story_off), dtype=torch.long, device=device)
        self.n = len(self.story_off) - 1
        self.longest = int((self.sent_off[1:] - self.sent_off[:-1]).max())

    @classmethod
    def from_files(cls, root, prefix, device):
        return cls(*[np.load(os.path.join(root, prefix + s)) for s in ("_ids.npy", "_sentence_offsets.npy", "_offsets.npy")],
                   device)

    @classmethod
    def from_lists(cls, stories, device):
        """stories: hikaye -> cumle -> kelime kimlikleri."""
        sents = [s for st in stories for s in st]
        return cls([w for s in sents for w in s], np.r_[0, np.cumsum([len(s) for s in sents])],
                   np.r_[0, np.cumsum([len(st) for st in stories])], device)

    def batch(self, rows):
        first, last = self.story_off[rows], self.story_off[rows + 1]
        T = int((last - first).max())
        sent = first[:, None] + torch.arange(T, device=first.device)[None]
        has = sent < last[:, None]
        sent = sent.clamp(max=len(self.sent_off) - 2)
        a, b = self.sent_off[sent], self.sent_off[sent + 1]
        W = int(((b - a) * has).max())
        pos = a[..., None] + torch.arange(W, device=a.device)
        mask = (pos < b[..., None]) & has[..., None]
        ids = torch.where(mask, self.flat[pos.clamp(max=len(self.flat) - 1)].long(), torch.zeros_like(pos))
        return ids, mask


def story_z(keys, ids, mask, mode="true"):
    """Batch'in butun cumlelerinin z'si bir kez (cihazda) -> (B, T, z).  mode karisik: hikaye b'nin k. z'si hikaye b+1'in
    k. (yoksa son) cumlesinden; sifir: z yok."""
    B, T, W = ids.shape
    zgrid = torch.zeros(B, T, keys["z"], device=ids.device)
    if mode == "zero":
        return zgrid
    has = mask.any(-1)
    b, t = has.nonzero(as_tuple=True)
    for c in range(0, len(b), 8192):
        zgrid[b[c:c + 8192], t[c:c + 8192]] = encode_z(keys, ids[b[c:c + 8192], t[c:c + 8192]], mask[b[c:c + 8192], t[c:c + 8192]])
    if mode == "shuffle":
        donor = (torch.arange(B, device=ids.device) + 1) % B
        last = has.sum(1)[donor] - 1
        zgrid = zgrid[donor[:, None], torch.minimum(torch.arange(T, device=ids.device)[None], last[:, None])]
    return zgrid


def examples(ids, mask, zgrid, END, EOS):
    """Hikaye batch'i -> ornekler (her cumle + hikaye sonu): [BOS][z_1..z_k] w_1..w_L.  Satir k (son ozet) w_1'i ya da (k = n)
    EOS'u, satir k+i w_{i+1}'i, son kelime END'i tahmin eder.  -> dict: kind, tok, zvec, pos, target, kategori maskeleri."""
    dev = ids.device
    B, T, W = ids.shape
    has = mask.any(-1)
    n = has.sum(1)
    L = mask.sum(-1)
    cnt = n + 1
    E = int(cnt.sum())
    ex_b = torch.repeat_interleave(torch.arange(B, device=dev), cnt)
    ex_k = torch.arange(E, device=dev) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
    eos_ex = ex_k == n[ex_b]
    kc = ex_k.clamp(max=T - 1)
    Lk = torch.where(eos_ex, torch.zeros_like(ex_k), L[ex_b, kc])
    Tx = int((1 + ex_k + Lk).max())
    j = torch.arange(Tx, device=dev)[None]
    k_, L_ = ex_k[:, None], Lk[:, None]
    kind = torch.where(j == 0, BOS, torch.where(j <= k_, ZTOK, torch.where(j <= k_ + L_, WORD, PAD)))
    tok = torch.where(kind == WORD, ids[ex_b[:, None], kc[:, None], (j - k_ - 1).clamp(0, W - 1)], 0)
    zvec = zgrid[ex_b[:, None], (j - 1).clamp(0, T - 1)] * (kind == ZTOK)[..., None]
    word_next = ids[ex_b[:, None], kc[:, None], (j - k_).clamp(0, W - 1)]
    first = j == k_
    mid = (j > k_) & (j < k_ + L_)
    last = (j == k_ + L_) & (L_ > 0)
    target = torch.full((E, Tx), -100, device=dev)
    target = torch.where(first & eos_ex[:, None], EOS, target)
    target = torch.where((first & ~eos_ex[:, None]) | mid, word_next, target)
    target = torch.where(last, END, target)
    return dict(kind=kind, tok=tok, zvec=zvec, pos=j.expand(E, -1), target=target, first=first & ~eos_ex[:, None],
                eos=first & eos_ex[:, None], end=last, mid=mid, story=ex_b, k=ex_k)


@torch.no_grad()
def exam_scores(model, st, rows, keys, batch, mode="true", nbytes=None):
    """Sinav hikayeleri -> kayip, ppl, dogruluk (acc butun hedefler; acc_word yalniz kelime, END / EOS haric), cumle /
    hikaye bitirme; nbytes verilirse bits per byte (model_y exam_simplestories gibi: esas EOS haric, bits_per_byte_eos EOS
    dahil).  bpb'de Z bosluk / satir sonunu odemez (~0,02 bit/bayt; belge/model_z_temel/14 2.4)."""
    sums = dict(nll=0.0, nll_eos=0.0, n=0, hit=0, end_hit=0, end_n=0, false_end=0, word_n=0, word_hit=0, eos_hit=0,
                eos_n=0, false_eos=0, first_n=0)
    bounds = list(range(0, len(rows), batch))
    if len(bounds) > 1 and len(rows) - bounds[-1] == 1:                 # karisik z: tek hikayelik parca kendiyle karisir
        bounds = bounds[:-1]
    for i, c in enumerate(bounds):
        stop = bounds[i + 1] if i + 1 < len(bounds) else len(rows)
        ids, mask = st.batch(rows[c:stop])
        ex = examples(ids, mask, story_z(keys, ids, mask, mode), model.END, model.EOS)
        h = model.hidden(ex["kind"], ex["tok"], ex["zvec"], ex["pos"], None)
        keep = ex["target"] >= 0
        hk, tgt = h[keep], ex["target"][keep]
        nll = torch.empty(len(tgt), device=h.device)
        pred = torch.empty(len(tgt), dtype=torch.long, device=h.device)
        for r in range(0, len(tgt), 4096):                                # logit tablosu parca parca (bellek)
            lg = (hk[r:r + 4096] @ model.E.weight.T).float()
            nll[r:r + 4096] = F.cross_entropy(lg, tgt[r:r + 4096], reduction="none")
            pred[r:r + 4096] = lg.argmax(-1)
        is_eos = ex["eos"][keep]
        sums["nll"] += float(nll[~is_eos].sum())
        sums["nll_eos"] += float(nll[is_eos].sum())
        sums["n"] += len(tgt)
        sums["hit"] += int((pred == tgt).sum())
        for name in ("end", "eos", "first", "mid"):
            ex[name] = ex[name][keep]
        sums["end_hit"] += int((pred[ex["end"]] == model.END).sum())
        sums["end_n"] += int(ex["end"].sum())
        words = ex["first"] | ex["mid"]
        sums["false_end"] += int((pred[words] == model.END).sum())
        sums["word_n"] += int(words.sum())
        sums["word_hit"] += int((pred[words] == tgt[words]).sum())
        sums["eos_hit"] += int((pred[ex["eos"]] == model.EOS).sum())
        sums["eos_n"] += int(ex["eos"].sum())
        sums["false_eos"] += int((pred[ex["first"]] == model.EOS).sum())
        sums["first_n"] += int(ex["first"].sum())
    nll = (sums["nll"] + sums["nll_eos"]) / sums["n"]
    out = dict(loss=round(nll, 4), ppl=round(math.exp(nll), 2), acc=round(sums["hit"] / sums["n"], 4),
               acc_word=round(sums["word_hit"] / sums["word_n"], 4),
               end_ok=round(sums["end_hit"] / sums["end_n"], 4), false_end=round(sums["false_end"] / sums["word_n"], 4),
               eos_ok=round(sums["eos_hit"] / sums["eos_n"], 4), false_eos=round(sums["false_eos"] / sums["first_n"], 4))
    if nbytes is not None:
        out["bits_per_byte"] = round(sums["nll"] / math.log(2) / nbytes, 4)
        out["bits_per_byte_eos"] = round((sums["nll"] + sums["nll_eos"]) / math.log(2) / (nbytes + len(rows)), 4)
    return out


def load_countries():
    """-> sozluk, hikayeler (split, ulke, cumleler, olgular, valid_next kumeleri, fact_of: ulkenin cumlesi -> olgusu)."""
    folder = os.path.join(MODEL_Z, "data", "countries")
    vocab = json.load(open(os.path.join(folder, "country_vocab.json"), encoding="utf-8"))
    ix = {w: i for i, w in enumerate(vocab)}
    ids = lambda s: [ix.get(w, 0) for w in s]  # noqa: E731
    fact_of = {}
    for line in open(os.path.join(folder, "country_sentences.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        fact_of.setdefault(r["country"], {})[tuple(ids(r["words"]))] = r["fact"]
    stories = []
    for line in open(os.path.join(folder, "country_stories.jsonl"), encoding="utf-8"):
        s = json.loads(line)
        stories.append(dict(split=s["split"], country=ids(s["country"].split()), sents=[ids(x) for x in s["sentences"]],
                            facts=s["facts"], fact_of=fact_of[s["country"]],
                            valid=[{tuple(ids(x)) for x in step} for step in s.get("valid_next", [])]))
    return vocab, stories


def pad_sentences(sents, device="cpu"):
    L = max(len(s) for s in sents)
    ids = torch.zeros(len(sents), L, dtype=torch.long, device=device)
    mask = torch.zeros(len(sents), L, dtype=torch.bool, device=device)
    for i, s in enumerate(sents):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, :len(s)] = True
    return ids, mask


@torch.no_grad()
def country_generation(model, exam, keys, device, show):
    """Ulke: gercek onceki cumlelerin z'leriyle sonraki cumle (acgozlu) -> gecerli, dogru olgu, tekrar, ayni olgu
    (repeat_fact: soylenmis olgunun herhangi bir kalibi), ... + ornekler.  z_ok: uretilen cumlelerden geri acilan pay."""
    vocab = model.vocab
    text = lambda s: " ".join(vocab[w] for w in s)  # noqa: E731
    longest = max(len(s) for st in exam for s in st["sents"])
    zs = [encode_z(keys, *pad_sentences(st["sents"], device)) for st in exam]
    by_k = {}
    for si, st in enumerate(exam):
        for k in range(1, len(st["sents"])):
            by_k.setdefault(k, []).append(si)
    res = dict(n=0, valid=0, true_fact=0, repeat=0, repeat_fact=0, eos=0, has_country=0, ended=0, loop=0, exact=0)
    shown, outs = [], []
    for k, items in sorted(by_k.items()):
        for c in range(0, len(items), 512):
            part = items[c:c + 512]
            for si, o in zip(part, model.generate(torch.stack([zs[si][:k] for si in part]), longest + 3)):
                st = exam[si]
                if o == [model.EOS]:
                    res["n"] += 1
                    res["eos"] += 1
                    continue
                prev = {tuple(s) for s in st["sents"][:k]}
                facts = set().union(*st["valid"]) | {tuple(s) for s in st["sents"]}
                name = st["country"]
                ok = tuple(o) in st["valid"][k - 1]
                res["n"] += 1
                res["valid"] += ok
                res["true_fact"] += tuple(o) in facts
                res["repeat"] += tuple(o) in prev
                res["repeat_fact"] += st["fact_of"].get(tuple(o)) in set(st["facts"][:k])
                res["has_country"] += any(o[i:i + len(name)] == name for i in range(len(o)))
                res["ended"] += len(o) <= longest + 2
                res["loop"] += any(o[i:i + 3] == o[i + 3:i + 6] for i in range(max(0, len(o) - 5)))
                res["exact"] += o == st["sents"][k]
                if 0 < len(o) <= longest:
                    outs.append(o)
                if len(shown) < show and si % 37 == 0 and k >= 3:
                    tag = "GECERLI" if ok else ("TEKRAR" if tuple(o) in prev else
                                                ("DOGRU OLGU" if tuple(o) in facts else "-"))
                    shown.append(([text(s) for s in st["sents"][:k]], text(o), tag))
    n = res.pop("n")
    out = {k: round(v / n, 4) for k, v in res.items()}
    out["n"] = n
    if outs:                                                            # z kontrolu: uretilen cumle geri aciliyor mu
        back = decode_z(keys, encode_z(keys, *pad_sentences(outs, device)), max_len=longest)
        out["z_ok"] = round(sum(b == o for b, o in zip(back, outs)) / len(outs), 4)
    for prev, o, tag in shown:
        print("   onceki : " + "\n            ".join(prev) + "\n   model  : %s   [%s]" % (o, tag), flush=True)
    return out


@torch.no_grad()
def story_generation(model, keys, prompts, max_sentences=80, max_words=64, generator=None, mode="true"):
    """Kapali dongu: istem cumlelerinin z'leri, sonra model her cumleden sonra KENDI cumlesinin z'siyle devam eder; EOS'ta
    ya da max_sentences'te durur (80 > SS sinavinin en uzun hikayesi 77 cumle; 64 kelime > sinav cumlelerinin %99,99'u).
    generator None: acgozlu; verilirse ornekleme (sentence.generate).  mode "zero": model z yerine sifir gorur (--train_z
    zero ile egitilmis model; story_z gibi).  prompts: istem -> cumle -> kelime kimlikleri.
    -> (uretilen hikayeler, olculer): eos_rate (EOS ile bitti), sentences (istem basina uretilen cumle), sentence_repeat
    (uretilen cumlelerden o hikayede -- istem + once uretilenler -- birebir daha once gecmis olanlarin orani), loop (cumle
    icinde ardisik 3'lu tekrar), no_end (max_words'e dayandi), empty (bos cumle), unk_rate, z_ok (END'le biten cumlelerden
    geri acilan pay; kesilen cumle olculmez: SIC maliyeti ~ cumle x boy^2 x V x z)."""
    dev = next(model.parameters()).device
    max_words = min(max_words, len(keys["signs"]) - 1)                  # encode_z en uzun cumleyi asamaz
    stories, ended, finished = [[] for _ in prompts], [False] * len(prompts), []
    z_of = (lambda ids, mask: encode_z(keys, ids, mask)) if mode == "true" else \
        (lambda ids, mask: torch.zeros(len(ids), keys["z"], device=dev))
    res = dict(sentences=0, sentence_repeat=0, loop=0, no_end=0, empty=0, words=0, unk=0)
    by_k = {}
    for i, p in enumerate(prompts):
        by_k.setdefault(len(p), []).append(i)
    for items in by_k.values():                                         # generate: ayni turda ayni sayida z
        zs = torch.stack([z_of(*pad_sentences(prompts[i], dev)) for i in items])
        for _ in range(max_sentences):
            new = []
            for i, o in zip(items, model.generate(zs, max_words, generator)):
                if not ended[i] and o == [model.EOS]:
                    ended[i] = True
                if ended[i]:
                    new.append([])
                    continue
                cut = o[:max_words]
                res["sentences"] += 1
                res["sentence_repeat"] += cut in prompts[i] + stories[i]
                res["loop"] += any(o[j:j + 3] == o[j + 3:j + 6] for j in range(max(0, len(o) - 5)))
                res["no_end"] += len(o) > max_words
                res["empty"] += not cut
                res["words"] += len(cut)
                res["unk"] += sum(w == 0 for w in cut)
                stories[i].append(cut)
                if cut and len(o) <= max_words:
                    finished.append(cut)
                new.append(cut)
            if all(ended[i] for i in items):
                break
            zs = torch.cat([zs, z_of(*pad_sentences(new, dev))[:, None]], 1)
    z_ok = 0                                    # yalniz END'le biten cumleler; boya gore (SIC ~ cumle x (L+1)^2 x V x z)
    for L in sorted({len(s) for s in finished}):
        part = [s for s in finished if len(s) == L]
        z_ok += sum(b == s for b, s in zip(decode_z(keys, encode_z(keys, *pad_sentences(part, dev)), max_len=L), part))
    n = max(res["sentences"], 1)
    out = dict(stories=len(prompts), eos_rate=round(sum(ended) / len(prompts), 4),
               sentences=round(res["sentences"] / len(prompts), 2), unk_rate=round(res["unk"] / max(res["words"], 1), 4),
               z_ok=round(z_ok / len(finished), 4) if finished else None,
               **{k: round(res[k] / n, 4) for k in ("sentence_repeat", "loop", "no_end", "empty")})
    return stories, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="countries", choices=("countries", "simplestories"))
    ap.add_argument("--root", default="G:/Drive'ım/model_z/simplestories_full", help="SS: ss_vocab.json ve ss_*story_*.npy")
    ap.add_argument("--meaning", default=None, help="verilmezse AGENTS[--data]")
    ap.add_argument("--grammar", default=None, help="verilmezse AGENTS[--data]")
    ap.add_argument("--z", type=int, default=512)
    ap.add_argument("--z_roles", type=int, default=1, help="1: z'de grammar rol terimi (bugunku varsayilan); 0: yalniz konum")
    ap.add_argument("--bag_channel", type=int, default=0, help="1: z'ye torba kanali (z boyu 2 kat; sentence_z, belge 15)")
    ap.add_argument("--train_z", default="true", choices=("true", "zero"), help="zero: z'siz kontrol (z yerine sifir)")
    ap.add_argument("--decode", default="greedy,sample", help="kapali dongu: greedy (argmax), sample (modelin kendi "
                    "dagilimi, sicaklik 1, tohum --seed); virgulle ikisi")
    ap.add_argument("--prompts", default=None, help="kapali dongu istemleri (json: stories = sinav hikayesi indeksi, "
                    "sentences = istem cumle sayilari); SS'te verilmezse transformer_baseline/ss_prompts.json (varsa)")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16, help="hikaye (ulke 16 ~ 160 cumle)")
    ap.add_argument("--lr", type=float, default=None, help="verilmezse LR_D64 * 64 / d (olcut yukarida)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile", type=int, default=1, help="1: torch.compile (yalniz cuda)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", type=int, default=0)
    ap.add_argument("--eval_only", default=None, help="agent.pt: egitim yok, yalniz sinav + kapali dongu (results.json'a "
                    "dokunmaz; --out/eval_<zaman>.json)")
    args = ap.parse_args(argv)
    args.meaning = args.meaning or AGENTS[args.data][0]
    args.grammar = args.grammar or AGENTS[args.data][1]
    if args.lr is None:
        args.lr = LR_D64 * 64 / args.d
    for name in ("meaning", "grammar"):                                  # z bu dosyalara bagli: iz (kural 9)
        args.__dict__[name + "_sha256"] = hashlib.sha256(open(vars(args)[name], "rb").read()).hexdigest()
    agent = None
    if args.eval_only:                                  # olculecek ajan: boyu ve z ayari ajandan (eski ajan: rol acik, z true)
        agent = torch.load(args.eval_only, map_location="cpu", weights_only=False)
        a = dict(dict(z_roles=1, train_z="true", bag_channel=0), **agent["args"])
        assert a["data"] == args.data, "ajan %s verisiyle egitilmis" % a["data"]
        for name in ("meaning", "grammar"):             # iz varsa iz, yoksa kosu klasoru adi (Colab / yerel yol farkli)
            same = a.get(name + "_sha256") == vars(args)[name + "_sha256"] if name + "_sha256" in a else \
                os.path.basename(os.path.dirname(a[name])) == os.path.basename(os.path.dirname(vars(args)[name]))
            assert same, "%s ajani egitimdekinden farkli: %s" % (name, a[name])
        for k in ("z", "d", "layers", "heads", "z_roles", "train_z", "bag_channel", "seed", "batch"):   # batch: karisik z esi
            vars(args)[k] = a[k]
    default_prompts = os.path.join(os.path.dirname(MODEL_Z), "transformer_baseline", "ss_prompts.json")
    if args.prompts is None and args.data == "simplestories" and os.path.exists(default_prompts):
        args.prompts = default_prompts
    print("kapali dongu istemleri:", args.prompts or "YOK (story_generation atlanir)", flush=True)
    torch.manual_seed(args.seed)
    cuda = args.device.startswith("cuda")
    if not cuda and os.name == "nt":
        print("guc kisitlamasi (EcoQoS) kapali:", _no_power_throttling(), flush=True)
    if cuda:
        torch.set_float32_matmul_precision("high")
    t0 = time.time()

    exam_lists, nbytes = None, None
    if args.data == "countries":
        vocab, stories = load_countries()
        train = Stories.from_lists([s["sents"] for s in stories if s["split"] == "train"], args.device)
        exam_lists = [s for s in stories if s["split"] == "exam"]
        exam = Stories.from_lists([s["sents"] for s in exam_lists], args.device)
        exam_rows = torch.arange(exam.n, device=args.device)
        longest = max(train.longest, exam.longest)
    else:
        vocab = json.load(open(os.path.join(args.root, "ss_vocab.json"), encoding="utf-8"))
        exam = Stories.from_files(args.root, "ss_exam_story", args.device)
        if agent is None:
            train = Stories.from_files(args.root, "ss_story", args.device)
            longest = max(train.longest, exam.longest)
        else:                                           # egitim verisi yuklenmez; anahtarlar egitimdeki boyla (P ona bagli)
            longest = max(int(np.diff(np.load(os.path.join(args.root, "ss_story_sentence_offsets.npy"))).max()),
                          exam.longest)
        # sinav: model_y'nin kumesi (sizintisiz ve sigan valid hikayeleri; valid sirasi = Z sinav hikayesi sirasi) ve
        # exam_simplestories.exam_rows'un kurali -> ayni 1000 hikaye
        args.exam_set = os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(args.root))), "simplestories",
                                     "exam_stories.npy")
        allowed = np.load(args.exam_set)
        args.exam_set_sha256 = hashlib.sha256(np.ascontiguousarray(allowed)).hexdigest()
        saved = json.load(open(args.exam_set[:-4] + ".json", encoding="utf-8"))["sha256"]
        assert args.exam_set_sha256 == saved, "sinav kumesi izi tutmuyor: %s" % args.exam_set
        assert allowed.max() < exam.n, "sinav kumesi Z sinav hikayeleriyle hizali degil"
        pick = np.sort(allowed[np.random.default_rng(0).permutation(len(allowed))[:EXAM_STORIES]])
        exam_rows = torch.as_tensor(pick, device=args.device)
        story_bytes = np.load(os.path.join(args.root, "ss_exam_story_bytes.npy"))      # make_ss_sentences --stories
        assert len(story_bytes) == exam.n, "ss_exam_story_bytes.npy sinav hikayeleriyle hizali degil"
        nbytes = int(story_bytes[pick].sum())
    keys = keys_to(build_keys(args.meaning, args.grammar, args.z, longest, roles=bool(args.z_roles),
                              bag_channel=bool(args.bag_channel)), args.device)
    assert keys["vocab"] == vocab, "meaning / grammar sozlugu veriyle ayni degil"

    model = SentenceTransformer(vocab, keys["z"], args.d, args.layers, args.heads).to(args.device)   # z: konum (+ torba)
    if cuda and args.compile:
        for block in model.blocks:
            block.compile(dynamic=True)
        model.loss_fn = torch.compile(output_loss, dynamic=True)          # cikis carpimi + kayip tek grafikte
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=WEIGHT_DECAY), dict(params=no_decay, weight_decay=0.0)],
                            lr=args.lr, betas=BETAS)
    n = train.n if agent is None else 0
    steps_per = -(-n // args.batch)
    total_steps = steps_per * args.epochs
    first, start, ckpt = 1, 0, os.path.join(args.out, "checkpoint.pt") if args.out else None
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    history = []
    if args.resume:
        pack = torch.load(ckpt, map_location=args.device, weights_only=False)
        keys_ = ("data", "z", "d", "layers", "heads", "batch", "lr", "epochs", "seed", "meaning", "grammar", "z_roles",
                 "train_z", "bag_channel", "meaning_sha256", "grammar_sha256", "exam_set_sha256")
        old = dict(dict(z_roles=1, train_z="true", bag_channel=0), **pack["args"])     # eski checkpoint; iz yoksa atlanir
        diff = {k: (old[k], vars(args).get(k)) for k in keys_ if k in old and old[k] != vars(args).get(k)}
        assert not diff, "surdurme ayari checkpoint'ten farkli (checkpoint, simdi): %s" % diff
        if pack["epoch"] > args.epochs:
            print("KOSU ZATEN BITMIS (checkpoint epok %d > %d): hicbir dosyaya dokunulmadi" % (pack["epoch"], args.epochs),
                  flush=True)
            return None
        model.load_state_dict(pack["state"])
        opt.load_state_dict(pack["opt"])
        first, start, history = pack["epoch"], pack["next"], pack.get("history", [])
        print("SURDURULDU: epok %d, hikaye %d'den" % (first, start), flush=True)

    def save(epoch, nxt):
        if ckpt:
            torch.save(dict(vocab=vocab, state=model.state_dict(), opt=opt.state_dict(), epoch=epoch, next=nxt,
                            history=history, args=vars(args)), ckpt + ".part")
            os.replace(ckpt + ".part", ckpt)

    def measure(last):
        """Sinav (egitimdeki z, karisik, sifir) + ulke uretimi + son epokta kapali dongu -> (e, s, z0, gen, story)."""
        model.eval()
        with torch.autocast(args.device.split(":")[0], dtype=torch.bfloat16, enabled=cuda):
            e = exam_scores(model, exam, exam_rows, keys, args.batch * 4, args.train_z, nbytes)   # ana okuma: egitimdeki z
            s = exam_scores(model, exam, exam_rows, keys, args.batch * 4, "shuffle")
            z0 = exam_scores(model, exam, exam_rows, keys, args.batch * 4, "zero")
            gen = country_generation(model, exam_lists, keys, args.device, SHOW if last else 0) if exam_lists else None
            story = None
            if last and args.prompts:                                     # kapali dongu: metin (kural 12) + olcu
                spec = json.load(open(args.prompts, encoding="utf-8"))
                p_ids, p_mask = exam.batch(torch.as_tensor(spec["stories"], device=args.device))
                sents = [[r[m].tolist() for r, m in zip(p_ids[b], p_mask[b]) if m.any()] for b in range(len(p_ids))]
                prompts = [st[:c] for st in sents for c in spec["sentences"]]
                text = lambda s_: " ".join(vocab[w] for w in s_)  # noqa: E731
                story, written = {}, {}
                for mode in args.decode.split(","):
                    gen_ = torch.Generator(args.device).manual_seed(args.seed) if mode == "sample" else None
                    written[mode], story[mode] = story_generation(model, keys, prompts, generator=gen_, mode=args.train_z)
                    story[mode]["texts"] = [[text(x) for x in w] for w in written[mode]]
                story["prompts"] = [[text(x) for x in p] for p in prompts]
                for j, p in enumerate(story["prompts"]):                  # decode'lar yan yana (goz)
                    print("   istem  : " + "\n            ".join(p), flush=True)
                    for mode in written:
                        print("   %-7s: " % mode + "\n            ".join(story[mode]["texts"][j]), flush=True)
        model.train()
        if story:
            for mode in args.decode.split(","):
                print("kapali dongu %s: %s" % (mode, {k: v for k, v in story[mode].items() if k != "texts"}), flush=True)
        return e, s, z0, gen, story

    if agent is not None:                               # yalniz olcum: egitim, checkpoint, agent.pt, results.json YOK
        model.load_state_dict(agent["state"])
        print("YALNIZ OLCUM: %s | sinav %d hikaye (olcu %d%s) | z %d (meaning+konum%s), egitimde z %s | d %d, katman %d" % (
            args.eval_only, exam.n, len(exam_rows), ", kume sha256 %s" % args.exam_set_sha256[:12] if "exam_set" in vars(args)
            else "", keys["z"], ("+grammar rol" if args.z_roles else "") + ("+torba" if args.bag_channel else ""),
            args.train_z, args.d, args.layers), flush=True)
        e, s, z0, gen, story = measure(True)
        print("sinav (z %s) %s | karisik z kayip %.3f, sifir z kayip %.3f%s" % (
            args.train_z, e, s["loss"], z0["loss"], " | uretim %s" % gen if gen else ""), flush=True)
        if args.out:
            path = os.path.join(args.out, "eval_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
            json.dump(dict(args=vars(args), exam=e, exam_z=args.train_z, exam_shuffled_z=s, exam_zero_z=z0,
                           generation=gen, story_generation=story), open(path, "w"), indent=1)
            print("kaydedildi:", path, flush=True)
        return model
    print("veri %s: egitim %d hikaye (%d cumle, en uzun %d kelime), sinav %d hikaye (olcu %d%s) | z %d (meaning+konum%s), "
          "egitimde z %s | d %d, katman %d, head %d, %d parametre | batch %d hikaye, lr %g, %d adim | cihaz %s compile %s "
          "(%.0f sn hazirlik)"
          % (args.data, n, len(train.sent_off) - 1, train.longest, exam.n, len(exam_rows),
             ", kume %s sha256 %s" % (args.exam_set, args.exam_set_sha256[:12]) if "exam_set" in vars(args) else "",
             keys["z"], ("+grammar rol" if args.z_roles else "") + ("+torba" if args.bag_channel else ""), args.train_z,
             args.d, args.layers, args.heads,
             sum(p.numel() for p in model.parameters()), args.batch, args.lr, total_steps, args.device,
             bool(cuda and args.compile), time.time() - t0), flush=True)
    for epoch in range(first, args.epochs + 1):
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed + epoch)).to(args.device)
        total, count, sents_done = torch.zeros((), device=args.device), 0, 0
        t_epoch = t_shown = t_saved = time.time()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        b0 = start if epoch == first else 0
        for b in range(b0, n, args.batch):
            if time.time() - t_shown > PROGRESS_SECS:                     # ara satir: kayip yalniz burada okunur
                t_shown, el = time.time(), time.time() - t_epoch
                left = (n - b) // args.batch + (args.epochs - epoch) * steps_per
                print("  epok %d hikaye %d / %d (%%%.0f)  kayip %.3f  %.0f cumle/sn  kalan ~%.0f dk (butun egitim)%s" % (
                    epoch, b, n, 100 * b / n, total.item() / max(count, 1), sents_done / el, left * el / max(count, 1) / 60,
                    ", GPU tepe %.1f GB" % (torch.cuda.max_memory_allocated() / 1e9) if cuda else ""), flush=True)
            if time.time() - t_saved > CHECKPOINT_SECS:
                t_saved = time.time()
                save(epoch, b)
            step = (epoch - 1) * steps_per + b // args.batch
            warm = max(1, int(WARMUP * total_steps))
            lr = args.lr * (step + 1) / warm if step < warm else \
                args.lr * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total_steps - warm)))
            for group in opt.param_groups:
                group["lr"] = lr
            ids, mask = train.batch(perm[b:b + args.batch])
            ex = examples(ids, mask, story_z(keys, ids, mask, args.train_z), model.END, model.EOS)
            sents_done += int(mask.any(-1).sum())
            with torch.autocast(args.device.split(":")[0], dtype=torch.bfloat16, enabled=cuda):
                loss = model(ex["kind"], ex["tok"], ex["zvec"], ex["pos"], None, ex["target"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            total += loss.detach()
            count += 1
        t_train = time.time() - t_epoch
        e, s, z0, gen, story = measure(epoch == args.epochs)
        print("epok %d  kayip %.3f | sinav (z %s) %s | karisik z kayip %.3f, sifir z kayip %.3f%s  (egitim %.0f sn, olcum %.0f sn)" % (
            epoch, total.item() / count, args.train_z, e, s["loss"], z0["loss"], " | uretim %s" % gen if gen else "", t_train,
            time.time() - t_epoch - t_train), flush=True)
        history.append(dict(epoch=epoch, loss=total.item() / count, exam=e, exam_z=args.train_z, exam_shuffled_z=s,
                            exam_zero_z=z0, generation=gen, story_generation=story))
        save(epoch + 1, 0)
    if args.out:
        torch.save(dict(vocab=vocab, state=model.state_dict(), args=vars(args)), os.path.join(args.out, "agent.pt"))
        json.dump(dict(args=vars(args), history=history), open(os.path.join(args.out, "results.json"), "w"), indent=1)
        print("kaydedildi:", args.out, flush=True)
    return model


if __name__ == "__main__":
    main()
