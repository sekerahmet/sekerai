"""train_sentence -- SentenceTransformer egitimi (ajan: sentence.py; z: sentence_z.py).  Hikaye hikaye: her cumle, onceki
cumlelerinin z'leri verilerek kelime kelime tahmin edilir (teacher forcing); son cumleden sonra hikaye sonu (EOS).
Ulke (data/countries) ve SS (make_ss_sentences --stories dosyalari) ayni yoldan: batch = hikayeler; her cumlenin z'si
batch'te bir kez cihazda hesaplanir (meaning + grammar + konum, formul); ornekler [BOS][z_1..z_k] kelimeler cihazda
vektorel kurulur.  Olcu (sinav hikayeleri, yalniz sonraki 1 cumle; kullanici: "Safece 1 cümle sonrasına bakacak";
"normal transformer ölçüsü ne ise ona bakalım"):
    kayip, ppl, dogruluk, bits per byte (SS: model_y exam_simplestories ile ayni olcu, sinav hikayelerinin baytlari)
    cumle bitirme   son kelimeden sonra END dogru mu; cumle icinde yanlis END
    hikaye bitirme  son cumleden sonra EOS dogru mu; hikaye icinde yanlis EOS
    z kullanimi     z'ler baska hikayeden (karisik) ve sifir z ile kayip
    ulke            ayrica uretim: gecerli sonraki cumle (valid_next), dogru olgu, tekrar, END, dongu + goz icin ornekler
checkpoint.pt her CHECKPOINT_SECS'te ve epok sonunda (kaldigi hikayeden surdurur); sonda agent.pt ve results.json.

    python train_sentence.py [--data countries|simplestories] [--root SS klasoru] [--epochs 4] [--d 64] [--layers 4]
                             [--heads 4] [--batch 16 (hikaye)] [--lr LR_D64*64/d] [--z 512] [--device cpu|cuda]
                             [--out klasor] [--resume 1]
"""
import argparse
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
EXAM_STORIES = 1000     # SS: sinavin sabit alt kumesi (tohum 0; model_y exam_simplestories ile ayni sayi)
SHOW = 8


class Stories:
    """Hikayeler (kelimeler uc uca + cumle ve hikaye sinirlari) -> hikaye batch'leri (ids (B, T, W), mask) cihazda.  SS:
    make_ss_sentences --stories dosyalari (<prefix>_ids, _sentence_offsets, _offsets); ulke: from_lists.  Arsivdeki context
    egitiminden (arsiv/model_z_20261005); kesme yok (kullanici: "maks kelime diye birşey yok"), <unk> da kelimedir."""

    def __init__(self, flat, sent_off, story_off, device):
        self.flat = torch.as_tensor(np.asarray(flat), dtype=torch.long, device=device)
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
        ids = torch.where(mask, self.flat[pos.clamp(max=len(self.flat) - 1)], torch.zeros_like(pos))
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
    """Sinav hikayeleri -> kayip, ppl, dogruluk, cumle / hikaye bitirme; nbytes verilirse bits per byte (model_y
    exam_simplestories gibi: esas EOS haric, bits_per_byte_eos EOS dahil)."""
    sums = dict(nll=0.0, nll_eos=0.0, n=0, hit=0, end_hit=0, end_n=0, false_end=0, word_n=0, eos_hit=0, eos_n=0,
                false_eos=0, first_n=0)
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
        sums["eos_hit"] += int((pred[ex["eos"]] == model.EOS).sum())
        sums["eos_n"] += int(ex["eos"].sum())
        sums["false_eos"] += int((pred[ex["first"]] == model.EOS).sum())
        sums["first_n"] += int(ex["first"].sum())
    nll = (sums["nll"] + sums["nll_eos"]) / sums["n"]
    out = dict(loss=round(nll, 4), ppl=round(math.exp(nll), 2), acc=round(sums["hit"] / sums["n"], 4),
               end_ok=round(sums["end_hit"] / sums["end_n"], 4), false_end=round(sums["false_end"] / sums["word_n"], 4),
               eos_ok=round(sums["eos_hit"] / sums["eos_n"], 4), false_eos=round(sums["false_eos"] / sums["first_n"], 4))
    if nbytes is not None:
        out["bits_per_byte"] = round(sums["nll"] / math.log(2) / nbytes, 4)
        out["bits_per_byte_eos"] = round((sums["nll"] + sums["nll_eos"]) / math.log(2) / (nbytes + len(rows)), 4)
    return out


def load_countries():
    """-> sozluk, hikayeler (split, ulke, cumleler, valid_next kumeleri)."""
    folder = os.path.join(MODEL_Z, "data", "countries")
    vocab = json.load(open(os.path.join(folder, "country_vocab.json"), encoding="utf-8"))
    ix = {w: i for i, w in enumerate(vocab)}
    ids = lambda s: [ix.get(w, 0) for w in s]  # noqa: E731
    stories = []
    for line in open(os.path.join(folder, "country_stories.jsonl"), encoding="utf-8"):
        s = json.loads(line)
        stories.append(dict(split=s["split"], country=ids(s["country"].split()), sents=[ids(x) for x in s["sentences"]],
                            valid=[{tuple(ids(x)) for x in step} for step in s.get("valid_next", [])]))
    return vocab, stories


def pad_sentences(sents, device="cpu"):
    L = max(len(s) for s in sents)
    ids = torch.zeros(len(sents), L, dtype=torch.long, device=device)
    mask = torch.zeros(len(sents), L, dtype=torch.bool, device=device)
    for i, s in enumerate(sents):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = True
    return ids, mask


@torch.no_grad()
def country_generation(model, exam, keys, device, show):
    """Ulke: gercek onceki cumlelerin z'leriyle sonraki cumle (acgozlu) -> gecerli, dogru olgu, tekrar, ... + ornekler."""
    vocab = model.vocab
    text = lambda s: " ".join(vocab[w] for w in s)  # noqa: E731
    longest = max(len(s) for st in exam for s in st["sents"])
    zs = [encode_z(keys, *pad_sentences(st["sents"], device)) for st in exam]
    by_k = {}
    for si, st in enumerate(exam):
        for k in range(1, len(st["sents"])):
            by_k.setdefault(k, []).append(si)
    res = dict(n=0, valid=0, true_fact=0, repeat=0, eos=0, has_country=0, ended=0, loop=0, exact=0, z_ok=0)
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
    for c in range(0, len(outs), 2048):                                 # z kontrolu: uretilen cumle geri aciliyor mu
        part = outs[c:c + 2048]
        back = decode_z(keys, encode_z(keys, *pad_sentences(part, device)))
        res["z_ok"] += sum(b == o for b, o in zip(back, part))
    n = res.pop("n")
    out = {k: round(v / n, 4) for k, v in res.items()}
    out["n"] = n
    for prev, o, tag in shown:
        print("   onceki : " + "\n            ".join(prev) + "\n   model  : %s   [%s]" % (o, tag), flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="countries", choices=("countries", "simplestories"))
    ap.add_argument("--root", default="G:/Drive'ım/model_z/simplestories_full", help="SS: ss_vocab.json ve ss_*story_*.npy")
    ap.add_argument("--meaning", default=None, help="verilmezse AGENTS[--data]")
    ap.add_argument("--grammar", default=None, help="verilmezse AGENTS[--data]")
    ap.add_argument("--z", type=int, default=512)
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
    args = ap.parse_args(argv)
    args.meaning = args.meaning or AGENTS[args.data][0]
    args.grammar = args.grammar or AGENTS[args.data][1]
    if args.lr is None:
        args.lr = LR_D64 * 64 / args.d
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
    else:
        vocab = json.load(open(os.path.join(args.root, "ss_vocab.json"), encoding="utf-8"))
        train = Stories.from_files(args.root, "ss_story", args.device)
        exam = Stories.from_files(args.root, "ss_exam_story", args.device)
        pick = np.sort(np.random.default_rng(0).permutation(exam.n)[:EXAM_STORIES])
        exam_rows = torch.as_tensor(pick, device=args.device)
        bpath = os.path.join(args.root, "ss_exam_story_bytes.npy")
        if os.path.exists(bpath):
            nbytes = int(np.load(bpath)[pick].sum())
    keys = keys_to(build_keys(args.meaning, args.grammar, args.z, max(train.longest, exam.longest)), args.device)
    assert keys["vocab"] == vocab, "meaning / grammar sozlugu veriyle ayni degil"

    model = SentenceTransformer(vocab, args.z, args.d, args.layers, args.heads).to(args.device)
    if cuda and args.compile:
        for block in model.blocks:
            block.compile(dynamic=True)
        model.loss_fn = torch.compile(output_loss, dynamic=True)          # cikis carpimi + kayip tek grafikte
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=WEIGHT_DECAY), dict(params=no_decay, weight_decay=0.0)],
                            lr=args.lr, betas=BETAS)
    n = train.n
    steps_per = -(-n // args.batch)
    total_steps = steps_per * args.epochs
    first, start, ckpt = 1, 0, os.path.join(args.out, "checkpoint.pt") if args.out else None
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    if args.resume:
        pack = torch.load(ckpt, map_location=args.device, weights_only=False)
        keys_ = ("data", "z", "d", "layers", "heads", "batch", "lr", "epochs", "seed", "meaning", "grammar")
        diff = {k: (pack["args"][k], vars(args)[k]) for k in keys_ if pack["args"][k] != vars(args)[k]}
        assert not diff, "surdurme ayari checkpoint'ten farkli (checkpoint, simdi): %s" % diff
        model.load_state_dict(pack["state"])
        opt.load_state_dict(pack["opt"])
        first, start = pack["epoch"], pack["next"]
        print("SURDURULDU: epok %d, hikaye %d'den" % (first, start), flush=True)

    def save(epoch, nxt):
        if ckpt:
            torch.save(dict(vocab=vocab, state=model.state_dict(), opt=opt.state_dict(), epoch=epoch, next=nxt,
                            args=vars(args)), ckpt + ".part")
            os.replace(ckpt + ".part", ckpt)

    print("veri %s: egitim %d hikaye (%d cumle, en uzun %d kelime), sinav %d hikaye (olcu %d) | z %d (meaning+grammar+konum) "
          "| d %d, katman %d, head %d, %d parametre | batch %d hikaye, lr %g, %d adim | cihaz %s compile %s (%.0f sn hazirlik)"
          % (args.data, n, len(train.sent_off) - 1, train.longest, exam.n, len(exam_rows), args.z, args.d, args.layers,
             args.heads, sum(p.numel() for p in model.parameters()), args.batch, args.lr, total_steps, args.device,
             bool(cuda and args.compile), time.time() - t0), flush=True)
    history = []
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
            ex = examples(ids, mask, story_z(keys, ids, mask), model.END, model.EOS)
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
        model.eval()
        last = epoch == args.epochs
        with torch.autocast(args.device.split(":")[0], dtype=torch.bfloat16, enabled=cuda):
            e = exam_scores(model, exam, exam_rows, keys, args.batch * 4, "true", nbytes)
            s = exam_scores(model, exam, exam_rows, keys, args.batch * 4, "shuffle")
            z0 = exam_scores(model, exam, exam_rows, keys, args.batch * 4, "zero")
            gen = country_generation(model, exam_lists, keys, args.device, SHOW if last else 0) if exam_lists else None
        model.train()
        print("epok %d  kayip %.3f | sinav %s | karisik z kayip %.3f, sifir z kayip %.3f%s  (egitim %.0f sn, olcum %.0f sn)" % (
            epoch, total.item() / count, e, s["loss"], z0["loss"], " | uretim %s" % gen if gen else "", t_train,
            time.time() - t_epoch - t_train), flush=True)
        history.append(dict(epoch=epoch, loss=total.item() / count, exam=e, exam_shuffled_z=s, exam_zero_z=z0,
                            generation=gen))
        save(epoch + 1, 0)
    if args.out:
        torch.save(dict(vocab=vocab, state=model.state_dict(), args=vars(args)), os.path.join(args.out, "agent.pt"))
        json.dump(dict(args=vars(args), history=history), open(os.path.join(args.out, "results.json"), "w"), indent=1)
        print("kaydedildi:", args.out, flush=True)
    return model


if __name__ == "__main__":
    main()
