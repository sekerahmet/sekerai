"""train_baseline -- BaselineTransformer egitimi ve olcusu, Model Z'nin train_sentence.py'si ile ayni veri, ayni hedef, ayni
tarif, ayni sira (adillik sozlesmesi: belge/model_z_temel/17_transformer_taban_tasarimi.md).  Hikaye tek dizi
(story_sequences); hedefler Model Z'nin examples() hedefleriyle hikaye hikaye birebir ayni (tests_baseline T1).
Sinav: model_y'nin sizintisiz sinav kumesi (<root>/../../simplestories/exam_stories.npy, valid hikaye sirasi; sha256
exam_stories.json ile dogrulanir) icinden default_rng(0) ile 1.000 hikaye, sirali (model_y exam_rows ile ayni secim).
Olcu, exam_scores:
    kayip, ppl, dogruluk, bits per byte (EOS haric; payda ss_exam_story_bytes.npy)
    cumle bitirme (END), hikaye bitirme (EOS), hedef turune gore kayip / dogruluk (first, mid, end, eos)
    baglam: full (hikaye), none (her cumle tek basina), shuffled (onceki cumleler baska hikayeden)
checkpoint.pt her CHECKPOINT_SECS'te ve epok sonunda (kaldigi hikayeden surdurur); sonda agent.pt, results.json ve
generate_baseline ile sabit istemlerden hikaye (samples.txt / samples.json).

    python train_baseline.py [--root SS klasoru] [--epochs 4] [--d 256] [--layers 4] [--heads 4] [--batch 16 (hikaye)]
                             [--lr LR_D64*64/d] [--seed 0] [--compile 1] [--device cpu|cuda] [--out klasor]
                             [--resume 1] [--train-stories N (yalniz ilk N egitim hikayesi; duman testi)]
"""
import argparse
import hashlib
import json
import math
import os
import time

import torch  # noqa: I001  (Windows: torch numpy'den once)
import numpy as np
import torch.nn.functional as F

from baseline import BaselineTransformer, output_loss

ROOT = "G:/Drive'ım/model_z/simplestories_full"
# lr: Model Z ile ayni olcut (train_sentence LR_D64 * 64 / d; d 256 -> 1,75e-3).
LR_D64 = 7e-3
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
CLIP = 1.0
WARMUP = 0.05           # adimlarin bu kadari dogrusal isinma, sonra cosine
PROGRESS_SECS = 60      # epok icinde ara satir araligi
CHECKPOINT_SECS = 600
EXAM_STORIES = 1000     # sinavin sabit alt kumesi: model_y exam_stories.npy (sizintisiz) icinden tohum 0
CONTEXTS = ("full", "none", "shuffled")


def _no_power_throttling():
    """Windows: surecin guc kisitlamasini (EcoQoS) kapat; arka plandaki surec ~10 kat yavasliyordu (Model Z train_grammar)."""
    import ctypes
    from ctypes import wintypes

    class State(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]
    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    s = State(1, 0x1, 0)
    return bool(k.SetProcessInformation(k.GetCurrentProcess(), 4, ctypes.byref(s), ctypes.sizeof(s)))


class Stories:
    """Hikayeler (kelimeler uc uca + cumle ve hikaye sinirlari) -> hikaye batch'leri (ids (B, T, W), mask) cihazda.  Model Z
    train_sentence.Stories'in kopyasi (ayni batch; tests_baseline T9); from_files ayni dosyalar, mmap ile okur."""

    def __init__(self, flat, sent_off, story_off, device):
        self.flat = torch.as_tensor(np.array(flat), dtype=torch.long, device=device)      # np.array: mmap'ten kopya
        self.sent_off = torch.as_tensor(np.array(sent_off), dtype=torch.long, device=device)
        self.story_off = torch.as_tensor(np.array(story_off), dtype=torch.long, device=device)
        self.n = len(self.story_off) - 1
        self.longest = int((self.sent_off[1:] - self.sent_off[:-1]).max())

    @classmethod
    def from_files(cls, root, prefix, device, stories=None):
        """stories: yalniz ilk N hikaye (dosyanin geri kalani okunmaz)."""
        flat, sent_off, story_off = [np.load(os.path.join(root, prefix + s), mmap_mode="r")
                                     for s in ("_ids.npy", "_sentence_offsets.npy", "_offsets.npy")]
        if stories is not None:
            story_off = story_off[:stories + 1]
            sent_off = sent_off[:int(story_off[-1]) + 1]
            flat = flat[:int(sent_off[-1])]
        return cls(flat, sent_off, story_off, device)

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


def story_sequences(ids, mask, END, EOS, context="full"):
    """Hikaye batch'i (ids (B, S, W), mask) -> dizi + hedef (cihazda, vektorel).  -> dict: tok (N, T) (konum 0 BOS, dolgu
    sagda 0), target (-100 hedef yok), tur maskeleri first / mid / end / eos, story (N,).
      full      N = B: [BOS] w.. END w.. END ...; konum t sonraki token'i, son END EOS'u tahmin eder.
      none      N = ornek (Model Z gibi her cumle + hikaye sonu): [BOS] w_1..w_L; hikaye sonu ornegi [BOS] -> EOS.
      shuffled  none gibi ama k. cumleden once baska hikayenin (b+1) ilk k cumlesi (yoksa son cumlesi) END'li; Model Z
                story_z(mode="shuffle") ile ayni bagisci kurali.  Batch tek hikayeyse bagisci kendisi: full ile ayni tahmin.
    Hedefler uc baglamda da ayni: hikaye basina W + S + 1."""
    dev = ids.device
    B, S, W = ids.shape
    L = mask.sum(-1)
    has = mask.any(-1)
    n = has.sum(1)
    if context == "full":
        col = torch.arange(W + 1, device=dev)
        is_end = (col == L[..., None]) & has[..., None]          # her cumlenin sonuna END
        x = torch.where(is_end, END, torch.cat([ids, torch.zeros_like(ids[..., :1])], -1)).view(B, -1)
        keep = (torch.cat([mask, torch.zeros_like(mask[..., :1])], -1) | is_end).view(B, -1)
        is_end = is_end.view(B, -1)
        Tlen = 1 + keep.sum(1)                                   # girdi boyu = hedef sayisi
        Tx = int(Tlen.max())
        at = torch.cumsum(keep, 1)                               # BOS'tan sonraki konum
        rb = torch.arange(B, device=dev)[:, None].expand_as(keep)
        tok = torch.zeros(B, Tx + 1, dtype=torch.long, device=dev)
        tok[rb[keep], at[keep]] = x[keep]
        end_in = torch.zeros(B, Tx + 1, dtype=torch.bool, device=dev)
        end_in[rb[keep], at[keep]] = is_end[keep]
        j = torch.arange(Tx, device=dev)[None]
        live = j < Tlen[:, None] - 1
        end = live & end_in[:, 1:]
        eos = j == Tlen[:, None] - 1
        first = live & ((j == 0) | end_in[:, :Tx])
        target = torch.where(live, tok[:, 1:], -100)
        target = torch.where(eos, EOS, target)
        return dict(tok=tok[:, :Tx], target=target, first=first, mid=live & ~first & ~end, end=end, eos=eos,
                    story=torch.arange(B, device=dev))
    assert context in ("none", "shuffled"), context
    cnt = n + 1
    E = int(cnt.sum())
    ex_b = torch.repeat_interleave(torch.arange(B, device=dev), cnt)
    ex_k = torch.arange(E, device=dev) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
    src = (ex_b + 1) % B
    m = ex_k if context == "shuffled" else torch.zeros_like(ex_k)
    seg_e = torch.repeat_interleave(torch.arange(E, device=dev), m)          # baglam cumleleri
    seg_j = torch.arange(len(seg_e), device=dev) - torch.repeat_interleave(torch.cumsum(m, 0) - m, m)
    seg_b = src[seg_e]
    seg_s = torch.minimum(seg_j, n[seg_b] - 1)
    seg_len = L[seg_b, seg_s] + 1
    ctx = torch.zeros(E, dtype=torch.long, device=dev).index_add_(0, seg_e, seg_len)
    own = torch.where(ex_k < n[ex_b], L[ex_b, ex_k.clamp(max=S - 1)], 0)
    Tx = int((1 + ctx + own).max())
    tok = torch.zeros(E, Tx + 1, dtype=torch.long, device=dev)
    if len(seg_e):
        t_seg = torch.repeat_interleave(torch.arange(len(seg_e), device=dev), seg_len)
        cum = torch.cumsum(seg_len, 0) - seg_len
        off = torch.arange(len(t_seg), device=dev) - cum[t_seg]
        start = 1 + cum - (torch.cumsum(ctx, 0) - ctx)[seg_e]                # orneginin icinde
        word = ids[seg_b[t_seg], seg_s[t_seg], off.clamp(max=W - 1)]
        tok[seg_e[t_seg], start[t_seg] + off] = torch.where(off < seg_len[t_seg] - 1, word, END)
    t_own = torch.repeat_interleave(torch.arange(E, device=dev), own)
    off = torch.arange(len(t_own), device=dev) - torch.repeat_interleave(torch.cumsum(own, 0) - own, own)
    tok[t_own, 1 + ctx[t_own] + off] = ids[ex_b[t_own], ex_k[t_own], off]
    j = torch.arange(Tx, device=dev)[None]
    p0, o = ctx[:, None], own[:, None]
    eos_ex = (ex_k == n[ex_b])[:, None]
    first = (j == p0) & ~eos_ex
    eos = (j == p0) & eos_ex
    mid = (j > p0) & (j < p0 + o)
    end = (j == p0 + o) & (o > 0)
    target = torch.where(first | mid, tok[:, 1:], -100)
    target = torch.where(end, END, torch.where(eos, EOS, target))
    return dict(tok=tok[:, :Tx], target=target, first=first, mid=mid, end=end, eos=eos, story=ex_b)


@torch.no_grad()
def exam_scores(model, st, rows, batch, context="full", nbytes=None):
    """Sinav hikayeleri -> kayip, ppl, dogruluk, cumle / hikaye bitirme, hedef turune gore kayip / dogruluk; nbytes
    verilirse bits per byte (Model Z exam_scores ile ayni tanim: esas EOS haric, bits_per_byte_eos EOS dahil)."""
    kinds = ("first", "mid", "end", "eos")
    sums = dict(nll=0.0, nll_eos=0.0, n=0, hit=0, end_hit=0, end_n=0, false_end=0, word_n=0, eos_hit=0, eos_n=0,
                false_eos=0, first_n=0)
    per = {k: [0.0, 0, 0] for k in kinds}                                 # nll, dogru, sayi
    bounds = list(range(0, len(rows), batch))
    if len(bounds) > 1 and len(rows) - bounds[-1] == 1:                 # karisik: tek hikayelik parca kendiyle karisir
        bounds = bounds[:-1]
    for i, c in enumerate(bounds):
        stop = bounds[i + 1] if i + 1 < len(bounds) else len(rows)
        ids, mask = st.batch(rows[c:stop])
        ex = story_sequences(ids, mask, model.END, model.EOS, context)
        h = model.hidden(ex["tok"])
        keep = ex["target"] >= 0
        hk, tgt = h[keep], ex["target"][keep]
        nll = torch.empty(len(tgt), device=h.device)
        pred = torch.empty(len(tgt), dtype=torch.long, device=h.device)
        for r in range(0, len(tgt), 4096):                                # logit tablosu parca parca (bellek)
            lg = (hk[r:r + 4096] @ model.E.weight.T).float()
            nll[r:r + 4096] = F.cross_entropy(lg, tgt[r:r + 4096], reduction="none")
            pred[r:r + 4096] = lg.argmax(-1)
        for name in kinds:
            ex[name] = ex[name][keep]
            per[name][0] += float(nll[ex[name]].sum())
            per[name][1] += int((pred[ex[name]] == tgt[ex[name]]).sum())
            per[name][2] += int(ex[name].sum())
        sums["nll"] += float(nll[~ex["eos"]].sum())
        sums["nll_eos"] += float(nll[ex["eos"]].sum())
        sums["n"] += len(tgt)
        sums["hit"] += int((pred == tgt).sum())
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
               eos_ok=round(sums["eos_hit"] / sums["eos_n"], 4), false_eos=round(sums["false_eos"] / sums["first_n"], 4),
               targets=sums["n"])
    out["by_type"] = {k: dict(n=v[2], loss=round(v[0] / max(v[2], 1), 4), acc=round(v[1] / max(v[2], 1), 4))
                      for k, v in per.items()}
    if nbytes is not None:
        out["bits_per_byte"] = round(sums["nll"] / math.log(2) / nbytes, 4)
        out["bits_per_byte_eos"] = round((sums["nll"] + sums["nll_eos"]) / math.log(2) / (nbytes + len(rows)), 4)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=ROOT, help="ss_vocab.json, ss_story_*.npy, ss_exam_story_*.npy")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16, help="hikaye")
    ap.add_argument("--lr", type=float, default=None, help="verilmezse LR_D64 * 64 / d (Model Z ile ayni)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile", type=int, default=1, help="1: torch.compile (yalniz cuda)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", type=int, default=0)
    ap.add_argument("--train-stories", type=int, default=None, help="yalniz ilk N egitim hikayesi (duman testi)")
    args = ap.parse_args(argv)
    if args.lr is None:
        args.lr = LR_D64 * 64 / args.d
    torch.manual_seed(args.seed)
    cuda = args.device.startswith("cuda")
    if cuda:                                                             # GPU kapisi (CLAUDE.md kural 2)
        assert torch.cuda.is_available(), "GPU YOK"
        free = torch.cuda.mem_get_info()[0] / 1e9
        assert free > 12, "GPU'da yalniz %.1f GB bos" % free
        torch.set_float32_matmul_precision("high")
    elif os.name == "nt":
        print("guc kisitlamasi (EcoQoS) kapali:", _no_power_throttling(), flush=True)
    t0 = time.time()

    vocab = json.load(open(os.path.join(args.root, "ss_vocab.json"), encoding="utf-8"))
    train = Stories.from_files(args.root, "ss_story", args.device, args.train_stories)
    exam = Stories.from_files(args.root, "ss_exam_story", args.device)
    exam_path = os.path.normpath(os.path.join(args.root, "..", "..", "simplestories", "exam_stories.npy"))
    exam_set = np.load(exam_path)
    exam_sha = hashlib.sha256(np.ascontiguousarray(exam_set)).hexdigest()
    saved = json.load(open(exam_path[:-4] + ".json", encoding="utf-8"))
    assert exam_sha == saved["sha256"] and len(exam_set) == saved["stories"], "sinav kumesi izi tutmuyor: %s" % exam_path
    assert exam_set.max() < exam.n, "sinav kumesi bu sinav dosyasina ait degil"
    pick = np.sort(exam_set[np.random.default_rng(0).permutation(len(exam_set))[:EXAM_STORIES]])
    exam_rows = torch.as_tensor(pick, device=args.device)
    bpath = os.path.join(args.root, "ss_exam_story_bytes.npy")
    assert os.path.exists(bpath), "bpb paydasi yok: %s" % bpath
    story_bytes = np.load(bpath)
    assert len(story_bytes) == exam.n, "bayt dosyasi sinav hikayeleriyle hizali degil"
    nbytes = int(story_bytes[pick].sum())
    args.exam_set, args.exam_set_sha256 = exam_path, exam_sha

    model = BaselineTransformer(vocab, args.d, args.layers, args.heads).to(args.device)
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
    params = sum(p.numel() for p in model.parameters())
    first, start, ckpt = 1, 0, os.path.join(args.out, "checkpoint.pt") if args.out else None
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    if args.resume:
        assert ckpt and os.path.exists(ckpt), "surdurme paketi yok: %s (baştan baslamaz)" % ckpt
        pack = torch.load(ckpt, map_location=args.device, weights_only=False)
        keys_ = ("d", "layers", "heads", "batch", "lr", "epochs", "seed", "train_stories", "exam_set_sha256")
        diff = {k: (pack["args"][k], vars(args)[k]) for k in keys_ if pack["args"][k] != vars(args)[k]}
        assert not diff, "surdurme ayari checkpoint'ten farkli (checkpoint, simdi): %s" % diff
        model.load_state_dict(pack["state"])
        opt.load_state_dict(pack["opt"])
        first, start = pack["epoch"], pack["next"]
        print("SURDURULDU: epok %d, hikaye %d'den" % (first, start), flush=True)
    elif args.out:
        json.dump(dict(args=vars(args), params=params, train_stories=n, train_sentences=len(train.sent_off) - 1,
                       train_words=len(train.flat), exam_stories=exam.n, exam_rows=pick.tolist(), exam_bytes=nbytes,
                       steps=total_steps), open(os.path.join(args.out, "config.json"), "w"), indent=1)

    def save(epoch, nxt):
        if ckpt:
            torch.save(dict(vocab=vocab, state=model.state_dict(), opt=opt.state_dict(), epoch=epoch, next=nxt,
                            args=vars(args)), ckpt + ".part")
            os.replace(ckpt + ".part", ckpt)

    print("veri simplestories: egitim %d hikaye (%d cumle, %d kelime, hedef %d), sinav %d hikaye (olcu %d, %d bayt) | "
          "d %d, katman %d, head %d, %d parametre | batch %d hikaye, lr %g, %d adim | cihaz %s compile %s (%.0f sn hazirlik)"
          % (n, len(train.sent_off) - 1, len(train.flat), len(train.flat) + len(train.sent_off) - 1 + n, exam.n,
             len(pick), nbytes, args.d, args.layers, args.heads, params, args.batch, args.lr, total_steps, args.device,
             bool(cuda and args.compile), time.time() - t0), flush=True)
    history = []
    for epoch in range(first, args.epochs + 1):
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed + epoch)).to(args.device)
        total, count, sents_done, toks_done = torch.zeros((), device=args.device), 0, 0, 0
        t_epoch = t_shown = t_saved = time.time()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        b0 = start if epoch == first else 0
        for b in range(b0, n, args.batch):
            if time.time() - t_shown > PROGRESS_SECS:                     # ara satir: kayip yalniz burada okunur
                t_shown, el = time.time(), time.time() - t_epoch
                left = (n - b) // args.batch + (args.epochs - epoch) * steps_per
                print("  epok %d hikaye %d / %d (%%%.0f)  kayip %.3f  %.0f cumle/sn  %.0f token/sn  kalan ~%.0f dk "
                      "(butun egitim)%s" % (
                          epoch, b, n, 100 * b / n, total.item() / max(count, 1), sents_done / el, toks_done / el,
                          left * el / max(count, 1) / 60,
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
            ex = story_sequences(ids, mask, model.END, model.EOS)
            sents_done += int(mask.any(-1).sum())
            toks_done += int(mask.sum()) + int(mask.any(-1).sum()) + len(ids)
            with torch.autocast(args.device.split(":")[0], dtype=torch.bfloat16, enabled=cuda):
                loss = model(ex["tok"], ex["target"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            total += loss.detach()
            count += 1
        t_train = time.time() - t_epoch
        model.eval()
        with torch.autocast(args.device.split(":")[0], dtype=torch.bfloat16, enabled=cuda):
            scores = {c: exam_scores(model, exam, exam_rows, args.batch * 4, c, nbytes if c == "full" else None)
                      for c in CONTEXTS}
        model.train()
        print("epok %d  kayip %.3f | sinav %s | bagimsiz cumle kayip %.3f, karisik baglam kayip %.3f  (egitim %.0f sn, "
              "%.0f cumle/sn, %.0f token/sn; olcum %.0f sn)" % (
                  epoch, total.item() / max(count, 1), scores["full"], scores["none"]["loss"],
                  scores["shuffled"]["loss"], t_train, sents_done / t_train, toks_done / t_train,
                  time.time() - t_epoch - t_train), flush=True)
        history.append(dict(epoch=epoch, loss=total.item() / max(count, 1), exam=scores["full"],
                            exam_no_context=scores["none"], exam_shuffled_context=scores["shuffled"],
                            train_seconds=round(t_train), sentences_per_sec=round(sents_done / t_train),
                            tokens_per_sec=round(toks_done / t_train)))
        save(epoch + 1, 0)
    if args.out:
        torch.save(dict(vocab=vocab, state=model.state_dict(), args=vars(args)), os.path.join(args.out, "agent.pt"))
        json.dump(dict(args=vars(args), params=params, history=history), open(os.path.join(args.out, "results.json"), "w"),
                  indent=1)
        print("kaydedildi:", args.out, flush=True)
        import generate_baseline
        generate_baseline.main(["--out", args.out, "--root", args.root, "--device", args.device])
    return model


if __name__ == "__main__":
    main()
