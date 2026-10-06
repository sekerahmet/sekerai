"""train_meaning (V2) -- meaning agent'in GPT-2 token duzeyinde egitimi (ad onayli, kullanici 6 Ekim; belge 22 s2, 25, 27).
V1 core/meaning/meaning.py + train_meaning.py kopyasi + degisiklik (V2 kurali: V1'den import yok).  Tarif V1 ile ayni (d 256,
buyuyen pencere 1..5 cumle, tam softmax, seyreltme t 3e-5, batch 2048, lr 3e-3, Adam); degisen: birim kelime -> GPT-2
token'i (V 50.257), veri yalniz train, takvim wsd (recipe.wsd_lr; --schedule cosine V1'inki; kullanici karari).

    R[i, j]  = source_i . target_j / sqrt(d)
    P(j | i) = softmax_j(bias_j + R[i, j])                     bias: token'in genel sikligi
    P(j gizli | pencere) = ortalama_{i oy veren} P(j | i)       oy veren: penceredeki obur token'lar
Pencere: cumle s'de biter, hikaye icinde en cok W cumle geriye, son MAX_TOKENS token; farkli token'lar tek torba (END ve
EOS hedef degil: akista cumle disinda).  Seyreltme (word2vec): token pencerede min(1, sqrt(t/f) + t/f) olasilikla kalir
(f train sikligi).  --mask_same_word 1: ayni kelimenin parcalari (" sw" + "am") birbirine oy vermez (belge 25 s1).
Kalite (belge 25 s3): valid'de held-out kayip (ayni pencere, sabit tohumlu seyreltme, ayni-kelime maskesi hep acik, train'de
gecmeyen gizli token sayilmaz), ayni olcu train'in sabit alt kumesinde; taban = gizli token'larin unigram entropisi, kazanc =
taban - kayip; siklik bantlari ve parca bandi.  Goz: EYE token'larinin log P komsulari.
Kayit: her epok sonu ve epok icinde --checkpoint_minutes'te bir checkpoint.pt; wsd inisinin ilk adiminda
decay_start/checkpoint.pt.  --resume 1: ayni --epochs kaldigi yerden; buyuk --epochs uzatma, inis basindan (eski ciktilar
<out>/epochs_<eski>/; common/train.py duzeni).  Sonda agent.pt (state["source.weight"] -> sentence_z.build_keys),
neighbors.pt, metrics.json.  CUDA'da torch.compile, fused Adam, bf16 indirgeme kapali; CPU'da eager.

    python train_meaning.py --root <simplestories (gpt2/)> --offsets <v2/simplestories_gpt2> --out <kosu klasoru>
                            [--device cuda] [--epochs 1] [--schedule wsd] [--mask_same_word 1] [--resume 1]
"""
import argparse
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
from data import EOS_ID, TokenStories  # noqa: E402
import recipe as R  # noqa: E402  (wsd_lr: train.py ile ayni takvim)

V = EOS_ID + 1          # GPT-2 sozlugu; END burada yok (sentence_z kendi satirini ekler)
PAD = EOS_ID            # dolgu: pencerede EOS yok (hikaye siniri); en buyuk kimlik, siralamada sona duser
D = 256
WINDOW = 5              # kullanici, 4 Ekim: "Meaning agent window 5"
BATCH = 2048
LR = 3e-3
EPOCHS = 3
SUBSAMPLE = 3e-5
MAX_TOKENS = 256        # pencerede en cok token (uzun pencerede son token'lar)
BANDS = {"sik": 1000, "orta": 10000, "seyrek": None}    # kalite bantlari: train siklik sirasi (belge 23 s D)
PROGRESS_SECS = 60
DECAY = 0.2                 # recipe.wsd_lr varsayilani: son %20 dogrusal inis; inis basinda decay_start/checkpoint.pt
CHECKPOINT_MINUTES = 10     # epok ici kayit araligi, duvar saati (kullanici, 6 Ekim: "en fazla 10 dk kayıp yaşansın";
                            # buyuk koşuda 30); --checkpoint_minutes, surdurme kimligine girmez
EYE = (" dragon", " forest", " rain", " cookie", " school", " ocean", " Lily", " happy", " said", " the", ".",
       "Lily", "Mia", " swam", " glowed", " kite", " riddle", " sparkled", " Flamewing")
# glue: token basi / sonu sinifi; onceki token'in kelimesine katilir <=> start[cur] & end[prev] != 0 (V1 _WORD kelimesi:
# harf+('harf)? | rakam+)
_ALPHA, _DIGIT, _APOS = 1, 2, 4


def _no_power_throttling():
    """Windows: surecin EcoQoS kisitlamasini kapat (yoksa ~10 kat yavas; kullanici, 6 Ekim)."""
    import ctypes
    from ctypes import wintypes

    class State(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]
    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    s = State(1, 0x1, 0)
    return bool(k.SetProcessInformation(k.GetCurrentProcess(), 4, ctypes.byref(s), ctypes.sizeof(s)))


def glue_tables(text):
    """Token metinleri (V) -> (start, end) int8: kelime devami olabilir mi (harf / rakam / 'harf ile baslar), neyle biter."""
    start = np.array([(_ALPHA if s[:1].isalpha() else 0) | (_DIGIT if s[:1].isdigit() else 0)
                      | (_APOS if len(s) > 1 and s[0] in "'’" and s[1].isalpha() else 0) for s in text], np.int8)
    end = np.array([(_ALPHA | _APOS if s[-1:].isalpha() else 0) | (_DIGIT if s[-1:].isdigit() else 0) for s in text],
                   np.int8)
    return torch.from_numpy(start), torch.from_numpy(end)


def _log_z(src, tgt, bias):
    return (src @ tgt.T / src.shape[-1] ** 0.5 + bias).logsumexp(1)


def _mix(u, v, b, lz, voters):
    """s[b, i, j] = log P(j | i); oy verenler uzerinden ortalama P -> (log P (B, k), oy veren var mi)."""
    s = u @ v.transpose(1, 2) / u.shape[-1] ** 0.5 + b[:, None, :] - lz[:, :, None]
    n = voters.sum(1)
    return s.masked_fill(~voters, -1e9).logsumexp(1) - n.clamp(min=1).log(), n > 0


_COMPILED = {}


def _fn(f, device):
    """CUDA'da torch.compile (kullanici kurali; sekiller her adim degisir: dynamic), CPU'da eager."""
    if device.type != "cuda":
        return f
    if f not in _COMPILED:
        _COMPILED[f] = torch.compile(f, dynamic=True)
    return _COMPILED[f]


class MeaningAgent(torch.nn.Module):
    """V1 MeaningAgent (core/meaning/meaning.py), sozluk GPT-2 kimlikleri (n satir); same: oy veremeyen ciftler."""

    def __init__(self, n=V, d=D):
        super().__init__()
        self.source = torch.nn.Embedding(n, d)          # oy veren; meaning m = source (sentence_z.build_keys)
        self.target = torch.nn.Embedding(n, d)          # oy alan
        self.bias = torch.nn.Parameter(torch.zeros(n))
        for e in (self.source, self.target):
            torch.nn.init.normal_(e.weight, std=0.1)

    def forward(self, ids, present, same=None, chunk=8192):
        """ids (B, k) penceredeki farkli token'lar, present (B, k) gercek yuva, same (B, k, k) oy veremez ya da None ->
        (log P(yuvadaki token | penceredeki obur token'lar) (B, k), oy veren var mi (B, k))."""
        dev = ids.device
        u, v = self.source(ids), self.target(ids)
        uq, inv = ids.unique(return_inverse=True)
        lz = torch.cat([_fn(_log_z, dev)(self.source(uq[c:c + chunk]), self.target.weight, self.bias)
                        for c in range(0, len(uq), chunk)])
        k = ids.shape[1]
        voters = present[:, :, None] & present[:, None, :] & ~torch.eye(k, dtype=torch.bool, device=dev)
        if same is not None:
            voters = voters & ~same
        # bias[ids], lz[inv]: embedding yolu (gelismis indeksin geri yayilimi PAD kopyalarini seri topluyordu; belge 27 D1)
        b = F.embedding(ids, self.bias[:, None]).squeeze(-1)
        return _fn(_mix, dev)(u, v, b, F.embedding(inv, lz[:, None]).squeeze(-1), voters)


class Windows:
    """Token akisi + cumle [bas, son) (N, 2) + hikaye ilk cumlesi (H+1) -> pencere batch'leri cihazda."""

    def __init__(self, stream, sent, story, W, device, glue=None):
        self.stream = torch.empty(len(stream), dtype=torch.int32, device=device)
        for c in range(0, len(stream), 1 << 26):                       # 64M'lik parcalar: host'ta int64 kopya yok
            self.stream[c:c + (1 << 26)] = torch.from_numpy(np.asarray(stream[c:c + (1 << 26)]).astype(np.int32))
        sent, story = np.asarray(sent), np.asarray(story)
        first = np.repeat(story[:-1], np.diff(story))                   # cumlenin hikayesinin ilk cumlesi
        s = np.arange(len(sent))
        self.lo = torch.as_tensor(sent[np.maximum(first, s - W + 1), 0], dtype=torch.long, device=device)
        self.hi = torch.as_tensor(sent[:, 1], dtype=torch.long, device=device)
        self.n = len(sent)
        self.glue = None if glue is None else tuple(g.to(device) for g in glue)

    def batch(self, rows, same_word=False):
        """-> ids (B, k) farkli token'lar artan sirada (dolgu PAD sonda), present (B, k), same (B, k, k) ayni kelime
        parcasi ciftleri (same_word ve glue varsa; yoksa None)."""
        hi = self.hi[rows]
        lo = torch.maximum(self.lo[rows], hi - MAX_TOKENS)
        L = int((hi - lo).max())
        ar = torch.arange(L, device=hi.device)
        pos = lo[:, None] + ar[None]
        ok = pos < hi[:, None]
        tok = torch.where(ok, self.stream[pos.clamp(max=len(self.stream) - 1)].long(), PAD)
        srt = tok.sort(1).values
        present = (srt != PAD) & torch.cat([torch.ones_like(srt[:, :1], dtype=torch.bool), srt[:, 1:] != srt[:, :-1]], 1)
        order = (~present).to(torch.int8).argsort(dim=1, stable=True)
        k = int(present.sum(1).max().clamp(min=1))
        present = present.gather(1, order)[:, :k]
        ids = torch.where(present, srt.gather(1, order)[:, :k], PAD)   # artan; dolgu PAD (searchsorted icin sirali)
        if not (same_word and self.glue is not None):
            return ids, present, None
        start, end = self.glue
        g = torch.zeros_like(ok)
        g[:, 1:] = ((start[tok[:, 1:]] & end[tok[:, :-1]]) != 0) & ok[:, 1:]
        idx = ar.expand(len(tok), -1)
        depth = idx - torch.where(g, 0, idx).cummax(1).values              # kelime icindeki sira (0: kelime basi)
        same = torch.zeros(len(tok), k, k, dtype=torch.bool, device=tok.device)
        slot = torch.searchsorted(ids, tok)
        b = torch.arange(len(tok), device=tok.device)[:, None]
        for d in range(1, int(depth.max()) + 1):
            m = depth[:, d:] >= d                                          # (i, i - d) ayni kelimede
            bi, x, y = b.expand(-1, L - d)[m], slot[:, d:][m], slot[:, :-d][m]
            same[bi, x, y] = True
            same[bi, y, x] = True
        kk = torch.arange(k, device=tok.device)
        same[:, kk, kk] = False                                            # ayni token kelimede iki kez (kosegen)
        return ids, present, same


def _subsample(present, ids, keep, gen):
    return present & (torch.rand(ids.shape, generator=gen, device=ids.device) < keep[ids])


def loss_of(agent, ids, present, same, keep=None, gen=None):
    """Pencereler -> her token gizliyken -log P ortalamasi.  keep (V,): seyreltme (atilan token ne gizlenir ne oy verir)."""
    if keep is not None:
        present = _subsample(present, ids, keep, gen)
    with torch.autocast(ids.device.type, dtype=torch.bfloat16, enabled=ids.device.type == "cuda"):
        logp, ok = agent(ids, present, same)
    use = present & ok
    return -(logp.float() * use).sum() / use.sum().clamp(min=1)


@torch.no_grad()
def evaluate(agent, data, rows, keep, band, batch, piece=None, seed=12345):
    """Sabit pencereler -> dict(nll, unigram, gain, targets, bands).  Seyreltme sabit tohumlu, ayni-kelime maskesi hep acik
    (kalite = farkli kelimeler arasi iliski).  band (V,): train sikligi bandi (BANDS), -1 train'de gecmeyen (sayilmaz).
    unigram: gizli token'larin kendi dagiliminin entropisi (pencereye bakmayan en iyi tahmin); gain = unigram - nll (nat /
    gizli token); bands: bant basina ayni ikili (unigram = o banttaki gizli token'larin -log u ortalamasi); piece (V,) bool
    verilirse bands["parca"]: gecislerinin cogu cok parcali kelimede olan token'lar."""
    dev = data.hi.device
    gen = torch.Generator(device=dev).manual_seed(seed)
    seen = band >= 0
    tot = torch.zeros(len(seen), dtype=torch.float64, device=dev)
    cnt = torch.zeros(len(seen), dtype=torch.float64, device=dev)
    was = agent.training
    agent.eval()
    for b in range(0, len(rows), batch):
        ids, present, same = data.batch(rows[b:b + batch], same_word=True)
        present = _subsample(present, ids, keep, gen)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logp, ok = agent(ids, present, same)
        use = present & ok & seen[ids]
        tot.index_add_(0, ids[use], -logp[use].double())
        cnt += torch.bincount(ids[use], minlength=len(seen)).double()
    agent.train(was)
    N = cnt.sum().clamp(min=1)
    neg_log_u = -(cnt / N).clamp_min(1e-300).log()

    def pair(m):
        n = cnt[m].sum().clamp(min=1)
        nll, H = (tot[m].sum() / n).item(), ((cnt * neg_log_u)[m].sum() / n).item()
        return dict(nll=round(nll, 4), unigram=round(H, 4), gain=round(H - nll, 4), targets=int(cnt[m].sum().item()))
    out = pair(seen)
    out["bands"] = {name: pair(band == i) for i, name in enumerate(BANDS)}
    if piece is not None:
        out["bands"]["parca"] = pair(seen & piece)
    return out


@torch.no_grad()
def build_neighbor_table(agent, m, seen):
    """-> {"ids": (V, m), "scores": (V, m)}: her token i icin log P(j | i) en buyuk m token j (kendisi, PAD ve train'de
    gecmeyenler haric)."""
    src, tgt = agent.source.weight.float(), agent.target.weight.float()
    ids, scores = [], []
    for c in range(0, len(src), 4096):
        rows = torch.arange(c, min(c + 4096, len(src)), device=src.device)
        r = src[rows] @ tgt.T / src.shape[1] ** 0.5 + agent.bias.float()
        r = r - r.logsumexp(1, keepdim=True)
        r[:, ~seen] = -1e9
        r[torch.arange(len(rows)), rows] = -1e9
        top = r.topk(m, dim=1)
        ids.append(top.indices.cpu())
        scores.append(top.values.cpu())
    return dict(ids=torch.cat(ids), scores=torch.cat(scores))


def eye(agent, seen, encode, text, words=EYE, m=15):
    """EYE dizgileri -> token'lari ve her token'in ilk m komsusu (P(j | i)); satirlar listesi (gunluge de yazilir)."""
    table = build_neighbor_table(agent, m, seen)
    lines = ["", "   BAG TABLOSU (token duzeyi, ilk %d, P(j | i)):" % m]
    for w in words:
        for t in encode(w):
            lines.append("   %-12r %-10s %s" % (text[t], "(%s)" % w.strip() if len(encode(w)) > 1 else "", ", ".join(
                "%r %.3f" % (text[j], math.exp(s)) for j, s in zip(table["ids"][t].tolist(), table["scores"][t].tolist()))))
    print("\n".join(lines), flush=True)
    return lines


def _save(obj, path):
    torch.save(obj, path + ".part")
    os.replace(path + ".part", path)


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None, help="simplestories klasoru: gpt2/train.npy, valid.npy, tokenizer.json")
    ap.add_argument("--offsets", default=None, help="V2 sinir klasoru: <split>_sentence_offsets.npy, _story_offsets.npy")
    ap.add_argument("--out", default=None, help="kosu klasoru: her epok checkpoint.pt, sonda agent.pt, neighbors.pt")
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=D)
    ap.add_argument("--batch", type=int, default=BATCH, help="pencere")
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--subsample", type=float, default=SUBSAMPLE, help="seyreltme esigi t (0: kapali)")
    ap.add_argument("--mask_same_word", type=int, default=1, help="1: ayni kelimenin parcalari birbirine oy vermez")
    ap.add_argument("--schedule", default="wsd", choices=("wsd", "cosine"),
                    help="lr takvimi: wsd (recipe.wsd_lr, isinma %%1, son %%20 inis; kullanici karari) | cosine (V1)")
    ap.add_argument("--checkpoint_minutes", type=float, default=CHECKPOINT_MINUTES, help="epok ici kayit araligi (dk)")
    ap.add_argument("--resume", type=int, default=0, help="1: --out'taki checkpoint.pt'den surdur; --epochs buyukse "
                    "uzatma (wsd: inis basindan, decay_start/)")
    return ap.parse_args(argv)


def load(args):
    """-> (train, valid) TokenStories (stream mmap, sent, story), token metinleri, tokenizer sha256."""
    import hashlib
    from tokenizers import Tokenizer
    path = os.path.join(args.root, "gpt2", "tokenizer.json")
    tok = Tokenizer.from_file(path)
    assert tok.get_vocab_size() == V and tok.token_to_id("<|endoftext|>") == EOS_ID
    text = [tok.decode([i]) for i in range(V)]
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    return (TokenStories(args.root, args.offsets, "train"), TokenStories(args.root, args.offsets, "valid"), text, sha,
            lambda s: tok.encode(s, add_special_tokens=False).ids)


def _train(args, train, valid, text, encode=None, tokenizer_sha=None, eval_rows=None, stop_after=None):
    """Egitim.  train / valid: stream, sent (N, 2), story (H+1) tasiyan nesne (TokenStories).  eval_rows: valid ve train
    olcusunde en cok kac pencere (None: valid'in tamami; train alt kumesi ayni sayida).  stop_after: bu epoktan sonra kes
    (surdurme testi)."""
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    t0 = time.time()
    glue = glue_tables(text)
    data = Windows(train.stream, train.sent, train.story, args.window, dev, glue)
    vdata = Windows(valid.stream, valid.sent, valid.story, args.window, dev, glue)
    count = torch.zeros(V, dtype=torch.float64, device=dev)
    lo0, hi0 = int(train.sent[0, 0]), int(train.sent[-1, 1])           # train sikligi: cumlelerin kapsadigi akis
    for c in range(lo0, hi0, 1 << 26):
        count += torch.bincount(data.stream[c:min(c + (1 << 26), hi0)].long(), minlength=V).double()
    count = count.cpu()
    count[EOS_ID] = 0
    seen = (count > 0).to(dev)
    rank = torch.empty(V, dtype=torch.long)
    rank[count.argsort(descending=True)] = torch.arange(V)
    band = torch.full((V,), len(BANDS) - 1, dtype=torch.long)
    for i, top in reversed(list(enumerate(BANDS.values()))):
        if top is not None:
            band[rank < top] = i
    band = torch.where(count > 0, band, -1).to(dev)
    start, end = data.glue
    piece_count = torch.zeros(V, dtype=torch.float64, device=dev)       # cok parcali kelimenin parcasi olarak gecis
    for c in range(lo0, hi0, 1 << 26):
        a, b = max(c - 1, lo0), min(c + (1 << 26) + 1, hi0)
        st = data.stream[a:b].long()
        g = (start[st[1:]] & end[st[:-1]]) != 0                         # g[k]: st[k + 1] onceki token'a katilir
        multi = torch.zeros(len(st), dtype=torch.bool, device=dev)
        multi[1:] |= g
        multi[:-1] |= g
        own = slice(c - a, c - a + min(1 << 26, hi0 - c))
        piece_count += torch.bincount(st[own][multi[own]], minlength=V).double()
    piece = ((piece_count.cpu() >= 0.5 * count) & (count >= 5)).to(dev)
    f = count / count.sum()
    keep = None
    if args.subsample > 0:
        keep = ((args.subsample / f.clamp_min(1e-12)).sqrt() + args.subsample / f.clamp_min(1e-12)).clamp(max=1)
        keep = keep.float().to(dev)
    n_eval = vdata.n if eval_rows is None else min(eval_rows, vdata.n)
    vrows = torch.arange(n_eval, device=dev)
    trows = torch.randperm(data.n, generator=torch.Generator().manual_seed(args.seed + 1000))[:n_eval].sort().values.to(dev)
    agent = MeaningAgent(V, args.d).to(dev)
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr, **({"fused": True} if dev.type == "cuda" else {}))
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    n = data.n
    print("veri: train %d pencere (en cok %d cumle), valid %d, train'de gecen token %d / %d | d %d, lr %g %s, batch %d, "
          "%d parametre, ayni-kelime maskesi %d, parca token turu %d | yukleme %.0f sn" % (
              n, args.window, vdata.n, int(seen.sum()), V, args.d, args.lr, args.schedule, args.batch,
              sum(p.numel() for p in agent.parameters()), args.mask_same_word, int(piece.sum()), time.time() - t0), flush=True)
    if keep is not None and encode is not None:
        print("seyreltme t %g: kalma olasiligi %s" % (args.subsample, ", ".join(
            "%r %.2f" % (w, keep[encode(w)[0]].item()) for w in (" happy", " said", " the", "."))), flush=True)
    cuda = dev.type == "cuda"
    first, start, carried, history = 1, 0, 0.0, []
    steps = -(-n // args.batch)                                        # epok basina adim
    total_steps = args.epochs * steps
    down = total_steps - round(DECAY * total_steps)                    # wsd inisinin ilk adimi (recipe.wsd_lr)
    ckpt = os.path.join(args.out, "checkpoint.pt") if args.out else None
    decay_ckpt = os.path.join(args.out, "decay_start", "checkpoint.pt") if args.out and args.schedule == "wsd" else None

    def save_ckpt(epoch, pos, total, path=None):
        """epoch: biten son epok; pos: epok + 1'de siradaki pencere ofseti (0: epok basi); total: o epokun kayip toplami."""
        os.makedirs(os.path.dirname(path or ckpt), exist_ok=True)
        _save(dict(state=agent.state_dict(), opt=opt.state_dict(), gen=gen.get_state(), epoch=epoch, pos=pos,
                   total=total, args=vars(args), history=history, data=[n, vdata.n]), path or ckpt)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    if args.resume:
        pack = torch.load(ckpt, map_location=dev, weights_only=False)
        keys = ("window", "seed", "d", "batch", "lr", "subsample", "mask_same_word", "schedule")   # epochs yok: uzatma
        diff = {k: (pack["args"].get(k, "cosine" if k == "schedule" else None), vars(args)[k]) for k in keys
                if pack["args"].get(k, "cosine" if k == "schedule" else None) != vars(args)[k]}
        assert not diff, "surdurme ayari checkpoint'ten farkli (checkpoint, simdi): %s" % diff
        assert pack["data"] == [n, vdata.n], "veri checkpoint'tekinden farkli"
        old_epochs = pack["args"]["epochs"]
        assert args.epochs >= old_epochs, "kisaltma yok: --epochs %d < checkpoint'teki %d" % (args.epochs, old_epochs)
        if args.epochs > old_epochs:                                   # uzatma (kural 3): wsd'de inis basindan
            assert args.schedule == "wsd", "uzatma yalniz wsd'de (cosine'in egrisi toplam adima bagli)"
            old_total = old_epochs * steps
            old_down = old_total - round(DECAY * old_total)
            if pack["epoch"] * steps + pack["pos"] // args.batch > old_down:
                pack = torch.load(decay_ckpt, map_location=dev, weights_only=False)
                assert pack["epoch"] * steps + pack["pos"] // args.batch == old_down, "decay_start inis basinda degil"
            arch = os.path.join(args.out, "epochs_%d" % old_epochs)
            for name in ("agent.pt", "neighbors.pt", "metrics.json"):
                if os.path.exists(os.path.join(args.out, name)):
                    os.makedirs(arch, exist_ok=True)
                    shutil.move(os.path.join(args.out, name), os.path.join(arch, name))
            print("UZATMA: %d -> %d epok, genel adim %d'den (inis basi; eski ciktilar %s)" % (
                old_epochs, args.epochs, pack["epoch"] * steps + pack["pos"] // args.batch, arch), flush=True)
        agent.load_state_dict(pack["state"])
        opt.load_state_dict(pack["opt"])
        gen.set_state(pack["gen"].cpu())
        first, start, carried, history = pack["epoch"] + 1, pack["pos"], pack["total"], pack["history"]
        print("SURDURULDU: epok %d, pencere %d / %d'den (%s)" % (first, start, n, ckpt), flush=True)
    t_run = t_ckpt = time.time()
    for epoch in range(first, args.epochs + 1):
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed + epoch)).to(dev)
        total, t_epoch = torch.full((), carried, device=dev), time.time()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        t_shown, b0 = time.time(), start
        for step, b in enumerate(range(0, n, args.batch), 1):
            if b < start:                                      # epok ortasindan surdurme: islenmis pencereler
                continue
            g = (epoch - 1) * steps + step - 1                         # genel adim, 0'dan
            if decay_ckpt and g == down:
                save_ckpt(epoch - 1, b, total.item(), decay_ckpt)      # uzatma buradan (train.py ile ayni)
            if ckpt and time.time() - t_ckpt >= 60 * args.checkpoint_minutes:     # epok ici kayit
                save_ckpt(epoch - 1, b, total.item())
                t_ckpt = time.time()
            if time.time() - t_shown > PROGRESS_SECS:
                t_shown, el = time.time(), time.time() - t_epoch
                print("  epok %d adim %d / %d (%%%.0f)  kayip %.3f  %.0f pencere/sn  kalan ~%.0f dk (butun egitim)" % (
                    epoch, step, steps, 100 * step / steps, total.item() / step, (b - b0) / el,
                    ((steps - step) + (args.epochs - epoch) * steps) * el / (step - b0 // args.batch) / 60), flush=True)
            if args.schedule == "wsd":
                lr = R.wsd_lr(g, total_steps, args.lr, decay=DECAY)
            else:
                lr = args.lr * 0.5 * (1 + math.cos(math.pi * ((epoch - 1) * n + b) / (args.epochs * n)))
            for group in opt.param_groups:
                group["lr"] = lr
            ids, present, same = data.batch(perm[b:b + args.batch], same_word=bool(args.mask_same_word))
            loss = loss_of(agent, ids, present, same, keep, gen)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.detach()
        secs = time.time() - t_epoch
        tv = time.time()
        ev = dict(epoch=epoch, train_loss=round(total.item() / steps, 4), seconds=round(secs, 1),
                  valid=evaluate(agent, vdata, vrows, keep, band, args.batch, piece),
                  train=evaluate(agent, data, trows, keep, band, args.batch, piece))
        history.append(ev)
        print("epok %d  kayip %.4f  (%.0f sn) | %.0f pencere/sn%s\n  KALITE valid: kayip %.4f, unigram %.4f, kazanc %.4f "
              "(%d gizli token) | train alt kumesi: kayip %.4f, kazanc %.4f | fark (valid - train) %.4f  (%.0f sn)" % (
                  epoch, ev["train_loss"], time.time() - t_run, n / secs,
                  ", GPU tepe %.1f GB" % (torch.cuda.max_memory_allocated() / 1e9) if cuda else "",
                  ev["valid"]["nll"], ev["valid"]["unigram"], ev["valid"]["gain"], ev["valid"]["targets"],
                  ev["train"]["nll"], ev["train"]["gain"], ev["valid"]["nll"] - ev["train"]["nll"], time.time() - tv)
              + "\n  bantlar (valid kazanc / train kazanc, gizli token): " + ", ".join("%s %.4f / %.4f (%d)" % (
                  k, ev["valid"]["bands"][k]["gain"], ev["train"]["bands"][k]["gain"], ev["valid"]["bands"][k]["targets"])
                  for k in ev["valid"]["bands"]), flush=True)
        start, carried = 0, 0.0
        if ckpt:
            save_ckpt(epoch, 0, 0.0)
            t_ckpt = time.time()
        if stop_after == epoch:
            return agent, history
    agent.eval()
    lines = eye(agent, seen, encode, text) if encode is not None else []
    if args.out:
        _save(dict(state=agent.state_dict(), args=vars(args), count=count.float(), tokenizer_sha256=tokenizer_sha,
                   history=history), os.path.join(args.out, "agent.pt"))
        _save(build_neighbor_table(agent, 50, seen), os.path.join(args.out, "neighbors.pt"))
        json.dump(dict(args=vars(args), history=history, eye=lines), open(os.path.join(args.out, "metrics.json"), "w",
                                                                          encoding="utf-8"), indent=1)
        print("kaydedildi:", args.out, flush=True)
    return agent, history


def main(argv=None):
    args = parse(argv)
    if args.device.startswith("cuda"):              # uzun K (V) toplamlari fp32'de indirgensin (belge 27 D4; maliyet ~0)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if args.device == "cpu":
        torch.set_num_threads(4)
        if os.name == "nt":
            print("guc kisitlamasi (EcoQoS) kapali:", _no_power_throttling(), flush=True)
    train, valid, text, sha, encode = load(args)
    return _train(args, train, valid, text, encode, sha)


if __name__ == "__main__":
    main()
