"""train_codec -- codec agent'in egitimi (ajan: codec.py).  Hikayesiz, tek tek cumleler: z = encode(cumle), kayip =
-log P(decode ozgun cumleyi kelime kelime yazar); encoder ve decoder birlikte.  --noise > 0: encoder'a kelimeleri
silinmis cumle gider, decoder ozgunu yazar (TSDAE).
Veri: ulke (data/countries, egitim / sinav hikayelerinin cumleleri) ya da SS (ss_story_* egitim, ss_exam_story_* sinav).
Olcu (sinav cumleleri; kullanici: "codec cümle cümle encode yazıp decode yapabiliyor mu"): birebir geri yazim (egitimde
gecmis / gecmemis ayri), z'ye gurultu eklenince birebir geri yazim, goz icin ornekler.  checkpoint.pt her
CHECKPOINT_SECS'te ve epok sonunda (epok icinde kaldigi cumleden surdurur); sonda agent.pt ve results.json.

    python train_codec.py [--data countries|simplestories] [--root klasor] [--epochs 10] [--device cpu|cuda]
                          [--d 256] [--z 256] [--layers 3] [--batch 1024] [--lr 1e-3] [--noise 0] [--out klasor] [--resume 1]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

from codec import D, LAYERS, Z, CodecAgent, loss_of

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_Z = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "grammar"))
from train_grammar import _no_power_throttling  # noqa: E402

BATCH = 1024            # cumle
LR = 1e-3
PROGRESS_SECS = 60      # epok icinde ara satir araligi
CHECKPOINT_SECS = 600   # epok icinde checkpoint araligi (SS epoku uzun; Colab kopabilir)
EXAM_N = 2000           # olcude kullanilan sinav cumlesi
NOISE_Z = (0.1, 0.3, 0.5)  # olcude z'ye eklenen gurultu (z boyutu basina birim olcekli)
SHOW = 8                # goz icin ornek


class Sentences:
    """Cumleler (kelimeler uc uca + cumle baslari) -> sirali batch (ids, mask) cihazda; kesme yok."""

    def __init__(self, flat, off, device):
        self.flat = torch.as_tensor(np.asarray(flat), dtype=torch.long, device=device)
        self.off = torch.as_tensor(np.asarray(off), dtype=torch.long, device=device)
        self.n = len(off) - 1
        self.longest = int((self.off[1:] - self.off[:-1]).max())

    def batch(self, rows):
        a, b = self.off[rows], self.off[rows + 1]
        n = b - a
        L = -(-int(n.max()) // 8) * 8                                  # 8'in kati: compile her boyda yeniden derlemesin
        pos = a[:, None] + torch.arange(L, device=a.device)[None]
        mask = torch.arange(L, device=a.device)[None] < n[:, None]
        ids = torch.where(mask, self.flat[pos.clamp(max=len(self.flat) - 1)], torch.zeros_like(pos))
        return ids, mask

    def hashes(self):
        """-> (n,) int64 cumle ozeti (sirali kelimeler; 2^64'te tasan carpim): ayni cumle ayni ozet."""
        n = self.off[1:] - self.off[:-1]
        powers = torch.ones(self.longest, dtype=torch.long, device=self.flat.device)
        for i in range(1, self.longest):
            powers[i] = powers[i - 1] * 1000003
        pos = torch.arange(len(self.flat), device=self.flat.device) - torch.repeat_interleave(self.off[:-1], n)
        cs = torch.cat([torch.zeros(1, dtype=torch.long, device=self.flat.device), ((self.flat + 1) * powers[pos]).cumsum(0)])
        return (cs[self.off[1:]] - cs[self.off[:-1]]) * 31 + n


def load(args):
    """-> (sozluk, egitim Sentences, sinav Sentences, sinav cumlesi egitimde gecti mi (ya da None))."""
    if args.data == "countries":
        folder = os.path.join(MODEL_Z, "data", "countries")
        vocab = json.load(open(os.path.join(folder, "country_vocab.json"), encoding="utf-8"))
        ix = {w: i for i, w in enumerate(vocab)}
        stories = [json.loads(line) for line in open(os.path.join(folder, "country_stories.jsonl"), encoding="utf-8")]
        parts = {}
        for split in ("train", "exam"):
            sents = [[ix.get(w, 0) for w in x] for s in stories if s["split"] == split for x in s["sentences"]]
            parts[split] = (np.array([w for x in sents for w in x]), np.r_[0, np.cumsum([len(x) for x in sents])], sents)
        if args.holdout > 0:
            # ulkenin butun cumleleri egitimde de geciyor (2.374 farkli soyleyisin hepsi): genelleme icin farkli
            # cumlelerin bir kismi butun gecisleriyle egitimden cikarilir, sinav onlar olur
            distinct = sorted(set(tuple(x) for x in parts["train"][2]))
            rng = np.random.default_rng(args.seed)
            held = {distinct[i] for i in rng.permutation(len(distinct))[:int(len(distinct) * args.holdout)]}
            keep = [x for x in parts["train"][2] if tuple(x) not in held]
            parts["train"] = (np.array([w for x in keep for w in x]), np.r_[0, np.cumsum([len(x) for x in keep])], keep)
            exam = [list(x) for x in sorted(held)]
            parts["exam"] = (np.array([w for x in exam for w in x]), np.r_[0, np.cumsum([len(x) for x in exam])], exam)
        seen = set(tuple(x) for x in parts["train"][2])
        exam_seen = torch.tensor([tuple(x) in seen for x in parts["exam"][2][:EXAM_N]])
        tr, ex = parts["train"][:2], parts["exam"][:2]
    else:
        vocab = json.load(open(os.path.join(args.root, "ss_vocab.json"), encoding="utf-8"))
        tr = [np.load(os.path.join(args.root, "ss_story" + s), mmap_mode="r") for s in ("_ids.npy", "_sentence_offsets.npy")]
        ex = [np.load(os.path.join(args.root, "ss_exam_story" + s), mmap_mode="r") for s in ("_ids.npy", "_sentence_offsets.npy")]
        train, exam = Sentences(tr[0], tr[1], args.device), Sentences(ex[0], ex[1], args.device)
        exam_seen = torch.isin(exam.hashes()[:EXAM_N], train.hashes()).cpu()     # gorulmemis = asil kritik olcu
        return vocab, train, exam, exam_seen
    return vocab, Sentences(tr[0], tr[1], args.device), Sentences(ex[0], ex[1], args.device), exam_seen


@torch.no_grad()
def evaluate(agent, exam, exam_seen, show):
    """Sinav cumleleri -> birebir geri yazim (gurultusuz ve z'ye gurultu eklenmis) + ornekler."""
    rows = torch.arange(min(EXAM_N, exam.n), device=exam.flat.device)
    z_all, true = [], []
    for c in range(0, len(rows), 256):
        ids, mask = exam.batch(rows[c:c + 256])
        z_all.append(agent.encode(ids, mask))
        true += [r[:int(m.sum())] for r, m in zip(ids.tolist(), mask)]
    z = torch.cat(z_all)
    longest = max(len(t) for t in true) + 5
    gen = torch.Generator(device=z.device).manual_seed(0)
    res, outs = {}, {}
    for sigma in (0.0,) + NOISE_Z:
        zz = z + sigma * torch.randn(z.shape, generator=gen, device=z.device)
        out = [w for c in range(0, len(zz), 256) for w in agent.decode(zz[c:c + 256], longest)]
        ok = torch.tensor([o == t for o, t in zip(out, true)])
        key = "exact" if sigma == 0 else "exact_z_noise_%g" % sigma
        res[key] = round(float(ok.float().mean()), 4)
        if exam_seen is not None and sigma == 0:
            res["exact_seen"] = round(float(ok[exam_seen].float().mean()), 4) if exam_seen.any() else None
            res["exact_unseen"] = round(float(ok[~exam_seen].float().mean()), 4) if (~exam_seen).any() else None
            res["n_unseen"] = int((~exam_seen).sum())
        outs[sigma] = out
    V = agent.vocab
    text = lambda x: " ".join(V[w] if w < len(V) else "<?>" for w in x)
    picks = list(range(0, len(true), max(1, len(true) // max(show, 1))))[:show]
    if exam_seen is not None and (~exam_seen).any():                    # gorulmemis cumlelerden de ornek
        picks = picks[:show // 2] + (~exam_seen).nonzero().flatten()[:show - show // 2].tolist()
    for i in picks:
        print("   asil   : " + text(true[i]) + ("" if exam_seen is None else ("  (egitimde var)" if exam_seen[i] else "  (GORULMEMIS)")))
        print("   geri   : " + ("AYNI" if outs[0.0][i] == true[i] else text(outs[0.0][i])))
        print("   z+0.3  : " + ("AYNI" if outs[0.3][i] == true[i] else text(outs[0.3][i])), flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="countries", choices=("countries", "simplestories"))
    ap.add_argument("--root", default=None, help="SS: ss_vocab.json ve ss_story_*.npy klasoru")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=D)
    ap.add_argument("--z", type=int, default=Z)
    ap.add_argument("--layers", type=int, default=LAYERS)
    ap.add_argument("--batch", type=int, default=BATCH, help="cumle")
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--noise", type=float, default=0.0, help="encoder girdisinde kelime silme orani (TSDAE en iyisi 0,6)")
    ap.add_argument("--holdout", type=float, default=0.0, help="ulke: farkli cumlelerin bu kadari egitimden cikar, sinav olur")
    ap.add_argument("--compile", type=int, default=1, help="1: torch.compile (yalniz cuda)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--out", default=None, help="kosu klasoru: her epok checkpoint.pt, sonda agent.pt ve results.json")
    ap.add_argument("--resume", type=int, default=0, help="1: --out'taki checkpoint.pt'den kaldigi epoktan surdur")
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    cuda = args.device.startswith("cuda")
    if not cuda:
        torch.set_num_threads(8)
        if os.name == "nt":
            print("guc kisitlamasi (EcoQoS) kapali:", _no_power_throttling(), flush=True)
    else:
        torch.set_float32_matmul_precision("high")
    vocab, train, exam, exam_seen = load(args)
    agent = CodecAgent(vocab, d=args.d, z=args.z, layers=args.layers).to(args.device)
    compiled = bool(cuda and args.compile)
    if compiled:                                   # agir katmanlar derlenir; gercek satir secimi (boyu veriye bagli) disarida
        agent.encoder.compile(dynamic=True)        # (butun forward'u derlemek G4'te InductorError verdi)
        for layer in agent.decoder:
            layer.compile(dynamic=True)
    forward = agent
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr)
    noise_gen = torch.Generator(device=args.device).manual_seed(args.seed + 1)
    first, start, ckpt = 1, 0, os.path.join(args.out, "checkpoint.pt") if args.out else None
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    if args.resume:
        pack = torch.load(ckpt, map_location=args.device, weights_only=False)
        assert pack["vocab"] == vocab, "sozluk checkpoint'tekinden farkli"
        keys = ("data", "seed", "d", "z", "layers", "batch", "lr", "noise", "holdout", "epochs")
        diff = {k: (pack["args"][k], vars(args)[k]) for k in keys if pack["args"][k] != vars(args)[k]}
        assert not diff, "surdurme ayari checkpoint'ten farkli (checkpoint, simdi): %s" % diff
        agent.load_state_dict(pack["state"])
        opt.load_state_dict(pack["opt"])
        noise_gen.set_state(pack["noise_gen"].cpu() if not cuda else pack["noise_gen"])
        first, start = pack["epoch"], pack["next"]
        print("SURDURULDU: epok %d, cumle %d'den" % (first, start), flush=True)

    def save(epoch, nxt):
        if ckpt:
            torch.save(dict(vocab=vocab, state=agent.state_dict(), opt=opt.state_dict(), noise_gen=noise_gen.get_state(),
                            epoch=epoch, next=nxt, args=vars(args)), ckpt + ".part")
            os.replace(ckpt + ".part", ckpt)
    n = train.n
    steps = -(-n // args.batch)
    print("veri %s: egitim %d cumle (en uzun %d kelime), sinav %d (olcu ilk %d) | d %d, z %d, katman %d, lr %g cosine, "
          "batch %d, gurultu %g, %d parametre | cihaz %s compile %s" % (
              args.data, n, train.longest, exam.n, min(EXAM_N, exam.n), args.d, args.z, args.layers, args.lr, args.batch,
              args.noise, sum(p.numel() for p in agent.parameters()), args.device, compiled), flush=True)
    t0, results = time.time(), {}
    for epoch in range(first, args.epochs + 1):
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed + epoch)).to(args.device)
        total, t_epoch, t_shown, t_saved = torch.zeros((), device=args.device), time.time(), time.time(), time.time()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        b0 = start if epoch == first else 0
        for step, b in enumerate(range(b0, n, args.batch), 1):
            if time.time() - t_shown > PROGRESS_SECS:                 # ara satir: kayip yalniz burada okunur
                t_shown, el = time.time(), time.time() - t_epoch
                left = (n - b) // args.batch + (args.epochs - epoch) * steps
                print("  epok %d adim %d / %d (%%%.0f)  kayip %.3f  %.0f cumle/sn  kalan ~%.0f dk (butun egitim)%s" % (
                    epoch, b // args.batch, steps, 100 * b / n, total.item() / step, (b - b0) / el, left * el / step / 60,
                    ", GPU tepe %.1f GB" % (torch.cuda.max_memory_allocated() / 1e9) if cuda else ""), flush=True)
            if time.time() - t_saved > CHECKPOINT_SECS:
                t_saved = time.time()
                save(epoch, b)
            done = ((epoch - 1) * n + b) / (args.epochs * n)
            for group in opt.param_groups:
                group["lr"] = args.lr * 0.5 * (1 + math.cos(math.pi * done))
            ids, mask = train.batch(perm[b:b + args.batch])
            with torch.autocast(args.device.split(":")[0], dtype=torch.bfloat16, enabled=cuda):
                loss = loss_of(forward, ids, mask, args.noise, noise_gen)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.detach()
        line = "epok %d  kayip %.4f  (%.0f sn)" % (epoch, total.item() / step, time.time() - t0)
        if epoch == args.epochs:
            agent.eval()
            results = evaluate(agent, exam, exam_seen, SHOW)
            agent.train()
            line += " | sinav: %s" % results
        print(line, flush=True)
        save(epoch + 1, 0)
    if args.out:
        torch.save(dict(vocab=vocab, state=agent.state_dict(), args=vars(args)), os.path.join(args.out, "agent.pt"))
        json.dump(dict(args=vars(args), results=results), open(os.path.join(args.out, "results.json"), "w"), indent=1)
        print("kaydedildi:", args.out, flush=True)
    return agent


if __name__ == "__main__":
    main()
