"""train_context -- baglam ajaninin egitimi ve olcumu (ajan: context.py; matematik: belge/model_z_temel/07, 09).

Kayip: sonraki cumlenin gercek torbasinin kesin karisim olasiligi, -log sum_k pi_k P(B | k); olu yon icin RELAX:
(1 - relax) * karisim + relax * yonlerin ortalamasi (her yon biraz ogrenir; Rupprecht ve ark. MHP).  Etiket yok.
Olculer (sinav hikayeleri, her cumle gecisi; 07 §7):
    bag_in_top_k     gercek torba (kelime + sayi, 3+ birlesik) ilk 1 / 5 / K aday icinde mi; taban: onceki cumlenin torbasi
    tutarlilik       aday torbalarin kaci veride gecen gercek bir cumle torbasi (ilk aday / butun adaylar)
    calibration      adaylarin olasiligi ile gercekten dogru cikma sikligi (5 dilim, ortalama sapma)
    direction_health farkli torba / K, exp H(pi), olu yon (sinavda pi hic 0,01'i gecmeyen)
Goz: birkac sinav hikayesinde her cumleden sonra ilk 3 aday torba ve gercek sonraki cumle.

    python train_context.py [--data countries] [--epochs 30] [--device cpu|cuda] [--directions 20] [--out klasor]
"""
import argparse
import json
import math
import os
import time
from collections import defaultdict

import torch

from context import D, DIRECTIONS, LEVELS, SLOTS, ContextAgent

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_Z = os.path.dirname(os.path.dirname(HERE))
FILES = {"countries": "country_"}   # data/<ad>/<onek>{stories.jsonl, vocab.json}
BATCH = 64              # hikaye
LR = 1e-3
SCHEDULE = "cosine"     # constant | cosine (adim adim --epochs sonunda 0)
RELAX = 0.05            # olu yon gevsetmesi; Rupprecht ve ark. (MHP) degeri, bu modelde olculmedi
SHOW = 2                # goz: kac sinav hikayesi yazilir


def _no_power_throttling():
    """Windows: bu surecin guc kisitlamasini (EcoQoS) kapat.  Arka planda baslayan surec 1-3 sn sonra yavas cekirdege
    alinip ~10 kat yavasliyordu (senior-developer, 4 Ekim; belge/model_z_temel/06 H1)."""
    import ctypes
    from ctypes import wintypes

    class State(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]
    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    s = State(1, 0x1, 0)                                 # ProcessPowerThrottling, EXECUTION_SPEED denetimi, kapali
    return bool(k.SetProcessInformation(k.GetCurrentProcess(), 4, ctypes.byref(s), ctypes.sizeof(s)))


def _cached(path, sources, build):
    """path, kaynaklarin (veri ve bu kod) hepsinden yeniyse okunur; degilse build() -> path (her koşuda kodlanmasın)."""
    if os.path.exists(path) and all(os.path.getmtime(path) > os.path.getmtime(s) for s in sources):
        return torch.load(path, weights_only=False)
    out = build()
    torch.save(out, path + ".part")
    os.replace(path + ".part", path)
    return out


def _encode(agent, stories):
    """Hikayeler -> ids (N, T, W) dolgulu, mask (N, T, W)."""
    T = max(len(s["sentences"]) for s in stories)
    W = max(len(x) for s in stories for x in s["sentences"])
    ids = torch.zeros(len(stories), T, W, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for n, s in enumerate(stories):
        for t, x in enumerate(s["sentences"]):
            ids[n, t, :len(x)] = torch.tensor(agent.ids(x))
            mask[n, t, :len(x)] = True
    return ids, mask


def _bag(ids, mask, vocab_size):
    """Bir cumle (B, W) -> torbadaki farkli kelimeler (B, W) ve sayilari (B, W; 0 = dolgu).  <unk> torbaya girmez."""
    counts = torch.zeros(len(ids), vocab_size, dtype=torch.long, device=ids.device).scatter_add_(1, ids, mask.long())
    counts[:, 0] = 0
    c, words = counts.topk(ids.shape[1], dim=1)
    return words, c


def loss_of(agent, ids, mask, relax):
    """Hikaye batch'i -> ortalama kayip (gecis basina, nat): (1 - relax) * -log sum_k pi_k P(B | k) + relax * yon ortalamasi."""
    state = agent.initial_state(len(ids))
    LP, PI = [], []
    for t in range(ids.shape[1] - 1):
        state = agent.read(state, ids[:, t], mask[:, t])
        valid = mask[:, t + 1].any(1)
        if not valid.any():
            break
        words, c = _bag(ids[:, t + 1], mask[:, t + 1], len(agent.vocab))
        log_pi, lp = agent.bag_log_prob(state, words, c)
        LP.append(lp[valid])
        PI.append(log_pi[valid])
    lp, log_pi = torch.cat(LP), torch.cat(PI)
    return ((1 - relax) * -(log_pi + lp).logsumexp(1) + relax * -lp.mean(1)).mean()


def _key(counts_row, V):
    """Sayi vektoru (V,) -> torba anahtari: ((kelime, sayi), ...) sirali."""
    nz = counts_row.nonzero().squeeze(1)
    return tuple(zip(nz.tolist(), counts_row[nz].tolist()))


def evaluate(agent, ids, mask, real_bags, names, show=SHOW):
    """-> bag_in_top_k (ilk 1 / 5 / K, taban: onceki cumle), tutarlilik, calibration, direction_health; goz satirlari."""
    K, V = agent.directions, len(agent.vocab)
    hit = defaultdict(int)
    n, coherent_first, coherent_all, n_cand, distinct, ent = 0, 0, 0, 0, 0.0, 0.0
    bins = [[0.0, 0, 0] for _ in range(5)]           # olasilik toplami, aday, dogru
    pi_max = torch.zeros(K)
    eye = []
    with torch.no_grad():
        for b0 in range(0, len(ids), 256):
            bi, bm = ids[b0:b0 + 256], mask[b0:b0 + 256]
            state = agent.initial_state(len(bi))
            for t in range(bi.shape[1] - 1):
                state = agent.read(state, bi[:, t], bm[:, t])
                counts, logp = agent.next_bags(state)
                log_pi = agent._heads(state)[0]
                pi_max = torch.maximum(pi_max, log_pi.exp().max(0).values.cpu())
                for r in range(len(bi)):
                    if not bm[r, t + 1].any():
                        continue
                    words, c = _bag(bi[r:r + 1, t + 1], bm[r:r + 1, t + 1], V)
                    true = tuple(sorted((w, min(x, LEVELS - 1)) for w, x in zip(words[0].tolist(), c[0].tolist()) if x))
                    prev_w, prev_c = _bag(bi[r:r + 1, t], bm[r:r + 1, t], V)
                    prev = tuple(sorted((w, min(x, LEVELS - 1)) for w, x in zip(prev_w[0].tolist(), prev_c[0].tolist()) if x))
                    merged = defaultdict(float)
                    for k in range(K):                                       # ayni torbayi veren yonler birlesir
                        merged[_key(counts[r, k], V)] += math.exp(logp[r, k].item())
                    ranked = sorted(merged.items(), key=lambda x: -x[1])
                    keys = [key for key, _ in ranked]
                    n += 1
                    hit["ilk1"] += true in keys[:1]
                    hit["ilk5"] += true in keys[:5]
                    hit["ilkK"] += true in keys
                    hit["taban_onceki"] += prev == true
                    coherent_first += keys[0] in real_bags
                    coherent_all += sum(key in real_bags for key in keys)
                    n_cand += len(keys)
                    distinct += len(keys) / K
                    p = log_pi[r].exp()
                    ent += math.exp(-(p * p.clamp_min(1e-12).log()).sum().item())
                    for key, prob in ranked:
                        s = bins[min(int(prob * 5), 4)]
                        s[0] += prob
                        s[1] += 1
                        s[2] += key == true
                    if b0 + r < show:
                        eye.append((b0 + r, t, bi[r, t][bm[r, t]].tolist(), ranked[:3], bi[r, t + 1][bm[r, t + 1]].tolist()))
    calib = sum(abs(s[0] / s[1] - s[2] / s[1]) * s[1] for s in bins if s[1]) / max(sum(s[1] for s in bins), 1)
    out = dict(bag_in_top_k={k: round(v / n, 4) for k, v in hit.items()},
               tutarlilik=dict(ilk_aday=round(coherent_first / n, 4), butun_adaylar=round(coherent_all / max(n_cand, 1), 4)),
               calibration=round(calib, 4),
               direction_health=dict(farkli_torba_orani=round(distinct / n, 4), exp_H_pi=round(ent / n, 2),
                                     olu_yon=int((pi_max < 0.01).sum())))
    bag_text = lambda key: " ".join(agent.vocab[w] + ("x%d" % c if c > 1 else "") for w, c in key)
    for s, t, read, top, real in eye:
        if t == 0:
            print("   --- %s" % names[s])
        print("   okunan: %s" % " ".join(agent.vocab[i] for i in read))
        for key, prob in top:
            print("      %.3f  {%s}" % (prob, bag_text(key)))
        print("      gercek: %s" % " ".join(agent.vocab[i] for i in real))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="countries", choices=sorted(FILES))
    ap.add_argument("--root", default=None, help="veri klasoru (varsayilan data/<data>)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=D)
    ap.add_argument("--slots", type=int, default=SLOTS, help="durumun yuva sayisi")
    ap.add_argument("--directions", type=int, default=DIRECTIONS, help="aday torba (yon) sayisi")
    ap.add_argument("--relax", type=float, default=RELAX, help="olu yon gevsetmesi")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--schedule", default=SCHEDULE, choices=("constant", "cosine"))
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--every", type=int, default=10, help="kac epokta bir olcum")
    ap.add_argument("--out", default=None, help="kosu klasoru: her epok checkpoint.pt, sonda agent.pt ve results.json")
    ap.add_argument("--resume", type=int, default=0, help="1: --out'taki checkpoint.pt'den kaldigi epoktan surdur")
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(4)
        if os.name == "nt":
            print("guc kisitlamasi (EcoQoS) kapali:", _no_power_throttling(), flush=True)
    folder, prefix = args.root or os.path.join(MODEL_Z, "data", args.data), FILES[args.data]
    vocab_path, stories_path = os.path.join(folder, prefix + "vocab.json"), os.path.join(folder, prefix + "stories.jsonl")
    vocab = json.load(open(vocab_path, encoding="utf-8"))
    agent = ContextAgent(vocab, d=args.d, slots=args.slots, directions=args.directions).to(args.device)

    def build():
        stories = [json.loads(line) for line in open(stories_path, encoding="utf-8")]
        train, exam = [s for s in stories if s["split"] == "train"], [s for s in stories if s["split"] == "exam"]
        return dict(train=_encode(agent, train), exam=_encode(agent, exam),
                    exam_names=[s.get("country", i) for i, s in enumerate(exam)])
    data = _cached(os.path.join(folder, prefix + "stories.pt"),
                   (vocab_path, stories_path, __file__, os.path.join(HERE, "context.py")), build)
    ids, mask = (t.to(args.device) for t in data["train"])
    exam_ids, exam_mask = (t.to(args.device) for t in data["exam"])
    n_train = len(ids)
    real_bags = set()                                       # veride gecen butun cumle torbalari (tutarlilik olcusu)
    for a, m in (data["train"], data["exam"]):
        for n in range(len(a)):
            for t in range(a.shape[1]):
                if m[n, t].any():
                    real_bags.add(tuple(sorted((w, min(c, LEVELS - 1)) for w, c in
                                               zip(*torch.unique(a[n, t][m[n, t]], return_counts=True)) if w)))
    real_bags = {tuple((int(w), int(c)) for w, c in b) for b in real_bags}
    sent = mask.any(2)
    presence = torch.zeros(len(vocab), device=args.device)
    for t in range(ids.shape[1]):
        present = torch.zeros(n_train, len(vocab), device=args.device).scatter_(1, ids[:, t], mask[:, t].float())
        presence += (present * sent[:, t:t + 1]).sum(0)
    agent.set_prior(presence / sent.sum())
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr)
    gen = torch.Generator().manual_seed(args.seed)
    print("veri %s: egitim %d hikaye, sinav %d, sozluk %d, gercek torba %d | d %d, slots %d, directions %d, relax %g, "
          "lr %g %s, batch %d, %d parametre | cihaz %s" % (
              args.data, n_train, len(exam_ids), len(vocab), len(real_bags), args.d, args.slots, args.directions,
              args.relax, args.lr, args.schedule, args.batch, sum(p.numel() for p in agent.parameters()), args.device),
          flush=True)
    t0, history, first = time.time(), [], 1
    ckpt = os.path.join(args.out, "checkpoint.pt") if args.out else None
    if args.out:
        assert args.resume or not os.path.exists(ckpt), "kosu klasoru dolu (%s): yeni ad ya da --resume 1" % args.out
        os.makedirs(args.out, exist_ok=True)
    if args.resume:
        pack = torch.load(ckpt, map_location=args.device, weights_only=False)
        assert pack["vocab"] == vocab, "sozluk checkpoint'tekinden farkli"
        keys = ("data", "seed", "d", "slots", "directions", "relax", "batch", "lr", "schedule") + (
            ("epochs",) if args.schedule == "cosine" else ())
        diff = {k: (pack["args"].get(k), vars(args)[k]) for k in keys if pack["args"].get(k) != vars(args)[k]}
        assert not diff, "surdurme ayari checkpoint'ten farkli (checkpoint, simdi): %s" % diff
        agent.load_state_dict(pack["state"])
        opt.load_state_dict(pack["opt"])
        gen.set_state(pack["gen"].cpu())
        first, history = pack["epoch"] + 1, pack["history"]
        print("SURDURULDU: epok %d'den" % pack["epoch"], flush=True)
    for epoch in range(first, args.epochs + 1):
        perm = torch.randperm(n_train, generator=gen)
        total, t_epoch = 0.0, time.time()
        for b in range(0, n_train, args.batch):
            if args.schedule == "cosine":
                done = ((epoch - 1) * n_train + b) / (args.epochs * n_train)
                for group in opt.param_groups:
                    group["lr"] = args.lr * 0.5 * (1 + math.cos(math.pi * done))
            rows = perm[b:b + args.batch].to(args.device)
            loss = loss_of(agent, ids[rows], mask[rows], args.relax)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(rows)
        if epoch % args.every == 0 or epoch == args.epochs:
            agent.eval()
            res = evaluate(agent, exam_ids, exam_mask, real_bags, data["exam_names"],
                           show=SHOW if epoch == args.epochs else 0)
            agent.train()
            k = res["bag_in_top_k"]
            print("epok %3d  kayip %.3f  (%.0f sn, epok %.1f sn) | dogru torba ilk1 %.3f ilk5 %.3f ilk%d %.3f (taban onceki "
                  "%.3f) | tutarlilik ilk %.3f butun %.3f | calibration %.3f | yon: farkli %.2f expH %.1f olu %d" % (
                      epoch, total / n_train, time.time() - t0, time.time() - t_epoch, k["ilk1"], k["ilk5"],
                      args.directions, k["ilkK"], k["taban_onceki"], res["tutarlilik"]["ilk_aday"],
                      res["tutarlilik"]["butun_adaylar"], res["calibration"], res["direction_health"]["farkli_torba_orani"],
                      res["direction_health"]["exp_H_pi"], res["direction_health"]["olu_yon"]), flush=True)
            history.append(dict(epoch=epoch, loss=total / n_train, result=res))
        if ckpt:
            torch.save(dict(vocab=vocab, state=agent.state_dict(), opt=opt.state_dict(), gen=gen.get_state(),
                            epoch=epoch, history=history, args=vars(args)), ckpt + ".part")
            os.replace(ckpt + ".part", ckpt)
    if args.out:
        torch.save(dict(vocab=vocab, state=agent.state_dict(), args=vars(args)), os.path.join(args.out, "agent.pt"))
        json.dump(history, open(os.path.join(args.out, "results.json"), "w"), indent=1)
        print("kaydedildi:", args.out, flush=True)
    return agent


if __name__ == "__main__":
    main()
