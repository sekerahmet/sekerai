"""train_grammar -- gramer ajaninin egitimi ve olcumu (ajan: grammar.py).

Kayip: ardil (satir) + oncel (sutun) CE; ayni kelimenin kopyalari esdeger hedef.  Gorev ve yuva etiketi yok.
Olculer (sinav bolmeleri ve cumle boyu ayri): tam dogru (ayni torbadan kurulabilen herhangi gecerli cumle), dogru yuva,
dogru komsu, deneme sayisi (Gumbel gurultulu G ile dogru bulunana kadar, en cok TRIES), cumle basina dizme suresi.

    python train_grammar.py [--data countries|simplestories] [--epochs 30] [--device cpu|cuda] [--d 64] [--lr 3e-3]
                            [--schedule constant|cosine] [--compile 1] [--precision bf16] [--out klasor] [--resume 1]
                            [--missing 1 --drop 0.25 --add 0.25]
"""
import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict

import numpy as np
import torch

from grammar import D, NEG, UNK, GrammarAgent, is_complete, order_alternatives, order_by_relation, relation_matrix

MODEL_Z = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILES = {"countries": "country_", "simplestories": "ss_"}   # data/<ad>/<onek>{train,exam,sentences}.jsonl
BATCH = 64              # tam SS gibi buyuk veride Colab hucresi 1024 verir
LR = 3e-3               # Adam; d=256'da bu degerle 6. epoktan sonra dagildi
SCHEDULE = "constant"   # constant | cosine (adim adim --epochs sonunda 0)
UNK_RATE = 0.1          # egitimde kelimenin <unk> yapilma olasiligi: bilinmeyen kelimeye yer bulmayi da ogrensin
ALTERNATIVES = 5        # top_k: dogru cumle ilk bu kadar aday icinde mi (order_alternatives)
TRIES = 100             # deneme sayisi olcusunun ust siniri
DROP_RATE = 0.25        # --missing 1: torbadan kelime cikarma olasiligi (olculmedi)
ADD_RATE = 0.25         # --missing 1: torbaya baska cumleden kelime ekleme olasiligi (olculmedi)
LENGTH_BANDS = ((1, 10), (11, 20), (21, 1000))   # olculer cumle boyuna gore de


def _no_power_throttling():
    """Windows: bu surecin guc kisitlamasini (EcoQoS) kapat; arka plandaki surec ~10 kat yavasliyordu (train_context ile
    ayni; belge/model_z_temel/06 H1)."""
    import ctypes
    from ctypes import wintypes

    class State(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]
    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    s = State(1, 0x1, 0)                                 # ProcessPowerThrottling, EXECUTION_SPEED denetimi, kapali
    return bool(k.SetProcessInformation(k.GetCurrentProcess(), 4, ctypes.byref(s), ctypes.sizeof(s)))


def _load(path, required=True):
    if not os.path.exists(path):
        assert not required, "veri dosyasi yok: %s (--root dogru mu?)" % path
        return []
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def _encode(agent, rows):
    """Cumleler -> bir kez: ids (N, L) dolgulu, boy (N,), kelimeler."""
    L = max(len(r["words"]) for r in rows)
    ids = torch.zeros(len(rows), L, dtype=torch.long)
    for b, r in enumerate(rows):
        ids[b, :len(r["words"])] = torch.tensor(agent.ids(r["words"]))
    return dict(ids=ids, length=torch.tensor([len(r["words"]) for r in rows]), words=[r["words"] for r in rows])


def _batch(enc, rows, gen):
    """Satirlar -> karisik torbalar (sinav).  order[b, i] = torbadaki i. kelimenin cumledeki konumu."""
    L = enc["ids"].shape[1]
    mask = torch.arange(L)[None] < enc["length"][rows][:, None]
    order = torch.rand(len(rows), L, generator=gen).masked_fill(~mask, 2.0).argsort(1)
    return enc["ids"][rows].gather(1, order), mask, order


def loss_of(agent, enc, rows, gen, unk_rate, forward=None, bf16=False, L=None, drop=0.0, add=0.0):
    """Egitim torbasi cumlenin kendisi: okuyucu konumsuz, G kelimelerin veriliş sirasiyla birlikte permute olur, kayip
    sirasizdir; karistirmak bir sey degistirmez.  Hedefler sabit: ardil i+1, oncel i-1, boundary (indeks L) ilk / son
    kelimeye.  enc (flat: cumleler uc uca, start, length), rows ve gen ajanin cihazinda; dolgu batch'in en uzun cumlesine
    (8'e yuvarli) kirpilir.  forward: agent'in derlenmis hali (torch.compile) ya da agent; bf16: okuyucu autocast (GPU; G
    hep fp32).  L: dolgu boyu, CPU'dan (verilmezse cihazdan okunur: GPU'yu bekletir).
    Bozuk torba (missing'li ajan): drop olasiligiyla cumleden 1..n-1 kelime cikar, gercek komsusu cikan kelimenin hedefi
    missing; add olasiligiyla batch'teki baska cumleden, bu cumlede olmayan 1-2 kelime eklenir, ardili ve onceli missing."""
    dev = next(agent.parameters()).device
    B = len(rows)
    length = enc["length"][rows]
    L = L or -(-int(length.max()) // 8) * 8            # 8'in kati: compile her boyda yeniden derlemesin
    pos = torch.arange(L, device=dev)[None]
    mask = pos < length[:, None]
    flat = enc["flat"]
    orig = flat[(enc["start"][rows][:, None] + pos).clamp(max=len(flat) - 1)].long().masked_fill(~mask, 0)
    keep = mask
    if drop > 0:
        cut = (torch.rand(B, generator=gen, device=dev) < drop) & (length > 1)
        k = (torch.rand(B, generator=gen, device=dev) * (length - 1).float()).long() + 1          # 1..n-1 kelime
        rank = torch.rand(B, L, generator=gen, device=dev).masked_fill(~mask, 2.0).argsort(1).argsort(1)
        keep = mask & ~(cut[:, None] & (rank < k[:, None]))
    A = 2 if add > 0 else 0                             # eklenen kelime yuvasi
    if A:
        other = (torch.arange(B, device=dev) + 1) % B
        at = (torch.rand(B, A, generator=gen, device=dev) * length[other][:, None].float()).long()
        extra = orig[other].gather(1, at)
        count = (torch.rand(B, generator=gen, device=dev) < add).long() * (
            1 + (torch.rand(B, generator=gen, device=dev) < 0.5).long())                         # 0, 1 ya da 2
        extra_ok = (torch.arange(A, device=dev)[None] < count[:, None]) & ~(
            orig.masked_fill(~mask, -1)[:, :, None] == extra[:, None, :]).any(1)
    else:
        extra, extra_ok = orig[:, :0], mask[:, :0]
    bound, miss = L + A, L + A + 1                      # dugum indeksleri (missing'siz ajanda miss hic hedef olmaz)
    nxt = torch.cat([keep[:, 1:], keep[:, :1] & False], 1)
    prv = torch.cat([keep[:, :1] & False, keep[:, :-1]], 1)
    last = keep.gather(1, (length - 1)[:, None])[:, 0]
    # olmayan yuva (dolgu, cikan kelime) kayba girmez; hedefi boundary (gecerli indeks)
    succ = torch.cat([torch.where(keep, torch.where(pos + 1 < length[:, None], torch.where(nxt, pos + 1, miss), bound),
                                  bound),
                      torch.where(extra_ok, miss, bound),
                      torch.where(keep[:, 0], 0, miss)[:, None]], 1)
    pred = torch.cat([torch.where(keep, torch.where(pos > 0, torch.where(prv, pos - 1, miss), bound), bound),
                      torch.where(extra_ok, miss, bound),
                      torch.where(last, length - 1, miss)[:, None]], 1)
    present = torch.cat([keep, extra_ok], 1)
    words = torch.cat([orig, extra], 1)
    ids = words.masked_fill((torch.rand(words.shape, generator=gen, device=dev) < unk_rate) & present, agent.index[UNK])
    key = torch.cat([words.masked_fill(~present, -5), torch.full((B, 1), -7, device=dev)]
                    + ([torch.full((B, 1), -9, device=dev)] if agent.missing is not None else []), 1)
    same = key[:, :, None] == key[:, None, :]           # ayni kelimenin kopyalari esdeger hedef
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bf16):
        G = (agent if forward is None else forward)(ids, present)
    G = G.float()
    L1, R = G.shape[1], succ.shape[1]                   # R: kelime yuvalari + boundary (missing'in hedefi yok)
    node = torch.cat([present, torch.ones(B, 1, dtype=torch.bool, device=dev)], 1).float()
    hit_s = same.gather(1, succ[:, :, None].expand(-1, -1, L1))     # j, gercek ardille ayni kelime
    hit_p = same.gather(1, pred[:, :, None].expand(-1, -1, L1))
    ls = G[:, :R].log_softmax(2).masked_fill(~hit_s, NEG).logsumexp(2)                    # ardil (satir)
    lp = G.log_softmax(1).transpose(1, 2)[:, :R].masked_fill(~hit_p, NEG).logsumexp(2)   # oncel (sutun)
    return -((ls + lp) * node).sum() / node.sum()


def evaluate(agent, enc, valid, seed=0, k_best=ALTERNATIVES):
    """-> butun ve cumle boyu bantlari: tam dogru, ilk k_best aday icinde (top_k), dogru yuva, dogru komsu, deneme (en cok
    TRIES; bulunamayan TRIES + 1), dizme ms.  k_best 0: top_k yok.  Butun sinav parca parca tek ileri hesapla."""
    gen = torch.Generator().manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    dev = next(agent.parameters()).device
    rows = torch.arange(len(enc["words"]))
    stats = defaultdict(Counter)
    gate = agent.missing is not None
    t_order = 0.0
    for c in range(0, len(rows), 512):
        part = rows[c:c + 512]
        ids, mask, order = _batch(enc, part, gen)
        with torch.no_grad():
            G = agent(ids.to(dev), mask.to(dev)).cpu().double().numpy()
        L = ids.shape[1]
        for b, r in enumerate(part.tolist()):
            words = enc["words"][r]
            n = len(words)
            bag = [words[o] for o in order[b, :n].tolist()]
            idx = list(range(n)) + [L]
            full = G[b][np.ix_(idx + [L + 1], idx + [L + 1])] if gate else None
            g = G[b][np.ix_(idx, idx)]
            if gate:
                # kapi: gercek torba gecmeli; bozuk torba (bir kelime cikmis / baska sinav cumlesinden bir kelime
                # eklenmis; gecerli bir cumle torbasi olursa sayilmaz) elenmeli
                stats["all"]["real_n"] += 1
                stats["all"]["real_ok"] += is_complete(full, order_by_relation(g))
                other = enc["words"][(r + 1) % len(enc["words"])]
                broken = []
                if n > 1:
                    j = int(np_rng.integers(n))
                    broken.append(words[:j] + words[j + 1:])
                foreign = [w for w in other if w not in words]
                if foreign:
                    broken.append(words + [foreign[int(np_rng.integers(len(foreign)))]])
                for x in broken:
                    if tuple(sorted(x)) in valid:
                        continue
                    gx = relation_matrix(agent, x).double().numpy()
                    stats["all"]["broken_n"] += 1
                    stats["all"]["broken_out"] += not is_complete(gx, order_by_relation(gx[:len(x) + 1, :len(x) + 1]))
            t = time.perf_counter()
            built = tuple(bag[i] for i in order_by_relation(g))
            t_order += time.perf_counter() - t
            options = valid.get(tuple(sorted(words)), {tuple(words)})
            best = max(options, key=lambda o: sum(a == x for a, x in zip(built, o)))
            top = built in options or (k_best > 0 and any(tuple(bag[i] for i in seq) in options
                                                          for seq in order_alternatives(g, k_best, bag)[1:]))
            k, found = 1, built in options
            noise = np_rng.gumbel(size=(TRIES,) + g.shape) if not found else None
            while not found and k < TRIES:
                found = tuple(bag[i] for i in order_by_relation(g + noise[k])) in options
                k += 1
            band = next(f"{lo}-{hi}" if hi < 1000 else f"{lo}+" for lo, hi in LENGTH_BANDS if lo <= n <= hi)
            for key in ("all", band):
                s = stats[key]
                s["n"] += 1
                s["exact"] += built in options
                s["top_k"] += top
                s["slot"] += sum(a == x for a, x in zip(built, best))
                s["slots"] += n
                s["pairs"] += sum((Counter(zip(built, built[1:])) & Counter(zip(best, best[1:]))).values())
                s["n_pairs"] += n - 1
                s["tries"] += k if found else TRIES + 1
    out = {}
    for key, s in stats.items():
        out[key] = dict(n=s["n"], exact=round(s["exact"] / s["n"], 4),
                        top_k=round(s["top_k"] / s["n"], 4) if k_best else None, slot=round(s["slot"] / s["slots"], 4),
                        neighbor=round(s["pairs"] / max(s["n_pairs"], 1), 4), tries=round(s["tries"] / s["n"], 2))
    out["all"]["ms"] = round(1000 * t_order / max(len(rows), 1), 3)
    if gate:
        s = stats["all"]
        out["all"]["complete_real"] = round(s["real_ok"] / s["real_n"], 4)
        out["all"]["rejected_broken"] = round(s["broken_out"] / max(s["broken_n"], 1), 4)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="countries", choices=sorted(FILES))
    ap.add_argument("--root", default=None, help="veri klasoru (varsayilan data/<data>; SS: Drive'daki hazir dosyalar)")
    ap.add_argument("--epochs", type=int, default=30)          # ulke verisinde 60 ile ayni sonuc (olculdu)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=D, help="kelime temsili boyu")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--schedule", default=SCHEDULE, choices=("constant", "cosine"))
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--compile", type=int, default=1, help="1: torch.compile (yalniz cuda; CPU'da kendiliginden kapali)")
    ap.add_argument("--precision", default="bf16", choices=("bf16", "fp32"), help="egitim ileri hesabi (yalniz cuda); "
                    "sinav hep fp32")
    ap.add_argument("--alternatives", type=int, default=ALTERNATIVES, help="top_k olcusu: ilk kac aday")
    ap.add_argument("--missing", type=int, default=0, help="1: missing dugumu + bozuk torbalarla egitim (is_complete kapisi)")
    ap.add_argument("--drop", type=float, default=DROP_RATE, help="--missing 1: kelime cikarma olasiligi")
    ap.add_argument("--add", type=float, default=ADD_RATE, help="--missing 1: kelime ekleme olasiligi")
    ap.add_argument("--every", type=int, default=10, help="kac epokta bir olcum")
    ap.add_argument("--out", default=None, help="kosu klasoru: her epok checkpoint.pt (ajan, optimizer, epok, rastgelelik), "
                    "sonda agent.pt; olcumler results.json")
    ap.add_argument("--resume", type=int, default=0, help="1: --out'taki checkpoint.pt'den kaldigi epoktan surdur")
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(4)                        # kucuk model: 16 is parcacigi 4'ten yavas (olculdu)
        if os.name == "nt":
            print("guc kisitlamasi (EcoQoS) kapali:", _no_power_throttling(), flush=True)
    folder, prefix = args.root or os.path.join(MODEL_Z, "data", args.data), FILES[args.data]
    exam = _load(os.path.join(folder, prefix + "exam.jsonl"))
    valid = defaultdict(set)                            # ayni torbadan kurulabilen gecerli cumleler
    if os.path.exists(os.path.join(folder, prefix + "train_ids.npy")):
        # buyuk veri (make_ss_sentences): sozluk, cumleler uc uca kimlik dizisi; gecerli siralar sinav satirinda
        vocab = json.load(open(os.path.join(folder, prefix + "vocab.json"), encoding="utf-8"))
        assert vocab[0] == UNK
        offsets = torch.from_numpy(np.load(os.path.join(folder, prefix + "train_offsets.npy")))
        flat = torch.from_numpy(np.load(os.path.join(folder, prefix + "train_ids.npy")))
        for r in exam:
            valid[tuple(sorted(r["words"]))] |= {tuple(o) for o in r["valid"]}
    else:
        train = _load(os.path.join(folder, prefix + "train.jsonl"))
        for r in _load(os.path.join(folder, prefix + "sentences.jsonl"), required=False) or train + exam:
            valid[tuple(sorted(r["words"]))].add(tuple(r["words"]))
        vocab = [UNK] + sorted({w for r in train for w in r["words"]})
        index = {w: i for i, w in enumerate(vocab)}
        flat = torch.tensor([index[w] for r in train for w in r["words"]], dtype=torch.int32)
        offsets = torch.tensor([0] + [len(r["words"]) for r in train]).cumsum(0)
    n_train = len(offsets) - 1
    agent = GrammarAgent(vocab, d=args.d, missing=bool(args.missing)).to(args.device)
    drop, add = (args.drop, args.add) if args.missing else (0.0, 0.0)
    cuda = torch.device(args.device).type == "cuda"
    forward = torch.compile(agent, dynamic=True) if (cuda and args.compile) else agent
    bf16 = cuda and args.precision == "bf16"
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr)
    splits = {s: _encode(agent, [r for r in exam if r["split"] == s]) for s in sorted({r["split"] for r in exam})}
    length_host = (offsets[1:] - offsets[:-1]).long()   # dolgu boyu CPU'dan: her adimda GPU beklenmez
    enc = dict(flat=flat.to(args.device), start=offsets[:-1].long().to(args.device),
               length=length_host.to(args.device))      # egitim verisi bir kez cihazda
    gen = torch.Generator().manual_seed(args.seed)                     # epok sirasi (CPU)
    gen_unk = torch.Generator(device=args.device).manual_seed(args.seed)   # <unk> secimi (cihazda)
    print("veri %s: egitim %d cumle, sozluk %d, sinav %s, en uzun %d kelime | d %d, lr %g %s, batch %d, %d parametre | "
          "cihaz %s compile %s %s | missing %s" % (
              args.data, n_train, len(vocab), {s: len(e["words"]) for s, e in splits.items()}, int(length_host.max()),
              args.d, args.lr, args.schedule, args.batch, sum(p.numel() for p in agent.parameters()), args.device,
              forward is not agent, "bf16" if bf16 else "fp32",
              "drop %g add %g" % (drop, add) if args.missing else "yok"), flush=True)
    t0 = time.time()
    res, history, first = None, [], 1
    ckpt = os.path.join(args.out, "checkpoint.pt") if args.out else None
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    if args.resume:
        pack = torch.load(ckpt, map_location=args.device, weights_only=False)
        assert pack["vocab"] == vocab, "sozluk checkpoint'tekinden farkli: ayni veriyle surdurulur"
        # surdurme ayni tarifle: farkli ayar sessizce yok sayilmasin (lr optimizer durumundan gelir)
        keys = ("data", "seed", "d", "batch", "lr", "schedule", "precision", "missing", "drop", "add") + (
            ("epochs",) if args.schedule == "cosine" else ())    # cosine'in bitisi --epochs: uzatma zamanlamayi degistirir
        old = {k: pack["args"].get(k, ap.get_default(k)) for k in keys}     # eski checkpoint'te olmayan ayar: varsayilan
        diff = {k: (old[k], vars(args)[k]) for k in keys if old[k] != vars(args)[k]}
        assert not diff, "surdurme ayari checkpoint'ten farkli (checkpoint, simdi): %s" % diff
        agent.load_state_dict(pack["state"])
        opt.load_state_dict(pack["opt"])
        gen.set_state(pack["gen"].cpu())                 # map_location RNG durumunu da cihaza tasir; set_state CPU ister
        gen_unk.set_state(pack["gen_unk"].cpu())
        first, history = pack["epoch"] + 1, pack["history"]
        print("SURDURULDU: epok %d'den (%s)" % (pack["epoch"], ckpt), flush=True)
    for epoch in range(first, args.epochs + 1):
        perm_host = torch.randperm(n_train, generator=gen)
        perm = perm_host.to(args.device)
        total, t_epoch = torch.zeros((), device=args.device), time.time()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        for b in range(0, n_train, args.batch):
            if args.schedule == "cosine":                     # adim epok ve batch'ten: surdurmede ayni lr
                done = ((epoch - 1) * n_train + b) / (args.epochs * n_train)
                for group in opt.param_groups:
                    group["lr"] = args.lr * 0.5 * (1 + math.cos(math.pi * done))
            rows = perm[b:b + args.batch]
            L = -(-int(length_host[perm_host[b:b + args.batch]].max()) // 8) * 8
            loss = loss_of(agent, enc, rows, gen_unk, UNK_RATE, forward, bf16, L, drop, add)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.detach() * len(rows)                # .item() yok: her adimda GPU beklenmez
        total = total.item()
        if cuda:
            torch.cuda.synchronize()
        perf = dict(train_secs=round(time.time() - t_epoch, 2), sentences_per_sec=round(n_train / (time.time() - t_epoch)),
                    gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else None)
        if epoch % args.every == 0 or epoch == args.epochs:
            agent.eval()
            t_eval = time.time()
            k_best = args.alternatives if epoch == args.epochs else 0     # top_k yavas (~50 ms / yanlis cumle): yalniz sonda
            res = {s: evaluate(agent, e, valid, k_best=k_best) for s, e in splits.items()}
            perf["eval_secs"] = round(time.time() - t_eval, 1)
            agent.train()
            print("epok %3d  kayip %.3f  (%.0f sn) | egitim %.1f sn/epok, %d cumle/sn%s | sinav %.1f sn" % (
                epoch, total / n_train, time.time() - t0, perf["train_secs"], perf["sentences_per_sec"],
                (", gpu tepe %.2f GB" % perf["gpu_peak_gb"]) if cuda else "", perf["eval_secs"]), flush=True)
            for s, v in res.items():
                print("   %-8s %s" % (s, "  ".join(
                    "%s: n %d tam %.3f ilk%d %s yuva %.3f komsu %.3f deneme %.1f%s" % (
                        k, x["n"], x["exact"], args.alternatives, "-" if x["top_k"] is None else "%.3f" % x["top_k"], x["slot"], x["neighbor"], x["tries"],
                        (" %.2f ms" % x["ms"]) if "ms" in x else "") + (
                        " | kapi: complete_real %.3f rejected_broken %.3f" % (x["complete_real"], x["rejected_broken"])
                        if "complete_real" in x else "") for k, x in v.items())), flush=True)
            history.append(dict(epoch=epoch, loss=total / n_train, result=res, perf=perf))
        if ckpt:                                         # her epok: once .part, sonra yerine (yarim dosya kalmaz)
            torch.save(dict(vocab=vocab, state=agent.state_dict(), opt=opt.state_dict(), gen=gen.get_state(),
                            gen_unk=gen_unk.get_state(), epoch=epoch, history=history, args=vars(args)), ckpt + ".part")
            os.replace(ckpt + ".part", ckpt)
    if args.out:
        torch.save(dict(vocab=vocab, state=agent.state_dict(), args=vars(args)), os.path.join(args.out, "agent.pt"))
        json.dump(history, open(os.path.join(args.out, "results.json"), "w"), indent=1)
        print("kaydedildi:", args.out, flush=True)
    return agent, res


if __name__ == "__main__":
    main()
