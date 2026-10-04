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

import torch

from context import D, DIRECTIONS, LEVELS, SLOTS, ContextAgent

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_Z = os.path.dirname(os.path.dirname(HERE))
FILES = {"countries": "country_", "countries_fixed": "country_", "countries_orders": "country_"}   # data/<ad>/<onek>{stories.jsonl, vocab.json}
# veriye gore baslangic lr'si (kullanici, 4 Ekim: "hepsi cosine sadece başlangıç lr farklı veriye göre"); verilmeyende LR
DATA_LR = {"countries": 3e-3, "countries_fixed": 3e-3, "countries_orders": 3e-3}       # ulke: sabit 3e-3 600 adimda sinav ilk20 0,650 (olculdu)
RAMP = 0.5              # kademeli tamamlama: egitimin bu payinda eksik kelime 1'den butun torbaya cikar (0: kapali)
BATCH = 64              # hikaye
LR = 1e-3              # genel baslangic lr'si (gramer d 256 ile ayni); veriye ozel deger DATA_LR
SCHEDULE = "cosine"     # her veride cosine (kullanici, 4 Ekim: "hepsi cosine"); constant yalniz denemek icin
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


def loss_of(agent, ids, mask, relax, progress=1.0, ramp=0.0):
    """Hikaye batch'i -> ortalama kayip (gecis basina, nat): (1 - relax) * -log sum_k pi_k P(B | k) + relax * yon ortalamasi.
    Kademeli tamamlama (kullanici, 4 Ekim: "hedef cümle 10 kelime ise 9 nu verelim sadece 1 tanesini tahmin etsin. sonra 8 ni
    verelim 2 sini tahmin etsin"; "aynen bunu istiyorum"): asama a = progress / ramp; sonraki cumlenin kelimelerinden rastgele
    max(1, a * n) tanesi saklanir, gerisi verilir; ajan saklananlari bulur.  a >= 1: hepsi saklanir (uretimdeki gibi)."""
    a = min(progress / ramp, 1.0) if ramp > 0 else 1.0
    state = agent.initial_state(len(ids))
    LP, PI = [], []
    for t in range(ids.shape[1] - 1):
        state = agent.read(state, ids[:, t], mask[:, t])
        valid = mask[:, t + 1].any(1)
        if not valid.any():
            break
        nxt, nm = ids[:, t + 1], mask[:, t + 1]
        real = nm & (nxt != 0)
        hide = (real.sum(1, keepdim=True).float() * a).ceil().clamp(min=1)
        rank = torch.rand(nxt.shape, device=nxt.device).masked_fill(~real, 2.0).argsort(1).argsort(1)
        hidden = real & (rank < hide)
        pool = _counts(nxt, real & ~hidden, len(agent.vocab))                       # verilen kisim
        words, c = _bag(nxt, hidden, len(agent.vocab))                               # eksik kisim
        log_pi, lp = agent.bag_log_prob(state, words, c, pool)
        LP.append(lp[valid])
        PI.append(log_pi[valid])
    lp, log_pi = torch.cat(LP), torch.cat(PI)
    return ((1 - relax) * -(log_pi + lp).logsumexp(1) + relax * -lp.mean(1)).mean()


def _bag_hash(counts, weights):
    """Sayi vektorleri (..., V) -> torba ozeti (...,) int64: sum_w min(c_w, 3) * r_w (r_w rastgele; carpisma ~0)."""
    return (counts.clamp(max=LEVELS - 1).long() * weights).sum(-1)


def _counts(ids, mask, V):
    """Cumleler (B, W) -> sayi vektoru (B, V); <unk> sayilmaz."""
    c = torch.zeros(len(ids), V, device=ids.device).scatter_add_(1, ids, mask.float())
    c[:, 0] = 0
    return c


def evaluate(agent, ids, mask, real_hash, weights, names, show=SHOW):
    """-> bag_in_top_k (ilk 1 / 5 / K, taban: onceki cumle), tutarlilik, calibration, direction_health; goz satirlari.
    Toplu: ayni torbayi veren yonler ozetle birlesir, olasiliklari toplanir.  Hikayede daha once gecmis torba aday sayilmaz
    (kullanici, 4 Ekim: "birebir aynı torba olmadığı sürece bence sıkıntı yok aynı şeyin farklı ifade edilmesinde")."""
    K, V = agent.directions, len(agent.vocab)
    acc = torch.zeros(9, device=ids.device)          # n, ilk1, ilk5, ilkK, taban, tutarli ilk, tutarli hepsi, aday, expH
    bins = torch.zeros(3, 5, device=ids.device)      # olasilik toplami, aday, dogru (5 dilim)
    pi_max = torch.zeros(K, device=ids.device)
    eye = []
    earlier = torch.ones(K, K, device=ids.device).tril(-1).bool()
    with torch.no_grad():
        for b0 in range(0, len(ids), 256):
            bi, bm = ids[b0:b0 + 256], mask[b0:b0 + 256]
            state = agent.initial_state(len(bi))
            said = []                                                                 # hikayede gecmis torbalar
            for t in range(bi.shape[1] - 1):
                state = agent.read(state, bi[:, t], bm[:, t])
                said.append(_bag_hash(_counts(bi[:, t], bm[:, t], V), weights))
                counts, logp = agent.next_bags(state)
                hk = _bag_hash(counts, weights)
                logp = logp.masked_fill((hk[:, :, None] == torch.stack(said, 1)[:, None, :]).any(-1), -1e9)
                pi_max = torch.maximum(pi_max, agent._heads(state)[0].exp().max(0).values)
                valid = bm[:, t + 1].any(1)
                true = _bag_hash(_counts(bi[:, t + 1], bm[:, t + 1], V), weights)     # (B,)
                prev = _bag_hash(_counts(bi[:, t], bm[:, t], V), weights)
                same = hk[:, :, None] == hk[:, None, :]                              # (B, K, K)
                p = logp.exp()
                merged = (same * p[:, None, :]).sum(-1)                               # ayni torbanin toplam olasiligi
                first = ~(same & earlier).any(-1) & (logp > -1e8)                     # ilk gorulen yon; tekrar torba yok
                score = torch.where(first, merged, torch.full_like(merged, -1.0))
                order = score.argsort(1, descending=True)
                ranked_h = hk.gather(1, order)
                ranked_first = first.gather(1, order)
                hit = (ranked_h == true[:, None]) & ranked_first                     # (B, K) sirali
                real = torch.isin(hk, real_hash) & first
                n_first = first.sum(1).float()
                q = torch.where(first, merged, torch.zeros_like(merged))
                q = q / q.sum(1, keepdim=True).clamp_min(1e-30)
                ent = (-(q * q.clamp_min(1e-30).log()).sum(1)).exp()
                v = valid.float()
                acc += torch.stack([v.sum(), (hit[:, :1].any(1) * v).sum(), (hit[:, :5].any(1) * v).sum(),
                                    (hit.any(1) * v).sum(), ((prev == true) * v).sum(),
                                    (real.gather(1, order[:, :1]).squeeze(1) * v).sum(), (real.sum(1) * v).sum(),
                                    (n_first * v).sum(), (ent * v).sum()])
                correct = (hk == true[:, None]) & first
                dil = (merged * 5).long().clamp(0, 4)
                w = (first & valid[:, None]).float()
                for j, val in enumerate((merged * w, w, correct.float() * w)):
                    bins[j].scatter_add_(0, dil.flatten(), val.flatten())
                for r in range(min(len(bi), max(show - b0, 0))):
                    if valid[r]:
                        top = [(counts[r, order[r, j]].clone(), merged[r, order[r, j]].item()) for j in range(K)
                               if first[r, order[r, j]]][:3]
                        eye.append((b0 + r, t, bi[r, t][bm[r, t]].tolist(), top, bi[r, t + 1][bm[r, t + 1]].tolist()))
    n, acc = acc[0].item(), acc.tolist()
    calib = ((bins[0] - bins[2]).abs().sum() / bins[1].sum().clamp_min(1)).item()
    out = dict(bag_in_top_k=dict(ilk1=round(acc[1] / n, 4), ilk5=round(acc[2] / n, 4), ilkK=round(acc[3] / n, 4),
                                 taban_onceki=round(acc[4] / n, 4)),
               tutarlilik=dict(ilk_aday=round(acc[5] / n, 4), butun_adaylar=round(acc[6] / max(acc[7], 1), 4)),
               calibration=round(calib, 4),
               direction_health=dict(farkli_torba_orani=round(acc[7] / n / K, 4), exp_H_pi=round(acc[8] / n, 2),
                                     olu_yon=int((pi_max < 0.01).sum())))
    bag_text = lambda c: " ".join(agent.vocab[w] + ("x%d" % c[w] if c[w] > 1 else "") for w in c.nonzero().flatten().tolist())
    for s_, t, read, top, real in eye:
        if t == 0:
            print("   --- %s" % names[s_])
        print("   okunan: %s" % " ".join(agent.vocab[i] for i in read))
        for c, prob in top:
            print("      %.3f  {%s}" % (prob, bag_text(c)))
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
    ap.add_argument("--lr", type=float, default=None, help="baslangic lr'si; verilmezse veriye gore (DATA_LR) ya da LR")
    ap.add_argument("--schedule", default=SCHEDULE, choices=("constant", "cosine"))
    ap.add_argument("--ramp", type=float, default=RAMP, help="kademeli tamamlama: egitimin bu payinda eksik 1 kelimeden hepsine")
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--every", type=int, default=10, help="kac epokta bir olcum")
    ap.add_argument("--out", default=None, help="kosu klasoru: her epok checkpoint.pt, sonda agent.pt ve results.json")
    ap.add_argument("--resume", type=int, default=0, help="1: --out'taki checkpoint.pt'den kaldigi epoktan surdur")
    args = ap.parse_args(argv)
    args.lr = args.lr if args.lr is not None else DATA_LR.get(args.data, LR)
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
    weights = torch.randint(1, 2 ** 62, (len(vocab),), generator=torch.Generator().manual_seed(0)).to(args.device)
    real_hash = torch.cat([_bag_hash(_counts(a[:, t], m[:, t], len(vocab)), weights)[m[:, t].any(1)]
                           for a, m in ((ids, mask), (exam_ids, exam_mask)) for t in range(a.shape[1])]).unique()
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
              args.data, n_train, len(exam_ids), len(vocab), len(real_hash), args.d, args.slots, args.directions,
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
        keys = ("data", "seed", "d", "slots", "directions", "relax", "batch", "lr", "schedule", "ramp", "epochs")
        diff = {k: (pack["args"].get(k), vars(args)[k]) for k in keys if pack["args"].get(k) != vars(args)[k]}
        assert not diff, "surdurme ayari checkpoint'ten farkli (checkpoint, simdi): %s" % diff
        agent.load_state_dict(pack["state"])
        opt.load_state_dict(pack["opt"])
        gen.set_state(pack["gen"].cpu())
        first, history = pack["epoch"] + 1, pack["history"]
        print("SURDURULDU: epok %d'den" % pack["epoch"], flush=True)
    for epoch in range(first, args.epochs + 1):
        perm = torch.randperm(n_train, generator=gen)
        total, t_epoch = torch.zeros((), device=args.device), time.time()
        for b in range(0, n_train, args.batch):
            if args.schedule == "cosine":
                done = ((epoch - 1) * n_train + b) / (args.epochs * n_train)
                for group in opt.param_groups:
                    group["lr"] = args.lr * 0.5 * (1 + math.cos(math.pi * done))
            rows = perm[b:b + args.batch].to(args.device)
            progress = ((epoch - 1) * n_train + b) / (args.epochs * n_train)
            loss = loss_of(agent, ids[rows], mask[rows], args.relax, progress, args.ramp)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.detach() * len(rows)
        total = total.item()                             # adim basina .item() yok: GPU her adimda beklenmez
        if epoch % args.every == 0 or epoch == args.epochs:
            agent.eval()
            t_train = time.time() - t_epoch
            res = evaluate(agent, exam_ids, exam_mask, real_hash, weights, data["exam_names"],
                           show=SHOW if epoch == args.epochs else 0)
            agent.train()
            k = res["bag_in_top_k"]
            print("epok %3d  kayip %.3f  (%.0f sn; egitim %.1f sn/epok, olcum %.1f sn) | dogru torba ilk1 %.3f ilk5 %.3f ilk%d %.3f (taban onceki "
                  "%.3f) | tutarlilik ilk %.3f butun %.3f | calibration %.3f | yon: farkli %.2f expH %.1f olu %d" % (
                      epoch, total / n_train, time.time() - t0, t_train, time.time() - t_epoch - t_train,
                      k["ilk1"], k["ilk5"],
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
