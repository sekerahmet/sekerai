"""train_context -- baglam ajaninin egitimi ve olcumu (ajan: context.py; matematik: belge/model_z_temel/07, 09).

Kayip: sonraki cumlenin gercek torbasinin kesin karisim olasiligi, -log sum_k pi_k P(B | k); olu yon icin RELAX:
(1 - relax) * karisim + relax * yonlerin ortalamasi (her yon biraz ogrenir; Rupprecht ve ark. MHP).  Etiket yok.
Olculer (sinav hikayeleri, her cumle gecisi; 07 §7):
    bag_in_top_k     gercek torba (kelime + sayi, 3+ birlesik) ilk 1 / 5 / K aday icinde mi; taban: onceki cumlenin torbasi
    tutarlilik       aday torbalarin kaci veride gecen gercek bir cumle torbasi (ilk aday / butun adaylar)
    calibration      adaylarin olasiligi ile gercekten dogru cikma sikligi (5 dilim, ortalama sapma)
    direction_health farkli torba / K, exp H(pi), olu yon (sinavda pi hic 0,01'i gecmeyen)
Secimde kapi (--grammar; kullanici, 4 Ekim: "Tamam son karar eğitimden sonra"): grammar agent'in is_complete'inden
gecemeyen aday, hikayede gecmis torba gibi aday sayilmaz; butun olculer kalan adaylarla.  Egitim kapisizdir.
Goz: birkac sinav hikayesinde her cumleden sonra ilk 3 aday torba ve gercek sonraki cumle.

    python train_context.py [--data countries] [--epochs 30] [--device cpu|cuda] [--grammar agent.pt] [--out klasor]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

from context import D, DIRECTIONS, LEVELS, SLOTS, ContextAgent

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "meaning"))
from meaning import build_neighbor_table, shortlist  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_Z = os.path.dirname(os.path.dirname(HERE))
FILES = {"countries": "country_"}   # data/<ad>/<onek>{stories.jsonl, vocab.json}
# veriye gore baslangic lr'si (kullanici, 4 Ekim: "hepsi cosine sadece başlangıç lr farklı veriye göre"); verilmeyende LR
DATA_LR = {"countries": 3e-3}       # ulke: sabit 3e-3 600 adimda sinav ilk20 0,650 (olculdu)
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


def _judge(gate, h, counts):
    """Yon torbalari -> kapidan gecemeyen mi (h'nin bicimi, bool).  h torba ozetleri, counts (..., V) sayilari.  Her farkli
    torba bir kez gramerden gecer (donuk grammar agent: dizme + is_complete); sonuc ozetle saklanir.  Bos torba gecemez."""
    flat, V = h.flatten(), counts.shape[-1]
    known, ok = gate["known"], gate["ok"]
    found = torch.zeros_like(flat, dtype=torch.bool)
    if len(known):
        found = known[torch.searchsorted(known, flat).clamp(max=len(known) - 1)] == flat
    new, inv = flat[~found].unique(return_inverse=True)
    if len(new):
        at = torch.zeros(len(new), dtype=torch.long, device=flat.device).scatter_(
            0, inv, (~found).nonzero().flatten())                      # her yeni torbanin bir gorulen yeri
        rows = counts.reshape(-1, V)[at].long().cpu()
        g, Gm = gate["grammar"], gate["module"]
        bags = [[gate["vocab"][w] for w in r.nonzero().flatten().tolist() for _ in range(int(r[w]))] for r in rows]
        L = max(1, max(len(b) for b in bags))
        dev = next(g.parameters()).device
        gid = torch.zeros(len(bags), L, dtype=torch.long)
        gm = torch.zeros(len(bags), L, dtype=torch.bool)
        for i, b in enumerate(bags):
            gid[i, :len(b)] = torch.tensor(g.ids(b), dtype=torch.long)
            gm[i, :len(b)] = True
        with torch.no_grad():
            G = g(gid.to(dev), gm.to(dev)).double().cpu().numpy()
        res = []
        for i, b in enumerate(bags):
            n = len(b)
            if n == 0:
                res.append(False)
                continue
            idx = list(range(n)) + [L, L + 1]
            full = G[i][np.ix_(idx, idx)]
            res.append(Gm.is_complete(full, Gm.order_by_relation(full[:n + 1, :n + 1])))
        known = torch.cat([known, new])
        ok = torch.cat([ok, torch.tensor(res, device=ok.device)])
        order = known.argsort()
        gate["known"], gate["ok"] = known[order], ok[order]
        known, ok = gate["known"], gate["ok"]
    return ~ok[torch.searchsorted(known, flat)].view(h.shape)


def loss_of(agent, ids, mask, relax, lists=None):
    """Hikaye batch'i -> ortalama kayip (gecis basina, nat): (1 - relax) * -log sum_k pi_k P(B | k) + relax * yon ortalamasi;
    B sonraki cumlenin butun torbasi.  lists (B, T - 1, L): gecis basina kisa liste (verilirse torba yalniz listeden)."""
    state = agent.initial_state(len(ids))
    LP, PI = [], []
    for t in range(ids.shape[1] - 1):
        state = agent.read(state, ids[:, t], mask[:, t])
        valid = mask[:, t + 1].any(1)
        if not valid.any():
            break
        words, c = _bag(ids[:, t + 1], mask[:, t + 1], len(agent.vocab))
        log_pi, lp = agent.bag_log_prob(state, words, c, None if lists is None else lists[:, t])
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


def _lists(table, stories, T, gold):
    """Hikayeler (cumle -> kimlik) -> gecis basina kisa liste (N, T - 1, L), dolgu 0: okunan cumlenin kelimelerinin
    komsulari + sik kelimeler + hikayede o ana kadar gecenler; gold: egitimde gercek sonraki cumlenin kelimeleri de (torba
    olasiligi ancak boyle tanimli)."""
    rows = []
    for st in stories:
        seen, story = set(), []
        for t in range(len(st) - 1):
            seen |= set(st[t])
            lst = shortlist(table, sorted(set(st[t]))) | seen
            if gold:
                lst |= set(st[t + 1])
            lst.discard(0)
            story.append(sorted(lst))
        rows.append(story)
    L = max(len(x) for st in rows for x in st)
    out = torch.zeros(len(rows), T - 1, L, dtype=torch.long)
    for n, st in enumerate(rows):
        for t, x in enumerate(st):
            out[n, t, :len(x)] = torch.tensor(x)
    return out


def valid_hashes(valid_next, agent, weights, T):
    """Hikayelerin gecerli devamlari (hikaye -> gecis -> kelime listeleri) -> ozet tensoru (N, T - 1, M), bos -1."""
    V = len(agent.vocab)
    rows = [[[_bag_hash(torch.bincount(torch.tensor(agent.ids(b)), minlength=V).float(), weights.cpu()).item()
              for b in step] for step in story] for story in valid_next]
    M = max((len(step) for story in rows for step in story), default=1)
    out = torch.full((len(rows), T - 1, M), -1, dtype=torch.long)
    for n, story in enumerate(rows):
        for t, step in enumerate(story):
            out[n, t, :len(step)] = torch.tensor(step, dtype=torch.long)
    return out


def evaluate(agent, ids, mask, real_hash, weights, names, show=SHOW, valid_next=None, gate=None, lists=None):
    """-> bag_in_top_k (ilk 1 / 5 / K, taban: onceki cumle), tutarlilik, calibration, direction_health; goz satirlari.
    gecerli_devam (valid verilirse; kullanici, 4 Ekim: "modelin ürettiği çıktı olası bir çıktı olabilir yani bizim istediğimiz
    değil ama doğru"): ilk aday veride tanimli gecerli devamlardan biri mi, ilk 5 adayin kaci gecerli.
    Toplu: ayni torbayi veren yonler ozetle birlesir, olasiliklari toplanir.  Hikayede daha once gecmis torba aday sayilmaz
    (kullanici, 4 Ekim: "birebir aynı torba olmadığı sürece bence sıkıntı yok aynı şeyin farklı ifade edilmesinde").
    gate verilirse kapidan (is_complete) gecemeyen torba da aday sayilmaz (secimde kapi)."""
    K, V = agent.directions, len(agent.vocab)
    acc = torch.zeros(12, device=ids.device)         # n, ilk1, ilk5, ilkK, taban, tutarli ilk, tutarli hepsi, aday, expH,
    #                                                  gecerli ilk1, gecerli ilk5 sayisi, ilk5 aday sayisi
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
                counts, logp = agent.next_bags(state, None if lists is None else lists[b0:b0 + 256, t])
                hk = _bag_hash(counts, weights)
                logp = logp.masked_fill((hk[:, :, None] == torch.stack(said, 1)[:, None, :]).any(-1), -1e9)
                if gate is not None:
                    logp = logp.masked_fill(_judge(gate, hk, counts), -1e9)
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
                v = valid.float()
                if valid_next is not None:
                    vh = valid_next[b0:b0 + len(bi), t].to(ids.device)
                    ok = (hk[:, :, None] == vh[:, None, :]).any(-1) & first
                    ok_ranked = ok.gather(1, order)[:, :5]
                    in5 = ranked_first[:, :5].sum(1).float()
                    extra = torch.stack([(ok_ranked[:, 0] * v).sum(), (ok_ranked.sum(1) * v).sum(), (in5 * v).sum()])
                else:
                    extra = torch.zeros(3, device=ids.device)
                q = torch.where(first, merged, torch.zeros_like(merged))
                q = q / q.sum(1, keepdim=True).clamp_min(1e-30)
                ent = (-(q * q.clamp_min(1e-30).log()).sum(1)).exp()
                acc += torch.cat([torch.stack([v.sum(), (hit[:, :1].any(1) * v).sum(), (hit[:, :5].any(1) * v).sum(),
                                               (hit.any(1) * v).sum(), ((prev == true) * v).sum(),
                                               (real.gather(1, order[:, :1]).squeeze(1) * v).sum(),
                                               (real.sum(1) * v).sum(), (n_first * v).sum(), (ent * v).sum()]), extra])
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
               gecerli_devam=dict(ilk_aday=round(acc[9] / n, 4), ilk5_orani=round(acc[10] / max(acc[11], 1), 4))
               if valid_next is not None else None,
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
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--every", type=int, default=10, help="kac epokta bir olcum")
    ap.add_argument("--out", default=None, help="kosu klasoru: her epok checkpoint.pt, sonda agent.pt ve results.json")
    ap.add_argument("--resume", type=int, default=0, help="1: --out'taki checkpoint.pt'den kaldigi epoktan surdur")
    ap.add_argument("--shortlist", type=int, default=0, help="N: torba kisa listeden (meaning komsu tablosu, kelime basina N "
                    "komsu + sik kelimeler + hikayede gecenler); 0 kapali")
    ap.add_argument("--grammar", default=None, help="secimde kapi: missing'li grammar agent'in agent.pt'si (sinavda)")
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
                    exam_names=[s.get("country", i) for i, s in enumerate(exam)],
                    exam_valid=[s["valid_next"] for s in exam] if all("valid_next" in s for s in exam) else None)
    data = _cached(os.path.join(folder, prefix + "stories.pt"),
                   (vocab_path, stories_path, __file__, os.path.join(HERE, "context.py")), build)
    ids, mask = (t.to(args.device) for t in data["train"])
    exam_ids, exam_mask = (t.to(args.device) for t in data["exam"])
    n_train = len(ids)
    weights = torch.randint(1, 2 ** 62, (len(vocab),), generator=torch.Generator().manual_seed(0)).to(args.device)
    lists = exam_lists = None
    if args.shortlist:
        t_list = time.time()
        as_lists = lambda a, m: [[a[n, t][m[n, t]].tolist() for t in range(a.shape[1]) if m[n, t].any()] for n in range(len(a))]
        train_s, exam_s = as_lists(ids.cpu(), mask.cpu()), as_lists(exam_ids.cpu(), exam_mask.cpu())
        table = build_neighbor_table(train_s, len(vocab), n=args.shortlist)
        lists = _lists(table, train_s, ids.shape[1], gold=True).to(args.device)
        exam_lists = _lists(table, exam_s, exam_ids.shape[1], gold=False).to(args.device)
        cover = [len(set(st[t + 1]) & set(exam_lists[n, t].tolist())) / len(set(st[t + 1]))
                 for n, st in enumerate(exam_s) for t in range(len(st) - 1)]
        print("kisa liste: kelime basina %d komsu; liste ort %.0f kelime (sozluk %d); sinavda sonraki cumlenin kelimeleri "
              "listede %.3f; %.0f sn" % (args.shortlist, (exam_lists > 0).sum(-1).float()[exam_lists.sum(-1) > 0].mean(),
                                        len(vocab), np.mean(cover), time.time() - t_list), flush=True)
    real_hash = torch.cat([_bag_hash(_counts(a[:, t], m[:, t], len(vocab)), weights)[m[:, t].any(1)]
                           for a, m in ((ids, mask), (exam_ids, exam_mask)) for t in range(a.shape[1])]).unique()
    exam_valid = (valid_hashes(data["exam_valid"], agent, weights, exam_ids.shape[1])
                  if data.get("exam_valid") else None)
    sent = mask.any(2)
    presence = torch.zeros(len(vocab), device=args.device)
    for t in range(ids.shape[1]):
        present = torch.zeros(n_train, len(vocab), device=args.device).scatter_(1, ids[:, t], mask[:, t].float())
        presence += (present * sent[:, t:t + 1]).sum(0)
    agent.set_prior(presence / sent.sum())
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr)
    gen = torch.Generator().manual_seed(args.seed)
    gate = None
    if args.grammar:
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "grammar"))
        import grammar as Gm
        gp = torch.load(args.grammar, map_location=args.device, weights_only=False)
        grm = Gm.GrammarAgent(gp["vocab"], d=gp["args"]["d"], missing=bool(gp["args"].get("missing", 0)))
        grm.load_state_dict(gp["state"])
        assert grm.missing is not None, "kapi missing'li gramer ister (train_grammar --missing 1)"
        grm.to(args.device).eval().requires_grad_(False)
        gate = dict(grammar=grm, module=Gm, vocab=vocab, known=torch.empty(0, dtype=torch.long, device=args.device),
                    ok=torch.empty(0, dtype=torch.bool, device=args.device))
        print("secimde kapi:", args.grammar, flush=True)
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
        keys = ("data", "seed", "d", "slots", "directions", "relax", "batch", "lr", "schedule", "epochs", "shortlist")
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
            loss = loss_of(agent, ids[rows], mask[rows], args.relax, None if lists is None else lists[rows])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.detach() * len(rows)
        total = total.item()                             # adim basina .item() yok: GPU her adimda beklenmez
        if epoch % args.every == 0 or epoch == args.epochs:
            agent.eval()
            t_train = time.time() - t_epoch
            res = evaluate(agent, exam_ids, exam_mask, real_hash, weights, data["exam_names"], valid_next=exam_valid, lists=exam_lists,
                           show=SHOW if epoch == args.epochs else 0, gate=gate)
            agent.train()
            k = res["bag_in_top_k"]
            print("epok %3d  kayip %.3f  (%.0f sn; egitim %.1f sn/epok, olcum %.1f sn) | dogru torba ilk1 %.3f ilk5 %.3f ilk%d %.3f (taban onceki "
                  "%.3f) | tutarlilik ilk %.3f butun %.3f | calibration %.3f | yon: farkli %.2f expH %.1f olu %d" % (
                      epoch, total / n_train, time.time() - t0, t_train, time.time() - t_epoch - t_train,
                      k["ilk1"], k["ilk5"],
                      args.directions, k["ilkK"], k["taban_onceki"], res["tutarlilik"]["ilk_aday"],
                      res["tutarlilik"]["butun_adaylar"], res["calibration"], res["direction_health"]["farkli_torba_orani"],
                      res["direction_health"]["exp_H_pi"], res["direction_health"]["olu_yon"]), flush=True)
            if res["gecerli_devam"]:
                print("          gecerli devam: ilk aday %.3f, ilk 5 adayin %.3f'i" % (
                    res["gecerli_devam"]["ilk_aday"], res["gecerli_devam"]["ilk5_orani"]), flush=True)
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
