"""bag_report -- aday havuzu (torba) A0: kayitli Model Z DONUK, yalniz secici (BagHead) SS train'de uydurulur, sinav alt
kumesinde kaynak kaynak kapsama olculur (belge 53 s8-9, 54, 55; adlar onayli, kullanici 7 Ekim: "A0 ve isimler ok").

Torba B_k = C u P_k u L_k (model_z/sentence.py bag_*): C = train sayiminda en sik --bag_core token + END + EOS (belge 55
O1; kullanici, 7 Ekim: "şu an ilk 50 diye başlayalım"); P_k ayni hikayenin onceki token'lari; L_k secici sirasiyla.
Secici puani: log sum_j exp(q_j . sg(e_v)) + b_v (+ --bag_prior: log p_meaning, belge 54 s6).  Torba kaybi token basina
havuzlanir (belge 54 s4.1), yalniz secicinin agirliklari ogrenir (model ve E donuk; ileri gecis no_grad).
Olcu (sinav: exam_pack_plan.npz; ezber kontrolu: uydurmada gorulen ilk train batch'leri): L boyu M ve toplam boy K'ya gore
C / P / L / kacan, tahmin edilen cumlenin sirasina ve hedef turune gore kacan; ayni M'de tabanlar (siklik dolgusu,
meaning-yalniz); ornek: cumle cumle sonraki cumle + L'nin ilk 20'si.
Kisa deneme (kural 3 istisnasi): surdurme yok; --out'ta bag_head.pt varsa DURUR.

    python bag_report.py --run <kosu (agent.pt)> --out <klasor> [--data <v2/simplestories_gpt2>] [--stream <simplestories>]
                         [--bag_core 50] [--bag_prior <meaning kosusu>] [--queries 1] [--steps 2000] [--lr 1e-3]
                         [--local /content/v2_cache] [--device cuda]
Sayim <data>/train_token_counts.npy; yoksa data.token_counts ile BIR kez uretilir (belge 55 O1).
Cikti: <out>/config.json, bag_head.pt, bag_report.json, bag_examples.txt; sonda tek satir ozet.
"""
import torch  # noqa: I001  (Windows: torch once)

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(os.path.dirname(HERE), "common")
sys.path.insert(0, COMMON)
sys.path.insert(0, HERE)
import data as D  # noqa: E402
import train as T  # noqa: E402  (_attn: egitimle ayni maske yolu)
import generate_readings as GR  # noqa: E402  (load_run)

sys.path.insert(0, os.path.join(os.path.dirname(HERE), "model_z"))
from sentence import BagHead, bag_copy, bag_index, bag_loss, bag_rank  # noqa: E402

PRIOR_WINDOW, PRIOR_SKIP = 5, 128      # belge 54 s5.2: son 5 cumlenin siklik sirasi >= 128 token'lari oy verir
M_LIST = (0, 64, 128, 256, 512, 1024, 1536, 2048)        # L boyu
K_LIST = (64, 128, 256, 512, 1024, 2048)                 # toplam boy: M_k = max(0, K - |C| - |P_k|)
SENT_BINS = ((0, 0), (1, 1), (2, 5), (6, 10 ** 6))       # tahmin edilen cumlenin hikayedeki sirasi (0'dan)
KINDS = {D.TargetKind.FIRST: "first", D.TargetKind.MID: "mid", D.TargetKind.END: "end", D.TargetKind.EOS: "eos"}
SRC_C, SRC_P, SRC_REST = 0, 1, 2
CHUNK = 512                            # torba dilimi (puan (dilim, VOCAB) fp32)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def core_mask(counts, n, dev):
    """-> (core (VOCAB,) bool, rank (VOCAB,) siklik sirasi): en sik n token + END + EOS (END sayimda yok)."""
    order = np.argsort(-np.asarray(counts), kind="stable")
    rank = np.full(D.VOCAB, len(order), np.int64)
    rank[order] = np.arange(len(order))
    core = np.zeros(D.VOCAB, bool)
    core[order[:n]] = True
    core[[D.END_ID, D.EOS_ID]] = True
    return torch.as_tensor(core, device=dev), torch.as_tensor(rank, device=dev)


def load_prior(path, tok_sha, dev):
    """Meaning (v2_meaning_*; arsiv kodu v2-before-cleanup-20261006): P(j | i) = softmax_j(bias_j + s_i . t_j / sqrt(d))."""
    f = os.path.join(path, "agent.pt")
    pack = torch.load(f, map_location="cpu", weights_only=False)
    assert pack["tokenizer_sha256"] == tok_sha, "meaning baska tokenizer'la: %s" % pack["tokenizer_sha256"]
    st = pack["state"]
    return dict(src=st["source.weight"].float().to(dev), tgt=st["target.weight"].float().to(dev),
                bias=st["bias"].float().to(dev), seen=(pack["count"] > 0).to(dev), sha256=_sha256(f))


def prior_logp(prior, batch, ids, rows, cols, rank):
    """-> (n_torba, VOCAB) log p_meaning(v | torbanin son PRIOR_WINDOW cumlesi): oy veren = farkli TOKEN'lar, siklik sirasi
    >= PRIOR_SKIP ve meaning'de gorulmus; oylarin P(. | i) ortalamasi; oy yoksa softmax(bias).  END sutunu ~0."""
    T_, n, Vm = batch.kind.shape[1], len(rows), len(prior["bias"])
    r, c = (batch.kind == D.Kind.TOKEN).nonzero(as_tuple=True)
    v = batch.tokens[r, c]
    ok = (rank[v] >= PRIOR_SKIP) & prior["seen"][v]
    r, c, v = r[ok], c[ok], v[ok]
    last = torch.searchsorted(rows * T_ + batch.doc[rows, cols].long(), r * T_ + batch.doc[r, c].long(), right=True) - 1
    bags = ids[r, c][:, None] + 1 + torch.arange(PRIOR_WINDOW, device=v.device)        # kendi Z'si ve sonraki W-1
    pair = torch.unique((bags * D.VOCAB + v[:, None])[bags <= last[:, None]])
    bag, word = pair // D.VOCAB, pair % D.VOCAB
    uv, col = torch.unique(word, return_inverse=True)
    A = torch.zeros(n, len(uv), device=v.device)
    A[bag, col] = 1.0
    cnt = A.sum(1, keepdim=True)
    rows_p = torch.softmax(prior["bias"] + prior["src"][uv] @ prior["tgt"].T / math.sqrt(prior["src"].shape[1]), -1)
    p = (A / cnt.clamp_min(1)) @ rows_p
    p = torch.where(cnt > 0, p, torch.softmax(prior["bias"], -1))
    out = torch.full((n, D.VOCAB), math.log(1e-30), device=v.device)
    out[:, :Vm] = p.clamp_min(1e-30).log()
    return out


def bag_batch(model, mask_fn, batch, core, cuda, prior=None, rank=None):
    """Donuk ileri gecis + torba yapisi -> dict(h ozet durumlari, P, prior, hedefler: bag, y, kind, izinli)."""
    with torch.no_grad(), torch.autocast(batch.tokens.device.type, dtype=torch.bfloat16, enabled=cuda):
        h = model._batch_hidden(batch, T._attn(batch, mask_fn, cuda))
    ids, rows, cols = bag_index(batch)
    P = bag_copy(batch, ids, rows, cols, core)
    keep = batch.target >= 0
    bag, y = ids[keep], batch.target[keep]
    src = torch.where(core[y], SRC_C, torch.where(P[bag, y], SRC_P, SRC_REST))
    pri = prior_logp(prior, batch, ids, rows, cols, rank) if prior is not None else None
    return dict(h=h[rows, cols].float(), P=P, prior=pri, bag=bag, y=y, src=src, kind=batch.target_kind[keep],
                sent=batch.sent[rows, cols][bag].long() + 1, n=len(rows))


def scores(head, E, b, b0, b1, prior_on):
    """Torba dilimi [b0, b1) -> puan (dilim, VOCAB)."""
    s = head(b["h"][b0:b1], E)
    return s + b["prior"][b0:b1] if prior_on else s


def _slices(b):
    """-> [(b0, b1, t0, t1)]: torba dilimleri ve hedef araliklari (hedefler torba sirasinda)."""
    out = []
    for b0 in range(0, b["n"], CHUNK):
        b1 = min(b0 + CHUNK, b["n"])
        t0, t1 = (int(torch.searchsorted(b["bag"], torch.tensor(x, device=b["bag"].device))) for x in (b0, b1))
        out.append((b0, b1, t0, t1))
    return out


def fit_step(head, opt, E, b, core, prior_on):
    """Bir uydurma adimi: torba kaybi (token basina havuz) dilim dilim geri yayilir -> (kayip, hedef sayisi)."""
    rest = b["src"] == SRC_REST
    total = int(rest.sum())
    opt.zero_grad(set_to_none=True)
    loss_sum = torch.zeros((), device=b["y"].device)
    for b0, b1, t0, t1 in _slices(b):
        m = rest[t0:t1]
        s = scores(head, E, b, b0, b1, prior_on)
        allowed = ~core[None] & ~b["P"][b0:b1]
        loss, _ = bag_loss(s, allowed, b["bag"][t0:t1][m] - b0, b["y"][t0:t1][m])
        (loss / max(total, 1)).backward()
        loss_sum += loss.detach()
    opt.step()
    return float(loss_sum) / max(total, 1), total


@torch.no_grad()
def ranks(b, core, scorers):
    """-> {ad: sira (hedef basina; C / P hedefinde -1)}.  scorers: ad -> f(b0, b1) puan dilimi."""
    out = {k: torch.full_like(b["y"], -1) for k in scorers}
    rest = b["src"] == SRC_REST
    for b0, b1, t0, t1 in _slices(b):
        m = rest[t0:t1]
        if not bool(m.any()):
            continue
        allowed = ~core[None] & ~b["P"][b0:b1]
        bag, y = b["bag"][t0:t1][m] - b0, b["y"][t0:t1][m]
        for k, f in scorers.items():
            out[k][t0:t1][m] = bag_rank(f(b0, b1), allowed, bag, y)
    return out


def _bin(lo, hi):
    return "%d+" % lo if hi >= 10 ** 6 else "%d" % lo if lo == hi else "%d-%d" % (lo, hi)


def summarize(rows, n_core):
    """rows: hedef basina birlesik dict(src, kind, sent, pk, rank_<ad>) (numpy) -> kapsama tablolari (hepsi hedef
    agirlikli; p_mean ve k_mean = |C| + ort |P_k| + M de)."""
    src, n = rows["src"], len(rows["src"])
    out = dict(targets=n, core=n_core, share_c=round(float((src == SRC_C).mean()), 4),
               share_p=round(float((src == SRC_P).mean()), 4), p_mean=round(float(rows["pk"].mean()), 1))
    for name in [k[5:] for k in rows if k.startswith("rank_")]:
        r = rows["rank_" + name]
        rest = src == SRC_REST
        miss = lambda m, sel=slice(None): float((rest[sel] & (r[sel] >= m)).sum() / max(1, len(src[sel])))  # noqa: E731
        by_m = {str(m): dict(miss=round(miss(m), 4), share_l=round(float((rest & (r < m)).mean()), 4),
                             k_mean=round(n_core + float(rows["pk"].mean()) + m, 1)) for m in M_LIST}
        mk = lambda K: np.maximum(0, K - n_core - rows["pk"])  # noqa: E731
        by_k = {str(K): dict(miss=round(float((rest & (r >= mk(K))).mean()), 4),
                             over=round(float((n_core + rows["pk"] > K).mean()), 4)) for K in K_LIST}
        by_sent = {_bin(*lo_hi): {str(m): round(miss(m, (rows["sent"] >= lo_hi[0]) & (rows["sent"] <= lo_hi[1])), 4)
                                     for m in (0, 512, 2048)} for lo_hi in SENT_BINS}
        by_kind = {nm: {str(m): round(miss(m, rows["kind"] == k), 4) for m in (0, 512, 2048)} for k, nm in KINDS.items()}
        out[name] = dict(by_m=by_m, by_k=by_k, by_sent=by_sent, by_kind=by_kind)
    return out


def _collect(acc, b, rk):
    pk = b["P"].sum(1)[b["bag"]]
    for k, v in dict(src=b["src"], kind=b["kind"], sent=b["sent"], pk=pk, **{"rank_" + n: r for n, r in rk.items()}).items():
        acc.setdefault(k, []).append(v.cpu().numpy())


def _concat(acc):
    return {k: np.concatenate(v) for k, v in acc.items()}


def examples(b, batch, core, scorers, tok, n_docs=3, top=20):
    """Ilk n_docs hikaye (satir 0): her torba icin sonraki cumle ve her puanin L'sinin ilk top kelimesi."""
    txt = []
    ids, rows, cols = bag_index(batch)
    docs = batch.doc[rows, cols].cpu()
    for j in ((rows.cpu() == 0) & (docs < n_docs)).nonzero()[:, 0].tolist():
        r, c = 0, int(cols[j])
        nxt = ((batch.doc[r] == batch.doc[r, c]) & (batch.sent[r] == int(batch.sent[r, c]) + 1) &
               (batch.kind[r] == D.Kind.TOKEN))
        words = batch.tokens[r][nxt].tolist()
        txt.append("--- hikaye %d, torba %d (cumle %d icin) | sonraki: %s" % (
            int(batch.doc[r, c]), j, int(batch.sent[r, c]) + 1, tok.decode(words) if words else "(EOS)"))
        allowed = ~core & ~b["P"][j]
        for name, f in scorers.items():
            s = f(j, j + 1)[0].masked_fill(~allowed, float("-inf"))
            got = s.topk(top).indices.tolist()
            txt.append("  %-9s %s" % (name, " | ".join(tok.decode([w]).replace("\n", "\\n") +
                                                      ("*" if w in words else "") for w in got)))
    return txt


def _args(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="kayitli Model Z kosusu (agent.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="/content/drive/MyDrive/v2/simplestories_gpt2")
    ap.add_argument("--stream", default="/content/drive/MyDrive/simplestories")
    ap.add_argument("--local", default=None, help="ham akisin yerel kopyasi (train.py --local ile ayni; sha denetimli)")
    ap.add_argument("--bag_core", type=int, default=50, help="C: train sayiminda en sik N (+ END + EOS)")
    ap.add_argument("--bag_prior", default=None, help="meaning kosusu (agent.pt); yoksa onsel yok")
    ap.add_argument("--queries", type=int, default=1, help="secicinin sorgu sayisi J (belge 54 s6)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seen_batches", type=int, default=5, help="ezber kontrolu: uydurmada gorulen ilk N train batch'i")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args(argv)


def main(argv=None):
    args = _args(argv)
    t0 = time.time()
    log = lambda msg: print("[%7.1f sn] %s" % (time.time() - t0, msg), flush=True)  # noqa: E731
    dev = torch.device(args.device)
    cuda = dev.type == "cuda"
    if cuda:
        assert torch.cuda.is_available(), "GPU YOK"
        for k in ("cache_size_limit", "recompile_limit"):                # train.py ile ayni (son batch'ler yeni sekil)
            if hasattr(torch._dynamo.config, k):
                setattr(torch._dynamo.config, k, 32)
    elif os.name == "nt":
        T._no_power_throttling()
    if os.path.exists(os.path.join(args.out, "bag_head.pt")):
        sys.exit("DUR: %s'de bag_head.pt var (kisa deneme, surdurme yok); yeni klasor" % args.out)
    os.makedirs(args.out, exist_ok=True)
    cpath = os.path.join(args.data, "train_token_counts.npy")
    if not os.path.exists(cpath):
        log("train_token_counts.npy yok -> data.token_counts ile uretiliyor (bir kez, belge 55 O1)")
        D.token_counts(args.stream, args.data)
    counts = np.load(cpath)
    core, rank = core_mask(counts, args.bag_core, dev)
    tok_path = os.path.join(args.stream, "gpt2", "tokenizer.json")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    prior = load_prior(args.bag_prior, _sha256(tok_path), dev) if args.bag_prior else None
    model, idt = GR.load_run(args.run, args.data, dev)
    model.eval().requires_grad_(False)
    if cuda:
        for block in model.blocks:
            block.compile(dynamic=False)
    E = model.E.weight
    stream = T._local_copy(args.stream, args.local, args.data) if args.local else args.stream
    train = D.TokenStories(stream, args.data, "train")
    valid = D.TokenStories(stream, args.data, "valid")
    plan = np.load(os.path.join(args.data, "train_pack_plan_e1.npz"))
    ro, rs, row_len = plan["row_offsets"], plan["row_stories"], int(plan["row_len"])
    ep = np.load(os.path.join(args.data, "exam_pack_plan.npz"))
    batches = lambda o, r, a, b: [r[o[i]:o[i + 1]].tolist() for i in range(a, min(b, len(o) - 1))]  # noqa: E731
    torch.manual_seed(args.seed)
    bias = None if prior is not None else np.log(np.r_[counts, 0][:D.VOCAB] + 1.0)
    head = BagHead(model.E.weight.shape[1], args.queries, bias).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.0)
    config = dict(args=vars(args), run_identity=idt, core=int(core.sum()), core_tokens=core.nonzero()[:, 0].tolist(),
                  counts_sha256=_sha256(cpath), bag_prior_sha256=prior["sha256"] if prior else None,
                  prior_window=PRIOR_WINDOW, prior_skip=PRIOR_SKIP, git=T._git(),
                  exam_set_sha256=str(ep["exam_set_sha256"]), torch=torch.__version__,
                  device=torch.cuda.get_device_name(0) if cuda else "cpu")
    json.dump(config, open(os.path.join(args.out, "config.json"), "w"), indent=1)
    log("A0 | kosu %s | C %d (en sik %d + END + EOS) | onsel %s | J %d | %d adim, lr %g | cihaz %s" % (
        os.path.basename(os.path.normpath(args.run)), int(core.sum()), args.bag_core, "meaning" if prior else "yok",
        args.queries, args.steps, args.lr, config["device"]))
    prior_on = prior is not None
    win, t_win = [], time.time()
    for step in range(args.steps):
        rows = batches(ro, rs, step * D.BATCH_ROWS, (step + 1) * D.BATCH_ROWS)
        if not rows:
            sys.exit("DUR: plan bitti (%d adim)" % step)
        b = bag_batch(model, model.mask_fn, D.build_batch(train, rows, "model_z", dev, row_len), core, cuda, prior, rank)
        win.append(fit_step(head, opt, E, b, core, prior_on))
        if (step + 1) % 100 == 0 or step + 1 == args.steps:
            if cuda:
                torch.cuda.synchronize()
            ls, ns = zip(*win)
            log("adim %d / %d  torba kaybi %.4f  hedef/batch %d  %.0f ms/adim" % (
                step + 1, args.steps, sum(l * n for l, n in win) / sum(ns), sum(ns) / len(ns),
                1000 * (time.time() - t_win) / len(win)))
            win, t_win = [], time.time()
    torch.save(dict(state=head.state_dict(), config=config), os.path.join(args.out, "bag_head.pt"))
    head.eval()

    freq = torch.log(torch.as_tensor(np.r_[counts, 0][:D.VOCAB] + 1.0, device=dev, dtype=torch.float))[None]

    def scorer_set(b):
        f = dict(selector=lambda b0, b1: scores(head, E, b, b0, b1, prior_on).detach(),
                 freq=lambda b0, b1: freq.expand(b1 - b0, -1))
        if prior_on:
            f["meaning"] = lambda b0, b1: b["prior"][b0:b1]
        return f

    report, ex = {}, []
    for name, (st, o, r, n) in dict(exam=(valid, ep["row_offsets"], ep["row_stories"], len(ep["row_offsets"]) - 1),
                                    train_seen=(train, ro, rs, args.seen_batches * D.BATCH_ROWS)).items():
        acc = {}
        for a in range(0, n, D.BATCH_ROWS):
            batch = D.build_batch(st, batches(o, r, a, min(a + D.BATCH_ROWS, n)), "model_z", dev, row_len)
            b = bag_batch(model, model.mask_fn, batch, core, cuda, prior, rank)
            sc = scorer_set(b)
            _collect(acc, b, ranks(b, core, sc))
            if name == "exam" and a == 0:
                with torch.no_grad():
                    ex = examples(b, batch, core, sc, tok)
        report[name] = summarize(_concat(acc), int(core.sum()))
        log("%s: %d hedef, C %.3f P %.3f | kacan M 512 / 2048: %s" % (
            name, report[name]["targets"], report[name]["share_c"], report[name]["share_p"], ", ".join(
                "%s %.4f / %.4f" % (k, report[name][k]["by_m"]["512"]["miss"], report[name][k]["by_m"]["2048"]["miss"])
                for k in scorer_set(b))))
    json.dump(dict(config=config, report=report), open(os.path.join(args.out, "bag_report.json"), "w"), indent=1)
    open(os.path.join(args.out, "bag_examples.txt"), "w", encoding="utf-8").write("\n".join(ex) + "\n")
    e = report["exam"]
    print("BAG A0 | C %d | onsel %s | J %d | sinav kacan M 512: %s | M 2048: %s | ezber (train_seen - sinav, M 512) %+.4f"
          % (int(core.sum()), "meaning" if prior_on else "yok", args.queries,
             " ".join("%s %.4f" % (k, e[k]["by_m"]["512"]["miss"]) for k in e if isinstance(e[k], dict)),
             " ".join("%s %.4f" % (k, e[k]["by_m"]["2048"]["miss"]) for k in e if isinstance(e[k], dict)),
             report["train_seen"]["selector"]["by_m"]["512"]["miss"] - e["selector"]["by_m"]["512"]["miss"]),
          flush=True)
    return report


if __name__ == "__main__":
    main()
