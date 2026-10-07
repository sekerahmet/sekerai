"""bag_report -- torba (aday havuzu) kapsama raporu (belge 53-55; adlar onayli, kullanici 7 Ekim: "A0 ve isimler ok").

Iki is, ayni olcu: (1) A0 -- kayitli torbasiz model (Model Z ya da transformer) DONUK, yalniz secici (recipe.Bag, egitimdeki ayni secici) SS train'de
--steps adim uydurulur; (2) torbali egitilmis kosu (--bag_k) -- modelin kendi torbasi olculur, uydurma yok.
Torba B_k = C u P_k u L_k (recipe.bag_index / bag_copy / Bag): C en sik --bag_core + END + EOS (A0; egitilmis kosuda
modelinki), P_k ayni hikayenin onceki token'lari, L_k secici sirasiyla.  Sinav alt kumesinde (exam_pack_plan.npz) ve
uydurmada gorulen ilk train batch'lerinde (ezber kontrolu): L boyu M ve toplam boy K'ya gore C / P / L / kacan, tahmin
edilen cumlenin sirasina ve hedef turune gore kacan; ayni M'de siklik dolgusu tabani; ornek: cumle cumle sonraki cumle + L'nin
ilk 20'si (yakalanan *).  Maliyet: secici FLOP / hedef.  Kisa deneme (kural 3 istisnasi): surdurme yok; --out'ta
bag_report.json varsa DURUR.

    python bag_report.py --run <kosu (agent.pt)> --out <klasor> [--data <v2/simplestories_gpt2>] [--stream <simplestories>]
                         [--local /content/v2_cache] [--bag_core 50] [--steps 2000] [--lr 1e-3] [--device cuda]
Sayim <data>/train_token_counts.npy (data.token_counts ile bir kez uretilir; yoksa DURUR).
Cikti: <out>/config.json, bag_head.pt (A0), bag_report.json, bag_examples.txt; sonda tek satir ozet.
"""
import torch  # noqa: I001  (Windows: torch once)

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(os.path.dirname(HERE), "common")
sys.path.insert(0, COMMON)
sys.path.insert(0, HERE)
import data as D  # noqa: E402
import recipe as R  # noqa: E402
import train as T  # noqa: E402  (_attn: egitimle ayni maske yolu)
import generate_readings as GR  # noqa: E402  (load_run)

M_LIST = (0, 64, 128, 256, 512, 1024, 1536, 2048)        # L boyu
K_LIST = (64, 128, 256, 512, 1024, 2048)                 # toplam boy: M_k = max(0, K - |C| - |P_k|)
SENT_BINS = ((0, 0), (1, 1), (2, 5), (6, 10 ** 6))       # tahmin edilen cumlenin hikayedeki sirasi (0'dan)
KINDS = {D.TargetKind.FIRST: "first", D.TargetKind.MID: "mid", D.TargetKind.END: "end", D.TargetKind.EOS: "eos"}
SRC_C, SRC_P, SRC_REST = 0, 1, 2
CHUNK = 512                            # sira hesabinda torba dilimi
NEVER = 10 ** 9                        # secicinin hic secemeyecegi hedef (train'de gorulmemis) sirasi


def bag_batch(model, mask_fn, bag, batch, cuda):
    """Donuk ileri gecis + torba -> Bag.batch_select ciktisi + hedefler (bag, y, src, kind, sent)."""
    with torch.no_grad(), torch.autocast(batch.tokens.device.type, dtype=torch.bfloat16, enabled=cuda):
        h = model._batch_hidden(batch, T._attn(batch, mask_fn, cuda))
    sel = bag.batch_select(batch, h, model.E.weight)
    keep = batch.target >= 0
    b, y = sel["ids"][keep], batch.target[keep]
    src = torch.where(R.bag_mask(bag.core)[y], SRC_C, torch.where(sel["pbit"][b, y], SRC_P, SRC_REST))
    return dict(sel, bag=b, y=y, src=src, kind=batch.target_kind[keep],
                sent=batch.sent[sel["rows"], sel["cols"]][b].long() + 1, n=len(sel["rows"]))


def fit_step(bag, opt, b):
    """A0 uydurma adimi: Bag.selector_loss (token basina havuz) -> (kayip, hedef sayisi)."""
    opt.zero_grad(set_to_none=True)
    loss, n = bag.selector_loss(b, b["bag"], b["y"])
    (loss / n.clamp_min(1)).backward()
    opt.step()
    return float(loss.detach()) / max(int(n), 1), int(n)


def _slices(b):
    """-> [(b0, b1, t0, t1)]: torba dilimleri ve hedef araliklari (hedefler torba sirasinda)."""
    out = []
    for b0 in range(0, b["n"], CHUNK):
        b1 = min(b0 + CHUNK, b["n"])
        t0, t1 = (int(torch.searchsorted(b["bag"], torch.tensor(x, device=b["bag"].device))) for x in (b0, b1))
        out.append((b0, b1, t0, t1))
    return out


@torch.no_grad()
def ranks(b, scorers):
    """-> {ad: L sirasi (hedef basina; C / P hedefinde -1, secilemeyen NEVER)}.  scorers: ad -> f(b0, b1) puan dilimi;
    aday kumesi her puanda ayni (Bag: C u P_k disi, gorulmus)."""
    out = {k: torch.full_like(b["y"], -1) for k in scorers}
    rest = b["src"] == SRC_REST
    ok = b["allowed"][b["bag"], b["y"]]
    for b0, b1, t0, t1 in _slices(b):
        m = rest[t0:t1] & ok[t0:t1]
        for k in scorers:
            out[k][t0:t1][rest[t0:t1] & ~ok[t0:t1]] = NEVER
        if not bool(m.any()):
            continue
        bag, y = b["bag"][t0:t1][m] - b0, b["y"][t0:t1][m]
        for k, f in scorers.items():
            out[k][t0:t1][m] = bag_rank(f(b0, b1), b["allowed"][b0:b1], bag, y)
    return out


def bag_rank(score, allowed, bag, y):
    """Hedefin L sirasi (0'dan): izinli kelimelerden puani buyuk olanlar + esitlerden kelime no'su kucuk olanlar."""
    s, sy = score[bag], score[bag, y][:, None]
    v = torch.arange(score.shape[1], device=score.device)
    return (allowed[bag] & ((s > sy) | ((s == sy) & (v < y[:, None])))).sum(1)


def _bin(lo, hi):
    return "%d+" % lo if hi >= 10 ** 6 else "%d" % lo if lo == hi else "%d-%d" % (lo, hi)


def summarize(rows, n_core, k_list):
    """rows: hedef basina birlesik dict(src, kind, sent, pk, rank_<ad>) (numpy) -> kapsama tablolari (hedef agirlikli;
    k_mean = |C| + ort |P_k| + M)."""
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
                             over=round(float((n_core + rows["pk"] > K).mean()), 4)) for K in k_list}
        by_sent = {_bin(*lo_hi): {str(m): round(miss(m, (rows["sent"] >= lo_hi[0]) & (rows["sent"] <= lo_hi[1])), 4)
                                  for m in (0, 512, 2048)} for lo_hi in SENT_BINS}
        by_kind = {nm: {str(m): round(miss(m, rows["kind"] == k), 4) for m in (0, 512, 2048)} for k, nm in KINDS.items()}
        out[name] = dict(by_m=by_m, by_k=by_k, by_sent=by_sent, by_kind=by_kind)
    return out


def _collect(acc, b, rk):
    pk = b["pbit"].sum(1)[b["bag"]]
    for k, v in dict(src=b["src"], kind=b["kind"], sent=b["sent"], pk=pk, **{"rank_" + n: r for n, r in rk.items()}).items():
        acc.setdefault(k, []).append(v.cpu().numpy())


def examples(b, batch, scorers, tok, n_docs=3, top=20):
    """Ilk n_docs hikaye (satir 0): her torba icin sonraki cumle ve her puanin L'sinin ilk top kelimesi (yakalanan *)."""
    txt = []
    rows, cols = b["rows"], b["cols"]
    docs = batch.doc[rows, cols].cpu()
    for j in ((rows.cpu() == 0) & (docs < n_docs)).nonzero()[:, 0].tolist():
        c = int(cols[j])
        nxt = ((batch.doc[0] == batch.doc[0, c]) & (batch.sent[0] == int(batch.sent[0, c]) + 1) &
               (batch.kind[0] == D.Kind.TOKEN))
        words = batch.tokens[0][nxt].tolist()
        txt.append("--- hikaye %d, torba %d (cumle %d icin) | sonraki: %s" % (
            int(batch.doc[0, c]), j, int(batch.sent[0, c]) + 1, tok.decode(words) if words else "(EOS)"))
        for name, f in scorers.items():
            got = f(j, j + 1)[0].masked_fill(~b["allowed"][j], float("-inf")).topk(top).indices.tolist()
            txt.append("  %-9s %s" % (name, " | ".join(tok.decode([w]).replace("\n", "\\n") +
                                                      ("*" if w in words else "") for w in got)))
    return txt


def _args(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="kayitli kosu (agent.pt): torbasizsa A0, torbaliysa kendi torbasi")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="/content/drive/MyDrive/v2/simplestories_gpt2")
    ap.add_argument("--stream", default="/content/drive/MyDrive/simplestories")
    ap.add_argument("--local", default=None, help="ham akisin yerel kopyasi (train.py --local ile ayni; sha denetimli)")
    ap.add_argument("--bag_core", type=int, default=50, help="A0: C = train sayiminda en sik N (+ END + EOS)")
    ap.add_argument("--steps", type=int, default=2000, help="A0 uydurma adimi")
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
    if os.path.exists(os.path.join(args.out, "bag_report.json")):
        sys.exit("DUR: %s'de bag_report.json var (kisa deneme, surdurme yok); yeni klasor" % args.out)
    cpath = os.path.join(args.data, "train_token_counts.npy")
    if not os.path.exists(cpath):
        sys.exit("DUR: %s yok (data.token_counts ile bir kez uretilir)" % cpath)
    counts = np.load(cpath)
    os.makedirs(args.out, exist_ok=True)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(args.stream, "gpt2", "tokenizer.json"))
    model, idt = GR.load_run(args.run, args.data, dev)
    model.eval().requires_grad_(False)
    if cuda:
        for block in model.blocks:
            block.compile(dynamic=False)
    trained = hasattr(model, "bag")
    layout, mask_fn = ("model_z", model.mask_fn) if idt["model"] == "model_z" else ("transformer", R.document_mask)
    torch.manual_seed(args.seed)
    if trained:
        bag = model.bag
    else:
        core = R.core_ids(counts, args.bag_core)
        bag = R.Bag(model.E.weight.shape[1], len(core) + max(M_LIST), len(core)).to(dev)
        bag.fill(core, counts)
    stream = T._local_copy(args.stream, args.local, args.data) if args.local else args.stream
    train = D.TokenStories(stream, args.data, "train")
    valid = D.TokenStories(stream, args.data, "valid")
    plan = np.load(os.path.join(args.data, "train_pack_plan_e1.npz"))
    ro, rs, row_len = plan["row_offsets"], plan["row_stories"], int(plan["row_len"])
    ep = np.load(os.path.join(args.data, "exam_pack_plan.npz"))
    batches = lambda o, r, a, b: [r[o[i]:o[i + 1]].tolist() for i in range(a, min(b, len(o) - 1))]  # noqa: E731
    d = model.E.weight.shape[1]
    config = dict(args=vars(args), run_identity=idt, mode="trained" if trained else "A0", bag_k=bag.k,
                  core=len(bag.core), core_tokens=bag.core.tolist(), counts_sha256=T._sha256(cpath), git=T._git(),
                  selector_flop_per_bag=2 * d * (d + D.VOCAB), exam_set_sha256=str(ep["exam_set_sha256"]),
                  torch=torch.__version__, device=torch.cuda.get_device_name(0) if cuda else "cpu")
    json.dump(config, open(os.path.join(args.out, "config.json"), "w"), indent=1)
    log("%s | kosu %s | C %d | K %d | cihaz %s" % (config["mode"], os.path.basename(os.path.normpath(args.run)),
                                                  len(bag.core), bag.k, config["device"]))
    if not trained:
        bag.requires_grad_(True)
        opt = torch.optim.AdamW(bag.parameters(), lr=args.lr, weight_decay=0.0)
        win, t_win = [], time.time()
        for step in range(args.steps):
            rows = batches(ro, rs, step * D.BATCH_ROWS, (step + 1) * D.BATCH_ROWS)
            if not rows:
                sys.exit("DUR: plan bitti (%d adim)" % step)
            b = bag_batch(model, mask_fn, bag, D.build_batch(train, rows, layout, dev, row_len), cuda)
            win.append(fit_step(bag, opt, b))
            if (step + 1) % 100 == 0 or step + 1 == args.steps:
                if cuda:
                    torch.cuda.synchronize()
                ns = [n for _, n in win]
                log("adim %d / %d  secici kaybi %.4f  hedef/batch %d  %.0f ms/adim" % (
                    step + 1, args.steps, sum(l_ * n for l_, n in win) / max(sum(ns), 1), sum(ns) / len(ns),
                    1000 * (time.time() - t_win) / len(win)))
                win, t_win = [], time.time()
        torch.save(dict(state=bag.state_dict(), config=config), os.path.join(args.out, "bag_head.pt"))
        bag.requires_grad_(False)
    freq = torch.log(torch.as_tensor(np.r_[counts, 0][:D.VOCAB] + 1.0, device=dev, dtype=torch.float))[None]
    k_list = sorted(set(K_LIST) | {bag.k})
    report, ex, cost = {}, [], {}
    for name, (st, o, r, n) in dict(exam=(valid, ep["row_offsets"], ep["row_stories"], len(ep["row_offsets"]) - 1),
                                    train_seen=(train, ro, rs, args.seen_batches * D.BATCH_ROWS)).items():
        acc = {}
        for a in range(0, n, D.BATCH_ROWS):
            batch = D.build_batch(st, batches(o, r, a, min(a + D.BATCH_ROWS, n)), layout, dev, row_len)
            with torch.no_grad():
                b = bag_batch(model, mask_fn, bag, batch, cuda)
                sc = dict(selector=lambda b0, b1, b=b: b["score"][b0:b1], freq=lambda b0, b1: freq.expand(b1 - b0, -1))
                _collect(acc, b, ranks(b, sc))
                if name == "exam" and a == 0:
                    ex = examples(b, batch, sc, tok)
                    cost = dict(bags_per_target=round(b["n"] / len(b["y"]), 4),
                                selector_flop_per_target=round(config["selector_flop_per_bag"] * b["n"] / len(b["y"])))
        report[name] = summarize({k: np.concatenate(v) for k, v in acc.items()}, len(bag.core), k_list)
        log("%s: %d hedef, C %.3f P %.3f | kacan M 512 / 2048 / K %d: %s" % (
            name, report[name]["targets"], report[name]["share_c"], report[name]["share_p"], bag.k, ", ".join(
                "%s %.4f / %.4f / %.4f" % (k, report[name][k]["by_m"]["512"]["miss"],
                                           report[name][k]["by_m"]["2048"]["miss"], report[name][k]["by_k"][str(bag.k)]["miss"])
                for k in ("selector", "freq"))))
    report["cost"] = cost
    json.dump(dict(config=config, report=report), open(os.path.join(args.out, "bag_report.json"), "w"), indent=1)
    open(os.path.join(args.out, "bag_examples.txt"), "w", encoding="utf-8").write("\n".join(ex) + "\n")
    e = report["exam"]
    print("BAG %s | C %d | K %d | sinav kacan M 512: secici %.4f siklik %.4f | M 2048: %.4f / %.4f | K: %.4f / %.4f | ezber "
          "(train_seen - sinav, M 512) %+.4f | secici %d FLOP/hedef" % (
              config["mode"], len(bag.core), bag.k, e["selector"]["by_m"]["512"]["miss"], e["freq"]["by_m"]["512"]["miss"],
              e["selector"]["by_m"]["2048"]["miss"], e["freq"]["by_m"]["2048"]["miss"],
              e["selector"]["by_k"][str(bag.k)]["miss"], e["freq"]["by_k"][str(bag.k)]["miss"],
              report["train_seen"]["selector"]["by_m"]["512"]["miss"] - e["selector"]["by_m"]["512"]["miss"],
              cost["selector_flop_per_target"]), flush=True)
    return report


if __name__ == "__main__":
    main()
