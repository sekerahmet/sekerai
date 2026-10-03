"""lookahead_ss -- SimpleStories, egitimsiz ileri bakis sinamasi (bulgular 34 §5.1; kullanici, 3 Ekim: "illeri bakışa
bakabilirsin", adlar "adlar uygun").  Deneysel: ise yaramazsa silinir.

Ayni istemlerden iki uretim, cumle cumle:
    acgozlu      her cumle sinirinda en olasi token, cumle sonuna kadar acgozlu (duz acgozlu uretimle ayni metin)
    ileri bakis  her cumle sinirinda en olasi TOP token ve eot denenir; her aday cumle sonuna kadar acgozlu yazilir;
                 birebir cumle tekrari (exam_simplestories'in tanimi, >= 5 birim) OLMAYANLARIN en olasisi secilir
                 (eot hep izinli; hepsi tekrarsa en olasi -- 'zorlandi' sayilir)
Istemler sinav hikayelerinden (exam_rows): ilk 6 token / ilk cumle / ilk yari.  Olculer exam_simplestories._count
(cumle tekrari, /1k, 8'li dongu, bitirme, uydurma).  Metinler gozle okunur (kural 12).

    python analysis/lookahead_ss.py <kosu> --data <simplestories koku> [--device cuda] [--n 300] [--weights both]
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # deneme2/model_y
for _p in (os.path.join(HERE, "train_simplestories"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402  (pyarrow'dan once: Windows DLL sirasi)
import data_simplestories as DS  # noqa: E402
import exam_simplestories as ES  # noqa: E402
import internals_y as I  # noqa: E402
from model_y import AttentionCache  # noqa: E402
from train_y import generate_cached  # noqa: E402

TOP = 4             # ileri bakista denenen en olasi baslangic token'i (+ eot)
SENTENCE_MAX = 40   # aday cumle en cok bu kadar token; cumle sonu gelmezse oldugu gibi
BUDGET = 256        # devam en cok bu kadar token
CHUNK = 64          # son konum logit'i icin satir parcasi


@torch.no_grad()
def _last_logprobs(model, prefixes, eos):
    """Onekler (degisken boy) -> son konumdaki log olasiliklar (cpu).  Onbellekli: onek son token'i haric hidden'dan."""
    dev = next(model.parameters()).device
    out = []
    for c in range(0, len(prefixes), CHUNK):
        part = prefixes[c:c + CHUNK]
        lengths = torch.tensor([len(p) for p in part], device=dev)
        width = int(lengths.max())
        ids = torch.full((len(part), width), eos, dtype=torch.long)
        for i, p in enumerate(part):
            ids[i, :len(p)] = torch.tensor(p)
        ids, rows = ids.to(dev), torch.arange(len(part), device=dev)
        if width > 1:
            caches = [AttentionCache(lengths - 1, width) for _ in range(model.turns)]
            model.hidden(ids[:, :width - 1], caches)
            z = model.logits(ids[rows, lengths - 1][:, None], caches)[:, 0]
        else:
            z = model.logits(ids)[:, -1]
        out.append(z.float().log_softmax(-1).cpu())
    return torch.cat(out)


def _sentence(first, rollout, end, eos):
    """Aday: ilk token + acgozlu devam -> (cumle token'lari, eot'la bitti).  Ilk cumle sonunda (dahil) ya da eot'ta kesilir."""
    out = [first]
    if end[first]:
        return out, False
    for t in rollout:
        if t == eos:
            return out, True
        out.append(t)
        if end[t]:
            break
    return out, False


def _repeat(context, sent, tab):
    """context + sent'in son cumlesi (>= 5 birim) daha once tam gecmis mi (exam_simplestories cumle tanimi)."""
    s = ES._sentences(DS._units(context + sent, tab))
    return len(s) > 1 and len(s[-1]) >= ES._LONG_SENTENCE and s[-1] in set(s[:-1])


@torch.no_grad()
def continue_by_sentence(model, prompts, vocab, lookahead, budget=BUDGET):
    """prompts ([eos] + istem) -> (devamlar, bitti, istatistik).  Cumle cumle; lookahead False: her sinirda en olasi
    token (acgozlu ile ayni metin)."""
    eos = vocab.index(DS.EOS_TOKEN)
    end = DS._sentence_ends(vocab, len(vocab))[0]
    tab = DS._token_table(vocab)
    gens, ended = [[] for _ in prompts], [False] * len(prompts)
    stats = dict(boundaries=0, changed=0, forced=0, stopped=0)
    active = list(range(len(prompts)))
    while active:
        prefixes = [prompts[i] + gens[i] for i in active]
        lp = _last_logprobs(model, prefixes, eos)
        k = TOP if lookahead else 1
        top = lp.topk(k, -1)
        cands = []
        for r, i in enumerate(active):
            c = [(float(v), int(t)) for v, t in zip(top.values[r], top.indices[r])]
            if lookahead and eos not in [t for _, t in c]:
                c.append((float(lp[r, eos]), eos))
            cands.append(sorted(c, reverse=True))
        jobs = [(r, t) for r, c in enumerate(cands) for _, t in c if t != eos]
        seqs = [prefixes[r] + [t] for r, t in jobs]   # parca parca: istem gecisi butun konumlarda sozluk logit'i uretir
        rolls = [x for c in range(0, len(seqs), CHUNK) for x in generate_cached(model, seqs[c:c + CHUNK], SENTENCE_MAX - 1)]
        roll = {job: x for job, x in zip(jobs, rolls)}
        nxt = []
        for r, i in enumerate(active):
            stats["boundaries"] += 1
            choice = None
            for rank, (_, t) in enumerate(cands[r]):
                if t == eos:
                    choice = ([], True)
                    break
                sent, stop = _sentence(t, roll[(r, t)], end, eos)
                if not lookahead or not _repeat(prompts[i][1:] + gens[i], sent, tab):
                    choice = (sent, stop)
                    break
            if choice is None:                       # hepsi tekrar: en olasi (zorlandi)
                t = next(t for _, t in cands[r] if t != eos)
                choice = _sentence(t, roll[(r, t)], end, eos)
                stats["forced"] += 1
            chosen = choice[0][0] if choice[0] else eos
            stats["changed"] += chosen != cands[r][0][1]
            sent, stop = choice
            gens[i] = (gens[i] + sent)[:budget]
            if stop and len(gens[i]) < budget:
                ended[i] = True
                stats["stopped"] += 1
            if not ended[i] and len(gens[i]) < budget:
                nxt.append(i)
        active = nxt
    return gens, ended, stats


def prompt_sets(data, n):
    """Sinav hikayelerinden uc istem kumesi: ilk 6 token / ilk cumle / ilk yari ([eos] ile)."""
    vocab, a = data["vocab"], data["valid"]
    eos = vocab.index(DS.EOS_TOKEN)
    end = DS._sentence_ends(vocab, len(vocab))[0]
    rows = ES.exam_rows(data)[:n]
    half = ES.story_prompts(data, rows)[0]
    first6, sent1 = [], []
    for s, L in zip(data["valid_start"][rows].tolist(), data["valid_length"][rows].tolist()):
        x = a[s:s + L].tolist()
        first6.append([eos] + x[:6])
        cut = next((j + 1 for j, t in enumerate(x[:60]) if end[t]), 12)
        sent1.append([eos] + x[:cut])
    return dict(first6=first6, sentence1=sent1, half=half)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run")
    ap.add_argument("--data", required=True, help="simplestories koku (gpt2/...)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--weights", default="both", choices=("last", "ema", "both"))
    ap.add_argument("--examples", type=int, default=4)
    ap.add_argument("--budget", type=int, default=BUDGET)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    cfg = json.load(open(os.path.join(args.run, "config.json")))
    assert cfg.get("tag") == "gpt2", "yalniz etiketsiz SimpleStories gpt2 kosusu: %s" % cfg.get("tag")
    data = DS.build(args.data, "gpt2", seq_len=cfg["seq_len"], log=lambda s: None)
    vocab, tab = data["vocab"], DS._token_table(data["vocab"])
    words, xax = data["text_reference"]["words"], data["text_reference"]["xax"]
    stock = ES._stock_reference(data, ES.STORY_CONTINUATIONS)
    sets = prompt_sets(data, args.n)
    report = dict(run=os.path.basename(os.path.normpath(args.run)), n=args.n, top=TOP, budget=BUDGET, results={})
    lines = []
    for w in (["last", "ema"] if args.weights == "both" else [args.weights]):
        model = I._load_model(args.run, w).to(args.device)
        for name, prompts in sets.items():
            for mode in ("greedy", "lookahead"):
                t0 = time.time()
                gens, ended, st = continue_by_sentence(model, prompts, vocab, mode == "lookahead", args.budget)
                gens = [DS.strip_sentence_ids(g, vocab) for g in gens]
                c = ES._count([DS.strip_sentence_ids(p, vocab) for p in prompts], gens, ended, tab, words, xax, stock)
                c.update(st, secs=round(time.time() - t0, 1))
                c["texts"] = [dict(prompt=DS.decode(p[1:], vocab), model=DS.decode(g, vocab))
                              for p, g in zip(prompts[:args.examples], gens[:args.examples])]
                report["results"]["%s/%s/%s" % (w, name, mode)] = c
                lines.append("%-4s %-9s %-9s | cumle tekrari %3.0f%% /1k %5.2f | 8li dongu %3.0f%% | bitti %3.0f%% | "
                             "uydurma /1k %5.1f | sinir %d degisti %d zorlandi %d | %.0f sn" % (
                                 w, name, mode, 100 * c["sentence_repeat"], c["sentence_repeat_per1k"],
                                 100 * c["loop"], 100 * c["ended"], c["nonword_per1k"], st["boundaries"],
                                 st["changed"], st["forced"], c["secs"]))
                print(lines[-1], flush=True)
        del model
    print("\n# ORNEKLER (gozle)")
    for key, c in report["results"].items():
        if key.split("/")[0] != "last":
            continue
        print("\n## " + key)
        for t in c["texts"]:
            print("   [%s] || %s" % (t["prompt"][-80:], t["model"][:600].replace("\n", " / ")))
    out = os.environ.get("KUYRUK_SONUC", args.run)
    path = os.path.join(out, "lookahead_ss_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
    json.dump(report, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nyazildi:", path)


if __name__ == "__main__":
    main()
