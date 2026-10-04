"""generate -- Model Z uretim dongusu (kullanici, 4 Ekim: "ilk hedef döngülü olarak ülke hikayesi yazdırmak olsun";
"generate çok mantıklı. yani istediğimiz çıktı aslında generate olacak").

Bir cumleden baslar, metni cumle cumle surdurur:
    1. context agent okudugu cumleye gore sonraki cumlenin aday torbalarini uretir (meaning tablosu verilirse yalniz kisa
       listeden: okunan cumlenin kelimelerinin komsulari + metinde gecenler)
    2. grammar agent her adayi dizer; metinde birebir gecmis torba elenir; gramer missing'li ise kapidan (is_complete)
       gecemeyen de (kullanici, 4 Ekim: "ilk eğitimi kapısız yapalım")
    3. en olasi aday secilir, dizilmis cumle context agent'a geri verilir

    python generate.py --context agent.pt --grammar agent.pt [--meaning neighbors.pt --shortlist 30 --depth 2]
                       [--steps 15] --start "Turkey is famous for baklava ."
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for part in ("context", "grammar", "meaning"):
    sys.path.insert(0, os.path.join(HERE, part))
import grammar as Gm  # noqa: E402
from context import ContextAgent  # noqa: E402
from meaning import DEPTH, NEIGHBORS, shortlist  # noqa: E402


def load(context_path, grammar_path, meaning_path=None):
    """-> (context agent, grammar agent, meaning tablosu ya da None); hepsi CPU'da, egitim kipinde degil."""
    pc = torch.load(context_path, map_location="cpu", weights_only=False)
    a = pc["args"]
    ctx = ContextAgent(pc["vocab"], d=a["d"], slots=a["slots"], directions=a["directions"])
    ctx.load_state_dict(pc["state"])
    pg = torch.load(grammar_path, map_location="cpu", weights_only=False)
    grm = Gm.GrammarAgent(pg["vocab"], d=pg["args"]["d"], missing=bool(pg["args"].get("missing", 0)))
    grm.load_state_dict(pg["state"])
    table = torch.load(meaning_path, weights_only=False) if meaning_path else None
    if table is not None:
        assert table["vocab"] == pc["vocab"], "meaning tablosunun sozlugu context agent'inkinden farkli"
    return ctx.eval(), grm.eval(), table


@torch.no_grad()
def generate(ctx, grm, start, steps, table=None, n=NEIGHBORS, depth=DEPTH):
    """Ilk cumle (kelime listesi) -> uretilen cumleler (kelime listeleri; ilki haric).  Gecen aday kalmazsa durur."""
    state = ctx.initial_state(1)
    sentence, said, seen, out = list(start), set(), set(), []
    for _ in range(steps):
        ids = torch.tensor([ctx.ids(sentence)])
        state = ctx.read(state, ids, torch.ones_like(ids, dtype=torch.bool))
        said.add(tuple(sorted(sentence)))
        seen |= set(ids[0].tolist())
        lst = None
        if table is not None:
            lst = torch.tensor([sorted((shortlist(table, sorted(set(ids[0].tolist())), n, depth) | seen) - {0})])
        counts, logp = ctx.next_bags(state, lst)
        chosen = None
        for k in logp[0].argsort(descending=True).tolist():
            words = [ctx.vocab[w] for w in counts[0, k].nonzero().flatten().tolist() for _ in range(int(counts[0, k, w]))]
            if not words or tuple(sorted(words)) in said:
                continue
            g = Gm.relation_matrix(grm, words).double().numpy()
            order = Gm.order_by_relation(g[:len(words) + 1, :len(words) + 1])
            if grm.missing is None or Gm.is_complete(g, order):
                chosen = [words[i] for i in order]
                break
        if chosen is None:
            break
        out.append(chosen)
        sentence = chosen
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--context", required=True, help="context agent'in agent.pt'si")
    ap.add_argument("--grammar", required=True, help="grammar agent'in agent.pt'si (missing'li ise kapi da)")
    ap.add_argument("--meaning", default=None, help="meaning agent'in neighbors.pt'si (verilirse kisa liste)")
    ap.add_argument("--shortlist", type=int, default=NEIGHBORS, help="kisa liste: kelime basina komsu")
    ap.add_argument("--depth", type=int, default=DEPTH, help="kisa liste: komsularin komsulari kac adim")
    ap.add_argument("--steps", type=int, default=15, help="en cok kac cumle uretilir")
    ap.add_argument("--start", required=True, help="ilk cumle (kelimeler bosluklu, nokta ayri: 'Turkey is famous for baklava .')")
    args = ap.parse_args(argv)
    ctx, grm, table = load(args.context, args.grammar, args.meaning)
    start = args.start.split()
    print(" 1. " + " ".join(start))
    for i, s in enumerate(generate(ctx, grm, start, args.steps, table, args.shortlist, args.depth)):
        print("%2d. %s" % (i + 2, " ".join(s)))


if __name__ == "__main__":
    main()
