# -*- coding: utf-8 -*-
"""train_20 -- adim 1 kosusu, CPU: data_20'nin egitim cumlelerinden ardisik token ciftleri, butun veri tek batch, Adam.
Uc ayar yan yana: sabit noktalar, serbest (capa 0), capali.

    python train_20.py           secili token'lar
    python train_20.py --all     butun token'lar
"""
import sys
import time

import torch

import data_20
from model_20 import CONFIDENCE, NextTokenModel, deviation, neighbors, transitions

SETTINGS = {                      # capa lr/wd gibi deneme sayisi; 1e-2 fazla sertti (kayip 1,284 > 1,270)
    "fixed": dict(learn_points=False),
    "free": dict(learn_points=True, anchor=0.0),
    "anchored": dict(learn_points=True, anchor=1e-3),
}
STEPS, LR = 2000, 0.01
LOG_AT = (0, 10, 50, 200, 500, 1000, 2000)
SHOW = ("Alice", "Tom", "Smith", "Hill", "'s", "father", "aunt", "is", "Who", "?", ".")


def token_pairs(data):
    """Her cumle <eos> ... <eos>; ardisik (girdi, hedef) ciftleri."""
    inputs, targets = [], []
    for s in data["train"]:
        ids = data_20.encode(s, data["vocab"])
        inputs += ids[:-1]
        targets += ids[1:]
    return torch.tensor(inputs), torch.tensor(targets)


def bigram_floor(inputs, targets, n):
    """Yalniz son token'a bakan bir modelin ulasabilecegi en dusuk kayip (sayimlardan)."""
    C = torch.zeros(n, n, dtype=torch.float64)
    C.index_put_((inputs, targets), torch.ones(len(inputs), dtype=torch.float64), accumulate=True)
    p = C / C.sum(1, keepdim=True).clamp_min(1)
    nz = C > 0
    return float(-(C[nz] * p[nz].log()).sum() / C.sum())


def train(setting, inputs, targets, n, steps=STEPS, lr=LR, log_at=LOG_AT):
    model = NextTokenModel(n, **SETTINGS[setting])
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    curve = []
    for step in range(steps + 1):
        total, nll = model.loss(inputs, targets)
        if step in log_at:
            curve.append(dict(step=step, nll=nll.item(), W=model.next.W.norm().item(), dev=deviation(model).clone()))
        if step == steps:
            break
        opt.zero_grad()
        total.backward()
        opt.step()
    return model, curve


def report(results, vocab, floor, n_pairs, fingerprint, show):
    ix = {w: i for i, w in enumerate(vocab)}
    rows = [ix[w] for w in show]
    scale = next(iter(results.values()))[0].scale
    out = ["ADIM 1  veri iz %s  cift %d  sozluk %d  scale %.3f (guven %.2f)  esit olasilik %.3f  bigram tavani %.3f" % (
        fingerprint, n_pairs, len(vocab), scale, CONFIDENCE, torch.log(torch.tensor(float(len(vocab)))), floor), ""]
    out.append("ayar       " + "  ".join("%11s" % ("adim %d" % c["step"]) for c in results["fixed"][1]))
    for name, (model, curve) in results.items():
        out.append("%-9s  " % name + "  ".join("%11s" % ("%.3f" % c["nll"]) for c in curve) + "   nll")
        out.append("%-9s  " % "" + "  ".join("%11s" % ("%.1f" % c["W"]) for c in curve) + "   |W|")
        out.append("%-9s  " % "" + "  ".join("%11s" % ("%.1f°" % float(c["dev"].mean())) for c in curve)
                   + "   ort. sapma")
    for name, (model, curve) in results.items():
        if not model.tokens.learn:
            continue
        out += ["", "%s: NEREDEN NEREYE (sapma = PF ile PL arasi aci)" % name,
                "  %-8s %6s   %-36s %-36s %s" % ("token", "sapma", "PF'de en yakin 3", "PL'de en yakin 3", "egri (adim:aci)")]
        nf, af = neighbors(model.tokens.fixed_points)
        nl, al = neighbors(model.tokens.points())
        for r in rows:
            near = lambda idx, ang: ", ".join("%s %.0f°" % (vocab[j], a) for j, a in zip(idx[r].tolist(), ang[r].tolist()))
            out.append("  %-8s %5.1f°   %-36s %-36s %s" % (
                vocab[r], float(curve[-1]["dev"][r]), near(nf, af), near(nl, al),
                " ".join("%d:%.0f" % (c["step"], float(c["dev"][r])) for c in curve)))
    out += ["", "GECIS (sonraki token, en olasi 5)"]
    for r in rows:
        for name, (model, _) in results.items():
            i, v = transitions(model)
            out.append("  %-8s %-9s %s" % (vocab[r] if name == "fixed" else "", name,
                                           "  ".join("%s %.2f" % (vocab[j], p) for j, p in zip(i[r].tolist(), v[r].tolist()))))
    return "\n".join(out)


if __name__ == "__main__":
    torch.set_num_threads(1)
    data = data_20.build()
    data_20.audit(data)
    inputs, targets = token_pairs(data)
    n = len(data["vocab"])
    results = {}
    for name in SETTINGS:
        t0 = time.time()
        results[name] = train(name, inputs, targets, n)
        print("%s %.1f sn" % (name, time.time() - t0), flush=True)
    show = data["vocab"][2:] if "--all" in sys.argv else SHOW
    print()
    print(report(results, data["vocab"], bigram_floor(inputs, targets, n), len(inputs), data["fingerprint"], show))
