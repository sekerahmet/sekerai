# -*- coding: utf-8 -*-
"""analyze_capacity -- Model Y kapasite analizi (ajan A, 2 Ekim 2026).

Kullanici: "bu bozulmanin nedeni model kapasitesinin eksik olmasi olabilir mi? birsey ogrenirken onu saklayacak yerin
olmamasi mesela".  Soru: bir olgu checkpoint'ler boyunca ogreniliyor mu, sonra unutuluyor mu, hic mi ogrenilmiyor.

    facts   FACTS listesindeki her istem icin dogru cevabin (ve 2 yanlis ama makul cevabin) ogretmen zorlamali nll'i,
            cevabin ilk token'inin sirasi (0 = en olasi), o konumda modelin ilk 5 tahmini; checkpoint'ler boyunca
            (her yedekte son agirlik + optimizer'daki EMA), sonda model.pt ve model_weight_ema.pt.

    python analysis/analyze_capacity.py <kosu klasoru> facts --data <FineWeb koku> --device cuda --steps 2000,4000
    python analysis/analyze_capacity.py <kosu klasoru> facts --data <FineWeb koku> --selftest      (CPU, kucuk rastgele model)
Cikti: $KUYRUK_SONUC (yoksa <kosu>/analysis) altina capacity_facts_<etiket>.json; ozet tablo stdout'a.
"""
import argparse
import json
import math
import os
import sys
import time

import torch  # pyarrow / tokenizers'tan once (Windows c10.dll)
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path[:0] = [SRC, os.path.join(SRC, "train_fineweb"), os.path.join(SRC, "train_simplestories")]
import internals_y as I  # noqa: E402

EOT = "<|endoftext|>"

# kind: frequent_entity / rare_entity / definition / date_event / number / arithmetic / attribution.  answer: olgu
# token'lari; tail: dogal devam (cevaba kosullu, karsilastirmaya girmez); wrong: ayni yere konan makul yanlislar.
# Ilk 10'u exam_fineweb.QUESTIONS (ayni istem).
FACTS = (
    dict(kind="frequent_entity", prompt="The capital of France is", answer=" Paris",
         tail=", which is also the largest city in the country.", wrong=(" Lyon", " London")),
    dict(kind="frequent_entity", prompt="The largest planet in our solar system is", answer=" Jupiter",
         tail=", a gas giant more than eleven times wider than Earth.", wrong=(" Saturn", " Earth")),
    dict(kind="number", prompt="Water boils at a temperature of", answer=" 100 degrees Celsius",
         tail=" (212 degrees Fahrenheit) at sea level.", wrong=(" 50 degrees Celsius", " 0 degrees Celsius")),
    dict(kind="definition", prompt="Plants need sunlight, water and", answer=" carbon dioxide",
         tail=" to make their own food through photosynthesis.", wrong=(" oxygen", " nitrogen")),
    dict(kind="rare_entity", prompt="The chemical symbol for gold is", answer=" Au",
         tail=", from the Latin word aurum.", wrong=(" Cu", " Ag")),
    dict(kind="attribution", prompt="Isaac Newton is famous for", answer=" his laws of motion",
         tail=" and the law of universal gravitation.", wrong=(" his theory of relativity", " his theory of evolution")),
    dict(kind="date_event", prompt="World War II ended in the year", answer=" 1945",
         tail=", when Germany and Japan surrendered.", wrong=(" 1918", " 1939")),
    dict(kind="frequent_entity", prompt="The Amazon rainforest is located in", answer=" South America",
         tail=", mostly in Brazil.", wrong=(" Africa", " Asia")),
    dict(kind="arithmetic", prompt="If a rectangle is 3 meters long and 4 meters wide, its area is",
         answer=" 12 square meters", tail=".", wrong=(" 7 square meters", " 14 square meters")),
    dict(kind="definition", prompt="The seasons on Earth are caused by", answer=" the tilt of Earth's axis",
         tail=" as it orbits the Sun.", wrong=(" the distance between Earth and the Sun", " the rotation of Earth")),
    dict(kind="definition", prompt="The water cycle is", answer=" the continuous movement of water",
         tail=" through evaporation, condensation and precipitation.",
         wrong=(" the continuous movement of air", " the continuous movement of rocks")),
    dict(kind="definition", prompt="The process by which plants make food using sunlight is called",
         answer=" photosynthesis", tail=".", wrong=(" respiration", " transpiration")),
    dict(kind="definition", prompt="The powerhouse of the cell is the", answer=" mitochondria",
         tail=", which produces energy for the cell.", wrong=(" nucleus", " ribosome")),
    dict(kind="date_event", prompt="The French Revolution began in", answer=" 1789",
         tail=", when the people of Paris stormed the Bastille.", wrong=(" 1776", " 1815")),
    dict(kind="date_event", prompt="The American Declaration of Independence was signed in", answer=" 1776",
         tail=" in Philadelphia.", wrong=(" 1789", " 1812")),
    dict(kind="date_event", prompt="Christopher Columbus first reached the Americas in", answer=" 1492",
         tail=".", wrong=(" 1607", " 1588")),
    dict(kind="date_event", prompt="The Berlin Wall fell in", answer=" 1989",
         tail=", leading to the reunification of Germany.", wrong=(" 1961", " 1991")),
    dict(kind="date_event", prompt="The first humans landed on the Moon in", answer=" 1969",
         tail=", during the Apollo 11 mission.", wrong=(" 1959", " 1979")),
    dict(kind="attribution", prompt="The play Romeo and Juliet was written by", answer=" William Shakespeare",
         tail=".", wrong=(" Charles Dickens", " Christopher Marlowe")),
    dict(kind="attribution", prompt="The novel Pride and Prejudice was written by", answer=" Jane Austen",
         tail=".", wrong=(" Charles Dickens", " Mary Shelley")),
    dict(kind="attribution", prompt="The telephone was invented by", answer=" Alexander Graham Bell",
         tail=" in 1876.", wrong=(" Thomas Edison", " Nikola Tesla")),
    dict(kind="attribution", prompt="The theory of evolution by natural selection was proposed by",
         answer=" Charles Darwin", tail=".", wrong=(" Isaac Newton", " Gregor Mendel")),
    dict(kind="rare_entity", prompt="The capital of Australia is", answer=" Canberra",
         tail=".", wrong=(" Sydney", " Melbourne")),
    dict(kind="rare_entity", prompt="The capital of Mongolia is", answer=" Ulaanbaatar",
         tail=".", wrong=(" Beijing", " Astana")),
    dict(kind="rare_entity", prompt="The chemical symbol for sodium is", answer=" Na",
         tail=".", wrong=(" So", " K")),
    dict(kind="number", prompt="The speed of light in a vacuum is about", answer=" 300,000 kilometers per second",
         tail=".", wrong=(" 300,000 kilometers per hour", " 3,000 kilometers per second")),
    dict(kind="arithmetic", prompt="Seven plus five equals", answer=" twelve",
         tail=".", wrong=(" eleven", " thirteen")),
)


def load_tokenizer(fw_root):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(fw_root, "gpt2", "tokenizer.json"))
    eot = tok.token_to_id(EOT)
    assert eot is not None, "tokenizer'da %s yok" % EOT
    return tok, eot


def build_sequences(tok, eot):
    """Her olgu x aday (0 = dogru, 1.. yanlis) -> dict(ids, answer_start, answer_end, tail_end).  Dizi: eot + istem +
    cevap + devam (belge basindan, konus'taki gibi).  Ayri kodlama birlesik kodlamayla tutmali (bosluk sinirinda)."""
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids
    seqs = []
    for i, f in enumerate(FACTS):
        for j, ans in enumerate((f["answer"],) + tuple(f["wrong"])):
            p, a, t = enc(f["prompt"]), enc(ans), enc(f["tail"])
            joined = enc(f["prompt"] + ans + f["tail"])
            ids = [eot] + p + a + t
            seqs.append(dict(fact=i, candidate=j, text=ans, ids=ids, answer_start=1 + len(p),
                             answer_end=1 + len(p) + len(a), tail_end=len(ids), joint_ok=joined == p + a + t))
    return seqs


@torch.no_grad()
def score(model, seqs, tok, device, eot):
    """Tek ileri hesap (sagdan dolgu; nedensel, dolgu onceki konumlari etkilemez).  -> aday basina sayilar."""
    T = max(len(s["ids"]) for s in seqs)
    ids = torch.full((len(seqs), T), eot, dtype=torch.long)
    for k, s in enumerate(seqs):
        ids[k, :len(s["ids"])] = torch.tensor(s["ids"])
    ids = ids.to(device)
    logp = F.log_softmax(model.logits(ids[:, :-1]).float(), -1)        # konum t -> token t+1
    out = []
    for k, s in enumerate(seqs):
        a0, a1, t1 = s["answer_start"], s["answer_end"], s["tail_end"]
        tgt = ids[k, 1:t1]
        nll = -logp[k, :t1 - 1].gather(-1, tgt[:, None])[:, 0]
        first = logp[k, a0 - 1]
        rank = int((first > first[ids[k, a0]]).sum())
        top = first.topk(5)
        ans_nll = nll[a0 - 1:a1 - 1]
        tail_nll = nll[a1 - 1:t1 - 1]
        out.append(dict(
            answer_nll=float(ans_nll.sum()), answer_tokens=int(a1 - a0), answer_token_nll=[round(float(x), 4) for x in ans_nll],
            first_rank=rank, first_p=float(first[ids[k, a0]].exp()),
            tail_nll_mean=float(tail_nll.mean()) if len(tail_nll) else None,
            top5=[(tok.decode([int(t)]), round(float(p.exp()), 4)) for p, t in zip(top.values, top.indices)]
            if s["candidate"] == 0 else None))
    return out


def per_fact(seqs, scores):
    """Aday sonuclari olgu basina: dogru, yanlislar, dogru en dusuk nll'li mi (cevap kismi toplam nll)."""
    rows = []
    for i, f in enumerate(FACTS):
        cand = [(s, r) for s, r in zip(seqs, scores) if s["fact"] == i]
        right = cand[0][1]
        wrong = [dict(text=s["text"], answer_nll=r["answer_nll"], first_rank=r["first_rank"]) for s, r in cand[1:]]
        rows.append(dict(fact=i, kind=f["kind"], prompt=f["prompt"], answer=f["answer"],
                         answer_nll=right["answer_nll"], answer_tokens=right["answer_tokens"],
                         answer_token_nll=right["answer_token_nll"], first_rank=right["first_rank"],
                         first_p=right["first_p"], tail_nll_mean=right["tail_nll_mean"], top5=right["top5"],
                         wrong=wrong, margin=min(w["answer_nll"] for w in wrong) - right["answer_nll"],
                         correct_best=all(right["answer_nll"] < w["answer_nll"] for w in wrong)))
    return rows


def weight_sets(run_dir, steps, finals, config):
    """(etiket, adim, agirlik turu, model) uretici; her yedek bir kez okunur: once son agirlik, sonra EMA."""
    packs = I._checkpoints(run_dir)
    missing = [s for s in steps if s not in packs]
    assert not missing, "yedek yok: %s" % missing
    for s in steps:
        t0 = time.time()
        pack = torch.load(packs[s], map_location="cpu", weights_only=True)
        assert pack["step"] == s, (pack["step"], s)
        model = I._build(config)
        model.load_state_dict(pack["model"])
        print("   yedek %d okundu (%.0f sn)" % (s, time.time() - t0), flush=True)
        yield "t%06d_last" % s, s, "last", model.eval()
        I._load_weight_ema(model, pack["optimizer"], config)
        del pack
        yield "t%06d_ema" % s, s, "ema", model.eval()
    for name in finals:
        path = os.path.join(run_dir, {"last": "model.pt", "ema": "model_weight_ema.pt"}[name])
        model = I._build(config)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        yield "final_%s" % name, None, name, model.eval()


def selftest_sets(config):
    """Kucuk rastgele model (CPU): akisin sinamasi, sayilarin anlami yok."""
    small = I._override(config, ["d=64", "units=64", "turns=2", "layers=2", "heads=2", "t_max=128"])
    yield "selftest", None, "last", I._build(small).eval()


def summary_lines(results):
    labels = [r["label"] for r in results]
    L = ["## olgu basina: cevap nll (toplam, nat) / ilk token sirasi / dogru en iyi mi (+: dogru < iki yanlis)",
         "%-44s %s" % ("", " ".join("%16s" % x[-16:] for x in labels))]
    for i, f in enumerate(FACTS):
        cells = []
        for r in results:
            x = r["facts"][i]
            cells.append("%16s" % ("%.2f/%d%s" % (x["answer_nll"], x["first_rank"], "+" if x["correct_best"] else "-")))
        L.append("%-44s %s" % (("%-6s %s" % (f["kind"][:6], f["prompt"]))[:44], " ".join(cells)))
    L.append("%-44s %s" % ("dogru en iyi (sayi)", " ".join("%16d" % sum(x["correct_best"] for x in r["facts"])
                                                           for r in results)))
    L.append("%-44s %s" % ("ilk token sira 0 (sayi)", " ".join("%16d" % sum(x["first_rank"] == 0 for x in r["facts"])
                                                               for r in results)))
    L.append("%-44s %s" % ("cevap nll ortalama", " ".join("%16.3f" % (sum(x["answer_nll"] for x in r["facts"]) / len(FACTS))
                                                          for r in results)))
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(prog="analyze_capacity.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="kosu klasoru")
    ap.add_argument("measure", choices=("facts",))
    ap.add_argument("--data", required=True, help="FineWeb koku (gpt2/tokenizer.json)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", default="", help="checkpoint adimlari: '2000,4000' ya da 'every:K' ya da 'all'")
    ap.add_argument("--finals", default="", help="'last,ema': model.pt / model_weight_ema.pt")
    ap.add_argument("--label", default="", help="cikti adina ek")
    ap.add_argument("--selftest", action="store_true", help="kucuk rastgele model, CPU")
    args = ap.parse_args(argv)
    t0 = time.time()
    config = I._config(args.run)
    tok, eot = load_tokenizer(args.data)
    seqs = build_sequences(tok, eot)
    bad = [s["text"] for s in seqs if not s["joint_ok"]]
    print("olgu %d, aday %d; ayri/birlesik kodlama farkli: %s" % (len(FACTS), len(seqs), bad or "yok"), flush=True)
    if args.selftest:
        sets = selftest_sets(config)
    else:
        packs = I._checkpoints(args.run)
        spec = args.steps
        steps = ([] if not spec else list(packs) if spec == "all" else
                 [s for s in packs if s % int(spec[6:]) == 0] if spec.startswith("every:") else [int(s) for s in spec.split(",")])
        sets = weight_sets(args.run, steps, [x for x in args.finals.split(",") if x], config)
    results = []
    for label, step, kind, model in sets:
        model = model.to(args.device)
        scores = score(model, seqs, tok, args.device, eot)
        results.append(dict(label=label, step=step, weights=kind, facts=per_fact(seqs, scores)))
        print("   %s: dogru en iyi %d / %d" % (label, sum(x["correct_best"] for x in results[-1]["facts"]), len(FACTS)), flush=True)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    lines = summary_lines(results)
    print("\n".join(lines))
    out_dir = os.environ.get("KUYRUK_SONUC") or os.path.join(args.run, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    name = "capacity_facts%s_%s" % ("_" + args.label if args.label else "", time.strftime("%Y%m%d_%H%M%S"))
    path = os.path.join(out_dir, name + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(measure="facts", run=config.get("name"), selftest=args.selftest, secs=round(time.time() - t0, 1),
                       facts=[dict(f, wrong=list(f["wrong"])) for f in FACTS],
                       token_ids=[dict(fact=s["fact"], candidate=s["candidate"], ids=s["ids"]) for s in seqs],
                       results=results), f, ensure_ascii=False, indent=1)
    print("yazildi: %s  (%.0f sn)" % (path, time.time() - t0))


if __name__ == "__main__":
    main()
