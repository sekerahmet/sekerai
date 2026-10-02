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


# Kabul edilen baska dogrular (exam_fineweb'in "100 C / 212 F" gibi); FACTS degismez, eski sonuclarla kiyas surer.
ALSO = {
    "Water boils at a temperature of": (" 212 degrees Fahrenheit",),
    "Isaac Newton is famous for": (" his law of gravity", " his laws of gravity"),
    "The Amazon rainforest is located in": (" Brazil",),
    "If a rectangle is 3 meters long and 4 meters wide, its area is": (" 12 square metres",),
    "The seasons on Earth are caused by": (" the tilt of the Earth's axis",),
    "The powerhouse of the cell is the": (" mitochondrion",),
    "The American Declaration of Independence was signed in": (" Philadelphia",),
    "The capital of Mongolia is": (" Ulan Bator",),
    "The speed of light in a vacuum is about": (" 186,000 miles per second",),
    "Seven plus five equals": (" 12",),
}
EMA_DECAY = 0.999        # config weight_ema; EMA baslangic agirligiyla baslar (duzeltmesiz): payi EMA_DECAY^adim


def load_tokenizer(fw_root):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(fw_root, "gpt2", "tokenizer.json"))
    eot = tok.token_to_id(EOT)
    assert eot is not None, "tokenizer'da %s yok" % EOT
    return tok, eot


def build_sequences(tok, eot, facts=FACTS):
    """Her olgu x aday (0 = dogru, sonra ALSO'daki dogrular, sonra yanlislar; role right / also / wrong) ->
    dict(ids, answer_start, answer_end, tail_end).  Dizi: eot + istem +
    cevap + devam (belge basindan, konus'taki gibi).  Ayri kodlama birlesik kodlamayla tutmali (bosluk sinirinda)."""
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids
    seqs = []
    for i, f in enumerate(facts):
        also = tuple(f.get("also", ())) or ALSO.get(f["prompt"], ())
        roles = ["right"] + ["also"] * len(also) + ["wrong"] * len(f["wrong"])
        for j, ans in enumerate((f["answer"],) + tuple(also) + tuple(f["wrong"])):
            p, a, t = enc(f["prompt"]), enc(ans), enc(f["tail"])
            joined = enc(f["prompt"] + ans + f["tail"])
            ids = [eot] + p + a + t
            seqs.append(dict(fact=i, candidate=j, role=roles[j], text=ans, ids=ids, answer_start=1 + len(p),
                             answer_end=1 + len(p) + len(a), tail_end=len(ids), joint_ok=joined == p + a + t))
    return seqs


@torch.no_grad()
def score(model, seqs, tok, device, eot, batch=256):
    """Batch'ler halinde ileri hesap (sagdan dolgu; nedensel, dolgu onceki konumlari etkilemez).  -> aday basina sayilar."""
    if len(seqs) > batch:
        return [x for i in range(0, len(seqs), batch) for x in score(model, seqs[i:i + batch], tok, device, eot, batch)]
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
        pos = logp[k, a0 - 1:a1 - 1]                                    # cevap token'larinin konumlari
        own = pos.gather(-1, ids[k, a0:a1, None])
        token_rank = (pos > own).sum(-1)
        out.append(dict(
            answer_nll=float(ans_nll.sum()), answer_tokens=int(a1 - a0), answer_token_nll=[round(float(x), 4) for x in ans_nll],
            answer_token_rank=[int(x) for x in token_rank], answer_ids=[int(x) for x in ids[k, a0:a1]],
            first_rank=rank, first_p=float(first[ids[k, a0]].exp()),
            tail_nll_mean=float(tail_nll.mean()) if len(tail_nll) else None,
            top5=[(tok.decode([int(t)]), round(float(p.exp()), 4)) for p, t in zip(top.values, top.indices)]
            if s["candidate"] == 0 else None))
    return out


def _diverging(right, other):
    """Ilk AYRISAN token (ortak onek ayni baglamda ayni nll'i alir): (indeks, dogrunun orada sirasi, dogru - oteki log p)."""
    a, b = right["answer_ids"], other["answer_ids"]
    d = next((n for n in range(min(len(a), len(b))) if a[n] != b[n]), None)
    if d is None:
        return None
    return dict(index=d, right_rank=right["answer_token_rank"][d],
                logp_gap=other["answer_token_nll"][d] - right["answer_token_nll"][d])


def per_fact(seqs, scores, facts=FACTS):
    """Aday sonuclari olgu basina.  margin: en iyi yanlis - dogru (cevap kismi toplam nll); margin_any: dogru ya da ALSO'dan
    en iyisiyle; margin_per_token: token basina ortalamayla (uzunluk / bolunme duyarliligi icin); diverging: yanlis basina
    ilk ayrisan token'da dogrunun sirasi ve log p farki."""
    rows = []
    by_fact = {}
    for s, r in zip(seqs, scores):
        by_fact.setdefault(s["fact"], []).append((s, r))
    for i, f in enumerate(facts):
        cand = by_fact[i]
        right = cand[0][1]
        alts = [dict(text=s["text"], answer_nll=r["answer_nll"], answer_tokens=r["answer_tokens"], first_rank=r["first_rank"],
                     answer_token_nll=r["answer_token_nll"]) for s, r in cand if s["role"] == "also"]
        wrong_raw = [(s, r) for s, r in cand if s["role"] == "wrong"]
        wrong = [dict(text=s["text"], answer_nll=r["answer_nll"], answer_tokens=r["answer_tokens"], first_rank=r["first_rank"],
                      answer_token_nll=r["answer_token_nll"], diverging=_diverging(right, r)) for s, r in wrong_raw]
        best_wrong = min(w["answer_nll"] for w in wrong)
        best_right = min([right["answer_nll"]] + [a["answer_nll"] for a in alts])
        rows.append(dict(fact=i, kind=f["kind"], prompt=f["prompt"], answer=f["answer"], first_diverging_rank=max(
                             [w["diverging"]["right_rank"] for w in wrong if w["diverging"]] or [right["first_rank"]]),
                         answer_nll=right["answer_nll"], answer_tokens=right["answer_tokens"],
                         answer_token_nll=right["answer_token_nll"], first_rank=right["first_rank"],
                         first_p=right["first_p"], tail_nll_mean=right["tail_nll_mean"], top5=right["top5"],
                         also=alts, wrong=wrong, margin=best_wrong - right["answer_nll"],
                         correct_best=all(right["answer_nll"] < w["answer_nll"] for w in wrong),
                         margin_any=best_wrong - best_right, correct_best_any=best_right < best_wrong,
                         margin_per_token=min(w["answer_nll"] / w["answer_tokens"] for w in wrong)
                         - right["answer_nll"] / right["answer_tokens"]))
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
    L.append("%-44s %s" % ("marj ortalama (nat)", " ".join("%16.3f" % (sum(x["margin"] for x in r["facts"]) / len(FACTS))
                                                           for r in results)))
    L.append("%-44s %s" % ("marj ALSO dahil ortalama", " ".join("%16.3f" % (sum(x["margin_any"] for x in r["facts"]) / len(FACTS))
                                                                for r in results)))
    L.append("%-44s %s" % ("cevap nll ortalama", " ".join("%16.3f" % (sum(x["answer_nll"] for x in r["facts"]) / len(FACTS))
                                                          for r in results)))
    return L


# ---- siklik esigi (kullanici, 2 Ekim: oneri 1 "Olur eklet."): olgu derlemde kac kez geciyor, model onu biliyor mu;
# esik yedekler boyunca asagi kayiyorsa egitim sinirli, duruyorsa kapasite sinirli.

# (ulke istemdeki yazimiyla, sayim icin ozne, baskent, kabul edilen baska yazimlar)
CAPITALS = (
    ("Afghanistan", "Afghanistan", "Kabul", ()), ("Albania", "Albania", "Tirana", ()),
    ("Algeria", "Algeria", "Algiers", ()), ("Argentina", "Argentina", "Buenos Aires", ()),
    ("Armenia", "Armenia", "Yerevan", ()), ("Australia", "Australia", "Canberra", ()),
    ("Austria", "Austria", "Vienna", ()), ("Azerbaijan", "Azerbaijan", "Baku", ()),
    ("Bangladesh", "Bangladesh", "Dhaka", ()), ("Belarus", "Belarus", "Minsk", ()),
    ("Belgium", "Belgium", "Brussels", ()), ("Brazil", "Brazil", "Brasília", ("Brasilia",)),
    ("Bulgaria", "Bulgaria", "Sofia", ()), ("Cambodia", "Cambodia", "Phnom Penh", ()),
    ("Canada", "Canada", "Ottawa", ()), ("Chile", "Chile", "Santiago", ()),
    ("China", "China", "Beijing", ()), ("Colombia", "Colombia", "Bogotá", ("Bogota",)),
    ("Croatia", "Croatia", "Zagreb", ()), ("Cuba", "Cuba", "Havana", ()),
    ("the Czech Republic", "Czech Republic", "Prague", ()), ("Denmark", "Denmark", "Copenhagen", ()),
    ("Ecuador", "Ecuador", "Quito", ()), ("Egypt", "Egypt", "Cairo", ()),
    ("Estonia", "Estonia", "Tallinn", ()), ("Ethiopia", "Ethiopia", "Addis Ababa", ()),
    ("Finland", "Finland", "Helsinki", ()), ("France", "France", "Paris", ()),
    ("Germany", "Germany", "Berlin", ()), ("Ghana", "Ghana", "Accra", ()),
    ("Greece", "Greece", "Athens", ()), ("Hungary", "Hungary", "Budapest", ()),
    ("Iceland", "Iceland", "Reykjavik", ()), ("India", "India", "New Delhi", ()),
    ("Indonesia", "Indonesia", "Jakarta", ()), ("Iran", "Iran", "Tehran", ()),
    ("Iraq", "Iraq", "Baghdad", ()), ("Ireland", "Ireland", "Dublin", ()),
    ("Italy", "Italy", "Rome", ()), ("Jamaica", "Jamaica", "Kingston", ()),
    ("Japan", "Japan", "Tokyo", ()), ("Jordan", "Jordan", "Amman", ()),
    ("Kenya", "Kenya", "Nairobi", ()), ("Laos", "Laos", "Vientiane", ()),
    ("Latvia", "Latvia", "Riga", ()), ("Lebanon", "Lebanon", "Beirut", ()),
    ("Libya", "Libya", "Tripoli", ()), ("Lithuania", "Lithuania", "Vilnius", ()),
    ("Madagascar", "Madagascar", "Antananarivo", ()), ("Malaysia", "Malaysia", "Kuala Lumpur", ()),
    ("Mali", "Mali", "Bamako", ()), ("Mexico", "Mexico", "Mexico City", ()),
    ("Mongolia", "Mongolia", "Ulaanbaatar", ("Ulan Bator",)), ("Morocco", "Morocco", "Rabat", ()),
    ("Mozambique", "Mozambique", "Maputo", ()), ("Nepal", "Nepal", "Kathmandu", ()),
    ("the Netherlands", "Netherlands", "Amsterdam", ()), ("New Zealand", "New Zealand", "Wellington", ()),
    ("Nicaragua", "Nicaragua", "Managua", ()), ("Niger", "Niger", "Niamey", ()),
    ("Nigeria", "Nigeria", "Abuja", ()), ("North Korea", "North Korea", "Pyongyang", ()),
    ("Norway", "Norway", "Oslo", ()), ("Pakistan", "Pakistan", "Islamabad", ()),
    ("Paraguay", "Paraguay", "Asunción", ("Asuncion",)), ("Peru", "Peru", "Lima", ()),
    ("the Philippines", "Philippines", "Manila", ()), ("Poland", "Poland", "Warsaw", ()),
    ("Portugal", "Portugal", "Lisbon", ()), ("Qatar", "Qatar", "Doha", ()),
    ("Romania", "Romania", "Bucharest", ()), ("Russia", "Russia", "Moscow", ()),
    ("Rwanda", "Rwanda", "Kigali", ()), ("Saudi Arabia", "Saudi Arabia", "Riyadh", ()),
    ("Senegal", "Senegal", "Dakar", ()), ("Serbia", "Serbia", "Belgrade", ()),
    ("Slovakia", "Slovakia", "Bratislava", ()), ("Slovenia", "Slovenia", "Ljubljana", ()),
    ("Somalia", "Somalia", "Mogadishu", ()), ("South Korea", "South Korea", "Seoul", ()),
    ("Spain", "Spain", "Madrid", ()), ("Sudan", "Sudan", "Khartoum", ()),
    ("Sweden", "Sweden", "Stockholm", ()), ("Switzerland", "Switzerland", "Bern", ("Berne",)),
    ("Syria", "Syria", "Damascus", ()), ("Taiwan", "Taiwan", "Taipei", ()),
    ("Tanzania", "Tanzania", "Dodoma", ()), ("Thailand", "Thailand", "Bangkok", ()),
    ("Tunisia", "Tunisia", "Tunis", ()), ("Turkey", "Turkey", "Ankara", ()),
    ("Uganda", "Uganda", "Kampala", ()), ("Ukraine", "Ukraine", "Kyiv", ("Kiev",)),
    ("the United Kingdom", "United Kingdom", "London", ()), ("the United States", "United States", "Washington", ()),
    ("Uruguay", "Uruguay", "Montevideo", ()), ("Uzbekistan", "Uzbekistan", "Tashkent", ()),
    ("Venezuela", "Venezuela", "Caracas", ()), ("Vietnam", "Vietnam", "Hanoi", ()),
    ("Zambia", "Zambia", "Lusaka", ()), ("Zimbabwe", "Zimbabwe", "Harare", ()),
    ("Angola", "Angola", "Luanda", ()), ("Botswana", "Botswana", "Gaborone", ()),
    ("Namibia", "Namibia", "Windhoek", ()), ("Bhutan", "Bhutan", "Thimphu", ()),
    ("Kyrgyzstan", "Kyrgyzstan", "Bishkek", ()), ("Tajikistan", "Tajikistan", "Dushanbe", ()),
    ("Turkmenistan", "Turkmenistan", "Ashgabat", ()), ("North Macedonia", "Macedonia", "Skopje", ()),
    ("Montenegro", "Montenegro", "Podgorica", ()), ("Bosnia and Herzegovina", "Bosnia", "Sarajevo", ()),
    ("Malta", "Malta", "Valletta", ()), ("Cyprus", "Cyprus", "Nicosia", ()),
    ("Haiti", "Haiti", "Port-au-Prince", ()), ("the Dominican Republic", "Dominican Republic", "Santo Domingo", ()),
    ("Honduras", "Honduras", "Tegucigalpa", ()), ("Guatemala", "Guatemala", "Guatemala City", ()),
    ("El Salvador", "El Salvador", "San Salvador", ()), ("Costa Rica", "Costa Rica", "San José", ("San Jose",)),
    ("the Bahamas", "Bahamas", "Nassau", ()), ("Fiji", "Fiji", "Suva", ()),
    ("Papua New Guinea", "Papua New Guinea", "Port Moresby", ()), ("Liberia", "Liberia", "Monrovia", ()),
    ("Sierra Leone", "Sierra Leone", "Freetown", ()), ("Burkina Faso", "Burkina Faso", "Ouagadougou", ()),
    ("Eritrea", "Eritrea", "Asmara", ()), ("Malawi", "Malawi", "Lilongwe", ()),
    ("Gabon", "Gabon", "Libreville", ()), ("Togo", "Togo", "Lomé", ("Lome",)),
    ("Mauritania", "Mauritania", "Nouakchott", ()), ("Oman", "Oman", "Muscat", ()),
    ("Kuwait", "Kuwait", "Kuwait City", ()), ("Bahrain", "Bahrain", "Manama", ()),
    ("the United Arab Emirates", "United Arab Emirates", "Abu Dhabi", ()),
)
# (element adi, sembol)
ELEMENTS = (
    ("hydrogen", "H"), ("helium", "He"), ("lithium", "Li"), ("beryllium", "Be"), ("boron", "B"), ("carbon", "C"),
    ("nitrogen", "N"), ("oxygen", "O"), ("fluorine", "F"), ("neon", "Ne"), ("sodium", "Na"), ("magnesium", "Mg"),
    ("aluminum", "Al"), ("silicon", "Si"), ("phosphorus", "P"), ("sulfur", "S"), ("chlorine", "Cl"), ("argon", "Ar"),
    ("potassium", "K"), ("calcium", "Ca"), ("scandium", "Sc"), ("titanium", "Ti"), ("vanadium", "V"),
    ("chromium", "Cr"), ("manganese", "Mn"), ("iron", "Fe"), ("cobalt", "Co"), ("nickel", "Ni"), ("copper", "Cu"),
    ("zinc", "Zn"), ("gallium", "Ga"), ("germanium", "Ge"), ("arsenic", "As"), ("selenium", "Se"), ("bromine", "Br"),
    ("krypton", "Kr"), ("rubidium", "Rb"), ("strontium", "Sr"), ("yttrium", "Y"), ("zirconium", "Zr"),
    ("niobium", "Nb"), ("molybdenum", "Mo"), ("technetium", "Tc"), ("ruthenium", "Ru"), ("rhodium", "Rh"),
    ("palladium", "Pd"), ("silver", "Ag"), ("cadmium", "Cd"), ("indium", "In"), ("tin", "Sn"), ("antimony", "Sb"),
    ("tellurium", "Te"), ("iodine", "I"), ("xenon", "Xe"), ("cesium", "Cs"), ("barium", "Ba"), ("lanthanum", "La"),
    ("cerium", "Ce"), ("neodymium", "Nd"), ("europium", "Eu"), ("gadolinium", "Gd"), ("hafnium", "Hf"),
    ("tantalum", "Ta"), ("tungsten", "W"), ("rhenium", "Re"), ("osmium", "Os"), ("iridium", "Ir"),
    ("platinum", "Pt"), ("gold", "Au"), ("mercury", "Hg"), ("thallium", "Tl"), ("lead", "Pb"), ("bismuth", "Bi"),
    ("polonium", "Po"), ("astatine", "At"), ("radon", "Rn"), ("francium", "Fr"), ("radium", "Ra"),
    ("actinium", "Ac"), ("thorium", "Th"), ("uranium", "U"), ("plutonium", "Pu"), ("americium", "Am"),
    ("curium", "Cm"), ("einsteinium", "Es"), ("nobelium", "No"),
)
# (istem, sayim icin ozne, yil)
DATES = (
    ("The French Revolution began in", "French Revolution", "1789"),
    ("The American Declaration of Independence was signed in", "Declaration of Independence", "1776"),
    ("World War I began in", "World War I", "1914"), ("World War I ended in", "World War I", "1918"),
    ("World War II began in", "World War II", "1939"), ("World War II ended in", "World War II", "1945"),
    ("The Berlin Wall fell in", "Berlin Wall", "1989"), ("The Berlin Wall was built in", "Berlin Wall", "1961"),
    ("The Apollo 11 mission landed on the Moon in", "Apollo 11", "1969"),
    ("Christopher Columbus first reached the Americas in", "Columbus", "1492"),
    ("The Titanic sank in", "Titanic", "1912"), ("The Magna Carta was signed in", "Magna Carta", "1215"),
    ("The Battle of Hastings took place in", "Battle of Hastings", "1066"),
    ("The Russian Revolution took place in", "Russian Revolution", "1917"),
    ("The American Civil War began in", "American Civil War", "1861"),
    ("The American Civil War ended in", "American Civil War", "1865"),
    ("Abraham Lincoln was assassinated in", "Lincoln", "1865"),
    ("The stock market crash that started the Great Depression happened in", "Great Depression", "1929"),
    ("The Soviet Union collapsed in", "Soviet Union", "1991"),
    ("The attack on Pearl Harbor took place in", "Pearl Harbor", "1941"),
    ("The atomic bomb was dropped on Hiroshima in", "Hiroshima", "1945"),
    ("The September 11 attacks happened in", "September 11", "2001"),
    ("The Wright brothers made their first powered flight in", "Wright brothers", "1903"),
    ("Martin Luther posted his Ninety-five Theses in", "Martin Luther", "1517"),
    ("The Battle of Waterloo was fought in", "Waterloo", "1815"),
    ("The United Nations was founded in", "United Nations", "1945"),
    ("Nelson Mandela was released from prison in", "Mandela", "1990"),
    ("The Chernobyl disaster happened in", "Chernobyl", "1986"),
    ("Charles Darwin published On the Origin of Species in", "Origin of Species", "1859"),
    ("The Boston Tea Party took place in", "Boston Tea Party", "1773"),
    ("The Great Fire of London happened in", "Great Fire of London", "1666"),
    ("The Spanish Armada was defeated in", "Spanish Armada", "1588"),
    ("The Treaty of Versailles was signed in", "Treaty of Versailles", "1919"),
    ("The Emancipation Proclamation was issued in", "Emancipation Proclamation", "1863"),
    ("The Louisiana Purchase took place in", "Louisiana Purchase", "1803"),
    ("The Cuban Missile Crisis took place in", "Cuban Missile Crisis", "1962"),
    ("Sputnik was launched in", "Sputnik", "1957"),
    ("Yuri Gagarin became the first human in space in", "Gagarin", "1961"),
    ("The Korean War began in", "Korean War", "1950"), ("The Vietnam War ended in", "Vietnam War", "1975"),
    ("Martin Luther King Jr. gave his I Have a Dream speech in", "I Have a Dream", "1963"),
    ("John F. Kennedy was assassinated in", "Kennedy", "1963"),
    ("The Nineteenth Amendment was ratified in", "Nineteenth Amendment", "1920"),
    ("The Ottoman Turks captured Constantinople in", "Constantinople", "1453"),
    ("The Battle of Gettysburg was fought in", "Gettysburg", "1863"),
    ("The Panama Canal opened in", "Panama Canal", "1914"), ("The Suez Canal opened in", "Suez Canal", "1869"),
    ("The Eiffel Tower was completed in", "Eiffel Tower", "1889"),
    ("The Statue of Liberty was dedicated in", "Statue of Liberty", "1886"),
    ("Alexander Fleming discovered penicillin in", "penicillin", "1928"),
    ("Albert Einstein published the theory of special relativity in", "special relativity", "1905"),
    ("The first iPhone was released in", "iPhone", "2007"),
    ("The People's Republic of China was founded in", "People's Republic of China", "1949"),
    ("The State of Israel was founded in", "Israel", "1948"),
    ("The Mayflower arrived in America in", "Mayflower", "1620"), ("Jamestown was founded in", "Jamestown", "1607"),
    ("The Battle of Trafalgar took place in", "Trafalgar", "1805"),
    ("The Hundred Years' War began in", "Hundred Years", "1337"),
    ("Queen Victoria became queen in", "Queen Victoria", "1837"),
    ("The Glorious Revolution took place in", "Glorious Revolution", "1688"),
    ("The Peace of Westphalia was signed in", "Westphalia", "1648"),
    ("Galileo was tried by the Inquisition in", "Galileo", "1633"),
    ("Isaac Newton published the Principia in", "Principia", "1687"),
    ("The Meiji Restoration began in", "Meiji Restoration", "1868"),
    ("The Mexican-American War began in", "Mexican-American War", "1846"),
    ("The Great Chicago Fire happened in", "Chicago Fire", "1871"),
    ("The great San Francisco earthquake happened in", "San Francisco earthquake", "1906"),
    ("Hurricane Katrina struck New Orleans in", "Katrina", "2005"),
    ("The Hubble Space Telescope was launched in", "Hubble", "1990"),
    ("The Space Shuttle Challenger disaster happened in", "Challenger", "1986"),
    ("The Human Genome Project was completed in", "Human Genome Project", "2003"),
    ("Dolly the sheep was cloned in", "Dolly", "1996"),
    ("Watson and Crick described the structure of DNA in", "Watson and Crick", "1953"),
    ("Marie Curie won her first Nobel Prize in", "Marie Curie", "1903"),
    ("The Rwandan genocide took place in", "Rwandan genocide", "1994"),
    ("Barack Obama was first elected president in", "Obama", "2008"),
    ("Germany was reunified in", "reunification", "1990"),
    ("The Indian Ocean tsunami happened in", "Indian Ocean tsunami", "2004"),
    ("The Battle of Stalingrad ended in", "Stalingrad", "1943"), ("D-Day took place in", "D-Day", "1944"),
    ("Thomas Edison invented the phonograph in", "phonograph", "1877"),
    ("Alexander Graham Bell patented the telephone in", "Graham Bell", "1876"),
    ("Gutenberg's printing press was developed around", "Gutenberg", "1440"),
    ("The Battle of Bunker Hill was fought in", "Bunker Hill", "1775"),
    ("The US Constitution was signed in", "Constitution", "1787"),
    ("India gained independence from Britain in", "India", "1947"),
    ("The Prohibition era in the United States began in", "Prohibition", "1920"),
    ("The Bolsheviks seized power in Russia in", "Bolsheviks", "1917"),
    ("Napoleon Bonaparte crowned himself emperor in", "Napoleon", "1804"),
    ("The Battle of Midway was fought in", "Midway", "1942"),
    ("The Treaty of Paris ended the American Revolutionary War in", "Treaty of Paris", "1783"),
    ("The Wall Street Journal was first published in", "Wall Street Journal", "1889"),
    ("The first Olympic Games of the modern era were held in", "Olympic Games", "1896"),
)
YEAR_OFFSETS = (-10, -3, 3, 10)      # tarih celdiricileri: dogru yilin komsulari (kesin yil bilgisi olculur)
FREQ_DISTRACTORS = 4                 # baskent ve sembolde: listeden rastgele baska cevaplar (tohum 0)
# sayimi gurultulu semboller: Ingilizce kelime ya da tek harf (bas harf, zamir) -- esik iki kez: hepsiyle ve bunlarsiz
NOISY_SYMBOLS = {"H", "B", "C", "N", "O", "F", "P", "S", "K", "V", "Y", "W", "U", "I", "He", "Be", "In", "As", "At",
                 "No", "Am", "Co", "Es"}
FREQ_WINDOW = 16                     # sayim: ozne ile cevap arasinda en cok bu kadar token (iki yon), ayni belge


def freq_facts():
    """Uc tur olgu -> FACTS bicimi (+ subject: sayim icin ozne, answers: dogru yazimlarin hepsi).  Celdiriciler tohum 0."""
    import random
    rng = random.Random(0)
    out = []
    caps = [c[2] for c in CAPITALS]
    for country, subject, capital, also in CAPITALS:
        wrong = rng.sample([c for c in caps if c != capital], FREQ_DISTRACTORS)
        out.append(dict(kind="capital", prompt="The capital of %s is" % country, answer=" " + capital,
                        also=tuple(" " + a for a in also), wrong=tuple(" " + w for w in wrong), tail=".",
                        subject=" " + subject))
    syms = [e[1] for e in ELEMENTS]
    for name, sym in ELEMENTS:
        wrong = rng.sample([x for x in syms if x != sym], FREQ_DISTRACTORS)
        out.append(dict(kind="element", prompt="The chemical symbol for %s is" % name, answer=" " + sym, also=(),
                        wrong=tuple(" " + w for w in wrong), tail=".", subject=" " + name, noisy=sym in NOISY_SYMBOLS))
    for prompt, subject, year in DATES:
        wrong = tuple(" %d" % (int(year) + k) for k in YEAR_OFFSETS)
        out.append(dict(kind="date", prompt=prompt, answer=" " + year, also=(), wrong=wrong, tail=".",
                        subject=" " + subject))
    return out


def _token_positions(a, firsts):
    """a icinde firsts token'larinin konumlari, token'a gore gruplu: {token: artan konumlar}.  Tek gecis (arama tablosu)."""
    import numpy as np
    lut = np.zeros(1 << 16, dtype=bool)
    lut[list(firsts)] = True
    idx = np.flatnonzero(lut[a])
    order = np.argsort(a[idx], kind="stable")
    idx, vals = idx[order], a[idx][order]
    cut = np.flatnonzero(np.diff(vals)) + 1
    return {int(g[0]): p for g, p in zip(np.split(vals, cut), np.split(idx, cut)) if len(g)}


def _phrase_positions(a, ids, by_first):
    import numpy as np
    cand = by_first.get(ids[0], np.array([], dtype=np.int64))
    cand = cand[cand + len(ids) <= len(a)]
    for k, t in enumerate(ids[1:], 1):
        if not len(cand):
            break
        cand = cand[a[cand + k] == t]
    return cand


def freq_count(fw_root, tok, facts, window=FREQ_WINDOW, shards=None, log=print):
    """Olgu basina derlem sayimi (butun parcalar; shard 13'un valid belgeleri dahil, ~%0,04):
    n_subject  ozne (bosluklu yazim) gecisi
    near     cevabin (dogru yazimlarindan biri) ozneden en cok `window` token once ya da sonra, AYNI belgede basladigi ozne
             gecisi.  Ana sikilik olcusu
    docs     ozne ve cevabin ikisini birden iceren belge sayisi
    n_answer   cevabin gecisi (ozneden bagimsiz)"""
    import numpy as np
    d = os.path.join(fw_root, "gpt2")
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids
    subj = [enc(f["subject"]) for f in facts]
    ans = [[enc(x) for x in (f["answer"],) + tuple(f.get("also", ()))] for f in facts]
    firsts = {x[0] for x in subj} | {y[0] for xs in ans for y in xs}
    tot = [dict(n_subject=0, near=0, docs=0, n_answer=0) for _ in facts]
    found = sorted(int(f[6:9]) for f in os.listdir(d) if f.startswith("shard_") and f.endswith(".bin"))
    tokens = 0
    for i in found if shards is None else [s for s in found if s in shards]:
        t0 = time.time()
        a = np.fromfile(os.path.join(d, "shard_%03d.bin" % i), dtype=np.uint16)
        offsets = np.load(os.path.join(d, "shard_%03d_offsets.npy" % i))
        tokens += len(a)
        by_first = _token_positions(a, firsts)
        doc = lambda p: np.searchsorted(offsets, p, side="right") - 1
        for k, f in enumerate(facts):
            sp = _phrase_positions(a, subj[k], by_first)
            ap = np.unique(np.concatenate([_phrase_positions(a, x, by_first) for x in ans[k]] + [np.array([], np.int64)]))
            t = tot[k]
            t["n_subject"] += len(sp)
            t["n_answer"] += len(ap)
            if len(sp) and len(ap):
                lo = np.searchsorted(ap, sp - window)                  # ozneden once en cok window token ...
                j = np.minimum(lo, len(ap) - 1)
                hit = (lo < len(ap)) & (ap[j] <= sp + len(subj[k]) + window) & (doc(ap[j]) == doc(sp))
                t["near"] += int(hit.sum())
                t["docs"] += len(np.intersect1d(np.unique(doc(sp)), np.unique(doc(ap))))
        log("parca %03d: %.0f M token, %.0f sn" % (i, len(a) / 1e6, time.time() - t0))
        del a, by_first
    return dict(tokens=tokens, window=window, counts=[dict(t, prompt=f["prompt"], answer=f["answer"], kind=f["kind"],
                                                           subject=f["subject"], noisy=f.get("noisy", False))
                                                      for t, f in zip(tot, facts)])


def _logistic_floor(x, y, floor):
    """p = floor + (1 - floor) sigmoid(a + b x), en buyuk olabilirlik: standartlastirilmis x'te (a, b) izgarasi (b >= 0).
    -> (a, b) x'in kendi biriminde."""
    import numpy as np
    xm, xs = x.mean(), x.std() + 1e-9
    z = (x - xm) / xs
    A, B = np.meshgrid(np.linspace(-6, 6, 61), np.linspace(0, 8, 41), indexing="ij")
    s = 1 / (1 + np.exp(-(A.reshape(-1, 1) + B.reshape(-1, 1) * z)))
    p = np.clip(floor + (1 - floor) * s, 1e-6, 1 - 1e-6)
    ll = (y * np.log(p) + (1 - y) * np.log(1 - p)).sum(1)
    k = int(ll.argmax())
    a, b = A.reshape(-1)[k], B.reshape(-1)[k]
    return a - b * xm / xs, b / xs


def freq_threshold(count_res, fact_res, key="near", criterion="correct_best_any", weights="ema", min_step=4000,
                   boot=100, kinds_only=None, drop_noisy=False):
    """Bilinme orani ~ log10(1 + sayim), yedek basina: kutu tablosu (Wilson araligi) ve sans tabanli lojistik esik
    (p = taban + (1 - taban)/2'de sayim), olgulara gore bootstrap %90 araligi.  EMA'da min_step oncesi baslangic agirligiyla
    kirli (E2 R1).  -> (satirlar, ozet)."""
    import numpy as np
    # cevap ozneyle basliyorsa (Mexico City / Mexico) ozne cevabin icinde sayilir: sayim anlamsiz, disarida
    keep = np.array([(kinds_only is None or c["kind"] in kinds_only) and not (drop_noisy and c["noisy"])
                     and not c["answer"].startswith(c["subject"]) for c in count_res["counts"]])
    counts = np.array([c[key] for c in count_res["counts"]], dtype=float)[keep]
    kinds = np.array([c["kind"] for c in count_res["counts"]])[keep]
    x = np.log10(1 + counts)
    floor = 1.0 / (1 + FREQ_DISTRACTORS)
    edges = [0, 1, 10, 100, 1000, 10000, 1e9]
    L = ["## bilinme orani (%s) ~ sayim (%s, pencere %d) | agirlik %s, adim >= %d; sans %.2f | olgu %d (%s%s)" % (
        criterion, key, count_res["window"], weights, min_step, floor, keep.sum(), ",".join(kinds_only or ["hepsi"]),
        ", gurultulu semboller haric" if drop_noisy else ""),
         "%-12s %s | esik (sayim, %%90 araligi)" % ("adim", " ".join("%13s" % ("[%g,%g)" % (lo, hi))
                                                              for lo, hi in zip(edges[:-1], edges[1:])))]
    L.append("%-12s %s" % ("olgu sayisi", " ".join("%13d" % ((counts >= lo) & (counts < hi)).sum()
                                                   for lo, hi in zip(edges[:-1], edges[1:]))))
    rng = np.random.default_rng(0)
    summary = []
    for r in fact_res:
        if r["weights"] != weights or (r["step"] is not None and r["step"] < min_step):
            continue
        y = np.array([float(f[criterion]) for f in r["facts"]])[keep]
        cells = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (counts >= lo) & (counts < hi)
            n, k = m.sum(), y[m].sum()
            if not n:
                cells.append("%13s" % "-")
                continue
            p, z = k / n, 1.645
            c = (p + z * z / (2 * n)) / (1 + z * z / n)
            h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
            cells.append("%13s" % ("%.2f[%.2f,%.2f]" % (p, max(0, c - h), min(1, c + h))))

        def thr(xx, yy):
            a, b = _logistic_floor(xx, yy, floor)
            return 10 ** min(-a / b, 12.0) - 1 if b > 0 else 1e12
        t = thr(x, y)
        bs = []
        for _ in range(boot):
            i = rng.integers(0, len(x), len(x))
            bs.append(thr(x[i], y[i]))
        lo, hi = np.percentile(bs, [5, 95])
        per_kind = {k: float(y[kinds == k].mean()) for k in sorted(set(kinds))}
        summary.append(dict(label=r["label"], step=r["step"], threshold=t, ci90=[float(lo), float(hi)],
                            known=float(y.mean()), per_kind=per_kind))
        L.append("%-12s %s | %.0f [%.0f, %.0f]  bilinen %.2f  %s" % (
            r["label"][-12:], " ".join(cells), t, lo, hi, y.mean(), " ".join("%s %.2f" % kv for kv in per_kind.items())))
    return L, summary


# ---- yon asimetrisi (ajan L, Sorun 3): ayni 6 olay-yil olgusu iki yonde, kapali 6'li aday kumesiyle
DIRECTION_EVENTS = (
    ("The French Revolution began in", "1789", " the French Revolution began"),
    ("The Declaration of Independence was signed in", "1776", " the Declaration of Independence was signed"),
    ("The Russian Revolution began in", "1917", " the Russian Revolution began"),
    ("The Berlin Wall fell in", "1989", " the Berlin Wall fell"),
    ("World War II ended in", "1945", " World War II ended"),
    ("Christopher Columbus reached the Americas in", "1492", " Christopher Columbus reached the Americas"),
)
DIRECTION_PRIORS = ("It happened in", "In that year,")      # aday onselleri: kalibrasyon icin notr istemler


def direction_facts():
    """forward: olay -> yil (6 aday yil); reverse: "In <yil>," -> olay (6 aday olay); prior_*: ayni adaylar notr istemle
    (kalibre skor = nll(aday | istem) - nll(aday | notr))."""
    years = [" " + y for _, y, _ in DIRECTION_EVENTS]
    events = [e for _, _, e in DIRECTION_EVENTS]
    out = []
    for k, (prompt, year, event) in enumerate(DIRECTION_EVENTS):
        out.append(dict(kind="forward", prompt=prompt, answer=years[k], wrong=tuple(y for y in years if y != years[k]),
                        tail="."))
        out.append(dict(kind="reverse", prompt="In %s," % year, answer=event, wrong=tuple(e for e in events if e != event),
                        tail="."))
    out.append(dict(kind="prior_forward", prompt=DIRECTION_PRIORS[0], answer=years[0], wrong=tuple(years[1:]), tail="."))
    out.append(dict(kind="prior_reverse", prompt=DIRECTION_PRIORS[1], answer=events[0], wrong=tuple(events[1:]), tail="."))
    return out


def direction_summary(result):
    """Ham ve kalibre dogruluk (6'li, sans 1/6) ve marj, yon basina."""
    facts = result["facts"]
    prior = {}
    for x in facts:
        if x["kind"].startswith("prior_"):
            prior[x["kind"][6:]] = {x["answer"]: x["answer_nll"], **{w["text"]: w["answer_nll"] for w in x["wrong"]}}
    out = {}
    for kind in ("forward", "reverse"):
        acc = cal = 0
        margins, cal_margins = [], []
        rows = [x for x in facts if x["kind"] == kind]
        for x in rows:
            raw = {x["answer"]: x["answer_nll"], **{w["text"]: w["answer_nll"] for w in x["wrong"]}}
            adj = {k: v - prior[kind][k] for k, v in raw.items()}
            for d, store in ((raw, margins), (adj, cal_margins)):
                store.append(min(v for k, v in d.items() if k != x["answer"]) - d[x["answer"]])
        out[kind] = dict(n=len(rows), acc=sum(m > 0 for m in margins) / len(rows),
                         acc_calibrated=sum(m > 0 for m in cal_margins) / len(rows),
                         margin=sum(margins) / len(rows), margin_calibrated=sum(cal_margins) / len(rows),
                         margins=[round(m, 3) for m in margins], margins_calibrated=[round(m, 3) for m in cal_margins])
    return out


def _write(args, name, payload, path=None):
    """JSON yaz; path verilirse ayni dosyanin uzerine (kosu kesilirse o ana kadarki yedekler kalir)."""
    if path is None:
        out_dir = os.environ.get("KUYRUK_SONUC") or os.path.join(args.run, "analysis")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "%s%s_%s.json" % (name, "_" + args.label if args.label else "",
                                                       time.strftime("%Y%m%d_%H%M%S")))
    with open(path + ".part", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(path + ".part", path)
    return path


def freq_lines(results, facts):
    kinds = sorted({f["kind"] for f in facts})
    L = ["## siklik olgulari: tur basina dogru en iyi (ALSO dahil) / ilk ayrisan token sira 0 orani; n = %s" % (
        ", ".join("%s %d" % (k, sum(f["kind"] == k for f in facts)) for k in kinds))]
    for r in results:
        cells = []
        for k in kinds:
            xs = [x for x in r["facts"] if x["kind"] == k]
            cells.append("%s %.2f / %.2f" % (k, sum(x["correct_best_any"] for x in xs) / len(xs),
                                             sum(x["first_diverging_rank"] == 0 for x in xs) / len(xs)))
        L.append("%-12s %s" % (r["label"][-12:], "   ".join(cells)))
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(prog="analyze_capacity.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="kosu klasoru")
    ap.add_argument("measure", choices=("facts", "freq_facts", "freq_count", "freq_threshold", "direction"))
    ap.add_argument("--data", required=True, help="FineWeb koku (gpt2/tokenizer.json, gpt2/shard_*)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", default="", help="checkpoint adimlari: '2000,4000' ya da 'every:K' ya da 'all'")
    ap.add_argument("--finals", default="", help="'last,ema': model.pt / model_weight_ema.pt")
    ap.add_argument("--label", default="", help="cikti adina ek")
    ap.add_argument("--selftest", action="store_true", help="kucuk rastgele model, CPU")
    ap.add_argument("--shards", default="", help="freq_count: yalniz bu parcalar ('0,1'; bos = hepsi)")
    ap.add_argument("--counts", help="freq_threshold: freq_count ciktisi (.json)")
    ap.add_argument("--inputs", nargs="+", help="freq_threshold: freq_facts ciktilari (.json)")
    args = ap.parse_args(argv)
    t0 = time.time()
    tok, eot = load_tokenizer(args.data)
    if args.measure == "freq_count":
        facts = freq_facts()
        shards = [int(x) for x in args.shards.split(",")] if args.shards else None
        res = freq_count(args.data, tok, facts, shards=shards, log=lambda m: print(m, flush=True))
        for c in sorted(res["counts"], key=lambda c: -c["near"])[:: max(1, len(facts) // 40)]:
            print("%8d near %8d docs %9d ozne  %-40s %s" % (c["near"], c["docs"], c["n_subject"], c["prompt"][:40], c["answer"]))
        path = _write(args, "capacity_freq_count", dict(measure="freq_count", secs=round(time.time() - t0, 1), **res))
        print("yazildi: %s  (%.0f sn)" % (path, time.time() - t0))
        return
    if args.measure == "freq_threshold":
        counts = json.load(open(args.counts, encoding="utf-8"))
        results = [r for p in args.inputs for r in json.load(open(p, encoding="utf-8"))["results"]]
        results.sort(key=lambda r: (r["step"] is None, r["step"] or 0))
        out = {}
        for weights in ("ema", "last"):
            for kinds, noisy in ((None, True), (["capital"], False), (["element"], True), (["date"], False)):
                L, summ = freq_threshold(counts, results, weights=weights, kinds_only=kinds, drop_noisy=noisy,
                                         min_step=4000 if weights == "ema" else 0)
                print("\n".join(L) + "\n", flush=True)
                out["%s|%s|%s" % (weights, ",".join(kinds or ["all"]), "clean" if noisy else "raw")] = summ
        path = _write(args, "capacity_freq_threshold", dict(measure="freq_threshold", counts=args.counts, inputs=args.inputs,
                                                            summary=out))
        print("yazildi: %s" % path)
        return
    config = I._config(args.run)
    facts = FACTS if args.measure == "facts" else freq_facts() if args.measure == "freq_facts" else direction_facts()
    seqs = build_sequences(tok, eot, facts)
    bad = [s["text"] for s in seqs if not s["joint_ok"]]
    print("olgu %d, aday %d; ayri/birlesik kodlama farkli: %s" % (len(facts), len(seqs), bad or "yok"), flush=True)
    if args.selftest:
        sets = selftest_sets(config)
    else:
        packs = I._checkpoints(args.run)
        spec = args.steps
        steps = ([] if not spec else list(packs) if spec == "all" else
                 [s for s in packs if s % int(spec[6:]) == 0] if spec.startswith("every:") else [int(s) for s in spec.split(",")])
        sets = weight_sets(args.run, steps, [x for x in args.finals.split(",") if x], config)
    results = []
    name = dict(facts="capacity_facts", freq_facts="capacity_freq_facts", direction="capacity_direction")[args.measure]
    payload = lambda: dict(measure=args.measure, run=config.get("name"), selftest=args.selftest,
                           secs=round(time.time() - t0, 1),
                           facts=[dict(f, wrong=list(f["wrong"]), also=list(f.get("also", ()))) for f in facts],
                           token_ids=[dict(fact=s["fact"], candidate=s["candidate"], role=s["role"], ids=s["ids"])
                                      for s in seqs], results=results)
    path = None
    for label, step, kind, model in sets:
        model = model.to(args.device)
        scores = score(model, seqs, tok, args.device, eot)
        results.append(dict(label=label, step=step, weights=kind, facts=per_fact(seqs, scores, facts),
                            ema_init_share=EMA_DECAY ** step if kind == "ema" and step else None))
        if args.measure == "direction":
            results[-1]["direction"] = direction_summary(results[-1])
        print("   %s: dogru en iyi %d / %d" % (label, sum(x["correct_best_any"] for x in results[-1]["facts"]), len(facts)),
              flush=True)
        path = _write(args, name, payload(), path)              # her yedekten sonra: kesilirse kismi sonuc kalir
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    if args.measure == "direction":
        print("## yon: 6'li kapali aday, sans 0,17 -- ham / kalibre dogruluk, ortalama marj (nat)")
        for r in results:
            print("%-14s %s" % (r["label"][-14:], "   ".join("%s %.2f / %.2f  marj %.2f / %.2f" % (
                k, v["acc"], v["acc_calibrated"], v["margin"], v["margin_calibrated"]) for k, v in r["direction"].items())))
    else:
        print("\n".join(summary_lines(results) if args.measure == "facts" else freq_lines(results, facts)))
    path = _write(args, name, payload(), path)
    print("yazildi: %s  (%.0f sn)" % (path, time.time() - t0))


if __name__ == "__main__":
    main()
