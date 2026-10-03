"""make_country_sentences -- Model Z ilk verisi: ulke olgularindan duz cumleler (kullanici, 3 Ekim: "Önce tüm veriyi
çıkartan sonra eğitim ve sınav").  Ayni olgu farkli kaliplarla: ulke bazen ozne, bazen nesne, tamlayan ya da yer.

1. Butun veri: her ulke x her dolu olgu x o olgunun her kalibi -> country_sentences.jsonl
2. Egitim / sinav (tohum sabit):
     her ulkenin dolu olgularindan 3'u ayrilir (sinav, gorulmemis olgu); kalanlarin her biri egitimde TEK kalipla
     (kalip ulkeden ulkeye doner: ayni olgu turu farkli ulkelerde farkli gorevle gecer)
     sinav bolmeleri: seen (egitim cumlelerinden ornek), unseen_template (egitimdeki olgu, baska kalip),
     unseen_fact (ayrilan olgu, butun kaliplari)
   -> country_train.jsonl, country_exam.jsonl
Kelime basina gorev (roles) yalniz OLCUM icin; model egitimde gormez.

    python make_country_sentences.py
"""
import json
import os
import random
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
HELD_OUT = 3            # ulke basina sinava ayrilan olgu
SEEN_EXAM = 100         # sinavin 'seen' bolmesi: egitim cumlelerinden bu kadar
SEED = 0
THE = {"Netherlands", "United Kingdom", "United States", "Czech Republic", "Philippines", "United Arab Emirates"}

# kalip: (parca, gorev) listesi; {C} ulke, {F} olgu.  Gorevler: subject verb object complement genitive location function
TEMPLATES = {
    "capital": [
        [("The capital", "subject"), ("of", "function"), ("{C}", "genitive"), ("is", "verb"), ("{F}", "complement")],
        [("{F}", "subject"), ("is", "verb"), ("the capital", "complement"), ("of", "function"), ("{C}", "genitive")],
        [("{C}", "subject"), ("has", "verb"), ("{F}", "object"), ("as its capital", "complement")],
    ],
    "continent": [
        [("{C}", "subject"), ("is", "verb"), ("in", "function"), ("{F}", "location")],
        [("{F}", "subject"), ("includes", "verb"), ("{C}", "object")],
    ],
    "language": [
        [("People", "subject"), ("in", "function"), ("{C}", "location"), ("speak", "verb"), ("{F}", "object")],
        [("{F}", "subject"), ("is spoken", "verb"), ("in", "function"), ("{C}", "location")],
    ],
    "currency": [
        [("{C}", "subject"), ("uses", "verb"), ("the {F}", "object")],
        [("The {F}", "subject"), ("is", "verb"), ("the currency", "complement"), ("of", "function"), ("{C}", "genitive")],
        [("People", "subject"), ("in", "function"), ("{C}", "location"), ("pay", "verb"), ("with", "function"),
         ("the {F}", "object")],
    ],
    "city": [
        [("{F}", "subject"), ("is", "verb"), ("a large city", "complement"), ("in", "function"), ("{C}", "location")],
        [("{C}", "subject"), ("has", "verb"), ("a large city called {F}", "object")],
    ],
    "neighbor": [
        [("{C}", "subject"), ("borders", "verb"), ("{F}", "object")],
        [("{F}", "subject"), ("borders", "verb"), ("{C}", "object")],
        [("{F}", "subject"), ("is", "verb"), ("a neighbor", "complement"), ("of", "function"), ("{C}", "genitive")],
    ],
    "river": [
        [("The {F}", "subject"), ("flows", "verb"), ("through", "function"), ("{C}", "location")],
        [("The {F}", "subject"), ("is", "verb"), ("a river", "complement"), ("in", "function"), ("{C}", "location")],
    ],
    "dish": [
        [("People", "subject"), ("in", "function"), ("{C}", "location"), ("eat", "verb"), ("{F}", "object")],
        [("{C}", "subject"), ("is famous", "verb"), ("for", "function"), ("{F}", "object")],
    ],
    "landmark": [
        [("{F}", "subject"), ("is", "verb"), ("in", "function"), ("{C}", "location")],
        [("Tourists", "subject"), ("in", "function"), ("{C}", "location"), ("visit", "verb"), ("{F}", "object")],
        [("{C}", "subject"), ("is home", "verb"), ("to", "function"), ("{F}", "object")],
    ],
    "sea": [
        [("{C}", "subject"), ("lies", "verb"), ("on", "function"), ("the {F}", "location")],
        [("The {F}", "subject"), ("touches", "verb"), ("{C}", "object")],
    ],
}


def country_name(c):
    return ("the " + c) if c in THE else c


def build(country, fact, value, k):
    """-> kayit: cumle, kelimeler, kelime basina gorev, ulkenin ve olgunun gorevi."""
    words, roles, c_role, f_role = [], [], None, None
    for text, role in TEMPLATES[fact][k]:
        if "{C}" in text:
            c_role = role
        if "{F}" in text:
            f_role = role
        part = text.replace("{C}", country_name(country)).replace("{F}", value).split()
        words += part
        roles += [role] * len(part)
    if words[0] == "the":                              # cumle basi
        words[0] = "The"
    words.append(".")
    roles.append("punct")
    return dict(sentence=" ".join(words[:-1]) + ".", words=words, roles=roles, country=country, fact=fact, template=k,
                country_role=c_role, fact_role=f_role)


def main():
    table = json.load(open(os.path.join(DATA, "country_facts.json"), encoding="utf-8"))
    facts = [f for f in table["fields"]]
    everything = []
    for row in table["countries"]:
        for fact in facts:
            if row[fact] is not None:
                for k in range(len(TEMPLATES[fact])):
                    everything.append(build(row["country"], fact, row[fact], k))
    for i, r in enumerate(everything):
        r["id"] = i
    rng = random.Random(SEED)
    train, exam = [], []
    turn = Counter()                                   # olgu turu basina kalip sirasi: ulkeden ulkeye doner
    for row in table["countries"]:
        mine = [f for f in facts if row[f] is not None]
        held = set(rng.sample(mine, HELD_OUT))
        for fact in mine:
            rows = [r for r in everything if r["country"] == row["country"] and r["fact"] == fact]
            if fact in held:
                exam += [dict(r, split="unseen_fact") for r in rows]
                continue
            k = turn[fact] % len(TEMPLATES[fact])
            turn[fact] += 1
            train += [dict(r, split="train") for r in rows if r["template"] == k]
            exam += [dict(r, split="unseen_template") for r in rows if r["template"] != k]
    exam += [dict(r, split="seen") for r in rng.sample(train, SEEN_EXAM)]
    for name, rows in (("country_sentences", everything), ("country_train", train), ("country_exam", exam)):
        with open(os.path.join(DATA, name + ".jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("butun veri %d cumle | egitim %d | sinav %s" % (
        len(everything), len(train), dict(Counter(r["split"] for r in exam))))
    print("egitimde ulkenin gorevi:", dict(Counter(r["country_role"] for r in train)))
    print("cumle boyu (kelime): en kisa %d, en uzun %d, sozluk %d kelime" % (
        min(len(r["words"]) for r in everything), max(len(r["words"]) for r in everything),
        len({w for r in everything for w in r["words"]})))


if __name__ == "__main__":
    main()
