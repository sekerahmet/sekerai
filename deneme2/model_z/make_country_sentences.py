"""make_country_sentences -- Model Z ilk verisi: ulke olgularindan duz cumleler (kullanici, 3 Ekim: "Önce tüm veriyi
çıkartan sonra eğitim ve sınav").  Ayni olgu farkli kaliplarla: ulke bazen ozne, bazen nesne, tamlayan ya da yer.
Gorev etiketi YOK: gorevi makine kendisi bulur (kullanici, 3 Ekim: "Görev etiketi olmasın dedim ya artık tasarımdan kalktı").

1. Butun veri: her ulke x her dolu olgu x o olgunun her kalibi -> country_sentences.jsonl
2. Egitim / sinav (tohum sabit):
     her ulkenin dolu olgularindan 3'u ayrilir (sinav, gorulmemis olgu); kalanlarin her biri egitimde TEK kalipla
     (kalip ulkeden ulkeye doner)
     sinav bolmeleri: seen (egitim cumlelerinden ornek), unseen_template (egitimdeki olgu, baska kalip),
     unseen_fact (ayrilan olgu, butun kaliplari)
   -> country_train.jsonl, country_exam.jsonl
Hazir veri (data/countries/): bir kez uretilir, sonra hep okunur; yeniden calistirmak ayni dosyalari verir.

    python make_country_sentences.py
"""
import json
import os
import random
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "countries")
HELD_OUT = 3            # ulke basina sinava ayrilan olgu
SEEN_EXAM = 100         # sinavin 'seen' bolmesi: egitim cumlelerinden bu kadar
SEED = 0
THE = {"Netherlands", "United Kingdom", "United States", "Czech Republic", "Philippines", "United Arab Emirates"}

# {C} ulke, {F} olgu
TEMPLATES = {
    "capital": ["The capital of {C} is {F}", "{F} is the capital of {C}", "{C} has {F} as its capital"],
    "continent": ["{C} is in {F}", "{F} includes {C}"],
    "language": ["People in {C} speak {F}", "{F} is spoken in {C}"],
    "currency": ["{C} uses the {F}", "The {F} is the currency of {C}", "People in {C} pay with the {F}"],
    "city": ["{F} is a large city in {C}", "{C} has a large city called {F}"],
    "neighbor": ["{C} borders {F}", "{F} borders {C}", "{F} is a neighbor of {C}"],
    "river": ["The {F} flows through {C}", "The {F} is a river in {C}"],
    "dish": ["People in {C} eat {F}", "{C} is famous for {F}"],
    "landmark": ["{F} is in {C}", "Tourists in {C} visit {F}", "{C} is home to {F}"],
    "sea": ["{C} lies on the {F}", "The {F} touches {C}"],
}


def build(country, fact, value, k):
    """-> kayit: cumle ve kelimeleri (nokta ayri kelime)."""
    words = TEMPLATES[fact][k].replace("{C}", ("the " + country) if country in THE else country).replace(
        "{F}", value).split()
    words[0] = words[0][0].upper() + words[0][1:]      # cumle basi buyuk harf
    return dict(sentence=" ".join(words) + ".", words=words + ["."], country=country, fact=fact, template=k)


def main():
    table = json.load(open(os.path.join(DATA, "country_facts.json"), encoding="utf-8"))
    facts = list(table["fields"])
    everything = [build(row["country"], fact, row[fact], k) for row in table["countries"] for fact in facts
                  if row[fact] is not None for k in range(len(TEMPLATES[fact]))]
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
        with open(os.path.join(DATA, name + ".jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("butun veri %d cumle | egitim %d | sinav %s" % (
        len(everything), len(train), dict(Counter(r["split"] for r in exam))))
    print("cumle boyu (kelime): en kisa %d, en uzun %d, sozluk %d kelime" % (
        min(len(r["words"]) for r in everything), max(len(r["words"]) for r in everything),
        len({w for r in everything for w in r["words"]})))


if __name__ == "__main__":
    main()
