"""make_country_sentences -- Model Z ilk verisi: ulke olgularindan duz cumleler (kullanici, 3 Ekim: "Önce tüm veriyi
çıkartan sonra eğitim ve sınav").  Ayni olgu farkli kaliplarla: ulke bazen ozne, bazen nesne, tamlayan ya da yer.
Gorev kavrami yok (kullanici, 3 Ekim: "Görevi makine bulmayacak görev diye birşey yok"): veri yalniz kelimeler.

1. Butun veri: her ulke x her dolu olgu x o olgunun her kalibi -> country_sentences.jsonl
2. Egitim / sinav (tohum sabit):
     her ulkenin dolu olgularindan 3'u ayrilir; ayrilan olgunun BIR kalibi egitime, otekiler sinava (unseen): sinav
     cumlesi egitimde gecmemis ama butun kelimeleri egitimde gecmis (kullanici, 3 Ekim: "Unk içinse demekki soru ve
     eğitim setimiz uyumsuz"); kalan olgular butun kaliplariyla egitimde.  Egitimde birebir gecen cumle sinavdan cikar
     (komsuluk iki ulkeden de yazilabiliyor).  sinav bolmeleri: seen (egitim cumlelerinden ornek), unseen
   -> country_train.jsonl, country_exam.jsonl
3. Baglam ajani icin hikayeler (kullanici, 4 Ekim: "Önce bizim ülke verisinden denesek"): hikaye = bir ulkenin dolu
   olgulari rastgele sirayla, her olgu rastgele bir kalipla; egitime STORIES, sinava EXAM_STORIES hikaye / ulke (ayri
   tohum; 1-2'nin dosyalari degismez).  Sozluk: butun verinin kelimeleri, indeks 0 '<unk>'.
   -> country_stories.jsonl (split train / exam), country_vocab.json
4. Sabit hikayeler (kullanici, 4 Ekim: "sabit bir hikaye"; "her ülkenin hikayesi sabit ama sırası farklı"): her ulke tek
   hikaye, kalip 0; countries_fixed: butun ulkelerde ayni olgu sirasi, countries_orders: ulkeye ozel sabit sira (tohum 7);
   sinav = egitim.  -> data/countries_fixed/, data/countries_orders/
Her hikayede valid_next (kullanici, 4 Ekim: "modelin ürettiği çıktı olası bir çıktı olabilir yani bizim istediğimiz değil
ama doğru"): her cumleden sonra gecerli devamlar = o ulkenin henuz soylenmemis olgularinin butun kaliplari (kelime listesi).
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
STORIES = 30           # egitim hikayesi / ulke
EXAM_STORIES = 3       # sinav hikayesi / ulke
THE = {"Netherlands", "United Kingdom", "United States", "Czech Republic", "Philippines", "United Arab Emirates"}

# {C} ulke, {F} olgu
TEMPLATES = {
    "capital": ["The capital of {C} is {F}", "{F} is the capital of {C}", "{C} has {F} as its capital"],
    "continent": ["{C} is in {F}", "{F} includes {C}"],
    "language": ["People in {C} speak {F}", "{F} is spoken in {C}"],
    "currency": ["{C} uses the {F}", "The {F} is the currency of {C}", "People in {C} pay with the {F}"],
    "city": ["{F} is a large city in {C}", "{C} has a large city called {F}"],
    "neighbor": ["{C} borders {F}", "{F} borders {C}", "{F} is a neighbor of {C}", "{C} is a neighbor of {F}"],   # iki yon
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


def _valid_next(story, everything):
    """Her cumleden sonra gecerli devamlar: ulkenin henuz soylenmemis olgularinin butun kaliplari."""
    out = []
    for t in range(len(story["facts"]) - 1):
        said = set(story["facts"][:t + 1])
        out.append([r["words"] for r in everything if r["country"] == story["country"] and r["fact"] not in said])
    return out


def main():
    table = json.load(open(os.path.join(DATA, "country_facts.json"), encoding="utf-8"))
    facts = list(table["fields"])
    everything = [build(row["country"], fact, row[fact], k) for row in table["countries"] for fact in facts
                  if row[fact] is not None for k in range(len(TEMPLATES[fact]))]
    for i, r in enumerate(everything):
        r["id"] = i
    rng = random.Random(SEED)
    train, exam = [], []
    for row in table["countries"]:
        mine = [f for f in facts if row[f] is not None]
        held = set(rng.sample(mine, HELD_OUT))
        for fact in mine:
            rows = [r for r in everything if r["country"] == row["country"] and r["fact"] == fact]
            if fact in held:                           # bir kalip egitime: kelimeleri ogrenilsin
                k = rng.randrange(len(rows))
                train.append(dict(rows[k], split="train"))
                exam += [dict(r, split="unseen") for i, r in enumerate(rows) if i != k]
                continue
            train += [dict(r, split="train") for r in rows]
    seen_text = {r["sentence"] for r in train}
    known = {w for r in train for w in r["words"]}
    exam = [r for r in exam if r["sentence"] not in seen_text]
    assert all(w in known for r in exam for w in r["words"]), "sinavda egitimde gecmeyen kelime var"
    exam += [dict(r, split="seen") for r in rng.sample(train, SEEN_EXAM)]
    for name, rows in (("country_sentences", everything), ("country_train", train), ("country_exam", exam)):
        with open(os.path.join(DATA, name + ".jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    story_rng = random.Random(SEED + 1)
    stories = []
    for row in table["countries"]:
        mine = [f for f in facts if row[f] is not None]
        for i in range(STORIES + EXAM_STORIES):
            order = story_rng.sample(mine, len(mine))
            sents = [story_rng.choice([r for r in everything if r["country"] == row["country"] and r["fact"] == f])
                     for f in order]
            stories.append(dict(country=row["country"], split="train" if i < STORIES else "exam",
                                facts=order, sentences=[r["words"] for r in sents]))
    vocab = ["<unk>"] + sorted({w for r in everything for w in r["words"]})
    fixed_rng = random.Random(7)
    fixed, orders = [], []
    for row in table["countries"]:
        mine = [f for f in facts if row[f] is not None]
        own = list(mine)
        fixed_rng.shuffle(own)
        for out, order in ((fixed, mine), (orders, own)):
            out.append(dict(country=row["country"], facts=order,
                            sentences=[build(row["country"], f, row[f], 0)["words"] for f in order]))
    sets = ((DATA, stories), (DATA + "_fixed", [dict(s, split=sp) for sp in ("train", "exam") for s in fixed]),
            (DATA + "_orders", [dict(s, split=sp) for sp in ("train", "exam") for s in orders]))
    for folder, rows in sets:
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "country_stories.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(dict(r, valid_next=_valid_next(r, everything)), ensure_ascii=False) + "\n")
        json.dump(vocab, open(os.path.join(folder, "country_vocab.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("hikaye %s, sozluk %d" % (dict(Counter(r["split"] for r in stories)), len(vocab)))
    print("butun veri %d cumle | egitim %d | sinav %s" % (
        len(everything), len(train), dict(Counter(r["split"] for r in exam))))
    print("cumle boyu (kelime): en kisa %d, en uzun %d, sozluk %d kelime" % (
        min(len(r["words"]) for r in everything), max(len(r["words"]) for r in everything),
        len({w for r in everything for w in r["words"]})))


if __name__ == "__main__":
    main()
