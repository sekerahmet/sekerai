"""Olgu testi: modelin dünya bilgisini kayıptan ayrı ölçer.  Her soru 4 seçenekli (1 doğru, aynı kategoriden 3 yanlış);
model her seçeneğin istemden sonra gelme log-olasılığını hesaplar, en yükseği seçilir.  Şans %25.

    python scripts/fact_probe.py --runs fw_s000_d512_tf_yg fw_s000_d512_tf_yg_ozel_rms fw_s000_d512_ozel_3epok

Yanlış seçenekler mümkünse doğru cevapla aynı token uzunluğunda seçilir (toplam log-olasılık kısa seçeneği kayırır).
Puan: toplam log-olasılıkla doğruluk; kategori başına, toplam ve öznenin (ülke, element, kitap) tek / çok token olmasına göre
(kelimeye özel nöron girdi token'ına bağlı: tek token'lı öznenin olgusu kendi satırında durabilir).
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from rgn.train.checkpoint import load_run  # noqa: E402

CAPITALS = [("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Portugal", "Lisbon"),
            ("the United Kingdom", "London"), ("Ireland", "Dublin"), ("the Netherlands", "Amsterdam"),
            ("Belgium", "Brussels"), ("Austria", "Vienna"), ("Switzerland", "Bern"), ("Sweden", "Stockholm"),
            ("Norway", "Oslo"), ("Denmark", "Copenhagen"), ("Finland", "Helsinki"), ("Poland", "Warsaw"),
            ("Greece", "Athens"), ("Turkey", "Ankara"), ("Russia", "Moscow"), ("Hungary", "Budapest"),
            ("the Czech Republic", "Prague"), ("Romania", "Bucharest"), ("Iceland", "Reykjavik"), ("Egypt", "Cairo"),
            ("Kenya", "Nairobi"), ("Nigeria", "Abuja"), ("Morocco", "Rabat"), ("Ethiopia", "Addis Ababa"),
            ("Ghana", "Accra"), ("China", "Beijing"), ("Japan", "Tokyo"), ("South Korea", "Seoul"),
            ("India", "New Delhi"), ("Pakistan", "Islamabad"), ("Bangladesh", "Dhaka"), ("Thailand", "Bangkok"),
            ("Vietnam", "Hanoi"), ("Indonesia", "Jakarta"), ("the Philippines", "Manila"),
            ("Malaysia", "Kuala Lumpur"), ("Iran", "Tehran"), ("Iraq", "Baghdad"), ("Saudi Arabia", "Riyadh"),
            ("Afghanistan", "Kabul"), ("Australia", "Canberra"), ("New Zealand", "Wellington"), ("Canada", "Ottawa"),
            ("Mexico", "Mexico City"), ("Cuba", "Havana"), ("Argentina", "Buenos Aires"), ("Chile", "Santiago"),
            ("Peru", "Lima"), ("Venezuela", "Caracas"), ("Ecuador", "Quito"), ("Jamaica", "Kingston")]
SYMBOLS = [("gold", "Au"), ("silver", "Ag"), ("iron", "Fe"), ("copper", "Cu"), ("sodium", "Na"), ("potassium", "K"),
           ("lead", "Pb"), ("mercury", "Hg"), ("tin", "Sn"), ("hydrogen", "H"), ("oxygen", "O"), ("carbon", "C"),
           ("nitrogen", "N"), ("helium", "He"), ("calcium", "Ca"), ("chlorine", "Cl"), ("sulfur", "S"), ("zinc", "Zn"),
           ("magnesium", "Mg"), ("aluminum", "Al"), ("uranium", "U"), ("phosphorus", "P"), ("neon", "Ne"),
           ("silicon", "Si"), ("nickel", "Ni"), ("platinum", "Pt"), ("iodine", "I"), ("fluorine", "F"),
           ("lithium", "Li"), ("tungsten", "W")]
CONTINENTS = [(c, "Europe") for c in ("France", "Germany", "Italy", "Spain", "Poland", "Sweden", "Greece", "Norway")] + \
             [(c, "Asia") for c in ("China", "Japan", "India", "Thailand", "Vietnam", "Pakistan", "Mongolia", "Nepal")] + \
             [(c, "Africa") for c in ("Egypt", "Kenya", "Nigeria", "Ghana", "Ethiopia", "Morocco", "Senegal",
                                      "Tanzania")] + \
             [(c, "South America") for c in ("Brazil", "Argentina", "Chile", "Peru", "Colombia", "Bolivia")] + \
             [(c, "North America") for c in ("Canada", "Mexico", "Guatemala", "Honduras")]
LANGUAGES = [("Brazil", "Portuguese"), ("Mexico", "Spanish"), ("Argentina", "Spanish"), ("France", "French"),
             ("Germany", "German"), ("Italy", "Italian"), ("Japan", "Japanese"), ("Russia", "Russian"),
             ("Portugal", "Portuguese"), ("Greece", "Greek"), ("Turkey", "Turkish"), ("Poland", "Polish"),
             ("the Netherlands", "Dutch"), ("Sweden", "Swedish"), ("Iran", "Persian"), ("Egypt", "Arabic"),
             ("Saudi Arabia", "Arabic"), ("Thailand", "Thai"), ("Vietnam", "Vietnamese"), ("South Korea", "Korean"),
             ("Austria", "German"), ("Chile", "Spanish"), ("Hungary", "Hungarian"), ("Norway", "Norwegian"),
             ("Denmark", "Danish")]
SCIENCE = [("The largest planet in our solar system is", "Jupiter", ["Saturn", "Mars", "Venus"]),
           ("The planet closest to the Sun is", "Mercury", ["Venus", "Mars", "Earth"]),
           ("The planet known as the Red Planet is", "Mars", ["Venus", "Jupiter", "Saturn"]),
           ("Water is made of hydrogen and", "oxygen", ["nitrogen", "carbon", "helium"]),
           ("The powerhouse of the cell is the", "mitochondria", ["nucleus", "ribosome", "membrane"]),
           ("Plants absorb carbon dioxide and release", "oxygen", ["nitrogen", "hydrogen", "helium"]),
           ("The hardest natural substance on Earth is", "diamond", ["gold", "iron", "quartz"]),
           ("At sea level, water boils at 100 degrees", "Celsius", ["Fahrenheit", "Kelvin", "Rankine"]),
           ("Water freezes at 0 degrees", "Celsius", ["Fahrenheit", "Kelvin", "Rankine"]),
           ("The force that keeps us on the ground is called", "gravity", ["friction", "magnetism", "inertia"]),
           ("The largest ocean on Earth is the", "Pacific", ["Atlantic", "Indian", "Arctic"]),
           ("The longest river in Africa is the", "Nile", ["Congo", "Niger", "Zambezi"]),
           ("The tallest mountain in the world is Mount", "Everest", ["Kilimanjaro", "Fuji", "Blanc"]),
           ("The largest desert in Africa is the", "Sahara", ["Kalahari", "Gobi", "Namib"]),
           ("The Earth revolves around the", "Sun", ["Moon", "Mars", "Jupiter"]),
           ("Insulin is produced by the", "pancreas", ["liver", "kidney", "heart"]),
           ("The study of living things is called", "biology", ["geology", "astronomy", "chemistry"]),
           ("Animals that eat only plants are called", "herbivores", ["carnivores", "omnivores", "predators"]),
           ("Sound cannot travel through a", "vacuum", ["liquid", "solid", "gas"]),
           ("The gas that plants need for photosynthesis is carbon", "dioxide", ["monoxide", "oxide", "sulfide"])]
AUTHORS = ["William Shakespeare", "Jane Austen", "Charles Dickens", "Leo Tolstoy", "George Orwell", "Mark Twain",
           "Homer", "Miguel de Cervantes", "Victor Hugo", "Dante Alighieri"]
BOOKS = [("Romeo and Juliet", "William Shakespeare"), ("Hamlet", "William Shakespeare"),
         ("Pride and Prejudice", "Jane Austen"), ("Oliver Twist", "Charles Dickens"), ("War and Peace", "Leo Tolstoy"),
         ("Nineteen Eighty-Four", "George Orwell"), ("The Adventures of Tom Sawyer", "Mark Twain"),
         ("The Odyssey", "Homer"), ("Don Quixote", "Miguel de Cervantes"), ("Les Miserables", "Victor Hugo"),
         ("The Divine Comedy", "Dante Alighieri"), ("A Tale of Two Cities", "Charles Dickens")]
SCIENTISTS = ["Alexander Graham Bell", "Thomas Edison", "Albert Einstein", "Isaac Newton", "Alexander Fleming",
              "Johannes Gutenberg", "Tim Berners-Lee", "Charles Darwin", "Nicolaus Copernicus", "Galileo Galilei"]
DISCOVERY = [("The telephone was invented by", "Alexander Graham Bell"),
             ("The theory of relativity was developed by", "Albert Einstein"),
             ("The laws of motion and universal gravitation were formulated by", "Isaac Newton"),
             ("Penicillin was discovered by", "Alexander Fleming"),
             ("The printing press with movable type was invented in Europe by", "Johannes Gutenberg"),
             ("The World Wide Web was invented by", "Tim Berners-Lee"),
             ("The theory of evolution by natural selection was proposed by", "Charles Darwin"),
             ("The heliocentric model of the solar system was proposed by", "Nicolaus Copernicus")]
YEARS = [("World War II ended in", "1945", ["1939", "1918", "1950"]),
         ("World War I began in", "1914", ["1918", "1939", "1905"]),
         ("World War I ended in", "1918", ["1914", "1945", "1925"]),
         ("The United States declared independence in", "1776", ["1789", "1812", "1865"]),
         ("The Berlin Wall fell in", "1989", ["1991", "1961", "1975"]),
         ("Neil Armstrong first walked on the Moon in", "1969", ["1959", "1975", "1981"]),
         ("The French Revolution began in", "1789", ["1776", "1815", "1848"]),
         ("The Titanic sank in", "1912", ["1905", "1920", "1898"]),
         ("Christopher Columbus reached the Americas in", "1492", ["1519", "1607", "1453"]),
         ("The American Civil War ended in", "1865", ["1861", "1776", "1898"]),
         ("The Soviet Union collapsed in", "1991", ["1989", "1985", "1979"]),
         ("The September 11 attacks took place in", "2001", ["1999", "2003", "1993"]),
         ("The Great Depression began in", "1929", ["1919", "1939", "1907"]),
         ("India gained independence from Britain in", "1947", ["1950", "1937", "1962"]),
         ("Pearl Harbor was attacked in", "1941", ["1939", "1945", "1937"]),
         ("The Magna Carta was signed in", "1215", ["1066", "1492", "1315"]),
         ("The Battle of Hastings took place in", "1066", ["1215", "1166", "966"]),
         ("The first iPhone was released in", "2007", ["2003", "2010", "2001"])]


def _ntok(enc, w):
    return len(enc.encode(" " + w))


def questions(seed=0, enc=None):
    """-> [(kategori, istem, doğru, [seçenekler], özne ya da None)]; yanlışlar aynı kategoriden, mümkünse doğru cevapla
    aynı token uzunluğunda (enc verilirse), tohumla sabit."""
    rnd = random.Random(seed)
    qs = []

    def pick(pool, ans, k=3):
        cand = sorted({p for p in pool if p != ans})
        if enc is not None:
            same = [p for p in cand if _ntok(enc, p) == _ntok(enc, ans)]
            if len(same) >= k:
                cand = same
        return rnd.sample(cand, k)

    caps = [c for _, c in CAPITALS]
    qs += [("başkent", "The capital of %s is" % c, a, pick(caps, a), c) for c, a in CAPITALS]
    syms = [s for _, s in SYMBOLS]
    qs += [("kimyasal simge", "The chemical symbol for %s is" % e, s, pick(syms, s), e) for e, s in SYMBOLS]
    conts = ["Europe", "Asia", "Africa", "Australia", "Antarctica", "South America", "North America", "Central America"]
    qs += [("kıta", "%s is a country located in" % c, a, pick(conts, a), c) for c, a in CONTINENTS]
    langs = sorted({l for _, l in LANGUAGES})
    qs += [("dil", "Most people in %s speak" % c, a, pick(langs, a), c) for c, a in LANGUAGES]
    qs += [("bilim", p, a, d, None) for p, a, d in SCIENCE]
    qs += [("yazar", "%s was written by" % b, a, pick(AUTHORS, a), b) for b, a in BOOKS]
    qs += [("buluş", p, a, pick(SCIENTISTS, a), None) for p, a in DISCOVERY]
    qs += [("yıl", p, a, d, None) for p, a, d in YEARS]
    return qs


@torch.no_grad()
def score(m, enc, prompt, options, dev):
    """Her seçenek için (toplam log-olasılık, karakter başına log-olasılık)."""
    p = enc.encode(prompt)
    seqs = [p + enc.encode(" " + o) for o in options]
    n = max(len(s) for s in seqs)
    x = torch.zeros(len(seqs), n, dtype=torch.long, device=dev)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    lp = F.log_softmax(m(x).float(), -1)
    out = []
    for i, (s, o) in enumerate(zip(seqs, options)):
        tot = sum(lp[i, j - 1, s[j]].item() for j in range(len(p), len(s)))
        out.append((tot, tot / len(o)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    a = ap.parse_args()
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qs = questions(enc=enc)
    cats = sorted({q[0] for q in qs}, key=[q[0] for q in qs].index)
    single = [None if s is None else _ntok(enc, s) == 1 for *_, s in qs]
    ok = {}
    for run in a.runs:
        m, step = load_run(run, dev)
        m.float()
        ok[run] = [int(max(range(4), key=lambda i: sc[i][0]) == 0)
                   for sc in (score(m, enc, p, [ans] + list(w), dev) for _, p, ans, w, _ in qs)]
        del m
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    print("%d soru, şans %%25; yanlış seçenekler mümkünse aynı token uzunluğunda.  Hücre: doğruluk %%" % len(qs))
    print("%-22s %5s" % ("grup", "n") + "".join("%14s" % r[-14:] for r in a.runs))
    groups = [(c, [i for i, q in enumerate(qs) if q[0] == c]) for c in cats] + [("TOPLAM", list(range(len(qs))))]
    groups += [("özne TEK token", [i for i, s in enumerate(single) if s is True]),
               ("özne ÇOK token", [i for i, s in enumerate(single) if s is False])]
    for name, ix in groups:
        print("%-22s %5d" % (name, len(ix)) + "".join("%14.0f" % (100 * sum(ok[r][i] for i in ix) / len(ix))
                                                     for r in a.runs))


if __name__ == "__main__":
    main()
