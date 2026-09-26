# -*- coding: utf-8 -*-
"""data_20 -- model_20 adim 0: kucuk Ingilizce akrabalik dunyasi.

8 aile, 3 nesil, 48 kisi.  Her ciftin bir oglu bir kizi var; aile i'nin oglu aile i+1'in kiziyla evlenir (halka), evlenen
kadin kocasinin soyadini alir, her ad benzersiz.  Boylece her torunun aunt'u babasinin kiz kardesi, uncle'i annesinin erkek
kardesi.  Ogretme hanelerinin torunlarinda turemis iliskiler zincir ve adiyla yazilir; tutulan hanelerde HIC yazilmaz,
sorulur.

    python data_20.py          ozet
    python data_20.py --all    butun cumleler ve sorular
"""
import hashlib
import sys

FAMILIES = ("Smith", "Brown", "Clark", "Davis", "Evans", "Fisher", "Green", "Hill")
# aile basina: dede, nine, ogul, kiz, torun oglan, torun kiz (torunlar ailenin OGLUNUN hanesinde)
NAMES = {
    "Smith": ("George", "Mary", "John", "Emma", "Tom", "Alice"),
    "Brown": ("Henry", "Rose", "Peter", "Linda", "Jack", "Lily"),
    "Clark": ("Walter", "Ruth", "Paul", "Kate", "Sam", "Grace"),
    "Davis": ("Frank", "Helen", "Mark", "Sarah", "Luke", "Chloe"),
    "Evans": ("Arthur", "Edith", "David", "Laura", "Owen", "Ella"),
    "Fisher": ("Harold", "Doris", "Simon", "Julia", "Adam", "Ruby"),
    "Green": ("Albert", "Irene", "James", "Claire", "Leo", "Zoe"),
    "Hill": ("Ernest", "Agnes", "Robert", "Diana", "Ben", "Mia"),
}
# Bitisik: kuzenleri iki yandaki haneler oldugu icin kenardaki iki hanenin (Evans, Hill) kuzen sorulari co_written olur.
HELD_HOUSEHOLDS = ("Evans", "Fisher", "Green", "Hill")
BASE = ("father", "mother", "brother", "sister", "son", "daughter")
# turemis iliski = bu zincirlerin birlesimi; dunyada var olan zincirler yazilir/sorulur
DERIVED = {
    "grandfather": (("father", "father"), ("mother", "father")),
    "grandmother": (("father", "mother"), ("mother", "mother")),
    "aunt": (("father", "sister"), ("mother", "sister")),
    "uncle": (("father", "brother"), ("mother", "brother")),
    "cousin": tuple((p, s, c) for p in ("father", "mother") for s in ("sister", "brother") for c in ("son", "daughter")),
}
PAD, EOS = "<pad>", "<eos>"
PUNCT = ("'s", "?", ".")


def build_world():
    """people: tam ad -> ozellikler; rel: tam ad -> iliski -> tam adlar (sirali)."""
    people, rel = {}, {}
    n = len(FAMILIES)

    def person(first, surname, gender, generation, birth_family, household=None):
        full = "%s %s" % (first, surname)
        assert full not in people
        people[full] = dict(first=first, surname=surname, gender=gender, generation=generation, birth_family=birth_family,
                            household=household)
        rel[full] = {}
        return full

    def link(a, r, b):
        rel[a].setdefault(r, []).append(b)

    def family(father, mother, son, daughter):
        for child in (son, daughter):
            link(child, "father", father)
            link(child, "mother", mother)
        for parent in (father, mother):
            link(parent, "son", son)
            link(parent, "daughter", daughter)
        link(son, "sister", daughter)
        link(daughter, "brother", son)

    gen2 = {}
    for i, f in enumerate(FAMILIES):
        gf, gm, son, dau = NAMES[f][:4]
        husband, wife = person(gf, f, "m", 1, f), person(gm, f, "f", 1, f)
        s = person(son, f, "m", 2, f)
        d = person(dau, FAMILIES[(i - 1) % n], "f", 2, f)          # aile i-1'in ogluyla evli
        family(husband, wife, s, d)
        gen2[f] = (s, d)
    for i, f in enumerate(FAMILIES):
        father, mother = gen2[f][0], gen2[FAMILIES[(i + 1) % n]][1]
        boy, girl = NAMES[f][4:]
        family(father, mother, person(boy, f, "m", 3, f, household=f), person(girl, f, "f", 3, f, household=f))
    for x in rel:
        for r in rel[x]:
            rel[x][r] = sorted(rel[x][r])
    return people, rel


def follow(rel, x, path):
    """x'ten iliski zinciriyle varilan kisiler (sirali)."""
    now = {x}
    for r in path:
        now = {y for z in now for y in rel[z].get(r, ())}
    return sorted(now)


def facts(people, rel):
    """(ozne, zincir, cevap, tur): tur base / chain / named.  Turemisler yalniz zinciri dunyada olan kisilerde."""
    out = []
    for x in people:
        for r in BASE:
            out += [(x, (r,), y, "base") for y in rel[x].get(r, ())]
    for x in people:
        for name, paths in DERIVED.items():
            union = set()
            for p in paths:
                ys = follow(rel, x, p)
                out += [(x, p, y, "chain") for y in ys]
                union.update(ys)
            out += [(x, (name,), y, "named") for y in sorted(union)]
    return out


def phrase(x, path):
    """'Alice Smith 's father 's sister' token'lari."""
    toks = x.split()
    for r in path:
        toks += ["'s", r]
    return toks


def statement(x, path, y):
    return phrase(x, path) + ["is"] + y.split() + ["."]


def question(x, path):
    return ["Who", "is"] + phrase(x, path) + ["?"]


def qa(x, path, y):
    return question(x, path) + y.split() + ["."]


def detokenize(tokens):
    s = " ".join(tokens)
    for p in PUNCT:
        s = s.replace(" " + p, p)
    return s


def tokenize(text):
    for p in PUNCT:
        text = text.replace(p, " " + p)
    return text.split()


def hops(path):
    return 3 if path == ("cousin",) else 2 if path[0] in DERIVED else len(path)


def build():
    """Butun veri: dunya, egitim cumleleri (token listeleri), sinav, sozluk, iz."""
    people, rel = build_world()
    all_facts = facts(people, rel)
    held = {x for x, p in people.items() if p["household"] in HELD_HOUSEHOLDS}
    written = [(x, path, y, kind) for x, path, y, kind in all_facts if kind == "base" or x not in held]
    train = []
    for x, path, y, kind in written:
        train += [statement(x, path, y), qa(x, path, y)]
    pairs = {(a, b) for a, _, b, _ in written} | {(b, a) for a, _, b, _ in written}

    exam = []
    grouped = {}
    for x, path, y, kind in all_facts:
        grouped.setdefault((x, path, kind), []).append(y)
    for (x, path, kind), ys in grouped.items():
        if kind == "base":
            cls = "memory_base"
        elif x not in held:
            cls = "memory_derived"
        else:
            cls = "%s%d" % (kind, hops(path))
        exam.append(dict(cls=cls, subject=x, path=path, prompt=question(x, path), answers=sorted(ys),
                         co_written=any((x, y) in pairs for y in ys)))

    vocab = [PAD, EOS] + sorted({t for s in train for t in s} | {t for e in exam for t in e["prompt"]})
    h = hashlib.sha256()
    for s in train:
        h.update((" ".join(s) + "\n").encode("utf-8"))
    for e in exam:
        h.update(("%s|%s|%s\n" % (e["cls"], " ".join(e["prompt"]), ",".join(e["answers"]))).encode("utf-8"))
    return dict(people=people, rel=rel, facts=all_facts, held=sorted(held), train=train, exam=exam, vocab=vocab,
                fingerprint=h.hexdigest()[:12])


def encode(tokens, vocab):
    """<eos> cumle <eos> -> id listesi."""
    ix = {w: i for i, w in enumerate(vocab)}
    return [ix[EOS]] + [ix[t] for t in tokens] + [ix[EOS]]


def audit(data):
    """Tutulan sorunun zinciri ya da adi egitim metninde HIC gecmez; zincir cevabi ozneye ya da son iliskinin ozneye
    uygulanmasina donmez; her ad benzersiz.  Bozuksa DURUR."""
    text = "\n".join(detokenize(s) for s in data["train"])
    held_q = [e for e in data["exam"] if not e["cls"].startswith("memory")]
    for e in held_q:
        assert detokenize(phrase(e["subject"], e["path"])) not in text, e["prompt"]
    for x, path, y, kind in data["facts"]:
        if kind == "chain":
            assert y != x and y not in follow(data["rel"], x, path[-1:]), (x, path, y)
    firsts = [p["first"] for p in data["people"].values()]
    assert len(set(firsts)) == len(firsts)
    assert not set(firsts) & set(FAMILIES)
    return len(held_q)


def report(data, full=False):
    people, rel = data["people"], data["rel"]
    out = ["data_20  iz %s" % data["fingerprint"], ""]
    out.append("HANELER (torunlar; * = tutulan)")
    for f in FAMILIES:
        kids = [x for x, p in people.items() if p["household"] == f]
        father, mother = rel[kids[0]]["father"][0], rel[kids[0]]["mother"][0]
        out.append("  %s %-8s %-14s + %-14s -> %s" % ("*" if f in HELD_HOUSEHOLDS else " ", f, father, mother,
                                                        ", ".join(kids)))
    out.append("")
    kinds = {}
    for x, path, y, kind in data["facts"]:
        if kind == "base" or x not in data["held"]:
            kinds[kind] = kinds.get(kind, 0) + 1
    out.append("kisi %d  yazili olgu %d (base %d, chain %d, named %d)  egitim cumlesi %d  sozluk %d" % (
        len(people), sum(kinds.values()), kinds.get("base", 0), kinds.get("chain", 0), kinds.get("named", 0),
        len(data["train"]), len(data["vocab"])))
    out.append("")
    out.append("SINAV            soru  co_written  ornek")
    classes = {}
    for e in data["exam"]:
        classes.setdefault(e["cls"], []).append(e)
    for cls in ("memory_base", "memory_derived", "chain2", "chain3", "named2", "named3"):
        es = classes.get(cls, [])
        ex = es[0] if es else None
        out.append("  %-15s %4d  %10d  %s" % (cls, len(es), sum(e["co_written"] for e in es),
                                              "%s -> %s" % (detokenize(ex["prompt"]), " | ".join(ex["answers"])) if ex else ""))
    out.append("")
    sample = "Alice Smith"
    out.append("ORNEK EGITIM CUMLELERI (%s)" % sample)
    out += ["  " + detokenize(s) for s in data["train"] if detokenize(s[:2]) == sample or s[2:4] == sample.split()]
    if full:
        out += ["", "BUTUN EGITIM CUMLELERI"] + ["  " + detokenize(s) for s in data["train"]]
        out += ["", "BUTUN SORULAR"]
        for e in data["exam"]:
            out.append("  %-15s %s  %s -> %s" % (e["cls"], "co" if e["co_written"] else "  ", detokenize(e["prompt"]),
                                                 " | ".join(e["answers"])))
    return "\n".join(out)


if __name__ == "__main__":
    d = build()
    audit(d)
    print(report(d, full="--all" in sys.argv))
