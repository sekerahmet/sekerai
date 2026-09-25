"""Modelle KONUS -- bir hikaye baslangici yaz, model_19 devamini yazsin.

Agirlik Colab'da uretilip Drive'a yaziliyor; burasi yalniz OKUYOR.  Model CPU'da
calisir.  Kosu surerken de calisir: en yeni w/t paketi alinir.  Model <eos>'tan
baslar ve kendi <eos>'unu yazinca durur (sonda "---").

  Once upon a time           yaz, devamini gorursun
  <bos satir>                makalenin degerlendirme istemlerinden RASTGELE biri
  serbest                    istemsiz, bos sayfadan bir hikaye
  n=120                      en fazla kac kelime (varsayilan 120, istem+n <= T)
  s=0.8                      sicaklik.  0 = hep en yuksek puan (belirlenimci, decompose'un gordugu)
  r=1.3                      tekrar cezasi: son 20 kelimede gecenler 1,3'e bolunur (1 = kapali)
  p=0.9                      top-p: en olasi kelimelerden toplami 0,9 olan kumeden sec (1 = kapali; s > 0 ister)
  yasak                      <bilinmeyen>/<dolgu> uretimi kapat (varsayilan) / ac
  yedek                      hangi paketler var, hangisi yuklu
  yedek t20000               baska bir ani yukle (ayni kosuda); PR_TS_ARCH_BASE_S0/t20000 da olur
  ?                          yardim
  q                          cik
"""
import math
import os
import random
import re
import sys
import textwrap
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Agir moduller ilk kullanimda (_heavy): pencere hemen acilir.  Olculdu, 25 Eylul: import torch ~7 sn; yeni paketin
# Drive'dan ilk okunmasi 10-19 sn (ikinci kez 0,4 sn).
torch = PointRelation = DS = None

MODEL_DIRS = [r"G:\Drive'ım\model_19", r"G:\Drivem\model_19"]
TS_DIRS = [r"G:\Drive'ım\tinystories", r"G:\Drivem\tinystories"]
NEW_PARTS = ("embed", "readout", "chain_sim", "distance")


def _heavy():
    """torch, model_19 ve data_stories_19 -- ilk cagrida yuklenir."""
    global torch, PointRelation, DS
    if DS is None:
        import torch as _torch
        from model_19 import PointRelation as _point_relation
        import data_stories_19 as _data
        torch, PointRelation, DS = _torch, _point_relation, _data


def _first_existing(candidates):
    return next((d for d in candidates if os.path.isdir(d)), None)


_CACHE = {}       # path -> paket: find_package() bir kez yukler, load_package() yeniden okumaz


def _run_dirs():
    """Kosu klasorleri, en son degiseni once (Drive'da yuzlerce dosyanin tarihine bakmaktan hizli)."""
    k = _first_existing(MODEL_DIRS)
    if not k:
        return []
    d = [os.path.join(k, x) for x in os.listdir(k) if "_eski_" not in x]
    return sorted((x for x in d if os.path.isdir(x)), key=os.path.getmtime, reverse=True)


def packages(run_dirs=None):
    """(kosu/dosya, yol) -- yeniden eskiye.  Yalniz kosu klasorlerindeki t/w paketleri."""
    found = []
    for d in (_run_dirs() if run_dirs is None else run_dirs):
        run_name = os.path.basename(d)
        for f in os.listdir(d):
            if re.fullmatch(r"[tw]\d+\.pt", f):
                found.append(("%s/%s" % (run_name, f[:-3]), os.path.join(d, f)))
    return sorted(found, key=lambda x: os.path.getmtime(x[1]), reverse=True)


def find_package(pattern=None, run_name=None):
    """Desen yoksa EN YENI hikaye paketi.  'w500' yuklu kosuda, 'KOSU/w500' her yerde;
    ad BIREBIR (t500, t5000 degil)."""
    _heavy()
    if not pattern:
        for run_name in _run_dirs():                         # en son degisen kosu klasoru once
            for name, path in packages([run_name]):
                k = torch.load(path, weights_only=False, map_location="cpu", mmap=True)
                if len(k.get("vocab") or []) > 13:
                    _CACHE[path] = k
                    return path
                break                                   # matematik kosusu: sonraki klasor
        print("HIKAYE PAKETI YOK.  Arananlar: %s\\<kosu>\\t*.pt, w*.pt" % " ya da ".join(MODEL_DIRS))
        sys.exit(1)
    all_packages = packages()
    d = pattern.replace("\\", "/")
    d = d[:-3] if d.endswith(".pt") else d
    e = [y for a, y in all_packages if a == d or (run_name and a == "%s/%s" % (run_name, d))]
    if not e:
        print("'%s' ile eslesen paket yok.  'yedek' yazip listeye bak." % pattern)
        return None
    return e[0]


def load_package(path):
    _heavy()
    k = _CACHE.pop(path, None) or torch.load(path, weights_only=False, map_location="cpu", mmap=True)
    return PointRelation.from_package(k).eval(), k


def prompts():
    _heavy()
    d = _first_existing(TS_DIRS)
    return DS.prompts(d) if d else []


def print_wrapped(title, text, g=70):
    print(title)
    for s in textwrap.wrap(text, g) or [""]:
        print("   " + s)


def summary(path, k):
    g = k.get("heldout_diag") or {}
    ce = k.get("heldout_ce")
    c = k["config"]
    parts = [p for p in NEW_PARTS if c.get(p)] + (["ikinci attention"] if len(c.get("attn_after", ())) > 1 else [])
    print("=" * 74)
    print("paket   %s/%s%s" % (os.path.basename(os.path.dirname(path)), os.path.basename(path),
                               "   (BITTI)" if k.get("done") else ""))
    print("adim    %s   parametre %s   sozluk %d   D %d (%d + %d)   T %d"
          % ("{:,}".format(k["step"]), "{:,}".format(k.get("n_params", 0)), c["n"], c["d_order"] + c["d_content"],
             c["d_order"], c["d_content"], c["t_max"]))
    print("model   yeni parcalar: %s   deneme turu: %s" % (", ".join(parts) or "YOK (zemin)", k.get("trial", "-")))
    if "heldout_acc" in k:
        print("olcum   accuracy train %.4f   heldout %.4f   ppl %s"
              % (k["train_acc"], k["heldout_acc"], "-" if ce is None else "%.1f" % math.exp(ce)))
    if g:
        print("        " + "   ".join("%s %.4f" % (a, v) for a, v in g.items()))
    print("=" * 74)


def main():
    print(__doc__.split("\n\n")[-1].rstrip())
    print("\nmodel arka planda yukleniyor -- istemini simdiden yazabilirsin\n")
    loaded = {}

    def _load():
        try:
            loaded["path"] = find_package(sys.argv[1] if len(sys.argv) > 1 else None)
            if loaded["path"]:
                loaded["m"], loaded["k"] = load_package(loaded["path"])
                loaded["prompts"] = prompts()
        except BaseException as h:                   # find_package bulamazsa sys.exit: ipligi degil programi durdur
            loaded["error"] = h

    loader = threading.Thread(target=_load, daemon=True)
    loader.start()
    path = m = k = VOCAB = TOKEN_ID = PROMPTS = None

    n, temp, banned, penalty, top_p = 120, 0.0, True, 1.0, 1.0      # DS.REPEAT_PENALTY, DS.TOP_P: ikisi de kapali
    while True:
        try:
            g = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if g in ("q", "quit", "cik"):
            break
        if m is None and not re.match(r"(\?|n=|s=|r=|p=|yasak$)", g):
            if loader.is_alive():
                print("   model yukleniyor, bekle...")
                loader.join()
            if "error" in loaded or not loaded.get("path"):
                raise SystemExit(loaded.get("error") or 1)
            path, m, k, PROMPTS = loaded["path"], loaded["m"], loaded["k"], loaded["prompts"]
            VOCAB = DS.to_general_eos(list(k["vocab"]))
            TOKEN_ID = {a: i for i, a in enumerate(VOCAB)}
            summary(path, k)
            print("%d degerlendirme istemi" % len(PROMPTS))
        if g == "?":
            print(__doc__.split("\n\n")[-1].rstrip())
            continue
        if g.startswith("n="):
            n = max(1, int(g[2:]))
            print("   en fazla %d kelime" % n)
            continue
        if g.startswith("s="):
            temp = max(0.0, float(g[2:]))
            print("   sicaklik %.2f%s" % (temp, "  (belirlenimci)" if not temp else ""))
            continue
        if g.startswith("r="):
            penalty = max(1.0, float(g[2:]))
            print("   tekrar cezasi %.2f%s" % (penalty, "  (kapali)" if penalty == 1 else ""))
            continue
        if g.startswith("p="):
            top_p = min(1.0, max(0.01, float(g[2:])))
            print("   top-p %.2f%s" % (top_p, "  (kapali)" if top_p == 1 else ""))
            continue
        if g == "yasak":
            banned = not banned
            print("   <bilinmeyen>/<dolgu> %s" % ("URETILMEZ" if banned else "uretilebilir"))
            continue
        if g.startswith("yedek"):
            p = g[5:].strip()
            if not p:
                for name_, path_ in packages()[:15]:
                    print("   %s%s" % (name_, "   <- yuklu" if os.path.normcase(path_)
                                       == os.path.normcase(path) else ""))
                continue
            path2 = find_package(p, os.path.basename(os.path.dirname(path)))
            if path2:
                path = path2
                m, k = load_package(path)
                VOCAB = DS.to_general_eos(list(k["vocab"]))
                TOKEN_ID = {a: i for i, a in enumerate(VOCAB)}
                summary(path, k)
            continue
        if g == "serbest":
            g = ""
        elif not g:
            if not PROMPTS:
                print("   istem dosyasi yok; kendin bir seyler yaz")
                continue
            g = random.choice(PROMPTS)
        n_unknown = sum(1 for t in DS.tokenize(DS.normalize(g)) if t not in TOKEN_ID)
        banned_tokens = (DS.PAD_TOKEN, DS.UNK_TOKEN) if banned else ()
        prompt_text, generated = DS.generate(m, g, VOCAB, TOKEN_ID, steps=n, device="cpu", temperature=temp,
                            seed=random.randrange(10 ** 6) if temp else None, banned=banned_tokens,
                            penalty=penalty, top_p=top_p)
        print()
        print_wrapped("ISTEM%s" % ("   (%d kelime sozlukte YOK)" % n_unknown if n_unknown else ""), prompt_text or "(bos)")
        print_wrapped("MODEL  (sicaklik %.2f, top-p %.2f, tekrar cezasi %.2f)" % (temp, top_p, penalty), generated)
        print()


if __name__ == "__main__":
    main()
