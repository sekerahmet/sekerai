"""Modelle KONUS -- bir hikaye baslangici yaz, PV devamini yazsin.

Agirlik Colab'da uretilip Drive'a yaziliyor; burasi yalniz OKUYOR.  Model CPU'da
calisir.  Kosu surerken de calisir: en yeni w/t paketi alinir.  Model <eos>'tan
baslar ve kendi <eos>'unu yazinca durur (sonda "---").

  Once upon a time           yaz, devamini gorursun
  <bos satir>                makalenin degerlendirme istemlerinden RASTGELE biri
  serbest                    istemsiz, bos sayfadan bir hikaye
  n=120                      en fazla kac kelime (varsayilan 120, istem+n <= T)
  s=0.8                      sicaklik.  0 = hep en yuksek puan (belirlenimci)
  r=1.3                      tekrar cezasi: son 20 kelimede gecenler 1,3'e bolunur (1 = kapali)
  p=0.9                      top-p: en olasi kelimelerden toplami 0,9 olan kumeden sec (1 = kapali; s > 0 ister)
  yasak                      <bilinmeyen>/<dolgu> uretimi kapat (varsayilan) / ac
  yedek                      hangi paketler var, hangisi yuklu
  yedek t4000                baska bir ani yukle (ayni kosuda); TS_PV_D1024/w500 da olur
  ?                          yardim
  q                          cik
"""
import math
import os
import random
import re
import sys
import textwrap

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_18 import PV                                       # noqa: E402
import data_stories as DS                                     # noqa: E402

ADAY = [r"G:\Drive'ım\model_18", r"G:\Drivem\model_18"]
TS = [r"G:\Drive'ım\tinystories", r"G:\Drivem\tinystories"]


def _kok(liste):
    return next((d for d in liste if os.path.isdir(d)), None)


def paketler():
    """(kosu/dosya, yol) -- yeniden eskiye.  Yalniz kosu klasorlerindeki t/w paketleri."""
    k = _kok(ADAY)
    if not k:
        return []
    cik = []
    for kosu in os.listdir(k):
        d = os.path.join(k, kosu)
        if not os.path.isdir(d) or "_eski_" in kosu:
            continue
        for f in os.listdir(d):
            if re.fullmatch(r"[tw]\d+\.pt", f):
                cik.append(("%s/%s" % (kosu, f[:-3]), os.path.join(d, f)))
    return sorted(cik, key=lambda x: os.path.getmtime(x[1]), reverse=True)


def bul(desen=None, kosu=None):
    """Desen yoksa EN YENI hikaye paketi.  'w500' yuklu kosuda, 'KOSU/w500' her yerde;
    ad BIREBIR (t500, t5000 degil)."""
    hepsi = paketler()
    if not desen:
        for ad, yol in hepsi:
            if len(torch.load(yol, weights_only=False, map_location="cpu").get("vocab") or []) > 13:
                return yol
        print("HIKAYE PAKETI YOK.  Arananlar: %s\\<kosu>\\t*.pt, w*.pt" % " ya da ".join(ADAY))
        sys.exit(1)
    d = desen.replace("\\", "/")
    d = d[:-3] if d.endswith(".pt") else d
    e = [y for a, y in hepsi if a == d or (kosu and a == "%s/%s" % (kosu, d))]
    if not e:
        print("'%s' ile eslesen paket yok.  'yedek' yazip listeye bak." % desen)
        return None
    return e[0]


def yukle(yol):
    k = torch.load(yol, weights_only=False, map_location="cpu")
    m = PV(k["n"], d=k["d"], vectors=k["vectors"], active=k["active"], layers=k["layers"],
           t_max=k["t_max"], seed=k["seed"], squared=k.get("squared", True),
           S_p=k.get("S_p", 1.0), lam=k.get("lam", 1.0),
           chain=k.get("chain", "absolute"), c_cache=k.get("c_cache", False),
           cache_topk=k.get("cache_topk", 8), cache_skip=k.get("cache_skip", 3))
    m.load_state_dict(k["weights"])
    return m.eval(), k


def istemler():
    d = _kok(TS)
    return DS.istemler(d) if d else []


def yaz(baslik, metin, g=70):
    print(baslik)
    for s in textwrap.wrap(metin, g) or [""]:
        print("   " + s)


def ozet(yol, k):
    g = k.get("heldout_diag") or {}
    ce = k.get("heldout_ce")
    print("=" * 74)
    print("paket   %s/%s%s" % (os.path.basename(os.path.dirname(yol)), os.path.basename(yol),
                               "   (BITTI)" if k.get("done") else ""))
    print("adim    %s   parametre %s   sozluk %d   d %d   T %d"
          % ("{:,}".format(k["step"]), "{:,}".format(k.get("n_params", 0)), k["n"], k["d"],
             k["t_max"]))
    print("olcum   accuracy train %.4f   heldout %.4f   ppl %s"
          % (k["train_acc"], k["heldout_acc"], "-" if ce is None else "%.1f" % math.exp(ce)))
    if g:
        print("        " + "   ".join("%s %.4f" % (a, v) for a, v in g.items()))
    print("=" * 74)


def main():
    yol = bul(sys.argv[1] if len(sys.argv) > 1 else None)
    if not yol:
        sys.exit(1)
    m, k = yukle(yol)
    SOZ = DS.genel(list(k["vocab"]))
    IX = {a: i for i, a in enumerate(SOZ)}
    IST = istemler()
    ozet(yol, k)
    print("%d degerlendirme istemi" % len(IST))
    print(__doc__.split("\n\n")[-1].rstrip())
    print()

    n, sic, yasak, ceza, top_p = 120, 0.0, True, DS.TEKRAR_CEZASI, DS.TOP_P
    while True:
        try:
            g = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if g in ("q", "quit", "cik"):
            break
        if g == "?":
            print(__doc__.split("\n\n")[-1].rstrip())
            continue
        if g.startswith("n="):
            n = max(1, int(g[2:]))
            print("   en fazla %d kelime" % n)
            continue
        if g.startswith("s="):
            sic = max(0.0, float(g[2:]))
            print("   sicaklik %.2f%s" % (sic, "  (belirlenimci)" if not sic else ""))
            continue
        if g.startswith("r="):
            ceza = max(1.0, float(g[2:]))
            print("   tekrar cezasi %.2f%s" % (ceza, "  (kapali)" if ceza == 1 else ""))
            continue
        if g.startswith("p="):
            top_p = min(1.0, max(0.01, float(g[2:])))
            print("   top-p %.2f%s" % (top_p, "  (kapali)" if top_p == 1 else ""))
            continue
        if g == "yasak":
            yasak = not yasak
            print("   <bilinmeyen>/<dolgu> %s" % ("URETILMEZ" if yasak else "uretilebilir"))
            continue
        if g.startswith("yedek"):
            p = g[5:].strip()
            if not p:
                for ad_, p_ in paketler()[:15]:
                    print("   %s%s" % (ad_, "   <- yuklu" if os.path.normcase(p_)
                                       == os.path.normcase(yol) else ""))
                continue
            y2 = bul(p, os.path.basename(os.path.dirname(yol)))
            if y2:
                yol = y2
                m, k = yukle(yol)
                SOZ = DS.genel(list(k["vocab"]))
                IX = {a: i for i, a in enumerate(SOZ)}
                ozet(yol, k)
            continue
        if g == "serbest":
            g = ""
        elif not g:
            if not IST:
                print("   istem dosyasi yok; kendin bir seyler yaz")
                continue
            g = random.choice(IST)
        bil = sum(1 for t in DS.JETON.findall(g.translate(DS.DUZLE)) if t not in IX)
        y = (DS.DOLGU, DS.BILINMEYEN) if yasak else ()
        bas, hep = DS.devam(m, g, SOZ, IX, adim=n, aygit="cpu", sicaklik=sic,
                            tohum=random.randrange(10 ** 6) if sic else None, yasak=y,
                            tekrar=ceza, top_p=top_p)
        print()
        yaz("ISTEM%s" % ("   (%d kelime sozlukte YOK)" % bil if bil else ""), bas or "(bos)")
        yaz("MODEL  (sicaklik %.2f, top-p %.2f, tekrar cezasi %.2f)" % (sic, top_p, ceza), hep)
        print()


if __name__ == "__main__":
    main()
