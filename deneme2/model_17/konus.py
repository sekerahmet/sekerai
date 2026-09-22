"""Modelle KONUS -- bir hikaye baslangici yaz, devamini gor.

Egitilmis agirlik Colab'da uretilip Drive'a yaziliyor; burasi sadece okuyor.
Model CPU'da calisir, GPU gerekmez.

Sayi degil METIN: hukum CLAUDE.md'nin DIL alaninda gozle veriliyor.
Kayip perplexity DEGIL (puan -||o-E||^2, olcegi kalibre degil), o yuzden
burada kayip gosterilmiyor -- yalniz modelin YAZDIGI.

  Once upon a time           yaz, devamini gorursun
  <bos satir>                makalenin 44 degerlendirme isteminden RASTGELE biri
  n=120                      kac kelime uretilecek  (varsayilan 80)
  s=0.8                      sicaklik.  0 = hep en yakin kelime (belirlenimci)
  yedek                      hangi yedekler var, hangisi yuklu
  yedek t500                 baska bir ani yukle -- egri boyunca gezinmek icin
  ?                          yardim
  q                          cik
"""
import os
import random
import re
import sys
import textwrap

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_17 import Yol                                     # noqa: E402
import veri_t17 as V                                         # noqa: E402
import olcme_17 as OL                                        # noqa: E402

ADAY = [r"G:\Drive'ım\model_17",
        r"G:\Drivem\model_17",
        os.path.dirname(os.path.abspath(__file__))]
TS = [r"G:\Drive'ım\tinystories", r"G:\Drivem\tinystories"]


def _kok():
    for d in ADAY:
        if os.path.isdir(d):
            return d
    return None


def yedekler():
    """(ad, yol, adim) -- yeniden eskiye."""
    k = _kok()
    if not k:
        return []
    cik = []
    for kok, _, dosyalar in os.walk(k):
        for f in dosyalar:
            if re.fullmatch(r"t\d+\.pt", f) or re.fullmatch(r"model_.+\.pt", f):
                p = os.path.join(kok, f)
                n = re.findall(r"\d+", f)
                cik.append((os.path.relpath(p, k), p,
                            int(n[0]) if n and f[0] == "t" else -1))
    return sorted(cik, key=lambda x: os.path.getmtime(x[1]), reverse=True)


def bul(desen=None):
    """Agirlik dosyasini bul.  Desen verilmezse EN YENISI."""
    hepsi = yedekler()
    if not hepsi:
        print("AGIRLIK DOSYASI YOK.  Arananlar:")
        for d in ADAY:
            print("   %s\\*\\t*.pt" % d)
        print()
        print("Colab defterinde kosuyu baslat; yedek her 50 adimda")
        print("Drive'a yaziliyor, eslesince buraya duser.")
        sys.exit(1)
    if desen:
        e = [x for x in hepsi if desen in x[0]]
        if not e:
            print("'%s' ile eslesen yedek yok.  'yedek' yazip listeye bak."
                  % desen)
            return None
        return e[0][1]
    return hepsi[0][1]


def yukle(yol):
    k = torch.load(yol, weights_only=False, map_location="cpu")
    m = Yol(k["n"], boyut=k["boyut"], durum=k["durum"])
    m.load_state_dict(k["agirlik"])
    m.eval()
    return m, k


def sozluk(k):
    """Sozluk KAYITTAN degil, ONBELLEKTEN -- kayit yalniz boyutu tasiyor.
    Iz tutmazsa DURUR: yanlis sozlukle uretilen metin sessizce sacmalar."""
    for d in TS:
        p = os.path.join(d, "onbellek")
        if not os.path.isdir(p):
            continue
        for f in sorted(os.listdir(p)):
            if f.startswith("sozluk_"):
                import numpy as np
                ad = list(np.load(os.path.join(p, f), allow_pickle=True))
                if len(ad) == k["n"]:
                    return ad, f
    print("SOZLUK BULUNAMADI (%d birim).  Aranan: <tinystories>\\onbellek\\"
          "sozluk_*.npy" % k["n"])
    sys.exit(1)


def istemler():
    """Makalenin 44 degerlendirme istemi -- varsa."""
    for d in TS:
        p = os.path.join(d, "Evaluation prompts.yaml")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                y = f.read()
            return [re.sub(r"\s+", " ", b).strip()
                    for b in re.split(r"\n- \|-\n", y)[1:]]
    return []


def yaz(baslik, metin, g=70):
    print(baslik)
    for s in textwrap.wrap(metin, g) or [""]:
        print("   " + s)


def main():
    yol = bul(sys.argv[1] if len(sys.argv) > 1 else None)
    m, k = yukle(yol)
    AD, sz = sozluk(k)
    IX = {a: i for i, a in enumerate(AD)}
    IST = istemler()

    print("=" * 74)
    print("yedek   %s" % os.path.basename(yol))
    print("adim    %s   parametre %s   sozluk %d"
          % ("{:,}".format(k.get("adim", -1)),
             "{:,}".format(k.get("parametre", 0)), k["n"]))
    print("olcum   egitim %.4f   dogrulama %.4f   (sonraki jeton)"
          % (k.get("egitim", float("nan")), k.get("dogrulama", float("nan"))))
    print("sozluk  %s   |   %d degerlendirme istemi" % (sz, len(IST)))
    print("=" * 74)
    print(__doc__.split("\n\n")[-1].rstrip())
    print()

    n, sic = 80, 0.0
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
            print("   uretim %d kelime" % n)
            continue
        if g.startswith("s="):
            sic = max(0.0, float(g[2:]))
            print("   sicaklik %.2f%s" % (sic, "  (belirlenimci)" if not sic
                                          else ""))
            continue
        if g.startswith("yedek"):
            p = g[5:].strip()
            if not p:
                for ad_, _, _ in yedekler()[:12]:
                    print("   %s%s" % (ad_, "   <- yuklu"
                                       if ad_ in yol else ""))
                continue
            y2 = bul(p)
            if y2:
                yol = y2
                m, k = yukle(yol)
                print("   yuklendi %s   adim %s"
                      % (os.path.basename(yol), "{:,}".format(k.get("adim", -1))))
            continue
        if not g:
            if not IST:
                print("   istem dosyasi yok; kendin bir seyler yaz")
                continue
            g = random.choice(IST)

        onek = V.JETON.findall(g.translate(V.DUZLE))[-60:]
        if not onek:
            continue
        kod = [IX.get(t, IX[V.BILINMEYEN]) for t in onek]
        bil = sum(1 for t in onek if t not in IX)
        bas = V.coz(torch.tensor(kod), AD)
        hep = OL.devam(m, onek, AD, IX, V.coz, adim=n, aygit="cpu",
                       sicaklik=sic)
        print()
        yaz("ISTEM%s" % ("   (%d kelime sozlukte YOK)" % bil if bil else ""),
            bas)
        yaz("MODEL  (%d kelime, sicaklik %.2f)" % (n, sic),
            hep[len(bas):].strip())
        print()


if __name__ == "__main__":
    main()
