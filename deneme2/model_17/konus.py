"""Modelle KONUS -- bir hikaye baslangici yaz, devamini gor.

Egitilmis agirlik Colab'da uretilip Drive'a yaziliyor; burasi sadece okuyor.
Model CPU'da calisir, GPU gerekmez.

Sayi degil METIN: hukum CLAUDE.md'nin DIL alaninda gozle veriliyor.
Sayilar (dogrulama perplexity'si) basliktaki paketten; burada yalniz
modelin YAZDIGI.  BOS/EOS'la egitilmis model <eos>'tan baslar ve kendi
<eos>'unu yazinca durur (sonda "---").

  Once upon a time           yaz, devamini gorursun
  <bos satir>                makalenin 44 degerlendirme isteminden RASTGELE biri
  serbest                    istemsiz, bos sayfadan bir hikaye (BOS/EOS'lu model)
  n=120                      kac kelime uretilecek  (varsayilan 80, istem+n <= T)
  s=0.8                      sicaklik.  0 = hep en yakin kelime (belirlenimci)
  yasak                      <bilinmeyen>/<dolgu> uretimi kapat (varsayilan) / ac
  yedek                      hangi yedekler var, hangisi yuklu
  yedek TAM1/t500            baska bir ani yukle -- ad BIREBIR (t500, t5000 degil)
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
from model_17 import Yol, DT                                 # noqa: E402
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


def _ad(x):
    """Yedegin goreli yolu, '/' ile ve .pt'siz: TAM1/t500."""
    return x[0].replace("\\", "/")[:-3]


def bul(desen=None):
    """Agirlik dosyasini bul.  Desen verilmezse EN YENISI.

    Desen dosya adiyla BIREBIR eslesir: t500 -> t500.pt, t5000 DEGIL.
    Birden cok kosuda varsa klasorle yazilir (TAM1/t500); TS2 gibi bir
    kosu adi model_TS2.pt'yi getirir."""
    hepsi = yedekler()
    if not hepsi:
        print("AGIRLIK DOSYASI YOK.  Arananlar:")
        for d in ADAY:
            print("   %s\\*\\t*.pt" % d)
        print()
        print("Colab defterinde kosuyu baslat; yedek her `yedek` adimda")
        print("Drive'a yaziliyor, eslesince buraya duser.")
        sys.exit(1)
    if not desen:
        return hepsi[0][1]
    d = desen.replace("\\", "/")
    d = d[:-3] if d.endswith(".pt") else d
    e = [x for x in hepsi if _ad(x) == d or _ad(x).endswith("/" + d)]
    e = e or [x for x in hepsi if _ad(x) == "model_" + d]
    if not e:
        print("'%s' ile eslesen yedek yok.  'yedek' yazip listeye bak."
              % desen)
        return None
    if len(e) > 1:
        print("'%s' birden cok yerde var, klasorle yaz:" % desen)
        for x in e:
            print("   " + _ad(x))
        return None
    return e[0][1]


def yukle(yol):
    """Paketin mimarisine gore: "dt" hedef mimari, yoksa kelime = matris."""
    k = torch.load(yol, weights_only=False, map_location="cpu")
    if k.get("mimari") == "dt":
        m = DT(k["n"], genislik=k["genislik"], durum=k["durum"],
               blok=k["blok"], bellek=k["bellek"], yansima=k["yansima"],
               pay=k["pay"], gb_W=k.get("gb_W", 0), gb_kafa=k.get("gb_kafa", 4),
               gb_oncul=k.get("gb_oncul", 0.0))
    else:
        m = Yol(k["n"], boyut=k["boyut"], durum=k["durum"],
                **{a: k[a] for a in ("norm", "pay") if k.get(a) is not None})
    m.load_state_dict(k["agirlik"])
    m.eval()
    return m, k


def sozluk(k, m):
    """Sozluk PAKETTEN.  Eski paketler tasimiyor; onlar icin ONBELLEKTEN.

    Uzunluk YETMEZ: 64mb ve tam sozluklerinin ikisi de 4.003 birim ama
    4.003 konumun 3.784'u farkli kelime (olculdu).  O yuzden MODELE
    SORULUR -- her adayin kendi dogrulama akisindan bir pencere alinip
    kayip hesaplanir; dogru sozluk belirgin dusuk cikar."""
    if k.get("sozluk"):
        assert len(k["sozluk"]) == k["n"], "paketteki sozluk n ile tutmuyor"
        return V.genel(list(k["sozluk"])), "paketten"
    import numpy as np
    aday = []
    for d in TS:
        ob = os.path.join(d, "onbellek")
        if not os.path.isdir(ob):
            continue
        for f in sorted(os.listdir(ob)):
            if not f.startswith("sozluk_"):
                continue
            ad = V.genel(list(np.load(os.path.join(ob, f), allow_pickle=True)))
            if len(ad) != k["n"]:
                continue
            av = os.path.join(ob, f.replace("sozluk_", "akis_valid_"))
            if not os.path.exists(av):
                continue
            a_ = np.asarray(np.load(av, mmap_mode="r")[:16 * 256], "int64")
            with torch.no_grad():
                kay = float(m.kayip(torch.from_numpy(a_.reshape(16, 256))))
            aday.append((kay, f, ad))
    if not aday:
        print("SOZLUK BULUNAMADI (%d birim)." % k["n"])
        sys.exit(1)
    aday.sort()
    if len(aday) > 1:
        print("sozluk adaylari (kayip -- KUCUK olan secilir):")
        for kay, f, _ in aday:
            print("   %8.4f  %s" % (kay, f))
    return aday[0][2], aday[0][1]

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
    if not yol:
        sys.exit(1)
    m, k = yukle(yol)
    AD, sz = sozluk(k, m)
    IX = {a: i for i, a in enumerate(AD)}
    IST = istemler()

    print("=" * 74)
    print("yedek   %s" % os.path.basename(yol))
    print("adim    %s   parametre %s   sozluk %d   mimari %s"
          % ("{:,}".format(k.get("adim", -1)),
             "{:,}".format(k.get("parametre", 0)), k["n"],
             k.get("mimari", "kelime_matris")))
    print("olcum   egitim %.4f   dogrulama %.4f   (sonraki jeton dogrulugu)"
          % (k.get("egitim", float("nan")), k.get("dogrulama", float("nan"))))
    if k.get("dogrulama_ce") is not None:
        g_ = (k.get("dogrulama_tur") or {}).get("ce_govde")
        print("        dogrulama perplexity %.3f%s" % (
            math.exp(k["dogrulama_ce"]),
            "   govde %.3f" % math.exp(g_) if g_ is not None else ""))
    print("BOS/EOS %s" % ("VAR -- <eos>'tan baslar, <eos>'ta durur"
                          if k.get("bos_eos") else "YOK (eski paket)"))
    print("sozluk  %s   |   %d degerlendirme istemi" % (sz, len(IST)))
    print("=" * 74)
    print(__doc__.split("\n\n")[-1].rstrip())
    print()

    n, sic, yasak = 80, 0.0, True
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
        if g == "yasak":
            yasak = not yasak
            print("   <bilinmeyen>/<dolgu> %s"
                  % ("URETILMEZ" if yasak else "uretilebilir"))
            continue
        if g.startswith("yedek"):
            p = g[5:].strip()
            if not p:
                for ad_, p_, _ in yedekler()[:12]:
                    print("   %s%s" % (ad_, "   <- yuklu"
                                       if os.path.normcase(p_)
                                       == os.path.normcase(yol) else ""))
                continue
            y2 = bul(p)
            if y2:
                yol = y2
                m, k = yukle(yol)
                # Baska kosunun paketi baska sozlukle egitilmis olabilir.
                AD, sz = sozluk(k, m)
                IX = {a: i for i, a in enumerate(AD)}
                print("   yuklendi %s   adim %s   sozluk %s"
                      % (os.path.basename(yol),
                         "{:,}".format(k.get("adim", -1)), sz))
            continue
        bos = k.get("hikaye_id") if k.get("bos_eos") else None
        if g == "serbest":
            if bos is None:
                print("   bu paket BOS/EOS'suz egitildi; istemsiz uretemez")
                continue
            g = ""
        elif not g:
            if not IST:
                print("   istem dosyasi yok; kendin bir seyler yaz")
                continue
            g = random.choice(IST)

        # Model yalniz hikaye BASINDAN baslayan pencere gordu: istemin
        # basi kesilmez, sigmazsa uretim kisalir.
        T = k.get("T", 256) - (1 if bos is not None else 0)
        onek = V.JETON.findall(g.translate(V.DUZLE))
        if not onek and bos is None:
            continue
        if len(onek) >= T:
            print("   UYARI: istem %d kelime, T=%d -- BASI kesildi; model "
                  "hikaye ortasindan baslayan pencere gormedi" % (len(onek), T))
            onek = onek[-(T - 1):]
        n_ = min(n, T - len(onek))
        if n_ < n:
            print("   uretim %d -> %d kelime: istem + uretim T=%d'yi asamaz"
                  % (n, n_, T))
        kod = [IX.get(t, IX[V.BILINMEYEN]) for t in onek]
        bil = sum(1 for t in onek if t not in IX)
        Y = [IX[t] for t in (V.BILINMEYEN, V.DOLGU) if yasak and t in IX]
        bas = V.coz(torch.tensor(kod, dtype=torch.long), AD)
        hep = OL.devam(m, onek, AD, IX, V.coz, adim=n_, aygit="cpu",
                       sicaklik=sic, yasak=Y, bos=bos)
        print()
        yaz("ISTEM%s" % ("   (%d kelime sozlukte YOK)" % bil if bil else ""),
            bas)
        yaz("MODEL  (%d kelime, sicaklik %.2f%s)"
            % (n_, sic, "   yasak: " + " ".join(AD[i] for i in Y) if Y else ""),
            hep[len(bas):].strip())
        print()


if __name__ == "__main__":
    main()
