"""Modelle KONUS -- soru sor, cevabini ve yolu gor.

Egitilmis agirlik Colab'da uretilip Drive'a yaziliyor; burasi sadece okuyor.
Model CPU'da calisir, GPU gerekmez.

  472+182        sor
  ayrinti        dikkat agirliklarini ac / kapa
  ?              yardim
  q              cik
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_15 import Yol                                   # noqa: E402
from veri_15 import AD, ARTI, ESIT, PAD, HA, HC, ENB, soru  # noqa: E402

ADAY = [
    r"G:\Drive'ım\model_15\model_Mb.pt",
    r"G:\Drivem\model_15\model_Mb.pt",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_Mb.pt"),
]


def yukle(yol=None):
    """Agirligi bul ve modeli kur.  Dosya yoksa NE YAPILACAGINI soyler."""
    adaylar = [yol] if yol else ADAY
    for p in adaylar:
        if p and os.path.exists(p):
            k = torch.load(p, weights_only=False, map_location="cpu")
            m = Yol(k["n"], boyut=k["boyut"], durum=k["durum"])
            m.load_state_dict(k["agirlik"])
            m.eval()
            return m, k, p
    print("AGIRLIK DOSYASI YOK.  Arananlar:")
    for p in adaylar:
        print("   " + str(p))
    print()
    print("Colab defterinde 'MODELI DRIVE'A KAYDET' hucresini calistir,")
    print("Drive eslesince dosya buraya duser.")
    sys.exit(1)


def cevapla(m, a, b, ayrinti=False):
    """Serbest uretim: her adimin CIKTISI bir sonraki adimin GIRDISI."""
    yol = list(soru(a, b))
    cikan = []
    for adim in range(HC):
        with torch.no_grad():
            o, ag = m.dikkat(torch.tensor(yol))
            uzak = ((m.E - o) ** 2).sum(-1).sqrt()
        c = int(uzak.argmin())
        cikan.append(c)

        if ayrinti:
            print(f"    -- adim {adim + 1} --")
            print("       yol     " + " ".join(AD[t] for t in yol))
            print("       agirlik " + " ".join(
                f"{float(x):.2f}" for x in ag))
            ilk = uzak.argsort()[:3].tolist()
            print("       en yakin " + "   ".join(
                f"{AD[j]} {float(uzak[j]):.3f}" for j in ilk))
        yol.append(c)
    return cikan, yol


def bicim(x):
    """Sayiyi HC haneli yaz, bos basamaklar PAD."""
    return "".join(AD[t] for t in
                   [PAD] * (HC - len(str(x))) + [int(c) for c in str(x)])


def coz(satir):
    """'472+182' ya da '472 + 182 =' -> (472, 182).  Olmazsa None."""
    s = satir.replace("=", "").replace(" ", "")
    if "+" not in s:
        return None
    p = s.split("+")
    if len(p) != 2 or not all(q.isdigit() for q in p):
        return None
    return int(p[0]), int(p[1])


def yardim():
    print(f"  472+182     sor  (her toplanan en fazla {HA} hane)")
    print(f"  ayrinti     dikkat agirliklarini ac / kapa")
    print(f"  ?           bu yazi")
    print(f"  q           cik")


def main():
    m, k, p = yukle(sys.argv[1] if len(sys.argv) > 1 else None)
    par = sum(t.numel() for t in m.parameters())
    print("=" * 62)
    print(f"  {os.path.basename(p)}   kod {k.get('kod', '?')}"
          f"   veri {k.get('veri', '?')}")
    print(f"  adim {k['adim']}   parametre {par}"
          f"   boyut {k['boyut']}  durum {k['durum']}")
    print(f"  egitim {k['egitim']:.4f}   tutulan {k['tutulan']:.4f}")
    print(f"  yuva " + " ".join(f"{x:.4f}" for x in k["yuva"])
          + "   (binler yuzler onlar birler)")
    print("=" * 62)
    yardim()
    print()

    ayrinti = False
    while True:
        try:
            satir = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not satir:
            continue
        if satir.lower() in ("q", "cik", "exit", "quit"):
            break
        if satir in ("?", "yardim", "help"):
            yardim(); continue
        if satir.lower().startswith("ayrinti"):
            ayrinti = not ayrinti
            print("  ayrinti " + ("ACIK" if ayrinti else "KAPALI"))
            continue

        ab = coz(satir)
        if ab is None:
            print("  anlamadim.  ornek:  472+182      ('?' yardim)")
            continue
        a, b = ab
        if a > 10 ** HA - 1 or b > 10 ** HA - 1:
            print(f"  toplananlar en fazla {HA} hane olabilir.")
            continue

        cikan, yol = cevapla(m, a, b, ayrinti)
        verdi = "".join(AD[c] for c in cikan)
        gercek = bicim(a + b)
        ok = verdi == gercek

        print(f"  yol    " + " ".join(AD[t] for t in yol))
        print(f"  MODEL  {verdi}      DOGRU  {gercek}"
              f"      {'DOGRU' if ok else 'YANLIS'}")
        if a > ENB or b > ENB:
            print(f"  NOT: egitim araligi 0..{ENB}; bu soru DISARIDA.")
        print()


if __name__ == "__main__":
    main()
