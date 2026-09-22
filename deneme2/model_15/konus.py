"""Modelle KONUS -- soru sor, cevabini ve yolu gor.

Egitilmis agirlik Colab'da uretilip Drive'a yaziliyor; burasi sadece okuyor.
Model CPU'da calisir, GPU gerekmez.

Iki sinav bicimini de acar, hangisi oldugunu KAYITTAN anlar:
  DOLGULU   _ _ 1 + _ _ 1 = _ _ _ 2     cevap hep 4 rakam    (veri_15)
  DOLGUSUZ  1 + 1 = 2                   cevap DUR ile biter  (veri_dur)

  472+182        sor
  ayrinti        dikkat agirliklarini ac / kapa
  ?              yardim
  q              cik
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_15 import Yol                                     # noqa: E402

ADAY = [
    r"G:\Drive'ım\model_15",
    r"G:\Drivem\model_15",
    os.path.dirname(os.path.abspath(__file__)),
]


def bul(yol=None):
    """Agirlik dosyasini bul.  Verilmezse klasorlerdeki model_*.pt'lerden
    EN YENISI secilir -- 'en guncel model' istendiginde tahmin ettirmemek icin."""
    if yol:
        return yol
    hepsi = []
    for d in ADAY:
        if os.path.isdir(d):
            hepsi += [os.path.join(d, f) for f in os.listdir(d)
                      if f.startswith("model_") and f.endswith(".pt")]
    if not hepsi:
        print("AGIRLIK DOSYASI YOK.  Arananlar:")
        for d in ADAY:
            print(f"   {d}\\model_*.pt")
        print()
        print("Colab defterinde kosuyu baslat; agirlik her 2000 adimda")
        print("Drive'a yaziliyor, eslesince buraya duser.")
        sys.exit(1)
    return max(hepsi, key=os.path.getmtime)


def yukle(yol):
    k = torch.load(yol, weights_only=False, map_location="cpu")
    m = Yol(k["n"], boyut=k["boyut"], durum=k["durum"])
    m.load_state_dict(k["agirlik"])
    m.eval()
    # BICIM KAYITTAN okunur, tahmin edilmez.
    if "DUR" in k:
        import veri_dur as V
        bicim = "DOLGUSUZ (DUR ile biter)"
    else:
        import veri_15 as V
        bicim = "DOLGULU (cevap hep %d rakam)" % V.HC
    return m, k, V, bicim


def cevapla(m, V, a, b, ayrinti=False):
    """Serbest uretim: her adimin CIKTISI bir sonraki adimin GIRDISI."""
    dur = getattr(V, "DUR", None)
    enfazla = 4 if dur is not None else V.HC
    yol, cikan = list(V.soru(a, b)), []
    for adim in range(enfazla):
        with torch.no_grad():
            o, ag = m.dikkat(torch.tensor(yol))
            uzak = ((m.E - o) ** 2).sum(-1).sqrt()
        c = int(uzak.argmin())
        if ayrinti:
            print(f"    -- adim {adim + 1} --")
            print("       yol     " + " ".join(V.AD[t] for t in yol))
            print("       agirlik " + " ".join(f"{float(x):.2f}" for x in ag))
            print("       en yakin " + "   ".join(
                f"{V.AD[j]} {float(uzak[j]):.3f}" for j in uzak.argsort()[:3]))
        yol.append(c)
        if c == dur:
            break
        cikan.append(c)
    return cikan, yol


def coz(satir):
    """'472+182' ya da '472 + 182 =' -> (472, 182).  Olmazsa None."""
    s = satir.replace("=", "").replace(" ", "")
    if "+" not in s:
        return None
    p = s.split("+")
    if len(p) != 2 or not all(q.isdigit() for q in p):
        return None
    return int(p[0]), int(p[1])


def main():
    yol = bul(sys.argv[1] if len(sys.argv) > 1 else None)
    m, k, V, bicim = yukle(yol)
    par = sum(t.numel() for t in m.parameters())
    print("=" * 66)
    print(f"  {os.path.basename(yol)}   kod {k.get('kod','?')}"
          f"   veri {k.get('veri','?')}")
    print(f"  SINAV  {bicim}")
    print(f"  adim {k['adim']}   parametre {par}"
          f"   boyut {k['boyut']}  durum {k['durum']}")
    print(f"  egitim {k['egitim']:.4f}   tutulan {k['tutulan']:.4f}")
    print("=" * 66)
    print("  472+182   sor      ayrinti   dikkati ac/kapa      ?  yardim"
          "      q  cik")
    print()

    ayrinti = False
    while True:
        try:
            satir = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not satir:
            continue
        if satir.lower() in ("q", "cik", "exit", "quit"):
            break
        if satir in ("?", "yardim", "help"):
            print(f"  472+182     sor  (toplananlar 0..{V.ENB} ogretildi)")
            print("  ayrinti     dikkat agirliklarini ac / kapa")
            print("  q           cik")
            continue
        if satir.lower().startswith("ayrinti"):
            ayrinti = not ayrinti
            print("  ayrinti " + ("ACIK" if ayrinti else "KAPALI"))
            continue

        ab = coz(satir)
        if ab is None:
            print("  anlamadim.  ornek:  472+182      ('?' yardim)")
            continue
        a, b = ab
        if max(a, b) > 999:
            print("  toplananlar en fazla 3 hane olabilir.")
            continue

        cikan, izlek = cevapla(m, V, a, b, ayrinti)
        verdi = "".join(V.AD[c] for c in cikan)
        gercek = "".join(V.AD[t] for t in V.rak(a + b)) \
            if hasattr(V, "DUR") else "".join(V.AD[t] for t in V.rak(a + b, V.HC))
        ok = verdi == gercek
        print("  yol    " + " ".join(V.AD[t] for t in izlek))
        print(f"  MODEL  {verdi or '(bos)'}      DOGRU  {gercek}"
              f"      {'DOGRU' if ok else 'YANLIS'}")
        if a > V.ENB or b > V.ENB:
            print(f"  NOT: egitim araligi 0..{V.ENB}; bu soru DISARIDA.")
        print()


if __name__ == "__main__":
    main()
