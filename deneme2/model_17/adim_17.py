"""Bir islemin TAM hesabi -- katman katman, yuva yuva."""
import torch
from veri_17 import AD, N, ARTI, ESIT
from model_17 import Yol

m = Yol(N, tohum=0)        # ayarlar model_17'ten
v = lambda t: "[" + " ".join(f"{x:+5.2f}" for x in t[:6]) + "]"


def hesapla(a):
    w = torch.tensor([a[0]] + [t for x in a[1:] for t in (ARTI, x)] + [ESIT])
    ad = [AD[i] for i in w.tolist()]
    print("=" * 74)
    print("ISLEM  " + " ".join(ad) + f"      dogru {sum(a)}")
    print("=" * 74)

    with torch.no_grad():
        S = m.gez(w)[1:][None]
        print("1) GEZ   s_t = normalize(M[token] @ s_{t-1} + b[token])")
        for i, t in enumerate(ad):
            print(f"   s_{i:<2d} {t:<3s} {v(S[0, i])} ...")

        o, ag = m.dikkat(w)
        print()
        print(f"2) DIKKAT   son yuva neye bakiyor"
              f"  (agirlik toplami {float(ag.sum()):.3f})")
        for i, t in enumerate(ad):
            a_ = float(ag[i])
            print(f"   yuva {i} {t:<3s} {a_:.4f}  {'#' * int(max(a_, 0) * 30)}")

        print()
        print(f"3) CIKTI = sum(agirlik * deger) = {v(o)} ...")
        print(f"   en yakin token: {AD[m.oku(o)]}")
        print()


if __name__ == "__main__":
    hesapla([3, 5])
    hesapla([3, 5, 9])
