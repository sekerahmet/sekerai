"""Bir islemin TAM hesabi -- katman katman, yuva yuva."""
import torch
from veri_15 import AD, N, ARTI, ESIT
from model_15 import Yol

m = Yol(N, tohum=0)        # ayarlar model_15'ten
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

        for L in range(m.katman):
            u, ag = m.blok(S, L)
            print()
            print(f"2.{L}) KATMAN {L}   son yuvanin baktigi yerler"
                  f"  (agirlik toplami {float(ag[0, -1].sum()):.3f})")
            for i, t in enumerate(ad):
                a_ = float(ag[0, -1, i])
                print(f"   yuva {i} {t:<3s} {a_:.4f}  {'#' * int(a_ * 30)}")
            S = S + u
            if m.norm:
                S = torch.nn.functional.normalize(S, dim=-1)

        o = S[0, -1] @ m.Wson
        print()
        print(f"3) CIKTI = s_son @ Wson = {v(o)} ...")
        print(f"   en yakin token: {AD[m.oku(o)]}")
        print()


if __name__ == "__main__":
    hesapla([3, 5])
    hesapla([3, 5, 9])
