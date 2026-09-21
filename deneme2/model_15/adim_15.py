"""Bir islemin TAM hesabi -- her ara sayi basiliyor."""
import torch
from toplama_15 import AD, N, ARTI, ESIT
from model_15 import Yol

m = Yol(N, boyut=2, durum=16, tohum=0)
v = lambda t: "[" + " ".join(f"{x:+5.2f}" for x in t) + "]"


def hesapla(a):
    w = torch.tensor([a[0]] + [t for x in a[1:] for t in (ARTI, x)] + [ESIT])
    ad = [AD[i] for i in w.tolist()]
    print("=" * 74)
    print("ISLEM  " + " ".join(ad) + f"      dogru {sum(a)}")
    print("=" * 74)

    S = m.gez(w)
    print("1) GEZ   s_t = M[token] @ s_{t-1} + b[token]   (norm ACIK)")
    for i in range(len(S)):
        et = "baslangic" if i == 0 else ad[i - 1]
        print(f"   s_{i:<2d} {et:<9s} {v(S[i][:8])} ...")
    S = S[1:]

    q = S[-1] @ m.Wq
    print()
    print(f"2) SORU   q = s_son @ Wq = ({q[0]:+.3f} {q[1]:+.3f})")

    puan = (S @ m.Wk) @ q
    ag = puan.softmax(0)
    print()
    print("3) PUAN ve AGIRLIK")
    for i, t in enumerate(ad):
        print(f"   yuva {i} {t:<3s} puan {puan[i]:+7.3f}   agirlik {ag[i]:.4f}"
              f"  {'#' * int(ag[i] * 40)}")

    V = S @ m.Wv
    o = ag @ V
    print()
    print(f"4) CIKTI = sum agirlik_i * v_i  =  ({o[0]:+.3f} {o[1]:+.3f})")
    print(f"   en yakin token: {AD[m.oku(o)]}")
    print()


if __name__ == "__main__":
    with torch.no_grad():
        hesapla([3, 5])
        hesapla([3, 5, 9])
