"""Bir islemin TAM hesabi -- her ara sayi basiliyor."""
import torch
from veri_15 import AD, N, ARTI, ESIT
from model_15 import Yol

m = Yol(N, tohum=0)        # ayarlar model_15'ten
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
    print("2) SORU   q = s_son @ Wq = " + v(q[:6]) + " ...")

    puan = (S @ m.Wk) @ q
    ag = puan.softmax(0) if m.pay else puan.relu()
    print()
    print(f"3) PUAN ve AGIRLIK   ({'softmax' if m.pay else 'relu'};"
          f" toplam {float(ag.sum()):.3f})")
    for i, t in enumerate(ad):
        print(f"   yuva {i} {t:<3s} puan {puan[i]:+7.3f}   agirlik {ag[i]:.4f}"
              f"  {'#' * int(float(ag[i]) * 40)}")

    V = S @ m.Wv
    o = ag @ V
    print()
    print("4) CIKTI = sum agirlik_i * v_i  =  " + v(o[:6]) + " ...")
    print(f"   en yakin token: {AD[m.oku(o)]}")
    print()


if __name__ == "__main__":
    with torch.no_grad():
        hesapla([3, 5])
        hesapla([3, 5, 9])
