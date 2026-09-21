"""Bir yolun TAM hesabi -- her ara sayi basiliyor."""
import torch
from sehir_15 import AD, IX, N, DUR, dizi
from model_15 import Yol

m = Yol(N, boyut=2, durum=6, dur=IX[DUR], tohum=0)
v6 = lambda t: "[" + " ".join(f"{x:+6.2f}" for x in t) + "]"


def hesapla(onek):
    w = dizi(onek)
    print("=" * 72)
    print("YOL  " + " ".join(onek))
    print("=" * 72)

    S = m.gez(w)
    print("1) YOLU GEZ   s_t = M[sehir] @ s_{t-1} + b[sehir]")
    print(f"   s_0  {v6(S[0])}   baslangic")
    for i, a in enumerate(onek):
        print(f"   s_{i+1}  {v6(S[i+1])}   {a} uygulandi")
    S = S[1:]

    q = S[-1] @ m.Wq
    print()
    print(f"2) SORU   q = s_son @ Wq  =  ({q[0]:+.3f} {q[1]:+.3f})")

    K = S @ m.Wk
    puan = K @ q
    print()
    print("3) ANAHTAR ve PUAN   k_i = s_i @ Wk    puan_i = k_i . q")
    for i, a in enumerate(onek):
        print(f"   yuva {i} {a:<10s} k=({K[i,0]:+.3f} {K[i,1]:+.3f})"
              f"   puan = ({K[i,0]:+.3f})({q[0]:+.3f}) + ({K[i,1]:+.3f})({q[1]:+.3f})"
              f" = {puan[i]:+.3f}")

    ag = puan.softmax(0)
    print()
    print("4) AGIRLIK   softmax(puan)")
    for i, a in enumerate(onek):
        print(f"   yuva {i} {a:<10s} {ag[i]:.4f}  {'#' * int(ag[i] * 50)}")

    V = S @ m.Wv
    print()
    print("5) DEGER   v_i = s_i @ Wv")
    for i, a in enumerate(onek):
        print(f"   yuva {i} {a:<10s} ({V[i,0]:+.3f} {V[i,1]:+.3f})")

    o = ag @ V
    print()
    print("6) CIKTI = sum agirlik_i * v_i")
    print("   " + "  +  ".join(f"{ag[i]:.3f}*({V[i,0]:+.3f} {V[i,1]:+.3f})"
                               for i in range(len(onek))))
    print(f"   = ({o[0]:+.3f} {o[1]:+.3f})    en yakin token: {AD[m.oku(o)]}")
    print()
    return o


with torch.no_grad():
    a = hesapla(["Istanbul", "Ankara", "Mersin"])
    b = hesapla(["Mersin", "Ankara", "Istanbul"])
    print(f">>> IKISININ FARKI  {(a - b).norm():.4f}")
    print()
    c = hesapla(["Istanbul", "Ankara", "Mersin", "Sivas"])
    print(">>> 4. SEHIR GELINCE NE DEGISTI")
    print("    s_4 hesaplandi (M[Sivas] @ s_3 + b[Sivas])")
    print("    soru q ARTIK s_4'ten geliyor -- tamamen degisti")
    print("    yuva sayisi 3 -> 4, yani 4 anahtar / 4 puan / 4 agirlik")
    print("    ilk uc yuvanin k ve v DEGERLERI AYNI kaldi (s_1..s_3 degismedi)")
    print("    ama AGIRLIKLARI degisti, cunku soru degisti")
