"""SIRA gorunuyor mu.  Iki ayri soru, ayri cevaplar."""
import torch
from sehir_15 import AD, IX, N, DUR, dizi
from model_15 import Yol

m = Yol(N, boyut=2, durum=6, dur=IX[DUR], tohum=0)

with torch.no_grad():
    print("1) AYNI SEHIRLER, FARKLI SIRA -- cikti degisiyor mu")
    for a, b in ((["Istanbul", "Ankara", "Mersin"], ["Mersin", "Ankara", "Istanbul"]),
                 (["Izmir", "Bursa", "Mersin"],     ["Mersin", "Bursa", "Izmir"])):
        oa, aga = m.dikkat(dizi(a))
        ob, agb = m.dikkat(dizi(b))
        print(f"   {' '.join(a):<26s} ({oa[0]:+.4f} {oa[1]:+.4f})  agirlik {[round(float(x),3) for x in aga]}")
        print(f"   {' '.join(b):<26s} ({ob[0]:+.4f} {ob[1]:+.4f})  agirlik {[round(float(x),3) for x in agb]}")
        print(f"   fark {(oa - ob).norm():.4f}"
              f"   -> {'SIRA GORUNUYOR' if (oa-ob).norm() > 1e-6 else 'SIRA KOR'}")
        print()

    print("2) AYNI SEHIR IKI KEZ -- iki kopya ayni agirligi mi aliyor")
    yol = ["Ankara", "Mersin", "Ankara", "Sivas"]
    o, ag = m.dikkat(dizi(yol))
    for i, (a, w) in enumerate(zip(yol, ag)):
        print(f"   yuva {i}  {a:<10s} agirlik {w:.4f}")
    i0, i2 = [i for i, a in enumerate(yol) if a == "Ankara"]
    print(f"   iki Ankara farki {abs(ag[i0]-ag[i2]):.6f}"
          f"   -> {'AYIRT EDIYOR' if abs(ag[i0]-ag[i2]) > 1e-6 else 'AYIRT ETMIYOR'}")
