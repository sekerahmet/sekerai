"""Simdiki hali ne veriyor, ve kayip ne kadar.  EGITIM YOK -- tek ileri gecis."""
import math
import torch
from sehir_15 import AD, IX, N, DUR, dizi
from model_15 import Yol

m = Yol(N, boyut=2, durum=6, dur=IX[DUR], tohum=0)

ORNEK = [(["Istanbul", "Ankara", "Mersin"], "Sivas"),
         (["Mersin", "Ankara", "Istanbul"], "Izmir")]

with torch.no_grad():
    toplam = 0.0
    for yol, hedef in ORNEK:
        o, _ = m.dikkat(dizi(yol))
        u = (m.E - o).norm(dim=-1)              # her token'a uzaklik
        puan = -(u ** 2)                        # yakin = yuksek puan
        p = puan.softmax(0)
        h = IX[hedef]
        kayip = -math.log(float(p[h]))
        toplam += kayip

        print("=" * 58)
        print(f"{' '.join(yol)}   ->  hedef {hedef}")
        print(f"cikti ({o[0]:+.3f} {o[1]:+.3f})")
        print()
        print("  token       uzaklik    puan     olasilik")
        for i, a in enumerate(AD):
            im = "  <- HEDEF" if i == h else ("  <- SECILEN" if i == int(p.argmax()) else "")
            print(f"  {a:<10s} {u[i]:7.3f}  {puan[i]:8.3f}   {p[i]:7.4f}{im}")
        print()
        print(f"  dogru olasiligi {p[h]:.4f}    kayip = -log({p[h]:.4f}) = {kayip:.4f}")
        print()

    print("=" * 58)
    print(f"TOPLAM KAYIP  {toplam:.4f}   ortalama {toplam/2:.4f}")
    print(f"SANS SEVIYESI 1/{N} = {1/N:.4f}  ->  kayip -log(1/{N}) = {math.log(N):.4f}")
