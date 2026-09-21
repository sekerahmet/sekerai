"""Ozyineleme gercekten oluyor mu -- ciktı bir sonraki adimin girdisi mi."""
import torch
from sehir_15 import AD, dizi, N
from model_15 import Yol

m = Yol(N, boyut=2, durum=6, tohum=0)
onek = ["Istanbul", "Ankara", "Mersin"]

with torch.no_grad():
    v = m.yol(dizi(onek))
    print(f"girdi   {' '.join(onek):<38s} {len(v)} sayi")
    for adim in range(3):
        onceki = v.clone()
        c = m.oku_yol(v)
        v = m.ekle(v, c)
        onek = onek + [AD[c]]

        # DENETIM
        assert len(v) == len(onceki) + 2,           "uzunluk 2 artmadi"
        assert torch.equal(v[:-2], onceki),         "eski yol KORUNMADI"
        assert torch.equal(v[-2:], m.E[c]),         "eklenen sey E[c] degil"

        print(f"  -> {AD[c]:<10s}  eklenen {v[-2].item():6.3f} {v[-1].item():6.3f}"
              f"   = E[{c}] {'EVET' if torch.equal(v[-2:], m.E[c]) else 'HAYIR'}")
        print(f"girdi   {' '.join(onek):<38s} {len(v)} sayi")

    print()
    print("DENETIM  her adimda:")
    print("  uzunluk tam 2 artti           GECTI")
    print("  eski yol degismeden korundu   GECTI")
    print("  eklenen sayilar E[c] ile ayni GECTI")
