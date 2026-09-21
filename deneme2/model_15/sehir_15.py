"""Sehirler ve guzergahlar.  ADIM 1.

Koordinatlar SABIT ve VERILI: egitilmez, oynatilmaz.
Boyut 2 -- 3'e cikarmak ADIM 2'nin karari, simdi gerekcesi yok.
"""
import torch

BOYUT = 2

# Kullanicinin verdigi iki deger AYNEN; kalani ornegi tamamliyor.
SEHIR = {
    "Istanbul": (1.0, 1.0),      # kullanici
    "Ankara":   (0.5, 0.5),      # kullanici
    "Izmir":   (-1.0, 0.3),
    "Bursa":    (0.2, 1.1),
    "Mersin":   (0.4, -0.9),
    "Sivas":    (1.4, 0.1),
    "Adana":    (0.9, -0.8),
    "Konya":    (0.6, -0.3),
}

AD = list(SEHIR)
IX = {a: i for i, a in enumerate(AD)}
N = len(AD)

# (N, BOYUT) -- okumanin hedefleri.  buffer olarak tasinacak, Parameter degil.
KONUM = torch.tensor([SEHIR[a] for a in AD], dtype=torch.float32)

# Ornek: son sehir AYNI (Mersin), devam FARKLI -> AYIRMA gerekiyor.
#        ucuncusu devam AYNI -> BIRLESTIRME gerekiyor.
YOL = [
    ["Istanbul", "Ankara", "Mersin", "Sivas"],
    ["Izmir",    "Bursa",  "Mersin", "Adana"],
    ["Konya",    "Ankara", "Mersin", "Sivas"],
]


def dizi(yol):
    """Ad listesi -> indeks tensoru."""
    return torch.tensor([IX[a] for a in yol], dtype=torch.long)


def _goster():
    print(f"SEHIR  {N} adet, boyut {BOYUT}\n")
    print("  ix  ad          x       y      uzunluk   aci")
    for i, a in enumerate(AD):
        x, y = SEHIR[a]
        u = (x * x + y * y) ** 0.5
        ac = torch.atan2(torch.tensor(y), torch.tensor(x)).item() * 180 / 3.14159265
        print(f"  {i:2d}  {a:<10s} {x:6.2f}  {y:6.2f}   {u:6.3f}  {ac:7.1f}")

    print("\nYOL")
    for y in YOL:
        print("  " + " -> ".join(y) + f"      {dizi(y).tolist()}")

    print("\nDENETIM")
    k = KONUM
    d = (k[:, None, :] - k[None, :, :]).norm(dim=-1)
    d.fill_diagonal_(9.9)
    a, b2 = divmod(int(d.argmin()), N)
    print(f"  en yakin iki sehir  {AD[a]} / {AD[b2]}   uzaklik {d.min():.3f}")

    b = k / k.norm(dim=-1, keepdim=True)
    c = b @ b.t()
    c.fill_diagonal_(-1.0)
    i, j = divmod(int(c.argmax()), N)
    print(f"  en yakin iki DOGRULTU  cos {c.max():.4f}"
          f"   {AD[i]} / {AD[j]}")
    if c.max() > 0.9999:
        print("  !! UZUNLUK ATILIRSA bu iki sehir AYNI NOKTA olur.")


if __name__ == "__main__":
    _goster()
