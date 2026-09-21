"""Yazdigimiz sey calisiyor mu.  Egitim YOK -- yalniz mekanizma."""
import torch
from sehir_15 import AD, IX, N, DUR, dizi
from model_15 import Yol

m = Yol(N, boyut=2, durum=6, dur=IX[DUR], tohum=0)
p = dict(m.named_parameters())
print("PARAMETRE")
for ad, t in p.items():
    print(f"  {ad:<3s} {str(tuple(t.shape)):<12s} {t.numel():4d}")
print(f"  toplam {sum(t.numel() for t in p.values())}")

with torch.no_grad():
    print()
    print("UZUNLUK SERBEST -- ayni matrislerle 1, 3, 6 sehir")
    for onek in (["Mersin"], ["Istanbul", "Ankara", "Mersin"],
                 ["Izmir", "Bursa", "Konya", "Ankara", "Sivas", "Mersin"]):
        o, ag = m.dikkat(dizi(onek))
        print(f"  {len(onek)} sehir -> {len(ag)} agirlik, cikti 2 sayi"
              f"   ({o[0]:+.3f} {o[1]:+.3f}) -> {AD[m.oku(o)]}")

    print()
    print("OZYINELEME -- cikti girdiye ekleniyor mu")
    onek = ["Istanbul", "Ankara", "Mersin"]
    w = dizi(onek).tolist()
    for _ in range(3):
        o, ag = m.dikkat(torch.tensor(w))
        c = m.oku(o)
        onceki = list(w)
        w.append(c)
        assert w[:-1] == onceki, "eski yol bozuldu"
        assert len(ag) == len(onceki), "agirlik sayisi yol uzunluguna esit degil"
        print(f"  {' '.join(AD[i] for i in onceki):<40s} -> {AD[c]}")
    print()
    print("  eski yol korunuyor            GECTI")
    print("  agirlik sayisi = yol uzunlugu GECTI")

    print("\nSIRA GORUNUYOR MU")
    for a, b in ((["Istanbul", "Ankara", "Mersin"], ["Mersin", "Ankara", "Istanbul"]),):
        oa, _ = m.dikkat(dizi(a)); ob, _ = m.dikkat(dizi(b))
        f = (oa - ob).norm()
        print(f"  {' '.join(a)} / ters   fark {f:.4f}")
        assert f > 1e-6, "SIRA KOR"
    print("  ayni sehirler ters sirada FARKLI  GECTI")

    print("\nAYNI SEHIR IKI KEZ")
    yol = ["Ankara", "Mersin", "Ankara", "Sivas"]
    _o, ag = m.dikkat(dizi(yol))
    i0, i2 = [i for i, a in enumerate(yol) if a == "Ankara"]
    f = abs(float(ag[i0] - ag[i2]))
    for i, (a, x) in enumerate(zip(yol, ag)):
        print(f"  yuva {i} {a:<10s} {float(x):.4f}")
    assert f > 1e-6, "iki kopya AYIRT EDILMIYOR"
    print(f"  iki Ankara farki {f:.6f}   GECTI")
