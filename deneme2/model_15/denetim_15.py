"""Yazdigimiz sey calisiyor mu.  Egitim YOK -- yalniz mekanizma."""
import torch
from veri_15 import AD, N, ARTI, ESIT
from model_15 import Yol

ix = lambda *a: torch.tensor(list(a))
m = Yol(N, tohum=0)

print("PARAMETRE")
p = dict(m.named_parameters())
for ad, t in p.items():
    print(f"  {ad:<3s} {str(tuple(t.shape)):<14s} {t.numel():5d}")
print(f"  toplam {sum(t.numel() for t in p.values())}")

with torch.no_grad():
    print()
    print("UZUNLUK SERBEST -- ayni matrislerle farkli uzunluklar")
    for a in ([3, 5], [3, 5, 9], [3, 5, 9, 2, 7]):
        w = ix(a[0], *[t for x in a[1:] for t in (ARTI, x)], ESIT)
        o, ag = m.dikkat(w)
        print(f"  {' '.join(AD[i] for i in w.tolist()):<26s}"
              f" {len(w)} token -> {ag.shape[-1]} agirlik -> {AD[m.oku(o)]}")

    print("\nSIRA  --  OPERAND sirasi onemsiz, SAYI ICINDEKI sira ONEMLI")
    from veri_15 import rak
    cik = lambda a, b: m.dikkat(
        torch.tensor(rak(a, 3) + [ARTI] + rak(b, 3) + [ESIT]))[0]
    f1 = float((cik(53, 65) - cik(65, 53)).norm())
    f2 = float((cik(53, 65) - cik(35, 65)).norm())
    print(f"  53+65 / 65+53   fark {f1:.6f}   cevap AYNI   (118 = 118)")
    print(f"  53+65 / 35+65   fark {f2:.6f}   cevap FARKLI (118 != 100)")
    assert f2 > 1e-6, "SAYI ICINDEKI SIRA GORUNMUYOR -- yol bir TORBA"
    print("  GECTI")
    print("\nAYNI TOKEN IKI YERDE  --  ayri PUAN aliyor mu")
    print("  Agirliga degil PUANA bakilir -- relu ikisini de sifirlamis")
    print("  olabilir; o zaman fark gorunmez ama mekanizma calisiyordur.")
    w = ix(4, ARTI, 7, ARTI, 4, ESIT)
    Y = m.yol(w[None])[0]
    q = Y[-1] @ m.Wq
    pu = (Y @ m.Wk) @ q + m.hb
    y = [i for i, t in enumerate(w.tolist()) if t == 4]
    for i, t in enumerate(w.tolist()):
        print(f"  yuva {i} {AD[t]:<3s} puan {float(pu[i]):+8.3f}")
    d = abs(float(pu[y[0]] - pu[y[1]]))
    assert d > 1e-6, "iki kopya AYIRT EDILMIYOR"
    print(f"  iki 4 puan farki {d:.6f}   GECTI")