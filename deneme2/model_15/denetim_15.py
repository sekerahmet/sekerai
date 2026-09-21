"""Yazdigimiz sey calisiyor mu.  Egitim YOK -- yalniz mekanizma."""
import torch
from veri_15 import AD, N, ARTI, ESIT
from model_15 import Yol

ix = lambda *a: torch.tensor(list(a))
m = Yol(N, boyut=2, durum=16, tohum=0)

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

    print()
    print("SIRA GORUNUYOR MU  (toplamada cevap ayni ama DURUM ayrilmali)")
    a = ix(3, ARTI, 5, ARTI, 9, ESIT)
    b = ix(9, ARTI, 5, ARTI, 3, ESIT)
    f = (m.dikkat(a)[0] - m.dikkat(b)[0]).norm()
    print(f"  3+5+9= / 9+5+3=   cikti farki {f:.4f}")
    assert f > 1e-6, "SIRA KOR"
    print("  GECTI")

    print()
    print("AYNI TOKEN IKI KEZ  ayri agirlik aliyor mu")
    w = ix(4, ARTI, 7, ARTI, 4, ESIT)
    _o, ag = m.dikkat(w)
    ag = ag.sum(0)                     # kafalar toplanir
    y = [i for i, t in enumerate(w.tolist()) if t == 4]
    for i, (t, x) in enumerate(zip(w.tolist(), ag)):
        print(f"  yuva {i} {AD[t]:<3s} {float(x):.4f}")
    d = abs(float(ag[y[0]] - ag[y[1]]))
    assert d > 1e-6, "iki kopya AYIRT EDILMIYOR"
    print(f"  iki '4' farki {d:.6f}   GECTI")
