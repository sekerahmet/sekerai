"""veri_15 uzerinde egitim.  Cesitli uzunluk, ikinci dereceden kural."""
import math
import torch
import torch.nn.functional as F
from sehir_15 import AD, IX, N, DUR
from model_15 import Yol
from veri_15 import ORNEK

# 80 / 20 -- TUTULAN kume egitimde HIC gorulmuyor.
_g = torch.Generator().manual_seed(1)
_k = torch.randperm(len(ORNEK), generator=_g)
_b = int(len(ORNEK) * 0.8)
EGIT = [ORNEK[i] for i in _k[:_b]]
TUT  = [ORNEK[i] for i in _k[_b:]]
W  = [(torch.tensor(o), h) for o, h in EGIT]
WT = [(torch.tensor(o), h) for o, h in TUT]
m = Yol(N, boyut=2, durum=6, dur=IX[DUR], tohum=0)
opt = torch.optim.Adam(m.parameters(), lr=0.02)


def gec(egit, kume=None):
    kume = W if kume is None else kume
    t, d = 0.0, 0
    for w, h in kume:
        o, _ = m.dikkat(w)
        puan = -((m.E - o) ** 2).sum(-1)
        t = t + F.cross_entropy(puan[None], torch.tensor([h]))
        d += int(puan.argmax()) == h
    k = t / len(kume)
    if egit:
        opt.zero_grad(); k.backward(); opt.step()
    return k.item(), d / len(kume)


print(f"egitim {len(W)}   tutulan {len(WT)}   sans kaybi {math.log(N):.4f}")
print("adim    EGITIM kayip  dogruluk   TUTULAN kayip  dogruluk")
for i in range(1201):
    k, d = gec(True)
    if i % 150 == 0:
        with torch.no_grad():
            kt, dt = gec(False, WT)
        print(f"{i:4d}      {k:.4f}     {d:.3f}        {kt:.4f}     {dt:.3f}")

with torch.no_grad():
    k, d = gec(False)
    print(f"\nSON   kayip {k:.4f}   dogruluk {d:.3f}")

    print("\nUZUNLUGA GORE DOGRULUK")
    print("  (TUTULAN kume)")
    for L in sorted({len(o) for o, _ in TUT}):
        alt = [(w, h) for (w, h), (o, _) in zip(WT, TUT) if len(o) == L]
        dd = sum(int((-((m.E - m.dikkat(w)[0]) ** 2).sum(-1)).argmax()) == h
                 for w, h in alt)
        print(f"  girdi {L} sehir   {dd}/{len(alt)}   {dd/len(alt):.3f}")

    print("\nDIKKAT NEREYE BAKIYOR  -- butun orneklerde")
    print("  geriden kacinci yuva   toplam agirlik")
    pay = {}
    for w, _h in W:
        _o, ag = m.dikkat(w)
        for j, a in enumerate(ag):
            pay[len(ag) - 1 - j] = pay.get(len(ag) - 1 - j, 0.0) + float(a)
    top = sum(pay.values())
    for gg in sorted(pay):
        ad = "SON" if gg == 0 else str(gg) + " geri"
        print(f"  {ad:<10s} {pay[gg]/top:.3f}  {chr(35) * int(pay[gg]/top*50)}")

    print("\nBIR ORNEK")
    w, h = max(W, key=lambda x: len(x[0]))
    o, ag = m.dikkat(w)
    for i, (t, a) in enumerate(zip(w.tolist(), ag)):
        print(f"  yuva {i} {AD[t]:<10s} {float(a):.3f}  {'#' * int(a * 50)}")
    print(f"  -> {AD[int((-((m.E - o) ** 2).sum(-1)).argmax())]}   hedef {AD[h]}")
