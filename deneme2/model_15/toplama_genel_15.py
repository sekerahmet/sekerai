"""GENELLEME: hic gosterilmemis toplamlar.

Bolme DIZI bazinda -- bir dizinin butun onekleri ayni tarafta, sizinti yok.
  TAVAN 1.000   toplama cikarilabilir bir kural, her soru BILINEBILIR
  TABAN 0.033   en sik cevap
"""
import statistics
from collections import Counter
import torch
import torch.nn.functional as F
from model_15 import Yol
from toplama_15 import AD, N, DIZI, islem, ix

ADIM, TOHUM = 300, 1


def bol(t):
    g = torch.Generator().manual_seed(t)
    k = torch.randperm(len(DIZI), generator=g)
    b = int(len(DIZI) * 0.8)
    yap = lambda I: [(islem(a[:i]), ix(sum(a[:i])))
                     for j in I for a in [DIZI[j]] for i in range(2, len(a) + 1)]
    return yap(k[:b].tolist()), yap(k[b:].tolist())


def obek(S):
    d = {}
    for o, h in S:
        d.setdefault(len(o), ([], []))
        d[len(o)][0].append(o); d[len(o)][1].append(h)
    return [(torch.tensor(w), torch.tensor(h)) for w, h in d.values()]


def gec(m, O, opt=None):
    kay, dog, say = 0.0, 0, 0
    for w, h in O:
        o, _ = m.dikkat(w)
        puan = -((m.E[None] - o[:, None]) ** 2).sum(-1)
        kay = kay + F.cross_entropy(puan, h, reduction="sum")
        dog += int((puan.argmax(-1) == h).sum()); say += len(h)
    k = kay / say
    if opt is not None:
        opt.zero_grad(); k.backward(); opt.step()
    return k.item(), dog / say


def kos(durum, wd, tohum):
    EG, TU = bol(tohum)
    OE, OT = obek(EG), obek(TU)
    m = Yol(N, boyut=8, durum=durum, tohum=tohum, norm=True)
    opt = torch.optim.Adam(m.parameters(), lr=0.02, weight_decay=wd)
    for i in range(ADIM):
        gec(m, OE, opt)
        if i % 100 == 0:                       # ilerleme BASILIR (kural 8)
            with torch.no_grad():
                _, de = gec(m, OE); _, dt = gec(m, OT)
            print(f"      adim {i:4d}   egitim {de:.3f}   tutulan {dt:.3f}", flush=True)
    with torch.no_grad():
        _, de = gec(m, OE); _, dt = gec(m, OT)
    return m, de, dt, TU


EG0, TU0 = bol(0)
print(f"egitim {len(EG0)}   tutulan {len(TU0)}   TABAN "
      f"{Counter(h for _, h in TU0).most_common(1)[0][1]/len(TU0):.3f}   TAVAN 1.000")
print()
print("  durum   wd     HAFIZA(egitim)    GENELLEME(tutulan)")
son = None
for durum in (16,):
    for wd in (0.0,):
        print(f"  -- durum {durum}  wd {wd}", flush=True)
        r = [kos(durum, wd, t) for t in range(TOHUM)]
        e = [x[1] for x in r]; u = [x[2] for x in r]
        sap = lambda v: statistics.stdev(v) if len(v) > 1 else 0.0
        print(f"  {durum:5d}   {wd:<5} {statistics.mean(e):.3f} +- {sap(e):.3f}"
              f"      {statistics.mean(u):.3f} +- {sap(u):.3f}")
        son = r[0]

print()
print("HATA ile TERIM SAYISI")
print("  terim  ornek   dogru    ORT HATA (model - dogru)   |hata|")
m, _, _, TU = son
with torch.no_grad():
    d = {}
    for o, h in TU:
        w = torch.tensor(o)
        c = int((-((m.E - m.dikkat(w)[0]) ** 2).sum(-1)).argmax())
        t = (len(o) + 1) // 2
        d.setdefault(t, []).append((c - h, c == h))
    for t in sorted(d):
        v = d[t]
        hata = sum(x for x, _ in v) / len(v)
        mut = sum(abs(x) for x, _ in v) / len(v)
        dog = sum(1 for _, y in v if y) / len(v)
        print(f"  {t:5d}  {len(v):5d}   {dog:.3f}   {hata:+8.2f}"
              f"                {mut:6.2f}")
print("TUTULANDAN ORNEKLER  (son modelin cevaplari)")
m, _, _, TU = son
with torch.no_grad():
    for o, h in TU[:10]:
        w = torch.tensor(o)
        c = int((-((m.E - m.dikkat(w)[0]) ** 2).sum(-1)).argmax())
        print(f"  {' '.join(AD[i] for i in o):<26s} {AD[c]:>4s}   dogru {AD[h]:>4s}"
              f"  {'DOGRU' if c == h else 'YANLIS'}")
