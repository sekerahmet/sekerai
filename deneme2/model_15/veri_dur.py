"""VERI -- DOLGU YOK.  Sayi kendi uzunluguyla yazilir, cevap DUR ile biter.

Kullanici, 22 Eylul: "artik bosluk doldurma yok.  pad kullanmadan
1+1 = 2 diye ogret.  21+23 = 44, yani uzunluklar farkli olacak."

Bir onceki sinavda her sey SABIT GENISLIKTI (_ _ 1 + _ _ 1 = _ _ _ 2) ve
"cevap 4 rakam" disaridan biliniyordu.  Burada bilinmiyor:

  1 + 1 =            ->  2
  1 + 1 = 2          ->  DUR

  2 1 + 2 3 =        ->  4
  2 1 + 2 3 = 4      ->  4
  2 1 + 2 3 = 4 4    ->  DUR

Bir cift  hane(a+b) + 1  ornek verir.  Girdi uzunlugu da cikti uzunlugu da
degisken; model NEREDE DURACAGINI da ogrenmek zorunda.  Fazladan rakam
uretmek de erken durmak da YANLIS.

Sozluk boyu degismedi (13): PAD'in yeri DUR'a gecti.
"""
import torch

ENB = 500                   # toplananlar 0..500
ARTI, ESIT, DUR = 10, 11, 12
N = 13
AD = [str(i) for i in range(10)] + ["+", "=", "DUR"]

# OGRETME ORANI, CEVABIN HANE SAYISINA gore.  Kullanici karari, 22 Eylul:
#   "tek haneli toplamlarin HEPSINI ogret, iki hanelilerin %75'ini,
#    uc hanelilerin %50'sini."
# Gerekce: evren carpik.  a,b duzgun 0..500 secilince toplam 500 civarina
# yigiliyor ve tek haneli toplam yalnizca 55 cift.  Onceki sinavda bunun
# yarisi egitime dusmustu ve cevabi 1 olan HIC ornek yoktu.
#
ORAN = {1: 1.00, 2: 0.75, 3: 0.50}

# 500+500=1000 CIKARILDI (kullanici, 22 Eylul: "500+500 kaldir").
# Tek basina 4 haneli cevap veren TEK cift oydu; ne ogretilebilir ne
# olculebilir bir sinifti, 1 ornek.
HEPSI = [(a, b) for a in range(ENB + 1) for b in range(ENB + 1)
         if a + b < 1000]

hane = lambda x: len(str(x))


def rak(x):
    """Sayinin KENDI rakamlari.  Dolgu YOK."""
    return [int(c) for c in str(x)]


def soru(a, b):
    return rak(a) + [ARTI] + rak(b) + [ESIT]


def ornekler(ciftler):
    """Her cift -> hane(a+b)+1 ornek; sonuncusunun hedefi DUR.

    Doner: obek basina (w, h, u).
      w  girdi
      h  hedef token
      u  CEVABIN kac haneli oldugu -- cekiliste sinif olarak kullaniliyor.

    Girdi uzunlugu hem toplananlarin hem uretilmis onekin uzunluguna gore
    degisiyor, o yuzden UZUNLUGA GORE gruplanip yiginlaniyor.  Obek bir
    MUFREDAT DEGIL, yalnizca tensor sekli; cekilis obekten bagimsiz."""
    g = {}
    for a, b in ciftler:
        q, c = soru(a, b), rak(a + b)
        for i in range(len(c) + 1):
            w = q + c[:i]
            h = c[i] if i < len(c) else DUR
            g.setdefault(len(w), ([], [], []))
            g[len(w)][0].append(w)
            g[len(w)][1].append(h)
            g[len(w)][2].append(len(c))
    return [(torch.tensor(w), torch.tensor(h), torch.tensor(u))
            for _, (w, h, u) in sorted(g.items())]


def bol(tohum=0):
    """CIFT bazinda bolme -- bir ciftin butun ornekleri ayni tarafta.

    Oran SABIT degil, cevabin hane sayisina gore (ORAN)."""
    g = torch.Generator().manual_seed(tohum)
    kova = {}
    for a, b in HEPSI:
        kova.setdefault(hane(a + b), []).append((a, b))
    eg, tu = [], []
    for h in sorted(kova):
        c = kova[h]
        k = torch.randperm(len(c), generator=g)
        n = round(len(c) * ORAN[h])
        eg += [c[j] for j in k[:n].tolist()]
        tu += [c[j] for j in k[n:].tolist()]
    return eg, tu


def sor(m, ciftler, aygit="cuda", en=20000, parca=20000):
    """TEK ANALIZ: soruyu sor, cevabi al, DOGRU MU.

    Kullanici, 22 Eylul: "yuva 1 hic bir zaman olcu olmadi, o bir analizdi.
    burda tek analiz var: soru sorduk, cevap dogru mu yanlis mi, bu kadar."

    Serbest uretim -- her adimin CIKTISI bir sonraki adimin GIRDISI.
    DUR'a kadar uretilen dizi dogru cevaba BIREBIR esit olmali:
      erken durmak     YANLIS
      fazla rakam      YANLIS
      rakam yanlis     YANLIS
    """
    kova = {}
    for a, b in ciftler[:en]:
        q = soru(a, b)
        kova.setdefault(len(q), []).append((q, rak(a + b)))

    K = max(len(c) for c in map(rak, (a + b for a, b in ciftler[:en]))) + 1
    dog = say = 0
    with torch.no_grad():
        for kalem in kova.values():
            for i in range(0, len(kalem), parca):
                oh = kalem[i:i + parca]
                W = torch.tensor([q for q, _ in oh], device=aygit)
                B = len(oh)
                U = torch.tensor([len(c) for _, c in oh], device=aygit)
                T = torch.full((B, K), -1, dtype=torch.long, device=aygit)
                for r, (_, c) in enumerate(oh):
                    T[r, :len(c)] = torch.tensor(c, device=aygit)

                yol = W
                cik = []
                for _ in range(K):
                    o, _ag = m.dikkat(yol)
                    t = (-((m.E[None] - o[:, None]) ** 2).sum(-1)).argmax(-1)
                    cik.append(t)
                    yol = torch.cat([yol, t[:, None]], 1)
                C = torch.stack(cik, 1)                      # (B, K)

                var = C == DUR
                ilk = torch.where(var.any(1), var.float().argmax(1),
                                  torch.full_like(U, K))     # ilk DUR nerede
                p = torch.arange(K, device=aygit)
                gec = p[None] < U[:, None]                   # rakam olan yerler
                dog += int((((C == T) | ~gec).all(1) & (ilk == U)).sum())
                say += B
    return dog / say


def yaz(yol="veri_dur.pt", tohum=0):
    """Bolmeyi DOSYAYA yaz.  Colab bunu Drive'dan okur, uretmez (kural 9)."""
    EG, TU = bol(tohum)
    d = dict(ENB=ENB, N=N, ARTI=ARTI, ESIT=ESIT, DUR=DUR, AD=AD,
             ORAN=ORAN, tohum=tohum, cift_eg=EG, cift_tu=TU,
             eg=ornekler(EG), tu=ornekler(TU))
    torch.save(d, yol)
    return d


def _ozet():
    EG, TU = bol()
    E = set(EG)
    print(f"sozluk {N}   " + " ".join(AD))
    print(f"toplananlar 0..{ENB}   cift {len(HEPSI)}")
    print()
    print("  cevap    teorik   egitim  tutulan    ORAN   egitim ORNEGI")
    for h in sorted(ORAN):
        c = [ab for ab in HEPSI if hane(ab[0] + ab[1]) == h]
        ne = sum(1 for ab in c if ab in E)
        print(f"  {h} hane  {len(c):8d} {ne:8d} {len(c)-ne:8d}"
              f"   %{100*ne/len(c):5.1f}   {ne*(h+1):8d}")
    no = sum(1 for ab in EG for _ in range(hane(ab[0]+ab[1]) + 1))
    nt = sum(1 for ab in TU for _ in range(hane(ab[0]+ab[1]) + 1))
    print(f"  TOPLAM   {len(HEPSI):8d} {len(EG):8d} {len(TU):8d}"
          f"            {no:8d}  (tutulan {nt})")
    print()
    print("ORNEKLER")
    for ab in [(1, 1), (21, 23), (472, 182)]:
        q, c = soru(*ab), rak(sum(ab))
        for i in range(len(c) + 1):
            h = AD[c[i]] if i < len(c) else "DUR"
            print("  " + " ".join(AD[t] for t in q + c[:i]) + "   ->  " + h)
        print()


if __name__ == "__main__":
    import sys
    if "--yaz" in sys.argv:
        import hashlib, os
        d = yaz()
        hh = hashlib.sha256()
        # IZ, girdilerin YANINDA sinif alanini da kapsar -- alan eklenince
        # dosya degisip iz ayni kalmasin diye.
        for w, _, u in d["eg"] + d["tu"]:
            hh.update(w.numpy().tobytes())
            hh.update(u.numpy().tobytes())
        print(f"veri_dur.pt yazildi   "
              f"{os.path.getsize('veri_dur.pt')/1e6:.1f} MB")
        print(f"  egitim {sum(len(h) for _, h, _ in d['eg'])}"
              f"   tutulan {sum(len(h) for _, h, _ in d['tu'])}")
        print("  obekler: " + "  ".join(
            f"{w.shape[1]}tk:{len(h)}" for w, h, _ in d["eg"]))
        print(f"  iz {hh.hexdigest()[:16]}")
    else:
        _ozet()
