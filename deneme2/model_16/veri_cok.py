"""VERI -- COK TERIMLI toplama.  2 ve 3 terim KARISIK, dolgu yok, DUR ile biter.

Kullanici, 22 Eylul: "artik yeni veri seti olsun ama 3 adimli toplamada
su an model ne yapabilecek.  yani 1+1+5 = 7, 23+1+120 = gibi."

OLCULDU once: 2 terimle egitilmis model (tutulan 0,7770) uc terimli
400 soruda SIFIR dogru yapti.  Iki '+' gormemisti.

  1 + 1 + 5 =          ->  7
  1 + 1 + 5 = 7        ->  DUR

  2 3 + 1 + 1 2 0 =    ->  1
  2 3 + 1 + 1 2 0 = 1  ->  4
  ...                  ->  4
  ... = 1 4 4          ->  DUR

Tasarimin iddiasi "yol uzayabilir" idi; bu onu dogrudan sinar.  Model
degismiyor -- E, b, M, s0, Wq/Wk/Wv, hb; hicbiri terim sayisina bagli degil.

EVREN ARTIK SAYILAMIYOR.  Iki terimde 251.000 ciftin HEPSINI kullaniyorduk.
Uc terimde 501^3 = 125.751.501 uclu var; ORNEKLENIYOR.  Bu ilk kez oluyor
ve ornekleme BELIRLENIMCI: ayni tohum ayni kumeyi verir, iz dosyaya yazilir.
"""
import torch

ENB = 500                   # her terim 0..500
ARTI, ESIT, DUR = 10, 11, 12
N = 13
AD = [str(i) for i in range(10)] + ["+", "=", "DUR"]

# Uc terimde en buyuk toplam 1500 -> 4 haneli cevaplar var.  Kisilmadi:
# DUR mekanizmasi icin fazladan bir sinav, ve dogal olan bu.
ORAN = {1: 1.00, 2: 0.75, 3: 0.50, 4: 0.50}

UCLU = 251_000              # kac uclu ORNEKLENECEK (iki terimlinin sayisi
                            # kadar -- iki yari denk olsun diye)

hane = lambda x: len(str(x))


def rak(x):
    """Sayinin KENDI rakamlari.  Dolgu YOK."""
    return [int(c) for c in str(x)]


def soru(ts):
    """ts: terimler.  ->  d..d + d..d [+ d..d] ="""
    w = []
    for i, x in enumerate(ts):
        if i:
            w.append(ARTI)
        w += rak(x)
    return w + [ESIT]


def ikililer():
    """Iki terimli evren -- ONCEKI SINAVIN TA KENDISI.

    500+500=1000 orada cikarilmisti (tek basina 4 haneli cevap veren tek
    cift); burada da cikariliyor ki iki terimli yari BIREBIR ayni kalsin
    ve 0,7770 ile dogrudan kiyaslanabilsin."""
    return [(a, b) for a in range(ENB + 1) for b in range(ENB + 1)
            if a + b < 1000]


def ucluler(tohum=0):
    """ORNEKLENMIS ucluler.  Evren 501^3 = 125.751.501, sayilamaz."""
    g = torch.Generator().manual_seed(tohum)
    uc, gor = [], set()
    while len(uc) < UCLU:                       # tekrari at, evren cok buyuk
        t = torch.randint(0, ENB + 1, (UCLU, 3), generator=g)
        for r in t.tolist():
            k = tuple(r)
            if k not in gor:
                gor.add(k); uc.append(k)
                if len(uc) == UCLU:
                    break
    return uc


def sorular(tohum=0):
    return ikililer() + ucluler(tohum)


def ornekler(sorlar):
    """Her soru -> hane(toplam)+1 ornek; sonuncusunun hedefi DUR.

    Doner: obek basina (w, h, u).  u = CEVABIN hane sayisi.
    Obek = GIRDI UZUNLUGU, yalnizca tensor sekli icin; mufredat DEGIL."""
    g = {}
    for ts in sorlar:
        q, c = soru(ts), rak(sum(ts))
        for i in range(len(c) + 1):
            w = q + c[:i]
            h = c[i] if i < len(c) else DUR
            g.setdefault(len(w), ([], [], []))
            g[len(w)][0].append(w)
            g[len(w)][1].append(h)
            g[len(w)][2].append(len(c))
    return [(torch.tensor(w), torch.tensor(h), torch.tensor(u))
            for _, (w, h, u) in sorted(g.items())]


def _bol(sorlar, tohum):
    """Cevabin hane sayisina gore ORAN kadarini egitime ayir.
    SORU bazinda -- bir sorunun butun ornekleri ayni tarafta."""
    g = torch.Generator().manual_seed(tohum)
    kova = {}
    for ts in sorlar:
        kova.setdefault(hane(sum(ts)), []).append(ts)
    eg, tu = [], []
    for h in sorted(kova):
        c = kova[h]
        k = torch.randperm(len(c), generator=g)
        n = round(len(c) * ORAN[h])
        eg += [c[j] for j in k[:n].tolist()]
        tu += [c[j] for j in k[n:].tolist()]
    return eg, tu


def bol(tohum=0):
    """IKILILER ve UCLULER AYRI bolunur.

    Ikili bolme, onceki sinavin bolmesiyle BIREBIR ayni cikar (ayni evren,
    ayni oran, ayni tohum).  Boylece iki terimli yaridaki sonuc 0,7770 ile
    dogrudan kiyaslanir.  Birlesik havuzda bolunseydi ikililer kayardi."""
    ie, it = _bol(ikililer(), tohum)
    ue, ut = _bol(ucluler(tohum), tohum + 1)
    return ie + ue, it + ut


def sor(m, sorlar, aygit="cuda", en=20000, parca=20000, ayrinti=False):
    """TEK ANALIZ: soruyu sor, cevabi al, DOGRU MU.

    Serbest uretim -- her adimin CIKTISI bir sonraki adimin GIRDISI.
    DUR'a kadar uretilen dizi dogru cevaba BIREBIR esit olmali:
      erken durmak  YANLIS    fazla rakam  YANLIS    rakam yanlis  YANLIS

    ayrinti=True -> (SAYI, UZUNLUK).  UZUNLUK = rakamlar ne olursa olsun
    DOGRU YERDE durdu mu."""
    if en < len(sorlar):                        # ALT KUME RASTGELE, tohum sabit
        g = torch.Generator().manual_seed(12345)
        k = torch.randperm(len(sorlar), generator=g)[:en].tolist()
        sorlar = [sorlar[j] for j in k]

    kova = {}
    for ts in sorlar:
        q = soru(ts)
        kova.setdefault(len(q), []).append((q, rak(sum(ts))))
    K = max(hane(sum(ts)) for ts in sorlar) + 1

    dog = uzn = say = 0
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

                yol, cik = W, []
                for _ in range(K):
                    o, _ag = m.dikkat(yol)
                    t = (-((m.E[None] - o[:, None]) ** 2).sum(-1)).argmax(-1)
                    cik.append(t)
                    yol = torch.cat([yol, t[:, None]], 1)
                C = torch.stack(cik, 1)

                var = C == DUR
                ilk = torch.where(var.any(1), var.float().argmax(1),
                                  torch.full_like(U, K))
                p = torch.arange(K, device=aygit)
                gec = p[None] < U[:, None]
                dog += int((((C == T) | ~gec).all(1) & (ilk == U)).sum())
                uzn += int((ilk == U).sum())
                say += B
    return (dog / say, uzn / say) if ayrinti else dog / say


def kirilim(m, sorlar, aygit="cuda", olcu="terim"):
    """SONUC istatistigi: {anahtar: (SAYI, UZUNLUK, n)}

    olcu="terim"  kac terimli soru (2 / 3)   <- asil soru bu
    olcu="hane"   cevap kac haneli (1..4)
    Yuva/basamak kirilimi DEGIL -- yine "cevap dogru mu"."""
    f = (lambda ts: len(ts)) if olcu == "terim" else (lambda ts: hane(sum(ts)))
    g = {}
    for ts in sorlar:
        g.setdefault(f(ts), []).append(ts)
    return {a: (*sor(m, c, aygit=aygit, en=10**9, ayrinti=True), len(c))
            for a, c in sorted(g.items())}


def yaz(yol="veri_cok.pt", tohum=0):
    """Bolmeyi DOSYAYA yaz.  Colab bunu Drive'dan okur, uretmez (kural 9)."""
    EG, TU = bol(tohum)
    d = dict(ENB=ENB, N=N, ARTI=ARTI, ESIT=ESIT, DUR=DUR, AD=AD,
             ORAN=ORAN, UCLU=UCLU, tohum=tohum, soru_eg=EG, soru_tu=TU,
             eg=ornekler(EG), tu=ornekler(TU))
    torch.save(d, yol)
    return d


def _ozet():
    from collections import Counter
    EG, TU = bol()
    E = set(EG)
    hepsi = sorular()
    print(f"sozluk {N}   " + " ".join(AD))
    print(f"terimler 0..{ENB}   soru {len(hepsi)}"
          f"   (ikili {len(hepsi)-UCLU}  uclu {UCLU})")
    print()
    print("  cevap    teorik   egitim  tutulan    ORAN   egitim ORNEGI")
    c = Counter(hane(sum(t)) for t in hepsi)
    ce = Counter(hane(sum(t)) for t in EG)
    for h in sorted(c):
        print(f"  {h} hane  {c[h]:8d} {ce[h]:8d} {c[h]-ce[h]:8d}"
              f"   %{100*ce[h]/c[h]:5.1f}   {ce[h]*(h+1):8d}")
    print()
    print("  terim    teorik   egitim  tutulan")
    t_ = Counter(len(t) for t in hepsi); te = Counter(len(t) for t in EG)
    for k in sorted(t_):
        print(f"  {k} terim {t_[k]:8d} {te[k]:8d} {t_[k]-te[k]:8d}")
    no = sum(hane(sum(t)) + 1 for t in EG)
    nt = sum(hane(sum(t)) + 1 for t in TU)
    print(f"\n  ORNEK  egitim {no}   tutulan {nt}")
    print()
    print("ORNEKLER")
    for ts in [(1, 1, 5), (23, 1, 120), (472, 182), (500, 500, 500)]:
        q, c = soru(ts), rak(sum(ts))
        for i in range(len(c) + 1):
            print("  " + " ".join(AD[t] for t in q + c[:i]) + "   ->  "
                  + (AD[c[i]] if i < len(c) else "DUR"))
        print()


if __name__ == "__main__":
    import sys
    if "--yaz" in sys.argv:
        import hashlib, os
        d = yaz()
        hh = hashlib.sha256()
        for w, _, u in d["eg"] + d["tu"]:
            hh.update(w.numpy().tobytes())
            hh.update(u.numpy().tobytes())
        print(f"veri_cok.pt yazildi   "
              f"{os.path.getsize('veri_cok.pt')/1e6:.1f} MB")
        print(f"  soru  egitim {len(d['soru_eg'])}   tutulan {len(d['soru_tu'])}")
        print(f"  ornek egitim {sum(len(h) for _, h, _ in d['eg'])}"
              f"   tutulan {sum(len(h) for _, h, _ in d['tu'])}")
        print("  obekler: " + "  ".join(
            f"{w.shape[1]}tk:{len(h)}" for w, h, _ in d["eg"]))
        print(f"  iz {hh.hexdigest()[:16]}")
    else:
        _ozet()
