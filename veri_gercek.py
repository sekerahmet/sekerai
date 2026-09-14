# -*- coding: utf-8 -*-
"""VERI_GERCEK — elle tasarlanmis TIPLI bilgi grafigi + kendi sozlugumuz.

Mevcut `sifirdan.build_data()` RASTGELE graf kuruyor: her varligin her
iliskisi var, hedefi rastgele. O kurulumda KISAYOL f(e,r2) HER ZAMAN bir
cevap uretebiliyor.

Burada graf TIPLI. Sonucu su:

    ULKE komsu baskent   -> kisayol "ULKE baskent"  VAR   (olgu mevcut)
    SEHIR ulke baskent   -> kisayol "SEHIR baskent" YOK   (sehrin baskenti olmaz)

Yani deneyin ICINE GOMULU kontrol: ayni modelde, kisayol hedefi OLAN
zincirlerle OLMAYANLARI ayri ayri olceriz.

TASARIM KARARLARI (13 Eylul):
  * SAYISAL YAPRAK YOK (nufus/yuzolcum cikarildi). 2.160 sayi varligi
    sozlugun yarisiydi ve hicbirinin iliskisi yoktu -> model sozlugun
    yarisini "yalniz cevap olarak cikan sey" diye ogrenirdi. Eski
    kurulumda bu asimetri yoktu.
  * PARA BIRIMI cikarildi: yapraktı, ve "sira" iliskisi uydurmaydi.
  * HER TIP hem KAYNAK hem HEDEF. Boylece sozluk simetrik kullanilir.
  * Tip buyuklukleri yakin tutuldu; 6 degerli KITA gibi bir tip, cevap
    uzayi 1.200 olan bir tiple ayni modelde kiyaslanamaz sayi uretir.

Sozluk kelime duzeyinde ve aciktir; her varlik/iliski TEK token.
Dil bilgisi yok: "Turkiye baskent Ankara" yeterli.

    python veri_gercek.py [--olgu 10000] [--dok cikti.txt]
"""
import argparse
import numpy as np

# --------------------------------------------------------------- SEMA
# iliski -> (kaynak tipi, hedef tipi).  HER tip hem kaynak hem hedef.
# TIP ORTUSMESI KASITLI: bir iliski BIRDEN COK tipten cikabilir (sehrin de
# dili var, ulkenin de bolgesi var). Ortusme olmazsa kisayollu zincir cok
# azalir -- ilk tasarimda %17'ye dusmustu, gomulu kontrol ise yaramaz olur.
SEMA = {
    "baskent": (("ULKE",),                   "SEHIR"),  # ulkenin baskenti
    "komsu":   (("ULKE",),                   "ULKE"),   # ulkenin komsusu
    "bolgesi": (("ULKE", "SEHIR"),           "BOLGE"),  # ulkenin/sehrin bolgesi
    "dili":    (("ULKE", "SEHIR", "BOLGE"),  "DIL"),    # ulkenin/sehrin/bolgenin dili
    "ulkesi":  (("SEHIR", "BOLGE"),          "ULKE"),   # sehrin/bolgenin ulkesi
    "merkezi": (("BOLGE",),                  "SEHIR"),  # bolgenin merkez sehri
    "yanbolge": (("BOLGE",),                 "BOLGE"),  # bitisik bolge
    "kokeni":  (("DIL",),                    "DIL"),    # dilin koken dili
}
ILISKI = list(SEMA)
TIPLER = ["ULKE", "SEHIR", "BOLGE", "DIL"]
CIKIS = {t: sum(1 for r in ILISKI if t in SEMA[r][0]) for t in TIPLER}

_KOK = {
    "ULKE": ["Turkiye", "Fransa", "Japonya", "Almanya", "Italya", "Ispanya",
             "Yunanistan", "Misir", "Hindistan", "Cin", "Rusya", "Brezilya",
             "Kanada", "Meksika", "Peru", "Sili", "Kenya", "Fas", "Iran",
             "Irak", "Suriye", "Lubnan", "Urdun", "Katar", "Umman", "Kibris"],
    "SEHIR": ["Ankara", "Istanbul", "Izmir", "Paris", "Lyon", "Tokyo", "Osaka",
              "Berlin", "Munih", "Roma", "Milano", "Madrid", "Atina", "Kahire",
              "Delhi", "Pekin", "Sanghay", "Moskova", "Brasilia", "Ottawa",
              "Lima", "Nairobi", "Rabat", "Tahran", "Bagdat", "Sam", "Beyrut"],
    "BOLGE": ["Marmara", "Ege", "Akdeniz", "Karadeniz", "Ic_Anadolu",
              "Dogu_Anadolu", "Guneydogu", "Normandiya", "Provence", "Bavyera",
              "Toskana", "Katalonya", "Endulus", "Sibirya", "Kafkasya"],
    "DIL": ["turkce", "fransizca", "japonca", "almanca", "italyanca",
            "ispanyolca", "yunanca", "arapca", "hintce", "cince", "rusca",
            "portekizce", "latince", "ingilizce", "farsca", "kurtce"],
}


def _adlar(t, n):
    k = _KOK[t]
    return list(k[:n]) if n <= len(k) else \
        list(k) + [f"{t.lower()}_{i}" for i in range(len(k), n)]


def kur(hedef_olgu=10000, tohum=0):
    rng = np.random.RandomState(tohum)
    # ULKE:SEHIR:BOLGE:DIL = 3:3:1:1 orani, olgu sayisi hedefe gore olcekli
    birim = 1
    while True:
        n = {"ULKE": 3 * birim, "SEHIR": 3 * birim,
             "BOLGE": max(4, birim), "DIL": max(4, birim)}
        if sum(CIKIS[t] * n[t] for t in TIPLER) >= hedef_olgu:
            break
        birim += 1
    ad = {t: _adlar(t, n[t]) for t in TIPLER}

    # --- SOZLUK: her varlik/iliski TEK token -----------------------------
    ozel = ["<pad>", "<soru>", "?", "<son>"]
    varliklar = [(t, a) for t in TIPLER for a in ad[t]]
    sozluk = ozel + ILISKI + [a for _, a in varliklar]
    tip = {a: t for t, a in varliklar}

    # --- OLGULAR: TIP KURALINA GORE --------------------------------------
    olgu = {}
    for r, (kts, ht) in SEMA.items():
        for kt in kts:                          # iliski BIRDEN COK tipten cikabilir
            for s in ad[kt]:
                h = ad[ht][rng.randint(len(ad[ht]))]
                while h == s:                   # kendine gitmesin
                    h = ad[ht][rng.randint(len(ad[ht]))]
                olgu[(s, r)] = h
    return dict(ad=ad, n=n, sozluk=sozluk, kim={s: i for i, s in enumerate(sozluk)},
                tip=tip, olgu=olgu, sema=SEMA, iliski=ILISKI)


def zincirler(G):
    """Tip olarak GECERLI 2 adimli zincirler + kisayol hedefi var mi."""
    out = []
    for (e, r1), b in G["olgu"].items():
        for r2 in G["iliski"]:
            if r2 == r1:
                # r1 == r2 ise KISAYOL ile KOPRU AYNI varlik olur ve
                # siniflandirma bozulur. Eski sentetik kurulum da disliyordu.
                continue
            if G["tip"][b] not in G["sema"][r2][0]:    # ikinci hop gecersiz
                continue
            out.append((e, r1, r2, b, G["olgu"][(b, r2)], G["olgu"].get((e, r2))))
    return out


def yaz(G, z, f=print):
    ks_var = [x for x in z if x[5] is not None]
    ks_yok = [x for x in z if x[5] is None]
    f("=" * 76)
    f("TIPLI BILGI GRAFIGI")
    f("=" * 76)
    f("  tip buyuklukleri : " + "  ".join(f"{k}={v}" for k, v in G["n"].items()))
    f(f"  SOZLUK           : {len(G['sozluk'])} token  "
      f"({len(G['iliski'])} iliski + {sum(G['n'].values())} varlik + 4 ozel)")
    f(f"  ATOMIK OLGU      : {len(G['olgu'])}")
    f(f"  2-ADIMLI ZINCIR  : {len(z)}")
    f(f"     kisayol hedefi VAR : {len(ks_var):6d}  ({100*len(ks_var)/len(z):.0f}%)")
    f(f"     kisayol hedefi YOK : {len(ks_yok):6d}  ({100*len(ks_yok)/len(z):.0f}%)")

    f("\n--- SEMA ---")
    for r, (kts, ht) in G["sema"].items():
        f(f"   {r:10s} {'|'.join(kts):18s} -> {ht}")

    f("\n--- ZINCIR TURLERI (r1 r2 -> kisayol var mi) ---")
    tur = {}
    for e, r1, r2, b, a, ks in z:
        tur.setdefault((r1, r2), [0, 0])[0 if ks is not None else 1] += 1
    for (r1, r2), (v, y) in sorted(tur.items()):
        f(f"   {r1:10s} {r2:10s}  {v+y:5d} zincir   "
          + ("KISAYOL VAR" if v else "kisayol YOK"))

    f("\n--- ATOMIK OLGULAR ---")
    for (e, r), h in list(G["olgu"].items())[:10]:
        f(f"   {e:16s} {r:10s} {h}")

    f("\n--- ZINCIR, KISAYOL HEDEFI *VAR* ---")
    for e, r1, r2, b, a, ks in ks_var[:6]:
        f(f"   {e} {r1} {r2} ?  ->  {a}")
        f(f"       kopru: {b}   |  KISAYOL '{e} {r2}' = {ks}")

    f("\n--- ZINCIR, KISAYOL HEDEFI *YOK* ---")
    for e, r1, r2, b, a, ks in ks_yok[:6]:
        f(f"   {e} {r1} {r2} ?  ->  {a}")
        f(f"       kopru: {b}   |  KISAYOL '{e} {r2}' = YOK (tip izin vermiyor)")

    f("\n--- TOKEN DIZILIMI (kelime duzeyi, her varlik TEK token) ---")
    for e, r1, r2, b, a, ks in ks_var[:3]:
        d = ["<soru>", e, r1, r2, "?", a, "<son>"]
        f(f"   {' '.join(d)}")
        f(f"       -> {[G['kim'][t] for t in d]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--olgu", type=int, default=10000)
    ap.add_argument("--dok", default="")
    a = ap.parse_args()
    G = kur(a.olgu)
    z = zincirler(G)
    yaz(G, z)
    if a.dok:
        import io
        with io.open(a.dok, "w", encoding="utf-8", newline="") as fh:
            def f(s=""):
                fh.write(str(s) + "\n")
            yaz(G, z, f)
            f("\n\n" + "=" * 76)
            f("TUM ATOMIK OLGULAR")
            f("=" * 76)
            for (e, r), h in G["olgu"].items():
                f(f"{e}\t{r}\t{h}")
            f("\n\n" + "=" * 76)
            f("TUM 2-ADIMLI ZINCIRLER  (soru / kopru / cevap / kisayol)")
            f("=" * 76)
            for e, r1, r2, b, ans, ks in z:
                f(f"{e} {r1} {r2} ?\t-> {ans}\tkopru={b}\tkisayol={ks or 'YOK'}")
        print(f"\n-> tam dokum: {a.dok}")


if __name__ == "__main__":
    main()
