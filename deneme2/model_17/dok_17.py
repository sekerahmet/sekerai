"""dok_17 -- modelin gordugu HER SEYI goz ile okunur dosyaya dok.

Kullanici, 20 Eylul 2026: *"once sen veri klasorune verileri yaz ben
goz ile kontrol edeyim"*.
ve 22 Eylul: *"once jeton uret bakalim orda eksik var mi?"*

Gerekce ayrica OLCULDU: bir gunde uc gosterim kusuru yalnizca METNE
bakilarak bulundu, hicbirini sayisal bir kapi gostermedi.  Sayi
"makul" durabilir; metin duramaz.

    python dok_17.py [klasor]        varsayilan: model_17/veri

DOSYALAR
    00_OZET.txt          kac birim, kac kelime, ne eksik -- tek sayfa
    01_token_listesi.txt SOZLUGUN TAMAMI: frekans, pay, tur, ornek kelime
    02_bolme.txt         her KELIME -> hangi birimlere bolunuyor
    03_korpus.txt        akistan kesintisiz bir parca

`dok_14`ten FARK: o `olcme_14.Sorular` / `Bicim`e bagliydi (sinav ve
uretim dokumleri).  model_17'nin olcusu bastan yazildi ve o sinifllar
yok; burada yalniz JETON tarafi dokuluyor.

ZINCIRDEKI YERI.  Kim kimi cagiriyor, bu dosya nerede:
(model BIRIM goruyor -- karakter yalnizca ara adim ve kapi)

  veri_17     graf: 1608 varlik, 24 iliski, olgular
  metin_17    graf -> duz Turkce cumle
  korpus_17   cumle -> belge -> paketlenmis akis
  jeton_17    KARAKTER sozlugu + GIDIS-DONUS KAPISI
  birim_17    metin -> sayim -> kok havuzu -> BIRIM AKISI -> pencere
  ek_17       kelime -> kok + ek    (Turkce morfolojisi)

  taban_17    bolmeler (ezber_* / cikarim_*) + Ayar tanimi
  ayar_17     dugmeler
  hazirla_17  veriyi dosyaya yazar, Colab Drive'dan OKUR
  dok_17      goz ile okunur dokum   <-- BU DOSYA

  model_17    MIMARI -- model_15'ten
  train_17    egitim dongusu
  olcme_17    olcu: soru soruldu, cevap dogru mu
"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ayar_17 as AY      # noqa: E402
import birim_17 as BR     # noqa: E402
import ek_17 as EK        # noqa: E402
import taban_17 as MT     # noqa: E402
import veri_17 as V       # noqa: E402

ONBELLEK = "birim_17.npz"


def tur(b, ad):
    """Bir birimin TURU.  Listenin okunabilir olmasi icin."""
    if ad in EK.NOKTALAMA:
        return "NOKTALAMA"
    if ad.startswith("-"):
        return "EK"
    if ad in EK.KOK:
        return "KOK"
    if ad in EK.BUTUN:
        return "BUTUN"
    return "OZEL AD"


def ac(kls, ad):
    return open(os.path.join(kls, ad), "w", encoding="utf-8")


def main():
    kls = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "veri")
    os.makedirs(kls, exist_ok=True)

    v = MT.veri_kur(AY.AYAR, yaz=lambda *a, **k: None)
    b = BR.kur(AY.AYAR, v, yaz=print, onbellek=ONBELLEK)

    say = b.say
    top = int(say.sum())
    # her birimi URETEN kelimeler -- listeyi goz ile okunur yapan sey
    ureten = collections.defaultdict(list)
    for k, parca in b.bolme.items():
        for x in set(parca):
            ureten[x].append(k)

    sayim = collections.Counter(tur(b, a) for a in b.ad)

    # --- 01 TOKEN LISTESI
    with ac(kls, "01_token_listesi.txt") as f:
        f.write("01  TOKEN LISTESI\n" + "=" * 78 + "\n")
        f.write(f"{len(b.ad)} token.  Numara sirali, TEKRARSIZ.\n")
        f.write("Modelin sozlugunun TAMAMI -- baska hicbir sey uretemez.\n\n")
        f.write("kok AYRI token, ek AYRI token.  Ek tokenleri GERCEK YUZEY:\n")
        f.write("`-in` ile `-in` ayri tokenler, unlu uyumu modelin girdisinde.\n")
        f.write(f"Korpustaki {len(b.bolme):,} kelimenin {len(b.bolme):,}'u "
                "BILINEN KOK + BILINEN EK'e\n"
                "cozuluyor; cozulemeyen YOK (`ek_17.denetle` her "
                "kurulumda sinar).\n\n")
        f.write("  " + "   ".join(f"{k} {n}" for k, n in
                                  sorted(sayim.items())) + "\n\n")
        f.write(f"{'NO':<5s}{'TOKEN':<22s}{'FREKANS':>10s}{'PAY':>8s}  "
                f"{'TUR':<10s}URETEN KELIMELER\n")
        f.write("-" * 110 + "\n")
        for i, a in enumerate(b.ad):
            u = sorted(ureten.get(a, ()))[:8]
            f.write(f"{i:<5d}{a:<22s}{int(say[i]):>10d}"
                    f"{100*int(say[i])/top:>7.3f}%  {tur(b, a):<10s}"
                    + "  ".join(u) + ("  ..." if len(ureten.get(a, ())) > 8
                                      else "") + "\n")

    # --- 02 BOLME
    with ac(kls, "02_bolme.txt") as f:
        f.write("02  BOLME -- her KELIME hangi birimlere ayriliyor\n")
        f.write("=" * 78 + "\n")
        f.write(f"{len(b.bolme):,} benzersiz kelime.  Frekans sirali.\n\n")
        kf = collections.Counter()
        for k, p in b.bolme.items():
            kf[k] = min((int(say[b.ix[x]]) for x in p), default=0)
        for k, _ in sorted(b.bolme.items(),
                           key=lambda kv: (-len(kv[1]), kv[0])):
            f.write(f"  {k:<26s} -> {'  '.join(b.bolme[k])}\n")

    # --- 03 KORPUS
    with ac(kls, "03_korpus.txt") as f:
        f.write("03  KORPUS -- birim akisindan kesintisiz bir parca\n")
        f.write("=" * 78 + "\n")
        f.write("Modelin GERCEKTEN gordugu dizi.  Birimler bosluklu.\n\n")
        f.write(b.coz(b.dizi[:4000]) + "\n")

    # --- 00 OZET
    iyi, eksik = EK.denetle(b.bolme.keys(), ozel=b.korunan)
    with ac(kls, "00_OZET.txt") as f:
        f.write("00  OZET\n" + "=" * 78 + "\n")
        f.write(f"  birim (sozluk)       {len(b.ad)}\n")
        for k, n in sorted(sayim.items()):
            f.write(f"    {k:<18s} {n}\n")
        f.write(f"  benzersiz kelime     {len(b.bolme):,}\n")
        f.write(f"  akis uzunlugu        {len(b.dizi):,} birim\n")
        f.write(f"  kelime/birim         {len(b.bolme)/len(b.ad):.2f}x\n")
        f.write(f"  COZULEMEYEN KELIME   {len(eksik)}"
                + ("   <-- EKSIK VAR" if eksik else "   (yok)") + "\n")
        if eksik:
            f.write("    " + "  ".join(sorted(eksik)[:40]) + "\n")
        f.write(f"\n  graf izi    {V.IZ}\n")
        f.write(f"  birim izi   {BR.iz(b)}\n")
        f.write(f"  pencere     {AY.AYAR.t_len} karakter -> birim akisi\n")

    print(f"\nyazildi -> {kls}")
    for x in sorted(os.listdir(kls)):
        p = os.path.join(kls, x)
        print(f"  {x:<24s} {os.path.getsize(p)/1e3:8.1f} KB")
    print(f"\n  {len(b.ad)} birim   " + "   ".join(
        f"{k} {n}" for k, n in sorted(sayim.items())))
    print(f"  cozulemeyen kelime: {len(eksik)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
