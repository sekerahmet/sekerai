# -*- coding: utf-8 -*-
"""dokum_10 — model_10 korpusu, OKUNABILIR metin dosyalari.

    python dokum_10.py [--kopya 5] [--tlen 512] [--ornek 60]

Amac: veriyi GOZLE gormek. Kapilarin gectigi ayri bir sey, metnin
duzgun Turkce olmasi ayri.
"""
from __future__ import annotations

import argparse
import io
import os
import sys

_K = os.path.dirname(os.path.abspath(__file__))
if _K not in sys.path:
    sys.path.insert(0, _K)

import jeton_10 as J                                          # noqa: E402
import korpus_10 as K                                         # noqa: E402
import metin_10 as MT                                         # noqa: E402
import model_10 as MM                                         # noqa: E402
import taban_10 as M                                          # noqa: E402
import veri_10 as V                                           # noqa: E402

NL = chr(10)


def _yaz(yol, baslik, giris, bloklar, n):
    with io.open(yol, "w", encoding="utf-8", newline="") as f:
        f.write(baslik + NL + "=" * 72 + NL + NL)
        for x in giris:
            f.write(x + NL)
        for ad, not_, sat in bloklar:
            f.write(NL + f"---- {ad} " + "-" * max(4, 66 - len(ad)) + NL)
            f.write(f"     {len(sat):,} satir, asagida {min(n, len(sat))}" + NL)
            for x in not_:
                f.write("     " + x + NL)
            f.write("-" * 72 + NL)
            k = max(1, len(sat) // n)
            for x in sat[::k][:n]:
                # <EOS> GORUNUR OLSUN. Ham hali chr(10) ve dosyaya oyle
                # yazilinca BIR DILIM BIRKAC SATIR gibi gorunuyordu --
                # okuyan dilimin nerede bittigini ayirt edemiyordu.
                f.write(x.replace(K.EOS_IM, " <EOS> ") + NL)
    return os.path.getsize(yol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kopya", type=int, default=5)
    ap.add_argument("--tlen", type=int, default=512)
    ap.add_argument("--ornek", type=int, default=60)
    a = ap.parse_args()
    kl = os.path.join(_K, "veri", "model_10")
    os.makedirs(kl, exist_ok=True)
    sus = lambda *x, **y: None

    print("veri kuruluyor ...")
    v = M.veri_kur(MM.AYAR, yaz=sus)
    G = V.kur(MM.AYAR.veri_tohum)
    E = [x for t in V.TIPLER for x in G["ad"][t]]
    tipi = {x: t for t in V.TIPLER for x in G["ad"][t]}
    print(f"  graf izi {V.IZ}   olcme izi "
          f"{M.olcme_izi(M.olcme_listeleri(MM.AYAR, v))}")

    print("korpus kuruluyor ...")
    bb, bs = K.biyografiler(v, G, kopya=a.kopya)
    zb, zs = K.zincirler(v, G, kopya=a.kopya)
    kim = K.kimlik_belgeleri(v, G, n=6)
    yasak = K.cakisan_ciftler(v)
    pak, olg, ham = {}, [], []
    for ad, L in (("biyografi", bb), ("biyografi_soru", bs),
                  ("zincir", zb), ("zincir_soru", zs), ("kimlik", kim)):
        d, o = K.paketle(L, a.tlen, yasak=yasak)
        pak[ad] = d
        olg += o
        ham.append("".join(d))
    K.sizinti_kapisi(v, olg)
    S = J.Sozluk("".join(ham).replace(K.EOS_IM, ""))
    print(f"  {S}")

    # --- dosyalar ---------------------------------------------------
    n = a.ornek
    yaz = []
    yaz.append(("01_sozluk.txt", "01 — SOZLUK (karakter)",
        [f"{S.vocab} jeton: <PAD>, <EOS> ve {len(S.harf)} karakter.",
         "BPE DEGIL, KARAKTER. Olculdu: bu korpusta BPE (vocab 1024)",
         "`danismaninin`i TEK jetona birlestiriyor -- model_08'in atomik",
         "iliski sorununu yeniden uretiyor. 227.189 kelimenin yalniz 491",
         "benzersizi var; parcalamak icin baski YOK.", "",
         "Unlu uyumunu ('in/'in/'un/'un) model HARF HARF kendi ogrenecek."],
        [("KARAKTERLER", [], [f"{i + 2}\t{c}" for i, c in enumerate(S.harf)])]))
    orn = {}
    for r in V.ILISKI:
        for (o, rr), c in G["olgu"].items():
            if rr == r:
                orn[r] = [MT.cumle(o, r, c, b, tipi[c])
                          for b in range(MT.N_BICIM)]
                break
    yaz.append(("02_bicimler.txt", "02 — BICIMLER (ayni olgu, bes yuzey)",
        ["model_08'in 'ILISKI BASTA' bicimi (`Kardesidir Elif Aydin,`)",
         "BURAYA ALINMADI -- dogal Turkce degil."],
        [(MT.BICIM[b], [], [orn[r][b] for r in V.ILISKI])
         for b in range(MT.N_BICIM)]))
    yaz.append(("03_iliskiler.txt", "03 — HER ILISKIDEN CUMLE",
        [f"{len(V.ILISKI)} iliski, {len(v.one):,} olgu.",
         "BIR CUMLE = BIR OLGU, istisnasiz."],
        [("KANONIK", [], [orn[r][0] for r in V.ILISKI])]))
    for no, ad, not_ in (
            ("04", "biyografi", ["Bir varligin ATOMIK olgulari, sirasi "
                                 "KARISTIRILMIS, bicimi rastgele.",
                                 "bioS'in karsiligi. 2-hop BURAYA GIRMEZ."]),
            ("05", "biyografi_soru", ["Ayni biyografi, SORU yuzeyinde."]),
            ("06", "zincir", ["2-HOP. Kopru cumlede GECMIYOR -- "
                              "bu kolun butun sorusu bu."]),
            ("07", "zincir_soru", ["Ayni zincirler, SORU yuzeyinde."]),
            ("08", "kimlik", ["Sifir adim ozdeslik (identity bridge)."])):
        yaz.append((f"{no}_{ad}.txt", f"{no} — {ad.upper()} DILIMLERI",
            [f"Paket: CUMLE SINIRINDA bol, t_len {a.tlen}.",
             "Her dilim CUMLE BASINDA basliyor -- sinav da konum 0'da",
             "basliyor, ikisi uyussun diye. Bedeli ~%6 dolgu.",
             "<EOS> = belge siniri."],
            [(f"DILIM (t_len {a.tlen})", not_, pak[ad])]))
    icerik = []
    for dosya, baslik, giris, blok in yaz:
        b = _yaz(os.path.join(kl, dosya), baslik, giris, blok, n)
        icerik.append((dosya, baslik, sum(len(s) for _a, _n, s in blok), b))
        print(f"yazildi: {dosya:<24}{b / 1024:>7.0f} KB")

    with io.open(os.path.join(kl, "00_ICINDEKILER.txt"), "w",
                 encoding="utf-8", newline="") as f:
        f.write("model_10 — KORPUS" + NL + "=" * 72 + NL + NL)
        f.write(f"graf izi   {V.IZ}    (model_08 ile AYNI)" + NL)
        f.write(f"sozluk     {S.vocab} karakter jetonu (iz {S.iz})" + NL)
        f.write(f"kopya      {a.kopya}   (bioS multi{a.kopya})" + NL)
        f.write(f"t_len      {a.tlen}" + NL)
        tk = sum(len("".join(pak[k])) for k in pak)
        f.write(f"korpus     {tk:,} karakter" + NL)
        f.write(f"maruziyet  {2 * a.kopya}/epok  ->  44 epokta "
                f"{88 * a.kopya}  (model_08: 442)" + NL + NL)
        for dosya, baslik, ns, by in icerik:
            f.write(f"{dosya:<24} {baslik}" + NL)
            f.write(f"{'':<24} {ns:,} satir  ·  {by / 1024:.0f} KB" + NL)
        f.write(NL + "SIZINTI KAPISI GECTI: tutulan hicbir zincirin iki "
                "kenari" + NL)
        f.write("ayni dilime dusmuyor (kisitli siralama)." + NL)
    print(f"\nklasor: {kl}")


if __name__ == "__main__":
    main()
