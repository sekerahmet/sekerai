# -*- coding: utf-8 -*-
"""dok_14 -- modelin gordugu HER SEYI goz ile okunur dosyaya dok.

Kullanici, 20 Eylul 2026: *"once sen veri klasorune verileri yaz ben
goz ile kontrol edeyim"*.

Gerekce ayrica OLCULDU: bir gunde uc gosterim kusuru yalnizca METNE
bakilarak bulundu, hicbirini sayisal bir kapi gostermedi. Sayi
"makul" durabilir; metin duramaz.

    python dok_14.py [klasor]        varsayilan: model_14/veri

DOSYALAR
    00_OZET.txt        kac birim, kac kelime, kac soru -- tek sayfa
    01_birim.txt       451 birimin tamami, frekansiyla
    02_bolme.txt       her KELIME -> hangi birimlere bolunuyor
    03_korpus.txt      akistan kesintisiz bir parca, cumle cumle
    04_iskelet.txt     BICIM'in korpustan turettigi kalip kumesi
    05_ek_kurali.txt   her ekin ONUNDE korpusta hangi birimler var
    06_sinav.txt       sinav sorulari, BEKLENEN cevabiyla, sinifiyla
    07_uretim.txt      MODELIN YAZDIGI   (agirlik varsa)
"""
from __future__ import annotations

import collections
import os
import sys

import numpy as np

import ayar_14 as AY
import birim_14 as BR
import olcme_14 as O
import taban_14 as MT
import veri_14 as V

BIRIM = os.environ.get("BIRIM_NPZ", r"G:/Drive'ım/model_14/birim_14.npz")
AGIRLIK = os.environ.get("AGIRLIK", r"G:/Drive'ım/model_14/t0/model_t0.pt")


def ac(kls, ad):
    f = open(os.path.join(kls, ad), "w", encoding="utf-8")
    print("  ", ad)
    return f


def main() -> int:
    kls = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "veri")
    kls = os.path.abspath(kls)
    os.makedirs(kls, exist_ok=True)
    print("klasor:", kls)

    b = BR.yukle(BIRIM, yaz=lambda *a: None)
    v = MT.veri_kur(AY.AYAR, yaz=lambda *a, **k: None)
    L = MT.olcme_listeleri(AY.AYAR, v)
    G = V.kur(AY.AYAR.veri_tohum)
    E_ad = [x for t in V.TIPLER for x in G["ad"][t]]
    S = O.Sorular(v, E_ad, list(V.ILISKI), V.TIPLER, V.TR, V.TR_ILISKI,
                  b.kok, b.ix, b.korunan)
    sor = S.tum(L, b.dizi, yaz=lambda *a: None)

    bitis = set(b.ix[x] for x in ".?!" if x in b.ix)
    ek_ix = set(i for i, a in enumerate(b.ad) if a.startswith("-"))
    var_ix = set(w for ad in E_ad for w in S.birim(ad))
    bc = O.Bicim(b.dizi, var_ix, bitis, ek_ix, n=len(b))

    # ---- 01 birim ----
    with ac(kls, "01_birim.txt") as f:
        f.write("MODELIN SOZLUGU -- %d birim.  Model bunlardan baska\n"
                "hicbir sey uretemez; okuma bu noktalar arasinda en\n"
                "yakini secmek demek.\n\n" % len(b))
        f.write("%-5s %-26s %12s %8s\n" % ("#", "birim", "frekans", "pay"))
        tp = float(b.say.sum())
        for i in np.argsort(-b.say):
            f.write("%-5d %-26s %12s %7.3f%%\n"
                    % (i, b.ad[i], format(int(b.say[i]), ","),
                       100 * b.say[i] / tp))

    # ---- 02 bolme ----
    with ac(kls, "02_bolme.txt") as f:
        f.write("KELIME -> BIRIM.  Korpustaki %d kelimenin tamami.\n"
                "Model kelimeyi gormuyor, SAGDAKI birimleri goruyor.\n\n"
                % len(b.bolme))
        for w in sorted(b.bolme, key=lambda x: (x.lower(), x)):
            p = b.bolme[w]
            f.write("%-30s %s%s\n" % (w, "  ".join(p),
                                      "" if len(p) > 1 else "        (bolunmedi)"))

    # ---- 03 korpus ----
    with ac(kls, "03_korpus.txt") as f:
        f.write("BIRIM AKISI -- modelin egitildigi sey.  Akisin basindan\n"
                "kesintisiz 4.000 cumle, her satir bir cumle.\n"
                "Bir egitim ornegi bu akistan PENCERE=%d birimlik bir\n"
                "dilim (atla=%d).\n\n" % (AY.PENCERE, AY.ATLA))
        n = bas = 0
        for i in np.isin(b.dizi, list(bitis)).nonzero()[0]:
            f.write("%s\n" % b.coz(b.dizi[bas:i + 1]))
            bas, n = i + 1, n + 1
            if n >= 4000:
                break
        f.write("\n--- AYNI YERDEN, PENCERE olarak (ilk 12 pencere) ---\n")
        W = b.pencere(AY.PENCERE, AY.ATLA)
        for j in range(12):
            f.write("p%-4d %s\n" % (j * AY.ATLA, b.coz(W[j])))

    # ---- 04 iskelet ----
    with ac(kls, "04_iskelet.txt") as f:
        f.write("BICIM/kalip bunlara bakiyor: korpusun OGRETTIGI cumle\n"
                "iskeletleri. Varlik birimlerinden olusan her kesintisiz\n"
                "kosu tek bir <V> ile degistirilmis hali. %d ayri iskelet.\n"
                "Elle dilbilgisi YAZILMADI -- hepsi korpustan sayildi.\n\n"
                % len(bc.iskelet))
        for isk, c in bc.iskelet.most_common():
            f.write("%9s  %s\n" % (format(c, ","), " ".join(
                "<V>" if w == O.YUVA else b.ad[w] for w in isk)))

    # ---- 05 ek kurali ----
    with ac(kls, "05_ek_kurali.txt") as f:
        f.write("BICIM/ek bunlara bakiyor: her ekin ONUNDE korpusta\n"
                "hangi birimler gorulmus. Hem 'koke mi takildi' hem\n"
                "'ek SIRASI dogru mu' ayni tabloda.\n\n")
        for e in sorted(bc.ek_oncesi, key=lambda x: -len(bc.ek_oncesi[x])):
            on = sorted(bc.ek_oncesi[e], key=lambda i: -b.say[i])
            f.write("%-14s %4d farkli birimden sonra gelebilir\n"
                    % (b.ad[e], len(on)))
            f.write("      en siklari: %s\n\n"
                    % "  ".join(b.ad[i] for i in on[:25]))

    # ---- 06 sinav ----
    with ac(kls, "06_sinav.txt") as f:
        f.write("SINAV.  %d zincir; sinif ETIKETTEN degil KORPUSTAN:\n"
                "  OGRETILEN  cevabi soyleyen cumle korpusta GECIYOR\n"
                "  CIKARIM    GECMIYOR -- model birlestirmek zorunda\n\n"
                % (len(sor["OGRETILEN"]) + len(sor["CIKARIM"])))
        for grup in ("OGRETILEN", "CIKARIM"):
            f.write("\n%s  %s  (%d soru, ilk 1500'u yaziliyor)\n%s\n"
                    % ("=" * 22, grup, len(sor[grup]), "=" * 60))
            for s in sor[grup][:1500]:
                f.write("%-62s -> %s%s\n"
                        % (b.coz(s.onek), b.coz(s.cevap),
                           "     [kisayol: %s]" % b.coz(s.kisayol)
                           if s.kisayol else ""))

    # ---- 07 uretim ----
    if os.path.exists(AGIRLIK):
        import torch
        import konus_14 as K
        _, m, a = K.yukle(BIRIM, AGIRLIK, "cpu")
        with ac(kls, "07_uretim.txt") as f:
            f.write("MODELIN YAZDIGI.  agirlik %s\n" % AGIRLIK)
            f.write("ayar  %s\n\n" % "  ".join("%s=%s" % kv
                                               for kv in a.items()))
            for grup in ("OGRETILEN", "CIKARIM"):
                f.write("\n%s  %s\n%s\n" % ("=" * 22, grup, "=" * 60))
                for s in sor[grup][:300]:
                    onek = [int(x) for x in s.onek]
                    with torch.no_grad():
                        cik, puan = K.uret(m, b, onek, 12, 0, a["R_CAPA"],
                                           False, yaz=lambda *x: None)
                    f.write("SORU      %s\nBEKLENEN  %s\nURETILEN  %s\n\n"
                            % (b.coz(onek), b.coz(s.cevap), b.coz(cik)))
    else:
        print("   (07_uretim.txt atlandi -- agirlik yok: %s)" % AGIRLIK)

    # ---- 00 ozet ----
    with ac(kls, "00_OZET.txt") as f:
        cum = collections.Counter()
        bas = 0
        for i in np.isin(b.dizi, list(bitis)).nonzero()[0]:
            cum[i + 1 - bas] += 1
            bas = i + 1
        n_c = sum(cum.values())
        f.write("model_14 -- VERI OZETI\n%s\n\n" % ("=" * 46))
        f.write("SOZLUK      %d birim   (kelime %d -> birim %d, %.2fx)\n"
                % (len(b), len(b.bolme), len(b), len(b.bolme) / len(b)))
        f.write("AKIS        %s birim   %s cumle   cumle basi %.1f birim\n"
                % (format(len(b.dizi), ","), format(n_c, ","),
                   len(b.dizi) / n_c))
        f.write("PENCERE     L=%d  atla=%d  ->  %s egitim ornegi\n"
                % (AY.PENCERE, AY.ATLA,
                   format((len(b.dizi) - AY.PENCERE) // AY.ATLA + 1, ",")))
        n_, ort, kap = BR.kapsama(b, AY.PENCERE)
        f.write("            soru+cevap %s cift, ort %.1f birim, "
                "L=%d kapsama %%%.1f\n" % (format(n_, ","), ort,
                                           AY.PENCERE, 100 * kap))
        f.write("ISKELET     %d ayri cumle kalibi\n" % len(bc.iskelet))
        f.write("SINAV       %d soru   OGRETILEN %d   CIKARIM %d\n"
                % (len(sor["OGRETILEN"]) + len(sor["CIKARIM"]),
                   len(sor["OGRETILEN"]), len(sor["CIKARIM"])))
        f.write("IZ          graf %s   olcme %s   birim %s\n"
                % (V.IZ, MT.olcme_izi(L), BR.iz(b)))
        f.write("\nCUMLE UZUNLUGU (birim)\n")
        for u in sorted(cum)[:26]:
            f.write("  %3d %s %s\n" % (u, format(cum[u], ">8,"),
                                       "#" * int(60 * cum[u] / max(cum.values()))))
    print("bitti.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
