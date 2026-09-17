# -*- coding: utf-8 -*-
"""egitim_dok_06 — VERININ KENDISI, duz metin, ACIKLAMA YOK.

Kullanici karari, 17 Eylul 2026: *"ben tum egitim verisini aciklama
olmadan bu egitim verisidir gormek istiyorum. her satirdaki aciklamaya
gerek yok."*  Ve model_06 icin: *"verileri olustur bakayim bi."*

!! YERLESIM. `analiz_06`in dokumu DENETIM icin (graf oku, kisayol,
jeton dizisi). Ikisi ayni klasorde durunca hep once o aciliyordu --
ayrimi ADLANDIRMA degil YERLESIM tasiyor:

    veri/model_06/            VERININ KENDISI -- yalniz cumleler
        EGITIM_bildirim_1hop.txt   EGITIM_bildirim_2hop.txt
        EGITIM_bosluk.txt          EGITIM_kimlik.txt
    veri/model_06/denetim/    `analiz_06`in dokumu (aciklamali)

model_05'TEN FARKI:
    SORU BICIMI YOK.  Cevap artik soruyu birebir tekrar etmiyor;
    olculmustu ki o 7 jetonun kosullu entropisi 0,000 idi (gradyan
    uretmiyorlardi) ve t_len'i 24'te tutuyorlardi. Simdi t_len 16.
    "Soru sorma" isini BOSLUK DOLDURMA yapiyor.

Cumleler MODELIN GORDUGU DIZIDEN okunuyor (`analiz_06.Dok.oku`),
yeniden uretilmiyor -- burada ne goruyorsan havuzda o var.

    python egitim_dok_06.py [--klasor <yol>]
"""
from __future__ import annotations

import argparse
import io
import os
import sys

_K = os.path.dirname(os.path.abspath(__file__))
if _K not in sys.path:
    sys.path.insert(0, _K)

import numpy as np                                             # noqa: E402
import taban_06 as M                                           # noqa: E402
import analiz_06 as AZ                                         # noqa: E402

NL = chr(10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--klasor", default=None)
    a = ap.parse_args()
    kl = a.klasor or os.path.join(_K, "veri", "model_06")
    os.makedirs(kl, exist_ok=True)
    d = AZ.Dok("model_06")
    v, ayar = d.v, d.ayar
    n_bic = max(1, ayar.bicim)

    def yaz(ad, satirlar, not_=""):
        yol = os.path.join(kl, ad)
        with io.open(yol, "w", encoding="utf-8") as f:
            for s in satirlar:
                f.write(s + NL)
        print(f"yazildi: {ad:<26} {len(satirlar):>9,} satir"
              f"   {os.path.getsize(yol)/1024/1024:>5.1f} MB   {not_}")

    # --- BILDIRIM satirlari, UC TURKCE SIRADA ---------------------------
    for ad, lst, kodla in (("04_egitim_bildirim_1hop.txt", v.one, M.kodla_1hop),
                           ("05_egitim_bildirim_2hop.txt", v.tr2, M.kodla_2hop)):
        s = []
        for x in lst:
            for b in range(n_bic):
                s.append(d.oku(kodla(v, [x], b)[0][0]))
        yaz(ad, s, f"({n_bic} sira)")

    # --- BOSLUK DOLDURMA ------------------------------------------------
    # Havuzdakinin AYNISI: ayni tohum, ayni konumlar. Yani burada
    # gordugun satirin birebir esi egitimde var.
    if getattr(ayar, "fim_kat", 0) > 0:
        rs = np.random.default_rng(1000 + ayar.veri_tohum)
        s = []
        for lst, kodla in ((v.one, M.kodla_1hop), (v.tr2, M.kodla_2hop)):
            for b in range(n_bic):
                X = kodla(v, list(lst), b)[0]
                for r in X:
                    dz = [int(t) for t in r if int(t) != M.PAD]
                    for p in rs.choice(len(dz),
                                       size=min(ayar.fim_kat, len(dz)),
                                       replace=False):
                        s.append(d.oku(M.kodla_fim(v, dz, int(p))))
        yaz("12_egitim_bosluk.txt", s, f"(satir basina {ayar.fim_kat} varyant)")

    # --- KIMLIK ---------------------------------------------------------
    if ayar.ident_frac > 0:
        X = M.kodla_kimlik_q1(v, range(v.n_ent))[0]
        yaz("13_egitim_kimlik.txt", [d.oku(r) for r in X],
            f"(havuzda x{ayar.ident_frac:.0%} paya kadar tekrarlanir)")

    # !! JETON TABLOSU BURADA YAZILMIYOR: `analiz_06` onu
    # 01_tokenlar.txt olarak zaten yaziyor. Ikinci bir kopya,
    # kullanicinin 17 Eylul'de sordugu "niye iki tane egitim
    # var?" durumunun aynisi olurdu.
    # --- DIZILER: cumle + jeton numaralari yan yana --------------------
    ornek = []
    for lst, kodla, et in ((v.one, M.kodla_1hop, "1hop"),
                           (v.tr2, M.kodla_2hop, "2hop")):
        for x in list(lst)[:200]:
            for b in range(n_bic):
                r = kodla(v, [x], b)[0][0]
                dz = [int(t) for t in r if int(t) != M.PAD]
                ornek.append(d.oku(dz))
                ornek.append("   jeton : " + " ".join(d.jeton_ad(t) for t in dz))
                ornek.append("   numara: " + " ".join(str(t) for t in dz))
                ornek.append("")
    yaz("14_diziler_numarali.txt", ornek, "(ilk 200 olgu, uc sira, numarali)")

    print(f"{NL}t_len {v.t_len}   vocab {v.vocab}   "
          f"<BOS> {v.bosluk}  <AYIR> {v.ayir}")
    print(f"klasor: {kl}")


if __name__ == "__main__":
    main()
