# -*- coding: utf-8 -*-
"""konus_11 — model_11 ile DUZ KONUSMA. Sablon YOK, kisit YOK.

    python konus_11.py [klasor] [--adim N | --pencere 5] [--panel]
    python konus_11.py --soru "Ahmet Aydin'in kardesi kimdir?"

Kullanici, defalarca: *"istatistik birsey ifade etmiyor, ornek ile
sorman lazim"*  ve  *"bu da GOZLE OLCUM demek"*. Sinav bir sayi verir;
bu dosya modelin AGZINDAN CIKANI verir.

!! BU BIR OLCU DEGIL. Hukum `pencere_11` ile verilir. Buradaki her sey
elle secilmis sorulardir ve oyle okunur.

==========================================================================
KARAKTER SURUMU -- 878 SATIR SILINDI, YENIDEN YAZILDI

Onceki hali model_08'in JETON surumuydu ve `TASINMADI = True` diye
kendini kilitliyordu: `kodla_1hop`, `kodla_2hop`, `v.yuva_ara`,
`jeton_ad` semasi -- hicbiri karakter modelinde YOK.

Karakterde "modelle konusmak" cok daha basit: dizgeyi `S.kodla` ile
jetona cevir, uret, `S.coz` ile geri oku. Yuva yok, maske yok, sablon
yok. O yuzden dosya TASINMADI, YENIDEN yazildi -- 878 satirin buyuk
kismi artik var olmayan bir semanin etrafindaydi.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

import numpy as np
import torch

_K = os.path.dirname(os.path.abspath(__file__))
if _K not in sys.path:
    sys.path.insert(0, _K)

# --- TURKCE CIKTI: KODLAMA BETIGIN ISI, BASLATICININ DEGIL ---------------
# !! KUSUR (18 Eylul). `konus.bat` `chcp 65001` yaziyordu, yani CIFT
# TIKLAYINCA dogru basiyordu. Terminalden / IDE'den Windows yerel
# kodlamasi devreye giriyor ve cikti soyle cikiyordu:
#     Elif Ayd?n'd?r.        yerine       Elif Aydın'dır.
# Bu arac "kendi gozlerimle gormek istiyorum" diye istendi; okunmayan
# cikti o isi GORMEZ, ustelik bozuk metin bir arizayi GIZLEYEBILIR.
for _akis in (sys.stdout, sys.stderr):
    try:
        _akis.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

import korpus_11 as KOR                                        # noqa: E402
import metin_11 as MT                                          # noqa: E402
import olcme_11 as OLC                                         # noqa: E402
import pencere_11 as P                                         # noqa: E402
import taban_11 as M                                           # noqa: E402
import veri_11 as V                                            # noqa: E402
from model_11 import ModelSade                                 # noqa: E402

SUS = lambda *a, **k: None
# Ham kosu verisi Drive'da, yerelde bagli (CLAUDE.md "Nerede ne okunur").
KLASOR = os.path.join("G:" + os.sep, "Drive'im", "model_11", "t0")


def yukle(klasor: str, adim=None, pencere=0):
    """(net, S, ayar, G, hangi) -- model + sozluk + graf.

    `pencere > 0` ise SON N anlik goruntunun AGIRLIK ORTALAMASI
    yuklenir; hukum okumasi (`pencere_11`) da bunu kullanir, yani
    konusulan model OLCULEN modeldir. Aksi halde tek anlik goruntu.

    !! Korpus `M.egitim_havuzu(ayar, v)` ile kuruluyor, `KOR.havuz`
    dogrudan CAGRILMIYOR -- okuma araclari 10 parametrenin 6'sini
    geciyor ve EGITILENDEN BASKA bir korpus kuruyordu (test_11 §0b)."""
    ayar = P.ayar_oku(klasor)
    snap = P.anlik_goruntuler(klasor)
    assert snap, f"anlik goruntu YOK: {klasor}"
    v = M.veri_kur(ayar, yaz=SUS)
    _X, S = M.egitim_havuzu(ayar, v, yaz=SUS)
    G = importlib.import_module(ayar.veri_ad).kur(ayar.veri_tohum)
    net = ModelSade(ayar, S.vocab).to(M.DEV)
    if pencere:
        yol = [snap[a] for a in sorted(snap)[-pencere:]]
        net.load_state_dict(P.agirlik_ortalamasi(yol))
        hangi = (f"PENCERE {len(yol)} anlik goruntu "
                 f"({sorted(snap)[-pencere]}..{max(snap)})")
    else:
        a = adim if adim is not None else max(snap)
        assert a in snap, f"adim {a} YOK: {sorted(snap)}"
        net.load_state_dict(torch.load(snap[a], map_location=M.DEV))
        hangi = f"TEK anlik goruntu, adim {a}"
    net.eval()
    return net, S, ayar, G, hangi


@torch.no_grad()
def sor(net, S, metin: str, n=120, nokta_durur=True) -> str:
    """Bir dizge ver, modelin DEVAMINI al. Kisit YOK -- serbest uretim.

    !! Uretim `olcme_11.uret` ile, yani SINAVIN kullandigi kodun
    AYNISI. Ayri bir uretim yolu yazilsaydi "konusta boyle diyor ama
    sinavda baska" gibi bir ayrisma dogardi."""
    j = S.kodla(metin)
    X = np.full((1, len(j) + n + 1), 0, np.int64)
    X[0, :len(j)] = j
    bas = np.array([len(j)], np.int64)
    return OLC.uret(net, S, X, bas, n, M.DEV, bs=1, nokta_durur=nokta_durur)[0]


# ======================= NITELIK PANELI ==================================
# Onkayit model_11.md §4: DOGRULUK / DURUSTLUK / TUTARLILIK / AKICILIK.
# Her olcum noktasinda AYNI 12 sonda sorulur ki degisim GORULEBILSIN.
#
# !! SORULAR GRAFTAN URETILIYOR, elle yazilmiyor -- veri degisince soru
# da degisir ve "eski soruyu yeni veriye sormak" kusuru olmaz.
# !! DURUSTLUK sondalari TUTULAN veriden (`reddetme_bolme`): o ad ve o
# (tip, iliski) cifti egitimde HIC gecmez.


def _soru_cevap(cumle: str):
    """'Soru? Cevap.' -> (soru, cevap)."""
    i = cumle.index("? ") + 1
    return cumle[:i], cumle[i:].strip()


def panel_sorulari(G):
    """[(boyut, soru, beklenen | 'REDDETMELI' | None)] -- 12 sonda."""
    tipi = {a: t for t in V.TIPLER for a in G["ad"][t]}
    olgu = G["olgu"]
    hedef = {r: tipi[h] for (_a, r), h in olgu.items()}
    P_ = []

    # --- DOGRULUK: AYNI ozneden 1, 2 ve 3 adim. Zincir derinlestikce
    #     gizlenen kopru sayisi artiyor; ucu birlikte okunur.
    k = next(a for a in G["ad"]["KISI"] if (a, "bolumu") in olgu)
    b1 = olgu[(k, "bolumu")]
    f1 = olgu[(b1, "fakultesi")]
    u1 = olgu[(f1, "universitesi")]
    P_.append(("DOGRULUK 1 adim",
               *_soru_cevap(MT.soru(k, "bolumu", b1, tipi[b1], 0))))
    P_.append(("DOGRULUK 2 adim  (1 kopru gizli)",
               *_soru_cevap(MT.zincir_soru(k, "bolumu", "fakultesi", f1,
                                           tipi[f1], 0))))
    P_.append(("DOGRULUK 3 adim  (2 kopru gizli)",
               *_soru_cevap(MT.yol_soru(k, ("bolumu", "fakultesi",
                                            "universitesi"), u1, tipi[u1], 0))))

    # --- DURUSTLUK: olmayan varlik, imkansiz iliski, VE bildigi bir soru.
    #     Ucuncusu sart: her seye "yok" diyen model ilk ikisinde tavana
    #     cikar. `kacamak` olmadan `dogru ret` OKUNMAZ.
    bl = KOR.reddetme_bolme(G, 0.20, 0)
    _t, _adlar = next((t, L) for t, L in sorted(bl["ad_tut"].items()) if L)
    _ad = sorted(_adlar)[0]
    _r = sorted({r for (a_, r) in olgu if tipi[a_] == _t})[0]
    P_.append(("DURUSTLUK olmayan ad  (TUTULAN)",
               _soru_cevap(MT.yok_varlik(_ad, _t, _r, hedef[_r], 0))[0],
               "REDDETMELI"))
    _tt, _rr = sorted(bl["cift_tut"])[0]
    P_.append(("DURUSTLUK imkansiz iliski  (TUTULAN)",
               _soru_cevap(MT.yok_iliski(G["ad"][_tt][0], _tt, _rr,
                                         hedef[_rr], 0))[0],
               "REDDETMELI"))
    k2 = next(a for a in G["ad"]["KISI"] if (a, "annesi") in olgu)
    a2 = olgu[(k2, "annesi")]
    _s2, _c2 = _soru_cevap(MT.soru(k2, "annesi", a2, tipi[a2], 0))
    P_.append(("DURUSTLUK kacamak sinami  (BILIYOR)", _s2, _c2))

    # --- TUTARLILIK: ayni olgu BASKA yuzeyde, ters yon, ve tip.
    P_.append(("TUTARLILIK ayni olgu, 4. soru kalibi",
               _soru_cevap(MT.soru(k, "bolumu", b1, tipi[b1], 3))[0],
               _soru_cevap(MT.soru(k, "bolumu", b1, tipi[b1], 0))[1]))
    te = next(a for a in G["ad"]["TEZ"] if (a, "yazari") in olgu)
    P_.append(("TUTARLILIK ters yon (tez -> yazar)",
               *_soru_cevap(MT.soru(te, "yazari", olgu[(te, "yazari")],
                                    tipi[olgu[(te, "yazari")]], 0))))
    P_.append(("TUTARLILIK tip  (X nedir?)",
               *_soru_cevap(MT.tip_sorusu(k, tipi[k], 0))))

    # --- AKICILIK: serbest devam. BEKLENEN YOK -- gozle okunur.
    P_.append(("AKICILIK ad ile basla", MT._tr(k) + " ", None))
    P_.append(("AKICILIK tip cumlesinden devam",
               MT.tip_cumlesi(k, tipi[k], 0) + " ", None))
    P_.append(("AKICILIK sehir", MT._tr(G["ad"]["SEHIR"][0]) + " ", None))
    return P_


def panel(net, S, G, uzun=200):
    P_ = panel_sorulari(G)
    print("=" * 78)
    print(" NITELIK PANELI -- 12 sonda. OLCU DEGIL, GOZLE OKUNUR.")
    print(" (Hukum `pencere_11` + `durustluk_11` ile verilir.)")
    print("=" * 78)
    son = None
    sayac = {}
    for boyut, soru_, bek in P_:
        b = boyut.split()[0]
        if b != son:
            print()
            son = b
        akici = boyut.startswith("AKICILIK")
        c = sor(net, S, soru_, n=uzun if akici else 120,
                nokta_durur=not akici)
        print(f"  [{boyut}]")
        print(f"    SORU   {soru_}")
        print(f"    MODEL  {c}")
        if bek == "REDDETMELI":
            _ok = OLC.ret_mi(c)
            print(f"    BEKLEN REDDETMELI        -> "
                  f"{'RET  (dogru)' if _ok else 'RET DEGIL  (kacirdi)'}")
            sayac[b] = sayac.get(b, 0) + int(_ok)
        elif bek is not None:
            _ok = (c == bek)
            _kac = OLC.ret_mi(c)
            print(f"    BEKLEN {bek:<24} -> "
                  + ("DOGRU" if _ok else
                     "KACAMAK  ('yok' dedi ama BILIYOR)" if _kac else
                     "tutmuyor"))
            sayac[b] = sayac.get(b, 0) + int(_ok)
    print()
    print("  " + "  ".join(f"{b} {n}" for b, n in sorted(sayac.items()))
          + "   (AKICILIK sayilmaz -- gozle okunur)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("klasor", nargs="?", default=KLASOR)
    ap.add_argument("--adim", type=int, default=None)
    ap.add_argument("--pencere", type=int, default=0,
                    help="son N anlik goruntunun AGIRLIK ORTALAMASI")
    ap.add_argument("--panel", action="store_true")
    ap.add_argument("--soru", default=None)
    ap.add_argument("--uzun", type=int, default=200)
    a = ap.parse_args()

    net, S, ayar, G, hangi = yukle(a.klasor, a.adim, a.pencere)
    print(f"model_11  t{ayar.tohum}   {hangi}")
    print(f"  sozluk {S.vocab} jeton   korpus izi {S.korpus_izi}")

    if a.panel:
        panel(net, S, G, a.uzun)
        return
    if a.soru:
        print(f"\n  SORU   {a.soru}")
        print(f"  MODEL  {sor(net, S, a.soru, n=a.uzun)}")
        return

    print("\n  Bir sey yaz, model DEVAMINI yazsin. Bos satir = cikis.")
    print("  Basina '>' koyarsan UZUN uretim (nokta durdurmaz).")
    while True:
        try:
            s = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not s:
            break
        if s.startswith(">"):
            print(sor(net, S, s[1:].lstrip(), n=a.uzun, nokta_durur=False))
        else:
            print(sor(net, S, s, n=120))


if __name__ == "__main__":
    main()
