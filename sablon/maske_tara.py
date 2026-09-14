# -*- coding: utf-8 -*-
"""MASKE TARA — 24 maske adayini ASIL OLCUYLE degerlendir.

NEDEN VAR. Asama B adaylari su skorla siraliyor:

    skor = (r1 BAGIMSIZLIGINI dusur) - (COMP'a ZARAR)

Bu bir VEKIL. Mekanizma iddiasi ise KISAYOL hakkinda: "maskeleme modelin
f(e,r2) yolunu kapatir". Kisayol oranini ZATEN olcuyoruz (`ent_shortcut`),
yani vekile ihtiyac yok -- adaylari dogrudan asil olcuyle siralayabiliriz.

Bu betik HICBIR SEY SECMEZ ve FAZ 3'u etkilemez. Sorusu tek:

    >> Sectigimiz nokta, kacis yolunu GERCEKTEN kapatiyor mu?

Her aday icin uc sayi basar (maske INFERENCE'ta acik, egitim YOK):

    ent      ENT-ARAMA dogrulugu      (yukselmeli ya da en azindan dusmemeli)
    kisayol  ENT-ARAMA kisayol orani  (DUSMELI -- iddia bu)
    comp     COMP dogrulugu           (COKMEMELI)

DIKKAT -- BU EGITIM DEGIL. Maskeyi egitilmis bir modele SONRADAN takmak,
o maskeyle EGITMEKLE ayni sey degildir. Burasi "kacis yolu bu mu" sorusunu
cevaplar; "maskelersem ogrenir mi" sorusunu DEGIL. Hukum yalniz G/GM
kiyasindan gelir (CLAUDE.md 0).

OLCUM KUMESI: ENT-ARAMA. Hukum kumesi (ENT-AYIRT) BILEREK kullanilmiyor --
arama neye bakiyorsa tani da ona bakmali, yoksa hukum kumesi kirlenir.

    VERI=okul PRESET=grok_uzun python maske_tara.py \
        --klasor /content/calis_g/cikti_g --adim 45000
    # ya da agirlik ortalamasi:
    #   --adim 100000,105000,110000,115000,120000
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sifirdan as S
from gpu_kapisi import gpu_bos_mu


def yukle(klasor, tohum, adimlar):
    top, bulunan = None, []
    for a in adimlar:
        y = os.path.join(klasor, f"snap_A_s{tohum}_{a:06d}.pt")
        if not os.path.exists(y):
            continue
        sd = torch.load(y, map_location="cpu")
        top = ({k: v.float() for k, v in sd.items()} if top is None
               else {k: top[k] + v.float() for k, v in sd.items()})
        bulunan.append(a)
    if not bulunan:
        raise FileNotFoundError(f"anlik goruntu YOK: {klasor} {adimlar}")
    return {k: v / len(bulunan) for k, v in top.items()}, bulunan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--klasor", required=True)
    ap.add_argument("--adim", required=True, help="tek sayi ya da virgullu")
    ap.add_argument("--tohum", type=int, default=0)
    ap.add_argument("--n", type=int, default=2000, help="ornek sayisi")
    ap.add_argument("--zorla", action="store_true",
                    help="egitim kosuyor olsa bile baslat")
    a = ap.parse_args()

    # EGITIM KOSARKEN BASLAMA (14 Eylul: uc runtime geri donusumu).
    gpu_bos_mu(zorla=a.zorla, ad="maske taramasi")
    assert S.VERI, "VERI bayragi YOK"
    facts, _p, _one, _tr2, comp, ent, _se, _ue, _e2 = S.build_data()
    assert S.ENT_ARAMA, "ENT_ARAMA bos -> arama kumesi ayrilmamis"

    LA = S.ENT_ARAMA[:a.n]
    LC = comp[:a.n]
    EA, EC = S.enc_two(LA), S.enc_two(LC)
    brA = np.array([S.ENT_OFF + b for _, _, _, b, _ in LA], np.int64)
    scA = np.array([S.ENT_OFF + int(facts[e, r2]) for e, _, r2, _, _ in LA],
                   np.int64)
    brC = np.array([S.ENT_OFF + b for _, _, _, b, _ in LC], np.int64)
    scC = np.array([S.ENT_OFF + int(facts[e, r2]) for e, _, r2, _, _ in LC],
                   np.int64)

    sd, bulunan = yukle(a.klasor, a.tohum,
                        [int(x) for x in a.adim.split(",")])
    net = S.Net("A", S.CFG).to(S.DEV)
    net.load_state_dict(sd)
    net.eval()
    L = S.CFG["L"]

    def olc(poz, b0):
        for i, blk in enumerate(net.blocks):
            blk.mask_key = poz if (poz is not None and i >= b0) else None
        zA, _ = S.zengin(net, EA, brA, scA)
        zC, _ = S.zengin(net, EC, brC, scC)
        for blk in net.blocks:
            blk.mask_key = None
        return zA["acc"], zA["shortcut"], zC["acc"]

    print("=" * 78)
    print(f"MASKE TARAMA   {a.klasor}   adimlar {bulunan}")
    print(f"  olcum: ENT-ARAMA {len(LA)} ornek (HUKUM kumesi DEGIL), "
          f"COMP {len(LC)}")
    print("  DIKKAT: maske INFERENCE'ta takiliyor, EGITIM yok.")
    print("=" * 78)
    e0, k0, c0 = olc(None, 0)
    print(f"  MASKESIZ      ent {e0:.4f}   kisayol {k0:.4f}   comp {c0:.4f}")
    print()
    print(f"  {'poz':>4} {'bloklar':>9} {'ent':>8} {'kisayol':>9} {'comp':>8}"
          f"   {'d(kisayol)':>11} {'d(comp)':>9}")
    sonuc = []
    for poz in (1, 2, 3):
        for b0 in range(L):
            e1, k1, c1 = olc(poz, b0)
            sonuc.append((poz, b0, e1, k1, c1))
            print(f"  {poz:>4} {f'{b0}..{L-1}':>9} {e1:8.4f} {k1:9.4f} "
                  f"{c1:8.4f}   {k1-k0:+11.4f} {c1-c0:+9.4f}")
    # ETKI DENETIMI (CLAUDE.md 7). Butun adaylar tabanla AYNI cikarsa bulgu
    # "etki yok" DEGIL, "olcum bozuk"tur: maske takilmamis ya da model zaten
    # her seye ayni cevabi veriyor olabilir. Ikisi de raporlanacak sey degil.
    _fark = max(abs(x[2] - e0) + abs(x[3] - k0) + abs(x[4] - c0)
                for x in sonuc)
    assert _fark > 1e-6, (
        "MASKE ETKISIZ -> olcum bozuk (CLAUDE.md 7)." + chr(10)
        + f"  taban ent {e0:.4f} kisayol {k0:.4f} comp {c0:.4f}; "
          f"24 adayin HEPSI ayni." + chr(10)
        + "  olasi sebep: model henuz hicbir sey ogrenmemis "
          "(hepsi 0.0000) ya da mask_key takilmiyor.")

    print()
    # ASIL SORU: kisayolu EN COK dusuren, COMP'u COKERTMEYEN aday hangisi?
    # 'Cokertmeyen' esigi ONCEDEN degil BURADA yaziliyor -- bu bir TANI,
    # secim degil. Esik 0.10: D3.3'un SAGLIK-comp kapisiyla ayni buyukluk.
    uygun = [x for x in sonuc if x[4] >= c0 - 0.10]
    if uygun:
        en = min(uygun, key=lambda x: x[3])
        print(f"  KISAYOLU EN COK DUSUREN (comp > {c0-0.10:.3f} sartiyla):")
        print(f"     poz {en[0]}  bloklar {en[1]}..{L-1}   "
              f"kisayol {k0:.4f} -> {en[3]:.4f}   comp {c0:.4f} -> {en[4]:.4f}"
              f"   ent {e0:.4f} -> {en[2]:.4f}")
    else:
        print(f"  !! HICBIR aday comp'u {c0-0.10:.3f} ustunde tutamiyor")
    print()
    print("  Bu bir TANIDIR, SECIM DEGIL. FAZ 3 kendi aramasini yapar.")


if __name__ == "__main__":
    main()
