# -*- coding: utf-8 -*-
"""odul_tani_04 — ODUL CIKTISININ DENETIMI. TEK BASINA DURUR.

Kullanici, 17 Eylul 2026: *"bizim kurdugumuz modelin ciktisini iyi
denetlemek lazim: hangi kapida en cok tikandi, nerde yanlis yapti, niye
tahmin edemedi vs."*

Uc soruyu ayri ayri cevaplar:

  1) MERDIVEN DAGILIMI -- ornekler hangi basamakta birikiyor, ve her
     basamak ortalama odule NE KADAR katkı veriyor. "Odul 0.13" tek
     basina hicbir sey soylemez; 0.13'un NEREDEN geldigi soyler.
  2) NEREDE TIKANIYOR -- ayni dagilim cevabin TIPINE ve son ILISKIYE
     gore. Bir tipte tikaniyorsa merdiven o tipte yanlis kurulmustur.
  3) NIYE TUTTURAMADI -- H'ye ulasamayan sorularda: dogru cevabin sirasi
     kacinci, kopru okunabiliyor mu, model onun YERINE ne diyor.

HUKUM ARACI DEGIL. Hukum `pencere_04` ile verilir (onkayit model_04.md 6).
Bu arac arizanin SEKLINI gosterir.

    python odul_tani_04.py <t_klasoru> [--bolme ent] [--genislik 5]
                           [--son ADIM] [--ornek 8]
"""
from __future__ import annotations

import argparse
import collections
import io
import os
import sys

import numpy as np
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")
_D = os.path.dirname(os.path.abspath(__file__))
if _D not in sys.path:
    sys.path.insert(0, _D)

import taban_04 as M                                          # noqa: E402
import odul_04 as OD                                          # noqa: E402
import pencere_04 as PEN                                      # noqa: E402
import veri_04 as V                                           # noqa: E402

# MERDIVEN SIRASI -- `odul_04.Puanlayici.puanla` ile AYNI OLMAK ZORUNDA.
# KAPI 2 (jeton sayisi) 17 Eylul'de KAPI olmaktan cikip BASAMAK oldu:
# kapi oldugunda menzile isabet eden cevaplarin %55'i zemine dusuyordu
# ve "menzilden rastgele sec", "tipten rastgele sec"ten daha az aliyordu.
BASAMAK = ("ZEMIN KAPI1 <YOK> sagda degil",
           "ZEMIN KAPI3 olmayan varlik",
           "KISAYOL (1 adimda ulasilir)",
           "TIP  gecerli varlik, YANLIS tip",
           "E    dogru tip, menzil DISI",
           "F    2 adimda, aile/tur yanlis",
           "G    2 adimda + aile/tur dogru",
           "H    TAM DOGRU")


def basamakla(pz, u, ozne, ksy, dogru):
    """(N,3) ornek -> (N,) basamak indeksi (BASAMAK ile ayni sira)."""
    n = len(u)
    b = np.full(n, -1, np.int64)
    k1 = pz.sag_hizali(u)
    b[~k1] = 0
    kalan = k1
    ent = pz.varlik_ara(u)
    b[kalan & (ent < 0)] = 1
    kalan = kalan & (ent >= 0)
    e_ = np.clip(ent, 0, None)
    ksy_m = pz.menzil.h1[ozne, e_] | ((ksy >= 0) & (ent == ksy))
    b[kalan & ksy_m] = 2
    kalan = kalan & ~ksy_m
    dt = (pz.par[e_] != pz.yok).sum(1) == pz.n_jeton[dogru]
    b[kalan & ~dt] = 3
    kalan = kalan & dt
    m2 = pz.menzil.h2[ozne, e_]
    b[kalan & ~m2] = 4
    kalan = kalan & m2
    ai = pz.par[e_, 1] == pz.par[dogru, 1]
    b[kalan & ~ai] = 5
    kalan = kalan & ai
    tam = ent == dogru
    b[kalan & ~tam] = 6
    b[kalan & tam] = 7
    assert (b >= 0).all(), "siniflandirilamayan ornek VAR -- merdiven eksik"
    return b, ent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("klasor")
    ap.add_argument("--bolme", default="ent",
                    help="ent (TUTULMUS sinav) | ent_arama (ODULUN kostugu) "
                         "| comp | ent_yok")
    ap.add_argument("--genislik", type=int, default=5)
    ap.add_argument("--son", type=int, default=None)
    ap.add_argument("--ornek", type=int, default=8, help="soru basina cekim")
    a = ap.parse_args()

    ayar = PEN.ayar_oku(a.klasor)
    snap = PEN.anlik_goruntuler(a.klasor)
    adimlar = list(snap)
    if a.son is not None:
        assert a.son in adimlar, f"{a.son} adiminda anlik goruntu YOK"
        adimlar = adimlar[:adimlar.index(a.son) + 1]
    pen = adimlar[-a.genislik:]
    # BASKA BIR KOLUN klasorunu okuyabilmek icin: model_03'un ayarinda
    # `veri_ad="veri_03"` yaziyor ve o modul model_04/ icinde YOK.
    # Icerik AYNI (ikisi de veri_okul4 kopyasi) ama bunu VARSAYMIYORUZ:
    # veri kurulduktan sonra OLCME IZI, kosunun kendi kunyesindekiyle
    # karsilastiriliyor. Tutmazsa cikiyoruz -- baska bir sinavin
    # sayilarini bu tabloda okumak, bu projede bir kez oldu.
    try:
        __import__(ayar.veri_ad)
    except ModuleNotFoundError:
        print(f"  veri modulu {ayar.veri_ad!r} bu klasorde YOK -> veri_04 ile "
              "kuruluyor; "
              "OLCME IZI ile dogrulanacak")
        ayar = ayar.degistir(veri_ad="veri_04")
    v = M.veri_kur(ayar, yaz=lambda *x: None)
    _kn = PEN.kunye_oku(a.klasor)
    _iz_kayitli = _kn.get("olcme_izi")
    _iz_simdi = M.olcme_izi(M.olcme_listeleri(ayar, v))
    if _iz_kayitli and _iz_kayitli != _iz_simdi:
        raise SystemExit(
            f"\n!! OLCME IZI TUTMUYOR: kosu {_iz_kayitli}, simdi {_iz_simdi}."
            "\n   Bu klasordeki model BASKA bir sinavda egitilmis; asagidaki"
            "\n   tablo onunla kiyaslanamaz.")
    print(f"  olcme izi {_iz_simdi}"
          + ("  <- kosunun kunyesiyle AYNI" if _iz_kayitli else
             "  (kunyede iz YOK, dogrulanamadi)"))
    net = PEN.MODEL_SINIFI(ayar, v.vocab).to(M.DEV)
    net.load_state_dict(PEN.agirlik_ortalamasi([snap[s] for s in pen]))
    net.eval()

    par = np.asarray(v.par)
    adlar = v.par_ad[0]
    yok = [i for i, x in enumerate(adlar) if str(x) == "<YOK>"][0]
    oa = OD.OdulAyar(ayar.odul_zemin, ayar.odul_kisayol, ayar.odul_tip,
                     ayar.odul_e, ayar.odul_f, ayar.odul_g_aile, ayar.odul_h)
    pz = OD.Puanlayici(par, yok, v.facts, oa)
    PUAN = np.array([oa.zemin, oa.zemin, oa.kisayol, oa.yanlis_tip,
                     oa.e_menzil_disi, oa.f_menzil, oa.g_aile, oa.h_tam])
    tip = np.asarray(v.tip)
    TA = list(v.tip_ad)
    REL = list(V.kur(0)["iliski"])

    def isim(e):
        return " ".join(str(adlar[j]) for j in par[e] if str(adlar[j]) != "<YOK>")

    lst = {"ent": v.ent, "ent_arama": v.ent_arama, "comp": v.comp,
           "ent_yok": v.ent_yok}[a.bolme]
    L = M.olcme_listeleri(ayar, v)
    if a.bolme in L and L[a.bolme]:
        lst = L[a.bolme]               # olcme listesi varsa ONU kullan
    X, P, T = M.kodla_2hop(v, lst)
    ozne = np.array([z[0] for z in lst], np.int64)
    r1 = np.array([z[1] for z in lst], np.int64)
    r2 = np.array([z[2] for z in lst], np.int64)
    kopru = np.array([z[3] for z in lst], np.int64)
    dogru = np.array([z[4] for z in lst], np.int64)
    ksy = v.facts[ozne, r2]

    print(f"=== odul_tani_04   {ayar.ad}   bolme {a.bolme}   "
          f"{len(lst)} soru x {a.ornek} cekim")
    print(f"  pencere {pen[0]}-{pen[-1]}   odul_ac={ayar.odul_ac} "
          f"lr={ayar.lr}   merdiven {tuple(PUAN[[0,3,4,5,6,7]])}")
    if a.bolme == "ent":
        print("  !! BU BOLME TUTULMUS SINAV -- odul burada KOSMADI.")
    elif a.bolme == "ent_arama":
        print("  !! ODULUN KOSTUGU BOLME -- burada iyi olmak BEKLENIR.")

    G = a.ornek
    rs = np.random.RandomState(0)
    bas_hep, ent_hep, sira_hep = [], [], []
    with torch.no_grad():
        for i in range(0, len(X), 128):
            xb = torch.from_numpy(X[i:i + 128]).to(M.DEV)
            lg = net(xb).float()
            ar = torch.arange(xb.shape[0], device=M.DEV)[:, None]
            pb = torch.from_numpy(P[i:i + 128]).to(M.DEV)
            lo, hi = v.yuva_ara[0]
            ly = lg[ar, pb][:, :, lo:hi]                    # (B,3,n)
            pr = torch.softmax(ly, -1).cpu().numpy()
            b = slice(i, i + xb.shape[0])
            # DOGRU cevabin ORTAK sirasi (yuvalar bagimsiz -> log toplami)
            lp = torch.log_softmax(ly, -1).cpu().numpy()
            pu = sum(lp[:, j, par[:, j]] for j in range(3))  # (B, n_ent)
            ph = pu[np.arange(len(pu)), dogru[b]]
            sira_hep.append((pu > ph[:, None]).sum(1) + 1)
            u = np.empty((len(pr) * G, 3), np.int64)
            for g in range(G):
                for j in range(3):
                    c = pr[:, j].cumsum(1)
                    r = rs.rand(len(pr), 1)
                    u[g::G, j] = (c < r).sum(1).clip(0, pr.shape[2] - 1)
            rep = np.repeat(np.arange(len(pr)), G)
            bb, ee = basamakla(pz, u, ozne[b][rep], ksy[b][rep], dogru[b][rep])
            bas_hep.append(bb)
            ent_hep.append(ee)
    bas = np.concatenate(bas_hep)
    ent = np.concatenate(ent_hep)
    sira = np.concatenate(sira_hep)
    N = len(bas)

    print()
    print("=== 1) MERDIVEN DAGILIMI -- odul NEREDEN geliyor ===")
    print(f"  {'basamak':<34}{'pay':>9}{'odul':>8}{'KATKI':>9}")
    for k in range(8):
        pay = float((bas == k).mean())
        print(f"  {BASAMAK[k]:<34}{pay:>9.4f}{PUAN[k]:>8.2f}{pay*PUAN[k]:>9.4f}")
    print(f"  {'-'*60}")
    print(f"  {'ORTALAMA ODUL':<34}{'':>9}{'':>8}{float(PUAN[bas].mean()):>9.4f}")
    _tik = int(np.argmax(np.bincount(bas, minlength=8)))
    print(f"  EN COK TIKANDIGI YER: {BASAMAK[_tik]}  (%{100*(bas==_tik).mean():.1f})")
    _kapi = float((bas <= 1).mean())
    print(f"  BICIM KAPILARINDA takilan (KAPI 1/3): %{100*_kapi:.1f}"
          + ("   <- kapilar bugun BOS, gradyani onlar uretmiyor"
             if _kapi < 0.02 else "   <- kapilar CALISIYOR"))

    print()
    print("=== 2) NEREDE TIKANIYOR -- cevabin TIPINE gore ===")
    tc = np.repeat(tip[dogru], G)
    print(f"  {'tip':<8}{'n':>7}" + "".join(f"{x:>8}" for x in
                                            ("BICIM", "KSY", "TIP", "E", "F",
                                             "G", "H")) + f"{'ODUL':>9}")
    for t in range(len(TA)):
        m = tc == t
        if not m.any():
            continue
        bb = bas[m]
        pay = [float((bb <= 2).mean()), float((bb == 3).mean())] + \
              [float((bb == k).mean()) for k in (4, 5, 6, 7)]
        print(f"  {TA[t]:<8}{int(m.sum()):>7}" + "".join(f"{x:>8.3f}" for x in pay)
              + f"{float(PUAN[bb].mean()):>9.4f}")

    print()
    print("=== 2b) son ILISKIYE (r2) gore -- H orani ve odul ===")
    rc = np.repeat(r2, G)
    sat = []
    for r in sorted(set(r2.tolist())):
        m = rc == r
        if m.sum() < 30 * G // 8:
            continue
        sat.append((float(PUAN[bas[m]].mean()), REL[r], int(m.sum()),
                    float((bas[m] == 7).mean()), float((bas[m] == 2).mean()),
                    float((bas[m] <= 1).mean())))
    print(f"  {'r2':<11}{'n':>7}{'H (TAM)':>10}{'KISAYOL':>9}{'BICIM':>8}{'ODUL':>9}")
    for od, ad, n, h, ks_, kp in sorted(sat):
        print(f"  {ad:<11}{n:>7}{h:>10.4f}{ks_:>9.4f}{kp:>8.4f}{od:>9.4f}")

    print()
    print("=== 3) NIYE TUTTURAMADI -- dogru cevap NEREDE duruyor ===")
    print(f"  dogru cevabin ORTAK sirasi (1060 varlik icinde):")
    for et, q in (("en iyi %25", 25), ("ORTANCA", 50), ("en kotu %25", 75)):
        print(f"    {et:<12}{np.percentile(sira, q):>8.0f}")
    for k in (1, 5, 20, 100):
        print(f"    top-{k:<8}{float((sira <= k).mean()):>8.4f}")
    print("  -> sira YUKSEKSE bilgi modelde YOK; DUSUKSE var ama argmax'a")
    print("     cikmiyor demektir. Odulun yapabilecegi is ikincisidir.")

    print()
    print("=== 3b) H'ye ulasamayan sorularda model NE diyor ===")
    ust = ent.reshape(-1, G)
    kac = 0
    ornekler = []
    for i in range(len(lst)):
        if (bas.reshape(-1, G)[i] == 7).any():
            continue
        kac += 1
        if len(ornekler) < 6:
            c = collections.Counter(int(x) for x in ust[i] if x >= 0)
            enc = c.most_common(1)[0] if c else (-1, 0)
            ornekler.append((i, enc))
    print(f"  {kac}/{len(lst)} soruda {G} cekimin HICBIRI dogru degil")
    for i, (enc, adet) in ornekler:
        e, _r1, _r2, kp, cv = lst[i]
        _k = int(v.facts[e, _r2])
        nd = ("KISAYOL" if enc == _k else
              "KOPRU" if enc == kp else
              "2 adimda" if enc >= 0 and pz.menzil.h2[e, enc] else
              "menzil disi" if enc >= 0 else "gecerli varlik DEGIL")
        print(f"    {isim(e)} + {REL[_r1]} + {REL[_r2]}")
        print(f"       DOGRU {isim(cv):<22} kopru {isim(kp)}")
        print(f"       model {isim(enc) if enc >= 0 else '(uydurma)':<22}"
              f" {adet}/{G} kez   [{nd}]   dogrunun sirasi {sira[i]}")


if __name__ == "__main__":
    main()
