# -*- coding: utf-8 -*-
"""tani_a — model_a ailesinin TANI aracı.  Yeni egitim GEREKTIRMEZ.

    python tani_a.py <klasor> [--genislik 5] [--cikti out.json]

Sordugu soru: `ent`te model YANLIS cevap verdiginde NE diyor?

`pencere_a` uc sayi veriyor -- DOGRU, KISAYOL, ve geri kalan. Olculdu
(15 Eylul, model_a t0): kazancin %72'si kisayol havuzundan geliyor ama
"geri kalan" %35'te KIPIRDAMIYOR (0.365 -> 0.342, alti pencere boyunca).
Tek sabit duran havuz o, ve ne oldugu bilinmiyor. Bu dosya onu aciyor.

--------------------------------------------------------------------------
KATEGORILER  --  zincir (e, r1, r2, b, a),  b = facts[e,r1], a = facts[b,r2]

    DOGRU          pred == a
    KOPRU          pred == b            <- KOPRUYU BULDU, ORADA DURDU
    KISAYOL        pred == facts[e,r2]  <- kopruyu atladi
    VARLIK         pred == e            <- soruyu kopyaladi
    KOPRU_YANLIS   pred == facts[b, r], r != r2
                                        <- kopruye ULASTI, YANLIS ILISKI
    VARLIK_BASKA   pred == facts[e, r]  <- varliga baska bir iliski uyguladi
    ILGISIZ        hicbiri

Sira ONEMLI: kategoriler bu sirayla denenir, ilk tutan kazanir. Ornegin
`a` zaten `facts[b, :]` icindedir; DOGRU once bakildigi icin KOPRU_YANLIS'a
dusmez.

Ayrica DOGRU olmayan orneklerde dogru cevabin SIRASI raporlanir: model
2. sirada diyorsa "bilmiyor" ile "az kaldi" ayrilir.

--------------------------------------------------------------------------
YAPISAL KORUMALAR -- pencere_a'dakiyle AYNI, cunku ONDAN IMPORT EDILIYOR

`ayar_oku`, `anlik_goruntuler`, `agirlik_ortalamasi` kopyalanmadi;
`pencere_a`dan geliyor. Ayni klasore iki arac bakiyorsa ayni agirligi ayni
sekilde kurmalari MEKANIK olmali, iddia degil.
"""
from __future__ import annotations

import argparse, os, sys

# `tani_b` bunu doldurur; None -> model_a.Model. `pencere_a`daki kancanin
# AYNISI. 16 Eylul: burada YOKTU ve `tani_a`, model_b anlik goruntusunu
# "Unexpected key(s): dar_norm.g" ile REDDEDERDI.
MODEL_SINIFI = None

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_a as M
import pencere_a as P

KATEGORI = ("DOGRU", "KOPRU", "KISAYOL", "VARLIK", "KOPRU_YANLIS",
            "VARLIK_BASKA", "ILGISIZ")


@torch.no_grad()
def tahmin_ve_sira(net, v: M.Veri, lst, bs=512):
    """Her ornek icin (tahmin edilen varlik id, dogru cevabin SIRASI).

    Sira 0 = model dogru cevabi ilk siraya koydu. Varlik-kisitli:
    iliski/ozel token'lar yarismaya SOKULMAZ (dogruluk() ile ayni kural)."""
    X, Pz, T = M.kodla_2hop(v, lst)
    lo, hi = v.ent_off, v.ent_off + v.n_ent
    onceki = net.training
    net.eval()
    tah, sira = [], []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(M.DEV)
        with torch.autocast(M.DEV, dtype=torch.float16,
                            enabled=(M.DEV == "cuda")):
            lg = net(xb)
        idx = torch.from_numpy(Pz[i:i + bs]).to(M.DEV)
        ar = torch.arange(len(idx), device=M.DEV)
        lg = lg.float()[ar, idx][:, lo:hi]
        g = torch.from_numpy(T[i:i + bs]).to(M.DEV) - lo
        tah.append(lg.argmax(-1).cpu().numpy())
        # dogru cevaptan KESIN BUYUK kac varlik var -> onun sirasi
        dogru_skor = lg.gather(1, g[:, None])
        sira.append((lg > dogru_skor).sum(-1).cpu().numpy())
    net.train(onceki)
    return np.concatenate(tah), np.concatenate(sira)


def ayristir(v: M.Veri, lst, tah: np.ndarray) -> list:
    """Her ornegi tek bir kategoriye koyar (yukaridaki siraya gore)."""
    out = []
    for (e, r1, r2, b, a), p in zip(lst, tah):
        ksy = int(v.facts[e, r2])
        if p == a:
            k = "DOGRU"
        elif p == b:
            k = "KOPRU"
        elif ksy >= 0 and p == ksy:
            k = "KISAYOL"
        elif p == e:
            k = "VARLIK"
        elif (v.facts[b] == p).any():
            k = "KOPRU_YANLIS"
        elif (v.facts[e] == p).any():
            k = "VARLIK_BASKA"
        else:
            k = "ILGISIZ"
        out.append(k)
    return out


def ozet(ad: str, kat: list, sira: np.ndarray, yaz=print) -> dict:
    n = len(kat)
    say = {k: kat.count(k) for k in KATEGORI}
    yaz(f"\n  {ad}   n={n}")
    for k in KATEGORI:
        if say[k]:
            yaz(f"    {k:<14}{say[k]:>6}   {say[k]/n:6.3f}")
    yanlis = sira[np.array([k != "DOGRU" for k in kat])]
    d = dict(n=n, oran={k: say[k] / n for k in KATEGORI}, say=say)
    if len(yanlis):
        for esik in (1, 2, 5, 10):
            d[f"yanlis_sira_alt_{esik}"] = float((yanlis <= esik).mean())
        d["yanlis_sira_ortanca"] = float(np.median(yanlis))
        yaz(f"    -- YANLIS olan {len(yanlis)} ornekte dogru cevabin SIRASI:")
        yaz(f"       ortanca {np.median(yanlis):.0f}. sirada   "
            f"ilk 2'de %{100*(yanlis <= 1).mean():.1f}   "
            f"ilk 3'te %{100*(yanlis <= 2).mean():.1f}   "
            f"ilk 11'de %{100*(yanlis <= 10).mean():.1f}")
    return d


def hedef_uzayi(v: M.Veri, lst, tah: np.ndarray, yaz=print) -> dict:
    """YANLIS cevaplar DOGRU UZAYDA mi?

    Ilk ayrisim "ILGISIZ" havuzunun buyuk oldugunu gosterdi (ENT'te 0.318):
    tahmin edilen varlik ne kopruyle ne soru varligiyla graf iliskisi olan
    bir sey. Ama "ilgisiz" demek "rastgele" demek DEGIL. Iki soru:

      TIP      tahmin, dogru cevapla AYNI TIPTE mi? (KISI/OKUL/SEHIR/DERS)
      MENZIL   tahmin, r2'nin CIKTI KUMESINDE mi? Yani grafta bir yerde
               `facts[x, r2] == tahmin` olan bir x var mi?

    Ikisi de SANS SEVIYESIYLE birlikte raporlanir; cunku KISI tipi
    varliklarin %66'si zaten KISI'dir ve "dogru tipi buldu" demek ancak
    sanstan yukarideyse bir sey ifade eder."""
    yanlis = [(x, int(p)) for x, p in zip(lst, tah) if p != x[4]]
    if not yanlis:
        return {}
    tip = v.tip
    # menzil[r] = r iliskisinin cikti kumesi
    menzil = {}
    for _, _, r2, _, _ in lst:
        if r2 not in menzil:
            s = v.facts[:, r2]
            menzil[r2] = set(int(x) for x in np.unique(s[s >= 0]))

    tip_ayni = np.mean([tip[p] == tip[x[4]] for x, p in yanlis])
    # SANS: rastgele bir varlik, dogru cevapla ayni tipte olma olasiligi
    tip_pay = np.bincount(tip, minlength=len(v.tip_ad)) / len(tip)
    tip_sans = np.mean([tip_pay[tip[x[4]]] for x, _ in yanlis])

    men_ic = np.mean([p in menzil[x[2]] for x, p in yanlis])
    men_sans = np.mean([len(menzil[x[2]]) / v.n_ent for x, _ in yanlis])

    yaz(f"    -- YANLIS {len(yanlis)} ornekte tahmin NEREDE:")
    yaz(f"       dogru cevapla AYNI TIP : %{100*tip_ayni:5.1f}   "
        f"(sans %{100*tip_sans:.1f})")
    yaz(f"       r2'nin MENZILINDE      : %{100*men_ic:5.1f}   "
        f"(sans %{100*men_sans:.1f})")
    return dict(n_yanlis=len(yanlis), tip_ayni=float(tip_ayni),
                tip_sans=float(tip_sans), menzil_ic=float(men_ic),
                menzil_sans=float(men_sans))


def iliski_karsilastir(v: M.Veri, L: dict, yaz=print) -> dict:
    """ENT ile ENT-YOK ayni iliskilerden mi olusuyor?

    ENT-YOK'ta dogruluk ENT'ten DUSUK cikti (0.273 < 0.424). Bu "kisayol
    engel degil" diye okunabilir -- AMA iki kume farkli zincir tipinden
    (AYIRT / YOK) geliyor, yani iliski karisimi farkli olabilir. Karisim
    farkliysa kiyas GECERSIZ. Burada olculuyor."""
    def dag(lst, i):
        c = np.bincount([x[i] for x in lst], minlength=v.n_rel)
        return c / max(1, c.sum())
    out = {}
    for i, ad in ((1, "r1"), (2, "r2")):
        a, b = dag(L["ent"], i), dag(L["ent_yok"], i)
        # toplam varyasyon uzakligi: 0 = ayni karisim, 1 = hic ortusmuyor
        tv = float(0.5 * np.abs(a - b).sum())
        out[ad] = tv
        yaz(f"    {ad}: toplam varyasyon uzakligi {tv:.3f}"
            + ("   <- KARISIM FARKLI, kiyas gecersiz" if tv > 0.2 else
               "   <- karisim benzer"))
    ortak = sum(1 for x in L["ent_yok"] if x[2] in {y[2] for y in L["ent"]})
    out["ent_yok_r2_ent_icinde"] = ortak / max(1, len(L["ent_yok"]))
    return out


def zincir_testi(net, v: M.Veri, lst, ad: str, yaz=print) -> dict:
    """Model iki olguyu AYRI AYRI biliyor mu, birlestiremiyor mu?

    ENT zincirinin (e, r1, r2, b, a) iki atomik olgusu da egitimde VAR
    (olculdu: 6665/6665). Ikisini TEK TEK soruyoruz, [Q1] cercevesinde:

        1. HOP    [Q1] e  r1 ?  -> b
        2. HOP    [Q1] b  r2 ?  -> a

    Ikisi de dogruysa ama [Q2] e r1 r2 ? yanlissa, arizanin yeri BELLI:
    model olgulari BILIYOR, BIRLESTIREMIYOR. Bu, "bilgi eksik" ile
    "kompozisyon eksik" arasindaki farki ayirir -- CLAUDE.md'nin
    'depth-local storage' tartismasinin tam ortasindaki soru."""
    h1 = [(e, r1, b) for e, r1, _, b, _ in lst]
    h2 = [(b, r2, a) for _, _, r2, b, a in lst]
    d1 = M.dogruluk(net, v, *M.kodla_1hop(v, h1))
    d2 = M.dogruluk(net, v, *M.kodla_1hop(v, h2))
    iki = M.dogruluk(net, v, *M.kodla_2hop(v, lst))
    yaz(f"\n  {ad}   n={len(lst)}")
    yaz(f"    1. HOP tek basina   [Q1] e  r1 ? -> b     {d1:.4f}")
    yaz(f"    2. HOP tek basina   [Q1] b  r2 ? -> a     {d2:.4f}")
    yaz(f"    IKISI BIRDEN        [Q2] e r1 r2 ? -> a   {iki:.4f}")
    yaz(f"    -> parcalar {min(d1, d2):.4f}'e kadar biliniyor, "
        f"birlesince {iki:.4f}.  KAYIP {min(d1, d2) - iki:+.4f}")
    return dict(hop1=d1, hop2=d2, iki_hop=iki, kayip=min(d1, d2) - iki)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("klasor")
    ap.add_argument("--genislik", type=int, default=5)
    ap.add_argument("--cikti", default=None)
    a = ap.parse_args()

    ayar = P.ayar_oku(a.klasor)
    kunye = P.kunye_oku(a.klasor)
    snap = P.anlik_goruntuler(a.klasor)
    adimlar = list(snap)
    print(f"=== tani_a  {ayar.ad} tohum {ayar.tohum} ===")
    P.kunye_bas(kunye)

    veri = M.veri_kur(ayar, yaz=lambda *x: None)
    L = M.olcme_listeleri(ayar, veri)
    iz = M.olcme_izi(L)
    _k = (kunye or {}).get("olcme_izi")
    if _k and _k != iz:
        raise SystemExit(f"!! OLCME SETI DEGISMIS: egitim {_k}, tani {iz}")
    print(f"  parmak izi {iz}" + ("  <- egitimle AYNI" if _k == iz else ""))

    g = min(a.genislik, len(adimlar))
    pen = adimlar[-g:]
    print(f"  pencere {pen[0]}-{pen[-1]} ({g} anlik goruntu) "
          f"-- pencere_a'nin SON penceresi")
    sd = P.agirlik_ortalamasi([snap[x] for x in pen])
    net = (MODEL_SINIFI or M.Model)(ayar, veri.vocab).to(M.DEV)
    net.load_state_dict(sd)
    net.eval()

    sonuc = {}
    print(f"\n{'='*66}\nENT CEVAPLARI NE? -- yanlis olanlar dahil hepsi")
    for ad in ("ent", "ent_yok", "comp", "seen"):
        if not L.get(ad):
            continue
        tah, sira = tahmin_ve_sira(net, veri, L[ad])
        kat = ayristir(veri, L[ad], tah)
        sonuc[ad] = ozet(ad, kat, sira)
        sonuc[ad]["uzay"] = hedef_uzayi(veri, L[ad], tah)

    print(f"\n{'='*66}\nOLGULARI BILIYOR MU, BIRLESTIREMIYOR MU?")
    for ad in ("ent", "comp", "seen"):
        if L.get(ad):
            sonuc[ad]["zincir"] = zincir_testi(net, veri, L[ad], ad)

    print(f"\n{'='*66}\nENT ile ENT-YOK ayni iliskilerden mi? "
          "(0.273 < 0.424 kiyasi gecerli mi)")
    sonuc["_iliski"] = iliski_karsilastir(veri, L)

    yol = a.cikti or os.path.join(
        a.klasor, f"tani_{ayar.ad}_t{ayar.tohum}_g{g}.json")
    M._yaz_json(yol, dict(ad=ayar.ad, tohum=ayar.tohum, pencere=pen,
                          genislik=g, parmak_izi=iz, kunye=kunye,
                          kategoriler=list(KATEGORI), sonuc=sonuc))
    print(f"\n-> {yol}")


if __name__ == "__main__":
    main()
