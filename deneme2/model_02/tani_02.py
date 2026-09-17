# -*- coding: utf-8 -*-
"""tani_02 — model_02'in KENDI AYRISTIRMASI. TEK BASINA DURUR.

Kullanici karari, 16 Eylul 2026: *"bunlarin hepsi model_02 folderi
altinda olmali. model_02 diger hicbir model ile ayni seyi
kullanmamali."*

`model_a/tani_a.py`nin KOPYASI (uretici: scratchpad/kur_okuma00.py). Modeli
`ModelSade` ile kurar. Paylasilan surumde yapilan bir degisiklik buraya
GECMEZ; `test_02.py` ikisinin AYNI SEYI olctugunu her kosuda siniyor.
"""
from __future__ import annotations

import argparse, os, sys

# `tani_b` bunu doldurur; None -> model_02.ModelSade. `pencere_a`daki kancanin
# AYNISI. 16 Eylul: burada YOKTU ve `tani_a`, model_b anlik goruntusunu
# "Unexpected key(s): dar_norm.g" ile REDDEDERDI.
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import taban_02 as M                                          # noqa: E402
import pencere_02 as P                                        # noqa: E402
from model_02 import ModelSade                                # noqa: E402

MODEL_SINIFI = ModelSade

KATEGORI = ("DOGRU", "KOPRU", "KISAYOL", "VARLIK", "KOPRU_YANLIS",
            "VARLIK_BASKA", "VARLIK_DEGIL", "ILGISIZ")
# VARLIK_DEGIL yalniz COK JETONLU kodlamada olabilir: yuva basina argmax
# HICBIR varliga denk gelmeyen bir jeton birlesimi uretebilir
# ("Ayse" + "Lisesi"). Tek jetonda bu kategori HEP BOS kalir.


def varlik_sozlugu(v: M.Veri):
    """(yuva jetonlari) -> varlik id.  Birebirlik `veri_kur`da ASSERT'li."""
    if v.par is None:
        return None
    return {tuple(int(x) for x in v.par[e]): e for e in range(v.n_ent)}


def yuva_araliklari(v: M.Veri, yuva: int):
    """Yuva basina (lo, hi). TEK KAYNAK `v.yuva_ara` -- `dogruluk()` ve
    kayip da onu kullaniyor, burada yeniden turetilmiyor."""
    if v.par is None:
        return [(v.ent_off, v.ent_off + v.n_ent)]
    return list(v.yuva_ara[:yuva])


def cevap_uzunlugu(v: M.Veri, e: int) -> int:
    """Varligin KAC jetonla yazildigi (<YOK> dolgusu sayilmaz)."""
    if v.par is None:
        return 1
    return sum(1 for j in range(v.yuva)
               if v.par_ad[j][int(v.par[e, j])] != "<YOK>")


@torch.no_grad()
def tahmin_ve_sira(net, v: M.Veri, lst, bs=512):
    """Her ornek icin (tahmin edilen varlik id, dogru cevabin SIRASI).

    Sira 0 = model dogru cevabi ilk siraya koydu. Varlik-kisitli:
    iliski/ozel token'lar yarismaya SOKULMAZ (dogruluk() ile ayni kural)."""
    X, Pz, _T = M.kodla_2hop(v, lst)
    ARA = yuva_araliklari(v, Pz.shape[1])
    soz = varlik_sozlugu(v)
    # EJ[e, j] = e varliginin j. yuvadaki jetonunun YUVA ICI indisi.
    EJ = (np.arange(v.n_ent, dtype=np.int64)[:, None] if v.par is None
          else v.par[:, :len(ARA)])
    EJt = torch.from_numpy(np.ascontiguousarray(EJ)).to(M.DEV)
    onceki = net.training
    net.eval()
    tah, sira = [], []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(M.DEV)
        with torch.autocast(M.DEV, dtype=torch.float16,
                            enabled=(M.DEV == "cuda")):
            lg = net(xb)
        idx = torch.from_numpy(Pz[i:i + bs]).to(M.DEV)
        ar = torch.arange(len(idx), device=M.DEV)[:, None]
        lgp = lg.float()[ar, idx]                      # (B, yuva, V)
        # TAHMIN: yuva basina KISITLI argmax -- `dogruluk()` ile AYNI
        # kural. Tek yuvada eski davranisin birebir aynisi.
        pj = torch.stack([lgp[:, j, lo:hi].argmax(-1)
                          for j, (lo, hi) in enumerate(ARA)], 1)
        pj = pj.cpu().numpy()
        if soz is None:
            tah.append(pj[:, 0])
        else:
            tah.append(np.array([soz.get(tuple(int(x) for x in r), -1)
                                 for r in pj], np.int64))
        # SIRA: varlik puani = yuvalarin LOG-OLASILIKLARI TOPLAMI. Kayip
        # yuvalari bagimsiz kabul ediyor, puan da oyle. Tek yuvada
        # log_softmax siralamayi DEGISTIRMEZ -> eski sira ile ayni.
        puan = None
        for j, (lo, hi) in enumerate(ARA):
            lp = torch.log_softmax(lgp[:, j, lo:hi], -1)     # (B, slot)
            p = lp[:, EJt[:, j]]                             # (B, n_ent)
            puan = p if puan is None else puan + p
        g = torch.as_tensor([x[4] for x in lst[i:i + bs]], device=M.DEV)
        dogru_skor = puan.gather(1, g[:, None])
        sira.append((puan > dogru_skor).sum(-1).cpu().numpy())
    net.train(onceki)
    return np.concatenate(tah), np.concatenate(sira)


def ayristir(v: M.Veri, lst, tah: np.ndarray) -> list:
    """Her ornegi tek bir kategoriye koyar (yukaridaki siraya gore)."""
    out = []
    for (e, r1, r2, b, a), p in zip(lst, tah):
        ksy = int(v.facts[e, r2])
        if p < 0:
            out.append("VARLIK_DEGIL")      # jeton birlesimi bir varlik DEGIL
            continue
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
    # p < 0 -> tahmin hicbir varliga denk gelmiyor; TIP ve MENZIL
    # tanimsiz. Sayisi ayrica raporlanir, ortalamaya KARISTIRILMAZ.
    dis = sum(1 for p in tah if int(p) < 0)
    yanlis = [(x, int(p)) for x, p in zip(lst, tah)
              if p != x[4] and int(p) >= 0]
    if not yanlis:
        return dict(varlik_degil=dis) if dis else {}
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
    if dis:
        yaz(f"       VARLIK DEGIL           : {dis} ornek (uzay disinda)")
    return dict(n_yanlis=len(yanlis), tip_ayni=float(tip_ayni),
                tip_sans=float(tip_sans), menzil_ic=float(men_ic),
                menzil_sans=float(men_sans), varlik_degil=dis)


def uzunluga_gore(v: M.Veri, lst, kat: list, ad: str, yaz=print) -> dict:
    """Ayni bolme, CEVABIN JETON SAYISINA gore ayrilmis.

    Onkayit model_b6.md 2.6 (BAGLAYICI, kosudan ONCE yazildi): `ood`
    homojen DEGIL -- orneklerin %26,1'inde cevap TEK jetonlu (SEHIR,
    DERS), yani cok-jetonlu baglama o orneklerde HIC SINANMIYOR. Tek
    sayi olarak okunursa kolay vakalar zor vakalari gizler.

    Tek jetonlu kodlamada butun cevaplar 1 jetonludur; tablo tek satir
    olur ve toplamla AYNI sayiyi verir -- yani bu rapor eski kollari
    degistirmez."""
    u = [cevap_uzunlugu(v, x[4]) for x in lst]
    d = {}
    yaz("")
    yaz(f"  {ad} -- CEVABIN JETON SAYISINA GORE  (onkayit model_b6 2.6)")
    yaz(f"    {'jeton':<7}{'n':>6}{'DOGRU':>9}{'KISAYOL':>9}{'KOPRU':>8}")
    for uz in sorted(set(u)):
        ix = [i for i, x in enumerate(u) if x == uz]
        n = len(ix)
        pay = {k: sum(1 for i in ix if kat[i] == k) / n
               for k in ("DOGRU", "KISAYOL", "KOPRU")}
        d[str(uz)] = dict(n=n, **pay)
        yaz(f"    {uz:<7}{n:>6}{pay['DOGRU']:>9.4f}"
            f"{pay['KISAYOL']:>9.4f}{pay['KOPRU']:>8.4f}")
    if len(d) > 1:
        e, z = d[str(min(u))]["DOGRU"], d[str(max(u))]["DOGRU"]
        yaz(f"    -> en KISA cevap {e:.4f}  vs  en UZUN cevap {z:.4f}"
            f"   fark {z - e:+.4f}")
    return d


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
    # ETIKET DIZIYE GORE. KUSUR (16 Eylul hakemligi): burada [Q1]/[Q2]
    # SABIT yaziliydi; `ek_kip="tr"` dizisinde oyle bir jeton YOK (rolu
    # POZISYON degil EK tasiyor). Sayilar dogruydu, ETIKET yanlisti --
    # rapora bakan kisi dizinin sekli hakkinda yanlis sey okurdu.
    if v.ek_kip:
        _e1 = "e ' <NIN> r1 <SI> ? -> b  "
        _e2 = "b ' <NIN> r2 <SI> ? -> a  "
        _ei = "e ' <NIN> r1 <SI> <NIN> r2 <SI> ? -> a"
    else:
        _e1, _e2, _ei = ("[Q1] e  r1 ? -> b         ",
                         "[Q1] b  r2 ? -> a         ",
                         "[Q2] e r1 r2 ? -> a")
    yaz(f"    1. HOP tek basina   {_e1}   {d1:.4f}")
    yaz(f"    2. HOP tek basina   {_e2}   {d2:.4f}")
    yaz(f"    IKISI BIRDEN        {_ei}   {iki:.4f}")
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
    # `ood` EKLENDI: model_b6'nin BIRINCIL olcusu o (bolme KENAR
    # duzeyinde, jetonlamadan etkilenmiyor). Once yoktu -- birincil
    # olcunun ayrisimi CIKMIYORDU.
    for ad in ("ent", "ent_yok", "ood", "comp", "seen"):
        if not L.get(ad):
            continue
        tah, sira = tahmin_ve_sira(net, veri, L[ad])
        kat = ayristir(veri, L[ad], tah)
        sonuc[ad] = ozet(ad, kat, sira)
        sonuc[ad]["uzay"] = hedef_uzayi(veri, L[ad], tah)
        if ad in ("ood", "ent"):
            sonuc[ad]["uzunluk"] = uzunluga_gore(veri, L[ad], kat, ad)

    print(f"\n{'='*66}\nOLGULARI BILIYOR MU, BIRLESTIREMIYOR MU?")
    for ad in ("ent", "ood", "comp", "seen"):
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
