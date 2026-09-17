# -*- coding: utf-8 -*-
"""asama1_04 — model_04'in KENDI ASAMA-1 TESHISI. TEK BASINA DURUR.

Kullanici karari, 16 Eylul 2026: *"bunlarin hepsi model_04 folderi
altinda olmali. model_04 diger hicbir model ile ayni seyi
kullanmamali."*

`model_a/asama1.py`nin KOPYASI (uretici: scratchpad/kur_okuma00.py). Modeli
`ModelSade` ile kurar. Paylasilan surumde yapilan bir degisiklik buraya
GECMEZ; `test_04.py` ikisinin AYNI SEYI olctugunu her kosuda siniyor.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

# `asama1_b` bunu doldurur; None -> model_04.ModelSade. `pencere_a` ve
# `tani_a`daki kancanin AYNISI.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import taban_04 as M                                          # noqa: E402
import pencere_04 as P                                        # noqa: E402
from model_04 import ModelSade                                # noqa: E402

MODEL_SINIFI = ModelSade


def _tek_dongu(ayar, vocab, sd):
    """K=1 model. AGIRLIK AYNI -- dongulu transformerda bloklar
    PAYLASIMLI, yani "f_theta'yi bir kez uygulamak" tam olarak budur.
    Yeni parametre YOK, yeniden egitim YOK."""
    a1 = ayar.degistir(dongu=1)
    net = (MODEL_SINIFI or M.Model)(a1, vocab).to(M.DEV)
    net.load_state_dict(sd)
    net.eval()
    return net


def _jetonlar(v, e):
    return [int(t) for t in M._e(v, e)]


@torch.no_grad()
def olc(net1, net, v, lst, bs=256):
    """Her pozisyon icin: yuva-kisitli argmax koprunun j. jetonu mu?

    Doner: (poz, yuva) sekilli dogruluk matrisi + asama-2 cevap dogrulugu.
    """
    X, Pz, T = M.kodla_2hop(v, lst)
    kopru = np.array([_jetonlar(v, x[3]) for x in lst])
    ARA = v.yuva_ara[:Pz.shape[1]]
    n, tlen = X.shape
    tut = np.zeros((tlen, len(ARA)))
    as2 = np.zeros(n, bool)
    for i in range(0, n, bs):
        xb = torch.from_numpy(X[i:i + bs]).to(M.DEV)
        lg1 = net1(xb).float()
        lg2 = net(xb).float()
        kb = torch.from_numpy(kopru[i:i + bs]).to(M.DEV)
        for t in range(tlen):
            for j, (lo, hi) in enumerate(ARA):
                z = lg1[:, t, lo:hi]
                tut[t, j] += (z.argmax(-1) == kb[:, j] - lo).sum().item()
        d = None
        ar = torch.arange(xb.shape[0], device=M.DEV)
        tb = torch.from_numpy(T[i:i + bs]).to(M.DEV)
        pb = torch.from_numpy(Pz[i:i + bs]).to(M.DEV)
        for j, (lo, hi) in enumerate(ARA):
            z = lg2[ar, pb[:, j], lo:hi]
            e = z.argmax(-1) == tb[:, j] - lo
            d = e if d is None else (d & e)
        as2[i:i + xb.shape[0]] = d.cpu().numpy()
    return tut / n, as2.mean()


@torch.no_grad()
def gizli(net1, v, lst, poz, bs=256):
    """poz'daki nf(h) -- ILK LOOP'tan sonra, Phi UYGULANMADAN."""
    X, _, _ = M.kodla_2hop(v, lst)
    cik = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(M.DEV)
        # ILERI GECISI ELLE KURMA -- modelin kendi govdesini cagir.
        # Eski hali `net1.emb + net1.pos` + tek argumanli `blk(h)` idi;
        # ikisi de `model_a.Model`e ozgu ve `ModelSade`de YOK. Bu satir
        # 16 Eylul hakemliginde AttributeError ile yakalandi.
        cik.append(net1.govde(xb)[:, poz].cpu())
    return torch.cat(cik).numpy()


@torch.no_grad()
def ornek_bazli(net1, v, lst, poz, bs=256):
    """poz'da ORNEK BASINA: kopru yuva0 dogru mu (bool dizi)."""
    X, _, _ = M.kodla_2hop(v, lst)
    lo, hi = (v.yuva_ara[0] if v.par is not None
              else (v.ent_off, v.ent_off + v.n_ent))
    kop = np.array([_jetonlar(v, x[3])[0] for x in lst])
    ok = np.zeros(len(X), bool)
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(M.DEV)
        z = net1(xb).float()[:, poz, lo:hi]
        h = torch.from_numpy(kop[i:i + bs]).to(M.DEV) - lo
        ok[i:i + xb.shape[0]] = (z.argmax(-1) == h).cpu().numpy()
    return ok


def sonda(q, v, lst, yaz=print):
    """LINEER SONDA: kopru, hidden state'ten DOGRUSAL olarak cikarilabiliyor
    mu? Yuva basina ayri.

    (e, r1) CIFTLERINE GORE AYRIK bolme -- ayni cift hem egitimde hem
    sinavda olsaydi sonda GRAFI EZBERLERDI ve her zaman yuksek cikardi.

    Kiyas SANS degil EN SIK SINIF: slot 2 cogunlukla <YOK> dolgusu, sans
    0.50 ama en sik sinif 0.95 -- sansla kiyaslayan "bilgi var" der.
    Ilk yazimda tam bunu yaptim ve slot 0 icin YANLIS etiket bastim.

    IKINCI TABAN -- KOPYA (16 Eylul, model_c sonrasi eklendi). En sik
    sinif da YETMIYOR: kopru cogu zaman soru varligiyla AYNI AILEDEN
    ("Ahmet Kilic --cocuk--> Huseyin Kilic"), yani slot 1 (soyad)
    GIRDIDE zaten duruyor. Sonda o slotu "cozunce" kopruyu bildigi
    icin degil, girdiden KOPYALADIGI icin cozmus olabilir. OLCULDU:
    comp'ta kopya orani 0.4450 iken sonda 0.3780 cikti -- yani sonda
    TRIVIAL kopyanin bile ALTINDA, ama en sik sinifa (0.0976) gore
    "BILGI VAR" yaziyordu. Taban artik max(en_sik, kopya)."""
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:                                    # pragma: no cover
        yaz("    (sklearn yok, sonda atlandi)")
        return {}
    import collections
    lo = v.yuva_ara[0][0] if v.par is not None else v.ent_off
    kop = np.array([[int(t) for t in M._e(v, x[3])] for x in lst])
    cift = np.array([hash((x[0], x[1])) % 5 for x in lst])
    tr, te = cift != 0, cift == 0
    yaz(f"    egitim {int(tr.sum())} / sinav {int(te.sum())}  "
        "((e,r1) AYRIK)")
    # KOPYA TABANI: soru varliginin ayni slot'u kopruye ESIT mi. Model
    # hicbir sey bilmeden bunu yapabilir; sonda da yapabilir.
    ent_ = np.array([[int(t) for t in M._e(v, x[0])] for x in lst])
    yaz(f"    {'slot':<6}{'aday':>6}{'SONDA':>9}{'EN SIK':>9}{'KOPYA':>9}"
        f"{'sans':>8}   hukum")
    out = {}
    for j in range(kop.shape[1]):
        y = kop[:, j] - lo
        ns = len(set(y.tolist()))
        if ns < 2 or te.sum() < 20:
            continue
        clf = LogisticRegression(max_iter=300).fit(q[tr], y[tr])
        acc = float(clf.score(q[te], y[te]))
        c = collections.Counter(y[te].tolist())
        sik = c.most_common(1)[0][1] / int(te.sum())
        kpy = float((ent_[te, j] == kop[te, j]).mean())
        taban = max(sik, kpy)
        hukum = ("TABAN>0.5 (%s), bilgi DEGIL" % ("kopya" if kpy >= sik
                                                 else "dolgu")
                 if taban > 0.5 else
                 "BILGI VAR" if acc > 2 * taban else "bilgi YOK")
        out[str(j)] = dict(aday=ns, sonda=acc, en_sik=sik, kopya=kpy,
                           taban=taban, hukum=hukum)
        yaz(f"    {j:<6}{ns:>6}{acc:>9.4f}{sik:>9.4f}{kpy:>9.4f}"
            f"{1/ns:>8.4f}   {hukum}")
    return out


def uzunluga_gore(v, lst, mat, poz, yaz=print):
    """ASAMA-1, KOPRUNUN token sayisina gore. Ayni model, ayni kosu --
    confound YOK. Hipotez: Phi pozisyon-yerel oldugu icin COK token'li
    kopruyu enjekte edemez; dogruysa TEK token'li koprulerde daha
    yuksek cikmali."""
    import collections
    if v.par is None:
        return {}
    uz = [sum(1 for j in range(v.yuva)
              if v.par_ad[j][int(v.par[x[3], j])] != "<YOK>") for x in lst]
    tip = [v.tip_ad[v.tip[x[3]]] for x in lst]
    yaz(f"    KOPRUNUN token sayisina gore (poz {poz}):")
    yaz(f"    {'kopru':<9}{'n':>6}{'ASAMA-1':>10}   kopru tipi")
    out = {}
    for u in sorted(set(uz)):
        ix = [i for i, x in enumerate(uz) if x == u]
        t = collections.Counter(tip[i] for i in ix)
        d = float(mat[ix].mean())
        out[str(u)] = dict(n=len(ix), asama1=d)
        yaz(f"    {u} token{'':<2}{len(ix):>6}{d:>10.4f}   "
            + "/".join(f"{k}:{n}" for k, n in t.most_common(3)))
    yaz("    ^ TIP ile KARISIK: tek token'li koprular SEHIR/DERS, aday")
    yaz("      kumesi kucuk. Tek basina hukum vermez.")
    return out


@torch.no_grad()
def head_teshisi(net, v, poz_h, yaz=print):
    """model_c: her head NE cozuyor, karisim agirliklari NE?

    Tasarim "her head bir yuva cozsun" diyor ama KAYIPTA bunu zorlayan
    HICBIR SEY YOK -- kayip yalniz son cevabi goruyor. Head'ler cokerse
    null sonuc YORUMLANAMAZ: hipotez mi yanlis, uygulama mi tutmadi,
    ayrilamaz. Bu yuzden olculuyor."""
    if not getattr(net, "cok_bas", False):
        return {}
    m = net.kafa.shape[0]
    W = net.emb.weight
    q = net.nf(poz_h)                                  # (N, d)
    yaz(f"    {m} head, {len(q)} pozisyon ornegi")
    gs, ss, tepe = [], [], []
    for j in range(m):
        p = torch.softmax((q @ net.kafa[j].T) @ W.T / net.ayar.dar_tau, -1)
        gs.append(p @ W)
        ss.append((gs[j] * net.kar_w[j]).sum(-1) + net.kar_b[j])
        tepe.append(p.argmax(-1))
    a = torch.softmax(torch.stack(ss, -1), -1)         # (N, m)
    out = dict(karisim=[float(x) for x in a.mean(0)])
    yaz(f"    karisim agirligi (ortalama): "
        + "  ".join(f"h{j}={a[:, j].mean():.3f}" for j in range(m)))
    yaz(f"    karisim SACILIMI (std)     : "
        + "  ".join(f"h{j}={a[:, j].std():.3f}" for j in range(m)))
    ayni = [[float((tepe[i] == tepe[j]).float().mean())
             for j in range(m)] for i in range(m)]
    out["ayni_token"] = ayni
    yaz("    head'ler AYNI token'i mi seciyor (1.0 = tamamen COKMUS):")
    for i in range(m):
        yaz("      " + "  ".join(f"{ayni[i][j]:.3f}" for j in range(m)))
    cok = max(ayni[i][j] for i in range(m) for j in range(m) if i != j)
    out["coktu"] = bool(cok > 0.9)
    yaz(f"    -> {'!! HEAD''LER COKTU' if cok > 0.9 else 'head''ler AYRI'}"
        f"  (en buyuk ortusme {cok:.3f})")
    return out


def birim_teshisi(net, v, lst, yaz=print):
    """VARLIK BIRIM MI: soru varliginin ICINDEKI next-token gecisi.

    `tam_kayip` (model_b10) bunu iddia ediyor: kayip HER pozisyonda
    hesaplanirsa model 'Ahmet' -> 'Yilmaz' gecisini de ogrenir, yani
    varligi BIRIM olarak baglar. Iddia edilen sey kuruldu mu, olculur.

    model_b6/b9'da bu pozisyonlar HIC gradyan ALMADI -> oradaki tahmin
    rastgeledir. Kiyas SANS DEGIL, EN SIK KOSULLU SINIF: 'Ahmet' 14 ayri
    kisi, tam tahmin ZATEN IMKANSIZ; tavan verinin kendi kosullu
    dagilimidir. (Ayni hataya sonda'da iki kez dustuk: once sansla,
    sonra en sik sinifla kiyasladik -- her ikisi de yanlis tabandi.)
    """
    if v.par is None or v.yuva < 2:
        return {}
    import collections
    X, _, _ = M.kodla_2hop(v, lst)
    xb = torch.from_numpy(X).to(M.DEV)
    with torch.no_grad():
        lg = net(xb)
    lo, hi = v.yuva_ara[0]
    tah = (lg[:, :, lo:hi].argmax(-1) + lo).cpu().numpy()

    # VERI TAVANI: onceki yuvalar verilince en sik gelen yuva
    sik = [collections.defaultdict(collections.Counter)
           for _ in range(v.yuva)]
    for e in range(v.n_ent):
        satir = tuple(int(v.par[e, j]) for j in range(v.yuva))
        for j in range(1, v.yuva):
            sik[j][satir[:j]][satir[j]] += 1

    yaz(f"    {'yuva':<6}{'MODEL':>9}{'VERI TAVANI':>13}   hukum")
    out = {}
    for j in range(1, v.yuva):
        # poz j, X[:, j+1]'i tahmin eder (varligin j. yuvasi)
        d = float((tah[:, j] == X[:, j + 1]).mean())
        t = 0.0
        for x in lst:
            satir = tuple(int(v.par[x[0], q]) for q in range(v.yuva))
            c = sik[j][satir[:j]]
            t += int(c.most_common(1)[0][0] == satir[j])
        t /= max(1, len(lst))
        hukum = ("TAVANDA" if d >= t - 1e-9 else
                 "BIRIM OGRENILDI" if d > 0.5 * t else "ogrenilmedi")
        out[str(j)] = dict(model=d, tavan=t, hukum=hukum)
        yaz(f"    {j:<6}{d:>9.4f}{t:>13.4f}   {hukum}")
    return out


def bas(ad, mat, as2, v, yaz=print):
    """EN IYI pozisyon + SORU SONU pozisyonu. IKISI DE basilir.

    !! 16 Eylul, hakemlikte bulundu: "en iyi poz" TEK BASINA
    KODLAMALAR ARASINDA KIYASLANAMAZ. `ek_kip="tr"` + `bicim>=3`
    kollarinda 1-hop BILDIRIM bicimi ("Ayse'nin annesi Fatma'dir")
    modele `e ' <NIN> r <SI>` onekinden sonra KOPRUYU yaz diye
    ogretiyor -- ve 2-hop sorusunun ilk alti jetonu bununla BIREBIR
    AYNI. Tarama o pozisyonu buluyor ve fiilen `one`i raporluyor.

    OLCULDU (model_b15, 12000-16000 penceresi):
        poz        6      7      8      9     10     ASAMA-2
        comp    1.000  0.000  0.000  0.033  0.025    0.0400
        ood     1.000  0.000  0.000  0.029  0.033    0.0143
    Kopru hop sinirinda 1.000, BIR JETON sonra 0.000.

    KIYASLANABILIR olan SORU SONU pozisyonudur: cevabin ILK jetonunu
    tahmin eden pozisyon. Eski kodlamada o poz 6 (QM), ek_kip'te poz
    10. `model_b13`/`b14`te orada 0.0275 cikmisti.
    """
    eniyi = int(mat[:, 0].argmax())
    # SORU SONU = ilk cevap jetonunun BIR ONCESI. `kopru_hedefi` DEGIL
    # (o r1/r2 pozisyonlarini veriyor).  2-hop bicim 0:
    #   eksiz  [S2] e(yuva) r1 r2 ?  -> 3 + yuva
    #   ek_kip e(yuva) ' <NIN> r1 <SI> <NIN> r2 <SI> ?  -> yuva + 7
    _son = min(mat.shape[0] - 1, (v.yuva + 7) if v.ek_kip else (3 + v.yuva))
    yaz(f"\n  {ad}")
    yaz(f"    ASAMA-1 kopru YUVA 0 (ayirt edici):  "
        f"en iyi poz {eniyi} -> {mat[eniyi, 0]:.4f}")
    yaz(f"    SORU SONU poz {_son} -> {mat[_son, 0]:.4f}"
        + ("   <- KODLAMALAR ARASI KIYASTA BU OKUNUR" if v.ek_kip else ""))
    if v.ek_kip and eniyi != _son:
        yaz("    !! en iyi poz, BILDIRIM biciminin OGRETTIGI yer olabilir")
        yaz("       (1-hop bildiriminde cevap tam orada baslar).")
    if v.yuva > 1:
        yaz(f"    (ayni pozda diger yuvalar: "
            + "  ".join(f"j{j}={mat[eniyi, j]:.4f}"
                        for j in range(1, mat.shape[1]))
            + "   <- dolgu yuvasi ALDATIR)")
    yaz(f"    ASAMA-2 cevap dogrulugu:             {as2:.4f}")
    return dict(en_iyi_poz=eniyi, asama1_yuva0=float(mat[eniyi, 0]),
                soru_sonu_poz=int(_son),
                asama1_soru_sonu=float(mat[_son, 0]),
                asama1_hepsi_poz=[[float(x) for x in r] for r in mat],
                asama2=float(as2))


def tara_bas(mat, X0, v, yaz=print):
    yaz(f"    {'poz':>4} {'girdi':<8}"
        + "".join(f"{'kopru j' + str(j):>11}" for j in range(mat.shape[1])))
    for t in range(mat.shape[0]):
        g = int(X0[t])
        ad = ("[S2]" if g == M.Q2 else "[S1]" if g == M.Q1 else
              "?" if g == M.QM else "<SON>" if g == M.EOS else
              "<PAD>" if g == M.PAD else
              "@iliski" if M.SPECIAL <= g < M.SPECIAL + v.n_rel else "jeton")
        yaz(f"    {t:>4} {ad:<8}"
            + "".join(f"{mat[t, j]:>11.4f}" for j in range(mat.shape[1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("klasor")
    ap.add_argument("--genislik", type=int, default=5)
    ap.add_argument("--tara", action="store_true",
                    help="BUTUN pozisyonlarin tablosunu bas")
    ap.add_argument("--bolme", default="ood,ent,comp,seen")
    ap.add_argument("--sonda", action="store_true",
                    help="LINEER SONDA: kopru hidden state'te DOGRUSAL mi")
    ap.add_argument("--head", action="store_true",
                    help="model_c: head'ler AYRISTI mi, karisim agirliklari")
    ap.add_argument("--birim", action="store_true",
                    help="model_b10: varlik BIRIM olarak ogrenildi mi")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--cikti", default=None)
    a = ap.parse_args()

    ayar = P.ayar_oku(a.klasor)
    kunye = P.kunye_oku(a.klasor)
    snap = P.anlik_goruntuler(a.klasor)
    adimlar = list(snap)
    g = min(a.genislik, len(adimlar))
    pen = adimlar[-g:]
    print(f"=== asama1  {ayar.ad} tohum {ayar.tohum} ===   cihaz {M.DEV}")
    P.kunye_bas(kunye)
    print(f"  pencere {pen[0]}-{pen[-1]} ({g} anlik goruntu)")

    v = M.veri_kur(ayar, yaz=lambda *x: None)
    L = M.olcme_listeleri(ayar, v)
    iz = M.olcme_izi(L)
    _k = (kunye or {}).get("olcme_izi")
    if _k and _k != iz:
        raise SystemExit(f"!! OLCME SETI DEGISMIS: egitim {_k}, asama1 {iz}")
    print(f"  parmak izi {iz}   yuva {v.yuva}   vocab {v.vocab}")

    sd = P.agirlik_ortalamasi([snap[x] for x in pen])
    net = (MODEL_SINIFI or M.Model)(ayar, v.vocab).to(M.DEV)
    net.load_state_dict(sd)
    net.eval()
    net1 = _tek_dongu(ayar, v.vocab, sd)

    sonuc = {}
    for bol in a.bolme.split(","):
        lst = (L.get(bol) or [])[:a.n]
        if not lst:
            continue
        mat, as2 = olc(net1, net, v, lst)
        sonuc[bol] = bas(f"{bol}  n={len(lst)}", mat, as2, v)
        ep = sonuc[bol]["en_iyi_poz"]
        if a.tara:
            tara_bas(mat, M.kodla_2hop(v, lst[:1])[0][0], v)
        # --- KOPRU UZUNLUGUNA GORE (ayni model, confound YOK) ----------
        if v.par is not None:
            ob = ornek_bazli(net1, v, lst, ep)
            sonuc[bol]["uzunluk"] = uzunluga_gore(v, lst, ob, ep)
        # --- LINEER SONDA ----------------------------------------------
        if a.sonda:
            print("    -- LINEER SONDA --")
            q = gizli(net1, v, lst, ep)
            sonuc[bol]["sonda"] = sonda(q, v, lst)
        # --- BIRIM TESHISI (model_b10) ---------------------------------
        if a.birim:
            print("    -- BIRIM TESHISI (varligin ICINDEKI gecis) --")
            sonuc[bol]["birim"] = birim_teshisi(net, v, lst)
        # --- HEAD TESHISI (model_c) ------------------------------------
        if a.head and getattr(net, "cok_bas", False):
            print("    -- HEAD TESHISI --")
            X0, _, _ = M.kodla_2hop(v, lst)
            with torch.no_grad():
                # (Bu dal `cok_bas` modellere ozgu -- ModelSade'de
                # CALISMAZ ve zaten `getattr(net,"cok_bas",False)` ile
                # kapali. Yine de elle ileri gecis BIRAKILMIYOR.)
                hh = net1.govde(torch.from_numpy(X0).to(M.DEV))
            sonuc[bol]["head"] = head_teshisi(net, v, hh[:, ep])

    yol = a.cikti or os.path.join(
        a.klasor, f"asama1_{ayar.ad}_t{ayar.tohum}_g{g}.json")
    M._yaz_json(yol, dict(ad=ayar.ad, tohum=ayar.tohum, pencere=pen,
                          genislik=g, parmak_izi=iz, yuva=v.yuva,
                          sonuc=sonuc))
    print(f"\n-> {yol}")


if __name__ == "__main__":
    main()
