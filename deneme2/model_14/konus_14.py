# -*- coding: utf-8 -*-
"""konus_14 -- MODELLE KONUSMA.  Sayi degil, modelin YAZDIGI.

Kullanici, 20 Eylul 2026: *"model ile konusabilecegim bir uygulama
yaz"*.

CLAUDE.md, calisma bicimi: *"METIN: sayi degil, MODELIN YAZDIGI
okunur"*. Bir gunde uc gosterim kusuru yalnizca metne bakilarak
bulundu; hicbirini sayisal bir kapi gostermedi.

    python konus_14.py

Yazdiginiz cumle KORPUSLA AYNI bolucuden gecer (`ek_14.bol`), ayni
birim sozlugune dusurulur ve model onu onek kabul edip devam eder.
Bolme ya da uretim burada AYRI bir yoldan yapilsaydi gordugunuz sey
olculen sey OLMAZDI -- ikisi de `model_14` / `ek_14`ten cagriliyor.

KOMUTLAR
    <cumle>        modele ver, devamini uret
    /soru [n]      sinavdan rastgele n soru, BEKLENEN cevabiyla
    /adim          adim adim okumayi ac/kapa  (her adimda en yakin uc)
    /isin N        isin genisligi. 0 = acgozlu
    /yeni N        en cok kac birim uretilsin
    /r X           capa yaricapi
    /birim <söz>   bir sozcugun nasil bolundugunu goster
    /ayar          yuklenen agirligin ayarini goster
    /yardim  /q
"""
from __future__ import annotations

import os
import random
import sys

import torch
import torch.nn.functional as F

import ayar_14 as AY
import birim_14 as BR
import ek_14 as EK
import model_14 as M

VARSAYILAN_BIRIM = r"G:/Drive'ım/model_14/birim_14.npz"
VARSAYILAN_AGIRLIK = r"G:/Drive'ım/model_14/t0/model_t0.pt"


def yukle(birim_yol: str, agirlik_yol: str, dev: str):
    b = BR.yukle(birim_yol, yaz=lambda *a: None)
    pk = torch.load(agirlik_yol, map_location=dev, weights_only=False)
    a = pk["ayar"]
    m = M.Yol(len(b), D=a["D_DURUM"], d=a["D_OKUMA"], K=a["K_KOD"],
              tam=M.sinif_ayir(b.say, a["K_TAM"]), saat=a["SAAT"]).to(dev)
    m.load_state_dict(pk["model"])
    m.eval()
    return b, m, a


def parcala(b: BR.Birim, w: str) -> tuple[list[str], bool]:
    """Bir sozcugun birimleri.  (parcalar, korpusta_var_mi)

    !! KORPUSUN KENDI TABLOSU ONCE. `b.bolme` korpus kurulurken
    yazildi; bolucu sonradan degisirse dogru olan O'dur, cunku model
    korpustaki bolmeyle egitildi. Olculdu: ikisi su an BIREBIR ayni
    (5 ornekte), ama ayrismalari MUMKUN ve o zaman ekranda gordugumuz
    sey modelin gordugu sey olmaz."""
    if w in b.bolme:
        return list(b.bolme[w]), True
    return EK.bol(w, b.kok, korunan=b.korunan, serbest=b.serbest), False


def bol(b: BR.Birim, satir: str):
    """Kullanici metni -> (birim indeksleri, bilinmeyen, korpus disi)."""
    ix, bilinmeyen, yeni = [], [], []
    for w in satir.split():
        p, korpusta = parcala(b, w)
        if not korpusta:
            yeni.append(w)
        for x in p:
            (ix.append(b.ix[x]) if x in b.ix else bilinmeyen.append(x))
    return ix, bilinmeyen, yeni


def ters_tablo(b: BR.Birim) -> dict:
    """birim dizisi -> KELIME.  `b.bolme`nin tersi.

    Model ekleri ayri birim goruyor (`Yılmaz'ın` = Yılmaz +
    -TAMLAYAN); ekranda okunmasi gereken sey Turkce cumle. Tabloyu
    korpusun KENDI bolmesinden cikariyoruz -- elle ek kurali
    yazsaydik ekranda gordugumuz sey modelin gordugu sey olmazdi.

    Ayni birim dizisine birden cok kelime dusuyorsa korpusta SIK
    olani secilir."""
    t = {}
    for w, p in b.bolme.items():
        k = tuple(p)
        f = min((b.say[b.ix[x]] for x in p if x in b.ix), default=0)
        if k not in t or f > t[k][1]:
            t[k] = (w, f)
    return {k: v[0] for k, v in t.items()}


def turkce(b: BR.Birim, ters: dict, dizi) -> str:
    """Birim dizisini KELIMELERE geri topla. En uzun eslesme."""
    d = [int(x) for x in dizi]
    en = max((len(k) for k in ters), default=1)
    cik, i = [], 0
    while i < len(d):
        for L in range(min(en, len(d) - i), 0, -1):
            k = tuple(b.ad[x] for x in d[i:i + L])
            if k in ters:
                cik.append(ters[k])
                i += L
                break
        else:
            cik.append(b.ad[d[i]])
            i += 1
    s = " ".join(cik)
    for n in (".", ",", "?", "!", ":", ";"):
        s = s.replace(" " + n, n)
    return s


@torch.no_grad()
def uret(m: M.Yol, b: BR.Birim, onek: list[int], n_yeni: int, isin: int,
         r: float, adim_adim: bool, yaz=print):
    """Onek zorlanir, sonrasi serbest.  `m.uret` ISIN aramasi yapar;
    acgozlu yol (isin=0) adim adim okumayi gosterebilsin diye ayri."""
    dur = tuple(b.ix[c] for c in ".?!" if c in b.ix)
    if isin and not adim_adim:
        yol, puan = m.uret(onek, en_cok=n_yeni, isin=isin, dur=dur, r=r)
        return yol[len(onek):], puan

    R, C, P = m.donme(), m.kod(), m.p
    z = P.new_zeros(1, m.D)
    z[:, :m.d] = P[onek[0]]
    yol, puan = [onek[0]], 0.0
    if adim_adim:
        yaz("   adim  girdi           |Pz|  capa   en yakin uc")
    for j in range(1, len(onek) + n_yeni):
        z = torch.bmm(R[yol[-1]][None], z.unsqueeze(-1)).squeeze(-1)
        yak2, k = (2 - 2 * (z @ C.T)).clamp(min=0).min(1)
        vur = bool(yak2[0] < r * r)
        if vur:
            z = C[k].clone()
        q = F.normalize(z[:, :m.d], dim=-1)
        v3, i3 = (q @ P.T)[0].topk(3)
        zorla = j < len(onek)
        sonra = onek[j] if zorla else int(i3[0])
        if adim_adim:
            yaz("   %4d  %-15s %5.3f %-5s  %s%s"
                % (j, b.ad[yol[-1]][:15], float(z[0, :m.d].norm()),
                   "CAPA" if vur else "-",
                   "  ".join("%s(%.2f)" % (b.ad[int(x)][:12], float(c))
                             for c, x in zip(v3, i3)),
                   "   <- onek" if zorla else ""))
        puan += float(2 - 2 * v3[0]) if not zorla else 0.0
        yol.append(sonra)
        if not zorla and sonra in dur:
            break
    return yol[len(onek):], puan


def sinav(b: BR.Birim):
    """Sinav sorulari -- olcumun KENDI kurdugu sorular, ayni koddan."""
    import olcme_14 as O
    import taban_14 as MT
    import veri_14 as V
    v = MT.veri_kur(AY.AYAR, yaz=lambda *a, **k: None)
    L = MT.olcme_listeleri(AY.AYAR, v)
    G = V.kur(AY.AYAR.veri_tohum)
    E_ad = [x for t in V.TIPLER for x in G["ad"][t]]
    soru_tip = {i: b.ix.get({"KISI": "kim", "SEHIR": "neresi",
                             "BOLGE": "neresi"}.get(t), b.ix["hangisi"])
                for i, t in enumerate(V.TIPLER)}
    S = O.Sorular(v, E_ad, list(V.ILISKI), V.TR, V.TR_ILISKI, b.kok,
                  b.ix, b.korunan, soru_tip)
    return S.tum(L, b.dizi, yaz=lambda *a: None)


def main() -> int:
    # Windows konsolu varsayilan cp1254; Turkce girdi/cikti bozuluyor.
    for ak in (sys.stdin, sys.stdout):
        try:
            ak.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    by = os.environ.get("BIRIM_NPZ", VARSAYILAN_BIRIM)
    ay = os.environ.get("AGIRLIK", VARSAYILAN_AGIRLIK)
    for yol, ne in ((by, "birim akisi"), (ay, "agirlik")):
        if not os.path.exists(yol):
            print("%s yok: %s" % (ne, yol))
            print("  BIRIM_NPZ / AGIRLIK ortam degiskeniyle yol verebilirsiniz.")
            return 1
    print("yukleniyor...  (%s)" % dev)
    b, m, a = yukle(by, ay, dev)
    ters = ters_tablo(b)
    print("model_14   %d birim   D=%d d=%d K=%d   parametre %s"
          % (len(b), a["D_DURUM"], a["D_OKUMA"], a["K_KOD"],
             format(M.n_par(m), ",")))
    print("agirlik    %s" % ay)
    print("/yardim  komutlar   |   /q  cikis\n")

    n_yeni, isin, r, adim = 12, 0, a["R_CAPA"], False
    sor = None
    while True:
        try:
            s = input("> ").strip().lstrip("﻿")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not s:
            continue
        if s in ("/q", "/cik", "/exit"):
            return 0
        if s == "/yardim":
            print(__doc__[__doc__.index("KOMUTLAR"):])
            continue
        if s == "/ayar":
            for k, x in a.items():
                print("   %-9s %s" % (k, x))
            print("   %-9s %s   isin %d   yeni %d   adim %s"
                  % ("(oturum)", "r=%.3f" % r, isin, n_yeni, adim))
            continue
        if s == "/adim":
            adim = not adim
            print("   adim adim okuma:", "ACIK" if adim else "kapali")
            continue
        if s.startswith("/isin"):
            isin = int(s.split()[1]) if len(s.split()) > 1 else 0
            print("   isin genisligi %d%s" % (isin, "  (acgozlu)" if not isin else ""))
            continue
        if s.startswith("/yeni"):
            n_yeni = int(s.split()[1])
            print("   en cok %d birim" % n_yeni)
            continue
        if s.startswith("/r"):
            r = float(s.split()[1])
            print("   capa yaricapi %.3f" % r)
            continue
        if s.startswith("/birim"):
            for w in s.split()[1:]:
                p, korpusta = parcala(b, w)
                e = EK.bol(w, b.kok, korunan=b.korunan, serbest=b.serbest)
                print("   %-22s -> %s%s" % (w, "  ".join(
                    "%s%s" % (x, "" if x in b.ix else " [SOZLUKTE YOK]")
                    for x in p), "" if korpusta else "   [KORPUSTA YOK]"))
                if korpusta and list(p) != list(e):
                    print("      !! bolucu AYRISMIS -- ek_14 simdi: %s"
                          % "  ".join(e))
            continue
        if s.startswith("/soru"):
            if sor is None:
                print("   sinav kuruluyor (bir kerelik, ~30 sn)...")
                sor = sinav(b)
                print("   OGRETILEN %d   CIKARIM %d"
                      % (len(sor["OGRETILEN"]), len(sor["CIKARIM"])))
            n = int(s.split()[1]) if len(s.split()) > 1 else 3
            for _ in range(n):
                grup = random.choice(["OGRETILEN", "CIKARIM"])
                q = random.choice(sor[grup])
                onek = [int(x) for x in q.onek]
                print("\n   [%s]  %s" % (grup, b.coz(onek)))
                print("   BEKLENEN  %s" % b.coz(q.cevap))
                cik, puan = uret(m, b, onek, n_yeni, isin, r, adim)
                print("   URETILEN  %s   (puan %.3f)" % (b.coz(cik), puan))
            continue
        if s.startswith("/"):
            print("   bilinmeyen komut. /yardim")
            continue

        onek, bilinmeyen, yeni = bol(b, s)
        if yeni:
            print("   KORPUSTA GECMEYEN sozcuk: %s" % "  ".join(yeni))
        if bilinmeyen:
            print("   SOZLUKTE YOK (atlandi): %s" % "  ".join(bilinmeyen))
        if not onek:
            print("   hicbir birim taninmadi -- /birim <sozcuk> ile bakin")
            continue
        print("   birim     %s" % b.coz(onek))
        cik, puan = uret(m, b, onek, n_yeni, isin, r, adim)
        print("   URETILEN  %s" % turkce(b, ters, cik))
        print("   birim     %s   (puan %.3f)" % (b.coz(cik), puan))


if __name__ == "__main__":
    sys.exit(main())
