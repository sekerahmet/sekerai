# -*- coding: utf-8 -*-
"""SOR — egitilmis modele ELLE soru sor, cevabi KELIMEYLE gor.

Model dil bilmiyor. Tek bir bicim taniyor:

    [S2] <varlik> <iliski1> <iliski2> ?     -> POZ 4'te TEK varlik token'i
    [S1] <varlik> <iliski>  ?               -> POZ 3'te TEK varlik token'i

Bu betik o token'a ad takar ve NE TUR bir hata oldugunu soyler:

    DOGRU    : iki adimi da yapti
    KISAYOL  : ikinci iliskiyi DOGRUDAN ilk varliga uyguladi  (f(e,r2))
    KOPRU    : ara cevabi yazdi, ikinci adimi atladi
    BASKA    : ucu de degil

IKI KOLU YAN YANA koyar: ayni soruya maskesiz (G) ne diyor, maskeli (GM) ne
diyor. Deneyin butun anlami bu iki sutunun farkinda.

    # tek anlik goruntu
    VERI=okul PRESET=grok_uzun python sor.py --kol "G:/content/calis_g/cikti_g:120000"

    # IKI kol, agirlik ortalamasi (BIRINCIL olcumle ayni okuma)
    VERI=okul PRESET=grok_uzun python sor.py \
        --kol "G:/content/calis_g/cikti_g:100000,105000,110000,115000,120000:yok" \
        --kol "GM:/content/calis_g/cikti_gm:100000,105000,110000,115000,120000:1@1-7"

Kol bicimi:  <ad>:<klasor>:<adimlar>[:<maske>]
  adimlar tek sayiysa o anlik goruntu; virgullu ise AGIRLIK ORTALAMASI
  maske    "yok" ya da "<poz>@<b0>-<b1>"   (verilmezse "yok")

KOMUTLAR
    <varlik> <r1> <r2>     2 adimli soru
    <varlik> <r>           1 adimli soru (olgu)
    :v [TIP] [n]           varlik listele        :i        iliski listele
    :r [KUME]              rastgele soru sec     (KUME: ent / comp / egitim)
    :b <varlik>            varligin butun olgulari
    :z <varlik> <r> <r>..  ZINCIR: adim adim yurut (KESIF, onkayit DISI)
    :q                     cik
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sablon"))
import sifirdan as S


def maske_coz(m):
    if not m or m == "yok":
        return None, None
    p, b = m.split("@")
    a1, a2 = b.split("-")
    return int(p), tuple(range(int(a1), int(a2) + 1))


def agirlik_ortalamasi(klasor, tohum, adimlar):
    """N anlik goruntunun ELEMAN BAZINDA ortalamasi. Birincil olcu bu
    (CLAUDE.md 0); egri ortalamasi DEGIL, arada ~2x fark var."""
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


class Kol:
    def __init__(self, tanim, tohum=0):
        parca = tanim.split(":")
        # Windows yolu "C:\..." icerebilir -> SAGDAN ayristir
        if len(parca) >= 4:
            ad = parca[0]
            maske = parca[-1]
            adimlar = parca[-2]
            klasor = ":".join(parca[1:-2])
        else:
            ad, maske = parca[0], "yok"
            adimlar = parca[-1]
            klasor = ":".join(parca[1:-1])
        self.ad = ad
        self.adimlar = [int(x) for x in adimlar.split(",")]
        self.poz, self.bloklar = maske_coz(maske)
        sd, self.bulunan = agirlik_ortalamasi(klasor, tohum, self.adimlar)
        self.net = S.Net("A", S.CFG).to(S.DEV)
        self.net.load_state_dict(sd)
        self.net.eval()
        # KOL KENDI MASKESIYLE olculur (CLAUDE.md 0). Maskeyle EGITILMIS bir
        # kolu maske KAPALI olcmek, Deney 7'de iki belgedeki sayilari
        # duzeltmek zorunda biraktigi hatanin aynisidir.
        for i, blk in enumerate(self.net.blocks):
            blk.mask_key = self.poz if (self.bloklar and i in self.bloklar) else None

    def tahmin(self, X, poz, k=5):
        lo, hi = S.ENT_OFF, S.ENT_OFF + S.CFG["N_ENT"]
        with torch.no_grad():
            lg, _ = self.net(torch.from_numpy(X).to(S.DEV))
        p = lg.float()[0, poz][lo:hi].softmax(-1)
        v, i = torch.topk(p, k)
        return [(int(a), float(b)) for a, b in zip(i.cpu(), v.cpu())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kol", action="append", required=True)
    ap.add_argument("--tohum", type=int, default=0)
    ap.add_argument("--ust", type=int, default=5, help="kac aday basilsin")
    a = ap.parse_args()

    assert S.VERI, "VERI bayragi YOK -> sor.py elle yapilmis veriyle calisir"
    VM = __import__("veri_" + S.VERI)
    G = VM.kur()
    E = [x for t in VM.TIPLER for x in G["ad"][t]]
    R = list(VM.ILISKI)
    eid = {x: i for i, x in enumerate(E)}
    rid = {r: i for i, r in enumerate(R)}
    tip = G["tip"]

    facts, pairs, one, tr2, comp, ent, se, ue, e2 = S.build_data()
    # (e,r1,r2) -> hangi bolmede.  Soru sorunca "bu soru nerede" desin diye.
    nerede = {}
    for ad_, lst in (("EGITIM", tr2), ("COMP", comp), ("ENT-AYIRT", ent),
                     ("ENT-YOK", S.ENT_YOK), ("ENT-ARAMA", S.ENT_ARAMA)):
        for e_, r1, r2, *_ in lst:
            nerede[(e_, r1, r2)] = ad_

    kollar = [Kol(t, a.tohum) for t in a.kol]
    print("=" * 78)
    print(f"SOR   veri={S.VERI}  sozluk={S.VOCAB}  "
          f"({len(R)} iliski + {len(E)} varlik)")
    for k in kollar:
        m = "yok" if k.poz is None else f"poz {k.poz} @ blok {k.bloklar[0]}-{k.bloklar[-1]}"
        print(f"   kol {k.ad:4s} maske {m:24s} adimlar {k.bulunan}")
    print("=" * 78)
    print("  <varlik> <r1> <r2>   2 adimli     |  :v [TIP] [n]  varlik listele")
    print("  <varlik> <r>         1 adimli     |  :i            iliski listele")
    print("  :r [ent|comp|egitim] rastgele     |  :b <varlik>   olgulari")
    print("  :q  cik")
    rng = np.random.RandomState(0)

    def ad(i):
        return E[i]

    def sor2(e, r1, r2):
        b = facts[eid[e], rid[r1]]
        if b < 0:
            print(f"  !! '{e}' ({tip[e]}) icin '{r1}' iliskisi YOK")
            return
        cev = facts[b, rid[r2]]
        if cev < 0:
            print(f"  !! kopru '{ad(b)}' ({tip[ad(b)]}) icin '{r2}' YOK "
                  f"-> zincir gecersiz")
            return
        ks = facts[eid[e], rid[r2]]
        X = np.zeros((1, S.T_LEN), np.int64)
        X[0, :7] = [S.Q2, S.ENT_OFF + eid[e], S.REL_OFF + rid[r1],
                    S.REL_OFF + rid[r2], S.QM, S.ENT_OFF + int(cev), S.EOS]
        print(f"\nSORU  [S2] {e} {r1} {r2} ?")
        print(f"  kopru   {ad(b):22s} ('{e} {r1}')")
        print(f"  DOGRU   {ad(cev):22s} ('{ad(b)} {r2}')")
        print(f"  KISAYOL {('YOK (tip izin vermiyor)' if ks < 0 else ad(ks)):22s}"
              + ("" if ks < 0 else f" ('{e} {r2}')"))
        _n = nerede.get((eid[e], rid[r1], rid[r2]), "(hicbir bolmede)")
        print(f"  bolme   {_n}")
        etiket = lambda t: ("DOGRU" if t == cev else
                            "KISAYOL" if (ks >= 0 and t == ks) else
                            "KOPRU" if t == b else "BASKA")
        ustler = [k.tahmin(X, 4, a.ust) for k in kollar]
        print()
        print("   " + "".join(f"{k.ad:^34s}" for k in kollar))
        for i in range(a.ust):
            sat = f" {i+1}. "
            for u in ustler:
                t, p = u[i]
                sat += f"{ad(t):20s}{p:5.3f} {etiket(t):8s}"
            print(sat)

    def sor1(e, r):
        h = facts[eid[e], rid[r]]
        if h < 0:
            print(f"  !! '{e}' ({tip[e]}) icin '{r}' iliskisi YOK")
            return
        X = np.zeros((1, S.T_LEN), np.int64)
        X[0, :6] = [S.Q1, S.ENT_OFF + eid[e], S.REL_OFF + rid[r], S.QM,
                    S.ENT_OFF + int(h), S.EOS]
        print(f"\nSORU  [S1] {e} {r} ?      DOGRU {ad(h)}   (olgu, EGITIMDE)")
        ustler = [k.tahmin(X, 3, a.ust) for k in kollar]
        print("   " + "".join(f"{k.ad:^34s}" for k in kollar))
        for i in range(a.ust):
            sat = f" {i+1}. "
            for u in ustler:
                t, p = u[i]
                sat += f"{ad(t):20s}{p:5.3f} {'DOGRU' if t == h else '':8s}"
            print(sat)

    def zincir(e, iliskiler):
        """Modele zinciri ADIM ADIM yurut: her adim 1-ADIMLI soru, cikan
        cevap bir sonraki adima GERI BESLENIR.

        DIKKAT -- KESIF ARACI, ONKAYIT DISI. Olctugu sey KOMPOZISYON DEGIL:
        her adim ayri bir OLGU sorusu, yani modelin BILGISI + hatanin
        BIRIKMESI. Hukum veren olcum yalniz pencere.py'dendir (CLAUDE.md 0);
        buradaki hicbir sayi birincil okumaya girmez."""
        print(chr(10) + "ZINCIR  " + e + " -> " + " -> ".join(iliskiler))
        g = [eid[e]]
        for r in iliskiler:
            h = facts[g[-1], rid[r]] if g[-1] >= 0 else -1
            g.append(int(h))
        print("  gercek  " + " -> ".join(
            (ad(x) if x >= 0 else "(YOK)") for x in g))
        print()
        print("  adim iliski      " + "".join(f"{k.ad:^30s}" for k in kollar))
        cur = {k.ad: eid[e] for k in kollar}
        for i, r in enumerate(iliskiler):
            sat = f"  {i+1:4d} {r:11s} "
            for k in kollar:
                c = cur[k.ad]
                if c < 0 or tip[ad(c)] not in VM.SEMA[r]:
                    sat += f"{'(bu tipte iliski YOK)':22s}{'':8s}"
                    cur[k.ad] = -1
                    continue
                X = np.zeros((1, S.T_LEN), np.int64)
                X[0, :6] = [S.Q1, S.ENT_OFF + c, S.REL_OFF + rid[r], S.QM, 0, S.EOS]
                t = k.tahmin(X, 3, 1)[0][0]
                sat += f"{ad(t):22s}{('yolda' if t == g[i+1] else 'SAPTI'):8s}"
                cur[k.ad] = t
            print(sat)
        print()
        for k in kollar:
            son = cur[k.ad]
            print(f"  {k.ad:4s} son cevap "
                  f"{(ad(son) if son >= 0 else '(YOK)'):22s}"
                  f"{'DOGRU' if son == g[-1] else 'YANLIS'}"
                  f"   (gercek {ad(g[-1]) if g[-1] >= 0 else '(YOK)'})")


    while True:
        try:
            s = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not s:
            continue
        if s in (":q", ":quit", "q"):
            break
        if s.startswith(":i"):
            for t in VM.TIPLER:
                rr = [r for r in R if t in VM.SEMA[r]]
                print(f"  {t:6s} -> " + ", ".join(
                    f"{r}({VM.SEMA[r][t]})" for r in rr))
            continue
        if s.startswith(":v"):
            p = s.split()
            t = p[1].upper() if len(p) > 1 else "KISI"
            n = int(p[2]) if len(p) > 2 else 12
            print(f"  {t} ({len(G['ad'][t])}): " + ", ".join(G["ad"][t][:n]) + " ...")
            continue
        if s.startswith(":b"):
            p = s.split()
            if len(p) < 2 or p[1] not in eid:
                print("  !! :b <varlik>"); continue
            e = p[1]
            for r in R:
                h = facts[eid[e], rid[r]]
                if h >= 0:
                    print(f"   {e:20s} {r:12s} {ad(h)}")
            continue
        if s.startswith(":z"):
            q = s.split()
            if len(q) < 3 or q[1] not in eid:
                print("  !! :z <varlik> <r1> <r2> [<r3> ...]"); continue
            if any(x not in rid for x in q[2:]):
                print("  !! bilinmeyen iliski  (:i ile listele)"); continue
            zincir(q[1], q[2:])
            continue
        if s.startswith(":r"):
            p = s.split()
            kume = (p[1].lower() if len(p) > 1 else "ent")
            lst = {"ent": ent, "comp": comp, "egitim": tr2}.get(kume, ent)
            e_, r1, r2, *_ = lst[rng.randint(len(lst))]
            sor2(E[e_], R[r1], R[r2])
            continue
        p = s.split()
        if p[0] not in eid:
            print(f"  !! '{p[0]}' sozlukte yok  (:v ile listele)"); continue
        if any(x not in rid for x in p[1:]):
            print(f"  !! bilinmeyen iliski  (:i ile listele)"); continue
        if len(p) == 3:
            sor2(p[0], p[1], p[2])
        elif len(p) == 2:
            sor1(p[0], p[1])
        else:
            print("  !! <varlik> <r1> <r2>  ya da  <varlik> <r>")


if __name__ == "__main__":
    main()
