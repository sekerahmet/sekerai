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

IKI KULLANIM VAR.

1) COLAB HUCRESI -- `input()` Colab'da calismaz, o yuzden REPL DEGIL:

       import sys, os
       sys.path.insert(0, "/content/kod_g")
       os.environ.update(VERI="okul", PRESET="grok_uzun", ARMS="A",
                         MEM_AT="4", ENT_PAY="0.20", COMP_PAY="0.10")
       from sor import Oturum
       P = "100000,105000,110000,115000,120000"
       o = Oturum([f"G:/content/calis_g/cikti_g:{P}:yok",
                   f"GM:/content/calis_g/cikti_gm:{P}:1@1-7"])
       o("Ahmet_Yilmaz arkadas anne")

2) TERMINAL (REPL):

       VERI=okul PRESET=grok_uzun python sor.py \
           --kol "G:/content/calis_g/cikti_g:120000"

Kol bicimi:  <ad>:<klasor>:<adimlar>[:<maske>]
  adimlar tek sayiysa o anlik goruntu; virgullu ise AGIRLIK ORTALAMASI
  maske    "yok" ya da "<poz>@<b0>-<b1>"   (verilmezse "yok")

KOMUTLAR (ikisinde de ayni)
    <varlik> <r1> <r2>     2 adimli soru
    <varlik> <r>           1 adimli soru (olgu)
    :v [TIP] [n]           varlik listele        :i   iliski listele
    :r [ent|comp|egitim]   rastgele soru sec
    :b <varlik>            varligin butun olgulari
    :z <varlik> <r> <r>..  ZINCIR: adim adim yurut (KESIF, onkayit DISI)
    :q                     cik (yalniz REPL)
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "sablon"))
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
            ad, maske, adimlar = parca[0], parca[-1], parca[-2]
            klasor = ":".join(parca[1:-2])
        else:
            ad, maske, adimlar = parca[0], "yok", parca[-1]
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


class Oturum:
    """Kurulumu BIR KEZ yapar, sonra sor(metin) ile sorgulanir."""

    def __init__(self, kol, tohum=0, ust=5, sessiz=False):
        assert S.VERI, "VERI bayragi YOK -> sor.py elle yapilmis veriyle calisir"
        self.ust = ust
        self.VM = __import__("veri_" + S.VERI)
        self.G = self.VM.kur()
        self.E = [x for t in self.VM.TIPLER for x in self.G["ad"][t]]
        self.R = list(self.VM.ILISKI)
        self.eid = {x: i for i, x in enumerate(self.E)}
        self.rid = {r: i for i, r in enumerate(self.R)}
        self.tip = self.G["tip"]

        (self.facts, _p, _one, self.tr2, self.comp,
         self.ent, _se, _ue, _e2) = S.build_data()
        # (e,r1,r2) -> hangi bolmede. Soru sorunca "bu soru nerede" desin diye.
        self.nerede = {}
        for ad_, lst in (("EGITIM", self.tr2), ("COMP", self.comp),
                         ("ENT-AYIRT", self.ent), ("ENT-YOK", S.ENT_YOK),
                         ("ENT-ARAMA", S.ENT_ARAMA)):
            for e_, r1, r2, *_ in lst:
                self.nerede[(e_, r1, r2)] = ad_

        self.kollar = [Kol(t, tohum) for t in kol]
        self.rng = np.random.RandomState(0)
        if not sessiz:
            self.bilgi()

    # ------------------------------------------------------------------
    def bilgi(self):
        print("=" * 78)
        print(f"SOR   veri={S.VERI}  sozluk={S.VOCAB}  "
              f"({len(self.R)} iliski + {len(self.E)} varlik)")
        for k in self.kollar:
            m = ("yok" if k.poz is None
                 else f"poz {k.poz} @ blok {k.bloklar[0]}-{k.bloklar[-1]}")
            print(f"   kol {k.ad:4s} maske {m:24s} adimlar {k.bulunan}")
        print("=" * 78)
        print("  <varlik> <r1> <r2>   2 adimli     |  :v [TIP] [n]  varlik")
        print("  <varlik> <r>         1 adimli     |  :i            iliski")
        print("  :r [ent|comp|egitim] rastgele     |  :b <varlik>   olgular")
        print("  :z <varlik> <r> <r>..  zincir (KESIF, onkayit DISI)")

    def ad(self, i):
        return self.E[i]

    # ------------------------------------------------------------------
    def sor2(self, e, r1, r2):
        f, eid, rid, ad = self.facts, self.eid, self.rid, self.ad
        b = f[eid[e], rid[r1]]
        if b < 0:
            print(f"  !! '{e}' ({self.tip[e]}) icin '{r1}' iliskisi YOK")
            return
        cev = f[b, rid[r2]]
        if cev < 0:
            print(f"  !! kopru '{ad(b)}' ({self.tip[ad(b)]}) icin '{r2}' YOK "
                  f"-> zincir gecersiz")
            return
        ks = f[eid[e], rid[r2]]
        X = np.zeros((1, S.T_LEN), np.int64)
        X[0, :7] = [S.Q2, S.ENT_OFF + eid[e], S.REL_OFF + rid[r1],
                    S.REL_OFF + rid[r2], S.QM, S.ENT_OFF + int(cev), S.EOS]
        print(f"\nSORU  [S2] {e} {r1} {r2} ?")
        print(f"  kopru   {ad(b):22s} ('{e} {r1}')")
        print(f"  DOGRU   {ad(cev):22s} ('{ad(b)} {r2}')")
        print(f"  KISAYOL {('YOK (tip izin vermiyor)' if ks < 0 else ad(ks)):22s}"
              + ("" if ks < 0 else f" ('{e} {r2}')"))
        print(f"  bolme   "
              f"{self.nerede.get((eid[e], rid[r1], rid[r2]), '(hicbir bolmede)')}")
        etiket = lambda t: ("DOGRU" if t == cev else
                            "KISAYOL" if (ks >= 0 and t == ks) else
                            "KOPRU" if t == b else "BASKA")
        ustler = [k.tahmin(X, 4, self.ust) for k in self.kollar]
        print()
        print("   " + "".join(f"{k.ad:^34s}" for k in self.kollar))
        for i in range(self.ust):
            sat = f" {i+1}. "
            for u in ustler:
                t, p = u[i]
                sat += f"{ad(t):20s}{p:5.3f} {etiket(t):8s}"
            print(sat)

    def sor1(self, e, r):
        f, eid, rid, ad = self.facts, self.eid, self.rid, self.ad
        h = f[eid[e], rid[r]]
        if h < 0:
            print(f"  !! '{e}' ({self.tip[e]}) icin '{r}' iliskisi YOK")
            return
        X = np.zeros((1, S.T_LEN), np.int64)
        X[0, :6] = [S.Q1, S.ENT_OFF + eid[e], S.REL_OFF + rid[r], S.QM,
                    S.ENT_OFF + int(h), S.EOS]
        print(f"\nSORU  [S1] {e} {r} ?      DOGRU {ad(h)}   (olgu, EGITIMDE)")
        ustler = [k.tahmin(X, 3, self.ust) for k in self.kollar]
        print("   " + "".join(f"{k.ad:^34s}" for k in self.kollar))
        for i in range(self.ust):
            sat = f" {i+1}. "
            for u in ustler:
                t, p = u[i]
                sat += f"{ad(t):20s}{p:5.3f} {'DOGRU' if t == h else '':8s}"
            print(sat)

    def zincir(self, e, iliskiler):
        """Modele zinciri ADIM ADIM yurut: her adim 1-ADIMLI soru, cikan
        cevap bir sonraki adima GERI BESLENIR.

        KESIF ARACI, ONKAYIT DISI. Olctugu sey KOMPOZISYON DEGIL: her adim
        ayri bir OLGU sorusu, yani BILGI + hatanin BIRIKMESI. Hukum veren
        olcum yalniz pencere.py'dendir (CLAUDE.md 0)."""
        f, eid, rid, ad = self.facts, self.eid, self.rid, self.ad
        print(chr(10) + "ZINCIR  " + e + " -> " + " -> ".join(iliskiler))
        g = [eid[e]]
        for r in iliskiler:
            g.append(int(f[g[-1], rid[r]]) if g[-1] >= 0 else -1)
        print("  gercek  " + " -> ".join(
            (ad(x) if x >= 0 else "(YOK)") for x in g))
        print()
        print("  adim iliski      " + "".join(f"{k.ad:^30s}" for k in self.kollar))
        cur = {k.ad: eid[e] for k in self.kollar}
        for i, r in enumerate(iliskiler):
            sat = f"  {i+1:4d} {r:11s} "
            for k in self.kollar:
                c = cur[k.ad]
                if c < 0 or self.tip[ad(c)] not in self.VM.SEMA[r]:
                    sat += f"{'(bu tipte iliski YOK)':22s}{'':8s}"
                    cur[k.ad] = -1
                    continue
                X = np.zeros((1, S.T_LEN), np.int64)
                X[0, :6] = [S.Q1, S.ENT_OFF + c, S.REL_OFF + rid[r], S.QM,
                            0, S.EOS]
                t = k.tahmin(X, 3, 1)[0][0]
                sat += f"{ad(t):22s}{('yolda' if t == g[i+1] else 'SAPTI'):8s}"
                cur[k.ad] = t
            print(sat)
        print()
        for k in self.kollar:
            son = cur[k.ad]
            print(f"  {k.ad:4s} son cevap "
                  f"{(ad(son) if son >= 0 else '(YOK)'):22s}"
                  f"{'DOGRU' if son == g[-1] else 'YANLIS'}"
                  f"   (gercek {ad(g[-1]) if g[-1] >= 0 else '(YOK)'})")

    # ------------------------------------------------------------------
    def __call__(self, s):
        return self.sor(s)

    def sor(self, s):
        s = (s or "").strip()
        if not s:
            return
        eid, rid, R, E, G, VM = (self.eid, self.rid, self.R, self.E,
                                 self.G, self.VM)
        if s.startswith(":i"):
            for t in VM.TIPLER:
                print(f"  {t:6s} -> " + ", ".join(
                    f"{r}({VM.SEMA[r][t]})" for r in R if t in VM.SEMA[r]))
            return
        if s.startswith(":v"):
            q = s.split()
            t = q[1].upper() if len(q) > 1 else "KISI"
            n = int(q[2]) if len(q) > 2 else 12
            print(f"  {t} ({len(G['ad'][t])}): "
                  + ", ".join(G["ad"][t][:n]) + " ...")
            return
        if s.startswith(":b"):
            q = s.split()
            if len(q) < 2 or q[1] not in eid:
                print("  !! :b <varlik>"); return
            for r in R:
                h = self.facts[eid[q[1]], rid[r]]
                if h >= 0:
                    print(f"   {q[1]:20s} {r:12s} {E[h]}")
            return
        if s.startswith(":z"):
            q = s.split()
            if len(q) < 3 or q[1] not in eid:
                print("  !! :z <varlik> <r1> <r2> [<r3> ...]"); return
            if any(x not in rid for x in q[2:]):
                print("  !! bilinmeyen iliski  (:i ile listele)"); return
            return self.zincir(q[1], q[2:])
        if s.startswith(":r"):
            q = s.split()
            lst = {"ent": self.ent, "comp": self.comp, "egitim": self.tr2}.get(
                (q[1].lower() if len(q) > 1 else "ent"), self.ent)
            e_, r1, r2, *_ = lst[self.rng.randint(len(lst))]
            return self.sor2(E[e_], R[r1], R[r2])
        q = s.split()
        if q[0] not in eid:
            print(f"  !! '{q[0]}' sozlukte yok  (:v ile listele)"); return
        if any(x not in rid for x in q[1:]):
            print("  !! bilinmeyen iliski  (:i ile listele)"); return
        if len(q) == 3:
            return self.sor2(q[0], q[1], q[2])
        if len(q) == 2:
            return self.sor1(q[0], q[1])
        print("  !! <varlik> <r1> <r2>  ya da  <varlik> <r>")

    def repl(self):
        while True:
            try:
                s = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print(); break
            if s in (":q", ":quit", "q"):
                break
            self.sor(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kol", action="append", required=True)
    ap.add_argument("--tohum", type=int, default=0)
    ap.add_argument("--ust", type=int, default=5, help="kac aday basilsin")
    a = ap.parse_args()
    Oturum(a.kol, a.tohum, a.ust).repl()


if __name__ == "__main__":
    main()
