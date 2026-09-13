# -*- coding: utf-8 -*-
"""PENCERE — birincil okuma: N anlik goruntunun AGIRLIK ORTALAMASI, sonra olc.

Ham egrideki ENT degerlerinin ortalamasi DEGIL. Once kontrol noktalarinin
tensorleri eleman eleman ortalanir, ortaya cikan TEK model olculur. Bu projede
ikisi arasinda ~2x fark var (D3: egri ort. 0.174, agirlik ort. 0.365) ve
karistirmak bulgunun kendisini degistirir.

TEK OLCUM YOLU (CLAUDE.md 4): butun kollar bu kodla olculur; belgelerdeki
sayilar alinti yapilmaz.

MASKE (CLAUDE.md 7): maskeyle EGITILMIS kol, maske ACIKKEN olculur. Deney 7'de
D ve E1 maske kapaliyken degerlendirildi ve iki belgedeki sayilar duzeltilmek
zorunda kaldi. `--maske` her kol icin ayri verilir.

    python pencere.py --cikti sonuc.json \
        --kol "D3.1:/content/calis_d31/cikti:60000,65000,70000,75000,80000:1@1-7" \
        --kol "A:/content/drive/.../deney4_tekrarli_erisim:60000,...:yok" \
        --kol "D3:/content/drive/.../deney9_D3/D3:60000,...:1@1-7"

Kol bicimi:  <ad>:<klasor>:<adimlar>:<maske>
Maske bicimi: "yok"  ya da  "<poz>@<ilk>-<son>"   ornek 1@1-7
"""
import os, sys, json, glob, argparse
import numpy as np
import torch

# Python betigin KENDI klasorunu sys.path'e koyar, calisma dizinini degil.
# Bu dosya sablon/ icinde; sifirdan.py depo KOKUNDE. Koku elle ekliyoruz.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sifirdan as S


def maske_coz(m):
    if m in ("yok", "", "-"):
        return None, None
    poz, blok = m.split("@")
    a, b = blok.split("-")
    return int(poz), tuple(range(int(a), int(b) + 1))


def agirlik_ortalamasi(klasor, kol, tohum, adimlar):
    """Eleman eleman ortalama. Dosya adi bicimini VARSAYMA, glob'dan bul (§7)."""
    toplam, n = None, 0
    bulunan = []
    for a in adimlar:
        y = glob.glob(os.path.join(klasor, f"snap_{kol}_s{tohum}_*{a}.pt"))
        y = [p for p in y if int(os.path.basename(p).split("_")[-1][:-3]) == a]
        assert y, f"anlik goruntu YOK: {klasor} adim {a}"
        sd = torch.load(y[0], map_location="cpu")
        toplam = ({k: v.float() for k, v in sd.items()} if toplam is None
                  else {k: toplam[k] + v.float() for k, v in sd.items()})
        n += 1
        bulunan.append(a)
    assert n == len(adimlar), f"{n}/{len(adimlar)} nokta"
    return {k: v / n for k, v in toplam.items()}, bulunan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kol", action="append", required=True)
    ap.add_argument("--cikti", required=True)
    ap.add_argument("--tohum", type=int, default=0)
    # ONCEDEN YAZILAN KAPILARI KOD DEGERLENDIRSIN. Kapiyi sadece insan
    # okursa unutulabilir ya da sonucu gorup gevsetilebilir; burada
    # pazarlik yok. Bicim:  "<kol> <alan> <op> <esik>"  ya da
    #                       "<kol>/<kol> <alan> <op> <esik>"  (oran)
    ap.add_argument("--kapi", action="append", default=[])
    a = ap.parse_args()

    # Olcme setleri: kol() ile AYNI kurulum. Kopyalanmis sabit yok (§6).
    facts, pairs, one, tr2, comp, ent_ev, seen_ent, unseen_ent, ent2_ev = S.build_data()

    def sub(lst, salt):
        r = np.random.RandomState(S.DATA_SEED + 7 + salt)
        if len(lst) > S.N_EVAL_MAX:
            lst = [lst[i] for i in r.permutation(len(lst))[:S.N_EVAL_MAX]]
        return lst

    L1, LS, LC, LE = sub(one, 0), sub(tr2, 1), sub(comp, 2), sub(ent_ev, 3)
    L2 = sub(ent2_ev, 4)
    E1, ES, EC, EE = S.enc_one(L1), S.enc_two(LS), S.enc_two(LC), S.enc_two(LE)
    E2 = S.enc_two(L2) if L2 else None
    # `seen` = EGITIMDE gorulmus 2-hop. Ezber ile genellemeyi ayirir ve
    # D3.2'nin teshisini tek basina bu sutun verdi; olmamasi eksikti.
    brS = np.array([S.ENT_OFF + b for _, _, _, b, _ in LS], np.int64)
    scS = np.array([S.ENT_OFF + int(facts[e, r2]) for e, _, r2, _, _ in LS], np.int64)
    brE = np.array([S.ENT_OFF + b for _, _, _, b, _ in LE], np.int64)
    scE = np.array([S.ENT_OFF + int(facts[e, r2]) for e, _, r2, _, _ in LE], np.int64)
    brC = np.array([S.ENT_OFF + b for _, _, _, b, _ in LC], np.int64)
    scC = np.array([S.ENT_OFF + int(facts[e, r2]) for e, _, r2, _, _ in LC], np.int64)
    br2 = np.array([S.ENT_OFF + b for _, _, _, b, _ in L2], np.int64)
    sc2 = np.array([S.ENT_OFF + int(facts[e, r2]) for e, _, r2, _, _ in L2], np.int64)
    print(f"olcme seti: ent {len(LE)}  comp {len(LC)}  ent2 {len(L2)}  1hop {len(L1)}")

    net = S.Net("A", S.CFG).to(S.DEV)
    sonuc = {}
    for tanim in a.kol:
        # Yol iki nokta icerebilir (Windows "C:\..."), o yuzden SAGDAN bol:
        # son iki alan adimlar ve maske; kalanin ILK ikinoktasi ad/klasor ayraci.
        govde, adimlar, mask = tanim.rsplit(":", 2)
        ad, klasor = govde.split(":", 1)
        adimlar = [int(x) for x in adimlar.split(",")]
        poz, bloklar = maske_coz(mask)
        sd, bulunan = agirlik_ortalamasi(klasor, "A", a.tohum, adimlar)
        net.load_state_dict(sd)
        net.eval()
        for i, blk in enumerate(net.blocks):
            blk.mask_key = poz if (bloklar and i in bloklar) else None

        zE, _ = S.zengin(net, EE, brE, scE)
        zC, _ = S.zengin(net, EC, brC, scC)
        zS, _ = S.zengin(net, ES, brS, scS)
        z2, _ = S.zengin(net, E2, br2, sc2)
        r = dict(kol=ad, klasor=klasor, adimlar=bulunan, maske=mask,
                 ent=zE["acc"], ent_kisayol=zE["shortcut"],
                 comp=zC["acc"], comp_kisayol=zC["shortcut"],
                 seen=zS["acc"], ent2=z2["acc"],
                 bir_hop=float(S.evaluate(net, *E1[:3])[0].mean()))
        sonuc[ad] = r
        print(f"\n{ad:8s} maske {mask:8s} adimlar {bulunan}")
        print(f"   ENT {r['ent']:.4f}   (kisayol {r['ent_kisayol']:.3f})")
        print(f"   comp {r['comp']:.3f}  seen {r['seen']:.3f}  "
              f"ent2 {r['ent2']:.4f}  1hop {r['bir_hop']:.3f}")
        for blk in net.blocks:
            blk.mask_key = None

    # --- ONCEDEN YAZILAN KAPILAR ------------------------------------------
    OP = {">=": lambda x, y: x >= y, "<=": lambda x, y: x <= y,
          ">": lambda x, y: x > y, "<": lambda x, y: x < y}
    kapilar = []
    if a.kapi:
        print(f"\n{'='*62}\nONCEDEN YAZILAN KAPILAR")
    for spec in a.kapi:
        sol, alan, op, esik = spec.split()
        esik = float(esik)
        if "/" in sol:                       # ORAN kapisi
            p, q = sol.split("/")
            deger = sonuc[p][alan] / sonuc[q][alan]
            nasil = f"{sonuc[p][alan]:.4f} / {sonuc[q][alan]:.4f}"
        elif "-" in sol:                     # FARK kapisi
            # OZGULLUK ("ENT2(D5) - ENT2(A5) < 0.05") ve SAGLIK
            # ("comp(D5) >= comp(A5) - 0.10") fark kapilari; oran bicimi
            # bunlari IFADE EDEMIYORDU, yani onceden yazilan iki kapi
            # uygulanamaz durumdaydi.
            p, q = sol.split("-")
            deger = sonuc[p][alan] - sonuc[q][alan]
            nasil = f"{sonuc[p][alan]:.4f} - {sonuc[q][alan]:.4f}"
        else:
            deger = sonuc[sol][alan]
            nasil = f"{deger:.4f}"
        gecti = OP[op](deger, esik)
        kapilar.append(dict(kural=spec, deger=deger, gecti=bool(gecti)))
        print(f"  {'GECTI ' if gecti else '!! KALDI'}  {sol} {alan} {op} {esik}"
              f"   ->  {nasil} = {deger:.4f}")
    sonuc["_kapilar"] = kapilar
    if kapilar:
        k = sum(x["gecti"] for x in kapilar)
        print(f"  {k}/{len(kapilar)} kapi gecti")

    json.dump(sonuc, open(a.cikti, "w"), indent=1)
    print(f"\n-> {a.cikti}")


if __name__ == "__main__":
    main()
