# -*- coding: utf-8 -*-
"""KAYAN PENCERE — birincil olcunun PENCEREYE ne kadar bagli oldugunu olc.

`pencere.py` tek bir pencereyi (60-80k) okur. O pencere bir ON SEZIYDI:
D'nin kendi egrisine bakilarak secildi, sonra E1'de disarida-birakma
testinden gecti -- ama phi 3.03'te. phi degisince ayni ADIM araligi ayni
ILERLEMEYE karsilik gelmiyor (CLAUDE.md 4: "esit adimda degil esit
ilerlemede kiyasla").

Bu betik pencereyi DEGISTIRMEZ. Butun ardisik pencereleri tarar ve
sonucun nasil degistigini basar. En iyisi SECILMEZ -- hepsi raporlanir.

    python kayan_pencere.py --cikti X.json \
        --kol "A5:/content/calis_d33/cikti_a5:yok" \
        --kol "D5:/content/calis_d33/cikti_d5:1@1-7" \
        --kol "K5:/content/calis_d33/cikti_k5:2@1-7" \
        --oran D5/A5 --oran D5/K5 --genislik 5

Kol bicimi:  <ad>:<klasor>:<maske>     (adimlar DISKTEN bulunur)
"""
import os, sys, json, glob, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sifirdan as S
from pencere import maske_coz


def adimlari_bul(klasor, kol, tohum):
    r = []
    for p in glob.glob(os.path.join(klasor, f"snap_{kol}_s{tohum}_*.pt")):
        try:
            r.append(int(os.path.basename(p).split("_")[-1][:-3]))
        except ValueError:
            pass
    return sorted(set(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kol", action="append", required=True)
    ap.add_argument("--cikti", required=True)
    ap.add_argument("--tohum", type=int, default=0)
    ap.add_argument("--genislik", type=int, default=5)
    ap.add_argument("--oran", action="append", default=[])
    a = ap.parse_args()

    # Olcme setleri: pencere.py ile AYNI kurulum, kopyalanmis sabit yok.
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
    brE = np.array([S.ENT_OFF + b for _, _, _, b, _ in LE], np.int64)
    scE = np.array([S.ENT_OFF + int(facts[e, r2]) for e, _, r2, _, _ in LE], np.int64)
    brC = np.array([S.ENT_OFF + b for _, _, _, b, _ in LC], np.int64)
    scC = np.array([S.ENT_OFF + int(facts[e, r2]) for e, _, r2, _, _ in LC], np.int64)
    br2 = np.array([S.ENT_OFF + b for _, _, _, b, _ in L2], np.int64) if L2 else None
    sc2 = (np.array([S.ENT_OFF + int(facts[e, r2]) for e, _, r2, _, _ in L2], np.int64)
           if L2 else None)
    # ENT-YOK (gomulu kontrol). Salt 5 = egitim dongusuyle AYNI ornekleme.
    LY = sub(S.ENT_YOK, 5) if getattr(S, "ENT_YOK", None) else []
    EY = S.enc_two(LY) if LY else None
    brY = np.array([S.ENT_OFF + b for _, _, _, b, _ in LY], np.int64) if LY else None
    scY = (np.array([S.ENT_OFF + int(facts[e, r2]) for e, _, r2, _, _ in LY], np.int64)
           if LY else None)

    kollar = []
    for t in a.kol:
        govde, mask = t.rsplit(":", 1)
        ad, klasor = govde.split(":", 1)
        kollar.append((ad, klasor, mask, adimlari_bul(klasor, "A", a.tohum)))

    # ORTAK adimlar: kollar HEP esit adimda kiyaslanir (CLAUDE.md 4).
    ortak = sorted(set.intersection(*[set(x[3]) for x in kollar]))
    G = a.genislik
    pencereler = [ortak[i:i + G] for i in range(len(ortak) - G + 1)]
    print(f"kollar: {[x[0] for x in kollar]}")
    print(f"ortak adim: {len(ortak)} nokta  {ortak[0]}..{ortak[-1]}")
    print(f"pencere: {G} nokta genisliginde, {len(pencereler)} adet")
    for ad, _, mask, ad_l in kollar:
        print(f"   {ad:4s} maske {mask:8s}  diskte {len(ad_l)} anlik goruntu")
    if not pencereler:
        print("!! yeterli ortak nokta yok"); sys.exit(3)

    net = S.Net("A", S.CFG).to(S.DEV)
    sonuc = {ad: {} for ad, _, _, _ in kollar}

    for ad, klasor, mask, _ in kollar:
        poz, bloklar = maske_coz(mask)
        # HER PENCERE BASTAN HESAPLANIR. Yuvarlanan toplam (cikar-ekle)
        # denendi: bastan hesapla arasinda 2.4e-07 fark birakti -- kucuk,
        # ama OLCUM hattinda "ihmal edilebilir" diye not dusmek yerine
        # optimizasyonu attik. Maliyeti ~1 dk (CLAUDE.md 6, bit-ayni).
        for w in pencereler:
            top = None
            for st in w:
                sd = torch.load(os.path.join(
                    klasor, f"snap_A_s{a.tohum}_{st:06d}.pt"), map_location="cpu")
                top = ({k: v.float() for k, v in sd.items()} if top is None
                       else {k: top[k] + v.float() for k, v in sd.items()})
            net.load_state_dict({k: v / G for k, v in top.items()})
            net.eval()
            for i, blk in enumerate(net.blocks):
                blk.mask_key = poz if (bloklar and i in bloklar) else None
            zE, _ = S.zengin(net, EE, brE, scE)
            zC, _ = S.zengin(net, EC, brC, scC)
            z2 = (S.zengin(net, E2, br2, sc2)[0] if E2 is not None
                  else {"acc": None})
            zY = (S.zengin(net, EY, brY, scY)[0] if EY is not None
                  else {"acc": None})
            for blk in net.blocks:
                blk.mask_key = None
            sonuc[ad][f"{w[0]}-{w[-1]}"] = dict(
                adimlar=w, ent=zE["acc"], ent_kisayol=zE["shortcut"],
                comp=zC["acc"], ent2=z2["acc"], ent_yok=zY["acc"])
        print(f"  {ad} bitti ({len(pencereler)} pencere)")

    # ------------------------------------------------------------- TABLO
    anah = [f"{w[0]}-{w[-1]}" for w in pencereler]
    print(f"\n{'='*78}\nKAYAN PENCERE — ENT  (her kol KENDI maskesiyle)")
    bas = "  " + "".join(f"{ad:>10s}" for ad, _, _, _ in kollar)
    print(f"  {'pencere':>15s}" + bas + "".join(f"{o:>12s}" for o in a.oran))
    for k in anah:
        sat = f"  {k:>15s}  "
        for ad, _, _, _ in kollar:
            sat += f"{sonuc[ad][k]['ent']:>10.4f}"
        for o in a.oran:
            p, q = o.split("/")
            v = sonuc[q][k]["ent"]
            sat += f"{(sonuc[p][k]['ent']/v if v else float('inf')):>12.2f}"
        print(sat + ("   <- ONCEDEN YAZILAN" if k == "60000-80000" else ""))

    print(f"\n{'='*78}\nAYNI PENCERELERDE comp / kisayol")
    print(f"  {'pencere':>15s}" + "".join(
        f"{ad+' comp':>12s}{ad+' ksy':>11s}" for ad, _, _, _ in kollar))
    for k in anah:
        sat = f"  {k:>15s}"
        for ad, _, _, _ in kollar:
            sat += f"{sonuc[ad][k]['comp']:>12.3f}{sonuc[ad][k]['ent_kisayol']:>11.3f}"
        print(sat)

    # GOMULU KONTROL, pencere pencere. Onkayit 5.3'un hukmu tek pencereye
    # baglanmamali: ENT-AYIRT'ta oran yuksek ama ENT-YOK'ta da yuksekse
    # kazanc kisayola OZGU degildir.
    if all(sonuc[ad][anah[0]].get("ent_yok") is not None
           for ad, _, _, _ in kollar):
        print(chr(10) + "=" * 78)
        print("ENT-YOK (gomulu kontrol: kisayol IMKANSIZ)")
        print(f"  {'pencere':>15s}"
              + "".join(f"{ad:>10s}" for ad, _, _, _ in kollar)
              + "".join(f"{o+' YOK':>12s}" for o in a.oran))
        for k in anah:
            sat = f"  {k:>15s}  "
            for ad, _, _, _ in kollar:
                sat += f"{sonuc[ad][k]['ent_yok']:>10.4f}"
            for o in a.oran:
                p_, q_ = o.split("/")
                v = sonuc[q_][k]["ent_yok"]
                sat += f"{(sonuc[p_][k]['ent_yok']/v if v else float('inf')):>12.2f}"
            print(sat)

    # ONKAYIT 5-EK: oran pencerelerin bir kisminda esigin ustunde, bir
    # kisminda altindaysa hukum "pencereye bagli" -> BIRINCIL ZAYIFLAR.
    print(f"\n{'='*78}\nPENCERE BAGIMLILIGI")
    for o in a.oran:
        p, q = o.split("/")
        vals = [sonuc[p][k]["ent"] / sonuc[q][k]["ent"] if sonuc[q][k]["ent"]
                else float("inf") for k in anah]
        esik = 3.0 if q == "A5" else 2.0
        ust = sum(1 for v in vals if v >= esik)
        print(f"  {o:8s} esik {esik}:  {ust}/{len(vals)} pencerede GECIYOR   "
              f"aralik {min(vals):.2f} .. {max(vals):.2f}")
        print(f"           -> " + ("HER pencerede ayni yon: hukum pencereye BAGLI DEGIL"
                                   if ust in (0, len(vals)) else
                                   "!! PENCEREYE BAGLI -> birincil sonuc ZAYIFLAR"))

    json.dump(dict(pencereler=anah, kollar=sonuc, oranlar=a.oran),
              open(a.cikti, "w"), indent=1)
    print(f"\n-> {a.cikti}")


if __name__ == "__main__":
    main()
