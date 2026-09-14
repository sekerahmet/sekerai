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
import os, sys, json, glob, argparse, hashlib
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
        if not y:
            raise FileNotFoundError(f"anlik goruntu YOK: {klasor} adim {a}")
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

    # OLCME SETLERI: sifirdan.py'nin egitim dongusuyle AYNI FONKSIYONDAN
    # (AUDIT B2, 14 Eylul). Eskiden burada `sub()` mantigi KOPYA duruyordu;
    # bir salt/bolme degisirse hicbir hata vermeden farkli ornek olculurdu.
    L1, LS, LC, LE, L2, LY = S.olcme_listeleri(one, tr2, comp, ent_ev, ent2_ev)
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
    # ENT-YOK kodlamasi. LY yukarida olcme_listeleri()'nden geldi; salt 5
    # artik TEK YERDE tanimli, burada tekrar edilmiyor.
    EY = S.enc_two(LY) if LY else None
    brY = np.array([S.ENT_OFF + b for _, _, _, b, _ in LY], np.int64)
    scY = np.array([S.ENT_OFF + int(facts[e, r2]) for e, _, r2, _, _ in LY], np.int64)
    print(f"olcme seti: ent {len(LE)}  comp {len(LC)}  ent2 {len(L2)}  "
          f"ent_yok {len(LY)}  1hop {len(L1)}")

    # --- OLCME SETI PARMAK IZI (AUDIT B2, 14 Eylul) ----------------------
    # Setler artik S.olcme_listeleri()'nden geliyor (tek kaynak), yani
    # "iki dosya farkli salt kullanir" arizasi KAPANDI. Bu damga geriye
    # kalani yakalar: build_data / veri ureteci / konfig / N_EVAL_MAX
    # degisirse ayni VERI ile farkli bir iz cikar ve OLCUM_IZ assert'i
    # atesler. Yani "ayni deneyin iki olcumu ayni seti mi gordu" sorusu
    # artik MEKANIK olarak cevaplanabiliyor.
    _iz = hashlib.md5(repr([len(x) for x in (L1, LS, LC, LE, L2, LY)]).encode()
                      + b"|" + repr([tuple(x[:3]) for x in
                                     (L1, LS, LC, LE, L2, LY) if x]).encode()
                      ).hexdigest()[:12]
    print(f"olcme seti parmak izi: {_iz}   VERI={S.VERI or '(yok)'}")
    _BEKLENEN = os.environ.get("OLCUM_IZ", "")
    assert not _BEKLENEN or _BEKLENEN == _iz, (
        f"OLCME SETI DEGISMIS: beklenen {_BEKLENEN}, kurulan {_iz}." + chr(10)
        + "  Ayni VERI/konfig ile onceki kosuyla AYNI seti olcmuyorsun ->"
          " sayilar kiyaslanamaz." + chr(10)
        + "  Bilerek degistiysen OLCUM_IZ'i guncelle; degilse sifirdan.py'de"
          " ne degisti diye bak.")

    net = S.Net("A", S.CFG).to(S.DEV)
    sonuc, eksik = {}, []
    for tanim in a.kol:
        # Yol iki nokta icerebilir (Windows "C:\..."), o yuzden SAGDAN bol:
        # son iki alan adimlar ve maske; kalanin ILK ikinoktasi ad/klasor ayraci.
        govde, adimlar, mask = tanim.rsplit(":", 2)
        ad, klasor = govde.split(":", 1)
        adimlar = [int(x) for x in adimlar.split(",")]
        poz, bloklar = maske_coz(mask)
        # BIR kol eksikse BUTUN okuma coluyordu: assert ilk eksik anlik
        # goruntude patliyor, HESAPLANMIS kollarin sonucu da yazilmiyordu.
        # Kismi sonuc sifir sonuctan iyidir; eksik olan ciktida ISARETLENIR.
        try:
            sd, bulunan = agirlik_ortalamasi(klasor, "A", a.tohum, adimlar)
            net.load_state_dict(sd)
        except Exception as e:
            print()
            print(f"{ad:8s} !! OLCULEMEDI: {e}")
            sonuc[ad] = dict(kol=ad, klasor=klasor, maske=mask, hata=str(e))
            eksik.append(ad)
            continue
        net.eval()
        for i, blk in enumerate(net.blocks):
            blk.mask_key = poz if (bloklar and i in bloklar) else None

        zE, _ = S.zengin(net, EE, brE, scE)
        zC, _ = S.zengin(net, EC, brC, scC)
        zS, _ = S.zengin(net, ES, brS, scS)
        # ENT2 seti BOS olabilir (kucuk konfig). Yukarida "E2 = ... if L2
        # else None" diye bir koruma vardi ama SAHTEYDI: asagisi None'i ele
        # almiyordu ve zengin() TypeError ile BUTUN okumayi olduruyordu.
        z2 = (S.zengin(net, E2, br2, sc2)[0] if E2 is not None
              else {"acc": None})
        zY = (S.zengin(net, EY, brY, scY)[0] if EY is not None
              else {"acc": None, "shortcut": None})
        r = dict(kol=ad, klasor=klasor, adimlar=bulunan, maske=mask,
                 ent=zE["acc"], ent_kisayol=zE["shortcut"],
                 comp=zC["acc"], comp_kisayol=zC["shortcut"],
                 seen=zS["acc"], ent2=z2["acc"],
                 ent_yok=zY["acc"], ent_yok_kisayol=zY["shortcut"],
                 bir_hop=float(S.evaluate(net, *E1[:3])[0].mean()))
        # TURETILMIS ALAN: onkayit 5.3 "fayda(AYIRT) > fayda(YOK)" bir ORANIN
        # ORANI ve kapi dilbilgisi bunu IFADE EDEMIYOR. Cebirsel olarak
        #   (GM.ent/G.ent) / (GM.ent_yok/G.ent_yok)
        #     = (GM.ent/GM.ent_yok) / (G.ent/G.ent_yok)
        # yani kol basina TEK alan yetiyor ve kapi "GM/G ent_bolu_yok > 1.0"
        # olarak yazilabiliyor. Payda 0 ise alan None -> kapi ATLANDI der
        # (sessizce inf/NaN uretip "gecti" demesindense).
        r["ent_bolu_yok"] = (r["ent"] / r["ent_yok"]
                             if r["ent_yok"] else None)
        sonuc[ad] = r
        print(f"\n{ad:8s} maske {mask:8s} adimlar {bulunan}")
        print(f"   ENT {r['ent']:.4f}   (kisayol {r['ent_kisayol']:.3f})")
        _e2 = "YOK" if r["ent2"] is None else f"{r['ent2']:.4f}"
        if r["ent_yok"] is not None:
            # GOMULU KONTROL: kisayolun IMKANSIZ oldugu ENT zincirleri.
            # ent_yok_kisayol 0.000 CIKMALI -- cikmiyorsa scY yanlis kurulmus.
            print(f"   ENT-YOK {r['ent_yok']:.4f}   "
                  f"(kisayol {r['ent_yok_kisayol']:.3f} -- 0.000 OLMALI)")
        print(f"   comp {r['comp']:.3f}  seen {r['seen']:.3f}  "
              f"ent2 {_e2}  1hop {r['bir_hop']:.3f}")
        for blk in net.blocks:
            blk.mask_key = None

    # --- ONCEDEN YAZILAN KAPILAR ------------------------------------------
    OP = {">=": lambda x, y: x >= y, "<=": lambda x, y: x <= y,
          ">": lambda x, y: x > y, "<": lambda x, y: x < y}
    kapilar = []
    if a.kapi:
        print(f"\n{'='*62}\nONCEDEN YAZILAN KAPILAR")
    for spec in a.kapi:
        # "ETIKET: X alan op esik"  — etiket istege bagli ama olmadan
        # cikti okunmuyor: hangi kapi oldugu ve gecmenin ne demek oldugu
        # belli olmuyordu, ustelik bir kapi TERS yonlu.
        etiket, _, kural = spec.rpartition(":")
        kural = (kural or spec).strip()
        # KAPI DENETIMI. Onceki hali bozuk kurali ya ValueError ile KOSUNUN
        # SONUNDA dusuruyordu ya da bilinmeyen alan adini sessizce "ATLANDI"
        # yapiyordu. 14 Eylul: deney G'nin 6 kapisinin 4'u bozuktu --
        # ikisi parantezli not yuzunden (split 4'u asiyor), biri sag tarafi
        # sayi degil kol adi, biri `ent_shortcut` (egrinin adi; pencere.py'de
        # alan `ent_kisayol`). Hicbiri kosmadan once fark edilmiyordu.
        _p = kural.split()
        assert len(_p) == 4, (
            f"KAPI BOZUK (4 sozcuk olmali, {len(_p)} var): {spec!r}" + chr(10) +
            f"   bicim: '<etiket>: <sol> <alan> <op> <sayi>'  "
            f"-- etikette BOSLUK, kuralda parantezli not OLAMAZ")
        sol, alan, op, esik = _p
        assert op in OP, f"KAPI BOZUK (op '{op}' taninmiyor): {spec!r}"
        try:
            esik = float(esik)
        except ValueError:
            raise AssertionError(
                f"KAPI BOZUK (sag taraf SAYI olmali, '{esik}' geldi): {spec!r}"
                f"   iki kolu kiyaslamak icin fark kullan: 'GM-G {alan} < 0.0'")
        _kol = sol.replace("/", " ").replace("-", " ").split()
        # ALAN ADI DENETIMI: olculmus bir koldaki anahtarlarla karsilastir.
        # "alan yok" (programci hatasi, GURULTULU olmali) ile "alan None"
        # (kume bos, ATLANDI dogru) ayni sey degil.
        _olculen = [v for v in sonuc.values()
                    if isinstance(v, dict) and "hata" not in v]
        if _olculen:
            assert alan in _olculen[0], (
                f"KAPI BOZUK (alan '{alan}' yok): {spec!r}" + chr(10) +
                f"   gecerli alanlar: "
                + ", ".join(k for k in _olculen[0] if k not in
                            ("kol", "klasor", "adimlar", "maske")))
        for k in _kol:
            assert k in sonuc, f"KAPI BOZUK (kol '{k}' yok): {spec!r}"
        # kol hic olculemedi YA DA o alan bu kolda yok/None
        if any(k in eksik or sonuc.get(k, {}).get(alan) is None
               for k in _kol):
            kapilar.append(dict(etiket=etiket.strip(), kural=kural,
                                deger=None, gecti=None))
            print(f"  ATLANDI   {etiket.strip() or sol:26s} {kural}"
                  f"   ->  kol OLCULEMEDI")
            continue
        if "/" in sol:                       # ORAN kapisi
            p, q = sol.split("/")
            # PAYDA SIFIR: eskiden ZeroDivisionError ile BUTUN okumayi
            # oldururdu -- 2 x 120.000 adimlik kosunun EN SONUNDA. ENT
            # phi 5'te 0.005 civari; bir pencerede 0.000 cikmasi mumkun.
            if not sonuc[q][alan]:
                kapilar.append(dict(etiket=etiket.strip(), kural=kural,
                                    deger=None, gecti=None))
                print(f"  ATLANDI   {etiket.strip() or sol:26s} {kural}"
                      f"   ->  payda SIFIR ({q}.{alan} = {sonuc[q][alan]})")
                continue
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
        kapilar.append(dict(etiket=etiket.strip(), kural=kural,
                            deger=deger, gecti=bool(gecti)))
        print(f"  {'GECTI ' if gecti else '!! KALDI'}  "
              f"{etiket.strip() or sol:26s} {sol} {alan} {op} {esik}"
              f"   ->  {nasil} = {deger:.4f}")
    sonuc["_kapilar"] = kapilar
    if kapilar:
        k = sum(1 for x in kapilar if x["gecti"])
        at = sum(1 for x in kapilar if x["gecti"] is None)
        print(f"  {k}/{len(kapilar)} kapi gecti"
              + (f"   ({at} kapi OLCULEMEDI)" if at else ""))

    sonuc["_eksik_kol"] = eksik
    json.dump(sonuc, open(a.cikti, "w"), indent=1)
    print(f"\n-> {a.cikti}")
    if eksik:
        print(f"!! OLCULEMEYEN KOL: {', '.join(eksik)}"
              f"   -> yukaridaki sonuclar KISMI, hukum verilemez")
        sys.exit(3)


if __name__ == "__main__":
    main()
