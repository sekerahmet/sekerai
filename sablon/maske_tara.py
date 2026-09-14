# -*- coding: utf-8 -*-
"""MASKE TARA — 24 maske adayini ASIL OLCUYLE degerlendir.

NEDEN VAR. Asama B adaylari su skorla siraliyor:

    skor = (r1 BAGIMSIZLIGINI dusur) - (COMP'a ZARAR)

Bu bir VEKIL. Mekanizma iddiasi ise KISAYOL hakkinda: "maskeleme modelin
f(e,r2) yolunu YENIDEN OKUMASINI engeller". Kisayol oranini ZATEN olcuyoruz
(`ent_shortcut`), yani vekile ihtiyac yok -- adaylari dogrudan asil olcuyle
siralayabiliriz.

  DIL DUZELTMESI (14 Eylul, AUDIT): eskiden "yolunu kapatir" yaziyordu ve
  YANLISTI. Bu betigin taradigi maske bir KUYRUK (`i >= b0`, :125). b0 > 0
  ise blok 0 maskesiz kalir ve varligin icerigi artik akista TASINIR; yani
  yol kapanmaz, YENIDEN OKUNMASI engellenir. Yalniz b0 = 0 tam kesmedir --
  ve o satirlar comp'u cokertir, :167'deki filtre onlari ZATEN ELER.
  Dolayisiyla bu tablodan okunabilen HICBIR satir "yol kapandi" demiyor.

Bu betik HICBIR SEY SECMEZ ve FAZ 3'u etkilemez. Sorusu tek:

    >> Sectigimiz nokta, kacis yolunu GERCEKTEN kapatiyor mu?

Her aday icin uc sayi basar (maske INFERENCE'ta acik, egitim YOK):

    ent      ENT-ARAMA dogrulugu      (yukselmeli ya da en azindan dusmemeli)
    kisayol  ENT-ARAMA kisayol orani  (DUSMELI -- iddia bu)
    comp     COMP dogrulugu           (COKMEMELI)

DIKKAT -- BU EGITIM DEGIL. Maskeyi egitilmis bir modele SONRADAN takmak,
o maskeyle EGITMEKLE ayni sey degildir. Burasi "kacis yolu bu mu" sorusunu
cevaplar; "maskelersem ogrenir mi" sorusunu DEGIL. Hukum yalniz G/GM
kiyasindan gelir (CLAUDE.md 0).

OLCUM KUMESI: ENT-ARAMA. Hukum kumesi (ENT-AYIRT) BILEREK kullanilmiyor --
arama neye bakiyorsa tani da ona bakmali, yoksa hukum kumesi kirlenir.

    VERI=okul PRESET=grok_uzun python maske_tara.py \
        --klasor /content/calis_g/cikti_g --adim 45000
    # ya da agirlik ortalamasi:
    #   --adim 100000,105000,110000,115000,120000
"""
import argparse
import re
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sifirdan as S
from gpu_kapisi import gpu_bos_mu


def yukle(klasor, tohum, adimlar):
    """Dosya adi bicimini VARSAYMA, glob'dan oku; EKSIKSE DUR (§7).

    AUDIT B4, 14 Eylul: eski hali `if not os.path.exists(y): continue` ile
    eksik adimi SESSIZCE atliyordu. Bu kosuda runtime UC kez geri donusturuldu
    (g.py:250); 5 adimdan 2'si eksik olsaydi 3 noktanin ortalamasi alinir,
    hukum "5 noktanin ortalamasi" diye raporlanir, HATA MESAJI CIKMAZDI.
    pencere.py:40-55 zaten glob+assert kullaniyordu; ikisi ayni davranmali.
    """
    var = {}
    for p in glob.glob(os.path.join(klasor, f"snap_A_s{tohum}_*.pt")):
        m = re.search(r"_(\d+)\.pt$", os.path.basename(p))
        if m:
            var[int(m.group(1))] = p
    eksik = [a for a in adimlar if a not in var]
    assert not eksik, (
        f"anlik goruntu EKSIK: {eksik}" + chr(10)
        + f"  klasor {klasor}" + chr(10)
        + f"  diskte olanlar: {sorted(var)}" + chr(10)
        + "  Eksigi SESSIZCE atlamak, hukmu baska bir modelden vermektir.")
    top = None
    for a in adimlar:
        sd = torch.load(var[a], map_location="cpu")
        top = ({k: v.float() for k, v in sd.items()} if top is None
               else {k: top[k] + v.float() for k, v in sd.items()})
    return {k: v / len(adimlar) for k, v in top.items()}, list(adimlar)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--klasor", required=True)
    ap.add_argument("--adim", required=True, help="tek sayi ya da virgullu")
    ap.add_argument("--tohum", type=int, default=0)
    ap.add_argument("--n", type=int, default=2000, help="ornek sayisi")
    ap.add_argument("--zorla", action="store_true",
                    help="egitim kosuyor olsa bile baslat")
    a = ap.parse_args()

    # EGITIM KOSARKEN BASLAMA (14 Eylul: uc runtime geri donusumu).
    gpu_bos_mu(zorla=a.zorla, ad="maske taramasi")
    assert S.VERI, "VERI bayragi YOK"
    facts, _p, _one, _tr2, comp, ent, _se, _ue, _e2 = S.build_data()
    assert S.ENT_ARAMA, "ENT_ARAMA bos -> arama kumesi ayrilmamis"

    # ORNEKLEME: TOHUMLU PERMUTASYON, prefiks DEGIL (AUDIT B3, 14 Eylul).
    # Eski hali `S.ENT_ARAMA[:n]` idi. build_data_dis zincirleri VARLIK VARLIK
    # ekliyor (sifirdan.py:312 `for e in E:`), yani prefiks = "ilk varliklarin
    # TUM zincirleri". Olculdu: comp[:2000] varlik uzayinin ilk %36'sindan
    # geliyordu (309/848 ayri varlik, id 0..384). r1/r2 karisimi oransal
    # korundugu icin G'nin sayilarini degistirmedi (iki ornekleme arasindaki
    # fark butun hucrelerde <= 0.002, G5_ZAMAN.txt) -- ama pencere.py ve
    # sifirdan.py tohumlu sub() kullaniyor, bu dosya kullanmiyordu; uc aracin
    # comp/kisayol sayilari YAN YANA KONULAMIYORDU ve hicbir uyari yoktu.
    def _sub(lst, salt, n):
        r = np.random.RandomState(S.DATA_SEED + 7 + salt)
        return [lst[i] for i in r.permutation(len(lst))[:n]] if len(lst) > n else list(lst)

    LA = _sub(S.ENT_ARAMA, 3, a.n)
    LC = _sub(comp, 2, a.n)          # salt 2 = pencere.py'nin comp salt'i
    EA, EC = S.enc_two(LA), S.enc_two(LC)
    brA = np.array([S.ENT_OFF + b for _, _, _, b, _ in LA], np.int64)
    scA = np.array([S.ENT_OFF + int(facts[e, r2]) for e, _, r2, _, _ in LA],
                   np.int64)
    brC = np.array([S.ENT_OFF + b for _, _, _, b, _ in LC], np.int64)
    scC = np.array([S.ENT_OFF + int(facts[e, r2]) for e, _, r2, _, _ in LC],
                   np.int64)
    # ENT-YOK (AUDIT (a), 14 Eylul): kisayolun TIP OLARAK imkansiz oldugu ENT
    # zincirleri. Bu betik onu HIC olcmuyordu. Neden gerekli: AYIRT'ta r1'i
    # kesince kisayol firliyor (+0.33) ve bunun iki okumasi var --
    #   "kopru hesabi yikildi, kutle kisayola dustu"  (genel)
    #   "r1 gidince erisilebilir tek varlik e kaldi"  (AYIRT'a ozgu degil)
    # ENT-YOK'ta kisayol KOVASI YOK; kutlenin nereye gittigi temiz gorunur.
    # Ayni oranda coker  -> kopru hesabi yikiliyor, genel
    # Az duser           -> AYIRT'a ozgu bir sey oluyor
    LY = _sub(S.ENT_YOK, 5, a.n) if getattr(S, "ENT_YOK", None) else []
    EY = S.enc_two(LY) if LY else None
    brY = (np.array([S.ENT_OFF + b for _, _, _, b, _ in LY], np.int64)
           if LY else None)
    scY = (np.array([S.ENT_OFF + int(facts[e, r2]) for e, _, r2, _, _ in LY],
                    np.int64) if LY else None)

    sd, bulunan = yukle(a.klasor, a.tohum,
                        [int(x) for x in a.adim.split(",")])
    net = S.Net("A", S.CFG).to(S.DEV)
    net.load_state_dict(sd)
    net.eval()
    L = S.CFG["L"]

    def olc(poz, b0, tek=False):
        """tek=False -> KUYRUK maskesi (i >= b0).  tek=True -> YALNIZ blok b0.

        TEK BLOK modu (AUDIT, 14 Eylul). CLAUDE.md 0 bunu dort aydir bilinen
        gedik diye yaziyor: "Asama B bir blok kuyrugu (b0) da donduruyor ama
        kullanilmiyor -- pozisyon aramadan geliyor, DERINLIK gelmiyor."
        Kuyruk taramasi "kisayol hangi blokta hesaplaniyor" sorusunu SORAMAZ.
        Ayrica iki canli tarifi ayiran tek olcum bu:
          "maskenin bu veride HEDEFI YOKTU"   vs
          "hedef VARDI ama kuyruk maskesinin ULASAMAYACAGI yerdeydi"
        """
        for i, blk in enumerate(net.blocks):
            acik = (i == b0) if tek else (i >= b0)
            blk.mask_key = poz if (poz is not None and acik) else None
        zA, _ = S.zengin(net, EA, brA, scA)
        zC, _ = S.zengin(net, EC, brC, scC)
        # COMP'un KISAYOL orani (AUDIT (b)): `zengin` bunu ZATEN hesapliyordu,
        # eski kod zC["shortcut"]'i DUSURUYORDU. COMP'ta varlik GORULMUS; orada
        # da kisayol benzer sekilde firliyorsa etki ENT'e OZGU DEGILDIR.
        zY = (S.zengin(net, EY, brY, scY)[0] if EY is not None
              else {"acc": float("nan")})
        for blk in net.blocks:
            blk.mask_key = None
        return (zA["acc"], zA["shortcut"], zC["acc"], zC["shortcut"],
                zY["acc"])

    print("=" * 78)
    print(f"MASKE TARAMA   {a.klasor}   adimlar {bulunan}")
    print(f"  olcum: ENT-ARAMA {len(LA)} ornek (HUKUM kumesi DEGIL), "
          f"COMP {len(LC)}")
    print("  DIKKAT: maske INFERENCE'ta takiliyor, EGITIM yok.")
    print("=" * 78)
    e0, k0, c0, ck0, y0 = olc(None, 0)
    print(f"  MASKESIZ      ent {e0:.4f}   kisayol {k0:.4f}   comp {c0:.4f}"
          f"   comp-ksy {ck0:.4f}   ent-yok {y0:.4f}")
    print(f"  ENT-YOK ornek: {len(LY)}   (kisayol TIP OLARAK imkansiz)")
    print()
    _bas = (f"  {'poz':>4} {'bloklar':>9} {'ent':>8} {'kisayol':>9} {'comp':>8}"
            f" {'cmpKSY':>7} {'entYOK':>7}   {'d(ksy)':>9} {'d(comp)':>9}"
            f" {'d(cKSY)':>9} {'d(eYOK)':>9}")

    def tur(tek):
        """tek=False: KUYRUK (b0..L-1).  tek=True: TEK BLOK (yalniz b)."""
        print(("  --- TEK BLOK TARAMASI (yalniz o blok maskeli) ---" if tek
               else "  --- KUYRUK TARAMASI (b0'dan sonuna) ---"))
        print(_bas)
        r = []
        for poz in (1, 2, 3):
            for b0 in range(L):
                e1, k1, c1, ck1, y1 = olc(poz, b0, tek=tek)
                r.append((poz, b0, e1, k1, c1, ck1, y1))
                et = f"{b0}" if tek else f"{b0}..{L-1}"
                print(f"  {poz:>4} {et:>9} {e1:8.4f} {k1:9.4f} {c1:8.4f} "
                      f"{ck1:7.3f} {y1:7.3f}   {k1-k0:+9.4f} {c1-c0:+9.4f} "
                      f"{ck1-ck0:+9.4f} {y1-y0:+9.4f}")
        print()
        return r

    sonuc = tur(tek=False)
    tur(tek=True)
    # ETKI DENETIMI (CLAUDE.md 7). Butun adaylar tabanla AYNI cikarsa bulgu
    # "etki yok" DEGIL, "olcum bozuk"tur: maske takilmamis ya da model zaten
    # her seye ayni cevabi veriyor olabilir. Ikisi de raporlanacak sey degil.
    _fark = max(abs(x[2] - e0) + abs(x[3] - k0) + abs(x[4] - c0)
                for x in sonuc)
    assert _fark > 1e-6, (
        "MASKE ETKISIZ -> olcum bozuk (CLAUDE.md 7)." + chr(10)
        + f"  taban ent {e0:.4f} kisayol {k0:.4f} comp {c0:.4f}; "
          f"24 adayin HEPSI ayni." + chr(10)
        + "  olasi sebep: model henuz hicbir sey ogrenmemis "
          "(hepsi 0.0000) ya da mask_key takilmiyor.")

    print()
    # ASIL SORU: kisayolu EN COK dusuren, COMP'u COKERTMEYEN aday hangisi?
    # 'Cokertmeyen' esigi ONCEDEN degil BURADA yaziliyor -- bu bir TANI,
    # secim degil. Esik 0.10: D3.3'un SAGLIK-comp kapisiyla ayni buyukluk.
    uygun = [x for x in sonuc if x[4] >= c0 - 0.10]
    if uygun:
        en = min(uygun, key=lambda x: x[3])
        print(f"  KISAYOLU EN COK DUSUREN (comp > {c0-0.10:.3f} sartiyla):")
        print(f"     poz {en[0]}  bloklar {en[1]}..{L-1}   "
              f"kisayol {k0:.4f} -> {en[3]:.4f}   comp {c0:.4f} -> {en[4]:.4f}"
              f"   ent {e0:.4f} -> {en[2]:.4f}")
    else:
        print(f"  !! HICBIR aday comp'u {c0-0.10:.3f} ustunde tutamiyor")
    print()
    print("  Bu bir TANIDIR, SECIM DEGIL. FAZ 3 kendi aramasini yapar.")


if __name__ == "__main__":
    main()
