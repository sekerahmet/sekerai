# -*- coding: utf-8 -*-
"""D3.3 — maskeleme kazanci phi ile hayatta kaliyor mu.

Onceden kayit: ONKAYIT_D33_PHI_EKSENI.md  (kosudan once yazildi, degismez)

    A5   maskesiz,                        phi 5.06, 0 -> 120.000
    D5   MASK_KEY=1 MASK_BLK=1..7,        phi 5.06, 0 -> 120.000
         (D3'un konfiginin AYNISI, sadece phi farkli)

Dusuk-phi karsiliklari ZATEN olculdu (A 0.0603 / D3 0.3607, oran 5.98x),
o yuzden burada sadece iki yeni kol var.

ARAMA YOK: bu deney tarifi degil, ALTINDAKI ETKIYI siniyor. Etki yoksa
tarif zaten konusuz. Arama olmayinca parcalamaya da gerek yok -> her kol
TEK surecte kosar (D3.2'de ek yukun %95 oldugu olculdu).

    python d33.py <konfig_giris.json>
"""
import os, sys, json, time, glob, subprocess

KON = json.load(open(sys.argv[1]))
sys.path.insert(0, KON["KOD"])
sys.path.insert(0, os.path.join(KON["KOD"], "sablon"))

for _v in ("MASK_KEY", "MASK_BLK", "RESUME_FROM", "INIT_FROM", "OUT",
           "SHARE", "IDENT_MODE", "STEPS", "WARM_OF"):
    os.environ.pop(_v, None)

SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    KON["ORT"] = dict(KON["ORT"], PRESET="smoke", MEM_AT="2", EVERY="200",
                      COMPILE="0", N_ENT="200", N_REL="6", N_PAIR="20",
                      P_TRAIN="16")
# Elle kopyalanmis filtre YOK: ORT tek kaynak (defterdeki liste ile burasi
# birbirini tutmuyordu; veriyi belirleyen bir anahtar eklenirse surucu ile
# pencere.py FARKLI veri kurabilirdi).
assert not (set(KON["ORT"]) & {"MASK_KEY", "MASK_BLK", "RESUME_FROM",
                               "INIT_FROM", "OUT", "STEPS"}),     f"ORT kosuya ozel degisken tasiyor: {KON['ORT']}"
os.environ.update(KON["ORT"])

import numpy as np
import torch
import sifirdan as S
from kosu import Kosu

A5, D5, K5 = "cikti_a5", "cikti_d5", "cikti_k5"

KON.update(
    DERINLIK=list(range(1, S.CFG["L"])),           # D3'un konfigi
    HEDEF_SON=120000,
    KOL_SAYISI=3,          # rapor ilerlemeyi UC kol uzerinden gostersin
    RAPOR_TOPLAM=3 * 120000,
    PENCERE=[60000, 65000, 70000, 75000, 80000],
    PHI_BEKLENEN=5.06,     # onkayit 3'te olculen deger; asagida ASSERT edilir
    OLGUNLUK=0.50,                                 # comp(A5 @ pencere)
    MEKANIZMA=0.30,                                # kisayol(A5 @ pencere)
    TOL_YORUNGE=9.9,                               # referans kol yok
    REF_AD="A5",
    KONTROL_ALT=A5,
    # FAZ 1 yazilmali: bu deneyde maske ADIM 0'dan acik, faz gecisi YOK.
    # rapor.py fazi atesleme adimindan turetiyor; asamaB olmadigi icin
    # at=None -> f hep 1. Faz-2 yazilsaydi uyarilar HIC atesmezdi.
    UYARI={"SAGLIK":  ["comp", 0.10, -1, 1],
           "YOL":     ["ent", 0.00, -1, 1],
           "DOLANMA": ["ent_shortcut", 0.00, +1, 1]},
)
if SMOKE:
    KON.update(RAPOR_TOPLAM=3 * 600,
               HEDEF_SON=600, PENCERE=[400, 600], OLGUNLUK=-1.0,
               MEKANIZMA=-1.0, PHI_BEKLENEN=2.17)   # smoke'un gercek phi'si
# YALNIZ DUMAN TESTINDE ezilebilir. Gecen tur bunu kosulsuz eklemistim:
# deneyin TANIMLAYICI buyuklugunu koruyan kapiyi, test kolayligi ugruna
# her ortam degiskenine acmis oldum — ve HUCRE 3'un temizlik listesinde de
# yoktu, yani kosular arasi SIZABILIRDI. Tam da N_PAIR=80'i basimiza acan
# hata sinifi. Gercek kosuda kapi artik susturulamaz.
if SMOKE and "PHI_BEKLENEN" in os.environ:
    KON["PHI_BEKLENEN"] = float(os.environ["PHI_BEKLENEN"])

K = Kosu(KON)
DERINLIK = tuple(KON["DERINLIK"])
HEDEF_SON = KON["HEDEF_SON"]
MBLK = ",".join(str(b) for b in DERINLIK)

# phi'yi KODDAN hesapla, elle yazma (CLAUDE.md 6)
_d = S.build_data()
_atom = S.CFG["N_ENT"] * S.CFG["N_REL"]
PHI = len(_d[3]) / _atom
K.log(f"D3.3 basliyor  commit {KON['commit']}")
K.log(f"  veri {S.CFG['N_ENT']}x{S.CFG['N_PAIR']} ({S.CFG['P_TRAIN']} egitimde) "
      f"R={S.CFG['N_REL']}  ->  PHI = {PHI:.2f}   (dusuk-phi referans: 3.03)")
K.log(f"  atomik olgu {_atom}  egitim-2hop {len(_d[3])}  "
      f"COMP {len(_d[4])}  ENT {len(_d[5])}  ENT2 {len(_d[8])}")
K.log(f"  kollar: A5 (maskesiz) | D5 (poz 1 @ {MBLK}) | "
      f"K5 (poz 2 @ {MBLK}, KONTROL: eslesmis maliyet, yanlis yer)")

# PHI KAPISI — bu deneyin TANIMLAYICI buyuklugu. Loglamak yetmez: yanlis
# P_TRAIN ile baslarsak phi 3.03'te kalir ve bir saat boyunca D3'un
# kopyasini kosariz, sonucu da "phi ile hayatta kaldi" diye okuruz.
if KON.get("PHI_BEKLENEN") is not None:
    assert abs(PHI - KON["PHI_BEKLENEN"]) < 0.05, (
        f"PHI={PHI:.3f} ama onkayit {KON['PHI_BEKLENEN']} diyor -> DURDURULDU. "
        f"(N_ENT={S.CFG['N_ENT']} N_REL={S.CFG['N_REL']} "
        f"N_PAIR={S.CFG['N_PAIR']} P_TRAIN={S.CFG['P_TRAIN']})")
    K.log(f"  PHI KAPISI GECTI   {PHI:.3f} ~ {KON['PHI_BEKLENEN']}")
K.kaydet(rapor_ek=[f"phi = {PHI:.2f}  (dusuk-phi referans 3.03)",
                   f"A5 maskesiz | D5 poz 1 @ {MBLK} | K5 poz 2 (KONTROL)",
                   f"dusuk-phi sonuc: A 0.0603  D3 0.3607  oran 5.98x"])

# ============================================================== ANA DONGU
#   Her kol TEK surecte. Surdurme paketi varsa kaldigi yerden devam eder,
#   yani kopma halinde HUCRE 0-4 tekrar kosulunca kayip olmaz.
# K5: ESLESMIS MALIYET, YANLIS YER. Deney 7'de K kolu bir bulguyu oldurdu
# ("maskeleme ogrenmeyi hizlandiriyor") ve digerini kurtardi. CLAUDE.md 10b:
# "Kontrolu sonraya birakma. Kontrol kolu tasariminin parcasidir, ek degil."
for ad, alt, ek in (("A5", A5, None),
                    ("D5", D5, dict(MASK_KEY="1", MASK_BLK=MBLK)),
                    ("K5", K5, dict(MASK_KEY="2", MASK_BLK=MBLK))):
    # Bayrak YETMEZ: durum.json "bitti" derken klasor bos olabilir
    # (Drive geri yuklemesi yarim kaldi). Atlanirsa olcum eksik kalirdi.
    if K.bitti_mi(ad, alt, KON["PENCERE"]):
        K.log(f"{ad} zaten bitmis (olcum noktalari diskte dogrulandi), atlaniyor")
        continue
    K.kaydet(faz=1 if ad == "A5" else 2,
             faz_ad=f"{ad} kosuyor ({'maskesiz' if ek is None else 'maskeli'}), "
                    f"hedef {HEDEF_SON}")
    t = time.time()
    sn = K.egit(HEDEF_SON, ek, alt=alt)
    # Ilerleme UC kolun toplami uzerinden. Onceki hali her kol bitince
    # adim=HEDEF_SON yaziyordu -> A5 biter bitmez rapor %100 diyordu,
    # oysa isin ucte ikisi duruyordu.
    _bitti = sum(1 for x in ("A5", "D5", "K5") if K.st.get(f"{x}_bitti"))
    K.kaydet(**{f"{ad}_bitti": True}, adim=(_bitti + 1) * HEDEF_SON)
    K.log(f"{ad} bitti ({sn} sn = {sn/60:.0f} dk)")
    if ek is not None:
        K.konfig_kapisi(dict(MASK_KEY=ek["MASK_KEY"],
                             MASK_BLK=list(DERINLIK)), alt=alt)

K.kaydet(faz=3, faz_ad="BITTI", adim=KON["KOL_SAYISI"] * HEDEF_SON)
_np = len(KON["PENCERE"])          # elle "/5" yazma yok (CLAUDE.md 6)
h = lambda a: sum(os.path.exists(
    K.y(a, f"snap_{KON['KOL']}_s{KON['SEED']}_{s:06d}.pt")) for s in KON["PENCERE"])
K.log(f"BITTI.  pencere noktalari: A5 {h(A5)}/{_np}  D5 {h(D5)}/{_np}  "
      f"K5 {h(K5)}/{_np}")

# Birincil okuma komutu — ONCEDEN YAZILAN KAPILAR da komutun icinde.
# Kapiyi sadece insan okursa unutulabilir ya da sonucu gorup gevsetilebilir.
_p = ",".join(str(s) for s in KON["PENCERE"])
_mb = f"{DERINLIK[0]}-{DERINLIK[-1]}"
ARG = ["--cikti", K.y("BIRINCIL_D33.json"),
       "--kol", f"A5:{K.y(A5)}:{_p}:yok",
       "--kol", f"D5:{K.y(D5)}:{_p}:1@{_mb}",
       "--kol", f"K5:{K.y(K5)}:{_p}:2@{_mb}",
       # Etiketli: ciktida hangi kapi oldugu ve GECMEK ne demek okunsun.
       # ANTITEZ kapisi TERS yonlu — gecmesi "antitez KAYBETTI" demek.
       "--kapi", f"OLGUNLUK: A5 comp >= {KON['OLGUNLUK']}",
       "--kapi", f"MEKANIZMA-kisayol-var: A5 ent_kisayol >= {KON['MEKANIZMA']}",
       "--kapi", "BIRINCIL-kazanc-yasiyor: D5/A5 ent >= 3.0",
       "--kapi", "YER-mi-MALIYET-mi: D5/K5 ent >= 2.0",
       "--kapi", "ANTITEZ-KAYBETTI: A5 ent < 0.18",
       "--kapi", "OZGULLUK-ENT2-oynamadi: D5-A5 ent2 < 0.05",
       "--kapi", "SAGLIK-comp: D5-A5 comp >= -0.10",
       "--kapi", "SAGLIK-1hop: D5 bir_hop >= 0.98"]
open(K.y("BIRINCIL_KOMUT.sh"), "w").write(
    "python sablon/pencere.py "
    + " ".join((f"'{x}'" if " " in x else x) for x in ARG) + "\n")

# BIRINCIL OKUMAYI SURUCU KOSAR. Kapilari bir .sh dosyasina birakmak
# "kapiyi kod degerlendirsin" amacini bosa cikariyordu: dosyayi kosmak yine
# insana kaliyordu, unutulabilir ya da sonucu gorup atlanabilirdi.
K.log("\n  BIRINCIL OKUMA (surucu kosuyor):")
_r = subprocess.run([sys.executable, "-u",
                     os.path.join(KON["KOD"], "sablon", "pencere.py")] + ARG,
                    cwd=KON["KOD"], env=dict(KON["TEMIZ"], **KON["ORT"]),
                    capture_output=True, text=True)
for _l in (_r.stdout or _r.stderr).splitlines():
    K.log("    " + _l)
if _r.returncode == 3:
    K.log("  !! BIRINCIL OKUMA KISMI — bazi kollar olculemedi. Sonuc dosyasi "
          "YAZILDI ama eksik kola bagli kapilar ATLANDI, hukum verilemez.")
elif _r.returncode != 0:
    K.log("  !! BIRINCIL OKUMA COKTU — elle kos: BIRINCIL_KOMUT.sh")
K.not_(f"BITTI — pencere noktalari A5 {h(A5)}/{_np} D5 {h(D5)}/{_np} K5 {h(K5)}/{_np}",
       f"phi = {PHI:.2f}   dusuk-phi referans: A 0.0603 / D3 0.3607 = 5.98x",
       f"okuma komutu: BIRINCIL_KOMUT.sh")
K.log("SURUCU BITTI")
