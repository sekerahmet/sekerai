# -*- coding: utf-8 -*-
"""veri_00 — model_00'in KENDI veri modulu.

Kullanici karari, 16 Eylul 2026:
    *"verisi de veri_00 olsun lutfen, okul degil veri_00 olacak"*

`ayar_00.py` ile ayni gerekce: `model_00` "her seyden bagimsiz" bir
referans kol. Ayarini artik kendi dosyasindan okuyor; verisini de
kendi adiyla cagiriyor -- `model_b` ailesinin dosya adina bagli DEGIL.

==========================================================================
ICERIK model_b15 ILE AYNI -- VE BU KASITLI

Bagimsizlik MIMARI icin gecerli, veri icin DEGIL. Ayni veriyi gormezse
kol hicbir sey olcmez: sinav bolmeleri ayrisir, `olcme_izi` degisir ve
sayilar ayni tabloda okunamaz. Bu yuzden bugun icerik `veri_okul4`un
AYNISI ve modul ona DEVREDIYOR:

    veri_okul4 = veri_okul (1x graf) + veri_okul3'un sekiz alan acilimi
                 1060 varlik   10.960 olgu   104.940 zincir   |R| = 17
                 phi TAVANI 9,57

==========================================================================
!! DEVRETME TEK BASINA YETMEZ -- IZ BU YUZDEN VAR

Duz bir `from veri_okul4 import *` sessiz bir tuzak olurdu: yarin bir
`model_b` kolu `veri_okul4`u degistirirse `veri_00` de onunla birlikte
kayar, ve `test_00.py`nin "model_b15 ile birebir ayni" karsilastirmasi
bunu YAKALAMAZ -- cunku iki taraf birlikte hareket eder.

Bu yuzden asagidaki `IZ`, `model_00`un gordugu grafin parmak izidir ve
`kur()` her cagrilista onu DENETLER. `veri_okul4` degisirse kol sessizce
baska bir veriyle kosmaz: burada DURUR.

O gun verilecek karar ikiden biridir ve ikisi de bilinerek verilir:

    1) degisikligi KABUL ET   -> IZ yenilenir, onkayda NOT DUSULUR
                                 (olcme izi degisir; CLAUDE.md
                                 "KIYAS ARTIK ARKA PLANDA" -> kol iptal
                                 degil, not duselir)
    2) SABITLE                -> devretme kaldirilir, veri_00 grafi
                                 kendi icinde kurar

IZ yalniz tohum 0 icin sabittir (`model_b15` ve `model_00` ikisi de
`veri_tohum=0` kullaniyor). Baska tohum denetlenmez -- denetlenseydi
her tohum icin ayri bir sabit tutmak gerekirdi ve o sabitler kullanilmaz
halde eskirdi.
"""
from __future__ import annotations

import hashlib

import veri_okul4 as V4

# --- model_a.veri_kur SOZLESMESI: kur / zincirler / TIPLER / ILISKI -----
# (veri_dok.py ayrica SEMA, graf_dok.py kur kullaniyor.)
TIPLER = V4.TIPLER
ILISKI = V4.ILISKI                     # 17 SEMBOL
BLOK = V4.BLOK
SEMA = V4.SEMA
GEREKTIRIR = V4.GEREKTIRIR

zincirler = V4.zincirler
yaz = V4.yaz
turetilebilir = V4.turetilebilir

# tohum 0 grafinin parmak izi -- olculdu 16 Eylul 2026.
IZ = "42a423f41000"


def graf_izi(G):
    """Grafin ICERIGINDEN tureyen sabit parmak izi.

    Sozluk sirasina ve id()'lere bagli DEGIL: olgular, iliski sembolleri
    ve tip basina varlik adlari SIRALANIP karma alinir.
    """
    h = hashlib.sha256()
    h.update(repr(sorted(map(str, G["olgu"]))).encode())
    h.update(repr(sorted(G["iliski"])).encode())
    h.update(repr({k: sorted(v) for k, v in sorted(G["ad"].items())}).encode())
    return h.hexdigest()[:12]


def kur(tohum=0):
    """`veri_okul4` grafi -- tohum 0'da ICERIGI IZ ile denetlenir."""
    G = V4.kur(tohum)
    if tohum == 0:
        _iz = graf_izi(G)
        assert _iz == IZ, (
            f"veri_00: graf DEGISTI  {IZ} -> {_iz}\n"
            "  `veri_okul4` (ya da dayandigi veri_okul / veri_okul3)\n"
            "  degistirilmis. model_00 baska bir veriyle SESSIZCE kosmaz.\n"
            "  Karar: (1) degisikligi kabul et -> IZ'i yenile + onkayda\n"
            "  not dus, ya da (2) veri_00'i devretmekten cikarip sabitle.")
    return G


if __name__ == "__main__":
    G = kur(0)
    z = zincirler(G)
    print("veri_00  varlik %d  olgu %d  zincir %d  |R| %d  phi TAVANI %.2f"
          % (sum(G["n"].values()), len(G["olgu"]), len(z), len(ILISKI),
             len(z) / len(G["olgu"])))
    print("graf izi", graf_izi(G), "  (beklenen", IZ + ")")
