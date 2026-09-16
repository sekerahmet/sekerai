# -*- coding: utf-8 -*-
"""test_sabit — MODEL_A DEGISMEZ. Bunu her degisiklikten sonra kanitlar.

    python test_sabit.py

Kullanici karari, 15 Eylul 2026: *"veri kismini degiskene bagla ve model_a
sabit olsun, bozulmadigindan emin ol."*

NEDEN VAR
---------
`model_a.py` dort kolun (a, a1, a2, a3) ORTAK TABANI ve dordunun de
sonuclari kayitli. Tabana dokunan her degisiklik, kayitli sonuclari
sessizce gecersiz kilabilir.

15 Eylul'de `veri_ad` alani eklendi (veri kaynagini DEGISKENE baglamak
icin) ve bu degisiklik IKI YERI KIRDI:
    surdurme_oku        butun eski kosularin SURDURULMESI
    pencere_a.ayar_oku  butun eski kosularin OLCULMESI
Ikisi de duzeltildi. Ama "duzelttim, baktim, tamam" bir kereliktir;
bu dosya onu MEKANIK yapar.

BURADAKI SAYILAR ELLE DEGISTIRILMEZ. Test duserse iki ihtimal var:
    1. Bir seyi bozdun            -> KODU duzelt
    2. Bilerek degistirdin        -> sayiyi degil, GEREKCEYI once yaz
                                    (belge/onkayit/), sonra sayiyi guncelle
Sayiyi sessizce guncellemek, testi yok saymakla ayni seydir.
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model_a as M                                          # noqa: E402

# ------------------------------------------------------------------ KILIT
# 15 Eylul 2026, commit 481c886 (veri_ad EKLENMEDEN ONCEKI hal) ile olculdu.
AYAR_KILIT = dict(
    ad="model_a", adim=20000, arama_pay=0.25, batch=512, betas=(0.9, 0.999),
    comp_pay=0.1, d=256, dff=1024, dongu=2, ent_pay=0.2, ident_frac=0.0,
    ident_kip="", isinma=2000, l=4, lr=0.001, mask_blok=(), mask_poz=None,
    n_olcum_max=3000, nh=4, olc_her=2000, sabit_lr=True, tohum=0,
    veri_tohum=0, wd=0.1,
    veri_ad="veri_okul",          # 15 Eylul'de EKLENDI, varsayilani eski yol
    # GERI BESLEMELI ORTALAMA (Lookahead) -- 15 Eylul'de EKLENDI.
    # Kilit bu uc alani YAKALADI ve dogru davrandi: 12 kontrol dustu,
    # 8'i eski kosu klasorlerinin artik OKUNAMAYACAGINI gosteriyordu.
    # Gerekce ONCE yazildi: belge/onkayit/model_a5.md (testin kendi
    # kurali). Sonra `ESKI_VARSAYILAN`a eklenip kilit guncellendi.
    # UCU DE VARSAYILAN KAPALI -- model_a..a4'un davranisi degismiyor,
    # ve bunu asagidaki VERI/PARAMETRE kilidi dogruluyor.
    ort_bas=0, ort_her=0, ort_alfa=0.5,
    # ENT-KATI bolmesi 15 Eylul'de eklendi (Wang 2405.15071'in OOD'si).
    # Kilit bunu da YAKALADI: alan sayisi 28 -> 29, 2 kontrol dustu.
    # Gerekce ONCE yazildi: belge/onkayit/model_a7.md §6.
    # VARSAYILAN KAPALI -- asagidaki VERI kilidi model_a'nin gordugu
    # verinin BIT DUZEYINDE ayni kaldigini dogruluyor.
    kati_pay=0.0,
    # WANG'IN KENAR BOLMESI (ood_pay) 15 Eylul'de eklendi. Kilit yine
    # YAKALADI (alan 29 -> 30). Gerekce ONCE: belge/onkayit/model_a8.md §1
    # -- makale bastan sona okundu ve `kati_pay`in (model_a7) makalenin
    # %0'ini ureten kosulu SAGLAMADIGI olculdu (2. hop kenari egitimde
    # 2. hop olarak %93,7 geciyordu). VARSAYILAN KAPALI.
    ood_pay=0.0,
    # DARBOGAZ alanlari 16 Eylul'de eklendi -- YENI AILE `model_b` icin
    # (DiscoLoop, arXiv 2607.00341 Denk. 4-6). Gerekce ONCE yazildi:
    # belge/onkayit/model_b.md §3.
    # `model_a.Model` bu alanlari HIC OKUMAZ. Kanit asagida, 4. bolumde:
    # `parametre 3427840` ve `agirlik sha` DEGISMEDI. Ayrica
    # model_b/test_sabit_b.py, dar_alfa=0 iken ModelB'nin model_a.Model
    # ile BIT DUZEYINDE ayni cikti verdigini her kosuda dogruluyor.
    dar_alfa=0.0, dar_tau=1.0, dar_kapi=False,
    # `dar_sdpa` 16 Eylul'de eklendi -- `model_b2` icin. Kilit YINE
    # YAKALADI (alan 33 -> 34, "dar_sdpa KILITTE YOK"). Gerekce kilide
    # DOKUNULMADAN ONCE yazildi: belge/onkayit/model_b2.md.
    # Bu alan MODELI degistirmez, `Phi`nin HESAP YOLUNU degistirir:
    # Phi(h) = softmax(nf(h) Wᵀ/tau) @ W  ==  Attention(Q=nf(h), K=V=W).
    # `model_a.Model` `Phi`yi zaten hic cagirmaz. VARSAYILAN KAPALI --
    # asagidaki `parametre 3427840` ve `agirlik sha` bunu dogruluyor,
    # ayrica model_b/test_sabit_b.py §6 iki yolun ayni fonksiyonu
    # verdigini (fp32 bagil fark < 1e-5) her kosuda olcuyor.
    dar_sdpa=False,
    # `dar_sert` 16 Eylul'de eklendi -- `model_b3` icin. Kilit YINE
    # YAKALADI (alan 34 -> 35). Gerekce kilide DOKUNULMADAN ONCE
    # yazildi: belge/onkayit/model_b3.md.
    # Phi'nin `tau -> 0` limiti: W[argmax(nf(h) Wᵀ)] + straight-through.
    # `model_a.Model` Phi'yi zaten hic cagirmaz. VARSAYILAN KAPALI.
    dar_sert=False,
    # `jeton_ad` 16 Eylul'de eklendi -- `model_b5` icin (varliklar IKI
    # JETON: Ayse_Yilmaz -> Ayse + Yilmaz). Kilit YINE YAKALADI
    # (alan 35 -> 36). Gerekce kilide DOKUNULMADAN ONCE yazildi:
    # belge/onkayit/model_b5.md.
    # ORTAK CEKIRDEGE dokunan ILK kol: t_len 8 -> 11, cevap IKI jeton,
    # kayip IKI hedef. HEPSI `jeton_ad`a bagli; VARSAYILAN KAPALI ve
    # asagidaki `parametre 3427840` + `agirlik sha` + VERI kilidi
    # model_a'nin BIT DUZEYINDE degismedigini dogruluyor.
    jeton_ad=False,
)

VERI_KILIT = dict(
    n_ent=1060, vocab=1085, phi=5.085714,
    facts="74c98461df78cddf", tip="d2dbc79fec1cf08c",
    tr2=42720, comp=4281, ent=6665, ent_yok=1640, ent_arama=2064,
    tr2_h="94662b1d18b7d0df", ent_h="9459c4904ca96014",
    comp_h="78754be8545394c9", entyok_h="de5132c03940aaf4",
)
GRAF_SHA = "44a1016bd74f894b"      # veri_okul.kur(0) olgu sozlugu
PARAMETRE = 3427840
AGIRLIK_SHA = "b27ae8bada5893d7"   # torch.manual_seed(0) -> Model(AYAR, 1085)

_h = lambda x: hashlib.sha256(np.asarray(x).tobytes()).hexdigest()[:16]
_iyi = _kotu = 0


def _bak(ad, beklenen, olculen, sert=True):
    global _iyi, _kotu
    ok = beklenen == olculen
    if ok:
        _iyi += 1
    elif sert:
        _kotu += 1
    im = "  ok  " if ok else ("!! BOZUK" if sert else "  ~~  ")
    if not ok or os.environ.get("AYRINTI"):
        print(f"  {im} {ad:<24} beklenen {beklenen!r:<22} olculen {olculen!r}")
    return ok


def main():
    print("=== 1) model_a.AYAR degismedi mi ===")
    d = M.AYAR.sozluk()
    _bak("alan sayisi", len(AYAR_KILIT), len(d))
    for k in sorted(AYAR_KILIT):
        _bak(k, AYAR_KILIT[k], d.get(k))
    for k in sorted(set(d) - set(AYAR_KILIT)):
        print(f"  !! BOZUK {k:<24} KILITTE YOK, Ayar'a eklenmis: {d[k]!r}")
        globals()["_kotu"] = globals()["_kotu"] + 1

    print("=== 2) veri_okul grafi degismedi mi ===")
    import veri_okul as VO
    import json as _j
    G = VO.kur(0)
    sha = hashlib.sha256(_j.dumps(
        {f"{a}|{b}": c for (a, b), c in G["olgu"].items()},
        sort_keys=True).encode()).hexdigest()[:16]
    _bak("olgu sha", GRAF_SHA, sha)

    print("=== 3) model_a'nin GORDUGU veri degismedi mi ===")
    v = M.veri_kur(M.AYAR, yaz=lambda *a, **k: None)
    o = dict(n_ent=v.n_ent, vocab=v.vocab, phi=round(v.phi, 6),
             facts=_h(v.facts), tip=_h(v.tip), tr2=len(v.tr2),
             comp=len(v.comp), ent=len(v.ent), ent_yok=len(v.ent_yok),
             ent_arama=len(v.ent_arama), tr2_h=_h(np.array(v.tr2)),
             ent_h=_h(np.array(v.ent)), comp_h=_h(np.array(v.comp)),
             entyok_h=_h(np.array(v.ent_yok)))
    for k in sorted(VERI_KILIT):
        _bak(k, VERI_KILIT[k], o[k])

    print("=== 4) model BASLANGICI degismedi mi ===")
    torch.manual_seed(0)
    net = M.Model(M.AYAR, v.vocab)
    _bak("parametre", PARAMETRE, net.n_param())
    w = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes()
                                for p in net.parameters())).hexdigest()[:16]
    # YUMUSAK: agirlik baslangici torch surumune bagli olabilir. Duserse
    # KOSU gecersiz demek DEGIL; ama ayni makinede duserse init degismistir.
    if not _bak("agirlik sha", AGIRLIK_SHA, w, sert=False):
        print(f"       (yumusak: torch {torch.__version__} -- surum farkiysa "
              f"beklenir, AYNI makinede degistiyse INIT degismistir)")

    print("=== 5) AILENIN DUGME YAPISI ===")
    # Her kol, TABANINDAN tam olarak SU alanlarda ayrilmali. `ad` her zaman
    # ayrilir (cikti dosya adlarina giriyor). Fazladan bir alan ayrilirsa
    # "tek dugme" iddiasi coker ve kiyas yorumlanamaz hale gelir.
    import model_a1, model_a2, model_a3, model_a4, model_a5, model_a6, model_a7, model_a8, model_a9, model_a10
    DUGME = (
        ("model_a1", M.AYAR, model_a1.AYAR, {"ad", "dongu"}),
        ("model_a2", M.AYAR, model_a2.AYAR, {"ad", "l", "dongu"}),
        ("model_a3", M.AYAR, model_a3.AYAR, {"ad", "wd"}),
        # model_a4'un TABANI model_a DEGIL model_a3 -- sorulan sey
        # "wd calisirken olcek ne yapiyor", o yuzden wd=0.5 zemininde.
        ("model_a4", model_a3.AYAR, model_a4.AYAR, {"ad", "veri_ad"}),
        # model_a5'in TABANI model_a4 -- "model_a4 standartlarinda"
        # geri beslemeli ortalama (kullanici, 15 Eylul).
        ("model_a5", model_a4.AYAR, model_a5.AYAR, {"ad", "ort_bas"}),
        # model_a6'nin TABANI model_a5 -- ayrilan soru "model_a5'in
        # sonucu FIKRIN mi `k`NIN mi", o yuzden model_a5 zemininde.
        ("model_a6", model_a5.AYAR, model_a6.AYAR, {"ad", "ort_her"}),
        # model_a7'nin TABANI model_a5 (model_a6 DEGIL): `k` ekseni
        # kapandi (a6-a5 = +0.0024), o yuzden sinav zorlasirken ailenin
        # en iyi TEK anlik goruntusunu veren kola donuluyor.
        ("model_a7", model_a5.AYAR, model_a7.AYAR, {"ad", "kati_pay"}),
        # model_a8'in TABANI model_a5 (model_a7 DEGIL): kati_pay ayri bir
        # soru, ikisini ayni kosuda karistirmak tek dugme ilkesini bozar.
        ("model_a8", model_a5.AYAR, model_a8.AYAR, {"ad", "ood_pay"}),
        # model_a9  : model_a8'in grafi degisti (veri_okul2 -> veri_wang)
        # model_a10 : a9'un PAYLASIMSIZ kontrolu. IKI dugme ama TEK kavram:
        #   `dongu`yu tek basina 1 yapmak hesabi da yariya indirirdi ve
        #   "paylasim mi yoktu, hesap mi yetmedi" AYRILAMAZDI. `l=8` hesabi
        #   esitler. model_a <-> model_a2 ciftiyle AYNI dugme yapisi.
        ("model_a9", model_a8.AYAR, model_a9.AYAR, {"ad", "veri_ad"}),
        ("model_a10", model_a9.AYAR, model_a10.AYAR, {"ad", "l", "dongu"}),
    )
    for ad, taban, kol, bek in DUGME:
        f = set(taban.fark(kol)) | {"ad"}
        _bak(f"{ad} dugmeleri", sorted(bek), sorted(f))

    print("=== 6) ESKI kosular hala okunuyor / surdurulebiliyor mu ===")
    import glob
    kok = os.environ.get("KOSU_KOK", r"G:/Drive'\u0131m".encode().decode("unicode_escape"))
    bulunan = sorted(glob.glob(os.path.join(kok, "model_a*", "t*", "ayar_t*.json")))
    if not bulunan:
        print(f"  ~~  kosu klasoru yok ({kok}) -- bu adim ATLANDI")
    else:
        import pencere_a as P
        for y in bulunan:
            kl = os.path.dirname(y)
            try:
                a = P.ayar_oku(kl)
                _bak(os.path.relpath(kl, kok).replace("\\", "/"), True, True)
            except Exception as e:
                print(f"  !! BOZUK {os.path.relpath(kl, kok)}: {e}")
                globals()["_kotu"] = globals()["_kotu"] + 1

    print()
    print(f"{_iyi} gecti, {_kotu} BOZUK")
    if _kotu:
        print()
        print("!! MODEL_A DEGISMIS. Kayitli dort kolun sonuclari bu tabana")
        print("   dayaniyor. Once KODU duzelt; bilerek degistiysen once")
        print("   gerekceyi belge/onkayit/'a yaz, sonra kiliti guncelle.")
    return 1 if _kotu else 0


if __name__ == "__main__":
    raise SystemExit(main())
