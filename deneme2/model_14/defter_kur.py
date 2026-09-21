# -*- coding: utf-8 -*-
"""model_14.ipynb'yi KURAR.  Defter elle duzenlenmez, BURADAN uretilir.

GEREKCE OLCULDU (21 Eylul): yerel defter 6 hucrede takili kalmisti,
Colab'daki ise 24 hucreye cikmisti -- 13'u tek seferlik olcum hucresi.
Ikisi arasinda hicbir bag yoktu ve `test_14`te defteri sinayan hicbir
kapi yoktu.  Defter artik kodun bir parcasi: burada uretilir, kapi 42
derlenebilirligini ve ayarla tutarliligini denetler.

CLAUDE.md: "Hucre sayisi sabit degil. Kolun sorusuna hizmet etmeyen
hucre duser."  Olcum hucreleri DUSTU -- sonuclari DENKLEM.md'de, ve
tek seferlik bir olcum defterin kalici parcasi degildir.
"""
from __future__ import annotations

import json
import os

MD = """# model_14 — YOL MODELI  ·  t0

`z_j = capa(R[w_{j-1}] z_{j-1})`,  okuma `w_j = argmax_w <Pi z_j / |Pi z_j|, p_w>`

**TASARIM 4** (`DENKLEM.md` §12c):

```
capa      KAPALI   (r = 0)      adresin TEK bozucusuydu      §5.1/T
VQ        KAPALI   (a2 = 0)     C'yi niceleyiciye zorluyordu §3.1b
kapi      ReLU(<zp,C> + hb)     softmax DEGIL -- yuvalar
                                yarisinca 8.192'nin 10'u
                                kaliyordu                    §5.1/U
okuma     TOPLAMSAL             z <- norm(z + g @ V)
kullanim  BEDELLI  (a4, butce)  ikameyi engeller             §12c
```

**Sira:** `H → 0 → 1 → 2 → 3 → 4`, arada `5 NABIZ`.
Izleme `R` ile, hicbir sey kosturmadan (CLAUDE.md kural 8).
`Y` yedekler, `X` durdurur, `G` modelin YAZDIGINA gozle baktirir.

Egitim her epok sonunda **surdurme paketi** yaziyor: agirlik + ADAM
durumu + RNG.  Cekirdek olurse `3` yeniden kosulur ve KALDIGI YERDEN
devam eder (kural 1).
"""

H = '''# H HAZIRLIK  |  CPU  |  KOSUDAN ONCE  |  eski deger BIRAKMAZ
# Uc yerde eski deger kalabilir; ucu de burada kapatiliyor.
#   1  Drive'daki t0/   YEDEGI BULUNUR, bayt bayt dogrulanir, silinir
#   2  cekirdekteki sonuc degiskenleri
#   3  olcum hucrelerinin biraktigi buyuk tensorler (GPU)
# !! Cekirdekteki ESKI `model_14` modulu ayri bir tehlike ve onu
# hucre 1 hallediyor (rm -rf + clone + sys.modules temizligi).
import os, gc, hashlib, json

_KAY = "/content/drive/MyDrive/model_14"
if not os.path.isdir(_KAY):
    from google.colab import drive
    drive.mount("/content/drive")
_T0 = _KAY + "/t0"
os.makedirs(_T0, exist_ok=True)


def _oz(y):
    h = hashlib.sha256()
    with open(y, "rb") as f:
        for p in iter(lambda: f.read(1 << 20), b""):
            h.update(p)
    return h.hexdigest()[:16]


def _ayni(a, c):
    if sorted(os.listdir(a)) != sorted(os.listdir(c)):
        return False
    return all(_oz(a + "/" + f) == _oz(c + "/" + f) for f in os.listdir(a))


print("=" * 66)
print("1) DRIVE t0/")
if os.listdir(_T0):
    _k = json.load(open(_T0 + "/dogruluk.json", encoding="utf-8"))
    print("   iceride: commit %s  %s" % (_k["commit"], _k["zaman"]))
    # !! YEDEK ARANIR, ADI VARSAYILMAZ.
    _bul = [d for d in sorted(os.listdir(_KAY))
            if d != "t0" and os.path.isdir(_KAY + "/" + d)
            and _ayni(_T0, _KAY + "/" + d)]
    assert _bul, ("t0/ icin bayt bayt ayni bir yedek YOK -- silmiyorum."
                  "  Once Y hucresini kosun.")
    print("   yedek BULUNDU: %s" % ", ".join(_bul))
    for _f in sorted(os.listdir(_T0)):
        print("   %-18s %s" % (_f, _oz(_T0 + "/" + _f)))
        os.remove(_T0 + "/" + _f)
    print("   -> t0/ TEMIZ")
else:
    print("   zaten bos")

print()
print("2) CEKIRDEK -- sonuc degiskenleri")
for _v in ("EGITIM", "OLCUM", "SONUC", "MDL", "ELOG", "OLOG", "EP",
           "T_EG", "T_OL", "KURULUM", "KLOG"):
    if _v in globals():
        print("   sil  %-8s %s" % (_v, str(globals()[_v])[:46]))
        del globals()[_v]

print()
print("3) BUYUK NESNELER")
_n = 0
for _v in [k for k in list(globals())
           if k.startswith("_") and k not in
           ("_KAY", "_T0", "_oz", "_ayni", "_v", "_n", "_bul", "_k", "_f")]:
    _o = globals()[_v]
    if hasattr(_o, "numel") or hasattr(_o, "nbytes"):
        del globals()[_v]
        _n += 1
print("   %d nesne silindi" % _n)
gc.collect()
try:
    import torch
    torch.cuda.empty_cache()
    _t, _b = torch.cuda.mem_get_info()
    print("   GPU  %.2f / %.1f GB" % ((_b - _t) / 1e9, _b / 1e9))
except Exception as _e:
    print("   GPU okunamadi:", _e)

print()
print("SIRA: 0 -> 1 (KLON, YENI kodu getirir) -> 2 -> 3")
'''

C0 = '''# 0 GPU KAPISI  |  GPU  |  tekrar: GUVENLI
# CLAUDE.md kural 2: GPU'yu kullanacak hucre GPU'yu KENDI ICINDE sorar.
# Ayri bir "GPU var mi" hucresi hucre SIRASINA bagli bir kuraldir --
# insan hatirlarsa calisir.  Gerekce olculdu: cekirdek kosunun
# ortasinda oldu ve defterdeki ayri kontrol hucresi kosturulmamisti.
import os, sys, time, json, threading, subprocess
import torch

assert torch.cuda.is_available(), "GPU YOK -- Runtime > Change runtime type"
_bos = torch.cuda.mem_get_info()[0] / 1e9
assert _bos > 2.0, "GPU'da sadece %.1f GB bos" % _bos
print("GPU kapisi GECTI: %s  bos %.1f GB"
      % (torch.cuda.get_device_name(0), _bos))
print("torch", torch.__version__)
'''

C1 = '''# 1 DEPO + DRIVE  |  CPU  |  tekrar: GUVENLI  |  ~15 sn
from google.colab import drive
drive.mount("/content/drive")

DEPO = "https://github.com/sekerahmet/sekerai.git"
KOD = "/content/kod"
subprocess.run(["rm", "-rf", KOD], check=True)
subprocess.run(["git", "clone", "-q", "--depth", "1", DEPO, KOD], check=True)
KOL = KOD + "/deneme2/model_14"
if KOL not in sys.path:
    sys.path.insert(0, KOL)

# !! MODUL ONBELLEGI. rm -rf + clone dosyayi degistirir ama `import`
# ONCEKI modulu dondurur -- model_13'te bir kez bunun kurbani olundu.
for _m in [m for m in list(sys.modules) if m.endswith("_14")]:
    del sys.modules[_m]

CIKTI = "/content/drive/MyDrive/model_14/t0"
BIRIM_YOL = "/content/drive/MyDrive/model_14/birim_14.npz"
os.makedirs(CIKTI, exist_ok=True)

# CLAUDE.md KURAL 9 -- MEKANIK KAPI.  "Drive'dan alinabilecek her sey
# Drive'dan."  Birim akisi Drive'dan GELIR; Colab onu URETMEZ.  Dosya
# yoksa burasi DURUR, sessizce 300 sn uretmeye baslamaz.
assert os.path.exists(BIRIM_YOL), (
    "BIRIM AKISI DRIVE'DA YOK: " + BIRIM_YOL + "\\n"
    "Colab korpus URETMEZ (kural 9). Yerelde uretip koyun.")

COMMIT = subprocess.run(["git", "-C", KOD, "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
print("commit", COMMIT)
print("cikti ", CIKTI)
print("birim ", BIRIM_YOL, "%.1f MB" % (os.path.getsize(BIRIM_YOL) / 1e6))
'''

C2 = '''# 2 VERI + SINAV  |  CPU  |  ARKA PLAN  (CLAUDE.md kural 8)
#   graf         tohumdan uretilir   ~1 sn
#   BIRIM AKISI  Drive'dan OKUNUR    ~4 sn
#
# !! `BR.yukle` -- `BR.kur` DEGIL.  `kur`, dosya yoksa URETIR; burada
# uretmek YASAK (kural 9).  Iz okunmuyor, akistan YENIDEN HESAPLANIP
# karsilastiriliyor: dosya bozuksa ya da baska bir bolucuyle
# uretilmisse burada patlar, sessizce gecmez.
KURULUM, KLOG = "kosuyor", []
T_KUR = [0.0, 0.0]


def _kur():
    global KURULUM
    T_KUR[0] = time.time()
    try:
        g = globals()
        import ayar_14 as AY, taban_14 as MT, veri_14 as V14
        import birim_14 as BR, olcme_14 as OL, model_14 as M14
        g.update(AY=AY, MT=MT, V14=V14, BR=BR, OL=OL, M14=M14)
        A = AY.AYAR

        v = MT.veri_kur(A, yaz=lambda *a, **k: None)
        L = MT.olcme_listeleri(A, v)
        assert V14.IZ == AY.IZ_GRAF, V14.IZ
        assert MT.olcme_izi(L) == AY.IZ_OLCME, MT.olcme_izi(L)
        KLOG.append("izler TUTUYOR  graf %s  olcme %s  (%.0f sn)"
                    % (V14.IZ, MT.olcme_izi(L), time.time() - T_KUR[0]))

        b = BR.yukle(BIRIM_YOL, yaz=KLOG.append)
        # PENCERE de denetleniyor: sinav cevabi sorunun OZNESINDEN
        # uretiyor; ikisi ayni zincire sigmazsa model o baglantiyi HIC
        # gormez.  Kapi bos degil, 16 ve 20 reddediliyor.
        KLOG.append(BR.kapi(b, AY.PENCERE))
        KLOG.append("ISINMA %d  (kayip konum %d..%d) -- DENKLEM §5.2"
                    % (AY.ISINMA, AY.ISINMA, AY.PENCERE - 1))

        G = V14.kur(A.veri_tohum)
        E_ad = [x for t in V14.TIPLER for x in G["ad"][t]]
        g.update(v=v, L=L, b=b, E_ad=E_ad)
        KURULUM = "bitti"
    except Exception as e:
        import traceback
        KURULUM = "HATA: %s: %s" % (type(e).__name__, e)
        KLOG.append(traceback.format_exc()[-700:])
    T_KUR[1] = time.time()
    for s in KLOG:
        print(s, flush=True)
    print("KURULUM", KURULUM, flush=True)


threading.Thread(target=_kur, daemon=True).start()
print("kurulum ARKA PLANDA basladi -- bitince ciktisi BU hucreye duser")
'''

C3 = '''# 3 EGITIM  |  GPU  |  ARKA PLAN
# TASARIM 4 (§12c).  Butun ayarlar `ayar_14`te, her birinin gerekcesi
# kendi yaninda.  Burada YALNIZ dongu var.
#
# !! ONKOSUL `YUVA` saglik satirinda.  Tasarim 3'te bu sayi bir epokta
# 8078 -> 796 dusuyordu; ReLU'da DUSMEMELI.  KOSU SIRASINDA okunur,
# sonda degil -- duserse BILGI okunmaz.
#
# !! SURDURME.  Her epok sonunda Drive'a paket: agirlik + ADAM durumu
# + RNG + ELOG.  GEREKCE, 21 Eylul'de odendi: bir kosu 4. epokta
# cekirdek sifirlanmasiyla oldu ve elde HICBIR SEY kalmadi, cunku ilk
# kayit hucre 4'un sonundaydi.  Kural 1 bunu zaten istiyordu.
# !! AGIRLIK TEK BASINA YETMEZ: ADAM momentleri ve RNG olmadan devam
# SESSIZCE baska bir yorunge uretir (kural 1'in kendi gerekcesi).
#
# !! Her adimda GRADYANIN sonlu olup olmadigina `op.step()` ONCESINDE
# bakiliyor: itme terimindeki sqrt tekilligi adim 1'in gradyanini NaN
# yapiyordu ve adim 2'de bakmak GEC KALIYORDU.
SURDUR = True
EGITIM, ELOG, MDL = "kosuyor", [], None
OLCUM, OLOG, SONUC = "-", [], None        # onceki olcum GECERSIZ
EP = [0, 0]
T_EG = [0.0, 0.0]
T_OL = [0.0, 0.0]
SUR_YOL = CIKTI + "/surdurme.pt"


def _egit():
    global EGITIM, MDL
    T_EG[0] = time.time()
    try:
        assert KURULUM == "bitti", "once 2 bitsin (durum: %s)" % KURULUM
        A = AY.AYAR
        torch.manual_seed(A.tohum)
        m = M14.Yol(len(b), D=AY.D_DURUM, d=AY.D_OKUMA, K=AY.K_KOD,
                    tam=M14.sinif_ayir(b.say, AY.K_TAM), saat=AY.SAAT,
                    hafiza=AY.HAFIZA, haf_b0=AY.HAF_B0).to("cuda")
        ELOG.append(M14.kapi(m).replace("\\n", "\\n  "))
        ELOG.append("HAFIZA %s  M=%d  kapi ReLU(<zp,C>+hb)  hb0=%.2f"
                    % (AY.HAFIZA, AY.K_KOD, AY.HAF_B0))
        ELOG.append("CAPA r=%.2f %s   VQ a2=%.1f %s   BUTCE a4=%.1f B=%.2f"
                    % (AY.R_CAPA, "KAPALI" if AY.R_CAPA == 0 else "acik",
                       AY.A2_CAPA, "KAPALI" if AY.A2_CAPA == 0 else "acik",
                       AY.A4_HAF, AY.HAF_BUTCE))

        P = torch.as_tensor(b.pencere(AY.PENCERE, AY.ATLA).copy(),
                            device="cuda")
        op = torch.optim.Adam(m.parameters(), lr=AY.LR)
        gg = torch.Generator(device="cuda").manual_seed(A.tohum)
        N, BS = len(P), AY.BATCH
        EP[1], ep0 = AY.EPOK, 0
        ELOG.append("pencere %s  atla %d  isinma %d  adim/epok %d  batch %d"
                    % (tuple(P.shape), AY.ATLA, AY.ISINMA,
                       (N + BS - 1) // BS, BS))

        if SURDUR and os.path.exists(SUR_YOL):
            pk = torch.load(SUR_YOL, map_location="cuda", weights_only=False)
            assert pk["commit"] == COMMIT, (
                "surdurme paketi BASKA commit'ten (%s != %s) -- baska bir "
                "model demek, devam ETMEM" % (pk["commit"], COMMIT))
            assert pk["iz_birim"] == BR.iz(b), "paket baska korpustan"
            m.load_state_dict(pk["model"])
            op.load_state_dict(pk["opt"])
            gg.set_state(pk["gg"])
            torch.set_rng_state(pk["cpu_rng"])
            torch.cuda.set_rng_state(pk["gpu_rng"])
            ELOG.extend(pk["elog"])
            ep0 = pk["epok"]
            EP[0] = ep0
            ELOG.append("  >> SURDURULDU: epok %d'ten devam" % (ep0 + 1))

        def dok(baslik, ad):
            ELOG.append("  !! %s -- adim %d" % (baslik, ad))
            ELOG.append("  " + M14.saglik(m).replace("\\n", "\\n  "))
            for k, p in m.named_parameters():
                if p.numel() == 0:
                    ELOG.append("     %-3s BOS" % k)
                    continue
                g_ = p.grad
                iy = torch.isfinite(g_) if g_ is not None else None
                ELOG.append(
                    "     %-3s |p|max %9.3e sonlu %-5s | |g|max %9.3e"
                    "  SONSUZ %s/%d"
                    % (k, float(p.abs().max()), bool(torch.isfinite(p).all()),
                       float(g_[iy].abs().max()) if iy is not None
                       and bool(iy.any()) else float("nan"),
                       int((~iy).sum()) if iy is not None else "grad YOK",
                       p.numel()))

        ad = ep0 * ((N + BS - 1) // BS)
        for ep in range(ep0 + 1, AY.EPOK + 1):
            perm = torch.randperm(N, device="cuda", generator=gg)
            tot = ns = 0.0
            for i in range(0, N, BS):
                ad += 1
                X = P[perm[i:i + BS]]
                k, uye = m.kayip(X, AY.A1_DIS, AY.A2_CAPA, AY.A3_DUZEN,
                                 AY.DELTA, AY.BETA, AY.R_CAPA, AY.ISINMA,
                                 AY.A4_HAF, AY.HAF_BUTCE)
                op.zero_grad(set_to_none=True)
                k.backward()
                # TEK senkron: butun gradyanlarin toplami sonlu mu.
                g = float(sum(p.grad.sum() for p in m.parameters()
                              if p.grad is not None))
                kf = float(k)
                if ad <= 2 or g != g or abs(g) == float("inf") \\
                        or kf != kf or abs(kf) == float("inf"):
                    bas = ("ILK ADIMLAR" if ad <= 2 and g == g
                           and abs(g) != float("inf") and kf == kf
                           else "SONLU DEGIL")
                    dok("%s  kayip %s  grad toplam %s" % (bas, kf, g), ad)
                    if bas == "SONLU DEGIL":
                        raise RuntimeError("adim %d: kayip %s grad %s"
                                           % (ad, kf, g))
                op.step()
                tot += float(uye) * len(X)
                ns += len(X)
            EP[0] = ep
            ELOG.append("  epok %d/%d  uye %.4f  (%.0f sn)"
                        % (ep, AY.EPOK, tot / ns, time.time() - T_EG[0]))
            ELOG.append("       " + M14.saglik(m).replace("\\n", "\\n       "))
            # ONCE gecici, SONRA yerine tasi: yazarken olursek yarim
            # dosya birakmayalim.
            torch.save({"model": m.state_dict(), "opt": op.state_dict(),
                        "gg": gg.get_state(),
                        "cpu_rng": torch.get_rng_state(),
                        "gpu_rng": torch.cuda.get_rng_state(),
                        "epok": ep, "elog": ELOG, "commit": COMMIT,
                        "iz_birim": BR.iz(b)}, SUR_YOL + ".tmp")
            os.replace(SUR_YOL + ".tmp", SUR_YOL)
        MDL = m
        EGITIM = "bitti"
    except Exception as e:
        import traceback
        EGITIM = "HATA: %s: %s" % (type(e).__name__, e)
        ELOG.append(traceback.format_exc()[-700:])
    T_EG[1] = time.time()


threading.Thread(target=_egit, daemon=True).start()
print("egitim ARKA PLANDA basladi -- durum ve log icin R hucresi")
print("SURDURME:", SUR_YOL, "(her epok sonunda)")
'''

C4 = '''# 4 DOGRULUK + SONUC KAYDI  |  GPU  |  ARKA PLAN
#   DOGRULUK = BICIM (kalip/ek/tip) + BILGI (OGRETILEN/CIKARIM)
#
# !! BOLME ADI KULLANILMAZ.  `tum()` butun zincirleri tek havuzda
# topluyor, adimi zincirin UZUNLUGUNDAN, sinifi KORPUSTAN cikariyor.
#
# !! URETIM HAFIZAYI OKUR.  `uret_toplu` adimi kendi yaziyordu ve
# `mdl.V`ye dokunmuyordu; bir kosu hafizayla egitilip HAFIZASIZ
# olculdu ve BICIM sayilari gecersiz cikti.  Tek adim artik
# `Yol.adim`da, kapi 37 iki yolu birbirine bagliyor.
#
# !! BILGI OLCUSU kuyruktaki eki IKI TARAFTA da atiyor.  Onceki hal
# BIREBIR esitlik ariyordu ve modelin `-dir`i yuzunden HIC
# ateslenemiyordu: dort kosu 0,0000 diye okundu, gercek deger
# 0,0003 / 0,0004 / 0,0008 / 0,0138 idi.  Kapi 40.
OLCUM, OLOG, SONUC = "kosuyor", [], None
T_OL = [0.0, 0.0]


def _olc():
    global OLCUM, SONUC
    T_OL[0] = time.time()
    try:
        assert EGITIM == "bitti", "once 3 bitsin (durum: %s)" % EGITIM
        S = OL.Sorular(v, E_ad, list(V14.ILISKI), V14.TIPLER,
                       V14.TR, V14.TR_ILISKI, b.kok, b.ix, b.korunan)
        sor = S.tum(L, b.dizi, yaz=OLOG.append)
        OLOG.append(OL.kapi(sor, b.dizi))

        bitis = set(b.ix[x] for x in ".?!" if x in b.ix)
        ek_ix = set(i for i, a in enumerate(b.ad) if a.startswith("-"))
        var_ix = set(w for ad in E_ad for w in S.birim(ad))
        bc = OL.Bicim(b.dizi, var_ix, bitis, ek_ix, n=len(b))
        SONUC = OL.dogruluk(MDL, bc, sor, bitis,
                            OL.tip_haritasi(v, E_ad, S), r=AY.R_CAPA)
        OL.yaz(SONUC, OLOG.append)

        # ONKOSUL ve tasarimin KENDI iddiasi.  Hukum vermez, aciklar.
        if MDL.V is not None:
            with torch.no_grad():
                Pw = torch.as_tensor(b.pencere(AY.PENCERE, AY.ATLA)
                                     [:65536].copy(), device="cuda")
                y_ = MDL.yol(Pw, AY.R_CAPA)
                mn, t_ = y_["mn"][:, 1:], Pw[:, 1:]
                vx = torch.zeros(len(b), dtype=torch.bool, device="cuda")
                vx[list(var_ix)] = True
                mv = vx[t_]
                OLOG.append("ONKOSUL  atesleyen yuva %d / %d   konum basina "
                            "aktif %.1f"
                            % (int(y_["ates"].sum()), AY.K_KOD,
                               float(y_["ak"][:, 1:].mean())))
                OLOG.append("HAFIZA   ort|m| %.4f (butce %.2f)   VARLIK %.4f"
                            "   DIGER %.4f   oran %.2f"
                            % (float(mn.mean()), AY.HAF_BUTCE,
                               float(mn[mv].mean()), float(mn[~mv].mean()),
                               float(mn[mv].mean())
                               / max(float(mn[~mv].mean()), 1e-9)))

        kayit = {"commit": COMMIT, "zaman": time.strftime("%Y-%m-%d %H:%M"),
                 "iz_graf": V14.IZ, "iz_birim": BR.iz(b),
                 "ayar": {k: getattr(AY, k) for k in
                          ("D_DURUM", "D_OKUMA", "K_KOD", "K_TAM", "SAAT",
                           "R_CAPA", "DELTA", "A1_DIS", "A2_CAPA",
                           "A3_DUZEN", "BETA", "A4_HAF", "HAF_BUTCE",
                           "HAF_B0", "PENCERE", "ATLA", "ISINMA", "HAFIZA",
                           "LR", "BATCH", "EPOK")},
                 "parametre": M14.n_par(MDL), "birim": len(b),
                 "egitim_log": ELOG, "dogruluk": SONUC}
        with open(CIKTI + "/dogruluk.json", "w", encoding="utf-8") as f:
            json.dump(kayit, f, ensure_ascii=False, indent=1)
        torch.save({"model": MDL.state_dict(), "ayar": kayit["ayar"]},
                   CIKTI + "/model_t0.pt")
        OLOG.append("KAYDEDILDI  " + CIKTI + "/{dogruluk.json, model_t0.pt}")
        OLCUM = "bitti"
    except Exception as e:
        import traceback
        OLCUM = "HATA: %s: %s" % (type(e).__name__, e)
        OLOG.append(traceback.format_exc()[-900:])
    T_OL[1] = time.time()


threading.Thread(target=_olc, daemon=True).start()
print("olcum ARKA PLANDA basladi -- durum ve log icin R hucresi")
'''

C5 = '''# 5 NABIZ  |  CPU  |  ARKA PLAN  |  oturum dusmesin
# Her turda uc kosunun da durumunu basiyor, boylece izleme icin
# ayrica bir sey kosturmak gerekmiyor (CLAUDE.md kural 8).
# !! Nabiz YALNIZ bosta kalma kopmasini onler.  Calisma zamaninin
# geri alinmasini ONLEMEZ -- ona karsi olan sey hucre 3'un surdurme
# paketi (21 Eylul'de bir kosu boyle kayboldu).
NABIZ = True


def _nabiz():
    while NABIZ:
        print("[%s] kurulum=%s  egitim=%s (%d/%d)  olcum=%s"
              % (time.strftime("%H:%M:%S"), globals().get("KURULUM", "-"),
                 globals().get("EGITIM", "-"), *globals().get("EP", [0, 0]),
                 globals().get("OLCUM", "-")), flush=True)
        time.sleep(180)


threading.Thread(target=_nabiz, daemon=True).start()
print("nabiz basladi (3 dk) -- durdurmak icin NABIZ = False")
'''

CY = '''# Y YEDEK  |  CPU  |  tekrar: GUVENLI  |  UZERINE YAZMAZ
# Biten kosu t0/'dan kendi klasorune tasinir.  Hedef adi AYARDAN
# turiyor, elle yazilmiyor: iki kosu ayni ada dusemesin.
import os, json, shutil

_KAY = "/content/drive/MyDrive/model_14"
_KAYNAK = _KAY + "/t0"
assert os.path.exists(_KAYNAK + "/dogruluk.json"), "yedeklenecek kosu yok"
_k = json.load(open(_KAYNAK + "/dogruluk.json", encoding="utf-8"))
_a = _k["ayar"]
_HEDEF = "%s/t0_%s_d%d_M%d" % (_KAY, _k["commit"], _a["D_OKUMA"], _a["K_KOD"])

print("KAYNAK  commit %s  %s" % (_k["commit"], _k["zaman"]))
print("  " + "  ".join("%s=%s" % (x, _a.get(x)) for x in
                       ("D_OKUMA", "K_KOD", "R_CAPA", "A2_CAPA", "A4_HAF",
                        "HAF_B0")))
print("  parametre %s" % format(_k.get("parametre", 0), ","))
for _f in sorted(os.listdir(_KAYNAK)):
    print("  %-18s %8.1f MB" % (_f, os.path.getsize(_KAYNAK + "/" + _f) / 1e6))

assert not os.path.exists(_HEDEF), "HEDEF ZATEN VAR: " + _HEDEF
shutil.copytree(_KAYNAK, _HEDEF)
print("\\nYEDEKLENDI ->", _HEDEF)

print("\\nDRIVE'DAKI KOSULAR")
for _d in sorted(os.listdir(_KAY)):
    _j = _KAY + "/" + _d + "/dogruluk.json"
    if os.path.exists(_j):
        _q = json.load(open(_j, encoding="utf-8"))
        _b = _q["dogruluk"]["BICIM"]["OGRETILEN"]
        _g = _q["dogruluk"]["BILGI"]["OGRETILEN"]
        print("  %-26s kalip %.4f  kapanmadi %.4f  |  BILGI tam %.4f"
              % (_d, _b["kalip"], _b["kapanmadi"], _g["tam"]))
'''

CG = '''# G GOZLE  |  GPU  |  URETIM  |  sayilar ne olctugunu, METIN nasil
# CLAUDE.md: "sayi degil, MODELIN YAZDIGI okunur."
import numpy as np

assert globals().get("OLCUM") == "bitti", "once 4 bitsin"
_S = OL.Sorular(v, E_ad, list(V14.ILISKI), V14.TIPLER,
                V14.TR, V14.TR_ILISKI, b.kok, b.ix, b.korunan)
_G = V14.kur(AY.AYAR.veri_tohum)
_Ei = {a: i for i, a in enumerate(E_ad)}
_IL = list(V14.ILISKI)
_Ri = {r: i for i, r in enumerate(_IL)}

# ILISKILERE YAYARAK sec -- tek iliskiden 20 tane degil
_ilg = {}
for (_o, _r), _c in _G["olgu"].items():
    if _o in _Ei and _c in _Ei and _r in _Ri:
        _ilg.setdefault(_r, []).append((_o, _c))
_rng = np.random.default_rng(20)
_rl = [r for r in _IL if r in _ilg]
_sec = []
for _i in range(200):
    if len(_sec) == 20:
        break
    _r = _rl[_i % len(_rl)]
    _o, _c = _ilg[_r][int(_rng.integers(len(_ilg[_r])))]
    _q = _S.kur([_Ei[_o], _Ri[_r], _Ei[_c]], 1)
    if _q and all(_o != s[0] or _r != s[1] for s in _sec):
        _sec.append((_o, _r, _q[0]))

_cik = OL.uret_toplu(MDL, [s[2].onek for s in _sec], n_yeni=12,
                     r=AY.R_CAPA, bs=64)
_bitis = set(b.ix[x] for x in ".?!" if x in b.ix)
_eki = set(i for i, a in enumerate(b.ad) if a.startswith("-"))


def _kirp(p):
    p = list(p)
    while p and int(p[-1]) in _eki:
        p.pop()
    return tuple(p)


print("=" * 78)
_dog = 0
for _k, (_o, _r, _q) in enumerate(_sec):
    _sp = OL._span(_cik[_k], _bitis)
    _ok = _kirp(_sp) == _kirp(_q.cevap)
    _dog += _ok
    print("%2d  %s" % (_k + 1, b.coz(_q.onek)))
    print("    beklenen : %s" % b.coz(_q.cevap))
    print("    model    : %s   %s"
          % (b.coz(list(_sp)), "DOGRU" if _ok else ""))
    print()
print("=" * 78)
print("20 soruda TAM dogru: %d   (olcu ek-toleransli, kapi 40)" % _dog)
'''

CR = '''# R RAPOR  |  CPU  |  ANINDA doner  |  HER ZAMAN GUVENLI
# Hicbir sey baslatmaz, GPU'ya dokunmaz.  Henuz kosmamis hucrelerin
# degiskenleri olmasa da calisir -- yoksa "her zaman guvenli" iddiasi
# bos olur (bir kez NameError verdi).
import time as _t0
_g = globals()
_al = lambda ad, ilk: _g.get(ad, ilk)


def _sure(t):
    if not t[0]:
        return "-"
    return "%.0f sn%s" % ((t[1] or _t0.time()) - t[0],
                          "" if t[1] else " (suruyor)")


print("=" * 64)
print("model_14 t0  commit %s   %s"
      % (_al("COMMIT", "?"), _t0.strftime("%H:%M:%S")))
print("=" * 64)
for ad, dur, t in (("KURULUM", _al("KURULUM", "-"), _al("T_KUR", [0, 0])),
                   ("EGITIM", _al("EGITIM", "-"), _al("T_EG", [0, 0])),
                   ("OLCUM", _al("OLCUM", "-"), _al("T_OL", [0, 0]))):
    print("  %-9s %-36s %s" % (ad, dur, _sure(t)))

_ep = _al("EP", [0, 0])
if _ep[1]:
    kalan = ""
    if _ep[0] and _al("EGITIM", "") == "kosuyor":
        kalan = "   kalan ~%.0f sn" % (
            (_t0.time() - _al("T_EG", [_t0.time(), 0])[0]) / _ep[0]
            * (_ep[1] - _ep[0]))
    print("  epok      %d/%d%s" % (_ep[0], _ep[1], kalan))

try:
    import torch as _th
    _t, _b = _th.cuda.mem_get_info()
    print("  GPU       %.1f / %.1f GB kullanimda" % ((_b - _t) / 1e9, _b / 1e9))
except Exception:
    print("  GPU       okunamadi (cekirdek sifirlanmis olabilir)")

for baslik, ad in (("KURULUM", "KLOG"), ("EGITIM", "ELOG"), ("OLCUM", "OLOG")):
    log = _al(ad, [])
    if log:
        print("\\n--- " + baslik + " ---")
        for s in log:
            print(s if s.startswith(" ") else "  " + s)
'''

CX = '''# X DURDUR  |  CPU  |  tekrar: GUVENLI  |  kosan EGITIMI ve OLCUMU keser
# Iplikler daemon; durdurma bayragi YOK.  Cekirdegi oldurmek KURULUMU
# da goturur (v/L/b/E_ad).  Onun yerine ipligin CAGIRDIGI fonksiyon
# gecici olarak hata firlatir yapiliyor: iplik kendi `except`ine
# duser, geri kalan her sey ayakta kalir.
import gc

NABIZ = False
_yedek = {}
if globals().get("EGITIM") == "kosuyor":
    _yedek["kayip"] = M14.Yol.kayip
    M14.Yol.kayip = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("DURDURULDU"))
if globals().get("OLCUM") == "kosuyor":
    for _ad in ("_gecer", "uret_toplu"):
        if hasattr(OL, _ad):
            _yedek[_ad] = getattr(OL, _ad)
            setattr(OL, _ad, lambda *a, **k: (_ for _ in ()).throw(
                RuntimeError("DURDURULDU")))

for _ in range(120):
    if globals().get("EGITIM") != "kosuyor" \\
            and globals().get("OLCUM") != "kosuyor":
        break
    time.sleep(0.5)

if "kayip" in _yedek:
    M14.Yol.kayip = _yedek.pop("kayip")
for _ad, _f in _yedek.items():
    setattr(OL, _ad, _f)          # hepsi geri takildi

gc.collect()
torch.cuda.empty_cache()
_t, _b = torch.cuda.mem_get_info()
print("kurulum:", globals().get("KURULUM"))
print("egitim :", globals().get("EGITIM"))
print("olcum  :", globals().get("OLCUM"))
print("GPU    : %.1f / %.1f GB" % ((_b - _t) / 1e9, _b / 1e9))
'''

HUCRE = [("md", MD), ("H", H), ("0", C0), ("1", C1), ("2", C2), ("3", C3),
         ("4", C4), ("5", C5), ("Y", CY), ("G", CG), ("R", CR), ("X", CX)]


def kur(yol="model_14.ipynb"):
    cells = []
    for ad, kaynak in HUCRE:
        sat = kaynak.rstrip("\n").split("\n")
        src = [s + "\n" for s in sat[:-1]] + [sat[-1]]
        if ad == "md":
            cells.append({"cell_type": "markdown", "metadata": {},
                          "source": src})
        else:
            compile(kaynak, "<hucre %s>" % ad, "exec")   # KAPI: derleniyor mu
            cells.append({"cell_type": "code", "metadata": {},
                          "execution_count": None, "outputs": [],
                          "source": src})
    nb = {"cells": cells, "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "L4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 0}
    with open(yol, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    return len(cells)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    n = kur()
    print("model_14.ipynb kuruldu: %d hucre" % n)
    for ad, k in HUCRE:
        print("  %-3s %s" % (ad, k.lstrip().split("\n")[0][:64]))
