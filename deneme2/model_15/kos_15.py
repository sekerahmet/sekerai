"""EGITIM DONGUSU.  Defterde DEGIL, depoda -- defter sifirlaninca kaybolmasin.

Kullanici, 22 Eylul: defteri sifirdan kurarken "MODELI DRIVE'A KAYDET"
hucresi silindi ve 8000 adimlik kosu (449 sn) kayboldu.  Kaydetme artik
AYRI BIR HUCRE DEGIL, kosunun ICINDE:  her olcum noktasinda Drive'a
yaziliyor.  Gerekce CLAUDE.md kural 2 ile ayni -- ayri hucredeki adim,
hucre sirasina bagli bir kuraldir, insan hatirlarsa calisir.

Kural 8: baslat() ARKA PLANDA calisir ve HEMEN doner.  Izleme, GUNLUK'u
basan nabiz hucresiyle OKUYARAK yapilir.
"""
import os
import shutil
import time
import threading

import torch
import torch.nn.functional as F

from model_15 import Yol, BOYUT, DURUM, LR, WD

GUNLUK, SONUC, DURDUR = [], {}, set()


def olc(m, OBEK, en=20000):
    """Butun yuvalarin ortalamasi.  Okuma: en yakin E[token]."""
    dog = say = 0
    with torch.no_grad():
        for w, h in OBEK:
            w, h = w[:en], h[:en]
            o, _ = m.dikkat(w)
            c = (-((m.E[None] - o[:, None]) ** 2).sum(-1)).argmax(-1)
            dog += int((c == h).sum()); say += len(h)
    return dog / say


def olc_obek(m, OBEK, en=20000):
    """OBEK BASINA dogruluk.  Obek = GIRDI UZUNLUGU, tensor sekli geregi.

    Sabit genislikli sinavda obek "cevabin kacinci rakami" demekti
    (yuva1..yuva4).  Dolgu kalkinca bu anlam GITTI: obek artik yalnizca
    "kac token" demek.  Bu yuzden TANI amaclidir, HUKUM vermez -- hukum
    SAYI olcutuyle verilir (DUR'a kadar uretilen dizinin tamami dogru mu).
    """
    r = []
    with torch.no_grad():
        for w, h in OBEK:
            w, h = w[:en], h[:en]
            o, _ = m.dikkat(w)
            c = (-((m.E[None] - o[:, None]) ** 2).sum(-1)).argmax(-1)
            r.append(float((c == h).float().mean()))
    return r


def _yaz(kok, ad, m, bilgi):
    """Agirligi Drive'a yaz.  Her yedek noktasi AYRI dosya + 'son' kopyasi.

    Eski anlik goruntuler SILINMEZ: model 21 KB, Drive'da 2 TB var, ve
    seyreltmek bu projede daha once uc kez kosu yeniden baslatmaya mal oldu.
    """
    if not kok:
        return
    d = f"{kok}/{ad}"
    os.makedirs(d, exist_ok=True)
    p = dict(bilgi)
    p["agirlik"] = {k: v.detach().cpu() for k, v in m.state_dict().items()}
    torch.save(p, f"{d}/t{bilgi['adim']}.pt")
    torch.save(p, f"{kok}/model_{ad}.pt")          # konus.py bunu okur


def _koru(kok, ad):
    """Ayni adla yeni kosu ESKI YEDEKLERI SILMEZ -- yan klasore tasir.

    CLAUDE.md kural 1 ile ayni yaklasim: `--ustune` de silmiyor, eskisini
    `t<N>_eski_<zaman>/` diye kenara aliyor.  Burada da oyle; kaybedilen
    bir kosu geri getirilemiyor, ama 21 KB'lik bir klasor hep saklanabilir.
    Kosu BASLAMADAN once, IPLIK DISINDA calisir ki hata gorunur olsun.
    """
    if not kok:
        return None
    d = f"{kok}/{ad}"
    if not os.path.isdir(d) or not os.listdir(d):
        return None
    yeni = f"{kok}/{ad}_eski_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.move(d, yeni)
    son = f"{kok}/model_{ad}.pt"
    if os.path.exists(son):
        shutil.move(son, f"{yeni}/model_{ad}.pt")
    return os.path.basename(yeni)


def _kos(ad, EG, TU, N, aygit, kok, ek, boyut, durum,
         lr, wd, adim, tohum, yigin, bas, yedek):
    not_ = GUNLUK.append
    torch.manual_seed(tohum)
    m = Yol(N, boyut=boyut, durum=durum, tohum=tohum).to(aygit)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    t0, ob = time.time(), len(EG)
    par = sum(p.numel() for p in m.parameters())
    uret = torch.Generator(device=aygit).manual_seed(tohum)
    pay = torch.tensor([float(len(h)) for _, h in EG], device=aygit)
    pay = pay / pay.sum()                    # obek buyuklugu kadar sik
    not_(f"[{ad}] boyut {boyut} durum {durum} lr {lr} wd {wd} tohum {tohum}"
         f"  parametre {par}")
    not_(f"[{ad}]   adim   egitim  tutulan   "
     + " ".join(f"{int(w.shape[1]):2d}tk" for w, _ in EG) + "     sn")

    i = 0
    for i in range(adim + 1):
        if ad in DURDUR:
            not_(f"[{ad}] DURDURULDU  adim {i}")
            break
        # OBEK SECIMI ORANTILI -- "hepsini karistirip cek" ile ayni sey.
        # Obekler yalnizca TENSOR SEKLI icin var (dolgu kalkinca ayni
        # uzunluktakiler bir arada yiginlanmak zorunda); bir mufredat
        # DEGIL.  Sirayla gezmek her obege esit sure veriyordu ve bu,
        # kimsenin vermedigi bir agirliklandirmaydi: 83 ornekli obek ile
        # 125.419 ornekli obek ayni sureyi aliyordu.
        g = int(torch.multinomial(pay, 1, generator=uret))
        w, h = EG[g]
        j = torch.randint(0, len(h), (yigin,), device=aygit, generator=uret)
        o, _ = m.dikkat(w[j])
        puan = -((m.E[None] - o[:, None]) ** 2).sum(-1)
        k = F.cross_entropy(puan, h[j])
        opt.zero_grad(); k.backward(); opt.step()
        if i % bas == 0:
            de, dt = olc(m, EG), olc(m, TU)
            r = olc_obek(m, TU)
            bilgi = dict(ek or {}, n=N, boyut=boyut, durum=durum, adim=i,
                         lr=lr, wd=wd, tohum=tohum, parametre=par,
                         egitim=de, tutulan=dt, obek=r)
            SONUC[ad] = dict(bilgi, model=m)
            iz = ""
            if i % yedek == 0:                   # YEDEK -- kosunun ICINDE
                _yaz(kok, ad, m, bilgi)
                iz = "  yedek"
            not_(f"[{ad}] {i:6d}  {de:.4f}  {dt:.4f}  "
                 + " ".join(f"{x:.4f}" for x in r)
                 + f"   {time.time()-t0:5.0f}{iz}")

    de, dt = olc(m, EG, 10**9), olc(m, TU, 10**9)
    r = olc_obek(m, TU, 10**9)
    bilgi = dict(ek or {}, n=N, boyut=boyut, durum=durum, adim=i,
                 lr=lr, wd=wd, tohum=tohum, parametre=par,
                 egitim=de, tutulan=dt, obek=r, biti=True)
    SONUC[ad] = dict(bilgi, model=m)
    _yaz(kok, ad, m, bilgi)
    not_(f"[{ad}] BITTI  egitim {de:.4f}  tutulan {dt:.4f}  obek "
         + " ".join(f"{x:.4f}" for x in r) + f"  ({time.time()-t0:.0f} sn)")


def baslat(ad, EG, TU, N, *, aygit="cuda", kok=None, ek=None,
           boyut=BOYUT, durum=DURUM, lr=LR, wd=WD,
           adim=8000, tohum=0, yigin=25000, bas=200, yedek=2000):
    """ARKA PLANDA baslatir, HEMEN doner (kural 8).  DURDUR.add(ad) durdurur.

    bas    kac adimda bir OLCULUR   -- gunluge satir duser
    yedek  kac adimda bir KAYDEDILIR -- Drive'a yazilir.  Kosu bitince
           her halukarda yazilir; oturum duserse son yedekten devam edilir.
    kok    Drive klasoru.  Verilmezse HICBIR SEY kaydedilmez ve uyarilir.
    """
    if kok is None:
        GUNLUK.append(f"[{ad}] UYARI: kok YOK, agirlik KAYDEDILMIYOR")
    tasinan = _koru(kok, ad)           # iplikten ONCE -- hata gorunur olsun
    if tasinan:
        print(f"  ESKI YEDEKLER KORUNDU -> {tasinan}/")
        GUNLUK.append(f"[{ad}] eski yedekler tasindi -> {tasinan}/")
    DURDUR.discard(ad)
    threading.Thread(
        target=_kos, daemon=True,
        args=(ad, EG, TU, N, aygit, kok, ek, boyut, durum,
              lr, wd, adim, tohum, yigin, bas, yedek)).start()
    return f"{ad} basladi"


def nabiz(son=40):
    """Gunlugu bas.  HICBIR SEY KOSTURMAZ."""
    print(f"gunluk {len(GUNLUK)} satir   SONUC: {list(SONUC)}"
          f"   DURDUR: {sorted(DURDUR)}")
    for s in GUNLUK[-son:]:
        print(s)


def durdur(ad=None):
    """Bayrak koyar; iplik bir sonraki adimda kendi kendine cikar."""
    for a in ([ad] if ad else list(SONUC)):
        DURDUR.add(a)
    print("durdurma bayragi:", sorted(DURDUR))
