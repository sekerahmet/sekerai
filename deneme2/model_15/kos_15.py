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

# Cevabi bu kadar ya da daha az haneli olan ornekler HER yigina TAMAMI ile
# girer.  Kullanici karari, 22 Eylul.  Bedeli olculdu ve soylendi: 2 haneli
# toplamlar gercek dagilimda %2,23 iken yiginda ~%45 olacak.
SABIT_SINIF = 2

GUNLUK, SONUC, DURDUR = [], {}, set()


def _olcut_yok(m, taraf):
    raise RuntimeError("olcut verilmedi -- baslat(..., olcut=...) sart")


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
    # GUNLUK de yazilir: yalniz bellekte kalirsa cekirdekle birlikte gider
    # ve kosunun EGRISI kaybolur -- agirlik kalir ama nasil gelindigi gitmis
    # olur.  Ayni ders, ayni yer: kaydetme kosunun ICINDE.
    with open(f"{d}/gunluk.txt", "w") as f:
        f.write("\n".join(GUNLUK) + "\n")


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


def _kos(ad, EG, TU, N, olcut, aygit, kok, ek, boyut, durum,
         lr, wd, adim, tohum, yigin, bas, yedek):
    not_ = GUNLUK.append
    torch.manual_seed(tohum)
    m = Yol(N, boyut=boyut, durum=durum, tohum=tohum).to(aygit)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    t0, ob = time.time(), len(EG)
    par = sum(p.numel() for p in m.parameters())
    uret = torch.Generator(device=aygit).manual_seed(tohum)
    # --- YIGIN KURULUSU.  Kullanici karari, 22 Eylul:
    #   "25.000'lik yigin icinde 1+1 ve iki hanelerin hepsi her seferinde
    #    olsun."
    # Kisa cevapli ornekler (u <= SABIT_SINIF) HER adimda TAMAMI ile
    # yigina giriyor; kalan yer uzun cevaplilardan CEKILIYOR.
    # Obek yalnizca tensor sekli oldugu icin her obekten pay aliniyor --
    # tek obek secmek kucuk obegi "ya hepsi ya hicbiri" yapiyordu ve
    # 8000 adimda 4 tokenlik obek SIFIR kez secilmisti.
    sabit, havuz = [], []
    for _w, _h, _u in EG:
        m_ = _u <= SABIT_SINIF
        sabit.append(m_.nonzero(as_tuple=True)[0])
        havuz.append((~m_).nonzero(as_tuple=True)[0])
    n_sabit = sum(len(x) for x in sabit)
    n_havuz = sum(len(x) for x in havuz)
    kalan = yigin - n_sabit
    assert kalan > 0, (f"sabit sinif {n_sabit} ornek, yigin {yigin} -- "
                       f"yigini buyut ya da SABIT_SINIF'i kucult")
    cek = [int(round(kalan * len(x) / n_havuz)) for x in havuz]
    not_(f"[{ad}] YIGIN {yigin}: sabit {n_sabit} (cevap <= {SABIT_SINIF} hane, "
         f"HEPSI her adimda) + cekilis {sum(cek)}")
    not_(f"[{ad}] boyut {boyut} durum {durum} lr {lr} wd {wd} tohum {tohum}"
         f"  parametre {par}")
    not_(f"[{ad}]   adim   egitim  tutulan     sn")
    not_(f"[{ad}]   TEK OLCUT: soru soruldu, cevap DOGRU MU.  "
         f"Serbest uretim, DUR'a kadar, tamami birebir.")

    i = 0
    for i in range(adim + 1):
        if ad in DURDUR:
            not_(f"[{ad}] DURDURULDU  adim {i}")
            break
        # HER OBEKTEN pay alinir: sabit kisim tamami, gerisi cekilis.
        opt.zero_grad()
        toplam = 0.0
        for g, (w, h, _u) in enumerate(EG):
            j = sabit[g]
            if cek[g] and len(havuz[g]):
                r = torch.randint(0, len(havuz[g]), (cek[g],),
                                  device=aygit, generator=uret)
                j = torch.cat([j, havuz[g][r]])
            if not len(j):
                continue
            o, _ = m.dikkat(w[j])
            puan = -((m.E[None] - o[:, None]) ** 2).sum(-1)
            # 'sum' -> butun obeklerin toplami tek yigin gibi davranir
            (F.cross_entropy(puan, h[j], reduction="sum") / yigin).backward()
            toplam += len(j)
        opt.step()
        if i % bas == 0:
            de, dt = olcut(m, "eg"), olcut(m, "tu")
            bilgi = dict(ek or {}, n=N, boyut=boyut, durum=durum, adim=i,
                         lr=lr, wd=wd, tohum=tohum, parametre=par,
                         egitim=de, tutulan=dt)
            SONUC[ad] = dict(bilgi, model=m)
            iz = ""
            if i % yedek == 0:                   # YEDEK -- kosunun ICINDE
                _yaz(kok, ad, m, bilgi)
                iz = "  yedek"
            not_(f"[{ad}] {i:6d}   {de:.4f}   {dt:.4f}"
                 f"   {time.time()-t0:5.0f}{iz}")

    de, dt = olcut(m, "eg", tam=True), olcut(m, "tu", tam=True)
    bilgi = dict(ek or {}, n=N, boyut=boyut, durum=durum, adim=i,
                 lr=lr, wd=wd, tohum=tohum, parametre=par,
                 egitim=de, tutulan=dt, biti=True)
    SONUC[ad] = dict(bilgi, model=m)
    _yaz(kok, ad, m, bilgi)
    not_(f"[{ad}] BITTI   egitim {de:.4f}   tutulan {dt:.4f}"
         f"   ({time.time()-t0:.0f} sn)")


def baslat(ad, EG, TU, N, *, olcut, aygit="cuda", kok=None, ek=None,
           boyut=BOYUT, durum=DURUM, lr=LR, wd=WD,
           adim=8000, tohum=0, yigin=25000, bas=200, yedek=2000):
    """ARKA PLANDA baslatir, HEMEN doner (kural 8).  DURDUR.add(ad) durdurur.

    olcut  olcut(m, "eg"|"tu", tam=False) -> oran.  TEK analiz:
           soru soruldu, cevap dogru mu.  Yuva/obek kirilimi YOK.
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
        args=(ad, EG, TU, N, olcut, aygit, kok, ek, boyut, durum,
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
