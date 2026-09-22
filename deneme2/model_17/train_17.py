# -*- coding: utf-8 -*-
"""train_17 -- egitim dongusu: optimizer, yigin, kayit, SURDURME.

GOREV SONRAKI JETON.  Etiket yok; her konum bir sonrakini tahmin eder.

    kayip = m.kayip(w)          w (B,T) pencere
                                dizi() her yuvayi okur, nedensel maske

Veri TEK TENSOR: (n, T) pencere yigini, hepsi ayni uzunlukta.  Obek YOK
-- akis kesintisiz ve pencereler sabit T, yani dolgu da yok.

AdamW, Adam DEGIL.  Adam'in weight_decay'i L2'yi gradyana katar ve
1/sqrt(v) ile normalize eder; gorev gradyani sadelesen bir bilesende
adim -lr*sign(t) olur ve bilesen wd'den BAGIMSIZ olarak lr hiziyla
silinir.  model_16'da olculdu: token basina parametreler cokuyordu
(E 15,8x, b 21x, M 31x), paylasilanlar cokmuyordu.
dim < 2 (s0, hb) decay disinda.

KOSU ARKA PLANDA (CLAUDE.md kural 8).  `baslat` hemen doner, ilerleme
`nabiz` ile OKUNUR.  Gunluk her `yedek` adimda diske de yazilir.
"""
from __future__ import annotations

import os
import shutil
import threading
import time

import torch

from model_17 import Yol, BOYUT, DURUM, LR, WD

YIGIN = 512
GUNLUK, SONUC, DURDUR = [], {}, set()


def _olcut_yok(m, taraf, tam=False):
    raise RuntimeError("olcut verilmedi -- baslat(..., olcut=...) sart")


def _tam(m, opt, uret, bilgi):
    """TAM anlik goruntu -- surdurmeye yeten her sey."""
    return dict(bilgi, agirlik=m.state_dict(), opt=opt.state_dict(),
                rng=torch.get_rng_state(), uret_rng=uret.get_state())


def _yaz(kok, ad, p):
    """Anlik goruntuyu ve gunlugu diske yaz -- KOSUNUN ICINDE.
    Ayri hucre olarak dururken unutulur; model_16'da 8000 adimlik bir
    kosu oyle gitmisti.  Eskiler silinmez."""
    if not kok:
        return
    d = f"{kok}/{ad}"
    os.makedirs(d, exist_ok=True)
    torch.save(p, f"{d}/t{p['adim']}.pt")
    torch.save(p, f"{kok}/model_{ad}.pt")
    with open(f"{d}/gunluk.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(GUNLUK) + "\n")


def _koru(kok, ad):
    """Ayni adla yeni kosu eskiyi EZMEZ, yan klasore TASIR."""
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


def _kos(ad, EG, N, olcut, aygit, kok, ek, boyut, durum,
         lr, wd, adim, tohum, yigin, bas, yedek, surdur, derle):
    not_ = GUNLUK.append
    torch.manual_seed(tohum)
    m = Yol(N, boyut=boyut, durum=durum, tohum=tohum).to(aygit)

    # torch.compile.  BURADA OLCULDU, 22 Eylul, L4
    # (T=256, yigin 512, sozluk 4002, boyut=durum=32, 5 isinma + 500 adim):
    #
    #                 sn/adim    kayip@100  @200    @300    @400    @500
    #   derlemesiz      0,557     8,6488  6,0457  5,2187  4,8805  4,6358
    #   DERLEMELI       0,143     8,6488  6,0457  5,2187  4,8805  4,6358
    #                   3,90x     BES OLCUMUN BESI DE BIREBIR AYNI
    #
    #   derleme maliyeti 466,8 sn (7,8 dk).  Adim basina tasarruf 0,415 sn
    #   -> BASABAS 1.125 ADIM.  Kisa kosuda ZARARLI:
    #        500 adim   derlemesiz 279 sn   DERLEMELI 538 sn
    #      2.000 adim            1.114 sn              751 sn
    #      8.400 adim            4.679 sn            1.660 sn   (tam korpus 1 epok)
    #
    # model_11'de 1,83 kat cikmisti; burada daha buyuk cunku darbogaz tam
    # olarak derlemenin cozdugu sey -- gez()'in T kez donen dongusunde
    # cok sayida KUCUK cekirdek baslatma.
    #
    # KALICI OLSUN DIYE BURAYA YAZILDI.  model_10/11/12'de olculup
    # model_13'ten sonra dusmustu: her kol kodunu ebeveyninden kopyaliyor
    # ve model_15 sifirdan yazildi.  Olculmus kazanc, olculmemis
    # varsayimlarla birlikte gitmisti.
    egit = m.kayip
    if derle:
        egit = torch.compile(m.kayip)
        not_(f"[{ad}] torch.compile ACIK -- ilk adim DERLEME yuzunden yavas")
    dec = [p for p in m.parameters() if p.dim() >= 2]
    nodec = [p for p in m.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": wd},
                             {"params": nodec, "weight_decay": 0.0}], lr=lr)
    uret = torch.Generator(device="cpu").manual_seed(tohum)
    bas_adim = 0

    if surdur:
        p = torch.load(surdur, weights_only=False, map_location=aygit)
        m.load_state_dict(p["agirlik"])
        opt.load_state_dict(p["opt"])
        torch.set_rng_state(p["rng"].cpu())
        uret.set_state(p["uret_rng"].cpu())
        bas_adim = p["adim"]
        not_(f"[{ad}] SURDURULUYOR  {os.path.basename(surdur)}  adim {bas_adim}")

    # PENCERELER CPU'DA KALIR, yigin yigin tasinir.  64 MB dilimde
    # tumu GPU'ya sigiyordu (253 MB) ama tam korpusta 8,8 GB eder ve
    # aktivasyonlarin yanina sigmaz.  Tasima maliyeti adim basina
    # 512x256 int32 = 0,5 MB -- olcusuz.
    # EG ya (W, M) ya da tek W.  M dolgu maskesi -- HER HIKAYE BIR
    # PENCERE oldugu icin hikaye bitince kalan yer <dolgu>.
    W, M = EG if isinstance(EG, (tuple, list)) else (EG, None)
    W = W if W.device.type == "cpu" else W.cpu()
    M = M if M is None or M.device.type == "cpu" else M.cpu()
    n, T = W.shape
    par = sum(p.numel() for p in m.parameters())
    _et = 1.0 if M is None else float(M.float().mean())
    not_(f"[{ad}] pencere {n:,} x {T}   {n * T:,} yuva"
         + ("" if M is None else f"   dolgu %{100 * (1 - _et):.1f}"
                                 f"   ETKIN {n * T * _et:,.0f}"))
    not_(f"[{ad}] boyut {boyut} durum {durum} lr {lr} wd {wd} tohum {tohum}"
         f"  sozluk {N}  parametre {par:,}")
    not_(f"[{ad}] yigin {yigin}   adim basina {yigin * (T - 1):,} tahmin"
         f"   epok = {n / yigin:,.0f} adim")
    not_(f"[{ad}] OLCUT: SONRAKI JETON dogrulugu.  sans {1 / N:.5f}")
    not_(f"[{ad}]   adim    kayip   egitim  dogrulama      sn")

    t0, i = time.time(), bas_adim
    for i in range(bas_adim, adim + 1):
        if ad in DURDUR:
            not_(f"[{ad}] DURDURULDU  adim {i}")
            break
        j = torch.randint(0, n, (yigin,), generator=uret)
        opt.zero_grad()
        w_ = W[j].to(aygit, non_blocking=True).long()
        m_ = None if M is None else M[j].to(aygit, non_blocking=True)
        kay = egit(w_, m_)
        kay.backward()
        opt.step()

        if i % bas == 0:
            gecen = time.time() - t0
            e, d = olcut(m, "eg"), olcut(m, "dg")
            bilgi = dict(ek or {}, n=N, adim=i, boyut=boyut, durum=durum,
                         lr=lr, wd=wd, tohum=tohum, yigin=yigin, T=T,
                         parametre=par, kayip=float(kay.detach()),
                         egitim=e, dogrulama=d, derle=derle)
            SONUC[ad] = dict(bilgi, model=m)
            im = ""
            if i % yedek == 0:
                _yaz(kok, ad, _tam(m, opt, uret, bilgi)); im = "  yedek"
            not_(f"[{ad}] {i:6d}  {bilgi['kayip']:7.3f}  {e:7.4f}  {d:9.4f}"
                 f"  {gecen:6.0f}{im}")

    e, d = olcut(m, "eg", tam=True), olcut(m, "dg", tam=True)
    bilgi = dict(ek or {}, n=N, adim=i, boyut=boyut, durum=durum, lr=lr,
                 wd=wd, tohum=tohum, yigin=yigin, T=T, parametre=par,
                 kayip=float(kay.detach()), egitim=e, dogrulama=d,
                 derle=derle, biti=True)
    SONUC[ad] = dict(bilgi, model=m)
    _yaz(kok, ad, _tam(m, opt, uret, bilgi))
    not_(f"[{ad}] BITTI   egitim {e:.4f}   dogrulama {d:.4f}   "
         f"({time.time() - t0:.0f} sn)")


def baslat(ad, EG, N, *, olcut=_olcut_yok, aygit="cuda", kok=None, ek=None,
           boyut=BOYUT, durum=DURUM, lr=LR, wd=WD,
           adim=20000, tohum=0, yigin=YIGIN, bas=100, yedek=500,
           surdur=None, derle=False):
    """ARKA PLANDA baslatir, HEMEN doner (kural 8).

    EG      (n, T) pencere yigini -- hepsi ayni uzunlukta, dolgu YOK
    olcut   olcut(m, "eg"|"dg", tam=False) -> sonraki jeton dogrulugu
    surdur  bir anlik goruntu yolu verilirse KALDIGI YERDEN devam eder
            (agirlik + optimizer + RNG).  Kural 1: uzatma SURDURMEDIR.
    derle   torch.compile.  model_11'de 1,83 kat OLCULDU ama BASKA bir
            mimaride; burada olculmeden acilmaz.
    """
    if kok is None:
        GUNLUK.append(f"[{ad}] UYARI: kok YOK, agirlik KAYDEDILMIYOR")
    if not surdur:
        tasinan = _koru(kok, ad)
        if tasinan:
            print(f"  ESKI YEDEKLER KORUNDU -> {tasinan}/")
            GUNLUK.append(f"[{ad}] eski yedekler tasindi -> {tasinan}/")
    DURDUR.discard(ad)
    threading.Thread(
        target=_kos, daemon=True,
        args=(ad, EG, N, olcut, aygit, kok, ek, boyut, durum, lr, wd,
              adim, tohum, yigin, bas, yedek, surdur, derle)).start()
    return f"{ad} basladi" + (f"  ({os.path.basename(surdur)}'den)"
                              if surdur else "")


def nabiz(son=40):
    """Gunlugu bas.  HICBIR SEY KOSTURMAZ (kural 8)."""
    print(f"gunluk {len(GUNLUK)} satir   SONUC: {list(SONUC)}"
          f"   DURDUR: {sorted(DURDUR)}")
    for s in GUNLUK[-son:]:
        print(s)


def durdur(ad=None):
    """Bayrak koyar; iplik bir sonraki adimda cikar ve cikmadan KAYDEDER."""
    for a in ([ad] if ad else list(SONUC)):
        DURDUR.add(a)
    print("durdurma bayragi:", sorted(DURDUR))
