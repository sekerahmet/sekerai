# -*- coding: utf-8 -*-
"""train_17 -- egitim dongusu: optimizer, yigin, kayit, SURDURME.

GOREV SONRAKI JETON.  Etiket yok; her konum bir sonrakini tahmin eder.

    kayip = m.kayip(w, maske)   w (B,T) pencere, maske dolgu disi yuvalar
                                dizi() her yuvayi okur, nedensel maske

Veri (W, M): W (n, T) pencere yigini, M ayni bicimde dolgu maskesi.
HER HIKAYE BIR PENCERE; hikaye bitince kalan yer <dolgu>.

AdamW, Adam DEGIL.  Adam'in weight_decay'i L2'yi gradyana katar ve
1/sqrt(v) ile normalize eder; gorev gradyani sadelesen bir bilesende
adim -lr*sign(t) olur ve bilesen wd'den BAGIMSIZ olarak lr hiziyla
silinir.  model_16'da olculdu: token basina parametreler cokuyordu
(E 15,8x, b 21x, M 31x), paylasilanlar cokmuyordu.
dim < 2 (s0, hb) ve gecici bellegin konum yanliligi decay disinda (sonumlu).

KOSU ARKA PLANDA (CLAUDE.md kural 8).  `baslat` hemen doner, ilerleme
`nabiz` ile OKUNUR.  Her kosu KENDI gunluk.txt'sine her olcumde EKLER;
surdurmede eski satirlar durur.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import threading
import time
import traceback

import math

import torch
import torch.nn.functional as F

from model_17 import (Yol, DT, BOYUT, DURUM, LR, WD, DT_GENISLIK,
                      DT_DURUM, DT_BLOK, DT_BELLEK, DT_YANSIMA)
from olcme_17 import bos_kimligi

YIGIN = 512
GUNLUK, SONUC, DURDUR = [], {}, set()
IPLIK = {}      # ad -> Thread; nabiz canli mi diye bakar
_DISKE = {}     # ad -> gunluk.txt'ye henuz yazilmamis satirlar

# Surdurmede paketteki degerle AYNI olmali.  Farkliysa yorunge sessizce
# baskalasir: opt.load_state_dict lr/wd'yi paketten alir, gunluk cagriyi yazar.
SURDUR_ESIT = ("mimari", "n", "T", "yigin", "lr", "wd", "tohum", "boyut",
               "durum", "norm", "pay", "genislik", "blok", "bellek",
               "yansima", "gb_W", "gb_kafa", "gb_oncul", "egitim_iz", "sozluk")


def sonumlu(ad, p):
    """Weight decay alir mi: matrisler evet; dim < 2 (s0, hb) ve gecici
    bellegin konum yanliligi (gecici.b, 2 boyutlu ama bir yanlilik) hayir."""
    return p.dim() >= 2 and not ad.endswith("gecici.b")


def _olcut_yok(m, taraf, tam=False):
    raise RuntimeError("olcut verilmedi -- baslat(..., olcut=...) sart")


def _sonuc(r):
    """olcut dogruluk (float), (dogruluk, ce) ya da dict donebilir."""
    if isinstance(r, dict):
        return r
    if isinstance(r, (tuple, list)):
        return {"dogruluk": float(r[0]), "ce": float(r[1])}
    return {"dogruluk": float(r)}


def _ppl(ce):
    return "      -" if ce is None else f"{math.exp(ce):7.2f}"


def _not(ad, s):
    """Satir hem ortak gunluge (nabiz) hem kosunun disk kuyruguna."""
    s = f"[{ad}] {s}"
    GUNLUK.append(s)
    _DISKE.setdefault(ad, []).append(s)


def _gunluk(kok, ad):
    """Kosunun yeni satirlarini KENDI gunluk.txt'sine EKLER -- uzerine
    yazmaz, baska kosunun satirini tasimaz."""
    bek = _DISKE.get(ad)
    if not bek:
        return
    if kok:
        d = f"{kok}/{ad}"
        os.makedirs(d, exist_ok=True)
        with open(f"{d}/gunluk.txt", "a", encoding="utf-8") as f:
            f.write("\n".join(bek) + "\n")
    bek.clear()


def _iz(*t):
    """Egitim tensorlerinin parmak izi -- surdurme AYNI veriyle mi."""
    h = hashlib.sha256()
    for x in t:
        if x is not None:
            h.update(f"{tuple(x.shape)} {x.dtype}".encode())
            h.update(x.contiguous().numpy())
    return h.hexdigest()[:16]


def _tam(m, opt, uret, bilgi):
    """TAM anlik goruntu -- surdurmeye yeten her sey."""
    return dict(bilgi, agirlik=m.state_dict(), opt=opt.state_dict(),
                rng=torch.get_rng_state(), uret_rng=uret.get_state())


def _yaz(kok, ad, p):
    """Anlik goruntuyu diske yaz -- KOSUNUN ICINDE.
    Ayri hucre olarak dururken unutulur; model_16'da 8000 adimlik bir
    kosu oyle gitmisti.  Eskiler silinmez."""
    if not kok:
        return
    d = f"{kok}/{ad}"
    os.makedirs(d, exist_ok=True)
    torch.save(p, f"{d}/t{p['adim']}.pt")
    torch.save(p, f"{kok}/model_{ad}.pt")


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


def _denetle(p, sabit):
    """Surdurme paketi bu cagriyla AYNI kosu mu?  Degilse kosu BASLAMAZ."""
    kisa = lambda v: f"<{len(v)} birim>" if isinstance(v, list) else repr(v)
    fark = [f"{a}: paket {kisa(p[a])}, cagri {kisa(sabit[a])}"
            for a in SURDUR_ESIT
            if p.get(a) is not None and sabit.get(a) is not None
            and p[a] != sabit[a]]
    if fark:
        raise ValueError("surdurme paketi bu cagriyla uyusmuyor -- "
                         + "; ".join(fark))


def _kos(ad, EG, N, olcut, aygit, kok, ek, boyut, durum,
         lr, wd, adim, tohum, yigin, bas, yedek, surdur, derle, sozluk=None,
         mimari="kelime_matris", genislik=DT_GENISLIK, blok=DT_BLOK,
         bellek=DT_BELLEK, yansima=DT_YANSIMA, gb_W=0, gb_kafa=4,
         gb_oncul=0.0):
    not_ = lambda s: _not(ad, s)
    torch.manual_seed(tohum)
    dt = mimari == "dt"
    m = (DT(N, genislik=genislik, durum=durum, blok=blok, bellek=bellek,
            yansima=yansima, tohum=tohum, gb_W=gb_W, gb_kafa=gb_kafa,
            gb_oncul=gb_oncul) if dt
         else Yol(N, boyut=boyut, durum=durum, tohum=tohum)).to(aygit)

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
    def _kayip(w, mk, hd):
        """hd yoksa modelin kendi kaybi; varsa YALNIZ hd'deki hedefler sayilir
        (matematikte: cevabin rakamlari ve EOS, soru rakamlari degil)."""
        if hd is None:
            return m.kayip(w, mk)
        p = m.dizi(w, mk)[:, :-1].reshape(-1, m.n)
        a = hd[:, 1:].reshape(-1)
        k = F.cross_entropy(p, w[:, 1:].reshape(-1), reduction="none")
        return (k * a).sum() / a.sum()

    egit = _kayip
    if derle:
        egit = torch.compile(_kayip)
        not_("torch.compile ACIK -- ilk adim DERLEME yuzunden yavas")
    dec = [p for a, p in m.named_parameters() if sonumlu(a, p)]
    nodec = [p for a, p in m.named_parameters() if not sonumlu(a, p)]
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": wd},
                             {"params": nodec, "weight_decay": 0.0}], lr=lr)
    uret = torch.Generator(device="cpu").manual_seed(tohum)

    # PENCERELER CPU'DA KALIR, yigin yigin tasinir.  64 MB dilimde
    # tumu GPU'ya sigiyordu (253 MB) ama tam korpusta 8,8 GB eder ve
    # aktivasyonlarin yanina sigmaz.  Tasima maliyeti adim basina
    # 512x256 int32 = 0,5 MB -- olcusuz.
    # EG: tek W, (W, M) ya da (W, M, H).  M dolgu maskesi -- HER HIKAYE BIR
    # PENCERE oldugu icin hikaye bitince kalan yer <dolgu>.  H hedef maskesi:
    # verilirse kayip YALNIZ orada sayilir.
    W, M, H = ((list(EG) + [None, None])[:3] if isinstance(EG, (tuple, list))
               else (EG, None, None))
    W = W if W.device.type == "cpu" else W.cpu()
    M = M if M is None or M.device.type == "cpu" else M.cpu()
    H = H if H is None or H.device.type == "cpu" else H.cpu()
    n, T = W.shape
    par = sum(p.numel() for p in m.parameters())
    # Pakete giden kimlik: sozluk ve iz pakette durursa konus tahmin etmez,
    # surdurme de ayni veriyi dogrular.
    hk = bos_kimligi(W, M)
    mim = (dict(genislik=genislik, blok=blok, bellek=bellek, yansima=yansima,
                gb_W=gb_W, gb_kafa=gb_kafa, gb_oncul=gb_oncul)
           if dt else dict(boyut=boyut, norm=m.norm))
    sabit = dict(ek or {}, mimari=m.mimari, n=N, durum=durum, **mim,
                 pay=m.pay, lr=lr, wd=wd, tohum=tohum, yigin=yigin, T=T,
                 parametre=par, derle=derle, egitim_iz=_iz(W, M, H),
                 bos_eos=hk is not None, hikaye_id=hk,
                 sozluk=None if sozluk is None else [str(a) for a in sozluk])

    son, onceki = -1, None                # son TAMAMLANAN adim
    if surdur:
        p = torch.load(surdur, weights_only=False, map_location=aygit)
        _denetle(p, sabit)
        sabit["sozluk"] = sabit["sozluk"] or p.get("sozluk")
        m.load_state_dict(p["agirlik"])
        opt.load_state_dict(p["opt"])
        torch.set_rng_state(p["rng"].cpu())
        uret.set_state(p["uret_rng"].cpu())
        son, onceki = p["adim"], p.get("kayip_iz")
        not_(f"SURDURULUYOR  {os.path.basename(surdur)}  adim {son}")
    son_kayit = son                       # paket o adimin kaydi zaten

    # HER ADIMIN kaybi: egrinin tam cozunurlugu.  GPU'da birikir (adim basina
    # senkron yok), olcum noktasinda pakete girer, surdurmede geri yuklenir.
    kiz = torch.full((max(adim, son) + 1,), float("nan"), device=aygit)
    if onceki is not None:
        kiz[:len(onceki)] = onceki.to(aygit)

    _et = 1.0 if M is None else float(M.sum()) / M.numel()
    not_(f"pencere {n:,} x {T}   {n * T:,} yuva"
         + ("" if M is None else f"   dolgu %{100 * (1 - _et):.1f}"
                                 f"   ETKIN {n * T * _et:,.0f}"))
    not_(f"mimari {m.mimari}  "
         + (f"genislik {genislik} durum {durum}x{durum} blok {blok} "
            f"bellek {bellek} yansima {yansima}"
            + (f" gecici W {gb_W} kafa {gb_kafa}" if gb_W else "")
            + (f" oncul {gb_oncul}" if gb_oncul else "") if dt
            else f"boyut {boyut} durum {durum}")
         + f"  lr {lr} wd {wd} tohum {tohum}  sozluk {N}  parametre {par:,}")
    not_(f"yigin {yigin}   adim basina {yigin * (T - 1):,} tahmin"
         f"   epok = {n / yigin:,.0f} adim")
    not_(f"veri izi {sabit['egitim_iz']}   olcum her {bas}   yedek her {yedek}"
         f"   BOS/EOS {'VAR' if hk is not None else 'YOK'}"
         + ("" if sabit["sozluk"] else
            "   UYARI: sozluk YOK, konus tahmin etmek zorunda"))
    not_("OLCUT: soru soruldu, cevap BIREBIR dogru mu (EOS'a kadar serbest "
         "uretim); kayip yalniz cevapta" if H is not None else
         f"OLCUT: SONRAKI JETON.  dogruluk (sans {1 / N:.5f}) ve dogrulama "
         "perplexity'si; 'govde' TAM1'le AYNI hedefler")
    not_("  adim    kayip   egitim  dogrulama   dg ppl   "
         + ("  ilk  uzunluk" if H is not None else "govde") + "      sn")
    _gunluk(kok, ad)

    def olc(i, tam=False):
        e = _sonuc(olcut(m, "eg", tam=tam))
        d = _sonuc(olcut(m, "dg", tam=tam))
        b = dict(sabit, adim=i, kayip=float(kay.detach()),
                 kayip_iz=kiz[:i + 1].cpu(),
                 egitim=e["dogruluk"], dogrulama=d["dogruluk"],
                 egitim_ce=e.get("ce"), dogrulama_ce=d.get("ce"),
                 dogrulama_tur={k: v for k, v in d.items()
                                if k not in ("dogruluk", "ce")})
        SONUC[ad] = dict(b, model=m)
        return b

    def satir(b, gecen, im=""):
        t = b["dogrulama_tur"]
        son_ = (f"{t['ilk']:7.4f}  {t['uzunluk']:7.4f}" if "ilk" in t
                else _ppl(t.get("ce_govde")))         # matematik: ilk rakam
        return (f"{b['adim']:6d}  {b['kayip']:7.3f}  {b['egitim']:7.4f}  "
                f"{b['dogrulama']:9.4f}  {_ppl(b['dogrulama_ce'])} "
                f"{son_}  {gecen:6.0f}{im}")

    # Yedek, o adimin opt.step()'i BITTIKTEN sonra yazilir.  Yani t<N>
    # N adimi ICERIR ve surdurme N+1'den baslar; N'den baslamak o adimi
    # IKI KEZ atar ve yorunge kayar (test_17 §1).
    t0, kay, bilgi, durdu = time.time(), None, None, False
    for i in range(son + 1, adim + 1):
        if ad in DURDUR:
            durdu = True
            break
        j = torch.randint(0, n, (yigin,), generator=uret)
        opt.zero_grad()
        w_ = W[j].to(aygit, non_blocking=True).long()
        m_ = None if M is None else M[j].to(aygit, non_blocking=True)
        h_ = None if H is None else H[j].to(aygit, non_blocking=True)
        kay = egit(w_, m_, h_)
        kiz[i] = kay.detach()
        kay.backward()
        opt.step()
        son = i

        # yedek bas'in katina bagli DEGIL: bas 100, yedek 250 -> 250'de yazar.
        if i % bas == 0 or i % yedek == 0:
            gecen = time.time() - t0
            bilgi, im = olc(i), ""
            if i % yedek == 0:
                _yaz(kok, ad, _tam(m, opt, uret, bilgi))
                son_kayit, im = i, "  yedek"
            not_(satir(bilgi, gecen, im))
            _gunluk(kok, ad)

    if son < 0:
        not_("DURDURULDU -- hic adim atilmadi, kayit yok")
        _gunluk(kok, ad)
        return
    if kay is None:                       # surdurulen kosuda yeni adim yok
        with torch.no_grad():
            kay = _kayip(W[:yigin].to(aygit).long(),
                         None if M is None else M[:yigin].to(aygit),
                         None if H is None else H[:yigin].to(aygit))

    # ONCE KAYDET, SONRA OLC.  Tam olcum TAM1'de ~15 dk surdu; o arada
    # oturum duserse son yedekten beri yapilan is giderdi.
    if son_kayit != son:
        if bilgi is None or bilgi["adim"] != son:
            gecen = time.time() - t0
            bilgi = olc(son)
            _yaz(kok, ad, _tam(m, opt, uret, bilgi))
            not_(satir(bilgi, gecen, "  yedek"))
        else:
            _yaz(kok, ad, _tam(m, opt, uret, bilgi))
            not_(f"{son:6d}  yedek")
        _gunluk(kok, ad)

    if durdu:
        # Durdurmanin amaci GPU'yu HEMEN birakmak; tam olcum paketten alinir.
        SONUC[ad] = dict(bilgi or dict(sabit, adim=son), model=m,
                         biti=True, durduruldu=True)
        not_(f"DURDURULDU  son tamamlanan adim {son} kaydedildi"
             "  -- tam olcum ATLANDI")
        _gunluk(kok, ad)
        return

    bilgi = dict(olc(son, tam=True), biti=True)
    SONUC[ad] = dict(bilgi, model=m)
    _yaz(kok, ad, _tam(m, opt, uret, bilgi))
    not_(f"BITTI   egitim {bilgi['egitim']:.4f}   dogrulama "
         f"{bilgi['dogrulama']:.4f}   dg ppl {_ppl(bilgi['dogrulama_ce']).strip()}"
         f"   govde {_ppl(bilgi['dogrulama_tur'].get('ce_govde')).strip()}"
         f"   ({time.time() - t0:.0f} sn)")
    _gunluk(kok, ad)


def _korumali(**k):
    """_kos'u sarar: iplik olurse nabiz bunu GORSUN, kosu suruyor sanilmasin."""
    ad = k["ad"]
    try:
        _kos(**k)
    except Exception:
        tb = traceback.format_exc()
        for s in tb.rstrip().splitlines():
            _not(ad, "HATA  " + s)
        SONUC[ad] = dict(SONUC.get(ad, {}), hata=tb)
        try:
            _gunluk(k["kok"], ad)
        except Exception:
            pass


def baslat(ad, EG, N, *, olcut=_olcut_yok, aygit="cuda", kok=None, ek=None,
           boyut=BOYUT, durum=None, lr=LR, wd=WD,
           adim=20000, tohum=0, yigin=YIGIN, bas=100, yedek=500,
           surdur=None, derle=False, sozluk=None, mimari="kelime_matris",
           genislik=DT_GENISLIK, blok=DT_BLOK, bellek=DT_BELLEK,
           yansima=DT_YANSIMA, gb_W=0, gb_kafa=4, gb_oncul=0.0):
    """ARKA PLANDA baslatir, HEMEN doner (kural 8).

    EG      (W, M) ya da tek W -- W (n, T) pencere yigini, M dolgu maskesi
    olcut   olcut(m, "eg"|"dg", tam=False) -> olcme_17.olc sozlugu
            (dogruluk, ce, ce_bas/govde/son) ya da yalniz dogruluk
    bas     olcum araligi;  yedek  anlik goruntu araligi -- birbirinden BAGIMSIZ.
            HER ADIMIN kaybi ikisinden de bagimsiz: pakette `kayip_iz`.
    surdur  bir anlik goruntu yolu verilirse KALDIGI YERDEN devam eder
            (agirlik + optimizer + RNG).  Kural 1: uzatma SURDURMEDIR.
            Paketteki ayar ya da veri bu cagriyla tutmazsa kosu BASLAMAZ.
    sozluk  kelime listesi -- pakete yazilir, konus tahmin etmek zorunda kalmaz
    derle   torch.compile.  Eski mimaride 3,90 kat olculdu (_kos'taki not);
            kisa kosuda zararli.  DT'de OLCULMEDI.
    mimari  "kelime_matris" (TAM1/TAM2, varsayilan) ya da "dt" (hedef mimari:
            genislik, blok, bellek, yansima; durum verilmezse 64)
    gb_W    DT'de gecici bellek penceresi (0 = yok), gb_kafa okuma kafasi,
            gb_oncul kafa h'nin 'h+1 geri' baslangic egilimi (0 = duz)
    """
    if durum is None:
        durum = DT_DURUM if mimari == "dt" else DURUM
    t = IPLIK.get(ad)
    if t is not None and t.is_alive():
        raise RuntimeError(f"{ad} hala kosuyor -- once durdur('{ad}')")
    _DISKE.pop(ad, None)          # onceki kosunun yazilamamis satiri karismasin
    if kok is None:
        _not(ad, "UYARI: kok YOK, agirlik KAYDEDILMIYOR")
    if not surdur:
        tasinan = _koru(kok, ad)
        if tasinan:
            print(f"  ESKI YEDEKLER KORUNDU -> {tasinan}/")
            _not(ad, f"eski yedekler tasindi -> {tasinan}/")
    DURDUR.discard(ad)
    t = threading.Thread(target=_korumali, daemon=True, kwargs=dict(
        ad=ad, EG=EG, N=N, olcut=olcut, aygit=aygit, kok=kok, ek=ek,
        boyut=boyut, durum=durum, lr=lr, wd=wd, adim=adim, tohum=tohum,
        yigin=yigin, bas=bas, yedek=yedek, surdur=surdur, derle=derle,
        sozluk=sozluk, mimari=mimari, genislik=genislik, blok=blok,
        bellek=bellek, yansima=yansima, gb_W=gb_W, gb_kafa=gb_kafa,
        gb_oncul=gb_oncul))
    IPLIK[ad] = t
    t.start()
    return f"{ad} basladi" + (f"  ({os.path.basename(surdur)}'den)"
                              if surdur else "")


def nabiz(son=40):
    """Gunlugu bas.  HICBIR SEY KOSTURMAZ (kural 8)."""
    hal = {a: ("CANLI" if t.is_alive()
               else "HATA" if SONUC.get(a, {}).get("hata") else "bitti")
           for a, t in IPLIK.items()}
    print(f"gunluk {len(GUNLUK)} satir   iplik: {hal}"
          f"   DURDUR: {sorted(DURDUR)}")
    for s in GUNLUK[-son:]:
        print(s)


def durdur(ad=None):
    """Bayrak koyar; iplik bir sonraki adimda cikar ve cikmadan KAYDEDER.
    Tam olcum ATLANIR -- durdurmanin amaci GPU'yu hemen birakmak."""
    for a in ([ad] if ad else [a for a, t in IPLIK.items() if t.is_alive()]):
        DURDUR.add(a)
    print("durdurma bayragi:", sorted(DURDUR))
