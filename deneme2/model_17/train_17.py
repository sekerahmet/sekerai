# -*- coding: utf-8 -*-
"""train_17 -- egitim dongusu: optimizer, yigin, kayit, SURDURME.

AMAC model_15'inki: onek -> TEK hedef.  Uretim yok; bir zincir verilir,
bilesik iliski TEK bir etiket olarak okunur.

    o, _ = m.dikkat(w)                onek -> tek cikti noktasi
    puan = -||o - E||^2
    kayip = cross_entropy(puan, h)

YIGIN HER OBEKTEN PAY ALIR.  k degisken (2..10) ve tensor dikdortgen
olmak zorunda, o yuzden obek var; ama obek yalnizca TENSOR SEKLI.
Bir adimda tek obek secmek kucuk obegi "ya hepsi ya hicbiri" yapar --
kos_15'te OLCULDU: 8000 adimda 4 tokenlik obek SIFIR kez secilmisti.
Gradyan obekler boyunca BIRIKIR, sonra TEK optimizer adimi; kayip
toplam paya bolundugu icin hepsi tek bir yigin gibi davranir.

AdamW, Adam DEGIL.  Adam'in weight_decay'i L2'yi gradyana katar ve
1/sqrt(v) ile normalize eder; gorev gradyani sadelesen bir bilesende
adim -lr*sign(t) olur ve bilesen wd'den BAGIMSIZ olarak lr hiziyla
silinir.  model_16'da olculdu: token basina parametreler cokuyordu
(E 15,8x, b 21x, M 31x), paylasilanlar cokmuyordu.
dim < 2 (s0, hb) decay disinda.

KOSU ARKA PLANDA (CLAUDE.md kural 8).  `baslat` hemen doner, ilerleme
`nabiz` ile OKUNUR.
"""
from __future__ import annotations

import os
import shutil
import threading
import time

import torch
import torch.nn.functional as F

from model_17 import Yol, BOYUT, DURUM, LR, WD

YIGIN = 4096
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
         lr, wd, adim, tohum, yigin, bas, yedek, surdur):
    not_ = GUNLUK.append
    torch.manual_seed(tohum)
    m = Yol(N, boyut=boyut, durum=durum, tohum=tohum).to(aygit)
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

    # Obek (w, h) ya da (w, h, maske) -- C'de diziler dolgulu geliyor.
    OB = [(k, v[0].to(aygit), v[1].to(aygit),
           v[2].to(aygit) if len(v) > 2 else None)
          for k, v in sorted(EG.items())]
    n = [w.shape[0] for _, w, _, _ in OB]
    T = sum(n)
    # Obek basina cekilis: satir sayisiyla ORANTILI, ama EN AZ 1 --
    # tek satirlik obek bile her adimda gelir.
    CEK = [max(1, int(round(yigin * k / T))) for k in n]
    par = sum(p.numel() for p in m.parameters())
    not_(f"[{ad}] {len(OB)} obek (k = " + ", ".join(str(k) for k, *_ in OB)
         + f")   {T:,} ornek")
    not_(f"[{ad}] YIGIN {sum(CEK):,} = her obekten pay "
         f"({min(CEK)}..{max(CEK)})   en kucuk obek {min(n)} ornek")
    not_(f"[{ad}] boyut {boyut} durum {durum} lr {lr} wd {wd} tohum {tohum}"
         f"  sozluk {N}  parametre {par}")
    not_(f"[{ad}] TEK OLCUT: zincir verildi, bilesik iliski DOGRU MU."
         f"  sans {1/18:.4f}")
    not_(f"[{ad}]   adim    kayip   egitim  dogrulama  sinav      sn")

    t0, i = time.time(), bas_adim
    for i in range(bas_adim, adim + 1):
        if ad in DURDUR:
            not_(f"[{ad}] DURDURULDU  adim {i}")
            break
        opt.zero_grad()
        top = 0.0
        for b, (k, w, h, mk) in enumerate(OB):
            j = torch.randint(0, w.shape[0], (CEK[b],), generator=uret)
            o, _ = m.dikkat(w[j], None if mk is None else mk[j])
            puan = -torch.cdist(o, m.E) ** 2
            kay = F.cross_entropy(puan, h[j], reduction="sum") / sum(CEK)
            kay.backward()
            top += float(kay.detach())
        opt.step()

        if i % bas == 0:
            gecen = time.time() - t0
            e, d, s = (olcut(m, "eg"), olcut(m, "dg"), olcut(m, "si"))
            bilgi = dict(ek or {}, n=N, adim=i, boyut=boyut, durum=durum,
                         lr=lr, wd=wd, tohum=tohum, yigin=sum(CEK),
                         parametre=par, kayip=top, egitim=e,
                         dogrulama=d, sinav=s)
            SONUC[ad] = dict(bilgi, model=m)
            im = ""
            if i % yedek == 0:
                _yaz(kok, ad, _tam(m, opt, uret, bilgi)); im = "  yedek"
            not_(f"[{ad}] {i:6d}  {top:7.4f}  {e:7.4f}  {d:9.4f}  {s:5.4f}"
                 f"  {gecen:6.0f}{im}")

    e, d, s = (olcut(m, "eg", tam=True), olcut(m, "dg", tam=True),
               olcut(m, "si", tam=True))
    bilgi = dict(ek or {}, n=N, adim=i, boyut=boyut, durum=durum, lr=lr,
                 wd=wd, tohum=tohum, yigin=sum(CEK), parametre=par,
                 kayip=top, egitim=e, dogrulama=d, sinav=s, biti=True)
    SONUC[ad] = dict(bilgi, model=m)
    _yaz(kok, ad, _tam(m, opt, uret, bilgi))
    not_(f"[{ad}] BITTI   egitim {e:.4f}   dogrulama {d:.4f}   "
         f"sinav {s:.4f}   ({time.time()-t0:.0f} sn)")


def baslat(ad, EG, N, *, olcut=_olcut_yok, aygit="cuda", kok=None, ek=None,
           boyut=BOYUT, durum=DURUM, lr=LR, wd=WD,
           adim=20000, tohum=0, yigin=YIGIN, bas=200, yedek=1000,
           surdur=None):
    """ARKA PLANDA baslatir, HEMEN doner (kural 8).

    EG      {k: (w, h)}  --  w (n, k+1) girdi, h (n,) hedef
    olcut   olcut(m, "eg"|"dg"|"si", tam=False) -> oran
    surdur  bir anlik goruntu yolu verilirse KALDIGI YERDEN devam eder
            (agirlik + optimizer + RNG).  Kural 1: uzatma SURDURMEDIR.
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
              adim, tohum, yigin, bas, yedek, surdur)).start()
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
