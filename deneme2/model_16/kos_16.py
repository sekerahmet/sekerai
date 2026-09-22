"""DIL EGITIMI.  model_16'nin kosu dongusu -- defterde DEGIL, depoda.

Kullanici, 22 Eylul: "kos 16 sifirdan yazalim, kopya yok.  sakin model
icindeki ayarlari kaldirma, onlar model icinde kalacak."

Bu dosya MODELE DOKUNMAZ.  Mimari de ayarlari da `model_16.py`de durur;
burada yalniz EGITIM vardir: yigin, kayip, optimizer, yedek, gunluk.

FARKI NEREDE -- aritmetik kosusundan uc ayrilma:

  1  YIGIN X'TEN PENCERE.  Aritmetikte her ornek bir soruydu; burada bir
     ornek 512 karakterlik bir PENCERE ve hangi cumlenin nerede oldugu
     modele SOYLENMIYOR.

  2  KAYIP HER KONUMDA.  Pencerenin 512 konumunun her birinde "sonraki
     karakter ne" sorulur.  Aritmetikte bir soru bir gradyan isareti
     veriyordu; burada bir pencere 511 isaret verir.  Model bunun icin
     her yuvanin KENDI sorusunu sormasi gerekiyor (`m.dizi`).

  3  PAD KAYIPTAN DUSER.  Pencere kuyrugunun %10,8'i dolgu.  Sayilsaydi
     gradyanin %10,8'i "dolgu tahmin et" ogretirdi.

SURDURME.  CLAUDE.md kural 1: "Uzatma SURDURMEDIR, sifirdan kosu DEGIL."
Kural bunu istiyordu ama model_15'in kosusu YALNIZ AGIRLIK kaydediyordu;
Adam momentleri ve yigin RNG'si yoktu, yani agirliktan devam etmek sessizce
BASKA BIR YORUNGE uretirdi.  Arsivde bir kol tam boyle gecersiz kalmisti.
Burada anlik goruntu TAM: agirlik + optimizer + iki RNG durumu.

Kural 8: baslat() ARKA PLANDA calisir ve HEMEN doner; izleme nabiz()
hucresiyle OKUYARAK yapilir.

ZINCIRDEKI YERI.  Kim kimi cagiriyor, bu dosya nerede:
(model BIRIM goruyor -- karakter yalnizca ara adim ve kapi)

  veri_16     graf: 1608 varlik, 24 iliski, olgular
  metin_16    graf -> duz Turkce cumle
  korpus_16   cumle -> belge -> paketlenmis akis
  jeton_16    KARAKTER sozlugu + GIDIS-DONUS KAPISI
  birim_16    metin -> sayim -> kok havuzu -> BIRIM AKISI -> pencere
  ek_16       kelime -> kok + ek    (Turkce morfolojisi)

  taban_16    bolmeler (ezber_* / cikarim_*) + Ayar tanimi
  ayar_16     dugmeler
  hazirla_16  veriyi dosyaya yazar, Colab Drive'dan OKUR

  model_16    MIMARI -- model_15'ten
  kos_16      egitim dongusu   <-- BU DOSYA
  olcme_16    olcu: soru soruldu, cevap dogru mu
"""
import os
import shutil
import threading
import time

import torch
import torch.nn.functional as F

# Ayarlar MODELIN ICINDE.  Burada YENIDEN TANIMLANMAZ.
from model_16 import Yol, BOYUT, DURUM, LR, WD

GUNLUK, SONUC, DURDUR = [], {}, set()

# Yigin.  Sinir ozyineleme degil DIKKAT: nedensel dikkat (B,T,T) puan
# tutuyor ve geri gecis icin agirliklari da saklar, yani T=512'de ornek
# basina ~2,1 MB.  23 GB'lik bir kartta 4096 sinirda, 2048 rahat.
# Ozyineleme ise T kadar ARDISIK minik cekirdek -- suresi yigindan
# BAGIMSIZ sabit bir gecikme.  Yigin buyudukce o sabit daha cok jetona
# bolunur; hiz kazanci buradan gelir.
YIGIN = 2048


def _tam(m, opt, uret, bilgi):
    """TAM anlik goruntu -- surdurmeye yeten her sey."""
    return dict(bilgi,
                agirlik={k: v.detach().cpu() for k, v in m.state_dict().items()},
                opt=opt.state_dict(),
                rng=torch.get_rng_state(),
                uret_rng=uret.get_state())


def _yaz(kok, ad, p):
    """Anlik goruntuyu ve gunlugu Drive'a yaz -- KOSUNUN ICINDE.

    Ayri hucre olarak dururken unutuldu ve 8000 adimlik bir kosu gitti.
    Eskiler silinmez: model kucuk, Drive buyuk, ve seyreltmek bu projede
    daha once uc kez kosu yeniden baslatmaya mal oldu."""
    if not kok:
        return
    d = f"{kok}/{ad}"
    os.makedirs(d, exist_ok=True)
    torch.save(p, f"{d}/t{p['adim']}.pt")
    torch.save(p, f"{kok}/model_{ad}.pt")
    with open(f"{d}/gunluk.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(GUNLUK) + "\n")


def _koru(kok, ad):
    """Ayni adla yeni kosu eskiyi EZMEZ, yan klasore TASIR (kural 1'in
    `--ustune` davranisi).  Iplikten ONCE calisir ki hata gorunur olsun."""
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


def kayip(m, w, PAD):
    """SONRAKI KARAKTER, her konumda, PAD haric.

    w (B,T) -> puan (B,T,n).  Konum j, j+1'i tahmin eder; son konumun
    hedefi yok.  `ignore_index=PAD` hem kuyruk dolgusunu hem dolgudan
    sonra gelen her seyi kayiptan duser."""
    puan = m.dizi(w)                       # (B,T,n)
    return F.cross_entropy(puan[:, :-1].reshape(-1, puan.shape[-1]),
                           w[:, 1:].reshape(-1), ignore_index=PAD)


def _kos(ad, X, PAD, olcut, aygit, kok, ek,
         boyut, durum, lr, wd, adim, tohum, yigin, bas, yedek, surdur):
    not_ = GUNLUK.append
    torch.manual_seed(tohum)
    m = Yol(ek["n"], boyut=boyut, durum=durum, tohum=tohum).to(aygit)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
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

    par = sum(p.numel() for p in m.parameters())
    N, T = X.shape
    not_(f"[{ad}] boyut {boyut} durum {durum} lr {lr} wd {wd} tohum {tohum}"
         f"  parametre {par}")
    not_(f"[{ad}] pencere {N} x {T}   yigin {yigin}"
         f"   adim basina {yigin * T / 1e6:.1f}M jeton")
    not_(f"[{ad}]   adim    kayip   ezber  cikarim   jeton/sn      sn")

    t0, i = time.time(), bas_adim
    for i in range(bas_adim, adim + 1):
        if ad in DURDUR:
            not_(f"[{ad}] DURDURULDU  adim {i}")
            break
        j = torch.randint(0, N, (yigin,), generator=uret)
        k = kayip(m, X[j].to(aygit).long(), PAD)
        opt.zero_grad(); k.backward(); opt.step()

        if i % bas == 0:
            gecen = time.time() - t0
            e, c = olcut(m, "ezber"), olcut(m, "cikarim")
            bilgi = dict(ek, adim=i, boyut=boyut, durum=durum, lr=lr, wd=wd,
                         tohum=tohum, yigin=yigin, parametre=par,
                         kayip=float(k.detach()), ezber=e, cikarim=c,
                         jeton_sn=(i - bas_adim + 1) * yigin * T / max(gecen, 1e-9))
            SONUC[ad] = dict(bilgi, model=m)
            im = ""
            if i % yedek == 0:
                _yaz(kok, ad, _tam(m, opt, uret, bilgi)); im = "  yedek"
            not_(f"[{ad}] {i:6d}   {float(k.detach()):.4f}  {e:.4f}   {c:.4f}"
                 f"   {bilgi['jeton_sn']/1e6:8.2f}M  {gecen:6.0f}{im}")

    gecen = time.time() - t0
    bilgi = dict(ek, adim=i, boyut=boyut, durum=durum, lr=lr, wd=wd,
                 tohum=tohum, yigin=yigin, parametre=par, kayip=float(k.detach()),
                 ezber=olcut(m, "ezber", tam=True),
                 cikarim=olcut(m, "cikarim", tam=True),
                 jeton_sn=(i - bas_adim + 1) * yigin * T / max(gecen, 1e-9),
                 biti=True)
    SONUC[ad] = dict(bilgi, model=m)
    _yaz(kok, ad, _tam(m, opt, uret, bilgi))
    not_(f"[{ad}] BITTI   kayip {float(k.detach()):.4f}   ezber {bilgi['ezber']:.4f}"
         f"   cikarim {bilgi['cikarim']:.4f}   ({gecen:.0f} sn)")


def baslat(ad, X, PAD, n, *, olcut, aygit="cuda", kok=None, ek=None,
           boyut=BOYUT, durum=DURUM, lr=LR, wd=WD,
           adim=20000, tohum=0, yigin=YIGIN, bas=200, yedek=1000,
           surdur=None):
    """ARKA PLANDA baslatir, HEMEN doner (kural 8).

    X       (N, T) int8 pencere tablosu -- CPU'da durur, yigin GPU'ya gider
    PAD     kayiptan dusulecek jeton
    n       sozluk boyu
    olcut   olcut(m, "ezber"|"cikarim", tam=False) -> oran.
            TEK ANALIZ: soru soruldu, cevap dogru mu.
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
        args=(ad, X, PAD, olcut, aygit, kok, dict(ek or {}, n=n),
              boyut, durum, lr, wd, adim, tohum, yigin, bas, yedek,
              surdur)).start()
    return f"{ad} basladi" + (f"  ({os.path.basename(surdur)}'den)" if surdur else "")


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
