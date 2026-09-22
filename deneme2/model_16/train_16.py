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
  train_16    egitim dongusu   <-- BU DOSYA
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


def _kos(ad, X, PAD, olcut, aygit, kok, ek, boyut, durum,
         lr, wd, adim, tohum, yigin, bas, yedek, surdur, t_len, atla,
         ofset, hiza):
    not_ = GUNLUK.append
    torch.manual_seed(tohum)
    m = Yol(ek["n"], boyut=boyut, durum=durum, tohum=tohum).to(aygit)
    # AdamW, Adam DEGIL: Adam'in weight_decay'i L2'yi gradyana katar ve
    # 1/sqrt(v) ile normalize eder -- gorev gradyani sadelesen bir
    # bilesende adim -lr*sign(t) olur, yani bilesen wd'den BAGIMSIZ
    # olarak lr hiziyla silinir.  AdamW decay'i agirliga AYRI uygular.
    # dim < 2 (s0, hb) decay disinda -- model_00..02 ile ayni bolme.
    dec = [p for p in m.parameters() if p.dim() >= 2]
    nodec = [p for p in m.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": wd},
                             {"params": nodec, "weight_decay": 0.0}],
                            lr=lr)
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
    # X bir SOZLUK ise TEKER TEKER egitim: {uzunluk: (X, OFS)}.
    # Her uzunluk AYRI tensor, hepsi tam dolu -- DOLGU YOK.
    #
    # !! HER ADIMDA HER OBEKTEN PAY ALINIR.  Obek yalnizca tensorun
    # SEKLI; bir adimda tek obek secmek kucuk obegi "ya hepsi ya
    # hicbiri" yapar.  model_15'te OLCULDU: 8000 adimda 4 tokenlik obek
    # SIFIR kez secilmisti.  Bizde 30 birimlik obekte 1 satir var ve
    # orantili secimle 20.000 adimda beklentisi 0,02 idi -- en uzun
    # parcalar, yani cok adimli sorular, egitime HIC girmezdi.
    #
    # Gradyan obekler boyunca BIRIKIR, sonra TEK optimizer adimi:
    # kayip `yigin`a bolundugu icin butun obekler tek bir yigin gibi
    # davranir.
    OBEK = None
    if isinstance(X, dict):
        OBEK = sorted(X.items())
        _n = [a.shape[0] for _, (a, _) in OBEK]
        N = sum(_n)
        T = sum(k * L for k, (L, _) in zip(_n, OBEK)) / N
        # Obek basina cekilis: satir sayisiyla ORANTILI, ama EN AZ 1 --
        # tek satirlik obek bile her adimda gelir.
        CEK = [max(1, int(round(yigin * k / N))) for k in _n]
        not_(f"[{ad}] TEKER TEKER: {len(OBEK)} uzunluk obegi, {N:,} parca"
             f"   ort {T:.1f} birim   DOLGU YOK")
        not_(f"[{ad}] YIGIN {sum(CEK):,} = her obekten pay "
             f"({min(CEK)}..{max(CEK)})   en kucuk obek {min(_n)} satir")
    # X ya PENCERE TABLOSU (B,T) ya da tek uzun AKIS (N,).  Akis verilirse
    # pencere BURADA aciliyor: sliding_window_view bir GORUNUM, kopya yok.
    # Birim dosyasi akis sakliyor cunku atla=4 ile pencereler 20 birim
    # ortusuyor ve tablo 6 kat sisiyordu (192 MB -> 32,8 MB).
    if OBEK is not None:
        pass
    elif X.dim() == 1:
        import numpy as np
        P = np.lib.stride_tricks.sliding_window_view(X.numpy(), t_len)
        X = torch.from_numpy(P)                    # TAM gorunum
        # BAS: pencerenin hangi konumdan acilacagi.  hiza yoksa eski
        # davranis (her `atla` birimde bir) -- ayni sayida, ayni sirada,
        # yani RNG tuketimi ve cekilen yigin BIT DUZEYINDE ayni.
        if hiza is None:
            bs = np.arange(0, len(P), atla, dtype=np.int64)
            nasil = f"atla {atla}"
        else:
            bs = np.asarray(hiza, dtype=np.int64)
            bs = bs[bs < len(P)]
            nasil = "CUMLE BASI"
        BAS = torch.from_numpy(bs)
        if ofset is not None:
            # ofset AKISLA ayni uzunlukta; pencereler X ile AYNI
            # gorunumle aciliyor ki konumlar birebir ortussun.
            o1 = np.asarray(ofset)
            OF = np.lib.stride_tricks.sliding_window_view(ofset, t_len)
            ofset = torch.from_numpy(OF)           # GORUNUM, kopya yok
            not_(f"[{ad}] CEVAP KAPISI acik: aralik basi "
                 f"{int((o1[bs] == 0).sum()):,} pencerede")
        not_(f"[{ad}] akis {len(P) + t_len - 1:,} -> pencere "
             f"{len(bs):,} x {t_len}  ({nasil}, GORUNUM)")
    else:
        BAS = torch.arange(X.shape[0])
    if OBEK is None:
        N, T = len(BAS), X.shape[1]
    not_(f"[{ad}] boyut {boyut} durum {durum} lr {lr} wd {wd} tohum {tohum}"
         f"  parametre {par}")
    _jt = sum(c * L for c, (L, _) in zip(CEK, OBEK)) if OBEK else yigin * T
    not_(f"[{ad}] ornek {N:,} x {T:.1f} birim   yigin "
         f"{sum(CEK) if OBEK else yigin:,}   adim basina {_jt/1e6:.3f}M birim")
    not_(f"[{ad}]   adim    kayip   ezber  cikarim   jeton/sn      sn")

    t0, i = time.time(), bas_adim
    for i in range(bas_adim, adim + 1):
        if ad in DURDUR:
            not_(f"[{ad}] DURDURULDU  adim {i}")
            break
        opt.zero_grad()
        if OBEK is not None:
            # HER obekten pay; gradyan birikir, adim SONDA atilir.
            _top, _n_ok = 0.0, 0
            for _b, (_L, (_X, _O)) in enumerate(OBEK):
                j = torch.randint(0, _X.shape[0], (CEK[_b],), generator=uret)
                _k = m.kayip(_X[j].to(aygit).long(), PAD,
                             None if ofset is None else _O[j].to(aygit).long())
                # `yigin`a degil, obeklerin TOPLAM payina bolunur ki
                # butun obekler tek bir yigin gibi davransin.
                (_k * (CEK[_b] / sum(CEK))).backward()
                _top += float(_k.detach()) * CEK[_b]
                _n_ok += CEK[_b]
            k = torch.tensor(_top / _n_ok)
        else:
            j = BAS[torch.randint(0, N, (yigin,), generator=uret)]
            _o = None if ofset is None else ofset[j]
            k = m.kayip(X[j].to(aygit).long(), PAD,
                        None if _o is None else _o.to(aygit).long())
            k.backward()
        opt.step()

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
           surdur=None, t_len=None, atla=1, ofset=None, hiza=None):
    """ARKA PLANDA baslatir, HEMEN doner (kural 8).

    X       (N,T) PENCERE TABLOSU ya da (N,) tek uzun AKIS.  Akis
            verilirse t_len/atla ile pencere BURADA acilir (gorunum).
            CPU'da durur, yigin GPU'ya gider.
    PAD     kayiptan dusulecek jeton.  BIRIM akisinda dolgu YOK -> None.
    n       sozluk boyu
    olcut   olcut(m, "ezber"|"cikarim", tam=False) -> oran.
            TEK ANALIZ: soru soruldu, cevap dogru mu.
    hiza    pencere BASLANGIC konumlari (agirlik_16.cumle_basi).
            None -> her `atla` birimde bir, eski davranis.  Verilirse
            hicbir pencere cumle ortasindan baslamaz.
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
              surdur, t_len, atla, ofset, hiza)).start()
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
