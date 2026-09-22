"""KORPUSU DOSYAYA YAZ.  Colab bunu Drive'dan OKUR, uretmez (kural 9).

Uretim 166 sn suruyor ve her kosuda yeniden uretilirse korpus boluculerin
en ufak degisikliginde SESSIZCE kayar -- iki kosu kiyaslanamaz hale gelir.
Dosyanin izi sabittir ve Colab kapida YENIDEN HESAPLAYIP karsilastirir.

GRAF dosyaya girmiyor: `veri_16.kur(0)` bir saniyede kuruluyor ve KENDI
iz kapisi var (3cd9a2575e47).  Dosyaya konan sey PAHALI olan ve kaymamasi
gereken sey: X.

  python hazirla_16.py --yaz

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
  hazirla_16  veriyi dosyaya yazar, Colab Drive'dan OKUR   <-- BU DOSYA

  model_16    MIMARI -- model_15'ten
  kos_16      egitim dongusu
  olcme_16    olcu: soru soruldu, cevap dogru mu
"""
import hashlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ayar_16 as A          # noqa: E402
import jeton_16 as J         # noqa: E402
import taban_16 as T         # noqa: E402
import veri_16 as V          # noqa: E402

CIKTI = "korpus_16.pt"

# Bolmeler.  Ayrim TEK SORUDA: cevap korpusta YAZIYOR MU?
BOLME = ("ezber_olgu", "ezber_zincir",          # yaziyor  -> EZBER
         "cikarim_gorulmemis", "cikarim_yabanci",  # yazmiyor -> CIKARIM
         "kapi_kisayolsuz",                     # KAPI, yetenek olcmez
         "ayrik_arama", "ayrik_ood", "ayrik_kati")  # HUKUM VERMEZ


def kur(yaz=print):
    v = T.veri_kur(A.AYAR, yaz=lambda *a: None)
    G = V.kur(A.AYAR.veri_tohum)               # kendi iz kapisi var
    X, S = T.egitim_havuzu(A.AYAR, v, yaz=lambda *a: None)

    assert S.vocab < 128, f"sozluk {S.vocab} -- int8'e sigmaz"
    assert X.min() >= 0 and X.max() < 128
    X8 = X.astype(np.int8)                      # 64 jeton; int64 israf
    assert (X8.astype(np.int64) == X).all(), "int8 cevrimi KAYIPLI"

    E = [a for t in V.TIPLER for a in G["ad"][t]]
    d = dict(
        X=torch.from_numpy(X8),                 # (N, t_len) int8
        t_len=A.AYAR.t_len,
        harf=S.harf, vocab=S.vocab, PAD=J.PAD, EOS=J.EOS,
        harf_izi=S.iz, korpus_izi=S.korpus_izi,
        varlik=E, iliski=list(V.ILISKI),
        tip=v.tip, tip_ad=v.tip_ad, n_ent=v.n_ent, n_rel=v.n_rel,
        bolme={k: list(getattr(v, k)) for k in BOLME},
        ayar={k: getattr(A.AYAR, k) for k in A.SABIT},
        graf_izi=V.IZ,
    )
    d["iz"] = iz(d)
    return d


def iz(d):
    """X ve BOLMELER uzerinden.  Bolmeler de ize girer: X ayni kalip
    bolme kayarsa olcum sessizce baska bir sinav olur."""
    h = hashlib.sha256()
    h.update(d["X"].numpy().tobytes())
    for k in BOLME:
        h.update(repr(d["bolme"][k]).encode())
    return h.hexdigest()[:16]


if __name__ == "__main__":
    d = kur()
    if "--yaz" not in sys.argv:
        print("KURU CALISMA -- yazmak icin --yaz")
    else:
        torch.save(d, CIKTI)
        print(f"{CIKTI} yazildi   {os.path.getsize(CIKTI)/1e6:.1f} MB")
    X = d["X"]
    print(f"  X {tuple(X.shape)} {X.dtype}   {X.numel()/1e6:.1f} milyon jeton"
          f"   PAD %{100*float((X == d['PAD']).float().mean()):.1f}")
    print(f"  sozluk {d['vocab']}   harf izi {d['harf_izi']}"
          f"   korpus izi {d['korpus_izi']}   graf izi {d['graf_izi']}")
    print(f"  varlik {len(d['varlik'])}   iliski {len(d['iliski'])}")
    for k in BOLME:
        print(f"    {k:<20s} {len(d['bolme'][k]):7d}")
    print(f"  IZ {d['iz']}")
