# -*- coding: utf-8 -*-
"""kos_10 — BU KOLUN KENDI kosucusu. TEK BASINA DURUR.

Kullanici karari, 16 Eylul 2026: *"bunlarin hepsi model_09 folderi
altinda olmali. model_09 diger hicbir model ile ayni seyi
kullanmamali."*

`deneme2/kos.py`nin KOPYASI (uretici: elle, tek seferlik). Farklar:

    --model YOK       bu kosucu yalniz model_09'i baslatir; aile klasoru
                      ARANMAZ (paylasilan kos.py `deneme2/*/<ad>.py`
                      glob'u yapiyordu -- model_09 artik o aramaya
                      girmiyor bile)
    yol               yalniz KENDI klasoru sys.path'e girer

    python kos_10.py --ev <cikti koku> [--tohum 0] [--commit X]
                     [--ustune] [--adim N] [--surdur]
"""
from __future__ import annotations

import argparse, importlib, os, sys

KOK = os.path.dirname(os.path.abspath(__file__))       # deneme2/model_10/
MODEL = "model_10"


OKU = """# {model} — ham koşu çıktısı

Bu klasörü **kod yazdı**, elle düzenleme. Depoda değil, Drive'da; koşu
çıktısı ikili ve büyük (`.gitignore`: `*.pt *.npz *.parquet`).

```
{model}/
    OKU.md                        bu dosya (her koşuda yenilenir)
    log/
        kos_<zaman>.txt           KOŞUNUN logu (tohumun değil -- bir koşu
                                  birden çok tohum sürebilir)
    t0/   t1/   t2/               her TOHUM kendi klasöründe
        ayar_t0.json              NE İSTEDİK  + _olcme_izi
        kosu_t0.json              NE KOŞTU: commit, GPU, torch, durum, süre
        egri_model_a_t0.json      her ölçüm noktası (adım 0 dahil)
        pencere_model_a_t0.json   pencere_a'nın çıktısı (birincil okuma)
        snap/
            snap_model_a_t0_00002000.pt    ağırlıklar, fp16
```

## Önce nereye bakılır

1. `kosu_t<N>.json` → `durum`. `BITTI` değilse koşu **tamamlanmamıştır**;
   anlık görüntüler geçerli ama bütçe sorusu sorulamaz.
2. `egri_*.json` → eğri. **Ön okuma.**
3. `pencere_*.json` → **birincil okuma**: N anlık görüntünün ağırlık
   ortalaması alınıp tek model ölçülür. Eğri değerleriyle pencere
   değerleri **aynı şey değildir**.

## Tohumlar, ve tekrar koşmak

`t0`, `t1`, `t2` ayrı model değil, **aynı modelin tekrarı** — tek fark
başlangıç ağırlıkları ve batch sırası. Veri hepsinde aynı (`veri_tohum`
ayrı bir alan).

**Hiçbir koşu ezilmez:**

```
başka tohum          ayrı klasöre yazar, öncekine dokunmaz
aynı tohum           REDDEDİLİR -- "ZATEN DOLU"
aynı tohum --ustune  eskisi SİLİNMEZ, t<N>_eski_<zaman>/ diye TAŞINIR
```

`t<N>_eski_*` klasörleri eski koşulardır; `kosu_t<N>.json`'larındaki
`baslangic` ve `commit` hangisi olduğunu söyler.

Üreten kod: `deneme2/{model}/` — `kosu_t<N>.json` içindeki `commit`
hangi sürüm olduğunu söyler.
"""


def main():
    ap = argparse.ArgumentParser()
    # --model YOK: bu kosucu yalniz model_09'i baslatir.
    ap.set_defaults(model=MODEL)
    ap.add_argument("--ev", required=True, help="cikti koku; t<N>/ altina yazar")
    ap.add_argument("--tohum", type=int, nargs="+", default=[0])
    ap.add_argument("--commit", default=None,
                    help="defterin klonladigi commit; kunyeye yazilir")
    ap.add_argument("--ustune", action="store_true",
                    help="DOLU klasoru _eski_<zaman>'a TASI, yenisini bos ac")
    ap.add_argument("--adim", type=int, default=None,
                    help="butce tavanini degistir (CLAUDE.md kural 1: "
                         "uzatmak KULLANICI kararidir). ayar_t<N>.json'a yazilir.")
    ap.add_argument("--surdur", action="store_true",
                    help="surdurme_t<N>.pt'den KALDIGI YERDEN devam et. "
                         "--adim ile birlikte kullanilir.")
    a = ap.parse_args()

    if KOK not in sys.path:
        sys.path.insert(0, KOK)
    M = importlib.import_module(MODEL)
    # Cikti dosyalari `AYAR.ad` ile adlandirilir. Dosya adi ile AYAR.ad
    # ayrilirsa dosya adi YALAN SOYLER: model_b.py, snap_model_a_*.pt yazar.
    # Arsivde tam bu oldu -- butun kollar `snap_A_s0_*.pt` yaziyordu.
    assert M.AYAR.ad == a.model, (
        f"AD TUTMUYOR: dosya {a.model}.py ama AYAR.ad {M.AYAR.ad!r}. "
        "Kopyalanan bir model dosyasinda `ad` degistirilmemis olabilir "
        "(deneme2/ISIMLENDIRME.md).")

    print(f"kos.py   model {a.model}   ev {a.ev}   tohum {a.tohum}   "
          f"commit {a.commit or '(yerel)'}", flush=True)

    # !! `--hiz-dogrula` KALDIRILDI (kullanici karari, 18 Eylul).
    # O bayrak tek bir seyi siniyordu: egitim havuzunun GPU'da mi CPU'da
    # mi tutuldugunu. `GPU_HAVUZ` model_09'de YOK (taban_10.egit'teki
    # nota bak), dolayisiyla kiyaslanacak iki dal da yok.

    # Klasor KENDINI anlatsin: Drive'i uc ay sonra acan kisi (biz) hangi
    # dosyaya once bakacagini bilmeli. Her kosuda yenilenir, tohum
    # klasorlerine DOKUNMAZ.
    os.makedirs(a.ev, exist_ok=True)
    with open(os.path.join(a.ev, "OKU.md"), "w", encoding="utf-8") as f:
        f.write(OKU.format(model=a.model))
    for t in a.tohum:
        ayar = M.AYAR.degistir(tohum=t)
        if a.adim is not None:
            # Butce degisikligi bir IDDIA degil, CIKTI olsun: hem ekrana
            # basilir hem ayar_t<N>.json'a yazilir.
            M.fark_bas(M.AYAR, ayar.degistir(adim=a.adim))
            ayar = ayar.degistir(adim=a.adim)
        M.egit(ayar, alt=f"{a.ev}/t{t}", ustune=a.ustune,
               commit=a.commit, surdur=a.surdur)


if __name__ == "__main__":
    main()
