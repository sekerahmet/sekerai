# -*- coding: utf-8 -*-
"""kos_06 — model_05'in KENDI kosucusu. TEK BASINA DURUR.

Kullanici karari, 16 Eylul 2026: *"bunlarin hepsi model_05 folderi
altinda olmali. model_05 diger hicbir model ile ayni seyi
kullanmamali."*

`deneme2/kos.py`nin KOPYASI (uretici: elle, tek seferlik). Farklar:

    --model YOK       bu kosucu yalniz model_05'i baslatir; aile klasoru
                      ARANMAZ (paylasilan kos.py `deneme2/*/<ad>.py`
                      glob'u yapiyordu -- model_05 artik o aramaya
                      girmiyor bile)
    yol               yalniz KENDI klasoru sys.path'e girer

    python kos_05.py --ev <cikti koku> [--tohum 0] [--commit X]
                     [--ustune] [--adim N] [--surdur]
"""
from __future__ import annotations

import argparse, glob, importlib, os, sys, time

KOK = os.path.dirname(os.path.abspath(__file__))       # deneme2/model_05/
MODEL = "model_06"


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
    # --model YOK: bu kosucu yalniz model_05'i baslatir.
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
    ap.add_argument("--hiz-dogrula", type=int, default=0, metavar="N",
                    dest="hiz_dogrula",
                    help="HIZ OPTIMIZASYONU YORUNGEYI DEGISTIRIYOR MU? "
                         "N adim iki kez kosar (havuz GPU'da / CPU'da) ve "
                         "agirliklari BIT DUZEYINDE kiyaslar. Egitim YAPMAZ.")
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

    # --- HIZ DOGRULAMASI ------------------------------------------------
    # "Hizlandirdim" demek yetmez; YORUNGE AYNI MI, o kanitlanir.
    # Bu projede daha once ayni sey surdurme icin yapildi: 8+8 surdurulmus
    # kosu ile kesintisiz 16 adimlik kosu BIT DUZEYINDE ayni cikti,
    # agirlik farki 0.000e+00 (CLAUDE.md kural 1).
    #
    # Batch indisleri `rs.randint` ile NUMPY tarafinda uretiliyor; havuzun
    # GPU'da olmasi hangi satirlarin secildigini DEGISTIRMEZ. Beklenen
    # sonuc TAM SIFIR fark.
    if a.hiz_dogrula:
        import tempfile, torch
        _sonuc = {}
        for _ad, _bayrak in (("GPU havuzu", "1"), ("CPU havuzu", "0")):
            os.environ["GPU_HAVUZ"] = _bayrak
            importlib.reload(M.M)          # taban_06'yi tazele
            importlib.reload(M)
            _ayar = M.AYAR.degistir(tohum=a.tohum[0], adim=a.hiz_dogrula,
                                    olc_her=a.hiz_dogrula)
            with tempfile.TemporaryDirectory() as _d:
                t0 = time.time()
                M.egit(_ayar, alt=_d, yaz=lambda *x: None, commit="hiz")
                _sn = time.time() - t0
                _sd = torch.load(sorted(glob.glob(f"{_d}/snap/*.pt"))[-1],
                                 map_location="cpu")
            _sonuc[_ad] = (_sd, _sn)
            print(f"  {_ad:<12} {a.hiz_dogrula} adim  {_sn:.1f} sn"
                  f"  ({_sn/a.hiz_dogrula*1000:.1f} ms/adim)", flush=True)
        (g, gs), (c, cs) = _sonuc["GPU havuzu"], _sonuc["CPU havuzu"]
        _fark = max(float((g[k].float() - c[k].float()).abs().max())
                    for k in g if k in c)
        print(f"\n  EN BUYUK AGIRLIK FARKI: {_fark:.3e}")
        print("  " + ("YORUNGE BIREBIR AYNI -- optimizasyon GUVENLI"
                      if _fark == 0.0 else
                      "!! YORUNGE DEGISTI -- optimizasyon KULLANILMAZ"))
        print(f"  HIZLANMA: {cs/gs:.2f}x  ({cs:.1f} sn -> {gs:.1f} sn)")
        os.environ.pop("GPU_HAVUZ", None)
        return 0

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
