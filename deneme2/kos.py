# -*- coding: utf-8 -*-
"""kos.py — defterin BASLATTIGI surec.  Modele ait DEGIL, altyapi.

    python kos.py --model model_a --ev <cikti koku> --tohum 0 [--commit X]
                  [--ustune]

Cikti:  <ev>/t<tohum>/     her tohum KENDI klasorune yazar.

--------------------------------------------------------------------------
NEDEN AYRI DOSYA

Eskiden defter bu betigi KENDI ICINDE metin olarak kuruyordu:

    kos = "\\n".join(['import sys; sys.path.insert(0, %r)' % ...,
                      'import %s as M' % MODEL, ...])
    open("/content/kos.py", "w").write(kos)

Yani kosuyu fiilen baslatan kod DEPODA DEGILDI. Gozden gecirilemiyordu,
test edilemiyordu, commit'e girmiyordu ve `%`/tirnak kacislari sessizce
bozulabilirdi. Simdi klonla birlikte geliyor: neyi kosturdugun, kosan
kodun kendisiyle AYNI commit'te.
"""
from __future__ import annotations

import argparse, glob, importlib, os, sys

KOK = os.path.dirname(os.path.abspath(__file__))       # deneme2/


def aile_yolu(model: str) -> str:
    """`<model>.py` hangi AILE klasorunde?  deneme2/model_a/model_a1.py gibi.

    Klasor adini isimden TURETMIYORUZ (model_a1 -> model_a gibi bir kural
    sessizce yanlis klasoru secebilirdi). Dosyayi ARIYORUZ ve tam bir tane
    bulmasini SART kosuyoruz."""
    aday = glob.glob(os.path.join(KOK, "*", f"{model}.py"))
    assert len(aday) == 1, (
        f"{model}.py deneme2/*/ altinda TAM BIR KEZ bulunmali, "
        f"{len(aday)} bulundu: {aday}")
    return os.path.dirname(aday[0])


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

## Tohumlar

`t0`, `t1`, `t2` ayrı model değil, **aynı modelin tekrarı** — tek fark
başlangıç ağırlıkları ve batch sırası. Veri hepsinde aynı (`veri_tohum`
ayrı bir alan). Dolu bir tohum klasörüne tekrar koşmak **reddedilir**.

Üreten kod: `deneme2/{model}/` — `kosu_t<N>.json` içindeki `commit`
hangi sürüm olduğunu söyler.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="orn. model_a")
    ap.add_argument("--ev", required=True, help="cikti koku; t<N>/ altina yazar")
    ap.add_argument("--tohum", type=int, nargs="+", default=[0])
    ap.add_argument("--commit", default=None,
                    help="defterin klonladigi commit; kunyeye yazilir")
    ap.add_argument("--ustune", action="store_true",
                    help="DOLU klasorun uzerine yaz (varsayilan: REDDET)")
    a = ap.parse_args()

    sys.path.insert(0, aile_yolu(a.model))
    M = importlib.import_module(a.model)
    # Cikti dosyalari `AYAR.ad` ile adlandirilir. Dosya adi ile AYAR.ad
    # ayrilirsa dosya adi YALAN SOYLER: model_b.py, snap_model_a_*.pt yazar.
    # Arsivde tam bu oldu -- butun kollar `snap_A_s0_*.pt` yaziyordu.
    assert M.AYAR.ad == a.model, (
        f"AD TUTMUYOR: dosya {a.model}.py ama AYAR.ad {M.AYAR.ad!r}. "
        "Kopyalanan bir model dosyasinda `ad` degistirilmemis olabilir "
        "(deneme2/ISIMLENDIRME.md).")

    print(f"kos.py   model {a.model}   ev {a.ev}   tohum {a.tohum}   "
          f"commit {a.commit or '(yerel)'}", flush=True)

    # Klasor KENDINI anlatsin: Drive'i uc ay sonra acan kisi (biz) hangi
    # dosyaya once bakacagini bilmeli. Her kosuda yenilenir, tohum
    # klasorlerine DOKUNMAZ.
    os.makedirs(a.ev, exist_ok=True)
    with open(os.path.join(a.ev, "OKU.md"), "w", encoding="utf-8") as f:
        f.write(OKU.format(model=a.model))
    for t in a.tohum:
        M.egit(M.AYAR.degistir(tohum=t), alt=f"{a.ev}/t{t}",
               ustune=a.ustune, commit=a.commit)


if __name__ == "__main__":
    main()
