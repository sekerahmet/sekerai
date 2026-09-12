# sekerai — sifirdan uc kol

Bellegin **adresi** nereden gelmeli: girdi token'larindan mi, gizli durumdan mi?

Sifirdan egitilen 3 transformer, esit parametre / esit veri / esit seed:

| kol | bellek | sorgu kaynagi |
|---|---|---|
| A | yok (FFN genisletilip parametre esitlenir) | — |
| B | var | katman 0 (token-adresli) |
| C | var | katman L/2 (latent-adresli) |

**B ile C arasindaki tek fark bellek modulunun okudugu tensordur.** Ayni
parametre sayisi, ayni sekiller, ayni enjeksiyon noktasi.

Karar kurali kod calismadan once `sifirdan.py` docstring'inde sabitlenmistir.

## Colab

```python
!git clone -q https://github.com/sekerahmet/sekerai.git
%cd sekerai
!PRESET=small SEEDS=0 python -u sifirdan.py
```

Cikti: `sifirdan_out/verdict.json` ve `sifirdan_out/log.txt`.

## Ayarlar

Ortam degiskeniyle her sey ezilebilir:

- `PRESET` = `smoke` | `small` | `full`
- `SEEDS` = `0,1,2` (>=3 seed olmadan sonuc "ON BULGU"dur)
- `ARMS` = `ABC` (alt kume calistirmak icin, or. `BC`)
- `STEPS`, `BATCH`, `LR`, `D`, `L`, `M`, ... — CFG'deki her anahtar
