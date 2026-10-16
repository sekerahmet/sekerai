"""RGN_colab.ipynb üretici (hücre içerikleri burada; not defteri bu dosyadan yeniden yazılır).

    python colab/make_notebook.py
"""
import json
from pathlib import Path

HEADER = """« YYG + KELİMEYE ÖZEL NÖRON · COLAB KOŞULARI »

**Model:** softmax + RoPE + SwiGLU transformer, yerel / genel katman düzeni (YYGYGYYG, pencere 128), isteğe bağlı dikkat kapısı;
her katmanda her kelimenin kendi nöronu (adres token kimliğinden belli, yalnız o kelimede çalışır ve güncellenir).
Kod: `rgn/` (sekerai deposu, kol `BRANCH`); deneyler `rgn/configs/`; veri Drive'dan; çıktılar `Drive/RGN/runs/<koşu>/` (yerelde `G:\\Drive'ım\\RGN\\runs`).
Eski RGN (çekim dikkati, rasyonel MLP) deneyleri git etiketi `arsiv/rgn`'de.

| hücre | iş |
|---|---|
| 0 HAZIRLIK | Drive, kod (GitHub'dan taze, `COMMIT` yazdırılır), bağımlılık; canlı iş varken yeniden kurmaz |
| 1 TEST | test takımı arka planda (CUDA: Flex pencere, özel nöron çekirdeği) |
| 2 KOŞU | deney kuyruğu arka planda (`R.kuyruk`); `stop_step` ile kesilir, aynı adla sürer |
| 3 NABIZ | çalışan işler: durum, adım, kayıp, kalan süre, son değerlendirme (beklemez) |
| 4 KIYAS | 3000. adım tablosu: kayıp, tabana fark, ms/adım |
| 5 OLGU TESTİ | çoktan seçmeli olgu testi (arka planda) |
| 6 SONUÇ | biten koşuların son değerlendirmesi |
| X DURDUR | çalışan işleri checkpoint kaydedip durdurur |
| Z GPU KAPAT | çalışan iş yoksa çalışma zamanını kapatır |

Bekleme döngülü (time.sleep) hücre yok: bir iş başka birini bekleyecekse arka plan sürecinde bekler.
"""

HAZIRLIK = """# 0 HAZIRLIK  |  CPU  |  tekrar: GÜVENLİ
# Çekirdek düştüyse ÖNCE bu hücre.  Kod git'ten (önce commit + push), veri Drive'dan.
REPO, BRANCH = 'https://github.com/sekerahmet/sekerai.git', 'temizlik/yyg-ozel'
import os, sys, importlib
if 'runner' in sys.modules:                                   # canlı iş varken kod değiştirilmez
    _R = sys.modules['runner']
    _live = _R.canli() if hasattr(_R, 'canli') else [n for n, j in _R._jobs().items() if not n.startswith('_') and _R._alive(j['pid'])]
    assert not _live, 'ÇALIŞAN İŞ VAR: %s -- HAZIRLIK gerekmiyor, 3 NABIZ' % _live
from google.colab import drive
drive.mount('/content/drive')
%cd /content
!rm -rf /content/_repo && git clone -q --depth 1 --branch {BRANCH} {REPO} /content/_repo
!python /content/_repo/rgn/colab/setup_colab.py --git {REPO} --branch {BRANCH} --subdir rgn
for _m in [m for m in sys.modules if m in ('rgn', 'runner') or m.startswith('rgn.')]:   # eski kod bellekte kalmasın
    del sys.modules[_m]
sys.path.insert(0, '/content/RGN/colab'); import runner as R
%cd /content/RGN
COMMIT = !git -C /content/_repo log --oneline -1
print('COMMIT', COMMIT[0])"""

TEST = """# 1 TEST  |  GPU  —  test takımı arka planda (sonuç: R.nabiz('testler', satir=30))
R.arkaplan('testler', 'python tests/run_tests.py')"""

KOSU = """# 2 KOŞU  |  GPU  —  deney kuyruğu arka planda; 3000. adımda kesilir; aynı adla tekrar = sürdürür
_B = 'configs/fineweb/fw_s000_d512_tf_yg_ozel_rms.json train.stop_step=3000 '
R.kuyruk([_B + 'model.d_ff=640 model.n_layers=12 model.global_layers=4 name=fw_s000_e1_L12',
          _B + 'model.d_ff=640 model.n_layers=16 model.global_layers=5 name=fw_s000_e1_L16',
          _B + 'model.d_ff=1024 model.d_model=768 model.n_heads=12 model.n_kv_heads=12 name=fw_s000_e1_d768'],
         name='olcek', resume=True)"""

NABIZ = """# 3 NABIZ  |  CPU  —  çalışan işler (beklemez); başka bir iş için: R.nabiz('ad', satir=30)
R.nabiz()"""

KIYAS = """# 4 KIYAS  |  CPU  —  3000. adım tablosu (kayıp, tabana fark, ms/adım)
import json
from rgn.paths import runs_dir
K = [('taban d512 L8', 'fw_s000_d512_tf_yg'), ('özel d512 L8', 'fw_s000_d512_tf_yg_ozel_rms_v2'),
     ('E1 d512 L8', 'fw_s000_d512_ozel_ff640'), ('E1 d512 L12', 'fw_s000_e1_L12'), ('E1 d512 L16', 'fw_s000_e1_L16'),
     ('E1 d768 L8', 'fw_s000_e1_d768'), ('özel d768 L8', 'fw_s000_d768_ozel')]

def _recs(run):
    p = runs_dir() / run / 'metrics.jsonl'
    out = []
    for l in (open(p, encoding='utf-8') if p.exists() else []):
        try: out.append(json.loads(l))
        except Exception: pass
    return out
ev = lambda r, s: next((x['loss'] for x in r if x.get('type') == 'eval' and x.get('step') == s), None)
t = ev(_recs(K[0][1]), 3000)
print('%-14s %9s %9s %9s %10s %8s' % ('model', '1000', '2000', '3000', 'taban farkı', 'ms/adım'))
for ad, run in K:
    r = _recs(run)
    v = [ev(r, s) for s in (1000, 2000, 3000)]
    ms = sorted(x['ms'] for x in r if x.get('type') == 'train' and 100 < x.get('step', 0) <= 3000)
    print('%-14s %9s %9s %9s %10s %8s' % (ad, *['%.4f' % x if x else '—' for x in v],
          '%+.4f' % (v[2] - t) if v[2] else '—', '%.0f' % ms[len(ms) // 2] if ms else '—'))"""

OLGU = """# 5 OLGU TESTİ  |  GPU (arka planda; sonuç: R.nabiz('olgu', satir=20))  —  scripts/fact_probe.py
KOSULAR = 'fw_s000_d512_tf_yg fw_s000_d512_ozel_ff640_epok fw_s000_d512_tf_yg_ozel_rms fw_s000_d512_ozel_3epok'
R.arkaplan('olgu', 'python scripts/fact_probe.py --runs ' + KOSULAR)"""

SONUC = """# 6 SONUÇ  |  CPU  —  biten koşuların son değerlendirmesi ve hızı
R.sonuc(metin=0)"""

DURDUR = """# X DURDUR  |  CPU  —  çalışan bütün işleri checkpoint kaydedip durdurur (aynı hücreyle sürdürülür)
for _n in R.canli():
    R.kapat(_n)"""

GPU = """# Z GPU KAPAT  |  CPU  —  çalışan iş yoksa çalışma zamanını kapatır
assert not R.canli(), 'çalışan iş var: %s' % R.canli()
from google.colab import runtime; runtime.unassign()"""

CELLS = [("markdown", HEADER), ("code", HAZIRLIK), ("code", TEST), ("code", KOSU), ("code", NABIZ), ("code", KIYAS),
         ("code", OLGU), ("code", SONUC), ("code", DURDUR), ("code", GPU)]


def main():
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
          "cells": []}
    for i, (kind, src) in enumerate(CELLS):
        cell = {"cell_type": kind, "id": "c%02d" % i, "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb["cells"].append(cell)
    out = Path(__file__).with_name("RGN_colab.ipynb")
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("yazıldı:", out, len(CELLS), "hücre")


if __name__ == "__main__":
    main()
