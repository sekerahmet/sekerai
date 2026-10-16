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
| 1 TEST | GPU kapısı; test takımı arka planda (CUDA: Flex pencere, özel nöron çekirdeği) |
| 2 KOŞU | GPU kapısı; deney listesi sırayla arka planda (`R.kuyruk`); `stop_step` ile kesilir, aynı adla sürer |
| 3 NABIZ | işler: durum, adım, kayıp, kalan süre, son değerlendirme |
| 4 KIYAS | taban ve deneyler yan yana, adım adım (parantezde tabana fark) + ms/adım |
| 5 PROFİL | tam eğitim adımının bölüm bölüm süresi (arka planda; koşu sürerken çalıştırma) |
| 6 DÖNGÜ | greedy üretimde tekrar testi (arka planda) |
| X DURDUR | SIGTERM: checkpoint kaydedip çıkar; 2 KOŞU aynı listeyle kaldığı yerden sürer |
| Y DRIVE'DAKİ KAYIT | koşu klasörleri ve dosyaları |
| Z GPU | nvidia-smi; iş yoksa çalışma zamanını kapat |
"""

HAZIRLIK = """# 0 HAZIRLIK  |  CPU  |  tekrar: GÜVENLİ
# Çekirdek düştüyse ÖNCE bu hücre.  Kod git'ten (önce commit + push), veri Drive'dan.
REPO, BRANCH = 'https://github.com/sekerahmet/sekerai.git', 'temizlik/yyg-ozel'
import os, sys, importlib
if 'runner' in sys.modules:                                   # canlı iş varken kod değiştirilmez
    _R = sys.modules['runner']
    _live = [n for n, j in _R._jobs().items() if not n.startswith('_') and _R._alive(j['pid'])]
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

TEST = """# 1 TEST  |  GPU  |  tekrar: GÜVENLİ  (arka planda; sonuç: R.nabiz('testler', satir=30))
import torch; assert torch.cuda.is_available(), 'GPU yok: Çalışma zamanı > Çalışma zamanı türünü değiştir'
R.arkaplan('testler', 'python tests/run_tests.py')"""

KOSU = """# 2 KOŞU  |  GPU  |  tekrar: SÜRDÜRÜR (aynı ad --resume ile kaldığı yerden; temiz başlatmak için klasörü sil)
# FineWeb-Edu s000 tam epok (~11.250 adım): taban YYG ve kelimeye özel nöron (momentsiz satır optimizer'ı).
import torch; assert torch.cuda.is_available(), 'GPU yok: Çalışma zamanı > Çalışma zamanı türünü değiştir'
KOSULAR = [
    'configs/fineweb/fw_s000_d512_tf_yg.json',
    'configs/fineweb/fw_s000_d512_tf_yg_ozel_rms.json',
]
_free = torch.cuda.mem_get_info()[0] / 2**30; print('boş GPU belleği %.1f GB' % _free); assert _free > 30, 'GPU dolu'
R.kuyruk(KOSULAR, name='deney', resume=True)"""

NABIZ = """# 3 NABIZ  |  CPU  |  tekrar: GÜVENLİ
R.nabiz('deney')"""

KIYAS = """# 4 KIYAS  |  CPU  |  tekrar: GÜVENLİ  —  kayıp (parantezde tabana fark) + hız; SS ve FineWeb ayrı tablolar
import json
from rgn.paths import runs_dir

def _recs(run):
    out, p = [], runs_dir() / run / 'metrics.jsonl'
    if p.exists():
        for l in open(p, encoding='utf-8'):
            try: out.append(json.loads(l))
            except Exception: pass
    return out

def tablo(baslik, K, adimlar):
    R_ = {ad: _recs(run) for ad, run in K}
    ev = lambda r, s: next((x['loss'] for x in r if x.get('type') == 'eval' and (x.get('step') == s or (s == 'SON' and x.get('final')))), None)
    base = K[0][0]
    print(baslik)
    print('%-9s' % 'adım' + ''.join('%22s' % ad for ad, _ in K))
    for s in adimlar:
        t = ev(R_[base], s)
        row = []
        for ad, _ in K:
            v = ev(R_[ad], s)
            row.append('—' if v is None else '%.4f' % v if (ad == base or t is None) else '%.4f (%+.4f)' % (v, v - t))
        print('%-9s' % s + ''.join('%22s' % c for c in row))
    for ad, _ in K:
        ms = sorted(x['ms'] for x in R_[ad] if x.get('type') == 'train' and 'ms' in x and x.get('step', 0) > 100)
        print('   %-18s ms/adım ~%s' % (ad, '%.0f' % ms[len(ms) // 2] if ms else '-'))
    print()

tablo('SS (tam epok programı)', [('taban YYG', 'ss_d512_tf_yg_epok'), ('kelimeye özel', 'ss_d512_tf_yg_ozel')],
      (1000, 2000, 3000, 6000, 9000, 'SON'))
tablo('FineWeb-Edu s000', [('FW taban YYG', 'fw_s000_d512_tf_yg'), ('FW özel momentsiz', 'fw_s000_d512_tf_yg_ozel_rms')],
      (1000, 2000, 3000, 6000, 9000, 'SON'))"""

PROFIL = """# 5 PROFİL  |  GPU  |  tekrar: GÜVENLİ  (arka planda; koşu sürerken çalıştırma — ölçüm bozulur; sonuç: R.nabiz('profil', satir=90))
# Önce model testleri (CUDA çekirdek doğruluğu dahil), sonra tam eğitim adımı bölüm bölüm.
R.arkaplan('profil', 'python tests/run_tests.py model && '
           'python scripts/profile_step.py --config configs/fineweb/fw_s000_d512_tf_yg_ozel_rms.json')"""

DONGU = """# 6 DÖNGÜ  |  GPU  |  tekrar: GÜVENLİ  (arka planda; sonuç: R.nabiz('dongu', satir=40))
# Greedy üretimde tekrar: SS hikâyelerinin ilk 32 token'ı, 384 token devam (yalnız bitmiş koşular: runs/<koşu>/model.pt).
R.arkaplan('dongu', 'python scripts/loop_test.py --runs ss_d512_tf_yg_epok ss_d512_tf_yg_ozel --stories 256 --prompt 32 --new 384')"""

DURDUR = """# X DURDUR  |  CPU  |  tekrar: GÜVENLİ
# Çalışan kuyruk checkpoint kaydedip çıkar.  Sürdürmek: 2 KOŞU (aynı ad, --resume).
R.kapat('deney')"""

DRIVE = """# Y DRIVE'DAKİ KAYIT  |  CPU  |  tekrar: GÜVENLİ
from rgn.paths import runs_dir
for _d in sorted(p for p in runs_dir().iterdir() if p.is_dir()):
    _fs = sorted(f.name for f in _d.iterdir())
    print('%-34s %s' % (_d.name, ' '.join(_fs)))"""

GPU = """# Z GPU  |  CPU  |  tekrar: GÜVENLİ  —  iş yoksa çalışma zamanını kapatır
!nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
_live = [n for n, j in R._jobs().items() if not n.startswith('_') and R._alive(j['pid'])]
print('çalışan iş:', _live or 'yok')
assert not _live, 'çalışan iş var, kapatılmadı'
from google.colab import runtime; runtime.unassign()"""

CELLS = [("markdown", HEADER), ("code", HAZIRLIK), ("code", TEST), ("code", KOSU), ("code", NABIZ), ("code", KIYAS),
         ("code", PROFIL), ("code", DONGU), ("code", DURDUR), ("code", DRIVE), ("code", GPU)]


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
