"""RGN_colab.ipynb üretici (hücre içerikleri burada; not defteri bu dosyadan yeniden yazılır).

    python colab/make_notebook.py
"""
import json
from pathlib import Path

HEADER = """« RGN — Rasyonel Çekim Ağı · COLAB KOŞULARI »

**Soru: softmax'sız çekim dikkati ve cebirsel MLP ile kurulan decoder, aynı bütçeyle transformer'a ne kadar yaklaşıyor; transformer'ın q / k / v'sine eklenen önem / odak / kapı ne getiriyor?**
Kod: `rgn/` (sekerai deposu, kol `BRANCH`); deneyler `rgn/configs/`; veri Drive'dan; çıktılar `Drive/RGN/runs/<koşu>/` (yerelde `G:\\Drive'ım\\RGN\\runs`).
Tasarım dokümanı ve günlük özetler yerelde (`rgn/docs/`, git dışı).

| hücre | iş |
|---|---|
| 0 HAZIRLIK | Drive, kod (GitHub'dan taze, `COMMIT` yazdırılır), bağımlılık; canlı iş varken yeniden kurmaz |
| 1 TEST | GPU kapısı; test takımı arka planda (CUDA: Flash-RGN = dense, pencere, tamsayı k) |
| 2 KOŞU | GPU kapısı; deney listesi sırayla arka planda (`R.kuyruk`); program tam epok, `stop_step` ile kesilir |
| 3 NABIZ | işler: durum, adım, kayıp, kalan süre, son değerlendirme |
| 4 SONUÇ | kıyas tabloları: 1000 / 2000 / 3000. adım ve tam epok, transformer'a fark |
| 4b KIYAS | transformer ve q/k/v koşuları yan yana, adım adım |
| 5 HIZ | tam eğitim adımı hızı (arka planda; koşu sürerken çalıştırma) |
| 6 DÖNGÜ | greedy üretimde tekrar testi (arka planda) |
| X DURDUR | SIGTERM: checkpoint kaydedip çıkar; 2 KOŞU aynı listeyle kaldığı yerden sürer |
| Y DRIVE'DAKİ KAYIT | koşu klasörleri ve dosyaları |
| Z GPU | nvidia-smi; iş yoksa çalışma zamanını kapat |
"""

HAZIRLIK = """# 0 HAZIRLIK  |  CPU  |  tekrar: GÜVENLİ
# Çekirdek düştüyse ÖNCE bu hücre.  Kod git'ten (önce commit + push), veri Drive'dan.
REPO, BRANCH = 'https://github.com/sekerahmet/sekerai.git', 'deneme/rgn-asama0'
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
# Transformer + q/k/v düzeyi eklemeler (kullanıcı: "softmax kalsın, q k v değişsin").  Program birebir tam epok
# (9241 adım, iniş 7392'de), 3000. adımda kesilir.  Kıyas (tam epok koşularından): transformer 1000: 1,9076 |
# 2000: 1,7927 | 3000: 1,7409;  RGN (öğrenilen k + GLU) 3000: 1,7508.
import torch; assert torch.cuda.is_available(), 'GPU yok'
_free = torch.cuda.mem_get_info()[0] / 2**30; print('boş GPU belleği %.1f GB' % _free); assert _free > 30, 'GPU dolu'
R.kuyruk([
    'configs/ss/ss_d512_tf_onem.json train.stop_step=3000',    # önem (kütle): anahtar başına taban puan
    'configs/ss/ss_d512_tf_odak.json train.stop_step=3000',    # odak: sorgu başına keskinlik
    'configs/ss/ss_d512_tf_kapi.json train.stop_step=3000',    # kapı: head başına çıktı kapısı
    'configs/ss/ss_d512_tf_hepsi.json train.stop_step=3000',   # üçü birlikte
], name='ss_qkv', resume=False)"""

NABIZ = """# 3 NABIZ  |  CPU  |  tekrar: GÜVENLİ
R.nabiz('ss_qkv')"""

SONUC = """# 4 SONUÇ  |  CPU  |  tekrar: GÜVENLİ   —  KIYAS TABLOLARI (Drive'daki bütün koşulardan)
import json
from rgn.paths import runs_dir

def _recs(d):
    out, p = [], d / 'metrics.jsonl'
    if p.exists():
        for l in open(p, encoding='utf-8'):
            try: out.append(json.loads(l))
            except Exception: pass
    return out

def _desc(d):
    try: m = json.load(open(d / 'config.json', encoding='utf-8')).get('model', {})
    except Exception: m = {}
    if m.get('attn') == 'softmax':
        a = 'softmax' + ''.join(s for s, on in ((' +önem', m.get('attn_importance')), (' +odak', m.get('attn_focus')),
                                                  (' +kapı', m.get('attn_gate'))) if on)
    else:
        k = ('k≤%g' % m.get('sharpness_range', [1, 8])[1]) if m.get('sharpness_learn') else 'k=%s' % m.get('sharpness', 1)
        a = 'çekim ' + k + (' tam' if m.get('sharpness_int') else '')
    if m.get('local_window'):
        a += ' YG%d' % m['local_window']
    return a, m.get('mlp', '?')

runs = {d.name: _recs(d) for d in sorted(runs_dir().iterdir()) if d.is_dir() and (d / 'metrics.jsonl').exists()}
ev = lambda r, s: next((x for x in r if x.get('type') == 'eval' and x.get('step') == s), None)
TF = runs.get('ss_d512_transformer', [])

def tablo(adim, baslik):
    rows = []
    for name, r in runs.items():
        e = ev(r, adim)
        if not e: continue
        tr = sorted(x['ms'] for x in r if x.get('type') == 'train' and 'ms' in x and x.get('step', 0) > 100)
        rows.append((e['loss'], name, e, tr[len(tr) // 2] if tr else float('nan')))
    if not rows: return
    t = ev(TF, adim)
    print(baslik)
    print('%-34s %-28s %-8s %8s %8s %9s %8s' % ('koşu', 'dikkat', 'mlp', 'kayıp', 'bpb', 'Δ tf', 'ms/adım'))
    for loss, name, e, ms in sorted(rows):
        a, mlp = _desc(runs_dir() / name)
        print('%-34s %-28s %-8s %8.4f %8.4f %+9.4f %8.0f' % (name[:34], a[:28], mlp, loss, e['bpb'],
              loss - t['loss'] if t else float('nan'), ms))
    print()

tablo(1000, 'TABLO 1 — 1000. adım (SS, pencere 512, aynı veri sırası, tam epok programı)')
tablo(2000, 'TABLO 2 — 2000. adım')
tablo(3000, 'TABLO 3 — 3000. adım')
print('TABLO 4 — tam epok koşuları: transformer ile adım adım')
for name in [n for n in runs if n.endswith('_epok')]:
    r = runs[name]
    print('%-8s %12s %12s %9s     (%s)' % ('adım', 'transformer', 'koşu', 'Δ', name))
    for e in [x for x in r if x.get('type') == 'eval']:
        t = next((x for x in TF if x.get('type') == 'eval' and (x['step'] == e['step'] or (e.get('final') and x.get('final')))), None)
        print('%-8s %12s %12.4f %9s' % (('SON ' if e.get('final') else '') + str(e['step']), '%.4f' % t['loss'] if t else '-',
              e['loss'], '%+.4f' % (e['loss'] - t['loss']) if t else '-'))"""

KIYAS = """# 4b KIYAS  |  CPU  |  tekrar: GÜVENLİ  —  transformer ve q/k/v koşuları yan yana (kayıp; parantezde transformer'a fark)
import json
from rgn.paths import runs_dir
K = [('transformer', 'ss_d512_transformer'), ('önem', 'ss_d512_tf_onem'), ('odak', 'ss_d512_tf_odak'),
     ('kapı', 'ss_d512_tf_kapi'), ('hepsi', 'ss_d512_tf_hepsi')]

def _recs(run):
    out, p = [], runs_dir() / run / 'metrics.jsonl'
    if p.exists():
        for l in open(p, encoding='utf-8'):
            try: out.append(json.loads(l))
            except Exception: pass
    return out

R_ = {ad: _recs(run) for ad, run in K}
ev = lambda r, s: next((x['loss'] for x in r if x.get('type') == 'eval' and x.get('step') == s), None)
print('%-9s' % 'adım' + ''.join('%19s' % ad for ad, _ in K))
for s in (1000, 2000, 3000):
    t = ev(R_['transformer'], s)
    row = []
    for ad, _ in K:
        v = ev(R_[ad], s)
        row.append('—' if v is None else '%.4f' % v if (ad == 'transformer' or t is None) else '%.4f (%+.4f)' % (v, v - t))
    print('%-9s' % s + ''.join('%19s' % c for c in row))
ms = []
for ad, _ in K:
    tr = sorted(x['ms'] for x in R_[ad] if x.get('type') == 'train' and 'ms' in x and 100 < x.get('step', 0) <= 3000)
    ms.append('%.0f' % tr[len(tr) // 2] if tr else '—')
print('%-9s' % 'ms/adım' + ''.join('%19s' % m for m in ms))
print('\\n(transformer ms/adım kendi koşusundan: o koşuda kayıp checkpoint açıktı; hız testinde 112 ms)')"""

HIZ = """# 5 HIZ  |  GPU  |  tekrar: GÜVENLİ  (arka planda; koşu sürerken çalıştırma — ölçüm bozulur; sonuç: R.nabiz('hiz', satir=20))
# Tam eğitim adımı 128 × 512 = 65.536 token (ileri + geri + optimizer), derlenmiş.
R.arkaplan('hiz', 'python scripts/bench_speed.py model --seq 512 --batch 128 --configs '
           'configs/ss/ss_d512_transformer.json configs/ss/ss_d512_tf_hepsi.json configs/ss/ss_d512_rgn_kint_glu_yg.json')"""

DONGU = """# 6 DÖNGÜ  |  GPU  |  tekrar: GÜVENLİ  (arka planda; sonuç: R.nabiz('dongu', satir=40))
# Greedy üretimde tekrar: SS hikâyelerinin ilk 32 token'ı, 384 token devam.  Model ağırlıkları runs/<koşu>/model.pt
# (yalnız bitmiş koşularda).  Önceki sonuçlar: runs/_loop_test_384.json, _loop_test_1024.json.
R.arkaplan('dongu', 'python scripts/loop_test.py --runs ss_d512_transformer ss_d512_rgn_klearn_glu_epok --stories 256 --prompt 32 --new 384')"""

DURDUR = """# X DURDUR  |  CPU  |  tekrar: GÜVENLİ
# Çalışan kuyruk checkpoint kaydedip çıkar.  Sürdürmek: 2 KOŞU (aynı ad, --resume).
R.kapat('ss_qkv')"""

DRIVE = """# Y DRIVE'DAKİ KAYIT  |  CPU  |  tekrar: GÜVENLİ
from rgn.paths import runs_dir
for _d in sorted(p for p in runs_dir().iterdir() if p.is_dir()):
    _fs = sorted(f.name for f in _d.iterdir())
    print('%-34s %s' % (_d.name, ' '.join(_fs)))"""

GPU = """# Z GPU  |  CPU  |  tekrar: GÜVENLİ
!nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
_live = [n for n, j in R._jobs().items() if not n.startswith('_') and R._alive(j['pid'])]
print('çalışan iş:', _live or 'yok')
# İş yoksa çalışma zamanını kapat (aşağıdaki satırı aç):
# assert not _live; from google.colab import runtime; runtime.unassign()"""

CELLS = [("markdown", HEADER), ("code", HAZIRLIK), ("code", TEST), ("code", KOSU), ("code", NABIZ), ("code", SONUC), ("code", KIYAS),
         ("code", HIZ), ("code", DONGU), ("code", DURDUR), ("code", DRIVE), ("code", GPU)]


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
