# S HIZ OLCUMU -- sayisallik ve derleyici secenekleri  |  GPU  |  tekrar: GUVENLI (egitim baslatmaz, Drive'a yazmaz)
# Once TinyStories defterinin 0 HAZIRLIK hucresi (SRC, DATA).  Sure: ~15 dk (her secenek sabit 45 adim; compile'lar
# derleme suresi ekler).  Kural 0: kullanici onayiyla.  Kural 8: GROUPS'tan bir kismi secilip hucre bolunebilir.
import os, sys, time
import torch

# --- GPU KAPISI (CLAUDE.md kural 2); esik hesaptan: logits fp32 64 x 511 x 8.004, ileri + geri ~4 kopya (olculmedi)
assert torch.cuda.is_available(), 'GPU YOK -- Runtime > Change runtime type'
_bos = torch.cuda.mem_get_info()[0] / 1e9
_need = max(2.0, 4 * 64 * 512 * 8004 * 4 / 1e9)
assert _bos > _need, "GPU'da sadece %.1f GB bos, gereken ~%.1f GB" % (_bos, _need)
print('GPU kapisi GECTI: %s  bos %.1f GB' % (torch.cuda.get_device_name(0), _bos))

# Kosan egitim varken olculmez: sure paylasilir, iki is de bozulur
_live = [n for m in ('colab_tinystories', 'colab_kinship', 'colab_math') if m in sys.modules
         for n, r in sys.modules[m].RUNS.items() if r['thread'].is_alive()]
assert not _live, 'EGITIM KOSUYOR: %s -- olcum egitim bitince' % _live

sys.path.insert(0, SRC + '/speed')
for _m in ('bench_speed', 'model_20', 'train_20'):
    sys.modules.pop(_m, None)
import train_20 as TR
assert hasattr(TR, 'numerics'), 'kod eski: speed dali main\'e alinmadi'
import bench_speed as BS

GROUPS = ['environment', 'compile_repro', 'tinystories_synthetic', 'tinystories_real', 'math']
# ileri sapma egitilmis agirlikla: kosunun son surdurme paketi (yalniz okunur)
_run_dir = ROOT + '/tinystories_modelx_s0'
_packs = sorted(f for f in os.listdir(_run_dir) if f.startswith('checkpoint_t')) if os.path.isdir(_run_dir) else []
CHECKPOINT = _run_dir + '/' + _packs[-1] if _packs else None
print('ileri sapma agirligi:', CHECKPOINT or 'ortak baslangic (paket yok)')
_t = time.time()
if 'environment' in GROUPS:
    BS.environment()
if 'compile_repro' in GROUPS:
    BS.compile_repro()
if 'tinystories_synthetic' in GROUPS:
    BS.tinystories(checkpoint=CHECKPOINT)
if 'tinystories_real' in GROUPS and 'DATA' in globals():
    BS.tinystories(data=DATA, checkpoint=CHECKPOINT)
if 'math' in GROUPS:
    BS.math()
print('BITTI  %.0f sn' % (time.time() - _t))
