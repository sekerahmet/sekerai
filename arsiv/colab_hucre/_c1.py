# ============================================================================
# HUCRE 1 - KURULUM + UC DENETIM KAPISI
# ============================================================================
import subprocess, sys, os, hashlib, shutil, inspect, glob, re, json, time
import numpy as np, torch

assert torch.cuda.is_available(), "GPU YOK -> Runtime > Change runtime type > T4"
print("GPU:", torch.cuda.get_device_name(0))
from google.colab import drive
drive.mount('/content/drive')

DR  = "/content/drive/MyDrive"
EV  = f"{DR}/deney9_D3"        # bu deneyin kalici klasoru
OUT = "/content/out9"          # calisma
KOD = "/content/_d3"           # klon
REF = {"D":  (f"{DR}/deney7_maske/D",          (6, 7)),      # referans kollar
       "A":  (f"{DR}/deney4_tekrarli_erisim",  ())}
MASKE_D3 = (1, 2, 3, 4, 5, 6, 7)      # TEK YERDE tanimli, elle kopyalanmaz
os.makedirs(f"{EV}/kod", exist_ok=True); os.makedirs(OUT, exist_ok=True)

subprocess.run(["bash", "-lc", f"""
pkill -f '[k]os9.py' ; sleep 1
cp -ru {EV}/. {OUT}/ 2>/dev/null
rm -rf {KOD} ; git clone -q https://github.com/sekerahmet/sekerai.git {KOD}
"""])
print("klon:", subprocess.run(["bash", "-lc", f"cd {KOD} && git log --oneline -1"],
                              capture_output=True, text=True).stdout.strip())

# DEPODAKI SURUM YAMASIZ (65381 char). Yamali surum yalnizca Drive'da (67224).
# Kaynak SIRAYLA aranir; bulunamazsa acik mesajla durulur.
_adaylar = [f"{EV}/kod/sifirdan.py",
            f"{DR}/deney8_tohum1/kod/sifirdan.py",
            f"{DR}/deney7_maske/kod/sifirdan.py"]
src = next((p for p in _adaylar if os.path.exists(p)), None)
assert src, ("Yamali sifirdan.py yok. Depodaki surum YAMASIZ; sunlardan biri "
             "Drive'da bulunmali: " + " | ".join(_adaylar))
shutil.copy(src, f"{KOD}/sifirdan.py")
print("  sifirdan.py <-", src)
KOD_MD5 = hashlib.md5(open(f"{KOD}/sifirdan.py", "rb").read()).hexdigest()
print("  calisacak sifirdan.py md5 =", KOD_MD5[:10])
if src != f"{EV}/kod/sifirdan.py":                     # bu deneye kendi kopyasi
    shutil.copy(src, f"{EV}/kod/sifirdan.py.tmp")
    os.replace(f"{EV}/kod/sifirdan.py.tmp", f"{EV}/kod/sifirdan.py")
open(f"{EV}/kod_md5.txt", "w").write(KOD_MD5 + "  " + src)   # CLAUDE.md 9

# --- DENETIM 1: DOSYADA yama var mi --------------------------------------
_k = open(f"{KOD}/sifirdan.py", encoding="utf-8").read()
for g in ('MASK_KEY = os.environ.get("MASK_KEY"',
          'self.mask_key = None',
          'm[:, self.mask_key] = True'):
    assert g in _k, f"sifirdan.py YAMASIZ (eksik: {g[:34]}...)"
print("DENETIM 1 GECTI   dosyada maske yamasi var")

# --- DENETIM 2: DEFTERE IMPORT EDILEN modul de yamali mi? ----------------
# 13 Eylul arizasi tam buradaydi: egitim yamali klondan kosuyordu, defter
# BASKA bir klondan import ediyordu. blk.mask_key = 1 olu oznitelik yaratti,
# hata vermedi, maskeyle egitilmis modeli maskesiz olctu.
for _v in ("MASK_KEY", "MASK_BLK", "RESUME_FROM", "INIT_FROM", "OUT"):
    os.environ.pop(_v, None)                   # analiz modulu MASKESIZ import edilir
os.environ.update(PRESET="grok_uzun", ARMS="A", SEEDS="0",
                  HOP2_FRAC="0.10", MEM_AT="4")
sys.path.insert(0, KOD)
sys.modules.pop("sifirdan", None)
import sifirdan as S

print("  S.__file__ =", S.__file__)
assert os.path.realpath(S.__file__).startswith(os.path.realpath(KOD)), \
    f"YANLIS MODUL yuklendi: {S.__file__}"
assert "self.mask_key" in inspect.getsource(S.Block.forward), \
    "Yuklenen modulun Block.forward'i maskeyi OKUMUYOR"
print("DENETIM 2 GECTI   import edilen modul yamali")

# --- DENETIM 3: maske etkili mi + kapatinca BIT-AYNI mi? -----------------
_n = S.Net("A", S.CFG).to(S.DEV).eval()
_x = torch.randint(0, _n.emb.num_embeddings, (8, S.T_LEN), device=S.DEV)
with torch.no_grad():
    _a = _n(_x)[0].float().clone()
    for _b in _n.blocks: _b.mask_key = 1
    _m = _n(_x)[0].float().clone()
    for _b in _n.blocks: _b.mask_key = None
    _c = _n(_x)[0].float().clone()
assert torch.equal(_a, _c), "maske kapatilinca eski hale donmuyor (no-op DEGIL)"
assert float((_a - _m).abs().max()) > 1e-3, "MASKE ETKISIZ -> olcum bozuk olurdu"
print(f"DENETIM 3 GECTI   maske etkili (fark {float((_a-_m).abs().max()):.1f}), "
      f"kapatinca bit-ayni")
del _n, _x, _a, _m, _c

def snaplar(klasor, kol="A", tohum=0):
    """Anlik goruntuleri GLOB ile bul. Dosya adi bicimini VARSAYMA:
    snap_A_s0_070000.pt SIFIR DOLGULU; elle format kurmak hata kaynagi."""
    d = {}
    for p in glob.glob(f"{klasor}/snap_{kol}_s{tohum}_*.pt"):
        m = re.search(r"_(\d+)\.pt$", os.path.basename(p))
        if m: d[int(m.group(1))] = p
    return dict(sorted(d.items()))

TEMIZ = {k: os.environ[k] for k in
         ("PATH", "HOME", "LANG", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES")
         if k in os.environ}          # alt surece SIZINTISIZ ortam
print("\nKURULUM TAMAM")
