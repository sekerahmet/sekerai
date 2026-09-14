# ============================================================================
# DENEY 7 — D: kisayol yolunu EGITIM BOYUNCA kapat
# Onceden kayit: DENEY7_ONKAYIT_D_MASKE.md
#
#   D  MASK_KEY=1 MASK_BLK=6,7   blok 6-7'de hicbir sorgu poz1'e (varlik) bakamaz
#   K  MASK_KEY=2 MASK_BLK=6,7   KONTROL: poz2 (r1) kapali, maliyeti eslesmis
#   A  maske yok                 ZATEN VAR, yeniden kosulmaz
#
# ONCE Drive'a koy:  MyDrive/deney7_maske/kod/sifirdan.py   (yamali surum)
# Hucre imzalari dogrulamadan baslamaz.
# ============================================================================
import subprocess, sys, os, hashlib, shutil, torch
assert torch.cuda.is_available(), "GPU YOK -> Runtime > Change runtime type > T4"
print("GPU:", torch.cuda.get_device_name(0))

from google.colab import drive
drive.mount('/content/drive')

H   = "/content/drive/MyDrive/deney7_maske"      # deney4 klasorune DOKUNMAZ
OUT = "/content/out7"
os.makedirs(f"{H}/kod", exist_ok=True); os.makedirs(OUT, exist_ok=True)

subprocess.run(["bash", "-lc", f"""
pkill -f '[k]os7.py' ; pkill -f '[s]ifirdan.py' ; pkill -f '[y]edek.sh' ; sleep 2
cp -ru {H}/. {OUT}/ 2>/dev/null
rm -rf /content/_d ; git clone -q https://github.com/sekerahmet/sekerai.git /content/_d
"""])
print("klon:", subprocess.run(["bash","-lc","cd /content/_d && git log --oneline -1"],
                              capture_output=True, text=True).stdout.strip())

# --- Drive'daki yamali kod klonun UZERINE, sonra IMZA DOGRULA ---------------
src = f"{H}/kod/sifirdan.py"
if os.path.exists(src):
    shutil.copy(src, "/content/_d/sifirdan.py")
    print(f"  sifirdan.py <- Drive/kod  md5={hashlib.md5(open(src,'rb').read()).hexdigest()[:10]}")
kod = open("/content/_d/sifirdan.py", encoding="utf-8").read()
for g in ('MASK_KEY = os.environ.get("MASK_KEY"',
          'self.mask_key = None',
          'm[:, self.mask_key] = True'):
    assert g in kod, (f"sifirdan.py GUNCEL DEGIL (eksik: {g[:34]}...).\n"
                      f"Yamali dosyayi {src} icine koy ve hucreyi tekrar calistir.")
print("  sifirdan.py DOGRULANDI (MASK_KEY + Block.mask_key + maske kurulumu)")

# --- sirali surucu ----------------------------------------------------------
open("/content/kos7.py", "w").write(f'''
import os, sys, subprocess, time
ORTAK = dict(PRESET="grok_uzun", STEPS="120000", ARMS="A", SEEDS="0",
             HOP2_FRAC="0.10", MEM_AT="4", RESUME_EVERY="1")
KOSULAR = [("D", {{"MASK_KEY": "1", "MASK_BLK": "6,7"}}),   # mudahale
           ("K", {{"MASK_KEY": "2", "MASK_BLK": "6,7"}})]   # KONTROL
for ad, ek in KOSULAR:
    d = "{OUT}/" + ad
    os.makedirs(d, exist_ok=True)
    if os.path.exists(d + "/verdict.json"):
        print(f"[{{ad}}] zaten bitmis, atlandi", flush=True); continue
    sur = d + "/surdur_A_s0.pt"
    env = dict(os.environ, OUT=d, **ORTAK, **ek,
               RESUME_FROM=(sur if os.path.exists(sur) else ""))
    print(f"[{{ad}}] basliyor {{ek}}" + ("  [SURDURULUYOR]" if os.path.exists(sur) else ""),
          flush=True)
    t = time.time()
    r = subprocess.run([sys.executable, "-u", "sifirdan.py"], cwd="/content/_d", env=env,
                       stdout=open(f"/content/log_{{ad}}.txt", "a"), stderr=subprocess.STDOUT)
    print(f"[{{ad}}] bitti rc={{r.returncode}}  {{(time.time()-t)/60:.1f}} dk", flush=True)
    if r.returncode != 0:
        print(f"[{{ad}}] HATA -> log_{{ad}}.txt", flush=True); break
print("BITTI", flush=True)
''')
p = subprocess.Popen([sys.executable, "-u", "/content/kos7.py"],
                     stdout=open("/content/surucu7.txt", "w"), stderr=subprocess.STDOUT)

# yedek + NABIZ (ps|grep kendi komutunu yakalar -> nabiz dosyasinin YASINA bak)
open("/content/yedek.sh", "w").write(
    "while true; do\n"
    f"  cp -ru {OUT}/. {H}/ 2>/dev/null\n"
    f"  cp -f /content/log_*.txt /content/surucu7.txt {H}/ 2>/dev/null\n"
    "  touch /content/yedek_nabiz\n  sleep 300\ndone\n")
subprocess.Popen(["bash", "/content/yedek.sh"],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print(f"""
pid: {p.pid}   2 kol x 120.000 adim   ~110 dk   yedek: {H} (5 dk)

BIRINCIL (adim 120.000, tam kumeler):
  ENT(D) - ENT(A) >= +0.10      A = 0.029
  ENT(D) - ENT(K) >= +0.08      kontrol maskesi aciklamamali
OZGULLUK:
  ENT2(D) - ENT2(A) < +0.05     ikisi birden duzelirse SONUC GECERSIZ
SAGLIK:
  comp >= 0.80   |   1hop >= 0.98  (maskenin olculmus yapisal maliyeti var)

Kopma olursa bu hucreyi tekrar calistir; biten kol atlanir, yarim kalan surdurur.
""")

# ============================================================================
# DURUM (ayri hucre):
#   import os, re, time
#   print(open("/content/surucu7.txt").read())
#   print("nabiz:", round(time.time()-os.path.getmtime("/content/yedek_nabiz")), "sn once")
#   for ad in ("D","K"):
#       y = f"/content/log_{ad}.txt"
#       if os.path.exists(y):
#           s=[l.rstrip()[:150] for l in open(y) if re.match(r"^\s+\d+\s+loss", l)]
#           print(f"--- {ad} ---"); print("\n".join(s[-3:]))
# ============================================================================
