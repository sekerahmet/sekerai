# ============================================================================
# KOL C — KOPMADAN SONRA DEVAM   (eski baslatma hucresini KULLANMA)
#
# Eski hucrede iki tuzak vardi:
#   1) rm -f ... surdur_C_s0.pt   -> surdurme paketini SILIYOR
#   2) RESUME_FROM yok            -> adim 1'den basliyor
# Bu hucre once DURUMU YAZAR, sonra paketten devam eder. Hicbir sey silmez.
# ============================================================================
import subprocess, sys, os, torch, time
assert torch.cuda.is_available(), "GPU YOK -> Runtime > Change runtime type > T4 GPU"
print("GPU:", torch.cuda.get_device_name(0))

from google.colab import drive
drive.mount('/content/drive')

H   = "/content/drive/MyDrive/deney4_tekrarli_erisim"
OUT = "/content/out4"
assert os.path.isdir(H), "Drive klasoru yok: " + H

# --- 1) ONCE DURUM: Drive'da ne var? (hicbir sey silmeden) ------------------
os.makedirs(OUT, exist_ok=True)
subprocess.run(["bash", "-lc", f"cp -ru {H}/. {OUT}/ 2>/dev/null"])

SUR = f"{OUT}/surdur_C_s0.pt"
print("\n--- Drive'daki C dosyalari ---")
for f in sorted(os.listdir(OUT)):
    if "_C_" in f or f.startswith("surdur_C"):
        print(f"  {f:34s} {os.path.getsize(os.path.join(OUT,f))/1e6:8.1f} MB")

if not os.path.exists(SUR):
    print("\n*** SURDURME PAKETI YOK. Devam edilemez. ***")
    print("    Yedek dongusu Drive'a yazmadan mi koptu, yoksa hic yazilmadi mi?")
    print("    egri_C_s0.json varsa kac adima kadar gittigi oradan gorulur.")
    raise SystemExit

_ck = torch.load(SUR, map_location="cpu", weights_only=False)
ADIM = int(_ck["step"]); HEDEF = int(_ck["cfg"]["STEPS"])
assert _ck["arm"] == "C" and _ck["seed"] == 0, f"paket baska kola ait: {_ck['arm']}"
print(f"\n  SURDURME PAKETI: kol={_ck['arm']} seed={_ck['seed']}")
print(f"  adim {ADIM} / {HEDEF}   (%{100*ADIM/HEDEF:.1f})   kalan {HEDEF-ADIM} adim")
print(f"  MEM_AT={_ck['cfg'].get('MEM_AT')} HOP2_FRAC={_ck['cfg'].get('HOP2_FRAC')}")
print(f"  paket yasi: {(time.time()-os.path.getmtime(SUR))/60:.0f} dk once yazilmis")
del _ck

# --- 2) kod (SILMEDEN) ------------------------------------------------------
subprocess.run(["bash", "-lc", """
pkill -f '[s]ifirdan.py' ; pkill -f '[y]edek.sh' ; sleep 2
rm -rf /content/_s ; git clone -q https://github.com/sekerahmet/sekerai.git /content/_s
"""])
print("\nkod :", subprocess.run(["bash", "-lc", "cd /content/_s && git log --oneline -1"],
                                capture_output=True, text=True).stdout.strip())

# --- 3) DEVAM ---------------------------------------------------------------
env = dict(os.environ, PRESET="grok_uzun", MEM_AT="2,4,6", HOP2_FRAC="0.10",
           ARMS="C", SEEDS="0", OUT=OUT, RESUME_EVERY="1", RESUME_FROM=SUR)
p = subprocess.Popen([sys.executable, "-u", "sifirdan.py"], cwd="/content/_s", env=env,
                     stdout=open("/content/log4.txt", "a"), stderr=subprocess.STDOUT)

# yedek + NABIZ. (ps|grep kendi komut satirini yakalar -> nabiz dosyasinin
#  YASINA bak, surec listesine degil. Bu hatayi iki kez yaptik.)
open("/content/yedek.sh", "w").write(
    "while true; do\n"
    f"  cp -ru {OUT}/. {H}/ 2>/dev/null\n"
    f"  cp -f /content/log4.txt {H}/egitim_log_C.txt 2>/dev/null\n"
    "  touch /content/yedek_nabiz\n  sleep 300\ndone\n")
subprocess.Popen(["bash", "/content/yedek.sh"],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print(f"""
pid: {p.pid}   adim {ADIM} -> {HEDEF}   (~{(HEDEF-ADIM)/1500:.0f} dk)
yedek: {H} (5 dk'da bir)

Tekrar koparsa BU hucreyi yeniden calistir; hep son paketten devam eder.
""")

# ============================================================================
# DURUM (ayri hucre):
#   import os, time, re
#   print("nabiz:", round(time.time()-os.path.getmtime("/content/yedek_nabiz")), "sn once")
#   s = [l.rstrip()[:150] for l in open("/content/log4.txt")
#        if re.match(r"^\s+\d+\s+loss", l)]
#   print("\n".join(s[-4:]))
# ============================================================================
