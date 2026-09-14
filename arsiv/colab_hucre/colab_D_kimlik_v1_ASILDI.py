# ============================================================================
# DENEY 5 — KIMLIK DENETIMI (kol D)
# Yeni Colab defterine bu hucreyi yapistir ve calistir.
# Onceden kayit: DENEY5_ONKAYIT_KIMLIK.md
#
# D, kol A ile MIMARI OLARAK OZDES (ayni VOCAB, ayni parametre sayisi).
# Tek fark: egitim havuzuna "[Q1] e IDENT ? -> e" ornekleri ekleniyor.
# ============================================================================
import subprocess, sys, os, torch
assert torch.cuda.is_available(), "GPU YOK -> Runtime > Change runtime type > T4 GPU"
print("GPU:", torch.cuda.get_device_name(0))

from google.colab import drive
drive.mount('/content/drive')

H = "/content/drive/MyDrive/deney5_kimlik"          # C'nin klasorune DOKUNMAZ
os.makedirs(H, exist_ok=True)

subprocess.run(["bash", "-lc", f"""
pkill -f '[s]ifirdan.py' ; pkill -f '[y]edek.sh' ; sleep 2
rm -rf /content/_d /content/outD /content/logD.txt
mkdir -p /content/outD
cp -ru {H}/. /content/outD/ 2>/dev/null
git clone -q https://github.com/sekerahmet/sekerai.git /content/_d
"""])
assert os.path.isdir("/content/_d"), "klon basarisiz"
print("kod :", subprocess.run(["bash", "-lc", "cd /content/_d && git log --oneline -1"],
                              capture_output=True, text=True).stdout.strip())

# --- yarida kalmis kosu varsa DEVAM ET ---
surdur = "/content/outD/surdur_A_s0.pt"
resume = surdur if os.path.exists(surdur) else ""
if resume:
    print(">>> yarida kalmis kosu bulundu, DEVAM EDILIYOR:", resume)

env = dict(os.environ,
           PRESET="grok_uzun",      # 120.000 adim
           MEM_AT="4",              # bellek yok (kol A), deger onemsiz
           HOP2_FRAC="0.10",        # kol A ile AYNI veri
           IDENT_FRAC="0.15",       # <-- TEK FARK
           ARMS="A",                # belleksiz: A ile ozdes mimari
           SEEDS="0", OUT="/content/outD",
           RESUME_EVERY="1", RESUME_FROM=resume)
p = subprocess.Popen([sys.executable, "-u", "sifirdan.py"], cwd="/content/_d", env=env,
                     stdout=open("/content/logD.txt", "w"), stderr=subprocess.STDOUT)

open("/content/yedek.sh", "w").write(
    "while true; do\n"
    f"  cp -ru /content/outD/. {H}/ 2>/dev/null\n"
    f"  cp -f /content/logD.txt {H}/egitim_log_D.txt 2>/dev/null\n"
    "  touch /content/yedek_nabiz\n  sleep 300\ndone\n")
subprocess.Popen(["bash", "/content/yedek.sh"],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print(f"""
pid: {p.pid}   ~65 dk   yedek: {H} (5 dk)

BIRINCIL : ENT > 0.20          (kol A ayni veride: 0.029)
IKINCIL  : ENT2 > 0.10         (kol A: 0.049)
           comp >= 0.80        (kol A: 0.851)
SAGLIK   : 1hop >= 0.99        (kol A: 0.996)
           kimlik >= 0.95      <-- DUSUK KALIRSA SONUC GECERSIZ
                                   ("recete uygulanamadi", "ise yaramadi" degil)

kopma olursa: bu hucreyi tekrar calistir, surdurme paketinden devam eder.
""")


# ============================================================================
# LOG (ayri hucre):     print(open("/content/logD.txt").read())
#
# DURUM (ayri hucre):
#   import re
#   for l in open("/content/logD.txt"):
#       if re.match(r"^\s+\d+\s+loss", l): print(l.rstrip()[:150])
#
# BITINCE (ayri hucre):
#   !cd /content/_d && python analiz.py --out /content/outD
# ============================================================================
