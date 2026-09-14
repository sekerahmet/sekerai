# =================== TAM KOSU ===================
# Arka planda baslatir, hemen doner. Drive'a her 5 dk yedekler.
# Drive'in ONCEDEN baglanmis olmasi gerekir (drive.mount hucre icinden).
import os, subprocess, time

os.chdir("/content")
subprocess.run(["bash", "-lc", "pkill -f sifirdan.py; pkill -f yedek.sh; rm -rf sekerai"])
subprocess.run(["git", "clone", "-q", "https://github.com/sekerahmet/sekerai.git"])
print("kod:", subprocess.run(["bash", "-lc", "cd /content/sekerai && git log --oneline -1"],
                             capture_output=True, text=True).stdout.strip())

OUT = "/content/out"
os.makedirs(OUT, exist_ok=True)
env = dict(os.environ, PRESET="grok", SEEDS="0", OUT=OUT)
p = subprocess.Popen(["python", "-u", "sifirdan.py"], cwd="/content/sekerai", env=env,
                     stdout=open("/content/log.txt", "w"), stderr=subprocess.STDOUT)
print("egitim baslatildi, pid", p.pid)

# ---- Drive yedegi: --delete YOK (once bir kez veri kaybettik), sadece ekler
H = "/content/drive/MyDrive/sifirdan_final"
if os.path.isdir("/content/drive/MyDrive"):
    os.makedirs(H, exist_ok=True)
    open("/content/yedek.sh", "w").write(
        "while true; do\n"
        f"  cp -ru {OUT}/. {H}/ 2>/dev/null\n"
        f"  cp -f /content/log.txt {H}/log.txt 2>/dev/null\n"
        "  sleep 300\n"
        "done\n")
    subprocess.Popen(["bash", "/content/yedek.sh"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("yedek ->", H, "(her 5 dk, --delete YOK)")
else:
    print("UYARI: Drive bagli degil, yedek YOK. Once drive.mount calistir.")

print("\nIlerleme icin:  !tail -20 /content/log.txt")
print("Beklenen sure : ~1.4 saat (3 kol x 60000 adim, torch.compile acik)")
