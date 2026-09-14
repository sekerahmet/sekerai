# ============================================================================
# HUCRE 7 - KURTARMA BETIGI -> Drive  (defter kaybolursa TEK hucre yeter)
# ============================================================================
_mb = ",".join(str(b) for b in MASKE_D3)
open(f"{EV}/kod/kurtar.py", "w").write(f'''
# !python /content/drive/MyDrive/deney9_D3/kod/kurtar.py
import os, sys, subprocess, shutil
EV, OUT, KOD = "{EV}", "{OUT}", "{KOD}"
os.makedirs(OUT, exist_ok=True)
subprocess.run(["bash","-lc", "cp -ru " + EV + "/. " + OUT + "/ 2>/dev/null; "
                "rm -rf " + KOD + "; git clone -q "
                "https://github.com/sekerahmet/sekerai.git " + KOD])
src = EV + "/kod/sifirdan.py"
if os.path.exists(src): shutil.copy(src, KOD + "/sifirdan.py")
k = open(KOD + "/sifirdan.py", encoding="utf-8").read()
for g in ('MASK_KEY = os.environ.get("MASK_KEY"', "self.mask_key = None",
          "m[:, self.mask_key] = True"):
    assert g in k, "sifirdan.py YAMASIZ: " + g[:30]
print("yama dogrulandi")
d = OUT + "/D3"; os.makedirs(d, exist_ok=True)
if os.path.exists(d + "/verdict.json"):
    print("D3 zaten bitmis"); sys.exit()
sur = d + "/surdur_A_s0.pt"
env = dict(os.environ, OUT=d, PRESET="grok_uzun", STEPS="120000", ARMS="A",
           SEEDS="0", HOP2_FRAC="0.10", MEM_AT="4", RESUME_EVERY="1",
           MASK_KEY="1", MASK_BLK="{_mb}",
           RESUME_FROM=(sur if os.path.exists(sur) else ""))
print("basliyor" + ("  [SURDURULUYOR]" if os.path.exists(sur) else ""))
subprocess.run([sys.executable, "-u", "sifirdan.py"], cwd=KOD, env=env)
''')
print("kurtar.py yazildi, maske =", _mb)
