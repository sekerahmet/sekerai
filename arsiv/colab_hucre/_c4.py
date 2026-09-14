# ============================================================================
# HUCRE 4 - D3'U BASLAT   (~85 dk, tek kol)
# Kopma olursa bu hucreyi tekrar calistir: biten atlanir, yarim kalan surdurur.
# ============================================================================
ORTAK = dict(PRESET="grok_uzun", STEPS="120000", ARMS="A", SEEDS="0",
             HOP2_FRAC="0.10", MEM_AT="4", RESUME_EVERY="1")
KOSULAR = [("D3", {"MASK_KEY": "1",
                   "MASK_BLK": ",".join(str(b) for b in MASKE_D3)})]
print("kosulacak:", KOSULAR)

open("/content/kos9.py", "w").write(f"""
import os, sys, subprocess, time
TEMIZ = {TEMIZ!r}
for ad, ek in {KOSULAR!r}:
    d = "{OUT}/" + ad
    os.makedirs(d, exist_ok=True)
    if os.path.exists(d + "/verdict.json"):
        print(f"[{{ad}}] zaten bitmis, atlandi", flush=True); continue
    sur = d + "/surdur_A_s0.pt"
    env = dict(TEMIZ, OUT=d, **{ORTAK!r}, **ek,
               RESUME_FROM=(sur if os.path.exists(sur) else ""))
    print(f"[{{ad}}] basliyor {{ek}}" + ("  [SURDURULUYOR]" if os.path.exists(sur) else ""),
          flush=True)
    t = time.time()
    r = subprocess.run([sys.executable, "-u", "sifirdan.py"], cwd="{KOD}", env=env,
                       stdout=open(f"/content/log_{{ad}}.txt", "a"), stderr=subprocess.STDOUT)
    print(f"[{{ad}}] bitti rc={{r.returncode}}  {{(time.time()-t)/60:.1f}} dk", flush=True)
    if r.returncode != 0:
        print(f"[{{ad}}] HATA -> log_{{ad}}.txt", flush=True); break
print("BITTI", flush=True)
""")
p = subprocess.Popen([sys.executable, "-u", "/content/kos9.py"],
                     stdout=open("/content/surucu9.txt", "w"), stderr=subprocess.STDOUT)

# Yedekleme. surdur_*.pt ~102 MB ve HER OLCUMDE yeniden yazilir: duz cp
# ortasinda runtime olurse Drive'da YARIM dosya kalir, /content ise silinmistir.
# CLAUDE.md 7 -> buyuk dosya .tmp + mv ile. Nabiz dosyasi; ps|grep KULLANMA.
_yed = """while true; do
  ( cd __OUT__ && find . -type f ! -name 'surdur_*.pt' -print0 |
      xargs -0 -I@ cp -u --parents @ __EV__/ ) 2>/dev/null
  for f in __OUT__/*/surdur_*.pt; do
    [ -e "$f" ] || continue
    rel="${f#__OUT__/}"
    mkdir -p "__EV__/$(dirname "$rel")"
    cp -f "$f" "__EV__/$rel.tmp" && mv -f "__EV__/$rel.tmp" "__EV__/$rel"
  done
  cp -f /content/log_*.txt /content/surucu9.txt __EV__/ 2>/dev/null
  touch /content/yedek_nabiz9
  sleep 300
done
"""
open("/content/yedek9.sh", "w").write(
    _yed.replace("__OUT__", OUT).replace("__EV__", EV))
subprocess.Popen(["bash", "/content/yedek9.sh"],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(f"pid {p.pid}   ~85 dk   yedek: {EV} (5 dk)")
