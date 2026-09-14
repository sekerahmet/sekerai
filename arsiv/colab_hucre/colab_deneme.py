# =================== COLAB DENEME HUCRESI ===================
# Kisa kosu (~8 dk). Tam kosudan once her seyin calistigini dogrular.
import os, subprocess, time, json, glob, math, sys

# ---- 0. Drive envanteri (once neyin kaldigini gor)
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception as e:
    print("Drive baglanmadi:", e)
print("\n=== DRIVE'DA NE VAR ===")
for d in sorted(glob.glob("/content/drive/MyDrive/sifirdan*")):
    f = os.listdir(d)
    mb = sum(os.path.getsize(os.path.join(d, x)) for x in f
             if os.path.isfile(os.path.join(d, x))) / 1e6
    print(f"  {d}  ->  {len(f)} dosya, {mb:.1f} MB")
    for x in sorted(f)[:8]:
        print(f"       {x}")

# ---- 1. Repo
os.chdir("/content")
subprocess.run(["bash", "-lc", "pkill -f sifirdan.py; rm -rf sekerai"])
subprocess.run(["git", "clone", "-q", "https://github.com/sekerahmet/sekerai.git"])
os.chdir("/content/sekerai")
print("\n=== KOD ===")
print(subprocess.run(["bash", "-lc", "git log --oneline -1"],
                     capture_output=True, text=True).stdout.strip())
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())

# ---- 2. KISA kosu: 3 kol, 2000 adim
env = dict(os.environ, PRESET="grok", SEEDS="0", ARMS="ABC",
           STEPS="2000", EVERY="1000", OUT="/content/deneme")
subprocess.run(["bash", "-lc", "rm -rf /content/deneme"])
print("\n=== DENEME KOSUSU (3 kol x 2000 adim) ===")
t0 = time.time()
p = subprocess.run(["python", "-u", "sifirdan.py"], env=env,
                   capture_output=True, text=True)
print(p.stdout[-3500:])
if p.returncode != 0:
    print("HATA:\n", p.stderr[-2500:])
print(f"sure: {time.time()-t0:.0f} sn")

# ---- 3. KENDI KENDINI DENETLE
print("\n" + "=" * 60)
print("DENETIM")
print("=" * 60)
ok = True
def chk(ad, kosul, detay=""):
    global ok
    ok &= bool(kosul)
    print(f"  [{'GECTI' if kosul else 'KALDI'}] {ad}  {detay}")

chk("torch.compile acik", "torch.compile acik" in p.stdout)
for a in "ABC":
    try:
        e = json.load(open(f"/content/deneme/egri_{a}_s0.json"))
    except Exception as ex:
        chk(f"kol {a} egrisi", False, str(ex)[:50]); continue
    c = e[-1]
    pr = c.get("kopru_profil", [])
    chk(f"kol {a} katman profili", len(pr) == 9, f"{len(pr)} katman (beklenen 9)")
    chk(f"kol {a} problar sayisal",
        all(isinstance(c.get(k), float) and math.isfinite(c[k])
            for k in ("kopru", "kopru_L0", "kopru_null", "cevap", "ho_kopru")))
    if a != "A":
        sc = c.get("mem_scale", float("nan"))
        chk(f"kol {a} bellek sicakligi cokmedi", sc > 8, f"scale={sc:.2f} (baslangic 16)")
        chk(f"kol {a} bellek secici", c.get("mem_H", 99) < 8.0,
            f"memH={c.get('mem_H', float('nan')):.2f} (maks {math.log(4096):.2f})")

f = os.listdir("/content/deneme") if os.path.exists("/content/deneme") else []
for tip, ad in (("snap_", "ara kontrol noktasi"), ("kayit_", "ornek tablosu"),
                ("kv_", "bellek K/V"), ("dikkat_", "dikkat haritasi"),
                ("kafa_", "kafa ablasyonu")):
    chk(ad, any(x.startswith(tip) for x in f),
        f"{sum(x.startswith(tip) for x in f)} dosya")

print("\n" + ("TUMU GECTI -> tam kosuya hazir" if ok else
              "BAZILARI KALDI -> once bunlari duzeltelim"))
