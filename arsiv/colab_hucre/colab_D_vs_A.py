# ============================================================================
# DENEY 6 — D'nin A ile kiyasi   (onceden kayit: DENEY6_ONKAYIT_D_vs_A.md)
#
# YENI Colab defterine bu hucreyi yapistir. C'nin defterine DOKUNMA.
#
# Uc kol da A'nin EGITILMIS agirliklarindan basliyor (A yeniden egitilmiyor):
#     artiA : havuz A ile ayni              <- KONTROL, atlanamaz
#     D2    : [Q2] e r1 KENDISI ? -> kopru  <- BIRINCIL hipotez
#
# Her kol 30.000 adim (~16 dk), toplam ~33 dk. Sirali kosar, kopmada devam eder.
# ============================================================================
import subprocess, sys, os, torch
assert torch.cuda.is_available(), "GPU YOK -> Runtime > Change runtime type > T4 GPU"
print("GPU:", torch.cuda.get_device_name(0))

from google.colab import drive
drive.mount('/content/drive')

KAYNAK = "/content/drive/MyDrive/deney4_tekrarli_erisim/model_A_s0.pt"
H      = "/content/drive/MyDrive/deney6_D_vs_A"      # C'nin klasorune DOKUNMAZ
assert os.path.exists(KAYNAK), f"A'nin modeli yok: {KAYNAK}"
os.makedirs(H, exist_ok=True)

# --- kaynagin GERCEKTEN aradigimiz A oldugunu dogrula (koru korune baslama) --
_c = torch.load(KAYNAK, map_location="cpu", weights_only=False)
print(f"kaynak: kol={_c['arm']} seed={_c['seed']} adim={_c['cfg']['STEPS']} "
      f"HOP2_FRAC={_c['cfg'].get('HOP2_FRAC')} "
      f"IDENT_FRAC={_c['cfg'].get('IDENT_FRAC')} "
      f"MEM_AT={_c['cfg'].get('MEM_AT')} commit={str(_c['cfg'].get('_commit'))[:8]}")
assert _c["arm"] == "A" and float(_c["cfg"].get("HOP2_FRAC", -1)) == 0.10, \
    "kaynak model beklenen A kolu degil"
assert float(_c["cfg"].get("IDENT_FRAC", 0)) == 0.0, "kaynak zaten kimlik gormus"
del _c

subprocess.run(["bash", "-lc", f"""
pkill -f '[k]osD.py' ; pkill -f '[s]ifirdan.py' ; pkill -f '[y]edek.sh' ; sleep 2
rm -rf /content/_d ; mkdir -p /content/out
cp -ru {H}/. /content/out/ 2>/dev/null
git clone -q https://github.com/sekerahmet/sekerai.git /content/_d
"""])
assert os.path.isdir("/content/_d"), "klon basarisiz"
print("klon:", subprocess.run(["bash", "-lc", "cd /content/_d && git log --oneline -1"],
                              capture_output=True, text=True).stdout.strip())

# --- Drive'daki kod klonun UZERINE yazilir --------------------------------
# INIT_FROM ve IDENT_MODE GitHub'a itilmediyse klon ESKI kodu getirir ve
# kosu sessizce yanlis seyi olcer. O yuzden once Drive'a bak, sonra DOGRULA.
# {H}/kod/ icine sifirdan.py ve kiyas_D.py koy.
import hashlib, shutil
for _f in ("sifirdan.py", "kiyas_D.py"):
    _s, _h = f"{H}/kod/{_f}", f"/content/_d/{_f}"
    _k = "klon"
    if os.path.exists(_s):
        shutil.copy(_s, _h); _k = "Drive/kod"
    if os.path.exists(_h):
        _m = hashlib.md5(open(_h, "rb").read()).hexdigest()[:10]
        print(f"  {_f:14s} <- {_k:10s} md5={_m}")
    else:
        print(f"  {_f:14s} YOK")

_kod = open("/content/_d/sifirdan.py", encoding="utf-8").read()
for _gerek in ('INIT_FROM    = os.environ.get("INIT_FROM"',
               'IDENT_MODE = os.environ.get("IDENT_MODE"',
               'def enc_ident_q2son('):
    assert _gerek in _kod, (
        f"sifirdan.py bu deney icin GUNCEL DEGIL (eksik: {_gerek[:34]}...).\n"
        f"Guncel dosyayi {H}/kod/sifirdan.py icine koy ve hucreyi tekrar calistir.")
print(f"  sifirdan.py DOGRULANDI (INIT_FROM + IDENT_MODE + q2son mevcut)")

# --- sirali surucu: bitmis kolu atlar, yarim kalani surdurur -----------------
open("/content/kosD.py", "w").write(f'''
import os, sys, subprocess, json, time
KAYNAK = {KAYNAK!r}
ORTAK = dict(PRESET="grok_uzun", STEPS="30000", EVERY="2000",
             ARMS="A", SEEDS="0", HOP2_FRAC="0.10", MEM_AT="4",
             INIT_FROM=KAYNAK, RESUME_EVERY="1")
# SIRA onemli: kontrol once, sonra birincil hipotez, sonra ikincil.
# D1 (q1) KESILDI (13 Eylul, kullanici karari): bagli gomme yuzunden
# zaten dogru olan bir seyi ogretiyor ve kopyalamayla cozulebiliyor.
KOSULAR = [("artiA", {{}}),
           ("D2",    {{"IDENT_FRAC": "0.15", "IDENT_MODE": "q2son"}})]
for ad, ek in KOSULAR:
    d = f"/content/out/{{ad}}"
    os.makedirs(d, exist_ok=True)
    if os.path.exists(f"{{d}}/verdict.json"):
        print(f"[{{ad}}] zaten bitmis, atlandi", flush=True); continue
    sur = f"{{d}}/surdur_A_s0.pt"
    env = dict(os.environ, OUT=d, **ORTAK, **ek,
               RESUME_FROM=(sur if os.path.exists(sur) else ""))
    print(f"[{{ad}}] basliyor  {{ek or 'kimlik YOK (kontrol)'}}"
          + ("  [SURDURULUYOR]" if os.path.exists(sur) else ""), flush=True)
    t = time.time()
    r = subprocess.run([sys.executable, "-u", "sifirdan.py"], cwd="/content/_d",
                       env=env, stdout=open(f"/content/log_{{ad}}.txt", "a"),
                       stderr=subprocess.STDOUT)
    print(f"[{{ad}}] bitti rc={{r.returncode}}  {{(time.time()-t)/60:.1f}} dk", flush=True)
    if r.returncode != 0:
        print(f"[{{ad}}] HATA -> log_{{ad}}.txt", flush=True); break
print("TUM KOLLAR BITTI", flush=True)
''')

p = subprocess.Popen([sys.executable, "-u", "/content/kosD.py"],
                     stdout=open("/content/surucu.txt", "w"),
                     stderr=subprocess.STDOUT)

# yedek + NABIZ dosyasi. (ps|grep kendini yakaliyor -> nabiz dosyasinin
#  yasina bak, surec listesine DEGIL. Bu hatayi iki kez yaptik.)
open("/content/yedek.sh", "w").write(
    "while true; do\n"
    f"  cp -ru /content/out/. {H}/ 2>/dev/null\n"
    f"  cp -f /content/log_*.txt /content/surucu.txt {H}/ 2>/dev/null\n"
    "  touch /content/yedek_nabiz\n  sleep 300\ndone\n")
subprocess.Popen(["bash", "/content/yedek.sh"],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print(f"""
pid: {p.pid}   ~33 dk (2 x 30.000 adim)   yedek: {H} (5 dk)

BIRINCIL (her D kolu icin):  ENT(D) - ENT(artiA) >= +0.10  VE  p < 0.01
SAGLIK  :  1hop >= 0.99 | comp >= 0.80 | ident >= 0.95 | ident_ent >= 0.95
           ident_ent DUSUKSE sonuc GECERSIZ ("recete uygulanamadi")

kopma olursa: bu hucreyi tekrar calistir. Biten kol atlanir, yarim kalan surdurur.
""")


# ============================================================================
# DURUM (ayri hucre):
#   import os, time, re
#   print(open("/content/surucu.txt").read())
#   print("nabiz:", round(time.time()-os.path.getmtime("/content/yedek_nabiz")), "sn once")
#   for ad in ("artiA","D2"):
#       y = f"/content/log_{ad}.txt"
#       if os.path.exists(y):
#           s = [l.rstrip()[:150] for l in open(y) if re.match(r"^\s+\d+\s+loss", l)]
#           print(f"--- {ad} ---"); print("\n".join(s[-3:]))
#
# BITINCE (ayri hucre) — onceden kayitli kiyas:
#   !cd /content/_d && python kiyas_D.py \
#       --artiA /content/out/artiA --d2 /content/out/D2 \
#       --json /content/out/kiyas.json
# ============================================================================
