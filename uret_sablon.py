# -*- coding: utf-8 -*-
"""SABLON_deney.ipynb uretici. Defter elle duzenlenmez, buradan uretilir.

    python uret_sablon.py <cikti.ipynb> <deney>

Deneye ozel HER SEY asagidaki DENEYLER sozlugunde. Defterin icinde elle
duzenlenecek alan YOKTUR -- sablon onceki deneyin ayarlarini tasiyordu ve
duzeltilmeden kosulursa sessizce ONCEKI deneyi baslatiyordu."""
import json, io, sys, textwrap, pprint

DENEYLER = {
    "d31": dict(
        ONKAYIT="ONKAYIT_D31_KENDI_BULSUN.md",
        IMZA={"sifirdan.py": [
            'WARM_OF      = int(os.environ.get("WARM_OF"',
            'self.mask_key = None',
            'm[:, self.mask_key] = True']},
        ORT=dict(PRESET="grok_uzun", HOP2_FRAC="0.10", MEM_AT="4",
                 RESUME_EVERY="1", WARM_OF="120000"),
    ),
    "d33": dict(
        ONKAYIT="ONKAYIT_D33_PHI_EKSENI.md",
        # IMZA = YAPISAL capa, MESAJ METNI DEGIL. 13 Eylul: kosu.py'deki
        # "COMMIT DEGISTI ama YARIM IS" mesajini yeniden yazdim, imza o
        # metni ariyordu ve kapi bosuna atesledi -- kod ESKI degildi,
        # IMZA eskiydi. Fonksiyon adi / cagri / atama sec; cumle secme.
        IMZA={
            "sifirdan.py": [
                'WARM_OF      = int(os.environ.get("WARM_OF"',
                'self.mask_key = None',
                'm[:, self.mask_key] = True',
                'assert NP <= len(allp)',
                'os.replace(_sy + ".tmp", _sy)'],
            "sablon/kosu.py": ['def bitti_mi', 'def _commit_kapisi'],
            "sablon/pencere.py": ['_eksik_kol', 'raise FileNotFoundError'],
            "sablon/arsivle.py": ['_alinan_', 'os.replace(gec, hy)'],
            "sablon/kayan_pencere.py": ['def adimlari_bul'],
            "sablon/bakici.sh": ['echo "--- tur'],
            "deney/d33.py": ['K.bitti_mi(ad, alt', '_SEC = [x.strip()']},
        # N_PAIR/P_TRAIN = phi 5.06'yi TANIMLAYAN iki sayi (onkayit 3).
        ORT=dict(PRESET="grok_uzun", HOP2_FRAC="0.10", MEM_AT="4",
                 RESUME_EVERY="1", WARM_OF="120000",
                 N_PAIR="56", P_TRAIN="50"),
    ),
    "g": dict(
        ONKAYIT="ONKAYIT_G_GERCEK_VERI.md",
        # IMZA = YAPISAL capa. VERI bayragini ve dis-veri yolunu ayrica
        # denetliyoruz: bayrak itilmeden kosarsak sessizce ESKI RASTGELE
        # GRAF egitilir ve sonuc "yeni veride de calisti" diye okunur.
        IMZA={
            "sifirdan.py": [
                'VERI = os.environ.get("VERI"',
                'def build_data_dis',
                'ENT_YOK, ENT_ARAMA = ent_yk, ent_ar',
                'global ENT_YOK, ENT_ARAMA',
                'ARAMA/HUKUM varlik sizintisi',
                '("ent_yok", EY, brY, scY, LY)',
                'self.mask_key = None',
                'm[:, self.mask_key] = True',
                'os.replace(_sy + ".tmp", _sy)'],
            # Kaydirma kaldirildi -> capa da degisti (14 Eylul). Eski capa
            # 'def _eb(' idi ve o fonksiyon artik YOK.
            "veri_okul.py": ['def zincirler', 'GEREKTIRIR = ',
                             'def _esle(', 'def _devirsiz(',
                             'aile kurulamadi'],
            "veri_kontrol.py": ['BOZUK KONTROL SAYISI',
                                'hicbir iliski SABIT KAYDIRMA degil',
                                'hicbir (r1,r2) zinciri SABIT KAYDIRMA degil'],
            "sablon/kosu.py": ['def bitti_mi', 'def _commit_kapisi'],
            "sablon/pencere.py": ['_eksik_kol', 'raise FileNotFoundError'],
            "sablon/arsivle.py": ['_alinan_', 'os.replace(gec, hy)'],
            "sablon/kayan_pencere.py": ['def adimlari_bul'],
            "sablon/bakici.sh": ['echo "--- tur'],
            "deney/d33_arama.py": ['ARAMA_CIKTI'],
            "deney/g.py": ['FAZ 2: ON KAPI', 'if SKOR <= 0',
                           'BUTCE = int(os.environ.get']},
        # VERI=okul  -> N_ENT/N_REL veri setinden TURETILIR, elle yazilmaz.
        # N_PAIR/P_TRAIN YOK: tipli grafta cift kumesi tipten geliyor.
        # BUTCE: adim sayisi. 80.000 ONCEDEN degil SONRADAN secildi --
        # G kolunda butun egrilerin 50.000'de doydugu OLCULDU, 120.000
        # ayni platoyu olcuyordu (iki kol icin ~78 dk bosa GPU).
        # WARM_OF BUTCE'ye ESIT olmali: isinma TOPLAM kosudan hesaplanir.
        # UZATMA KULLANICI KARARI -- buradaki iki sayi birlikte degisir.
        ORT=dict(PRESET="grok_uzun", MEM_AT="4", RESUME_EVERY="1",
                 BUTCE="80000", WARM_OF="80000", VERI="okul",
                 ENT_PAY="0.20", COMP_PAY="0.10"),
    ),
}

if len(sys.argv) < 3 or sys.argv[2] not in DENEYLER:
    sys.exit(f"kullanim: uret_sablon.py <cikti.ipynb> <deney>   "
             f"(deney: {', '.join(DENEYLER)})")
AD = sys.argv[2]
_d = DENEYLER[AD]
_ort = textwrap.fill(
    ", ".join(f'{k}="{v}"' for k, v in _d["ORT"].items()),
    width=62, initial_indent="", subsequent_indent=" " * 11)
_imza = pprint.pformat(_d["IMZA"], width=68, sort_dicts=False)

H = []


def hucre(kod):
    H.append(kod.strip("\n"))


hucre(r'''
# ============================================================================
# HUCRE 0 — AYAR.  DENEYE OZEL TEK YER. Baska hicbir hucre degismez.
# ============================================================================
DENEY   = "@AD@"                      # dosya adi MASKESI: her sey bu onekle
ONKAYIT = "@ONKAYIT@"
REPO    = "https://github.com/sekerahmet/sekerai.git"
SURUCU  = f"deney/{DENEY}.py"         # deneye ozel karar mantigi — DEPODA
KOL, SEED = "A", 0

# Kodda BULUNMASI gereken isaretler. Depoya itmeyi unutursak KAPI 3 patlar.
# {dosya: [imza]} -- SADECE sifirdan.py degil, ISKELE de denetlenir.
IMZA = @IMZA@

# Egitim ortam degiskenleri. WARM_OF: parcali kosuda isinma TOPLAM kosu
# uzerinden hesaplansin (yoksa ilk parcanin isinmasi 6000 degil 250 adim olur).
ORT = dict(ARMS=KOL, SEEDS=str(SEED), @ORT@)

# TUTARLILIK KAPISI — bu hucre uret_sablon.py'den URETILIR, elle yazilmaz.
# Eskiden sablon onceki deneyin ayarlarini tasiyordu: duzeltmeyi unutan
# kisi SESSIZCE onceki deneyi baslatiyordu, hicbir kapi yakalamiyordu.
assert DENEY.upper() in ONKAYIT.upper().replace("_", ""),     f"ONKAYIT ({ONKAYIT}) DENEY ({DENEY}) ile uyusmuyor — defteri yeniden uret"

REF_EGRI  = "/content/drive/MyDrive/deney4_tekrarli_erisim/egri_A_s0.json"
YEDEK_ARA = 300        # sn — yedek + rapor tazeleme araligi

# --- turetilen yollar (elleme) ---
CALIS = f"/content/calis_{DENEY}"                  # hizli disk, calisma
EV    = f"/content/drive/MyDrive/deney_{DENEY}"    # KALICI, yedek
KOD   = f"/content/kod_{DENEY}"                    # GitHub klonu — YAMA YOK

# --- KAPI ZINCIRI: sira ZORUNLU. Bir kapi gecilmezse sonrakiler CALISMAZ. ---
KAPI = {}


def gerek(*adlar):
    eksik = [a for a in adlar if not KAPI.get(a)]
    if eksik:
        raise SystemExit(f"SIRA HATASI — once su hucreleri kos: {eksik}\n"
                         f"  gecilen kapilar: {sorted(KAPI) or 'hicbiri'}")


print(f"deney {DENEY}   onkayit {ONKAYIT}")
print(f"  calisma {CALIS}")
print(f"  kalici  {EV}")
print(f"  kod     {KOD}  <- {REPO}")
print(f"  surucu  {SURUCU}")
''')

hucre(r'''
# ============================================================================
# HUCRE 1 — KAPI 1: DONANIM.  CPU'ya duserse 70 dk'lik is 20 saat olur.
# ============================================================================
import torch, subprocess
gerek()                       # HUCRE 0 kosmadiysa burada durur
# CPU ise HICBIR SEY calismasin: KAPI["1 DONANIM"] asagida set edilmez,
# HUCRE 2/3/4 gerek() ile reddeder.
assert torch.cuda.is_available(), \
    "GPU YOK -> Runtime > Change runtime type > T4/L4 sec, sonra bu hucreyi tekrar kos"
_p = torch.cuda.get_device_properties(0)
print(f"KAPI 1 GECTI   {_p.name}   {_p.total_memory/1e9:.0f} GB   "
      f"torch {torch.__version__}  cuda {torch.version.cuda}")
print(subprocess.run(["nvidia-smi",
                      "--query-gpu=memory.used,memory.total,utilization.gpu",
                      "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip())
KAPI["1 DONANIM"] = torch.cuda.get_device_name(0)
''')

hucre(r'''
# ============================================================================
# HUCRE 2 — KAPI 2: DRIVE.  Yedek yoksa kopan runtime = sifirdan kosu.
# ============================================================================
import os, shutil
gerek("1 DONANIM")            # GPU kapisi dustuyse buraya HIC gelinmez
from google.colab import drive
if not os.path.ismount("/content/drive"):
    drive.mount("/content/drive")
assert os.path.ismount("/content/drive"), "Drive baglanmadi"
os.makedirs(EV, exist_ok=True)
_t = f"{EV}/.yazma_testi"                    # IDDIA ETME — YAZ, OKU, SIL
open(_t, "w").write("ok")
assert open(_t).read() == "ok", "Drive'a yazilamiyor"
os.remove(_t)
_b = shutil.disk_usage("/content").free / 1e9
# Adim adli surdurme paketleri ~100 MB x durak sayisi yer kaplar.
assert _b > 20, f"/content'te sadece {_b:.0f} GB bos — adim adli yedek sigmaz"
_n = sum(len(fs) for _, _, fs in os.walk(EV))
_m = sum(os.path.getsize(os.path.join(r, f))
         for r, _, fs in os.walk(EV) for f in fs) / 1e6
print(f"KAPI 2 GECTI   {EV} yazilabilir   /content bos {_b:.0f} GB")
print(f"  Drive'da mevcut: {_n} dosya  {_m:.0f} MB")
KAPI["2 DRIVE"] = EV
''')

hucre(r'''
# ============================================================================
# HUCRE 3 — KAPI 3: KOD.  Colab'da YAMA YOK; kod GitHub'dan akar.
#   3a  imzalar kaynakta var mi              (depoya itmeyi unuttuk mu)
#   3b  iceri aktarilan modul AYNI klondan mi  (§7'nin en pahali hatasi)
#   3c  mudahale ETKILI mi + kapatinca BIT-AYNI mi  (§7: etkisizlik = ariza)
# ============================================================================
import os, sys, subprocess, hashlib, inspect, glob, json, gc, torch
gerek("1 DONANIM", "2 DRIVE")

subprocess.run(["bash", "-lc", f"rm -rf {KOD} && git clone -q {REPO} {KOD}"],
               check=True)
COMMIT = subprocess.run(["bash", "-lc", f"cd {KOD} && git rev-parse --short HEAD"],
                        capture_output=True, text=True).stdout.strip()
_src = open(f"{KOD}/sifirdan.py", encoding="utf-8").read()
MD5 = hashlib.md5(open(f"{KOD}/sifirdan.py", "rb").read()).hexdigest()
# IMZA yalniz sifirdan.py'yi denetliyordu: ISKELEYI (kosu/pencere/arsivle/
# bakici/surucu) itmeyi unutursak KAPI 3a GECIYOR ve Colab ESKI altyapiyla
# kosuyordu -- tam da bu kapinin onlemesi gereken hata, iskele icin aciktı.
_eksik = []
for _f, _gs in IMZA.items():
    _p = f'{KOD}/{_f}'
    if not os.path.exists(_p):
        _eksik.append((_f, 'DOSYA YOK')); continue
    _t = open(_p, encoding='utf-8').read()
    _eksik += [(_f, g) for g in _gs if g not in _t]
assert not _eksik, (f"DEPODAKI KOD ESKI — eksik imza: {_eksik}\n"
                    f"  yereli GitHub'a itmeyi unuttun")
_ni = sum(len(v) for v in IMZA.values())
print(f"KAPI 3a GECTI  commit {COMMIT}  md5 {MD5[:10]}  "
      f"{len(_src)} bayt  {_ni}/{_ni} imza  ({len(IMZA)} dosya)")

# KOSUYA OZEL degiskenler her zaman temizlenir; VERIYI belirleyenler de,
# yoksa onceki oturumdan sizan bir N_PAIR sessizce baska bir veri kurar
# (13 Eylul: N_PAIR=80 tam boyle sizmisti).
_KOSUYA_OZEL = ("MASK_KEY", "MASK_BLK", "RESUME_FROM", "INIT_FROM", "OUT",
                "STEPS", "SHARE", "IDENT_MODE")
# VERI/ENT_PAY/COMP_PAY bu listeye 14 Eylul'de EKLENDI. Olculdu: deney G
# VERI=okul ile kosuyor, d33'un ORT'unda VERI YOK ve eski liste onu
# silmiyordu -> ayni runtime'da G'den SONRA d33 kosulursa d33 sessizce
# OKUL VERISIYLE egitilirdi. Ustelik d33'un PHI kapisi bunu GORMEZ:
#   d33 bekleneni 5.06, tolerans 0.05;  okul verisinin phi'si 5.0857
#   fark 0.0257  ->  KAPI GECER.
# Yani sizintiyi yakalamasi gereken tek kapi, 0.024 farkla kor kaliyordu.
for _v in _KOSUYA_OZEL + ("PRESET", "ARMS", "SEEDS", "HOP2_FRAC", "MEM_AT",
                          "RESUME_EVERY", "WARM_OF", "N_ENT", "N_REL",
                          "N_PAIR", "P_TRAIN",
                          "VERI", "ENT_PAY", "COMP_PAY"):
    os.environ.pop(_v, None)
# Elle kopyalanmis filtre listesi YOK: ORT tek kaynak. Eskiden defterdeki
# liste ile surucudeki liste birbirini tutmuyordu ve KAPI 3c modeli
# KOSULACAK konfigde degil, VARSAYILAN konfigde kuruyordu.
assert not (set(ORT) & set(_KOSUYA_OZEL)),     f"ORT kosuya ozel degisken tasiyor: {set(ORT) & set(_KOSUYA_OZEL)}"
os.environ.update(ORT)
sys.path.insert(0, KOD)
sys.modules.pop("sifirdan", None)
import sifirdan as S
assert os.path.realpath(S.__file__).startswith(os.path.realpath(KOD)), \
    f"BASKA KLONDAN geliyor: {S.__file__}"
assert "self.mask_key" in inspect.getsource(S.Block.forward), "modul YAMASIZ"
# VERI KAPISI: liste elle tutuluyor ve bir dahakine yine unutulabilir.
# Bu assert dogrudan SONUCU denetliyor -- modul hangi veriyi kurdu?
assert S.VERI == ORT.get("VERI", ""), (
    f"VERI UYUSMUYOR: modul '{S.VERI}' kurdu, deney '{ORT.get('VERI','')}' "
    f"istiyor -> onceki oturumdan SIZMIS ortam degiskeni")
print(f"KAPI 3b GECTI  S.__file__ = {S.__file__}   VERI={S.VERI or '(yok)'}  "
      f"VOCAB={S.VOCAB}")

_n = S.Net(KOL, S.CFG).to(S.DEV).eval()
_x = torch.randint(0, _n.emb.num_embeddings, (8, S.T_LEN), device=S.DEV)
with torch.no_grad():
    _a = _n(_x)[0].float().clone()
    for _b in _n.blocks:
        _b.mask_key = 1
    _m = _n(_x)[0].float().clone()
    for _b in _n.blocks:
        _b.mask_key = None
    _c = _n(_x)[0].float().clone()
_fark = float((_a - _m).abs().max())
assert _fark > 1e-3, "MUDAHALE ETKISIZ -> olcum bozuk (§7)"
assert torch.equal(_a, _c), "kapatinca BIT-AYNI donmuyor"
print(f"KAPI 3c GECTI  maske etkili (fark {_fark:.2f}), kapatinca bit-ayni")
del _n, _x, _a, _m, _c
gc.collect()
torch.cuda.empty_cache()

TEMIZ = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LD_LIBRARY_PATH",
                                    "CUDA_VISIBLE_DEVICES") if k in os.environ}
_d = (json.load(open(f"{CALIS}/durum.json"))
      if os.path.exists(f"{CALIS}/durum.json") else None)
_s = sorted(glob.glob(f"{CALIS}/sur/**/*.pt", recursive=True))
KAPI["3 KOD"] = COMMIT
print(f"\nBASLANGIC  "
      f"{'SURDURULECEK: adim ' + str(_d['adim']) if _d else 'SIFIRDAN (durum yok)'}"
      f"   adim adli paket: {len(_s)}")
''')

hucre(r'''
# ============================================================================
# HUCRE 4 — BASLAT.  Surucu ve bakici AYRI SUREC -> cekirdek SERBEST kalir.
#   Bu hucre bitince defteri kapatabilirsin; kosu devam eder.
# ============================================================================
import subprocess, sys, os, time, json, shutil
gerek("1 DONANIM", "2 DRIVE", "3 KOD")
# Runtime arada degismis/kopmus olabilir: 70 dk'lik isi CPU'da baslatmayalim.
assert torch.cuda.is_available() and \
    torch.cuda.get_device_name(0) == KAPI["1 DONANIM"], \
    f"GPU degisti/kayboldu (kapida {KAPI['1 DONANIM']} vardi) -> HUCRE 1'den basla"
assert os.path.ismount("/content/drive"), "Drive dustu -> HUCRE 2'den basla"

os.makedirs(f"{CALIS}/log", exist_ok=True)
if os.path.exists(EV):                           # Drive'daki her seyi geri al
    subprocess.run(["bash", "-lc", f"cp -ru {EV}/. {CALIS}/ 2>/dev/null || true"])

json.dump(dict(DENEY=DENEY, ONKAYIT=ONKAYIT, commit=COMMIT, md5=MD5,
               CALIS=CALIS, EV=EV, KOD=KOD, KOL=KOL, SEED=SEED,
               ORT=ORT, TEMIZ=TEMIZ, REF_EGRI=REF_EGRI),
          open(f"{CALIS}/konfig_giris.json", "w"), indent=1)

subprocess.run(["bash", "-lc",
                f"pkill -f '[{DENEY[0]}]{DENEY[1:]}_surucu' ; "
                f"pkill -f '[s]ifirdan.py' ; pkill -f '[b]akici.sh' ; sleep 2"])
_lg = f"{CALIS}/log/surucu_{DENEY}.txt"
_sr = f"/content/{DENEY}_surucu.py"
shutil.copy2(f"{KOD}/{SURUCU}", _sr)             # ad maskesi: ps'te goruunsun

p1 = subprocess.Popen([sys.executable, "-u", _sr, f"{CALIS}/konfig_giris.json"],
                      stdout=open(_lg, "a"), stderr=subprocess.STDOUT)
# Bakicinin ciktisi DEVNULL'a gidiyordu: arsivle.py'nin "arsivlendi adim
# N" satiri da, aynalama hatasi da kayboluyordu. Yedekleme surecinin HIC
# adli izi yoktu — oysa varlik sebebi cekirdek kilitliyken bakabilmek.
_bl = f"{CALIS}/log/bakici_{DENEY}.txt"
p2 = subprocess.Popen(["bash", f"{KOD}/sablon/bakici.sh",
                       CALIS, EV, KOD, str(YEDEK_ARA)],
                      stdout=open(_bl, "a"), stderr=subprocess.STDOUT)
time.sleep(20)
print(f"BASLADI   surucu pid {p1.pid}   bakici pid {p2.pid}   cekirdek SERBEST")
print(f"  surucu logu : {_lg}")
print(f"  bakici logu : {_bl}")
print(f"  rapor       : {EV}/RAPOR.txt   (her {YEDEK_ARA//60} dk tazelenir)")
print("\n--- surucunun ilk satirlari ---")
print(open(_lg).read()[-1500:] or "(veri kuruluyor, ~1 dk)")
assert p1.poll() is None, "SURUCU HEMEN OLDU — yukaridaki loga bak"
''')

hucre(r'''
# ============================================================================
# HUCRE 5 — RAPOR.  Dosyayi basar, HESAP YAPMAZ, GPU kullanmaz.
#   Bakici zaten her 5 dk'da tazeliyor; burasi sadece okuyor.
#   Defter kapaliyken de Drive'dan ayni dosya okunabilir: EV/RAPOR.txt
# ============================================================================
import os, subprocess, sys
gerek()
_r = f"{CALIS}/RAPOR.txt"
if os.path.exists(_r):
    print(open(_r).read())
else:                       # bakici daha ilk turunu atmadi — kendimiz uretelim
    print(subprocess.run([sys.executable, f"{KOD}/sablon/rapor.py", CALIS],
                         capture_output=True, text=True).stdout)
''')

hucre(r'''
# ============================================================================
# HUCRE 6 — KURTARMA.  Runtime koparsa: yeni defter ac, HUCRE 0-4'u kos.
#   durum.json ve adim adli paketler Drive'da; kaldigi yerden devam eder.
# ============================================================================
import glob, os
gerek()
_p = sorted(glob.glob(f"{CALIS}/sur/**/*.pt", recursive=True))
print(f"adim adli surdurme paketleri ({len(_p)}):")
for x in _p:
    print(f"   {os.path.relpath(x, CALIS):44s} {os.path.getsize(x)/1e6:6.0f} MB")
print("\nGECMIS BIR ADIMA DONMEK icin (ornek 40000):")
print(f"   import shutil, json")
print(f"   # <kol> = yukarida gorunen klasor (cikti / cikti_a4 / cikti_d32)")
print(f"   shutil.copy2('{CALIS}/sur/<kol>/surdur_{KOL}_s{SEED}_040000.pt',")
print(f"                '{CALIS}/<kol>/surdur_{KOL}_s{SEED}.pt')")
print(f"   d = json.load(open('{CALIS}/durum.json'))")
print(f"   d['adim'] = 40000                    # sadece ilerleme cubugu")
print(f"   d.pop('<KOL>_bitti', None)           # <-- ZORUNLU. Surucu kolu")
print(f"                                        #     ATLARKEN bu bayraga")
print(f"                                        #     bakar, 'adim'a DEGIL.")
print(f"   json.dump(d, open('{CALIS}/durum.json','w'))")
print("   ...sonra HUCRE 4'u tekrar kos.")
print("\n   BITMIS bir kolu geri almak istiyorsan bayragi silmek SART;")
print("   yoksa paketi geri koysan bile kol atlanir ve hicbir sey degismez.")
''')

H = [k.replace("@AD@", AD).replace("@ONKAYIT@", _d["ONKAYIT"])
      .replace("@ORT@", _ort).replace("@IMZA@", _imza) for k in H]

# Defter SAF ASCII olsun: Colab'a MCP ile yazarken ve terminalde okurken
# tek bir kodlama surprizi bile hucreyi sessizce bozabilir.
_ASCII = {"—": "--", "§": "bolum ", "ı": "i",
          "ş": "s", "ç": "c", "ğ": "g", "ü": "u",
          "ö": "o", "İ": "I", "Ş": "S", "Ç": "C",
          "Ğ": "G", "Ü": "U", "Ö": "O", "’": "'",
          "“": '"', "”": '"', "→": "->"}
for _a, _b in _ASCII.items():
    H = [k.replace(_a, _b) for k in H]
_kalan = sorted({c for k in H for c in k if ord(c) > 127})
assert not _kalan, f"defterde ASCII disi karakter kaldi: {_kalan}"

nb = dict(nbformat=4, nbformat_minor=0,
          metadata=dict(colab=dict(provenance=[]),
                        kernelspec=dict(name="python3", display_name="Python 3"),
                        language_info=dict(name="python")),
          cells=[dict(cell_type="code", metadata={},
                      source=(k + "\n").splitlines(True),
                      execution_count=None, outputs=[]) for k in H])
import ast
for _i, _k in enumerate(H):
    try:
        ast.parse(_k)
    except SyntaxError as _e:
        _c = chr(10).join(_k.split(chr(10))[max(0, _e.lineno - 3):_e.lineno + 1])
        raise SystemExit(f'HUCRE {_i} GECERSIZ PYTHON, satir {_e.lineno}: '
                         f'{_e.msg}' + chr(10) + _c)

io.open(sys.argv[1], "w", encoding="utf-8", newline="\n").write(
    json.dumps(nb, indent=1, ensure_ascii=False))
print(f"{sys.argv[1]}   {len(H)} hucre   "
      f"{sum(k.count(chr(10)) + 1 for k in H)} satir")
