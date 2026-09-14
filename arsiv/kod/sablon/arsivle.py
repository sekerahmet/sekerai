# -*- coding: utf-8 -*-
"""ARSIVLE — surdurme paketini ADIM ADLI kopyala.

sifirdan.py paketi her olcumde AYNI ada yazar (surdur_<kol>_s<seed>.pt), yani
uzerine yazar; gecmis bir adima donmek imkansiz olur (CLAUDE.md 7 ve 9 —
A'nin 5.000-170.000 arasi ornek-basina kaydi tam bu yuzden kayboldu).

Bakici bunu her turda cagirir. Adim, paketin ICINDEN okunur (egri dosyasindan
DEGIL): egri paketten SONRA yaziliyor, aradaki yaris kopyaya yanlis ad verirdi.

Kopya yoksa alinir, varsa dokunulmaz -> her 5.000 adimda bir geri donus noktasi.

    python arsivle.py <CALIS>
"""
import sys, os, json, glob, shutil
import torch

C = sys.argv[1]
kon = json.load(open(os.path.join(C, "konfig.json")))
kol, seed = kon.get("KOL", "A"), kon.get("SEED", 0)

# Bir kosuda BIRDEN COK kol olabilir (D3.2: cikti_a4 + cikti_d32). Klasor
# adini VARSAYMA, diskten bul — "cikti" sabitini yazmak d32'de sessizce
# hicbir sey yedeklememeye yol aciyordu.
altlar = sorted(os.path.basename(p) for p in glob.glob(os.path.join(C, "cikti*"))
                if os.path.isdir(p))
if not altlar:
    sys.exit(0)

for alt in altlar:
    sp = os.path.join(C, alt, f"surdur_{kol}_s{seed}.pt")
    if not os.path.exists(sp):
        continue
    os.makedirs(os.path.join(C, "sur", alt), exist_ok=True)
    # ONCE KOPYALA, SONRA ADIMI KOPYADAN OKU.  Tersi yaristi: adim
    # kaynaktan okunup kopya sonra alininca, arada sifirdan.py os.replace
    # ile paketi degistirebiliyordu -> kopya YENI agirliklari tasiyip ESKI
    # adimin adini aliyordu. Sessiz, kalici, fark edilmez bir etiket hatasi.
    # shutil.copy2 dosyayi bir kez acar; os.replace dizin girdisini degistirse
    # bile acik tanitici eski inode'u okumaya devam eder -> kopya TUTARLI.
    gec = os.path.join(C, "sur", alt, f"_alinan_{kol}_s{seed}.tmp")
    try:
        shutil.copy2(sp, gec)
        adim = int(torch.load(gec, map_location="cpu",
                              weights_only=False)["step"])
    except Exception as e:                   # yarida yazilmis: sonraki turda
        if os.path.exists(gec):
            os.remove(gec)
        print(f"{alt}: paket okunamadi ({e}), sonraki turda tekrar")
        continue
    hy = os.path.join(C, "sur", alt, f"surdur_{kol}_s{seed}_{adim:06d}.pt")
    if os.path.exists(hy):
        os.remove(gec)
        continue
    os.replace(gec, hy)                      # atomik: yarim dosya gorunmez
    n = len(glob.glob(os.path.join(C, "sur", alt, "*.pt")))
    print(f"{alt}: arsivlendi adim {adim}  ({n} geri donus noktasi, "
          f"{os.path.getsize(hy)/1e6:.0f} MB)")
