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
sp = os.path.join(C, "cikti", f"surdur_{kol}_s{seed}.pt")
if not os.path.exists(sp):
    sys.exit(0)

try:
    adim = int(torch.load(sp, map_location="cpu", weights_only=False)["step"])
except Exception as e:                       # yarida yazilmis olabilir: sonraki tur
    print(f"paket okunamadi ({e}), sonraki turda tekrar denenecek")
    sys.exit(0)

os.makedirs(os.path.join(C, "sur"), exist_ok=True)
hy = os.path.join(C, "sur", f"surdur_{kol}_s{seed}_{adim:06d}.pt")
if os.path.exists(hy):
    sys.exit(0)
shutil.copy2(sp, hy + ".tmp")                # .tmp + mv: yarim dosya kalmasin
os.replace(hy + ".tmp", hy)
n = len(glob.glob(os.path.join(C, "sur", "*.pt")))
print(f"arsivlendi: adim {adim}  ({n} geri donus noktasi, "
      f"{os.path.getsize(hy)/1e6:.0f} MB)")
