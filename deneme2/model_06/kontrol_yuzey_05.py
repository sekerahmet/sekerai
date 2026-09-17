# -*- coding: utf-8 -*-
"""KONTROL: model_05, model_06'NIN YUZEYIYLE sinanirsa ne oluyor?

Soru (kullanici, 17 Eylul): "bizim ent sorulari eski ent sorulari gibi mi?"
Olculdu: ZINCIRLER ayni, YUZEY farkli. model_05'in sinavi soruyu ve iki
iliskiyi IKI KEZ veriyor (onek 16 jeton); model_06 BIR KEZ (onek 7).

model_05 EGITIMDE bicim 0/1/2'nin UCUNU DE gordu (ayar_05.bicim=3), ve
bicim 2 = BILDIRIM = model_06'nin sinav yuzeyinin AYNISI. Yani bu kontrol
DAGILIM ICI: model_05'e hic gormedigi bir sey sorulmuyor.

  bicim 0  Pinar Celik'in ogrencisinin cocugu kim? Pinar Celik'in
           ogrencisinin cocugu ___        <- model_05'in RAPORLANAN sinavi
  bicim 2  Pinar Celik'in ogrencisinin cocugu ___
                                          <- model_06'nin sinavi

comp 0.9454 -> 0.3970 dususu TAKAS mi, yoksa ONEK YARIYA INDIGI icin mi?
Bu kontrol onu ayirir.
"""
import sys, os, glob, json, io
sys.path.insert(0, r"C:\AI_NEW_MODEL\deneme2\model_05")
import numpy as np, torch
import taban_05 as M
import ayar_05 as A
import pencere_05 as P

ALT = r"G:\Drive'ım\model_05\t0"
GEN = 5

ayar = A.AYAR
v = M.veri_kur(ayar, yaz=lambda *a, **k: None)
L = M.olcme_listeleri(ayar, v)
iz = M.olcme_izi(L)
k = json.load(io.open(os.path.join(ALT, "kosu_t0.json"), encoding="utf-8"))
assert k["olcme_izi"] == iz, (k["olcme_izi"], iz)
print(f"olcme_izi TUTUYOR: {iz}", flush=True)

yollar = sorted(glob.glob(os.path.join(ALT, "snap", "*.pt")))[-GEN:]
print("pencere:", [os.path.basename(y).split("_")[-1][:-3] for y in yollar],
      flush=True)
sd = P.agirlik_ortalamasi(yollar)
net = (P.MODEL_SINIFI or M.Model)(ayar, v.vocab).to(M.DEV)
print("model sinifi:", type(net).__name__, flush=True)
net.load_state_dict(sd)
net.eval()
print("cihaz:", M.DEV, flush=True)

HANGI = ("one", "seen", "comp", "ent", "ent_yok", "ood")
sonuc = {}
for bic in (0, 2):
    r = {}
    for h in HANGI:
        kod = (M.kodla_1hop(v, L[h], bic) if h == "one"
               else M.kodla_2hop(v, L[h], bic))
        r[h] = M.dogruluk(net, v, *kod)
        print(f"  bicim {bic}  {h:8s} {r[h]:.4f}", flush=True)
    sonuc[bic] = r

print()
print("=" * 66)
print(f"model_05  pencere {os.path.basename(yollar[0]).split('_')[-1][:-3]}"
      f"-{os.path.basename(yollar[-1]).split('_')[-1][:-3]}   genislik {GEN}")
print("=" * 66)
print(f"{'':10s} {'bicim0 (SORULU)':>16s} {'bicim2 (BILDIRIM)':>18s} {'fark':>9s}")
for h in HANGI:
    a, b = sonuc[0][h], sonuc[2][h]
    print(f"{h:10s} {a:16.4f} {b:18.4f} {b-a:+9.4f}")
print()
print("model_06 (20.000, pencere 12000-20000, bicim0 = BILDIRIM):")
print("  one 0.9237  seen 0.9723  comp 0.3970  ent 0.3083  ood ----")
io.open("kontrol05_sonuc.json", "w", encoding="utf-8").write(
    json.dumps({str(a): b for a, b in sonuc.items()}, indent=1))
