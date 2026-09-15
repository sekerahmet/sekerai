# -*- coding: utf-8 -*-
"""model_a3 — AGIRLIK CURUMESI.  TEK FARK: wd 0.1 -> 0.5.

Onceden kayit: belge/onkayit/model_a3.md
Kullanici karari, 15 Eylul 2026.

SORU: wd, `ent`i hareket ettiriyor mu?

    model_a    l=4 dongu=2  wd=0.1    <- OLCULMEMIS ARALIKTA
    model_a3   l=4 dongu=2  wd=0.5    <- literaturun calisan araligina dogru
               ^ TEK FARK. Mimari, veri, tohum, lr, adim AYNI.

`model_a.py` `wd` alaninin yanina bunu yazmisti:

    2603.25009'un taramasi (modular addition, AdamW):
      lambda 0.01  ->  hic grokking YOK
      lambda 1.0   ->  3/3 tohum, gecikme 44.000 adim
      lambda 5.0   ->  3/3 tohum, gecikme 24.000 adim   (optimal)
    0.1 bu taramanin ALTINDA kaliyor (0.01 ile 1.0 arasi, OLCULMEMIS).
    SINIR: o tarama MODULAR ADDITION'da yapildi, bizim gorevde degil.
    Yani "0.1 yanlis" DIYEMEM; "olculmemis aralikta" diyebilirim.

Iste bu kol, o "olculmemis"i olcer. Ve `wd` dosyada acikca "KRITIK DUGME"
diye isaretlenmisti -- A ailesinde henuz hic cevrilmemis tek dugme bu.

--------------------------------------------------------------------------
NEDEN 0.5, NEDEN 1.0 DEGIL

0.1 ile 1.0 arasinin ORTASI (log olcekte ~0.32, biz 0.5 aldik). Amac
once ARALIGI taramak: 0.5 hicbir sey degistirmezse 1.0 denenir, cok
bozarsa 0.2 denenir. Tek atisla "en iyi wd"yi bulmaya calismiyoruz.

BEDELI VAR VE ONCEDEN YAZILI: wd buyudukce `egit()`teki grup secimi
onem kazaniyor. Kod gommeyi (dim 2) DECAY ICINE koyuyor:

    dec   = [p for p in model.parameters() if p.dim() >= 2]   # GOMME DAHIL
    nodec = [p for p in model.parameters() if p.dim() <  2]

Bu bir SECIM ve model_a.py'de "wd buyudukce onem kazanir; ACIK DUGME"
diye isaretli. 0.5'te sonuc bozulursa, sucu wd'ye atmadan once bu
secimin sorgulanmasi gerekir -- iki dugme ayni anda bastiriliyor olabilir.

BU DOSYA MIMARIYI YENIDEN TANIMLAMAZ: `model_a`dan import eder, yalniz
`AYAR`in tek alanini degistirir. `pencere_a.py` bu kolu da olcer --
mimariyi `ayar_t<N>.json`dan okuyor, VARSAYMIYOR.
"""
from __future__ import annotations

from model_a import AYAR as TABAN, egit, fark_bas           # noqa: F401

AYAR = TABAN.degistir(ad="model_a3", wd=0.5)
#      ^ SADECE FARK. Mimari/veri/tohum/lr/adim/d/dff TEKRAR YAZILMAZ.

if __name__ == "__main__":
    fark_bas(TABAN, AYAR)
    egit(AYAR)
