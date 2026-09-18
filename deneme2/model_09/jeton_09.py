# -*- coding: utf-8 -*-
"""jeton_09 — KARAKTER TOKENIZER. model_09'un sozlugu.

NEDEN KARAKTER, BPE DEGIL -- OLCULDU (18 Eylul). Korpusta 227.189
kelime var, yalniz 491 benzersizi. Kendi korpusumuzda egitilen BPE
(vocab 1024) `_danismaninin`i TEK JETONA birlestiriyor, yani model_08'in
atomik-iliski sorununu birebir yeniden uretiyor. Morfolojik ayristirma
ancak parcalar farkli sekillerde yeniden birlesince kazanc saglar; bu
korpusta birlesmiyorlar.

Karakterde model unlu uyumunu ('in / 'in / 'un / 'un) HARF HARF kendi
ogrenmek zorunda. `veri_09`un allomorf tablolari metni URETMEYE devam
eder ama modele ATOM olarak verilmez.

Sozluk KORPUSTAN turetilir, elle yazilmaz -- veri degisirse sozluk de
degisir ve `IZ` bunu yakalar.
"""
from __future__ import annotations

import hashlib

PAD, EOS = 0, 1
OZEL = ("<PAD>", "<EOS>")


class Sozluk:
    """Karakter <-> jeton. Deterministik: karakterler SIRALI."""

    def __init__(self, metin: str):
        self.harf = tuple(sorted(set(metin)))
        assert "\n" not in self.harf, "korpusta satir sonu OLMAMALI"
        self.ileri = {c: i + len(OZEL) for i, c in enumerate(self.harf)}
        self.geri = {i: c for c, i in self.ileri.items()}
        self.vocab = len(self.harf) + len(OZEL)
        # `korpus_09.havuz` dolduruyor: EGITIM TENSORUNUN izi.
        # `iz` yalniz karakter kumesini hash'liyor ve o neredeyse
        # hic degismez -- metin degisse bile ayni 61 harf cikar.
        self.korpus_izi = ""

    def kodla(self, s: str) -> list:
        eksik = set(s) - set(self.ileri)
        assert not eksik, f"sozlukte YOK: {sorted(eksik)}"
        return [self.ileri[c] for c in s]

    def coz(self, j) -> str:
        return "".join(self.geri[int(t)] for t in j
                       if int(t) not in (PAD, EOS))

    def ad(self, t: int) -> str:
        t = int(t)
        return OZEL[t] if t < len(OZEL) else self.geri[t]

    @property
    def iz(self) -> str:
        return hashlib.md5("".join(self.harf).encode()).hexdigest()[:12]

    def __repr__(self):
        return f"Sozluk(vocab={self.vocab}, iz={self.iz})"


def denetle(s: "Sozluk", satirlar, yaz=print) -> None:
    """GIDIS-DONUS: her satir kodlanip cozulunce AYNISI cikmali."""
    kotu = [x for x in satirlar if s.coz(s.kodla(x)) != x]
    assert not kotu, f"{len(kotu)} satir gidis-donusu GECMEDI: {kotu[:2]}"
    uz = [len(x) for x in satirlar]
    yaz(f"  gidis-donus {len(satirlar):,}/{len(satirlar):,} GECTI")
    yaz(f"  sozluk {s.vocab} jeton (iz {s.iz})   "
        f"satir {min(uz)}..{max(uz)} karakter")
