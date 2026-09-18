# -*- coding: utf-8 -*-
"""KAYIP TABANI: havuzun kendi kosullu entropisi.

Butun olgulari EZBERLEMIS mukemmel bir modelin bile odeyecegi alt sinir.
Kaybin bu tabana ne kadar yakin oldugu "ogrenilebilir kalan"i verir ve
bu proje uzatma kararlarini ona dayandiriyor (OLCULENLER §2).

Maskeleme EGITIMDEKININ AYNISI (taban_0X, tam_kayip dali):
    hedef = x[1:]           next-token
    hedef == PAD      -> atlanir
    hedef == <BOS>    -> PAD'e cevrilir, atlanir
Onek = dizinin BASINDAN o konuma kadarki JETONLARIN TAMAMI.

Dogrulama: model_06 icin kayitli sayi 0.5148. Bu betik onu yeniden
uretmezse betik yanlistir, havuz degil.
"""
import sys, collections, math
import numpy as np

def taban(kol, yaz=print):
    yol = r"C:\AI_NEW_MODEL\deneme2\model_" + kol
    sys.path.insert(0, yol)
    for m in list(sys.modules):
        if m.startswith(("taban_", "veri_", "ayar_", "analiz_")):
            del sys.modules[m]
    import importlib
    M = importlib.import_module("taban_" + kol)
    A = importlib.import_module("ayar_" + kol)
    v = M.veri_kur(A.AYAR, yaz=lambda *a, **k: None)
    X = M.egitim_havuzu(A.AYAR, v, yaz=lambda *a, **k: None)[0]
    sys.path.remove(yol)

    X = np.asarray(X, np.int64)
    BOS = getattr(v, "bosluk", 0)
    # onek -> sonraki jeton sayaci.  Onek ARTIMLI hash'leniyor.
    say = collections.defaultdict(collections.Counter)
    for satir in X:
        h = 0
        for i in range(len(satir) - 1):
            t = int(satir[i])
            if t == M.PAD:
                break
            h = (h * 1000003 + t + 1) % (1 << 61)
            y = int(satir[i + 1])
            if y == M.PAD or (BOS and y == BOS):
                continue
            say[h][y] += 1
    top, n = 0.0, 0
    for c in say.values():
        N = sum(c.values())
        for k in c.values():
            top -= k * math.log(k / N)
        n += N
    yaz(f"model_{kol}:  havuz {len(X):,} satir   olculen konum {n:,}   "
        f"farkli onek {len(say):,}")
    yaz(f"           KAYIP TABANI = {top / n:.4f} nat")
    return top / n

if __name__ == "__main__":
    for kol in sys.argv[1:] or ["06", "07"]:
        taban(kol)
