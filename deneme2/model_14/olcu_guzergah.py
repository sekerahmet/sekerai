# -*- coding: utf-8 -*-
"""KAC AYRI GUZERGAH OGRENILMELI -- dilin kendi kapasite sorusu.

Kullanici: "Istanbul'dan Ankara'ya gidiyorsam sonrasi Izmir ya da
Bursa; Mersin'den Ankara'ya gidiyorsam Sivas ya da Erzurum."

Yani durum, GECMISI degil gecmisin DEVAMI BELIRLEYEN kismini
tutmali.  Iki guzergahin devami AYNIYSA ayni duruma dusebilirler.
Olculecek sey: k uzunluktaki baglamlardan kac tanesi BIRBIRINDEN
AYIRT EDILMEK ZORUNDA.
"""
import sys, collections
sys.path.insert(0, r"C:\AI_NEW_MODEL\deneme2\model_14")
import veri_14 as V, metin_14 as M, ek_14 as EK

G = V.kur(0)
tipi = {a: t for t in V.TIPLER for a in G["ad"][t]}
_yuz = lambda a: [x for w in a.split("_") for x in V.TR.get(w, w).split()]
ozel = frozenset(w for t in V.TIPLER for a in G["ad"][t] for w in _yuz(a))

# korpusun kendi karisimi: 17 bildirim + 8 soru
cumleler = []
for (o, r), c in sorted(G["olgu"].items()):
    for k in range(M.N_BILDIRIM):
        cumleler.append(M.cumle(o, r, c, k))
    for k in range(M.N_SORU):
        cumleler.append(M.soru(o, r, c, tipi[c], k))
print("cumle %s" % format(len(cumleler), ","))

say = collections.Counter()
for s in cumleler:
    say.update(s.split())
kok = EK.kok_havuzu(say)
bolme = {w: EK.bol(w, kok, korunan=ozel) for w in say}
diz = [[u for w in s.split() for u in bolme[w]] for s in cumleler]
print("birim akisi %s" % format(sum(len(d) for d in diz), ","))

print()
print("  k   AYRI baglam      AYRI DEVAM KUMESI    dallanma   sikistirma")
for k in range(1, 7):
    dev = collections.defaultdict(set)
    for d in diz:
        for i in range(len(d) - k):
            dev[tuple(d[i:i + k])].add(d[i + k])
    nb = len(dev)
    # devam kumesi AYNI olanlar ayni duruma dusebilir
    sinif = len(set(frozenset(v) for v in dev.values()))
    dal = sum(len(v) for v in dev.values()) / nb
    print("  %d  %13s    %13s        %5.2f      %5.1f kat"
          % (k, format(nb, ","), format(sinif, ","), dal, nb / sinif))

print()
print("  AYRI baglam        = kac farkli k-li gecmis GORULUYOR")
print("  AYRI DEVAM KUMESI  = bunlardan kac tanesi BIRBIRINDEN")
print("                       ayirt edilmek ZORUNDA")
print("  sikistirma         = modelin bedava aldigi kazanc")
