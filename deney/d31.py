# -*- coding: utf-8 -*-
"""D3.1 — model kendi kacis yolunu bulsun ve kapatsin.

Onceden kayit: ONKAYIT_D31_KENDI_BULSUN.md  (kosudan once yazildi, degismez)

    FAZ 1  maskesiz egit, her 5.000 adimda ASAMA A (ucuz arama) kos
    esik   sinyal >= 0.03 olan ILK adimda ASAMA B'yi BIR KEZ kos -> hangi PARCA
    FAZ 2  maskeyi o parcaya, bloklar 1..L-1'e koy, 120.000'e kadar devam

Insan mudahalesi yok: argmax ne derse o. Yanlis parca secerse duzeltilmez.

    python d31.py <konfig_giris.json>
"""
import os, sys, json, time, glob

KON = json.load(open(sys.argv[1]))
sys.path.insert(0, KON["KOD"])
sys.path.insert(0, os.path.join(KON["KOD"], "sablon"))

for _v in ("MASK_KEY", "MASK_BLK", "RESUME_FROM", "INIT_FROM", "OUT",
           "SHARE", "IDENT_MODE", "STEPS", "WARM_OF"):
    os.environ.pop(_v, None)
# DUMAN TESTI: ayni kod yollari, kucuk sayilarla. Gercek kosuya DOKUNMAZ
# (ayri CALIS klasoru). Sadece SAYILAR degisir, mantik degismez.
SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:                       # PRESET/MEM_AT modul seviyesinde assert edilir
    KON["ORT"] = dict(KON["ORT"], PRESET="smoke", MEM_AT="2", EVERY="200",
                      WARM_OF="600", COMPILE="0")
os.environ.update({k: v for k, v in KON["ORT"].items()
                   if k in ("PRESET", "ARMS", "SEEDS", "HOP2_FRAC", "MEM_AT")})

import numpy as np
import torch
import sifirdan as S
from kosu import Kosu

# ---------------------------------------------------------------- ONCEDEN KAYIT
KON.update(
    ESIK=0.03,                                     # A ve E2'nin iki kosusundan
    DERINLIK=list(range(1, S.CFG["L"])),           # D3'te sabitlendi: 1..L-1
    DURAKLAR=list(range(5000, 85000, 5000)),       # ADIM EZBERI YOK
    HEDEF_SON=120000,
    NA=2000,                                       # olculen sd 0.0036
    PENCERE=[60000, 65000, 70000, 75000, 80000],   # birincil olcu
    TOL_YORUNGE=0.05,                              # FAZ 1 == referans kol mu
    # rapor.py icin: {ad: [alan, tolerans, yon, faz]}
    #   yon -1 referansin ALTINA duserse, +1 USTUNE cikarsa, 0 iki yonde de
    # DOLANMA: maske acikken kisayol, MASKESIZ kontrolu gecerse model maskeyi
    #   dolaniyor demektir. Mutlak degere bakmak YANILTIR: A'da kisayol zaten
    #   0.21'den 0.74'e tirmaniyor, maskelide de yukselmesi tek basina bir sey
    #   soylemez — kiyas KONTROL koluyla yapilir (CLAUDE.md 4, "eslestir").
    UYARI={"FAZ1-BUTUNLUK": ["comp", 0.05, 0, 1],
           "SAGLIK":        ["comp", 0.10, -1, 2],
           "YOL":           ["ent", 0.00, -1, 2],
           "DOLANMA":       ["ent_shortcut", 0.00, +1, 2]},
)
KON["ATES_SINIR"] = KON["PENCERE"][0] - 5000       # atesleme bunu asarsa pencere kirlenir
if SMOKE:
    KON.update(DURAKLAR=[200, 400], HEDEF_SON=600, ESIK=-1.0, NA=64,
               PENCERE=[200, 400], TOL_YORUNGE=9.0, ATES_SINIR=10**9)

K = Kosu(KON)
ESIK, DURAKLAR, DERINLIK = KON["ESIK"], KON["DURAKLAR"], tuple(KON["DERINLIK"])
HEDEF_SON, NA, ATES_SINIR = KON["HEDEF_SON"], KON["NA"], KON["ATES_SINIR"]
MBLK = ",".join(str(b) for b in DERINLIK)

K.log(f"D3.1 basliyor   commit {KON['commit']}   esik {ESIK}   "
      f"duraklar {DURAKLAR[0]}..{DURAKLAR[-1]}   derinlik {DERINLIK[0]}..{DERINLIK[-1]}")
K.log(f"  atesleme <= {ATES_SINIR} olmali, yoksa birincil pencere kirlenir")

# ------------------------------------------------------------------- OLCME SETI
facts, pairs, one, tr2, comp_, ent_ev, seen_ent, unseen_ent, ent2_ev = S.build_data()
lo, hi = S.ENT_OFF, S.ENT_OFF + S.CFG["N_ENT"]
rg = np.random.RandomState(0)                      # SABIT ornek: duraklar arasi kiyas
VER = {}
for ad, lst in (("COMP", comp_[:NA]), ("ENT", ent_ev[:NA])):
    E = S.enc_two(lst)
    VER[ad] = dict(X=E[0], poz=E[1],
                   gold=(S.ENT_OFF + np.array([x[4] for x in lst])).astype(np.int64),
                   boz={})
    for p in (1, 2, 3):                            # her girisi BOZ: 1=varlik, 2/3=iliski
        X = E[0].copy()
        off, N = (S.ENT_OFF, S.CFG["N_ENT"]) if p == 1 else (S.REL_OFF, S.CFG["N_REL"])
        cur = X[:, p] - off
        y = rg.randint(N - 1, size=len(X)); y += (y >= cur)
        X[:, p] = off + y
        VER[ad]["boz"][p] = X

net = S.Net(KON["KOL"], S.CFG).to(S.DEV)


def yukle(adim):
    y = K.y("cikti", f"snap_{KON['KOL']}_s{KON['SEED']}_{adim:06d}.pt")
    assert os.path.exists(y), f"anlik goruntu yok: {y}"
    net.load_state_dict({k: v.float() for k, v in
                         torch.load(y, map_location=S.DEV).items()})
    net.eval()


def maskele(p=None, b0=None):
    for i, blk in enumerate(net.blocks):
        blk.mask_key = p if (p is not None and i >= b0) else None


@torch.no_grad()
def tah(X, poz, bs=1000):
    o = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(S.DEV)
        ix = torch.from_numpy(poz[i:i + bs]).to(S.DEV)
        ar = torch.arange(len(ix), device=S.DEV)
        lg, _ = net(xb)
        o.append((lg.float()[ar, ix][:, lo:hi].argmax(-1) + lo).cpu().numpy())
    return np.concatenate(o)


def asamaA(adim):
    """UCUZ arama: hangi girdi parcasi, ENT'te COMP'tan FAZLA degismezlik veriyor."""
    yukle(adim); maskele()
    tmz = {a: tah(VER[a]["X"], VER[a]["poz"]) for a in ("COMP", "ENT")}
    deg = {a: {p: float((tmz[a] == tah(VER[a]["boz"][p], VER[a]["poz"])).mean())
               for p in (1, 2, 3)} for a in ("COMP", "ENT")}
    d = {p: deg["ENT"][p] - deg["COMP"][p] for p in (1, 2, 3)}
    ps = max(d, key=d.get)
    return dict(adim=adim, d={str(k): v for k, v in d.items()}, p_yildiz=ps,
                sinyal=d[ps], inv0=deg["ENT"][ps],
                comp=float((tmz["COMP"] == VER["COMP"]["gold"]).mean()))


def asamaB(adim, ps, inv0, acc0):
    """PAHALI arama, BIR KEZ: 3 parca x L blok-kuyrugu. Skor = r1 bagimsizligini
    dusur, COMP'a dokunma."""
    yukle(adim)
    tablo, etki = [], False
    for p in (1, 2, 3):
        for b0 in range(S.CFG["L"]):
            maskele(p, b0)
            iv = float((tah(VER["ENT"]["X"], VER["ENT"]["poz"]) ==
                        tah(VER["ENT"]["boz"][ps], VER["ENT"]["poz"])).mean())
            ac = float((tah(VER["COMP"]["X"], VER["COMP"]["poz"]) ==
                        VER["COMP"]["gold"]).mean())
            if abs(iv - inv0) > 1e-9 or abs(ac - acc0) > 1e-9:
                etki = True
            tablo.append(dict(p=p, b0=b0, inv=iv, comp=ac,
                              skor=(inv0 - iv) - (acc0 - ac)))
    maskele()
    # §7: butun adaylar tam +0.000 verdiyse bulgu "etki yok" degil "olcum bozuk".
    assert etki, "ASAMA B: maske HIC etki etmedi -> olcum bozuk"
    kaz = max(tablo, key=lambda r: r["skor"])
    return dict(adim=adim, tablo=tablo, kazanan=kaz, secilen_parca=kaz["p"])


def ham(ad, veri):
    y = K.y("ham", f"{ad}.json")
    if not os.path.exists(y):
        json.dump(veri, open(y, "w"), indent=1)


# ============================================================== ANA DONGU
while True:
    st = K.st

    # ---------------------------------------------------------- FAZ 1
    if st["faz"] == 1:
        nxt = next((a for a in DURAKLAR if a > st["adim"]), None)
        if nxt is None:
            K.log(f"!! {DURAKLAR[-1]}'e kadar ATESLEMEDI -> PROSEDUR BASARISIZ")
            K.kaydet(faz=3, faz_ad="BITTI — prosedur atesleyemedi")
            break
        K.kaydet(faz_ad=f"FAZ 1 — maskesiz, {nxt} adima egitiliyor")
        sn = K.egit(nxt)
        K.kaydet(adim=nxt)

        # KAPI: maskesiz faz referans kolun TEKRARI olmali (bugunku warm hatasi)
        K.yorunge_kapisi(nxt)

        a = asamaA(nxt)
        st.setdefault("asamaA", []).append(a)
        ham(f"asamaA_{nxt:06d}", a)
        gec = a["sinyal"] >= ESIK
        K.log(f"FAZ1 {nxt:6d} ({sn} sn)  comp {a['comp']:.3f}  p*={a['p_yildiz']}  "
              f"sinyal {a['sinyal']:+.4f}   {'>>> ESIK GECILDI' if gec else ''}")
        K.not_(f"esik {ESIK}   son sinyal {a['sinyal']:+.4f} (p*={a['p_yildiz']})"
               f"   {'GECTI' if gec else 'gecilmedi, tavan %d' % DURAKLAR[-1]}")

        if gec:
            b = asamaB(nxt, a["p_yildiz"], a["inv0"], a["comp"])
            ham(f"asamaB_{nxt:06d}", b)
            K.kaydet(faz=2, atesledi=True, secilen=b["secilen_parca"],
                     asamaB={k: b[k] for k in ("adim", "kazanan", "secilen_parca")})
            K.log(f"  ASAMA B -> PARCA {b['secilen_parca']} "
                  f"(skor {b['kazanan']['skor']:+.3f})   atesleme adimi {nxt}")
            K.log(f"  maske: poz {b['secilen_parca']}, bloklar {MBLK}")
            # ONKAYIT R4: parca 1 secilirse ek kontrol kolu gerekmez.
            K.log(f"  R4: ek kontrol kolu "
                  f"{'GEREKMEZ' if b['secilen_parca'] == 1 else 'GEREKIR'}")
            if nxt > ATES_SINIR:
                K.log(f"  !! PENCERE KIRLENMESI: atesleme {nxt} > {ATES_SINIR}; "
                      f"birincil olcu maskesiz bolgeyi de kapsiyor -> YORUMLANAMAZ")
            K.not_(f"ATESLEDI  adim {nxt}  ->  PARCA {b['secilen_parca']} "
                   f"(skor {b['kazanan']['skor']:+.3f})   maske bloklar {MBLK}",
                   f"pencere kirlenmesi: {'YOK' if nxt <= ATES_SINIR else 'VAR'}"
                   f"   R4 ek kontrol kolu: "
                   f"{'GEREKMEZ' if b['secilen_parca'] == 1 else 'GEREKIR'}")

    # ---------------------------------------------------------- FAZ 2
    elif st["faz"] == 2:
        mk = str(st["secilen"])
        ek = dict(MASK_KEY=mk, MASK_BLK=MBLK)
        at = st["asamaB"]["adim"]
        ara = min(at + 10000, HEDEF_SON)       # once KISA adim: konfigi ERKEN dogrula
        if st["adim"] < ara:
            K.kaydet(faz_ad=f"FAZ 2 — maskeli (parca {mk}), ara durak {ara}")
            sn = K.egit(ara, ek)
            K.kaydet(adim=ara)
            K.log(f"FAZ2 ara durak {ara} ({sn} sn)")
            K.konfig_kapisi(dict(MASK_KEY=mk, MASK_BLK=list(DERINLIK)))
        K.kaydet(faz_ad=f"FAZ 2 — maskeli (parca {mk}), {HEDEF_SON} adima")
        sn = K.egit(HEDEF_SON, ek)
        K.kaydet(adim=HEDEF_SON, faz=3, faz_ad="BITTI")
        K.log(f"FAZ2 bitti ({sn} sn) -> {HEDEF_SON}")
        K.konfig_kapisi(dict(MASK_KEY=mk, MASK_BLK=list(DERINLIK)))

    # ---------------------------------------------------------- BITTI
    else:
        v = [a for a in KON["PENCERE"]
             if os.path.exists(K.y("cikti",
                                   f"snap_{KON['KOL']}_s{KON['SEED']}_{a:06d}.pt"))]
        K.log(f"BITTI. Birincil pencere {KON['PENCERE'][0]}-{KON['PENCERE'][-1]}: "
              f"{len(v)}/{len(KON['PENCERE'])} nokta hazir.")
        K.not_(f"BITTI — birincil okuma icin {len(v)}/{len(KON['PENCERE'])} nokta hazir")
        break

K.log("SURUCU BITTI")
