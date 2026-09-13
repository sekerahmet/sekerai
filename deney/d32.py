# -*- coding: utf-8 -*-
"""D3.2 — tarifin tam hali, veri x4'te.

Onceden kayit: ONKAYIT_D32_VERI_X4.md  (kosudan once yazildi, degismez)

    FAZ 1  A4: maskesiz, 0 -> 120.000.  Hem KONTROL kolu hem arama substrati.
           Her 5.000'de ASAMA A. Esik gecilince ASAMA B kosar, yer KAYDEDILIR,
           ama egitim DURMAZ (A4'un kontrol olarak tamamlanmasi gerekiyor).
    FAZ 2  D3.2: bos klasorde, ayni tohumla SIFIRDAN -> baslangic parametreleri
           bit-ayni (52/52 olculdu). A4'un buldugu maskeyle 0 -> 120.000.

DONDURULMUS SABITLER (onkayit 4): esik 0.03, derinlik 1..L-1, duraklar 5.000,
pencere 60-80k, hedef 120.000. Yeni olcekte YENIDEN AYARLANMAZ.

    python d32.py <konfig_giris.json>
"""
import os, sys, json, time, glob

KON = json.load(open(sys.argv[1]))
sys.path.insert(0, KON["KOD"])
sys.path.insert(0, os.path.join(KON["KOD"], "sablon"))

for _v in ("MASK_KEY", "MASK_BLK", "RESUME_FROM", "INIT_FROM", "OUT",
           "SHARE", "IDENT_MODE", "STEPS", "WARM_OF"):
    os.environ.pop(_v, None)

SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    KON["ORT"] = dict(KON["ORT"], PRESET="smoke", MEM_AT="2", EVERY="200",
                      WARM_OF="600", COMPILE="0", N_ENT="200", N_PAIR="6",
                      P_TRAIN="4")
os.environ.update({k: v for k, v in KON["ORT"].items()
                   if k in ("PRESET", "ARMS", "SEEDS", "HOP2_FRAC", "MEM_AT",
                            "N_ENT", "N_REL", "N_PAIR", "P_TRAIN")})

import numpy as np
import torch
import sifirdan as S
from kosu import Kosu

A4, D32 = "cikti_a4", "cikti_d32"           # iki kolun klasorleri

KON.update(
    ESIK=0.03,                                     # DONDURULMUS
    DERINLIK=list(range(1, S.CFG["L"])),           # DONDURULMUS
    DURAKLAR=list(range(5000, 125000, 5000)),      # A4 sonuna kadar surer
    HEDEF_SON=120000,
    NA=2000,
    PENCERE=[60000, 65000, 70000, 75000, 80000],   # DONDURULMUS
    OLGUNLUK=0.30,                                 # comp(A4 @ 60-80k) esigi
    TOL_YORUNGE=9.9,        # A4 kendisi referans; yorunge kapisi ANLAMSIZ
    REF_AD="A4",
    KONTROL_ALT=A4,        # rapor kiyas sutunu: ayni kosudaki kontrol kolu
    UYARI={"SAGLIK":   ["comp", 0.10, -1, 2],
           "YOL":      ["ent", 0.00, -1, 2],
           "DOLANMA":  ["ent_shortcut", 0.00, +1, 2]},
)
KON["ATES_SINIR"] = KON["PENCERE"][0] - 5000
if SMOKE:
    KON.update(DURAKLAR=[200, 400, 600], HEDEF_SON=600, ESIK=-1.0, NA=64,
               PENCERE=[200, 400], ATES_SINIR=10**9, OLGUNLUK=-1.0)

K = Kosu(KON)
ESIK, DURAKLAR = KON["ESIK"], KON["DURAKLAR"]
DERINLIK = tuple(KON["DERINLIK"])
HEDEF_SON, NA, ATES_SINIR = KON["HEDEF_SON"], KON["NA"], KON["ATES_SINIR"]
MBLK = ",".join(str(b) for b in DERINLIK)

K.log(f"D3.2 basliyor  commit {KON['commit']}  "
      f"veri {S.CFG['N_ENT']}x{S.CFG['N_PAIR']} ({S.CFG['P_TRAIN']} egitimde)  "
      f"L={S.CFG['L']} D={S.CFG['D']}")
K.log(f"  DONDURULMUS: esik {ESIK}, derinlik {DERINLIK[0]}..{DERINLIK[-1]}, "
      f"duraklar 5000, pencere {KON['PENCERE'][0]}-{KON['PENCERE'][-1]}")

# ------------------------------------------------------------------- OLCME SETI
facts, pairs, one, tr2, comp_, ent_ev, seen_ent, unseen_ent, ent2_ev = S.build_data()
lo, hi = S.ENT_OFF, S.ENT_OFF + S.CFG["N_ENT"]
rg = np.random.RandomState(0)                      # SABIT ornek
VER = {}
for ad, lst in (("COMP", comp_[:NA]), ("ENT", ent_ev[:NA])):
    E = S.enc_two(lst)
    VER[ad] = dict(X=E[0], poz=E[1], boz={},
                   gold=(S.ENT_OFF + np.array([x[4] for x in lst])).astype(np.int64))
    for p in (1, 2, 3):
        X = E[0].copy()
        off, N = (S.ENT_OFF, S.CFG["N_ENT"]) if p == 1 else (S.REL_OFF, S.CFG["N_REL"])
        cur = X[:, p] - off
        y = rg.randint(N - 1, size=len(X)); y += (y >= cur)
        X[:, p] = off + y
        VER[ad]["boz"][p] = X

net = S.Net(KON["KOL"], S.CFG).to(S.DEV)


def yukle(alt, adim):
    y = K.y(alt, f"snap_{KON['KOL']}_s{KON['SEED']}_{adim:06d}.pt")
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
    yukle(A4, adim); maskele()
    tmz = {a: tah(VER[a]["X"], VER[a]["poz"]) for a in ("COMP", "ENT")}
    deg = {a: {p: float((tmz[a] == tah(VER[a]["boz"][p], VER[a]["poz"])).mean())
               for p in (1, 2, 3)} for a in ("COMP", "ENT")}
    d = {p: deg["ENT"][p] - deg["COMP"][p] for p in (1, 2, 3)}
    ps = max(d, key=d.get)
    return dict(adim=adim, d={str(k): v for k, v in d.items()}, p_yildiz=ps,
                sinyal=d[ps], inv0=deg["ENT"][ps],
                comp=float((tmz["COMP"] == VER["COMP"]["gold"]).mean()))


def asamaB(adim, ps, inv0, acc0):
    yukle(A4, adim)
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

    # ------------------------------------------------- FAZ 1: A4 + arama
    if st["faz"] == 1:
        nxt = next((a for a in DURAKLAR if a > st["adim"]), None)
        if nxt is None or st["adim"] >= HEDEF_SON:
            if not st.get("atesledi"):
                K.log("!! A4 boyunca ATESLEMEDI -> PROSEDUR BASARISIZ")
                K.kaydet(faz=3, faz_ad="BITTI — arama atesleyemedi")
                break
            K.kaydet(faz=2, faz_ad="FAZ 2 — D3.2 sifirdan, maskeli")
            continue
        K.kaydet(faz_ad=f"FAZ 1 — A4 maskesiz (kontrol + arama), {nxt} adima")
        sn = K.egit(nxt, alt=A4)
        K.kaydet(adim=nxt)

        a = asamaA(nxt)
        st.setdefault("asamaA", []).append(a)
        ham(f"asamaA_{nxt:06d}", a)
        gec = (a["sinyal"] >= ESIK) and not st.get("atesledi")
        K.log(f"A4 {nxt:6d} ({sn} sn)  comp {a['comp']:.3f}  p*={a['p_yildiz']}  "
              f"sinyal {a['sinyal']:+.4f}   {'>>> ESIK GECILDI' if gec else ''}")
        K.not_(f"A4 {nxt}/{HEDEF_SON}   esik {ESIK}   "
               f"son sinyal {a['sinyal']:+.4f} (p*={a['p_yildiz']})",
               (f"ATESLEDI adim {st['asamaB']['adim']} -> PARCA {st['secilen']}"
                if st.get("atesledi") else "henuz atesle(n)medi"))

        if gec:
            b = asamaB(nxt, a["p_yildiz"], a["inv0"], a["comp"])
            ham(f"asamaB_{nxt:06d}", b)
            K.kaydet(atesledi=True, secilen=b["secilen_parca"],
                     asamaB={k: b[k] for k in ("adim", "kazanan", "secilen_parca")})
            K.log(f"  ASAMA B -> PARCA {b['secilen_parca']} "
                  f"(skor {b['kazanan']['skor']:+.3f})  atesleme {nxt}")
            K.log(f"  maske olacak: poz {b['secilen_parca']}, bloklar {MBLK}")
            K.log(f"  A4 DURMUYOR — kontrol kolu olarak {HEDEF_SON}'e devam eder")
            if nxt > ATES_SINIR:
                K.log(f"  !! PENCERE KIRLENMESI RISKI: atesleme {nxt} > {ATES_SINIR}"
                      f"  (D3.2 sifirdan basladigi icin ETKILEMEZ; A4 icin not)")

    # ------------------------------------------------- FAZ 2: D3.2 sifirdan
    elif st["faz"] == 2:
        mk = str(st["secilen"])
        ek = dict(MASK_KEY=mk, MASK_BLK=MBLK)
        d32 = st.get("d32_adim", 0)
        ara = min(10000, HEDEF_SON)          # once KISA adim: konfigi ERKEN dogrula
        if d32 < ara:
            K.log(f"FAZ 2 — D3.2 SIFIRDAN basliyor (ayni tohum, bit-ayni baslangic), "
                  f"maske poz {mk} bloklar {MBLK}")
            sn = K.egit(ara, ek, alt=D32)
            K.kaydet(d32_adim=ara)
            K.log(f"  D3.2 ara durak {ara} ({sn} sn)")
            K.konfig_kapisi(dict(MASK_KEY=mk, MASK_BLK=list(DERINLIK)), alt=D32)
        K.kaydet(faz_ad=f"FAZ 2 — D3.2 maskeli (parca {mk}), {HEDEF_SON} adima")
        sn = K.egit(HEDEF_SON, ek, alt=D32)
        K.kaydet(d32_adim=HEDEF_SON, faz=3, faz_ad="BITTI")
        K.log(f"D3.2 bitti ({sn} sn) -> {HEDEF_SON}")
        K.konfig_kapisi(dict(MASK_KEY=mk, MASK_BLK=list(DERINLIK)), alt=D32)

    # ------------------------------------------------- BITTI
    else:
        h = lambda alt: sum(os.path.exists(
            K.y(alt, f"snap_{KON['KOL']}_s{KON['SEED']}_{a:06d}.pt"))
            for a in KON["PENCERE"])
        K.log(f"BITTI.  pencere noktalari: A4 {h(A4)}/5   D3.2 {h(D32)}/5")
        K.log(f"  birincil okuma:  python sablon/pencere.py --cikti sonuc.json \\")
        K.log(f"     --kol 'A4:{K.y(A4)}:60000,65000,70000,75000,80000:yok' \\")
        K.log(f"     --kol 'D3.2:{K.y(D32)}:60000,65000,70000,75000,80000:"
              f"{st['secilen']}@{DERINLIK[0]}-{DERINLIK[-1]}'")
        K.not_(f"BITTI — A4 {h(A4)}/5, D3.2 {h(D32)}/5 pencere noktasi hazir")
        break

K.log("SURUCU BITTI")
