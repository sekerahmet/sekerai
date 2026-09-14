# -*- coding: utf-8 -*-
"""D3.3 ARAMA -- maskeleme yerini A5'in kontrol noktalarindan BUL.

NEDEN AYRI BIR BETIK: d33.py ilk halinde D5'e D3'un ELLE bulunmus yerini
(poz 1) veriyordu. Oysa tezin zinciri su:

    D3    yeri tersine muhendislikle bulduk         -> A'yi gecti
    D3.1  yeri ARAYARAK da bulabiliyoruz            -> dogrulandi
    ->    o halde D3.3'te de yeri ARAMA belirlemeli, elle degil.

Yoksa "phi degisince mekanizma da degisti mi" sorusunu, mekanizmanin
degismedigini VARSAYARAK cevaplamis oluruz.

PROSEDUR D3.1'IN AYNISI, tek fark: arama A5'in DISKTEKI anlik
goruntulerinde kosuyor, egitimin icinde degil. Boylece A5'in yorungesine
dokunulmaz ve D5 bulunan yerle SIFIRDAN baslar (tezin 4. adimi:
"yeri bilince, ayni tohum ve veriyle baslangictan kosarsan").

ONCEDEN YAZILAN PARAMETRELER -- hepsi D3.1'in onkaydindan, HICBIRI
bu kosunun verisine bakilarak secilmedi:

    ESIK     0.03                  A ve E2'nin iki kosusundan
    DURAKLAR 5.000..80.000 / 5.000
    NA       2000                  olculen sd 0.0036
    DERINLIK 1..L-1                D3'te sabitlendi
    ATES_SINIR  55.000             asilirsa birincil pencere kirlenir

    python d33_arama.py <konfig_giris.json>
"""
import os, sys, json, glob

KON = json.load(open(sys.argv[1]))
sys.path.insert(0, KON["KOD"])
sys.path.insert(0, os.path.join(KON["KOD"], "sablon"))

for _v in ("MASK_KEY", "MASK_BLK", "RESUME_FROM", "INIT_FROM", "OUT",
           "SHARE", "IDENT_MODE", "STEPS", "WARM_OF"):
    os.environ.pop(_v, None)
SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    KON["ORT"] = dict(KON["ORT"], PRESET="smoke", MEM_AT="2", EVERY="200",
                      COMPILE="0", N_ENT="200", N_REL="6", N_PAIR="20",
                      P_TRAIN="16")
assert not (set(KON["ORT"]) & {"MASK_KEY", "MASK_BLK", "RESUME_FROM",
                               "INIT_FROM", "OUT", "STEPS"}), \
    f"ORT kosuya ozel degisken tasiyor: {KON['ORT']}"
os.environ.update(KON["ORT"])

import numpy as np
import torch
import sifirdan as S

ALT = os.environ.get("ARAMA_ALT", "cikti_a5")     # hangi kolun goruntuleri
C = KON["CALIS"]
KOL, SEED = KON["KOL"], KON["SEED"]

# --- ONCEDEN YAZILAN (D3.1 onkaydi, degistirilmedi) ---------------------
ESIK = 0.03
DURAKLAR = list(range(5000, 85000, 5000))
NA = 2000
DERINLIK = tuple(range(1, S.CFG["L"]))
ATES_SINIR = 55000
if SMOKE:
    ESIK, DURAKLAR, NA, ATES_SINIR = -1.0, [200, 400], 64, 10 ** 9
MBLK = ",".join(str(b) for b in DERINLIK)

print(f"D3.3 ARAMA   kol klasoru {ALT}   esik {ESIK}   NA {NA}")
print(f"  duraklar {DURAKLAR[0]}..{DURAKLAR[-1]}   derinlik "
      f"{DERINLIK[0]}..{DERINLIK[-1]}   atesleme siniri {ATES_SINIR}")

# ------------------------------------------------------------- OLCME SETI
facts, pairs, one, tr2, comp_, ent_ev, seen_ent, unseen_ent, ent2_ev = S.build_data()
lo, hi = S.ENT_OFF, S.ENT_OFF + S.CFG["N_ENT"]
rg = np.random.RandomState(0)          # SABIT ornek: duraklar arasi kiyas icin
VER = {}
for ad, lst in (("COMP", comp_[:NA]), ("ENT", ent_ev[:NA])):
    E = S.enc_two(lst)
    VER[ad] = dict(X=E[0], poz=E[1],
                   gold=(S.ENT_OFF + np.array([x[4] for x in lst])).astype(np.int64),
                   boz={})
    for p in (1, 2, 3):                # 1=varlik, 2=r1, 3=r2
        X = E[0].copy()
        off, N = (S.ENT_OFF, S.CFG["N_ENT"]) if p == 1 else (S.REL_OFF, S.CFG["N_REL"])
        cur = X[:, p] - off
        y = rg.randint(N - 1, size=len(X)); y += (y >= cur)
        X[:, p] = off + y
        VER[ad]["boz"][p] = X
print(f"  olcme seti: COMP {len(VER['COMP']['X'])}  ENT {len(VER['ENT']['X'])}")

net = S.Net(KOL, S.CFG).to(S.DEV)


def yukle(adim):
    y = os.path.join(C, ALT, f"snap_{KOL}_s{SEED}_{adim:06d}.pt")
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
    """UCUZ: hangi girdi parcasi ENT'te COMP'tan FAZLA degismezlik veriyor."""
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
    """PAHALI, BIR KEZ: 3 parca x L blok-kuyrugu. Skor = r1 bagimsizligini
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
    # CLAUDE.md 7: butun adaylar tam +0.000 verdiyse bulgu "etki yok" degil
    # "olcum bozuk".
    assert etki, "ASAMA B: maske HIC etki etmedi -> olcum bozuk"
    kaz = max(tablo, key=lambda r: r["skor"])
    return dict(adim=adim, tablo=tablo, kazanan=kaz, secilen_parca=kaz["p"])


# ================================================================= TARAMA
var = sorted(int(os.path.basename(p).split("_")[-1][:-3])
             for p in glob.glob(os.path.join(C, ALT, f"snap_{KOL}_s{SEED}_*.pt")))
hedef = [a for a in DURAKLAR if a in var]
print(f"  diskte {len(var)} anlik goruntu, duraklarla eslesen {len(hedef)}: "
      f"{hedef[:3]}..{hedef[-1] if hedef else '-'}\n")

sonuc = dict(alt=ALT, esik=ESIK, duraklar=DURAKLAR, NA=NA,
             derinlik=list(DERINLIK), ates_siniri=ATES_SINIR,
             asamaA=[], atesledi=False)
print(f"{'adim':>7} {'comp':>7} {'p*':>3} {'sinyal':>8}   "
      f"{'d(p=1)':>8} {'d(p=2)':>8} {'d(p=3)':>8}")
# TANI MODU: her durakta ASAMA B'yi de kos, KARAR VERME. D3.1'in
# proseduru ilk esik gecisinde durur; burada kazananin ve SKORUN adimla
# nasil degistigini gormek istiyoruz. Karar bu tablodan SONRA, ayrica.
TAM = os.environ.get("ARAMA_TAM") == "1"
if TAM:
    print("  [TANI MODU] her durakta ASAMA B de kosulur, prosedur DURMAZ\n")
for adim in hedef:
    a = asamaA(adim)
    sonuc["asamaA"].append(a)
    print(f"{adim:7d} {a['comp']:7.3f} {a['p_yildiz']:3d} {a['sinyal']:+8.4f}   "
          f"{a['d']['1']:+8.4f} {a['d']['2']:+8.4f} {a['d']['3']:+8.4f}"
          + ("   >>> ESIK GECILDI" if a["sinyal"] >= ESIK else ""))
    if TAM:
        b = asamaB(adim, a["p_yildiz"], a["inv0"], a["comp"])
        sonuc.setdefault("tani", []).append(
            dict(adim=adim, comp=a["comp"], sinyal=a["sinyal"],
                 p_yildiz=a["p_yildiz"], kazanan=b["kazanan"]))
        k = b["kazanan"]
        if os.environ.get("ARAMA_TABLO") == "1":
            # TAM TABLO: kazanani okumak yetmez. "b0=0 neden hic secilmiyor"
            # sorusu ancak butun adaylari gorunce cevaplanir.
            print(f"        --- adim {adim}: 24 adayin hepsi "
                  f"(inv0={a['inv0']:.3f} comp0={a['comp']:.3f})")
            for p in (1, 2, 3):
                sat = "          p=%d  " % p
                for r in [x for x in b["tablo"] if x["p"] == p]:
                    sat += f"b{r['b0']}:{r['skor']:+.3f} "
                print(sat)
        print(f"        ASAMA B -> parca {k['p']}  b0={k['b0']}  "
              f"skor {k['skor']:+.4f}"
              + ("   (D3.1 kazanani: parca 1, skor +0.0565)" if adim == hedef[0] else ""))
        continue
    if a["sinyal"] >= ESIK:
        b = asamaB(adim, a["p_yildiz"], a["inv0"], a["comp"])
        sonuc.update(atesledi=True, atesleme_adimi=adim,
                     secilen_parca=b["secilen_parca"],
                     kazanan=b["kazanan"], asamaB_tablo=b["tablo"],
                     MASK_KEY=str(b["secilen_parca"]), MASK_BLK=MBLK)
        print(f"\n  ASAMA B -> PARCA {b['secilen_parca']}   "
              f"skor {b['kazanan']['skor']:+.4f}   "
              f"(kazanan b0={b['kazanan']['b0']})")
        print(f"  D5 KONFIGI:  MASK_KEY={b['secilen_parca']}  MASK_BLK={MBLK}")
        print(f"  atesleme adimi {adim} "
              f"{'<= ' + str(ATES_SINIR) + ', pencere TEMIZ' if adim <= ATES_SINIR else '> ' + str(ATES_SINIR) + ' !! PENCERE KIRLENIR'}")
        break
else:
    print(f"\n  ESIK GECILMEDI (elde {len(hedef)} durak var). "
          f"Daha fazla anlik goruntu bekleniyor.")

y = os.path.join(C, "ARAMA_D33.json")
json.dump(sonuc, open(y, "w"), indent=1)
print(f"\n-> {y}")
