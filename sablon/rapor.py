"""RAPOR — kosunun o anki hali, tek metin dosyasi.

Bakici bunu 5 dk'da bir calistirir; cikti CALIS/RAPOR.txt ve Drive'a kopyalanir.
HICBIR SEY HESAPLAMAZ, sadece diskte olani okur -> cekirdek mesgulken de
okunabilir, GPU kullanmaz, kosuyu yavaslatmaz.

Deneye OZEL satirlar surucunun durum.json'a yazdigi "rapor_ek" listesinden
gelir; bu dosya onlari oldugu gibi basar. Boylece rapor genel kalir.

    python rapor.py <CALIS>
"""
import sys, os, json, time, glob

C = sys.argv[1]
P = lambda *a: os.path.join(C, *a)
jy = lambda y, v=None: (json.load(open(y)) if os.path.exists(y) else v)


def yas(y):
    return f"{(time.time()-os.path.getmtime(y))/60:.1f} dk once" if os.path.exists(y) else "YOK"


kon = jy(P("konfig.json"), {}) or {}
st = jy(P("durum.json"), {}) or {}
D = kon.get("DENEY", "?")
hedef = kon.get("HEDEF_SON", 0)
ref = {r["step"]: r for r in (jy(kon.get("REF_EGRI", ""), []) or [])}
egri = jy(P("cikti", f"egri_{kon.get('KOL','A')}_s{kon.get('SEED',0)}.json"), []) or []

CIZ = "=" * 76
print(CIZ)
print(f" {D.upper()}   {time.strftime('%d.%m.%Y %H:%M:%S')}   commit {kon.get('commit','?')}")
print(f" {kon.get('ONKAYIT','onkayit belirtilmemis')}")
print(CIZ)

# ---- 1) NEREDE -------------------------------------------------------------
ad = st.get("adim", 0)
print(f" {st.get('faz_ad','baslamadi')}")
if hedef:
    n = int(30 * ad / hedef)
    print(f" adim {ad:6d}/{hedef}  [{'#'*n}{'.'*(30-n)}] %{100*ad/hedef:.0f}"
          f"   son guncelleme {st.get('guncelleme','-')}")
for satir in st.get("rapor_ek", []):          # deneye OZEL blok
    print(" " + satir)

# ---- 2) EGITIM EGRISI + referans kol ---------------------------------------
if egri:
    rn = os.path.basename(kon.get("REF_EGRI", "")) or "-"
    print(f"\n EGRI   son {min(len(egri),10)} olcum        referans: {rn}")
    print(f" {'adim':>7} {'1hop':>6} {'comp':>6} {'ent':>6} {'ent2':>6} {'kisayol':>7}"
          f" |{'compREF':>8} {'entREF':>7} | durum")
    U = kon.get("UYARI", {})
    # Uyari FAZA bagli: atesleme oncesi satirlar maskesiz (FAZ 1 = referansin
    # TEKRARI olmali), sonrasi maskeli (FAZ 2 = referanstan SAPMASI beklenir).
    at = (st.get("asamaB") or {}).get("adim")
    for r in egri[-10:]:
        b = ref.get(r["step"], {})
        f = 1 if (at is None or r["step"] <= at) else 2
        u = [ad_ for ad_, (alan, tol, yon, fz) in U.items()
             if b and fz == f and
             ((r[alan] < b[alan] - tol) if yon < 0 else (abs(r[alan] - b[alan]) > tol))]
        print(f" {r['step']:7d} {r['one']:6.3f} {r['comp']:6.3f} {r['ent']:6.3f}"
              f" {r.get('ent2', float('nan')):6.3f} {r['ent_shortcut']:7.3f}"
              f" |{b.get('comp', float('nan')):8.3f} {b.get('ent', float('nan')):7.3f}"
              f" | F{f} {','.join(u) if u else 'tamam'}"
              + ("   <<< MASKE ACILDI" if at == r["step"] else ""))

# ---- 3) GERI DONULEBILIRLIK ------------------------------------------------
sur = sorted(glob.glob(P("sur", "*.pt")))
snap = sorted(glob.glob(P("cikti", "snap_*.pt")))
ham = sorted(glob.glob(P("ham", "*")))
print(f"\n KAYIT   (CLAUDE.md 9 — 'yapacagin analiz henuz icat edilmedi')")
print(f"   surdurme, ADIM ADLI : {len(sur):3d} paket"
      + (f"   {', '.join(os.path.basename(p).split('_')[-1][:-3] for p in sur[-5:])}"
         if sur else "   !! YOK — gecmis bir adima DONULEMEZ"))
print(f"   anlik goruntu       : {len(snap):3d} adet")
print(f"   ham analiz ciktisi  : {len(ham):3d} dosya")
pen = kon.get("PENCERE", [])
if pen:
    v = [a for a in pen if os.path.exists(P("cikti", f"snap_{kon.get('KOL','A')}"
                                            f"_s{kon.get('SEED',0)}_{a:06d}.pt"))]
    print(f"   birincil pencere {pen[0]}-{pen[-1]} : {len(v)}/{len(pen)} nokta hazir")

# ---- 4) CANLILIK -----------------------------------------------------------
_dp = f"[{D[0]}]{D[1:]}_surucu|[s]ifirdan.py"
# 'ps | grep <betik>' kendi komut satirini yakalar -> yanlis "calisiyor" der;
# ilk harfi köşeli paranteze alarak bunu onluyoruz (CLAUDE.md 7).
ps = (os.popen(f"ps -eo etime,cmd 2>/dev/null | grep -E '{_dp}'").read().strip()
      if os.name == "posix" else "(ps yok — bu makine Colab degil)")
print(f"\n CANLILIK")
print(f"   surucu   : {ps.splitlines()[0] if ps else '!! CALISAN SUREC YOK'}")
for l in ps.splitlines()[1:]:
    print(f"              {l}")
logs = sorted(glob.glob(P("log", f"egitim_{D}_*.txt")))
if logs:
    son = [l.rstrip()[:88] for l in open(logs[-1]).read().splitlines()
           if l.strip().startswith("adim")][-2:]
    print(f"   son log  : {os.path.basename(logs[-1])}")
    for l in son:
        print(f"              {l.strip()}")
_nb = os.path.join(kon.get("EV", ""), "nabiz.txt")
print(f"   Drive    : yedek nabzi {yas(_nb)}")
print(CIZ)
