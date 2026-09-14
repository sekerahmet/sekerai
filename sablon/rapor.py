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
# Coklu kollu kosuda ilerleme TOPLAM is uzerinden gosterilmeli; tek kolun
# hedefini kullanmak ilk kol bitince "%100" yaziyordu.
hedef = kon.get("RAPOR_TOPLAM") or kon.get("HEDEF_SON", 0)
KOL, SEED = kon.get("KOL", "A"), kon.get("SEED", 0)

# Bir kosuda BIRDEN COK kol olabilir (D3.2: A4 kontrolu + D3.2 maskeli kolu).
# Klasorleri diskten bul; "cikti" sabitini yazmak d32'de raporu BOS birakiyordu.
altlar = sorted(os.path.basename(p) for p in glob.glob(P("cikti*"))
                if os.path.isdir(p)) or ["cikti"]
egriler = {a: (jy(P(a, f"egri_{KOL}_s{SEED}.json"), []) or []) for a in altlar}

# Kiyas sutunu: ayni kosudaki KONTROL kolu varsa o, yoksa disaridan REF_EGRI.
kalt = kon.get("KONTROL_ALT")
if kalt and egriler.get(kalt):
    ref = {r["step"]: r for r in egriler[kalt]}
    ref_ad = kon.get("REF_AD") or kalt
else:
    ref = {r["step"]: r for r in (jy(kon.get("REF_EGRI", ""), []) or [])}
    ref_ad = kon.get("REF_AD") or os.path.basename(
        os.path.dirname(kon.get("REF_EGRI", "") or "")) or "REF"

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

# ---- 2) EGITIM EGRILERI (her kol icin ayri tablo) ---------------------------
U = kon.get("UYARI", {})
# Uyari FAZA bagli: atesleme oncesi satirlar maskesiz, sonrasi maskeli.
at = (st.get("asamaB") or {}).get("adim")
for alt in altlar:
    egri = egriler.get(alt) or []
    if not egri:
        continue
    bu = alt if alt != "cikti" else D.upper()
    kiyas = (alt != kalt) and bool(ref)      # kontrol kolunun kendine kiyasi olmaz
    print(f"\n EGRI [{bu}]   son {min(len(egri),10)} olcum")
    print(f" {'':7}{('|------- ' + bu + ' — bu kol -------|'):^39}"
          + (f"{('|-- ' + ref_ad + ' — kontrol --|'):^18}" if kiyas else ""))
    # `seen` = EGITIMDE gorulmus 2-hop zincirlerdeki dogruluk. Ezber ile
    # genellemeyi ayiran sutun bu: seen yuksek + comp dusuk = ezberliyor ama
    # genellemiyor; seen de dusuk = ezberleyemiyor bile (kapasite).
    # ENT2 slotu bos olan kosularda (VERI=... modu) yerine ENT-YOK basilir:
    # gomulu kontrol deney G'nin TEK ayirt edici olcusu ve rapor onu hic
    # gostermiyordu -- `ent2` sutunu saatlerce `nan` basiyordu. Rapor
    # kosu KILITLIYKEN bakilabilsin diye var (CLAUDE.md 7); goremedigimiz
    # sey icin var olmasinin anlami yok.
    _yok = any("ent_yok" in r for r in egri)
    _u4 = "ENTYOK" if _yok else "ent2"
    _a4 = (lambda r: r.get("ent_yok", float("nan"))) if _yok else           (lambda r: r.get("ent2", float("nan")))
    print(f" {'adim':>7} {'1hop':>6} {'seen':>6} {'comp':>6} {'ent':>6} {_u4:>6}"
          f" {'kisayol':>7}"
          + (f" |{'comp':>8} {'ent':>7} | durum" if kiyas else ""))
    for r in egri[-10:]:
        b = ref.get(r["step"], {}) if kiyas else {}
        f = 1 if (at is None or r["step"] <= at) else 2
        # yon: -1 referansin ALTINA duserse, +1 USTUNE cikarsa, 0 iki yonde de
        u = []
        for ad_, (alan, tol, yon, fz) in U.items():
            if not b or fz != f or alan not in r or alan not in b:
                continue
            d_ = r[alan] - b[alan]
            if (d_ < -tol) if yon < 0 else (d_ > tol) if yon > 0 else abs(d_) > tol:
                u.append(ad_)
        print(f" {r['step']:7d} {r['one']:6.3f} {r.get('seen', float('nan')):6.3f}"
              f" {r['comp']:6.3f} {r['ent']:6.3f}"
              f" {_a4(r):6.3f} {r['ent_shortcut']:7.3f}"
              + (f" |{b.get('comp', float('nan')):8.3f} "
                 f"{b.get('ent', float('nan')):7.3f} | F{f} "
                 f"{','.join(u) if u else 'tamam'}" if kiyas else "")
              + ("   <<< MASKE ACILDI" if at == r["step"] else ""))

# ---- 3) GERI DONULEBILIRLIK ------------------------------------------------
ham = sorted(glob.glob(P("ham", "*")))
print(f"\n KAYIT   (CLAUDE.md 9 — 'yapacagin analiz henuz icat edilmedi')")
pen = kon.get("PENCERE", [])
for alt in altlar:
    # tek kollu eski kosular sur/ altina duz yaziyordu, coklu kol sur/<alt>/
    sur = sorted(glob.glob(P("sur", alt, "*.pt"))
                 + (glob.glob(P("sur", "*.pt")) if alt == "cikti" else []))
    snap = sorted(glob.glob(P(alt, "snap_*.pt")))
    bu = alt if alt != "cikti" else D.upper()
    print(f"   [{bu}] surdurme ADIM ADLI: {len(sur):3d} paket"
          + (f"   son {', '.join(os.path.basename(p).split('_')[-1][:-3] for p in sur[-3:])}"
             if sur else "   !! YOK — gecmis bir adima DONULEMEZ")
          + f"   |  anlik goruntu {len(snap):3d}"
          + (f"   |  pencere {sum(os.path.exists(P(alt, f'snap_{KOL}_s{SEED}_{a:06d}.pt')) for a in pen)}/{len(pen)}"
             if pen else ""))
print(f"   ham analiz ciktisi  : {len(ham):3d} dosya")

# ---- 3b) DOYMA -- BUTCE FAZLA MI? (CLAUDE.md 8) ----------------------------
# Adim butcesi onceki deneyden KOPYALANIYOR ve kimse "gerekli miydi" diye
# sormuyor: 120.000 D3.3'ten G'ye gecti, oysa D3.3'un kendi egrisi 20.000'de
# doymustu. Insanin fark etmesini bekleme -- rapor her turda soylesin.
def _doyma(eg, alan, tol=0.02):
    """Alanin SON degerine %tol icinde girip BIR DAHA cikmadigi ILK adim."""
    v = [(x["step"], x[alan]) for x in eg if alan in x and x[alan] is not None]
    if len(v) < 4:
        return None
    son = v[-1][1]
    esik = max(abs(son) * tol, 0.005)
    for j in range(len(v)):
        if all(abs(w - son) <= esik for _, w in v[j:]):
            return v[j][0]
    return v[-1][0]


for _a in sorted(glob.glob(P("cikti*"))):
    _eg = jy(os.path.join(_a, f"egri_{KOL}_s{SEED}.json"), [])
    if len(_eg) < 4:
        continue
    _sn = _eg[-1]["step"]
    _d = {k: _doyma(_eg, k) for k in ("comp", "ent", "ent_yok", "ent_shortcut")
          if k in _eg[-1] and _eg[-1][k] is not None}
    if not _d:
        continue
    _en = max(v for v in _d.values() if v)
    print(chr(10) + f"DOYMA [{os.path.basename(_a)}]   son adim {_sn}")
    print("   " + "   ".join(f"{k} {v}" for k, v in _d.items()))
    if _en < 0.6 * _sn:
        print(f"   !! BUTCE FAZLA: butun alanlar {_en} adimda doymus, kosu "
              f"{_sn}'e kadar surmus.")
        print(f"      ~%{100*(_sn-_en)/_sn:.0f}'i ayni platoyu olcuyor. "
              f"Yeni deneyde BUTCE = doyma + pencere (CLAUDE.md 8).")
    else:
        print(f"   butce makul: doyma {_en}, son {_sn}")


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
