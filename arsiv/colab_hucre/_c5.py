# ============================================================================
# HUCRE 5 - DURUM + EGITIMIN MASKESI GERCEKTEN ACIK MI
# ============================================================================
import os, re, time
print(open("/content/surucu9.txt").read() if os.path.exists("/content/surucu9.txt") else "")
nb = "/content/yedek_nabiz9"
print("nabiz:", f"{time.time()-os.path.getmtime(nb):.0f} sn once"
      if os.path.exists(nb) else "yok")
y = "/content/log_D3.txt"
if os.path.exists(y):
    L = open(y).read().splitlines()
    for l in [x.rstrip()[:140] for x in L if re.match(r"^\s+\d+\s+loss", x)][-3:]: print("  ", l)
    for l in [x.rstrip()[:100] for x in L if re.match(r"^\s+adim\s+\d+", x)][-1:]: print("  ", l)

sn = snaplar(f"{OUT}/D3"); print("\nanlik goruntuler:", list(sn.keys())[-6:])

# --- EGITIM MASKESI: KAYITLI KONFIGDEN (hukum veren kontrol) -------------
_sp = glob.glob(f"{OUT}/D3/surdur_*.pt")
if _sp:
    _c = torch.load(_sp[0], map_location="cpu", weights_only=False).get("cfg", {})
    print(f"\nEGITIM KONFIGI  MASK_KEY={_c.get('MASK_KEY')!r}  "
          f"MASK_BLK={_c.get('MASK_BLK')!r}")
    assert _c.get("MASK_KEY") == "1" and tuple(_c.get("MASK_BLK") or ()) == MASKE_D3, \
        "EGITIM YANLIS MASKEYLE KOSUYOR -> DURDUR"
    print("   TAMAM: egitim D3 maskesiyle kosuyor")
else:
    print("\n(surdurme paketi henuz yok - ilk olcumden sonra kontrol edilecek)")

# --- ERKEN UYARI: D3'u D ile AYNI ADIMDA kiyasla, ta basindan ----------
# Amac: yanlisi 120.000'de degil, 20.000'de gormek. Bunlar UYARIDIR;
# birincil hukum yine 60-80 bin penceresinden verilir (ONKAYIT, degismez).
# Esikler kosudan ONCE yazildi:
#   SAGLIK   comp(D3)  >= comp(D) - 0.10      2 ardisik noktada ihlal -> bak
#   DOLANMA  kisayol(D3) <= 0.15              maske dolanilmis olabilir
#   YOL      ent(D3)   >= ent(D)  - 0.05      yetenek hic gelismiyorsa
_e3 = f"{OUT}/D3/egri_A_s0.json"
_ed = f"{REF['D'][0]}/egri_A_s0.json"
if os.path.exists(_e3) and os.path.exists(_ed):
    E3 = {r["step"]: r for r in json.load(open(_e3))}
    ED = {r["step"]: r for r in json.load(open(_ed))}
    ort = [s for s in sorted(E3) if s in ED]
    print(f"\nERKEN UYARI  (esikler kosudan once yazildi)")
    print(f"{'adim':>7s} {'compD3':>7s} {'compD':>6s} {'d':>7s} | {'entD3':>6s} "
          f"{'entD':>6s} | {'kisD3':>6s} {'kisD':>6s} | durum")
    _ihlal = []
    for s_ in ort[-8:]:
        a, b = E3[s_], ED[s_]
        dc = a["comp"] - b["comp"]
        u = []
        if dc < -0.10: u.append("SAGLIK")
        if a["ent_shortcut"] > 0.15: u.append("DOLANMA")
        if a["ent"] < b["ent"] - 0.05: u.append("YOL")
        if u: _ihlal.append((s_, u))
        print(f"{s_:7d} {a['comp']:7.3f} {b['comp']:6.3f} {dc:+7.3f} | "
              f"{a['ent']:6.3f} {b['ent']:6.3f} | {a['ent_shortcut']:6.3f} "
              f"{b['ent_shortcut']:6.3f} | {','.join(u) if u else 'tamam'}")
    _ard = [x for k, x in enumerate(_ihlal)
            if k and _ihlal[k-1][0] in ort and ort.index(x[0]) - ort.index(_ihlal[k-1][0]) == 1]
    print("  ->", f"{len(_ihlal)} noktada uyari, {len(_ard)} tanesi ardisik"
          if _ihlal else "butun kontrollerden gecti")
    if _ard:
        print("  !! ARDISIK IHLAL: kosuyu kesmeyi dusun, sonuca kadar bekleme")
    _p = [s_ for s_ in (60000, 65000, 70000, 75000, 80000) if s_ in E3]
    print(f"  birincil okuma icin gereken 5 nokta: {len(_p)}/5 hazir  "
          f"(80.000'de okunabilir, ~57 dk)")

# --- davranissal kiyas: BILGI AMACLI, hukum vermez ----------------------
_eg = f"{OUT}/D3/egri_A_s0.json"
if sn and os.path.exists(_eg):
    E = {r["step"]: r for r in json.load(open(_eg))}
    a = max(k for k in sn if k in E)
    print(f"  bilgi: adim {a}  egri {E[a]['ent']:.3f} | maskeli "
          f"{olc_snap(sn[a], MASKE_D3)['ENT']:.3f} | maskesiz "
          f"{olc_snap(sn[a], ())['ENT']:.3f}")
