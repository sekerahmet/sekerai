# ============================================================================
# HUCRE 6 - ONCEDEN KAYITLI OKUMA   60.000-80.000, 5 nokta
# Kural ONKAYIT_D3_ZEMIN_KAT.md'de kosudan once yazildi. DEGISTIRILMEZ.
# ============================================================================
BAND = 0.05
sn = snaplar(f"{OUT}/D3")
var = [a for a in PENCERE if a in sn]
if len(var) < 5:
    print(f"{len(var)}/5 nokta hazir {var} -> HENUZ ERKEN (5 nokta sart)")
else:
    tek = [olc_snap(sn[a], MASKE_D3)["ENT"] for a in var]
    o = olc_ortalama([sn[a] for a in var], MASKE_D3)
    v, dD = o["ENT"], o["ENT"] - REF_D
    hkm = ("DERINLIK ONEMSIZ (esdegerlik)" if abs(dD) < BAND else
           "GENIS DAHA IYI" if dD >= BAND else "GENIS DAHA KOTU")
    print(f"D3 tek noktalar {[f'{x:.3f}' for x in tek]}   en iyi {max(tek):.3f}")
    print(f"D3 ORTALAMA  ENT {v:.3f}  kisayol {o['ENT_kis']:.3f}  COMP {o['COMP']:.3f}  "
          f"ENT2 {o['ENT2']:.3f}  1hop {o['1hop']:.3f}")
    print(f"referans  D {REF_D:.3f}  (ayni hattan)   A "
          f"{REFERANS.get('A', {}).get('ort', {}).get('ENT', float('nan')):.3f}")
    print(f"\nBIRINCIL HUKUM: {hkm}   (D3 - D = {dD:+.3f}, band +/-{BAND})")
    e2 = o["ENT2"] - REFERANS["D"]["ort"]["ENT2"]
    print(f"OZGULLUK: ENT2 farki {e2:+.3f}  -> "
          f"{'TAMAM' if e2 < 0.05 else 'IHLAL - sonuc gecersiz'}")
    print(f"SAGLIK  : COMP {o['COMP']:.3f} (>=0.80)  1hop {o['1hop']:.3f} (>=0.98)")
    print(f"DOLANMA : kisayol {o['ENT_kis']:.3f}  (D: {REFERANS['D']['ort']['ENT_kis']:.3f})")
    json.dump(dict(pencere=var, tek=tek, ortalama=o, REF_D=REF_D, hukum=hkm,
                   maske=list(MASKE_D3)), open(f"{OUT}/onkayit_D3.json", "w"), indent=1)
    os.system(f"cp -f {OUT}/onkayit_D3.json {EV}/ 2>/dev/null")
