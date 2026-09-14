# ============================================================================
# HUCRE 3 - REFERANSLARI BU HATTAN OLC (D ve A, 60-80 bin penceresi)
# Sabit sayi (0.322) kullanmiyoruz: kiyasin iki tarafi ayni koddan gecsin.
# ============================================================================
PENCERE = [60000, 65000, 70000, 75000, 80000]
REFERANS = {}
for ad, (kl, mk) in REF.items():
    sn = snaplar(kl)
    var = [a for a in PENCERE if a in sn]
    if len(var) < 5:
        print(f"{ad}: {len(var)}/5 nokta -> atlandi ({kl})"); continue
    tek = [olc_snap(sn[a], mk)["ENT"] for a in var]
    o = olc_ortalama([sn[a] for a in var], mk)
    REFERANS[ad] = dict(tek=tek, ort=o, maske=list(mk))
    print(f"{ad:>3s} (maske {mk or 'yok'}): tek {[f'{v:.3f}' for v in tek]}  "
          f"ORT ENT {o['ENT']:.3f}  kis {o['ENT_kis']:.3f}  COMP {o['COMP']:.3f}  "
          f"ENT2 {o['ENT2']:.3f}")

REF_D = REFERANS["D"]["ort"]["ENT"]
print(f"\nREF_D (bu defterin olcumu) = {REF_D:.3f}")
print(f"onceki defterin olcumu      = 0.322   fark {abs(REF_D-0.322):.3f}")
if abs(REF_D - 0.322) > 0.02:
    print("!! FARK BUYUK - iki defterin olcum kumeleri ayni degil.")
    print("   Hukum BU defterin REF_D'sine gore verilir; belgeye bu not dusulur.")
json.dump({k: {"tek": v["tek"], "ort": v["ort"], "maske": v["maske"]}
           for k, v in REFERANS.items()}, open(f"{OUT}/referans_D_A.json", "w"), indent=1)
os.system(f"cp -f {OUT}/referans_D_A.json {EV}/ 2>/dev/null")
