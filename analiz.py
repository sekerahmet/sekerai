# -*- coding: utf-8 -*-
"""
ANALIZ — kosu bittikten sonra toplanan veriyi okur.

BIRINCIL karar sifirdan.py'de verildi ve burada TEKRAR EDILMEZ. Bu dosya
yalnizca TANI uretir; buradan cikan her sey "ayrica test edilecek" listesine
gider, "bulduk" listesine degil.

Uretilenler:
  1. egriler.png        comp / kopru / cevap / bellek — kol kol
  2. devre.png          kafa ablasyon haritasi (comp dususu - 1hop dususu)
  3. katman.png         logit lens: kopru ve cevap hangi katmanda
  4. slot_testi         ASIL MEKANISTIK TEST (asagida)
  5. ozet.md            hepsinin metin ozeti

SLOT TESTI — tezin dogrudan sinavi:
  C'nin bellegi gercekten KOPRU ile adresleniyorsa, ayni kopruye sahip iki
  FARKLI soru ayni slotlari yakmali. B bunu yapisal olarak yapamaz (token'lari
  farkli). Olculen: ayni-kopru cifti slot ortusmesi vs farkli-kopru cifti.
  Kontrol sart: ayni-e ciftleri de ayri olculur, cunku ayni kopru cogu zaman
  ayni e'den gelir ve etki koprununki degil e'ninki olabilir.

Kullanim:
    python analiz.py --out /content/out2
"""
import os, sys, json, glob, argparse
import numpy as np


def yukle_tablo(out, arm, seed=0):
    """Kolon adi -> numpy dizisi. pandas gerektirmez."""
    pq = os.path.join(out, f"kayit_{arm}_s{seed}.parquet")
    nz = os.path.join(out, f"kayit_{arm}_s{seed}.npz")
    if os.path.exists(pq):
        try:
            import pandas as pd
            return {k: v.to_numpy() for k, v in pd.read_parquet(pq).items()}
        except Exception:
            pass
    if os.path.exists(nz):
        z = np.load(nz, allow_pickle=True)
        return {k: z[k] for k in z.files}
    return None


def slot_testi(df, adim=None, kume="comp", nmax=4000, seed=0):
    """AYNI KOPRU -> AYNI SLOT mu?

    Uc grup, hepsi ayni tablodan:
      ayni_kopru_farkli_e : kopru ayni, soru varligi FARKLI   <- tezin tahmini
      ayni_e              : varlik ayni (kopru de cogu zaman ayni) <- kontrol
      farkli              : ikisi de farkli                   <- taban
    Olcu: iki sorunun en cok yanan 3 slotunun kesisim buyuklugu (0..3).
    """
    if df is None or "slot1" not in df:
        return None
    km = np.asarray(df["kume"]).astype(str)
    sel = km == kume
    st = np.asarray(df["step"])
    if adim is None:
        if not sel.any():
            return None
        adim = int(st[sel].max())
    sel = sel & (st == adim)
    if sel.sum() < 50:
        return None
    idx = np.where(sel)[0]
    r = np.random.RandomState(seed)
    if len(idx) > nmax:
        idx = idx[r.permutation(len(idx))[:nmax]]
    S = np.stack([np.asarray(df[f"slot{j}"])[idx] for j in (1, 2, 3)], 1)
    e = np.asarray(df["e"])[idx]; b = np.asarray(df["kopru"])[idx]
    n = len(idx)

    def ortusme(i, j):
        return len(set(S[i]) & set(S[j]))

    gruplar = {"ayni_kopru_farkli_e": [], "ayni_e": [], "farkli": []}
    hedef = 3000
    tries = 0
    while min(len(v) for v in gruplar.values()) < hedef and tries < hedef * 200:
        tries += 1
        i, j = r.randint(n), r.randint(n)
        if i == j:
            continue
        if e[i] == e[j]:
            g = "ayni_e"
        elif b[i] == b[j]:
            g = "ayni_kopru_farkli_e"
        else:
            g = "farkli"
        if len(gruplar[g]) < hedef:
            gruplar[g].append(ortusme(i, j))
    return {k: (float(np.mean(v)), len(v)) for k, v in gruplar.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/content/out2")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT, SEED = a.out, a.seed
    sat = []
    P = lambda s: (print(s), sat.append(s))

    arms = [x for x in "ABC" if os.path.exists(
        os.path.join(OUT, f"egri_{x}_s{SEED}.json"))]
    if not arms:
        raise SystemExit(f"egri bulunamadi: {OUT}")
    egri = {x: json.load(open(os.path.join(OUT, f"egri_{x}_s{SEED}.json")))
            for x in arms}
    P(f"# ANALIZ — {OUT}  (kollar: {''.join(arms)})\n")

    # ---- 1. eğriler
    P("## Egriler (son nokta)\n")
    P("| kol | adim | comp | seen | 1hop | ent | kopru(ONEK-TUT) | kopru_L0 | cevap | belleksiz | memH |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    for x in arms:
        c = egri[x][-1]
        g = lambda k, d="-": (f"{c[k]:.3f}" if isinstance(c.get(k), (int, float))
                              and np.isfinite(c.get(k, np.nan)) else d)
        P(f"| {x} | {c['step']} | {g('comp')} | {g('seen')} | {g('one')} | {g('ent')} | "
          f"{g('ho_kopru')} | {g('ho_kopru_L0')} | {g('cevap')} | "
          f"{g('comp_acc_nomem')} | {g('mem_H')} |")

    # ---- 2. kopru sabit mi, comp tirmaniyor mu?
    P("\n## Kopru okunabilirligi vs comp dogrulugu\n")
    for x in arms:
        cs = [c["comp"] for c in egri[x]]
        ks = [c.get("ho_kopru", float("nan")) for c in egri[x]]
        P(f"- **kol {x}**: comp {cs[0]:.3f} -> {cs[-1]:.3f} "
          f"({cs[-1]/max(cs[0],1e-9):.1f}x) | kopru {ks[0]:.3f} -> {ks[-1]:.3f}")
    P("\n> Kopru yatay kalip comp tirmaniyorsa: ogrenilen sey kopruyu BULMAK "
      "degil, onu KULLANMAK.")

    # ---- 3. bellek ablasyonu
    P("\n## Bellek ablasyonu (bellegi kapatinca comp)\n")
    for x in arms:
        c = egri[x][-1]
        if np.isfinite(c.get("comp_acc_nomem", np.nan)):
            P(f"- kol {x}: {c['comp']:.3f} -> {c['comp_acc_nomem']:.3f} "
              f"(dusus {c['comp']-c['comp_acc_nomem']:+.3f})")
    P("\n> Dusus ~0 ise bellek TASIYICI degil sus; model isi transformer'da yapiyor.")

    # ---- 4. kafa ablasyonu
    P("\n## Besteleme devresi (kafa ablasyonu)\n")
    for x in arms:
        f = os.path.join(OUT, f"kafa_{x}_s{SEED}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f); dc, d1 = z["dusus_comp"], z["dusus_1hop"]
        sec = dc - d1
        top = np.dstack(np.unravel_index(np.argsort(-sec, axis=None)[:5], sec.shape))[0]
        P(f"\n**kol {x}** (taban comp {float(z['taban_comp']):.3f}):\n")
        P("| kafa | comp dususu | 1hop dususu | fark |")
        P("|---|---|---|---|")
        for l, h in top:
            P(f"| K{l}H{h} | {dc[l,h]:+.3f} | {d1[l,h]:+.3f} | {sec[l,h]:+.3f} |")

    # ---- 5. logit lens
    P("\n## Logit lens — kopru ve cevap hangi katmanda\n")
    for x in arms:
        c = egri[x][-1]
        if "lens_kopru" in c:
            lk, lg = c["lens_kopru"], c["lens_gold"]
            P(f"\n**kol {x}** (0=tepe, 0.5=sans):\n")
            P("| katman | " + " | ".join(str(i) for i in range(len(lk))) + " |")
            P("|" + "---|" * (len(lk) + 1))
            P("| kopru | " + " | ".join(f"{v:.3f}" for v in lk) + " |")
            P("| cevap | " + " | ".join(f"{v:.3f}" for v in lg) + " |")

    # ---- 6. SLOT TESTI
    P("\n## SLOT TESTI — ayni kopru ayni slotu mu yakiyor?\n")
    P("| kol | ayni kopru/farkli e | ayni e | farkli (taban) | fark |")
    P("|---|---|---|---|---|")
    for x in arms:
        df = yukle_tablo(OUT, x, SEED)
        if df is None:
            continue
        st = slot_testi(df)
        if not st:
            P(f"| {x} | (bellek yok) | | | |")
            continue
        ak = st.get("ayni_kopru_farkli_e", (float('nan'), 0))[0]
        ae = st.get("ayni_e", (float('nan'), 0))[0]
        fk = st.get("farkli", (float('nan'), 0))[0]
        P(f"| {x} | {ak:.3f} | {ae:.3f} | {fk:.3f} | {ak-fk:+.3f} |")
    P("\n> C'de 'ayni kopru/farkli e' tabandan belirgin yuksekse ve B'de degilse:\n"
      "> bellek gercekten KOPRU ile adresleniyor. Bu tezin dogrudan kaniti olur.\n"
      "> 'ayni e' sutunu kontrol: etki koprunun mu, yoksa sadece ayni varliktan\n"
      "> gelmenin mi oldugunu ayirir.")

    # ---- grafikler
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for x in arms:
            st = [c["step"] for c in egri[x]]
            ax[0].plot(st, [c["comp"] for c in egri[x]], marker="o", label=f"kol {x}")
            ax[1].plot(st, [c.get("ho_kopru", np.nan) for c in egri[x]],
                       marker="o", label=f"{x} kopru")
            ax[1].plot(st, [c.get("ho_kopru_L0", np.nan) for c in egri[x]],
                       ls="--", alpha=.5, label=f"{x} L0")
            ax[2].plot(st, [c.get("mem_H", np.nan) for c in egri[x]],
                       marker="o", label=f"kol {x}")
        ax[0].set_title("COMP dogrulugu"); ax[0].set_xlabel("adim")
        ax[1].set_title("kopru okunabilirligi (0=tepe, .5=sans)")
        ax[1].axhline(.5, color="k", lw=.5)
        ax[2].set_title("bellek entropisi")
        for A in ax:
            A.legend(fontsize=7); A.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(os.path.join(OUT, "egriler.png"), dpi=110)
        P(f"\ngrafik -> {OUT}/egriler.png")
    except Exception as ex:
        P(f"\n(grafik atlandi: {ex})")

    open(os.path.join(OUT, "ozet.md"), "w", encoding="utf-8").write("\n".join(sat))
    print(f"\nozet -> {OUT}/ozet.md")


if __name__ == "__main__":
    main()
