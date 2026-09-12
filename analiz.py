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

ON KAYIT-2 (HIZ) — 12 Eylul 2026, kol A kosarken, B ve C HENUZ KOSMADAN yazildi.

  Gerekce: gercek egitimde kimse yakinsamadan 10x sonrasina kadar egitmez.
  Ayni son dogruluga DAHA AZ ADIMDA varmak gercek bir kazanctir ve grokking
  literaturunun asil baktigi nicelik de genellemenin BASLAMA anidir.

  Olcu     : tau = comp >= 0.30'a ulasilan ilk degerlendirme adimi; ayrica AUC.
  Iddia esigi: C'nin tau'su B'ninkinden en az 5000 adim kucuk OLMALI, VE AUC
             farki ayni yonde olmali.
  Esik neden 0.30: A kolunun egrisinden secildi (A ~30k civarinda geciyor),
             boylece butce icinde ulasilabilir bir orta nokta. A, C-B
             karsilastirmasinin PARCASI DEGIL; dolayisiyla bu secim C-B'yi
             yanlilamaz.
  Uyari    : B ve C yalnizca bellegin okudugu tensorde farklidir, bu yuzden
             aralarindaki hiz farki adresleme mekanizmasina atfedilebilir.
             A ile kiyas ayni kontrole sahip DEGILDIR (A'nin FFN'i genis).
  Statu    : tek seed ile sonuc ON BULGU. Grokking baslangici seed'e cok
             duyarlidir; hiz iddiasi >=3 seed ister.

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
    Olcu: iki sorunun en cok yanan N slotunun kesisim buyuklugu (0..N).
    N = tabloda kac slot sutunu varsa (8.5M kosusunda 3, sonrakilerde 8).
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
    ns = len([k for k in df if k.startswith("slot") and not k.endswith("_w")
              and k != "slot_H"])
    S = np.stack([np.asarray(df[f"slot{j}"])[idx] for j in range(1, ns + 1)], 1)
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


def hiz_olcusu(egri, esik=0.30):
    """IKINCIL ON KAYIT — bkz. dosya basindaki ON KAYIT-2 notu.

    tau  : comp >= esik'e ULASILAN ILK degerlendirme adimi (yoksa None)
    auc  : comp egrisinin adim-normalize alani (surekli surum)
    """
    st = [c["step"] for c in egri]; cs = [c["comp"] for c in egri]
    tau = next((s_ for s_, v in zip(st, cs) if v >= esik), None)
    auc = float(np.trapezoid(cs, st) / (st[-1] - st[0])) if len(st) > 1 else float("nan")
    return tau, auc


def cifte_kapi(OUT, arm, seed=0):
    """Composition Collapse (arXiv 2605.26789) konfundunu ayirt eder.

    ENT2 = 'ikinci hop olgusu egitimde HIC ikinci hop olarak kullanilmamis'.
    ENT2 basarisizligi iki sebepten olabilir:
      (a) model besteleyemiyor                 <- aradigimiz sonuc
      (b) o olguyu zaten guvenilir bilmiyor    <- OLCUM TUZAGI
    Genel 1hop dogrulugunun ~0.997 olmasi (b)'yi DISLAMAZ: o ortalama,
    tutulan olgularin kendi dogrulugu degil.

    Ayirt etme: ENT2 dogrulugunu, karsilik gelen (b,r2) olgusu 1-hop'ta
    DOGRU bilinen ornekler uzerinde yeniden hesapla. Son kontrol
    noktasindan dogrudan olculur; parquet gerekmez."""
    import torch
    v = json.load(open(os.path.join(OUT, "verdict.json")))
    cfg = v["cfg"]
    z = np.load(os.path.join(OUT, "veri.npz"))
    if "ent2" not in z.files or len(z["ent2"]) == 0:
        return None
    mp = os.path.join(OUT, "model_%s_s%d.pt" % (arm, seed))
    if not os.path.exists(mp):
        return None

    os.environ["PRESET"] = v.get("preset", "grok")
    os.environ["MEM_AT"] = ",".join(str(x) for x in cfg.get("MEM_AT", [cfg["L"] // 2]))
    os.environ["HOP2_FRAC"] = str(cfg.get("HOP2_FRAC", 0))
    os.environ["COMPILE"] = "0"
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import sifirdan as S

    facts = z["facts"]
    ent2 = [tuple(int(x) for x in row) for row in z["ent2"]]
    net = S.Net(arm, S.CFG).to(S.DEV)
    net.load_state_dict(torch.load(mp, map_location=S.DEV)["state"])
    net.eval()

    ciftler = sorted({(b, r2) for _, _, r2, b, _ in ent2})
    b1 = [(b, r2, int(facts[b, r2])) for b, r2 in ciftler]
    ok1, _ = S.evaluate(net, *S.enc_one(b1)[:3])
    bil = {(b, r2): bool(o) for (b, r2, _), o in zip(b1, ok1)}

    ok2, _ = S.evaluate(net, *S.enc_two(ent2)[:3])
    kapi = np.array([bil[(b, r2)] for _, _, r2, b, _ in ent2])
    return dict(n=len(ent2), n_olgu=len(b1),
                bir_hop_tutulan=float(ok1.mean()),
                ent2=float(ok2.mean()),
                ent2_kapili=float(ok2[kapi].mean()) if kapi.any() else float("nan"),
                n_kapili=int(kapi.sum()))


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
    _e2 = any("ent2" in egri[x][-1] for x in arms)
    _id = any("ident" in egri[x][-1] for x in arms)
    P("| kol | adim | comp | seen | 1hop |" + (" kimlik |" if _id else "")
      + " ent |" + (" ENT2 |" if _e2 else "")
      + " kopru(ONEK-TUT) | kopru_L0 | cevap | belleksiz | memH |")
    P("|---|---|---|---|---|" + ("---|" if _id else "") + "---|"
      + ("---|" if _e2 else "") + "---|---|---|---|---|")
    for x in arms:
        c = egri[x][-1]
        g = lambda k, d="-": (f"{c[k]:.3f}" if isinstance(c.get(k), (int, float))
                              and np.isfinite(c.get(k, np.nan)) else d)
        P(f"| {x} | {c['step']} | {g('comp')} | {g('seen')} | {g('one')} | "
          + (f"{g('ident')} | " if _id else "")
          + f"{g('ent')} | "
          + (f"{g('ent2')} | " if _e2 else "")
          + f"{g('ho_kopru')} | {g('ho_kopru_L0')} | {g('cevap')} | "
          f"{g('comp_acc_nomem')} | {g('mem_H')} |")

    # ---- 1b. ENT2: BIRINCIL (deney 4) + cifte kapi
    if _e2:
        P("")
        P("## ENT2 - ikinci-hop OOD (BIRINCIL, esik 0.10)")
        P("")
        P("> ENT  = 1. hop olgusu hic 1. hop olarak kullanilmamis (bizim bolme)")
        P("> ENT2 = 2. hop olgusu hic 2. hop olarak kullanilmamis "
          "(2405.15071 / 2606.20737 OOD tanimi)")
        P("")
        P("| kol | ENT2 max | hangi adimda | ENT2 son | esik 0.10 |")
        P("|---|---|---|---|---|")
        for x in arms:
            vs = [(c.get("ent2", float("nan")), c["step"]) for c in egri[x]]
            vs = [(v_, st) for v_, st in vs if np.isfinite(v_)]
            if not vs:
                continue
            mx, mst = max(vs)
            P(f"| {x} | {mx:.3f} | {mst} | {vs[-1][0]:.3f} | "
              f"{'**GECTI**' if mx > 0.10 else 'gecmedi'} |")
        P("")
        P("### Cifte kapi (Composition Collapse, arXiv 2605.26789)")
        P("")
        P("> ENT2 dusuk cikarsa iki aciklama var: model besteleyemiyor, ya da "
          "o olgulari zaten bilmiyor. 'ENT2 (kapili)' sutunu yalnizca 1-hop'u "
          "DOGRU bilinen olgular uzerinde hesaplandi.")
        P("")
        P("| kol | tutulan olgu | 1hop(tutulan) | ENT2 (hepsi) | ENT2 (kapili) | n |")
        P("|---|---|---|---|---|---|")
        for x in arms:
            try:
                d = cifte_kapi(OUT, x, SEED)
            except Exception as ex:
                P(f"| {x} | HATA: {str(ex)[:50]} | | | | |")
                continue
            if d is None:
                continue
            P(f"| {x} | {d['n_olgu']} | {d['bir_hop_tutulan']:.3f} | "
              f"{d['ent2']:.3f} | {d['ent2_kapili']:.3f} | {d['n_kapili']} |")

    # ---- 1c. KIMLIK DENETIMI (deney 5)
    if _id:
        P("")
        P("## Kimlik denetimi — saglik kapisi ve birincil olcu")
        P("")
        P("> `kimlik` DUSUK kalirsa sonuc 'recete ise yaramadi' DEGIL,")
        P("> 'recete uygulanamadi' demektir. Ikisi cok farkli.")
        P("")
        P("| kol | kimlik | ENT (birincil, esik 0.20) | ENT2 | comp | 1hop | hukum |")
        P("|---|---|---|---|---|---|---|")
        for x in arms:
            c = egri[x][-1]
            if "ident" not in c:
                continue
            idv = c["ident"]
            ent_mx = max(r.get("ent", 0) for r in egri[x])
            if idv < 0.95:
                h = "**GECERSIZ** (kimlik ogrenilmedi)"
            elif ent_mx > 0.20:
                h = "**ACTI**"
            else:
                h = "acmadi"
            P(f"| {x} | {idv:.3f} | {ent_mx:.3f} | {c.get('ent2', float('nan')):.3f} | "
              f"{c['comp']:.3f} | {c['one']:.3f} | {h} |")
        P("")
        P("Karsilastirma tabani — kol A (kimlik denetimi YOK, ayni veri, 120k):")
        P("`comp 0.851 | 1hop 0.996 | ENT 0.029 | ENT2 0.049`")

    # ---- 2. kopru sabit mi, comp tirmaniyor mu?
    P("\n## Kopru okunabilirligi vs comp dogrulugu\n")
    for x in arms:
        cs = [c["comp"] for c in egri[x]]
        ks = [c.get("ho_kopru", float("nan")) for c in egri[x]]
        P(f"- **kol {x}**: comp {cs[0]:.3f} -> {cs[-1]:.3f} "
          f"({cs[-1]/max(cs[0],1e-9):.1f}x) | kopru {ks[0]:.3f} -> {ks[-1]:.3f}")
    P("\n> Kopru yatay kalip comp tirmaniyorsa: ogrenilen sey kopruyu BULMAK "
      "degil, onu KULLANMAK.")

    # ---- 2b. IKINCIL ON KAYIT: hiz
    P("\n## Hiz (IKINCIL — B ve C kosmadan once kayda gecti)\n")
    P("| kol | tau(0.20) | tau(0.30) | tau(0.40) | tau(0.50) | AUC |")
    P("|---|---|---|---|---|---|")
    taus = {}
    for x in arms:
        rs = [hiz_olcusu(egri[x], e)[0] for e in (.20, .30, .40, .50)]
        auc = hiz_olcusu(egri[x])[1]
        taus[x] = rs[1]
        P(f"| {x} | " + " | ".join(str(v) if v else "-" for v in rs) + f" | {auc:.3f} |")
    if taus.get("B") and taus.get("C"):
        d = taus["B"] - taus["C"]
        P(f"\n**C - B hiz farki:** C esige {d:+d} adim {'ONCE' if d>0 else 'SONRA'} ulasti.")
        P("> ON KAYIT-2 esigi: en az bir degerlendirme araligi (5000 adim) ONCE,\n"
          "> VE AUC farki ayni yonde. Tek seed ile sonuc yine ON BULGU'dur;\n"
          "> grokking baslangici seed'e cok duyarlidir, hiz iddiasi >=3 seed ister.")
    elif taus.get("B") is None or taus.get("C") is None:
        P("\n(B ve/veya C esigi bu butcede gecmedi -> hiz karsilastirmasi yapilamaz)")

    # ---- 2c. GERCEK MALIYET (duvar saati)
    P("")
    P("## Hesap maliyeti — kollar gercekten esit mi?")
    P("")
    P("> Teoride esit: yogun bellekte her parametre token basina TAM BIR KEZ")
    P("> okunuyor (K skorda, V agirlikli toplamda), FFN de oyle. Yani parametre")
    P("> esitlemek = hesap esitlemek. Olculdu: ek MAC/token A 1.933.312,")
    P("> C 1.933.312 -- sifir fark. Asagisi bunun GERCEK cekirdeklerde de")
    P("> tuttugunun kontrolu; tutmazsa hiz/verimlilik iddiasi duzeltilmelidir.")
    P("")
    P("| kol | toplam sn | sn / 1000 adim | A'ya gore |")
    P("|---|---|---|---|")
    taban = None
    for x in arms:
        c = egri[x][-1]
        sn = c.get("secs")
        if sn is None:
            continue
        per = 1000.0 * sn / c["step"]
        if taban is None:
            taban = per
        P(f"| {x} | {sn:.0f} | {per:.2f} | {per/taban:.3f}x |")

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
