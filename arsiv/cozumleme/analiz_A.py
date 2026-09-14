# -*- coding: utf-8 -*-
"""
Kol A'nin TAM tersine muhendislik analizi. Yeni kosu YOK; kayitli kontrol
noktasi + veri.npz yeter. GPU gerekmez (~3-5 dk CPU).

Her sonuc DISKE yazilir: terminal ciktisina guvenme.
  <out>/analiz_A.json   tum skalerler ve egriler
  <out>/analiz_A.npz    diziler (dikkat, ablasyon, yon)
  <out>/analiz_A.png    ozet grafik (matplotlib varsa)

Kullanim:
    python analiz_A.py --model A_kontrol/model_A_s0.pt --out A_kontrol [--n 2000]
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tani_rol as T


def tahminler(S, net, E, kisayol, bs=256):
    """argmax tahmini + dogru/kisayol maskeleri."""
    X, poz, tt = E[0], E[1], E[2]
    lo, hi = S.ENT_OFF, S.ENT_OFF + S.CFG["N_ENT"]
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).to(S.DEV)
            idx = torch.from_numpy(poz[i:i + bs]).to(S.DEV)
            ar = torch.arange(len(idx), device=S.DEV)
            lg, _ = net(xb)
            out.append((lg.float()[ar, idx][:, lo:hi].argmax(-1) + lo).cpu().numpy())
    p = np.concatenate(out)
    return p, (p == tt).mean(), (p == kisayol).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="A_kontrol")
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = {"_model": os.path.abspath(a.model), "_n": a.n}
    A = {}

    S, net, cfg = T.kur(a.model)
    net.eval()
    R["_cfg"] = {k: (list(v) if isinstance(v, tuple) else v)
                 for k, v in cfg.items() if isinstance(v, (int, float, str, tuple))}
    facts, pairs, one, tr2, comp, ent_ev, seen_e, unseen_e, ent2 = S.build_data()
    kume, _ = T.kumeler(S, a.n, os.path.join(os.path.dirname(a.model), "veri.npz"))
    N, L, NH, D = S.CFG["N_ENT"], S.CFG["L"], S.CFG["NH"], S.CFG["D"]
    kis = {ad: np.array([S.ENT_OFF + int(facts[e, r2]) for e, _, r2, *_ in L_])
           for ad, L_ in kume.items()}

    # ---- 1) temel + KISAYOL orani ------------------------------------------
    print("[1/7] dogruluk ve kisayol")
    R["temel"] = {}
    for ad, lst in kume.items():
        E = S.enc_two(lst)
        p, acc, sc = tahminler(S, net, E, kis[ad])
        # ayni sorunun 1-hop hali: [Q1] e r2 ?
        E1 = S.enc_one([(int(e), int(r2), int(facts[e, r2])) for e, _, r2, *_ in lst])
        p1, _, _ = tahminler(S, net, E1, kis[ad])
        R["temel"][ad] = dict(n=len(lst), dogruluk=float(acc), kisayol=float(sc),
                              bir_hop_ortusme=float((p == p1).mean()))
        print(f"   {ad:10s} dogr {acc:.3f}  kisayol {sc:.3f}  1hop-ortusme {(p==p1).mean():.3f}")

    # ---- 2) r1 BOZMA (nedensel) --------------------------------------------
    print("[2/7] r1 bozma")
    rng = np.random.RandomState(0)
    R["r1_bozma"] = {}
    for ad, lst in kume.items():
        p0, _, _ = tahminler(S, net, S.enc_two(lst), kis[ad])
        yeni = []
        for e, r1, r2, b, _ in lst:
            r1b = int(rng.randint(S.CFG["N_REL"] - 1)); r1b += (r1b >= r1)
            bb = int(facts[e, r1b])
            yeni.append((int(e), r1b, int(r2), bb, int(facts[bb, r2])))
        ky = np.array([S.ENT_OFF + int(facts[e, r2]) for e, _, r2, *_ in yeni])
        p1, acc1, sc1 = tahminler(S, net, S.enc_two(yeni), ky)
        R["r1_bozma"][ad] = dict(ayni_kaldi=float((p0 == p1).mean()),
                                 yeni_dogru=float(acc1), yeni_kisayol=float(sc1))
        print(f"   {ad:10s} ayni {(p0==p1).mean():.3f}  yeni-dogru {acc1:.3f}")

    # ---- 3) katman egrileri + dikkat ---------------------------------------
    print("[3/7] katman egrileri ve dikkat")
    R["katman"] = {}
    for ad, lst in kume.items():
        E = S.enc_two(lst)
        kop = np.array([S.ENT_OFF + b for *_, b, _ in lst], np.int64)
        lg_, lb_ = S.logit_lens(net, E, kop)
        P, C = T.kopru_hizasi(S, net, E, kop)
        R["katman"][ad] = dict(altin_sira=lg_, kopru_sira=lb_, kopru_P=P, kopru_cos=C)
        A[f"dikkat_{ad}"] = S.dikkat(net, E[0], E[1], nmax=768)   # [L, NH, T]

    # ---- 4) gomme probu: "zincir basi" yonu --------------------------------
    print("[4/7] prob ve yon")
    W = net.emb.weight.detach()[S.ENT_OFF:S.ENT_OFF + N].float()
    w = (W[list(seen_e)].mean(0) - W[list(unseen_e)].mean(0)); w = w / w.norm()
    ps = (W[list(seen_e)] @ w); pu = (W[list(unseen_e)] @ w)
    R["yon"] = dict(seen_izdusum=float(ps.mean()), unseen_izdusum=float(pu.mean()),
                    seen_sd=float(ps.std()), unseen_sd=float(pu.std()),
                    ayrim_sd_kati=float((ps.mean() - pu.mean()) / ps.std()))
    A["yon_w"] = w.numpy(); A["gomme_izdusum"] = (W @ w).numpy()
    A["etiket_seen"] = np.isin(np.arange(N), list(seen_e))
    # AUC (esikten bagimsiz ayrilabilirlik)
    s = (W @ w).numpy(); y = A["etiket_seen"]
    o = np.argsort(s); r_ = np.empty(len(o)); r_[o] = np.arange(len(o))
    R["yon"]["AUC"] = float((r_[y].sum() - y.sum()*(y.sum()-1)/2) / (y.sum()*(~y).sum()))
    print(f"   ayrim {R['yon']['ayrim_sd_kati']:.2f} sd   AUC {R['yon']['AUC']:.3f}")

    # ---- 5) TEK BILESEN duzenlemesi + rastgele yon kontrolu ----------------
    print("[5/7] tek-bilesen nedensel duzenleme")
    @torch.no_grad()
    def duzenle(lst, yon, hedef, bs=256):
        E = S.enc_two(lst); X, poz, tt = E[0], E[1], E[2]
        lo, hi = S.ENT_OFF, S.ENT_OFF + N
        ky = np.array([S.ENT_OFF + int(facts[e, r2]) for e, _, r2, *_ in lst])
        ok, sc = [], []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i+bs]).to(S.DEV)
            idx = torch.from_numpy(poz[i:i+bs]).to(S.DEV)
            ar = torch.arange(len(idx), device=S.DEV)
            h = net.emb(xb) + net.pos(torch.arange(xb.shape[1], device=S.DEV))[None]
            if hedef is not None:
                h = h.clone()
                mev = net.emb(xb[:, 1]).float() @ yon
                h[:, 1] = h[:, 1] + ((hedef - mev)[:, None] * yon[None, :]).to(h.dtype)
            h0 = h
            for j, blk in enumerate(net.blocks):
                h = blk(h); h, _ = net._mem_uygula(h, h0, j)
            p = net.head(net.nf(h))[ar, idx].float()[:, lo:hi].argmax(-1) + lo
            ok.append((p == torch.from_numpy(tt[i:i+bs]).to(S.DEV)).cpu().numpy())
            sc.append((p == torch.from_numpy(ky[i:i+bs]).to(S.DEV)).cpu().numpy())
        return float(np.concatenate(ok).mean()), float(np.concatenate(sc).mean())

    g = torch.Generator().manual_seed(3)
    wr = torch.randn(D, generator=g); wr = wr / wr.norm()
    R["duzenleme"] = {}
    for ad, hedef_y, hedef_r in (("ENT", ps.mean().item(), (W[list(seen_e)] @ wr).mean().item()),
                                 ("COMP", pu.mean().item(), (W[list(unseen_e)] @ wr).mean().item())):
        t0 = duzenle(kume[ad], w, None)
        t1 = duzenle(kume[ad], w, hedef_y)
        t2 = duzenle(kume[ad], wr, hedef_r)
        R["duzenleme"][ad] = dict(taban=dict(dogruluk=t0[0], kisayol=t0[1]),
                                  yon=dict(dogruluk=t1[0], kisayol=t1[1]),
                                  rastgele_kontrol=dict(dogruluk=t2[0], kisayol=t2[1]))
        print(f"   {ad:5s} taban {t0[1]:.3f} -> yon {t1[1]:.3f} | rastgele {t2[1]:.3f} (kisayol)")

    # ---- 6) kafa ablasyonu -------------------------------------------------
    print("[6/7] 64 kafa ablasyonu")
    E1g = S.enc_one([(int(e), int(r), int(facts[e, r])) for e, r, _ in one[:800]])
    def olc():
        d = {}
        p, acc, _ = tahminler(S, net, E1g, np.zeros(len(E1g[0]), np.int64))
        d["bir_hop"] = float(acc)
        for ad in ("COMP", "ENT"):
            _, acc2, sc2 = tahminler(S, net, S.enc_two(kume[ad][:800]), kis[ad][:800])
            d[ad] = float(acc2); d[ad + "_kis"] = float(sc2)
        return d
    taban = olc()
    kf, dC, dK, dE, d1 = [], [], [], [], []
    for j in range(L):
        for hh in range(NH):
            net.blocks[j].abl = [hh]; r_ = olc(); net.blocks[j].abl = None
            kf.append((j, hh)); dC.append(r_["COMP"] - taban["COMP"])
            dK.append(r_["ENT_kis"] - taban["ENT_kis"])
            dE.append(r_["ENT"] - taban["ENT"]); d1.append(r_["bir_hop"] - taban["bir_hop"])
    A["abl_kafa"] = np.array(kf); A["abl_dCOMP"] = np.array(dC)
    A["abl_dKIS"] = np.array(dK); A["abl_dENT"] = np.array(dE); A["abl_d1hop"] = np.array(d1)
    i = int(np.argmin(dC))
    R["ablasyon"] = dict(taban=taban, en_kritik_kafa=f"K{kf[i][0]}H{kf[i][1]}",
                         en_kritik_dCOMP=float(dC[i]), en_kritik_dKIS=float(dK[i]),
                         korelasyon_COMP_KIS=float(np.corrcoef(dC, dK)[0, 1]))
    print(f"   en kritik {R['ablasyon']['en_kritik_kafa']}  "
          f"dCOMP {dC[i]:+.3f} dKIS {dK[i]:+.3f}  korelasyon {R['ablasyon']['korelasyon_COMP_KIS']:+.3f}")

    # ---- 7) iliski operator mu --------------------------------------------
    print("[7/7] operator testi")
    Wc = W - W.mean(0); Ms, r2l = [], []
    for r in range(S.CFG["N_REL"]):
        Y = Wc[torch.from_numpy(facts[:, r]).long()]
        M = torch.linalg.lstsq(Wc, Y).solution; Ms.append(M)
        r2l.append(float(1 - ((Y - Wc @ M) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()))
    komp = []
    for (r1, r2) in pairs[:12]:
        b = facts[np.arange(N), r1]; ans = facts[b, r2]
        Y = Wc[torch.from_numpy(ans).long()]
        Yh = Wc @ Ms[r1] @ Ms[r2]
        komp.append(float(1 - ((Y - Yh) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()))
    R["operator"] = dict(tek_hop_R2=r2l, tek_hop_ort=float(np.mean(r2l)),
                         kompozisyon_R2=komp, kompozisyon_ort=float(np.mean(komp)))
    print(f"   tek hop R2 {np.mean(r2l):.3f}  kompozisyon R2 {np.mean(komp):.3f}")

    # ---- yaz ---------------------------------------------------------------
    jy = os.path.join(a.out, "analiz_A.json")
    ny = os.path.join(a.out, "analiz_A.npz")
    json.dump(R, open(jy, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    np.savez_compressed(ny, **A)
    print(f"\n  -> {jy}\n  -> {ny}")
    try:
        cizim(R, A, os.path.join(a.out, "analiz_A.png"))
        print(f"  -> {os.path.join(a.out, 'analiz_A.png')}")
    except Exception as ex:
        print(f"  grafik atlandi: {str(ex)[:80]}")


def cizim(R, A, yol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f, ax = plt.subplots(2, 3, figsize=(16, 8))
    kl = ["COMP", "ENT", "ENT2"]

    b = ax[0, 0]
    x = np.arange(len(kl)); g = 0.35
    b.bar(x - g/2, [R["temel"][k]["dogruluk"] for k in kl], g, label="dogruluk")
    b.bar(x + g/2, [R["temel"][k]["kisayol"] for k in kl], g, label="kisayol f(e,r2)")
    b.set_xticks(x); b.set_xticklabels(kl); b.legend(); b.set_title("Dogruluk vs kisayol")

    b = ax[0, 1]
    for k in kl:
        b.plot(R["katman"][k]["kopru_sira"], "o-", label=f"{k} kopru")
        b.plot(R["katman"][k]["altin_sira"], "s--", alpha=.5, label=f"{k} altin")
    b.axhline(.5, color="k", lw=.5); b.set_xlabel("katman"); b.legend(fontsize=7)
    b.set_title("Logit lens (0=tepe, .5=sans)")

    b = ax[0, 2]
    for k in ("COMP", "ENT"):
        kat = A[f"dikkat_{k}"].mean(1)
        b.plot(kat[:, 1], "o-", label=f"{k} -> varlik")
        b.plot(kat[:, 3], "s--", label=f"{k} -> r2")
    b.set_xlabel("katman"); b.legend(fontsize=7); b.set_title("Cevap pozisyonundan dikkat")

    b = ax[1, 0]
    iz, y = A["gomme_izdusum"], A["etiket_seen"]
    b.hist(iz[y], 40, alpha=.6, density=True, label="seen (2-hop oznesi)")
    b.hist(iz[~y], 40, alpha=.6, density=True, label="unseen")
    b.legend(); b.set_title(f"'zincir basi' yonu  AUC={R['yon']['AUC']:.3f}")

    b = ax[1, 1]
    for i, k in enumerate(("ENT", "COMP")):
        d = R["duzenleme"][k]
        for j, (ad, c) in enumerate((("taban", "C0"), ("yon", "C1"), ("rastgele_kontrol", "C2"))):
            b.bar(i * 4 + j, d[ad]["kisayol"], color=c,
                  label=ad if i == 0 else None)
    b.set_xticks([1, 5]); b.set_xticklabels(["ENT", "COMP"]); b.legend(fontsize=7)
    b.set_title("Tek-bilesen duzenleme -> kisayol")

    b = ax[1, 2]
    b.scatter(A["abl_dCOMP"], A["abl_dKIS"], s=14)
    i = int(np.argmin(A["abl_dCOMP"])); j, h = A["abl_kafa"][i]
    b.annotate(f"K{j}H{h}", (A["abl_dCOMP"][i], A["abl_dKIS"][i]))
    b.set_xlabel("dCOMP"); b.set_ylabel("dKISAYOL")
    b.set_title(f"Kafa ablasyonu  r={R['ablasyon']['korelasyon_COMP_KIS']:+.2f}")
    f.tight_layout(); f.savefig(yol, dpi=110)


if __name__ == "__main__":
    main()
