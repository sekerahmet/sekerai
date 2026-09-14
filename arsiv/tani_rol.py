# -*- coding: utf-8 -*-
"""
TANI: kopru KURULUYOR mu, yoksa KULLANILAMIYOR mu?  (GPU gerekmez)

Iki canli hipotez zit tahmin veriyor:

  (A) arXiv 2608.07261: "alt katmanlar ara temsili dogru kuruyor, ust katmanlar
      akil yurutmek yerine cevaba eslemeyi ogreniyor."
      -> ENT'te de kopru alt katmanlarda BELIRIR. Ariza tuketimde.
      -> D2 (kopru URETIMINI ogreten ders) bosa.

  (B) Bizim olcum: gorulmemis varlik BAS/OZNE rolunde hic okunmamis
      (ama kopru rolunde 24.5, cevap rolunde 24.2 kez gecmis).
      -> ENT'te kopru HIC belirmez. Ariza ozne okumasinda.
      -> D2 tam o rolu veriyor, yerinde.

Olcum: logit lens ile her katmanda KOPRU varliginin normalize sirasi
(0 = en tepe, 0.5 = sans). COMP (calisiyor, 0.851) ile ENT (0.029) egrilerini
yan yana koy.

Kullanim:
    python tani_rol.py --model .../model_A_s0.pt [--n 2000] [--json cikti.json]
"""
import argparse, json, os, sys
import numpy as np
import torch


def kur(model_yolu):
    ck = torch.load(model_yolu, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    os.environ["PRESET"]    = os.environ.get("PRESET_ZORLA", "grok_uzun")
    os.environ["MEM_AT"]    = ",".join(str(x) for x in cfg.get("MEM_AT", [cfg["L"] // 2]))
    os.environ["HOP2_FRAC"] = str(cfg.get("HOP2_FRAC", 0))
    os.environ["COMPILE"]   = "0"
    os.environ["IDENT_FRAC"] = "0"
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import sifirdan as S
    for k in ("N_ENT", "N_REL", "N_PAIR", "P_TRAIN", "D", "L"):
        assert S.CFG.get(k) == cfg.get(k), \
            f"CFG uyusmuyor: {k} {S.CFG.get(k)} != {cfg.get(k)} (PRESET_ZORLA ile ez)"
    net = S.Net(ck["arm"], S.CFG).to(S.DEV)
    net.load_state_dict(ck["state"])
    net.eval()
    print(f"model: kol={ck['arm']} seed={ck['seed']} adim={cfg.get('STEPS')} "
          f"HOP2_FRAC={cfg.get('HOP2_FRAC')} IDENT_FRAC={cfg.get('IDENT_FRAC')} "
          f"cihaz={S.DEV}")
    return S, net, cfg


def kumeler(S, n_max, veri_npz=None):
    facts, pairs, one, tr2, comp, ent_ev, seen_e, unseen_e, ent2 = S.build_data()
    R = S.CFG["N_REL"]

    # veri.npz varsa bolmelerin AYNI oldugunu dogrula (sessiz kayma tuzagi)
    if veri_npz and os.path.exists(veri_npz):
        z = np.load(veri_npz, allow_pickle=True)
        if "ent" in z.files:
            kayit = {tuple(int(x) for x in r[:3]) for r in z["ent"]}
            simdi = {(int(e), int(r1), int(r2)) for e, r1, r2, *_ in ent_ev}
            assert kayit == simdi, (
                f"ENT bolmesi veri.npz ile AYNI DEGIL "
                f"({len(kayit)} vs {len(simdi)}) -> DATA_SEED/HOP2_FRAC farkli")
            print(f"  veri.npz ile bolme dogrulandi (ENT {len(simdi)})")

    kull = set(map(tuple, pairs))
    kullanilmayan = [(a, b) for a in range(R) for b in range(R)
                     if a != b and (a, b) not in kull]
    # gozlenebilir ikinci-hop yasagi (tam kume disaridan turetilemiyor)
    yasak = {(int(b), int(r2)) for _, _, r2, b, _ in ent2}
    rng = np.random.RandomState(12345)
    se = list(seen_e)
    yeni_cift = []
    for _ in range(n_max * 3):
        e = int(se[rng.randint(len(se))])
        r1, r2 = kullanilmayan[rng.randint(len(kullanilmayan))]
        b = int(facts[e, r1])
        if (b, r2) in yasak:
            continue
        yeni_cift.append((e, r1, r2, b, int(facts[b, r2])))
        if len(yeni_cift) >= n_max:
            break

    def kes(L):
        if len(L) <= n_max:
            return L
        i = np.random.RandomState(7).permutation(len(L))[:n_max]
        return [L[j] for j in i]

    return dict(COMP=kes(comp), YENI_CIFT=yeni_cift,
                ENT=kes(ent_ev), ENT2=kes(ent2)), facts


# ===========================================================================
# DiscoLoop (arXiv 2607.00341) imzasi ve EGITIMSIZ mudahalesi
#
# Onlarin olcumu (varlik-ayrik graf = bizim ENT'in birebir karsiligi):
#     test_id : P(b|h) = 1.000  ama  cos(h, W[b]) = 0.327
#     test_ood: P(b|h) = 1.000  ama  cos(h, W[b]) = 0.266
# Kopru OLASILIK olarak kusursuz cozuluyor ama GOMME YONU hizasiz.
#
# Mudahale:  h <- (1-alfa)*h + alfa*Norm(W[b])*||h||   (r1 pozisyonunda)
# Onlarda alfa=0.1 -> OOD %8.3 -> %25.9; alfa~0.5 -> ID ve OOD ~%100. EGITIM YOK.
# Iki kip: argmax (modelin kendi tahmini) / kahin (gercek kopru = UST SINIR).
# ===========================================================================
import torch.nn.functional as F

ALFALAR = (0.0, 0.1, 0.3, 0.5, 0.7, 1.0)


@torch.no_grad()
def kopru_hizasi(S, model, E, kopru, poz_r1=2, bs=256):
    """Her katmanda r1 pozisyonunda: P(kopru) ve cos(h, W[kopru])."""
    X = E[0]
    W = model.emb.weight
    L = len(model.blocks)
    P = [[] for _ in range(L + 1)]
    C = [[] for _ in range(L + 1)]
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(S.DEV)
        b = torch.from_numpy(kopru[i:i + bs]).to(S.DEV)
        h = model.emb(xb) + model.pos(torch.arange(xb.shape[1], device=S.DEV))[None]
        hs = [h]
        for j, blk in enumerate(model.blocks):
            h = blk(h)
            h, _ = model._mem_uygula(h, hs[0], j)
            hs.append(h)
        for j, hh in enumerate(hs):
            hr = hh[:, poz_r1].float()
            lg = model.head(model.nf(hh))[:, poz_r1].float()
            P[j].append(lg.softmax(-1).gather(1, b[:, None]).squeeze(1).cpu().numpy())
            C[j].append(F.cosine_similarity(hr, W[b].float(), dim=-1).cpu().numpy())
    return ([float(np.mean(np.concatenate(x))) for x in P],
            [float(np.mean(np.concatenate(x))) for x in C])


@torch.no_grad()
def alfa_mudahale(S, model, E, kopru, kat, alfa, kahin=False, poz_r1=2, bs=256):
    """kat. bloktan SONRA r1 pozisyonunu kopru gommesine cek, ileri devam et."""
    X, poz, tt = E[0], E[1], E[2]
    W = model.emb.weight
    lo, hi = S.ENT_OFF, S.ENT_OFF + S.CFG["N_ENT"]
    ok = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(S.DEV)
        idx = torch.from_numpy(poz[i:i + bs]).to(S.DEV)
        g = torch.from_numpy(tt[i:i + bs]).to(S.DEV)
        bg = torch.from_numpy(kopru[i:i + bs]).to(S.DEV)
        ar = torch.arange(len(idx), device=S.DEV)
        h = model.emb(xb) + model.pos(torch.arange(xb.shape[1], device=S.DEV))[None]
        h0 = h
        for j, blk in enumerate(model.blocks):
            h = blk(h)
            h, _ = model._mem_uygula(h, h0, j)
            if j + 1 == kat and alfa > 0:
                hr = h[:, poz_r1]
                if kahin:
                    b = bg
                else:
                    _l = model.head(model.nf(h))[:, poz_r1].float()
                    b = _l[:, lo:hi].argmax(-1) + lo
                e = (F.normalize(W[b].float(), dim=-1).to(hr.dtype)
                     * hr.norm(dim=-1, keepdim=True))
                h = h.clone()
                h[:, poz_r1] = (1 - alfa) * hr + alfa * e
        lg = model.head(model.nf(h))[ar, idx].float()
        ok.append((lg[:, lo:hi].argmax(-1) + lo == g).cpu().numpy())
    return float(np.concatenate(ok).mean())


def disco(S, net, kume):
    L = len(net.blocks)
    print("\n" + "=" * 74)
    print("DiscoLoop IMZASI: kopru olasiligi vs GOMME HIZASI (r1 pozisyonu)")
    print("=" * 74)
    print(f"{'kume':10s} {'olcu':5s}   " + "  ".join(f"K{j}" for j in range(L + 1)))
    hiza = {}
    for ad, lst in kume.items():
        if not lst:
            continue
        E = S.enc_two(lst)
        kopru = np.array([S.ENT_OFF + b for *_, b, _ in lst], np.int64)
        P, C = kopru_hizasi(S, net, E, kopru)
        print(f"{ad:10s} {'P(b)':5s}   " + "  ".join(f"{v:.3f}"[1:] for v in P))
        print(f"{'':10s} {'cos':5s}   " + "  ".join(f"{v:+.2f}" for v in C))
        hiza[ad] = dict(P=P, cos=C, P_kat=int(np.argmax(P)),
                        P_max=float(np.max(P)), cos_max=float(np.max(C)))
    print("\n  DiscoLoop referansi: P=1.000 iken cos 0.327 (ID) / 0.266 (OOD)")
    print("  -> P yuksek + cos dusuk = ariza TEMSIL HIZASIZLIGI, bilgi eksikligi degil")

    if not kume.get("ENT"):
        return hiza
    kat = max(1, min(hiza.get("COMP", hiza["ENT"])["P_kat"], L - 1))
    print("\n" + "=" * 74)
    print(f"EGITIMSIZ MUDAHALE: K{kat} cikisinda r1'i kopru gommesine cek")
    print("=" * 74)
    print(f"{'kume':10s} {'kip':7s}  " + "  ".join(f"a={a:<4.2f}" for a in ALFALAR))
    for ad in ("COMP", "ENT", "ENT2"):
        lst = kume.get(ad)
        if not lst:
            continue
        E = S.enc_two(lst)
        kopru = np.array([S.ENT_OFF + b for *_, b, _ in lst], np.int64)
        for kip, kh in (("argmax", False), ("kahin", True)):
            r = [alfa_mudahale(S, net, E, kopru, kat, a, kahin=kh) for a in ALFALAR]
            print(f"{ad:10s} {kip:7s}  " + "  ".join(f"{v:6.3f}" for v in r))
            hiza.setdefault(ad, {})[f"alfa_{kip}"] = r
    print("\n  a=0 sutunu mudahalesiz taban olmali (dogrulama).")
    print("  ENT argmax yukseliyorsa  -> ariza HIZALAMA, DiscoLoop tarzi kapi cozer")
    print("  argmax yatay KAHIN yukseliyorsa -> kopru TAHMINI kotu, ikinci hop saglam")
    print("  ikisi de yatay ise -> ikinci tablo bu varliklar icin BOS, mimari gerekir")
    return hiza


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--json", default="")
    ap.add_argument("--disco", action="store_true",
                    help="DiscoLoop imzasi + egitimsiz alfa mudahalesi")
    a = ap.parse_args()

    S, net, cfg = kur(a.model)
    veri = os.path.join(os.path.dirname(os.path.abspath(a.model)), "veri.npz")
    kume, facts = kumeler(S, a.n, veri)

    L = len(net.blocks)
    print(f"\n{'='*74}\nKOPRU, KATMAN KATMAN (normalize medyan sira: 0 = en tepe, "
          f"0.5 = sans)\n{'='*74}")
    basliklar = "  ".join(f"K{j}" for j in range(L + 1))
    print(f"{'kume':10s} {'n':>5s} {'dogr':>6s}   {basliklar}")

    sonuc = {}
    for ad, lst in kume.items():
        if not lst:
            print(f"{ad:10s}  (bos)"); continue
        E = S.enc_two(lst)
        kopru = np.array([S.ENT_OFF + b for *_, b, _ in lst], np.int64)
        dogr = float(S.evaluate(net, *E[:3])[0].mean())
        lg, lb = S.logit_lens(net, E, kopru)
        print(f"{ad:10s} {len(lst):5d} {dogr:6.3f}   "
              + "  ".join(f"{v:.3f}"[1:] for v in lb))
        sonuc[ad] = dict(n=len(lst), dogruluk=dogr, kopru_sira=lb, altin_sira=lg,
                         kopru_min=float(np.min(lb)), kopru_kat=int(np.argmin(lb)),
                         altin_min=float(np.min(lg)), altin_kat=int(np.argmin(lg)))

    print(f"\n{'='*74}\nKARAR\n{'='*74}")
    if "COMP" in sonuc and "ENT" in sonuc:
        c, e = sonuc["COMP"], sonuc["ENT"]
        print(f"  COMP : kopru en iyi K{c['kopru_kat']}'te, sira {c['kopru_min']:.3f}  "
              f"| dogruluk {c['dogruluk']:.3f}")
        print(f"  ENT  : kopru en iyi K{e['kopru_kat']}'te, sira {e['kopru_min']:.3f}  "
              f"| dogruluk {e['dogruluk']:.3f}")
        # SANS = 0.5. Ham orani kullanmak tuzak: iki egri de sansa yakinsa
        # oran ~1 cikar ve kod GUVENLE YANLIS hukum verir. Onun yerine
        # "sans altina ne kadar inmis" payini kiyasla.
        pay = lambda r: max(0.0, (0.5 - r) / 0.5)
        pc, pe = pay(c["kopru_min"]), pay(e["kopru_min"])
        print(f"  sans altina inis payi: COMP {pc:.3f}   ENT {pe:.3f}")
        if pc < 0.25:
            print()
            print("  >>> OLCUM GECERSIZ: COMP'ta bile kopru belirmiyor "
                  f"(pay {pc:.3f} < 0.25).")
            print("      Model yeterince egitilmemis ya da logit lens bu mimaride")
            print("      kopruyu gostermiyor. Hukum VERILMEZ.")
            if a.json:
                json.dump(sonuc, open(a.json, "w", encoding="utf-8"), indent=1)
            return
        oran = pc / max(pe, 1e-9)
        print(f"  COMP/ENT pay orani: {oran:.1f}x  (1.0 = kopru ayni kalitede kurulmus)")
        print()
        if oran < 1.5:
            print("  >>> (A) KOPRU ENT'TE DE KURULUYOR.")
            print("      Ariza ust katmanlarda, kopruyu KULLANMAKTA.")
            print("      D2 (kopru URETIMI dersi) yanlis hedefte -> tasarim degismeli.")
        elif oran > 4:
            print("  >>> (B) KOPRU ENT'TE KURULMUYOR.")
            print("      Ariza OZNE rolunde okumada.")
            print("      D2 tam o rolu veriyor -> tasarim yerinde.")
        else:
            print("  >>> ARA DEGER. Tek olcum karar vermiyor;")
            print("      YENI_CIFT sutunuyla (0.73 seviyesi) kiyasla, ve")
            print("      altin_sira egrisine bak: cevap nerede beliriyor?")
    if a.disco:
        sonuc["_disco"] = disco(S, net, kume)
    if a.json:
        json.dump(sonuc, open(a.json, "w", encoding="utf-8"), indent=1)
        print(f"\n  -> {a.json}")


if __name__ == "__main__":
    main()
