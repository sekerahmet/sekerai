# -*- coding: utf-8 -*-
"""OKUNABILIR — sentetik veriyi ve model hatalarini KELIMEYLE goster.

Model zaten tam sayi token goruyor; bu dosya EGITIMI HIC DEGISTIRMEZ.
Sadece token kimliklerine okunabilir ad takar:

    token  8..15    -> iliski adlari  (baskent, nufus, ...)
    token 16..1015  -> varlik adlari  (Turkiye, Ankara, ...)

DURUSTLUK NOTU: graf RASTGELE (facts = rng.randint). Yani "Turkiye baskent
Ankara" okunakli ama ANLAMI keyfi -- Ankara gercekten Turkiye'nin baskenti
oldugu icin degil, 417 numarali varligin 3 numarali iliskisi 881'e gittigi
icin oyle yaziyor. Amac anlam degil GORUNURLUK: modelin hangi varligi
sectigini (kopru mu, kisayol mu, baska mi) gozle gormek.

    python okunabilir.py <cikti_klasoru> [adim]
"""
import os, sys, glob, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sifirdan as S

# 8 iliski: kullanicinin ornegindeki gercek iliski adlari
ILISKI = ["baskent", "nufus", "yuzolcum", "para_birimi",
          "sira", "komsu", "dil", "kita"]

# Varlik adlari: okunabilir ve AYIRT EDILEBILIR olmasi yeterli.
_KOK = ["Turkiye", "Ankara", "Istanbul", "Fransa", "Paris", "Japonya", "Tokyo",
        "Almanya", "Berlin", "Italya", "Roma", "Ispanya", "Madrid", "Yunanistan",
        "Atina", "Misir", "Kahire", "Hindistan", "Delhi", "Cin", "Pekin",
        "Rusya", "Moskova", "Brezilya", "Brasilia", "Kanada", "Ottawa",
        "Meksika", "Peru", "Lima", "Sili", "Kenya", "Nairobi", "Fas", "Rabat",
        "Iran", "Tahran", "Irak", "Bagdat", "Suriye", "Sam", "Lubnan", "Beyrut",
        "Urdun", "Amman", "Katar", "Doha", "Umman", "Maskat", "Kibris"]


def adlar(n_ent, n_rel):
    """token kimligi -> ad.  Ilk 50 varlik tanidik isim, gerisi turetilmis."""
    ent = list(_KOK[:min(n_ent, len(_KOK))])
    i = 0
    while len(ent) < n_ent:
        ent.append(f"{_KOK[i % len(_KOK)]}_{len(ent)}")
        i += 1
    rel = [ILISKI[r % len(ILISKI)] + ("" if r < len(ILISKI) else f"_{r}")
           for r in range(n_rel)]
    return ent[:n_ent], rel[:n_rel]


def kur():
    E, R = adlar(S.CFG["N_ENT"], S.CFG["N_REL"])

    def ad(tok):
        if tok < S.SPECIAL:
            return {S.PAD: ".", S.Q1: "[S1]", S.Q2: "[S2]", S.QM: "?",
                    S.EOS: "<son>", S.IDENT: "[AYNI]"}.get(int(tok), f"<{tok}>")
        if tok < S.ENT_OFF:
            return R[int(tok) - S.REL_OFF]
        return E[int(tok) - S.ENT_OFF]
    return E, R, ad


def satir(x, ad):
    """token dizisini okunakli tek satira cevir."""
    return " ".join(ad(t) for t in x if t != S.PAD)


def main():
    klasor = sys.argv[1]
    E, R, ad = kur()
    facts, pairs, one, tr2, comp, ent_ev, seen_ent, unseen_ent, ent2_ev = S.build_data()

    print("=" * 78)
    print("SOZLUK")
    print(f"  iliskiler ({len(R)}): " + ", ".join(R))
    print(f"  varliklar ({len(E)}): " + ", ".join(E[:8]) + f", ... , {E[-1]}")
    print(f"  ZINCIR BASI OLMAYAN varlik sayisi: {len(unseen_ent)}"
          f"   ornek: " + ", ".join(E[i] for i in unseen_ent[:5]))

    print("\n" + "=" * 78)
    print("EGITIM VERISI — 1 ADIMLI olgular (atomik)")
    for e, r, a in one[:6]:
        print(f"   {E[e]:14s} {R[r]:12s} -> {E[a]}")

    print("\nEGITIM VERISI — 2 ADIMLI zincirler (gorulmus varliklar)")
    for e, r1, r2, b, a in tr2[:6]:
        print(f"   {E[e]:14s} {R[r1]:12s} {R[r2]:12s} -> {E[a]}"
              f"      (kopru: {E[b]})")

    print("\nSINAV — ENT: varlik HIC zincir basi olmamis")
    for e, r1, r2, b, a in ent_ev[:6]:
        print(f"   {E[e]:14s} {R[r1]:12s} {R[r2]:12s} -> {E[a]}"
              f"      (kopru: {E[b]}, KISAYOL cevabi: {E[facts[e, r2]]})")

    # --- model varsa: NE CEVAPLADI -----------------------------------------
    snaps = sorted(glob.glob(os.path.join(klasor, "snap_A_s*_*.pt")))
    if not snaps:
        print("\n(anlik goruntu yok, model cevaplari atlandi)")
        return
    adim = int(sys.argv[2]) if len(sys.argv) > 2 else None
    sec = ([p for p in snaps if f"_{adim:06d}.pt" in p] or [snaps[-1]])[0]
    poz, blok = None, None
    if os.environ.get("MASKE", "yok") != "yok":
        p, b = os.environ["MASKE"].split("@")
        a1, a2 = b.split("-")
        poz, blok = int(p), tuple(range(int(a1), int(a2) + 1))

    net = S.Net("A", S.CFG).to(S.DEV)
    net.load_state_dict({k: v.float() for k, v in
                         torch.load(sec, map_location=S.DEV).items()})
    net.eval()
    for i, blk in enumerate(net.blocks):
        blk.mask_key = poz if (blok and i in blok) else None

    # RASTGELE ornek: ent_ev VARLIGA GORE SIRALI, ilk 12 hep AYNI varlik
    # olur ve tek varligin davranisi genelin yerine gecer (13 Eylul: A5'te
    # 1/12 kisayol gorundu, gercek oran 0.82 idi).
    n = 12
    rg = np.random.RandomState(0)
    lst = [ent_ev[i] for i in rg.permutation(len(ent_ev))[:n]]
    X, tp, tt, _ = S.enc_two(lst)
    with torch.no_grad():
        lg, _ = net(torch.from_numpy(X).to(S.DEV))
    ix = torch.from_numpy(tp).to(S.DEV)
    ar = torch.arange(len(ix), device=S.DEV)
    tah = (lg.float()[ar, ix][:, S.ENT_OFF:S.ENT_OFF + S.CFG["N_ENT"]]
           .argmax(-1).cpu().numpy())

    print("\n" + "=" * 78)
    print(f"MODEL NE CEVAPLADI   {os.path.basename(sec)}"
          f"   maske {os.environ.get('MASKE', 'yok')}")
    print(f"{'SORU':44s} {'DOGRU':12s} {'MODEL':12s} NE OLDU")
    sayac = {"DOGRU": 0, "KISAYOL": 0, "KOPRU": 0, "BASKA": 0}
    for (e, r1, r2, b, a), t in zip(lst, tah):
        ks = int(facts[e, r2])
        ne = ("DOGRU" if t == a else "KISAYOL" if t == ks
              else "KOPRU" if t == b else "BASKA")
        sayac[ne] += 1
        print(f"   {E[e]:13s} {R[r1]:11s} {R[r2]:11s} ? "
              f"{E[a]:12s} {E[t]:12s} {ne}")
    print(f"\n   {n} ornekte: " + "  ".join(f"{k} {v}" for k, v in sayac.items()))

    # BILINEN DEGERE KARSI DOGRULA (CLAUDE.md 10a): ayni olcumu TUM ENT
    # setinde yap, egrideki sayiyla yan yana bas. Tutmuyorsa ARAC bozuk.
    TX, Ttp, _t, _e = S.enc_two(ent_ev)
    dg = ks_ = 0
    lo_, hi_ = S.ENT_OFF, S.ENT_OFF + S.CFG['N_ENT']
    with torch.no_grad():
        for i0 in range(0, len(TX), 512):
            xb = torch.from_numpy(TX[i0:i0 + 512]).to(S.DEV)
            ixb = torch.from_numpy(Ttp[i0:i0 + 512]).to(S.DEV)
            arb = torch.arange(len(ixb), device=S.DEV)
            pr = (net(xb)[0].float()[arb, ixb][:, lo_:hi_]
                  .argmax(-1).cpu().numpy())
            for k0, (e0, _r1, r2_, _b, a0) in enumerate(ent_ev[i0:i0 + 512]):
                dg += int(pr[k0] == a0)
                ks_ += int(pr[k0] == facts[e0, r2_])
    m = len(ent_ev)
    print(f'\n   TUM ENT setinde ({m} ornek):')
    print(f'      dogruluk (ent) {dg/m:.4f}    kisayol {ks_/m:.4f}')
    print('      -> EGRIDEKI ent / ent_shortcut ile tutmali; tutmuyorsa ARAC bozuk')
    print("   KISAYOL = ikinci iliskiyi DOGRUDAN ilk varliga uygulamis")
    print("   KOPRU   = ara cevabi yazmis, ikinci adimi atlamis")


if __name__ == "__main__":
    main()
