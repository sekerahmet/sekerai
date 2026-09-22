"""adim_16 -- TEK SORUDA modelin ADIM ADIM ne yaptigi.

Kullanici, 22 Eylul 2026: *"sorulari ve cevaplari gozle gorup bir yargiya
varmak benim isim sen de bakabilirsin ama yargiya varamazsin. sen adim
adim bir ornek soruda modelin davranisini incelemelisin"*.

Bu dosya HUKUM VERMEZ, SAYI GOSTERIR.  Bir soru secilir ve uretimin her
adiminda dort sey dokulur:

    YOL       her birim durumu NE KADAR kaydirdi   |s_t - s_(t-1)|
    DIKKAT    o adimda hangi yuvaya bakildi        relu agirliklari
    OKUMA     sozlukteki en yakin bes birim        cdist mesafeleri
    SECIM     hangisi secildi, DOGRUsu kacinci sirada

KAPI: iz `uret_dizi` ile BIT DUZEYINDE ayni diziyi vermeli.  Vermezse
burada anlatilan sey modelin gercekte yaptigi sey DEGILDIR.

    python adim_16.py                    (ornek: ezber_olgu #0)

ZINCIRDEKI YERI.  Kim kimi cagiriyor, bu dosya nerede:
(model BIRIM goruyor -- karakter yalnizca ara adim ve kapi)

  veri_16     graf: 1608 varlik, 24 iliski, olgular
  metin_16    graf -> duz Turkce cumle
  korpus_16   cumle -> belge -> paketlenmis akis
  jeton_16    KARAKTER sozlugu + GIDIS-DONUS KAPISI
  birim_16    metin -> sayim -> kok havuzu -> BIRIM AKISI -> pencere
  ek_16       kelime -> kok + ek    (Turkce morfolojisi)

  taban_16    bolmeler (ezber_* / cikarim_*) + Ayar tanimi
  ayar_16     dugmeler
  hazirla_16  veriyi dosyaya yazar, Colab Drive'dan OKUR
  dok_16      goz ile okunur dokum
  denetim_16  KAPILAR

  model_16    MIMARI -- model_15'ten
  kos_16      egitim dongusu
  olcme_16    olcu: soru soruldu, cevap dogru mu
  adim_16     TEK SORU, ADIM ADIM   <-- BU DOSYA
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ek_16 as EK        # noqa: E402
import metin_16 as MT     # noqa: E402
import olcme_16 as OL     # noqa: E402


def yuzey(d, ad, i):
    """Bolmenin i. zincirinden (onek, cevap).  yuzeyler() ile AYNI kod
    yolu -- sinavin yuzeyi ile izin yuzeyi ayrilmasin diye."""
    E, IL, TIP = d["varlik"], d["iliski"], d["tip_ad"]
    z = [int(x) for x in d["bolme"][ad][i]]
    rs = z[1:2] if len(z) == 3 else z[1:3]
    onek, cev, _ = MT.sinav_yuzeyi(E[z[0]], [IL[r] for r in rs],
                                   E[z[-1]], TIP[int(d["tip"][z[-1]])])
    return onek, cev


def yurut(m, w, adim):
    """uret_dizi ile AYNI hesap, her ara deger kaydedilir.  w: (1,L)."""
    with torch.no_grad():
        IZ = m.gez(w)                     # (1,L+1,durum) -- s0 dahil
        S = IZ[:, 1:]
        kayma = (IZ[0, 1:] - IZ[0, :-1]).norm(dim=-1)   # (L,)
        s = S[:, -1]
        kayit = []
        for _ in range(adim):
            q = s @ m.Wq
            pu = ((S @ m.Wk) @ q.unsqueeze(-1)).squeeze(-1) + m.hb
            ag = pu.softmax(-1) if m.pay else pu.clamp(min=0)
            o = (ag.unsqueeze(-1) * (S @ m.Wv)).sum(1)          # (1,boyut)
            mes = torch.cdist(o, m.E)[0]                        # (n,)
            t = int(mes.argmin())
            kayit.append(dict(ag=ag[0].cpu(), o=o[0].cpu(),
                              mesafe=mes.cpu(), secilen=t))
            tt = torch.tensor([t], device=w.device)
            s = torch.bmm(m.M[tt], s.unsqueeze(-1)).squeeze(-1) + m.b[tt]
            if m.norm:
                s = F.normalize(s, dim=-1)
            S = torch.cat([S, s[:, None]], 1)
    return kayma.cpu(), kayit


def dok(m, d, ad="ezber_olgu", i=0, onek=None, cevap="", adim=None,
        aygit="cuda", ust=5, yaz=print):
    """TEK SORU, ADIM ADIM.  Hicbir yargi yok -- yalniz sayilar."""
    if onek is None:
        onek, cevap = yuzey(d, ad, i)
        baslik = f"{ad} #{i}"
    else:
        baslik = "ELLE VERILEN SORU"
    kodla, coz = OL._kodlayici(d)
    ix, adlar = d["ix"], d["ad"]
    j = kodla(onek)
    adim = adim or OL.ENUZUN["birim" if "ad" in d else "karakter"]
    w = torch.tensor([j], device=aygit)

    kayma, kayit = yurut(m, w, adim)

    # --- KAPI: iz, uretimin KENDISI mi?
    with torch.no_grad():
        resmi = m.uret_dizi(w, adim)[0].tolist()
    izden = [k["secilen"] for k in kayit]
    assert izden == resmi, ("IZ uret_dizi ILE TUTMADI -- burada anlatilan "
                            f"sey modelin yaptigi sey DEGIL:\n  iz  {izden}\n"
                            f"  asil {resmi}")

    bek = [x for k in cevap.split() for x in EK.bol(k)] if cevap else []
    yaz("=" * 74)
    yaz(f"TEK SORU, ADIM ADIM      {baslik}      KAPI: iz == uret_dizi")
    yaz("=" * 74)
    yaz(f"SORU     {onek}")
    yaz(f"BEKLENEN {cevap!r}   ->  birim olarak: {' '.join(bek)}")

    yaz(f"\n1) GIRDI -- soru {len(j)} birime bolundu.  Yol s0'dan basliyor,")
    yaz("   her birim durumu kaydiriyor.  |s_t - s_(t-1)| BUYUKSE o birim")
    yaz("   yolu cok degistirdi, KUCUKSE neredeyse hic dokunmadi.")
    yaz(f"   {'#':>3s}  {'BIRIM':<18s}{'NO':>5s}{'KAYMA':>9s}")
    for t, (b, u) in enumerate(zip(j, kayma.tolist())):
        yaz(f"   {t:>3d}  {adlar[b]:<18s}{b:>5d}{u:>9.3f}")

    yaz(f"\n2) URETIM -- {adim} adim.  Dikkat RELU (softmax DEGIL): agirliklar")
    yaz("   1'e toplanmaz, sifir olabilir.  Okuma en yakin E[c].")
    uretilen = []
    for k, r in enumerate(kayit):
        ag = r["ag"]
        sec = r["secilen"]
        top = float(ag.sum())
        sifir = int((ag == 0).sum())
        yaz(f"\n   ADIM {k+1}   yuva sayisi {len(ag)}"
            f"   dikkat toplami {top:.3f}   sifir agirlikli {sifir}/{len(ag)}")
        sira = ag.argsort(descending=True)[:ust]
        for p in sira.tolist():
            et = adlar[j[p]] if p < len(j) else adlar[uretilen[p - len(j)]]
            im = "  <- SON YUVA" if p == len(ag) - 1 else ""
            yaz(f"        agirlik {float(ag[p]):7.3f}   yuva {p:3d}  "
                f"{et}{im}")
        mes = r["mesafe"]
        en = mes.argsort()[:ust]
        yaz(f"        okuma  |o| = {float(r['o'].norm()):.3f}")
        for p in en.tolist():
            im = "   <- SECILDI" if p == sec else ""
            yaz(f"        mesafe {float(mes[p]):7.3f}   {adlar[p]}{im}")
        if k < len(bek):
            dg = ix.get(bek[k])
            if dg is None:
                yaz(f"        DOGRUSU {bek[k]!r} SOZLUKTE YOK")
            else:
                s = int((mes < mes[dg]).sum()) + 1
                yaz(f"        DOGRUSU {bek[k]:<16s} mesafe {float(mes[dg]):7.3f}"
                    f"   {s}. sirada / {len(mes)}")
        uretilen.append(sec)

    urun = coz(uretilen)
    yaz(f"\n3) URETILEN   {urun!r}")
    yaz(f"   noktaya kadar {urun.split(OL.BITIS)[0].strip()!r}")
    yaz(f"   DOGRU MU      {OL._esit(urun.split(OL.BITIS)[0].strip(), cevap, True)}")
    yaz("\nNOT: ilk sapmadan SONRA onek artik beklenenden farkli, yani")
    yaz("     sonraki adimlarin 'DOGRUSU' satiri bilgi icindir, olcu degil.")
    return uretilen


if __name__ == "__main__":
    import hazirla_16 as H
    yol = os.environ.get("BIRIM_PT", r"G:/Drive'ım/model_16/birim_16.pt")
    agirlik = os.environ.get("AGIRLIK")
    assert os.path.exists(yol), f"{yol} YOK -- BIRIM_PT ile yol ver"
    d = torch.load(yol, weights_only=False)
    assert H.iz(d) == d["iz"], "veri izi TUTMADI"
    import model_16
    m = model_16.Yol(d["vocab"])
    if agirlik:
        m.load_state_dict(torch.load(agirlik, weights_only=False)["agirlik"])
    else:
        print("UYARI: AGIRLIK verilmedi -- EGITILMEMIS model iz suruluyor\n")
    m.eval()
    dok(m, d, ad=sys.argv[1] if len(sys.argv) > 1 else "ezber_olgu",
        i=int(sys.argv[2]) if len(sys.argv) > 2 else 0, aygit="cpu")
