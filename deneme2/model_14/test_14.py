# -*- coding: utf-8 -*-
"""test_14 -- DENKLEM.md'nin her iddiasi icin bir kapi.

Amac: matematik mi yanlis, kod mu yanlis -- ayirmak. Her kapinin
basliginda BELGENIN hangi cumlesini sinadigi yaziyor.

    python test_14.py
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

import model_14 as M

TOL = 1e-4
GECTI, KALDI = [], []


def kapi(ad):
    def sar(f):
        try:
            not_ = f()
            GECTI.append((ad, not_ or ""))
        except AssertionError as e:
            KALDI.append((ad, str(e)))
        except Exception as e:                       # kodun kendisi bozuk
            KALDI.append((ad, f"{type(e).__name__}: {e}"))
        return f
    return sar


def kur(n=40, D=12, d=4, K=16, k_tam=8, saat=False, tohum=0):
    frk = torch.arange(n).flip(0).float()            # ilk k_tam en sik
    return M.Yol(n, D=D, d=d, K=K, tam=M.sinif_ayir(frk, k_tam),
                 saat=saat, tohum=tohum)


# =====================================================================
@kapi("0  IZOLASYON -- kol disariya import etmez")
def _0():
    src = open(M.__file__, encoding="utf-8").read()
    kotu = [l for l in src.split("\n")
            if l.startswith("import ") or l.startswith("from ")]
    izin = {"math", "torch", "torch.nn", "torch.nn.functional",
            "__future__"}
    for l in kotu:
        mod = l.split()[1]
        assert mod in izin, f"disariya import: {l}"
    return f"{len(kotu)} import, hepsi izinli"


@kapi("1  ORTOGONALLIK -- 'donme normu ve mesafeyi korur'")
def _1():
    m = kur()
    R = m.donme()
    I = torch.eye(m.D)
    e = float((R @ R.transpose(-1, -2) - I).abs().max())
    assert e < TOL, f"R R^T != I : {e:.2e}"
    dt = float((torch.linalg.det(R) - 1).abs().max())
    assert dt < TOL, f"det != 1 (yansima var): {dt:.2e}"
    return f"max|RR^T-I| {e:.1e}   max|det-1| {dt:.1e}"


@kapi("2  RODRIGUES -- tek duzlem KAPALI FORMU dogru mu")
def _2():
    m = kur()
    R = m.donme()
    u = F.normalize(m.u, dim=-1)
    v = m.v - (m.v * u).sum(-1, keepdim=True) * u
    v = F.normalize(v, dim=-1)
    th = m.th
    Ra = R[m.ix_acik]
    bek_u = torch.cos(th)[:, None] * u + torch.sin(th)[:, None] * v
    bek_v = -torch.sin(th)[:, None] * u + torch.cos(th)[:, None] * v
    e1 = float((torch.einsum("mij,mj->mi", Ra, u) - bek_u).abs().max())
    e2 = float((torch.einsum("mij,mj->mi", Ra, v) - bek_v).abs().max())
    assert max(e1, e2) < TOL, f"Ru/Rv yanlis: {e1:.2e} {e2:.2e}"
    return f"Ru hata {e1:.1e}   Rv hata {e2:.1e}"


@kapi("3  TEK DUZLEM -- duzlem disinda OZDESLIK, ve 2/D iddiasi")
def _3():
    m = kur(D=32)
    R = m.donme()[m.ix_acik]
    u = F.normalize(m.u, dim=-1)
    v = m.v - (m.v * u).sum(-1, keepdim=True) * u
    v = F.normalize(v, dim=-1)
    g = torch.Generator().manual_seed(1)
    w = torch.randn(len(u), m.D, generator=g)
    w = w - (w * u).sum(-1, keepdim=True) * u
    w = w - (w * v).sum(-1, keepdim=True) * v          # duzleme DIK
    e = float((torch.einsum("mij,mj->mi", R, w) - w).abs().max())
    assert e < TOL, f"dik bilesen DEGISTI: {e:.2e}"

    # DENKLEM §4.3: rastgele farkin ancak 2/D'si dokunuluyor
    dlt = F.normalize(torch.randn(2000, m.D, generator=g), dim=-1)
    pay = ((dlt @ u.T) ** 2 + (dlt @ v.T) ** 2).mean()
    bek = 2.0 / m.D
    assert abs(float(pay) - bek) < 0.02, f"pay {float(pay):.4f} != {bek:.4f}"
    return f"dik hata {e:.1e}   duzlem payi {float(pay):.4f} (beklenen {bek:.4f})"


@kapi("4  IZOMETRI -- ||Ra-Rb|| = ||a-b||")
def _4():
    m = kur()
    R = m.donme()
    g = torch.Generator().manual_seed(2)
    a, b = torch.randn(m.n, m.D, generator=g), torch.randn(m.n, m.D, generator=g)
    Ra = torch.einsum("nij,nj->ni", R, a)
    Rb = torch.einsum("nij,nj->ni", R, b)
    e = float(((Ra - Rb).norm(dim=-1) - (a - b).norm(dim=-1)).abs().max())
    assert e < TOL, f"mesafe korunmadi: {e:.2e}"
    return f"max mesafe sapmasi {e:.1e}"


@kapi("5  CAPA TEOREMI -- 'gecmisten BAGIMSIZ' (comp'un tek sebebi)")
def _5():
    """DENKLEM §3.1: ayni koda capalanan IKI FARKLI gecmis, ayni
    sonekle AYNI yere gitmeli. Bu tutmuyorsa comp ~ a^2 de tutmaz."""
    m = kur(n=40, D=12, d=4, K=8)
    R = m.donme()
    C = m.C.detach()
    sonek = [3, 7, 11]

    def yuru(bas, zorla_kod):
        z = torch.zeros(1, m.D)
        z[:, :m.d] = m.p[bas]
        z = C[zorla_kod:zorla_kod + 1]                 # CAPA
        for w in sonek:
            z = z @ R[w].T
        return z

    z1 = yuru(0, 2)
    z2 = yuru(17, 2)                                   # BASKA gecmis, AYNI kod
    e = float((z1 - z2).abs().max())
    assert e < 1e-6, f"capa gecmisi silmedi: {e:.2e}"

    # ve FARKLI kod FARKLI yere gitmeli (kapinin bos olmadigi)
    z3 = yuru(0, 5)
    assert float((z1 - z3).norm()) > 0.1, "farkli kod ayni yere gitti"
    return f"ayni kod: fark {e:.1e}   farkli kod: {float((z1-z3).norm()):.3f}"


@kapi("6  GRADYAN SONMEZ -- 'tekil degerleri 1'")
def _6():
    m = kur()
    R = m.donme()
    J = torch.eye(m.D)
    for w in [1, 5, 9, 2, 30, 31]:
        J = R[w] @ J
    sv = torch.linalg.svdvals(J)
    e = float((sv - 1).abs().max())
    assert e < TOL, f"tekil degerler 1 degil: {e:.2e}"
    return f"6 adimlik bileske, max|sv-1| {e:.1e}"


@kapi("7  ACISAL OKUMA -- gizli kutleden BAGIMSIZ (§4.2)")
def _7():
    m = kur()
    g = torch.Generator().manual_seed(3)
    q = F.normalize(torch.randn(5, m.d, generator=g), dim=-1)
    z1 = torch.zeros(5, m.D); z1[:, :m.d] = q * 1.0           # gizli kutle 0
    z2 = torch.zeros(5, m.D); z2[:, :m.d] = q * 0.3
    z2[:, m.d:] = 0.5                                          # gizli kutle VAR
    t = torch.zeros(5, dtype=torch.long)
    d1 = m.oku(z1[:, None], t[:, None])
    d2 = m.oku(z2[:, None], t[:, None])
    e = float((d1 - d2).abs().max())
    assert e < TOL, f"gizli kutle okumayi degistirdi: {e:.2e}"
    return f"gizli kutle 0 vs 0.5 -> okuma farki {e:.1e}"


@kapi("8  D > d KAPISI")
def _8():
    try:
        M.Yol(10, D=8, d=8)
        raise RuntimeError("D=d kabul edildi")
    except AssertionError:
        pass
    return "D=d reddedildi"


@kapi("9  PARAMETRE SAYISI -- belgedeki formulle birebir")
def _9():
    m = kur(n=475, D=32, d=8, K=2048, k_tam=80)
    T = 32 * 31 // 2
    bek = 80 * T + 395 * (2 * 32 + 1) + 2048 * 32
    assert M.n_par(m) == bek, f"{M.n_par(m)} != {bek}"
    assert bek == 130891, f"belge 130.891 diyor, cikan {bek}"
    return f"{bek:,}  (kapali 80x{T} + acik 395x65 + kod 2048x32)"


@kapi("10  KAYIP -- sonlu, ve gradyan BUTUN parametrelere ulasiyor")
def _10():
    m = kur()
    g = torch.Generator().manual_seed(4)
    X = torch.randint(0, m.n, (6, 9), generator=g)
    L, uye = m.kayip(X)
    assert torch.isfinite(L), f"kayip sonlu degil: {L}"
    L.backward()
    bos = [ad for ad, p in m.named_parameters()
           if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not bos, f"gradyan ulasmadi / NaN: {bos}"
    nrm = {ad: float(p.grad.norm()) for ad, p in m.named_parameters()}
    assert nrm["C"] > 0, "kod defterine gradyan gitmiyor"
    return f"L={float(L):.4f} uye={float(uye):.4f}  grad " + \
           " ".join(f"{k}:{v:.1e}" for k, v in nrm.items())


@kapi("11  SAAT -- kapaliyken hedef SABIT, acikken DEGISIYOR")
def _11():
    mk = kur(saat=False)
    H = mk.hedef(4)
    assert float((H[0] - H[3]).abs().max()) < 1e-6, "saat kapali ama hedef kaydi"
    ma = kur(saat=True)
    Ha = ma.hedef(4)
    fark = float((Ha[0] - Ha[3]).abs().max())
    assert fark > 1e-3, f"saat acik ama hedef kaymadi: {fark:.2e}"
    nrm = float((Ha.norm(dim=-1) - 1).abs().max())
    assert nrm < TOL, f"hedef birim normda degil: {nrm:.2e}"
    return f"kapali: sabit   acik: t=0 vs t=3 farki {fark:.3f}"


@kapi("12  CAPA YARICAPI -- r=0 hic, r=buyuk hep")
def _12():
    m = kur()
    g = torch.Generator().manual_seed(5)
    X = torch.randint(0, m.n, (4, 8), generator=g)
    v0 = m.yol(X, r=0.0)["vur"]
    vb = m.yol(X, r=1e6)["vur"]
    assert not v0.any(), "r=0'da capa tetiklendi"
    assert vb[:, 1:].all(), "r=sonsuzda capa tetiklenmedi"
    return f"r=0 -> {int(v0.sum())} capa   r=inf -> {int(vb.sum())} capa"


@kapi("13  URETIM -- sekil ve onek sadakati")
def _13():
    m = kur()
    onek = [3, 9, 14]
    yol, puan = m.uret(onek, en_cok=5, isin=8, r=0.25)
    assert yol[:len(onek)] == onek, f"onek bozuldu: {yol[:len(onek)]}"
    assert len(yol) <= len(onek) + 5, f"cok uzun: {len(yol)}"
    assert all(0 <= w < m.n for w in yol), "sozluk disi birim"
    return f"onek {onek} -> {yol}  puan {puan:.3f}"


@kapi("14  DURUM KUREDE -- capa sonrasi da")
def _14():
    m = kur()
    g = torch.Generator().manual_seed(6)
    X = torch.randint(0, m.n, (4, 8), generator=g)
    z = m.yol(X, r=1e6)["z"]                           # her adimda capa
    nrm = z.norm(dim=-1)
    e = float((nrm - 1).abs().max())
    assert e < TOL, (f"capa sonrasi durum kurede DEGIL: max|,|z|-1| = {e:.3f}"
                     "  -> kod defteri normalize edilmeli")
    return f"max| |z|-1 | = {e:.1e}"


@kapi("15  BELLEK -- oku() (B,L,n)'den buyuk tensor kurmamali")
def _15():
    m = kur(n=200, D=16, d=4, K=32)
    g = torch.Generator().manual_seed(7)
    X = torch.randint(0, m.n, (64, 12), generator=g)
    y = m.yol(X); z, t = y["z"], y["t"]
    d2 = m.oku(z, t)
    assert d2.shape == (64, 12, 200), d2.shape
    return f"cikti {tuple(d2.shape)}  saat={m.saat}"


@kapi("16  KOD DEFTERI KURENIN USTUNDE KALMALI -- C bir Parameter, KAYAR")
def _16():
    """14 yalniz BASLANGICTA gecer: C normalize baslatiliyor. Egitimde
    C serbest bir parametre ve kureden cikar. O zaman capa durumu
    kureden atar ve 'norm korunur' iddiasi duser."""
    m = kur()
    with torch.no_grad():
        m.C.mul_(2.5)                                  # egitimin yapacagi sey
    g = torch.Generator().manual_seed(8)
    X = torch.randint(0, m.n, (4, 8), generator=g)
    z = m.yol(X, r=1e6)["z"]
    e = float((z.norm(dim=-1) - 1).abs().max())
    assert e < TOL, (f"C kayinca durum kureden cikti: max| |z|-1 | = {e:.3f}"
                     "  -> ileri geciste C normalize edilmeli")
    return f"C x2.5 sonrasi max| |z|-1 | = {e:.1e}"


@kapi("17  BELLEK -- saat kapaliyken hedef() HIC kurulmamali")
def _17():
    """oku() icinde H[t] (B,L,n,d) demek. B=8192 L=16 n=475 d=8 ->
    498 milyon float = 2 GB. Saat kapaliyken gerek YOK: hedef sabit,
    q @ p^T yeter -> (B,L,n)."""
    m = kur(n=60, D=12, d=4, K=16)
    g = torch.Generator().manual_seed(9)
    X = torch.randint(0, m.n, (8, 7), generator=g)
    y = m.yol(X); z, t = y["z"], y["t"]
    cagri = []
    ger = m.hedef
    m.hedef = lambda *a, **k: (cagri.append(1), ger(*a, **k))[1]
    d2 = m.oku(z, t)
    m.hedef = ger
    q = F.normalize(z[..., :m.d], dim=-1)
    duz = (2 - 2 * (q @ m.p.T)).clamp(min=0)
    assert float((d2 - duz).abs().max()) < TOL, "hizli yol yanlis sonuc veriyor"
    assert not cagri, "saat KAPALI ama hedef() kuruldu -- (B,L,n,d) tensoru"
    return "saat kapali -> hedef() kurulmadi, sonuc duz formulle ayni"


@kapi("18  CAPA TEOREMI, GERCEK KOD YOLUYLA")
def _18():
    """5 teoremi elle kuruyordu. Bu, `yol()`un KENDISINI sinar: iki
    FARKLI onek ayni koda capalanip AYNI sonekle yurusun."""
    m = kur(n=40, D=12, d=4, K=8)
    sonek = [3, 7, 11]
    X1 = torch.tensor([[0, 5, 9] + sonek])
    X2 = torch.tensor([[17, 22, 31] + sonek])
    # Teoremin KOSULU: ayni koda capalanmak. Egitimsiz modelde bu
    # tesadufi, o yuzden kosulu SAGLAYAN bir cift ARANIYOR -- yoksa
    # kapi bos gecer ve hicbir sey sinamaz.
    g = torch.Generator().manual_seed(11)
    onek = torch.randint(0, m.n, (60, 3), generator=g)
    son = torch.tensor(sonek).expand(60, 3)
    y = m.yol(torch.cat([onek, son], 1), r=1e6)
    kod3 = y["k"][:, 2]                                # 3. adimdaki kod
    ciftler = [(i, j) for i in range(60) for j in range(i + 1, 60)
               if int(kod3[i]) == int(kod3[j])
               and not torch.equal(onek[i], onek[j])]
    assert ciftler, "ayni koda dusen iki FARKLI onek bulunamadi -- kapi bos"
    i, j = ciftler[0]
    e = float((y["z"][i, -1] - y["z"][j, -1]).abs().max())
    assert e < 1e-5, f"ayni koda capalandi ama son durum farkli: {e:.2e}"
    # ve kapinin bos olmadigi: FARKLI koda dusen cift FARKLI bitmeli
    fark = [(a, b) for a in range(60) for b in range(60)
            if int(kod3[a]) != int(kod3[b])]
    a, b = fark[0]
    ay = float((y["z"][a, -1] - y["z"][b, -1]).norm())
    assert ay > 1e-3, "farkli kod ayni yere gitti"
    return (f"{len(ciftler)} cift ayni koda dustu; onek {onek[i].tolist()} vs "
            f"{onek[j].tolist()} -> son durum farki {e:.1e}   "
            f"(farkli kod: {ay:.3f})")


@kapi("19  KOD ARAMA -- ic carpim formu cdist ile BIREBIR, ve cdist YOK")
def _19():
    """Is yukunun %92'si kod aramada (olculdu). z ve C birim normda
    oldugu icin ||z-C||^2 = 2 - 2 z.C  TAM esitliktir; cdist ayni isi
    1,5 kat yavas yapiyordu. Ayrica kayip() yol()un buldugu kodu
    YENIDEN aramamali (B*(L-1) x K = 1 GB)."""
    m = kur()
    C = m.kod().detach()
    g = torch.Generator().manual_seed(10)
    z = F.normalize(torch.randn(32, m.D, generator=g), dim=-1)
    e = float((torch.cdist(z, C) ** 2 - (2 - 2 * (z @ C.T))).abs().max())
    assert e < 1e-4, f"ic carpim formu cdist'ten sapiyor: {e:.2e}"

    say = []
    ger = torch.cdist
    torch.cdist = lambda *a, **k: (say.append(1), ger(*a, **k))[1]
    try:
        X = torch.randint(0, m.n, (4, 9), generator=g)
        m.kayip(X)
    finally:
        torch.cdist = ger
    assert not say, f"cdist hala {len(say)} kez cagriliyor"
    return f"cdist ile fark {e:.1e}   kayip()ta cdist cagrisi {len(say)}"


@kapi("20  IZLER -- graf ve sinav model_13 ile BIREBIR")
def _20():
    import ayar_14 as AY, taban_14 as MT, veri_14 as V
    assert V.IZ == AY.IZ_GRAF, f"graf izi {V.IZ} != {AY.IZ_GRAF}"
    v = MT.veri_kur(AY.AYAR, yaz=lambda *a, **k: None)
    L = MT.olcme_listeleri(AY.AYAR, v)
    iz = MT.olcme_izi(L)
    assert iz == AY.IZ_OLCME, f"olcme izi {iz} != {AY.IZ_OLCME}"
    return (f"graf {V.IZ}  olcme {iz}   bolmeler "
            + " ".join(f"{k}:{len(x)}" for k, x in L.items()
                       if hasattr(x, '__len__')))


@kapi("21  KOPYA -- VERI URETEN dosyalar model_13 ile AYNI (ad haric)")
def _21():
    import os, re
    fark = {}
    # !! `ek_14` LISTEDE YOK: artik kopya DEGIL, kolun kendi kodu.
    # Bolucu bastan yazildi (cok ekli kelime bolunmuyordu, morfotaktik
    # yoktu, `Bolumu`/`bolumu` carpisiyordu, KOK_BILINEN eklendi).
    #
    # !! `metin` de LISTEDE YOK (21 Eylul). Sinav yuzeyi artik
    # `metin_14.sinav_yuzeyi`den geliyor -- `olcme_14` jetonu ELLE
    # dizmeyi birakti, cunku elle dizilen liste tokenizer degisince
    # sessizce eskidi ve SINAV HIC KOSMADI (kapi 34).
    # Ama korunmasi gereken sey METNIN ayniligi degil, KORPUSUN
    # DEGISMEMESI. Asagida metin degil DAVRANIS sinaniyor.
    for f in ("veri", "korpus", "jeton"):
        a = open("../model_13/%s_13.py" % f, encoding="utf-8").read()
        c = open("%s_14.py" % f, encoding="utf-8").read()
        A = re.sub("_13" + chr(92) + "b", "_14", a).replace("model_13", "model_14")
        if A != c:
            fark[f] = sum(1 for x, y in zip(A.split(chr(10)),
                                            c.split(chr(10))) if x != y)
    assert not fark, "kopya ADLANDIRMADAN BASKA yerde degismis: %s" % fark

    # --- metin_14 KORPUSA dokunmuyor mu.  Degisiklik:
    #       eski  " ".join(... for x in w[:-1]) + " " + w[-1]
    #       yeni  " ".join([... for x in w[:-1]] + [w[-1]])
    # N>=2'de CEBIRSEL OLARAK ayni; N=1'de eski CIFT BOSLUK uretiyordu
    # ("Yildiz'in  kardesi"). Korpus `yol()`u hep 2-3 iliskiyle
    # cagiriyor (korpus_14:250/381/388) -> korpus DEGISMEDI.
    # Kapi bunu VARSAYMIYOR, her ciftte hesapliyor.
    import metin_14 as MT, veri_14 as V
    eski = lambda w: (" ".join(x + MT._nin(x, False) for x in w[:-1])
                      + " " + w[-1])
    rl = list(V.ILISKI)
    n = 0
    for i in range(len(rl)):
        for j in range(len(rl)):
            for rr in ((rl[i], rl[j]),
                       (rl[i], rl[j], rl[(i + j) % len(rl)])):
                w = [V.TR_ILISKI[r] for r in rr]
                assert MT._yol_parca("Cem_Yildiz", rr,
                                     "Ceren_Yildiz")[1] == eski(w), (
                    "N=%d KORPUSU DEGISTIRIR: %s" % (len(rr), rr))
                n += 1
    w1 = [V.TR_ILISKI[rl[0]]]
    assert eski(w1).startswith(" "), "N=1 eski hali zaten dogruymus"
    assert not MT._yol_parca("Cem_Yildiz", (rl[0],),
                             "Ceren_Yildiz")[1].startswith(" ")
    return ("veri korpus jeton -- saf yeniden adlandirma   "
            "metin: %d N>=2 tamlamasi BIREBIR ayni (korpus degismedi), "
            "N=1 cift boslugu duzeldi" % n)


@kapi("22  AYAR -- SABIT alanlar model_13 degerleriyle ayni")
def _22():
    import ayar_14 as AY
    import importlib.util as iu
    sp = iu.spec_from_file_location("a13", "../model_13/ayar_13.py")
    # ayar_13 taban_13'u import ediyor; yalniz degerleri metinden okuyalim
    src = open("../model_13/ayar_13.py", encoding="utf-8").read()
    import re
    kayan = []
    for a in AY.SABIT:
        m = re.search(rf"^\s*{a}=([^,#]+),", src, re.M)
        if not m:
            continue
        bek = eval(m.group(1).strip())
        var = getattr(AY.AYAR, a)
        if bek != var:
            kayan.append(f"{a}: {var} != {bek}")
    assert not kayan, "SINAV AYRISIR: " + "; ".join(kayan)
    return f"{len(AY.SABIT)} sabit alan, hepsi model_13 ile ayni"


@kapi("23  BIRIM -- pencere sekli ve kapali sinif secimi")
def _23():
    import birim_14 as B
    import numpy as np
    dizi = np.array([3, 1, 0, 1, 2, 1, 0, 4, 1, 0], dtype=np.int64)
    b = B.Birim(["a", "b", "c", "d", "e"], dizi, {}, set(), frozenset())
    assert list(b.say) == [3, 4, 1, 1, 1], list(b.say)
    m = b.kapali(2)
    # nonzero() SIRALI doner, argsort sirasini degil -- en sik ikisi {0,1}
    assert set(m.nonzero()[0].tolist()) == {0, 1}, "en sik iki birim secilmedi"
    P = b.pencere(4)
    assert P.shape == (7, 4), P.shape          # len-L+1 = 7
    assert list(P[0]) == [3, 1, 0, 1], P[0]
    assert list(P[-1]) == [0, 4, 1, 0], P[-1]  # son pencere DUSMEMELI
    return f"say {list(b.say)}  kapali {sorted(m.nonzero()[0])}  pencere {P.shape}"


@kapi("24  OLCME -- iskelet, ek orani, BICIM|BILGI ayrimi")
def _24():
    import olcme_14 as O
    import numpy as np
    #  0..3 VARLIK   4=-TAMLAYAN(ek) 5=tezi 6=-BILDIRME(ek) 7='.' 8='?'
    akis = np.array([0, 1, 4, 5, 2, 3, 6, 7] * 3 + [0, 1, 4, 5, 8], np.int64)
    b = O.Bicim(akis, {0, 1, 2, 3}, {7, 8}, ek_ix={4, 6}, n=12)
    assert len(b.iskelet) == 2, dict(b.iskelet)
    assert b.puanla([0, 1, 4, 5, 2, 3, 6, 7])["kalip"] == 1
    y = b.puanla([0, 1, 5, 4, 2, 3, 6, 7])          # 4 yanlis yerde
    assert y["ek_n"] == 2 and y["ek_iyi"] == 1, y
    assert b.puanla([0, 1, 4, 4, 4, 6, 7])["yozlasma"] == 1
    assert b.puanla([0, 1, 4, 5, 2, 3, 6])["kapanmadi"] == 1

    # BICIM tutup BILGI dusen durum -- model_13'un arizasinin sekli
    sor = [O.Soru((0, 1, 4, 5), (2, 3), ())]
    yanlis_kisi = [[0, 1, 6, 7]]
    tip = lambda w: {"K"} if w in (0, 1, 2, 3) else {"X"}
    bc = O.bicim_puanla(sor, yanlis_kisi, b, {7, 8}, tip)
    bl = O.bilgi_puanla(sor, yanlis_kisi, {7, 8})
    assert bc["tip"] == 1.0, "tip BICIM'de sayilmali"
    assert bl["tam"] == 0.0, "yanlis kisi BILGI'de dusmeli"
    bos = O.bicim_puanla(sor, [[0, 1, 7]], b, {7, 8}, tip)
    assert bos["ek_n"] == 0 and bos["ek"] != bos["ek"], "ek yoksa NaN"
    return ("iskelet %d   yanlis kisi -> BICIM tip %.1f / BILGI tam %.1f"
            % (len(b.iskelet), bc["tip"], bl["tam"]))


@kapi("25  SINIF KORPUSTAN -- zincirin ETIKETINDEN DEGIL")
def _25():
    import olcme_14 as O
    import numpy as np
    ham = np.asarray([9, 0, 4, 5, 2, 6, 7], np.uint16).tobytes()
    assert O._gecer(ham, (0, 4, 5, 2)), "gecen dizi bulunamadi"
    assert not O._gecer(ham, (0, 4, 5, 3)), "gecmeyen dizi bulundu"
    h2 = np.asarray([0x0102, 0x0304], np.uint16).tobytes()
    assert not O._gecer(h2, (0x0203,)), "tek ofset yanlis eslesti"
    return "bayt aramasi + hizalama dogru"



class _Buyuk(torch.overrides.TorchFunctionMode):
    """Esikten buyuk her tensor uretimini kaydeder (ad, sekil, grad)."""

    def __init__(self, esik):
        self.esik, self.v = esik, []

    def __torch_function__(self, f, t, a=(), k=None):
        o = f(*a, **(k or {}))
        if isinstance(o, torch.Tensor) and o.numel() >= self.esik:
            self.v.append((f.__name__, tuple(o.shape), o.requires_grad))
        return o


@kapi("26  SICAK DONGU -- buyuk gecici tensor SAYISI")
def _26():
    """Bu kapi bir HIZ kapisi, ve gerekcesi OLCULDU: ilk kosuda epok
    220 sn'yi gecti (>117 ms/adim). Sebep FLOP degil BELLEK TRAFIGI --
    adim basina (B,K) uzerinde dort elemanwise tensor ve okuma
    tarafinda (B,L,n) uzerinde uc tane kuruluyordu.

    Iki sayi kilitleniyor:
      (B,L-W,n)   1 tane  -- yalniz cos.  2-2x ve clamp REDUKSIYONDAN
                  SONRA, (B,n) uzerinde yapilir.  W = ISINMA.
      (B,K)       L-1 tane -- ISINMA'dan BAGIMSIZ: yol butun pencere
                  boyunca kosar, puanlanmayan yalniz KAYIP tarafi.
                  Ve HICBIRI requires_grad DEGIL.
                  `k` bir indeks, `vur` bir bool; aramadan geri hicbir
                  sey akmaz. Gradyanli olsaydi 15 x 67 MB geri gecise
                  kadar TUTULURDU."""
    import ayar_14 as AY
    n, D, d, K, B, L, W = 451, 32, 8, 2048, 512, 16, AY.ISINMA
    tam = torch.zeros(n, dtype=torch.bool)
    tam[:80] = True
    m = M.Yol(n, D=D, d=d, K=K, tam=tam)
    g = torch.Generator().manual_seed(26)
    X = torch.randint(0, n, (B, L), generator=g)

    with _Buyuk(B * (L - W) * n) as s1:
        m.kayip(X, isin=W)
    with _Buyuk(B * K) as s2:
        m.kayip(X, isin=W)
    bk = [x for x in s2.v if x[1] == (B, K)]
    grad = [x for x in bk if x[2]]

    assert len(s1.v) == 1, "(B,L-W,n) boyunda %d tensor: %s" % (
        len(s1.v), [x[:2] for x in s1.v])
    assert len(bk) == L - 1, "(B,K) boyunda %d tensor (beklenen %d)" % (
        len(bk), L - 1)
    assert not grad, "%d kod-arama tensoru GRADYANLI -- geri gecise kadar tutulur" % len(grad)
    return ("(B,L-%d,n) 1 tensor   (B,K) %d tensor, gradyanli 0"
            % (W, len(bk)))

@kapi("27  IZ -- `son` sozlugu kayipla BIREBIR toplaniyor")
def _27():
    """Egitim hucresi NaN'i bu sozlukten teshis edecek. Terimler
    kayiptan AYRI hesaplansaydi ikisi sessizce ayrisirdi; burada
    ayni ifadeden kopyalandiklari SAYIYLA dogrulaniyor.

    Ayrica `saglik` uc normalize paydasini basiyor: olculdu, bir
    yonun normu F.normalize'in eps'inin (1e-12) hemen ustune
    duserse geri gecis 1e12 mertebesinde gradyan uretiyor."""
    m = kur()
    g = torch.Generator().manual_seed(27)
    X = torch.randint(0, m.n, (8, 9), generator=g)
    a1, a2, a3, beta = 1.3, 0.7, 2e-4, 0.25
    L, uye = m.kayip(X, a1, a2, a3, 0.4, beta, 0.25)
    s = m.son
    bek = s["uye"] + a1 * s["dis"] + a2 * (s["kod"] + beta * s["bag"])         + a3 * s["duzen"]
    e = float((bek - s["top"]).abs())
    assert e == 0.0, f"iz toplami kayiptan sapiyor: {e:.2e}"
    assert float(s["top"]) == float(L) and float(s["uye"]) == float(uye)
    for ad in ("capa", "Pz_min"):
        assert ad in s and torch.isfinite(s[ad]), ad
    assert "capa" in M.saglik(m) and "v_dik" in M.saglik(m)
    return f"iz kayipla birebir (fark {e:.1e})   saglik tablosu 8 alan"

@kapi("28  ITME TEKILLIGI -- sqrt'e SIFIR gidemez")
def _28():
    """OLCULDU (20 Eylul), egitim ADIM 1'DE NaN verdi.

    `kos` bir kosinus ama fp32'de 1'i asiyor: olculen en kucuk
    2-2kos degeri -2,384e-07. clamp(min=0) onu TAM 0 yapiyor ve
    sqrt'un turevi orada tanimsiz.

    !! KAPI DEGERI SINIYOR, NaN'i DEGIL. Cunku ClampBackward NaN'i
    CPU'da yutuyor, CUDA'da yutmuyor -- "gradyan sonlu mu" diye
    soran bir kapi CPU'da BOS cikar (denendi: eski kodu geciriyor).
    Sinanan sey `dmin >= sqrt(TABAN)`: sqrt'e sifir gitmiyorsa
    tekillik hicbir cihazda olusamaz."""
    m = kur(n=60, D=12, d=4, K=64)
    g = torch.Generator().manual_seed(28)
    X = torch.randint(0, 60, (8, 9), generator=g)
    B, L = X.shape
    taban = M.TABAN ** 0.5 * (1 - 1e-6)   # fp32 sqrt yuvarlamasi

    # 1) TAM cakisma + kosinusun 1'i ASMASI, hem maskeli hem maskesiz
    kos = torch.full((B, L - 1, 60), 0.3, requires_grad=True)
    with torch.no_grad():
        kos[0, 0, int(X[0, 0])] = 1.0          # maskeli birim, TAM 1
        kos[1, 0, int(X[1, 0])] = 1.0 + 1e-7   # maskeli, 1'i ASIYOR
        dis_ = {int(v) for v in X[2]}
        bos = next(i for i in range(60) if i not in dis_)
        kos[2, 0, bos] = 1.0                   # MASKESIZ birim, TAM 1
    dis, dmin = m._itme(kos, X, 0.4)
    assert float(dmin.min()) >= taban, (
        f"sqrt'e sifir gidiyor: dmin_min = {float(dmin.min()):.3e} "
        f"< {taban:.3e}")
    (gk,) = torch.autograd.grad(dis, kos)
    assert torch.isfinite(dis) and torch.isfinite(gk).all(), "sonsuz"

    # 2) Tekillik YOKKEN sayi degismemeli -- taban bir yaklasim degil
    kos2 = torch.rand(B, L - 1, 60, generator=g) * 1.6 - 0.8
    d2 = (2 - 2 * kos2.amax(1)).clamp(min=0)
    ic = torch.zeros_like(d2, dtype=torch.bool).scatter_(1, X, True)
    bek = ((0.4 - d2.sqrt()).clamp(min=0).pow(2)
           .masked_fill(ic, 0.).sum(1).mean())
    e = float((m._itme(kos2, X, 0.4)[0] - bek).abs())
    assert e == 0.0, f"taban sayiyi degistirdi: {e:.2e}"

    # 3) Gradyan TAVANI: 2*delta / (2*sqrt(TABAN))
    tavan = 0.4 / taban
    assert float(gk.abs().max()) <= tavan, (
        f"gradyan tavani asildi: {float(gk.abs().max()):.2e} > {tavan:.2e}")
    return (f"cakismada dmin_min {float(dmin.min()):.1e} >= {taban:.0e}   "
            f"tekillik yokken fark 0.0   grad <= {tavan:.0e}")

@kapi("29  ZINCIR DUZENI -- hicbir zincir SESSIZCE dusmemeli")
def _29():
    """*Gerekce OLCULDU (20 Eylul):* `tum()` `3 <= len <= 4` suzuyor
    ve `adim = len-2` diyordu. Gercek duzen:

        1 adim   (e, r, cevap)                uzunluk 3
        2 adim   (e, r1, r2, KOPRU, cevap)    uzunluk 5

    Uzunluk 5 hicbirine uymuyordu: 14.043 zincirin 11.043'u SESSIZCE
    dustu, olcum yalniz 1 adimi sindi ve CIKARIM 8 ornege indi.
    Hicbir sayi bunu gostermedi -- "zincir 3000" satiri makul
    duruyordu. Kapi HAVUZUN TAMAMINI sayiyor."""
    import numpy as np
    import ayar_14 as AY, taban_14 as MT, veri_14 as V, olcme_14 as O
    v = MT.veri_kur(AY.AYAR, yaz=lambda *a, **k: None)
    L = MT.olcme_listeleri(AY.AYAR, v)

    bek = 0
    for ad, zs in L.items():
        if not hasattr(zs, "__len__"):
            continue
        bek += len(zs)
        uz = sorted({len(z) for z in zs})
        assert uz and set(uz) <= {3, 5}, f"{ad}: uzunluk {uz}"

    # adim UZUNLUKTAN dogru cikiyor mu -- grafi yuruyerek
    F = v.facts
    for ad, zs in L.items():
        if not hasattr(zs, "__len__"):
            continue
        for z in list(zs)[:200]:
            z = [int(x) for x in z]
            adim = (len(z) - 1) // 2
            cur = z[0]
            for r in z[1:1 + adim]:
                cur = int(F[cur, r])
                if cur < 0:
                    break
            assert cur == z[-1], f"{ad}: adim {adim} zincirin sonunu vermiyor"

    say = {}

    class _Say(O.Sorular):
        def kur(self, z, adim):
            say[(len(z), adim)] = say.get((len(z), adim), 0) + 1
            return None

    # !! Bu kapi `kur`u OVERRIDE ediyor, yani SORU YUZEYINI hic
    # kurmuyor -- yalniz HAVUZU sayiyor. Yuzeyin kendisini kapi 34
    # siniyor; 21 Eylul'de sinavin hic kosmadigi tam bu bosluktan
    # gecmisti.
    S = _Say(v, [], [], [], {}, {}, {}, {}, set())
    S.tum(L, np.zeros(4, np.int64), yaz=lambda *a: None)
    gecen = sum(say.values())
    assert gecen == bek, (
        f"havuza {gecen} zincir girdi, hukumdeki bolmelerde {bek} var -- "
        f"{bek - gecen} tanesi SESSIZCE dustu")
    assert not hasattr(O.Sorular, "HUKUM_DISI"), (
        "olcu bir bolmeyi ADIYLA eliyor -- HUKUM_DISI geri gelmis")
    assert set(say) == {(3, 1), (5, 2)}, say
    return f"{bek} zincir, hepsi havuzda   (uzunluk, adim) -> {say}"


@kapi("30  ARAMA -- indeksli arama bayt aramasiyla BIREBIR")
def _30():
    """`Arama` bir HIZ duzeltmesi ve gerekcesi OLCULDU: `_gecer`
    BULUNMAYAN bir dizide 30,7 MB'lik akisin tamamini tariyor,
    sinavin 14.123 sorusunun 11.123'unde cevap korpusta YOK ve olcum
    5 dakikayi gecti -- her kosuda odenecekti.

    Hiz duzeltmesi ANCAK ayni cevabi veriyorsa duzeltmedir. Kapi
    ikisini gercek akista karsilastiriyor: gecen diziler (akistan
    KESILEREK alinir, yani kesin var), gecmeyenler (ayni diziler
    bozularak) ve kenar durumlar."""
    import numpy as np
    import olcme_14 as O

    g = torch.Generator().manual_seed(30)
    d = torch.randint(0, 60, (200000,), generator=g).numpy().astype(np.int64)
    ham = np.asarray(d, np.uint16).tobytes()
    ara = O.Arama(d)

    ornek = []
    for i in torch.randint(0, len(d) - 20, (400,), generator=g).tolist():
        for uz in (1, 2, 3, 5, 9):
            ornek.append(tuple(int(x) for x in d[i:i + uz]))     # VAR
    for q in list(ornek[:400]):
        ornek.append(q[:-1] + (60 + 1,) if len(q) > 1 else (60 + 1,))  # YOK
    ornek += [(int(d[-1]),), tuple(int(x) for x in d[-3:]),
              tuple(int(x) for x in d[-2:]) + (0,)]              # KENAR

    fark = [q for q in ornek if ara(q) != O._gecer(ham, q)]
    assert not fark, "%d dizide indeks bayt aramasindan AYRILIYOR: %s" % (
        len(fark), fark[:3])
    var = sum(ara(q) for q in ornek)
    assert 0 < var < len(ornek), "kapi bos: hepsi ayni cevabi veriyor"
    return "%d dizi, %d var / %d yok -- ikisi de AYNI cevabi verdi" % (
        len(ornek), var, len(ornek) - var)


@kapi("31  OPERATOR -- OLGUYU TASIYAN birim zayif donmeye dusmemeli")
def _31():
    """*Gerekce OLCULDU (20 Eylul, t0):* `K_TAM = 80` iken 451 birimin
    328'i VARLIK birimiydi (ad parcasi) ve bunlarin yalniz 24'u (%7,3)
    frekansla ilk 80'e giriyordu. Yani "Elif" ile "Hasan" arasindaki
    farki yaratmasi gereken 304 birim, rastgele bir farkin %6,2'sine
    dokunan TEK DUZLEM donmesine mahkumdu (§4.3'un kendi sayisi).

    Modelin yazdiginda birebir gorundu -- R[ad] ~ I:
        Mersin -> Mersin(0,98)   Bartin -> Bartin(0,94)
        Isikli -> Isikli(0,91)   Sanliurfa x4
    ve DOGRULUK BILGI tam = 0,0000 cikti.

    §4.3'un ayrimi FREKANSA dayaniyordu; olcum frekansin olguyu
    yanlis tarafa koydugunu gosterdi. Kapi frekansa degil ROLE
    bakiyor: varlik birimi TAM SO(D) almali.

    !! IKINCI KAPI (21 Eylul): `K_TAM` bir SAYIYSA sozlukten KUCUK
    olmali. 451 > 444 iken deger "hepsi" demek istiyor ama bunu
    TESADUFEN soyluyordu; sozluk 451'i gecse ayrim KENDILIGINDEN ve
    KEYFI bir kesimle geri gelir, ve YUKARIDAKI oran kapisi bunu
    yakalamaz -- cunku o gun varlik birimleri hala ilk 451'de
    kalabilir. "Hepsi" demek isteniyorsa `None` yazilir."""
    import numpy as np
    import ayar_14 as AY, veri_14 as V, birim_14 as BR, olcme_14 as O

    yol = os.environ.get("BIRIM_NPZ", r"G:/Drive'ım/model_14/birim_14.npz")
    if not os.path.exists(yol):
        return "ATLANDI -- birim dosyasi yok (%s)" % yol
    b = BR.yukle(yol, yaz=lambda *a: None)
    G = V.kur(AY.AYAR.veri_tohum)
    E_ad = [x for t in V.TIPLER for x in G["ad"][t]]
    varlik = {w for ad in E_ad
              for w in O.birimle(ad, V.TR, b.kok, b.ix, b.korunan)}
    assert varlik, "varlik birimi bulunamadi -- ad esleme bozuk"

    assert AY.K_TAM is None or AY.K_TAM < len(b.ad), (
        "K_TAM = %s, sozluk %d -- deger sozlugu ASIYOR, yani 'hepsi'yi "
        "TESADUFEN soyluyor. Niyet buysa `None` yazin; degilse kesim "
        "OLCUYLE secilsin. Boyle birakilirsa sozluk buyudugunde frekans "
        "ayrimi kendiliginden geri gelir ve hicbir kapi soylemez."
        % (AY.K_TAM, len(b.ad)))

    tam = M.sinif_ayir(b.say, AY.K_TAM)
    guclu = {i for i in varlik if bool(tam[i])}
    oran = len(guclu) / len(varlik)
    assert oran > 0.99, (
        "%d varlik biriminin yalniz %d'u (%%%.1f) TAM SO(D) aliyor; "
        "kalan %d tanesi farkin 2/D = %%%.1f'ine dokunan TEK DUZLEM "
        "donmesiyle olgu tasimak zorunda (K_TAM=%d)"
        % (len(varlik), len(guclu), 100 * oran, len(varlik) - len(guclu),
           100 * 2 / AY.D_DURUM, AY.K_TAM))
    return ("%d birimin %d'i varlik birimi; hepsi TAM SO(%d)   "
            "(K_TAM=%s, acik sinif %d)"
            % (len(b.ad), len(varlik), AY.D_DURUM,
               "None (hepsi)" if AY.K_TAM is None else AY.K_TAM,
               int((~tam).sum())))


@kapi("32  ISINMA -- kayip ilk ISINMA konumunu GORMUYOR")
def _32():
    """§13/A3'un kapisi. Karar: cop onek PUANLANMAZ (§5.2).

    Dilim `y["z"][:, isin:]` sessizce `1:`e donerse hicbir sayi
    patlamaz -- kayip yine hesaplanir, yalniz konum 1'in %45,1
    tavani geri gelir. Bu yuzden kapi DEGER ozdesligi kuruyor:
    elle kurulan `uye` ile kayibin bastigi `uye` ayni mi, VE
    isin=1'inkinden FARKLI mi."""
    import ayar_14 as AY
    n, D, d, K, B, L = 120, 16, 6, 256, 64, 24
    W = AY.ISINMA
    m = M.Yol(n, D=D, d=d, K=K, tam=torch.ones(n, dtype=torch.bool))
    g = torch.Generator().manual_seed(32)
    X = torch.randint(0, n, (B, L), generator=g)

    y = m.yol(X, 0.25)
    kos = m._kos(y["z"][:, W:], y["t"][:, W:])
    assert kos.shape[1] == L - W, "puanlanan konum %d, beklenen %d" % (
        kos.shape[1], L - W)
    elde = float(2 - 2 * kos.gather(2, X[:, W:, None]).squeeze(-1).mean())

    m.kayip(X, isin=W)
    e = abs(float(m.son["uye"]) - elde)
    assert e < 1e-6, "uye elle hesapla tutmuyor: fark %.2e" % e

    m.kayip(X, isin=1)
    eski = float(m.son["uye"])
    assert abs(eski - elde) > 1e-4, (
        "isin=%d ile isin=1 AYNI uye veriyor -- dilim ise yaramiyor" % W)

    # HESAP: bir gecis j = i mod ATLA sinifinda kalir; ISINMA ATLA'nin
    # kati oldugu icin dort sinif da ayni sayida konum tutmali.
    say = {c: len([j for j in range(1, AY.PENCERE)
                   if j % AY.ATLA == c and j >= W]) for c in range(AY.ATLA)}
    assert len(set(say.values())) == 1, (
        "kalinti siniflari esit degil: %s -- bir gecis sinifi otekilerden "
        "az puanlaniyor" % say)
    return ("konum %d..%d puanlaniyor (%d/%d)   uye ozdes, isin=1'den "
            "%.4f farkli   her kalinti sinifi %d konum"
            % (W, L - 1, L - W, L - 1, abs(eski - elde),
               next(iter(say.values()))))


@kapi("33  DALLANMA -- uye k=1'de siraya HIZALI, k buyudukce degil")
def _33():
    """§5.1/J'nin kapisi. Egitim YOK, veri YOK -- sabit okuma
    geometrisi (p) ve kaybin tanimi uzerinde aritmetik.

    IKI SEY kilitleniyor:
      k=1  TEOREM. q'yu p_t'ye eta kadar iterken hedefin kazanci
           eta*1, rakibin kazanci eta*<p_t,p_c> <= eta. Sira ASLA
           kotulesemez -> bozulma TAM SIFIR olmali.
      k>1  Gradyan hedeflerin AGIRLIK MERKEZINE gider ve merkez
           hicbir hedefin yeri degildir -> bozulma SIFIRDAN BUYUK.
           Bu satir sifirlanirsa §5.1/J'nin dayanagi gitmis demektir.

    !! Kapi p'nin GERCEK dagilimini kullaniyor (modelin kendi
    buffer'i), elle secilmis bir olcek degil."""
    import ayar_14 as AY
    n, d, B, ETA = 444, AY.D_OKUMA, 2000, 0.10
    m = M.Yol(n, D=AY.D_DURUM, d=d, K=64)
    p = m.p
    g = torch.Generator().manual_seed(33)
    boz = {}
    for k in (1, 11):
        q = F.normalize(torch.randn(B, d, generator=g), dim=-1)
        T = torch.randint(0, n, (B, k), generator=g)
        q2 = F.normalize(q + ETA * F.normalize(p[T].mean(1), dim=-1), dim=-1)
        sira = lambda qq: ((qq @ p.T)[:, None, :] > (qq @ p.T).gather(
            1, T)[..., None]).sum(2) + 1
        boz[k] = float((sira(q2) > sira(q)).float().mean())

    assert boz[1] == 0.0, (
        "k=1'de %.4f hedefin sirasi kotulesti -- TEOREME AYKIRI, ya p "
        "birim normda degil ya adim kureye geri izdusurulmuyor" % boz[1])
    assert boz[11] > 0.05, (
        "k=11'de bozulma %.4f -- §5.1/J'nin dayanagi bu satir; sifira "
        "yakinsa dallanma sulandirmasi YOK demektir" % boz[11])
    # Erisilebilir taban: k hedefin ortalamasinin boyu ~ 1/sqrt(k)
    tab = 2 - 2 / (11 ** 0.5)
    return ("k=1 bozulma %.4f (teorem)   k=11 bozulma %.3f   "
            "k=11 uye tabani %.4f" % (boz[1], boz[11], tab))


@kapi("34  SINAV YUZEYI -- GERCEK sozlukle kurulur, tamami cozulur")
def _34():
    """*Gerekce OLCULDU (21 Eylul):* olcum `bx["-TAMLAYAN"]` uzerinde
    KeyError verdi. Yani `ek_14` yeniden yazildigindan beri SINAV HIC
    KOSMAMISTI ve 34 kapinin hicbiri gormemisti -- cunku hicbiri
    `Sorular`i GERCEK `b.ix` ile kurmuyordu. Kapi 29 sahte bir bx
    veriyor ve `kur`u override ediyor.

    Bu kapi gercegini kuruyor. Sinanan sey: sinav yuzeyi artik
    `metin_14.sinav_yuzeyi`den geliyor ve uretilen her onek/cevap/
    kanit sozluge TAM oturuyor. Jetonu elle dizen bir ikinci liste
    KALMADI -- kalsaydi burada patlardi."""
    import ayar_14 as AY, taban_14 as MT, veri_14 as V
    import birim_14 as BR, olcme_14 as O

    yol = os.environ.get("BIRIM_NPZ", r"G:/Drive'ım/model_14/birim_14.npz")
    if not os.path.exists(yol):
        return "ATLANDI -- birim dosyasi yok (%s)" % yol
    b = BR.yukle(yol, yaz=lambda *a: None)
    v = MT.veri_kur(AY.AYAR, yaz=lambda *a, **k: None)
    G = V.kur(AY.AYAR.veri_tohum)
    E_ad = [x for t in V.TIPLER for x in G["ad"][t]]
    S = O.Sorular(v, E_ad, list(V.ILISKI), V.TIPLER, V.TR, V.TR_ILISKI,
                  b.kok, b.ix, b.korunan)

    # 1 ve 2 adimli zincirleri ELLE kur -- grafin kendi olgularindan
    F = v.facts
    ornek, n1, n2 = [], 0, 0
    for e in range(min(400, F.shape[0])):
        for r in range(F.shape[1]):
            a = int(F[e, r])
            if a < 0:
                continue
            if n1 < 60:
                ornek.append(((e, r, a), 1)); n1 += 1
            for r2 in range(F.shape[1]):
                a2 = int(F[a, r2])
                if a2 >= 0 and n2 < 60:
                    ornek.append(((e, r, r2, a, a2), 2)); n2 += 1
            if n1 >= 60 and n2 >= 60:
                break
    assert len(ornek) >= 20, "graftan ornek cikmadi"

    kotu, bos = [], 0
    for z, adim in ornek:
        s = S.kur(z, adim)
        if s is None:
            bos += 1
            continue
        soru, kanit = s
        for ad, dizi in (("onek", soru.onek), ("cevap", soru.cevap),
                         ("kanit", kanit)):
            if not dizi:
                kotu.append((ad, z))
        # yuzey KORPUSUN yazdigi bicimde mi -- `?` sonda, ek AYRI
        if b.ad[soru.onek[-1]] != "?":
            kotu.append(("onek '?' ile bitmiyor", b.coz(soru.onek)))
    assert not kotu, "%d sinav yuzeyi bozuk: %s" % (len(kotu), kotu[:3])
    assert bos * 2 < len(ornek), (
        "%d/%d ornek KURULAMADI -- yuzey sozlukle tutmuyor"
        % (bos, len(ornek)))

    o1 = S.kur(ornek[0][0], 1)[0]
    return ("%d ornek, kurulamayan %d   ornek onek: %s"
            % (len(ornek), bos, b.coz(o1.onek)))


@kapi("35  SAAT -- t_j = vur_j ? 0 : t_{j-1}+1  (artirma CAPADAN ONCE)")
def _35():
    """§1'in "kodun hesapladigi sey, BIREBIR" blogunda `t_j` YOKTU
    (21 Eylul, adim adim denetim sirasinda bulundu). Yazildi; kapisi
    bu.

    Ozellikle sinanan sey SIRA: `t = t + 1` capa kontrolunden ONCE,
    `masked_fill(vur, 0)` SONRA. Ters olsaydi capa adiminda t = 1
    cikardi ve saat acikken okuma hedefi BIR ADIM KAYARDI (H[t]).
    Saat varsayilan KAPALI oldugu icin bu sessizce gecerdi."""
    n, D, d, K, B, L = 60, 16, 6, 128, 32, 12
    m = M.Yol(n, D=D, d=d, K=K, tam=torch.ones(n, dtype=torch.bool))
    g = torch.Generator().manual_seed(35)
    X = torch.randint(0, n, (B, L), generator=g)

    # --- uc rejim: hic capa / hep capa / karisik
    for ad, r in (("r=0 hic capa", 0.0), ("r=inf hep capa", 1e6),
                  ("karisik", 0.25)):
        y = m.yol(X, r)
        t, vur = y["t"], y["vur"]
        assert (t[:, 0] == 0).all(), "%s: t_0 sifir degil" % ad
        # GENEL DEGISMEZ: her adimda t_j == (vur_j ? 0 : t_{j-1}+1)
        bek = torch.where(vur[:, 1:], torch.zeros_like(t[:, 1:]),
                          t[:, :-1] + 1)
        kotu = int((t[:, 1:] != bek).sum())
        assert not kotu, (
            "%s: %d adimda t_j denklemi TUTMUYOR -- artirma ile "
            "sifirlamanin SIRASI ters olabilir" % (ad, kotu))
    # --- ucun da NE VERDIGI: kapi bos kalmasin
    t0 = m.yol(X, 0.0)["t"]
    t1 = m.yol(X, 1e6)["t"]
    assert (t0[:, -1] == L - 1).all(), "capa yokken t son adimda L-1 olmali"
    assert (t1[:, 1:] == 0).all(), (
        "capa HER adimda tetiklendiginde t hep 0 olmali -- 1 cikiyorsa "
        "artirma sifirlamadan SONRA yapiliyor demektir")
    return ("t_0 = 0;  hic capa -> t son adimda %d;  hep capa -> t hep 0;"
            "  karisik rejimde %d adimin hepsi denklemi tutuyor"
            % (int(t0[0, -1]), B * (L - 1)))


@kapi("36  HAFIZA -- V=0'da OZDES, ve §3.1'i BILEREK kiriyor")
def _36():
    """§12b olgu hafizasi.  Kapi 18 bunu GOREMEZ: kapilar `Yol`u
    `hafiza=True` GECMEDEN kuruyor, yani V is None ve hafiza yolu hic
    kosmuyor. `ayar_14`de HAFIZA = True yaziyor; sinayan yer BURASI.

    Dort sey kilitleniyor:
      1  V = 0 iken cikti hafizasiz modelle BIREBIR.  Guvenli
         baslangic: model tam eskisi gibi baslar, hafizayi kendi
         buyutur.  Bozulursa `V` sifirdan baslamiyordur.
      2  V != 0 iken cikti DEGISIYOR ve |z| = 1 kaliyor.
      3  §3.1 KOSULU: hafiza KAPALI iken ayni koda dusen iki durum
         OZDES (teorem); ACIK iken DEGIL -- cunku okuma capa ONCESI
         `zp`den adresleniyor.  BU BILEREK: olculdu ki capa
         tetikleyen orneklerde ozne kimligi 1,10 kat, tetiklemeyende
         11,34 kat sans ustu; tarih bagimsizligi ozneyi de unutturuyor.
      4  Sicak dongu bedeli: hafiza GRADYANLI (B,K) tensor EKLEMEZ
         (kapi 26'nin korudugu sey). Yumusak okuma butun K uzerinde
         olsaydi adim basina 67 MB x 15 geri gecise kadar tutulurdu."""
    import ayar_14 as AY
    n, D, d, K, B, L = 80, 16, 8, 64, 48, 10
    ort = dict(n=n, D=D, d=d, K=K, tam=torch.ones(n, dtype=torch.bool))
    g = torch.Generator().manual_seed(36)
    X = torch.randint(0, n, (B, L), generator=g)

    m0 = M.Yol(**ort)                                   # hafizasiz
    mh = M.Yol(**ort, hafiza=True, haf_n=AY.HAFIZA_N,
               haf_tau=AY.HAFIZA_TAU)
    assert mh.V is not None and float(mh.V.abs().max()) == 0.0,         "V SIFIRDAN baslamali -- guvenli baslangic bozulmus"

    # 1  V = 0 -> OZDES
    z0, zh = m0.yol(X, 0.25)["z"], mh.yol(X, 0.25)["z"]
    e = float((z0 - zh).abs().max())
    assert e < 1e-6, "V=0 iken cikti hafizasizdan AYRILIYOR: %.2e" % e

    # 2  V != 0 -> DEGISIYOR, norm KORUNUYOR
    with torch.no_grad():
        mh.V.normal_(0, 0.3, generator=g)
    zh2 = mh.yol(X, 0.25)["z"]
    f = float((z0 - zh2).abs().max())
    assert f > 1e-3, "V dolu ama cikti DEGISMIYOR -- hafiza okunmuyor"
    nz = float((zh2.norm(dim=-1) - 1).abs().max())
    assert nz < 1e-5, "|z| = 1 bozuldu: sapma %.2e" % nz

    # 3  §3.1 KOSULU -- AYNI KODA DUSEN cift ARANIR (kapi 18 gibi);
    #    r=inf capayi her adimda tetikler ama HANGI kod, duruma bagli.
    y0, yh = m0.yol(X, r=1e6), mh.yol(X, r=1e6)
    # !! KURULUM SARTI: ayni koda dusen ama CAPA ONCESI DURUMU FARKLI
    # bir cift. Ilk surum yalniz "ayni kod" ariyordu ve ayni ILK
    # BIRIMDEN baslayan bir cift buluyordu -- o cift bastan sona
    # OZDES, hafizanin ayiracagi bir sey YOK, sinama anlamsiz.
    cift = None
    for j in range(1, L):
        kk, zpj = y0["k"][:, j], y0["zp"][:, j]
        for c in torch.unique(kk):
            ix = torch.nonzero(kk == c).squeeze(1)
            for a_ in range(len(ix)):
                for b_ in range(a_ + 1, len(ix)):
                    if float((zpj[ix[a_]] - zpj[ix[b_]]).abs().max()) > 1e-3:
                        cift = (j, int(ix[a_]), int(ix[b_]))
                        break
                if cift:
                    break
            if cift:
                break
        if cift:
            break
    assert cift, ("kapi kurulumu: ayni koda dusen ama capa ONCESI durumu "
                  "FARKLI cift YOK -- sinama anlamsiz")
    j, i1, i2 = cift
    dzp = float((y0["zp"][i1, j] - y0["zp"][i2, j]).abs().max())
    ayni0 = float((y0["z"][i1, j] - y0["z"][i2, j]).abs().max())
    aynih = float((yh["z"][i1, j] - yh["z"][i2, j]).abs().max())
    assert ayni0 < 1e-6, (
        "HAFIZA KAPALI iken §3.1 TUTMALI (ayni kod -> ozdes durum), "
        "fark %.2e" % ayni0)
    assert aynih > 1e-3, (
        "HAFIZA ACIK iken §3.1 tutmamali -- tutuyorsa hafiza capa "
        "ONCESI durumdan adreslenmiyor, yani ozneyi gormuyor")

    # 3b HIZALAMA -- a[b,i] agirligi V[kn[b,i]] degeriyle mi eslesiyor.
    #    Bir boyut kaysa model yine "calisiyor" gorunurdu ve yukaridaki
    #    kontrollerin HICBIRI gormezdi.  Ic ice DONGUYLE yeniden kurup
    #    karsilastiriyoruz.
    with torch.no_grad():
        R2, C2, P2 = mh.donme(), mh.kod(), mh.p
        z0 = P2.new_zeros(B, D); z0[:, :d] = P2[X[:, 0]]
        zp = torch.bmm(R2[X[:, 0]], z0.unsqueeze(-1)).squeeze(-1)
        sa = zp @ C2.t()
        kn = sa.topk(mh.haf_n, 1).indices
        aw = torch.softmax((C2[kn] * zp[:, None]).sum(-1) / mh.haf_tau, 1)
        mem = (aw[..., None] * F.embedding(kn, mh.V)).sum(1)
        mem2 = torch.zeros(B, D)
        for bb in range(B):
            for ii in range(mh.haf_n):
                mem2[bb] += float(aw[bb, ii]) * mh.V[int(kn[bb, ii])]
    assert torch.allclose(aw.sum(1), torch.ones(B), atol=1e-5),         "attention agirliklari 1'e toplanmiyor"
    assert bool((aw.argmax(1) == sa.gather(1, kn).argmax(1)).all()),         "en buyuk agirlik EN BUYUK ic carpima denk gelmiyor -- softmax "        "yanlis eksende"
    hz = float((mem - mem2).abs().max())
    assert hz < 1e-5, ("HIZALAMA BOZUK: a[b,i] ile V[kn[b,i]] eslesmiyor, "
                       "fark %.2e" % hz)

    # 4  gradyanli (B,K) EKLENMEDI
    with _Buyuk(B * K) as sk:
        mh.kayip(X, isin=2)
    grad = [x for x in sk.v if x[1] == (B, K) and x[2]]
    assert not grad, "%d gradyanli (B,K) tensor -- hafiza sicak donguyu "        "bozuyor (kapi 26)" % len(grad)
    return ("V=0 ozdes (%.1e);  V dolu fark %.3f, |z|=1 sapma %.1e;  "
            "§3.1 (ayni kod, adim %d, zp farki %.3f) kapali %.1e "
            "ACIK %.3f;  hizalama %.1e;  gradyanli (B,K) 0"
            % (e, f, nz, j, dzp, ayni0, aynih, hz))


@kapi("37  EGITIM ve URETIM AYNI ADIMI KOSAR")
def _37():
    """OLCULDU 21 Eylul: `uret_toplu` adimi KENDI yaziyordu ve
    `mdl.V`ye hic dokunmuyordu. §12b kosusu boylece hafizayla
    egitilip HAFIZASIZ olculdu; BICIM sayilari (kalip 0,0000,
    kapanmadi 1,0000) o yuzden gecersiz.

    Hicbir kapi bunu goremiyordu cunku hepsi ya yalniz `yol`u ya
    yalniz uretimi cagiriyordu. Burasi IKISINI BIRBIRINE bagliyor:
    ayni onek, ayni agirlik -> AYNI jetonlar.  Hafiza ACIK, yoksa
    fark zaten olmazdi."""
    import olcme_14 as OL
    n, D, d, K, B, L = 60, 16, 8, 48, 6, 9
    g = torch.Generator().manual_seed(37)
    m = M.Yol(n=n, D=D, d=d, K=K, tam=torch.ones(n, dtype=torch.bool),
              hafiza=True, haf_n=4, haf_tau=0.02)
    with torch.no_grad():                  # V = 0 olsa fark GORUNMEZDI
        m.V.normal_(0, 0.3, generator=g)
    X = torch.randint(0, n, (B, L), generator=g)
    r = 0.25

    # !! KONUM HIZASI: `yol`un L-1 indeksli durumu 0..L-2 jetonlarini
    # yemistir ve L-1'i OKUR.  Ayni yere uretimle varmak icin onek
    # X[:, :L-1] verilir; ilk URETILEN jeton tam o okumadir.
    y = m.yol(X, r)
    q = F.normalize(y["z"][:, :, :d], dim=-1)
    bek = (q @ m.p.T).argmax(-1)[:, -1]
    onek = [tuple(int(t) for t in x) for x in X[:, :L - 1]]

    cik = OL.uret_toplu(m, onek, n_yeni=1, r=r, bs=B, dev="cpu")
    ger = torch.tensor([c[0] for c in cik])
    assert torch.equal(bek, ger), (
        "egitim yolu ile uretim AYRISTI: %s vs %s" % (bek.tolist(),
                                                      ger.tolist()))

    # ve V'yi degistirince uretim de DEGISMELI -- yoksa test bos gecer
    with torch.no_grad():
        m.V.mul_(0)
    cik0 = OL.uret_toplu(m, onek, n_yeni=1, r=r, bs=B, dev="cpu")
    ger0 = torch.tensor([c[0] for c in cik0])
    assert not torch.equal(ger, ger0), (
        "V sifirlaninca uretim DEGISMEDI -- uretim hafizayi okumuyor")
    return ("ayni onekte yol() ve uret_toplu() AYNI jetonu verdi (%d/%d); "
            "V sifirlaninca %d jeton degisti"
            % (int((bek == ger).sum()), B, int((ger != ger0).sum())))


@kapi("38  CAPA KAPALI (r=0) -- hic tetiklenmez, VQ terimleri SIFIR")
def _38():
    """§12c capayi `R_CAPA = 0` ile kaldiriyor.  Iki sey kilitleniyor:

      1  r = 0'da `vur` HICBIR adimda dogru olmamali.  `esik` 1,0
         degil 2,0 donduruyor: fp32'de iki birim vektorun ic carpimi
         1'i birkac ulp asabiliyor ve 1,0 esigi SESSIZCE tetiklerdi.
      2  a2 = 0'da `kod` ve `bag` TAM SIFIR, ve C'ye VQ gradyani
         GITMEMELI -- C artik hafizanin ANAHTARI, niceleyici degil
         (§3.1b: `kod` terimi C'ye olguyu degil ILISKIYI kodlatiyor).
    """
    n, D, d, K, B, L = 60, 16, 8, 48, 32, 10
    g = torch.Generator().manual_seed(38)
    m = M.Yol(n=n, D=D, d=d, K=K, tam=torch.ones(n, dtype=torch.bool),
              hafiza=True, haf_n=4)
    X = torch.randint(0, n, (B, L), generator=g)

    assert M.Yol.esik(0.0) == 2.0, "r=0 esigi 2,0 olmali (fp32 tasmasi)"
    y0 = m.yol(X, 0.0)
    assert not bool(y0["vur"].any()), (
        "r = 0 iken capa %d adimda tetikledi" % int(y0["vur"].sum()))
    # !! r BUYUK secildi: egitilmemis modelde durumlar kodlara uzak,
    # r = 0,25 (esik 0,969) hic tetiklemez ve kapi BOS gecerdi.
    y1 = m.yol(X, 1.5)
    assert bool(y1["vur"].any()), "r = 1,5'te bile hic tetiklemedi -- kapi bos"

    # a2 = 0  ->  kod/bag tam sifir ve C'ye VQ gradyani yok
    m.zero_grad(set_to_none=True)
    top, _ = m.kayip(X, a2=0.0, a3=0.0, a4=0.0, r=0.0, isin=2)
    top.backward()
    gC0 = m.C.grad.abs().sum().item()
    assert float(m.son["kod"]) == 0.0 and float(m.son["bag"]) == 0.0, (
        "a2 = 0 ama kod %.3e / bag %.3e" % (float(m.son["kod"]),
                                            float(m.son["bag"])))
    m.zero_grad(set_to_none=True)
    m.kayip(X, a2=1.0, a3=0.0, a4=0.0, r=1.5, isin=2)[0].backward()
    gC1 = m.C.grad.abs().sum().item()
    assert gC1 > gC0, "a2 acikken C'ye daha cok gradyan gitmeli"
    return ("r=0'da capa 0/%d adim (r=1,5'te %d);  a2=0'da kod=bag=0 ve "
            "C gradyani %.3e -> a2=1'de %.3e"
            % (B * (L - 1), int(y1["vur"].sum()), gC0, gC1))


@kapi("39  HAFIZA BUTCESI -- mentese, ve DEGER duzeyinde dogru")
def _39():
    """§12c: duz L1 bedelinin araligi BOS cikti (ikameyi engellemek
    a4 > 0,639, kullanimi birakmak a4 < 0,410).  Mentese ikisini
    ayiriyor: butcenin ALTINDA maliyet TAM SIFIR, ustunde kareyle.

    Deger duzeyinde sinaniyor -- "terim var" yetmez, SAYISI tutmali:
        haf = (ort|m| - B)+^2      ve   top = ... + a4 * haf
    Ayrica `mn` gercekten okumanin normu mu: V'yi olcekleyince
    ayni oranda buyumeli.

    !! ORTALAMA BUTUN KONUMLARDAN, `isin` diliminden DEGIL.
    Onceki hal `[:, isin:]` idi ve pencerenin %17'sine yazmak
    BEDAVAYDI; izde `|m|` j=1'de 4,79 cikiyordu (§5.1/U).  Bu kapi
    o deligi kapali tutar: dilim geri gelirse sayi tutmaz."""
    n, D, d, K, B, L = 60, 16, 8, 48, 24, 8
    g = torch.Generator().manual_seed(39)
    m = M.Yol(n=n, D=D, d=d, K=K, tam=torch.ones(n, dtype=torch.bool),
              hafiza=True, haf_n=4)
    X = torch.randint(0, n, (B, L), generator=g)
    ort = dict(a1=0.0, a2=0.0, a3=0.0, r=0.0, isin=2)

    with torch.no_grad():
        m.V.normal_(0, 0.5, generator=g)
    mn = float(m.yol(X, 0.0)["mn"][:, 1:].mean())
    assert mn > 0, "V dolu ama |m| = 0"
    mn_dilim = float(m.yol(X, 0.0)["mn"][:, 2:].mean())
    assert abs(mn - mn_dilim) > 1e-6, (
        "dilimli ve dilimsiz ortalama AYNI -- kapi bos gecer")

    # 1) butcenin USTUNDE: top - uye  ==  a4 * (mn - B)^2
    Bd, a4 = mn / 2, 3.0
    top, uye = m.kayip(X, a4=a4, haf_b=Bd, **ort)
    bek = a4 * (mn - Bd) ** 2
    assert abs(float(top) - float(uye) - bek) < 1e-5, (
        "ustte: top-uye %.6f, beklenen %.6f" % (float(top) - float(uye), bek))
    # 2) butcenin ALTINDA: TAM sifir
    top2, uye2 = m.kayip(X, a4=a4, haf_b=mn * 2, **ort)
    assert abs(float(top2) - float(uye2)) < 1e-6, (
        "altta maliyet 0 olmali, %.3e" % (float(top2) - float(uye2)))
    # 3) a4 = 0 -> terim HIC hesaplanmaz, top = uye
    top3, uye3 = m.kayip(X, a4=0.0, haf_b=Bd, **ort)
    assert abs(float(top3) - float(uye3)) < 1e-6
    # 4) `mn` OKUMANIN normu: V iki katina -> mn iki katina
    with torch.no_grad():
        m.V.mul_(2)
    mn2 = float(m.yol(X, 0.0)["mn"][:, 1:].mean())
    assert abs(mn2 / mn - 2) < 0.02, "|m| V ile dogrusal degil: %.4f" % (mn2 / mn)
    # 5) gradyan V'ye AKIYOR
    m.zero_grad(set_to_none=True)
    m.kayip(X, a4=a4, haf_b=mn, **ort)[0].backward()
    assert m.V.grad is not None and float(m.V.grad.abs().sum()) > 0, (
        "butce terimi V'ye gradyan vermiyor")
    return ("ort|m| %.4f (dilimli olsa %.4f -- delik kapali);  "
            "ust: top-uye %.6f = a4(mn-B)^2 %.6f;  alt: %.1e;  a4=0: %.1e;  "
            "V x2 -> |m| x%.3f;  V gradyani AKIYOR"
            % (mn, mn_dilim, float(top) - float(uye), bek,
               abs(float(top2) - float(uye2)), abs(float(top3) - float(uye3)),
               mn2 / mn))


@kapi("40  BILGI OLCUSU ATESLENEBILIYOR MU")
def _40():
    """OLCULDU 21 Eylul: `bilgi_puanla` BIREBIR esitlik ariyordu.
    Beklenen cevap CIPLAK ad (`sinav_yuzeyi` `Y` donduruyor) ama
    modelin urettigi dilbilgisel Turkce, yani bildirme ekini tasiyor:
        beklenen [Agri]        span [Agri -dir]
    `tam` DORT kosuda da 0,0000 verdi ve bu SIFIR diye okundu; ek
    toleransiyla ucuncu kosu 0,0008, dorduncu 0,0138 (21,8 kat sans
    ustu).  Yani GERCEK bir sinyal sifir diye raporlandi.

    Hicbir kapi goremiyordu: hepsi `bilgi_puanla`yi ya hic cagirmiyor
    ya da UYDURMA cikti veriyordu.  Burasi DOGRU cevabi ekiyle
    birlikte veriyor -- olcu bunu DOGRU saymali."""
    import olcme_14 as OL

    class S:                                  # minik sahte soru
        def __init__(self, c, k=()):
            self.cevap, self.kisayol = c, k
    BITIS, EK = {9}, {7, 8}                   # 7,8 = ek;  9 = nokta
    #            cevap        model ne uretti
    hal = [(S((1, 2)),        [1, 2, 7, 9]),      # dogru + bildirme eki
           (S((1, 2)),        [1, 2, 9]),         # dogru, eksiz
           (S((1, 2)),        [3, 4, 7, 9]),      # bambaska
           (S((1, 2)),        [5, 2, 7, 9]),      # AILE: son parca dogru
           (S((1, 2), (3,)),  [3, 8, 9]),         # KISAYOL, ekli
           (S((1, 2)),        [9])]              # BOS
    r = OL.bilgi_puanla([h[0] for h in hal], [h[1] for h in hal],
                        BITIS, EK)
    n = len(hal)
    assert abs(r["tam"] - 2 / n) < 1e-9, (
        "ekli ve eksiz DOGRU cevabin ikisi de sayilmali, tam=%.4f" % r["tam"])
    assert abs(r["aile"] - 1 / n) < 1e-9, "aile %.4f" % r["aile"]
    assert abs(r["kisayol"] - 1 / n) < 1e-9, "kisayol %.4f" % r["kisayol"]
    assert abs(r["bos"] - 1 / n) < 1e-9, "bos %.4f" % r["bos"]

    # ek_ix VERILMEZSE eski (bozuk) davranis: ekli olan sayilmaz.
    r0 = OL.bilgi_puanla([h[0] for h in hal], [h[1] for h in hal], BITIS)
    assert r0["tam"] < r["tam"], (
        "ek_ix'siz cagri ekliyi saymamali -- kapi BOS")
    return ("6 halde  tam %.3f  aile %.3f  kisayol %.3f  bos %.3f;  "
            "ek_ix'siz eski hal tam %.3f (ekli cevabi KACIRIYOR)"
            % (r["tam"], r["aile"], r["kisayol"], r["bos"], r0["tam"]))


@kapi("41  YUK DENGELEME -- deger duzeyinde, ve ONKOSUL izleniyor")
def _41():
    """§12c/S1.  OLCULDU (§5.1/U): 8.192 yuvanin 10'u atesliyor.
    Terim:  denge = K * sum_i f_i * P_i
        f_i  yuva i'yi TOP-1 secen konum payi  (gradyansiz sayim)
        P_i  yuva i'ye giden ortalama olasilik (gradyan BURADAN)
    Tekduze kullanimda 1, tek yuvada K.

    Uc sey kilitleniyor:
      1  SINIRLAR: elle kurulmus tekduze dagilim 1, tek yuvaya
         cokmus dagilim K verir.
      2  MODELDEKI deger, BAGIMSIZ yeniden hesapla BIREBIR tutar --
         "terim var" yetmez, SAYISI tutmali (kapi 27'nin yontemi).
      3  ONKOSUL izleniyor: `son["yuva"]` gercekten kac AYRI yuvanin
         atestigi.  §12c'nin onceden kaydi bu sayiyi KOSU SIRASINDA
         okuyor; hesaplanmiyorsa kayit uygulanamaz."""
    K = 64
    # 1) SINIRLAR -- saf formul, modelden bagimsiz
    f = torch.full((K,), 1.0 / K)
    assert abs(float(K * (f * f).sum()) - 1.0) < 1e-6, "tekduze 1 vermeli"
    g = torch.zeros(K); g[0] = 1.0
    assert abs(float(K * (g * g).sum()) - K) < 1e-6, "tek yuva K vermeli"

    # 2) MODELDEKI deger
    n, D, d, B, L = 60, 16, 8, 24, 9
    gg = torch.Generator().manual_seed(41)
    m = M.Yol(n=n, D=D, d=d, K=K, tam=torch.ones(n, dtype=torch.bool),
              hafiza=True, haf_n=4, haf_tau=0.02)
    with torch.no_grad():
        m.V.normal_(0, 0.3, generator=gg)
    X = torch.randint(0, n, (B, L), generator=gg)
    ort = dict(a1=0.0, a2=0.0, a3=0.0, a4=0.0, r=0.0, isin=2)

    top0, uye0 = m.kayip(X, a5=0.0, **ort)
    assert abs(float(top0) - float(uye0)) < 1e-6, "a5=0'da terim OLMAMALI"
    assert float(m.son["denge"]) == 0.0

    A5 = 0.25
    top, uye = m.kayip(X, a5=A5, **ort)
    y = m.yol(X, 0.0)
    kn, ha, k1 = (y["kn"][:, 1:].reshape(-1), y["ha"][:, 1:].reshape(-1),
                  y["k"][:, 1:].reshape(-1))
    N = k1.numel()
    P = torch.zeros(K).index_add(0, kn, ha) / N
    fq = torch.zeros(K).index_add(0, k1, torch.ones(N)) / N
    bek = float(K * (fq * P).sum())
    assert abs(float(m.son["denge"]) - bek) < 1e-4, (
        "denge %.6f, bagimsiz hesap %.6f" % (float(m.son["denge"]), bek))
    assert abs(float(top) - float(uye) - A5 * bek) < 1e-4, (
        "top-uye %.6f, a5*denge %.6f" % (float(top) - float(uye), A5 * bek))
    assert 1.0 <= bek <= K, "denge [1,K] disinda: %.4f" % bek

    # 3) ONKOSUL sayaci + gradyan ANAHTARLARA akiyor mu
    ay = int(torch.unique(k1).numel())
    assert int(m.son["yuva"]) == ay, (
        "son['yuva'] %d, gercek %d" % (int(m.son["yuva"]), ay))
    m.zero_grad(set_to_none=True)
    m.kayip(X, a5=A5, **ort)[0].backward()
    assert m.C.grad is not None and float(m.C.grad.abs().sum()) > 0, (
        "denge terimi ANAHTARLARA (C) gradyan vermiyor -- olu yuva "
        "olu kalir, terimin tek isi buydu")
    return ("sinirlar 1 / %d;  modeldeki %.4f = bagimsiz hesap %.4f;  "
            "top-uye = a5*denge (fark %.1e);  yuva %d/%d;  C gradyani AKIYOR"
            % (K, float(m.son["denge"]), bek,
               abs(float(top) - float(uye) - A5 * bek), ay, K))


# =====================================================================
# VERI YOLU  --  kopyanin ve kurulumun kapilari
# =====================================================================


if __name__ == "__main__":
    for ad, not_ in GECTI:
        print(f"  GECTI   {ad}\n          {not_}" if not_ else f"  GECTI   {ad}")
    for ad, e in KALDI:
        print(f"  KALDI   {ad}\n          {e}")
    print(f"\n{len(GECTI)} gecti, {len(KALDI)} kaldi")
    sys.exit(1 if KALDI else 0)
