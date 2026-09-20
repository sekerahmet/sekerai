# -*- coding: utf-8 -*-
"""test_14 -- DENKLEM.md'nin her iddiasi icin bir kapi.

Amac: matematik mi yanlis, kod mu yanlis -- ayirmak. Her kapinin
basliginda BELGENIN hangi cumlesini sinadigi yaziyor.

    python test_14.py
"""
from __future__ import annotations

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
    # Ote yandan veri URETEN dosyalar birebir kalmali -- sinav sabit.
    for f in ("veri", "korpus", "metin", "jeton"):
        a = open(f"../model_13/{f}_13.py", encoding="utf-8").read()
        b = open(f"{f}_14.py", encoding="utf-8").read()
        if re.sub(r"_13\b", "_14", a).replace("model_13", "model_14") != b:
            A = re.sub(r"_13\b", "_14", a).replace("model_13", "model_14")
            fark[f] = sum(1 for x, y in zip(A.split("\n"), b.split("\n"))
                          if x != y)
    assert not fark, f"kopya ADLANDIRMADAN BASKA yerde degismis: {fark}"
    return ("veri korpus metin jeton -- saf yeniden adlandirma"
            "   (ek_14 haric: kolun KENDI bolucusu)")


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
