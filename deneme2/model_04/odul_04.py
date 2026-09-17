# -*- coding: utf-8 -*-
"""odul_04 — model_04'un ODUL CEKIRDEGI. TEK BASINA DURUR.

Kullanici karari, 17 Eylul 2026: *"ödül mekanizmasını model_03 üzerine
kuracağız"* ve *"basit bir matematik hesabıyla değil; şekil, yuva,
beklenen cevaba yaklaşma gibi biraz kompleks kriterlerle verilmeli."*

BU DOSYA `taban_04.py`YI ICE AKTARMAZ. Yalniz numpy/torch. Boylece
`test_04.py` odulu motordan BAGIMSIZ sinayabilir: odul yanlissa kosu
baslamamali, ve "odul mu bozuk motor mu" sorusu ayrilabilir olmali.

MERDIVEN (onkayit model_04.md 3)
--------------------------------
Bir cevap uc yuvadir: [y1][y2][y3]. Once UC KAPI, sonra basamaklar.

    KAPI 1   <YOK> SAGDA mi        -> cevaba BAKMAZ
    KAPI 2   JETON SAYISI dogru mu -> cevabin jeton sayisini kullanir
    KAPI 3   GERCEK bir varlik mi  -> cevaba BAKMAZ
    Herhangi biri duserse  ODUL = ZEMIN, alt katmanlara BAKILMAZ.

    Sonra varligin YERI:
      ozneden 1 ADIMDA ulasilir   -> KISAYOL cezasi
      ne 1 ne 2 adimda            -> E
      2 ADIMDA ulasilir           -> F
        + y2 dogru                -> G
        + y1 de dogru (TAM)       -> H

POZITIF BASAMAKLAR UYDURULMADI -- aramanin ne kadar daraldigindan
turetildi (olculdu, `belge/OLCULENLER.md` 1e):

    baslangic                    1060 varlik    0.00 bit   -
    E  gecerli + dogru jeton      574           1.23 bit   0.12
    F  2 adim menzilinde          104           3.38 bit   0.34
    G  + dogru aile / tur          11           6.78 bit   0.67
    H  TAM                          1          10.05 bit   1.00

Yani odul = "cevabin bilgisinin ne kadarini daraltti". Tek keyfi karar
H = 1.00 olmasi; o da normalizasyon.

ZEMIN ve KISAYOL cezasinin TURETMESI YOK -- arizanin tanimindan
geliyorlar, olcumden degil. Ikisi de ACIK DUGME (onkayit 6).
  zemin < kisayol : bozuk cevap, kopruyu atlamaktan KOTU sayilir.
  Aksi halde model belirsizlikte SACMALAMAYI ogrenir (0 zemin, -0.30
  kisayol olsaydi bozuk cevap daha karli olurdu).
"""
from __future__ import annotations

import dataclasses as dc

import numpy as np
import torch


@dc.dataclass(frozen=True)
class OdulAyar:
    """Merdivenin basamak degerleri. `ayar_04.py` bunlari YAZIYOR."""
    zemin: float = -0.50      # KAPI 1/2/3'ten biri dustu
    kisayol: float = -0.30    # ozneden 1 ADIMDA ulasilan cevap
    e_menzil_disi: float = 0.12
    f_menzil: float = 0.34
    g_aile: float = 0.67
    h_tam: float = 1.00

    def __post_init__(self):
        # SIRA BOZULAMAZ. Bir gun biri basamaklari elle degistirirse
        # merdiven sessizce ters donebilir; o an kosu BASLAMAMALI.
        s = [self.zemin, self.kisayol, self.e_menzil_disi,
             self.f_menzil, self.g_aile, self.h_tam]
        assert all(a < b for a, b in zip(s, s[1:])), (
            f"MERDIVEN SIRASI BOZUK: {s} -- zemin <= kisayol < E < F < G < H "
            "olmali (onkayit model_04.md 3)")
        assert self.h_tam == 1.0, "H olcegi sabitler; digerleri ona goredir"


class Menzil:
    """Grafin ONCEDEN hesaplanmis 1-adim / 2-adim komsulugu.

    `facts` (n_ent, n_rel), -1 = olgu YOK. Kosu basinda BIR KEZ kurulur;
    adim basina hesaplanmaz. 1060 varlikta iki (1060,1060) bool dizi =
    ~2 MB, onemsiz.
    """

    def __init__(self, facts: np.ndarray):
        n = facts.shape[0]
        self.n_ent = n
        h1 = np.zeros((n, n), bool)
        for e in range(n):
            h = facts[e]
            h1[e, h[h >= 0]] = True
        # 2 ADIM: h1 @ h1 (bool carpim). n=1060'ta anlik.
        h2 = (h1.astype(np.uint8) @ h1.astype(np.uint8)) > 0
        self.h1 = h1
        self.h2 = h2 & ~h1          # "2 ADIMDA VAR, 1 ADIMDA YOK"
        # NOT: kosegen ELENMIYOR. `veri_kur` "kendine giden olgu" yok diye
        # zaten assert ediyor, ama 2 adimda e->b->e mumkun ve o da gecerli
        # bir "2 adim menzili" uyesidir. Bolme tanimi (comp/ent) cevabin
        # ozneye esit olmadigini garanti ediyor (olculdu: 0/3000).


class Puanlayici:
    """Bir cevap yigini -> odul vektoru. MODELDEN BAGIMSIZ."""

    def __init__(self, par: np.ndarray, yok_ix: int, facts: np.ndarray,
                 oa: OdulAyar):
        """par: (n_ent, 3) her varligin YUVA INDEKSLERI (paylasilan sozluk).
        yok_ix: <YOK> jetonunun yuva-ici indeksi."""
        assert par.ndim == 2 and par.shape[1] == 3, f"par {par.shape}"
        self.par = par.astype(np.int64)
        self.yok = int(yok_ix)
        self.oa = oa
        self.menzil = Menzil(facts)
        self.n_jeton = (self.par != self.yok).sum(1)      # (n_ent,)
        # UCLU -> VARLIK aramasi. Sozluk 272 jeton -> 272^3 = 20M, yogun
        # dizi ISTEMEZ. Tek tamsayiya kodlayip searchsorted.
        self.taban = int(par.max()) + 1
        anahtar = self._kodla(self.par)
        d = np.argsort(anahtar)
        self.anahtar = anahtar[d]
        self.anahtar_ent = d.astype(np.int64)
        assert len(np.unique(self.anahtar)) == len(self.anahtar), \
            "IKI VARLIK AYNI UCLUYE sahip -- arama tablosu kurulamaz"

    def _kodla(self, u: np.ndarray) -> np.ndarray:
        b = self.taban
        return (u[..., 0] * b + u[..., 1]) * b + u[..., 2]

    def varlik_ara(self, u: np.ndarray) -> np.ndarray:
        """(..., 3) yuva indeksleri -> varlik id, YOKSA -1."""
        k = self._kodla(u)
        i = np.searchsorted(self.anahtar, k)
        i = np.clip(i, 0, len(self.anahtar) - 1)
        bulundu = self.anahtar[i] == k
        return np.where(bulundu, self.anahtar_ent[i], -1)

    def sag_hizali(self, u: np.ndarray) -> np.ndarray:
        """<YOK>'ler SAGDA mi. [X . .] [X Y .] [X Y Z] gecerli;
        [X . Z] ve [. Y Z] BOZUK."""
        dolu = u != self.yok                       # (..., 3)
        # sagdan sola: bir kez <YOK> gorunce sonrasi da <YOK> olmali
        return ~(dolu[..., 1] & ~dolu[..., 0]) & ~(dolu[..., 2] & ~dolu[..., 1])

    def puanla(self, ornek: np.ndarray, ozne: np.ndarray, kisayol: np.ndarray,
               dogru: np.ndarray) -> tuple:
        """ornek (B,G,3) yuva indeksleri; ozne/kisayol/dogru (B,) varlik id.
        kisayol < 0 ise "kisayol TANIMSIZ" demektir (ceza uygulanmaz).
        DONER: (odul (B,G) float32, ayrinti dict)"""
        B, G, _ = ornek.shape
        oa = self.oa
        u = ornek.reshape(-1, 3)
        n = B * G
        oz = np.repeat(ozne, G)
        ks = np.repeat(kisayol, G)
        dg = np.repeat(dogru, G)

        # --- KAPI 1: <YOK> SAGDA mi (cevaba BAKMAZ) --------------------
        k1 = self.sag_hizali(u)
        # --- KAPI 2: JETON SAYISI dogru mu -----------------------------
        k2 = (u != self.yok).sum(1) == self.n_jeton[dg]
        # --- KAPI 3: GERCEK bir varlik mi (cevaba BAKMAZ) --------------
        ent = self.varlik_ara(u)
        k3 = ent >= 0
        gecti = k1 & k2 & k3

        odul = np.full(n, oa.zemin, np.float32)
        if gecti.any():
            g = np.flatnonzero(gecti)
            e_, o_, k_, d_ = ent[g], oz[g], ks[g], dg[g]
            p = np.full(len(g), oa.e_menzil_disi, np.float32)
            # 2 ADIM menzili
            m2 = self.menzil.h2[o_, e_]
            p[m2] = oa.f_menzil
            # + dogru AILE / TUR (yuva2)
            ai = m2 & (self.par[e_, 1] == self.par[d_, 1])
            p[ai] = oa.g_aile
            # + TAM
            p[e_ == d_] = oa.h_tam
            # KISAYOL en son: 1 ADIM menzili ustteki her seyi EZER.
            # (Olculdu: comp/ent'te kisayol HICBIR ZAMAN dogru cevap
            #  degil -- 0/2558 ve 0/3000 -- yani H'yi ezme riski YOK.)
            ksy = (self.menzil.h1[o_, e_]) | ((k_ >= 0) & (e_ == k_))
            p[ksy] = oa.kisayol
            odul[g] = p

        # --- IZLEME SAYILARI. Hukum DEGIL, kosu sirasinda ne oldugunu
        # gormek icin. `ent_kisayol` ayri bir olcu ve `taban_04` onu
        # kendi hattinda hesaplamaya devam ediyor.
        gu = np.flatnonzero(gecti)
        e_g, o_g, k_g = ent[gu], oz[gu], ks[gu]
        ayr = dict(
            kapi1=float(k1.mean()), kapi2=float(k2.mean()),
            kapi3=float(k3.mean()), kapi=float(gecti.mean()),
            tam=float((gecti & (ent == dg)).mean()),
            ornek_kisayol=float(
                (self.menzil.h1[o_g, e_g] | ((k_g >= 0) & (e_g == k_g))).sum()
            ) / n if len(gu) else 0.0,
            ornek_menzil=float(self.menzil.h2[o_g, e_g].sum()) / n
            if len(gu) else 0.0,
        )
        return odul.reshape(B, G).astype(np.float32), ayr


def adim(net, xb: torch.Tensor, pb: torch.Tensor, lo: int, hi: int,
         puanlayici: "Puanlayici", ozne: np.ndarray, kisayol: np.ndarray,
         dogru: np.ndarray, g: int, sicaklik: float) -> tuple:
    """BIR ODUL ADIMI -- YUVALARI ZINCIRLEME ceker.

    xb (B, T) kodlanmis sorular, pb (B, 3) cevap yuvalarinin ONCEKI
    pozisyonlari (pozisyon p'nin logiti p+1'deki jetonu tahmin eder).
    DONER: (kayip, ortalama_odul, ayrinti)

    !! NEDEN ZINCIRLEME -- OLCULDU, 17 Eylul.
    Ilk surum uc yuvayi BAGIMSIZ cekiyordu: tek ileri gecis, `lg[ar, pb]`
    ile uc logit, uc ayri multinomial. Ucuzdu ve YANLISTI. Dizi
        [S2] ... ? a1 a2 a3 <EOS>
    ve `a2`nin logiti `a1`e BAKAR. Ogretmen zorlamali girdide oradaki
    jeton GERCEK `a1` -- yani bagimsiz cekimde `a2` ve `a3`, modelin
    kendi sectigi `a1`e degil DOGRU CEVABIN `a1`ine kosullu kaliyordu.
    Cekilen ucluler modelin POLITIKASINDAN GELMIYORDU; REINFORCE
    tahmincisi de bu yuzden YANLIYDI.

    Bedeli olculdu (model_03, ent, 8 cekim):
        BAGIMSIZ   gecerli varlik orani 0.9289
        ZINCIRLEME gecerli varlik orani 0.9914
    Yani KAPI 3'te takilanlarin cogu modelin kusuru degil, ORNEKLEMENIN
    kusuruymus -- ve OKUL tipinde ceza yikiciydi (kapilarda %47.6,
    ortalama odul -0.166). Uc yuvali okullarda etki en buyuk cunku uc
    icerik yuvasi da tutarli olmak zorunda.

    Bedel: yuva basina bir ileri gecis. Ilki B dizi uzerinde (butun G
    kopya o noktada AYNI), sonraki ikisi B*G uzerinde. `odul_batch`
    bu yuzden `batch`ten kucuk.

    KIRPMA YOK: gecersiz uclu uretilirse KAPI 3 onu zemine dusurur ve
    model duzgun varlik uretmeyi ODULDEN ogrenir. Gecerli kumeye
    kisitli ornekleme o dersi modelin elinden alirdi.
    """
    B = xb.shape[0]
    dev = xb.device
    ar_b = torch.arange(B, device=dev)
    # --- YUVA 1: tek ileri gecis, G kopya icin ORTAK -------------------
    lg1 = net(xb).float()[ar_b, pb[:, 0], lo:hi]                 # (B, n)
    lp1 = torch.log_softmax(lg1 / sicaklik, dim=-1)
    with torch.no_grad():
        s1 = torch.multinomial(lp1.exp(), g, replacement=True)   # (B, g)
    logp = lp1.gather(1, s1)                                     # (B, g)
    # --- G kopyaya ac, secilen jetonu DIZIYE yaz ----------------------
    pg = pb.repeat_interleave(g, 0)                              # (B*g, 3)
    ar_g = torch.arange(B * g, device=dev)
    cek = [s1.reshape(-1)]
    xg = xb.repeat_interleave(g, 0).clone()                      # (B*g, T)
    xg[ar_g, pg[:, 0] + 1] = s1.reshape(-1) + lo
    for j in (1, 2):
        lgj = net(xg).float()[ar_g, pg[:, j], lo:hi]             # (B*g, n)
        lpj = torch.log_softmax(lgj / sicaklik, dim=-1)
        with torch.no_grad():
            sj = torch.multinomial(lpj.exp(), 1).squeeze(1)      # (B*g,)
        logp = logp + lpj.gather(1, sj[:, None]).reshape(B, g)
        # !! KLON SART. `xg` bir onceki ileri gecisin GIRDISI ve autograd
        # onu tutuyor; YERINDE degistirmek o gecisin gradyanini gecersiz
        # kilar ("modified by an inplace operation"). Her yuva icin YENI
        # bir tensor yazilir.
        xg = xg.clone()
        xg[ar_g, pg[:, j] + 1] = sj + lo
        cek.append(sj)
    ornek = torch.stack(cek, 1).reshape(B, g, 3)                 # (B,G,3)

    od_np, ayr = puanlayici.puanla(ornek.cpu().numpy(), ozne, kisayol, dogru)
    od = torch.from_numpy(od_np).to(dev)
    a = avantaj(od)
    # REINFORCE, grup-goreli taban ile. Tek adimlik ve ON-POLICY oldugu
    # icin PPO orani/kirpmasi YOK: ornekler tam da guncellenen
    # politikadan geliyor, yani oran birebir 1.
    kayip = -(a.detach() * logp).mean()
    ayr["odul_ort"] = float(od.mean())
    ayr["avantajli_soru"] = float((od.std(dim=1) > 1e-6).float().mean())
    return kayip, float(od.mean()), ayr


def avantaj(odul: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """GRUP ICI avantaj: (r - ort) / std.  (B,G) -> (B,G)

    Sabit terim TAM OLARAK iptal olur: gruptaki herkese ayni puan eklemek
    ortalamayi da ayni kadar kaydirir. Bu yuzden modelin ZATEN %100
    yaptigi bir kriter (jeton sayisi, yuva3) odule ne agirlikla girerse
    girsin GRADYANI DEGISTIRMEZ -- kapi olarak durmasinin sebebi
    gradyan degil, GERILEME KORUMASI (onkayit model_04.md 3).

    Varyanssiz grup (herkes ayni puan) -> avantaj SIFIR, o soru
    gradyan URETMEZ. Olculdu: ent'te gruplarin ~%71'i boyle (G=8).
    """
    ort = odul.mean(dim=1, keepdim=True)
    sd = odul.std(dim=1, keepdim=True)
    a = (odul - ort) / (sd + eps)
    return torch.where(sd > eps, a, torch.zeros_like(a))
