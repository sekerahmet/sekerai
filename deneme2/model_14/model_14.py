# -*- coding: utf-8 -*-
"""model_14 -- YOL MODELI.  Tam tarif: `deneme2/DENKLEM.md`.

GENEL bir dil mimarisi. Grafimizdan, varlik listemizden, iliski
semamizdan HICBIR SEY almaz. Korpusumuz mimarinin kaynagi degil,
SINAVI.

    z_j = capa( R[w_{j-1}] @ z_{j-1} )          durum, kurede
    w_j = en yakin birim( PI z_j / |PI z_j| )   OKUMA -- ACISAL

Hesap yigini yok: ne dikkat, ne katman, ne cikti matrisi. Ileri
geciste yalniz donme carpimi ve en yakin komsu.

=====================================================================
UC IDDIA
=====================================================================
1 IZOMETRIK GECIS  Donme normu ve mesafeyi korur -> temsil cokemez,
                   sinyal sonmez, gradyan sonmez.
                   OLCULDU (model_13): oteleme tabanlisinda ozne
                   duyarliligi 0,22'ye soniyordu. ||Ra-Rb||=||a-b||
                   oldugu icin burada sonme IMKANSIZ.

2 NICELEME         Durum periyodik olarak bir KODA oturur -> durumlar
                   YENIDEN KULLANILABILIR. Gorulmemis bilesim,
                   gorulmus parcalara iner.  ASIL IDDIA BU.
                   Olcusu `comp`;  teorem: comp ~ a^2  (DENKLEM §3.1)

3 MESAFEYLE OKUMA  Cikti katmani yok. Sozluk buyudukce okuma maliyeti
                   artar, PARAMETRE artmaz.

=====================================================================
KARARLAR ve NEDEN
=====================================================================
KONUMLAR SABIT     Cokus imkansiz olsun. OLCULDU (model_13): kapanma
                   cezasi 2->20'de bulut 0,4666 -> 0,2025 buzuldu.
                   Konum parametre degilse buzulemez.

DONME, OTELEME DEGIL  Oteleme DEGISMELIDIR: "X'in annesinin kardesi"
                   ile tersi AYNI toplami verir. Sonum (lambda<1)
                   denendi, ELENDI: sirayi ayirmak icin lambda 0,8
                   gerekiyordu, o da 6 adim sonra ozneyi 0,26'ya
                   dusuruyordu -- model_13'un olculmus arizasinin
                   (0,22) aynisi.

D > d              Okuma IZDUSUM olmak ZORUNDA. Esit olsaydi ortak
                   sonek izometri olur ve butun iliskilerde AYNI
                   cevap-geometrisini dayatirdi. Sayi: gereken hata
                   <0,38 derece, cikan ~100 derece. 250 kat.

OKUMA ACISAL       ||PI z - hedef||^2 gizli kutleyi SIFIRA iter, oysa
                   D>d onu ZORUNLU kiliyor -- iki terim kavga eder.
                   Normalize edilmis yon kullanilir.

TEK DUZLEM YETMEZ  Givens donmesi kendi duzleminin disinda ozdesliktir;
                   rastgele bir farkin ancak 2/D'sine dokunur
                   (D=32 -> %6,3). D>d kacisi cogu ciftte calismaz.
                   COZUM DILSEL: kapali sinif (yuksek frekans) TAM
                   SO(D), acik sinif TEK DUZLEM. Bolme FREKANSTAN --
                   etiket, sozluk, oracle gerekmez.

CAPA = NICELEME    Varlik listesi YOK. Kod defteri ogrenilir; model
                   hangi durumlarin yeniden kullanilmaya deger
                   oldugunu KENDI bulur. Reddetme de buradan cikar:
                   kodun yaricapinda degilse "yok".

CAPADA GRADYAN KESILIR  Ileri gecis gecmisi siliyorsa geri gecis de
                   silmeli. Durumun koda dogru cekilmesi L_capa'nin
                   BAGLILIK teriminden gelir, straight-through'dan
                   degil.

ICSEL CAPA         Uretimde capa YAZILAN kelimeye degil durumun kendi
                   konumuna bakar -> kopru YAZILMADAN capalanir.
                   Kopruyu yazdirmak (CoT) CLAUDE.md'ye gore AYRI bir
                   sorudur; bu proje ORTUK cikarimi arastiriyor.

SAAT VARSAYILAN KAPALI  Tekrar ayrimini D>d boslugu ve farkli capalar
                   zaten yapiyor. Saatin maliyeti var: ayni olgu
                   bildirimde ve soruda FARKLI adim sayisinda gelir,
                   kisitlar ~3 kat olur. Acmak bir OLCUM karari.

r, delta PARAMETRE DEGIL   Karar esikleri; kayipta yok, TUTULAN
                   bolmede aranir (birer olcumdur).

Bu dosya DISARIYA HICBIR SEY IMPORT ETMEZ.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Mesafe TABANI -- `dis`teki sqrt'un 0'da tekilligini kesiyor.
# HESAP: taban 1e-8 -> dmin >= 1e-4, gradyan <= 0,5/1e-4 = 5.000
# (sonsuz yerine). Mentesenin en uc terimi (delta-0)^2 = 0,1600
# yerine (delta-1e-4)^2 = 0,1599 oluyor: %0,05 sapma.
TABAN = 1e-8


def _ust(D: int) -> torch.Tensor:
    """SO(D)'nin serbestlik indeksleri -- ust ucgen, kosegen haric."""
    return torch.triu_indices(D, D, offset=1)


def _ters_simetrik(v: torch.Tensor, D: int, iu: torch.Tensor) -> torch.Tensor:
    """(..., D(D-1)/2) -> (..., D, D).

    Lie cebirinden gidiyoruz cunku R dogrudan optimize edilemez: Adam
    ortogonalligi bozar. Kisit yonetimi, ceza terimi ve yeniden
    ortogonallestirme -- hicbiri gerekmiyor."""
    A = v.new_zeros(*v.shape[:-1], D, D)
    A[..., iu[0], iu[1]] = v
    return A - A.transpose(-1, -2)


def sinif_ayir(frekans, k_tam: int | None) -> torch.Tensor:
    """En sik `k_tam` birim KAPALI SINIF sayilir -> tam SO(D).

    `k_tam = None` HEPSI demek -- niyeti dogrudan soyleyen deger.
    Onceki hal sozluk boyunu ASAN bir sayi (451 > 444) yaziyordu ve
    ayni seyi TESADUFEN soyluyordu: sozluk 451'i gecse frekans ayrimi
    kendiliginden, keyfi bir kesimle geri gelirdi. Kapi 31 artik
    bunu yasakliyor.

    Dilbilimsel kapali sinif (ek, noktalama, kalip, iliski sozcugu)
    yuksek frekansli olandir. Etiket ya da sozluk gerekmez."""
    f = torch.as_tensor(frekans, dtype=torch.float)
    if k_tam is None:
        return torch.ones(len(f), dtype=torch.bool)
    m = torch.zeros(len(f), dtype=torch.bool)
    m[f.topk(min(k_tam, len(f))).indices] = True
    return m


class Yol(nn.Module):
    """Kurede yol modeli.

    n      birim sayisi
    D      DURUM uzayi -- donmeler burada
    d      OKUMA uzayi -- sabit noktalar burada.  d < D ZORUNLU
    K      kod defteri buyuklugu
    tam    (n,) bool -- hangi birimler tam SO(D).  `sinif_ayir` verir.
    saat   varsayilan KAPALI (docstring).
    """

    def __init__(self, n: int, D: int = 32, d: int = 8, K: int = 2048,
                 tam: torch.Tensor | None = None, saat: bool = False,
                 tohum: int = 0, hafiza: bool = False,
                 haf_b0: float = -0.29):
        super().__init__()
        assert d < D, "D > d ZORUNLU -- DENKLEM.md §4.1, izometri celiskisi"
        self.n, self.D, self.d, self.K, self.saat = n, D, d, K, saat
        g = torch.Generator().manual_seed(tohum)
        rn = lambda *s: torch.randn(*s, generator=g)

        self.register_buffer("iu", _ust(D))
        self.register_buffer("I", torch.eye(D))

        # SABIT konumlar -- `buffer` cunku OGRENILMEMELI (bulut buzulemesin).
        p = rn(n, d)
        self.register_buffer("p", p / p.norm(dim=-1, keepdim=True))

        if tam is None:                       # hicbiri tam degilse mimari
            tam = torch.zeros(n, dtype=torch.bool)   # zayif kalir (§4.3)
        self.register_buffer("tam", tam)
        self.register_buffer("ix_tam", tam.nonzero(as_tuple=True)[0])
        self.register_buffer("ix_acik", (~tam).nonzero(as_tuple=True)[0])

        # Baslangic olcegi: tipik donme acisi BIR KOMSU ARALIGI olsun.
        # R~I ile baslasa yol kimildamaz; cok buyukse rastgele yurur.
        #   aralik ~ n^(-1/(d-1)),  ters simetrigin acisi ~ sigma*sqrt(D)
        # HESAP, olcum degil (DENKLEM.md §9.4).
        aralik = n ** (-1.0 / (d - 1))
        sigma = aralik / math.sqrt(D)
        T = self.iu.shape[1]

        self.a = nn.Parameter(rn(len(self.ix_tam), T) * sigma)      # tam
        self.u = nn.Parameter(rn(len(self.ix_acik), D))             # tek duzlem
        self.v = nn.Parameter(rn(len(self.ix_acik), D))
        self.th = nn.Parameter(rn(len(self.ix_acik)) * aralik)
        self.C = nn.Parameter(F.normalize(rn(K, D), dim=-1))        # kod defteri
        # OLGU HAFIZASI (§12b): C KILIT, V DEGER.  V SIFIRDAN baslar,
        # yani model tam eskisi gibi baslar ve hafizayi kendi buyutur.
        self.V = nn.Parameter(torch.zeros(K, D)) if hafiza else None
        # HAFIZA KAPISI: ReLU(<z,K> + hb).  `hb` OGRENILIR -- sabit
        # olsaydi seyreklik bir VARSAYIM olarak kalirdi.
        # hb0 HESAP (§12c/Tasarim 4): <z,k> birim vektorlerde
        # std ~ 1/sqrt(D); %p atesleme icin hb0 = -z_p / sqrt(D).
        self.hb = nn.Parameter(torch.full((K,), haf_b0)) if hafiza else None
        self.s = nn.Parameter(rn(T) * sigma) if saat else None

    # ---------------- donmeler ----------------
    def donme(self) -> torch.Tensor:
        """(n, D, D). Kapali sinif exp ile, acik sinif KAPALI FORM ile.

        Tek duzlem icin matrix_exp gerekmez -- Rodrigues:
          R = I + sin0 (v u^T - u v^T) + (cos0 - 1)(u u^T + v v^T)"""
        R = self.I.expand(self.n, self.D, self.D).clone()
        if len(self.ix_tam):
            R = R.index_copy(0, self.ix_tam,
                             torch.matrix_exp(_ters_simetrik(
                                 self.a, self.D, self.iu)))
        if len(self.ix_acik):
            u = F.normalize(self.u, dim=-1)
            v = self.v - (self.v * u).sum(-1, keepdim=True) * u
            v = F.normalize(v, dim=-1)
            sn = torch.sin(self.th)[:, None, None]
            cs = (torch.cos(self.th) - 1)[:, None, None]
            uo, vo = u[:, :, None], v[:, :, None]
            R = R.index_copy(0, self.ix_acik,
                             self.I + sn * (vo * u[:, None, :]
                                            - uo * v[:, None, :])
                             + cs * (uo * u[:, None, :] + vo * v[:, None, :]))
        return R

    def kod(self) -> torch.Tensor:
        """(K, D) kod defteri, KUREDE.

        `C` serbest bir parametre; egitimde kureden cikar ve o zaman
        capa durumu kureden atar -- "norm korunur" iddiasi duser.
        Normalizasyon ileri geciste yapilir, ceza terimiyle degil."""
        return F.normalize(self.C, dim=-1)

    def hedef(self, T: int) -> torch.Tensor:
        """(T, n, d) -- saat t iken okuma hedefleri, birim normda.

        Saat kapaliysa her t icin ayni: p."""
        if not self.saat:
            return self.p[None].expand(T, self.n, self.d)
        A = _ters_simetrik(self.s, self.D, self.iu)
        t = torch.arange(T, device=A.device, dtype=A.dtype)
        St = torch.matrix_exp(t[:, None, None] * A)          # (T, D, D)
        L = self.p.new_zeros(self.n, self.D)
        L[:, :self.d] = self.p
        return F.normalize(torch.einsum("tij,nj->tni", St, L)[..., :self.d],
                           dim=-1)

    # ---------------- yol ----------------
    @staticmethod
    def esik(r: float) -> float:
        """ic carpim esigi.  2-2s < r^2  <=>  s > 1 - r^2/2.

        r = 0  ->  2.0, yani `s > esik` HIC dogru olmaz: CAPA KAPALI
        (§12c).  s fp32'de 1'i birkac ulp asabiliyor, o yuzden 1.0
        degil 2.0."""
        return 1.0 - 0.5 * r * r if r > 0 else 2.0

    def adim(self, z, w, Rd, C, Ct, esik):
        """TEK ADIM:  zp -> capa -> hafiza.  -> (z, zp, vur, k, m, hf)

        `hf` = kapinin aktivasyonu (B, K); hafiza kapaliysa None.
        `yol` onu ANINDA ozetliyor -- (B,L,K) yigmak 8192'de 1,5 GB.

        HAFIZA KAPISI ELEMAN BAZINDA (Tasarim 4, transformer FFN'inden):
            g = ReLU(<zp, K> + hb)        m = g @ V
        Once softmax(top-8), tau=0,02 idi ve bir yuvanin gradyan
        almasi icin 8.184 rakibi YENMESI gerekiyordu; kazanan anahtar
        sorgulara yaklasip daha cok kazaniyordu -- kendini besleyen
        dongu.  OLCULDU (§5.1/U): 8.192 yuvanin 10'u atesliyordu.
        ReLU'da i'nin ateslemesi j'yi BASTIRMIYOR: dongu yok.
        !! `sa` artik GRADYANLI ve L-1 adim tutuluyor (kapi 26).

        !! EGITIM ve URETIM BU AYNI KODU cagirir.  Ayri yazilmislardi
        ve uretim tarafi hafizayi HIC okumuyordu; §12b kosusu boylece
        hafizayla egitilip HAFIZASIZ olculdu ve BICIM sayilari
        gecersiz cikti (21 Eylul).  Kapi 38 ikisini birbirine baglar.

        ARAMA GRADYANSIZ: `k` indeks, `vur` bool -- geri hicbir sey
        akmaz. Gradyan C'ye C[k] ve C[kn] uzerinden gidiyor. no_grad
        olmadan (B,K) ara tensor geri gecis icin TUTULUYOR."""
        zp = torch.bmm(F.embedding(w, Rd).view(-1, self.D, self.D),
                       z.unsqueeze(-1)).squeeze(-1)
        V = self.V
        if V is None:
            with torch.no_grad():
                s, k = (zp @ Ct).max(1)
                vur = s > esik
            return torch.where(vur[:, None], C[k], zp), zp, vur, k, None, None
        # TEK matmul: `k`/`vur` ayni skorlardan, KOPARILARAK cikiyor.
        sa = zp @ Ct
        with torch.no_grad():
            s, k = sa.max(1)
            vur = s > esik
        z = torch.where(vur[:, None], C[k], zp)
        # OLGU HAFIZASI -- EKLEMELI, ve sorgu `zp`den (capa ONCESI):
        # capa tetiklendiginde z artik c_k ve ozne kimligi orada YOK
        # (olculdu: capa tetikleyen orneklerde 1,10 kat, otekilerde
        # 11,34 kat sans ustu).
        # !! YERINDE: `sa + hb` ve `relu` ayri tensor olsaydi adim
        # basina UC (B,K) tutulurdu.  B=2048, K=8192'de her biri 67 MB
        # x 23 adim = 4,6 GB; teki 1,54 GB.  `sa`ya baskasi bakmiyor
        # (matmul'un geri gecisi girdilerine bakar, ciktisina degil).
        g = sa.add_(self.hb).relu_()
        m = g @ V
        return F.normalize(z + m, dim=-1), zp, vur, k, m, g

    def yol(self, X: torch.Tensor, r: float = 0.25) -> dict:
        """X (B, L) -> sozluk:  z, zp, t, vur, k, mn   (hepsi (B,L,...))

            z    capa SONRASI durum      -- okuma bunu kullanir
            zp   capa ONCESI durum       -- VQ kaybi BUNU kullanir
            t    son capadan beri adim
            vur  capa tetiklendi mi
            k    en yakin kodun indeksi
            mn   hafiza okumasinin normu |m|  (hafiza yoksa 0)

        !! `zp` ayri donuyor cunku capa sonrasi z TAM OLARAK C[k]'dir;
        VQ baglilik terimi z uzerinden yazilirsa ||z - C[k]|| = 0 olur
        ve terim BOSA CALISIR. (Bu hatayi test 19/16 yakaladi.)

        Egitimde w_{j-1} GERCEK birim (ogretmen zorlamasi); z_j onekin
        kapali formlu bileskesidir, ozyineleme yok.

        CAPADA GRADYAN KESILIR: `torch.where` ile z kolu kopar, C kolu
        kalir. Durumu koda ceken sey L_capa'nin BAGLILIK terimi."""
        R, C = self.donme(), self.kod()
        B, L = X.shape
        # Donme ARAMASI `embedding` ile: ayni ileri gecis, ama geri
        # gecisi `embedding_dense_backward` (siralanmis, segmentli)
        # yapiyor -- genel `index_put` adim basina 8,4 M atomik
        # toplama demekti.
        Rd = R.reshape(self.n, self.D * self.D)
        # Kod defteri devrigi DONGU DISINDA: adim basina (D,K) kopyasi
        # cikariliyor.
        Ct = C.t().contiguous()
        esik = self.esik(r)
        # Pencere basi: ilk birim okuma uzayina, gizli kisim sifir.
        # Pencere akistan KEYFI yerden basliyor, yani bastaki durum COP.
        # "Ilk capada silinir" IDDIASI KALDIRILDI -- olculdu, capa adim
        # 1'de tetiklenmiyor (s = 0,5482, esik 0,969) ve kayip copu
        # adim 1'den itibaren puanliyor.  DENKLEM.md §5.1/G, acik A3.
        z = self.p.new_zeros(B, self.D)
        z[:, :self.d] = self.p[X[:, 0]]
        sf_b = torch.zeros(B, dtype=torch.bool, device=X.device)
        sf_l = torch.zeros(B, dtype=torch.long, device=X.device)
        sf_f = torch.zeros(B, device=X.device)
        o = {"z": [z], "zp": [z], "t": [sf_l], "vur": [sf_b], "k": [sf_l],
             "mn": [sf_f], "ak": [sf_f]}
        # !! AKTIVASYON ANINDA OZETLENIR.  (B,L,K) yigmak K=8192'de
        # 1,5 GB; oysa kayibin ihtiyaci iki SAYI: konum basina kac
        # yuva atesledi, ve HANGI yuvalar hic atesledi.
        ates = (torch.zeros(self.K, dtype=torch.bool, device=X.device)
                if self.V is not None else None)
        t = sf_l
        for j in range(1, L):
            z, zp, vur, k, m, g = self.adim(z, X[:, j - 1], Rd, C, Ct, esik)
            t = (t + 1).masked_fill(vur, 0)
            if g is not None:
                with torch.no_grad():
                    acik = g > 0
                    ates |= acik.any(0)
            for ad, u in (("z", z), ("zp", zp), ("t", t), ("vur", vur),
                          ("k", k),
                          ("mn", sf_f if m is None else m.norm(dim=-1)),
                          ("ak", sf_f if g is None
                           else acik.sum(1).to(sf_f.dtype))):
                o[ad].append(u)
        y = {a: torch.stack(u, 1) for a, u in o.items()}
        y["ates"] = ates
        return y

    def _kos(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """(B,L,n) -- cos(aci).  `oku` bunun 2-2x'i.

        Kayip dogrudan BUNU kullanir: `2-2x` ve `clamp` (B,L,n) boyunda
        IKI gecici tensor daha demek -- B=8192 L=16 n=451'de her biri
        222 MB, ve ikisinin de geri gecisi var. Ayni sayi, ucte bir
        trafik.

        !! Saat kapaliyken hedef sabittir ve `q @ p^T` yeter. Genel yol
        H[t] ile (B,L,n,d) bir tensor kurar -- 498 milyon float, 2 GB.
        (Bu hatayi test 17 yakaladi.)"""
        q = F.normalize(z[..., :self.d], dim=-1)
        if not self.saat:
            return q @ self.p.T
        H = self.hedef(int(t.max()) + 1)                     # (T, n, d)
        return (q[:, :, None, :] * H[t]).sum(-1)

    def oku(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """(B,L,n) -- ACISAL uzaklik.  2 - 2cos(aci).  DENKLEM §4.2."""
        return (2 - 2 * self._kos(z, t)).clamp(min=0)

    def _itme(self, kos, X, delta):
        """L_dis + yolun her birime EN YAKIN gectigi uzaklik.  (dis, dmin)

        AYRI BIR METOT cunku kapisi var (test 28): burada sqrt'un
        girdisi TABAN'in altina DUSEMEZ, ve kapi bunu cihazdan
        bagimsiz olarak sinar.

        OLCULDU (20 Eylul), egitim ADIM 1'de NaN verdi. `kos` bir
        kosinus ama fp32'de 1'i asabiliyor: olculen en kucuk 2-2kos
        degeri -2,384e-07. clamp(min=0) onu TAM 0 yapiyor ve
        sqrt'un turevi orada tanimsiz (0/0).

        !! Ve bu CIHAZA GORE DEGISIYOR: CPU'da ClampBackward NaN'i
        yutuyor, CUDA'da yutmuyor. Yerelde 50 adim temiz kosarken
        GPU'da adim 1'de patlamasinin sebebi buydu -- yani bu sinif
        hatayi yerel CPU dongusu GOREMEZ, kapi degeri sinamali.

        Cakisma kaza degil: acik sinif donmesi TEK DUZLEM (§4.3),
        duzlem z0'a dik dusunce R z0 ~ z0 kaliyor ve q1 onceki
        birimin KENDI noktasina esitleniyor. B=256'da hic yoktu,
        B=1024'te 13.888, B=8192'de 26.288 sonsuz gradyan."""
        d2 = (2 - 2 * kos.amax(1)).clamp(min=0)
        # Pencerede OLAN birim itilmez: delta'nin otesine koyuluyor,
        # boylece mentese zaten 0 ve sqrt'e 0 gitmiyor.
        ic = torch.zeros_like(d2, dtype=torch.bool).scatter_(1, X, True)
        dmin = d2.masked_fill(ic, delta + 1.).clamp(min=TABAN).sqrt()
        return (delta - dmin).clamp(min=0).pow(2).sum(1).mean(), dmin

    def forward(self, X: torch.Tensor, r: float = 0.25) -> torch.Tensor:
        y = self.yol(X, r)
        return -self.oku(y["z"], y["t"])

    # ---------------- kayip ----------------
    def kayip(self, X, a1=1.0, a2=1.0, a3=1e-4, delta=0.4, beta=0.25,
              r=0.25, isin=4, a4=0.0, haf_b=0.30):
        """YOLUN TAMAMINA bakar -- next token YOK.  DENKLEM.md §5.

        a1..a4 OLCULMEDEN secilmez; buradakiler baslangic.

        `a4`, `haf_b`  HAFIZA BUTCESI (§12c).  Duz L1 bedeli kagitta
        sinandi ve araligi BOS cikti: ikameyi engellemek a4 > 0,639,
        amaclanan kullanimi birakmak a4 < 0,410.  Mentese ile ikisi
        AYRISIYOR -- butcenin altinda BEDAVA, ustunde kareyle artiyor.

        !! `a5` (yuk dengeleme) KALDIRILDI.  Kapi artik ReLU, yani
        yuvalar YARISMIYOR ve "top-1 payi" diye bir sey yok; terim
        anlamsizlasti.  Cokusun KAYNAGI yarismaydi (§5.1/U) ve
        Tasarim 4 onu kokten kaldiriyor.  Yerine izlenen: `yuva`
        (hic atesleyen yuva sayisi) ve `ak` (konum basina ortalama
        aktif yuva).

        HAFIZA TERIMLERI BUTUN KONUMLARA bakar, `isin` dilimine DEGIL.
        ISINMA `uye`yi ilgilendirir (bastaki onek cop, §5.2) ama
        hafiza oraya da YAZIYOR ve yazdigi sey ileri tasiniyor.
        Onceki hal `[:, isin:]` idi ve pencerenin %17'sine yazmak
        BEDAVAYDI; izde `|m|` j=1'de 4,79 cikiyordu (§5.1/U).

        `isin` ISINMA: ilk `isin` konum PUANLANMAZ. Pencere akistan
        keyfi yerden basliyor, yani bastaki onek cop. HESAP (§5.2):
        gecis j = i mod `atla` kalinti sinifinda kalir, yani L=24 /
        atla=4'te her gecis 5-6 AYRI konumda puanlanir ve bunlarin
        EN COK BIRI 1..3 arasindadir. isin=4 o tek konumu atar --
        hicbir gecis egitimden dusmez, dort kalinti sinifi da tam 5
        konum tutar.  (kapi 32)"""
        y = self.yol(X, r)
        # !! DILIMLEME OKUMADAN ONCE. Adim 0 VERILMIS; once okuyup sonra
        # dilimlemek (B,L,n)'in tamamini hesaplamak demek.
        # !! `_kos` -- `oku` degil. Ihtiyacimiz olan her sey cos'tan
        # cikiyor; 2-2x ve clamp iki (B,L-isin,n) tensor daha ekliyordu.
        kos = self._kos(y["z"][:, isin:], y["t"][:, isin:])
        uye = 2 - 2 * kos.gather(2, X[:, isin:, None]).squeeze(-1).mean()

        # ITICI kuvvet. Pencerede OLMAYAN birim yolun yanindan gecmemeli.
        # Mentese: hepsi delta'yi gecince terim sifirlanir.
        # !! BILINEN TUZAK: bu birimlerin bir kismi GECERLI alternatif
        # (korpusta 17 bildirim kalibi var); mentese softmax'tan serttir.
        dis = self._itme(kos, X, delta)[0]

        # VQ. Kod terimi HER ADIMDA (kodlar ziyaret edilen durumlari
        # izlesin, k-ortalama gibi); BAGLILIK yalniz capa tetiklendiginde
        # (yoksa butun durumlar koda cekilir ve model sonlu otomata coker).
        # !! `zp` -- capa ONCESI durum. `z` kullanilirsa fark sifirdir.
        # !! `k` yol()tan geliyor; yeniden cdist B*(L-1) x K matris demek.
        # !! a2 = 0 iken HIC HESAPLANMAZ.  §12c capayi kaldiriyor ve C'yi
        # SERBEST ANAHTAR yapiyor; `kod` terimi C'yi k-ortalamaya zorlar
        # ve OLCULDU (§3.1b) ki o zorlama C'ye OLGUYU degil ILISKIYI
        # kodlatiyor -- hafizanin adresini bozan sey tam buydu.
        vf = y["vur"][:, isin:].reshape(-1)
        if a2:
            zf = y["zp"][:, isin:].reshape(-1, self.D)
            Ck = F.embedding(y["k"][:, isin:].reshape(-1), self.kod())
            kod = (zf.detach() - Ck).pow(2).sum(-1).mean()
            bag = ((zf - Ck.detach()).pow(2).sum(-1) * vf).sum() \
                / vf.sum().clamp(min=1)
        else:
            kod = bag = uye.new_zeros(())

        # HAFIZA BUTCESI -- butcenin altinda BEDAVA (docstring).
        # !! DILIM YOK: butun konumlar.  Gerekce docstring'de.
        mn = y["mn"][:, 1:].mean()
        haf = (mn - haf_b).clamp(min=0).pow(2) if a4 else mn.new_zeros(())

        # a3 = EZBER <-> GENELLEME dugmesi.
        duzen = self.a.pow(2).sum() + self.th.pow(2).sum()
        top = uye + a1 * dis + a2 * (kod + beta * bag) + a3 * duzen \
            + a4 * haf

        # IZ -- her cagride guncellenen kopuk skalerler. Senkron YOK
        # (kimse float() cagirmadikca), maliyeti yok. Kayip sayisal
        # olarak patlarsa hangi terimde patladigini bu soyler.
        # `mn` BUTCENIN kendisi: 0,30'u asarsa ikame basliyor demektir.
        # `yuva` ONKOSUL: KAC yuva hic atesledi.  Tasarim 3'te bu
        # sayi bir epokta 8078 -> 796 dusuyordu (§5.1/U).
        self.son = {"top": top.detach(), "uye": uye.detach(),
                    "dis": dis.detach(), "kod": kod.detach(),
                    "bag": bag.detach(), "duzen": duzen.detach(),
                    "capa": vf.float().mean().detach(), "mn": mn.detach(),
                    "ak": y["ak"][:, 1:].mean().detach(),
                    "yuva": (int(y["ates"].sum())
                             if y["ates"] is not None else 0),
                    "Pz_min": y["z"][..., :self.d].norm(dim=-1).min().detach(),
                    "V_max": (self.V.norm(dim=-1).max().detach()
                              if self.V is not None
                              else torch.zeros((), device=top.device))}
        return top, uye

    # ---------------- uretim ----------------
    @torch.no_grad()
    def uret(self, onek, en_cok=12, isin=50, dur=(), r=0.25):
        """ISIN ARAMASI + ICSEL CAPA.

        Capa YAZILAN kelimeye degil durumun konumuna bakar; kopru
        YAZILMADAN capalanabilir (ortuk cikarim).

        `r` buyurse ILGISIZ bir koda capalanir ve model KENDINDEN EMIN
        yanlis verir. Kucukse capa hic tetiklenmez ve comp garantisi
        duser. Olculecek esik."""
        R, C = self.donme(), self.kod()
        Rd, Ct, esik = (R.reshape(self.n, self.D * self.D),
                        C.t().contiguous(), self.esik(r))
        H = self.hedef(en_cok + len(onek) + 1) if self.saat else None
        dur = set(dur)
        z = self.p.new_zeros(1, self.D)
        z[:, :self.d] = self.p[onek[0]]
        t = torch.zeros(1, dtype=torch.long, device=z.device)
        yol = torch.tensor([[onek[0]]], device=z.device)
        puan = torch.zeros(1, device=z.device)

        for j in range(1, len(onek) + en_cok):
            z, _, vur, _, _, _ = self.adim(z, yol[:, -1], Rd, C, Ct, esik)
            t = (t + 1).masked_fill(vur, 0)

            q = F.normalize(z[:, :self.d], dim=-1)
            d2 = ((2 - 2 * (q @ self.p.T)) if H is None else
                  (2 - 2 * (q[:, None, :] * H[t]).sum(-1))).clamp(min=0)
            if j < len(onek):                            # onek VERILMIS
                yol = torch.cat([yol, torch.full_like(yol[:, :1], onek[j])], 1)
                puan = puan + d2[:, onek[j]]
                continue
            top = (puan[:, None] + d2).flatten()
            puan, ix = top.topk(min(isin, top.numel()), largest=False)
            sec = torch.div(ix, self.n, rounding_mode='floor')
            yol = torch.cat([yol[sec], (ix % self.n)[:, None]], 1)
            z, t = z[sec], t[sec]
            if int(yol[0, -1]) in dur:
                break
        return yol[0].tolist(), float(puan[0])


def n_par(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def saglik(m: Yol) -> str:
    """Son `kayip` cagrisinin SAGLIK tablosu.  TEK senkron noktasi.

    Normalize edilen her yon bir tehlike: paydasi sifira yaklasirsa
    geri gecis 1e12 mertebesinde gradyan uretir (olculdu; F.normalize
    eps=1e-12'nin hemen ustunde patlar). Uc tane var -- okuma yonu
    Pz, ve donmenin u / v_dik'i."""
    # !! ACIK SINIF BOS OLABILIR (K_TAM = n). Bos tensorde .min()
    # patliyor; bu satir yalniz TEK DUZLEM kolunu izliyor, o kol yoksa
    # izlenecek bir sey de yok.
    if len(m.ix_acik):
        u = F.normalize(m.u, dim=-1)
        vd = float((m.v - (m.v * u).sum(-1, keepdim=True) * u)
                   .norm(dim=-1).min())
        un = float(m.u.norm(dim=-1).min())
    else:
        vd = un = float("nan")
    s = getattr(m, "son", {})
    return ("terim  " + "  ".join("%s %.4f" % (k, float(s[k])) for k in
                                  ("uye", "dis", "kod", "bag", "duzen")
                                  if k in s)
            + "\n       capa %%%.2f   |m| %.4f   |V|max %.3f   YUVA %d/%d"
              "   aktif %.1f\n       |Pz|min %.2e   |u|min %.2e   "
              "|v_dik|min %.2e   (acik sinif %d)"
            % (100 * float(s.get("capa", 0)), float(s.get("mn", 0)),
               float(s.get("V_max", 0)), int(s.get("yuva", 0)), m.K,
               float(s.get("ak", 0)), float(s.get("Pz_min", 0)),
               un, vd, len(m.ix_acik)))


def kapi(m: Yol, tol: float = 1e-4) -> str:
    """Modelin kendi denetimi.

    Ortogonallik kapisi bos degil: Rodrigues'te u ve v dik degilse ya
    da matrix_exp yerine bir yaklasim kullanilirsa model SESSIZCE
    izometri olmaktan cikar ve butun gerekceler dusser."""
    R = m.donme()
    I = torch.eye(m.D, device=R.device)
    e = float((R @ R.transpose(-1, -2) - I).abs().max())
    assert e < tol, f"ortogonal DEGIL: {e:.2e}"
    assert float((m.p.norm(dim=-1) - 1).abs().max()) < tol, "konum kurede degil"
    assert m.d < m.D, "D > d"
    T = m.D * (m.D - 1) // 2
    bek = len(m.ix_tam) * T + len(m.ix_acik) * (2 * m.D + 1) + m.K * m.D \
        + (T if m.saat else 0) + (m.K * m.D if m.V is not None else 0)
    assert n_par(m) == bek, f"parametre {n_par(m)} != {bek}"
    return (f"GECTI  n={m.n} D={m.D} d={m.d} K={m.K}  saat={m.saat}  "
            f"hafiza={m.V is not None}\n"
            f"       parametre {n_par(m):,}   "
            f"kapali {len(m.ix_tam)} / acik {len(m.ix_acik)}"
            + (f"   V {m.K * m.D:,}" if m.V is not None else "")
            + f"\n       ortogonallik {e:.1e}   "
            f"komsu araligi {m.n ** (-1.0/(m.d-1)):.3f}")
