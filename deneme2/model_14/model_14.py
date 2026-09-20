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


def sinif_ayir(frekans, k_tam: int) -> torch.Tensor:
    """En sik `k_tam` birim KAPALI SINIF sayilir -> tam SO(D).

    Dilbilimsel kapali sinif (ek, noktalama, kalip, iliski sozcugu)
    yuksek frekansli olandir. Etiket ya da sozluk gerekmez."""
    f = torch.as_tensor(frekans, dtype=torch.float)
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
                 tohum: int = 0):
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
    def yol(self, X: torch.Tensor, r: float = 0.25) -> dict:
        """X (B, L) -> sozluk:  z, zp, t, vur, k   (hepsi (B,L,...))

            z    capa SONRASI durum      -- okuma bunu kullanir
            zp   capa ONCESI durum       -- VQ kaybi BUNU kullanir
            t    son capadan beri adim
            vur  capa tetiklendi mi
            k    en yakin kodun indeksi

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
        # ||z-C||^2 = 2 - 2 z.C  ve z,C birim normda; en YAKIN kod, ic
        # carpimi EN BUYUK olandir. Esigi de dogrudan ic carpima
        # ceviriyoruz:  2-2s < r^2  <=>  s > 1 - r^2/2.
        esik = 1.0 - 0.5 * r * r
        # Pencere basi: ilk birim okuma uzayina, gizli kisim sifir.
        # Pencere akistan KEYFI yerden basliyor, yani bastaki durum COP.
        # "Ilk capada silinir" IDDIASI KALDIRILDI -- olculdu, capa adim
        # 1'de tetiklenmiyor (s = 0,5482, esik 0,969) ve kayip copu
        # adim 1'den itibaren puanliyor.  DENKLEM.md §5.1/G, acik A3.
        z = self.p.new_zeros(B, self.D)
        z[:, :self.d] = self.p[X[:, 0]]
        sf_b = torch.zeros(B, dtype=torch.bool, device=X.device)
        sf_l = torch.zeros(B, dtype=torch.long, device=X.device)
        o = {"z": [z], "zp": [z], "t": [sf_l], "vur": [sf_b], "k": [sf_l]}
        t = sf_l
        for j in range(1, L):
            Rg = F.embedding(X[:, j - 1], Rd).view(B, self.D, self.D)
            zp = torch.bmm(Rg, z.unsqueeze(-1)).squeeze(-1)
            t = t + 1
            # !! ARAMA GRADYANSIZ. `k` bir INDEKS, `vur` bir BOOL --
            # ikisinden de geri hicbir sey akmaz. Gradyan C'ye asagida
            # C[k] uzerinden gidiyor. no_grad olmadan (B,K) ara tensor
            # geri gecis icin TUTULUYOR: adim basina 67 MB x 15 adim.
            with torch.no_grad():
                s, k = (zp @ Ct).max(1)
                vur = s > esik
            z = torch.where(vur[:, None], C[k], zp)
            t = t.masked_fill(vur, 0)
            for ad, v in (("z", z), ("zp", zp), ("t", t), ("vur", vur),
                          ("k", k)):
                o[ad].append(v)
        return {a: torch.stack(v, 1) for a, v in o.items()}

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
    def kayip(self, X, a1=1.0, a2=1.0, a3=1e-4, delta=0.4, beta=0.25, r=0.25):
        """YOLUN TAMAMINA bakar -- next token YOK.  DENKLEM.md §5.

        a1..a3 OLCULMEDEN secilmez; buradakiler baslangic."""
        y = self.yol(X, r)
        # !! DILIMLEME OKUMADAN ONCE. Adim 0 VERILMIS; once okuyup sonra
        # dilimlemek (B,L,n)'in tamamini hesaplamak demek.
        # !! `_kos` -- `oku` degil. Ihtiyacimiz olan her sey cos'tan
        # cikiyor; 2-2x ve clamp iki (B,L-1,n) tensor daha ekliyordu.
        kos = self._kos(y["z"][:, 1:], y["t"][:, 1:])
        uye = 2 - 2 * kos.gather(2, X[:, 1:, None]).squeeze(-1).mean()

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
        zf = y["zp"][:, 1:].reshape(-1, self.D)
        vf = y["vur"][:, 1:].reshape(-1)
        Ck = F.embedding(y["k"][:, 1:].reshape(-1), self.kod())  # BIR gather
        kod = (zf.detach() - Ck).pow(2).sum(-1).mean()
        bag = ((zf - Ck.detach()).pow(2).sum(-1) * vf).sum() \
            / vf.sum().clamp(min=1)
        capa = kod + beta * bag

        # a3 = EZBER <-> GENELLEME dugmesi.
        duzen = self.a.pow(2).sum() + self.th.pow(2).sum()
        top = uye + a1 * dis + a2 * capa + a3 * duzen

        # IZ -- her cagride guncellenen kopuk skalerler. Senkron YOK
        # (kimse float() cagirmadikca), maliyeti yok. Kayip sayisal
        # olarak patlarsa hangi terimde patladigini bu soyler; ve
        # `capa` sifir kalirsa mimarinin ASIL iddiasi (§3) hic
        # calismiyor demektir -- o da buradan gorulur.
        self.son = {"top": top.detach(), "uye": uye.detach(),
                    "dis": dis.detach(), "kod": kod.detach(),
                    "bag": bag.detach(), "duzen": duzen.detach(),
                    "capa": vf.float().mean().detach(),
                    "Pz_min": y["z"][..., :self.d].norm(dim=-1).min().detach()}
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
        H = self.hedef(en_cok + len(onek) + 1) if self.saat else None
        dur = set(dur)
        z = self.p.new_zeros(1, self.D)
        z[:, :self.d] = self.p[onek[0]]
        t = torch.zeros(1, dtype=torch.long, device=z.device)
        yol = torch.tensor([[onek[0]]], device=z.device)
        puan = torch.zeros(1, device=z.device)

        for j in range(1, len(onek) + en_cok):
            z = torch.bmm(R[yol[:, -1]], z.unsqueeze(-1)).squeeze(-1)
            t = t + 1
            yak2, k = (2 - 2 * (z @ C.T)).clamp(min=0).min(1)   # ICSEL CAPA
            vur = yak2 < r * r
            z = torch.where(vur[:, None], C[k], z)
            t = torch.where(vur, torch.zeros_like(t), t)

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
            + "\n       capa %%%.2f   |Pz|min %.2e   |u|min %.2e   "
              "|v_dik|min %.2e   (acik sinif %d)"
            % (100 * float(s.get("capa", 0)), float(s.get("Pz_min", 0)),
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
        + (T if m.saat else 0)
    assert n_par(m) == bek, f"parametre {n_par(m)} != {bek}"
    return (f"GECTI  n={m.n} D={m.D} d={m.d} K={m.K}  saat={m.saat}\n"
            f"       parametre {n_par(m):,}   "
            f"kapali {len(m.ix_tam)} / acik {len(m.ix_acik)}\n"
            f"       ortogonallik {e:.1e}   "
            f"komsu araligi {m.n ** (-1.0/(m.d-1)):.3f}")
