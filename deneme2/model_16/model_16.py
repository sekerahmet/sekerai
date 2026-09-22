"""model_16 -- YOL TUTULMAZ, HESAPLANIR.

  gez       s_t = normalize(M[token] @ s_{t-1} + b[token])
  dikkat    SON yuva sorar; her yuva Wk ile puanlanir, Wv ile katki verir
  dizi      `dikkat`in COGUL hali: HER yuva kendi sorusunu sorar,
            nedensel maske gelecegi kapatir
  agirlik   relu(puan + hb) -- softmax DEGIL.  1'e toplanmak zorunda
            olsaydi cikti hep ORTALAMA olurdu, TOPLAM tasinamazdi.
  puan      cikti noktasi -> sozluk puani.  TEK okuma noktasi:
            en yakin E[token].
  kayip     sonraki jeton, her konumda; `ofs` ile CEVAP KAPISI

OLCUMLER BU DOSYADA DEGIL -> `belge/onkayit/model_16.md`.
(CLAUDE.md kural 7: tarihsel kayit belgede durur, kaynak kodda ikinci
kez yasamasina gerek yok.)

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
  agirlik_16  cevap araliklari -> aralik ici sira (kapinin girdisi)
  dok_16      goz ile okunur dokum
  denetim_16  KAPILAR

  model_16    MIMARI + KAYIP   <-- BU DOSYA
  train_16    egitim dongusu: optimizer, yigin, kayit, SURDURME
  olcme_16    olcu: soru soruldu, cevap dogru mu
  adim_16     TEK SORU, ADIM ADIM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# BELIRLENIMCI KOSU -- ayni tohum ayni sonucu versin diye.  GPU'da bazi
# cekirdekler belirlenimci degil; orada zorlamiyoruz.
if not torch.cuda.is_available():
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

# --- AYARLAR.  Ayri bir ayar dosyasi YOK; gerekceler belge/onkayit/model_16.md
BOYUT = 16           # token kac sayiyla tarif ediliyor  (E ve q/k/v)
DURUM = 16           # s kac sayi
NORM = True          # |s| = 1
PAY = False          # False -> relu dikkat   True -> softmax
LR = 0.002
WD = 0.01
TEDIRGIN = 0.4       # M'nin tedirginligi, tokenin TOPLAMSAL kanalina gore.
                     # |b| = 1 oldugu icin bu dogrudan bir ORAN: 0,4 =
                     # "carpimsal kanal toplamsalin %40'i".  Boyuta bagli
                     # kismi asagida aritmetikle (1/sqrt(durum)); ORANIN
                     # KENDISI secildi, olculmedi.


class Yol(nn.Module):
    def __init__(self, n, boyut=BOYUT, durum=DURUM, tohum=0,
                 norm=NORM, pay=PAY):
        """Ayarlar dosyanin basinda -- ayri bir ayar dosyasi YOK."""
        super().__init__()
        g = torch.Generator().manual_seed(tohum)
        r = lambda *s: torch.randn(*s, generator=g)
        self.boyut, self.durum = boyut, durum
        self.norm, self.pay = norm, pay

        # TOKEN BASINA UC PARAMETRE -- ikisi ICERI, biri DISARI:
        #   b[w]  okununca duruma EKLENEN vektor     -> s = ... + b[w]
        #   M[w]  okununca duruma UYGULANAN matris   -> s = M[w] @ s + ...
        #         birim matris + kucuk tedirginlik: ogrenilmemis token
        #         durumu bozmaz, oldugu gibi gecirir
        #   E[c]  token YAZILIRKEN hedeflenen konum; okuma en yakin E
        # b ile E AYRI parametre: iceri giren gomme disari cikanla bagli
        # degil.  Baglamak denendi, aritmetik gorevinde daha kotuydu.
        #   E'nin satirlari TAM birim norm.  Puan acilinca
        #   -||o-E_c||^2 = -||o||^2 + 2 o.E_c - ||E_c||^2 olur; ilk terim
        #   c'den bagimsiz, softmaxta duser.  ||E_c||^2 de SABIT olursa o da
        #   duser ve siralamayi YALNIZ o.E_c belirler -- okuma ilk adimdan
        #   itibaren girdiye bagli olur.
        #   Olculdu (t0): duz randn ile |E|~3,94 ve girdiden BAGIMSIZ terimin
        #   yayilimi girdiye baglinin 19 KATI; model her soruya ayni tokeni
        #   veriyordu.  Bolmek yetmez -- olcek orani 1/k duzeltir ama
        #   ||E_c||^2'nin SIRALAMASINI degistirmez, secim yine ayni kalir.
        self.E = nn.Parameter(F.normalize(r(n, boyut), dim=-1))   # CIKIS gommesi
        self.b = nn.Parameter(r(n, durum) / durum ** 0.5)   # GIRIS, toplamsal
        #   /sqrt(durum) -> |b| ~ 1 = |s|: eklenen sey eklendigiyle ayni
        #   buyuklukte, biri otekini bastirmiyor
        #   TEDIRGINLIK olcegi b ile AYNI aritmetikten: N(0,s^2) girisli
        #   d x d matris icin |A@x| ~ s*sqrt(d)*|x|, yani s = TEDIRGIN/sqrt(d)
        #   verince |tedirginlik @ s| ~ TEDIRGIN * |s| -- oran d ne olursa
        #   olsun sabit.  Duz 0,1 yazilsaydi oran durumla buyurdu
        #   (16'da 0,40 ama 256'da 1,60) ve kimligi bastirirdi.
        #   durum=16'da TEDIRGIN/sqrt(16) = 0,1 -- bugunku deger.
        self.M = nn.Parameter(torch.eye(durum).repeat(n, 1, 1)  # GIRIS, carpimsal
                              + (TEDIRGIN / durum ** 0.5) * r(n, durum, durum))
        self.s0 = nn.Parameter(torch.zeros(durum))          # baslangic durumu

        o = boyut ** -0.5                # |q| ~ |k| ~ |s| = 1 olsun diye
        self.Wq = nn.Parameter(r(durum, boyut) * o)     # son yuva -> soru
        self.Wk = nn.Parameter(r(durum, boyut) * o)     # yuva -> anahtar
        self.Wv = nn.Parameter(r(durum, boyut) * o)     # yuva -> deger
        # YANLILIK -- hep var.  relu(q.k) secimi orijinden gecen yarim
        # uzaya hapsederdi; hb sinirini q.x = -hb yapar.  Maliyet 1 sayi.
        self.hb = nn.Parameter(torch.zeros(1))

    def gez(self, w):
        """w: (T,) ya da (B,T).  Doner: (B,T+1,durum) -- her adimdaki durum."""
        tek = w.dim() == 1
        w = w[None] if tek else w
        s = self.s0.expand(w.shape[0], self.durum)
        iz = [s]
        for t in range(w.shape[1]):
            wt = w[:, t]
            s = torch.bmm(self.M[wt], s.unsqueeze(-1)).squeeze(-1) + self.b[wt]
            if self.norm:
                s = F.normalize(s, dim=-1)
            iz.append(s)
        S = torch.stack(iz, 1)
        return S[0] if tek else S

    def puan(self, O):
        """Cikti noktasindan SOZLUK PUANI.  Buyuk = yakin.

        Tek yerde durmasi sart: dizi/uret_dizi/oku ucu de bunu cagirir,
        yoksa egitim bir kuralla, uretim baskasiyla calisir."""
        E = self.E.expand(O.shape[0], -1, -1) if O.dim() == 3 else self.E
        return -torch.cdist(O, E) ** 2

    def dikkat(self, w):
        """Son yuva sorar, butun yuvalar cevaplar.  Doner: (o, agirlik).

        Anahtar ve deger yuvanin TOKEN'INDAN degil DURUMUNDAN uretiliyor --
        boylece ayni token iki farkli yerde ayni sey demiyor.
        Maske gerekmiyor: yalniz son yuva soruyor, gelecegi yok.
        """
        tek = w.dim() == 1
        S = self.gez(w)[..., 1:, :]
        S = S[None] if tek else S
        q = S[:, -1] @ self.Wq
        pu = (S @ self.Wk) @ q.unsqueeze(-1)
        pu = pu.squeeze(-1) + self.hb
        ag = pu.softmax(-1) if self.pay else pu.clamp(min=0)
        o = (ag.unsqueeze(-1) * (S @ self.Wv)).sum(1)
        return (o[0], ag[0]) if tek else (o, ag)

    def dizi(self, w):
        """HER yuva kendi sorusunu sorar.  w (B,T) -> puan (B,T,n).

        `dikkat`in COGUL hali; mekanizma AYNI, degisen tek sey kimin
        sordugu.  Onek basina ayri ileri gecis T kat is ederdi.

        NEDENSEL MASKE ZORUNLU: butun yuvalar ayni dizide soruyor, maske
        gelecegi gormemeyi geri koyuyor.  relu ONCE maskelenir; maskeye
        -inf degil dogrudan agirlik 0 verilir (pay YOK).
        """
        assert w.dim() == 2, "dizi() (B,T) bekler"
        S = self.gez(w)[:, 1:]                      # (B,T,durum)
        Q, K, V = S @ self.Wq, S @ self.Wk, S @ self.Wv
        P = Q @ K.transpose(1, 2) + self.hb         # (B,T,T)
        gec = torch.ones(w.shape[1], w.shape[1], dtype=torch.bool,
                         device=w.device).tril()    # j yalniz <= j'ye bakar
        A = (P.masked_fill(~gec, -torch.inf).softmax(-1) if self.pay
             else P.clamp(min=0) * gec)
        O = A @ V                                   # (B,T,boyut)  bmm --
        #   ara (B,T,T,boyut) tensor URETMEZ; carpim olarak yazilmasi sart,
        #   yayilimla yazilsaydi B=64'te bile 1 GB olurdu.
        return self.puan(O)

    def kayip(self, w, PAD=None, ofs=None):
        """SONRAKI JETON, her konumda.  w (B,T) -> SKALER kayip.

        Konum j, j+1'i tahmin eder; son konumun hedefi yok.

        PAD  bicime bagli, zorunlu degil.  Paketlenmis pencerelerde dolgu
             sayilsaydi gradyanin bir kismi 'dolgu tahmin et' ogretirdi;
             kesintisiz akista dolgu YOK ve PAD=None verilir.
        ofs  (B,T) cevap araligi ici sira (0 = ilk token, 1+ = devam,
             -1 = disari).  None -> butun konumlar esit (duz kayip).
             Verilirse CEVAP KAPISI acilir: bir yuva, ayni araliktaki
             BUTUN onceki yuvalar dogru bilindiyse puanlanir; biri
             yanlissa araligin kalani PUANLANMAZ.  Sinav cevabin
             TAMAMINA bakiyor; onek bozulduysa devaminin dogrulugu
             sifir kazandiriyor.  Kapi argmax ile kuruluyor
             -- turevi yok, gradyan yalniz cross-entropy'den akar,
             ek ileri gecis YOK.  Aralik kurulumu: `agirlik_16`.
        """
        puan = self.dizi(w)                       # (B,T,n)
        ek = {} if PAD is None else {'ignore_index': PAD}
        if ofs is None:
            return F.cross_entropy(puan[:, :-1].reshape(-1, puan.shape[-1]),
                                   w[:, 1:].reshape(-1), **ek)
        k = F.cross_entropy(puan[:, :-1].reshape(-1, puan.shape[-1]),
                            w[:, 1:].reshape(-1), reduction="none", **ek)
        a = self._kapi(puan, w, ofs).reshape(-1)
        return (k * a).sum() / a.sum()

    @staticmethod
    def _kapi(puan, w, ofs):
        """CEVAP KAPISI -> (B,T-1) 0/1.  Bir yuva ancak ayni araliktaki
        BUTUN onceki yuvalar dogru bilindiyse puanlanir; ilk hatada
        araligin kalani kapanir.  Pencerenin BASINDAN once baslayan bir
        aralikin gorunmeyen yuvalari BILINMIYOR sayilir, kapi ACIK
        birakilir (bilmedigimiz icin cezalandirmiyoruz)."""
        hed = w[:, 1:]
        dogru = puan[:, :-1].argmax(-1) == hed      # (B,T-1) bool
        o1 = ofs[:, 1:]                             # hedefin aralik ici sirasi
        kapi = torch.ones_like(o1, dtype=torch.bool)
        # birikim: adim turunda "son `adim` yuvanin hepsi dogru mu".
        # o1 == adim olan yuva icin bu, aralikin 0..adim-1 onekidir.
        birikim = torch.ones_like(o1, dtype=torch.bool)
        for adim in range(1, int(o1.max().item()) + 1 if o1.numel() else 1):
            geri = torch.roll(dogru, shifts=adim, dims=1)
            geri[:, :adim] = True                   # pencere disi -> kapi ACIK
            birikim = birikim & geri                # her turda BIR ONCEKI yuva
            sec = o1 == adim
            if sec.any():
                kapi = torch.where(sec, birikim, kapi)
        return kapi.to(puan.dtype)

    def uret_dizi(self, w, adim):
        """w (B,L) -> uretilen (B,adim).  DURUM TASINIR.

        `dizi`nin SON satirini adim adim yurutur.  Durum tasindigi icin
        k adim yeter; bastan hesaplansa k*L adim ederdi.
        """
        S = self.gez(w)[:, 1:]                       # (B,L,durum)
        s = S[:, -1]
        cikan = []
        for _ in range(adim):
            q = s @ self.Wq
            pu = (S @ self.Wk) @ q.unsqueeze(-1)
            pu = pu.squeeze(-1) + self.hb
            ag = pu.softmax(-1) if self.pay else pu.clamp(min=0)
            o = (ag.unsqueeze(-1) * (S @ self.Wv)).sum(1)
            t = self.puan(o).argmax(-1)              # (B,)
            cikan.append(t)
            s = torch.bmm(self.M[t], s.unsqueeze(-1)).squeeze(-1) + self.b[t]
            if self.norm:
                s = F.normalize(s, dim=-1)
            S = torch.cat([S, s[:, None]], 1)
        return torch.stack(cikan, 1)

    def oku(self, o):
        """Sozluk uzayindaki noktaya en yakin token."""
        return int(self.puan(o.reshape(1, -1)).argmax())
