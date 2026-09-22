# -*- coding: utf-8 -*-
"""taban_16 -- model_16'un VERI YOLU. `taban_11`den KOPYA.

Kullanici karari, 20 Eylul 2026: *"evet onlari kopyala aynisi zaten"*.

TASINAN: Ayar, Veri, veri_kur, olcme_listeleri, olcme_izi ve
bagimliliklari. Kapanis AST ile cikarildi, elle secilmedi -- IMPORTLAR
DA OYLE: torch / jeton / dil / olcme kapanista HIC gecmiyor, bu dosya
torch'suz calisiyor.

TASINMAYAN (~1000 satir): egit, Model, Blok, RMSNorm,
surdurme_*, onbellek_*, erken_teshis, _atomik ... Hepsi TRANSFORMER
egitim motoru. model_16 metne hic bakmiyor, yalniz grafa.
CLAUDE.md kural 7: cagrisiz kod tasinmaz.

KOPYANIN KAPISI IZLER:
    veri_16.IZ      == 3cd9a2575e47
    olcme_izi(L)    == 44e6262e37f3

ZINCIRDEKI YERI.  Kim kimi cagiriyor, bu dosya nerede:
(model BIRIM goruyor -- karakter yalnizca ara adim ve kapi)

  veri_16     graf: 1608 varlik, 24 iliski, olgular
  metin_16    graf -> duz Turkce cumle
  korpus_16   cumle -> belge -> paketlenmis akis
  jeton_16    KARAKTER sozlugu + GIDIS-DONUS KAPISI
  birim_16    metin -> sayim -> kok havuzu -> BIRIM AKISI -> pencere
  ek_16       kelime -> kok + ek    (Turkce morfolojisi)

  taban_16    bolmeler (ezber_* / cikarim_*) + Ayar tanimi   <-- BU DOSYA
  ayar_16     dugmeler
  hazirla_16  veriyi dosyaya yazar, Colab Drive'dan OKUR

  model_16    MIMARI -- model_15'ten
  train_16    egitim dongusu
  olcme_16    olcu: soru soruldu, cevap dogru mu
"""
from __future__ import annotations

import dataclasses as dc
import hashlib
import importlib
import os

import numpy as np

import jeton_16 as J
import korpus_16 as KOR

SPECIAL = 3

T_LEN = 8

@dc.dataclass(frozen=True)
class Ayar:
    ad: str = "model_11"

    # --- veri (bir ailenin butun kollarinda AYNI olmali, yoksa
    #     'sartlar esit' bozulur ve kollar farkli veri gorur)
    veri_ad: str = "veri_11"     # HANGI GRAF. "veri_okul2" = tam IKI KATI.
    #   15 Eylul'de eklendi. Modul adi olarak yaziliyor ki `ayar_t<N>.json`a
    #   girsin: "bu kosu hangi veriyi gordu" sorusu SONRADAN cevaplanabilsin.
    #   Alan eklemek SURDURMEYI bozabilirdi (eski paketlerde bu anahtar YOK
    #   ve karsilastirma 'degismis' derdi); `surdurme_oku` icinde ESKI
    #   VARSAYILAN tablosu var, oraya bak.
    veri_tohum: int = 0
    # --- KORPUS (model_09) ------------------------------------------
    # `t_len` ARTIK TURETILMIYOR. model_08'de jeton semasindan
    # dusuyordu (2*yuva+11 = 17) cunku her satir TEK cumleydi ve
    # uzunlugu semanin sonucuydu. Burada pencere bir VERI KARARI:
    # metin 512'lik dilimlere bolunuyor, dilim sinirinin cumleyle
    # ilgisi yok. Turetilecek bir sey kalmadi.
    t_len: int = 512           # egitim penceresi (karakter)
    kopya: int = 5             # Physics 3.1 `multiM`: varlik basina M belge
    tetik: int = 0             # bu M belgenin kaci BIYOGRAFI SORUSU onegiyle
    # --- KORPUSUN BILESIMI (model_11). Hepsi AYARDAN gelir ki onkayit
    # ve defter tek yerden okusun; `havuz`un varsayilanina GUVENILMEZ.
    zincir_pay: float = 0.20   # sayfa cumlelerinin ~payi ZINCIR cumlesi
    n3: int = 0                # UC adimli zincir havuzu (0 = yok)
    ret_pay: float = 0.05      # soru cumlelerinin ~payi REDDETME
    ret_tut: float = 0.20      # reddetme verisinin SINAVA ayrilan payi
    yabanci_pay: float = 0.20      # varliklarin ne kadari ENT'e ayrilir
    cikarim_pay: float = 0.10     # zincirlerin ne kadari COMP'a ayrilir
    arama_pay: float = 0.25    # ENT'in ne kadari ARAMA'ya (HUKUMDEN AYRIK)

    # --- mimari
    #   MIMARI: 2604.07822 "Loop, Think & Generalize" satir 594, BIREBIR:
    #   "we use an embedding dimension of 768, 12 attention heads and a
    #    recurrent block of 4 transformer layers"
    #   Kafadan atilmadi; yayimlanmis ve ayni gorevde (2-hop OOD) calismis
    #   bir konfigurasyon. Kodu: github.com/OSU-NLP-Group/Loop-Think-Generalize
    d: int = 256               # 768 idi (Loop&Generalize). KUCULTULDU.

    # --- egitim
    tohum: int = 0
    jeton_ad: str = ""         # Varliklari COK JETON olarak kodla.
    ayrik_ood_pay: float = 0.0       # 0 = KAPALI. >0 ise ATOMIK OLGULARIN (kenar)
    #   bu orani atomic_OOD'ye ayrilir -- Wang 2405.15071 §3.1'in birebir
    #   tanimi: "The atomic facts are then the EDGES ... which we partition
    #   disjointly into atomic_ID and atomic_OOD (95%:5%)".
    #   EGITIM  = iki kenari da ID olan zincirler (train_inferred_ID)
    #   SINAV   = iki kenari da OOD olan zincirler (test_inferred_OOD)
    #   KARISIK = bir kenari OOD -> NE EGITIM NE SINAV, tamamen duser
    #   ATOMIK OLGULAR HEPSI EGITIMDE (Wang §2: "our training set includes
    #   ALL the atomic facts").
    #   `kati_pay`dan FARKI: orada VARLIK boluyorduk ve ikinci hop kenari
    #   egitimde 2. hop olarak %93,7 geciyordu -- yani makalenin %0'ini
    #   ureten mekanizma YOKTU. Burada tanim geregi %0.
    kati_pay: float = 0.0      # 0 = KAPALI. >0 ise varliklarin bu orani
    n_olcum_max: int = 3000    # her olcme kumesinden en fazla

    belge_pay: float = 0.0
    # --- EK ISARETLEYICILI KODLAMA (16 Eylul) --------------------------
    # Bugunku dilde rolu POZISYON tasiyor: "Fatma anne Ayse" ile
    # "Ayse anne Fatma" farkli seyler. Yani siradan bir permutasyon
    # ANLAMI BOZAR -- bu, bicim cesitliligini imkansiz kiliyordu.
    # Turkce'de sira serbesttir cunku rolu EK tasir:
    #   "Ayse'nin annesi Fatma'dir" = "Fatma'dir Ayse'nin annesi"
    # ek_kip="tr" dort jeton ekler -- ILISKI DEGIL, DILBILGISI:
    #   '     ozel adla ek arasina (Ayse'nin)
    #   <NIN> tamlayan (sahip)
    #   <SI>  tamlanan (iliski)
    #   <DIR> yuklem (cevap)
    # Jetonlar SOZLUGUN SONUNA ekleniyor -> REL_OFF/ent_off KAYMAZ,
    # ek_kip="" ile uretilen diziler BIT AYNI kalir.
    # ek_kip="ezber_zincir" -- KULLANICI KARARI, 17 Eylul:
    #   *"ben duzgun bir turkce ile egitim istiyorum. Fakultesi ile
    #    fakultesi ayri seyler, o da ilk harften ayrisiyor."*
    # "tr" kipinde iliski KOK jetonuydu ve iyeligi ayri bir <SI> jetonu
    # tasiyordu (fakulte + <SI>). "ezber_zincir"de iliski KELIMENIN KENDISI:
    #   fakultesi   kucuk harf, CINS isim   <- iliski
    #   Fakultesi   BUYUK harf, OZEL adin parcasi
    # Ikisini ayiran sey YAZIM; fazladan isaretleyici GEREKMIYOR.
    # <SI> KALKTI, yerine SORU SOZCUGU geldi (kim / neresi / hangisi).
    ek_kip: str = ""           # "" | "tr" | "ezber_zincir"
    tam_kayip: bool = False

    def degistir(self, **kw) -> "Ayar":
        bilinmeyen = set(kw) - {f.name for f in dc.fields(self)}
        assert not bilinmeyen, f"Ayar'da boyle alan yok: {bilinmeyen}"
        return dc.replace(self, **kw)

    def sozluk(self) -> dict:
        return dc.asdict(self)

    def fark(self, other: "Ayar") -> dict:
        a, b = self.sozluk(), other.sozluk()
        return {k: (a[k], b[k]) for k in a if a[k] != b[k] and k != "ad"}

KISAYOL_YOLU = (("fakultesi",    ("bolumu", "fakultesi")),
                ("konusu",       ("tezi", "konusu")),
                ("dekani",       ("bolumu", "fakultesi", "dekani")),
                ("universitesi", ("bolumu", "fakultesi", "universitesi")))

def _kisayol_kur(G, eid, rid, tut_bas, tut_kenar, tut_zincir):
    """(e, r_son, cevap) uclulerini uretir. Grafa DOKUNMAZ.

    Kisa yol satiri EGITIM verisidir ve COK ADIMLI bir gercegi 1-hop
    yuzeyle soyler. Dolayisiyla TUTULAN (held-out) her seyi ihlal
    edebilir. UC kapi var ve UCU DE gerekli -- biri eksik olursa
    sizinti SESSIZ olur:

      tut_bas     TUTULAN VARLIK. `cikarim_yabanci` tanimi "hic zincir basi
                  olmamis"tir; "X'in dekani = Y" satiri X'i tam o
                  konuma koyar. cikarim_yabanci / kapi_kisayolsuz / ayrik_arama / ayrik_kati.
      tut_kenar   TUTULAN KENAR. atomic_OOD kenarlari egitimde YOK;
                  yolu o kenardan gecen satir onu geri sokar.
      tut_zincir  TUTULAN ZINCIR. Iki adimli bir kisa yol, tam o
                  zincirin cevabidir.

    !! VARSAYILAN YOK, bilerek: izin verici bir varsayilan kapiyi
    sessizce acar. Doner: (satirlar, {sebep: sayi}).
    """
    olgu = G["olgu"]
    out, atilan = [], {"bas": 0, "kenar": 0, "zincir": 0}
    for r_son, yol in KISAYOL_YOLU:
        assert yol[-1] == r_son, (r_son, yol)
        for k in G["ad"]["KISI"]:
            if eid[k] in tut_bas:
                atilan["bas"] += 1
                continue
            c, kenar = k, False
            for r in yol:
                if (eid[c], rid[r]) in tut_kenar:
                    kenar = True
                    break
                c = olgu.get((c, r))
                if c is None:
                    break
            if kenar:
                atilan["kenar"] += 1
                continue
            if c is None:
                continue                 # yol EKSIK -- satir yok
            # !! KISA YOL BIR OLGUYU TEKRAR ETMEMELI: (k, r_son) zaten
            # kayitliysa satir yeni bir sey ogretmez. Semada KISI'den
            # bu DORT iliski CIKMIYOR, yani bu assert yapisal -- ama
            # sema degisirse SESSIZ kalmasin.
            assert (k, r_son) not in olgu, (
                f"kisayol bir OLGUYU tekrar ediyor: {k} {r_son}")
            if len(yol) == 2 and (eid[k], rid[yol[0]], rid[yol[1]]) in tut_zincir:
                atilan["zincir"] += 1
                continue
            out.append((eid[k], rid[r_son], eid[c]))
    return tuple(out), atilan

@dc.dataclass
# ======================================================================
# BOLMELER -- adlar EZBER / CIKARIM.  Kullanici, 22 Eylul:
#   "one / comp / ent gibi kavramlari bir turlu kaldiramadik.
#    onun yerine ezber cikarim gelmesi gerekiyor."
#
# Ayrim TEK SORUDA: cevap korpusta YAZIYOR MU?
#   yaziyor  -> EZBER    (bulup getirecek)
#   yazmiyor -> CIKARIM  (birlestirecek)
#
#   ezber_olgu            (one)      tek olgu, tek cumlede ogretildi
#   ezber_zincir          (tr2)      bilesim, egitimde GORULDU
#   cikarim_gorulmemis    (comp)     o (varlik,r1,r2) uclusu korpusta YOK
#   cikarim_yabanci       (ent)      varlik HIC zincir basi olmamis
#
#   kapi_kisayolsuz       (ent_yok)  YETENEK OLCMEZ -- kisayol TIP OLARAK
#                                    imkansiz.  Kirmizi cizgi: burada
#                                    kisayol sayiliyorsa OLCUM BOZUKTUR.
#   ayrik_arama           (ent_arama)  HUKUM VERMEZ
#   ayrik_ood             (ood)        HUKUM VERMEZ -- 80 ornek, cozunurluksuz
#   ayrik_kati            (ent_kati)   HUKUM VERMEZ
# ======================================================================
class Veri:
    facts: np.ndarray          # (n_ent, n_rel)  -1 = olgu YOK
    ezber_olgu: list                  # (e, r, hedef)              1hop
    ezber_zincir: list                  # (e, r1, r2, kopru, cevap)  egitim 2hop
    cikarim_gorulmemis: list                 #  ayni   -- egitimde GORULMEMIS (e,r1,r2) UCLUSU
    #   !! DIKKAT (15 Eylul hakemligi): buraya ve CLAUDE.md'ye "gorulmemis
    #   r1-r2 CIFTI" yazilmisti. OLCULDU, YANLIS: cikarim_gorulmemis'un 4281 orneginin
    #   4281'inin (r1,r2) cifti egitimde de var (115 ciftin 115'i). Bolme
    #   kodu ZINCIR tutuyor, CIFT tutmuyor. Yani cikarim_gorulmemis "bu iliski ciftini hic
    #   gormedi" demek DEGIL, "bu varligin bu ciftle zincirini gormedi"
    #   demek -- cok daha zayif bir genelleme sinavi. Kod degismedi
    #   (arsivdeki build_data_dis ile ayni), ETIKET duzeltildi.
    #   ILISKI-CIFTI genellemesini olcen bir bolme HENUZ YOK.
    cikarim_yabanci: list                  #  ayni   -- varlik hic zincir basi olmamis (HUKUM)
    kapi_kisayolsuz: list              #  ayni   -- kisayol TIP OLARAK imkansiz
    ayrik_arama: list            #  ayni   -- maske aramasi icin, HUKUMDEN AYRIK
    n_ent: int = 0
    n_rel: int = 0
    # TIP: her varligin tipi (KISI/OKUL/SEHIR/DERS). TANI icin gerekli --
    # "model YANLIS cevap verirken hic olmazsa DOGRU TIPTE bir sey mi
    # soyluyor?" sorusu bunsuz sorulamaz. `veri_kur` doldurur; tani_a
    # kendi basina TURETMEZ (turetirse iki dosya ayri siralama kurar).
    tip: np.ndarray | None = None      # (n_ent,) tip indeksi
    tip_ad: tuple = ()                 # tip indeksi -> ad
    ayrik_ood: list = dc.field(default_factory=list)
    #   WANG'IN test_inferred_OOD'si: zincirin IKI kenari da atomic_OOD.
    #   Yani ne birinci hop ne ikinci hop, egitimdeki HICBIR zincirde
    #   gecmiyor. Bas varlik ise BASKA kenarlariyla egitimde zincir basi
    #   OLMUS (Wang'da da oyle) -- tutulan sey VARLIK degil KENAR.
    #   ayrik_ood_pay=0 ise BOS kalir.
    ayrik_kati: list = dc.field(default_factory=list)
    #   varlik HICBIR egitim zincirinde gecmemis -- ne bas, ne kopru, ne
    #   cevap. Yalniz atomik olgularda var. Wang 2405.15071'in OOD'si bu;
    #   orada transformer %0 aliyor (22 milyon adimda bile). `cikarim_yabanci`ten FARKI:
    #   ENT varliklari kopru ve cevap olarak egitimde GORUNUYOR.
    #   kati_pay=0 ise BOS kalir ve hicbir sey degismez.

    par: np.ndarray | None = None      # (n_ent, 2) jeton ciftleri
    #   jeton_ad=False ise None ve hicbir sey degismez.
    par_ad: tuple = ()                 # (yuva1 adlari, yuva2 adlari)
    t_len: int = 0                     # veri_kur doldurur (ayar.t_len)
    ek_kip: str = ""                   # "" | "tr" | "ezber_zincir"  (ayar.ek_kip)
    soru_ad: tuple = ()                # ek_kip="ezber_zincir": SORU SOZCUKLERI,
    #   veri modulunun `SORU_SOZ`undan, SIRALI (determinizm).
    soru_tip: tuple = ()               # tip indeksi -> soru_ad indeksi.
    dir_soru: tuple = ()               # soru sozcugu -> ek_dir_ad indeksi
    #   ("kimdir", "neresidir") -- kimlik satiri icin.
    #   Motor tip ADLARINI BILMEZ; esleme VERI MODULUNDEN gelir.
    # --- EK ALLOMORFLARI (ek_kip="ezber_zincir") ---------------------------------
    # Turkce'de tamlayan eki sekiz bicimde: -in/-in/-un/-un ve sesliden
    # sonra -nin/... Ilk surumde TEK <NIN> jetonu hepsini ortuyordu;
    # kullanici fark etti (17 Eylul). Hangi bicimin gelecegi ONCEKI
    # KELIMEDEN belirli, yani YENI BILGI DEGIL -- dilin yuzeyi.
    # Unlu uyumunu VERI MODULU hesaplar, motor yalniz INDEKS tasir.
    ek_nin_ad: tuple = ()              # ("in","in","un",... ) 8 bicim
    ek_dir_ad: tuple = ()              # ("dir","dir","dur",...) 8 bicim
    nin_ent: tuple = ()                # varlik -> ek_nin_ad indeksi
    dir_ent: tuple = ()                # varlik -> ek_dir_ad indeksi
    nin_rel: tuple = ()                # iliski -> ek_nin_ad indeksi
    dir_rel: tuple = ()                # iliski -> ek_dir_ad indeksi
    kisayol: tuple = ()                # (e, r_son, cevap) IYELIK KISA YOLU
    kisayol_atilan: tuple = ()         # ((sebep, sayi), ...) URETILMEYEN satirlar
    #   `ezber_olgu` ile AYNI BICIMDE ama OLGU DEGIL: cevap `bolumu ->
    #   fakultesi -> ...` yolunun sonu. `facts`a GIRMEZ, `zincirler`
    #   gormez, `olcme_izi` DEGISMEZ.
    #  ^ model_06: 'annesi' -> 'annesidir'. YUKLEM ONE ALINMIS yuzey
    #    bicimleri icin; rolu KONUM degil EK tasisin diye.

    def __post_init__(self):
        self.n_ent, self.n_rel = self.facts.shape
        self.ent_off = SPECIAL + self.n_rel
        if self.par is None:
            self.yuva, self.vocab = 1, self.ent_off + self.n_ent
            self.p1_off = self.p2_off = self.ent_off
            self.n1 = self.n2 = self.n_ent
            # `yuva_ara` BURADA DA kurulur. Yoksa `dogruluk()` tek jetonlu
            # kollarda AttributeError ile duserdi -- 16 Eylul, model_b6
            # yamasinda gozden kacti, duman testinde yakalandi. Tek kaynak
            # olsun diye TEK ELEMANLI liste; sart yazmaya gerek kalmiyor.
            self.paylasilan = True
            self.yuva_ara = [(self.ent_off, self.vocab)]
        else:
            # IKI AYRIK BLOK -> yuva basina KISITLI argmax temiz kalir.
            self.yuva = self.par.shape[1]
            # PAYLASILAN sozluk ("tam") -> butun yuvalar AYNI blok.
            # AYRIK sozluk ("ilk")     -> yuva basina ayri blok.
            _ayni = all(x is self.par_ad[0] or x == self.par_ad[0]
                        for x in self.par_ad)
            self.paylasilan = _ayni
            if _ayni:
                n = len(self.par_ad[0])
                self.yuva_ara = [(self.ent_off, self.ent_off + n)] * self.yuva
                self.vocab = self.ent_off + n
            else:
                o, self.yuva_ara = self.ent_off, []
                for ad_ in self.par_ad:
                    self.yuva_ara.append((o, o + len(ad_))); o += len(ad_)
                self.vocab = o
            (self.p1_off, _h1) = self.yuva_ara[0]
            (self.p2_off, _h2) = self.yuva_ara[min(1, self.yuva - 1)]
            self.n1, self.n2 = _h1 - self.p1_off, _h2 - self.p2_off
        # EK ISARETLEYICILERI SOZLUGUN SONUNA. Boylece REL_OFF, ent_off
        # ve yuva_ara HIC KAYMAZ -- ek_kip kapaliyken uretilen diziler
        # BIT AYNI kalir, eski kosular gecerliligini korur.
        self.ek0 = 0
        if self.ek_kip:
            assert self.ek_kip == "ezber_zincir", (
                f"eksiz ve 'tr' kodlama yollari SILINDI: {self.ek_kip!r}")
            self.ek0 = self.vocab
            if True:
                # '  + NIN allomorflari + DIR allomorflari + SORU SOZCUKLERI
                assert self.soru_ad, "ezber_zincir SORU_SOZ ister -- veri modulu vermeli"
                assert len(self.soru_tip) == len(self.tip_ad), (
                    "soru_tip her TIP icin bir deger tasimali")
                assert self.ek_nin_ad and self.ek_dir_ad, (
                    "ezber_zincir EK ALLOMORFLARINI ister -- veri modulu vermeli")
                assert len(self.nin_ent) == len(self.dir_ent) == self.n_ent, (
                    "her VARLIK icin ek bicimi belli olmali")
                assert len(self.nin_rel) == self.n_rel, (
                    "her ILISKI icin ek bicimi belli olmali")
                self.nin0 = self.ek0 + 1
                self.dir0 = self.nin0 + len(self.ek_nin_ad)
                self.soru0 = self.dir0 + len(self.ek_dir_ad)
                self.vocab += (1 + len(self.ek_nin_ad) + len(self.ek_dir_ad)
                               + len(self.soru_ad))
                # BOSLUK DOLDURMA jetonlari -- SOZLUGUN EN SONUNA, ki
                # REL_OFF / ent_off / yuva_ara HIC KAYMASIN.
                #   <BOS>   cumleden CIKARILAN parcanin yerini tutar
                #   <AYIR>  cumle bitti, simdi o parca geliyor
                # Kullanici karari, 17 Eylul: *"Fatma ..... 'in annesi Ayse
                # Yilmaz'dir gibi"* -- yani ayni cumlede HER parca
                # sorulabilmeli. Model nedensel oldugu icin bosluk yerinde
                # sorulamaz (sagini goremez); parca SONA tasinir.
                self.bosluk = self.vocab
                self.ayir = self.vocab + 1
                # VIRGUL -- kullanici karari, 17 Eylul: *"burda , kavrami
                # devreye giriyor... kardesi'dir Ozlem Yilmaz, Ibrahim
                # Yilmaz'in"*.
                #
                # GEREKCESI OLCULDU: devrik bicimde cevap ile ozne YAN YANA
                # iki ad oluyor ve aralarinda HIC isaret yoktu --
                # "Kardesidir Ozlem Yilmaz Ibrahim Yilmaz'in": cevap
                # "Ozlem" mi "Ozlem Yilmaz" mi, dizide bunu soyleyen bir
                # sey yok. `havuz_06` yakaladi: 3000 satirin 300'unde
                # next-token sozlesmesi dusuyordu. Diger biciminde sinir
                # zaten isaretli ('dir / 'in); burada VIRGUL isaretliyor.
                self.virgul = self.vocab + 2
                self.vocab += 3
        # --- DEGISKEN UZUNLUKLU AD (17 Eylul, kullanici: "<YOK> sil,
        # gereksiz"). Once her varlik TAM `yuva` jeton kapliyordu ve kisa
        # adlar <YOK> ile SAGDAN dolduruluyordu -- havuzdaki butun
        # jetonlarin %11,5'i dolguydu ve "Adana <YOK> <YOK>" Turkce
        # DEGILDI. Artik ad kac kelimeyse o kadar jeton.
        #
        # SINIR ZATEN ISARETLI: her adin ardindan KESME ISARETI geliyor
        # ("Adana'nin"). Yani dolguya gerek yok -- model adi bitirip `'`
        # demeyi ogrenmek zorunda, ki bu DAHA GERCEK bir dil gorevi.
        if self.par is not None and self.ek_kip == "ezber_zincir":
            self.n_yuva = tuple(
                int((self.par[e] >= 0).sum()) for e in range(self.n_ent))
            assert min(self.n_yuva) >= 1, "her adin en az bir kelimesi olmali"
            # CEVAP pozisyonlarinda izin verilen kume: VARLIK kelimeleri
            # ARTI kesme isareti. Ikisi BITISIK (ent_off .. ek0 .. ek0+1),
            # yani tek aralik yetiyor.
            self.cevap_ara = (self.ent_off, self.ek0 + 1)
            self.cevap_yuva = self.yuva + 1      # en uzun ad + `'`
        if not self.t_len:
            self.t_len = 11 if self.par is not None else T_LEN

        # phi: TURETILMIS TANI SAYISI, kontrol parametresi DEGIL. Ayarlanamaz;
        # graf yogunlugundan ve yabanci_pay/cikarim_pay'den duser. "phi'yi 7 yapalim"
        # denemez -- veri ureticisi degistirilir.
        self._phi_hesapla()

    def _phi_hesapla(self):
        """phi + wang_phi. `ezber_zincir` DEGISIRSE yeniden cagrilir.

        !! METOT, cunku `zincir_butcesi` `ezber_zincir`yi BUDUYOR. Hesap
        `__post_init__` icinde tek satir kalsaydi `v.phi` budama ONCESI
        degeri tasirdi: rapor 4.00 basardi, model 1.12 gorurken.

        phi TURETILMIS bir tani sayisidir, kontrol parametresi DEGIL --
        graf yogunlugundan ve yabanci_pay/cikarim_pay'den duser.

        WANG'IN TANIMI AYNI DEGIL (15 Eylul hakemligi). Wang
        2405.15071:155 "phi = |train_inferredID| / |atomicID|" ve
        atomicID, OOD varliklarinin olgularini DISLAR; bizim paydamiz
        TUM olgular. Olculdu: ayni veride bizimki 5.09, Wang tanimiyla
        6.36 -- %25 fark. Wang'in 3.6-18.0 taramasina konumlanirken
        WANG_PHI kullanilmali, phi degil. KATI varliklari paydadan
        duser -- Wang'in OOD'si tam olarak o."""
        self.phi = len(self.ezber_zincir) / max(1, len(self.ezber_olgu))
        _ent = ({e for e, *_ in self.cikarim_yabanci}
                | {e for e, *_ in self.kapi_kisayolsuz}
                | {e for e, *_ in self.ayrik_arama}
                | {e for e, *_ in self.ayrik_kati})
        _id = sum(1 for e, _, _ in self.ezber_olgu if e not in _ent)
        self.wang_phi = len(self.ezber_zincir) / max(1, _id)

def veri_kur(ayar: Ayar, yaz=print) -> Veri:
    """Okul grafi -> bolmeler.  Arsivdeki build_data_dis ile AYNI mantik.

    Tohum `ayar.veri_tohum`; butun kollarda AYNI olmali, yoksa kollar farkli
    veri gorur ve 'sartlar esit' bozulur."""
    # KUSUR (15 Eylul hakemligi): burada `VO.kur()` yaziyordu, yani graf
    # HER ZAMAN tohum 0 ile uretiliyordu. `ayar.veri_tohum` yalniz BOLMEYI
    # etkiliyordu. Olculdu: veri_tohum 0 ve 1 ayni `facts`, farkli `ezber_zincir`.
    # Yani "veri tohumunu degistirdim" diyen biri grafin degismedigini
    # FARK ETMEZDI.
    # HANGI GRAF -- modul adi AYARDAN geliyor, sabit degil. "veri_okul"
    # (1060 varlik) ya da "veri_okul2" (2120). Modul `kur`/`zincirler`/
    # `TIPLER`/`ILISKI` sozlesmesini saglamak zorunda; saglamazsa burada
    # AttributeError verir, sessizce yanlis veri kurmaz.
    _V = importlib.import_module(ayar.veri_ad)
    for _g in ("kur", "zincirler", "TIPLER", "ILISKI"):
        assert hasattr(_V, _g), f"{ayar.veri_ad} modulunde {_g} yok"

    G = _V.kur(ayar.veri_tohum)
    zin = _V.zincirler(G)
    E = [a for t in _V.TIPLER for a in G["ad"][t]]

    # --- IKI JETONLU KODLAMA (ayar.jeton_ad) ----------------------------
    # ILK alt cizgiden bolunur, cunku anlamli olan o:
    #   Ayse_Yilmaz       -> (Ayse, Yilmaz)        ad + soyad
    #   Ankara_Fen_Lisesi -> (Ankara, Fen_Lisesi)  sehir + tur
    #   Ankara            -> (Ankara, <YOK>)
    # Soyadi PAYLASIMI zaten var (veri_okul: "cocuk/kardes/anne/baba AYNI
    # soyadi tasir"), yani hicbir jeton TEK BASINA kisiyi belirlemiyor.
    _par = _par_ad = None
    if ayar.jeton_ad == "tam":
        # DOGRU KODLAMA: butun alt cizgiler, TEK PAYLASILAN sozluk.
        # !! `<YOK>` SOZLUKTEN CIKTI (17 Eylul, kullanici: "sil,
        # gereksiz"). Kisa adlar artik SAGDAN DOLDURULMUYOR; `par`in
        # bos yuvalari -1 tasiyor ve `_e` onlari HIC uretmiyor.
        # Sinir zaten KESME ISARETI ile isaretli ("Adana'nin").
        _yuva = max(len(a.split("_")) for a in E)
        _pl = sorted({p for a in E for p in a.split("_")})
        _ix = {p: i for i, p in enumerate(_pl)}
        _par = np.array([[(_ix[p] if p is not None else -1) for p in
                          (a.split("_") + [None] * _yuva)[:_yuva]]
                         for a in E], np.int64)
        _sz = tuple(_pl)
        _par_ad = (_sz,) * _yuva                         # AYNI sozluk, her yuva
        assert not any("_" in p for p in _pl), "jeton icinde ALT CIZGI kaldi"
        assert len({tuple(r) for r in _par}) == len(E),             "BIREBIR DEGIL -- ayni jeton dizisi birden cok varliga denk"
        _coz = lambda r: "_".join(_sz[i] for i in r if i >= 0)
        _kt = [(E[i], _coz(_par[i])) for i in range(len(E))
               if _coz(_par[i]) != E[i]]
        assert not _kt, f"GIDIS-DONUS BOZUK: {_kt[:3]}"
        yaz(f"  jeton_ad=tam: {len(E)} varlik -> {_yuva} yuva, "
            f"PAYLASILAN sozluk {len(_sz)} jeton "
            f"(tek jetonda {len(E)} idi)")
    elif ayar.jeton_ad == "ilk":
        _ik = [(a.split("_", 1) + ["<YOK>"])[:2] for a in E]
        _y1 = sorted({p[0] for p in _ik})
        _y2 = sorted({p[1] for p in _ik})
        _i1 = {a: i for i, a in enumerate(_y1)}
        _i2 = {a: i for i, a in enumerate(_y2)}
        _par = np.array([[_i1[p[0]], _i2[p[1]]] for p in _ik], np.int64)
        _par_ad = (tuple(_y1), tuple(_y2))
        # BIREBIRLIK: iki jeton bir varligi TEK SEKILDE belirlemeli, yoksa
        # "dogru cevap" tanimsiz olurdu.
        assert len({tuple(p) for p in _par}) == len(E), \
            "IKI JETON birebir DEGIL -- ayni cift birden cok varliga denk"
        yaz(f"  jeton_ad ACIK: {len(E)} varlik -> yuva1 {len(_y1)}  "
            f"yuva2 {len(_y2)}  (tek jetonda {len(E)} idi)")
    # E, TIP SIRASIYLA kuruluyor -- tip dizisi AYNI comprehension'dan
    # cikarilir ki iki yerde iki siralama olmasin.
    E_tip = np.array([i for i, t in enumerate(_V.TIPLER)
                      for _ in G["ad"][t]], np.int64)
    R = list(_V.ILISKI)
    eid = {a: i for i, a in enumerate(E)}
    rid = {r: i for i, r in enumerate(R)}

    facts = np.full((len(E), len(R)), -1, np.int64)
    for (e, r), h in G["olgu"].items():
        facts[eid[e], rid[r]] = eid[h]

    bas = {}
    for x in zin:
        bas.setdefault(x[0], []).append(x)

    rng = np.random.RandomState(ayar.veri_tohum)
    ent_ad = set()
    for t in _V.TIPLER:                       # TABAKALI: tek tip secilirse
        a = [x for x in G["ad"][t] if x in bas]   # sinav o tipin karisimina
        if a:                                     # indirgenir
            k = int(round(len(a) * ayar.yabanci_pay))
            ent_ad |= {a[int(i)] for i in rng.permutation(len(a))[:k]}

    # KATI GRUBU -- ENT'ten AYRIK secilir. Bu varliklar egitim zincirinde
    # HICBIR ROLDE gorunmeyecek (bas, kopru, cevap). ENT ise yalniz bas
    # olmuyor. Tabakalama ENT ile AYNI: tek tip secilirse sinav o tipin
    # karisimina indirgenir.
    kati_ad = set()
    if ayar.kati_pay > 0:
        for t in _V.TIPLER:
            a = [x for x in G["ad"][t] if x in bas and x not in ent_ad]
            if a:
                k = int(round(len(a) * ayar.kati_pay))
                kati_ad |= {a[int(i)] for i in rng.permutation(len(a))[:k]}
        assert not (kati_ad & ent_ad), "KATI ve ENT gruplari ORTUSUYOR"

    # ARAMA / HUKUM ayrimi VARLIK duzeyinde: ayni varligin baska bir zinciri
    # de sizinti sayilir. Olculdu: ayrim olmadan hukum kumesinin %24'u
    # aramada zaten gorulmustu.
    _ea = sorted(ent_ad)
    _k = max(1, int(round(len(_ea) * ayar.arama_pay)))
    _ix = rng.permutation(len(_ea))
    arama_ad = {_ea[int(i)] for i in _ix[:_k]}
    hukum_ad = ent_ad - arama_ad

    # OOD KENARLARI -- rng'den EN SON cekilir. Onceki cekilislerin
    # (ent_ad, kati_ad, arama_ad) sirasi DEGISMEMELI; degisirse ayrik_ood_pay=0
    # olsa bile eski kollarin verisi kayar. `test_sabit` bunu dogruluyor.
    ood_k = set()
    if ayar.ayrik_ood_pay > 0:
        _kn = [(e, r) for e in E for r in R if facts[eid[e], rid[r]] >= 0]
        _k = int(round(len(_kn) * ayar.ayrik_ood_pay))
        ood_k = {_kn[int(i)] for i in rng.permutation(len(_kn))[:_k]}

    def _ood(x):
        """Zincirin kac kenari atomic_OOD'de?  x = (e, r1, r2, kopru, cevap)"""
        return ((x[0], x[1]) in ood_k) + ((x[3], x[2]) in ood_k)

    ezber_zincir, cikarim_gorulmemis, ent_ay, ent_yk, ent_ar, ent_kt, ood_ay = [], [], [], [], [], [], []
    for e in E:                                # E sirasi SABIT -> tekrarlanabilir
        lst = bas.get(e)
        if not lst:
            continue
        if e in kati_ad:                       # HICBIR ROLDE egitimde yok
            for x in lst:
                if x[6] in ("AYIRT", "YOK") and _ood(x) == 0:
                    ent_kt.append(x)
            continue                           # AYNI / DONUS: duser
        if e in ent_ad:
            hedef = ent_ay if e in hukum_ad else ent_ar
            for x in lst:
                if _ood(x):                    # OOD kenarli: ENT'e girmez
                    continue
                if x[6] == "AYIRT":
                    hedef.append(x)
                elif x[6] == "YOK" and e in hukum_ad:
                    ent_yk.append(x)
            continue                           # AYNI / DONUS: duser
        # WANG'IN UC YOLU (§2, §3.1):
        #   iki kenar da OOD -> test_inferred_OOD
        #   bir kenar OOD    -> KARISIK: ne egitim ne sinav, DUSER
        #   iki kenar da ID  -> normal (ezber_zincir / cikarim_gorulmemis)
        # `lst_id` ayrik_ood_pay=0 iken `lst`in KENDISI olur ve rng akisi
        # birebir korunur -- eski kollarin verisi degismez.
        lst_id = []
        for x in lst:
            d = _ood(x)
            if d == 2:
                if x[6] in ("AYIRT", "YOK"):
                    ood_ay.append(x)
            elif d == 0:
                lst_id.append(x)
        p = rng.permutation(len(lst_id))
        k = int(round(len(lst_id) * (1.0 - ayar.cikarim_pay)))
        for i, j in enumerate(p):
            x = lst_id[int(j)]
            if i < k:
                ezber_zincir.append(x)
            elif x[6] in ("AYIRT", "YOK"):
                cikarim_gorulmemis.append(x)                 # AYNI/DONUS sinava girmez

    # KATI varligi KOPRU ya da CEVAP olarak da gecmemeli -- bolmenin
    # tanimi "hicbir rolde yok". Bas konumunu yukaridaki `continue`
    # hallediyor; kalan iki konum BURADA siliniyor.
    # OLCULDU (15 Eylul): kati_pay=0.05'te bu, egitim zincirlerinin
    # %9,3'unu goturuyor. YERINE KOYULMUYOR -- bu kolun kiyasi KOSU ICI
    # (cikarim_gorulmemis vs cikarim_yabanci vs ayrik_kati, ayni model, ayni adim, ayni phi), o yuzden
    # phi'nin baska kollarla eslesmesi GEREKMIYOR. Dolgu zincir eklemek
    # egitim dagilimini bozardi (elde kalan havuz AYNI/DONUS turunden,
    # yani TRIVIAL zincirler).
    if kati_ad:
        _n0 = len(ezber_zincir)
        ezber_zincir = [x for x in ezber_zincir if x[3] not in kati_ad and x[4] not in kati_ad]
        # COMP ve ENT sinavlari da KATI'den arindirilir: yoksa o orneklerin
        # bir kismi gizliden ayrik_kati olur ve UC GRUBUN KARSITLIGI bulanir.
        cikarim_gorulmemis = [x for x in cikarim_gorulmemis if x[3] not in kati_ad and x[4] not in kati_ad]
        ent_ay = [x for x in ent_ay if x[3] not in kati_ad and x[4] not in kati_ad]
        ent_yk = [x for x in ent_yk if x[3] not in kati_ad and x[4] not in kati_ad]
        ent_ar = [x for x in ent_ar if x[3] not in kati_ad and x[4] not in kati_ad]
        yaz(f"  KATI: {len(kati_ad)} varlik hicbir zincirde yok. "
            f"egitim zinciri {_n0} -> {len(ezber_zincir)} "
            f"(-{_n0 - len(ezber_zincir)}, %{100*(_n0-len(ezber_zincir))/max(1,_n0):.1f})")

    # --- SORU SOZCUKLERI (ek_kip="ezber_zincir") -- VERI MODULUNDEN --------------
    # Motor tip ADLARINI bilmez: "KISI'ye kim denir"i veri modulu soyler.
    # Sirali, cunku jeton id'leri buradan duser ve DETERMINIST olmali.
    _soru_ad, _soru_tip = (), ()
    _ek_nin_ad = _ek_dir_ad = _nin_ent = _dir_ent = _nin_rel = ()
    _dir_rel = ()
    _dir_soru = ()
    if ayar.ek_kip == "ezber_zincir":
        _SS = getattr(_V, "SORU_SOZ", None)
        assert isinstance(_SS, dict) and set(_SS) == set(_V.TIPLER), (
            f"{ayar.veri_ad}.SORU_SOZ her TIP icin bir soru sozcugu "
            f"vermeli (ek_kip='ezber_zincir'): {_SS}")
        _soru_ad = tuple(sorted(set(_SS.values())))
        _soru_tip = tuple(_soru_ad.index(_SS[t]) for t in _V.TIPLER)
        _ES = _V.ek_secim(G)
        _ek_nin_ad, _ek_dir_ad = tuple(_V.EK_NIN), tuple(_V.EK_DIR)
        _nin_ent = tuple(_ES["nin_varlik"][a] for a in E)
        _dir_ent = tuple(_ES["dir_varlik"][a] for a in E)
        _nin_rel = tuple(_ES["nin_iliski"][r] for r in _V.ILISKI)
        _dir_rel = tuple(_ES["dir_iliski"][r] for r in _V.ILISKI)
        _dir_soru = tuple(_ES["dir_soru"][w] for w in _soru_ad)

    say = lambda L: [(eid[x[0]], rid[x[1]], rid[x[2]], eid[x[3]], eid[x[4]])
                     for x in L]
    # IYELIK KISA YOLU icin YASAK ucluler: butun SINAV bolmeleri.
    # --- IYELIK KISA YOLU: UC TUTMA KAPISI (bkz. `_kisayol_kur`)
    # !! `ent_kt` (KATI) de listede. Bu kolda bos (kati_pay=0) ama
    # `olcme_listeleri` doluysa onu da olcuyor -- dusurulseydi
    # kati_pay>0 olan bir kolda SESSIZ sizinti olurdu.
    _ksy_bas = {eid[x[0]] for L in (ent_ay, ent_yk, ent_ar, ent_kt) for x in L}
    _ksy_kenar = {(eid[a], rid[b]) for a, b in ood_k}
    _ksy_zincir = {(eid[x[0]], rid[x[1]], rid[x[2]])
                   for L in (cikarim_gorulmemis, ent_ay, ent_yk, ent_ar, ood_ay, ent_kt)
                   for x in L}
    _ksy, _ksy_at = _kisayol_kur(G, eid, rid, _ksy_bas, _ksy_kenar, _ksy_zincir)
    v = Veri(facts=facts, ezber_olgu=[(eid[e], rid[r], eid[h])
                               for (e, r), h in G["olgu"].items()],
             ezber_zincir=say(ezber_zincir), cikarim_gorulmemis=say(cikarim_gorulmemis), cikarim_yabanci=say(ent_ay),
             kapi_kisayolsuz=say(ent_yk), ayrik_arama=say(ent_ar),
             tip=E_tip, tip_ad=tuple(_V.TIPLER), ayrik_kati=say(ent_kt),
             ayrik_ood=say(ood_ay), par=_par, par_ad=_par_ad or (),
             t_len=ayar.t_len, ek_kip=ayar.ek_kip,
             soru_ad=_soru_ad, soru_tip=_soru_tip,
             ek_nin_ad=_ek_nin_ad, ek_dir_ad=_ek_dir_ad,
             nin_ent=_nin_ent, dir_ent=_dir_ent, nin_rel=_nin_rel,
             dir_rel=_dir_rel,
             dir_soru=_dir_soru,
             kisayol=_ksy, kisayol_atilan=tuple(sorted(_ksy_at.items())))

    # --- ZINCIR BUTCESI: `ezber_zincir` KORPUSUN YAZABILECEGI KADAR -------------
    # OLCULDU 19 Eylul: ezber_zincir 29.510 zincir tasiyordu, korpus 6.819 tanesini
    # yazabiliyordu. `zincir` sinavi TAMAMINDAN soruyordu -> tavan 0.23,
    # esik 0.95. Olcu kendi adini yalanliyordu ("gordugu bilesigi
    # hatirliyor mu" -- gormedigini soruyordu).
    #
    # !! BURADA, `veri_kur`in ICINDE. Cunku iki okuma yolu var ve ikisi
    # de once BURAYA ugruyor:
    #     egit()      : egitim_havuzu -> olcme_listeleri
    #     pencere_11  : olcme_listeleri -> egitim_havuzu   (TERS SIRA)
    # Budama `havuz`da yapilsaydi pencere BUDANMAMIS ezber_zincir ile sinav
    # kurardi ve iki yol FARKLI `zincir` olcerdi -- sessizce.
    if ayar.zincir_pay:
        v.ezber_zincir = KOR.zincir_butcesi(
            v.ezber_olgu, v.ezber_zincir, KOR.kopru_yasagi(v), ayar.kopya,
            ayar.zincir_pay, tohum=ayar.veri_tohum, yaz=yaz)
        v._phi_hesapla()          # ezber_zincir degisti -> phi/wang_phi DE degisir

    # --- SIZINTI DENETIMI -- sessiz gecmesin
    trset = {(e, a, b) for e, a, b, _, _ in v.ezber_zincir}
    for nm, st in (("COMP", v.cikarim_gorulmemis), ("ENT", v.cikarim_yabanci), ("ENT-YOK", v.kapi_kisayolsuz)):
        k = sum((e, a, b) in trset for e, a, b, _, _ in st)
        assert k == 0, f"{nm} sizintisi: {k} ornek egitimde de var"
    # !! KISA YOL SATIRLARI UC KAPIDAN DA GECIRILIR. Iki adimli bir
    # kisayol ("X'in fakultesi = Y") tam olarak "X'in bolumunun
    # fakultesi?" sorusunun cevabidir; ayrica X TUTULAN bir varliksa
    # onu ZINCIR BASI yapar, ve yol bir OOD kenarindan geciyorsa o
    # kenari egitime geri sokar. Ucu de OLCULDU ve UCU DE OLDU
    # (18 Eylul): sirasiyla cikarim_gorulmemis'ta 28 zincir, 256 tutulan-bas satiri,
    # 55 OOD satiri.
    _ksy_yol = {r: yol for r, yol in KISAYOL_YOLU}
    _sinav3 = {(x[0], x[1], x[2]) for L in
               (v.cikarim_gorulmemis, v.cikarim_yabanci, v.kapi_kisayolsuz, v.ayrik_arama, v.ayrik_ood, v.ayrik_kati)
               for x in L}
    _sz = []
    for e, r, a in v.kisayol:
        _y = _ksy_yol[_V.ILISKI[r]]
        if e in _ksy_bas:
            _sz.append(("TUTULAN BAS", e, _V.ILISKI[r]))
        if len(_y) == 2 and (e, rid[_y[0]], rid[_y[1]]) in _sinav3:
            _sz.append(("SINAV ZINCIRI", e, _V.ILISKI[r]))
        _c = e
        for _rr in _y:
            if (_c, rid[_rr]) in _ksy_kenar:
                _sz.append(("OOD KENARI", e, _V.ILISKI[r]))
                break
            _c = int(facts[_c, rid[_rr]])
            if _c < 0:
                break
    assert not _sz, f"KISA YOL sizintisi: {len(_sz)} satir -- {_sz[:5]}"
    _ent_id = {eid[a] for a in ent_ad}
    k = sum(e in _ent_id for e, *_ in v.ezber_zincir)
    assert k == 0, f"ENT varligi egitimde ZINCIR BASI olmus: {k}"
    _h = {e for e, *_ in v.cikarim_yabanci} | {e for e, *_ in v.kapi_kisayolsuz}
    _a = {e for e, *_ in v.ayrik_arama}
    assert not (_h & _a), f"ARAMA/HUKUM varlik sizintisi: {len(_h & _a)}"
    # KATI: BOLMENIN TANIMI BU. Bas/kopru/cevap UC KONUMDA da denetlenir --
    # `cikarim_yabanci` icin yalniz bas denetleniyor, cunku orada kopru/cevap SERBEST.
    if kati_ad:
        _kid = {eid[a] for a in kati_ad}
        _k = sum(1 for e, _, _, b, c in v.ezber_zincir
                 if e in _kid or b in _kid or c in _kid)
        assert _k == 0, f"KATI varligi egitim zincirinde gecti: {_k}"
        assert v.ayrik_kati, "kati_pay > 0 ama ayrik_kati BOS"
        _kt = {e for e, *_ in v.ayrik_kati}
        assert _kt <= _kid, "ayrik_kati'de KATI olmayan varlik var"
        assert not (_kt & (_h | _a)), "KATI ile ENT gruplari ORTUSUYOR"
        # Atomik olgular DURMALI -- Wang da atomicOOD'yi egitimde tutuyor.
        _ko = sum(1 for e, _, _ in v.ezber_olgu if e in _kid)
        assert _ko > 0, "KATI varliklarinin atomik olgulari da silinmis"
        _sz = {(e, a, b) for e, a, b, _, _ in v.ezber_zincir}
        _l = sum((e, a, b) in _sz for e, a, b, _, _ in v.ayrik_kati)
        assert _l == 0, f"ENT-KATI sizintisi: {_l}"

    if ood_k:
        _ok = {(eid[e], rid[r]) for e, r in ood_k}
        # 1) OOD kenari HICBIR egitim zincirinde, HICBIR hop'ta gecmemeli
        _x = sum(1 for e, r1, r2, b, _ in v.ezber_zincir
                 if (e, r1) in _ok or (b, r2) in _ok)
        assert _x == 0, f"OOD kenari egitim zincirinde gecti: {_x}"
        # 2) SINAV zincirinin IKI kenari da OOD olmali
        _y = sum(1 for e, r1, r2, b, _ in v.ayrik_ood
                 if not ((e, r1) in _ok and (b, r2) in _ok))
        assert _y == 0, f"ayrik_ood bolmesinde iki kenari OOD olmayan: {_y}"
        # 3) ASIL MEKANIZMA (Wang §3.3): ikinci hop kenari egitimde IKINCI
        #    HOP olarak gecmemeli. `ayrik_kati`de bu %93,7 geciyordu -- yani
        #    o bolme makalenin %0'ini ureten kosulu SAGLAMIYORDU.
        _ik = {(b, r2) for _, _, r2, b, _ in v.ezber_zincir}
        _z = sum(1 for _, _, r2, b, _ in v.ayrik_ood if (b, r2) in _ik)
        assert _z == 0, f"OOD 2. hop kenari egitimde 2. HOP olarak gecti: {_z}"
        # 4) ATOMIK OLGULAR DURMALI (Wang §2: "all the atomic facts")
        _ao = sum(1 for e, r, _ in v.ezber_olgu if (e, r) in _ok)
        assert _ao == len(_ok), f"OOD atomik olgulari silinmis: {_ao}/{len(_ok)}"
        _bs = {e for e, *_ in v.ezber_zincir}
        _hb = sum(1 for e, *_ in v.ayrik_ood if e in _bs)
        yaz(f"  OOD: {len(_ok)} kenar atomic_OOD (%{100*ayar.ayrik_ood_pay:.1f}). "
            f"sinav {len(v.ayrik_ood)} zincir. "
            f"bas varlik egitimde BASKA zincirlerde bas olmus: "
            f"{_hb}/{len(v.ayrik_ood)} (%{100*_hb/max(1,len(v.ayrik_ood)):.0f})")
    yaz(f"  veri: olgu {len(v.ezber_olgu)}  egitim2 {len(v.ezber_zincir)}  COMP {len(v.cikarim_gorulmemis)}  "
        f"ENT {len(v.cikarim_yabanci)}  ENT-YOK {len(v.kapi_kisayolsuz)}  "
        f"ENT-ARAMA {len(v.ayrik_arama)}"
        + (f"  ENT-KATI {len(v.ayrik_kati)}" if v.ayrik_kati else "")
        + (f"  OOD {len(v.ayrik_ood)}" if v.ayrik_ood else "")
        + f"  phi {v.phi:.2f}")
    yaz(f"        n_ent {v.n_ent}  n_rel {v.n_rel}  vocab {v.vocab}  "
        f"ent_off {v.ent_off}")
    yaz(f"        phi {v.phi:.2f} (bizim tanim)   {v.wang_phi:.2f} (Wang tanimi, "
        f"payda atomicID)   -- TURETILMIS, ayar DEGIL")

    # BOLMELER NE KADAR YENI -- her kosuda BASILIR, bir daha etiket kaymasin.
    # 15 Eylul'de cikarim_gorulmemis'a "gorulmemis iliski cifti" deniyordu; olculunce
    # ciftlerin TAMAMI egitimde cikti. Sayi gozukurse iddia kayamaz.
    _tc = {(a, b) for _, a, b, _, _ in v.ezber_zincir}
    _tv = {e for e, *_ in v.ezber_zincir}
    for _ad, _L in (("COMP", v.cikarim_gorulmemis), ("ENT", v.cikarim_yabanci), ("ENT-YOK", v.kapi_kisayolsuz),
                    ("ENT-KATI", v.ayrik_kati), ("OOD", v.ayrik_ood)):
        if not _L:
            continue
        _yc = sum(1 for _, a, b, _, _ in _L if (a, b) not in _tc)
        _yv = sum(1 for e, *_ in _L if e not in _tv)
        yaz(f"        {_ad:<8} YENI olan: iliski-cifti {_yc}/{len(_L)}   "
            f"zincir-basi varlik {_yv}/{len(_L)}")
    yaz("        ^ COMP'ta cift YENI DEGIL: cikarim_gorulmemis 'gorulmemis UCLU', "
        "'gorulmemis CIFT' DEGIL.")
    return v

def olcme_listeleri(ayar: Ayar, v: Veri):
    """Her kumeden en fazla n_olcum_max ornek. TEK KAYNAK: egitim dongusu de
    pencere_a.py de BURAYI cagirir. Arsivde bu mantik iki dosyada AYRI AYRI
    duruyordu ve bir salt degisirse farkli ornek olculurdu, sessizce."""
    def alt(lst, salt):
        if not lst:
            return []
        r = np.random.RandomState(ayar.veri_tohum + 7 + salt)
        if len(lst) > ayar.n_olcum_max:
            return [lst[i] for i in r.permutation(len(lst))[:ayar.n_olcum_max]]
        return list(lst)
    d = dict(ezber_olgu=alt(v.ezber_olgu, 0), seen=alt(v.ezber_zincir, 1), cikarim_gorulmemis=alt(v.cikarim_gorulmemis, 2),
             cikarim_yabanci=alt(v.cikarim_yabanci, 3), kapi_kisayolsuz=alt(v.kapi_kisayolsuz, 5))
    # ANAHTAR YALNIZ DOLUYSA EKLENIR. Bos liste bile eklense `olcme_izi`
    # DEGISIR ve BUTUN eski kosular "iz tutmuyor" diye olculemez hale
    # gelirdi -- pencere_a egitim izi ile olcme izini karsilastiriyor.
    if v.ayrik_kati:
        d["ayrik_kati"] = alt(v.ayrik_kati, 6)
    if v.ayrik_ood:
        d["ayrik_ood"] = alt(v.ayrik_ood, 8)
    return d

def olcme_izi(L: dict) -> str:
    """Olcme setlerinin PARMAK IZI -- setin KENDISINI temsil eder.

    Dis hakemlik (15 Eylul) hakli cikti. Onceki hali sadece
    `(anahtar, uzunluk, ilk 2 ornek)` hash'liyordu ve ORTADAN degisen bir
    ornegi GORMUYORDU. Olculdu: cikarim_yabanci[1500] degistirildi, uzunluk ve ilk iki
    ornek ayni kaldi -> parmak izi BIREBIR AYNI cikti. Yani tam yakalamasi
    gereken seyi kaciriyordu.

    Simdi BAYT DUZEYINDE: her kumenin tamami int64 dizisine cevrilip
    ham baytlari hash'leniyor. `repr()`ten hem daha hizli hem tam
    deterministik (repr float/int gosterimine bagli degil).

    TEK KAYNAK: egit() de pencere_a.py de BURAYI cagirir. Ayni veri ve
    ayarla ayni izi vermeleri MEKANIK olarak dogrulanabilsin diye."""
    import hashlib
    h = hashlib.md5()
    for k in sorted(L):
        h.update(k.encode())
        arr = np.asarray(L[k], dtype=np.int64)
        h.update(repr(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()[:12]

_K = os.path.dirname(os.path.abspath(__file__))

ONBELLEK_YOL = os.environ.get(
    "KORPUS_ONBELLEK", os.path.join(os.path.dirname(_K), "onbellek"))

ONBELLEK_KAPALI = os.environ.get("KORPUS_ONBELLEK_KAPALI", "") == "1"

ONBELLEK_KAYNAK = ("korpus", "metin", "jeton", "veri")

def _onbellek_anahtari(ayar, v) -> str:
    """AYAR + VERI + KAYNAK KOD -> 16 haneli anahtar."""
    import hashlib
    h = hashlib.sha256()
    for k in ("kopya", "t_len", "tohum", "veri_tohum", "tetik", "zincir_pay",
              "n3", "ret_pay", "ret_tut", "veri_ad", "ek_kip", "tam_kayip"):
        h.update(f"{k}={getattr(ayar, k, None)!r};".encode())
    for ad in ("ezber_olgu", "ezber_zincir", "cikarim_gorulmemis", "cikarim_yabanci", "kapi_kisayolsuz", "ayrik_arama",
               "ayrik_ood", "ayrik_kati"):
        L = getattr(v, ad, None) or []
        h.update(f"{ad}={len(L)}:".encode())
        h.update(repr(L).encode())
    _kol = os.path.basename(_K).rsplit("_", 1)[-1]
    for _ad in ONBELLEK_KAYNAK:
        _f = os.path.join(_K, f"{_ad}_{_kol}.py")
        if _ad == "veri":
            _f = os.path.join(_K, ayar.veri_ad + ".py")
        with open(_f, "rb") as _fh:
            h.update(_fh.read())
    return h.hexdigest()[:16]

def _onbellek_oku(yol, yaz):
    """(X, S) ya da None. Bozuk dosya SESSIZCE atlanir, patlamaz."""
    try:
        z = np.load(yol, allow_pickle=False)
        X = z["X"].astype(np.int64)
        S = J.Sozluk("".join(str(z["harf"])))
        S.korpus_izi = str(z["korpus_izi"])
        yaz(f"  KORPUS ONBELLEKTEN: {os.path.basename(yol)}  "
            f"{X.shape[0]:,} x {X.shape[1]}  iz {S.korpus_izi}")
        return X, S
    except Exception as e:                       # pragma: no cover
        yaz(f"  onbellek OKUNAMADI ({e}) -- yeniden kurulacak")
        return None

def _onbellek_yaz(yol, X, S, yaz):
    """int16 olarak yazar: sozluk 64 sembol, int64 dort kat israf."""
    try:
        os.makedirs(os.path.dirname(yol), exist_ok=True)
        assert int(X.max()) < 32767, "int16 tasar -- sozluk buyumus"
        _gec = yol + ".gecici"
        np.savez(_gec, X=X.astype(np.int16),
                 harf="".join(S.harf), korpus_izi=S.korpus_izi)
        os.replace(_gec + ".npz", yol)           # ATOMIK: yarim dosya kalmaz
        yaz(f"  KORPUS ONBELLEGE YAZILDI: {os.path.basename(yol)}  "
            f"{os.path.getsize(yol)/1e6:.0f} MB")
    except Exception as e:                       # pragma: no cover
        yaz(f"  onbellege YAZILAMADI ({e}) -- kosu etkilenmez")

def egitim_havuzu(ayar: Ayar, v: Veri, yaz=print):
    """KORPUS -> egitim tensoru. model_08'in SATIR TABLOSU YOK.

    model_08'de havuz `kodla_1hop` / `kodla_2hop` / FIM / soru / kimlik
    dilimlerinden kuruluyordu ve her satir TEK olguydu. Burada havuz
    PAKETLENMIS METIN: belgeler birlestirilip `t_len`lik pencerelere
    bolunuyor (Physics of LM 3.1 Ek C duzeni). Bir pencere 7..11 cumle
    tasiyor ve hangi cumlenin nerede oldugu MODELE SOYLENMIYOR.

    Doner: (X, S). `P`/`T` (cevap yuvasi) YOK -- kayip butun
    pozisyonlarda next-token, yani `tam_kayip` artik SECENEK degil
    TEK yol."""
    assert ayar.tam_kayip, (
        "model_09 DIL MODELI kaybiyla egitilir. Cevap yuvasi diye bir sey "
        "yok: pencerede 7-11 cumle var ve hangisinin cevap oldugu "
        "isaretlenmiyor.")
    V09 = importlib.import_module(ayar.veri_ad)
    # !! ADLI CAGRI. Ilk surumde `yaz` POZISYONEL gecti ve `zincir_pay`
    # yuvasina dustu -- korpus sessizce `len(sat) * print` deneyip
    # patladi. Sirasi kayan bir cagri, patlamasaydi YANLIS ORANLA
    # korpus kurardi.
    # ONBELLEK: anahtar AYAR + VERI + KAYNAK KOD (bkz. yukarisi).
    _yol = None
    if not ONBELLEK_KAPALI:
        _yol = os.path.join(ONBELLEK_YOL,
                            f"korpus_{_onbellek_anahtari(ayar, v)}.npz")
        _var = _onbellek_oku(_yol, yaz) if os.path.exists(_yol) else None
        if _var is not None:
            return _var
    X, S = KOR.havuz(v, V09.kur(ayar.veri_tohum), kopya=ayar.kopya,
                     t_len=ayar.t_len, tohum=ayar.veri_tohum,
                     tetik=ayar.tetik, zincir_pay=ayar.zincir_pay,
                     n3=ayar.n3, ret_pay=ayar.ret_pay,
                     ret_tut=ayar.ret_tut, yaz=yaz)
    if _yol:
        _onbellek_yaz(_yol, X, S, yaz)
    return X, S
