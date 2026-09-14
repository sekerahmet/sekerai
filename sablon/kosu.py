"""KOSU — deneyden BAGIMSIZ kosu altyapisi.

Her deneyde ayni olan sey burada, bir kez:
  * adim adli surdurme paketi   -> gecmis bir adima DONULEBILIR (CLAUDE.md 7/9)
  * parca basina ayri log       -> "bu satir hangi adimdi" karisikligi yok
  * durum.json + konfig.json    -> kaldigi yerden devam, rapor icin girdi
  * yorunge kapisi              -> referans kolun ayni adimiyla kiyas
  * konfig kapisi               -> surdurme paketindeki cfg beklenen mi

Deneye OZEL karar mantigi (ne zaman dur, neye bak, ne sec) surucu dosyasinda.
"""
import os, sys, json, time, glob, shutil, subprocess


def _sorulmayan(c, bekle, muaf):
    """Kayitli cfg'de olup ne SORULAN ne MUAF olan anahtarlar.

    Ayri fonksiyon: Kosu ornegi kurmadan test edilebilsin (§6 test 3).
    """
    return sorted(k for k in c if k not in bekle and k not in muaf)


class Kosu:
    def __init__(self, kon):
        self.kon = kon
        self.C = kon["CALIS"]
        self.KOL, self.SEED, self.D = kon["KOL"], kon["SEED"], kon["DENEY"]
        # "cikti" BURADA yaratilmaz: kol klasorunu egit() kendi acar. Bos bir
        # cikti/ yaratmak coklu-kollu kosuda rapora hayalet satir dusuruyordu.
        for a in ("sur", "log", "ham"):
            os.makedirs(self.y(a), exist_ok=True)
        kon["commit"] = kon.get("commit", "?")
        # COMMIT KAPISI: yeni konfigi YAZMADAN once diskte duran eskiye bak.
        _eski = self._oku_json(self.y("konfig.json"))
        self.st = self._oku_json(self.y("durum.json"),
                                 dict(faz=1, adim=0, faz_ad="", rapor_ek=[]))
        self._commit_kapisi(_eski)
        self._yaz_json(self.y("konfig.json"), kon)

    def _commit_kapisi(self, eski):
        """Drive'dan geri yuklenen yarim is, BU commit'le mi uretilmis?

        HUCRE 4 `cp -ru EV/. CALIS/` ile her seyi geri aliyor; durum.json
        eski commit'in "<kol>_bitti" bayragini da getiriyor. Kod arada
        degistiyse bir kol ESKI, digerleri YENI kodla egitilir ve ayni
        tabloya girer -> WARM_OF hatasinin tekrari, ama sessiz hali."""
        simdi = self.kon["commit"]
        if not eski or eski.get("commit") in (None, "?") or simdi == "?":
            return
        if eski["commit"] == simdi:
            return
        # CEKIRDEK AYNIYSA devam serbest: commit degisse de egitim
        # yorungesi degismez (ornek: kol listesi degisti, sifirdan.py degil).
        e_md5, y_md5 = eski.get("md5"), self.kon.get("md5")
        if e_md5 and e_md5 == y_md5:
            self.log(f"  COMMIT KAPISI   disk {eski['commit']} -> simdi "
                     f"{simdi}   (sifirdan.py md5 AYNI {e_md5[:10]}, devam)")
            return
        biten = [k[:-6] for k, v in self.st.items()
                 if k.endswith("_bitti") and v]
        # ILERLEMEYI DISKTEN OKU. durum.json'daki `adim` kol BITENE kadar 0
        # kalir; kol ortasindaki 85.000 adim "yarim is yok" gibi gorunuyordu.
        _e = glob.glob(self.y("cikti*", f"egri_{self.KOL}_s{self.SEED}.json"))
        yarim = [os.path.basename(os.path.dirname(p)) for p in _e
                 if self._oku_json(p, [])]
        assert not biten and not yarim and not self.st.get("adim"), (
            f"EGITIM CEKIRDEGI DEGISTI ama YARIM IS duruyor -> DURDURULDU." + chr(10) +
            f"  diskteki is : commit {eski['commit']}  md5 {str(e_md5)[:10]}" + chr(10) +
            f"                biten {biten or 'yok'}  yarim {yarim or 'yok'}" + chr(10) +
            f"  simdi kosan : commit {simdi}  md5 {str(y_md5)[:10]}" + chr(10) +
            f"  Iki kol FARKLI cekirdekle egitilip ayni tabloya girerdi." + chr(10) +
            f"  Ya o commit'e don, ya {self.C} ve Drive kopyasini silip sifirdan basla.")
        self.log(f"  COMMIT KAPISI   disk {eski['commit']} -> simdi {simdi}"
                 f"   (cekirdek farkli ama hic ilerleme yok, devam)")

    def bitti_mi(self, ad, alt, pencere=None):
        """durum.json 'bitti' diyorsa DISKTEN de dogrula.

        Bayrak var ama olcum noktalari yok (Drive geri yuklemesi yarim
        kaldi / klasor silindi) -> kol ATLANIR, sonra pencere.py o kolun
        anlik goruntulerini bulamaz. Kurtarma yolu kendini bozuyordu."""
        if not self.st.get(f"{ad}_bitti"):
            return False
        eksik = [s for s in (pencere or []) if not os.path.exists(
            self.y(alt, f"snap_{self.KOL}_s{self.SEED}_{s:06d}.pt"))]
        assert not eksik, (
            f"{ad} 'bitti' isaretli ama OLCUM NOKTALARI DISKTE YOK: {eksik}\n"
            f"  klasor: {self.y(alt)}\n"
            f"  Atlanirsa birincil okuma eksik kalir. Ya eksikleri Drive'dan "
            f"geri yukle, ya durum.json'daki '{ad}_bitti' bayragini sil.")
        return True

    # --- yollar ---------------------------------------------------------
    def y(self, *a):
        return os.path.join(self.C, *a)

    def _yaz_json(self, y, d):
        open(y + ".tmp", "w").write(json.dumps(d, indent=1, ensure_ascii=False))
        os.replace(y + ".tmp", y)

    def _oku_json(self, y, vars=None):
        return json.load(open(y)) if os.path.exists(y) else vars

    # --- durum ----------------------------------------------------------
    def kaydet(self, **k):
        self.st.update(k)
        self.st["guncelleme"] = time.strftime("%H:%M:%S")
        self._yaz_json(self.y("durum.json"), self.st)

    def not_(self, *satirlar):
        """Rapora deneye ozel satir ekle (rapor.py oldugu gibi basar)."""
        self.st["rapor_ek"] = list(satirlar)
        self.kaydet()

    def log(self, *a):
        print(*a, flush=True)

    # --- egri -----------------------------------------------------------
    # `alt`: bir kosuda BIRDEN COK kol olabilir (D3.2: A4 kontrolu + D3.2
    # maskeli kolu). Her kol kendi alt klasorunde; varsayilan "cikti".
    def egri(self, alt="cikti"):
        return self._oku_json(
            self.y(alt, f"egri_{self.KOL}_s{self.SEED}.json"), []) or []

    def surdur_yolu(self, alt="cikti"):
        return self.y(alt, f"surdur_{self.KOL}_s{self.SEED}.pt")

    # --- EGITIM ---------------------------------------------------------
    def egit(self, hedef, ek=None, alt="cikti"):
        """hedef adima kadar egit. Log adim adli, yedek adim adli.

        SIFIRDAN baslatmak icin: bos bir `alt` klasoru ver. sifirdan.py
        RESUME_FROM bos olunca ayni tohumla ayni baslangic parametrelerini
        kurar (52/52 tensor bit-ayni oldugu olculdu) -- ek kod gerekmez."""
        os.makedirs(self.y(alt), exist_ok=True)
        sp = self.surdur_yolu(alt)
        env = dict(self.kon["TEMIZ"], **self.kon["ORT"],
                   OUT=self.y(alt), STEPS=str(hedef),
                   RESUME_FROM=(sp if os.path.exists(sp) else ""))
        env.update(ek or {})
        # KONFIG KAPISI BURADA, EGITIMDEN ONCE (14 Eylul).
        #
        # Eskiden suruculer kapiyi egit()'ten SONRA cagiriyordu
        # (g.py:275 egit -> :276 "bitti" -> :278 kapi; d33.py:144 -> :152).
        # Ama sifirdan.py:1401-1402 surdurme paketini `cfg=CFG` ile, yani AZ
        # ONCE KULLANILAN konfigle yaziyor. Yani kapi kendi uretiminin
        # ciktisini denetliyordu: yanlis konfigle surdurme ZATEN OLUP BITMIS
        # oluyor, kol "bitti" isaretleniyor, sonra kapi tautolojik olarak
        # geciyordu. CLAUDE.md 7 bu kapi icin "yanlis MASK_KEY ile SESSIZCE
        # surdurmeyi ONLER" diyor -- o haliyle ONLEYEMIYORDU.
        #
        # Kapi yalniz SURDURMEDE anlamli: sifirdan baslarken paket yok.
        # Ve `ek`/`alt` artik surucude TEKRAR yazilmiyor -> ikisinin
        # ayrisma ihtimali YAPISAL olarak kalkti (AUDIT'in onerisi).
        if env["RESUME_FROM"]:
            self.konfig_kapisi_tam(ek=ek, alt=alt)
        ly = self.y("log", f"egitim_{self.D}_{alt}_{hedef:06d}.txt")
        t = time.time()
        r = subprocess.run([sys.executable, "-u", "sifirdan.py"],
                           cwd=self.kon["KOD"], env=env,
                           stdout=open(ly, "w"), stderr=subprocess.STDOUT)
        assert r.returncode == 0, \
            f"egitim coktu rc={r.returncode} -> {ly} (son satirlar:\n" + \
            "".join(open(ly).read().splitlines(True)[-15:]) + ")"
        # GERI DONULEBILIRLIK: paket her olcumde UZERINE yaziliyor; adim adli
        # bir kopya almazsak gecmis bir adima donulemez (CLAUDE.md 7).
        if os.path.exists(sp):
            os.makedirs(self.y("sur", alt), exist_ok=True)
            hy = self.y("sur", alt, f"surdur_{self.KOL}_s{self.SEED}_{hedef:06d}.pt")
            if not os.path.exists(hy):
                shutil.copy2(sp, hy + ".tmp"); os.replace(hy + ".tmp", hy)
        return round(time.time() - t)

    # --- KAPILAR --------------------------------------------------------
    def yorunge_kapisi(self, adim, alan="comp", tol=None, alt="cikti"):
        """Maskesiz faz, referans kolun TEKRARI olmali. Degilse burada dur.

        Bugunku warm=STEPS//20 hatasi bu kapiyla 10.000'de yakalandi; kapi
        yoksa 80.000 adim sonra 'prosedur basarisiz' diye raporlanacakti."""
        ref = {r["step"]: r for r in (self._oku_json(self.kon["REF_EGRI"], []) or [])}
        e = [r for r in self.egri(alt) if r["step"] == adim]
        if not e or adim not in ref:
            return None
        f = e[0][alan] - ref[adim][alan]
        tol = self.kon["TOL_YORUNGE"] if tol is None else tol
        ok = abs(f) <= tol
        self.log(f"  YORUNGE KAPISI  {alan}(bu)={e[0][alan]:.3f}  "
                 f"{alan}(ref)={ref[adim][alan]:.3f}  fark {f:+.3f}  tol {tol}"
                 f"  -> {'GECTI' if ok else 'KALDI'}")
        assert ok, (f"YORUNGE AYRISTI: {alan} farki {f:+.3f} > {tol}. Maskesiz faz "
                    f"referans kolun tekrari DEGIL -> kiyas gecersiz, DURDURULDU.")
        return f

    def _canli_cfg(self, ek=None):
        """Egitimin GORECEGI CFG -- ayni ortamla, AYRI surecte, GPU'suz.

        kosu.py `import sifirdan` YAPAMAZ: surucu sureci egitimi alt surec
        olarak baslatiyor (egit(), :subprocess.run) ve kendi ortami alt
        surecinkiyle ayni olmak zorunda degil. Import etseydik kapi YANLIS
        referansa bakardi.
        Cozum: alt sureci ZATEN aciyoruz -- bir saniyelik bir tanesini
        CFG'yi sormak icin acalim. ~2 sn, GPU yok (CUDA gizleniyor).
        """
        env = dict(self.kon["TEMIZ"], **self.kon["ORT"], CUDA_VISIBLE_DEVICES="")
        env.update(ek or {})
        r = subprocess.run(
            [sys.executable, "-c",
             "import json, sifirdan as S; print('CFG_JSON', json.dumps("
             "{k: (list(v) if isinstance(v, tuple) else v)"
             " for k, v in S.CFG.items()}))"],
            cwd=self.kon["KOD"], env=env, capture_output=True, text=True)
        assert r.returncode == 0, (
            "CANLI CFG okunamadi (rc=%d):" % r.returncode + chr(10)
            + (r.stderr or r.stdout)[-600:])
        satir = [l for l in r.stdout.splitlines() if l.startswith("CFG_JSON ")]
        assert satir, "CFG_JSON satiri yok:" + chr(10) + r.stdout[-400:]
        return json.loads(satir[-1][len("CFG_JSON "):])

    def konfig_kapisi_tam(self, bekle=None, muaf=("STEPS", "_commit"),
                          alt="cikti", ek=None):
        """konfig_kapisi'nin OPT-OUT hali: SORULMAYAN ANAHTAR DA ARIZADIR.

        NEDEN (14 Eylul, AUDIT). `konfig_kapisi` yalniz KENDISINE VERILEN
        anahtarlari denetliyor. Alti cagrinin altisi da yalniz MASK_KEY/
        MASK_BLK geciriyor; hicbiri MEM_AT gecirmiyor. Ve kol C tam bu
        yuzden gecersiz kaldi (belge/KOLLAR.md, C satiri):
          colab_C_devam.py:54 TEK okuma noktasiyla egitilmis paketten
          MEM_AT=2,4,6 ile SURDURDU. Memory tek paylasimli modul ve okuma
          sayisi parametre eklemiyor -> load_state_dict SESSIZCE gecti.

        Asil ariza MEM_AT'in eksik olmasi DEGIL, varsayilanin
        "sorulmayan denetlenmez" olmasi. Bir sonraki kol yeni bir anahtar
        getirir (SHARE, IDENT_FRAC, HOP2_FRAC, N_PAIR...), biri listeye
        eklemeyi unutur, ayni sessiz hata tekrar mumkun olur.

        Burada varsayilan TERSINE cevrildi: kayitli cfg'deki HER anahtar
        ya `bekle`de sorulmus ya `muaf`ta ACIKCA hariç tutulmus olmali.
        Unutmak ARTIK HATA VERIR.

        BEKLENEN DEGERLER DE ELLE YAZILMAZ (14 Eylul, AUDIT ikinci tur).
        `bekle=None` ise referans `_canli_cfg()`'den gelir: egitimin ayni
        ortamda gorecegi CFG. Boylece "ismi listeye eklemeyi unutma"
        problemini cozerken "degeri yanlis yazma" problemini ACMIYORUZ --
        ikisi de ayni siniftan olurdu. Gercek CFG 24 anahtar; elle yazmak
        21'ini kopyalamak demekti.

        MUAF LISTESI -- her biri GEREKCELI:
          STEPS    surucular parca parca egitiyor (:egit, STEPS=str(hedef))
          _commit  sifirdan.py:1522'de main()'de ekleniyor, yani KAYITLI
                   pakette VAR ama taze import'ta YOK; ayrica iki oturum
                   arasi mesru sekilde degisir ve ZATEN ayri bir kapida
                   denetleniyor (`_commit_kapisi`, CLAUDE.md 7).
        Baska mesru fark cikarsa muaf'a GEREKCESIYLE eklenir, sessizce degil.

        Eski `konfig_kapisi` geriye donuk uyumluluk icin duruyor; YENI
        surucular BUNU cagirmali.
        """
        import torch
        if bekle is None:
            bekle = self._canli_cfg(ek)
        sp = self.surdur_yolu(alt)
        assert os.path.exists(sp), "surdurme paketi yok"
        c = torch.load(sp, map_location="cpu", weights_only=False).get("cfg", {})
        sorulmayan = _sorulmayan(c, bekle, muaf)
        assert not sorulmayan, (
            f"SORULMAYAN KONFIG ANAHTARI: {sorulmayan}" + chr(10)
            + "  Bu anahtarlar surdurme paketinde KAYITLI ama denetlenmiyor."
              " Sessizce farkli bir konfigle surdurmek MUMKUN." + chr(10)
            + "  Ya `bekle` sozlugune ekle, ya `muaf`a GEREKCESIYLE koy."
              "  (kol C bu yuzden gecersiz kaldi -- belge/KOLLAR.md)")
        # MUAF anahtarlar KARSILASTIRMADAN da cikarilir. Yoksa `bekle` canli
        # CFG oldugunda STEPS mesru sekilde farkli olur (paket 80000, egitim
        # 120000 hedefliyor) ve kapi HAKSIZ YERE atesler. Bunu kendi testim
        # yakaladi: TEST 4'te fark listesi ['STEPS', 'MEM_AT'] cikti; orada
        # olmasi gereken yalniz MEM_AT.
        return self.konfig_kapisi({k: v for k, v in bekle.items()
                                   if k not in muaf}, alt=alt)

    def konfig_kapisi(self, bekle, alt="cikti"):
        """Surdurme paketi hangi konfigle yazilmis? Davranistan degil KAYITTAN.

        DIKKAT: yalniz `bekle`deki anahtarlari denetler. Yeni surucular
        `konfig_kapisi_tam` kullanmali (sorulmayan anahtar da ariza sayilir).
        """
        import torch
        sp = self.surdur_yolu(alt)
        assert os.path.exists(sp), "surdurme paketi yok"
        c = torch.load(sp, map_location="cpu", weights_only=False).get("cfg", {})
        # tuple/list ayrimi anlamsiz: paket tuple tutuyor, JSON list dondurur.
        n = lambda v: tuple(v) if isinstance(v, (list, tuple)) else v
        var = {k: n(c.get(k)) for k in bekle}
        bek = {k: n(v) for k, v in bekle.items()}
        self.log(f"  KONFIG KAPISI   kayitli {var}  beklenen {bek}")
        assert var == bek, \
            f"EGITIM YANLIS KONFIGLE KOSUYOR: {var} != {bek} -> DURDURULDU"
