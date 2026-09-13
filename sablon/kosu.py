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


class Kosu:
    def __init__(self, kon):
        self.kon = kon
        self.C = kon["CALIS"]
        self.KOL, self.SEED, self.D = kon["KOL"], kon["SEED"], kon["DENEY"]
        for a in ("cikti", "sur", "log", "ham"):
            os.makedirs(self.y(a), exist_ok=True)
        kon["commit"] = kon.get("commit", "?")
        self._yaz_json(self.y("konfig.json"), kon)
        self.st = self._oku_json(self.y("durum.json"),
                                 dict(faz=1, adim=0, faz_ad="", rapor_ek=[]))

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

    def konfig_kapisi(self, bekle, alt="cikti"):
        """Surdurme paketi hangi konfigle yazilmis? Davranistan degil KAYITTAN."""
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
