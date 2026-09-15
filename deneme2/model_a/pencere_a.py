# -*- coding: utf-8 -*-
"""pencere_a — model_a ailesinin BIRINCIL OKUMASI.

N anlik goruntunun AGIRLIK ORTALAMASI alinir, sonra TEK model olculur.
Egri degerlerinin ortalamasi DEGIL -- ikisi ayri seydir ve karistirmak
bulguyu degistirir.

    python pencere_a.py <klasor> [--genislik 5] [--cikti out.json]

    klasor:  bir TOHUMUN klasoru, orn.  G:\\Drive'im\\deneme2\\model_a\\t0

Butun kayan pencereler raporlanir, EN IYISI SECILMEZ. Secim yapilacaksa
sonuclara bakan insan yapar ve [SONRADAN SECIM] diye etiketler.

--------------------------------------------------------------------------
UC YAPISAL KORUMA -- arsivde bunlarin yoklugu pahaliya mal oldu

1. AYAR DOSYADAN OKUNUR, elle kurulmaz.
   Klasordeki `ayar_t<N>.json` neyse model o. Yani egitilen mimari ile
   olculen mimari AYNI OLMAK ZORUNDA -- kod bunu varsaymiyor, okuyor.
   Arsivde maskeyle EGITILMIS iki kol maske KAPALIYKEN degerlendirildi ve
   iki belgedeki sayilar duzeltilmek zorunda kaldi. Burada o mumkun degil:
   maske de dongu de d de ayar dosyasindan geliyor.

2. ADIM GLOB'DAN OKUNUR, dosya adi bicimi VARSAYILMAZ.
   Arsivde `f"..._{20000}.pt"` yazilmis, sifir dolgulu adlari bulamamis
   ama 190000'i bulmustu (zaten 6 haneydi) -- hata KISMEN gorunmustu.

3. OLCME SETLERI model_a'dan IMPORT edilir, KOPYALANMAZ.
   Arsivde bu mantik iki dosyada ayri ayri duruyordu; bir salt degisirse
   iki dosya FARKLI ornek olcerdi ve hicbir sey hata vermezdi.
"""
from __future__ import annotations

import argparse, glob, json, os, re, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_a as M


def anlik_goruntuler(klasor: str) -> dict:
    """{adim: yol}.  Adim dosya ADINDAN degil, glob + regex ile cikarilir.

    Iki yerleşime de bakar: `t<N>/snap/*.pt` (yeni) ve `t<N>/snap_*.pt`
    (eski duz). `.tmp` uzantili yarim dosyalar ikisinde de ELENIR --
    desen `.pt` ile BITMEK zorunda."""
    bul = {}
    for y in (glob.glob(os.path.join(klasor, "snap", "*.pt"))
              + glob.glob(os.path.join(klasor, "snap_*.pt"))):
        m = re.search(r"_(\d+)\.pt$", os.path.basename(y))
        if m:
            bul[int(m.group(1))] = y
    assert bul, f"anlik goruntu YOK: {klasor}"
    return dict(sorted(bul.items()))


def ayar_oku(klasor: str) -> M.Ayar:
    """Egitimin YAZDIGI ayari geri kur. Elle kurulmaz (koruma 1)."""
    y = glob.glob(os.path.join(klasor, "ayar_t*.json"))
    assert len(y) == 1, f"tam bir ayar_t*.json bekleniyordu, {len(y)} bulundu: {klasor}"
    ham = json.load(open(y[0], encoding="utf-8"))
    # `_` ile baslayanlar META (orn. _olcme_izi) -- Ayar alani degiller,
    # ayrilirlar. Assert'ler yalniz GERCEK alanlara bakar.
    d = {k: x for k, x in ham.items() if not k.startswith("_")}
    alan = {f.name for f in __import__("dataclasses").fields(M.Ayar)}
    fazla = set(d) - alan
    eksik = alan - set(d)
    assert not fazla, f"ayar dosyasinda BILINMEYEN alan: {fazla} ({y[0]})"
    assert not eksik, f"ayar dosyasinda EKSIK alan: {eksik} ({y[0]})"
    d["mask_blok"] = tuple(d["mask_blok"])
    d["betas"] = tuple(d["betas"])
    return M.Ayar(**d)


def kunye_oku(klasor: str) -> dict:
    """Kosunun kunyesi: HANGI KOD, HANGI GPU, kosu BITTI mi.

    `ayar` "ne isteyecektik"i yazar, kunye "fiilen ne kostu"yu. Ikisi ayri
    sey ve ikincisi daha once HICBIR YERDE durmuyordu: commit yalniz
    defterin ekranina basiliyordu, log ise her kosuda ustune yaziliyordu.
    Boylece "bu sayilar hangi koddan" sorusunun mekanik cevabi yoktu."""
    y = glob.glob(os.path.join(klasor, "kosu_t*.json"))
    if not y:
        return {}
    return json.load(open(y[0], encoding="utf-8"))


def kunye_bas(k: dict):
    if not k:
        print("  kunye    YOK (eski kosu) -- commit/GPU/durum bilinmiyor")
        return
    print(f"  kunye    commit {k.get('commit','?')}   "
          f"{k.get('gpu') or k.get('cihaz','?')}   torch {k.get('torch','?')}"
          f"   {k.get('baslangic','?')}")
    d = k.get("durum")
    if d == "BITTI":
        print(f"  durum    BITTI  {k.get('sure_dk','?')} dk, "
              f"son adim {k.get('son_adim','?')}")
    else:
        # Yarim kosunun anlik goruntuleri GECERLIDIR, ama "egri duzlesti mi"
        # sorusu BASKA bir soruya donusur: egri bitmedi, KESILDI.
        print(f"  !! durum {d}  -- kosu TAMAMLANMADI "
              f"(son adim {k.get('son_adim','?')})")
        if k.get("hata"):
            print(f"     hata: {k['hata']}")
        print("     Anlik goruntuler gecerli, ama BUTCE sorusu sorulamaz: "
              "egri doymadi, KESILDI.")

    # OLCEN KOD, EGITEN KODLA AYNI MI?
    # Parmak izi olcme SETINI koruyor, KODU korumuyor. Mimari degisirse
    # `load_state_dict` zaten patlar (strict=True), ama olcme mantigi
    # degisirse -- varlik-kisitli argmax, kisayol tanimi, pencere ortalamasi
    # -- hicbir sey ses cikarmaz ve sayilar sessizce baska bir seyi olcer.
    e = (k.get("commit") or "").split("+")[0]
    s = M._commit().split("+")[0]
    if e and s and e not in ("?",) and s not in ("?",) and e != s:
        print(f"  !! KOD FARKLI: egitim {k.get('commit')}, olcum {M._commit()}")
        print("     Bu sayilari uretecek kod, onlari EGITEN koddan baska bir "
              "surumde. Hata degil -- ama bulguya yazilmali.")


def agirlik_ortalamasi(yollar: list) -> dict:
    """Eleman eleman ortalama. fp16 kaydedildi -> float32'de toplanir."""
    toplam = None
    for y in yollar:
        sd = torch.load(y, map_location="cpu")
        if toplam is None:
            toplam = {k: v.float() for k, v in sd.items()}
        else:
            assert set(sd) == set(toplam), f"anahtar kumesi farkli: {y}"
            for k in toplam:
                toplam[k] += sd[k].float()
    return {k: v / len(yollar) for k, v in toplam.items()}


def olc(ayar: M.Ayar, veri: M.Veri, kod: dict, L: dict, sd: dict) -> dict:
    net = M.Model(ayar, veri.vocab).to(M.DEV)
    net.load_state_dict(sd)
    net.eval()
    r = {k: M.dogruluk(net, veri, *kod[k]) for k in kod}
    r["ent_kisayol"] = M.kisayol_orani(net, veri, L["ent"])
    r["ent_yok_kisayol"] = M.kisayol_orani(net, veri, L["ent_yok"])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("klasor")
    ap.add_argument("--genislik", type=int, default=5,
                    help="pencerede kac anlik goruntu (varsayilan 5)")
    ap.add_argument("--cikti", default=None)
    a = ap.parse_args()

    ayar = ayar_oku(a.klasor)
    kunye = kunye_oku(a.klasor)
    snap = anlik_goruntuler(a.klasor)
    adimlar = list(snap)
    print(f"=== pencere_a  {ayar.ad} tohum {ayar.tohum} ===")
    print(f"  klasor   {a.klasor}")
    print(f"  ayar     d={ayar.d} l={ayar.l} nh={ayar.nh} dff={ayar.dff} "
          f"dongu={ayar.dongu} lr={ayar.lr} wd={ayar.wd}")
    print(f"  maske    {'YOK' if ayar.mask_poz is None else f'{ayar.mask_poz}@{ayar.mask_blok}'}"
          f"   <- ayar dosyasindan, VARSAYILMADI")
    kunye_bas(kunye)
    print(f"  anlik    {len(adimlar)} goruntu: {adimlar[0]}..{adimlar[-1]}")

    veri = M.veri_kur(ayar)
    L = M.olcme_listeleri(ayar, veri)
    kod = {k: (M.kodla_1hop(veri, L[k]) if k == "one" else M.kodla_2hop(veri, L[k]))
           for k in L if L[k]}
    # OLCME SETI PARMAK IZI -- model_a.olcme_izi()'nden, KOPYA DEGIL.
    iz = M.olcme_izi(L)
    print("  olcme    " + "  ".join(f"{k} {len(L[k])}" for k in L if L[k]))
    print(f"  parmak izi {iz}")
    # EGITIMIN yazdigi izle KARSILASTIR: olcum, egitimin gordugu ornekleri
    # mi olcuyor? Ayni ayardan turemis olmalari YETMEZ -- veri ureteci ya da
    # bolme mantigi degistiyse iz tutmaz ve sayilar kiyaslanamaz hale gelir.
    _ay = glob.glob(os.path.join(a.klasor, "ayar_t*.json"))
    _kayitli = json.load(open(_ay[0], encoding="utf-8")).get("_olcme_izi")
    if _kayitli is None:
        print("  !! egitim izi KAYDETMEMIS (eski kosu) -- karsilastirilamadi")
    elif _kayitli != iz:
        raise SystemExit(
            f"\n!! OLCME SETI DEGISMIS: egitim {_kayitli}, olcum {iz}.\n"
            "   Bu kosunun egittigi orneklerle simdi olctuklerimiz AYNI DEGIL;\n"
            "   sayilar kiyaslanamaz. veri_okul.py ya da bolme mantigi"
            " degisti mi?")
    else:
        print(f"  iz EGITIMLE AYNI ({_kayitli})  <- ayni ornekler olculuyor")

    if len(adimlar) < a.genislik:
        print(f"\n!! {len(adimlar)} anlik goruntu var, pencere {a.genislik} "
              f"istiyor -- pencere {len(adimlar)}'e indirildi")
        a.genislik = len(adimlar)

    pencereler = [adimlar[i:i + a.genislik]
                  for i in range(len(adimlar) - a.genislik + 1)]
    print(f"\n{len(pencereler)} kayan pencere, genislik {a.genislik}. "
          f"EN IYISI SECILMIYOR, hepsi raporlaniyor.\n")
    bas = f"  {'pencere':<18}" + "".join(f"{k:>9}" for k in
                                         ("one", "seen", "comp", "ent", "ent_yok"))
    print(bas + f"{'ent_ksy':>9}{'yok_ksy':>9}")

    sonuc = []
    for p in pencereler:
        sd = agirlik_ortalamasi([snap[x] for x in p])
        r = olc(ayar, veri, kod, L, sd)
        etiket = f"{p[0]}-{p[-1]}"
        sonuc.append(dict(pencere=etiket, adimlar=p, **r))
        print(f"  {etiket:<18}"
              + "".join(f"{r.get(k, float('nan')):>9.4f}" for k in
                        ("one", "seen", "comp", "ent", "ent_yok"))
              + f"{r['ent_kisayol']:>9.4f}{r['ent_yok_kisayol']:>9.4f}")

    # --- ONCEDEN YAZILAN KAPILAR (belge/onkayit/model_a.md 5)
    #     Kapiyi insan degil KOD degerlendirir: unutulmasin, sonucu gorup
    #     gevsetilmesin.
    son = sonuc[-1]
    kapilar = [
        ("SAGLIK-1HOP",  "one >= 0.98",             son.get("one", 0) >= 0.98),
        ("SAGLIK-EZBER", "seen >= 0.95",            son.get("seen", 0) >= 0.95),
        ("OLGUNLUK",     "comp >= 0.50",            son.get("comp", 0) >= 0.50),
        ("BIRIM TESTI",  "ent_yok_kisayol == 0.000",
         abs(son.get("ent_yok_kisayol", 1)) < 1e-9),
    ]
    # KAPIYA ADIYLA ERIS, SIRA NUMARASIYLA DEGIL (15 Eylul hakemligi).
    # Once `kapilar[2]` / `kapilar[3]` yaziyordu; listeye basa bir kapi
    # eklendiginde bu satirlar SESSIZCE baska kapiyi okurdu.
    gecti_mi = {ad: g for ad, _, g in kapilar}
    print(f"\n{'='*62}\nONCEDEN YAZILAN KAPILAR  (son pencere: {son['pencere']})")
    for ad, kural, gecti in kapilar:
        print(f"  {'GECTI ' if gecti else '!! KALDI'}  {ad:<14} {kural}")
    if not gecti_mi["BIRIM TESTI"]:
        print("  !! BIRIM TESTI KALDI -> OLCUM KODU BOZUK, sayilar okunmaz.")
    if not gecti_mi["OLGUNLUK"]:
        print("  !! OLGUNLUK KALDI -> kol 'olgunlasmamis', ENT YORUMLANMAZ.")

    # --- BUTCE YETTI MI (onkayit 6, son satir)
    #     OLGUNLUK KAPISI GECMEDEN BU SORU SORULMAZ. Hicbir sey ogrenilmemis
    #     bir kosuda egri de "duz" gorunur ve bu satir "butce yetti" derdi --
    #     tam tersi dogruyken.
    # KUSUR (15 Eylul, ikinci hakemlik): burada `sonuc[-1]` ile `sonuc[-2]`
    # kiyaslaniyordu. Kayan pencereler ORTUSUYOR: genislik 5'te ardisik iki
    # pencere 5 anlik goruntunun 4'unu PAYLASIR. Yani fark YAPISAL OLARAK
    # kucuk cikar ve bu satir neredeyse her zaman "egri duzlesmis -> butce
    # yetti" derdi. Tam da cevap vermesi gereken soruda YANLIS TARAFA
    # dusen bir esik. Artik AYRIK iki pencere kiyaslaniyor.
    if len(sonuc) >= 2:
        if len(sonuc) > a.genislik:
            onceki, tip = sonuc[-1 - a.genislik], "AYRIK"
        else:
            onceki, tip = sonuc[0], "ORTUSEN"
        d = sonuc[-1].get("ent", 0) - onceki.get("ent", 0)
        print(f"\nBUTCE: {onceki['pencere']} -> {sonuc[-1]['pencere']} "
              f"({tip} pencereler) ent degisimi {d:+.4f}")
        if tip == "ORTUSEN":
            print(f"  !! {len(sonuc)} pencere var, ayrik cift icin "
                  f"{a.genislik + 1} gerekiyor. Bu iki pencere anlik "
                  "goruntu PAYLASIYOR, fark oldugundan KUCUK gorunur.")
        if kunye and kunye.get("durum") != "BITTI":
            print(f"  KOSU TAMAMLANMAMIS (durum {kunye.get('durum')}) -> BUTCE "
                  "SORUSU SORULMAZ. Egri doymadi, KESILDI.")
        elif not gecti_mi["OLGUNLUK"]:
            print("  OLGUNLUK kapisi KALDI -> BUTCE SORUSU SORULMAZ. Egri duz "
                  "cikabilir ama sebebi doyma degil, hic ogrenilmemis olmasi.")
        elif d > 0.005:
            print("  egri HALA TIRMANIYOR -> butce YETMEDI, bu kosu butce "
                  "sorusuna cevap VERMEDI")
        else:
            print("  egri duzlesmis -> butce yetti")

    yol = a.cikti or os.path.join(a.klasor, f"pencere_{ayar.ad}_t{ayar.tohum}.json")
    # M._yaz_json: atomik (.tmp -> replace). Egitim hala kosuyorken bu dosya
    # okunabilir; yarim yazilmis json JSONDecodeError verir (olculdu).
    M._yaz_json(yol, dict(
        ad=ayar.ad, tohum=ayar.tohum, klasor=a.klasor, genislik=a.genislik,
        parmak_izi=iz, ayar=ayar.sozluk(), kunye=kunye, pencereler=sonuc,
        kapilar=[dict(ad=x, kural=y, gecti=z) for x, y, z in kapilar]))
    print(f"\n-> {yol}")


if __name__ == "__main__":
    main()
