# -*- coding: utf-8 -*-
"""pencere_07 — model_05'in KENDI BIRINCIL OKUMASI. TEK BASINA DURUR.

Kullanici karari, 16 Eylul 2026: *"bunlarin hepsi model_05 folderi
altinda olmali. model_05 diger hicbir model ile ayni seyi
kullanmamali."*

`model_a/pencere_a.py`nin KOPYASI (uretici: scratchpad/kur_okuma00.py). Modeli
`ModelSade` ile kurar. Paylasilan surumde yapilan bir degisiklik buraya
GECMEZ; `test_05.py` ikisinin AYNI SEYI olctugunu her kosuda siniyor.
"""
from __future__ import annotations

import argparse, glob, json, os, re, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import taban_07 as M                                          # noqa: E402
from model_07 import ModelSade                                # noqa: E402

# model_05'in TEK modeli var; kanca yok, dogrudan yazili.
MODEL_SINIFI = ModelSade


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
    # ALANLAR SONRADAN EKLENEBILIR. Eski `ayar_t<N>.json`larda yeni alan
    # YOKTUR; tablo olmasa bu assert butun eski kosularin OLCULMESINI
    # kirardi. Olculdu (15 Eylul, `veri_ad` eklenince fiilen kirildi).
    # Tablo `taban_05.ESKI_VARSAYILAN` -- TEK yerde durur, `surdurme_oku`
    # da ayni tabloyu kullanir, ikisi ayrisamaz.
    for k in sorted(eksik & set(M.ESKI_VARSAYILAN)):
        d[k] = M.ESKI_VARSAYILAN[k]
        print(f"  ESKI AYAR: '{k}' dosyada yok -> {d[k]!r} varsayildi "
              f"(alan {k} sonradan eklendi)")
    eksik -= set(M.ESKI_VARSAYILAN)
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


@torch.no_grad()
def fim_dogruluk(net, v, lst, hangi, bs=256):
    """BOSLUK DOLDURMA dogrulugu -- AYNI zincirler, BASKA yonden sorulmus.

    Kullanici, 17 Eylul: *"soru cesitlerimiz de degisti, FIM yapiyoruz,
    bunun ent'i olur mesela degil mi?"*  Olur: ayni `ent` zinciri uc
    ayri soru verir.

        ILERI   Ibrahim Yilmaz'in arkadasinin memleketi <BOS>  -> Kocaeli
        OZNE    <BOS>'in arkadasinin memleketi Kocaeli'dir     -> Ibrahim Yilmaz
        ILISKI  Ibrahim Yilmaz'in <BOS>nin memleketi Kocaeli'dir -> arkadasi

    !! BUNLAR HUKUM VERMEZ. Birincil olcu ILERI yon: olcme izi
    `f4ce53fd1555` model_05 ile AYNI ve o kiyas yalniz ileri yonde
    gecerli. Burasi ek bilgi; "hangi olcu iyi ciktiysa onu sectik"
    durumuna dusmemek icin ayri tabloda ve ayri adla duruyor.

    !! OZNE sorusu ILERI YONDEN KOLAY: cevap varligi (Kocaeli) girdide
    duruyor, yani arama uzayi zaten dar. Ayni olcekte okunmamali.
    """
    if not lst:
        return float("nan")
    onceki = net.training
    net.eval()
    # !! DIZI KURULUMU `taban_07.kodla_fim_sinav`DA -- burada DEGIL.
    # Ayni kurulum dokum betigine de lazim; iki kopya birbirinden
    # kayarsa olcum bir seyi, dosya baska seyi gosterirdi.
    X, hedef_n = M.kodla_fim_sinav(v, lst, hangi)
    dizi = [[int(t) for t in r if int(t) != M.PAD] for r in X]
    ok = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(M.DEV)
        with torch.autocast(M.DEV, dtype=torch.float16,
                            enabled=(M.DEV == "cuda")):
            lg = net(xb)
        lg = lg.float()
        for j in range(len(xb)):
            n = len(dizi[i + j])
            u = hedef_n[i + j]
            # hedef jetonlar dizinin SONUNDA; konum t, t+1'i tahmin eder
            dogru = True
            for k in range(u):
                p = n - u + k - 1
                if int(lg[j, p].argmax()) != int(X[i + j, n - u + k]):
                    dogru = False
                    break
            ok.append(dogru)
    net.train(onceki)
    return float(np.mean(ok))


def olc(ayar: M.Ayar, veri: M.Veri, kod: dict, L: dict, sd: dict) -> dict:
    # MODEL SINIFI AILEYE GORE secilir. `pencere_b` darbogazli kendi
    # sinifini yazar. Yanlis sinifla olcmek ya SESSIZCE yanlis sayi
    # uretir ya da load_state_dict'te patlar; ikincisi iyi, birincisi
    # olumcul. `model_b/test_sabit_b.py` bu kancayi HER KOSUDA dogrular
    # -- cunku 16 Eylul'de bu yama bir kez SESSIZCE uygulanmadi.
    net = (MODEL_SINIFI or M.Model)(ayar, veri.vocab).to(M.DEV)
    net.load_state_dict(sd)
    net.eval()
    r = {k: M.dogruluk(net, veri, *kod[k]) for k in kod}
    r["ent_kisayol"] = M.kisayol_orani(net, veri, L["ent"])
    r["ent_yok_kisayol"] = M.kisayol_orani(net, veri, L["ent_yok"])
    # SORU BICIMI -- IKINCIL, ayri adla (`soru_`), ayri tabloda.
    #
    # Kullanici, 17 Eylul: *"ben su soruyu da sorabilmeliyim: Ibrahim
    # Yilmaz'in danismanin arkadasi kimdir?"* Bu kolun eklentisi o, ve
    # ISE YARAYIP YARAMADIGI olculmeli.
    #
    # !! HUKUM VERMEZ. Birincil olcu DUZ BILDIRIM (`kodla_2hop`,
    # bicim 0), cunku olcme izi `f4ce53fd1555` model_05 ve model_06 ile
    # AYNI ve o kiyas yalniz orada gecerli. Soru bicimi YENI bir yuzey;
    # onu birincil yapmak, uc kolu ayni tabloda okuma imkanini bitirir.
    # "Hangi olcu iyi ciktiysa onu sectik" durumuna dusmemek icin ayri
    # adla ve ayri tabloda duruyor.
    #
    # BEKLENEN GERILIM, onceden yaziliyor: soru satiri EGITIMDE var
    # (havuzun %8,0'i) ama SINAV zincirlerinin `ent` kismi icin
    # ZINCIR BASI olarak YOK -- `ent` tanimi yuzeyden bagimsiz. Yani
    # `soru_ent` ile `ent` ARASINDAKI FARK, yuzeyin tek basina ne
    # getirdigini soyler.
    if getattr(ayar, "soru_kat", 0) > 0:
        for _b in ("one", "seen", "comp", "ent", "ent_yok", "ood"):
            if L.get(_b):
                _hop = 1 if _b == "one" else 2
                r[f"soru_{_b}"] = M.dogruluk(
                    net, veri, *M.kodla_soru(veri, L[_b], _hop))
    # BOSLUK DOLDURMA -- IKINCIL, ayri adla (`fim_`), ayri tabloda.
    if getattr(veri, "bosluk", 0):
        for _b in ("comp", "ent", "ent_yok", "ood"):
            if L.get(_b):
                # !! ANAHTAR `fim_ozne1_` -- model_06'nin `fim_ozne_`
                # sayilari OZNENIN TAMAMINI bosaltan ESKI tanimla
                # olculdu. Ad ayni kalsa iki AYRI gorev ayni sutunda
                # okunurdu.
                r[f"fim_ozne1_{_b}"] = fim_dogruluk(net, veri, L[_b], "ozne")
                r[f"fim_rel_{_b}"] = fim_dogruluk(net, veri, L[_b], "iliski")
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
    # OLCME SETI PARMAK IZI -- taban_05.olcme_izi()'nden, KOPYA DEGIL.
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
    # SUTUNLAR OLCME SETINDEN TURETILIR, elle yazilmaz. `ent_kati`
    # (kati_pay > 0) ancak boyle gorunur; sabit liste olsaydi olculur ama
    # BASILMAZDI -- "olculup gosterilmeyen sayi, yok sayilan sayidir".
    # !! 15 Eylul: `ood` eklendiginde BU LISTEYE eklenmedi ve model_a8'in
    # 60.000'lik kosusunda `ood` OLCULDU ama BASILMADI -- json'a yazildi,
    # tabloya girmedi. Tam olarak bu dosyanin yorumunda yazan tuzak
    # ("olculup gosterilmeyen sayi, yok sayilan sayidir") ve `ent_yok`un
    # basina gelenin aynisi. Liste artik OLCME SETINDEN suzuluyor ama
    # SIRALAMA burada duruyor; YENI BOLME EKLEYEN BU SATIRA DA EKLEMELI.
    SUT = tuple(k for k in ("one", "seen", "comp", "ood", "ent", "ent_yok",
                            "ent_kati") if k in kod)
    # `fim_*` anahtarlari AYRI TABLODA basiliyor -> bu denetimden muaf.
    # (Denetimin amaci "olculup HIC gosterilmeyen sayi" yakalamak.)
    _atlanan = [k for k in kod if k not in SUT]
    assert not _atlanan, (
        f"OLCULUYOR AMA BASILMIYOR: {_atlanan}. pencere_a.py'deki SUT "
        f"listesine ekle -- sessiz kaybolmasin.")
    bas = f"  {'pencere':<18}" + "".join(f"{k:>9}" for k in SUT)
    print(bas + f"{'ent_ksy':>9}{'yok_ksy':>9}")

    sonuc = []
    for p in pencereler:
        sd = agirlik_ortalamasi([snap[x] for x in p])
        r = olc(ayar, veri, kod, L, sd)
        etiket = f"{p[0]}-{p[-1]}"
        sonuc.append(dict(pencere=etiket, adimlar=p, **r))
        print(f"  {etiket:<18}"
              + "".join(f"{r.get(k, float('nan')):>9.4f}" for k in SUT)
              + f"{r['ent_kisayol']:>9.4f}{r['ent_yok_kisayol']:>9.4f}")

    # --- SORU BICIMI TABLOSU (IKINCIL -- HUKUM VERMEZ) -----------------
    _sk = [k for k in sonuc[-1] if k.startswith("soru_")]
    if _sk:
        _sb = [b for b in ("one", "seen", "comp", "ent", "ent_yok", "ood")
               if f"soru_{b}" in sonuc[-1]]
        print()
        print("=" * 62)
        print("SORU BICIMI -- AYNI zincirler, '... kimdir?' diye sorulmus")
        print("  !! HUKUM VERMEZ. Birincil olcu yukaridaki DUZ BILDIRIM;")
        print("     olcme izi f4ce53fd1555 ile model_05/06 kiyasi YALNIZ")
        print("     orada gecerli. Burasi EKLENTININ ISE YARAYIP")
        print("     YARAMADIGINI soyler, kolun hukmunu DEGIL.")
        print(f"  {'pencere':<18}" + "".join(f"{b:>11}" for b in _sb))
        for r in sonuc:
            print(f"  {r['pencere']:<18}"
                  + "".join(f"{r.get('soru_'+b, float('nan')):>11.4f}"
                            for b in _sb))
        print("  KIYAS -- ayni satirin DUZ BILDIRIM hali:")
        for r in sonuc[-1:]:
            print(f"  {'(son pencere)':<18}"
                  + "".join(f"{r.get(b, float('nan')):>11.4f}" for b in _sb))

    # --- BOSLUK DOLDURMA TABLOSU (IKINCIL -- HUKUM VERMEZ) -------------
    _fk = [k for k in sonuc[-1] if k.startswith("fim_")]
    if _fk:
        _bol = [b for b in ("comp", "ent", "ent_yok", "ood")
                if f"fim_ozne1_{b}" in sonuc[-1]]
        print(f"\n{'='*62}")
        print("BOSLUK DOLDURMA -- AYNI zincirler, BASKA yonden sorulmus")
        print("  !! HUKUM VERMEZ. Birincil olcu yukaridaki ILERI yon;")
        print("     olcme izi f4ce53fd1555 ile model_05 kiyasi YALNIZ orada")
        print("     gecerli.")
        print("  IKI SUTUN DA TEK JETONLUK bosluk sorar -- egitimle uyumlu")
        print("  (egitimdeki 187.296 boslugun 187.296'si tek jetonluk).")
        print("  ozne1 = oznenin SON jetonu (soyad). model_06'nin kayitli")
        print("  `fim_ozne_` sayilari OZNENIN TAMAMINI bosaltiyordu ve")
        print("  DAGITIM DISIYDI -- bu sutunla KIYASLANMAZ, adi da ayri.")

        print(f"  {'pencere':<18}"
              + "".join(f"{'ozne1:'+b:>13}{'rel:'+b:>13}" for b in _bol))
        for r in sonuc:
            print(f"  {r['pencere']:<18}"
                  + "".join(f"{r.get('fim_ozne1_'+b, float('nan')):>13.4f}"
                            f"{r.get('fim_rel_'+b, float('nan')):>13.4f}"
                            for b in _bol))

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

    # DOSYA ADINDA GENISLIK VAR. Yoksa `--genislik 4` ile alinan okuma,
    # sonraki `--genislik 2` kosusu tarafindan SESSIZCE eziliyordu ve
    # kaybolduguna dair hicbir iz kalmiyordu. Ayni genislikle tekrar
    # kosmak ayni sayilari uretir (ayni anlik goruntuler, ayni kod), yani
    # o durumda ezmenin bir bedeli yok.
    yol = a.cikti or os.path.join(
        a.klasor, f"pencere_{ayar.ad}_t{ayar.tohum}_g{a.genislik}.json")
    # M._yaz_json: atomik (.tmp -> replace). Egitim hala kosuyorken bu dosya
    # okunabilir; yarim yazilmis json JSONDecodeError verir (olculdu).
    M._yaz_json(yol, dict(
        ad=ayar.ad, tohum=ayar.tohum, klasor=a.klasor, genislik=a.genislik,
        parmak_izi=iz, ayar=ayar.sozluk(), kunye=kunye, pencereler=sonuc,
        kapilar=[dict(ad=x, kural=y, gecti=z) for x, y, z in kapilar]))
    print(f"\n-> {yol}")


if __name__ == "__main__":
    main()
