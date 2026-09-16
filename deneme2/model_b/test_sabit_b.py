# -*- coding: utf-8 -*-
"""test_sabit_b — model_b ailesinin KILIT TESTI.

    python test_sabit_b.py

Once `model_a/test_sabit.py`yi kosturur (taban DEGISMEDI mi), sonra
model_b'ye ozgu DORT degismezi denetler. Defterin 3. hucresi bunu
`test_*.py` olarak bulup kosturuyor; duserse egitim BASLAMAZ.

EN ONEMLI KONTROL 2 NUMARA: `dar_alfa=0` iken ModelB, model_a.Model ile
BIT DUZEYINDE ayni cikti vermeli. Vermezse `model_a9` ile kiyas gecersiz
olur -- cunku o zaman fark yalniz darbogazdan degil, mimarinin kendisinden
de geliyor demektir.
"""
from __future__ import annotations

import io, os, subprocess, sys

_B = os.path.dirname(os.path.abspath(__file__))
_A = os.path.join(os.path.dirname(_B), "model_a")
sys.path[:0] = [_B, _A, os.path.dirname(_B)]

import numpy as np                                           # noqa: E402
import torch                                                 # noqa: E402
import torch.nn.functional as F                              # noqa: E402
import model_a as M                                          # noqa: E402
import model_b                                               # noqa: E402
from model_b import ModelB                                   # noqa: E402

_iyi = _kotu = 0


def _bak(ad, ok, not_=""):
    global _iyi, _kotu
    if ok:
        _iyi += 1
        if os.environ.get("AYRINTI"):
            print(f"    ok   {ad}")
    else:
        _kotu += 1
        print(f"  !! BOZUK {ad}  {not_}")


def main():
    print("=== 0) TABAN (model_a/test_sabit.py) ===")
    r = subprocess.run([sys.executable, os.path.join(_A, "test_sabit.py")],
                       cwd=_A, capture_output=True, text=True,
                       env={**os.environ})
    print(r.stdout.rstrip()[-1500:] if r.stdout else "(cikti yok)")
    _bak("model_a kilidi", r.returncode == 0, f"cikis {r.returncode}")

    print("\n=== 1) model_b'nin DUGMESI ===")
    import model_a9
    f = set(model_a9.AYAR.fark(model_b.AYAR)) | {"ad"}
    _bak(f"model_b dugmeleri {sorted(f)}", sorted(f) == ["ad", "dar_alfa"])

    import model_b1, model_a8
    f1 = set(model_b.AYAR.fark(model_b1.AYAR)) | {"ad"}
    _bak(f"model_b1 dugmeleri (model_b'den) {sorted(f1)}",
         sorted(f1) == ["ad", "veri_ad"])
    # ASIL KIYAS: model_a8'den TEK FARK darbogaz olmali. Bozulursa
    # "darbogazin etkisi" iddiasi gecersiz olur -- baska bir sey de
    # degismis demektir.
    f2 = set(model_a8.AYAR.fark(model_b1.AYAR)) | {"ad"}
    _bak(f"model_b1 <-> model_a8 farki {sorted(f2)}",
         sorted(f2) == ["ad", "dar_alfa"],
         "model_a8 kiyasi TEK DUGME olmali")
    for _g in ("AYAR", "egit", "fark_bas"):
        _bak(f"model_b1.{_g} var", hasattr(model_b1, _g), "kos.py duser")

    import model_b2, model_b3, model_b4, model_b5, model_b6, model_b7
    import model_b8, model_b9, model_b10, model_b11, model_b12
    import model_b13, model_b14, model_b15
    f7 = set(model_b1.AYAR.fark(model_b6.AYAR)) | {"ad"}
    _bak(f"model_b6 <-> model_b1 farki {sorted(f7)}",
         sorted(f7) == ["ad", "jeton_ad"],
         "model_b6 SADECE kodlamayi degistirmeli")
    _bak('model_b6 jeton_ad="tam"', model_b6.AYAR.jeton_ad == "tam",
         f'jeton_ad={model_b6.AYAR.jeton_ad!r}')
    _bak('model_b5 jeton_ad="ilk" (KUSURLU, yeniden uretilebilir)',
         model_b5.AYAR.jeton_ad == "ilk",
         f'jeton_ad={model_b5.AYAR.jeton_ad!r}')
    _bak('model_b1 jeton_ad="" (DEGISMEDI)', model_b1.AYAR.jeton_ad == "")
    # --- model_b7: MAKALENIN GATE'I -----------------------------------
    # Alti kol da alfa=0.5 kostu; makale 5.1/5.3'te alfa=1 kullaniyor.
    # 0.5 secimi EGITILMIS agirliga SONRADAN mudahaleden gelmisti ve o
    # kanit egitime TASINMAZ. Onkayit: belge/onkayit/model_b7.md
    f8 = set(model_b6.AYAR.fark(model_b7.AYAR)) | {"ad"}
    _bak(f"model_b7 <-> model_b6 farki {sorted(f8)}",
         sorted(f8) == ["ad", "dar_alfa"],
         "model_b7 SADECE gate'i degistirmeli")
    _bak("model_b7 dar_alfa=1.0 (MAKALENIN AYARI)",
         model_b7.AYAR.dar_alfa == 1.0, str(model_b7.AYAR.dar_alfa))
    _bak("model_b6 dar_alfa=0.5 (DEGISMEDI)",
         model_b6.AYAR.dar_alfa == 0.5, str(model_b6.AYAR.dar_alfa))
    for _g in ("AYAR", "egit", "fark_bas"):
        _bak(f"model_b7.{_g} var", hasattr(model_b7, _g), "kos.py duser")
    # --- model_b8: YARDIMCI KOPRU KAYBI (TESHIS) -----------------------
    f9 = set(model_b6.AYAR.fark(model_b8.AYAR)) | {"ad"}
    _bak(f"model_b8 <-> model_b6 farki {sorted(f9)}",
         sorted(f9) == ["ad", "kopru_kayip"],
         "model_b8 SADECE yardimci kaybi acmali")
    _bak("model_b8 kopru_kayip=1.0", model_b8.AYAR.kopru_kayip == 1.0,
         str(model_b8.AYAR.kopru_kayip))
    _bak("model_b6 kopru_kayip=0.0 (DEGISMEDI)",
         model_b6.AYAR.kopru_kayip == 0.0, str(model_b6.AYAR.kopru_kayip))
    for _g in ("AYAR", "egit", "fark_bas"):
        _bak(f"model_b8.{_g} var", hasattr(model_b8, _g), "kos.py duser")
    # Kopru pozisyonlari ANA KAYBIN pozisyonlariyla CAKISMAMALI -- caksaydi
    # ayni yerden hem kopru hem cevap istenirdi ve ikisi CELISIRDI.
    _v8 = M.veri_kur(model_b8.AYAR, yaz=lambda *x: None)
    _kp, _nk = M.kopru_hedefi(_v8)
    _X8, _P8, _T8 = M.kodla_2hop(_v8, _v8.tr2[:1])
    _bak(f"kopru pozisyonlari {_kp}, cevap pozisyonlari {list(_P8[0])} "
         "-- CAKISMIYOR", not (set(_kp) & set(int(z) for z in _P8[0])))
    _bak(f"kopru {_nk} token (yuva {_v8.yuva}, en fazla 2 poz)",
         _nk == min(_v8.yuva, 2))
    # Kopru NEDENSEL olarak belirli mi: en kucuk kopru pozisyonu, r1'in
    # yerinden KUCUK OLMAMALI, yoksa model henuz r1'i gormemis olurdu.
    _bak(f"en kucuk kopru pozisyonu {min(_kp)} >= r1 pozisyonu "
         f"{1 + _v8.yuva}", min(_kp) >= 1 + _v8.yuva)
    _X9, _P9, _T9, _k9, _KP9, _KT9 = M.egitim_havuzu(
        model_b8.AYAR, _v8, yaz=lambda *x: None)
    _bak("1hop satirlarinda kopru hedefi -1 (MASKELI)",
         bool((_KT9[:len(_v8.one), 0] < 0).all()))
    _bak("2hop satirlarinda kopru hedefi GECERLI",
         bool((_KT9[len(_v8.one):, 0] >= 0).all()))
    _bak(f"kopru hedefi {int((_KT9[:, 0] >= 0).sum())}/{len(_KT9)} satirda",
         int((_KT9[:, 0] >= 0).sum()) == len(_v8.tr2))
    # --- model_b9: LEARNABLE GATE, TEK BASINA --------------------------
    # model_c uc head ekliyor; ikisi AYNI kolda degisirse sonuc cikarsa
    # hangisinden geldigi AYRILAMAZ. Bu kol gate'i YALNIZ BASINA olcer.
    f10 = set(model_b6.AYAR.fark(model_b9.AYAR)) | {"ad"}
    _bak(f"model_b9 <-> model_b6 farki {sorted(f10)}",
         sorted(f10) == ["ad", "dar_kapi"],
         "model_b9 SADECE gate'i ogrenilebilir yapmali")
    _bak("model_b9 dar_kapi=True (learnable gate)",
         model_b9.AYAR.dar_kapi is True)
    _bak("model_b6 dar_kapi=False (fixed gate, DEGISMEDI)",
         model_b6.AYAR.dar_kapi is False)
    _bak("model_b9 dar_kafa=1 (model_c ile KARISMIYOR)",
         model_b9.AYAR.dar_kafa == 1, str(model_b9.AYAR.dar_kafa))
    for _g in ("AYAR", "egit", "fark_bas"):
        _bak(f"model_b9.{_g} var", hasattr(model_b9, _g), "kos.py duser")
    # 0. ADIMDA FIXED GATE ILE AYNI: sigmoid(0) = 0.5 = dar_alfa.
    # Boyle olmasaydi "gate mi ise yaradi, farkli baslangic mi"
    # ayrilamazdi.
    torch.manual_seed(0)
    _g6 = ModelB(model_b6.AYAR, _v8.vocab)
    torch.manual_seed(0)
    _g9 = ModelB(model_b9.AYAR, _v8.vocab)
    _bak("kapi_w ve kapi_b SIFIRDAN basliyor",
         bool(torch.all(_g9.kapi_w == 0)) and bool(torch.all(_g9.kapi_b == 0)))
    _bak("ek parametre d+1 = 257",
         sum(p.numel() for p in _g9.parameters())
         - sum(p.numel() for p in _g6.parameters()) == model_b6.AYAR.d + 1)
    _g9.load_state_dict(_g6.state_dict(), strict=False)
    _g6.eval(); _g9.eval()
    _xg = torch.randint(0, _v8.vocab, (4, model_b9.AYAR.t_len))
    with torch.no_grad():
        _dg = (_g6(_xg) - _g9(_xg)).abs().max().item()
    _bak(f"0. adimda model_b6 ile fark {_dg:.3e}", _dg == 0.0)
    _g9.train()
    _g9(_xg).sum().backward()
    _bak("gradyan kapi_w'ye AKIYOR",
         _g9.kapi_w.grad is not None and _g9.kapi_w.grad.abs().sum() > 0)
    # --- model_b10: DIL MODELI KAYBI ----------------------------------
    # Alti koldur egittigimiz sey bir SORU-CEVAP basligiydi: logit'lerin
    # %70'i atiliyordu. Bu kol kaybi HER pozisyona yayiyor.
    # TABAN model_b9 -- kullanici karari 16 Eylul ("b10 learnable gate
    # olacak"). Boylece kol YINE TEK DUGME ve merdiven bir basamak
    # uzuyor: b6 -> b9 (dar_kapi) -> b10 (tam_kayip) -> b8 (TAVAN).
    f11 = set(model_b9.AYAR.fark(model_b10.AYAR)) | {"ad"}
    _bak(f"model_b10 <-> model_b9 farki {sorted(f11)}",
         sorted(f11) == ["ad", "tam_kayip"],
         "model_b10 SADECE kaybin NEREDE hesaplandigini degistirmeli")
    f11b = set(model_b6.AYAR.fark(model_b10.AYAR)) | {"ad"}
    _bak(f"model_b10 <-> model_b6 farki {sorted(f11b)} (IKI dugme, "
         f"atfetme b9 uzerinden)",
         sorted(f11b) == ["ad", "dar_kapi", "tam_kayip"])
    _bak("model_b10 tam_kayip=True", model_b10.AYAR.tam_kayip is True)
    _bak("model_b9 tam_kayip=False (DEGISMEDI)",
         model_b9.AYAR.tam_kayip is False)
    _bak("model_b6 tam_kayip=False (DEGISMEDI)",
         model_b6.AYAR.tam_kayip is False)
    _bak("model_b10 dar_kapi=True (model_b9'dan DEVRALINDI)",
         model_b10.AYAR.dar_kapi is True)
    _bak("model_b10 dar_kafa=1, kopru_kayip=0 (c/b8 ile KARISMIYOR)",
         model_b10.AYAR.dar_kafa == 1
         and model_b10.AYAR.kopru_kayip == 0.0)
    _bak("model_b10 MIMARI, model_b9 ile AYNI: ek parametre YOK",
         sum(p.numel() for p in ModelB(model_b10.AYAR, _v8.vocab).parameters())
         == sum(p.numel() for p in ModelB(model_b9.AYAR, _v8.vocab).parameters()),
         "tam_kayip MIMARIYE dokunmaz, yalniz kaybin NEREDE oldugunu degistirir")
    for _g in ("AYAR", "egit", "fark_bas"):
        _bak(f"model_b10.{_g} var", hasattr(model_b10, _g), "kos.py duser")

    # CEVAP KAYBI, DIL KAYBININ ALT KUMESI MI -- yani tam_kayip yeni bir
    # sozlesme getirmiyor, mevcut next-token sozlesmesini BUTUN
    # pozisyonlara yayiyor. Tutmazsa iki kayip AYRI SEY olur ve
    # `kayip_ana` sutunu model_b6 ile kiyaslanamaz.
    _Xk, _Pk, _Tk = M.kodla_2hop(_v8, _v8.tr2[:32])
    _bak("cevap pozisyonlari next-token sozlesmesine uyuyor (2hop)",
         all(_Xk[i, _Pk[i, j] + 1] == _Tk[i, j]
             for i in range(len(_Xk)) for j in range(_Pk.shape[1])))
    _X1, _P1, _T1 = M.kodla_1hop(_v8, _v8.one[:32])
    _bak("cevap pozisyonlari next-token sozlesmesine uyuyor (1hop)",
         all(_X1[i, _P1[i, j] + 1] == _T1[i, j]
             for i in range(len(_X1)) for j in range(_P1.shape[1])))
    _bak("PAD dizinin ORTASINDA yok (ignore_index=PAD guvenli)",
         all((lambda nz: len(nz) == 0 or nz.max() - nz.min() + 1 == len(nz))
             (np.nonzero(r)[0]) for r in np.concatenate([_Xk, _X1])))
    torch.manual_seed(0)
    _n10 = ModelB(model_b10.AYAR, _v8.vocab).eval()
    _xk = torch.from_numpy(_Xk)
    with torch.no_grad():
        _lgt = _n10(_xk)
    _arr = torch.arange(_xk.shape[0])[:, None]
    _ana = F.cross_entropy(
        _lgt[_arr, torch.from_numpy(_Pk)].float().reshape(-1, _lgt.shape[-1]),
        torch.from_numpy(_Tk).reshape(-1))
    _hep = F.cross_entropy(
        _lgt[:, :-1].float().reshape(-1, _lgt.shape[-1]),
        _xk[:, 1:].reshape(-1), ignore_index=M.PAD,
        reduction="none").reshape(_xk.shape[0], -1)
    _sec = _hep[_arr, torch.from_numpy(_Pk)].mean()
    _bak(f"cevap kaybi DIL kaybinin ALT KUMESI "
         f"({abs(_ana.item() - _sec.item()):.1e})",
         abs(_ana.item() - _sec.item()) < 1e-5)
    _pay = _Pk.size / int((_xk[:, 1:] != M.PAD).sum())
    # ON KOSUL OLCUMU DEPODA MI: model_b10'un onkayit 11.1'deki sayisi
    # bir oturumluk betikten degil, asama1'den uretilmeli.
    import asama1 as _A1
    _bak("asama1.birim_teshisi var (onkayit model_b10.md 11.1)",
         hasattr(_A1, "birim_teshisi"),
         "ON KOSUL olcumu depoda DEGIL -> tekrar uretilemez")
    _bak("asama1 --birim bayragi var",
         "--birim" in io.open(_A1.__file__, encoding="utf-8").read())

    _bak(f"SEYRELME olculdu: cevap gorevi kayip terimlerinin "
         f"{_pay:.0%}'i", 0.2 < _pay < 0.45,
         "cevap gorevine dusen gradyan ~3 kat seyreliyor -- "
         "onkayit model_b10.md 5'te CONFOUND olarak yazili")

    # --- model_b11: BELGE -- iki olgu AYNI DIZIDE ----------------------
    f12 = set(model_b10.AYAR.fark(model_b11.AYAR)) | {"ad"}
    _bak(f"model_b11 <-> model_b10 farki {sorted(f12)}",
         sorted(f12) == ["ad", "belge_pay"],
         "model_b11 SADECE satirda kac olgu oldugunu degistirmeli")
    _bak("model_b11 belge_pay=0.5", model_b11.AYAR.belge_pay == 0.5)
    _bak("model_b10 belge_pay=0.0 (DEGISMEDI)",
         model_b10.AYAR.belge_pay == 0.0)
    _bak(f"model_b11 t_len 20 (model_b10 {model_b10.AYAR.t_len}, DEGISMEDI)",
         model_b11.AYAR.t_len == 20 and model_b10.AYAR.t_len == 11)
    _bak("model_b11 tam_kayip=True (belge_pay BUNU GEREKTIRIR)",
         model_b11.AYAR.tam_kayip is True)
    for _g in ("AYAR", "egit", "fark_bas"):
        _bak(f"model_b11.{_g} var", hasattr(model_b11, _g), "kos.py duser")

    # belge_pay tam_kayip GEREKTIRIYOR mu -- assert GERCEKTEN atiyor mu
    _v11 = M.veri_kur(model_b11.AYAR, yaz=lambda *a: None)
    try:
        M.egitim_havuzu(model_b11.AYAR.degistir(tam_kayip=False), _v11,
                        yaz=lambda *a: None)
        _ok = False
    except AssertionError:
        _ok = True
    _bak("belge_pay tam_kayip GEREKTIRIR (assert ATIYOR)", _ok,
         "kayip yalniz cevapta olsaydi belgenin ORTASI ogrenilmezdi")

    # BELGE SATIRI: iki olgu, kopru DIZIDE, ve SIZINTI YOK
    _z = _v11.tr2[0]
    _Xb, _Pb, _Tb = M.kodla_belge(_v11, [_z])
    _jb = set(int(t) for t in _Xb[0])
    _bak("BELGE satiri: kopru DIZIDE yazili",
         set(int(t) for t in M._e(_v11, _z[3])) <= _jb,
         "belgenin amaci tam BU: kopru baglamda GORULSUN")
    _bak("BELGE satiri: iki [S1] cercevesi var",
         int((_Xb[0] == M.Q1).sum()) == 2, str(int((_Xb[0] == M.Q1).sum())))
    _bak(f"BELGE satiri t_len'e SIGIYOR ({int((_Xb[0] != M.PAD).sum())}"
         f"/{_v11.t_len})",
         int((_Xb[0] != M.PAD).sum()) <= _v11.t_len)
    _bak("BELGE cevabi 2. olgunun cevabi",
         list(_Tb[0]) == [int(t) for t in M._e(_v11, _z[4])])

    _X11, _P11, _T11, _k11, _KP11, _KT11 = M.egitim_havuzu(
        model_b11.AYAR, _v11, yaz=lambda *a: None)
    _n_bel = int((_X11 == M.Q1).sum(1) == 2).sum() if False else int(
        ((_X11 == M.Q1).sum(1) == 2).sum())
    _bak(f"havuzun %{100*_n_bel/len(_X11):.0f}'i BELGE ({_n_bel}/{len(_X11)})",
         0.45 < _n_bel / len(_X11) < 0.55)
    _bak("BELGE satirlarinda kopru hedefi -1 (maskeli)",
         bool((_KT11[((_X11 == M.Q1).sum(1) == 2)][:, 0] < 0).all()))
    _sinav11 = {(x[0], x[1], x[2]) for lst in
                (_v11.comp, _v11.ent, _v11.ent_yok, _v11.ent_arama, _v11.ood)
                for x in lst}
    _bak("SIZINTI YOK: hicbir SINAV zinciri tr2'de degil",
         not any((x[0], x[1], x[2]) in _sinav11 for x in _v11.tr2),
         "belgeler tr2'den kuruluyor; tr2 sinavla kesisirse SIZINTI olur")
    _bak(f"olcme izi DEGISMEDI ({M.olcme_izi(M.olcme_listeleri(model_b11.AYAR, _v11))})",
         M.olcme_izi(M.olcme_listeleri(model_b11.AYAR, _v11)) == "57cf60a5e9af")

    # --- model_b12: PAKET KOL -- wd + LR programi BIRLIKTE -------------
    # Kullanici karari 16 Eylul: "LR ve wd = 0,1 ayni anda degissin."
    # ITIRAZ EDILDI, karar TEKRARLANDI. Onkayit 0: atfetme YAPILAMAZ,
    # olumluysa bisect kollari (b12a yalniz wd, b12b yalniz cosine).
    f13 = set(model_b11.AYAR.fark(model_b12.AYAR)) | {"ad"}
    _bak(f"model_b12 <-> model_b11 farki {sorted(f13)}",
         sorted(f13) == ["ad", "sabit_lr", "wd"],
         "PAKET KOL: tam IKI dugme, ucuncusu OLMAMALI")
    _bak(f"model_b12 wd=0.1 (model_b11 {model_b11.AYAR.wd}, DEGISMEDI)",
         model_b12.AYAR.wd == 0.1 and model_b11.AYAR.wd == 0.5,
         "referanslarin DORDU de 0.1: nanoGPT, Pythia-70m/160m, Qwen2.5 SFT")
    _bak("model_b12 sabit_lr=False (model_b11 True, DEGISMEDI)",
         model_b12.AYAR.sabit_lr is False
         and model_b11.AYAR.sabit_lr is True)
    _bak("model_b12 belge_pay/tam_kayip/dar_kapi model_b11'den DEVRALINDI",
         model_b12.AYAR.belge_pay == 0.5 and model_b12.AYAR.tam_kayip is True
         and model_b12.AYAR.dar_kapi is True)
    _bak("model_b12 MIMARI, model_b11 ile AYNI: ek parametre YOK",
         sum(p.numel() for p in ModelB(model_b12.AYAR, _v11.vocab).parameters())
         == sum(p.numel() for p in ModelB(model_b11.AYAR, _v11.vocab).parameters()))
    _bak("Ayar alan sayisi 42 (ek_kip + bicim, model_b15)",
         len(vars(M.Ayar())) == 42, str(len(vars(M.Ayar()))))
    for _g in ("AYAR", "egit", "fark_bas"):
        _bak(f"model_b12.{_g} var", hasattr(model_b12, _g), "kos.py duser")

    # COSINE PROGRAMI: gercekten AZALIYOR mu, ve SIFIRA mi iniyor
    # (bilinen sapma -- referanslar lr/10'da durur; onkayit model_b12.md 3)
    import math as _mt

    def _lr(a, adim):
        if adim < a.isinma:
            return a.lr * adim / max(1, a.isinma)
        if a.sabit_lr:
            return a.lr
        _alt = a.lr / 10.0
        _k = 0.5 * (1 + _mt.cos(
            _mt.pi * (adim - a.isinma) / max(1, a.adim - a.isinma)))
        return _alt + _k * (a.lr - _alt)

    _A12 = model_b12.AYAR
    _dizi = [_lr(_A12, x) for x in (2000, 5000, 10000, 15000, 20000)]
    _bak("cosine MONOTON azaliyor (isinmadan sonra)",
         all(x > y for x, y in zip(_dizi, _dizi[1:])),
         "  ".join(f"{x:.6f}" for x in _dizi))
    _bak(f"cosine TABANI lr/10 ({_dizi[-1]:.6f}), SIFIR DEGIL",
         abs(_dizi[-1] - _A12.lr / 10) < 1e-12,
         "nanoGPT/Pythia/Qwen SFT ucu de lr/10'da durur -- min_lr")
    _bak("cosine TEPESI lr (isinma bitisinde)",
         abs(_dizi[0] - _A12.lr) < 1e-12, f"{_dizi[0]:.6f}")
    _bak("model_b11'in LR'i SABIT kaldi (DEGISMEDI)",
         _lr(model_b11.AYAR, 20000) == model_b11.AYAR.lr)

    # --- model_b13: IDENTITY BRIDGE (arXiv 2509.24653) -----------------
    # Kahin testimiz makalenin teshisini DOGRULADI: kopru elden verilse
    # bile bilesim olmuyor ("contextual disconnect between the input and
    # output spaces"). Cozumleri: b -> b "zero-hop" satirlari.
    f14 = set(model_b10.AYAR.fark(model_b13.AYAR)) | {"ad"}
    _bak(f"model_b13 <-> model_b10 farki {sorted(f14)}",
         sorted(f14) == ["ad", "ident_frac", "ident_kip", "sabit_lr", "wd"],
         "EN IYI TAHMIN kolu: ident cifti + wd + cosine (onkayit 0)")
    _bak("model_b13 ident_kip='q1' (SIFIR-HOP, e -> e)",
         model_b13.AYAR.ident_kip == "q1",
         "q2son GRAFTAN kopruyu soyler ve ENT tanimini BOZAR")
    _bak("model_b13 ident_frac=0.2", model_b13.AYAR.ident_frac == 0.2)
    _bak("model_b10 ident_frac=0.0 (DEGISMEDI)",
         model_b10.AYAR.ident_frac == 0.0)
    _bak(f"model_b13 t_len 11 -- kimlik satiri SIGIYOR, belge_pay=0",
         model_b13.AYAR.t_len == 11 and model_b13.AYAR.belge_pay == 0.0)
    for _g in ("AYAR", "egit", "fark_bas"):
        _bak(f"model_b13.{_g} var", hasattr(model_b13, _g), "kos.py duser")

    # KIMLIK SATIRI: cok jetonlu, dogru bicimli, GRAF BILGISI YOK
    _v13 = M.veri_kur(model_b13.AYAR, yaz=lambda *a: None)
    _Xk, _Pk, _Tk = M.kodla_kimlik_q1(_v13, range(_v13.n_ent))
    _bak(f"kimlik satiri COK JETONLU ({int((_Xk[0] != M.PAD).sum())} jeton)",
         int((_Xk[0] != M.PAD).sum()) == 2 * _v13.yuva + 4)
    _bak("kimlik satiri t_len'e SIGIYOR",
         int((_Xk != M.PAD).sum(1).max()) <= _v13.t_len)
    _bak("kimlik: next-token sozlesmesi",
         all(_Xk[i, _Pk[i, j] + 1] == _Tk[i, j]
             for i in range(0, len(_Xk), 97) for j in range(_v13.yuva)))
    _bak("kimlik: cevap = SORUNUN KENDISI (e -> e)",
         bool((_Xk[:, 1:1 + _v13.yuva] == _Tk).all()))
    _bak(f"kimlik: BUTUN varliklar kapsandi ({len(_Xk)}/{_v13.n_ent})",
         len(_Xk) == _v13.n_ent)
    # GRAF BILGISI YOK: satirda IKINCI bir varlik ya da ILISKI gecmiyor
    _iliski_var = bool(((_Xk >= M.REL_OFF) & (_Xk < _v13.ent_off)).any())
    _bak("kimlik satirinda ILISKI jetonu YOK -> graf bilgisi KULLANILMIYOR",
         not _iliski_var, "kopru_kayip'tan (model_b8) FARKI tam BU")

    # HAVUZ: kimlik payi ve SEYRELME
    _X13, _P13, _T13, _k13, _KP13, _KT13 = M.egitim_havuzu(
        model_b13.AYAR, _v13, yaz=lambda *a: None)
    _kim_n = int((_X13 == M.IDENT).any(1).sum())
    _bak(f"havuzun %{100*_kim_n/len(_X13):.0f}'i KIMLIK "
         f"({_kim_n}/{len(_X13)})", 0.15 < _kim_n / len(_X13) < 0.25,
         "model_b11 SEYRELMENIN zararini olctu; pay 0.2'de tutuldu")
    _bak("KIMLIK satirlarinda kopru hedefi -1 (maskeli)",
         bool((_KT13[(_X13 == M.IDENT).any(1)][:, 0] < 0).all()))

    # --- model_b14: VERI YOGUNLUGU (veri_okul3) ------------------------
    # TEK DUGME: veri_ad. wd/cosine/ident model_b13'ten DEVRALINIR.
    # Sebep OLCULDU: model_b13'te one/seen 1.0000 oldu, comp 0.0577
    # kaldi -- "ogrenemedi" mazereti kapandi. Literatur (Wang 2024;
    # arXiv 2505.17923) tek bir yere isaret ediyor: phi.
    f15 = set(model_b13.AYAR.fark(model_b14.AYAR)) | {"ad"}
    _bak(f"model_b14 <-> model_b13 farki {sorted(f15)}",
         sorted(f15) == ["ad", "veri_ad"],
         "model_b14 SADECE GRAFI degistirmeli")
    _bak('model_b14 veri_ad="veri_okul3"',
         model_b14.AYAR.veri_ad == "veri_okul3")
    _bak('model_b13 veri_ad="veri_okul2" (DEGISMEDI)',
         model_b13.AYAR.veri_ad == "veri_okul2")
    _bak("model_b14 ident/wd/cosine model_b13'ten DEVRALINDI",
         model_b14.AYAR.ident_frac == 0.2 and model_b14.AYAR.wd == 0.1
         and model_b14.AYAR.sabit_lr is False,
         "CLAUDE.md kural 4")
    _bak("model_b14 t_len 11 (DEGISMEDI)", model_b14.AYAR.t_len == 11)
    for _g in ("AYAR", "egit", "fark_bas"):
        _bak(f"model_b14.{_g} var", hasattr(model_b14, _g), "kos.py duser")
    import veri_okul2 as _V2, veri_okul3 as _V3, veri_okul as _VO
    _G2, _G3 = _V2.kur(0), _V3.kur(0)
    _bak("veri_okul3: |R| = 17 (YENI SEMBOL YOK -- arama uzayi 17^2 SABIT)",
         len(_G3["iliski"]) == 17 and list(_G3["iliski"]) == list(_G2["iliski"]),
         "arXiv 2505.17923: |R|^k veri acliginin ASIL kaynagi")
    _bak("veri_okul2 olgulari veri_okul3'te BIREBIR duruyor",
         all(_G3["olgu"].get(k) == v for k, v in _G2["olgu"].items()),
         "kontrol (model_b13) ancak boyle gecerli")
    _z2, _z3 = _VO.zincirler(_G2), _VO.zincirler(_G3)
    _p2, _p3 = len(_z2) / len(_G2["olgu"]), len(_z3) / len(_G3["olgu"])
    _bak(f"phi TAVANI 7,04 -> 9,57", abs(_p3 - 9.57) < 0.01 and abs(_p2 - 7.04) < 0.01,
         f"{_p2:.2f} -> {_p3:.2f}")
    _bak("veri_okul3 YENI turetilebilir cift URETMEDI",
         [x for x in _V3.turetilebilir(_G3)
          if (x[0], x[1]) not in _V3.GEREKTIRIR] == [],
         "iki iliski ayni eslemeyse model birini digerinden OKUR")
    _v14 = M.veri_kur(model_b14.AYAR, yaz=lambda *a: None)
    _bak(f"model_b14 egitim phi'si YUKSELDI",
         _v14.phi > _v13.phi, f"{_v13.phi:.2f} -> {_v14.phi:.2f}")
    _bak("model_b14 varlik sayisi DEGISMEDI (2120)",
         _v14.n_ent == _v13.n_ent == 2120)
    _bak("model_b14 vocab DEGISMEDI", _v14.vocab == _v13.vocab)

    # --- model_b15: BICIM CESITLILIGI (ek isaretleyicili dil) ----------
    # BIR CIFT DUGME: ek_kip + bicim. ek_kip olmadan bicim ANLAMSIZ.
    f16 = set(model_b14.AYAR.fark(model_b15.AYAR)) | {"ad"}
    _bak(f"model_b15 <-> model_b14 farki {sorted(f16)}",
         sorted(f16) == ["ad", "bicim", "ek_kip", "veri_ad"],
         "UC dugme: olcek + kodlama cifti. ATFETME YAPILAMAZ, ve artik "
         "hedef atfetme DEGIL (CLAUDE.md: KIYAS ARTIK ARKA PLANDA).")
    _bak('model_b15 veri_ad="veri_okul4" (1060 varlik)',
         model_b15.AYAR.veri_ad == "veri_okul4")
    _bak('model_b14 veri_ad="veri_okul3" (DEGISMEDI)',
         model_b14.AYAR.veri_ad == "veri_okul3")
    _bak('model_b15 ek_kip="tr", bicim=3',
         model_b15.AYAR.ek_kip == "tr" and model_b15.AYAR.bicim == 3)
    _bak('model_b14 ek_kip="" (DEGISMEDI)', model_b14.AYAR.ek_kip == "")
    _bak("model_b15 t_len 17 (model_b14 11, DEGISMEDI)",
         model_b15.AYAR.t_len == 17 and model_b14.AYAR.t_len == 11)
    _bak("model_b15 ident/wd/cosine model_b14'ten DEVRALINDI",
         model_b15.AYAR.ident_frac == 0.2 and model_b15.AYAR.wd == 0.1
         and model_b15.AYAR.sabit_lr is False, "CLAUDE.md kural 4")
    for _g in ("AYAR", "egit", "fark_bas"):
        _bak(f"model_b15.{_g} var", hasattr(model_b15, _g), "kos.py duser")
    _v15 = M.veri_kur(model_b15.AYAR, yaz=lambda *a: None)
    _bak("model_b15 varlik 1060 (veri_okul4, YARIM havuz)",
         _v15.n_ent == 1060, str(_v15.n_ent))
    _bak("EK jetonlari SOZLUGUN SONUNDA -- ent_off/REL_OFF KAYMADI",
         _v15.ek0 == _v15.vocab - 4 and _v15.ent_off == _v13.ent_off,
         f"ek0={_v15.ek0} vocab={_v15.vocab} ent_off={_v15.ent_off}")
    # phi OLCEK-DEGISMEZ: varlik yariya indi ama TAVAN degismedi.
    import veri_okul3 as _V3, veri_okul4 as _V4, veri_okul as _VO
    _G3, _G4 = _V3.kur(0), _V4.kur(0)
    _p3 = len(_VO.zincirler(_G3)) / len(_G3["olgu"])
    _p4 = len(_VO.zincirler(_G4)) / len(_G4["olgu"])
    _bak("phi TAVANI OLCEK-DEGISMEZ: okul3 = okul4 = 9,57",
         abs(_p3 - _p4) < 0.01 and abs(_p4 - 9.57) < 0.01,
         f"okul3 {_p3:.4f}  okul4 {_p4:.4f}")
    _bak("veri_okul4 varlik 1060, olgu 10.960",
         sum(_G4["n"].values()) == 1060 and len(_G4["olgu"]) == 10960)
    _bak("veri_okul4 SEMA veri_okul3 ile AYNI (|R|=17, ayni acilimlar)",
         _V4.SEMA == _V3.SEMA and len(_G4["iliski"]) == 17)
    _bak("veri_okul4 YENI turetilebilir cift URETMEDI",
         [x for x in _V4.turetilebilir(_G4)
          if (x[0], x[1]) not in _V4.GEREKTIRIR] == [])
    _bak("model_b15 olcme izi DEGISTI (b14 b7a4e954b2a2) -- NOT DUSULDU",
         M.olcme_izi(M.olcme_listeleri(model_b15.AYAR, _v15)) != "b7a4e954b2a2",
         "varlik kumesi yarilandi -> sinav zincirleri BASKA. Kol iptal "
         "DEGIL: CLAUDE.md 'KIYAS ARTIK ARKA PLANDA'.")
    # --- uc bicim AYNI OLGUYU mu tasiyor -------------------------------
    _T15 = [M.kodla_1hop(_v15, _v15.one[:2000], _b)[2] for _b in range(3)]
    _bak("uc bicim AYNI cevabi tasiyor (1-hop, 2000 ornek)",
         all((_T15[0] == _T15[_b]).all() for _b in (1, 2)))
    _X15 = [M.kodla_1hop(_v15, _v15.one[:2000], _b)[0] for _b in range(3)]
    _bak("uc bicim DIZI olarak FARKLI (hicbiri ayni degil)",
         not any((_X15[0][i] == _X15[_b][i]).all()
                 for i in range(2000) for _b in (1, 2)))
    _P15, _TT15 = M.kodla_2hop(_v15, _v15.tr2[:500], 0)[1:]
    _XX15 = M.kodla_2hop(_v15, _v15.tr2[:500], 0)[0]
    _bak("next-token sozlesmesi tutuyor (2-hop, bicim 0)",
         all(_XX15[i][_P15[i][j] + 1] == _TT15[i][j]
             for i in range(500) for j in range(_v15.yuva)))
    _K15, _N15, _S15, _D15 = M._ekler(_v15)
    _kotu = 0
    for _b in range(3):
        _XB = M.kodla_1hop(_v15, _v15.one[:300], _b)[0]
        for _r in _XB:
            _d = list(_r)
            for _j, _t in enumerate(_d):
                if _t == _K15 and _d[_j + 1] not in (_N15, _D15):
                    _kotu += 1
                if _t == _S15 and not (M.REL_OFF <= _d[_j - 1] < _v15.ent_off):
                    _kotu += 1
    _bak("' her zaman <NIN>/<DIR> ONUNDE, <SI> her zaman ILISKI ardinda",
         _kotu == 0, f"{_kotu} hata")
    _X15h, _P15h, _T15h, _kim15, _kp15, _KT15 = M.egitim_havuzu(
        model_b15.AYAR, _v15, yaz=lambda *a: None)
    # !! `egitim_havuzu` kimligi COGALTILMADAN ONCEKI haliyle donuyor
    # (2120); havuzdakinin kac kati oldugu ident_frac'tan duser. O
    # yuzden kimlik satir sayisi ARTIKTAN cikarilir.
    _kim_n = len(_X15h) - (len(_v15.one) + len(_v15.tr2)) * 3
    _bak("BICIM cogaltmasi OLGU **ve** SORU satirlarinda",
         _kim_n > 0 and _kim_n % len(_kim15[0]) == 0,
         f"havuz {len(_X15h)} = (olgu {len(_v15.one)} + 2hop "
         f"{len(_v15.tr2)}) x3 + kimlik {_kim_n}")
    _bak("PAD dizinin ORTASINDA yok (ek_kip)",
         all((_X15h[i][int((_X15h[i] == M.PAD).argmax()):].sum() == 0)
             for i in range(0, len(_X15h), 1013) if (_X15h[i] == M.PAD).any()))
    # bicim>1 + kopru_kayip REDDEDILMELI: kopru pozisyonu bicimden
    # bicime degisiyor, tek pozisyon listesi YANLIS satira duserdi.
    try:
        M.egitim_havuzu(model_b15.AYAR.degistir(kopru_kayip=1.0), _v15,
                        yaz=lambda *a: None)
        _ok = False
    except AssertionError:
        _ok = True
    _bak("bicim>1 + kopru_kayip REDDEDILIYOR (assert ATIYOR)", _ok)
    # ek_kip olmadan bicim REDDEDILMELI
    try:
        M.egitim_havuzu(model_b14.AYAR.degistir(bicim=3), _v13,
                        yaz=lambda *a: None)
        _ok = False
    except AssertionError:
        _ok = True
    _bak("ek_kip'siz bicim>1 REDDEDILIYOR (eksiz dilde sira ANLAMI BOZAR)",
         _ok)

    # TEK JETONLU yol BOZULMADI mi (model_b1)
    _v1 = M.veri_kur(model_b1.AYAR, yaz=lambda *a: None)
    _X1, _P1, _T1 = M.kodla_kimlik_q1(_v1, range(5))
    _bak("TEK JETONLU kimlik hala calisiyor (model_b1, yuva 1)",
         _v1.yuva == 1 and _X1.shape[1] == _v1.t_len
         and bool((_X1[:, 1:2] == _T1).all()))

    for _g in ("AYAR", "egit", "fark_bas"):
        _bak(f"model_b6.{_g} var", hasattr(model_b6, _g), "kos.py duser")
    f6 = set(model_b1.AYAR.fark(model_b5.AYAR)) | {"ad"}
    _bak(f"model_b5 <-> model_b1 farki {sorted(f6)}",
         sorted(f6) == ["ad", "jeton_ad"],
         "model_b5 SADECE kodlamayi degistirmeli, MIMARIYI degil")
    _bak("model_b5 t_len 11", model_b5.AYAR.t_len == 11,
         f"t_len={model_b5.AYAR.t_len}")
    _bak("model_b1 t_len 8 (DEGISMEDI)", model_b1.AYAR.t_len == 8)
    for _g in ("AYAR", "egit", "fark_bas"):
        _bak(f"model_b5.{_g} var", hasattr(model_b5, _g), "kos.py duser")
    f5 = set(model_b1.AYAR.fark(model_b4.AYAR)) | {"ad"}
    _bak(f"model_b4 <-> model_b1 farki {sorted(f5)}",
         sorted(f5) == ["ad", "ort_bas"],
         "model_b4 SADECE geri beslemeli ortalamayi kapatmali")
    _bak("model_b4'te ortalama GERCEKTEN kapali",
         model_b4.AYAR.ort_bas == 0, f"ort_bas={model_b4.AYAR.ort_bas}")
    for _g in ("AYAR", "egit", "fark_bas"):
        _bak(f"model_b4.{_g} var", hasattr(model_b4, _g), "kos.py duser")
    f4 = set(model_b1.AYAR.fark(model_b3.AYAR)) | {"ad"}
    _bak(f"model_b3 <-> model_b1 farki {sorted(f4)}",
         sorted(f4) == ["ad", "dar_sert"],
         "model_b3 SADECE Phi'nin icini degistirmeli")
    for _g in ("AYAR", "egit", "fark_bas"):
        _bak(f"model_b3.{_g} var", hasattr(model_b3, _g), "kos.py duser")
    f3 = set(model_b1.AYAR.fark(model_b2.AYAR)) | {"ad"}
    _bak(f"model_b2 <-> model_b1 farki {sorted(f3)}",
         sorted(f3) == ["ad", "dar_sdpa"],
         "model_b2 SADECE hesap yolunu degistirmeli")
    for _g in ("AYAR", "egit", "fark_bas"):
        _bak(f"model_b2.{_g} var", hasattr(model_b2, _g), "kos.py duser")


    print()
    print("=== 1b) KOS.PY KOSULLARINDA IMPORT (ALT SUREC) ===")
    # 16 EYLUL: model_b1'in ILK Colab kosusu ilk saniyede coktu --
    #   ImportError: cannot import name 'egit' from 'model_a'
    #                (unknown location)
    # cunku `deneme2/model_a` bir KLASOR ve yolu once eklenmezse
    # `import model_a` onu NAMESPACE PAKETI olarak buluyor.
    # YERELDE GORULMEDI: hem duman testi hem bu kilit sys.path'i ELLE
    # kuruyordu. `kos.py` ise YALNIZ aile klasorunu ekler.
    # Artik ALT SURECTE, kos.py'nin TAM kosullariyla deneniyor.
    # HER MODUL KENDI ALT SURECINDE. Ikisini AYNI surecte denemek testi
    # ISE YARAMAZ hale getiriyordu: `model_b` once import edilince yolu
    # yan etki olarak duzeltiyor ve `model_b1` hazir yolu buluyor.
    # `kos.py` ise YALNIZ istenen modulu import eder. Bu kusur bilerek
    # bozulmus bir surumle sinandi: duzeltmeden ONCE test GECIYORDU.
    for _ad in ("model_b", "model_b1", "model_b2", "model_b3",
                "model_b4", "model_b5", "model_b6", "model_b7",
                "model_b8", "model_b9", "model_b10", "model_b11",
                "model_b12", "model_b13", "model_b14", "model_b15"):
        _satir = [
            "import sys, importlib",
            "sys.path.insert(0, %r)" % _B,
            "m = importlib.import_module(%r)" % _ad,
            "assert hasattr(m, 'egit'), 'egit YOK'",
            "assert hasattr(m, 'AYAR'), 'AYAR YOK'",
            "assert hasattr(m, 'fark_bas'), 'fark_bas YOK'",
        ]
        _r2 = subprocess.run([sys.executable, "-c", chr(10).join(_satir)],
                             cwd=os.path.dirname(_B),
                             capture_output=True, text=True)
        _bak(f"{_ad}: kos.py kosullarinda TEK BASINA import",
             _r2.returncode == 0,
             (_r2.stderr.strip().splitlines() or ["?"])[-1][:110])

    print("\n=== 2) dar_alfa=0 -> ModelB, model_a.Model ILE AYNI MI ===")
    # Bu kolun BUTUN kiyasi buna dayaniyor. Ek modul kurulmamali,
    # parametre sayisi degismemeli, cikti BIT DUZEYINDE ayni olmali.
    kapali = model_b.AYAR.degistir(dar_alfa=0.0, dar_kapi=False)
    torch.manual_seed(0); a = M.Model(kapali, 1208)
    torch.manual_seed(0); b = ModelB(kapali, 1208)
    _bak("parametre sayisi", a.n_param() == b.n_param(),
         f"{a.n_param()} != {b.n_param()}")
    _bak("ek modul YOK", not hasattr(b, "dar_norm"))
    x = torch.randint(0, 1208, (8, M.T_LEN))
    a.eval(); b.eval()
    with torch.no_grad():
        d = (a(x) - b(x)).abs().max().item()
    _bak(f"cikti farki {d:.3e}", d == 0.0)

    print("\n=== 3) dar_alfa>0 -> cikti DEGISMELI ===")
    torch.manual_seed(0); c = ModelB(model_b.AYAR, 1208)
    c.eval()
    with torch.no_grad():
        d2 = (a(x) - c(x)).abs().max().item()
    _bak(f"darbogaz etkili {d2:.3e}", d2 > 1e-6)
    _bak("dar_norm kuruldu", hasattr(c, "dar_norm"))
    # sabit gecit -> ogrenilebilir gecit parametresi OLMAMALI
    _bak("sabit gecitte ek parametre yok",
         c.n_param() == a.n_param() + model_b.AYAR.d,   # yalniz RMSNorm.g
         f"{c.n_param()} vs {a.n_param()}+{model_b.AYAR.d}")

    print("\n=== 4) SON TURDA enjeksiyon YOK (alfa(K-1)=0) ===")
    # dongu=1 iken darbogaz HIC calismamali -> Model ile ayni cikmali.
    tek = model_b.AYAR.degistir(dongu=1)
    torch.manual_seed(0); e = M.Model(tek, 1208)
    torch.manual_seed(0); f2 = ModelB(tek, 1208)
    e.eval(); f2.eval()
    with torch.no_grad():
        d3 = (e(x) - f2(x)).abs().max().item()
    _bak(f"dongu=1'de fark {d3:.3e}", d3 == 0.0)

    print()
    print("=== 5) pencere_b KANCASI GERCEKTEN CALISIYOR MU ===")
    # 16 EYLUL: bu yama BIR KEZ SESSIZCE UYGULANMADI. Bir kabuk
    # zincirinde `sed` hata verdi, `&&` koptu, yamayi yazan blok HIC
    # calismadi, ve commit "kanca eklendi" diyerek gecti. Hata ancak
    # UCTAN UCA tur testinde goruldu -- kilit o zaman bakmiyordu.
    # Artik bakiyor.
    import pencere_b, pencere_a
    _bak("pencere_a'da MODEL_SINIFI alani var",
         hasattr(pencere_a, "MODEL_SINIFI"))
    _bak("pencere_b onu ModelB yapti",
         getattr(pencere_a, "MODEL_SINIFI", None) is ModelB,
         f"su an: {getattr(pencere_a, 'MODEL_SINIFI', 'ALAN YOK')}")
    _bak("olc() sabit M.Model KULLANMIYOR",
         "MODEL_SINIFI or M.Model" in
         io.open(os.path.join(_A, "pencere_a.py"), encoding="utf-8").read())

    # 16 EYLUL: `tani_a` da sabit `M.Model` kullaniyordu -- model_b anlik
    # goruntusunu "Unexpected key(s): dar_norm.g" ile REDDEDERDI ve bu
    # ancak tani KOSULUNCA gorulurdu. pencere_a'daki hatanin AYNISI.
    import tani_b, tani_a
    _bak("tani_a'da MODEL_SINIFI alani var", hasattr(tani_a, "MODEL_SINIFI"))
    _bak("tani_b onu ModelB yapti",
         getattr(tani_a, "MODEL_SINIFI", None) is ModelB,
         f"su an: {getattr(tani_a, 'MODEL_SINIFI', 'ALAN YOK')}")
    _bak("tani_a sabit M.Model KULLANMIYOR",
         "MODEL_SINIFI or M.Model" in
         io.open(os.path.join(_A, "tani_a.py"), encoding="utf-8").read())

    print()
    print("=== 6) SDPA Phi, NAIF Phi ILE AYNI FONKSIYON MU ===")
    # model_b2'nin BUTUN iddiasi buna dayaniyor: ayni fonksiyon, ucuz
    # hesap. Bozulursa model_b2 "hizli" degil "BASKA BIR MODEL" olur.
    # BIT DUZEYINDE esitlik BEKLENMIYOR -- fp16'da toplama sirasi farkli.
    # Onkayit model_b2.md §4: fp32'de < 1e-5, fp16'da < 1e-3 (bagil).
    import model_b2
    naif = model_b2.AYAR.degistir(dar_sdpa=False)
    torch.manual_seed(0); m_n = ModelB(naif, 1208).eval()
    torch.manual_seed(0); m_s = ModelB(model_b2.AYAR, 1208).eval()
    _bak("ayni agirlik", all(
        torch.equal(a_, b_) for a_, b_ in zip(m_n.parameters(),
                                              m_s.parameters())))
    _h = torch.randn(4, M.T_LEN, model_b2.AYAR.d)
    with torch.no_grad():
        p_n, p_s = m_n._phi(_h), m_s._phi(_h)
    _d32 = (p_n - p_s).abs().max().item() / p_n.abs().max().item()
    _bak(f"fp32 bagil fark {_d32:.2e}", _d32 < 1e-5, "esik 1e-5")
    # ve UCTAN UCA: ayni x, iki yol
    _x = torch.randint(0, 1208, (4, M.T_LEN))
    with torch.no_grad():
        o_n, o_s = m_n(_x), m_s(_x)
    _de = (o_n - o_s).abs().max().item() / o_n.abs().max().item()
    _bak(f"model ciktisi bagil fark {_de:.2e}", _de < 1e-5, "esik 1e-5")
    # dar_sdpa, darbogaz KAPALIYKEN hicbir sey yapmamali
    _kap = model_b2.AYAR.degistir(dar_alfa=0.0, dar_kapi=False)
    torch.manual_seed(0); m_k = ModelB(_kap, 1208).eval()
    torch.manual_seed(0); m_a = M.Model(_kap, 1208).eval()
    with torch.no_grad():
        _dk = (m_k(_x) - m_a(_x)).abs().max().item()
    _bak(f"dar_alfa=0 + dar_sdpa=1 -> hala BIT AYNI ({_dk:.3e})", _dk == 0.0)

    print()
    print("=== 7) SERT Phi (model_b3) ===")
    # Uc sey denetleniyor:
    #   a) ILERI cikti GERCEKTEN bir gomme satiri mi (yuvarlama oldu mu)
    #   b) GRADYAN akiyor mu (straight-through kopmus olabilir -- o zaman
    #      darbogaz egitilmez ve kol SESSIZCE anlamsizlasir)
    #   c) tau -> 0 limiti gercekten tutuyor mu (kucuk tau ile yumusak
    #      Phi, sert Phi'ye YAKINSAMALI)
    import model_b3
    torch.manual_seed(0); m_s = ModelB(model_b3.AYAR, 1208)
    _h = torch.randn(3, M.T_LEN, model_b3.AYAR.d, requires_grad=True)
    _g = m_s._phi(_h)
    with torch.no_grad():
        _W = m_s.emb.weight
        _en = (_g.reshape(-1, 1, _W.shape[1]) - _W[None]).abs().sum(-1).min(-1)
    _bak(f"cikti bir GOMME SATIRI (en buyuk uzaklik {_en.values.max():.2e})",
         _en.values.max().item() < 1e-5, "yuvarlama OLMAMIS")
    _g.sum().backward()
    _bak("gradyan h'ye AKIYOR (straight-through)",
         _h.grad is not None and _h.grad.abs().sum().item() > 0,
         "STE kopmus -- darbogaz EGITILMEZ")
    # tau -> 0 LIMITI. Tek bir tau'da ESITLIK beklemek YANLIS test --
    # iddia bir LIMIT. Dogru test: tau kuculdukce fark MONOTON kuculsun
    # ve sonunda ihmal edilebilir olsun. (Ilk surum tau=0.001'de esitlik
    # istiyordu ve 2.98e-02 ile dustu; esik gevsetilmedi, TEST duzeltildi.)
    _hd = _h.detach()
    _ref = None; _dizi = []
    with torch.no_grad():
        _sert = m_s._phi(_hd)
        _olc = _sert.abs().max().item()
        for _t in (0.1, 0.01, 0.001, 1e-4, 1e-5):
            torch.manual_seed(0)
            _my = ModelB(model_b3.AYAR.degistir(dar_sert=False, dar_tau=_t),
                         1208)
            _dizi.append((_my._phi(_hd) - _sert).abs().max().item() / _olc)
    _bak("tau kuculdukce fark MONOTON azaliyor  "
         + " > ".join(f"{x:.1e}" for x in _dizi),
         all(a_ > b_ for a_, b_ in zip(_dizi, _dizi[1:])),
         "MONOTON DEGIL -- sert Phi bir limit DEGIL")
    _bak(f"tau=1e-5'te bagil fark {_dizi[-1]:.2e}", _dizi[-1] < 1e-3,
         "sert Phi, tau->0 limitine YAKINSAMIYOR")
    # dar_sert + dar_sdpa BIRLIKTE olmamali
    try:
        ModelB(model_b3.AYAR.degistir(dar_sdpa=True), 1208)
        _bak("dar_sert + dar_sdpa REDDEDILIYOR", False, "assert ATMADI")
    except AssertionError:
        _bak("dar_sert + dar_sdpa REDDEDILIYOR", True)
    # darbogaz kapaliyken dar_sert de bir sey yapmamali
    _k = model_b3.AYAR.degistir(dar_alfa=0.0, dar_kapi=False)
    _x2 = torch.randint(0, 1208, (4, M.T_LEN))
    torch.manual_seed(0); _mk = ModelB(_k, 1208).eval()
    torch.manual_seed(0); _ma = M.Model(_k, 1208).eval()
    with torch.no_grad():
        _dk = (_mk(_x2) - _ma(_x2)).abs().max().item()
    _bak(f"dar_alfa=0 + dar_sert=1 -> hala BIT AYNI ({_dk:.3e})", _dk == 0.0)

    print()
    print("=== 8) VERI DENETIMI -- jeton_ad='tam' kodlamasi ===")
    # Kullanici, 16 Eylul: "biz dogru veri kurup dogru sorulari
    # sormuyorsak zaten bastan yanlis demektir, olctugumuz de yanlis
    # demektir." `model_b5` tam bundan dustu. Bu bolum kodlamayi HER
    # KOSUDA denetler -- bir daha kusurlu kodlamayla kosulmasin.
    import model_b6, numpy as _np
    _v = M.veri_kur(model_b6.AYAR.degistir(n_olcum_max=50),
                    yaz=lambda *a, **k: None)
    _sz = _v.par_ad[0]
    _coz = lambda r: "_".join(_sz[i] for i in r if i)
    import importlib as _il
    _V = _il.import_module(model_b6.AYAR.veri_ad)
    _G = _V.kur(model_b6.AYAR.veri_tohum)
    _E = [a for t in _V.TIPLER for a in _G["ad"][t]]
    _bak(f"gidis-donus: coz(kodla(x)) == x  ({len(_E)} varlik)",
         all(_coz(_v.par[i]) == _E[i] for i in range(len(_E))),
         "KODLAMA GERI DONMUYOR")
    _bak("birebir: farkli varlik -> farkli jeton dizisi",
         len({tuple(r) for r in _v.par}) == len(_E))
    _bak("TEK PAYLASILAN sozluk (butun yuvalar ayni blok)",
         _v.paylasilan and len(set(_v.yuva_ara)) == 1,
         f"yuva_ara={_v.yuva_ara}")
    _bak("hicbir jetonun icinde ALT CIZGI yok",
         not any("_" in p for p in _sz), [p for p in _sz if "_" in p][:3])
    _bak("dolgu <YOK> hep SAGDA",
         all(all(_v.par[i][j] or not any(_v.par[i][k] for k in range(j, _v.yuva))
                 for j in range(_v.yuva)) for i in range(len(_E))))
    _bak("her varligin en az 1 gercek jetonu var",
         all(_v.par[i][0] for i in range(len(_E))))
    _kon = {}
    for i in range(len(_E)):
        for j in range(_v.yuva):
            if _v.par[i][j]:
                _kon.setdefault(int(_v.par[i][j]), set()).add(j)
    _pay = [k for k, s_ in _kon.items() if len(s_) > 1]
    _bak(f"ayni jeton BIRDEN COK yuvada gecebiliyor ({len(_pay)} tane)",
         len(_pay) > 0, "paylasim YOK -> sozluk fiilen ayrik")
    _bak(f"t_len ({model_b6.AYAR.t_len}) 2-hop dizisine yetiyor",
         1 + 2 * _v.yuva + 2 + 1 + 1 <= model_b6.AYAR.t_len)
    # --- BOLME GECERLILIGI ---------------------------------------------
    _tr = {(e, a, b) for e, a, b, _, _ in _v.tr2}
    _bak("comp: egitimde olan UCLU yok (UCLU duzeyi GECERLI)",
         sum((e, a, b) in _tr for e, a, b, _, _ in _v.comp) == 0)
    _i1 = {(e, r) for e, r, _, _, _ in _v.tr2}
    _i2 = {(b, r) for _, _, r, b, _ in _v.tr2}
    _bak("ood: egitim zincirinde gecen KENAR yok (KENAR duzeyi GECERLI)",
         sum(1 for e, r1, r2, b, _ in _v.ood
             if (e, r1) in _i1 or (b, r2) in _i2) == 0)
    _bas = {x[0] for x in _v.tr2}
    _bak("ent: egitimde ZINCIR BASI olan ENT varligi yok",
         sum(1 for e, *_ in _v.ent if e in _bas) == 0)
    for _ad2, _ls in (("comp", _v.comp), ("ent", _v.ent),
                      ("ent_yok", _v.ent_yok), ("ood", _v.ood)):
        _bak(f"{_ad2}: DONUS (cevap == soru varligi) YOK",
             sum(1 for e, _, _, _, a in _ls if e == a) == 0,
             "kopyalamayla gecilebilir!")

    # ===================================================================
    print()
    print("=== 9) TANI YUVA SAYISINDAN BAGIMSIZ MI ===")
    # Neden kilitte: `tani_a` TEK yuvaya sabitti ve jeton_ad ile
    # cokuyordu; `dogruluk()` ise `yuva_ara`yi sartsiz okuyup TEK
    # jetonlu kollarda AttributeError veriyordu. Ikisi de 16 Eylul'de
    # duman testinde yakalandi -- kilitte OLMADIGI icin 78 kontrol
    # GECIYORDU. Simdi her kosuda deneniyor.
    import tani_a as _T                                      # noqa: E402
    import model_b1 as _m1                                   # noqa: E402
    for _mod in (_m1, model_b6):
        _a = _mod.AYAR
        _vv = M.veri_kur(_a, yaz=lambda *x: None)
        _bak(f"{_a.ad}: yuva_ara kuruldu (yuva={_vv.yuva})",
             len(getattr(_vv, "yuva_ara", [])) == _vv.yuva)
        _LL = M.olcme_listeleri(_a, _vv)
        _lst = _LL["ood"][:16]
        # .to(M.DEV) SART: tahmin_ve_sira girdiyi DEV'e tasiyor. Yerelde
        # DEV=cpu oldugu icin eksikligi GORUNMEDI, Colab'da (cuda) dustu.
        _net = ModelB(_a, _vv.vocab).to(M.DEV)
        _net.eval()
        _th, _sr = _T.tahmin_ve_sira(_net, _vv, _lst)
        _bak(f"{_a.ad}: tahmin_ve_sira sekli ({len(_lst)},)",
             _th.shape == (len(_lst),) and _sr.shape == (len(_lst),),
             f"{_th.shape} {_sr.shape}")
        _kt = _T.ayristir(_vv, _lst, _th)
        _bak(f"{_a.ad}: butun kategoriler KATEGORI listesinde",
             set(_kt) <= set(_T.KATEGORI), set(_kt) - set(_T.KATEGORI))
        _uz = _T.uzunluga_gore(_vv, _lst, _kt, "ood", yaz=lambda *x: None)
        _bak(f"{_a.ad}: uzunluk tablosu TOPLAMI n'e esit",
             sum(x["n"] for x in _uz.values()) == len(_lst))
        _bak(f"{_a.ad}: dogruluk() dusmeden kosuyor",
             0.0 <= M.dogruluk(_net, _vv, *M.kodla_2hop(_vv, _lst)) <= 1.0)
    _bak("tek jetonluda uzunluk tablosu TEK satir",
         len(_T.uzunluga_gore(
             M.veri_kur(_m1.AYAR, yaz=lambda *x: None),
             M.olcme_listeleri(_m1.AYAR, M.veri_kur(
                 _m1.AYAR, yaz=lambda *x: None))["ood"][:16],
             ["DOGRU"] * 16, "ood", yaz=lambda *x: None)) == 1)

    print()
    print(f"{_iyi} gecti, {_kotu} BOZUK")
    if _kotu:
        print("\n!! model_b BOZUK. model_a9 ile kiyas GECERSIZ olur.")
    return 1 if _kotu else 0


if __name__ == "__main__":
    raise SystemExit(main())
