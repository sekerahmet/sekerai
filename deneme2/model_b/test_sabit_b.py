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

import torch                                                 # noqa: E402
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

    import model_b2
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
    for _ad in ("model_b", "model_b1", "model_b2"):
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
    print(f"{_iyi} gecti, {_kotu} BOZUK")
    if _kotu:
        print("\n!! model_b BOZUK. model_a9 ile kiyas GECERSIZ olur.")
    return 1 if _kotu else 0


if __name__ == "__main__":
    raise SystemExit(main())
