# -*- coding: utf-8 -*-
"""test_sabit_c — model_c ailesinin KILIT TESTI.

    python test_sabit_c.py

Once `model_b/test_sabit_b.py`yi kosturur (o da icinde
`model_a/test_sabit.py`yi kosturur), sonra model_c'ye ozgu degismezleri
denetler. Defterin 3. hucresi bunu `test_*.py` olarak bulup kosturuyor;
duserse egitim BASLAMAZ.

Zincir: test_sabit_c -> test_sabit_b -> test_sabit
Boylece VERI DENETIMI (15 kontrol, jeton_ad="tam") ve B ailesinin butun
degismezleri model_c kosarken de sinaniyor -- kopyalanmadan.

EN ONEMLI KONTROL 2 NUMARA: `dar_kafa=1` iken ModelC, ModelB ile BIT
DUZEYINDE ayni cikti vermeli. Vermezse `model_b6` ile kiyas gecersiz
olur -- o zaman fark yalniz cok basli Phi'den degil, sinifin kendisinden
de geliyor demektir.
"""
from __future__ import annotations

import os
import subprocess
import sys

_C = os.path.dirname(os.path.abspath(__file__))
_B = os.path.join(os.path.dirname(_C), "model_b")
_A = os.path.join(os.path.dirname(_C), "model_a")
sys.path[:0] = [_C, _B, _A, os.path.dirname(_C)]

import torch                                                  # noqa: E402
import model_a as M                                           # noqa: E402
import model_b6                                               # noqa: E402
from model_b import ModelB                                    # noqa: E402
import model_c                                                # noqa: E402
from model_c import ModelC                                    # noqa: E402

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


def _fark(a, b):
    return sorted(k for k in a.sozluk()
                  if a.sozluk()[k] != b.sozluk().get(k))


def main():
    print("=== 0) TABAN (model_b/test_sabit_b.py -> model_a/test_sabit.py) ===")
    r = subprocess.run([sys.executable, os.path.join(_B, "test_sabit_b.py")],
                       cwd=_B, capture_output=True, text=True,
                       env={**os.environ})
    print(r.stdout.rstrip()[-1200:] if r.stdout else "(cikti yok)")
    if r.stderr.strip():
        print("stderr:", r.stderr.rstrip()[-800:])
    _bak("model_b kilidi (ve icindeki model_a kilidi)", r.returncode == 0,
         f"cikis {r.returncode}")

    print("\n=== 1) model_c'nin DUGMESI ===")
    _bak("model_c <-> model_b6 farki ['ad', 'dar_kafa']",
         _fark(model_c.AYAR, model_b6.AYAR) == ["ad", "dar_kafa"],
         str(_fark(model_c.AYAR, model_b6.AYAR)))
    _bak("model_c dar_kafa=3", model_c.AYAR.dar_kafa == 3,
         str(model_c.AYAR.dar_kafa))
    _bak('model_c jeton_ad="tam" (model_b6\'dan devralindi)',
         model_c.AYAR.jeton_ad == "tam", model_c.AYAR.jeton_ad)
    _bak("model_b6 dar_kafa=1 (DEGISMEDI)", model_b6.AYAR.dar_kafa == 1,
         str(model_b6.AYAR.dar_kafa))
    for g in ("AYAR", "egit", "fark_bas"):
        _bak(f"model_c.{g} var", hasattr(model_c, g))

    # --- veri: kiyas ancak AYNI olcme setiyle gecerli -------------------
    v = M.veri_kur(model_c.AYAR, yaz=lambda *a: None)
    L = M.olcme_listeleri(model_c.AYAR, v)
    iz = M.olcme_izi(L)
    _bak("olcme izi 57cf60a5e9af (model_b6 ile AYNI)", iz == "57cf60a5e9af", iz)
    _bak("yuva 3, vocab 466", (v.yuva, v.vocab) == (3, 466),
         f"{v.yuva} {v.vocab}")
    _bak("dar_kafa == yuva (her bas bir yuva)",
         model_c.AYAR.dar_kafa == v.yuva,
         f"{model_c.AYAR.dar_kafa} != {v.yuva}")

    print("\n=== 2) dar_kafa=1 -> ModelC, ModelB ILE BIT AYNI MI ===")
    a1 = model_c.AYAR.degistir(dar_kafa=1)
    torch.manual_seed(0)
    b = ModelB(a1, v.vocab)
    torch.manual_seed(0)
    c = ModelC(a1, v.vocab)
    _bak("parametre sayisi",
         sum(p.numel() for p in b.parameters())
         == sum(p.numel() for p in c.parameters()),
         f"{sum(p.numel() for p in b.parameters())} vs "
         f"{sum(p.numel() for p in c.parameters())}")
    _bak("ek modul YOK (kafa/karisim parametresi kurulmadi)",
         not hasattr(c, "kafa") and not hasattr(c, "kar_w")
         and c.cok_bas is False)
    c.load_state_dict(b.state_dict())
    x = torch.randint(0, v.vocab, (4, a1.t_len))
    b.eval(); c.eval()
    with torch.no_grad():
        d = (b(x) - c(x)).abs().max().item()
    _bak(f"cikti farki {d:.3e}", d == 0.0)

    print("\n=== 3) dar_kafa=3 -> cikti DEGISMELI ===")
    torch.manual_seed(0)
    c3 = ModelC(model_c.AYAR, v.vocab)
    _bak("kafa parametresi kuruldu (m, d, d)",
         hasattr(c3, "kafa")
         and tuple(c3.kafa.shape) == (3, model_c.AYAR.d, model_c.AYAR.d),
         str(tuple(c3.kafa.shape)) if hasattr(c3, "kafa") else "YOK")
    _bak("ek parametre m*d^2 + m*(d+1) = 197.379",
         sum(p.numel() for p in c3.parameters())
         - sum(p.numel() for p in b.parameters()) == 3 * 256 * 256 + 3 * 257,
         str(sum(p.numel() for p in c3.parameters())
             - sum(p.numel() for p in b.parameters())))
    _bak("karisim agirligi kuruldu (m,d) + (m,)",
         hasattr(c3, "kar_w") and tuple(c3.kar_w.shape) == (3, 256)
         and tuple(c3.kar_b.shape) == (3,))
    # Phi'ye AIT parametreler disindaki AGIRLIKLAR ayni olsun ki fark
    # YALNIZ cok basli Phi'den gelsin -- ModelB'de kafa/karisim YOK.
    sd = {k: t for k, t in c3.state_dict().items()
          if k not in ("kafa", "kar_w", "kar_b")}
    b3 = ModelB(model_c.AYAR, v.vocab)
    b3.load_state_dict(sd)
    b3.eval(); c3.eval()
    with torch.no_grad():
        d3 = (b3(x) - c3(x)).abs().max().item()
    _bak(f"cok basli Phi ETKILI {d3:.3e}", d3 > 1e-4)

    print("\n=== 4) A_0 = I ILE BASLATILDI ===")
    I = torch.eye(model_c.AYAR.d)
    _bak("kafa[0] TAM birim matris", torch.equal(c3.kafa[0], I),
         f"{(c3.kafa[0] - I).abs().max().item():.3e}")
    for j in (1, 2):
        s = (c3.kafa[j] - I).abs().max().item()
        _bak(f"kafa[{j}] birimden AYRI (simetri kirildi) {s:.3e}",
             1e-6 < s < 0.5, f"{s:.3e}")
    # 0. bas TEK BASINA ModelB'nin Phi'si olmali
    with torch.no_grad():
        h = torch.randn(2, 5, model_c.AYAR.d)
        q = c3.nf(h)
        p = torch.softmax(c3.head(q) / model_c.AYAR.dar_tau, -1)
        phi_b = p @ c3.emb.weight
        pj = torch.softmax((q @ c3.kafa[0].T) @ c3.emb.weight.T
                           / model_c.AYAR.dar_tau, -1)
        phi_c0 = pj @ c3.emb.weight
        d0 = (phi_b - phi_c0).abs().max().item()
    _bak(f"Phi_3'un 0. basi == ModelB'nin Phi'si {d0:.3e}", d0 < 1e-5)

    print()
    print("=== 4b) KARISIM AGIRLIGI OGRENILEBILIR, BASLANGICTA 1/m ===")
    # Sabit 1/m yanlisti: model "bu pozisyonda bu head gereksiz"
    # diyememeliydi. Softmax secildi ki w=b=0 iken TAM 1/m ciksin --
    # yani kol, sade ortalamanin BASLADIGI yerden baslasin.
    _bak("kar_w ve kar_b SIFIRDAN basliyor",
         bool(torch.all(c3.kar_w == 0)) and bool(torch.all(c3.kar_b == 0)))
    with torch.no_grad():
        h2 = torch.randn(2, 5, model_c.AYAR.d)
        q2 = c3.nf(h2)
        _gs = [torch.softmax((q2 @ c3.kafa[j].T) @ c3.emb.weight.T
                             / model_c.AYAR.dar_tau, -1) @ c3.emb.weight
               for j in range(3)]
        _sade = sum(_gs) / 3
        dk = (c3._phi(h2) - _sade).abs().max().item()
    _bak(f"baslangicta karisim == SADE ORTALAMA {dk:.3e}", dk < 1e-6)
    # gradyan karisima AKIYOR mu
    _c4 = ModelC(model_c.AYAR, v.vocab)
    _X, _P, _T = M.kodla_2hop(v, L["seen"][:8])
    _lg = _c4(torch.from_numpy(_X))
    _ar = torch.arange(8)[:, None]
    torch.nn.functional.cross_entropy(
        _lg[_ar, torch.from_numpy(_P)].reshape(-1, v.vocab),
        torch.from_numpy(_T).reshape(-1)).backward()
    _bak("gradyan kar_w'ye AKIYOR",
         _c4.kar_w.grad is not None and _c4.kar_w.grad.abs().sum() > 0)
    _bak("gradyan HER head'in A_j'sine AKIYOR",
         all(_c4.kafa.grad[j].abs().sum() > 0 for j in range(3)))

    print("\n=== 5) SON TURDA enjeksiyon YOK (alfa(K-1)=0) ===")
    a_tek = model_c.AYAR.degistir(dongu=1)
    torch.manual_seed(0)
    c_tek = ModelC(a_tek, v.vocab)
    torch.manual_seed(0)
    b_tek = ModelB(a_tek.degistir(dar_kafa=1), v.vocab)
    b_tek.load_state_dict({k: t for k, t in c_tek.state_dict().items()
                           if k not in ("kafa", "kar_w", "kar_b")})
    c_tek.eval(); b_tek.eval()
    with torch.no_grad():
        dt = (c_tek(x) - b_tek(x)).abs().max().item()
    _bak(f"dongu=1'de fark {dt:.3e} (cok bas DEVREYE GIRMIYOR)", dt == 0.0)

    print("\n=== 6) KANCALAR (pencere_c / tani_c / asama1_c) ===")
    for mod, ad in (("pencere_a", "pencere_c"), ("tani_a", "tani_c"),
                    ("asama1", "asama1_c")):
        try:
            __import__(ad)
            m = sys.modules[mod]
            _bak(f"{ad} MODEL_SINIFI'ni ModelC yapti",
                 getattr(m, "MODEL_SINIFI", None) is ModelC,
                 str(getattr(m, "MODEL_SINIFI", None)))
        except Exception as e:                         # noqa: BLE001
            _bak(f"{ad} import edilebiliyor", False, repr(e))

    print()
    print(f"{_iyi} gecti, {_kotu} BOZUK")
    if _kotu:
        print("\n!! model_c BOZUK. model_b6 ile kiyas GECERSIZ olur.")
    return 1 if _kotu else 0


if __name__ == "__main__":
    raise SystemExit(main())
