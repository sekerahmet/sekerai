# -*- coding: utf-8 -*-
"""model_b13 — IDENTITY BRIDGE (literaturun cozumu).

Onceden kayit: belge/onkayit/model_b13.md
Taban model_b10.  UC DEGISIKLIK, ve bilerek:

    model_b10   ident_frac=0.0   wd=0.5   sabit_lr=True
    model_b13   ident_frac=0.2   wd=0.1   sabit_lr=False
                ident_kip="q1"

--------------------------------------------------------------------------
NEDEN -- KENDI OLCUMUMUZ MAKALENIN TESHISINI DOGRULADI

Kahin testi (belge/bulgu/kahin_testi.md): Phi'nin ciktisi yerine
GERCEK KOPRU gommesini koyduk, gate'i 1.0'a zorladik.

    comp  NORMAL 0.0717   KAHIN 0.0733   KONTROL 0.0717
    (mudahale ETKILI: logit farki max 18.04, tahminlerin %11'i degisti)

Yani kopru elden verilse BILE bilesim olmuyor.

arXiv 2509.24653 4.1, BIREBIR -- ayni deneyi yapmislar, ayni sonuc:
  "We feed the sequence (a, r1) into the model and extract the hidden
   states at the r1 position ... These extracted states are then
   injected into the position of b within the sequence (b, r2) ...
   The result shows that the model completely fails to decode c on OOD
   two-hop reasoning, which indicates token b doesn't bridge the gap
   between first hop and second hop."

TESHISLERI:
  "We attribute this failure to a contextual disconnect between the
   input and output spaces. The model is not explicitly required to
   establish an equivalence between the input token b and the output
   token b, which is a trivial capability for well-trained LLMs."

COZUMLERI:
  "a straightforward solution is to augment the training data with
   b -> b 'zero-hop' sequences, which we term as identity bridge."

--------------------------------------------------------------------------
BU KURAL DAYATMASI DEGIL -- kopru_kayip'tan FARKI

    kopru_kayip (b8)   GRAFTAN kopruyu soyler      -> bizim kuralimiz
    ident_frac  (b13)  varligin KENDI jetonlari    -> girdi/cikti
                       "okudugun = yazdigin"          HIZALAMASI

Identity satiri hicbir olgu, cevap ya da kopru icermez:

    [S1] Ahmet Yilmaz <YOK> <KENDISI> ? Ahmet Yilmaz <YOK> <EOS>

Gercek LLM'ler bunu on-egitimde BEDAVA aliyor ("trivial capability for
well-trained LLMs"); 20.000 adimlik bir model almiyor.

--------------------------------------------------------------------------
ident_frac = 0.2 -- NEDEN 0.5 DEGIL

Makale: "The effect of retaining a certain proportion of identity
bridge ... The model's accuracy closely aligns with the proportion of
retained data." Yani onemli olan KAPSAM: 2120 varligin HEPSI var.

Havuz payini 0.2'de tuttuk cunku model_b11 SEYRELMENIN zararini
olctu: havuz iki katina cikinca `seen` 0.8210 -> 0.3353'e dustu.
0.2'de kimlik satirlari havuzun 1/5'i, cevap gorevi seyrelmiyor.

--------------------------------------------------------------------------
wd=0.1 ve cosine NEDEN BURADA

Kullanici karari, 16 Eylul: "LR ve wd = 0,1 ayni anda degissin",
"min lr olsun, tabani lr/10 olsun". Gerekce: "ortada denenmis bazi
seyler var, millet boyle yapmis ve bir sonuc almis... benim derdim su
an iyi bir sonuca ulasmak."

nanoGPT / Pythia-70m / Pythia-160m / Qwen2.5 SFT -- DORDU de wd 0.1 ve
AZALAN LR (min_lr = lr/10).

!! BU KOL BIR ABLASYON DEGIL, EN IYI TAHMIN KONFIGURASYONU. model_b10'a
gore UC dugme dondu. Sonuc olumluysa bisect kollari kosulur:
    b13a  yalniz ident_frac (wd 0.5, sabit LR kalir)
    b13b  yalniz wd+cosine (ident_frac 0 kalir)  == zaten model_b12

--------------------------------------------------------------------------
BELGE (belge_pay) NEDEN YOK

model_b11 olctu: `comp` 0.0607 -> 0.0893 (40.000'de), ama hesap +%59
(t_len 11 -> 20) ve `seen` cokuyor. Identity satiri 10 jeton, t_len
11'e SIGIYOR -- bu kol t_len'e DOKUNMAZ, hiz model_b10 ile ayni kalir.
Belge istenirse SONRA eklenir.

MIMARI: bir satir bile degismedi. Ek parametre YOK.
"""
from __future__ import annotations

import os
import sys

_B = os.path.dirname(os.path.abspath(__file__))
_A = os.path.join(os.path.dirname(_B), "model_a")
for _p in (_A, _B):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import model_a as M                                          # noqa: E402
assert hasattr(M, "egit"), (
    f"model_a MODUL degil PAKET olarak yuklendi: {getattr(M,'__file__',None)}")

from model_b import ModelB                                   # noqa: E402
from model_b10 import AYAR as TABAN                          # noqa: E402

AYAR = TABAN.degistir(ad="model_b13", ident_frac=0.2, ident_kip="q1",
                      wd=0.1, sabit_lr=False)
#      ^ ident_frac + ident_kip BIR CIFT (frac>0 kip GEREKTIRIR).
#        tam_kayip=True ve dar_kapi=True model_b10'dan DEVRALINIR.
#        belge_pay=0.0 -> t_len 11 KALIR.

fark_bas = M.fark_bas


def egit(ayar=None, **kw):
    return M.egit(ayar or AYAR, model_kur=ModelB, **kw)


if __name__ == "__main__":
    M.fark_bas(TABAN, AYAR)
    egit(AYAR)
