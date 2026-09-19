# -*- coding: utf-8 -*-
"""asama1_12 — DOGRUSAL SONDA: kopru gizli durumda VAR MI?

    python asama1_12.py <kosu klasoru> [--adim N] [--bolme comp]

Bu kolun sorusu: model 2-hop cevabini veremiyor. IKI AYRI SEBEP olabilir
ve ikisi cok farkli seyler soyler:

    (a) kopru HIC KURULMUYOR          -> ilk hop calismiyor
    (b) kopru KURULUYOR ama OKUNMUYOR -> ikinci hop calismiyor

Sonda ikisini ayirir. Modele soru okutulur, CEVABI YAZMADAN ONCE
durdurulur, ve o andaki gizli durumdan koprunun DOGRUSAL olarak
cikarilip cikarilamadigina bakilir.

    "Ibrahim Yilmaz'in ogrencisinin yasadigi yer neresidir? "
     ^-- model burada duruyor; gizli durumda 'Derya Yilmaz' var mi?

!! model_08'IN POZISYON TARAMASI DUSTU. Orada dizi 17 jetondu ve her
pozisyon ayri ayri taraniyordu ("en iyi poz"). Karakterde dizi ~100
karakter ve tek anlamli konum var: ONEGIN SON KARAKTERI, yani modelin
cevabi uretmeye basladigi yer. Taranacak bir sey kalmadi.

Ayni sebeple `olc` / `ornek_bazli` / `uzunluga_gore` / `birim_teshisi`
de silindi: hepsi yuva-kisitli argmax ve `yuva_ara` uzerine kuruluydu.
"""
from __future__ import annotations

import argparse
import collections
import importlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import korpus_12 as KOR                                       # noqa: E402
import olcme_12 as OLC                                        # noqa: E402
import pencere_12 as P                                        # noqa: E402
import taban_12 as M                                          # noqa: E402
import veri_12 as V                                           # noqa: E402
from model_12 import ModelHibrit                                # noqa: E402

MODEL_SINIFI = ModelHibrit
SUS = lambda *a, **k: None


def _kelime_sozlugu(G):
    """Ad -> kelime dizisi, ve kelime -> id. 1.608 addan 491 kelime."""
    E = [a for t in V.TIPLER for a in G["ad"][t]]
    kel = sorted({w for a in E for w in a.split("_")})
    return E, {w: i for i, w in enumerate(kel)}


@torch.no_grad()
def gizli(net, sv: OLC.Sinav, bs=256):
    """ONEGIN SON KONUMUNDAKI gizli durum -- (N, d).

    `govde()` cagriliyor, ileri gecis ELLE KURULMUYOR. Eski surumde
    `net.emb + net.pos` yaziliydi ve `ModelHibrit`de `.pos` YOK (RoPE);
    16 Eylul hakemliginde AttributeError ile yakalandi."""
    onceki = net.training
    net.eval()
    cik = []
    for i in range(0, len(sv.X), bs):
        xb = torch.from_numpy(sv.X[i:i + bs]).to(M.DEV)
        pb = torch.from_numpy(sv.bas[i:i + bs]).to(M.DEV) - 1
        h = net.govde(xb)                                  # (B, T, d)
        cik.append(h[torch.arange(len(xb), device=M.DEV), pb].float().cpu())
    net.train(onceki)
    return torch.cat(cik).numpy()


def sonda(q, G, lst, yaz=print):
    """Kopru gizli durumdan DOGRUSAL olarak cikarilabiliyor mu?

    (e, r1) CIFTLERINE GORE AYRIK bolme -- ayni cift hem egitimde hem
    sinavda olsaydi sonda GRAFI EZBERLERDI ve her zaman yuksek cikardi.

    TABAN SANS DEGIL. Iki taban var ve buyugu alinir:

      EN SIK   sinavdaki en sik sinif. Adin son kelimesi cogu zaman
               ayni ("...Fakultesi"), sansla kiyaslamak "bilgi var"
               dedirtirdi.
      KOPYA    sorunun oznesinin ayni kelimesi kopruyle ESIT mi. Kopru
               cogu zaman ozneyle AYNI AILEDEN ("Ahmet Kilic --cocuk-->
               Huseyin Kilic"), yani SOYAD girdide ZATEN duruyor. Sonda
               o kelimeyi "cozunce" kopruyu bildigi icin degil,
               girdiden KOPYALADIGI icin cozmus olabilir.

    OLCULDU (model_08, 16 Eylul): comp'ta kopya orani 0.4450 iken sonda
    0.3780 cikti -- yani sonda TRIVIAL kopyanin ALTINDA, ama en sik
    sinifa (0.0976) gore "BILGI VAR" yaziyordu."""
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:                                    # pragma: no cover
        yaz("    (sklearn yok, sonda atlandi)")
        return {}
    E, kid = _kelime_sozlugu(G)
    n_kel = max(len(E[int(z[3])].split("_")) for z in lst)
    _w = lambda e, j: (lambda p: kid[p[j]] if j < len(p) else -1)(
        E[e].split("_"))
    kop = np.array([[_w(int(z[3]), j) for j in range(n_kel)] for z in lst])
    ozn = np.array([[_w(int(z[0]), j) for j in range(n_kel)] for z in lst])
    cift = np.array([hash((int(z[0]), int(z[1]))) % 5 for z in lst])
    tr, te = cift != 0, cift == 0
    yaz(f"    egitim {int(tr.sum())} / sinav {int(te.sum())}   "
        "((e,r1) CIFTLERINE GORE AYRIK)")
    yaz(f"    {'kelime':<8}{'aday':>6}{'SONDA':>9}{'EN SIK':>9}{'KOPYA':>9}"
        f"{'sans':>8}   hukum")
    out = {}
    for j in range(n_kel):
        m = kop[:, j] >= 0
        y, mt, mr = kop[:, j], te & m, tr & m
        ns = len(set(y[m].tolist()))
        if ns < 2 or int(mt.sum()) < 20:
            continue
        clf = LogisticRegression(max_iter=300).fit(q[mr], y[mr])
        acc = float(clf.score(q[mt], y[mt]))
        c = collections.Counter(y[mt].tolist())
        sik = c.most_common(1)[0][1] / int(mt.sum())
        kpy = float((ozn[mt, j] == kop[mt, j]).mean())
        taban = max(sik, kpy)
        hukum = ("TABAN>0.5 (%s), bilgi DEGIL"
                 % ("kopya" if kpy >= sik else "en sik")
                 if taban > 0.5 else
                 "BILGI VAR" if acc > 2 * taban else "bilgi YOK")
        out[str(j)] = dict(aday=ns, sonda=acc, en_sik=sik, kopya=kpy,
                           taban=taban, hukum=hukum)
        yaz(f"    {j:<8}{ns:>6}{acc:>9.4f}{sik:>9.4f}{kpy:>9.4f}"
            f"{1/ns:>8.4f}   {hukum}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("klasor")
    ap.add_argument("--adim", type=int, default=None,
                    help="hangi anlik goruntu (varsayilan: SONUNCU)")
    ap.add_argument("--bolme", default="comp,ent",
                    help="VIRGULLE ayrilmis: comp,ent,ent_yok,seen,ood")
    a = ap.parse_args()

    ayar = P.ayar_oku(a.klasor)
    snap = P.anlik_goruntuler(a.klasor)
    adim = a.adim if a.adim is not None else max(snap)
    assert adim in snap, f"adim {adim} YOK: {sorted(snap)}"
    print(f"=== asama1_12  {ayar.ad} t{ayar.tohum}  adim {adim} ===")

    v = M.veri_kur(ayar, yaz=SUS)
    G = importlib.import_module(ayar.veri_ad).kur(ayar.veri_tohum)
    # !! `KOR.havuz` DOGRUDAN CAGRILMAZ -- `M.egitim_havuzu(ayar, v)`.
    # Onceki hali `KOR.havuz(v, G, ayar.kopya, ayar.t_len, ...)` idi ve
    # 10 parametrenin 6'sini geciyordu: `zincir_pay`, `n3`, `ret_pay`,
    # `ret_tut` VARSAYILANA dusuyordu. `n3` egitimde 10.000, burada 0 --
    # yani okuma araci EGITILEN KORPUSTAN BASKA bir korpus kuruyordu.
    # Sozluk ayni cikabilir (ayni harfler) ama `korpus_izi` tutmaz ve
    # hicbir sey bunu SOYLEMEZDI. Tek yol: ayardan turet.
    _X, S = M.egitim_havuzu(ayar, v, yaz=SUS)
    L = M.olcme_listeleri(ayar, v)
    net = (MODEL_SINIFI or M.Model)(ayar, S.vocab).to(M.DEV)
    net.load_state_dict(torch.load(snap[adim], map_location=M.DEV))
    for _b in [x.strip() for x in a.bolme.split(",") if x.strip()]:
        lst = L.get(_b, ())
        if not len(lst):
            print(f"\n  -- {_b}: BOS, atlandi")
            continue
        sv = OLC.Sinav(S, v, G, lst, 1 if _b == "one" else 2, _b)
        print(f"\n  === {_b}  {len(lst):,} zincir   sinav genisligi "
              f"{sv.genislik} ===")
        d = OLC.olc(net, S, sv, M.DEV)
        print(f"  ASAMA-2 (cevap):  tam {d['tam']:.4f}  "
              f"yakin {d['yakin']:.4f}  kisayol {d['kisayol']:.4f}")
        if _b == "one":
            continue            # 1-hop'ta KOPRU YOK, sonda anlamsiz
        print("  ASAMA-1 (kopru), onegin SON konumunda:")
        sonda(gizli(net, sv), G, lst)


if __name__ == "__main__":
    main()
