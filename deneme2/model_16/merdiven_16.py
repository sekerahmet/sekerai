"""merdiven_16 -- IDEAL teste model_16'nin kisitlarini TEKER TEKER koy.

Kullanici, 22 Eylul: *"burda olcumun sonuclarina gore hangi yerde nasil bir
mudahele yapacagimiza karar verecegiz"*.

Her basamak bir oncekine GERCEK MODELIN TEK BIR KISITINI ekler ve SIFIRDAN
egitilir.  Nerede TOP-1 cokerse mudahale yeri orasi.  Cerrahi teshis
(egitilmis model uzerinde parca dondurma) bu soruyu cevaplayamaz: orada
diger parametreler eski hale gore oturmustur.  Burada her yapilandirma
kendi egitimini gorur.

  R0  serbest e_x,  o = Mr[r] e_x                     hedef VARLIK  (1608)
  R1  + e_x PAYLASILAN token b/M'lerinden ozyineleme
  R2a + tek paylasilan Wv
  R2b + cikis gommesi TOKEN bazinda                   hedef ILK TOKEN (444)
  R3a + iliski GERCEK TOKENLER, okuma son durumdan
  R3b + TAM sinav onegi, okuma serbest yuva agirliklariyla
  R3c + GERCEK dikkat  relu(q.k + hb)
  R4  + amac korpusta sonraki token   ( = model_16, OLCULDU 0,020 )

OLCUNUN SAGLIGI
  TAVAN   R0 = 1.000 olculdu.
  TABAN   hedefler KARISTIRILIR.  Bu bir EZBER testi ve gercek graf da
          keyfi; dogru kurulmus olcumde karisik hedef AYNI sonucu verir.
  ENIYILEME  bir basamak 0,9'un altina duserse lr taramasi + 3x adim.
          Cokus ancak bunu gecerse sayilir.  (GECMEK kesin bilgi,
          COKMEK belirsiz -- asimetri bilerek boyle.)
  SIZINTI hedef girdide hic gecmez: skor yalniz (x, r)'den kurulur.

ZINCIRDEKI YERI.  Kim kimi cagiriyor, bu dosya nerede:

  veri_16 / metin_16 / korpus_16 / jeton_16 / birim_16 / ek_16
  taban_16 / ayar_16 / hazirla_16 / dok_16 / denetim_16
  model_16    MIMARI
  kos_16      egitim dongusu
  olcme_16    olcu
  adim_16     TEK SORU, ADIM ADIM
  merdiven_16 YAPI NEREDE TASIYAMIYOR   <-- BU DOSYA
"""
import collections
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metin_16 as MT      # noqa: E402
import olcme_16 as OL      # noqa: E402

D = int(os.environ.get("MERDIVEN_D", 16))
AYGIT = os.environ.get("MERDIVEN_AYGIT", "cuda")
if AYGIT == "cuda":
    assert torch.cuda.is_available(), "GPU YOK -- MERDIVEN_AYGIT=cpu ile zorla"
    _bos = torch.cuda.mem_get_info()[0] / 1e9
    assert _bos > 2.0, f"GPU'da sadece {_bos:.1f} GB bos"
    print(f"GPU kapisi GECTI: {torch.cuda.get_device_name(0)}  bos {_bos:.1f} GB")


def veri(yol):
    d = torch.load(yol, weights_only=False)
    kodla = OL._kodlayici(d)[0]
    V, IL, TIP, tip = d["varlik"], d["iliski"], d["tip_ad"], d["tip"]
    ENT = [kodla(MT._tr(v)) for v in V]
    zin = [[int(x) for x in y] for y in d["bolme"]["ezber_olgu"]]
    rr = sorted({z[1] for z in zin})
    rmap = {r: i for i, r in enumerate(rr)}
    X = [z[0] for z in zin]
    Y = [z[-1] for z in zin]
    R = [rmap[z[1]] for z in zin]
    kisa, tam = [], []
    for z in zin:
        o_, c_ = V[z[0]], V[z[-1]]
        Xn, il = MT._yol_parca(o_, [IL[z[1]]], c_)[:2]
        kisa.append(kodla(Xn + " " + il))                  # varlik+iyelik+iliski
        tam.append(kodla(MT.sinav_yuzeyi(o_, [IL[z[1]]], c_,
                                         TIP[int(tip[z[-1]])])[0]))
    return dict(d=d, ENT=ENT, X=X, Y=Y, R=R, kisa=kisa, tam=tam,
                NE=len(V), NR=len(rr), NT=len(d["ad"]),
                HT=[ENT[y][0] for y in Y])                 # ILK CEVAP TOKENI


def obekle(seqs, aygit):
    """Degisken uzunluk -> uzunluga gore obek.  (indeks, token tablosu)"""
    g = collections.defaultdict(list)
    for i, s in enumerate(seqs):
        g[len(s)].append(i)
    return [(torch.tensor(v, device=aygit),
             torch.tensor([seqs[i] for i in v], device=aygit))
            for v in g.values()]


def gez(p, obek, n, iz=False):
    """s_t = normalize(M[w] s + b[w]).  iz=True butun yuvalari dondurur."""
    T = max(W.shape[1] for _, W in obek)
    out = torch.zeros(n, T, D, device=p["b"].device) if iz else \
        torch.zeros(n, D, device=p["b"].device)
    for idx, W in obek:
        s = p["s0"].expand(len(idx), D)
        ara = []
        for t in range(W.shape[1]):
            s = F.normalize(torch.bmm(p["M"][W[:, t]], s.unsqueeze(-1)).squeeze(-1)
                            + p["b"][W[:, t]], dim=-1)
            ara.append(s)
        if iz:
            S = torch.stack(ara, 1)
            S = F.pad(S, (0, 0, 0, T - S.shape[1]))
            out = out.index_copy(0, idx, S)
        else:
            out = out.index_copy(0, idx, s)
    return out


BASAMAK = {            # (varlik_token, Wv, E_token, onek, dikkat)
    "R0 ": (0, 0, 0, None, None),
    "R1 ": (1, 0, 0, None, None),
    "R2a": (1, 1, 0, None, None),
    "R2b": (1, 1, 1, None, None),
    "R3a": (1, 1, 1, "kisa", None),
    "R3b": (1, 1, 1, "tam", "serbest"),
    "R3c": (1, 1, 1, "tam", "gercek"),
}


def parametre(v, cfg, tohum, aygit, F_):
    vt, wv, et, onek, dik = cfg
    g = torch.Generator().manual_seed(tohum)
    r = lambda *s: torch.randn(*s, generator=g).to(aygit)
    p = {}
    if vt or onek:
        p["b"] = r(v["NT"], D) / D ** 0.5
        p["M"] = torch.eye(D, device=aygit).repeat(v["NT"], 1, 1) + 0.1 * r(v["NT"], D, D)
        p["s0"] = torch.zeros(D, device=aygit)
    if not vt:
        p["e"] = r(v["NE"], D) / D ** 0.5
    if onek is None:
        p["Mr"] = torch.eye(D, device=aygit).repeat(v["NR"], 1, 1) + 0.1 * r(v["NR"], D, D)
    if wv:
        p["Wv"] = r(D, D) * D ** -0.5
    if et:
        p["E"] = r(v["NT"], D) / D ** 0.5
    if dik == "serbest":
        p["al"] = torch.zeros(F_, v["T"])
        p["al"] = p["al"].to(aygit)
    if dik == "gercek":
        p["Wq"] = r(D, D) * D ** -0.5
        p["Wk"] = r(D, D) * D ** -0.5
        p["hb"] = torch.zeros(1, device=aygit)
    for x in p.values():
        x.requires_grad_(True)
    return p


def ileri(p, v, cfg, T):
    vt, wv, et, onek, dik = cfg
    if onek is None:
        ex = (gez(p, v["ENT_OB"], v["NE"]) if vt else p["e"])
        cik = ex
        q = torch.bmm(p["Mr"][v["R"]], ex[v["X"]].unsqueeze(-1)).squeeze(-1)
    else:
        cik = None
        if dik is None:
            q = gez(p, v[onek + "_OB"], len(v["X"]))
        else:
            S = gez(p, v[onek + "_OB"], len(v["X"]), iz=True)       # (F,T,D)
            if dik == "serbest":
                a = F.softplus(p["al"])
            else:
                s = S[:, -1]
                pu = ((S @ p["Wk"]) @ (s @ p["Wq"]).unsqueeze(-1)).squeeze(-1)
                a = (pu + p["hb"]).clamp(min=0)
            q = (a.unsqueeze(-1) * S).sum(1)
    if wv:
        q = q @ p["Wv"]
    return q @ (p["E"] if et else cik).T


def kos(ad, v, cfg, hedef, lr, adim, tohum=0, yaz=print):
    T = v["T"]
    p = parametre(v, cfg, tohum, hedef.device, len(v["X"]))
    opt = torch.optim.Adam(list(p.values()), lr=lr)
    for _ in range(adim):
        k = F.cross_entropy(ileri(p, v, cfg, T), hedef)
        opt.zero_grad(); k.backward(); opt.step()
    with torch.no_grad():
        s = ileri(p, v, cfg, T)
        t1 = float((s.argmax(1) == hedef).float().mean())
    n = sum(x.numel() for x in p.values())
    yaz(f"    {ad:<30s} lr {lr:<6} adim {adim:<5} kayip {float(k):7.4f}"
        f"  TOP-1 {t1:.3f}   ({n:,} par)")
    return t1


GUNLUK, SONUC = [], {}


def main(yaz=None, yol=None, kok=None):
    """Butun basamaklari sirayla kosar.  `yaz` gunluge eklemek icin."""
    if yaz is None:
        yaz = lambda *a: print(*a, flush=True)
    yol = yol or os.environ.get(
        "BIRIM_PT", "/content/drive/MyDrive/model_16/birim_16.pt")
    v = veri(yol)
    aygit = AYGIT
    v["ENT_OB"] = obekle(v["ENT"], aygit)
    v["kisa_OB"] = obekle(v["kisa"], aygit)
    v["tam_OB"] = obekle(v["tam"], aygit)
    v["T"] = max(len(s) for s in v["tam"])
    for k in ("X", "R", "Y", "HT"):
        v[k] = torch.tensor(v[k], device=aygit)
    yaz(f"OLGU {len(v['X']):,}  VARLIK {v['NE']:,}  ILISKI {v['NR']}"
        f"  TOKEN {v['NT']}  durum {D}  aygit {aygit}")
    yaz(f"onek uzunluklari: kisa {sorted({len(s) for s in v['kisa']})}"
        f"   tam {sorted({len(s) for s in v['tam']})}")
    yaz(f"sans: varlik 1/{v['NE']} = {1/v['NE']:.4f}"
        f"   ilk token 1/{v['NT']} = {1/v['NT']:.4f}")
    yaz("model_16'nin AYNI karardaki sayisi: 0.020  (konum 0, t20000)")
    yaz("R3c parametre sayisi model_16 ile AYNI olmali: 128.657\n")
    g = torch.Generator().manual_seed(1)
    kar = {}
    for ad, cfg in BASAMAK.items():
        et = cfg[2]
        hedef = v["HT"] if et else v["Y"]
        n_sinif = v["NT"] if et else v["NE"]
        if n_sinif not in kar:
            kar[n_sinif] = hedef[torch.randperm(len(hedef), generator=g)]
        yaz(f"{ad}  hedef {'ILK TOKEN' if et else 'VARLIK'} ({n_sinif} sinif)")
        t1 = kos("gercek hedef", v, cfg, hedef, 0.01, 2000, yaz=yaz)
        if t1 < 0.9:                       # ENIYILEME KONTROLU -- bkz. baslik
            for lr in (0.003, 0.03, 0.1):
                t1 = max(t1, kos("gercek (lr taramasi)", v, cfg, hedef,
                                 lr, 2000, yaz=yaz))
            t1 = max(t1, kos("gercek (3x adim)", v, cfg, hedef, 0.01,
                             6000, yaz=yaz))
        tb = kos("KARISIK hedef (taban)", v, cfg, kar[n_sinif], 0.01,
                 2000, yaz=yaz)
        SONUC[ad] = dict(top1=t1, taban=tb)
        yaz(f"  ==> {ad} EN IYI TOP-1 {t1:.3f}   (karisik taban {tb:.3f})\n")
    if kok:
        d = f"{kok}/MERDIVEN"
        os.makedirs(d, exist_ok=True)
        with open(f"{d}/kayit.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(GUNLUK) + "\n")
        yaz(f"yazildi -> {d}/kayit.txt")
    yaz("MERDIVEN BITTI")


def baslat(yol=None, kok=None):
    """ARKA PLANDA baslatir, HEMEN doner (kural 8)."""
    import threading
    GUNLUK.clear(); SONUC.clear()
    threading.Thread(target=lambda: main(GUNLUK.append, yol, kok),
                     daemon=True).start()
    return "merdiven basladi"


def nabiz(son=60):
    """Gunlugu basar.  HICBIR SEY KOSTURMAZ (kural 8)."""
    print(f"gunluk {len(GUNLUK)} satir   biten basamak: {list(SONUC)}")
    for s in GUNLUK[-son:]:
        print(s)


if __name__ == "__main__":
    main()
