# -*- coding: utf-8 -*-
"""data_stories -- TinyStories, SONRAKI KELIME (model_17 veri_t17 + olcme_17'den).

Kullanici, 24 Eylul: "matematik de çok aritmetik bir işlem var ama dil de bağlam
ve ilişki matematik gibi değil. o yüzden görmek istiyorum."

Veri Drive'daki onbellekten (kural 9): model_17'nin 4.000 kelimelik sozlugu
(+ <dolgu> <eos> <bilinmeyen> = 4.003), akis ve pencere AYNEN onunki.
Dosyadaki eski ad <hikaye> == <eos> (genel()), kimlik 1.

OLCUT  accuracy (sonraki kelime birebir), ce (ppl = e^ce), tani: konuma gore
       accuracy (acc_0_64, acc_64_256, acc_256_512) ve eos_ok.
METIN  devam(): istemden hikaye, GOZLE okunur.

KELIME SEVIYESI, buyuk harf korunmus; nadir kelime <bilinmeyen>.
LISANS  cdla-sharing-1.0.   Eldan & Li, arXiv 2305.07759
"""
from __future__ import annotations

import collections
import hashlib
import math
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

SINIR = "<|endoftext|>"       # dosyadaki ayrac
# GENEL SINIR: her birimin (hikaye, soru) basi ve sonu -- BOS ve EOS ayni
# token.  Kullanici: "hikaye ve ya sinir yerine genel bir terim. model nerde
# durması gerektiğini öğrenmesi adına".
SON = "<eos>"
HIKAYE = "<hikaye>"           # ESKI ad: onbellekteki sozluklerde bu yaziyor
BILINMEYEN = "<bilinmeyen>"
DOLGU = "<dolgu>"          # hikaye bitince kalan yer; maske ile duser
# Kesme isaretli kisaltma TEK birim (don't, Lily's); noktalama AYRI.
JETON = re.compile(r"[A-Za-z]+'[A-Za-z]+|[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]")

# KIVRIK TIRNAKLAR DUZLESTIRILIYOR.  Korpus ikisini de kullaniyor ve
# kisaltma kurali yalniz duz kesmeyi taniyordu -- OLCULDU (48 MB):
#   kisaltma 93.704:  duz 91.062,  kivrik 2.642  (%2,8 UCE BOLUNUYORDU)
#   "Let's" bazen tek birim, bazen  Let | ' | s
# Ayni kelimenin iki ayri jetonlanmasi.  Tire (- , en, em) DOKUNULMUYOR:
# kurali bozmuyorlar ve anlamlari ayri.
DUZLE = str.maketrans({"’": "'", "‘": "'",
                       "“": '"', "”": '"'})
YAPISIK = set(".,!?;:)]}'\"")      # oncesine bosluk KOYMA
ACAN = set("([{")                  # sonrasina bosluk KOYMA


def _hikayeler(yol, parca_mb=64, en_mb=None):
    """Dosyayi parca parca oku, HIKAYE SINIRINDA kes, hikaye hikaye ver."""
    return _hikayeler_kar(yol, parca_mb * 1024 * 1024,
                          None if en_mb is None else en_mb * 1024 * 1024)


def _hikayeler_kar(yol, parca, dur=None):
    """parca/dur KARAKTER.  En fazla `dur` karakter okunur, yarim kalan son
    hikaye atilir.  Eskiden bolumden artan sayilmiyordu: en_mb=64 fiilen
    128 MB okuyordu (onkayit model_17_TAM1)."""
    art, okunan = "", 0
    with open(yol, encoding="utf-8", errors="replace") as f:
        while dur is None or okunan < dur:
            p = f.read(parca if dur is None else min(parca, dur - okunan))
            if not p:
                if art.strip():
                    yield art.translate(DUZLE)          # dosya bitti: son hikaye tam
                return
            okunan += len(p)
            p = art + p
            k = p.rfind(SINIR)
            if k < 0:
                art = p
                continue
            art, p = p[k + len(SINIR):], p[:k]
            for h in p.split(SINIR):
                if h.strip():
                    yield h.translate(DUZLE)


def genel(ad):
    """Eski sozlukteki <hikaye>'yi genel <eos>'a cevir.  Kimlik AYNI kalir,
    yani onbellekteki akislar ve eski paketler oldugu gibi gecerli."""
    return [SON if a == HIKAYE else a for a in ad]


def sozluk(yol, en: int, parca_mb=64, en_mb=None, yaz=print):
    """En sik `en` kelime + <dolgu> + <eos> + <bilinmeyen>.  EGITIMDEN cikar."""
    say = collections.Counter()
    for h in _hikayeler(yol, parca_mb, en_mb):
        say.update(JETON.findall(h))
    top = sum(say.values())
    ad = [DOLGU, SON, BILINMEYEN] + [a for a, _ in say.most_common(en)]
    kap = sum(c for _, c in say.most_common(en)) / top
    yaz("  sozluk  %s farkli kelime gorundu -> en sik %s tutuldu"
        % ("{:,}".format(len(say)), "{:,}".format(en)))
    yaz("          kapsama %%%.3f   disarda %s kelime"
        % (100 * kap, "{:,}".format(max(0, len(say) - en))))
    return ad, {a: i for i, a in enumerate(ad)}


def akis(yol, ix, parca_mb=64, en_mb=None, yaz=print):
    """Hikayeleri TEK akisa diz, aralarina <eos>.  int16 dizi.

    int16: en buyuk kimlik 4.002 < 32.767, KAYIPSIZ.  Yarim dosya,
    yarim yukleme, Colab'da yarim RAM."""
    assert len(ix) <= np.iinfo(np.int16).max + 1, "sozluk int16'ya sigmiyor"
    bl, hk = ix[BILINMEYEN], ix[SON]
    cik, n = [], 0
    for h in _hikayeler(yol, parca_mb, en_mb):
        cik.append(np.fromiter((ix.get(t, bl) for t in JETON.findall(h)),
                               dtype=np.int16))
        cik.append(np.array([hk], dtype=np.int16))
        n += 1
    a = np.concatenate(cik)
    yaz("  akis    %s hikaye   %s jeton   bilinmeyen %%%.2f"
        % ("{:,}".format(n), "{:,}".format(len(a)),
           100 * float((a == bl).mean())))
    return a


def pencere(a: np.ndarray, T: int, uret=None, hikaye=None, dolgu=0):
    """HER HIKAYE = BIR PENCERE, basinda ve sonunda <eos>.  Doner: (P, M).

    Kullanici: *"her hikaye bence 1 pencere olmali yoksa modele dogru tam
    hikaye ogretmemis oluruz"* ve *"bos ve eos gerekli"*.  `hikaye`: <eos>'un
    kimligi.

        <eos> w1 w2 ... wL <eos> <dolgu> ...
        bastaki BOS: w1 de tahmin edilir.  sondaki EOS: model bitisi
        ogrenir, uretim durabilir.

    Kalan yer <dolgu>, maskede False.  Tasma YOK.  L + 2 > T olan hikayeler
    ATILIR (bolmek "tam hikaye" ilkesini bozardi).
    OLCULDU, T=256, BOS/EOS'SUZ hali: hikayelerin %89,8'i kaliyor, dolgu
    %33,7.  T=384 -> %95,7 ama dolgu %53,5;  T=512 -> %98,5 ama dolgu %63,7.

    hikaye verilmezse akis DUZ kesilir (maske hep True).
    """
    if hikaye is None:
        n = len(a) // T
        P = a[:n * T].reshape(n, T)
        M = np.ones(P.shape, dtype=bool)
    else:
        sn = np.flatnonzero(a == hikaye)
        bas = np.concatenate([[0], sn + 1])[:len(sn)]   # her hikayenin basi
        uz = sn - bas                                   # ayrac haric kelime
        tut = (uz > 0) & (uz + 2 <= T)                  # + BOS + EOS
        bas, uz = bas[tut], uz[tut]
        P = np.full((len(bas), T), dolgu, dtype=a.dtype)
        M = np.zeros((len(bas), T), dtype=bool)
        P[:, 0] = hikaye
        for i, (b, L) in enumerate(zip(bas, uz)):
            P[i, 1:L + 1] = a[b:b + L]
            P[i, L + 1] = hikaye
            M[i, :L + 2] = True
    if uret is not None:
        j = uret.permutation(len(P))
        P, M = P[j], M[j]
    return P, M


def coz(P, ad, i=0, en=None) -> str:
    """Pencereyi OKUNUR metne cevir.  Sayi degil METIN gormek icin."""
    d = P[i] if getattr(P, "ndim", 1) > 1 else P      # yigin ya da tek dizi
    d = d[:en]
    s, acik, tirnak = "", False, False
    for x in d:
        t = ad[int(x)]
        if t in (SON, HIKAYE):
            s += "\n\n---\n\n"
            tirnak = False
            continue
        # Tirnak hem acar hem kapar; sirayi sayarak ayiriyoruz.
        if t in "\"'" and t != "'":
            yapisik, acacak = tirnak, not tirnak
            tirnak = not tirnak
        else:
            yapisik, acacak = t in YAPISIK, t in ACAN
        s += t if (not s or s.endswith(("\n", " ")) or yapisik or acik) \
            else " " + t
        acik = acacak
    return s.strip()


def iz(*diziler) -> str:
    """Parmak izi -- veri kayarsa kapi yakalasin."""
    h = hashlib.sha256()
    for d in diziler:
        h.update(repr(d).encode() if isinstance(d, list)
                 else np.ascontiguousarray(d).tobytes())
    return h.hexdigest()[:16]


def kur(kok: str, T: int = 128, en: int = 4000, en_mb=None, tohum: int = 0,
        hizali: bool = True, yaz=print):
    """TinyStories -> (ad, EG, DG).

    kok    TinyStories dosyalarinin durdugu klasor (Drive)
    T      pencere uzunlugu.  dizi() dikkati (B,T,T) -- T ile KARESEL.
    en     sozluk kirpmasi (+ <dolgu> + <eos> + <bilinmeyen>)
    en_mb  yalniz ilk N MB (deneme icin).  None -> hepsi.
    hizali pencereler HIKAYE BASINA hizalansin mi.  Duz kesimde
           tahminlerin %40,8'i hikayesinin basini GORMUYORDU (olculdu).

    Uretilen akis onbellege yazilir; ikinci cagri OKUR (kural 9).
    Veri izini train_17 egitim tensorunden hesaplayip pakete yazar."""
    ob = os.path.join(kok, "onbellek")
    os.makedirs(ob, exist_ok=True)
    yol = {b: os.path.join(kok, "TinyStoriesV2-GPT4-%s.txt" % b)
           for b in ("train", "valid")}
    # SURUM onbellek anahtarinda: yapi degisince eski onbellek SESSIZCE
    # kullanilmasin.  en_mb'li akislar v3: eskisi (v2) iki kati okumustu.
    etiket = ("tam_n%d_v2" % en if en_mb is None
              else "%dmb_n%d_v3" % (en_mb, en))

    ps = os.path.join(ob, "sozluk_%s.npy" % etiket)
    if os.path.exists(ps):
        ad = genel(list(np.load(ps, allow_pickle=True)))
        yaz("  sozluk  onbellekten  %s birim" % "{:,}".format(len(ad)))
    else:
        ad, _ = sozluk(yol["train"], en, en_mb=en_mb, yaz=yaz)
        np.save(ps, np.array(ad, dtype=object))
    ix = {a: i for i, a in enumerate(ad)}

    A = {}
    for b in ("train", "valid"):
        p = os.path.join(ob, "akis_%s_%s.npy" % (b, etiket))
        if os.path.exists(p):
            A[b] = np.load(p)
            yaz("  %-6s onbellekten  %s jeton" % (b, "{:,}".format(len(A[b]))))
        else:
            A[b] = akis(yol[b], ix, en_mb=en_mb if b == "train" else None,
                        yaz=yaz)
            np.save(p, A[b])

    uret = np.random.default_rng(tohum)
    hk = ix[SON] if hizali else None
    EG, EM = pencere(A["train"], T, uret, hk, ix[DOLGU])
    DG, DM = pencere(A["valid"], T, uret, hk, ix[DOLGU])
    yaz("  pencere T=%d   %s   egitim %s   dogrulama %s"
        % (T, "HER HIKAYE BIR PENCERE, <eos> ... <eos>" if hizali
           else "duz kesim", "{:,}".format(len(EG)), "{:,}".format(len(DG))))
    yaz("  dolgu %%%.1f   etkin is %%%.1f   (L+2 > T olan hikayeler atildi)"
        % (100 * (1 - EM.mean()), 100 * EM.mean()))
    return ad, (EG, EM), (DG, DM)


# ============================================================
# OLCUT -- olcme_17'den.  Hedef: SONRAKI KELIME, dolgu disinda her konum.
# ============================================================
BANTLAR = ((0, 64), (64, 256), (256, 512))     # hedefin penceredeki konumu
# URETIM ayari (egitimi degistirmez): son TEKRAR_PENCERESI token'da gecmis KELIMELERIN
# olasiligi TEKRAR_CEZASI'na bolunur; noktalama cezasiz.  1,0 = kapali.
# REL07 sicaklik 0'da "The dog was very happy." dongusune giriyordu (24 Eylul).
TEKRAR_CEZASI = 1.0
TEKRAR_PENCERESI = 20
# URETIM ayari: nucleus (top-p).  Olasiliklari buyukten kucuge toplami TOP_P'ye ulasan kume
# disindaki kelimeler atilir, kalanlardan secilir.  1,0 = kapali.  Holtzman ve ark. 2019.
TOP_P = 1.0
# SAGLIK sondasi, her kosuda AYNI: ilk SONDA_PENCERE tutulan pencere (m.health) ve sabit istemler (dongu).
# Kullanici, 25 Eylul: "nereye bakacağımızı anlamak için ölçüm ... eğitim boyunca takip edebilmek için".
SONDA_PENCERE = 64
SONDA_ISTEM = 3        # istem dosyasindan butun kelimeleri sozlukte olan ilk 3 + isim istemi
ISIM_ISTEMI = ("Once upon a time, there was a girl named Lily. She had a red ball. One day, Lily went to the "
               "park with her ball. At the park, she met a boy named Tom. Tom asked,")


def tekrar_ikili(w, a):
    """w (B,T) token, a (B,T-1) sayilan hedef -> (B,T-1) bool: hedef w_(t+1), (w_t, w_(t+1)) ikilisini BU
    hikayede daha once tamamlanmis -- Zoology'nin (2312.04927) "AR hit"i, egitimdeki siklik suzgeci olmadan."""
    B, L = a.shape
    n = int(w.max()) + 1
    sira = torch.arange(B * L, device=w.device).view(B, L)
    satir = torch.arange(B, device=w.device)[:, None] * n * n
    anahtar = torch.where(a, satir + w[:, :-1] * n + w[:, 1:], -1 - sira).reshape(-1)   # sayilmayan eslesmez
    o = torch.sort(anahtar, stable=True).indices          # esit anahtarlar konum sirasini korur
    tekrar = torch.zeros_like(anahtar, dtype=torch.bool)
    tekrar[o[1:]] = anahtar[o[1:]] == anahtar[o[:-1]]     # ilk gecis haric hepsi
    return tekrar.view(B, L)


def olc(m, W, M, eos, aygit="cuda", parca=64):
    """accuracy, ce ve tani.  W, M CPU'da; parca parca tasinir.

      accuracy       sonraki kelime BIREBIR (en yuksek puan)
      ce             ayni hedeflerde kayip; ppl = e^ce
      acc_a_b        hedef konumu a..b arasinda olanlarda accuracy --
                     C hikayenin neresinde doyuyor
      eos_ok         hikaye BITTIGINDE en yuksek puan EOS mu
      acc_ar         hedef, hikayede daha once gecmis bir ikiliyi tamamliyor (tekrar_ikili): geri cagirma
      acc_other      geri kalan hedefler"""
    dg = tp = 0
    ce_top = 0.0
    bant = {b: [0, 0] for b in BANTLAR}
    eo = [0, 0]
    ar = {"ar": [0, 0], "other": [0, 0]}
    with torch.no_grad():
        for i in range(0, W.shape[0], parca):
            w = W[i:i + parca].to(aygit).long()
            a = M[i:i + parca].to(aygit)[:, 1:]
            p = m.scoreboard(w)[:, :-1]
            h = w[:, 1:]
            ce = F.cross_entropy(p.transpose(1, 2), h, reduction="none")
            d = (p.argmax(-1) == h) & a
            dg += int(d.sum())
            tp += int(a.sum())
            ce_top += float(ce[a].sum())
            k = torch.arange(1, h.shape[1] + 1, device=aygit)      # hedefin konumu
            for (b0, b1) in BANTLAR:
                s = a & (k >= b0) & (k < b1)
                bant[(b0, b1)][0] += int((d & s).sum())
                bant[(b0, b1)][1] += int(s.sum())
            s = a & (h == eos)
            eo[0] += int((d & s).sum())
            eo[1] += int(s.sum())
            r = tekrar_ikili(w, a)
            for ad, s in (("ar", a & r), ("other", a & ~r)):
                ar[ad][0] += int((d & s).sum())
                ar[ad][1] += int(s.sum())
    diag = {"acc_%d_%d" % b: (c / n if n else float("nan")) for b, (c, n) in bant.items()}
    diag["eos_ok"] = eo[0] / max(eo[1], 1)
    for ad, (c, n) in ar.items():
        diag["acc_" + ad] = c / n if n else float("nan")
    return {"accuracy": dg / max(tp, 1), "ce": ce_top / max(tp, 1), "diag": diag}


def olcut(EG, DG, eos, aygit="cuda", en=2000, en_tam=None, ad=None, istemler=()):
    """train'in bekledigi metric(m, "train"|"heldout", full, save) -> dict.
    Egitim sirasinda ilk `en` pencere.  full: held-out HEPSI; egitim bolmesinden
    `en_tam` pencere (None: held-out kadar) -- 2,6 milyon pencerenin tamami
    bir epokluk ileri gecis olurdu.
    health: held-out'ta her olcumde sabit sondada m.health(); ad (sozluk) verilirse tam yedekte (save)
    ve sonda (full) sabit istemlerle dongu ve metin de."""
    kume = {"train": EG, "heldout": DG}
    ix = {a: i for i, a in enumerate(ad)} if ad is not None else None
    sonda = sonda_istemleri(istemler, ix) if ix is not None else []
    w, mk = DG[0][:SONDA_PENCERE], DG[1][:SONDA_PENCERE]
    L = int(mk.any(0).nonzero().max()) + 1                    # sondanin en uzun hikayesine kirpilir

    def f(m, side, full=False, save=False):
        W, M = kume[side]
        if full:
            n = len(W) if side == "heldout" else (en_tam or len(DG[0]))
        else:
            n = en
        r = olc(m, W[:n], M[:n], eos, aygit)
        if side == "heldout" and hasattr(m, "health"):
            r["health"] = m.health(w[:, :L].to(aygit).long(), mk[:, :L].to(aygit))
            if sonda and (save or full):
                r["health"].update(dongu(m, ad, ix, sonda, aygit=aygit))
        return r

    f.health = True
    return f


def konum_tablo(m, W, M, aygit="cuda", dilim=8, parca=64, yaz=print):
    """accuracy'yi KONUMA gore boler: C'nin doydugu yer."""
    T = W.shape[1]
    dg, tp = torch.zeros(T - 1), torch.zeros(T - 1)
    with torch.no_grad():
        for i in range(0, W.shape[0], parca):
            w = W[i:i + parca].to(aygit).long()
            a = M[i:i + parca].to(aygit)[:, 1:]
            d = (m.scoreboard(w)[:, :-1].argmax(-1) == w[:, 1:]) & a
            dg += d.sum(0).float().cpu()
            tp += a.sum(0).float().cpu()
    n = len(dg)
    yaz("%12s %10s %11s" % ("konum", "accuracy", "hedef"))
    for k in range(dilim):
        a, b = k * n // dilim, (k + 1) * n // dilim
        s = float(tp[a:b].sum())
        yaz("%12s %10.4f %11s" % ("%d-%d" % (a + 1, b), float(dg[a:b].sum()) / max(s, 1),
                                  "{:,}".format(int(s))))
    yaz("%12s %10.4f %11s" % ("TUMU", float(dg.sum() / tp.sum().clamp(min=1)),
                              "{:,}".format(int(tp.sum()))))


# ============================================================
# METIN -- olcme_17.devam ve konus.istemler'den.  Sayi degil, modelin YAZDIGI.
# ============================================================
def istemler(kok):
    """Makalenin degerlendirme istemleri (Evaluation prompts.yaml)."""
    p = os.path.join(kok, "Evaluation prompts.yaml")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        y = f.read()
    return [re.sub(r"\s+", " ", b).strip() for b in re.split(r"\n- \|-\n", y)[1:]]


def devam(m, istem, ad, ix, adim=120, aygit="cuda", sicaklik=0.0, tohum=None,
          yasak=(DOLGU, BILINMEYEN), tekrar=TEKRAR_CEZASI, pencere=TEKRAR_PENCERESI,
          top_p=TOP_P):
    """Istemin devamini URET; model <eos> uretince durur.
    sicaklik 0: hep en yuksek puan.  yasak: uretilmeyecek token'lar.
    tekrar: son `pencere` token'da gecmis kelimelerin olasiligi tekrar'a bolunur.
    top_p: sicaklik > 0 iken yalniz olasilik toplami top_p'ye ulasan en olasi kelimelerden secilir."""
    kelime = torch.tensor([any(ch.isalpha() for ch in a) for a in ad], device=aygit)
    g = None if tohum is None else torch.Generator(device=aygit).manual_seed(tohum)
    w = [ix[SON]] + [ix.get(t, ix[BILINMEYEN]) for t in JETON.findall(istem.translate(DUZLE))]
    n_istem = len(w)
    y = torch.tensor([ix[t] for t in yasak], dtype=torch.long, device=aygit)
    with torch.no_grad():
        for _ in range(min(adim, m.t_max - len(w))):
            p = m.scoreboard(torch.tensor([w], device=aygit))[0, -1]
            p = torch.log_softmax(p.index_fill(0, y, -float("inf")), -1)
            if tekrar > 1.0:
                son = torch.tensor(w[-pencere:], device=aygit)
                son = son[kelime[son]]
                p = p.index_add(0, son.unique(), torch.full((len(son.unique()),),
                                                            -math.log(tekrar), device=aygit))
            if sicaklik <= 0:
                c = int(p.argmax())
            else:
                q = (p / sicaklik).softmax(-1)
                if top_p < 1.0:
                    qs, qi = q.sort(descending=True)
                    at = (qs.cumsum(0) - qs) >= top_p            # kumeden once toplam top_p'yi gecti
                    q = q.index_fill(0, qi[at], 0.0)
                c = int(torch.multinomial(q / q.sum(), 1, generator=g))
            w.append(c)
            if c == ix[SON]:
                break
    return coz(np.array(w[1:n_istem]), ad), coz(np.array(w[n_istem:]), ad)


def sonda_istemleri(istemler, ix, k=SONDA_ISTEM):
    """Sabit istemler: kelimelerinin hepsi sozlukte olan ilk k istem + isim istemi.  <bilinmeyen> uretimde
    yasak; istemdeki bir isim <bilinmeyen> ise model onu hic yazamaz (CMNORM t4000 okumasi, 25 Eylul)."""
    temiz = [s for s in istemler if all(t in ix for t in JETON.findall(s.translate(DUZLE)))]
    return temiz[:k] + [ISIM_ISTEMI]


def dongu(m, ad, ix, istemler, adim=60, aygit="cuda"):
    """Acgozlu uretim (sicaklik 0).  farkli4: farkli 4'lulerin orani (1: hic tekrar yok);
    dongu: tekrar eden 8'lisi olan istemlerin payi; metin: yazilanlar (pakete girer, sonradan gozle okunur)."""
    f4, d8, metin = [], [], []
    for istem in istemler:
        _, yazdi = devam(m, istem, ad, ix, adim=adim, aygit=aygit)
        t = JETON.findall(yazdi.replace("---", " "))
        g4 = [tuple(t[i:i + 4]) for i in range(len(t) - 3)]
        g8 = [tuple(t[i:i + 8]) for i in range(len(t) - 7)]
        f4.append(len(set(g4)) / len(g4) if g4 else 1.0)
        d8.append(len(set(g8)) < len(g8))
        metin.append(yazdi)
    return {"farkli4": sum(f4) / len(f4), "dongu": sum(d8) / len(d8), "metin": metin}
