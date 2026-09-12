# -*- coding: utf-8 -*-
"""
SIFIRDAN UC KOL — bellegin ADRESI nereden gelsin?

SORU (tek degisken):
    Ogrenilebilir bir arama bellegi, GIRDI TOKEN'LARINDAN mi yoksa
    GIZLI DURUMDAN mi adreslenmeli?

KOLLAR (parametre sayilari esitlenir):
    A  duz transformer, bellek YOK          (FFN genisletilerek esitlenir)
    B  bellek var, sorgu katman-0'dan       (token-adresli)   <- gercek rakip
    C  bellek var, sorgu katman-L/2'den     (latent-adresli)  <- onerilen
  B ve C ARASINDAKI TEK FARK: bellek modulunun okudugu tensor.
  Ayni parametreler, ayni sekiller, ayni enjeksiyon noktasi, ayni veri, ayni seed.

GOREV (sentetik, kontrollu):
    olgular  f(e, r) = e'          e: varlik, r: iliski
    1-hop    [Q1] e r ?  -> f(e,r)
    2-hop    [Q2] e r1 r2 ? -> f(f(e,r1), r2)
  Kopru varlik f(e,r1) soruda GECMEZ. Token-adresli bellek ona ancak
  sig bir fonksiyonla ulasabilir; latent-adresli bellek L/2 katman
  hesaptan sonra ulasabilir. Tez budur.

BOLME:
    EVAL_COMP  gorulmus varlik x GORULMEMIS iliski cifti   <- BIRINCIL
    EVAL_ENT   hic 2-hop egitimi gormemis varlik           <- ikincil (daha zor)
    EVAL_SEEN  egitilmis 2-hop kombinasyonlari             <- saglik kapisi
    EVAL_1HOP  1-hop dogruluk                              <- saglik kapisi

ONCEDEN YAZILMIS KARAR KURALI  (kod calismadan once sabit):
    KAPI-1  her uc kol da EVAL_1HOP >= 0.90 olmali. Degilse: YETERSIZ EGITIM.
    KAPI-2  en az bir kol EVAL_SEEN >= 0.80 olmali. Degilse: YETERSIZ EGITIM.
    KAPI-3  BELLEKLI HER KOLDA kullanim entropisi > log(32) olmali (slot cokmesi
            olursa sonuc "o kol kotu" degil, "bellek egitilemedi" demektir).
            Once yalnizca C kontrol ediliyordu; bu gozden kacmaydi -- 2000
            adimlik deneme kosusunda kol B memH=0.93 (~2.5 etkin slot) cikti
            ve kapiya takilmadi.
    BIRINCIL  d = acc_C - acc_B, EVAL_COMP uzerinde,
              varlik-kumeli esli bootstrap %95 GA.
        GA alt siniri > +0.05   -> LATENT ADRES KAZANDI
        GA sifiri iceriyor      -> BERABERE  (dort projenin dersi tekrar)
        GA genislik > 0.15      -> YETERSIZ VERI
    TABAN/TAVAN  butun kollar COMP'ta <0.10 ya da >0.95 ise fark olcmek anlamsiz;
                 sonuc "BERABERE" degil "SONUC YOK"tur.
    A kolu baglam icindir. A ~ C ise bellek hicbir sey katmiyor demektir;
    bu da rapor edilir.
    TEK SEED ile cikan sonuc "ON BULGU"dur. Kesin hukum icin >= 3 seed.

Colab:
    !pip -q install torch --upgrade   # zaten kurulu
    %run sifirdan.py
"""
import os, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================ AYARLAR =====================================
PRESET = os.environ.get("PRESET", "small")   # smoke | small | full
OUT    = os.environ.get("OUT", "./sifirdan_out")

CFG = dict(
    smoke=dict(N_ENT=200,  N_REL=6,  N_PAIR=6,  P_TRAIN=4,
               D=64,  L=4,  NH=4,  DFF=256,
               M=256,  DK=32,  DM=64,
               STEPS=300,   BATCH=64,  LR=3e-3, EVERY=100,  WD=0.01, SCHED=0),
    small=dict(N_ENT=4000, N_REL=8,  N_PAIR=12, P_TRAIN=9,
               D=384, L=8,  NH=6,  DFF=1536,
               M=4096, DK=128, DM=256,
               STEPS=8000,  BATCH=1024, LR=1e-3, EVERY=1000, WD=0.01, SCHED=0),
    full =dict(N_ENT=8000, N_REL=10, N_PAIR=16, P_TRAIN=12,
               D=512, L=8,  NH=8,  DFF=2048,
               M=8192, DK=128, DM=256,
               STEPS=20000, BATCH=384, LR=1e-3, EVERY=2000, WD=0.01, SCHED=0),
    # Besteleme (composition) egitim dogrulugu doyduktan COK SONRA ortaya cikiyor.
    # O yuzden karar kosusu BUYUK degil, KUCUK ve UZUN olmali: az olgu, yuksek
    # 2-hop/1-hop orani, cok adim.
    grok =dict(N_ENT=1000, N_REL=8,  N_PAIR=40, P_TRAIN=30,
               D=256, L=8,  NH=8,  DFF=1024,
               M=4096, DK=128, DM=256,
               STEPS=60000, BATCH=512, LR=1e-3, EVERY=5000, WD=0.1, SCHED=1),
    # TEKRARLI ERISIM deneyi. arXiv 2606.20737'nin Dense+Mem'i OOD grokking'i
    # 60-85 bin adimda goruyor; kosu 2 tam 60 binde kesilmisti. MEM_AT=2,4,6 ile
    # birlikte kullan.
    grok_uzun=dict(N_ENT=1000, N_REL=8,  N_PAIR=40, P_TRAIN=30,
               D=256, L=8,  NH=8,  DFF=1024,
               M=4096, DK=128, DM=256,
               STEPS=120000, BATCH=512, LR=1e-3, EVERY=5000, WD=0.1, SCHED=1),
    # ~150M. Bu gorev icin GEREKMEZ; karar 'small'dan cikar. T4'te kol basina
    # yaklasik 3-4 saat, uc kol bir gun. Sadece olcek merakiysa kullan.
    big  =dict(N_ENT=8000, N_REL=10, N_PAIR=16, P_TRAIN=12,
               D=1024, L=12, NH=16, DFF=4096,
               M=16384, DK=128, DM=512,
               STEPS=20000, BATCH=384, LR=6e-4, EVERY=2000, WD=0.01, SCHED=0),
)[PRESET]
for _k in list(CFG):                       # ortam degiskeniyle ezme: STEPS=4000 gibi
    if _k in os.environ:
        CFG[_k] = type(CFG[_k])(float(os.environ[_k]))
ARMS = tuple(os.environ.get("ARMS", "ABC"))

# Bellek KAC noktada okunuyor (1-tabanli blok sayisi; "4" = 4. bloktan sonra).
#   "4"      -> tek okuma, kosu 2 ile ayni
#   "2,4,6"  -> uc okuma; arXiv 2606.20737 Dense+Mem'i (12 katman, 3/6/9) 8
#               katmana olceklenmis hali. Son blok bilerek belleksiz kalir.
# Bellek MODULU paylasimli: okuma sayisi PARAMETRE EKLEMEZ, sadece hesap ekler.
# Dolayisiyla >1 okumada kollar arasi FLOP esitligi BOZULUR -- raporda yaz.
MEM_AT = tuple(sorted({int(v) for v in
                       os.environ.get("MEM_AT", str(CFG["L"] // 2)).split(",")}))
assert all(1 <= m < CFG["L"] for m in MEM_AT),     f"MEM_AT 1..L-1 araliginda olmali (son blok belleksiz): {MEM_AT}, L={CFG['L']}"
CFG["MEM_AT"] = MEM_AT

DATA_SEED   = 0          # veri TUM kollarda ve TUM seed'lerde AYNI
TRAIN_SEEDS = [int(s) for s in os.environ.get("SEEDS", "0").split(",")]
CTX         = 4          # bellek sorgusu kac pozisyonu birlestiriyor
N_BOOT      = 4000
UNSEEN_ENT_FRAC = 0.10
N_EVAL_MAX  = 3000       # her eval kumesinden en fazla bu kadar ornek
GATE_1HOP, GATE_SEEN, GATE_MEMENT = 0.90, 0.80, math.log(32)
DELTA_MIN, CI_TOO_WIDE = 0.05, 0.15
FLOOR, CEIL = 0.10, 0.95   # taban/tavan etkisi: fark olcmek anlamsiz
NL = chr(10)
COMPILE = os.environ.get("COMPILE", "1") != "0"
PROG    = int(os.environ.get("PROG", "1000"))   # kac adimda bir ilerleme satiri
ABORT_ON_FATAL = os.environ.get("ABORT_ON_FATAL", "1") != "0"
SAVE_CKPT  = os.environ.get("SAVE_CKPT", "1") != "0"
SAVE_TABLE = os.environ.get("SAVE_TABLE", "1") != "0"   # ornek basina uzun tablo
SAVE_KV    = os.environ.get("SAVE_KV", "1") != "0"      # bellek K/V anlik goruntusu
SAVE_CIRCUIT = os.environ.get("SAVE_CIRCUIT", "1") != "0"  # logit lens, dikkat, kafa ablasyonu
SAVE_SNAP  = os.environ.get("SAVE_SNAP", "1") != "0"    # ARA kontrol noktalari (fp16)
TOPSLOT    = int(os.environ.get("TOPSLOT", "8"))         # ornek basina saklanan slot sayisi
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "1"))     # kac degerlendirmede bir
# NOT: 12 Eylul 8.5M kosusunda 3 kullanildi (12 noktanin 4'u) ve bu cimrilikti.
# Depolama kisit degil (Drive 2 TB); 100M icin 12 anlik goruntu x 3 kol ~7 GB.
# Hangi olcumu yapmak isteyecegin onceden bilinmiyor: ayni gun uc olcum hatasi
# olcum YAPILDIKTAN SONRA bulundu, ikisinde ara kayit olmadigi icin kosu bastan
# baslatildi. TEK ISTISNA: kosu basladiktan sonra kollarin kayit sikligini
# DEGISTIRME -- kollar arasi tutarlilik cozunurlukten onemli.

os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ozel token'lar
PAD, Q1, Q2, QM, EOS = 0, 1, 2, 3, 4
SPECIAL = 8
REL_OFF = SPECIAL
ENT_OFF = SPECIAL + CFG["N_REL"]
VOCAB   = ENT_OFF + CFG["N_ENT"]
T_LEN   = 8


# ============================ VERI ========================================
def build_data():
    """Olgular + bolmeler. Hicbir egitim ornegi test kombinasyonunu icermez."""
    N, R, NP, PT = CFG["N_ENT"], CFG["N_REL"], CFG["N_PAIR"], CFG["P_TRAIN"]
    rng = np.random.RandomState(DATA_SEED)

    facts = rng.randint(0, N, size=(N, R))
    for _ in range(64):                                    # f(e,r) != e
        bad = facts == np.arange(N)[:, None]
        if not bad.any():
            break
        facts[bad] = rng.randint(0, N, size=int(bad.sum()))
    assert not (facts == np.arange(N)[:, None]).any(), "self-loop kaldi"

    # iliski ciftleri (r1 != r2), sabit kume
    allp = [(a, b) for a in range(R) for b in range(R) if a != b]
    pairs = [allp[i] for i in rng.permutation(len(allp))[:NP]]

    # varlik bolmesi
    perm = rng.permutation(N)
    n_unseen = max(1, int(round(UNSEEN_ENT_FRAC * N)))
    unseen_ent = np.sort(perm[:n_unseen])          # hic 2-hop egitimi yok
    seen_ent = np.sort(perm[n_unseen:])

    # gorulmus varliklar icin cift bolmesi: PT egitim / kalani test
    tr2, comp = [], []
    for e in seen_ent:
        pp = rng.permutation(NP)
        for j in pp[:PT]:
            tr2.append((e, pairs[j][0], pairs[j][1]))
        for j in pp[PT:]:
            comp.append((e, pairs[j][0], pairs[j][1]))
    ent_eval = [(e, p[0], p[1]) for e in unseen_ent for p in pairs]

    def chains(lst):
        out = []
        for e, r1, r2 in lst:
            b = int(facts[e, r1])
            out.append((int(e), int(r1), int(r2), b, int(facts[b, r2])))
        return out

    tr2, comp, ent_eval = chains(tr2), chains(comp), chains(ent_eval)

    # --- sizinti denetimi: egitimdeki (e,r1,r2) kumesi testle kesismemeli
    trset = {(e, a, b) for e, a, b, _, _ in tr2}
    for nm, st in (("COMP", comp), ("ENT", ent_eval)):
        k = sum((e, a, b) in trset for e, a, b, _, _ in st)
        assert k == 0, f"{nm} sizintisi: {k}"
    assert len(set(unseen_ent) & set(seen_ent)) == 0

    one = [(int(e), int(r), int(facts[e, r])) for e in range(N) for r in range(R)]
    return facts, pairs, one, tr2, comp, ent_eval, seen_ent, unseen_ent


def enc_one(batch):
    """[Q1] e r ? ans EOS  -> hedef pozisyon 3"""
    X = np.zeros((len(batch), T_LEN), np.int64)
    for i, (e, r, a) in enumerate(batch):
        X[i, :6] = [Q1, ENT_OFF + e, REL_OFF + r, QM, ENT_OFF + a, EOS]
    return (X, np.full(len(batch), 3, np.int64),
            np.array([ENT_OFF + a for _, _, a in batch], np.int64),
            np.array([e for e, _, _ in batch], np.int64))


def enc_two(batch):
    """[Q2] e r1 r2 ? ans EOS  -> hedef pozisyon 4"""
    X = np.zeros((len(batch), T_LEN), np.int64)
    for i, (e, r1, r2, b, a) in enumerate(batch):
        X[i, :7] = [Q2, ENT_OFF + e, REL_OFF + r1, REL_OFF + r2, QM, ENT_OFF + a, EOS]
    return (X, np.full(len(batch), 4, np.int64),
            np.array([ENT_OFF + a for *_, a in batch], np.int64),
            np.array([e for e, *_ in batch], np.int64))


# ============================ MODEL =======================================
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.g = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        f = x.float()
        return (self.g.float() * f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True)
                                                 + self.eps)).to(x.dtype)


class Block(nn.Module):
    def __init__(self, d, nh, dff):
        super().__init__()
        self.nh, self.hd = nh, d // nh
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.po = nn.Linear(d, d, bias=False)
        self.f1 = nn.Linear(d, dff, bias=False)
        self.f2 = nn.Linear(dff, d, bias=False)
        self.abl = None          # kapatilacak kafa indeksleri (tani icin)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, -1)
        sh = lambda t: t.view(B, T, self.nh, self.hd).transpose(1, 2)
        o = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), is_causal=True)
        if self.abl is not None:
            o = o.clone(); o[:, self.abl, :, :] = 0
        x = x + self.po(o.transpose(1, 2).reshape(B, T, D))
        return x + self.f2(F.gelu(self.f1(self.n2(x))))


def shift(x, n):
    return x if n == 0 else F.pad(x, (0, 0, n, 0))[:, :x.shape[1], :]


class Memory(nn.Module):
    """Ogrenilen anahtar/deger tablosu. Sorgu CTX pozisyonun birlesiminden."""
    def __init__(self, d, M, dk, dm, ctx=CTX):
        super().__init__()
        self.ctx = ctx
        self.n = RMSNorm(d)
        self.wg = nn.Linear(ctx * d, d, bias=False)
        self.wq = nn.Linear(d, dk, bias=False)
        self.K = nn.Parameter(torch.randn(M, dk) * 0.02)
        self.V = nn.Parameter(torch.randn(M, dm) * 0.02)
        self.wo = nn.Linear(dm, d, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(16.0)))

    def forward(self, src):
        x = self.n(src)
        u = self.wg(torch.cat([shift(x, i) for i in range(self.ctx)], -1))
        q = F.normalize(self.wq(u).float(), dim=-1)
        k = F.normalize(self.K.float(), dim=-1)
        s = q @ k.t() * self.logit_scale.exp().clamp(1.0, 100.0)
        w = s.softmax(-1)
        r = w @ self.V.float()
        return self.wo(r.to(src.dtype)), w


class Net(nn.Module):
    def __init__(self, arm, cfg):
        super().__init__()
        self.arm, d, L = arm, cfg["D"], cfg["L"]
        self.mem_at = tuple(cfg.get("MEM_AT", (L // 2,)))   # bu bloklardan SONRA
        dff = cfg["DFF"]
        if arm == "A":                           # parametre esitleme
            mp = mem_params(cfg)
            dff = cfg["DFF"] + int(round(mp / (2.0 * d * L)))
        self.emb = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(T_LEN, d)
        self.blocks = nn.ModuleList([Block(d, cfg["NH"], dff) for _ in range(L)])
        self.nf = RMSNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)
        self.head.weight = self.emb.weight        # bagli
        self.mem = None if arm == "A" else Memory(d, cfg["M"], cfg["DK"], cfg["DM"])
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def _mem_uygula(self, h, h0, j, no_mem=False):
        """Bellegi (varsa) j. bloktan SONRA uygula. Donen (h, w|None).

        B: sorgu h0'dan (token-adresli) -> okuma sayisi kacsa OLSUN ayni sorgu,
           ayni cevap. Tekrarli erisim B'ye yapisal olarak bir sey katamaz.
        C: sorgu h'den (latent)         -> her okumada GUNCELLENMIS sorgu.
        Tekrarli erisim deneyinin tum ayrimi bu iki satirda."""
        if self.mem is None or no_mem or (j + 1) not in self.mem_at:
            return h, None
        m, w = self.mem(h0 if self.arm == "B" else h)
        return h + m, w

    def forward(self, x, want_w=False, no_mem=False):
        h = self.emb(x) + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        h0, w = h, None
        for i, blk in enumerate(self.blocks):
            h = blk(h)
            h, w_ = self._mem_uygula(h, h0, i, no_mem)
            if w_ is not None:
                w = w_                                  # SON okumanin agirliklari
        return self.head(self.nf(h)), (w if want_w else None)


def mem_params(cfg):
    d, M, dk, dm = cfg["D"], cfg["M"], cfg["DK"], cfg["DM"]
    return CTX * d * d + d * dk + M * dk + M * dm + dm * d + d + 1


def nparam(m):
    seen, tot = set(), 0
    for p in m.parameters():
        if id(p) not in seen:
            seen.add(id(p)); tot += p.numel()
    return tot


# ============================ EGITIM / OLCUM ==============================
@torch.no_grad()
def evaluate(model, X, tp, tt, bs=512):
    """entity-kisitli argmax (birincil) + serbest argmax."""
    model.eval()
    ok_r, ok_f = [], []
    ent_lo, ent_hi = ENT_OFF, ENT_OFF + CFG["N_ENT"]
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(DEV)
        with torch.autocast(DEV, dtype=torch.float16, enabled=(DEV == "cuda")):
            lg, _ = model(xb)
        lg = lg.float()
        idx = torch.from_numpy(tp[i:i + bs]).to(DEV)
        lg = lg[torch.arange(len(idx), device=DEV), idx]
        g = torch.from_numpy(tt[i:i + bs]).to(DEV)
        ok_f.append((lg.argmax(-1) == g).cpu().numpy())
        ok_r.append((lg[:, ent_lo:ent_hi].argmax(-1) + ent_lo == g).cpu().numpy())
    model.train()
    return np.concatenate(ok_r), np.concatenate(ok_f)


@torch.no_grad()
def mem_entropy(model, X, bs=512, nmax=1024):
    if model.mem is None:
        return float("nan")
    acc = None
    for i in range(0, min(len(X), nmax), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(DEV)
        with torch.autocast(DEV, dtype=torch.float16, enabled=(DEV == "cuda")):
            _, w = model(xb, want_w=True)
        w = w.float().reshape(-1, w.shape[-1]).mean(0)
        acc = w if acc is None else acc + w
    p = (acc / acc.sum()).cpu().numpy()
    return float(-(p * np.log(p + 1e-12)).sum())


@torch.no_grad()
def zengin(model, E, gold_bridge, shortcut_tgt, bs=512):
    """TANI (kesifsel): dogruluk + altin olasilik + kopru sirasi + kisayol + ablasyon."""
    X, tp, tt = E[0], E[1], E[2]
    lo, hi = ENT_OFF, ENT_OFF + CFG["N_ENT"]
    ok, okab, gp, br, br5, sc = [], [], [], [], [], []
    pred, pp, sl, slw, slH = [], [], [], [], []
    model.eval()
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i+bs]).to(DEV)
        idx = torch.from_numpy(tp[i:i+bs]).to(DEV)
        ar = torch.arange(len(idx), device=DEV)
        with torch.autocast(DEV, dtype=torch.float16, enabled=(DEV == "cuda")):
            l1, w = model(xb, want_w=True)
            l2, _ = model(xb, no_mem=True)
        p = l1.float()[ar, idx][:, lo:hi].softmax(-1)
        g = torch.from_numpy(tt[i:i+bs]).to(DEV) - lo
        ok.append((p.argmax(-1) == g).cpu().numpy())
        gp.append(p[ar, g].cpu().numpy())
        okab.append((l2.float()[ar, idx][:, lo:hi].argmax(-1) == g).cpu().numpy())
        b = torch.from_numpy(gold_bridge[i:i+bs]).to(DEV) - lo   # varlik dilimine gore
        pb = p[ar, b]
        br.append(((p > pb[:, None]).sum(-1)).float().cpu().numpy())   # koprunun sirasi
        br5.append((torch.topk(p, 5, -1).indices == b[:, None]).any(-1).cpu().numpy())
        sc.append((p.argmax(-1) + lo ==
                   torch.from_numpy(shortcut_tgt[i:i+bs]).to(DEV)).cpu().numpy())
        pred.append((p.argmax(-1) + lo).cpu().numpy())
        pp.append(p.max(-1).values.cpu().numpy())
        if w is not None:                      # ornek basina EN COK YANAN 3 SLOT
            ww = w[torch.arange(len(idx), device=DEV), idx].float()
            tk = torch.topk(ww, TOPSLOT, -1)
            sl.append(tk.indices.cpu().numpy()); slw.append(tk.values.cpu().numpy())
            slH.append((-(ww * torch.log(ww + 1e-12)).sum(-1)).cpu().numpy())
    model.train()
    c = lambda v: np.concatenate(v)
    ozet = dict(acc=float(c(ok).mean()), acc_nomem=float(c(okab).mean()),
                gold_prob=float(c(gp).mean()),
                bridge_rank=float(np.median(c(br))), bridge_top5=float(c(br5).mean()),
                shortcut=float(c(sc).mean()))
    detay = dict(ok=c(ok).astype(np.int8), ok_nomem=c(okab).astype(np.int8),
                 ok_kisayol=c(sc).astype(np.int8), p_gold=c(gp).astype(np.float32),
                 p_pred=c(pp).astype(np.float32), pred=c(pred).astype(np.int32),
                 kopru_sira=c(br).astype(np.int32))
    if sl:
        S, W = c(sl), c(slw)
        for j in range(TOPSLOT):
            detay[f"slot{j+1}"] = S[:, j].astype(np.int32)
            detay[f"slot{j+1}_w"] = W[:, j].astype(np.float32)
        detay["slot_H"] = c(slH).astype(np.float32)
    return ozet, detay


@torch.no_grad()
@torch.no_grad()
def _hid_tum(model, X, poz, bs=512):
    """TEK forward'da katman 0..L icin CTX penceresi. Kollar arasi
    KIYASLANABILIR derinlik profili verir -- eski surum her kolu kendi bellek
    okuma noktasinda prubluyordu (A/C katman 4, B katman 0) ve bu iki sayi
    ayni seyi olcmuyordu."""
    L = len(model.blocks)
    out = [[] for _ in range(L + 1)]
    model.eval()
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i+bs]).to(DEV)
        idx = torch.from_numpy(poz[i:i+bs]).to(DEV)
        ar = torch.arange(len(idx), device=DEV)
        h = model.emb(xb) + model.pos(torch.arange(xb.shape[1], device=DEV))[None]
        h0 = h
        hs = [h]
        for j, blk in enumerate(model.blocks):
            h = blk(h)
            h, _ = model._mem_uygula(h, h0, j)
            hs.append(h)
        for j, hh in enumerate(hs):
            w = torch.stack([hh[ar, (idx - t).clamp(min=0)] for t in range(CTX)], 1)
            out[j].append(w.reshape(len(idx), -1).float().cpu().numpy())
    model.train()
    return [np.concatenate(o) for o in out]


def _ridge_rank(Ztr, btr, Zte, bte, Emb, lam=100.0):
    mu = Ztr.mean(0); Ztr, Zte = Ztr - mu, Zte - mu
    G = Ztr.T @ Ztr
    A = np.linalg.solve(G + lam * np.eye(G.shape[0]), Ztr.T @ Emb[btr])
    P = Zte @ A
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    S = P @ Emb.T
    g = S[np.arange(len(bte)), bte]
    return float(np.median((S > g[:, None]).sum(1) / max(1, Emb.shape[0] - 1)))


def kopru_profil(model, E, btr_idx, bte_idx, b_all, lam=100.0):
    """Katman katman kopru okunabilirligi. Donen: [L+1] liste, 0=tepe 0.5=sans.
    Onek tutulmus bolme kullanilir (ezber imkansiz)."""
    Z = _hid_tum(model, E[0], E[1])
    Emb = model.emb.weight.detach()[ENT_OFF:ENT_OFF + CFG["N_ENT"]].float().cpu().numpy()
    Emb = Emb / (np.linalg.norm(Emb, axis=1, keepdims=True) + 1e-8)
    return [_ridge_rank(z[btr_idx], b_all[btr_idx], z[bte_idx], b_all[bte_idx], Emb, lam)
            for z in Z]


@torch.no_grad()
def _hid(model, X, poz, depth=None, bs=512):
    """depth blok sonrasi gizli durum. depth=0 -> saf gomme (kontrol kolu)."""
    out = []
    model.eval()
    if depth is None:
        depth = 0 if model.arm == "B" else model.mem_at[0]   # ILK okuma noktasi
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i+bs]).to(DEV)
        h = model.emb(xb) + model.pos(torch.arange(xb.shape[1], device=DEV))[None]
        for j, blk in enumerate(model.blocks):
            if j >= depth:
                break
            h = blk(h)
        idx = torch.from_numpy(poz[i:i+bs]).to(DEV)
        ar = torch.arange(len(idx), device=DEV)
        # CTX penceresi: poz, poz-1, ... -> bellegin gordugu bilginin AYNISI.
        # Kritik: tek pozisyon alinirsa depth=0 kontrolu BOS olur ('?' her
        # ornekte ayni token) ve kontrol vakum olur.
        win = torch.stack([h[ar, (idx - t).clamp(min=0)] for t in range(CTX)], 1)
        out.append(win.reshape(len(idx), -1).float().cpu().numpy())
    model.train()
    return np.concatenate(out)


def kopru_probu(model, Etr, btr, Ete, bte, lam=100.0, depth=None):
    """TANI: gizli durumdan KOPRU varliginin gommesi dogrusal okunabiliyor mu?
    Donen deger normalize sira: 0 = mukemmel, 0.5 = sans.

    DIKKAT: tek basina yaniltici. Prob, (e,r1) onekini egitimde gorup kopruyu
    EZBERLEYEBILIR (tuttugumuz sey cift (r1,r2), tek basina r1 degil). O yuzden
    her zaman depth=0 KONTROLU ile birlikte okunmali: fark, modelin hesabinin
    EKLEDIGI bilgidir. depth=0 ile ayni cikiyorsa hesap bir sey eklemiyor."""
    Ztr, Zte = (_hid(model, Etr[0], Etr[1], depth),
                _hid(model, Ete[0], Ete[1], depth))
    Emb = model.emb.weight.detach()[ENT_OFF:ENT_OFF + CFG["N_ENT"]].float().cpu().numpy()
    Emb = Emb / (np.linalg.norm(Emb, axis=1, keepdims=True) + 1e-8)
    mu = Ztr.mean(0); Ztr, Zte = Ztr - mu, Zte - mu
    G = Ztr.T @ Ztr
    A = np.linalg.solve(G + lam * np.eye(G.shape[0]), Ztr.T @ Emb[btr])
    P = Zte @ A
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    S = P @ Emb.T
    g = S[np.arange(len(bte)), bte]
    return float(np.median((S > g[:, None]).sum(1) / max(1, Emb.shape[0] - 1)))


def _alt(E, idx):
    return (E[0][idx], E[1][idx], E[2][idx], E[3][idx])


def onek_bolme(LL):
    """Prob icin (e, r1) ONEKINE gore bolme.

    NEDEN: olgular rastgele bir tablo; f(e,r1) hicbir kuralla tahmin edilemez,
    sadece EZBERLENEBILIR. Prob'u, degerlendirdigi onekler uzerinde egitirsen
    kopruyu modelin hesabindan degil, (e,r1) kimliginden okur -- ham gommelerden
    bile. (Olculdu: L0 0.025.) Onek tutulunca ezber imkansiz olur; geriye kalan
    okunabilirlik modelin kalintisinda TUTARLI BIR YONDE yazili olmasindan gelir."""
    key = [(e, r1) for e, r1, _, _, _ in LL]
    uq = sorted(set(key))
    rs = np.random.RandomState(DATA_SEED + 77)
    half = {uq[i] for i in rs.permutation(len(uq))[:len(uq) // 2]}
    tr = np.array([i for i, k in enumerate(key) if k in half], np.int64)
    te = np.array([i for i, k in enumerate(key) if k not in half], np.int64)
    return tr, te


def prob_seti(model, Etr, Ete, btr, bte, atr, ate, salt=0, LS=None):
    """TANI paketi. Her biri ayni prob makinesiyle, KIYASLANABILIR.
      kopru      : kopru gizli durumdan okunabiliyor mu
      kopru_L0   : ayni sey HAM GOMMELERDEN (CTX penceresi) -- kontrol
      kopru_null : kopru etiketleri KARISTIRILMIS -- prob'un taban gurultusu
      cevap      : ayni sey ALTIN CEVAP icin (zamanlama: once hangisi belirir?)
    kopru ~ kopru_L0 ise modelin hesabi bir sey EKLEMIYOR.
    kopru ~ kopru_null ise sinyal yok."""
    r = np.random.RandomState(DATA_SEED + 500 + salt)
    sh = btr[r.permutation(len(btr))]
    out = {}
    for nm, tr, te, dep in (("kopru", btr, bte, None), ("kopru_L0", btr, bte, 0),
                            ("kopru_null", sh, bte, None), ("cevap", atr, ate, None)):
        try:
            out[nm] = kopru_probu(model, Etr, tr, Ete, te, depth=dep)
        except Exception as ex:
            out[nm] = float("nan"); out[nm + "_hata"] = str(ex)[:60]
    out["kopru_kazanc"] = out["kopru_L0"] - out["kopru"]      # hesabin EKLEDIGI
    out["kopru_net"] = out["kopru_null"] - out["kopru"]       # null uzerine net

    # --- ASIL OLCUM: (e,r1) onegi TUTULMUS prob. Ezber imkansiz.
    if LS is not None:
        ti, vi = onek_bolme(LS)
        if len(ti) > 32 and len(vi) > 32:
            Ea, Eb = _alt(Etr, ti), _alt(Etr, vi)
            ba, bb = btr[ti], btr[vi]
            for nm, dep in (("ho_kopru", None), ("ho_kopru_L0", 0)):
                try:
                    out[nm] = kopru_probu(model, Ea, ba, Eb, bb, depth=dep)
                except Exception as ex:
                    out[nm] = float("nan"); out[nm + "_hata"] = str(ex)[:60]
            try:
                out["ho_kopru_null"] = kopru_probu(
                    model, Ea, ba[r.permutation(len(ba))], Eb, bb)
            except Exception:
                out["ho_kopru_null"] = float("nan")
    return out


@torch.no_grad()
def bellek_istat(model, X, bs=512, nmax=2048):
    if model.mem is None:
        return dict(mem_H=float("nan"), mem_olu=float("nan"), mem_tepe=float("nan"))
    acc = None
    model.eval()
    for i in range(0, min(len(X), nmax), bs):
        xb = torch.from_numpy(X[i:i+bs]).to(DEV)
        with torch.autocast(DEV, dtype=torch.float16, enabled=(DEV == "cuda")):
            _, w = model(xb, want_w=True)
        w = w.float().reshape(-1, w.shape[-1]).sum(0)
        acc = w if acc is None else acc + w
    model.train()
    p = (acc / acc.sum()).cpu().numpy()
    return dict(mem_H=float(-(p * np.log(p + 1e-12)).sum()),
                mem_olu=float((p < 1e-6).mean()), mem_tepe=float(p.max()))


@torch.no_grad()
def logit_lens(model, E, bridge, bs=512):
    """TANI: her katmanin kalintisi cikisa projelenince ALTIN ve KOPRU kacinci?
    'Kopru hangi katmanda beliriyor, cevap hangi katmanda?' sorusunun cevabi.
    Donen: [L+1] normalize sira (0 en iyi, 0.5 sans), altin ve kopru icin."""
    X, poz, tt = E[0], E[1], E[2]
    lo, hi = ENT_OFF, ENT_OFF + CFG["N_ENT"]
    L = len(model.blocks)
    rg = [[] for _ in range(L + 1)]; rb = [[] for _ in range(L + 1)]
    model.eval()
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i+bs]).to(DEV)
        idx = torch.from_numpy(poz[i:i+bs]).to(DEV)
        ar = torch.arange(len(idx), device=DEV)
        g = torch.from_numpy(tt[i:i+bs]).to(DEV) - lo
        b = torch.from_numpy(bridge[i:i+bs]).to(DEV) - lo
        h = model.emb(xb) + model.pos(torch.arange(xb.shape[1], device=DEV))[None]
        hs = [h]
        for j, blk in enumerate(model.blocks):
            h = blk(h)
            h, _ = model._mem_uygula(h, hs[0], j)
            hs.append(h)
        for j, hh in enumerate(hs):
            lg = model.head(model.nf(hh))[ar, idx][:, lo:hi].float()
            for tgt, box in ((g, rg), (b, rb)):
                v = lg[ar, tgt]
                box[j].append(((lg > v[:, None]).sum(-1).float()
                               / max(1, hi - lo - 1)).cpu().numpy())
    model.train()
    return ([float(np.median(np.concatenate(x))) for x in rg],
            [float(np.median(np.concatenate(x))) for x in rb])


@torch.no_grad()
def dikkat(model, X, poz, bs=256, nmax=1024):
    """TANI: cevap pozisyonunda her (katman, kafa) hangi pozisyona bakiyor?
    Donen: [L, NH, T] ortalama dikkat."""
    L, NH = len(model.blocks), model.blocks[0].nh
    T = X.shape[1]
    acc = np.zeros((L, NH, T), np.float64); n = 0
    model.eval()
    for i in range(0, min(len(X), nmax), bs):
        xb = torch.from_numpy(X[i:i+bs]).to(DEV)
        idx = torch.from_numpy(poz[i:i+bs]).to(DEV)
        ar = torch.arange(len(idx), device=DEV)
        h = model.emb(xb) + model.pos(torch.arange(T, device=DEV))[None]
        h0 = h
        for j, blk in enumerate(model.blocks):
            B_ = h.shape[0]
            q, k, _ = blk.qkv(blk.n1(h)).chunk(3, -1)
            sh = lambda t: t.view(B_, T, NH, blk.hd).transpose(1, 2)
            sc = (sh(q) @ sh(k).transpose(-1, -2)) / math.sqrt(blk.hd)
            m = torch.triu(torch.ones(T, T, device=DEV, dtype=torch.bool), 1)
            sc = sc.masked_fill(m, float("-inf")).softmax(-1)
            acc[j] += sc[ar, :, idx, :].float().sum(0).cpu().numpy()
            h = blk(h)
            h, _ = model._mem_uygula(h, h0, j)
        n += len(idx)
    model.train()
    return acc / max(1, n)


@torch.no_grad()
def kafa_ablasyonu(model, EC, E1, bs=512, nmax=1500):
    """TANI: her kafayi tek tek kapat, comp ve 1hop dususunu olc.
    2-hop'u bozup 1-hop'u bozmayan kafa = besteleme devresi."""
    def acc(E):
        X, poz, tt = E[0][:nmax], E[1][:nmax], E[2][:nmax]
        lo, hi = ENT_OFF, ENT_OFF + CFG["N_ENT"]
        o = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i+bs]).to(DEV)
            idx = torch.from_numpy(poz[i:i+bs]).to(DEV)
            ar = torch.arange(len(idx), device=DEV)
            with torch.autocast(DEV, dtype=torch.float16, enabled=(DEV == "cuda")):
                lg, _ = model(xb)
            pr = lg.float()[ar, idx][:, lo:hi].argmax(-1) + lo
            o.append((pr == torch.from_numpy(tt[i:i+bs]).to(DEV)).cpu().numpy())
        return float(np.concatenate(o).mean())

    model.eval()
    b0c, b01 = acc(EC), acc(E1)
    L, NH = len(model.blocks), model.blocks[0].nh
    dc = np.zeros((L, NH), np.float32); d1 = np.zeros((L, NH), np.float32)
    for l in range(L):
        for hd in range(NH):
            model.blocks[l].abl = [hd]
            dc[l, hd] = b0c - acc(EC); d1[l, hd] = b01 - acc(E1)
            model.blocks[l].abl = None
    model.train()
    return dict(taban_comp=b0c, taban_1hop=b01, dusus_comp=dc, dusus_1hop=d1)


@torch.no_grad()
def agirlik_istat(model):
    """TANI: modul basina agirlik normu + bellek tayfi (etkin rank).
    Weight decay altinda hangi modul buyuyor, bellek kac yonu gercekten kullaniyor."""
    o = {}
    gr = {}
    for n, p_ in model.named_parameters():
        k = ("mem_K" if n.endswith("mem.K") else "mem_V" if n.endswith("mem.V")
             else "mem" if ".mem." in n else "emb" if "emb" in n or "pos" in n
             else "attn" if (".qkv." in n or ".po." in n)
             else "ffn" if (".f1." in n or ".f2." in n) else "diger")
        gr[k] = gr.get(k, 0.0) + float(p_.detach().float().pow(2).sum())
    for k, v in gr.items():
        o[f"w_{k}"] = float(np.sqrt(v))
    if model.mem is not None:
        for nm, M_ in (("K", model.mem.K), ("V", model.mem.V)):
            sv = torch.linalg.svdvals(M_.detach().float())
            pw = (sv ** 2); pw = pw / pw.sum()
            o[f"mem{nm}_etkin_rank"] = float(torch.exp(
                -(pw * torch.log(pw + 1e-12)).sum()))     # entropi-tabanli rank
            o[f"mem{nm}_sv1_orani"] = float(pw[0])
        o["mem_scale"] = float(model.mem.logit_scale.detach().exp().clamp(1, 100))
    return o


def yaz_tablo(tablo, arm, seed):
    """Ornek basina uzun tablo: sonradan egitim tekrar etmeden analiz icin.
    Ana soru: AYNI KOPRUYE sahip sorular AYNI SLOTLARI mi yakiyor?"""
    cols = {}
    for k in tablo[0]:
        cols[k] = np.concatenate([t[k] for t in tablo])
    cols["kol"] = np.full(len(cols["step"]), arm, object)
    cols["seed"] = np.full(len(cols["step"]), seed, np.int32)
    yol = os.path.join(OUT, f"kayit_{arm}_s{seed}")
    try:
        import pandas as pd
        pd.DataFrame(cols).to_parquet(yol + ".parquet", index=False)
    except Exception:
        np.savez_compressed(yol + ".npz",
                            **{k: (v.astype("U8") if v.dtype == object else v)
                               for k, v in cols.items()})


def saglik(rec, curve, arm, adim, toplam):
    """ERKEN TESHIS — her degerlendirmede. Butun degerler zaten hesaplandi,
    ek maliyet yok. Amac: 1.4 saat kosup sonunda 'bellek olmus' demek yerine
    ilk olcumde bagirmak. (12 Eylul: weight decay C'nin bellegini 10k adimda
    oldurdu, 40 dk sonra tesadufen fark edildi.)
    Doner: (uyari listesi, olumcul_mu)"""
    u, olumcul = [], False
    ilerleme = adim / max(1, toplam)

    # Bellek kontrolleri icin ISINMA PAYI: ilk %10'da bellek henuz oturmamis
    # olabilir, orada uyarmak gurultu olur. Weight decay patolojisi %17'de
    # (10k/60k) zaten net gorunuyordu.
    if arm != "A" and ilerleme >= 0.10:
        sc = rec.get("mem_scale", float("nan"))
        if np.isfinite(sc) and sc < 8.0:
            u.append(f"bellek sicakligi dustu ({sc:.2f}, baslangic 16) "
                     "-> softmax yayvanlasiyor")
        H = rec.get("mem_H", float("nan"))
        if np.isfinite(H):
            if H < GATE_MEMENT:
                u.append(f"bellek cokmesi: memH {H:.2f} (~{math.exp(H):.0f} etkin slot)")
            elif H > 0.95 * math.log(CFG["M"]):
                u.append(f"bellek SECICI DEGIL: memH {H:.2f} ~ duzgun dagilim "
                         f"({math.log(CFG['M']):.2f})")
        for nm in ("memK_etkin_rank", "memV_etkin_rank"):
            r_ = rec.get(nm, float("nan"))
            if np.isfinite(r_) and r_ < 5:
                u.append(f"{nm} cokmesi: {r_:.1f}")

    if ilerleme > 0.25 and rec.get("one", 1) < 0.5:
        u.append(f"1hop ogrenilmiyor ({rec['one']:.3f}) -> egitim bozuk olabilir")
    if ilerleme > 0.5 and rec.get("comp", 1) < FLOOR:
        u.append(f"comp tabanda ({rec['comp']:.3f}) -> gorev cok zor olabilir")
    for k in ("ho_kopru", "kopru", "cevap"):
        if k in rec and not np.isfinite(rec[k]):
            u.append(f"prob '{k}' nan donuyor")
    if not np.isfinite(rec.get("loss", 0)):
        u.append("loss nan/inf")

    # OLUMCUL: bellek ust uste UC olcumde olu -> devam etmenin anlami yok
    if arm != "A" and len(curve) >= 3:
        son3 = curve[-3:]
        if all(np.isfinite(c.get("mem_scale", np.nan)) and c["mem_scale"] < 3.0
               and np.isfinite(c.get("memK_etkin_rank", np.nan))
               and c["memK_etkin_rank"] < 3 for c in son3):
            olumcul = True
            u.append("OLUMCUL: bellek uc olcumdur olu (scale<3, anahtar rank<3). "
                     "Bu kolu surdurmek bos.")
    return u, olumcul


def run_arm(arm, seed, data, log):
    facts, pairs, one, tr2, comp, ent_ev, seen_e, unseen_e = data
    torch.manual_seed(seed); np.random.seed(seed)

    X1, P1, T1, C1 = enc_one(one)
    X2, P2, T2, C2 = enc_two(tr2)
    Xtr = np.concatenate([X1, X2]); Ptr = np.concatenate([P1, P2])
    Ttr = np.concatenate([T1, T2])

    def sub(lst, salt):
        r = np.random.RandomState(DATA_SEED + 7 + salt)
        if len(lst) > N_EVAL_MAX:
            lst = [lst[i] for i in r.permutation(len(lst))[:N_EVAL_MAX]]
        return lst

    L1, LS, LC, LE = (sub(one, 0), sub(tr2, 1), sub(comp, 2), sub(ent_ev, 3))
    E1, ES, EC, EE = enc_one(L1), enc_two(LS), enc_two(LC), enc_two(LE)
    # tani hedefleri: kopru varligi ve "r1'i atla" kisayolu
    brS = np.array([ENT_OFF + b for _, _, _, b, _ in LS], np.int64)
    brC = np.array([ENT_OFF + b for _, _, _, b, _ in LC], np.int64)
    brE = np.array([ENT_OFF + b for _, _, _, b, _ in LE], np.int64)
    scS = np.array([ENT_OFF + int(facts[e, r2]) for e, _, r2, _, _ in LS], np.int64)
    scC = np.array([ENT_OFF + int(facts[e, r2]) for e, _, r2, _, _ in LC], np.int64)
    scE = np.array([ENT_OFF + int(facts[e, r2]) for e, _, r2, _, _ in LE], np.int64)
    _pti, _pvi = onek_bolme(LS)          # (e,r1) onegine gore prob bolmesi

    log(f"  --- kol {arm} ---")
    log(f"    veri kodlandi: 1hop {len(L1)} | 2hop-egitim {len(LS)} | "
        f"COMP {len(LC)} | ENT {len(LE)}")
    model = Net(arm, CFG).to(DEV)
    npm = nparam(model)
    # torch.compile SADECE egitim adimina. Tani fonksiyonlari ham `model`i
    # kullanir: onlar no_mem/want_w bayraklari ve degisken batch ile cagriliyor,
    # derlenmis surumde her varyant yeniden derleme tetikler ve yavaslatir.
    # Parametreler ortak oldugu icin ikisi AYNI modeldir.
    trn = model
    if COMPILE and DEV == "cuda":
        try:
            trn = torch.compile(model)
            log("    torch.compile acik (egitim yolu)")
        except Exception as ex:
            log(f"    torch.compile atlandi: {str(ex)[:60]}")
    log(f"  kol {arm}: {npm/1e6:.2f}M parametre")

    # WEIGHT DECAY sadece 2-B matrislere. Skaler ve 1-B parametreler (RMSNorm
    # kazanci, bellegin logit_scale'i) HARIC.
    # NEDEN: 12 Eylul kosusunda logit_scale'e de decay uygulaniyordu ve kol C'de
    # sicaklik 12.5 -> 2.3'e cokup kelepce tabanina yaklasti; anahtar/deger etkin
    # rank'i 61 -> 1.8'e dustu, bellek tek sabit vektore dondu (tepe agirlik
    # 0.109 -> 0.001, duzgun dagilim 0.00024). Kol B'de gradyan decay'i
    # dengeledigi icin ayni cokus olmadi -> karsilastirma YANLIydi.
    dec = [p_ for p_ in model.parameters() if p_.dim() >= 2]
    nodec = [p_ for p_ in model.parameters() if p_.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": dec, "weight_decay": CFG.get("WD", 0.01)},
         {"params": nodec, "weight_decay": 0.0}],
        lr=CFG["LR"], betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler(DEV, enabled=(DEV == "cuda"))
    S, B = CFG["STEPS"], CFG["BATCH"]
    warm = max(10, S // 20)
    rs = np.random.RandomState(seed + 991)
    curve, tablo, t0 = [], [], time.time()

    for step in range(1, S + 1):
        if step < warm:                      # isinma
            lr = CFG["LR"] * step / warm
        elif CFG.get("SCHED", 0):            # SABIT: grokking icin LR sonmemeli
            lr = CFG["LR"]
        else:                                # kosinus
            lr = CFG["LR"] * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, S - warm)))
        for g in opt.param_groups:
            g["lr"] = lr
        j = rs.randint(0, len(Xtr), B)
        xb = torch.from_numpy(Xtr[j]).to(DEV)
        pb = torch.from_numpy(Ptr[j]).to(DEV)
        tb = torch.from_numpy(Ttr[j]).to(DEV)
        with torch.autocast(DEV, dtype=torch.float16, enabled=(DEV == "cuda")):
            lg, _ = trn(xb)
            lg = lg[torch.arange(B, device=DEV), pb]
            loss = F.cross_entropy(lg.float(), tb)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()

        if PROG and step % PROG == 0 and step % CFG["EVERY"] != 0:
            el = time.time() - t0
            hz = step / max(el, 1e-9)
            log(f"      adim {step:6d}/{S}  (%{100*step/S:4.1f})  loss {loss.item():.3f}"
                f"  {hz:5.1f} adim/s  gecen {el/60:4.1f} dk  kalan ~{(S-step)/hz/60:4.1f} dk")
        if step % CFG["EVERY"] == 0 or step == S:
            a1 = evaluate(model, *E1[:3])[0].mean()
            rec = dict(step=step, loss=float(loss.item()), one=float(a1),
                       lr=float(lr), secs=round(time.time() - t0, 1))
            # --- TANI (kesifsel): birincil karari ETKILEMEZ
            for tag, EE_, gb, st, LL in (("seen", ES, brS, scS, LS),
                                         ("comp", EC, brC, scC, LC),
                                         ("ent", EE, brE, scE, LE)):
                z, dt = zengin(model, EE_, gb, st)
                rec[tag] = z["acc"]
                for k, v in z.items():
                    if k != "acc":
                        rec[f"{tag}_{k}"] = v
                if SAVE_TABLE:
                    n = len(dt["ok"])
                    row = dict(step=np.full(n, step, np.int32),
                               kume=np.full(n, tag, object),
                               e=np.array([a for a, *_ in LL], np.int32),
                               r1=np.array([b for _, b, *_ in LL], np.int32),
                               r2=np.array([c for _, _, c, _, _ in LL], np.int32),
                               kopru=np.array([d for *_, d, _ in LL], np.int32),
                               gold=np.array([f for *_, f in LL], np.int32))
                    row.update(dt); tablo.append(row)
            rec.update(bellek_istat(model, EC[0]))
            rec.update(prob_seti(model, ES, EC, brS - ENT_OFF, brC - ENT_OFF,
                                 ES[2] - ENT_OFF, EC[2] - ENT_OFF, salt=step, LS=LS))
            try:                       # KATMAN PROFILI (kollar arasi kiyaslanabilir)
                pr = kopru_profil(model, ES, _pti, _pvi, brS - ENT_OFF)
                rec["kopru_profil"] = pr
                rec["profil_min"] = float(np.min(pr))
                rec["profil_kat"] = int(np.argmin(pr))
            except Exception as ex:
                rec["kopru_profil"] = []; rec["profil_hata"] = str(ex)[:80]
            rec.update(agirlik_istat(model))
            curve.append(rec)
            if SAVE_TABLE and tablo:
                yaz_tablo(tablo, arm, seed)
            if SAVE_CIRCUIT:
                lg_, lb_ = logit_lens(model, EC, brC)
                rec["lens_gold"] = lg_; rec["lens_kopru"] = lb_
                rec["lens_kopru_min"] = float(np.min(lb_))
                rec["lens_kopru_kat"] = int(np.argmin(lb_))
                rec["lens_gold_min"] = float(np.min(lg_))
                rec["lens_gold_kat"] = int(np.argmin(lg_))
                np.savez_compressed(
                    os.path.join(OUT, f"dikkat_{arm}_s{seed}_{step:06d}.npz"),
                    comp=dikkat(model, EC[0], EC[1]),
                    seen=dikkat(model, ES[0], ES[1]))
            if SAVE_SNAP and (len(curve) % CKPT_EVERY == 0):
                torch.save({k: v.half().cpu() for k, v in model.state_dict().items()},
                           os.path.join(OUT, f"snap_{arm}_s{seed}_{step:06d}.pt"))
            if SAVE_KV and model.mem is not None:
                np.savez_compressed(
                    os.path.join(OUT, f"kv_{arm}_s{seed}_{step:06d}.npz"),
                    K=model.mem.K.detach().half().cpu().numpy(),
                    V=model.mem.V.detach().half().cpu().numpy(),
                    scale=float(model.mem.logit_scale.detach().exp().clamp(1, 100)))
            # EGRI EN SONA yazilir: onceden yaziliyordu ve devre olcumleri
            # (lens_kopru/lens_gold) rec'e ONDAN SONRA ekleniyordu, yani her
            # noktanin lens verisi bir sonraki yazimda diske iniyor, SON
            # noktanınki ise hic kaydedilmiyordu -- analiz.py'nin katman
            # tablosu tam o noktayi kullaniyor.
            json.dump(curve, open(os.path.join(
                OUT, f"egri_{arm}_s{seed}.json"), "w"), indent=1)
            uy, olumcul = saglik(rec, curve, arm, step, S)
            rec["uyarilar"] = uy
            for w in uy:
                log(f"      >>> UYARI: {w}")
            if olumcul and ABORT_ON_FATAL:
                log(f"      >>> kol {arm} durduruldu (olumcul saglik hatasi)")
                break
            log(f"    {step:6d}  loss {loss.item():.3f}  1hop {a1:.3f}  "
                f"seen {rec['seen']:.3f}  comp {rec['comp']:.3f}  "
                f"| ent {rec['ent']:.3f}  kopru-sira {rec['comp_bridge_rank']:.0f}  "
                f"kopru {rec['kopru']:.3f} (L0 {rec['kopru_L0']:.3f} null {rec['kopru_null']:.3f}) "
                f"cevap {rec['cevap']:.3f} | ONEK-TUT kopru {rec.get('ho_kopru', float('nan')):.3f} "
                f"(L0 {rec.get('ho_kopru_L0', float('nan')):.3f} "
                f"null {rec.get('ho_kopru_null', float('nan')):.3f})  "
                f"profil {rec.get('profil_min', float('nan')):.3f}@K{rec.get('profil_kat', -1)}  "
                f"belleksiz {rec['comp_acc_nomem']:.3f}  "
                f"memH {rec['mem_H']:.2f}"
                + (f"  lens-kopru {rec['lens_kopru_min']:.3f}@K{rec['lens_kopru_kat']}"
                   if SAVE_CIRCUIT else "")
                + f"  ({time.time()-t0:.0f}s)")

    r = {}
    for nm, E in (("one", E1), ("seen", ES), ("comp", EC), ("ent", EE)):
        okr, okf = evaluate(model, *E[:3])
        r[nm] = dict(acc=float(okr.mean()), acc_free=float(okf.mean()),
                     ok=okr, clus=E[3])
    r["shortcut"] = float(evaluate(model, EC[0], EC[1], scC)[0].mean())
    r["mement"] = mem_entropy(model, EC[0])
    r["nparam"] = int(npm); r["curve"] = curve
    r["secs"] = time.time() - t0
    if SAVE_CIRCUIT:                   # DEVRE: pahali, sadece sonda
        ka = kafa_ablasyonu(model, EC, E1)
        np.savez_compressed(os.path.join(OUT, f"kafa_{arm}_s{seed}.npz"), **ka)
        dc, d1 = ka["dusus_comp"], ka["dusus_1hop"]
        sec = dc - d1                                  # comp'u bozup 1hop'u bozmayan
        top = np.dstack(np.unravel_index(np.argsort(-sec, axis=None)[:5], sec.shape))[0]
        log("    besteleme devresi (comp dususu - 1hop dususu, en buyuk 5):")
        for l, h in top:
            log(f"      K{l}H{h}: comp {dc[l,h]:+.3f}  1hop {d1[l,h]:+.3f}  "
                f"fark {sec[l,h]:+.3f}")
    if SAVE_CKPT:                      # oynama ortami icin agirliklar
        ck = os.path.join(OUT, f"model_{arm}_s{seed}.pt")
        torch.save(dict(arm=arm, seed=seed, cfg=CFG, vocab=VOCAB, t_len=T_LEN,
                        rel_off=REL_OFF, ent_off=ENT_OFF, ctx=CTX,
                        state=model.state_dict()), ck)
        log(f"    kaydedildi -> {ck}")
    del model; torch.cuda.empty_cache() if DEV == "cuda" else None
    return r


def boot_diff(a, b, clus, salt, nb=N_BOOT):
    """varlik-kumeli ESLI bootstrap (a ve b ayni orneklerde)."""
    a, b, clus = np.asarray(a, float), np.asarray(b, float), np.asarray(clus)
    uq = np.unique(clus); by = {c: np.where(clus == c)[0] for c in uq}
    d = a - b
    r = np.random.RandomState(DATA_SEED + salt); m = []
    for _ in range(nb):
        pk = uq[r.randint(0, len(uq), len(uq))]
        m.append(d[np.concatenate([by[c] for c in pk])].mean())
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ============================ ANA AKIS ====================================
def main():
    lines = []
    def log(s):
        print(s, flush=True); lines.append(s)

    log(f"preset={PRESET}  cihaz={DEV}  seeds={TRAIN_SEEDS}")
    log(f"vocab={VOCAB}  T={T_LEN}  bellek_param={mem_params(CFG)/1e6:.2f}M")
    log(f"bellek okuma noktasi: {list(MEM_AT)} ({len(MEM_AT)} okuma/ileri gecis)"
        + ("" if len(MEM_AT) == 1 else
           "  -- DIKKAT: >1 okumada kollar arasi FLOP esitligi BOZUK"))
    _np = nparam(Net("C", CFG))
    _tok = CFG["STEPS"] * CFG["BATCH"] * T_LEN
    _fl = 6.0 * _np * _tok
    log(f"kol basina ~{_np/1e6:.1f}M parametre, {_tok/1e6:.1f}M token, "
        f"{_fl/1e15:.2f} PFLOP")
    log(f"T4 kabasi (~5 TFLOP/s etkin): kol basina ~{_fl/5e12/60:.0f} dk, "
        f"{len(ARMS)} kol x {len(TRAIN_SEEDS)} seed = "
        f"~{_fl*len(ARMS)*len(TRAIN_SEEDS)/5e12/3600:.1f} saat")
    log("veri hazirlaniyor...")
    data = build_data()
    facts, pairs, one, tr2, comp, ent_ev, seen_e, unseen_e = data
    log(f"olgu {len(one)} | 2hop-egitim {len(tr2)} | COMP {len(comp)} "
        f"({len(set(e for e,*_ in comp))} varlik) | ENT {len(ent_ev)} "
        f"({len(unseen_e)} varlik)")

    np.savez(os.path.join(OUT, "veri.npz"), facts=facts,
             pairs=np.array(pairs), unseen=np.array(unseen_e),
             comp=np.array([(e, a, b, br, an) for e, a, b, br, an in comp]))
    R = {}
    for sd in TRAIN_SEEDS:
        log(f"\n--- seed {sd} ---")
        for arm in ARMS:
            R[(arm, sd)] = run_arm(arm, sd, data, log)

    log("\n" + "=" * 78)
    log(f"SONUCLAR  ({len(TRAIN_SEEDS)} seed)")
    log("=" * 78)
    log(f"{'kol':4s} {'param':>8s} {'1hop':>7s} {'seen':>7s} {'COMP':>7s} "
        f"{'ENT':>7s} {'kisayol':>8s} {'memH':>7s}")
    agg = {}
    for arm in ARMS:
        rs = [R[(arm, s)] for s in TRAIN_SEEDS]
        g = lambda k: float(np.mean([r[k]["acc"] for r in rs]))
        agg[arm] = dict(one=g("one"), seen=g("seen"), comp=g("comp"), ent=g("ent"),
                        shortcut=float(np.mean([r["shortcut"] for r in rs])),
                        mement=float(np.mean([r["mement"] for r in rs])),
                        nparam=rs[0]["nparam"])
        a = agg[arm]
        log(f"{arm:4s} {a['nparam']/1e6:7.2f}M {a['one']:7.3f} {a['seen']:7.3f} "
            f"{a['comp']:7.3f} {a['ent']:7.3f} {a['shortcut']:8.3f} {a['mement']:7.2f}")

    # ---- kapilar
    gates, why = True, []
    if min(agg[a]["one"] for a in ARMS) < GATE_1HOP:
        gates = False; why.append(f"KAPI-1 dustu: 1hop < {GATE_1HOP}")
    if max(agg[a]["seen"] for a in ARMS) < GATE_SEEN:
        gates = False; why.append(f"KAPI-2 dustu: seen < {GATE_SEEN}")
    for _a in ARMS:                      # KAPI-3 BELLEKLI HER KOLA
        _h = agg[_a].get("mement", float("nan"))
        if np.isfinite(_h) and _h < GATE_MEMENT:
            gates = False
            why.append(f"KAPI-3 dustu: kol {_a} bellek cokmesi "
                       f"(H={_h:.2f} < {GATE_MEMENT:.2f}, yani ~{math.exp(_h):.0f} "
                       f"etkin slot)")

    if not set("ABC") <= set(ARMS):          # tam olmayan kosu -> sadece tani
        log("-" * 78)
        log(f"UYARI: sadece {''.join(ARMS)} kolu kosuldu; karsilastirma YAPILMADI.")
        for w in why:
            log("  " + w)
        json.dump(dict(preset=PRESET, cfg=CFG, seeds=TRAIN_SEEDS, arms="".join(ARMS),
                       verdict="KISMI KOSU", agg=agg),
                  open(os.path.join(OUT, "verdict.json"), "w"), indent=1)
        open(os.path.join(OUT, "log.txt"), "w", encoding="utf-8").write(NL.join(lines))
        return

    # ---- birincil (seed'ler havuzlanir, kume = varlik x seed)
    okC = np.concatenate([R[("C", s)]["comp"]["ok"] for s in TRAIN_SEEDS])
    okB = np.concatenate([R[("B", s)]["comp"]["ok"] for s in TRAIN_SEEDS])
    okA = np.concatenate([R[("A", s)]["comp"]["ok"] for s in TRAIN_SEEDS])
    clus = np.concatenate([[f"{s}_{c}" for c in R[("C", s)]["comp"]["clus"]]
                           for s in TRAIN_SEEDS])
    dCB, loCB, hiCB = boot_diff(okC, okB, clus, 1)
    dCA, loCA, hiCA = boot_diff(okC, okA, clus, 2)
    dBA, loBA, hiBA = boot_diff(okB, okA, clus, 3)

    log("-" * 78)
    log("ESLI FARKLAR (COMP, varlik-kumeli %95 GA)")
    log(f"  C - B (BIRINCIL): {dCB:+.3f}  [{loCB:+.3f}, {hiCB:+.3f}]")
    log(f"  C - A           : {dCA:+.3f}  [{loCA:+.3f}, {hiCA:+.3f}]")
    log(f"  B - A           : {dBA:+.3f}  [{loBA:+.3f}, {hiBA:+.3f}]")
    log("=" * 78)

    cmax = max(agg[a]["comp"] for a in ARMS)
    cmin = min(agg[a]["comp"] for a in ARMS)
    if not gates:
        V, msg = "YETERSIZ EGITIM / GECERSIZ", " ; ".join(why)
    elif cmax < FLOOR:
        V, msg = "TABAN ETKISI — SONUC YOK", (
            f"Hicbir kol COMP'ta {FLOOR:.2f} esigini gecemedi (en iyi {cmax:.3f}). "
            "Iki kol da basarisizken fark olcmek anlamsiz; gorev bu olcekte cok zor. "
            "Cozum: P_TRAIN'i artir, N_ENT'i azalt ya da modeli buyut.")
    elif cmin > CEIL:
        V, msg = "TAVAN ETKISI — SONUC YOK", (
            f"Butun kollar COMP'ta {CEIL:.2f} ustunde (en dusuk {cmin:.3f}). "
            "Gorev cok kolay; ayirt edemez. Cozum: P_TRAIN'i azalt ya da N_ENT'i artir.")
    elif (hiCB - loCB) > CI_TOO_WIDE:
        V, msg = "YETERSIZ VERI", f"GA genisligi {hiCB-loCB:.3f} > {CI_TOO_WIDE}"
    elif loCB > DELTA_MIN:
        V, msg = "LATENT ADRES KAZANDI", ("Gizli durumdan adresleme token'dan "
                                          "adreslemeyi gecti.")
    elif hiCB < -DELTA_MIN:
        V, msg = "TOKEN ADRES KAZANDI", "Beklenenin tersi; tez dustu."
    else:
        V, msg = "BERABERE", ("Adresin nereden geldigi fark etmiyor. "
                              "Dort projenin dersi bir kez daha.")
    if gates and loCA <= 0 and loBA <= 0:
        msg += "  NOT: bellek hicbir kolda A'yi gecmiyor -> modul bos."
    if len(TRAIN_SEEDS) < 3:
        V = "ON BULGU — " + V
        msg += f"  ({len(TRAIN_SEEDS)} seed; hukum icin >=3 gerekir.)"

    log("SONUC: " + V); log("  " + msg)

    for s in TRAIN_SEEDS:
        for a in "ABC":
            for k in ("one", "seen", "comp", "ent"):
                R[(a, s)][k].pop("ok"); R[(a, s)][k].pop("clus")
    json.dump(dict(preset=PRESET, cfg=CFG, seeds=TRAIN_SEEDS, verdict=V, msg=msg,
                   agg=agg, dCB=[dCB, loCB, hiCB], dCA=[dCA, loCA, hiCA],
                   dBA=[dBA, loBA, hiBA],
                   runs={f"{a}_{s}": R[(a, s)] for a in "ABC" for s in TRAIN_SEEDS}),
              open(os.path.join(OUT, "verdict.json"), "w"), indent=1)
    open(os.path.join(OUT, "log.txt"), "w", encoding="utf-8").write("\n".join(lines))
    log(f"\nkaydedildi -> {OUT}/verdict.json  ve  log.txt")


if __name__ == "__main__":
    main()
