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
    KAPI-3  C kolunda bellek kullanim entropisi > log(32) olmali (slot cokmesi
            olursa sonuc "C kotu" degil, "bellek egitilemedi" demektir).
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

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, -1)
        sh = lambda t: t.view(B, T, self.nh, self.hd).transpose(1, 2)
        o = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), is_causal=True)
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
        self.mem_at = L // 2                     # bellek bu bloktan SONRA girer
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

    def forward(self, x, want_w=False):
        h = self.emb(x) + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        h0, w = h, None
        for i, blk in enumerate(self.blocks):
            h = blk(h)
            if self.mem is not None and i == self.mem_at - 1:
                src = h0 if self.arm == "B" else h      # <-- TEK FARK
                m, w = self.mem(src)
                h = h + m
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

    model = Net(arm, CFG).to(DEV)
    npm = nparam(model)
    log(f"  kol {arm}: {npm/1e6:.2f}M parametre")

    opt = torch.optim.AdamW(model.parameters(), lr=CFG["LR"],
                            weight_decay=CFG.get("WD", 0.01),
                            betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler(DEV, enabled=(DEV == "cuda"))
    S, B = CFG["STEPS"], CFG["BATCH"]
    warm = max(10, S // 20)
    rs = np.random.RandomState(seed + 991)
    curve, t0 = [], time.time()

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
            lg, _ = model(xb)
            lg = lg[torch.arange(B, device=DEV), pb]
            loss = F.cross_entropy(lg.float(), tb)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()

        if step % CFG["EVERY"] == 0 or step == S:
            a1 = evaluate(model, *E1[:3])[0].mean()
            asn = evaluate(model, *ES[:3])[0].mean()
            ac = evaluate(model, *EC[:3])[0].mean()
            curve.append(dict(step=step, loss=float(loss.item()),
                              one=float(a1), seen=float(asn), comp=float(ac)))
            log(f"    {step:6d}  loss {loss.item():.3f}  1hop {a1:.3f}  "
                f"seen {asn:.3f}  comp {ac:.3f}  ({time.time()-t0:.0f}s)")

    r = {}
    for nm, E in (("one", E1), ("seen", ES), ("comp", EC), ("ent", EE)):
        okr, okf = evaluate(model, *E[:3])
        r[nm] = dict(acc=float(okr.mean()), acc_free=float(okf.mean()),
                     ok=okr, clus=E[3])
    # kisayol tanisi: r1'i yok sayip f(e,r2) mi diyor? (AYNI alt-orneklem LC)
    sc = np.array([ENT_OFF + int(facts[e, r2]) for e, _, r2, _, _ in LC], np.int64)
    r["shortcut"] = float(evaluate(model, EC[0], EC[1], sc)[0].mean())
    r["mement"] = mem_entropy(model, EC[0])
    r["nparam"] = int(npm); r["curve"] = curve
    r["secs"] = time.time() - t0
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
    _np = nparam(Net("C", CFG))
    _tok = CFG["STEPS"] * CFG["BATCH"] * T_LEN
    _fl = 6.0 * _np * _tok
    log(f"kol basina ~{_np/1e6:.1f}M parametre, {_tok/1e6:.1f}M token, "
        f"{_fl/1e15:.2f} PFLOP")
    log(f"T4 kabasi (~5 TFLOP/s etkin): kol basina ~{_fl/5e12/60:.0f} dk, "
        f"{len(ARMS)} kol x {len(TRAIN_SEEDS)} seed = "
        f"~{_fl*len(ARMS)*len(TRAIN_SEEDS)/5e12/3600:.1f} saat")
    data = build_data()
    facts, pairs, one, tr2, comp, ent_ev, seen_e, unseen_e = data
    log(f"olgu {len(one)} | 2hop-egitim {len(tr2)} | COMP {len(comp)} "
        f"({len(set(e for e,*_ in comp))} varlik) | ENT {len(ent_ev)} "
        f"({len(unseen_e)} varlik)")

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
    if "C" in ARMS and agg["C"]["mement"] < GATE_MEMENT:
        gates = False; why.append(f"KAPI-3 dustu: bellek cokmesi (H={agg['C']['mement']:.2f})")

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
