"""train -- V2 ortak egitim: iki model ayni veri, ayni hedef, ayni tarif, ayni sinav; farkli olan yalniz model
(belge/model_z_temel/20 s4, 21, 22, 24 s9; ad onayli, kullanici 6 Ekim).

Adim (belge 24 s9):  attn = recipe.block_mask(batch, mask_fn) (CPU'da dense_mask: FlexAttention CPU'da geri yayilim
yapmiyor); h = model._batch_hidden(batch, attn); kayip = recipe.output_loss(h, E, hedef).  bf16 autocast ve bloklarda
compile(dynamic=False) CUDA'da; clip 1,0; AdamW (0,9 / 0,95, wd 0,1; CUDA'da fused), recipe.param_groups, recipe.wsd_lr.
--optimizer normuon VARSAYILAN (kullanici, 8 Ekim: "normuon bence standart yapalım. büyüdükçe etkisini gösterdi"; d1024 / L12
ayni adimda -0,012; --resume'da acik verilmezse kosunun kimliginden, alan yoksa adamw).  --optimizer muon (7 Ekim
varsayilani): bloklarin 2-B matrisleri
recipe.BatchedMuon'a (torch.optim.Muon matematigi; adjust_lr_fn match_rms_adamw: guncelleme RMS'i AdamW'ninki, ayni --lr
ve wd; liu2025_muonscalable), geri kalan ayni AdamW'ye; wsd_lr ikisine.  Muon yoksa kosu baslamadan DURUR.  --lr
zorunlu; Muon icin olculen 2e-3 (belge 39 kisa tarama 5e-4 / 1e-3 / 2e-3 + 1 epok, OLCULENLER_z).  --optimizer adamw:
eski tarif (olculen lr 5e-4).  --optimizer normuon (kullanici, 8 Ekim: "Ben nurmuon yapalım şimdiden dedim"):
recipe.NorMuon (li2025_normuon Algorithm 1), ayni --lr (guncelleme RMS'i 0,2 lr).
--fp8 none|tensorwise|rowwise (kullanici, 8 Ekim: "fp8 de dene bakalım. fp8 dikkatli dene"; varsayilan none): torchao
Float8Linear yalniz MLP'de (gate_up, down), compile'dan once; agirliklar fp32, state_dict adlari ayni (kosu basinda
denetlenir) -> ayni kosu --resume ile FP8 acik / kapali surer; KIMLIGE GIRMEZ, kip her segmentte gunlukte ve results.json
segments'ta.  CUDA ve torchao ister, yoksa DURUR (bf16'ya sessizce dusmez).
Veri: BATCH_ROWS satir x row_len (plan dosyasindan); epok 1 <data>/train_pack_plan_e1.npz, sonrakiler pack_plan(seed,
epok).  --local: ham akisin yerel kopyasi (yalniz onbellek; sha256 = <split>_boundaries.json'daki).
--summaries_last 1 (belge 66; kullanici, 8 Ekim: "Fikrine onay verdim"): batch'ler sentence.summaries_last ile [token'lar |
ozetler | dolgu] sirasinda (egitim ve sinav; sinav sonucu hedef sirasina geri), maske sorgu basina iki aralik; uretim
degismez.  Torba ile DUR.
--z_bow_weight W (kullanici, 8 Ekim: "Onayladım"; belge 68 fikir 1): amac = token CE + W x z_bow (Z_k'nin z_bow_layer
ciktisindan z_bow_norm + bagli E ile sonraki cumlenin token torbasi, sentence.z_bow_loss); varsayilan 0, torba ile DUR,
sinav degismez; gunlukte z_bow.
--stop_step N (kullanici, 8 Ekim): takvim degismeden adim N'de durur; checkpoint.pt + agent.pt + results.json (finished
False, stopped_at, readings_skipped "stop_step"), son sinav ve okuma yok; --resume 1 kaldigi yerden.
Surdurme: <out>/checkpoint.pt son kayittan --checkpoint_minutes sonraki ilk gunluk sinirinda, epok sonunda ve bitiste;
<out>/decay_start/ inisin ilk adiminda.  --resume 1: ayni toplam -> kaldigi yerden; buyuk toplam (--epochs / --steps) -> uzatma, inis basindan; eski
ciktilar <out>/total_<eski toplam>/'a.  Bitmis kosu durur.
Olcu: epok sonunda ve bitiste metrics.exam_scores (exam_pack_plan.npz; egitimle ayni maske yolu); hiz pencere pencere
(recipe.SpeedWindow; ilk pencere derleme icerir, ozete girmez).  Sonda agent.pt ve results.json, SONRA
metrics.story_generation (reading_prompts.json; okuma dusse de model kalir); layer_plan mid'de okuma yok (results.json
readings_skipped).  Cikti: config.json, checkpoint.pt, decay_start/, results.json, agent.pt, samples.txt, samples.json.
Ek okuma kayitli kosudan: diag/generate_readings.py.

Model Z yalniz ogrenilen z (belge 35 (b)); formullu z ve --learned_z kaldirildi (kullanici, 7 Ekim: "bence temizlik
başlasın"; belge 44; eski kod git etiketi v2-before-formula-cleanup-20261007).  Kimlikte learned_z (Model Z 1) eski
kosulari ayirir: formullu ve temizlik oncesi (6 Ekim; etiket v2-before-cleanup-20261006) Model Z kosulari yuklenmez /
surdurulmez (_archived); eski transformer yuklenir.  Temizlik oncesi kosular uzatilmaz (kullanici, 6 Ekim: "eski koşuları
uzatma niyetim yok").
--global_layers N (belge 40 s6.2 Deney G; yalniz Model Z): son N blok tam causal, gercek hikaye konumuyla; maske ikilisi
(yerel, global) _attn'dan, egitim / sinav / teshis ayni yol.  --global_layers 0: G'siz Model Z (kiyas).  Eski
checkpoint'te alan yoksa 0.
Varsayilanlar (kullanici, 8 Ekim: "Varsayılan yap ama kısa bir koşu ile son halin çalıştığından emin olalım"): model_z'de
global_layers MODEL_Z_GLOBAL_LAYERS (3) ve torbasiz summaries_last MODEL_Z_SUMMARIES_LAST (1); transformer ve --bag_k'da
summaries_last 0 (acik 1 DURUR).  --resume 1'de acikca verilmeyen global_layers / summaries_last kosunun kimliginden (alan
yoksa 0): eski G1 / summaries_last 0 kosulari degismeden surer.

--bag_k K (iki modelde; belge 53-55, adlar onayli 7 Ekim; kullanici, 7 Ekim: "burda öğrenme kalite ve hıza etkisi ne"):
ogrenen torba B_k = C u P_k u L_k, |B_k| <= K (recipe.Bag): C en sik --bag_core + END + EOS (<data>/train_token_counts.npy),
P_k hikayenin onceki token'lari, L_k kelime secicisi (puan q(h) . sg(e_v) + b_v, bf16).  Token olasiligi iki asamali (recipe.two_stage_logprobs; konum basina kendi B_k'si), egitim
recipe.bag_train_loss (torba basina aday bmm; tam sozluk yalniz kacan ve --bag_full_frac konumlarinda) + --bag_weight x
secici kaybi (Z'ye akar).  Gunlukte loss = iki asamali NLL (sinavla ayni tanim); bag: kaynak kaynak (C / P / L / kacan),
p(DIGER), tam CE, secici kaybi; cikis ve secici ileri ms (CUDA olaylari).

    python train.py --model transformer|model_z --lr LR --out <kosu> [--data <v2/simplestories_gpt2>]
                    [--stream <simplestories>] [--local /content/v2_cache] [--epochs 1] [--steps N] [--d 512]
                    [--layers 8] [--heads 8] [--seed 0] [--device cuda] [--resume 1]
                    [--optimizer normuon|muon|adamw (varsayilan normuon)] [--global_layers N (model_z; varsayilan 3)]
                    [--summaries_last 0|1 (model_z torbasiz; varsayilan 1)]
                    [--bag_k K [--bag_core 50] [--bag_weight 0.1] [--bag_full_frac 0.05]]
"""
import torch  # noqa: I001  (Windows: torch once)

import argparse
import dataclasses
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as D  # noqa: E402
import metrics as M  # noqa: E402
import recipe as R  # noqa: E402

BETAS, WEIGHT_DECAY, CLIP = (0.9, 0.95), 0.1, 1.0         # belge 20 s4
MUON = dict(momentum=0.95, nesterov=True, ns_steps=5, adjust_lr_fn="match_rms_adamw")   # torch varsayilanlari + Moonshot olcegi
DECAY = 0.2                                                # recipe.wsd_lr varsayilani; inis basi checkpoint'i
BATCH_ROWS = D.BATCH_ROWS
LOG_EVERY = 100             # adim; gunluk satiri = bir hiz penceresi
MODEL_Z_GLOBAL_LAYERS = 3   # model_z varsayilani (kullanici, 8 Ekim; G3 + aralik maskesi + summaries_last + Muon)
MODEL_Z_SUMMARIES_LAST = 1  # model_z torbasiz varsayilani (belge 66)
INHERIT = dict(global_layers=0, summaries_last=0, optimizer="adamw")   # --resume'da acik verilmezse kimlikten (alan
DEFAULT_OPTIMIZER = "normuon"                       # yoksa bu deger); optimizer varsayilani (kullanici, 8 Ekim)
FP8_MODULES = ("gate_up", "down")                   # --fp8 donusturulen Linear'lar (MLP)
COMPILE_MODE = "max-autotune-no-cudagraphs"   # bloklarin derleme modu (5w: torba K 1024 -2,9 ms/adim; kullanici, 7 Ekim)
READING_PROMPTS = os.path.join(HERE, "reading_prompts.json")
READING_LIMITS = dict(max_sentences=80, max_tokens=128)     # belge 21 (story_generation varsayilanlari)
SAMPLE_SEED = 0             # sample cozme tohumu (V1 generate_baseline ile ayni)
IDENTITY = ("model", "d", "layers", "heads", "lr", "seed", "longest", "row_len", "batch_rows", "train_stream_sha256",
            "learned_z", "optimizer", "global_layers", "layer_plan", "bag_k", "bag_core", "bag_weight", "bag_full_frac", "bag_core_sha256",
            "bag_sel_frac", "summaries_last", "z_bow_weight")
NO_BAG = dict(bag_k=0, bag_core=0, bag_weight=0.0, bag_full_frac=0.0, bag_core_sha256=None, bag_sel_frac=1.0)   # torbasiz / eski kosu
LEGACY = ("meaning_sha256", "shared_vocab", "own_vocab", "open_z")   # temizlik oncesi kimlik alanlari (belge 33)
TAG = "v2-before-cleanup-20261006"
OUTPUTS = ("results.json", "agent.pt", "samples.txt", "samples.json")


def _no_power_throttling():
    """Windows: surecin EcoQoS kisitini kapat (kural 2a)."""
    import ctypes
    from ctypes import wintypes

    class State(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]
    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    s = State(1, 0x1, 0)
    return bool(k.SetProcessInformation(k.GetCurrentProcess(), 4, ctypes.byref(s), ctypes.sizeof(s)))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def _local_copy(stream_root, local, data_dir, splits=("train", "valid")):
    """<stream_root>/gpt2/<split>.npy -> <local>/gpt2/<split>.npy; varsa ve sha256 tutuyorsa yeniden kopyalanmaz.  sha256,
    sinirlarin hesaplandigi akisinki (<split>_boundaries.json) olmali: kopya = sinirlarin akisi, bayt bayt."""
    for split in splits:
        want = json.load(open(os.path.join(data_dir, split + "_boundaries.json"), encoding="utf-8"))["stream_sha256"]
        src, dst = os.path.join(stream_root, "gpt2", split + ".npy"), os.path.join(local, "gpt2", split + ".npy")
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src) and _sha256(dst) == want:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst + ".part")
        got = _sha256(dst + ".part")
        assert got == want, "%s kopyasinin sha256'si sinir dosyasindakiyle ayni degil: %s != %s" % (split, got, want)
        os.replace(dst + ".part", dst)
    return local


def _decay_start(total):
    """recipe.wsd_lr'deki inisin ilk adimi."""
    return total - round(DECAY * total)


def _schedule(train, data_dir, seed, epochs, steps):
    """-> (epok planlari [(row_offsets, row_stories)], epok basina adim, row_len, toplam adim).  steps verilirse toplam o
    (gerektigi kadar epok), yoksa epochs epok."""
    f = np.load(os.path.join(data_dir, "train_pack_plan_e1.npz"))
    row_len = int(f["row_len"])
    lengths = train.lengths()
    plans, per = [], []
    while True:
        e = len(plans) + 1
        if e == 1 and int(f["seed"]) == seed and int(f["epoch"]) == 1:
            plans.append((f["row_offsets"], f["row_stories"]))
        else:
            plans.append(D.pack_plan(lengths, row_len, seed, e))
        per.append(-(-(len(plans[-1][0]) - 1) // BATCH_ROWS))
        if (steps is None and e >= epochs) or (steps is not None and sum(per) >= steps):
            return plans, per, row_len, steps if steps is not None else sum(per)


def _archived(idt):
    """Okuma ve surdurme: kimlik bu kodla kurulamiyorsa ileti, yoksa None.  Model Z yalniz ogrenilen z (learned_z 1);
    formullu (learned_z 0 ya da alan yok) ve temizlik oncesi (LEGACY alanli) Model Z durur: agirlik sekilleri cogunda bu
    modelle ayni degil, ayni olanlar (iota) sessizce yanlis yuklenirdi.  Transformer her zaman yuklenir."""
    if idt.get("model") != "model_z" or idt.get("learned_z") == 1:
        return None
    tag = TAG if any(k in idt for k in LEGACY) else "v2-before-formula-cleanup-20261007"
    return "kosu formullu z / temizlik oncesi bir Model Z yoluyla egitildi; bu kodla yuklenmez -- git etiketi %s " \
           "(git worktree add <klasor> %s)" % (tag, tag)


def _global_error(args):
    """--global_layers kurulamiyorsa ileti, yoksa None (args'ta yoksa 0)."""
    gl = int(getattr(args, "global_layers", 0))
    if gl and args.model != "model_z":
        return "--global_layers yalniz model_z (belge 40 s6.2)"
    if not 0 <= gl <= args.layers:
        return "--global_layers %d: 0..%d olmali" % (gl, args.layers)
    return None


def _bag_error(args):
    """Torba ve summaries_last bayraklari kurulamiyorsa ileti, yoksa None (args'ta yoksa kapali)."""
    if getattr(args, "summaries_last", 0) and (args.model != "model_z" or getattr(args, "bag_k", 0)):
        return "--summaries_last yalniz model_z, torbasiz (belge 66)"
    if getattr(args, "z_bow_weight", 0.0) and (args.model != "model_z" or getattr(args, "bag_k", 0)):
        return "--z_bow_weight yalniz model_z, torbasiz"
    if not getattr(args, "bag_k", 0):
        return None
    if not 0.0 <= args.bag_full_frac <= 1.0:
        return "--bag_full_frac 0..1 olmali"
    if not 0.0 < getattr(args, "bag_sel_frac", 1.0) <= 1.0:
        return "--bag_sel_frac (0, 1] olmali"
    return None


def _build(args, dev):
    """-> (model, mask_fn, layout).  Model dosyalari yalniz burada import edilir.  Model Z: ogrenilen z (belge 35 (b));
    global_layers (args'ta yoksa 0): mask_fn (yerel, global) ikilisi.  Maske kurali yalniz burada secilir (egitim, sinav,
    teshis ayni yol).  bag_k: ilk agirliklardan SONRA recipe.attach_bag (args.bag_n_core; degerleri main Bag.fill ile ya
    da state_dict yazar)."""
    root = os.path.dirname(HERE)
    if _global_error(args):
        sys.exit("DUR: " + _global_error(args))
    torch.manual_seed(args.seed)
    if args.model == "transformer":
        sys.path.insert(0, os.path.join(root, "transformer"))
        from baseline import BaselineTransformer
        model, mask_fn, layout = BaselineTransformer(args.d, args.layers, args.heads).to(dev), R.document_mask, "transformer"
    else:
        sys.path.insert(0, os.path.join(root, "model_z"))
        from sentence import SentenceTransformer
        model = SentenceTransformer(args.d, args.layers, args.heads, global_layers=int(getattr(args, "global_layers", 0)),
                                    layer_plan=getattr(args, "layer_plan", None))
        model, mask_fn, layout = model.to(dev), model.mask_fn, "model_z"
    if getattr(args, "bag_k", 0):
        R.attach_bag(model, args.bag_k, args.bag_n_core)
    if getattr(args, "z_bow_weight", 0.0):                                # ilk agirliklardan sonra (RNG tuketmez)
        if model.z_bow_layer() is None:
            sys.exit("DUR: --z_bow_weight: glob olmayan blok yok")
        model.z_bow_norm = torch.nn.RMSNorm(args.d).to(dev)
    return model, mask_fn, layout


def _fp8_missing(cuda):
    """--fp8 kurulamiyorsa ileti (CUDA yok / torchao yok), yoksa None."""
    if not cuda:
        return "FP8 (torchao Float8Linear) CUDA ister"
    try:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training  # noqa: F401
    except ImportError as e:
        return "torchao.float8 yok (%s)" % e
    return None


def _fp8(model, recipe):
    """MLP Linear'lari (FP8_MODULES) torchao Float8Linear'a, compile'dan ONCE; state_dict adlari ve parametre sekilleri ayni
    kalmali (checkpoint / agent.pt FP8 <-> bf16 birebir; Muon ayrimi ad desenine dayanir) -> denetlenir, degilse DUR."""
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    before = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    convert_to_float8_training(model, config=Float8LinearConfig.from_recipe_name(recipe),
                               module_filter_fn=lambda m, fqn: fqn.split(".")[-1] in FP8_MODULES)
    after = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    if before != after:
        sys.exit("DUR: --fp8: state_dict adlari / sekilleri degisti (%s)" % sorted(set(before) ^ set(after))[:5])
    return sum(type(m).__name__ == "Float8Linear" for m in model.modules())


def _muon_missing():
    """torch.optim.Muon ve MUON ayarlari bu torch'ta kurulamiyorsa ileti, yoksa None."""
    if not hasattr(torch.optim, "Muon"):
        return "torch %s'te torch.optim.Muon yok" % torch.__version__
    try:
        torch.optim.Muon([torch.nn.Parameter(torch.zeros(2, 2))], **MUON)
    except (TypeError, ValueError) as e:
        return "torch %s: torch.optim.Muon(%s) kurulamadi: %s" % (torch.__version__, MUON, e)
    return None


def _optimizer(model, kind, lr, cuda):
    """-> (optimizer, bilgi).  adamw: tek AdamW (recipe.param_groups).  muon: recipe.muon_params Muon'a (wd ayni),
    geri kalan ayni ayarli AdamW'ye; recipe.MuonAdamW.  bilgi["split"]: grup basina tensor / parametre sayisi, ad deseni."""
    fused = {"fused": True} if cuda else {}
    muon = [p for _, p in R.muon_params(model)] if kind in ("muon", "normuon") else []
    decay, no_decay = R.param_groups(model, WEIGHT_DECAY, skip=muon)
    adamw = torch.optim.AdamW([decay, no_decay] if not muon else [g for g in (decay, no_decay) if g["params"]],
                              lr=lr, betas=BETAS, **fused)                   # muon: transformer'da decay grubu bos
    cls = R.NorMuon if kind == "normuon" else R.BatchedMuon
    opt = R.MuonAdamW(cls(muon, lr=lr, weight_decay=WEIGHT_DECAY, **MUON), adamw) if muon else adamw
    names = {id(p): n for n, p in model.named_parameters()}

    def summary(params):
        pats = {}
        for p in params:
            k = re.sub(r"\.\d+\.", ".*.", names[id(p)])
            pats[k] = pats.get(k, 0) + 1
        return dict(tensors=len(params), params=sum(p.numel() for p in params), names=pats)
    split = dict(muon=summary(muon), adamw_decay=summary(decay["params"]), adamw_no_decay=summary(no_decay["params"]))
    return opt, dict(name=kind, split=split, adamw=dict(betas=BETAS, weight_decay=WEIGHT_DECAY, fused=cuda),
                     muon=dict(MUON, weight_decay=WEIGHT_DECAY, **(dict(beta2=R.NORMUON_BETA2, eps=R.NORMUON_EPS,
                                                                     scale="0.2 lr sqrt(mn) / ||O^||_F")
                                                                if kind == "normuon" else {})) if muon else None)


def _attn(batch, mask_fn, cuda):
    """Egitim, sinav (_Exam) ve teshis araclarinin tek maske yolu; mask_fn ikiliyse (global_layers) iki maske."""
    if isinstance(mask_fn, tuple):
        return tuple(_attn(batch, f, cuda) for f in mask_fn)
    return R.block_mask(batch, mask_fn) if cuda else R.dense_mask(batch, mask_fn)


def _step(model, batch, mask_fn, opt, cuda, full=None, weight=0.0, timer=None, plan=None, bow=None):
    """Tek egitim adimi -> (kayip, gradyan normu, torba ekleri ya da None) cihazda.  full: torbali modelde tam softmax payi
    -> recipe.bag_train_loss (kayip = iki asamali NLL; amac + secici kaybi; plan: recipe.bag_plan, tek senkron).  timer: dort CUDA olayi, cikis
    kalemi ve icindeki secici (yalniz ileri).  bow: (z_bow_weight, sentence.z_bow_targets cihazda) -> amac += agirlik x
    z_bow kaybi; ek {"z_bow": kayip, "z_bow_n": hedefli Z}."""
    with torch.autocast(batch.tokens.device.type, dtype=torch.bfloat16, enabled=cuda):
        if bow is None:
            h = model._batch_hidden(batch, _attn(batch, mask_fn, cuda))
        else:
            from sentence import z_bow_loss
            h, xt = model._batch_hidden(batch, _attn(batch, mask_fn, cuda), tap=True)
            zb, zn = z_bow_loss(model, xt, bow[1])
        if timer is not None:
            timer[0].record()
        if full is None:
            loss = goal = R.output_loss(h.flatten(0, 1), model.E.weight, batch.target.flatten())
            extra = None
        else:
            goal, loss, extra = R.bag_train_loss(model, batch, h, full, weight, timer and timer[2:], plan)
        if timer is not None:
            timer[1].record()
    if bow is not None:
        goal = goal + bow[0] * zb
        extra = dict(extra or {}, z_bow=zb.detach() * zn, z_bow_n=zn)
    opt.zero_grad(set_to_none=True)
    goal.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
    opt.step()
    return loss.detach(), gn.detach(), extra


def _to_device(batch, dev):
    """CPU PackedBatch -> cihaz; CUDA'da sabitlenmis bellekten non_blocking (sonraki batch GPU calisirken hazirlanir).
    Bos alan (real_pos None) None kalir."""
    if dev.type != "cuda":
        return batch
    mv = lambda t: None if t is None else t.pin_memory().to(dev, non_blocking=True)  # noqa: E731
    return D.PackedBatch(**{f.name: mv(getattr(batch, f.name)) for f in dataclasses.fields(batch)})


def _to_device_bag(full, plan, dev):
    """Torba payi ve recipe.bag_plan tensorleri -> cihaz (CUDA'da sabitlenmis bellekten non_blocking)."""
    if full is None or dev.type != "cuda":
        return full, plan
    mv = lambda t: t.pin_memory().to(dev, non_blocking=True)
    return mv(full), dict(plan, ids=mv(plan["ids"]), rows=mv(plan["rows"]), cols=mv(plan["cols"]), keep=mv(plan["keep"]),
                          pairs=tuple(mv(p) for p in plan["pairs"]))


class _Exam:
    """exam_scores icin model: loss_per_target egitimle ayni maske yolundan (CUDA'da block_mask; belge 24 s5 I)."""

    def __init__(self, model, mask_fn, cuda, last=None):
        self.model, self.mask_fn, self.cuda, self.last = model, mask_fn, cuda, last

    def loss_per_target(self, batch):
        """last (sentence.summaries_last): batch o duzende islenir, sonuclar build_batch duzeninin hedef sirasina geri."""
        if self.last is None:
            return self.model.loss_per_target(batch, _attn(batch, self.mask_fn, self.cuda))
        pb, perm = self.last(batch)
        nll, pred, tk = self.model.loss_per_target(pb, _attn(pb, self.mask_fn, self.cuda))
        T = perm.shape[1]
        old = (torch.arange(len(perm), device=perm.device)[:, None] * T + perm)[pb.target >= 0]
        order = torch.argsort(old)
        return nll[order], pred[order], tk[order]


def _exam(model, mask_fn, valid, plan, story_bytes, layout, dev, cuda, last=None):
    t = time.time()
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=cuda):
        out = M.exam_scores(_Exam(model, mask_fn, cuda, last), valid, plan, story_bytes, layout, dev, BATCH_ROWS)
    return dict(out, seconds=round(time.time() - t, 2))


def _readings(model, valid, tok, spec=None, path=None):
    """Istemler (spec: [dict(label, story, sentences, decode)]; yoksa reading_prompts.json) -> (satirlar, decode basina
    olculer).  Uretim fp32, autocast yok."""
    spec = json.load(open(path or READING_PROMPTS, encoding="utf-8"))["prompts"] if spec is None else spec
    rows, gen = [None] * len(spec), {}
    for decode in ("greedy", "sample"):
        idx = [i for i, p in enumerate(spec) if p["decode"] == decode]
        if not idx:
            continue
        prompts = [[s.tolist() for s in valid.sentences(spec[i]["story"])[:spec[i]["sentences"]]] for i in idx]
        texts, gen[decode] = M.story_generation(model, prompts, decode, SAMPLE_SEED, tokenizer=tok,
                                                labels=[spec[i]["label"] for i in idx], **READING_LIMITS)
        for i, t in zip(idx, texts):
            real = valid.sentences(spec[i]["story"])[spec[i]["sentences"]:]
            rows[i] = dict(spec[i], prompt=t["prompt"], story_text=t["story"], eos=t["eos"],
                           real=M.reading_view([s.tolist() for s in real], tok))
    return rows, gen


def _samples_text(rows):
    out = []
    for r in rows:
        out += ["=== %s | hikaye %d, istem %d cumle, %s | EOS %s" % (r["label"], r["story"], r["sentences"], r["decode"],
                                                                       "var" if r["eos"] else "YOK"),
                "--- istem", r["prompt"], "--- model", r["story_text"], "--- gercek devam", r["real"], ""]
    return "\n".join(out)


def _git():
    try:
        rev = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             timeout=20).stdout.strip()
        dirty = subprocess.run(["git", "-C", HERE, "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                               text=True, timeout=20).stdout.strip()
        return dict(commit=rev or None, dirty=bool(dirty))
    except Exception:  # noqa: BLE001
        return dict(commit=None, dirty=None)


def _args(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=("transformer", "model_z"))
    ap.add_argument("--lr", type=float, required=True,
                    help="tepe lr (WSD); olculen: muon 2e-3, adamw 5e-4 (belge 39, OLCULENLER_z)")
    ap.add_argument("--out", required=True, help="kosu klasoru")
    ap.add_argument("--data", default="/content/drive/MyDrive/v2/simplestories_gpt2", help="sinir ve plan dosyalari")
    ap.add_argument("--stream", default="/content/drive/MyDrive/simplestories", help="gpt2/{train,valid}.npy kok")
    ap.add_argument("--local", default=None, help="ham akisin yerel kopyasi (yalniz onbellek)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=None, help="toplam adim (WSD tam takvim bu sayiyla); lr taramasi")
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", type=int, default=0)
    ap.add_argument("--optimizer", default=None, choices=("adamw", "muon", "normuon"),
                    help="normuon (varsayilan, 8 Ekim): Muon + noron basina normalizasyon (recipe.NorMuon, li2025_normuon); "
                         "muon: bloklarin 2-B matrisleri Muon'a (match_rms_adamw, ayni --lr), geri kalan AdamW'ye; adamw: "
                         "tek AdamW; --resume'da verilmezse kosunun kimliginden")
    ap.add_argument("--global_layers", type=int, default=None,
                    help="model_z: son N blok tam causal, gercek konumla (belge 40 s6.2 Deney G); varsayilan model_z'de "
                         "3 (8 Ekim), transformer'da 0; 0: G'siz Model Z; --resume'da verilmezse kosunun kimliginden")
    ap.add_argument("--layer_plan", default=None,
                    help="model_z katman plani, ornek loc2,mid4,loc1,glob1 (mid: yalniz BOS + Z satirlari, belge 52); "
                         "verilirse --layers ve --global_layers ondan")
    ap.add_argument("--bag_k", type=int, default=0, help="ogrenen torba boyu K = |C| + |P_k u L_k| siniri (0: tam softmax)")
    ap.add_argument("--bag_core", type=int, default=50, help="C: train sayiminda en sik N token (+ END + EOS)")
    ap.add_argument("--bag_weight", type=float, default=0.1, help="secici kaybinin agirligi (lambda; olculmedi)")
    ap.add_argument("--bag_full_frac", type=float, default=0.05,
                    help="gecerli hedeflerin bu payinda tam softmax CE de eklenir (belge 54 s2.3 yol a)")
    ap.add_argument("--bag_sel_frac", type=float, default=1.0,
                    help="secici kaybi torbalarin bu payinda (rastgele, adim tohumlu); secim her torbada (kullanici, 7 Ekim)")
    ap.add_argument("--summaries_last", type=int, default=None,
                    help="model_z: satir bellekte [token'lar | ozetler | dolgu] (belge 66); model ayni, maske iki aralik; "
                         "varsayilan model_z torbasiz 1, aksi 0; --resume'da verilmezse kosunun kimliginden")
    ap.add_argument("--z_bow_weight", type=float, default=0.0,
                    help="model_z: Z_k'dan sonraki cumlenin token torbasi ek kaybi agirligi (0: kapali; kullanici, 8 Ekim)")
    ap.add_argument("--fp8", default="none", choices=("none", "tensorwise", "rowwise"),
                    help="MLP (gate_up, down) torchao Float8Linear tarifi; none: bf16 (kimlige girmez, --resume'da "
                         "degistirilebilir; kullanici, 8 Ekim)")
    ap.add_argument("--stop_step", type=int, default=None,
                    help="takvim (WSD, epok plani) degismeden adim N'de dur: checkpoint.pt (surdurulebilir) + agent.pt + "
                         "results.json (finished False, stopped_at N); son sinav ve okuma yok (kullanici, 8 Ekim)")
    ap.add_argument("--checkpoint_minutes", type=float, default=10,
                    help="en cok bu kadar duvar saati kaybi (sinav dahil); surdurmede degistirilebilir")
    args = ap.parse_args(argv)
    args.defaulted = [k for k in INHERIT if getattr(args, k) is None and not (k == "global_layers" and args.layer_plan)]
    if args.optimizer is None:                                           # 8 Ekim: NorMuon varsayilan
        args.optimizer = DEFAULT_OPTIMIZER
    if args.global_layers is None:                                       # 8 Ekim: G3 varsayilan
        args.global_layers = MODEL_Z_GLOBAL_LAYERS if args.model == "model_z" else 0
    if args.summaries_last is None:
        args.summaries_last = MODEL_Z_SUMMARIES_LAST if args.model == "model_z" and not args.bag_k else 0
    if args.layer_plan:
        if args.model != "model_z":
            sys.exit("DUR: --layer_plan yalniz model_z")
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "model_z"))
        from sentence import parse_layer_plan
        plan = parse_layer_plan(args.layer_plan)
        args.layers = len(plan)
        args.global_layers = plan.count("glob")
    return args


def main(argv=None):
    args = _args(argv)
    t0 = time.time()
    log = lambda msg: print("[%7.1f sn] %s" % (time.time() - t0, msg), flush=True)  # noqa: E731
    ckpt0 = os.path.join(args.out, "checkpoint.pt")
    if args.resume and args.defaulted and os.path.exists(ckpt0):        # varsayilan degisse de kosu kendi ayariyla surer
        was0 = torch.load(ckpt0, map_location="cpu", weights_only=False, mmap=True)["args"]
        for k in args.defaulted:
            setattr(args, k, was0.get(k, INHERIT[k]))
        log("surdurme: verilmeyen %s kosunun kimliginden: %s" % (args.defaulted, {k: getattr(args, k)
                                                                                 for k in args.defaulted}))
    for err in (_global_error(args), _bag_error(args)):                  # veri yuklenmeden
        if err:
            sys.exit("DUR: " + err)
    if args.optimizer in ("muon", "normuon") and _muon_missing():      # sessizce AdamW'ye dusulmez
        sys.exit("DUR: --optimizer %s: %s" % (args.optimizer, _muon_missing()))
    dev = torch.device(args.device)
    cuda = dev.type == "cuda"
    if args.fp8 != "none" and _fp8_missing(cuda):                        # bf16'ya sessizce dusulmez
        sys.exit("DUR: --fp8 %s: %s" % (args.fp8, _fp8_missing(cuda)))
    if cuda:                                                             # GPU kapisi (kural 5)
        assert torch.cuda.is_available(), "GPU YOK"
        for k in ("cache_size_limit", "recompile_limit"):                # beklenen giris ~4-6 (egitim / sinav x tam /
            if hasattr(torch._dynamo.config, k):                         # son batch); 8'i asarsa sessizce eager'a duser
                setattr(torch._dynamo.config, k, 32)
        if COMPILE_MODE and "max-autotune" in COMPILE_MODE:              # bozuk Triton adayi yalniz alt sureci dusurur
            import torch._inductor.config as inductor_config              # (8 Ekim sm_120: aday illegal memory access, kosu
            if hasattr(inductor_config, "autotune_in_subproc"):          # adim 0'da oldu; 5b tek blok: alt surec 67,6 ms
                inductor_config.autotune_in_subproc = True                # cokme yok, GEMM ATEN 70,1 ms)
            else:
                log("UYARI: torch %s'te inductor autotune_in_subproc yok; adaylar ayni surecte denenir" % torch.__version__)
    elif os.name == "nt":
        log("guc kisitlamasi (EcoQoS) kapali: %s" % _no_power_throttling())
    stream = _local_copy(args.stream, args.local, args.data) if args.local else args.stream
    log("ham akis: %s%s" % (stream, " (yerel kopya, sha256 sinir dosyasiyla ayni)" if args.local else ""))
    train = D.TokenStories(stream, args.data, "train")
    valid = D.TokenStories(stream, args.data, "valid")
    plans, per_epoch, row_len, total = _schedule(train, args.data, args.seed, args.epochs, args.steps)
    bounds = np.cumsum([0] + per_epoch)
    down = _decay_start(total)
    ep = np.load(os.path.join(args.data, "exam_pack_plan.npz"))
    exam_plan = (ep["row_offsets"], ep["row_stories"])
    story_bytes = np.load(os.path.join(args.stream, "gpt2", "valid_bytes.npy"))
    assert len(story_bytes) == valid.n, "valid_bytes hikaye sayisi valid ile ayni degil"
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(args.stream, "gpt2", "tokenizer.json"))
    bag, core_cpu = dict(NO_BAG), None
    if args.bag_k:                                                       # C sayimdan (belge 55 O1)
        cpath = os.path.join(args.data, "train_token_counts.npy")
        if not os.path.exists(cpath):
            sys.exit("DUR: %s yok (data.token_counts ile bir kez uretilir)" % cpath)
        counts = np.load(cpath)
        core = R.core_ids(counts, args.bag_core)
        args.bag_n_core = len(core)
        core_cpu = torch.as_tensor(core)
        bag = dict(bag_k=args.bag_k, bag_core=args.bag_core, bag_weight=args.bag_weight, bag_full_frac=args.bag_full_frac,
                   bag_sel_frac=args.bag_sel_frac,
                   bag_core_sha256=hashlib.sha256(core.tobytes()).hexdigest())
    model, mask_fn, layout = _build(args, dev)
    last = None
    if args.z_bow_weight:
        from sentence import z_bow_targets
    if args.summaries_last:                                              # belge 66: [token'lar | ozetler | dolgu]
        from sentence import summaries_last as last
        mask_fn = model._masks(True)
    if args.bag_k:
        model.bag.fill(core, counts)
    n_fp8 = _fp8(model, args.fp8) if args.fp8 != "none" else 0          # compile ve optimizer'dan once
    if cuda:
        for block in model.blocks:
            block.compile(dynamic=False, mode=COMPILE_MODE)
    opt, opt_info = _optimizer(model, args.optimizer, args.lr, cuda)
    ident = dict(model=args.model, d=args.d, layers=args.layers, heads=args.heads, lr=args.lr, seed=args.seed,
                 longest=train.max_sentence_tokens, row_len=row_len, batch_rows=BATCH_ROWS,
                 train_stream_sha256=train.meta["stream_sha256"], learned_z=int(args.model == "model_z"),
                 optimizer=args.optimizer, global_layers=args.global_layers, layer_plan=args.layer_plan,
                 summaries_last=args.summaries_last, z_bow_weight=args.z_bow_weight, **bag)   # learned_z: eski kosu ayrimi
    plan_meta = dict(total=total, decay_start=down, per_epoch=per_epoch,
                     plan_sha256=[hashlib.sha256(np.ascontiguousarray(rs)).hexdigest() for _, rs in plans])
    params = sum(p.numel() for p in model.parameters())
    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, "checkpoint.pt")
    decay_dir = os.path.join(args.out, "decay_start")
    res_path = os.path.join(args.out, "results.json")
    start, history = 0, dict(log=[], exams=[], epochs=[])
    if os.path.exists(ckpt) and not args.resume:
        sys.exit("DUR: %s'de checkpoint var; surdurmek icin --resume 1, yeni kosu icin yeni klasor" % args.out)
    if args.resume:
        if not os.path.exists(ckpt):
            sys.exit("DUR: surdurme paketi yok: %s (bastan baslamaz)" % ckpt)
        peek = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)   # kimlik YUKLEMEDEN once: d farki
        was, old = peek["args"], peek["plan"]                             # load_state_dict'te patlamasin
        del peek
        if any(k in was for k in LEGACY):                                 # kullanici, 6 Ekim: eski kosu uzatilmaz
            sys.exit("DUR: temizlik oncesi kosu surdurulmez / uzatilmaz (kullanici, 6 Ekim); eski kod: git etiketi %s" % TAG)
        if _archived(was):                                                # formullu Model Z (belge 44)
            sys.exit("DUR: " + _archived(was))
        was = {"learned_z": 0, "optimizer": "adamw", "global_layers": 0, "layer_plan": None, "summaries_last": 0, "z_bow_weight": 0.0, **NO_BAG, **was}   # alanlardan onceki kosu
        diff = {k: (was.get(k), ident[k]) for k in IDENTITY if was.get(k) != ident[k]}
        n = len(old["plan_sha256"])
        if old["plan_sha256"] != plan_meta["plan_sha256"][:n]:
            diff["plan_sha256"] = "veri sirasi farkli"
        if diff:
            sys.exit("DUR: surdurme ayari checkpoint'ten farkli (checkpoint, simdi): %s" % diff)
        info = R.Checkpoint.load(args.out, model, opt, "cpu")
        if total < old["total"]:
            sys.exit("DUR: toplam adim %d < checkpoint'teki %d (kisaltma yok)" % (total, old["total"]))
        if total == old["total"]:
            if info["step"] == total and os.path.exists(res_path) and json.load(open(res_path)).get("finished"):
                log("kosu bitmis (%d / %d adim, results.json var); uzatma icin --epochs / --steps buyutulur" % (
                    total, total))
                return json.load(open(res_path))
            start, history = info["step"], info["history"]
            log("SURDURULDU: adim %d / %d" % (start, total))
        else:                                                            # uzatma (kural 3): inis basindan
            if info["step"] > old["decay_start"]:
                info = R.Checkpoint.load(decay_dir, model, opt, "cpu")
                assert info["step"] == old["decay_start"], "decay_start checkpoint'i inis basinda degil"
            arch = os.path.join(args.out, "total_%d" % old["total"])
            for name in OUTPUTS:
                if os.path.exists(os.path.join(args.out, name)):
                    os.makedirs(arch, exist_ok=True)
                    shutil.move(os.path.join(args.out, name), os.path.join(arch, name))
            start, history = info["step"], info["history"]
            log("UZATMA: toplam %d -> %d, adim %d'den (inis basi; eski ciktilar %s)" % (old["total"], total, start, arch))
    config = dict(identity=ident, plan=plan_meta, args=vars(args), params=params, optimizer=opt_info, env=dict(
        torch=torch.__version__, device=torch.cuda.get_device_name(0) if cuda else "cpu", git=_git()),
        data=dict(train_stories=train.n, train_sentences=len(train.sent), exam_stories=int(len(exam_plan[1])),
                  exam_set_sha256=str(ep["exam_set_sha256"]) if "exam_set_sha256" in ep.files else None,
                  exam_bytes=int(story_bytes[exam_plan[1]].sum()), train_stream_sha256=train.meta["stream_sha256"],
                  valid_stream_sha256=valid.meta["stream_sha256"]))
    json.dump(config, open(os.path.join(args.out, "config.json"), "w"), indent=1)
    log("%s | d %d, katman %d, head %d, %d parametre | egitim %d hikaye, satir %d x %d, %s adim/epok, toplam %d, inis "
        "basi %d | lr %g | cihaz %s, compile %s | sinav %d hikaye" % (
            args.model, args.d, args.layers, args.heads, params, train.n, BATCH_ROWS, row_len, per_epoch, total, down,
            args.lr, config["env"]["device"], cuda, len(exam_plan[1])))
    if args.layer_plan:
        log("layer_plan %s: %s" % (args.layer_plan, ",".join(model.plan)))
    if args.global_layers:
        log("global_layers %d: son %d blok tam causal (model_z_global_mask), gercek hikaye konumu" % (
            args.global_layers, args.global_layers))
    if args.bag_k:
        log("torba: K %d, C %d (en sik %d + END + EOS), lambda %g, tam softmax payi %g, secici kaybi payi %g" % (
            args.bag_k, len(core), args.bag_core, args.bag_weight, args.bag_full_frac, args.bag_sel_frac))
    if cuda:
        import torch._inductor.config as inductor_config
        log("hiz: derleme modu %s (autotune alt surecte %s), attention blok %d, secici derlenmis %s, Muon %s" % (
            COMPILE_MODE, getattr(inductor_config, "autotune_in_subproc", None), R.ATTN_BLOCK, R.SELECT_COMPILED,
            type(getattr(opt, "muon", opt)).__name__))
    for k, g in opt_info["split"].items():
        if g["tensors"]:
            log("optimizer %s | %s: %d tensor, %d parametre | %s" % (args.optimizer, k, g["tensors"], g["params"],
                                                                     ", ".join("%s x%d" % kv for kv in g["names"].items())))

    lengths = train.lengths()

    def rows_of(step):
        e = int(np.searchsorted(bounds, step, "right")) - 1
        ro, rs = plans[e]
        r0, r1 = (step - bounds[e]) * BATCH_ROWS, min((step - bounds[e] + 1) * BATCH_ROWS, len(ro) - 1)
        return [rs[ro[r]:ro[r + 1]].tolist() for r in range(r0, r1)], int(lengths[rs[ro[r0]:ro[r1]]].sum())

    def cpu_batch(step):
        rows, real = rows_of(step)
        b = D.build_batch(train, rows, layout, "cpu", row_len)
        if last is not None:
            b = last(b)[0]
        if not args.bag_k:
            return b, real, None, z_bow_targets(b) if args.z_bow_weight else None
        rng = np.random.default_rng(np.random.SeedSequence(args.seed, spawn_key=(step,)))   # adim tohumlu: surdurmede ayni
        full = R.bag_full_mask(b.target.numpy(), args.bag_full_frac, rng, args.bag_sel_frac)
        return b, real, full, R.bag_plan(b, full, core_cpu)                # torba yapisi CPU'da (adimda senkron yok)

    def save(dir_, step):
        R.Checkpoint.save(dir_, model, opt, step, plan_meta, history, ident)

    end = total if args.stop_step is None else args.stop_step         # --stop_step: takvim total'den, dongu end'e
    if not start < end <= total:
        sys.exit("DUR: --stop_step %s: adim %d < N <= %d olmali" % (args.stop_step, start, total))
    history.setdefault("segments", []).append(dict(start=start, fp8=args.fp8, fp8_linears=n_fp8))   # kimlik disi kip
    log("fp8 %s (%d Linear)" % (args.fp8, n_fp8))
    sw, win = R.SpeedWindow(), None
    first_window, epoch_from, epoch_t0, saved_at = True, start, time.time(), time.time()
    ckpt_seconds = 60 * args.checkpoint_minutes                      # kimlige girmez (kullanici: buyuk kosuda 30 dk)
    nxt = cpu_batch(start) if start < total else None
    for step in range(start, end):
        if win is None:
            sw.start(step)
            win = dict(step0=step, loss=torch.zeros((), device=dev), gn=torch.zeros((), device=dev), tokens=0, bag={})
            if cuda:
                torch.cuda.reset_peak_memory_stats()
        lr = R.wsd_lr(step, total, args.lr, decay=DECAY)
        for g in opt.param_groups:
            g["lr"] = lr
        batch, real, full, plan = nxt
        timer = tuple(torch.cuda.Event(enable_timing=True) for _ in range(4)) if cuda else None
        bow = None
        if args.z_bow_weight:                                             # z_bow hedefleri plan yerinde tasinir
            bow = (args.z_bow_weight, {k: v.pin_memory().to(dev, non_blocking=True) if cuda else v
                                       for k, v in plan.items()})
            plan = None
        full, plan = _to_device_bag(full, plan, dev)
        loss, gn, extra = _step(model, _to_device(batch, dev), mask_fn, opt, cuda, full, args.bag_weight, timer, plan,
                                bow)
        nxt = cpu_batch(step + 1) if step + 1 < total else None          # GPU calisirken hazirlanir
        win["loss"] += loss
        win["gn"] += gn
        win["tokens"] += real
        for k, v in (extra or {}).items():                               # torba: toplamlar cihazda (senkron yok)
            win["bag"][k] = win["bag"].get(k, 0) + v
        done = step + 1
        epoch = int(np.searchsorted(bounds, step, "right"))               # 1'den
        epoch_end = done == bounds[epoch]
        ckpt_due = epoch_end or done in (total, end) or (done % LOG_EVERY == 0
                                                            and time.time() - saved_at >= ckpt_seconds)
        if not (done % LOG_EVERY == 0 or ckpt_due or done == down):
            continue
        s = sw.stop(done, win["tokens"])                                 # pencere kapanir: kayit / sinav disarida
        k = done - win["step0"]
        rec = dict(step=done, epoch=epoch, lr=lr, loss=round(win["loss"].item() / k, 4),
                   grad_norm=round(win["gn"].item() / k, 4), steps=k, tokens=s["tokens"], seconds=s["seconds"],
                   ms_per_step=round(1000 * s["seconds"] / k, 1), tokens_per_sec=s["tokens_per_sec"], first=first_window,
                   peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else None,
                   peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 1e9, 2) if cuda else None)
        if cuda:                                                         # son adimin cikis kalemi, yalniz ileri
            rec["out_fwd_ms"] = round(timer[0].elapsed_time(timer[1]), 2)
            if args.bag_k:
                rec["selector_fwd_ms"] = round(timer[2].elapsed_time(timer[3]), 2)
        if args.z_bow_weight:
            rec["z_bow"] = round(float(win["bag"]["z_bow"]) / max(float(win["bag"]["z_bow_n"]), 1), 4)
        if args.bag_k:                                                   # kalibrasyon: p_other ~ miss (belge 54 s1.4)
            w = {k: float(v) for k, v in win["bag"].items()}
            n_ = max(w["valid"], 1)
            rec["bag"] = dict(c=round(w["c"] / n_, 4), p=round(w["p"] / n_, 4), l=round(w["l"] / n_, 4),
                              miss=round(w["miss"] / n_, 4), p_other=round(w["p_other"] / n_, 4),
                              full_ce=round(w["full"] / w["n_full"], 4) if w["n_full"] else None,
                              loss_selector=round(w["loss_selector"] / max(w["n_selector"], 1), 4),
                              p_over=int(w["p_over"]))
        history["log"].append(rec)
        log("adim %d / %d (epok %d)  lr %.3g  kayip %.4f  grad %.3f | pencere %d adim %.1f sn  %.1f ms/adim  %.0f tok/sn%s%s%s"
            % (done, total, epoch, lr, rec["loss"], rec["grad_norm"], k, s["seconds"], rec["ms_per_step"],
               s["tokens_per_sec"], "  tepe %.1f GB (ayrilan %.1f)  cikis ileri %.1f ms%s" % (
                   rec["peak_gb"], rec["peak_reserved_gb"], rec["out_fwd_ms"],
                   " (secici %.1f)" % rec["selector_fwd_ms"] if args.bag_k else "") if cuda else "",
               "  | C %.3f P %.3f L %.3f kacan %.4f p(DIGER) %.4f tam CE %s secici %.3f" % tuple(
                   rec["bag"][k] for k in ("c", "p", "l", "miss", "p_other", "full_ce", "loss_selector"))
               if args.bag_k else "",
               ("  | z_bow %.4f" % rec["z_bow"] if args.z_bow_weight else "")
               + ("  (ilk pencere: derleme dahil)" if first_window else "")))
        assert np.isfinite(rec["loss"]), "kayip sonlu degil; checkpoint yazilmadi"
        first_window, win = False, None
        if done == down:
            save(decay_dir, done)                                        # uzatma buradan (belge 20 s4)
        if epoch_end or done == total:
            history["epochs"].append(dict(epoch=epoch, step=done, wall_seconds=round(time.time() - epoch_t0, 1),
                                          resumed=bool(epoch_from > bounds[epoch - 1])))
            ex = _exam(model, mask_fn, valid, exam_plan, story_bytes, layout, dev, cuda, last)
            history["exams"].append(dict(ex, step=done, epoch=epoch, full_epoch=bool(epoch_end)))
            log("SINAV adim %d: kayip %.4f  acc %.4f  acc_token %.4f  bpb %.4f  end_ok %.4f  eos_ok %.4f (%.1f sn)" % (
                done, ex["loss"], ex["acc"], ex["acc_token"], ex["bits_per_byte"], ex["end_ok"], ex["eos_ok"],
                ex["seconds"]))
            epoch_from, epoch_t0 = done, time.time()
        if ckpt_due:
            save(args.out, done)
            saved_at = time.time()
    if end < total:                                                     # --stop_step: sinav ve okuma yok
        results = dict(config, run=os.path.basename(os.path.normpath(args.out)), finished=False, stopped_at=end,
                       readings_skipped="stop_step", exam=history["exams"][-1] if history["exams"] else None,
                       exams=history["exams"], epochs=history["epochs"], generation=None, log=history["log"],
                       segments=history["segments"])
        torch.save(dict(state=model.state_dict(), identity=ident, args=vars(args)), os.path.join(args.out, "agent.pt"))
        json.dump(results, open(res_path, "w"), indent=1)
        log("DURDU: --stop_step %d / %d (checkpoint.pt surdurulebilir, agent.pt, results.json; sinav ve okuma yok)" % (
            end, total))
        return results
    if not history["exams"] or history["exams"][-1]["step"] != total:   # bitis checkpoint'i var, sinavi yok
        ex = _exam(model, mask_fn, valid, exam_plan, story_bytes, layout, dev, cuda, last)
        history["exams"].append(dict(ex, step=total, epoch=len(per_epoch), full_epoch=bool(total == bounds[-1])))
    speed = [w for w in history["log"] if not w["first"]]
    med = lambda key: float(np.median([w[key] for w in speed])) if speed else None  # noqa: E731
    results = dict(config, run=os.path.basename(os.path.normpath(args.out)), finished=True, exam=history["exams"][-1],
                   exams=history["exams"], epochs=history["epochs"], generation=None, log=history["log"],
                   segments=history["segments"],
                   speed=dict(windows=len(speed), tokens_per_sec_median=med("tokens_per_sec"),
                              ms_per_step_median=med("ms_per_step"), note="pencere ortancasi; ilk pencere (derleme) haric"))
    torch.save(dict(state=model.state_dict(), identity=ident, args=vars(args)), os.path.join(args.out, "agent.pt"))
    json.dump(results, open(res_path, "w"), indent=1)                   # okumadan ONCE: okuma dusse de model kalir
    if "mid" in (getattr(model, "plan", None) or ()):
        results["readings_skipped"] = "OKUMA YOK: layer_plan mid (uretim onbellegi yok)"
        log(results["readings_skipped"])
    else:
        t = time.time()
        own = os.path.join(args.data, "reading_prompts.json")          # veri klasorunun istemleri (FineWeb, belge 48)
        rows, gen = _readings(model, valid, tok, path=own if os.path.exists(own) else None)
        log("okuma uretimi %.0f sn: %s" % (time.time() - t, gen))
        open(os.path.join(args.out, "samples.txt"), "w", encoding="utf-8").write(_samples_text(rows))
        json.dump(dict(generation=gen, rows=rows), open(os.path.join(args.out, "samples.json"), "w", encoding="utf-8"),
                  indent=1, ensure_ascii=False)
        results["generation"] = gen
    json.dump(results, open(res_path, "w"), indent=1)
    gen = results["generation"] or {}
    log("BITTI: %s | sinav kayip %.4f bpb %.4f | sentence_repeat %s | story_loop %s" % (
        args.out, results["exam"]["loss"], results["exam"]["bits_per_byte"],
        {d: g["sentence_repeat"] for d, g in gen.items()},
        {d: (g["story_loop"], g["story_loop_prompts"]) for d, g in gen.items()}))
    return results


if __name__ == "__main__":
    main()
