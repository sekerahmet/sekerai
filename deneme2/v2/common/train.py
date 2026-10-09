"""train -- V2 ortak egitim: iki model ayni veri, ayni hedef, ayni tarif, ayni sinav; farkli olan yalniz model
(belge/model_z_temel/20 s4, 21, 22, 24 s9; ad onayli, kullanici 6 Ekim).

Adim (belge 24 s9):  attn = recipe.block_mask(batch, mask_fn) (CPU'da dense_mask: FlexAttention CPU'da geri yayilim
yapmiyor); h = model._batch_hidden(batch, attn); kayip = recipe.output_loss(h, E, hedef).  bf16 autocast ve bloklarda
compile(dynamic=False) CUDA'da; clip 1,0; AdamW (0,9 / 0,95, wd 0,1; CUDA'da fused), recipe.param_groups, recipe.wsd_lr.
--optimizer normuon VARSAYILAN (kullanici, 8 Ekim: "Ben nurmuon yapalım şimdiden dedim"; "normuon bence standart yapalım.
büyüdükçe etkisini gösterdi"; d1024 / L12 ayni adimda -0,012; --resume'da acik verilmezse kosunun kimliginden):
recipe.NorMuon (li2025_normuon Algorithm 1), ayni --lr (guncelleme RMS'i 0,2 lr).  --optimizer muon (7 Ekim
varsayilani): bloklarin 2-B matrisleri recipe.BatchedMuon'a (torch.optim.Muon matematigi; adjust_lr_fn match_rms_adamw:
guncelleme RMS'i AdamW'ninki, ayni --lr ve wd; liu2025_muonscalable), geri kalan ayni AdamW'ye; wsd_lr ikisine.  Muon yoksa kosu baslamadan DURUR.  --lr
sayi ya da auto (varsayilan; kullanici, 8 Ekim: "her lr kendi modeline özgü D ye göre"): muon / normuon'da LR_REF[0]
(LR_REF[1] / d) ^ LR_REF[2] (adim basina aci ~ lr 0,2 sqrt(d) sabit; d768'de olculen 2e-3, belge 39 + OLCULENLER_z);
adamw'de sayi sart (DUR).  --optimizer adamw: eski tarif (olculen lr 5e-4).
--fp8 none|tensorwise|rowwise (kullanici, 8 Ekim: "fp8 de dene bakalım. fp8 dikkatli dene"; varsayilan none): torchao
Float8Linear yalniz MLP'de (gate_up, down), compile'dan once; agirliklar fp32, state_dict adlari ayni (kosu basinda
denetlenir) -> ayni kosu --resume ile FP8 acik / kapali surer; KIMLIGE GIRMEZ, kip her segmentte gunlukte ve results.json
segments'ta.  CUDA ve torchao ister, yoksa DURUR (bf16'ya sessizce dusmez).
Veri: BATCH_ROWS satir x row_len (plan dosyasindan); epok 1 <data>/train_pack_plan_e1.npz, sonrakiler pack_plan(seed,
epok).  --local: ham akisin yerel kopyasi (yalniz onbellek; sha256 = <split>_boundaries.json'daki).  Epok sonu eksik batch
CUDA'da bos (dolgu) satirla BATCH_ROWS'a tamamlanir, sinavin eksik batch'i her yerde (kullanici, 8 Ekim: "epok sonu
derlemeyi de ekle"): derleme sekli sabit, veri atilmaz, dolgu hedefsiz.
Model Z egitim / sinav duzeni summaries_last (belge 66; kullanici, 8 Ekim: "Fikrine onay verdim"): batch'ler
sentence.summaries_last ile [token'lar | ozetler | dolgu] sirasinda (sinav sonucu hedef sirasina geri), maske sorgu basina
iki aralik; uretim degismez.  --summaries_last bayragi ve torba (--bag_k) kaldirildi (kullanici, 8 Ekim: "Torbada gereksiz
gibi"; belge 77); kimlikte summaries_last duzen isareti (model_z 1).
Sozluk dolgusu (belge 89; kullanici, 8 Ekim: "sözlük dolgusu ok, ekle"): yeni kosuda E VOCAB_ROWS (50.304, 64'un kati)
satir; dolgu satirlari sifir, egitim kaybinda dolgu sutunu -inf, cikis VOCAB'a kesilir (hedef olmaz, uretilmez).  Kimlikte
vocab_rows; alani olmayan eski kosu VOCAB (50.258) ile yuklenir ve surer.
--ngram_embed N (deneme; kullanici, 8 Ekim: "isimler ok, başlat"; belge 88b, 90b): Model Z bigram embedding tablosu N satir
(0 kapali, varsayilan; deneme 251520 = 5 x 50.304), sentence.bigram_ids; tablo AdamW wd 0, lr x recipe.NGRAM_LR_MULT.
Kimlikte; transformer ile DUR.  Hiz secenekleri (belge 90b ek; deneme): --ngram_layers K yalniz ilk K blok girdisine
(0 hepsi), --ngram_sparse 1 tabloyu yalniz okunan satirlarla gunceller (recipe.NgramRowAdam: beta1 0 Adam, yogun
AdamW(0, beta2) ile ayni matematik).  Varsayilan 0 (kullanici, 9 Ekim: "lan hepsinde model boyu vs demeden aynı katkıyı
verdi. demek ki modelden bağımsız bir katkısı var ayrıca döngü de arttı."; 8 Ekim'den 9 Ekim'e auto idi).  --ngram_embed auto:
model_z'de 5 x vocab_rows satir (50.304'te 251.520), transformer'da 0.  n-gram acikken (auto ya da acik N) --ngram_layers
verilmezse 1 (yalniz ilk blok girdisi; d768 3.000 adim 3,3405 / 219,8 ms, taban 3,3567 / 210,9, belge 93 ek); acik
--ngram_layers 0 butun katmanlar.  Acik --ngram_embed 0 eski davranis (bit ayni); --ngram_sparse varsayilani 0.  Uc alan
INHERIT'te (alan yoksa 0: eski kosu kendi ayariyla surer).
--glob_kv_heads N|auto (kullanici, 8 Ekim: "bu duurmda GOA yı da sıraya koy o zaman bakalım"; uretim hizi): GQA, k / v
N head yalniz tam causal katmanlarda (Model Z glob; transformer'da her katman, kiyas icin); yerel katmanlar tam head
(Z K/V kanali daralmaz).  Varsayilan auto (kullanici, 8 Ekim: "GQA'yı varsayılan yap, ona karar verdik son koşuda bu
yüzden yaptık"): Model Z ve global_layers > 0 ise heads / GLOB_KV_GROUP (bolunmezse DUR), aksi halde (transformer, G'siz
Model Z) 0 = heads.  Acik 0 eski davranis (bit ayni); kimlikte (sekil degisir).
--carry_summaries 1 / --carry_group G (belge 81b, 83; kullanici, 8 Ekim: "isimler ok, carry kodunu başlat"): belgenin
ardisik <= G parcasi ayni batch'te ardisik satirlarda (data.carry_pack_plan, kosu basinda, dosyasiz); parcanin son Z'si
gruptaki sonraki parcanin ilk token'ini hedefler.  carry_summaries 1: devam parcasi BOS'suz, konum onceki parcalarin
devami, her katmanda onceki parcalarin BOS + Z'lerini (glob'da yalniz Z'leri) bellek olarak okur (gradyan akar; KV =
[satir || bellek], M_max plandan); --carry_group G tek basina = K kontrolu (ayni plan / hedef, BOS'lu, bellek yok).  Yalniz
model_z.  Sinavda continuation_exam (valid'in uzun belgeleri), gunlukte loss_cont.
--attn_gate 1 (belge 88a, 90a; kullanici, 8 Ekim: "o zaman attention head yapalım mı"; yalniz model_z): her blokta head
basina sigmoid cikis kapisi (sentence.Block._gate; girdi n1(x), agirlik sifirdan, kapi 0,5); agirlik (heads, d) bloklarin
2-B matrisi oldugu icin Muon / NorMuon grubunda.  Acik 0 = kapisiz (bit ayni); kimlikte, --resume'da verilmezse kosunun
kimliginden (alan yoksa 0); kapisiz checkpoint kapili surdurulmez (DUR).  --attn_gate 2 (kullanici, 8 Ekim: "onaylıyorum,
ikinci kolu da ekle"; "onaylıyorum, d // 64 yap"): kapi girdisi n1(x)'in ilk d // 64 boyutu, W (heads, d // 64) (speedrun
124M tarifi; belge 88a s4.3, 90a; d768'de 12); d < 64 DUR.  Varsayilan auto (kullanici, 8 Ekim: "gate 2 varsayılan"):
model_z ve d >= 64 ise 2, aksi halde 0.
--stop_step N (kullanici, 8 Ekim): takvim degismeden adim N'de durur; checkpoint.pt + agent.pt + results.json (finished
False, stopped_at, readings_skipped "stop_step"), son sinav ve okuma yok; --resume 1 kaldigi yerden.
Surdurme: <out>/checkpoint.pt son kayittan --checkpoint_minutes sonraki ilk gunluk sinirinda, epok sonunda ve bitiste;
<out>/decay_start/ inisin ilk adiminda.  --resume 1: ayni toplam -> kaldigi yerden; buyuk toplam (--epochs / --steps) -> uzatma, inis basindan; eski
ciktilar <out>/total_<eski toplam>/'a.  Bitmis kosu durur.
Olcu: epok sonunda ve bitiste metrics.exam_scores (exam_pack_plan.npz; egitimle ayni maske yolu); hiz pencere pencere
(recipe.SpeedWindow; ilk pencere derleme icerir, ozete girmez).  Sonda agent.pt ve results.json, SONRA
metrics.story_generation (reading_prompts.json; okuma dusse de model kalir).  Cikti: config.json, checkpoint.pt, decay_start/, results.json, agent.pt, samples.txt, samples.json.
Ek okuma kayitli kosudan: diag/generate_readings.py.
--mtp N (deneme/mtp dali; kullanici, 8 Ekim: "A'yı onaylıyorum, başlat"; belge 88c, 90c): ayni-logit MTP, ek parametresiz
(modded-nanogpt kayit 53): kayip = ana CE + agirlikli k+1 sonraki hedeflerin CE'si, ayni logit'ten (recipe.output_loss_mtp).
Hedefler hikaye sirasinda (data.mtp_targets, summaries_last'tan once, ayni perm).  Agirlik recipe.mtp_weights(adim, toplam,
N): son 1 / (N + 1) payda (WSD inisi dahil) yalniz ana hedef.  Sinav / bpb / okuma yalniz ana bas (dokunulmaz); gunlukte
loss = ana CE, loss_mtp / mtp_w ayri.  Yalniz model_z; acik --mtp N carry ile DUR; kimlikte.  Varsayilan 0 (kullanici, 9 Ekim:
n-gram ile birlikte; 8 Ekim'den 9 Ekim'e auto idi).  --mtp auto: model_z 2 (carry'de 0, gunlukte yazilir), transformer
0; acik --mtp 0 eski davranis (bit ayni); INHERIT'te (alan yoksa 0: eski kosu kendi ayariyla surer).
--g_latent_rank r / --g_raw_sentences K (deneme, belge 102; kullanici, 9 Ekim: "token vektör ama r bir izdüşüm"; "kaç
cümle gördüğünü pencere yapalım. K=0 yani kendi cümlesi ham olur"): Model Z glob katmanlarinda K + 1 cumleden eski token'lar
token basina latent'ten (MLA turu).  Varsayilan 0 (kapali, bit ayni); yalniz model_z, global_layers > 0, carry'siz;
kimlikte, INHERIT'te (alan yoksa 0).  --g_latent_rope dr (belge 102 s9; kullanici, 9 Ekim: "Bir de üretimi hızlandırması
lazım"): glob'da ayrik RoPE (son dr boyut) + uretimde absorb; 0 sade yol (bit ayni).  --g_latent_score 1 (belge 102
s11; kullanici, 9 Ekim: "Olur kur"): yalniz olcek, latent K'ya kv head (x kademe) basina ogrenilen carpan (1 baslar; kayma
yok, score_mod yok).  Esnek r (belge 103; kullanici, 9 Ekim: "bu arada hazır olunca GPU da başlat beni bekleme isimler
onaylı"): --g_latent_tiers 32,64,128,256 (son = --g_latent_rank) ile --g_latent_budget R (onem sirasina gore paylar; 9 Ekim
importance_probe sonrasi, Taylor secicisi kaldirildi), --g_latent_explore p (varsayilan 0); gunlukte tier_hist / tier_mean /
tier_hit / tier_ce.

Eski kosular (kullanici, 8 Ekim: "V2 içinde temizlik kastettim"; belge 77): Model Z kimliginde summaries_last 1 degilse
(8 Ekim oncesi; formullu / temizlik oncesi dahil; summaries_last 0), learned_z 0 ya da kaldirilan bir ozellik (z_bow,
layer_plan, z_reads_all, glob_drop, torba) varsa yuklenmez / surdurulmez (_archived; ileti eski kodun git etiketini verir).
Transformer yuklenir.
--global_layers N|auto (belge 40 s6.2 Deney G; yalniz Model Z): son N blok tam causal, gercek hikaye konumuyla; maske
ikilisi (yerel, global) _attn'dan, egitim / sinav / teshis ayni yol.  --global_layers 0: G'siz Model Z (kiyas).
Varsayilanlar (kullanici, 8 Ekim: "Varsayılan yap ama kısa bir koşu ile son halin çalıştığından emin olalım"; "G yi de
ölçüye bağlayalım"): model_z'de global_layers auto = round(layers x MODEL_Z_GLOBAL_RATIO) (L10 3, L12 4, L24 8; transformer
0).  --resume 1'de acikca verilmeyen global_layers / optimizer / glob_kv_heads / lr / attn_gate kosunun kimliginden
(INHERIT): eski kosular varsayilan degisse de kendi ayariyla surer.  Kimlikte "auto" degil cozulmus sayi.

    python train.py --model transformer|model_z --out <kosu> [--lr LR|auto (varsayilan auto)] [--data <v2/simplestories_gpt2>]
                    [--stream <simplestories>] [--local /content/v2_cache] [--epochs 1] [--steps N] [--d 512]
                    [--layers 8] [--heads 8] [--seed 0] [--device cuda] [--resume 1]
                    [--optimizer normuon|muon|adamw (varsayilan normuon)] [--global_layers N|auto (varsayilan auto)]
                    [--glob_kv_heads N|auto (varsayilan auto)] [--carry_summaries 1] [--carry_group G]
                    [--attn_gate 0|1|2|auto (varsayilan auto)] [--ngram_embed N|auto (varsayilan 0)] [--ngram_layers K]
                    [--ngram_sparse 1]
                    [--mtp N|auto (varsayilan 0)]
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
MODEL_Z_GLOBAL_RATIO = 1 / 3   # global_layers auto (OLCULENLER_z: d768/L10 G1->G3 kazanc, d1024/L12 G3->G4 -0,0047)
GLOB_KV_GROUP = 4              # glob_kv_heads auto = heads / 4
GLOB_KV_DEFAULT = "auto"       # --glob_kv_heads verilmezse (kullanici, 8 Ekim: GQA varsayilan); testler eski 0'a sabitler
ATTN_GATE_DEFAULT = "auto"     # --attn_gate verilmezse (kullanici, 8 Ekim: "gate 2 varsayılan"); testler eski 0'a sabitler
NGRAM_DEFAULT = 0              # --ngram_embed verilmezse (kullanici, 9 Ekim: katkisi modelden bagimsiz + dongu artti; 8 Ekim auto)
MTP_DEFAULT = 0                # --mtp verilmezse (kullanici, 9 Ekim, n-gram ile birlikte; 8 Ekim auto)
LR_REF = (2e-3, 768, 0.5)      # lr auto = lr0 (d0 / d) ^ us (aci / adim ~ lr 0,2 sqrt(d) sabit); d1024 olcumu: 1,4 / 1,7e-3 duz, 1,73e-3 icinde
INHERIT = ("global_layers", "optimizer", "glob_kv_heads", "lr", "attn_gate", "ngram_embed", "ngram_layers",
           "ngram_sparse", "mtp", "g_latent_rank", "g_raw_sentences", "g_latent_rope", "g_latent_score", "g_latent_tiers",
           "g_latent_budget", "g_latent_price", "g_latent_explore")   # --resume: kimlikten
VOCAB_ROWS = -(-D.VOCAB // 64) * 64   # yeni kosuda E satiri: 50.304 (sozluk dolgusu; belge 89, OLCULENLER 5o -1,5 ms/adim)
DEFAULT_OPTIMIZER = "normuon"                                   # kullanici, 8 Ekim
FP8_MODULES = ("gate_up", "down")                   # --fp8 donusturulen Linear'lar (MLP)
COMPILE_MODE = "max-autotune-no-cudagraphs"   # bloklarin derleme modu (5w: torba K 1024 -2,9 ms/adim; kullanici, 7 Ekim)
CLIP_IN_OPTIMIZER = True       # clip katsayisi optimizer'a (grad yerinde carpilmaz; belge 94 s11.1), destekleyen optimizer'da
READING_PROMPTS = os.path.join(HERE, "reading_prompts.json")
READING_LIMITS = dict(max_sentences=80, max_tokens=128)     # belge 21 (story_generation varsayilanlari)
SAMPLE_SEED = 0             # sample cozme tohumu (V1 generate_baseline ile ayni)
IDENTITY = ("model", "d", "layers", "heads", "lr", "seed", "longest", "row_len", "batch_rows", "train_stream_sha256",
            "optimizer", "global_layers", "summaries_last", "glob_kv_heads", "carry_summaries", "carry_group", "vocab_rows",
            "attn_gate", "ngram_embed", "ngram_layers", "ngram_sparse", "mtp", "g_latent_rank", "g_raw_sentences",
            "g_latent_rope", "g_latent_score", "g_latent_tiers", "g_latent_budget", "g_latent_price", "g_latent_explore")
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
    sinirlarin hesaplandigi akisinki (<split>_boundaries.json) olmali: kopya = sinirlarin akisi, bayt bayt.  Yer yetmezse
    kopyadan once DURUR (10BT akisi 19,9 GB)."""
    os.makedirs(os.path.join(local, "gpt2"), exist_ok=True)
    pairs = [(os.path.join(stream_root, "gpt2", sp + ".npy"), os.path.join(local, "gpt2", sp + ".npy")) for sp in splits]
    need = sum(os.path.getsize(s) for s, d in pairs if not (os.path.exists(d) and os.path.getsize(d) == os.path.getsize(s)))
    free = shutil.disk_usage(os.path.join(local, "gpt2")).free
    if need > free:
        sys.exit("DUR: --local %s: kopya icin %.1f GB gerekli, %.1f GB bos" % (local, need / 1e9, free / 1e9))
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


def _schedule(train, data_dir, seed, epochs, steps, carry_group=0):
    """-> (epok planlari [(row_offsets, row_stories)], epok basina adim, row_len, toplam adim).  steps verilirse toplam o
    (gerektigi kadar epok), yoksa epochs epok.  carry_group: her epok data.carry_pack_plan (dosyasiz)."""
    f = np.load(os.path.join(data_dir, "train_pack_plan_e1.npz"))
    row_len = int(f["row_len"])
    lengths = train.lengths()
    plans, per = [], []
    while True:
        e = len(plans) + 1
        if carry_group:
            plans.append(D.carry_pack_plan(lengths, train.continues, row_len, seed, e, carry_group, BATCH_ROWS))
        elif e == 1 and int(f["seed"]) == seed and int(f["epoch"]) == 1:
            plans.append((f["row_offsets"], f["row_stories"]))
        else:
            plans.append(D.pack_plan(lengths, row_len, seed, e))
        per.append(-(-(len(plans[-1][0]) - 1) // BATCH_ROWS))
        if (steps is None and e >= epochs) or (steps is not None and sum(per) >= steps):
            return plans, per, row_len, steps if steps is not None else sum(per)


def _archived(idt):
    """Okuma ve surdurme: kimlik bu kodla kurulamiyorsa ileti, yoksa None.  Transformer her zaman yuklenir (torbali haric).
    Model Z: summaries_last 1 degilse (8 Ekim oncesi ya da summaries_last 0), learned_z 0 (formullu) ya da kaldirilan bir
    ozellik varsa durur (belge 77; sekilleri ayni olanlar sessizce yanlis yuklenirdi)."""
    if not idt.get("bag_k") and (idt.get("model") != "model_z" or (idt.get("summaries_last") == 1 and idt.get(
            "learned_z", 1) == 1 and not any(idt.get(k) for k in ("z_bow_weight", "layer_plan", "z_reads_all", "glob_drop")))):
        return None
    return "kosu bu kodun kaldirdigi bir yolla (Model Z ya da torba) egitildi; yuklenmez -- eski kod git etiketi " \
           "v2-before-cleanup-20261008 (8 Ekim oncesi, summaries_last 0, torba, z_reads_all, glob_drop); " \
           "z_bow / layer_plan: commit 7bec0ae; formullu z: " \
           "v2-before-formula-cleanup-20261007; 6 Ekim oncesi: v2-before-cleanup-20261006 (git worktree add <klasor> <etiket>)"


def _global_error(args):
    """--global_layers kurulamiyorsa ileti, yoksa None (args'ta yoksa 0)."""
    gl = int(getattr(args, "global_layers", 0))
    if gl and args.model != "model_z":
        return "--global_layers yalniz model_z (belge 40 s6.2)"
    if not 0 <= gl <= args.layers:
        return "--global_layers %d: 0..%d olmali" % (gl, args.layers)
    kv = int(getattr(args, "glob_kv_heads", 0) or 0)
    if kv and (kv < 0 or args.heads % kv):
        return "--glob_kv_heads %d: heads (%d) boleni olmali" % (kv, args.heads)
    if kv and args.model == "model_z" and not gl:
        return "--glob_kv_heads: model_z'de glob katmani yok"
    return None


def _carry_error(args):
    """carry bayraklari kurulamiyorsa ileti, yoksa None (args'ta yoksa kapali)."""
    cg = getattr(args, "carry_group", 0) or 0
    if (cg or getattr(args, "carry_summaries", 0)) and (args.model != "model_z" or cg < 2):
        return "--carry_summaries / --carry_group: yalniz model_z, grup >= 2"
    return None


def _latent_error(args):
    """--g_latent_rank / --g_raw_sentences kurulamiyorsa ileti, yoksa None (belge 102)."""
    r, k = args.g_latent_rank, args.g_raw_sentences
    if r < 0 or k < 0:
        return "--g_latent_rank / --g_raw_sentences >= 0"
    if k and not r:
        return "--g_raw_sentences yalniz --g_latent_rank ile"
    tiers, bud, pri = args.g_latent_tiers, args.g_latent_budget, args.g_latent_price
    if tiers and (not r or tiers != sorted(set(tiers)) or tiers[0] <= 0 or tiers[-1] != r or not bud or pri
                  or not tiers[0] <= bud <= tiers[-1] or not 0 <= args.g_latent_explore <= 1):
        return "--g_latent_tiers: artan, son = --g_latent_rank; --g_latent_budget (kademe araliginda) gerek; --g_latent_price yok"
    if not tiers and (bud or pri):
        return "--g_latent_budget / --g_latent_price yalniz --g_latent_tiers ile"
    if args.g_latent_score not in (0, 1) or (args.g_latent_score and not r):
        return "--g_latent_score 0 / 1, yalniz --g_latent_rank ile"
    dr, hd = args.g_latent_rope, args.d // args.heads
    if dr and (not r or dr % 2 or not 0 < dr < hd):
        return "--g_latent_rope %d: --g_latent_rank ile, cift, 0 < dr < head boyu %d" % (dr, hd)
    if r and (args.model != "model_z" or not args.global_layers or args.carry_group or args.carry_summaries):
        return "--g_latent_rank: yalniz model_z, global_layers > 0, carry'siz"
    return None


def _mtp_error(args):
    """--mtp kurulamiyorsa ileti, yoksa None (args'ta yoksa 0)."""
    n = getattr(args, "mtp", 0) or 0
    if n < 0:
        return "--mtp %d: 0 ya da pozitif olmali" % n
    if n and args.model != "model_z":
        return "--mtp: yalniz model_z (deneme/mtp)"
    if n and (getattr(args, "carry_group", 0) or getattr(args, "carry_summaries", 0)):
        return "--mtp ile --carry_summaries / --carry_group birlikte kurulmadi (parca siniri hedef zinciri; belge 88c s1.3)"
    return None


def _build(args, dev):
    """-> (model, mask_fn, layout).  Model dosyalari yalniz burada import edilir.  Model Z: ogrenilen z (belge 35 (b));
    global_layers (args'ta yoksa 0): mask_fn (yerel, global) ikilisi.  Maske kurali yalniz burada secilir (egitim, sinav,
    teshis ayni yol)."""
    root = os.path.dirname(HERE)
    if _global_error(args):
        sys.exit("DUR: " + _global_error(args))
    torch.manual_seed(args.seed)
    if args.model == "transformer":
        sys.path.insert(0, os.path.join(root, "transformer"))
        from baseline import BaselineTransformer
        model = BaselineTransformer(args.d, args.layers, args.heads, kv_heads=getattr(args, "glob_kv_heads", 0) or None,
                                    vocab_rows=getattr(args, "vocab_rows", D.VOCAB))
        model, mask_fn, layout = model.to(dev), R.document_mask, "transformer"
    else:
        sys.path.insert(0, os.path.join(root, "model_z"))
        from model import SentenceTransformer
        model = SentenceTransformer(args.d, args.layers, args.heads, global_layers=int(getattr(args, "global_layers", 0)),
                                    glob_kv_heads=getattr(args, "glob_kv_heads", 0) or None,
                                    carry_group=int(getattr(args, "carry_group", 0) or 0) if getattr(args, "carry_summaries", 0)
                                    else 0, vocab_rows=getattr(args, "vocab_rows", D.VOCAB),
                                    attn_gate=int(getattr(args, "attn_gate", 0) or 0),
                                    ngram_rows=getattr(args, "ngram_embed", 0) or 0,
                                    ngram_layers=getattr(args, "ngram_layers", 0) or 0,
                                    ngram_sparse=bool(getattr(args, "ngram_sparse", 0)),
                                    g_latent_rank=getattr(args, "g_latent_rank", 0) or 0,
                                    g_raw_sentences=getattr(args, "g_raw_sentences", 0) or 0,
                                    g_latent_rope=getattr(args, "g_latent_rope", 0) or 0,
                                    g_latent_score=getattr(args, "g_latent_score", 0) or 0,
                                    g_latent_tiers=tuple(getattr(args, "g_latent_tiers", None) or ()),
                                    g_latent_budget=getattr(args, "g_latent_budget", 0) or 0,
                                    g_latent_price=getattr(args, "g_latent_price", 0) or 0,
                                    g_latent_explore=getattr(args, "g_latent_explore", 0) or 0)
        model, mask_fn, layout = model.to(dev), model.mask_fn, "model_z"
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
    decay, no_decay, *table = R.param_groups(model, WEIGHT_DECAY, skip=muon)   # table: bigram tablosu (lr_mult)
    adamw = torch.optim.AdamW(([decay, no_decay] if not muon else [g for g in (decay, no_decay) if g["params"]]) + table,
                              lr=lr, betas=BETAS, **fused)                   # muon: transformer'da decay grubu bos
    cls = R.NorMuon if kind == "normuon" else R.BatchedMuon
    opt = R.MuonAdamW(cls(muon, lr=lr, weight_decay=WEIGHT_DECAY, **MUON), adamw) if muon else adamw
    if getattr(model, "ngram_sparse", False) and model.ngram is not None:   # tablo seyrek satir Adam'da
        opt = R.NgramRowAdam(opt, model, lr, BETAS[1])
    names = {id(p): n for n, p in model.named_parameters()}

    def summary(params):
        pats = {}
        for p in params:
            k = re.sub(r"\.\d+\.", ".*.", names[id(p)])
            pats[k] = pats.get(k, 0) + 1
        return dict(tensors=len(params), params=sum(p.numel() for p in params), names=pats)
    split = dict(muon=summary(muon), adamw_decay=summary(decay["params"]), adamw_no_decay=summary(no_decay["params"]),
                 **({"adamw_ngram": summary(table[0]["params"])} if table else {}))
    return opt, dict(name=kind, split=split, adamw=dict(betas=BETAS, weight_decay=WEIGHT_DECAY, fused=cuda),
                     muon=dict(MUON, weight_decay=WEIGHT_DECAY, **(dict(beta2=R.NORMUON_BETA2, eps=R.NORMUON_EPS,
                                                                     scale="0.2 lr sqrt(mn) / ||O^||_F")
                                                                if kind == "normuon" else {})) if muon else None)


def _attn(batch, mask_fn, cuda):
    """Egitim, sinav (_Exam) ve teshis araclarinin tek maske yolu; mask_fn ikiliyse (global_layers) iki maske."""
    if isinstance(mask_fn, tuple):
        return tuple(_attn(batch, f, cuda) for f in mask_fn)
    return R.block_mask(batch, mask_fn) if cuda else R.dense_mask(batch, mask_fn)


def _step(model, batch, mask_fn, opt, cuda, timer=None, cont=None, mtp=None):
    """Tek egitim adimi -> (kayip, gradyan normu, ek ya da None) cihazda.  timer: iki CUDA olayi, cikis kalemi (yalniz
    ileri).  cont (carry, _cont_mask): devam parcasi hedeflerinin kaybi yalniz olcu (gradyansiz) -> ek {"loss_cont":
    toplam, "n_cont": sayi}.  mtp (--mtp): (ek hedefler (B, T, K), agirlik (K,)) -> geri yayilim MTP kaybindan, donen
    kayip ana CE; ek {"mtp_ce": (K,), "mtp_steps": 1}."""
    with torch.autocast(batch.tokens.device.type, dtype=torch.bfloat16, enabled=cuda):
        h = model._batch_hidden(batch, _attn(batch, mask_fn, cuda))
        if timer is not None:
            timer[0].record()
        extra = None
        if mtp is None:
            loss = main = R.output_loss(h.flatten(0, 1), model.E.weight, batch.target.flatten())
        else:
            loss, main, ce = R.output_loss_mtp(h.flatten(0, 1), model.E.weight, batch.target.flatten(),
                                               mtp[0].flatten(0, 1), mtp[1])
            extra = dict(mtp_ce=ce.detach(), mtp_steps=1)
        if cont is not None:
            with torch.no_grad():
                lc = torch.nn.functional.cross_entropy(model._logits(h[cont]).float(), batch.target[cont],
                                                       reduction="sum")
            extra = dict(extra or {}, loss_cont=lc, n_cont=cont.sum())
        if timer is not None:
            timer[1].record()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if getattr(model, "tier_select", None) is not None:                # esnek r: secici onem etiketine (MSE)
        extra = dict(extra or {}, **model.tier_update((batch.target >= 0).sum()))
    seen = getattr(model, "ngram_seen", None)                          # seyrek bigram yapragi kirpmaya dahil
    params = list(model.parameters()) if seen is None else [*model.parameters(), seen[1]]
    if getattr(model, "tier_select", None) is not None:                # secici kirpma normuna girmez (ana adimi kucultmesin)
        sel = {id(p) for p in model.tier_select.parameters()}
        params = [p for p in params if id(p) not in sel]
    if CLIP_IN_OPTIMIZER and getattr(opt, "grad_coef_ok", False):       # clip_grad_norm_ ile ayni norm ve katsayi;
        gn = torch.nn.utils.get_total_norm([p.grad for p in params if p.grad is not None])   # carpim optimizer'da
        opt.step(grad_coef=torch.clamp(CLIP / (gn + 1e-6), max=1.0))
    else:
        gn = torch.nn.utils.clip_grad_norm_(params, CLIP)
        opt.step()
    return main.detach(), gn.detach(), extra


def _to_device(batch, dev):
    """CPU PackedBatch -> cihaz; CUDA'da sabitlenmis bellekten non_blocking (sonraki batch GPU calisirken hazirlanir).
    Bos alan (real_pos None) None kalir."""
    if dev.type != "cuda":
        return batch
    mv = lambda t: None if t is None else t.pin_memory().to(dev, non_blocking=True)  # noqa: E731
    return D.PackedBatch(**{f.name: mv(getattr(batch, f.name)) for f in dataclasses.fields(batch)})


class _Exam:
    """exam_scores icin model: loss_per_target egitimle ayni maske yolundan (CUDA'da block_mask; belge 24 s5 I)."""

    def __init__(self, model, mask_fn, cuda, last=None):
        self.model, self.mask_fn, self.cuda, self.last = model, mask_fn, cuda, last

    def loss_per_target(self, batch):
        """last (sentence.summaries_last): batch o duzende islenir, sonuclar build_batch duzeninin hedef sirasina geri.
        Eksik son batch build_batch'in bos satir dolgusuyla BATCH_ROWS'a tamamlanir (derleme sekli sabit; dolgu satiri
        hedefsiz, cikti ayni)."""
        n = BATCH_ROWS - len(batch.kind)
        if n > 0:
            fill = dict(tokens=0, kind=D.Kind.PAD, pos=0, doc=-1, sent=-1, target=-100, target_kind=-1, story_ids=-1)
            batch = D.PackedBatch(**{k: torch.cat([getattr(batch, k), getattr(batch, k).new_full(
                (n, getattr(batch, k).shape[1]), v)]) for k, v in fill.items()})
        if self.last is None:
            return self.model.loss_per_target(batch, _attn(batch, self.mask_fn, self.cuda))
        pb, perm = self.last(batch)
        nll, pred, tk = self.model.loss_per_target(pb, _attn(pb, self.mask_fn, self.cuda))
        T = perm.shape[1]
        old = (torch.arange(len(perm), device=perm.device)[:, None] * T + perm)[pb.target >= 0]
        order = torch.argsort(old)
        return nll[order], pred[order], tk[order]


def _cont_mask(batch, gpos, nsent):
    """carry: hedefi gruptaki bir devam parcasinin (gpos > 0) token'i olan konumlar -> bool (B, T): devam parcasinin
    konumlari (BOS haric) ve onceki parcanin son Z'si (hedefi devamin ilk token'i).  gpos, nsent: hikaye basina (tensor)."""
    doc = batch.doc.long()
    sid = batch.story_ids.gather(1, doc.clamp_min(0))
    nxt = torch.where(sid + 1 < len(gpos), gpos[(sid + 1).clamp(max=len(gpos) - 1)], 0)
    last_z = (batch.kind == D.Kind.ZTOK) & (batch.sent.long() == nsent[sid] - 1)
    return (doc >= 0) & (batch.target >= 0) & (((gpos[sid] > 0) & (batch.kind != D.Kind.BOS)) | (last_z & (nxt > 0)))


@torch.no_grad()
def continuation_exam(model, mask_fn, valid, tok, group, memory, m_max, dev, cuda, last, row_len, docs=500):
    """Devam parcasi sinavi (belge 81b s6, 83): valid'de model_z boyu > row_len belgelerden tohum 0 ile en cok docs tanesi,
    make_fineweb parca kuraliyla parcalara, group'luk zincirlere; zincirin her parcasi kendi satirinda (egitim plani gibi),
    memory'de bellekli.  Olcu yalniz devam hedeflerinde (_cont_mask) -> dict(loss, first, mid, buckets (parca ici konum),
    docs, chains, targets, seconds) ya da uzun belge yoksa None."""
    import make_fineweb as MF
    from types import SimpleNamespace
    t0 = time.time()
    pick = np.flatnonzero(valid.lengths() > row_len)
    pick = np.sort(pick[np.random.default_rng(0).permutation(len(pick))[:docs]])
    if not len(pick):
        return None
    sent = np.concatenate([valid.sent[valid.story[i]:valid.story[i + 1]] for i in pick])
    sub = SimpleNamespace(stream=valid.stream, sent=sent, n=len(pick),
                          story=np.r_[0, np.cumsum([valid.story[i + 1] - valid.story[i] for i in pick])])
    story, cont, _ = MF._pieces(sub, row_len, tok)
    pcs = SimpleNamespace(stream=valid.stream, sent=sent, story=story, continues=cont)
    first = np.r_[True, ~cont[:-1]]
    gpos = (np.arange(len(cont)) - np.flatnonzero(first)[np.cumsum(first) - 1]) % group
    starts = np.flatnonzero(gpos == 0)
    chains = [list(range(a, b)) for a, b in zip(starts.tolist(), np.r_[starts[1:], len(cont)].tolist()) if b - a > 1]
    idx = np.arange(len(cont))
    need = int(np.where(gpos > 0, 1 + story[idx] - story[idx - gpos], 0).max())
    M = max(m_max, -(-need // 128) * 128)
    batches, cur = [], []
    for c in chains:
        if len(cur) + len(c) > BATCH_ROWS:
            batches.append(cur)
            cur = []
        cur += [[x] for x in c]
    batches += [cur] if cur else []
    gp_t, ns_t = torch.as_tensor(gpos), torch.as_tensor(np.diff(story))
    edges = (0, 64, 256, 1024, 10 ** 9)
    acc = dict(all=[0.0, 0], first=[0.0, 0], mid=[0.0, 0], **{"%d-%d" % (a, b - 1) if b < 10 ** 9 else "%d+" % a: [0.0, 0]
                                                              for a, b in zip(edges[:-1], edges[1:])})
    for rows in batches:
        rows = rows + [[]] * (BATCH_ROWS - len(rows)) if cuda else rows
        b = D.build_batch(pcs, rows, "model_z", "cpu", row_len, carry=dict(gpos=gpos, memory=memory, m_max=M))
        cm = _cont_mask(b, gp_t, ns_t)
        sid = b.story_ids.gather(1, b.doc.long().clamp_min(0))
        rel = torch.where(gp_t[sid] > 0, torch.arange(row_len)[None] - (0 if memory else 1), 0)   # parca ici konum
        if last is not None:
            b, perm = last(b)
            cm, rel = cm.gather(1, perm), rel.gather(1, perm)
        b = _to_device(b, dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=cuda):
            nll, _, tk = model.loss_per_target(b, _attn(b, mask_fn, cuda))
        keep = (b.target >= 0).cpu()
        c, r, nll, tk = cm[keep], rel[keep], nll.float().cpu(), tk.cpu()
        for name, m in [("all", c), ("first", c & (tk == D.TargetKind.FIRST)), ("mid", c & (tk == D.TargetKind.MID))] + [
                (k, c & (r >= a) & (r < b_)) for k, a, b_ in zip(list(acc)[3:], edges[:-1], edges[1:])]:
            acc[name][0] += float(nll[m].double().sum())
            acc[name][1] += int(m.sum())
    avg = {k: round(v[0] / v[1], 4) if v[1] else None for k, v in acc.items()}
    return dict(loss=avg["all"], first=avg["first"], mid=avg["mid"], buckets={k: avg[k] for k in list(acc)[3:]},
                docs=int(len(pick)), chains=len(chains), targets=acc["all"][1], seconds=round(time.time() - t0, 2))


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
    ap.add_argument("--lr", type=lambda s: s if s == "auto" else float(s), default=None,
                    help="tepe lr (WSD); auto (varsayilan; muon / normuon): LR_REF[0] (LR_REF[1] / d) ^ LR_REF[2]; adamw'de "
                         "sayi sart (olculen 5e-4); --resume'da verilmezse kosunun kimliginden (belge 39, OLCULENLER_z)")
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
    ap.add_argument("--global_layers", type=lambda s: s if s == "auto" else int(s), default=None,
                    help="model_z: son N blok tam causal, gercek konumla (belge 40 s6.2 Deney G); auto (varsayilan): "
                         "round(layers x MODEL_Z_GLOBAL_RATIO), transformer'da 0; 0: G'siz Model Z; --resume'da "
                         "verilmezse kosunun kimliginden")
    ap.add_argument("--glob_kv_heads", type=lambda s: s if s == "auto" else int(s), default=None,
                    help="GQA: tam causal katmanlarda (model_z glob, transformer hepsi) k / v head sayisi, heads'in "
                         "boleni ya da auto (varsayilan: model_z G > 0 ise heads / GLOB_KV_GROUP, aksi 0); kimlikte; "
                         "--resume'da verilmezse kosunun kimliginden")
    ap.add_argument("--carry_summaries", type=int, default=0,
                    help="model_z: parcalar arasi Z bellegi (belge 81b, 83; carry_group varsayilani 4); 0 kapali")
    ap.add_argument("--ngram_embed", type=lambda s: s if s == "auto" else int(s), default=None,
                    help="Model Z bigram embedding tablosu satir sayisi; varsayilan 0 (9 Ekim); auto: model_z 5 x vocab_rows, "
                         "transformer 0; 0 kapali; kimlikte, --resume'da verilmezse kosunun kimliginden (belge 88b, 93)")
    ap.add_argument("--ngram_layers", type=int, default=None,
                    help="bigram yalniz ilk K blok girdisine (0: hepsi); verilmezse n-gram aciksa 1 (belge 93 ek)")
    ap.add_argument("--ngram_sparse", type=int, default=None,
                    help="1: bigram tablosu yalniz okunan satirlarla guncellenir (beta1 0 seyrek Adam; deneme)")
    ap.add_argument("--carry_group", type=int, default=None,
                    help="carry plani: belgenin ardisik en cok G parcasi ayni batch'te (carry_summaries 0 ile: K kontrolu, "
                         "bellek yok); varsayilan carry_summaries ise 4, degilse 0")
    ap.add_argument("--attn_gate", type=lambda s: s if s == "auto" else int(s), default=None,
                    help="model_z: head basina attention cikis kapisi (belge 88a, 90a); 1 girdi n1(x), 2 girdi n1(x)'in "
                         "ilk d // 64 boyutu; auto (varsayilan): model_z ve d >= 64 ise 2, aksi 0; kimlikte; "
                         "--resume'da verilmezse kosunun kimliginden")
    ap.add_argument("--g_latent_rank", type=int, default=None,
                    help="model_z glob: K + 1 cumleden eski token'lar token basina latent'ten, rank r (belge 102; 0 kapali)")
    ap.add_argument("--g_raw_sentences", type=int, default=None,
                    help="--g_latent_rank ile: glob'da ham kalan onceki cumle sayisi K (varsayilan 0: yalniz kendi cumlesi)")
    ap.add_argument("--g_latent_rope", type=int, default=None,
                    help="--g_latent_rank ile: glob'da ayrik RoPE boyu dr (cift, < head boyu) + uretimde absorb (belge 102 s9; 0 kapali)")
    ap.add_argument("--g_latent_tiers", default=None,
                    help="esnek r kademeleri, artan, son = --g_latent_rank (orn. 32,64,128,256; belge 103); bos: sabit r")
    ap.add_argument("--g_latent_budget", type=float, default=None, help="esnek r: ortalama r butcesi (onem sirasina gore paylar)")
    ap.add_argument("--g_latent_price", type=float, default=None, help="esnek r: bu surumde yok (DUR)")
    ap.add_argument("--g_latent_explore", type=float, default=None,
                    help="esnek r: egitimde rastgele kademe olasiligi (varsayilan 0; inis evresinde en cok 0,05)")
    ap.add_argument("--g_latent_score", type=int, default=None,
                    help="--g_latent_rank ile: 1 latent K'ya kv head basina ogrenilen olcek (belge 102 s11; 0 kapali)")
    ap.add_argument("--mtp", type=lambda s: s if s == "auto" else int(s), default=None,
                    help="model_z: ayni-logit MTP ek hedef sayisi N (belge 90c; resmi kod N 2); agirlik recipe.mtp_weights, "
                         "son 1 / (N + 1) payda 0; varsayilan 0 (9 Ekim); auto: model_z 2 (carry'de 0), transformer 0; 0 kapali; "
                         "--resume'da verilmezse kosunun kimliginden")
    ap.add_argument("--fp8", default="none", choices=("none", "tensorwise", "rowwise"),
                    help="MLP (gate_up, down) torchao Float8Linear tarifi; none: bf16 (kimlige girmez, --resume'da "
                         "degistirilebilir; kullanici, 8 Ekim)")
    ap.add_argument("--stop_step", type=int, default=None,
                    help="takvim (WSD, epok plani) degismeden adim N'de dur: checkpoint.pt (surdurulebilir) + agent.pt + "
                         "results.json (finished False, stopped_at N); son sinav ve okuma yok (kullanici, 8 Ekim)")
    ap.add_argument("--checkpoint_minutes", type=float, default=10,
                    help="en cok bu kadar duvar saati kaybi (sinav dahil); surdurmede degistirilebilir")
    args = ap.parse_args(argv)
    args.defaulted = [k for k in INHERIT if getattr(args, k) is None]
    ckpt = os.path.join(args.out, "checkpoint.pt")
    was = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)["args"] \
        if args.resume and os.path.exists(ckpt) else None               # varsayilan degisse de kosu kendi ayariyla surer
    for k in args.defaulted if was is not None else ():
        setattr(args, k, was.get(k, 0 if k in ("glob_kv_heads", "attn_gate", "mtp") or k.startswith(("ngram", "g_"))
                                 else None))
    args.g_latent_rank, args.g_raw_sentences = args.g_latent_rank or 0, args.g_raw_sentences or 0
    args.g_latent_rope, args.g_latent_score = args.g_latent_rope or 0, args.g_latent_score or 0
    t = args.g_latent_tiers
    args.g_latent_tiers = [int(x) for x in t.split(",") if x.strip()] if isinstance(t, str) else list(t or [])
    args.g_latent_budget, args.g_latent_price = float(args.g_latent_budget or 0), float(args.g_latent_price or 0)
    if args.g_latent_explore is None:
        args.g_latent_explore = 0.0
    args.vocab_rows = VOCAB_ROWS if was is None else was.get("vocab_rows", D.VOCAB)   # eski kosu kendi E boyuyla
    if args.optimizer is None:                                           # 8 Ekim: NorMuon varsayilan
        args.optimizer = DEFAULT_OPTIMIZER
    if args.global_layers is None:
        args.global_layers = "auto"
    if args.glob_kv_heads is None:
        args.glob_kv_heads = GLOB_KV_DEFAULT
    if args.lr is None:
        args.lr = "auto"
    if args.attn_gate is None:
        args.attn_gate = ATTN_GATE_DEFAULT
    if args.attn_gate == "auto":                                         # kimlige cozulmus sayi
        args.attn_gate = 2 if args.model == "model_z" and args.d >= 64 else 0
    if args.attn_gate not in (0, 1, 2):
        sys.exit("DUR: --attn_gate %s: 0, 1, 2 ya da auto" % args.attn_gate)
    if args.attn_gate and args.model != "model_z":
        sys.exit("DUR: --attn_gate yalniz model_z")
    if args.attn_gate == 2 and args.d < 64:
        sys.exit("DUR: --attn_gate 2: kapi girdisi d // 64 boyut, d %d < 64 -> 0 boyut; d >= 64 ya da --attn_gate 0 / 1"
                 % args.d)
    if args.ngram_embed is None:
        args.ngram_embed = NGRAM_DEFAULT
    if args.ngram_embed == "auto":                                       # 5 x sozluk satiri (belge 90b), yalniz Model Z
        args.ngram_embed = 5 * args.vocab_rows if args.model == "model_z" else 0
    if args.ngram_layers is None:
        args.ngram_layers = 1 if args.ngram_embed else 0                # n-gram aciksa tek katman (belge 93 ek); 0 = hepsi
    if args.ngram_sparse is None:
        args.ngram_sparse = 0
    args.summaries_last = int(args.model == "model_z")                   # Model Z duzeni (kimlikte isaret)
    if args.carry_group is None:
        args.carry_group = 4 if args.carry_summaries else 0
    auto = []                                                            # kimlige cozulmus sayi girer
    if args.mtp is None:
        args.mtp = MTP_DEFAULT
    if args.mtp == "auto":                                               # carry ile MTP kurulmadi (_mtp_error): 0
        carry = bool(args.carry_summaries or args.carry_group)
        args.mtp = 2 if args.model == "model_z" and not carry else 0
        if args.model == "model_z":
            auto.append("mtp %d%s" % (args.mtp, " (carry: MTP ile birlikte kurulmadi, auto 0)" if carry else ""))
    if args.global_layers == "auto":
        args.global_layers = round(args.layers * MODEL_Z_GLOBAL_RATIO) if args.model == "model_z" else 0
        if args.model == "model_z":
            auto.append("global_layers %d (round(%d x %.4g))" % (args.global_layers, args.layers, MODEL_Z_GLOBAL_RATIO))
    if args.glob_kv_heads == "auto" and (args.model != "model_z" or not args.global_layers):
        args.glob_kv_heads = 0                                           # transformer / G'siz Model Z: GQA yok
    if args.glob_kv_heads == "auto":
        if args.heads % GLOB_KV_GROUP:
            sys.exit("DUR: --glob_kv_heads auto: heads %d, %d'e bolunmuyor; sayi ver" % (args.heads, GLOB_KV_GROUP))
        args.glob_kv_heads = args.heads // GLOB_KV_GROUP
        auto.append("glob_kv_heads %d (%d / %d)" % (args.glob_kv_heads, args.heads, GLOB_KV_GROUP))
    if args.lr == "auto":
        if args.optimizer == "adamw":
            sys.exit("DUR: --lr auto (varsayilan) yalniz muon / normuon (LR_REF Muon'da olculdu); adamw'de --lr sayi ver")
        args.lr = LR_REF[0] * (LR_REF[1] / args.d) ** LR_REF[2]
        auto.append("lr %.6g (%g x (%d / %d) ^ %g)" % (args.lr, LR_REF[0], LR_REF[1], args.d, LR_REF[2]))
    if auto:
        print("auto: " + ", ".join(auto), flush=True)
    return args


def main(argv=None):
    args = _args(argv)
    t0 = time.time()
    log = lambda msg: print("[%7.1f sn] %s" % (time.time() - t0, msg), flush=True)  # noqa: E731
    ckpt0 = os.path.join(args.out, "checkpoint.pt")
    if args.resume and args.defaulted and os.path.exists(ckpt0):        # _args kimlikten aldi
        log("surdurme: verilmeyen %s kosunun kimliginden: %s" % (args.defaulted, {k: getattr(args, k)
                                                                                 for k in args.defaulted}))
    ng_err = "--ngram_embed: yalniz model_z, satir >= 0" if args.ngram_embed < 0 or (
        args.ngram_embed and args.model != "model_z") else None
    if (args.ngram_layers or args.ngram_sparse) and (not args.ngram_embed or not 0 <= args.ngram_layers <= args.layers
                                                     or args.ngram_sparse not in (0, 1)):
        ng_err = "--ngram_layers / --ngram_sparse: --ngram_embed ile, 0 <= K <= layers, sparse 0 / 1"
    for err in (_global_error(args), _carry_error(args), ng_err, _mtp_error(args), _latent_error(args)):   # veri yuklenmeden
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
    if args.carry_group and train.continues is None:
        sys.exit("DUR: --carry_group: veri parcali degil (train_story_continues.npy yok)")
    plans, per_epoch, row_len, total = _schedule(train, args.data, args.seed, args.epochs, args.steps, args.carry_group)
    carry, m_max = None, 0
    if args.carry_group:                                                 # grup sirasi, bellek boyu (BOS + onceki Z'ler)
        cont_ = np.asarray(train.continues, bool)
        first_ = np.r_[True, ~cont_[:-1]]
        gpos = (np.arange(train.n) - np.flatnonzero(first_)[np.cumsum(first_) - 1]) % args.carry_group
        idx_ = np.arange(train.n)
        m_max = max(128, -(-int(np.where(gpos > 0, 1 + train.story[idx_] - train.story[idx_ - gpos], 0).max()) // 128) * 128)
        carry = dict(gpos=gpos, memory=bool(args.carry_summaries), m_max=m_max)
        gp_t, ns_t = torch.as_tensor(gpos), torch.as_tensor(np.diff(train.story))
    bounds = np.cumsum([0] + per_epoch)
    down = _decay_start(total)
    ep = np.load(os.path.join(args.data, "exam_pack_plan.npz"))
    exam_plan = (ep["row_offsets"], ep["row_stories"])
    story_bytes = np.load(os.path.join(args.stream, "gpt2", "valid_bytes.npy"))
    assert len(story_bytes) == valid.n, "valid_bytes hikaye sayisi valid ile ayni degil"
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(args.stream, "gpt2", "tokenizer.json"))
    model, mask_fn, layout = _build(args, dev)
    if args.g_latent_tiers:                                              # esnek r onem etiketi: -log p(token), train sayimi
        cpath = os.path.join(args.data, "train_token_counts.npy")
        if not os.path.exists(cpath):
            sys.exit("DUR: esnek r icin %s gerek" % cpath)
        c = np.zeros(D.VOCAB)                                            # sayim END'siz (V - 1)
        cnt = np.load(cpath)[:D.VOCAB]
        c[:len(cnt)] = cnt
        c = torch.tensor(c, dtype=torch.float) + 1
        model.tok_surprisal.copy_(-(c / c.sum()).log().to(model.tok_surprisal.device))
    model.row_len = row_len                                              # uretim konum siniri, carry parca boyu
    last = None
    if args.summaries_last:                                              # belge 66: [token'lar | ozetler | dolgu]
        from model import summaries_last as last
        mask_fn = model._masks(True)
    n_fp8 = _fp8(model, args.fp8) if args.fp8 != "none" else 0          # compile ve optimizer'dan once
    if cuda:
        for block in model.blocks:
            block.compile(dynamic=False, mode=COMPILE_MODE)
    opt, opt_info = _optimizer(model, args.optimizer, args.lr, cuda)
    ident = dict(model=args.model, d=args.d, layers=args.layers, heads=args.heads, lr=args.lr, seed=args.seed,
                 longest=train.max_sentence_tokens, row_len=row_len, batch_rows=BATCH_ROWS,
                 train_stream_sha256=train.meta["stream_sha256"],
                 optimizer=args.optimizer, global_layers=args.global_layers,
                 summaries_last=args.summaries_last, glob_kv_heads=args.glob_kv_heads, carry_summaries=args.carry_summaries,
                 carry_group=args.carry_group, vocab_rows=args.vocab_rows, attn_gate=args.attn_gate,
                 ngram_embed=args.ngram_embed, ngram_layers=args.ngram_layers, ngram_sparse=args.ngram_sparse,
                 mtp=args.mtp, g_latent_rank=args.g_latent_rank, g_raw_sentences=args.g_raw_sentences,
                 g_latent_rope=args.g_latent_rope, g_latent_score=args.g_latent_score, g_latent_tiers=args.g_latent_tiers,
                 g_latent_budget=args.g_latent_budget, g_latent_price=args.g_latent_price,
                 g_latent_explore=args.g_latent_explore)
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
        if _archived(was):                                                # eski / kaldirilan yol (belge 77)
            sys.exit("DUR: " + _archived(was))
        was = {"glob_kv_heads": 0, "carry_summaries": 0, "carry_group": 0, "vocab_rows": D.VOCAB, "attn_gate": 0,
               "ngram_embed": 0, "ngram_layers": 0, "ngram_sparse": 0, "mtp": 0, "g_latent_rank": 0, "g_raw_sentences": 0, "g_latent_rope": 0, "g_latent_score": 0,
               "g_latent_tiers": [], "g_latent_budget": 0.0, "g_latent_price": 0.0, "g_latent_explore": 0.0,
               **was}                                                    # sonradan eklenenler
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
    if args.carry_group:
        log("carry: grup %d, bellek %s, M_max %d (devam parcasi %d / %d)" % (
            args.carry_group, bool(args.carry_summaries), m_max, int((gpos > 0).sum()), train.n))
    if args.mtp:
        log("mtp %d: ayni-logit MTP, agirlik %s (adim 0), ek hedefler kapali: adim >= %d; inis basi %d" % (
            args.mtp, R.mtp_weights(0, total, args.mtp), next(s for s in range(total + 1)
                                                              if not any(R.mtp_weights(s, total, args.mtp))), down))
    if args.global_layers:
        log("global_layers %d: son %d blok tam causal (model_z_global_mask), gercek hikaye konumu" % (
            args.global_layers, args.global_layers))
    if cuda:
        import torch._inductor.config as inductor_config
        log("hiz: derleme modu %s (autotune alt surecte %s), attention blok %d, Muon %s" % (
            COMPILE_MODE, getattr(inductor_config, "autotune_in_subproc", None), R.ATTN_BLOCK,
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
        if cuda:                                                         # epok sonu eksik batch: bos (dolgu) satirla tam
            rows += [[]] * (BATCH_ROWS - len(rows))                      # boy, derleme sekli sabit (CPU'da agirlik bit ayni
        b = D.build_batch(train, rows, layout, "cpu", row_len, carry)     # kalsin diye yok: dW toplama sirasi degisiyor)
        mt = D.mtp_targets(b, args.mtp) if args.mtp else None            # hikaye sirasinda, permutasyondan once
        if last is not None:
            b, perm = last(b)
            if mt is not None:
                mt = mt.gather(1, perm[..., None].expand(-1, -1, args.mtp))
        return b, real, _cont_mask(b, gp_t, ns_t) if carry else None, mt  # carry: devam maskesi

    def save(dir_, step):
        R.Checkpoint.save(dir_, model, opt, step, plan_meta, history, ident)

    end = total if args.stop_step is None else args.stop_step         # --stop_step: takvim total'den, dongu end'e
    finishing = start == total and args.stop_step is None              # bitis checkpoint'i var, results yok (belge 89b)
    if finishing:
        log("BITIS: adim %d / %d checkpoint'te; egitim yok, sinav (yoksa) + agent.pt + results.json + okuma" % (total, total))
    elif not start < end <= total:
        sys.exit("DUR: --stop_step %s: adim %d < N <= %d olmali" % (args.stop_step, start, total))
    else:
        history.setdefault("segments", []).append(dict(start=start, fp8=args.fp8, fp8_linears=n_fp8))   # kimlik disi kip
    log("fp8 %s (%d Linear)" % (args.fp8, n_fp8))
    sw, win = R.SpeedWindow(), None
    first_window, epoch_from, epoch_t0, saved_at = True, start, time.time(), time.time()
    ckpt_seconds = 60 * args.checkpoint_minutes                      # kimlige girmez (kullanici: buyuk kosuda 30 dk)
    nxt = cpu_batch(start) if start < total else None
    for step in range(start, end):
        if win is None:
            sw.start(step)
            win = dict(step0=step, loss=torch.zeros((), device=dev), gn=torch.zeros((), device=dev), tokens=0, extra={})
            if cuda:
                torch.cuda.reset_peak_memory_stats()
        lr = R.wsd_lr(step, total, args.lr, decay=DECAY)
        for g in opt.param_groups:
            g["lr"] = lr * g.get("lr_mult", 1.0)                         # bigram tablosu grubu: lr_mult
        batch, real, cont, mt = nxt
        timer = tuple(torch.cuda.Event(enable_timing=True) for _ in range(2)) if cuda else None
        if cont is not None and cuda:
            cont = cont.pin_memory().to(dev, non_blocking=True)
        w_mtp = R.mtp_weights(step, total, args.mtp) if args.mtp else None
        mtp = None
        if w_mtp and any(w_mtp):                                         # son evre: eski yol (saf NTP)
            mtp = (mt, torch.tensor(w_mtp))
            mtp = tuple(t.pin_memory().to(dev, non_blocking=True) for t in mtp) if cuda else mtp
        if getattr(model, "tier_select", None) is not None:            # kesif: inis evresinde tier_p_late
            model.tier_p = args.g_latent_explore if step < down else min(args.g_latent_explore, model.tier_p_late)
        loss, gn, extra = _step(model, _to_device(batch, dev), mask_fn, opt, cuda, timer, cont, mtp)
        nxt = cpu_batch(step + 1) if step + 1 < total else None          # GPU calisirken hazirlanir
        win["loss"] += loss
        win["gn"] += gn
        win["tokens"] += real
        for k, v in (extra or {}).items():                               # toplamlar cihazda (senkron yok)
            win["extra"][k] = win["extra"].get(k, 0) + v
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
        if carry:
            n_c = int(win["extra"]["n_cont"])
            rec["loss_cont"] = round(float(win["extra"]["loss_cont"]) / n_c, 4) if n_c else None
        if args.mtp:                                                     # ek hedef CE (k = 1..N), pencere ortalamasi
            n_m = win["extra"].get("mtp_steps", 0)
            rec["loss_mtp"] = [round(v / n_m, 4) for v in win["extra"]["mtp_ce"].tolist()] if n_m else None
            rec["mtp_w"] = [round(v, 6) for v in w_mtp]
        n_t = win["extra"].get("tier_steps", 0)
        if n_t:                                                          # esnek r: pencere ortalamalari (belge 103 s4)
            rec.update(tier_hist=[round(v / n_t, 4) for v in win["extra"]["tier_hist"].tolist()],
                       tier_mean=round(float(win["extra"]["tier_mean"]) / n_t, 2),
                       tier_hit=round(float(win["extra"]["tier_hit"]) / n_t, 4),
                       tier_ce=round(float(win["extra"]["tier_ce"]) / n_t, 4))
            log("  kademe: hist %s  ortalama r %.1f  onem isabeti (ust kademe) %.3f  secici mse %.4f" % (
                rec["tier_hist"], rec["tier_mean"], rec["tier_hit"], rec["tier_ce"]))
        history["log"].append(rec)
        log("adim %d / %d (epok %d)  lr %.3g  kayip %.4f  grad %.3f | pencere %d adim %.1f sn  %.1f ms/adim  %.0f tok/sn%s%s"
            % (done, total, epoch, lr, rec["loss"], rec["grad_norm"], k, s["seconds"], rec["ms_per_step"],
               s["tokens_per_sec"], "  tepe %.1f GB (ayrilan %.1f)  cikis ileri %.1f ms" % (
                   rec["peak_gb"], rec["peak_reserved_gb"], rec["out_fwd_ms"]) if cuda else "",
               ("  | devam %s" % rec["loss_cont"] if carry else "")
               + ("  | mtp w %s ce %s" % (rec["mtp_w"], rec["loss_mtp"]) if args.mtp else "")
               + ("  (ilk pencere: derleme dahil)" if first_window else "")))
        assert np.isfinite(rec["loss"]), "kayip sonlu degil; checkpoint yazilmadi"
        first_window, win = False, None
        if done == down:
            save(decay_dir, done)                                        # uzatma buradan (belge 20 s4)
        if epoch_end or done == total:
            history["epochs"].append(dict(epoch=epoch, step=done, wall_seconds=round(time.time() - epoch_t0, 1),
                                          resumed=bool(epoch_from > bounds[epoch - 1])))
            ex = _exam(model, mask_fn, valid, exam_plan, story_bytes, layout, dev, cuda, last)
            if carry:
                ex["continuation"] = continuation_exam(model, mask_fn, valid, tok, args.carry_group, carry["memory"], m_max,
                                                       dev, cuda, last, row_len)
                log("DEVAM SINAVI adim %d: %s" % (done, ex["continuation"]))
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
        if carry:
            ex["continuation"] = continuation_exam(model, mask_fn, valid, tok, args.carry_group, carry["memory"], m_max,
                                                   dev, cuda, last, row_len)
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
    t = time.time()
    own = os.path.join(args.data, "reading_prompts.json")              # veri klasorunun istemleri (FineWeb, belge 48)
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
