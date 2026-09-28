# -*- coding: utf-8 -*-
"""bench_speed_20 -- hiz olcumu: mimari ve sonuc degismeden kosular ne kadar hizlanir.  OLCUMDUR (kural 0): kullanici
onayiyla kosulur; egitim DEGIL, sabit kucuk is (rastgele ya da kayitli agirlik, birkac adim).

    generation(model, prompts, n)       tam yeniden hesap (eski generate) ve artimli (AttentionCache): sure, ayni token mi
    step_time(model_kw, n, batches, ...) egitim adimi (train_seq'in adimi) ms: taban ve oneriler
                                         prefetch    sonraki batch arka planda, pinned bellek + non_blocking (bit duzeyinde ayni)
                                         valid_rows  skorlar yalniz sayilan konumlarda (float yuvarlamasina kadar ayni)
                                         crop        batch en uzun hikayesine kirpilir (float yuvarlamasina kadar ayni)
                                         rope_off    RoPE tablosuz, her cagri yeniden (27 Eylul kodu; tablonun etkisi)
    loss_valid_rows, crop               onerilerin kendisi (adlar ONERI; kullanici onayi olmadan modele girmez)
    run / start                         hepsi sirayla, satirlar LINES'ta ve out/bench_<zaman>.txt'de; start arka planda

Colab hucresi (colab_tinystories.ipynb, 0 HAZIRLIK'tan sonra; kod bu dali icermeli):

    # B HIZ OLCUMU (bench_speed_20)  |  GPU  |  tekrar: GUVENLI
    # OLCUMDUR (kural 0): yalniz kullanici onayiyla.  Egitim DEGIL: sabit kucuk is, kosu klasorlerine dokunmaz; yalniz
    # ROOT/bench_speed/bench_<zaman>.txt yazar.  Arka planda (kural 8); izleme: for _l in BS.LINES: print(_l)
    # --- GPU KAPISI (CLAUDE.md kural 2); esik hesaptan: 64 x 511 x 8.004 fp32 skor ~1 GB, ileri + geri ~4 kopya
    import torch
    assert torch.cuda.is_available(), 'GPU YOK -- Runtime > Change runtime type'
    _bos = torch.cuda.mem_get_info()[0] / 1e9
    assert _bos > 6.0, "GPU'da sadece %.1f GB bos" % _bos
    print('GPU kapisi GECTI: %s  bos %.1f GB' % (torch.cuda.get_device_name(0), _bos))
    if 'colab_tinystories' in sys.modules:
        _live = [n for n, r in sys.modules['colab_tinystories'].RUNS.items() if r['thread'].is_alive()]
        assert not _live, 'egitim kosuyor %s: olcum GPU paylasir, iki sonuc da bozulur' % _live
    if SRC + '/train_math' not in sys.path:
        sys.path.insert(0, SRC + '/train_math')
    import bench_speed_20 as BS, data_math as DM, model_20 as M
    assert hasattr(M, 'AttentionCache'), 'kod bu oneriyi icermiyor (AttentionCache yok)'
    _ck = ROOT + '/tinystories_modelx_s0/checkpoint_t011000.pt'      # egitilmis agirlik: ayni token kontrolu anlamli
    BS.LINES.clear()
    BS_THREAD = BS.start(DATA, ROOT + '/bench_speed', device='cuda', checkpoint=_ck if os.path.exists(_ck) else None,
                         math_data=DM.build('dur'))
    print('arka planda basladi -- izleme: for _l in BS.LINES: print(_l)')
"""
import math
import tempfile
import threading
import time

import torch
import torch.nn.functional as F

import model_20 as M
import train_20 as TR


def loss_valid_rows(model, ids, mask):
    """BlockModel.loss ile ayni kayip; skorlar yalniz hedefi sayilan konumlarda: (B, T, n) skor tensoru kurulmaz."""
    P = model.tokens.points()
    valid = mask[:, 1:]
    h = model.hidden(ids[:, :-1])[-1][valid]
    if model.layer_norm:
        logits = model.norm_final(h) @ P.T
    else:
        logits = model.scale * (h if model.stream_norm else F.normalize(h, dim=-1)) @ P.T
    nll = F.cross_entropy(logits, ids[:, 1:][valid])
    return nll + model.tokens.anchor_loss(), nll


def crop(ids, mask):
    """Batch'i sayilan son konuma kadar kirpar (CPU'da, kopyadan once; sagdaki dolgu nedensel attention'da oncekini
    etkilemez).  mask sayilan hedeflerin girdisi degil konumu: son True sutunu dahil."""
    T = int(mask.any(0).nonzero().max()) + 1
    return ids[:, :T], mask[:, :T]


def apply_rope_old(x, positions=None):
    """RoPE'nin tablosuz hali (27 Eylul kodu): her cagri cos/sin'i yeniden hesaplar.  rope_off olcumu icin."""
    assert positions is None
    T, dh = x.shape[-2], x.shape[-1]
    freq = 10000.0 ** (-torch.arange(0, dh, 2, device=x.device, dtype=x.dtype) / dh)
    angle = torch.arange(T, device=x.device, dtype=x.dtype)[:, None] * freq[None, :]
    cos, sin = angle.cos(), angle.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


class old_generation:
    """with old_generation(): TR.generate'i cagiran her sinav (exam_math, exam_tinystories.texts) eski yoldan uretir."""

    def __enter__(self):
        self.saved = TR.generate.__defaults__
        TR.generate.__defaults__ = (False,)

    def __exit__(self, *a):
        TR.generate.__defaults__ = self.saved


def timed(fn, device, repeat=1):
    """-> (saniye, son sonuc)."""
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(repeat):
        out = fn()
    _sync(device)
    return (time.perf_counter() - t0) / repeat, out


@torch.no_grad()
def generation(model, prompts, n, repeat=1):
    """-> dict(old_s, cached_s, same, mismatch): ayni istemler ve n token, eski ve artimli uretimin suresi."""
    device = next(model.parameters()).device
    out = {}
    for name, cached in (("old", False), ("cached", True)):
        TR.generate(model, prompts[:1], 2, cached=cached)            # isinma
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(repeat):
            g = TR.generate(model, prompts, n, cached=cached)
        _sync(device)
        out[name + "_s"] = (time.perf_counter() - t0) / repeat
        out[name] = g
    out["same"] = out["old"] == out["cached"]
    out["mismatch"] = sum(a != b for a, b in zip(out["old"], out["cached"]))
    return out


class Prefetch:
    """batches(step)'i bir adim once arka planda kurar; GPU'da pinned bellek + non_blocking kopya.  Ayni step ayni parca."""

    def __init__(self, batches, device):
        self.batches, self.device, self.cuda = batches, device, str(device).startswith("cuda")
        self.box, self.thread = {}, None

    def _build(self, step):
        ids, mask = self.batches(step)
        self.box[step] = (ids.pin_memory(), mask.pin_memory()) if self.cuda else (ids, mask)

    def get(self, step):
        if self.thread is not None:
            self.thread.join()
        if step not in self.box:
            self._build(step)
        ids, mask = self.box.pop(step)
        self.thread = threading.Thread(target=self._build, args=(step + 1,), daemon=True)
        self.thread.start()
        return ids.to(self.device, non_blocking=True), mask.to(self.device, non_blocking=True)


def step_time(model_kw, n, batches, device, variant="base", steps=20, warmup=3, lr=TR.LR, total_steps=41602, seed=0,
              setting="shared"):
    """train_seq'in adimi (cosine lr, kayip, geri yayilim, clipping, Adam) `steps` kez -> ms/adim.  variant: base,
    prefetch, valid_rows, crop, rope_off ya da '+' ile birlesimleri (ornek 'prefetch+valid_rows+crop')."""
    parts = set(variant.split("+"))
    assert parts <= {"base", "prefetch", "valid_rows", "crop", "rope_off"}, variant
    model = M.BlockModel(n, seed=seed, **({"shared": True} if setting == "shared" else {"shared": False}),
                         **model_kw).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = (lambda i, m: loss_valid_rows(model, i, m)) if "valid_rows" in parts else model.loss
    source = (lambda s: crop(*batches(s))) if "crop" in parts else batches
    fetch = Prefetch(source, device).get if "prefetch" in parts else (lambda s: tuple(t.to(device) for t in source(s)))
    t0 = None
    saved = M.apply_rope
    if "rope_off" in parts:
        M.apply_rope = apply_rope_old
    try:
        for step in range(warmup + steps):
            if step == warmup:
                _sync(device)
                t0 = time.perf_counter()
            for group in opt.param_groups:
                group["lr"] = lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / total_steps)))
            ids, mask = fetch(step)
            total, nll = loss_fn(ids, mask)
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        _sync(device)
    finally:
        M.apply_rope = saved
    return 1000 * (time.perf_counter() - t0) / steps


def profile_step(model_kw, n, batches, device, steps=3, top=12):
    """Taban egitim adiminin profili: en cok sure alan top islem (GPU'da CUDA suresi, CPU'da CPU suresi) -> satirlar."""
    from torch.profiler import ProfilerActivity, profile
    step_time(model_kw, n, batches, device, "base", steps=1, warmup=1)          # isinma (bellek, cuBLAS)
    cuda = str(device).startswith("cuda")
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
    with profile(activities=acts) as prof:
        step_time(model_kw, n, batches, device, "base", steps=steps, warmup=0)
    key = "self_cuda_time_total" if cuda else "self_cpu_time_total"
    return prof.key_averages().table(sort_by=key, row_limit=top).splitlines()


def save_time(model_kw, n, folder):
    """Surdurme paketi boyunda (model + Adam'in iki momenti) torch.save suresi; dosya silinir.  -> (saniye, MB)."""
    import os
    model = M.BlockModel(n, **model_kw)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params)
    for p in params:
        p.grad = torch.zeros_like(p)
    opt.step()                                           # momentler dolsun (paketteki gibi)
    path = os.path.join(folder, "bench_save_%d.pt" % os.getpid())
    t0 = time.perf_counter()
    torch.save(dict(step=0, model=model.state_dict(), optimizer=opt.state_dict()), path)
    s = time.perf_counter() - t0
    mb = os.path.getsize(path) / 1e6
    os.remove(path)
    return s, mb


# ---- Colab: arka planda tek olcum (kural 8); satirlar LINES'ta ve out/bench_<zaman>.txt'de

LINES = []
TS_KW = dict(d=384, turns=2, units=1536, t_max=512, anchor=0.001)        # tinystories_modelx_s0 ile ayni
MATH_KW = dict(d=64, turns=2, units=256, t_max=512)                       # modelx_math_dur_d64_s0 ile ayni
VARIANTS = ("base", "rope_off", "prefetch", "valid_rows", "crop", "prefetch+valid_rows+crop")


def run(data, out, device="cuda", checkpoint=None, ts_kw=TS_KW, batch=64, steps=20, probe=(4, 80), final=(12, 200),
        stories=8, math_data=None, math_kw=MATH_KW, math_batch=256, math_steps=100, exam_limit=20000, log=print):
    """TinyStories (data: data_tinystories.build) ve istenirse toplama (math_data: data_math.build) uzerinde sureler.
    checkpoint: kayitli agirlik (uretimin ayni token kontrolu egitilmis modelde anlamli); yoksa rastgele."""
    import os
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    for sub in ("train_tinystories", "train_math"):
        if os.path.join(here, sub) not in sys.path:
            sys.path.insert(0, os.path.join(here, sub))
    import data_tinystories as DT
    import exam_tinystories as ET
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "bench_%s.txt" % time.strftime("%Y%m%d_%H%M%S"))
    cuda = str(device).startswith("cuda")

    def note(line):
        LINES.append(line)
        log(line)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    note("bench_speed_20  %s  torch %s  model %s  batch %d" % (
        torch.cuda.get_device_name(0) if cuda else "cpu", torch.__version__, ts_kw, batch))
    vocab = data["vocab"]
    eos = vocab.index(DT.EOS_TOKEN)
    model = M.BlockModel(len(vocab), **ts_kw)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"])
    model = model.to(device).eval()
    note("agirlik: %s" % (checkpoint or "rastgele"))

    # bu cihazda: RoPE tablosu tablosuz hesapla BIT DUZEYINDE ayni mi; onerilen kayiplar eski kayipla ne kadar ayni
    g = torch.Generator().manual_seed(0)
    M.ROPE_TABLES.clear()
    rope_same = all(torch.equal(M.apply_rope(x), apply_rope_old(x)) for x in
                    (torch.randn(4, T, ts_kw["d"], generator=g).to(device) for T in (100, 511, 37, 511, 1)))
    ids, mask = DT.batches(data, 8, 0)(0)
    ids, mask = ids.to(device), mask.to(device)
    with torch.no_grad():
        base = float(model.loss(ids, mask)[1])
        rows_nll = float(loss_valid_rows(model, ids, mask)[1])
        crop_nll = float(model.loss(*(t.to(device) for t in crop(ids.cpu(), mask.cpu())))[1])
    note("DENKLIK RoPE tablosu bit duzeyinde %s | nll eski %.7f  valid_rows %.7f  crop %.7f" % (
        rope_same, base, rows_nll, crop_nll))

    prompts = [[eos] + DT.encode(p, vocab) for p in ET.PROMPTS]
    rows = ET.exam_rows(data)
    halves, _ = ET.story_prompts(data, rows[:stories])
    for label, ps, n in (("sinav metinleri %d istem x %d" % probe, prompts[:probe[0]], probe[1]),
                         ("son istemler %d x %d" % final, prompts[:final[0]], final[1]),
                         ("son hikayeler %d x %d" % (stories, final[1]), halves, final[1])):
        n = min(n, data["seq_len"] - max(len(p) for p in ps))
        r = generation(model, ps, n)
        note("URETIM %-28s eski %7.2f sn  artimli %6.2f sn  x%5.1f  ayni token %s (farkli satir %d)" % (
            label, r["old_s"], r["cached_s"], r["old_s"] / max(r["cached_s"], 1e-9), r["same"], r["mismatch"]))

    s, e = timed(lambda: ET.exam(model, data, rows), device)
    note("SINAV exam %d hikaye: %.2f sn  (nll %.4f acc %.4f)" % (len(rows), s, e["nll"], e["accuracy"]))
    probes = prompts[:probe[0]]
    with old_generation():
        s_old, t_old = timed(lambda: ET.texts(model, data, probes, probe[1]), device)
    s_new, t_new = timed(lambda: ET.texts(model, data, probes, probe[1]), device)
    note("SINAV texts %d istem: eski %.2f sn  artimli %.2f sn  ayni metin %s" % (
        len(probes), s_old, s_new, [w["ids"] for w in t_old] == [w["ids"] for w in t_new]))
    del model
    if cuda:
        torch.cuda.empty_cache()

    batches = DT.batches(data, batch, 0)
    for v in VARIANTS:
        note("ADIM tinystories %-26s %8.1f ms/adim" % (v, step_time(ts_kw, len(vocab), batches, device, v, steps=steps)))
        if cuda:
            torch.cuda.empty_cache()

    for line in profile_step(ts_kw, len(vocab), batches, device):
        note("PROFIL " + line)
    if cuda:
        torch.cuda.empty_cache()

    for where, folder in (("Drive (out)", out), ("yerel disk", tempfile.gettempdir())):
        s, mb = save_time(ts_kw, len(vocab), folder)
        note("YEDEK torch.save %.0f MB -> %-12s %.2f sn  (egitim ipliginde, her save_every adimda)" % (mb, where, s))

    if math_data is not None:
        import data_math as DM
        import exam_math as EM
        mb = DM.batches(math_data, math_batch, 0)
        for v in ("base", "rope_off", "prefetch", "valid_rows"):
            ms = step_time(math_kw, DM.N, mb, device, v, steps=math_steps, warmup=10, total_steps=49500)
            note("ADIM toplama %-30s %8.2f ms/adim" % (v, ms))
        mm = M.BlockModel(DM.N, **math_kw).to(device)
        with old_generation():
            s_old, a_old = timed(lambda: EM.ask(mm, math_data["heldout"], device, exam_limit), device)
        s_new, a_new = timed(lambda: EM.ask(mm, math_data["heldout"], device, exam_limit), device)
        note("SINAV toplama ask %d soru: eski %.2f sn  artimli %.2f sn  ayni sonuc %s" % (
            a_new["n"], s_old, s_new, a_old == a_new))
    note("BITTI")
    return path


def start(data, out, **kw):
    """run'i arka planda baslatir, hemen doner (kural 8).  Izleme: bench_speed_20.LINES."""
    def job():
        try:
            run(data, out, log=lambda line: None, **kw)
        except Exception:
            import traceback
            LINES.append("HATA " + traceback.format_exc())
    thread = threading.Thread(target=job, daemon=True)
    thread.start()
    return thread

