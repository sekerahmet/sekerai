# B HIZ OLCUMU: veri duzeni  |  GPU  |  tekrar: GUVENLI (egitim DEGIL: Drive'a yazmaz, kosu baslatmaz)
# ONAYLA kosulur (CLAUDE.md kural 0).  Once 0 HAZIRLIK (DATA).  Egitim kosusu canliyken KOSMAZ: ayni GPU, sure bozulur.
# Her duzen: ayni tohumlu taze Model X, ISINMA + OLC adim (ileri + geri + clip + Adam), sure torch.cuda.synchronize ile.
# Duzenler hucrenin ICINDE tanimli (main'deki koda yama gerekmez); oneri yamasi ile ayni hesap.
#   0 bugun      (64, 512), cikis katmani butun konumlarda
#   1 gather     cikis katmani yalniz hedefli konumlarda (h[valid] @ PL^T)
#   2 kirpma     batch en uzun penceresine (8'in kati) kirpilir, cikis butun konumlarda
#   1+2          ikisi
#   3 kova       K=32 esit hedefli kovalama + kirpma + gather (batch'ler DEGISIR: ayri egitim tarifi)
import time
import numpy as np
import torch
import torch.nn.functional as F

ISINMA, OLC = 3, 30
BENCH_BATCH = 64                                         # 1 KOSU ile ayni
BENCH_KW = dict(d=384, turns=2, units=1536, t_max=512)   # 1 KOSU'daki MODEL_KW
MULT, BUCKET = 8, 32

# --- canli kosu kapisi: egitimle ayni GPU'yu paylasmak iki olcumu de bozar
_live = [n for n, r in C.RUNS.items() if r['thread'].is_alive()]
assert not _live, 'egitim kosusu canli: %s -- bitince ya da ayri runtime' % _live

# --- GPU KAPISI (CLAUDE.md kural 2); 0 bugun'un logits'i fp32 64 x 511 x 8.004 = 1,05 GB, ileri + geri ~4 kopya
assert torch.cuda.is_available(), 'GPU YOK -- Runtime > Change runtime type'
_bos = torch.cuda.mem_get_info()[0] / 1e9
assert _bos > 6.0, "GPU'da sadece %.1f GB bos" % _bos
print('GPU kapisi GECTI: %s  bos %.1f GB  | torch %s  tf32 matmul %s' % (
    torch.cuda.get_device_name(0), _bos, torch.__version__, torch.backends.cuda.matmul.allow_tf32))

dev = 'cuda'
lengths = DATA['train_length']


def full_loss(m, ids, mask):                              # 032e664'teki BlockModel.loss
    logits = m.logits(ids[:, :-1])
    valid = mask[:, 1:]
    nll = F.cross_entropy(logits[valid], ids[:, 1:][valid])
    return nll + m.tokens.anchor_loss(), nll


def gather_loss(m, ids, mask):                            # oneri: Model X (stream_norm, layer_norm yok) icin ayni hesap
    valid = mask[:, 1:]
    h = m.hidden(ids[:, :-1])[-1][valid]
    nll = F.cross_entropy(m.scale * h @ m.tokens.points().T, ids[:, 1:][valid])
    return nll + m.tokens.anchor_loss(), nll


def trimmed(batch):                                       # CPU'da, GPU'ya gitmeden once
    def f(step):
        ids, mask = batch(step)
        W = min(-(-int(mask.sum(1).max()) // MULT) * MULT, ids.shape[1])
        return ids[:, :W], mask[:, :W]
    return f


def bucket_batches(seed=0):                               # data_tinystories.bucket_plan ile ayni plan, epok 0
    n = len(lengths)
    per_epoch = n // BENCH_BATCH
    perm = np.random.default_rng([seed, 0]).permutation(n)
    out = []
    for s in range(0, per_epoch * BENCH_BATCH, BUCKET * BENCH_BATCH):
        chunk = perm[s:min(s + BUCKET * BENCH_BATCH, per_epoch * BENCH_BATCH)]
        k = len(chunk) // BENCH_BATCH
        chunk = chunk[np.argsort(lengths[chunk], kind='stable')]
        cum = np.cumsum(lengths[chunk] + 1)
        out += np.split(chunk, np.searchsorted(cum - (lengths[chunk] + 1) / 2, cum[-1] * np.arange(1, k) / k))
    plan = [out[j] for j in np.random.default_rng([seed, 0, 1]).permutation(per_epoch)]

    def f(step):
        rows = plan[step]
        W = min(-(-(int(lengths[rows].max()) + 2) // MULT) * MULT, DATA['seq_len'])
        ids, mask = DT.sequences(DATA, 'train', rows)
        return ids[:, :W], mask[:, :W]
    return f


def fresh():
    m = TR.BlockModel(len(DATA['vocab']), seed=0, **BENCH_KW).to(dev)
    assert m.stream_norm and not m.layer_norm and not m.copy_path, 'gather_loss Model X icin yazildi'
    return m


base = DT.batches(DATA, BENCH_BATCH, 0)
cases = [('0 bugun', base, full_loss), ('1 gather', base, gather_loss), ('2 kirpma', trimmed(base), full_loss),
         ('1+2 gather+kirpma', trimmed(base), gather_loss), ('3 kova K=32', bucket_batches(), gather_loss)]

# --- dogruluk: adim 0 batch'i, ayni model; kayip ve gradyan onceki loss'a gore (GPU, float32)
m = fresh()
with torch.no_grad():
    _g = torch.Generator(device=dev).manual_seed(1)
    for p in m.parameters():
        if p.requires_grad:
            p.add_(0.02 * torch.randn(p.shape, generator=_g, device=dev))
params = [p for p in m.parameters() if p.requires_grad]


def grads(fn, ids, mask):
    m.zero_grad()
    total, _ = fn(m, ids.to(dev), mask.to(dev))
    total.backward()
    return total.detach(), [p.grad.clone() for p in params]


ref = grads(full_loss, *base(0))
again = grads(full_loss, *base(0))
print('belirlenimcilik: bugunku yol ayni batch iki kez -> kayip %s, gradyan %s' % (
    'birebir' if torch.equal(ref[0], again[0]) else 'FARKLI',
    'birebir' if all(torch.equal(a, b) for a, b in zip(ref[1], again[1])) else 'FARKLI (en buyuk %.1e)' % max(
        float((a - b).abs().max()) for a, b in zip(ref[1], again[1]))))
for name, batch, fn in cases[1:4]:
    new = grads(fn, *batch(0))
    rel = max(float((a - b).abs().max()) / float(b.abs().max()) for a, b in zip(new[1], ref[1]))
    print('  %-20s kayip farki %.1e  gradyan en buyuk goreli fark %.1e' % (name, abs(float(new[0] - ref[0])), rel))
del m, params, ref, again

# --- sure
rows_out = []
for name, batch, fn in cases:
    torch.manual_seed(0)
    m = fresh()
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=TR.LR)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    targets = positions = 0
    for s in range(ISINMA + OLC):
        if s == ISINMA:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        ids, mask = batch(s)
        if s >= ISINMA:
            targets += int(mask[:, 1:].sum())
            positions += ids.shape[0] * (ids.shape[1] - 1)
        total, _ = fn(m, ids.to(dev), mask.to(dev))
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, TR.GRAD_CLIP)
        opt.step()
    torch.cuda.synchronize()
    ms = 1000 * (time.perf_counter() - t0) / OLC
    rows_out.append((name, ms, targets / OLC, positions / OLC, torch.cuda.max_memory_allocated() / 1e9))
    del m, opt, params, total
    torch.cuda.empty_cache()

print('\n%-20s %9s %7s %12s %14s %8s' % ('duzen', 'ms/adim', 'hiz', 'hedef/adim', 'blok konum/adim', 'tepe GB'))
for name, ms, t, p, gb in rows_out:
    print('%-20s %9.1f %6.2fx %12.0f %14.0f %8.2f' % (name, ms, rows_out[0][1] / ms, t, p, gb))
print('(hiz = 0 bugun / duzen; kova satirlari degisken, hedef/adim esit -> adim suresi dogrudan kiyaslanir)')
