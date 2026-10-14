"""Değerlendirme: doğrulama kaybı, perplexity, bits-per-byte, sonraki-token doğruluğu; örnek üretim."""
import math

import torch

from rgn.data.registry import EOS_ID

STORY_PROMPTS = ("Once upon a time", "The little dog", "One day, a girl named", "In a small village,")
BANDS = ((0, 64), (64, 256), (256, 512), (512, 1 << 30))      # pencere içi konum bantları (doğruluk)


@torch.no_grad()
def evaluate(model, windows, batch, device, ctx, bytes_per_token=0.0, chunk=8192):
    """windows (n, T + 1) -> {"loss", "ppl", "bpb", "acc", "acc_bands", "tokens"}.  Logit'ler parça parça hesaplanır.
    acc_bands: pencere içi konum bandına göre sonraki-token doğruluğu (bağlam büyüdükçe kazanç)."""
    was_training = model.training
    model.eval()
    W = model.lm_head.weight
    T = windows.shape[1] - 1
    bands = [(lo, min(hi, T)) for lo, hi in BANDS if lo < T]
    band_ok = [0] * len(bands)
    total, correct, count = 0.0, 0, 0
    for i in range(0, len(windows), batch):
        w = windows[i:i + batch].to(device, non_blocking=True)
        x, y = w[:, :-1], w[:, 1:].reshape(-1)
        with ctx:
            h = model.hidden(x)
        h = h.reshape(-1, h.shape[-1]).float()
        hit = torch.empty_like(y, dtype=torch.bool)
        for j in range(0, y.numel(), chunk):
            logits = h[j:j + chunk] @ W.float().T
            t = y[j:j + chunk]
            total += torch.nn.functional.cross_entropy(logits, t, reduction="sum").item()
            hit[j:j + chunk] = logits.argmax(-1) == t
        hit = hit.view(-1, T)
        correct += hit.sum().item()
        for b, (lo, hi) in enumerate(bands):
            band_ok[b] += hit[:, lo:hi].sum().item()
        count += y.numel()
    model.train(was_training)
    loss = total / max(count, 1)
    n_win = count // T
    acc_bands = {"%d-%d" % (lo, hi): band_ok[b] / max(1, n_win * (hi - lo)) for b, (lo, hi) in enumerate(bands)}
    return {"loss": loss, "ppl": math.exp(min(loss, 50)), "acc": correct / max(count, 1), "acc_bands": acc_bands,
            "tokens": count, "bpb": loss / math.log(2) / bytes_per_token if bytes_per_token else None}


def distinct4(ids):
    """Üretilen token'larda farklı 4-gram oranı (1: hiç tekrar yok) ve döngü (bir 4-gram >= 3 kez)."""
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    if not grams:
        return 1.0, False
    from collections import Counter
    c = Counter(grams)
    return len(c) / len(grams), max(c.values()) >= 3


@torch.no_grad()
def probes(model, device, max_new_tokens=48, prompts=STORY_PROMPTS[:2]):
    """Değerlendirmede nitel kontrol: kısa greedy devam + farklı-4 + döngü.  tiktoken yoksa boş liste."""
    enc = _tokenizer()
    if enc is None or max_new_tokens <= 0:
        return []
    was_training = model.training
    model.eval()
    out = []
    for p in prompts:
        ids = enc.encode_ordinary(p)
        gen = model.generate(torch.tensor([ids], device=device), max_new_tokens, eos_id=EOS_ID)[0].tolist()[len(ids):]
        d4, loop = distinct4(gen)
        out.append({"prompt": p, "text": p + enc.decode([t for t in gen if t < enc.n_vocab]), "distinct4": d4,
                    "loop": loop})
    model.train(was_training)
    return out


def _tokenizer():  # noqa: E302
    try:
        import tiktoken
        return tiktoken.get_encoding("gpt2")
    except ImportError:
        return None


@torch.no_grad()
def samples(model, device, max_new_tokens=64, prompts=STORY_PROMPTS):
    """Greedy (argmax) örnek üretim; tiktoken yoksa boş liste."""
    enc = _tokenizer()
    if enc is None or max_new_tokens <= 0:
        return []
    was_training = model.training
    model.eval()
    out = []
    for p in prompts:
        idx = torch.tensor([enc.encode_ordinary(p)], device=device)
        gen = model.generate(idx, max_new_tokens, eos_id=EOS_ID)[0].tolist()
        out.append({"prompt": p, "text": enc.decode([t for t in gen if t < enc.n_vocab])})
    model.train(was_training)
    return out
