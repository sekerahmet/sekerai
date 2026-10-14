"""Değerlendirme: doğrulama kaybı, perplexity, bits-per-byte, sonraki-token doğruluğu; örnek üretim."""
import math

import torch

from rgn.data.registry import EOS_ID

STORY_PROMPTS = ("Once upon a time", "The little dog", "One day, a girl named", "In a small village,")


@torch.no_grad()
def evaluate(model, windows, batch, device, ctx, bytes_per_token=0.0, chunk=8192):
    """windows (n, T + 1) -> {"loss", "ppl", "bpb", "acc", "tokens"}.  Logit'ler parça parça hesaplanır."""
    was_training = model.training
    model.eval()
    W = model.lm_head.weight
    total, correct, count = 0.0, 0, 0
    for i in range(0, len(windows), batch):
        w = windows[i:i + batch].to(device, non_blocking=True)
        x, y = w[:, :-1], w[:, 1:].reshape(-1)
        with ctx:
            h = model.hidden(x)
        h = h.reshape(-1, h.shape[-1]).float()
        for j in range(0, y.numel(), chunk):
            logits = h[j:j + chunk] @ W.float().T
            t = y[j:j + chunk]
            total += torch.nn.functional.cross_entropy(logits, t, reduction="sum").item()
            correct += (logits.argmax(-1) == t).sum().item()
        count += y.numel()
    model.train(was_training)
    loss = total / max(count, 1)
    return {"loss": loss, "ppl": math.exp(min(loss, 50)), "acc": correct / max(count, 1), "tokens": count,
            "bpb": loss / math.log(2) / bytes_per_token if bytes_per_token else None}


def _tokenizer():
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
