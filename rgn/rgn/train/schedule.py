"""Öğrenme hızı takvimi: adım -> çarpan (her grubun base_lr'si ile çarpılır)."""
import math


def lr_factor(step, total, kind="wsd", warmup=0.01, decay=0.2, min_frac=0.0):
    """wsd: ısınma -> sabit -> son `decay` oranında doğrusal düşüş | cosine: ısınma -> kosinüs | constant: ısınma -> sabit."""
    w = max(1, int(total * warmup))
    if step < w:
        return (step + 1) / w
    if kind == "constant":
        return 1.0
    if kind == "cosine":
        p = (step - w) / max(1, total - w)
        return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    start = int(total * (1 - decay))
    if step < start:
        return 1.0
    p = (step - start) / max(1, total - start)
    return min_frac + (1 - min_frac) * max(0.0, 1 - p)
