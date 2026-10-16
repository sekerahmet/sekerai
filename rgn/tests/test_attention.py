"""Dikkat: paylar causal ve satır toplamı 1; yerel pencere Flex = SDPA maskesi (CUDA); GQA."""
import torch

from _util import require_cuda, tiny
from rgn.model.attention import Attention, _flex_attention, _window_block_mask
from rgn.model.position import rope_tables


def test_weights_sum_to_one_and_causal():
    torch.manual_seed(0)
    for window in (0, 5):
        at = Attention(tiny(), window=window)
        x = torch.randn(2, 16, 64)
        cos, sin = rope_tables(16, 16)
        a = at.weights(x, (cos, sin))
        assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-5)
        assert (a.triu(1) == 0).all()


def test_forward_matches_weights():
    """İleri geçiş (SDPA) = açık paylarla hesap (GQA ve yerel pencere dahil)."""
    torch.manual_seed(0)
    for window in (0, 5):
        at = Attention(tiny(), window=window)
        x = torch.randn(2, 16, 64)
        rope = rope_tables(16, 16)
        q, k, v = at.project(x, rope)
        v = v.repeat_interleave(at.H // at.Hkv, dim=1)
        ref = at.w_o((at.weights(x, rope) @ v).transpose(1, 2).reshape(2, 16, 64))
        assert torch.allclose(at(x, rope), ref, atol=1e-5)


def test_window_flex_matches_mask_cuda():
    """Yerel katman: Flex blok maskesi = SDPA N×N maskesi (ileri + geri)."""
    require_cuda()
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (torch.randn(2, 4, 256, 64, device="cuda", generator=g, requires_grad=True) for _ in range(3))
    W = 37
    rel = torch.arange(256, device="cuda")[:, None] - torch.arange(256, device="cuda")[None, :]
    G = torch.randn(2, 4, 256, 64, device="cuda", generator=g)
    outs = []
    for fn in (lambda: _flex_attention()(q, k, v, block_mask=_window_block_mask(256, W, q.device), scale=0.125),
               lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=(rel >= 0) & (rel < W),
                                                                        scale=0.125)):
        o = fn()
        (o * G).sum().backward()
        outs.append([o.detach()] + [t.grad.clone() for t in (q, k, v)])
        for t in (q, k, v):
            t.grad = None
    for a, b in zip(*outs):
        assert torch.allclose(a, b, atol=2e-3, rtol=2e-3), (a - b).abs().max().item()


def test_gqa_sdpa_matches_enable_gqa():
    """gqa_sdpa (maskeli GQA, hızlı çekirdek yolu) = SDPA enable_gqa: Tq > 1 (yerel pencere, hikâye maskesi) ve Tq 1."""
    from rgn.model.attention import gqa_sdpa
    torch.manual_seed(0)
    F = torch.nn.functional
    for H, Hkv, Tq, S in ((8, 2, 12, 12), (8, 8, 12, 12), (8, 4, 1, 20), (6, 1, 1, 9)):
        q, k, v = torch.randn(2, H, Tq, 16), torch.randn(2, Hkv, S, 16), torch.randn(2, Hkv, S, 16)
        rel = torch.arange(Tq)[:, None] + (S - Tq) - torch.arange(S)[None]
        for mask in ((rel >= 0) & (rel < 5), ((rel >= 0) & (rel < 5))[None, None].expand(2, 1, Tq, S)):
            ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=Hkv != H, scale=0.3)
            assert torch.allclose(gqa_sdpa(q, k, v, mask, 0.3), ref, atol=1e-5), (H, Hkv, Tq, mask.shape)
