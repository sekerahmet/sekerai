"""Strata-x Voice okumasi: acik token turune gore (kullanici, 11 Ekim: "Kapasite yapacağı iş ile ölçülür kanal ise ihtiyaç duyduğu bilgi ile").

Hedef basina kayip: kosu (--run), ayni Writer'li Voice'suz kiyas (--ref), transformer (--tf).  Turler: hikayede ilk gecis / tekrar,
transformer belirsizligi H_tf kovasi.  Voice'lu kosuda ek: kapinin ortalamasi sigmoid(direct_gate) ve Voice kapatilinca (kapi 0) kayip artisi.
CPU, sinav alt kumesinin ilk --rows satiri.  Kullanim:
    python model_s/sx_voice_read.py --run <kosu> --ref <kiyas> --tf <transformer> [--rows 6]
"""
import argparse
import ctypes
import os
import sys
from ctypes import wintypes

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in ("deneme2/v2/common", "deneme2/v2/transformer", "deneme2/v2/diag", "model_s"):
    sys.path.insert(0, os.path.join(ROOT, p))
import numpy as np  # noqa: E402
import data as D  # noqa: E402
from generate_readings import load_run  # noqa: E402


def _no_power_throttling():
    if os.name != "nt":
        return False
    class State(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]
    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    s = State(1, 0x1, 0)
    return bool(k.SetProcessInformation(k.GetCurrentProcess(), 4, ctypes.byref(s), ctypes.sizeof(s)))


def sx_nll(m, b, keep):
    h = m.forward_states(b.tokens, b.pos, b.doc)[0]
    v = torch.zeros_like(b.target, dtype=torch.float)
    v[keep] = m._nll(h[keep], b.target[keep])
    return v


def first_seen(tokens, target, doc):
    """target[t] bu belgede tokens[<= t] icinde yoksa True (hikayede ilk gecis)."""
    out = torch.zeros_like(target, dtype=torch.bool)
    for r in range(tokens.shape[0]):
        seen, cur = set(), None
        for t in range(tokens.shape[1]):
            d = int(doc[r, t])
            if d != cur:
                seen, cur = set(), d
            seen.add(int(tokens[r, t]))
            out[r, t] = int(target[r, t]) not in seen
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--data", default=r"G:/Drive'ım/v2/simplestories_gpt2")
    ap.add_argument("--stream", default=r"G:/Drive'ım/simplestories")
    a = ap.parse_args(argv)
    print("EcoQoS kapali:", _no_power_throttling(), flush=True)
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    run, _ = load_run(a.run, a.data, "cpu")
    ref, _ = load_run(a.ref, a.data, "cpu")
    tf, _ = load_run(a.tf, a.data, "cpu")
    for m in (run, ref, tf):
        m.eval()
    valid = D.TokenStories(a.stream, a.data, "valid")
    ep = np.load(os.path.join(a.data, "exam_pack_plan.npz"))
    ro, rs = ep["row_offsets"], ep["row_stories"]
    rows = [rs[ro[r]:ro[r + 1]].tolist() for r in range(min(a.rows, len(ro) - 1))]
    b = D.build_batch(valid, rows, "transformer", "cpu")
    keep = b.target >= 0

    voice = getattr(run, "direct_gate", None) is not None
    cap, mode = {}, {"off": False}
    if voice:
        def hook(_, __, out):
            cap["g"] = torch.sigmoid(out.float())[..., 0]
            return torch.full_like(out, -30.0) if mode["off"] else out
        run.direct_gate.register_forward_hook(hook)
    with torch.no_grad():
        L = {"run": sx_nll(run, b, keep), "ref": sx_nll(ref, b, keep)}
        real_down = ref.down.forward
        ref.down.forward = lambda x: ref.g0.expand(*x.shape[:-1], -1).clone()   # kiyasta Thinker kapali: her token g0
        th_off = (sx_nll(ref, b, keep) - L["ref"])[keep]
        ref.down.forward = real_down
        gate = cap["g"][keep] if voice else None
        if voice:
            mode["off"] = True
            L["off"] = sx_nll(run, b, keep)
        tn = torch.zeros_like(b.target, dtype=torch.float)
        tn[keep] = tf.loss_per_target(b)[0]
        hT = tf._batch_hidden(b)[keep]
        H = torch.empty(hT.shape[0])
        for i in range(0, hT.shape[0], 4096):
            lt = torch.log_softmax(tf._logits(hT[i:i + 4096]).float(), -1)
            H[i:i + 4096] = -(lt.exp() * lt).sum(-1)
    first = first_seen(b.tokens, b.target, b.doc)[keep]
    r, f, t = L["run"][keep], L["ref"][keep], tn[keep]
    print("satir %d, hedef %d | kosu %.4f  kiyas %.4f  transformer %.4f" % (len(rows), int(keep.sum()), float(r.mean()), float(f.mean()),
          float(t.mean())))
    print("%-24s %6s %6s | %8s %8s %8s | %6s %8s" % ("tur", "n", "pay", "kiyas-tf", "kosu-tf", "kosu-kiy", "kapi", "kapali-k"))
    cats = [("hepsi", torch.ones_like(first)), ("ilk gecis", first), ("tekrar", ~first),
            ("H_tf < 0,5 (kalip)", H < 0.5), ("H_tf 0,5-2", (H >= 0.5) & (H < 2)), ("H_tf > 2 (yeni / secim)", H >= 2),
            ("ilk gecis & H > 2", first & (H >= 2)), ("tekrar & H < 0,5", ~first & (H < 0.5)),
            ("Thinker'a bagli (>+0,1)", th_off > 0.1), ("Thinker'siz (|.|<=0,1)", th_off.abs() <= 0.1),
            ("Thinker zararli (<-0,1)", th_off < -0.1)]
    for nm, mm in cats:
        n = int(mm.sum())
        row = "%-24s %6d %6.3f | %+8.4f %+8.4f %+8.4f" % (nm, n, n / len(mm), float((f - t)[mm].mean()), float((r - t)[mm].mean()),
                                                            float((r - f)[mm].mean()))
        if voice:
            row += " | %6.3f %+8.4f" % (float(gate[mm].mean()), float((L["off"][keep] - r)[mm].mean()))
        print(row)
    tot = (f - t).sum()
    print("acik payi (kiyas - tf toplam katkisi): ilk gecis %.2f, H > 2 %.2f | Thinker'a bagli %.2f, Thinker'siz %.2f, zararli %.2f" % (
        float((f - t)[first].sum() / tot), float((f - t)[H >= 2].sum() / tot), float((f - t)[th_off > 0.1].sum() / tot),
        float((f - t)[th_off.abs() <= 0.1].sum() / tot), float((f - t)[th_off < -0.1].sum() / tot)))
    print("kiyasta Thinker kapali: ortalama %+.4f" % float(th_off.mean()))


if __name__ == "__main__":
    main()
