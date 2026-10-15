"""Çatışma (getirilenlerin varyansı) ölçümü — eğitimden önce (kullanıcı, 15 Ekim: "ölçümü yap bakalım").

    python scripts/variance_probe.py --runs ss_d512_tf_yg_epok

Dikkatin çıktısı yalnız ortalama: o_i = Σ_j α_ij v_j.  Atılan bilgi, getirilenlerin dağılımı:
    çatışma_i = 1 − |o_i|² / Σ_j α_ij |v_j|²   ∈ [0, 1]
    0 = getirilenlerin hepsi aynı yönde (net, tek bir şey); 1 = birbirini götürüyor (karışım, çatışma)
(Σ_c varyans_c / Σ_c E_α[v_c²] ile aynı.)  Sorular:
    1. çatışma, sonraki-token kaybıyla ilişkili mi?  (Pearson / Spearman, çeyreklere göre kayıp)
    2. ilişki, MLP'nin zaten görebildiği şeylerin ötesinde mi?  Kısmi korelasyon: kayıp ve çatışma, çıktı büyüklüğü |o|
       ve dikkat entropisi üzerine doğrusal regresyonla arındırılır, kalanların korelasyonu.
    3. zamir ve isimlerde daha mı güçlü?
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from rgn.system import setup_process  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attn_anatomy import load, windows  # noqa: E402
from locality_probe import PRON  # noqa: E402


@torch.no_grad()
def run(m, idx):
    """-> token kaybı (B, T-1), katman başına [çatışma, |o|, entropi] (B, T, 3) (head ortalaması)."""
    T = idx.shape[1]
    rope = m.rope(T)
    x = m.embed(idx)
    stats = []
    for blk in m.blocks:
        at = blk.attn
        u = blk.norm1(x)
        a = at.weights(u, rope).float()                                        # (B, H, T, T)
        _, _, v = at.project(u, rope)
        v = v.float()
        if at.Hkv != at.H:
            v = v.repeat_interleave(at.H // at.Hkv, 1)
        o = a @ v                                                               # ortalama
        e2 = a @ (v * v)                                                        # E_α[v²]
        conflict = 1.0 - (o * o).sum(-1) / e2.sum(-1).clamp_min(1e-12)         # (B, H, T)
        ent = -(a.clamp_min(1e-12).log() * a).sum(-1)
        stats.append(torch.stack([conflict.mean(1), o.norm(dim=-1).mean(1), ent.mean(1)], -1))
        out = blk(x, rope)
        x = out[0] if isinstance(out, tuple) else out
    logits = m.lm_head(m.norm_f(x)).float()
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), idx[:, 1:].reshape(-1), reduction="none")
    return loss.view(idx.shape[0], -1), stats


def _rank(a):
    r = np.empty(len(a))
    r[np.argsort(a)] = np.arange(len(a))
    return r


def _resid(y, X):
    X1 = np.column_stack([X, np.ones(len(X))])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return y - X1 @ beta


def main():
    setup_process()
    ap = argparse.ArgumentParser(description="çatışma (varyans) ölçümü")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2)
    a = ap.parse_args()
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx_all = windows(a.windows)
    toks = [enc.decode([t]) for t in idx_all[:, 1:].flatten().tolist()]
    prev = [enc.decode([t]) for t in idx_all[:, :-1].flatten().tolist()]
    is_pron = np.array([t in PRON for t in toks])
    is_name = np.array([t.startswith(" ") and t[1:2].isupper() and p.strip() not in {".", "!", "?", '"', ""}
                        for t, p in zip(toks, prev)])
    for runname in a.runs:
        m, step = load(runname, device)
        losses, stats = [], None
        for i in range(0, len(idx_all), a.batch):
            lo, st = run(m, idx_all[i:i + a.batch].to(device))
            losses.append(lo.cpu())
            st = [s[:, :-1].cpu() for s in st]                                 # i -> i+1 tahminiyle hizalı
            stats = st if stats is None else [torch.cat([p, q]) for p, q in zip(stats, st)]
        L = torch.cat(losses).flatten().numpy()
        kinds = m.cfg.layer_kinds()
        print("\n== %s (adım %s) — ortalama kayıp %.4f  (zamir n %d, isim n %d)" % (runname, step, L.mean(), is_pron.sum(),
                                                                                    is_name.sum()))
        print("  katman  çatışma ort  Pearson  Spearman  kısmi(|o|,entropi)   kayıp: düşük → yüksek çatışma çeyreği"
              "        zamir: çatışma / kısmi   isim: çatışma / kısmi")
        for li, s in enumerate(stats):
            C, O, E = (s[..., k].flatten().numpy() for k in range(3))
            p = np.corrcoef(C, L)[0, 1]
            sp = np.corrcoef(_rank(C), _rank(L))[0, 1]
            X = np.column_stack([O, E])
            pc = np.corrcoef(_resid(C, X), _resid(L, X))[0, 1]
            qs = np.quantile(C, [0.25, 0.5, 0.75])
            b = np.digitize(C, qs)
            ql = "  ".join("%.3f" % L[b == k].mean() for k in range(4))
            grp = []
            for mask in (is_pron, is_name):
                if mask.sum() > 20:
                    pcg = np.corrcoef(_resid(C[mask], X[mask]), _resid(L[mask], X[mask]))[0, 1]
                    grp.append("%.3f / %+.3f" % (C[mask].mean(), pcg))
                else:
                    grp.append("—")
            print("  K%d %s %10.3f %9.3f %9.3f %12.3f        %s     %s   %s" % (li + 1, kinds[li], C.mean(), p, sp, pc, ql,
                                                                            grp[0], grp[1]))
        del m


if __name__ == "__main__":
    main()
