"""MLP girdi muhasebesi (kullanıcı, 15 Ekim: "MLP neler kullanıyor, hangi oranda kullanıyor, biz MLP'ye hangi oranda ne
vermeliyiz").

    python scripts/mlp_inputs.py --runs ss_d512_tf_yg_epok ss_d512_tf_yg_kapi ss_d512_tf_yg_konum

Akış (residual) parçaların toplamı: x = gömme + Σ dikkat_l + Σ MLP_l.  MLP'nin girdisi norm(x) = g ⊙ x / rms(x) — token
başına tek bölen, toplam yapısı korunur — o yüzden MLP'nin ilk doğrusal katmanının çıktısı (kapı G ve değer U) kaynakların
katkılarının TAM toplamıdır.  Katman başına dört kaynak grubu (+ varsa konum özeti):
    gömme          token'ın kelime gömmesi
    önceki dikkat  daha önceki katmanların dikkat çıktıları
    önceki MLP     daha önceki katmanların MLP çıktıları
    bu dikkat      aynı katmanın dikkat çıktısı
Ölçüler:
    pay    kaynağın ön-aktivasyona izdüşüm payı  <katkı, toplam> / |toplam|²  (işaretli; kaynaklar + bias toplamı = 1)
    etki   kaynak YALNIZ bu MLP'nin girdisinde ortalamasıyla değiştirilince (bilgisi silinir, ölçeği korunur) son kayıp
           ne kadar artıyor — akışın geri kalanı değişmez
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from rgn.system import setup_process  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attn_anatomy import load, windows  # noqa: E402

GROUPS = ("gömme", "önceki dikkat", "önceki MLP", "bu dikkat")


def _loc_extra(blk, loc, dtype):
    return blk.loc_proj((2.0 * loc - 1.0).to(dtype)) if loc is not None else None


@torch.no_grad()
def forward(m, idx, ablate=None):
    """ablate = (katman, grup) -> o katmanın MLP girdisinde grup ortalamasıyla değiştirilir (değişen MLP çıktısı akışa
    girer ve sonraki katmanlara yayılır).  -> (kayıp, katman başına kaynak payları)."""
    T = idx.shape[1]
    rope = m.rope(T)
    x = m.embed(idx)
    parts = {"gömme": x.clone(), "önceki dikkat": torch.zeros_like(x), "önceki MLP": torch.zeros_like(x)}
    recs = []
    for li, blk in enumerate(m.blocks):
        a = blk.attn(blk.norm1(x), rope)
        loc = None
        if blk.loc_proj is not None:
            a, loc = a
        s = x + a
        groups = {"gömme": parts["gömme"], "önceki dikkat": parts["önceki dikkat"], "önceki MLP": parts["önceki MLP"],
                  "bu dikkat": a}
        if ablate is not None and ablate == (li, "konum") and loc is not None:
            loc = loc.mean((0, 1), keepdim=True).expand_as(loc)
        extra = _loc_extra(blk, loc, x.dtype)
        xin = s
        if ablate is not None and ablate[0] == li and ablate[1] != "konum":
            c = groups[ablate[1]]
            xin = s - c + c.mean((0, 1), keepdim=True)
        mo = blk.mlp(blk.norm2(xin), extra)
        if ablate is None:                                      # kaynak payları: ilk doğrusal katmanın ön-aktivasyonu
            n2 = blk.norm2
            inv = torch.rsqrt(s.float().pow(2).mean(-1, keepdim=True) + n2.eps)
            W = blk.mlp.a.weight.float()
            contrib = {g: (c.float() * inv * n2.weight.float()) @ W.T for g, c in groups.items()}
            if extra is not None:                               # konum yalnız kapı yarısına (ilk ff) eklenir
                ex = extra.float()
                contrib["konum"] = torch.cat([ex, torch.zeros_like(ex)], -1) if ex.shape[-1] * 2 == W.shape[0] else ex
            total = sum(contrib.values())
            if blk.mlp.a.bias is not None:
                total = total + blk.mlp.a.bias.float()
            tt = (total * total).sum()
            recs.append({g: ((c * total).sum() / tt).item() for g, c in contrib.items()})
        parts["önceki dikkat"] = parts["önceki dikkat"] + a
        parts["önceki MLP"] = parts["önceki MLP"] + mo
        x = s + mo
    logits = m.lm_head(m.norm_f(x)).float()
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), idx[:, 1:].reshape(-1))
    return loss.item(), recs


def main():
    setup_process()
    ap = argparse.ArgumentParser(description="MLP girdi muhasebesi")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--windows", type=int, default=8)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx_all = windows(a.windows).to(device)
    for run in a.runs:
        m, step = load(run, device)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
        batches = [idx_all[i:i + 4] for i in range(0, len(idx_all), 4)]
        with ctx:
            base = [forward(m, b) for b in batches]
        L0 = sum(x[0] for x in base) / len(base)
        nl = len(m.blocks)
        shares = [{k: sum(b[1][li][k] for b in base) / len(base) for k in base[0][1][li]} for li in range(nl)]
        kinds = m.cfg.layer_kinds()
        print("\n== %s (adım %s) — kayıp %.4f" % (run, step, L0))
        keys = list(GROUPS) + (["konum"] if "konum" in shares[0] else [])
        print("  katman  " + "".join("%22s" % k for k in keys))
        print("          " + "".join("%22s" % "pay  /  etki" for _ in keys))
        for li in range(nl):
            row = []
            for k in keys:
                with ctx:
                    La = sum(forward(m, b, (li, k))[0] for b in batches) / len(batches)
                row.append("%7.1f%%  / %+7.4f" % (100 * shares[li][k], La - L0))
            print("  K%d %s  " % (li + 1, kinds[li]) + "".join("%22s" % r for r in row))
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
