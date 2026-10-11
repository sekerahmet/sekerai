"""sx_profile -- Strata-x hiz profili (kullanici, 11 Ekim: "Bir hız profili yazsan sanki daha iyi analiz edersin"): egitilmis bir
kosunun modelinde, gercek sinav satirlarinda, bilesen bilesen ileri + geri GPU suresi.

    tam adim        forward_states + cikis kaybi + geri (optimizer yok)
    kodlayici       E (+ tok_in) + yerel kodlayici bloklari
    kodlayici basi  enc_norm + yardimci CE (surpriz)
    buyuk cozucu    birimler uzerindeki bloklar (gercek Umax kovasiyla)
    kod cozucu      yerel kod cozucu bloklari + son norm
    cikis           ana CE (carpanli ya da tam)
    kalan           tam adim - toplam: sinir, toplama, maskeler, uzatma (yogun maske kurma dahil)

Bloklar egitimdeki gibi derlenir (dynamic=False, max-autotune-no-cudagraphs).  Birim / token ve kova boyu da yazilir.

    python model_s/sx_profile.py --run <kosu klasoru> --data <veri> [--stream <ham akis>] [--rows 32] [--iters 20]
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
for p in ("common", "diag", "transformer"):
    sys.path.insert(0, os.path.join(ROOT, "deneme2", "v2", p))
import data as D  # noqa: E402
import gap_v2 as G  # noqa: E402
import generate_readings as GR  # noqa: E402


def timed(fn, iters):
    """fn() bir kez ileri + geri; ortanca ms."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--stream", default=None, help="ham akis koku (varsayilan --data; SS: .../simplestories)")
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--mode", default="max-autotune-no-cudagraphs")
    a = ap.parse_args(argv)
    assert torch.cuda.is_available(), "GPU yok"
    dev = torch.device("cuda")
    m, idt = GR.load_run(a.run, a.data, dev)
    m.train()
    for b in m.blocks:
        b.compile(dynamic=False, mode=a.mode)
    st = D.TokenStories(a.stream or a.data, a.data, "valid")
    rows = G.exam_rows(a.data)
    rows = (rows * (a.rows // max(len(rows), 1) + 1))[:a.rows]
    batch = D.build_batch(st, rows, "transformer", dev)
    tgt = batch.target.flatten()
    ac = lambda: torch.autocast("cuda", torch.bfloat16)  # noqa: E731

    def full():
        with ac():
            h = m._batch_hidden(batch)
            loss = m.output_loss(h.flatten(0, 1), tgt)
        loss.backward()
        m.zero_grad(set_to_none=True)
    t_full = timed(full, a.iters)
    # ara tensorler (bir ileri; birim sayisi ve kova)
    with torch.no_grad(), ac():
        h, _, ends, info = m.forward_states(batch.tokens, batch.pos, batch.doc)
    real = (batch.doc >= 0)
    upr = ends.sum(1)
    import strata_x as SX
    Umax = max(1, -(-int(upr.max()) // SX.UNIT_BUCKET)) * SX.UNIT_BUCKET   # modeldeki kova
    B, T = batch.tokens.shape
    n_enc, n_glob = m.n_enc, m.n_glob
    m_loc = m._local_mask(batch.doc)
    dl = m.blocks[0].A.d
    dg = m.blocks[n_enc].A.d
    dt = m.E.weight.shape[1]

    def comp_enc():
        with ac():
            x = m.E(batch.tokens)
            if m.tok_in is not None:
                x = m.tok_in(x)
            for blk in m.blocks[:n_enc]:
                x = blk(x, batch.pos, m_loc)
        x.float().pow(2).mean().backward()
        m.zero_grad(set_to_none=True)
    xe = torch.randn(B, T, dl, device=dev, requires_grad=True)

    def comp_head():
        with ac():
            hn = m.enc_norm(m.tok_out(xe) if m.tok_out is not None else xe)
            ok = tgt >= 0
            loss = m._nll(hn.flatten(0, 1)[ok], tgt[ok]).mean()
        loss.backward()
        m.zero_grad(set_to_none=True)
    zu = torch.randn(B, Umax, dg, device=dev, requires_grad=True)
    upos = torch.arange(Umax, device=dev).expand(B, Umax)
    m_g = m._unit_mask(torch.zeros(B, Umax, dtype=torch.long, device=dev), torch.ones(B, Umax, dtype=torch.bool, device=dev))

    def comp_glob():
        with ac():
            z = zu
            for blk in m.blocks[n_enc:n_enc + n_glob]:
                z = blk(z, upos, m_g)
        z.float().pow(2).mean().backward()
        m.zero_grad(set_to_none=True)
    xd = torch.randn(B, T, dl, device=dev, requires_grad=True)

    def comp_dec():
        with ac():
            x = xd
            for blk in m.blocks[n_enc + n_glob:]:
                x = blk(x, batch.pos, m_loc)
            y = m.norm(m.tok_out(x) if m.tok_out is not None else x)
        y.float().pow(2).mean().backward()
        m.zero_grad(set_to_none=True)
    ho = torch.randn(B * T, dt, device=dev, requires_grad=True)

    def comp_out():
        with ac():
            ok = tgt >= 0
            loss = m._nll(ho[ok], tgt[ok]).mean()
        loss.backward()
        m.zero_grad(set_to_none=True)

    def comp_mask():
        with torch.no_grad():
            m._local_mask(batch.doc)
    parts = [("kodlayici", comp_enc), ("kodlayici basi (surpriz)", comp_head), ("buyuk cozucu (Umax %d)" % Umax, comp_glob),
             ("kod cozucu", comp_dec), ("cikis", comp_out), ("yerel maske kurma (ileri)", comp_mask)]
    res = [(n, timed(f, a.iters)) for n, f in parts]
    print("strata-x profil | %s | %s" % (os.path.basename(os.path.normpath(a.run)), torch.cuda.get_device_name(0)))
    print("satir %d x %d | birim / token %.3f | satir basina birim ort %.0f, en cok %d -> kova %d | yerel d %d, buyuk d %d, token d %d"
          % (B, T, float(ends.sum()) / float(real.sum()), float(upr.float().mean()), int(upr.max()), Umax, dl, dg, dt))
    print("%-30s | %9s | %6s" % ("bilesen (ileri + geri)", "ms", "pay"))
    for n, t in res:
        print("%-30s | %9.2f | %5.1f%%" % (n, t, 100 * t / t_full))
    rest = t_full - sum(t for _, t in res)
    print("%-30s | %9.2f | %5.1f%%" % ("kalan (sinir / toplama / uzatma)", rest, 100 * rest / t_full))
    print("%-30s | %9.2f | 100%%" % ("tam adim", t_full))


if __name__ == "__main__":
    main()
