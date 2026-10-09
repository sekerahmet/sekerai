"""tests_model_beta -- model_beta/beta.py (V4Small: v4_small taban, model_beta) testleri (CPU; belge 99).  Gruplar:
mask (maske + grup = bagimsiz dongu, gercek build_batch), model (havuzlama, sizinti, flex = dense, uretim satiri),
train (train.py uctan uca, DUR, surdurme, load_run + gap_v2).  Yardimcilar common/tests_v2'den.

    python tests_model_beta.py [--only mask,model,train]
"""
import torch

import os  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(HERE)
for p in (os.path.join(V2, "common"), os.path.join(V2, "diag"), HERE):
    sys.path.insert(0, p)
import tests_v2 as T2  # noqa: E402  (EcoQoS kapatma, gecici klasor, veri kurucu)
import beta as BT  # noqa: E402
import data as D  # noqa: E402

RESULTS = []
SMALL = dict(WINDOW=6, CSA=3, HCA=8)     # kucuk satirda (64) uc tur de girdi uretsin


def check(name, ok, info=""):
    RESULTS.append((name, bool(ok)))
    print("%-4s %s %s" % ("OK" if ok else "HATA", name, info), flush=True)


def _pin(**kw):
    old = {k: getattr(BT, k) for k in kw}
    for k, v in kw.items():
        setattr(BT, k, v)
    return old


def _rows():
    """Gercek veri: sinav plani satirlari (her satirda birden cok hikaye + dolgu) -> (TokenStories, batch)."""
    tp = T2.tokenizer_path()
    if tp is None:
        return None
    root, data, _ = T2._train_root(tp)
    va = D.TokenStories(root, data, "valid")
    f = np.load(os.path.join(data, "exam_pack_plan.npz"))
    ro, rs = f["row_offsets"], f["row_stories"]
    rows = [rs[ro[r]:ro[r + 1]].tolist() for r in range(len(ro) - 1)]
    return va, D.build_batch(va, rows, "transformer", "cpu", 64), rows


def _reference(batch, m, sentence):
    """Bagimsiz dongu: (B, T, 2T) gorunurluk, yuva -> grup token listesi, hikaye ici konum.  Hikaye BOS ile, cumle END
    ile ayrilir (batch.sent / doc kullanilmaz; yalniz kind)."""
    kind = batch.kind.tolist()
    B, T = batch.kind.shape
    vis = np.zeros((B, T, 2 * T), bool)
    slots, pin = [], np.zeros((B, T), int)
    for b in range(B):
        story, sidx, p0, sent = [-1] * T, [None] * T, 0, 0
        s = -1
        for j in range(T):
            if kind[b][j] == D.Kind.BOS:
                s, p0, sent = s + 1, j, 0
                story[j], sidx[j] = s, -1
            elif kind[b][j] != D.Kind.PAD:
                story[j], sidx[j] = s, sent
                sent += kind[b][j] == D.Kind.END
            pin[b, j] = j - p0
        grp = []                                                    # (son sutun, token listesi), son sutun sirasiyla
        if not m:
            pass
        elif not sentence:
            for st in sorted(set(x for x in story if x >= 0)):
                cols = [j for j in range(T) if story[j] == st]
                grp += [(cols[i + m - 1], cols[i:i + m]) for i in range(0, len(cols) - m + 1, m)]
        else:
            for st in sorted(set(x for x in story if x >= 0)):
                for k in sorted(set(sidx[j] for j in range(T) if story[j] == st and sidx[j] >= 0)):
                    cols = [j for j in range(T) if story[j] == st and sidx[j] == k]
                    step = len(cols) if m == BT.SENTENCE else m
                    grp += [(cols[min(i + step, len(cols)) - 1], cols[i:i + step]) for i in range(0, len(cols), step)]
        grp.sort()
        slots.append(grp)
        for q in range(T):
            if story[q] < 0:
                continue
            for j in range(q + 1):
                if story[j] != story[q]:
                    continue
                vis[b, q, j] = (sidx[j] == sidx[q] or sidx[j] == -1) if sentence else q - j < BT.WINDOW
            if m:
                for e, (end, cols) in enumerate(grp):
                    if story[end] == story[q] and end <= q and (not sentence or sidx[end] < sidx[q]):
                        vis[b, q, T + e] = True
    return vis, slots, pin


def t_mask():
    """Maske ve gruplar bagimsiz donguyle: taban (pencere, m'lik grup) ve Beta (cumle, m'lik parca, cumle girdisi);
    gercek build_batch satirlari (cok hikaye + dolgu); batch.pos = hikaye ici konum (RoPE ile yuva konumu ayni kaynaktan)."""
    got = _rows()
    if got is None:
        print("ATLA mask: GPT-2 tokenizer yok", flush=True)
        return
    _, b, _ = got
    old = _pin(**SMALL)
    try:
        real = (b.kind != D.Kind.PAD).numpy()
        for sentence, ms in ((False, (0, BT.CSA, BT.HCA)), (True, (0, BT.CSA, BT.SENTENCE))):
            for m in ms:
                f = BT.make_mask(m, sentence)
                mask = BT.dense(f, b.kind, b.doc, b.sent).numpy()
                vis, slots, pin = _reference(b, m, sentence)
                S = mask.shape[2]
                ok = np.array_equal(mask[real], vis[:, :, :S][real]) and mask.any(-1).all()
                n = sum(len(g) for g in slots)
                if m:
                    gid, slot_end, slot_pos = BT.groups(b.kind, b.doc, b.sent, m, sentence)
                    for r, grp in enumerate(slots):
                        ok &= slot_end[r, :len(grp)].tolist() == [e for e, _ in grp] and bool((slot_end[r, len(grp):] < 0).all())
                        for e, (_, cols) in enumerate(grp):
                            ok &= gid[r, cols].tolist() == [e] * len(cols) and int(slot_pos[r, e]) == int(pin[r, cols[0]])
                        inside = {c for _, cols in grp for c in cols}
                        ok &= all(int(gid[r, j]) == b.kind.shape[1] for j in range(b.kind.shape[1]) if j not in inside)
                check("mask %s m %s: dense maske = bagimsiz dongu (gercek satirlar); yuva / grup / konum ayni; her satirda "
                      "en az bir anahtar" % ("beta" if sentence else "taban", m), ok, "%d girdi" % n)
        check("mask: batch.pos = hikaye ici konum (BOS 0)", np.array_equal(b.pos.numpy()[real], _reference(b, 0, False)[2][real]))
    finally:
        _pin(**old)


def _model(sentence, seed=0, ratios=None):
    torch.manual_seed(seed)
    r = ratios or ((0, BT.CSA, BT.SENTENCE if sentence else BT.HCA))
    return BT.V4Small(32, len(r), 2, sentence=sentence, ratios=r).eval()


def t_model():
    """Compressor = elle grup havuzu; sizinti yok (gelecek token, sonraki cumle, ayni satirdaki baska hikaye); Beta'da
    onceki cumle yalniz sikistirilmis yoldan; flex BlockMask = dense; uretim satiri = build_batch satiri ve logit'i."""
    import recipe as R
    got = _rows()
    if got is None:
        print("ATLA model: GPT-2 tokenizer yok", flush=True)
        return
    va, b, rows = got
    old = _pin(**SMALL)
    try:
        for sentence in (False, True):
            name = "beta" if sentence else "taban"
            m = _model(sentence)
            blk = m.blocks[1]
            x = torch.randn(b.tokens.shape + (32,))
            h = blk.n1(x)
            gid, _, slot_pos = BT.groups(b.kind, b.doc, b.sent, BT.CSA, sentence)
            with torch.no_grad():
                got_c = blk.compressor(h, gid, slot_pos, blk.theta)
                kv, sc = blk.compressor.wkv(h), blk.compressor.wgate(h)
                err = 0.0
                for r in range(b.kind.shape[0]):
                    for e in range(int(gid[r][gid[r] < b.kind.shape[1]].max()) + 1 if (gid[r] < b.kind.shape[1]).any() else 0):
                        cols = (gid[r] == e).nonzero().flatten()
                        w = torch.softmax(sc[r, cols], 0)
                        ref = blk.compressor.norm((w * kv[r, cols]).sum(0))
                        ref = BT.rope(ref[None, None], slot_pos[r:r + 1, e:e + 1], blk.theta)[0, 0]
                        err = max(err, float((ref - got_c[r, e]).abs().max()))
            check("model %s: Compressor = elle softmax grup havuzu + RMSNorm + RoPE (fark %.1e)" % (name, err), err < 1e-5)

            with torch.no_grad():
                h0 = m._batch_hidden(b)
            r0 = 0
            cols = (b.doc[r0] == 0).nonzero().flatten().tolist()
            other = (b.doc[r0] == 1).nonzero().flatten()
            tok = b.tokens.clone()
            tok[r0, other] = (tok[r0, other] + 7) % D.VOCAB                         # ayni satirdaki ikinci hikaye
            last_sent = int(b.sent[r0, cols].max())
            later = [j for j in cols if int(b.sent[r0, j]) == last_sent and b.kind[r0, j] == D.Kind.TOKEN]
            tok[r0, later[-1]] = (tok[r0, later[-1]] + 11) % D.VOCAB                  # ilk hikayenin son token'i
            b2 = D.PackedBatch(**{**b.__dict__, "tokens": tok})
            with torch.no_grad():
                h1 = m._batch_hidden(b2)
            before = [j for j in cols if j < later[-1]]
            check("model %s: sizinti yok: ayni satirdaki sonraki hikaye ve gelecek token degisince onceki konumlar bit ayni; "
                  "degisen konum ve sonrasi farkli" % name,
                  torch.equal(h0[r0, before], h1[r0, before]) and not torch.equal(h0[r0, later[-1]], h1[r0, later[-1]]))
            if sentence:
                first = [j for j in cols if int(b.sent[r0, j]) == 0 and b.kind[r0, j] == D.Kind.TOKEN]
                tok2 = b.tokens.clone()
                tok2[r0, first[0]] = (tok2[r0, first[0]] + 5) % D.VOCAB
                b3 = D.PackedBatch(**{**b.__dict__, "tokens": tok2})
                for blk_ in m.blocks:                                               # sikistirilmis yol kapali
                    if blk_.compressor is not None:
                        blk_.compressor.wkv.weight.data.zero_()
                with torch.no_grad():
                    ha, hb = m._batch_hidden(b), m._batch_hidden(b3)
                s1 = [j for j in cols if int(b.sent[r0, j]) >= 1]
                m2 = _model(sentence)
                with torch.no_grad():
                    hc, hd = m2._batch_hidden(b), m2._batch_hidden(b3)
                check("model beta: onceki cumle yalniz sikistirilmis yoldan: Compressor.wkv 0 iken cumle 0'daki degisiklik "
                      "sonraki cumlelere ulasmaz, aciksa ulasir",
                      torch.equal(ha[r0, s1], hb[r0, s1]) and not torch.equal(hc[r0, s1], hd[r0, s1]))

            m = _model(sentence)
            attn = tuple(R.block_mask(b, f) for f in m.mask_fn)
            with torch.no_grad():
                nd = m.loss_per_target(b)[0]
                nf = m.loss_per_target(b, attn)[0]
            check("model %s: flex BlockMask (CPU) = dense (nll fark %.1e)" % (name, float((nd - nf).abs().max())),
                  float((nd - nf).abs().max()) < 1e-4)

            story = rows[0][0]
            one = D.build_batch(va, [[story]], "transformer", "cpu", 64)
            n = int((one.kind[0] != D.Kind.PAD).sum())
            seq = one.tokens[0, :n].tolist()
            rw = m.row(seq)
            same = all(torch.equal(getattr(rw, k)[0], getattr(one, k)[0, :n]) for k in ("tokens", "kind", "pos", "doc", "sent"))
            with torch.no_grad():
                full = m._logits(m._batch_hidden(one))[0, :n]
                d = max(float((m.next_logits(seq[:i + 1]) - full[i]).abs().max()) for i in range(0, n, 3))
            gen = m.generate([[seq[1:4]]], 2, 4)[0][0]
            check("model %s: uretim satiri (row) = build_batch satiri (kind / pos / doc / sent); next_logits = tam ileri "
                  "(fark %.1e); generate calisir" % (name, d), same and d < 1e-5 and 1 <= len(gen) <= 2)
    finally:
        _pin(**old)


def t_train():
    """train.py uctan uca iki model (kucuk CPU): kayip duser, ilk adim = loss_per_target; v1'de olmayan bayrak DURUR;
    kesilip surdurulen = kesintisiz (bit); load_run sabitleri denetler; gap_v2 --mz Beta calisir (nll = sinav kaybi)."""
    import traceback
    import gap_v2 as G
    import generate_readings as GR
    import train as TR
    tp = T2.tokenizer_path()
    if tp is None:
        print("ATLA train: GPT-2 tokenizer yok", flush=True)
        return
    root, data, prompts = T2._train_root(tp)
    saved = (TR.BATCH_ROWS, TR.LOG_EVERY, TR.READING_PROMPTS, TR.READING_LIMITS)
    TR.BATCH_ROWS, TR.LOG_EVERY = 4, 1
    TR.READING_PROMPTS, TR.READING_LIMITS = prompts, dict(max_sentences=3, max_tokens=4)
    old = _pin(**SMALL)
    base = ["--data", data, "--stream", root, "--device", "cpu", "--d", "32", "--layers", "3", "--heads", "2",
            "--lr", "1e-2", "--checkpoint_minutes", "0", "--optimizer", "adamw"]
    out = lambda name: os.path.join(T2.TMP, "runs_beta", name)  # noqa: E731
    state = lambda o: torch.load(os.path.join(o, "agent.pt"), weights_only=False)["state"]  # noqa: E731
    try:
        runs = {}
        for model in TR.BETA_MODELS:
            cmd = base + ["--model", model, "--steps", "12"]
            a = TR.main(cmd + ["--out", out(model)])
            runs[model] = out(model)
            L = [w["loss"] for w in a["log"]]
            args = TR._args(cmd + ["--out", "x"])
            mm, mask_fn, layout = TR._build(args, torch.device("cpu"))
            st = D.TokenStories(root, data, "train")
            f = np.load(os.path.join(data, "train_pack_plan_e1.npz"))
            ro, rs = f["row_offsets"], f["row_stories"]
            bb = D.build_batch(st, [rs[ro[r]:ro[r + 1]].tolist() for r in range(4)], layout, "cpu", 64)
            with torch.no_grad():
                want = mm.loss_per_target(bb, TR._attn(bb, mask_fn, False))[0].mean().item()
            check("train %s: 12 adim, kayip duser; ilk adim kaybi = loss_per_target (plan e1 satirlari 0-3); kimlikte beta "
                  "sabitleri" % model,
                  np.mean(L[-3:]) < np.mean(L[:3]) - 0.3 and abs(L[0] - want) < 1e-4
                  and a["identity"]["beta"] == mm.config(), "kayip %.3f -> %.3f, ilk %.4f / %.4f" % (
                      np.mean(L[:3]), np.mean(L[-3:]), L[0], want))
        msgs = [T2._exit_msg(TR.main, base + ["--model", "model_beta", "--steps", "2", "--out", out("dur")] + f)
                for f in (["--mtp", "2"], ["--glob_kv_heads", "2"], ["--fp8", "rowwise"])]
        check("train: model_beta + --mtp 2 / --glob_kv_heads 2 / --fp8 DURUR (v1'de yok)",
              all(m is not None for m in msgs) and "model_beta" in msgs[1] and "model_beta" in msgs[2], str(msgs))

        cmd = base + ["--model", "model_beta", "--steps", "8"]
        stopped, _ = T2._cut_and_resume(TR, cmd, out("cut"))
        check("train model_beta: adim 4'te kesilip surdurulen = kesintisiz (agirliklar bit ayni)",
              stopped and _same_as_full(TR, cmd, out("full8"), out("cut"), state))

        mz, idt = GR.load_run(runs["model_beta"], data, torch.device("cpu"))
        ok_load = idt["model"] == "model_beta" and mz.sentence and mz.config() == idt["beta"]
        BT.WINDOW = 7
        try:
            GR.load_run(runs["v4_small"], data, torch.device("cpu"))
            refused = False
        except AssertionError:
            refused = True
        BT.WINDOW = SMALL["WINDOW"]
        check("train: load_run Beta'yi kurar; sabit farkli kodla yukleme DURUR", ok_load and refused)

        tf = out("tf")
        TR.main(base + ["--model", "transformer", "--steps", "6", "--out", tf])
        exam = __import__("json").load(open(os.path.join(runs["model_beta"], "results.json"), encoding="utf-8"))["exam"]["loss"]
        res = G.main(["--data", data, "--stream", root, "--tf", tf, "--mz", runs["model_beta"], "--out", out("gap")])
        check("train: gap_v2 --mz model_beta calisir; nll = egitimin sinav kaybi (%.4f / %.4f)" % (res["toplam"]["mz"], exam),
              abs(res["toplam"]["mz"] - exam) < 1e-4)
    except Exception:  # noqa: BLE001
        check("train", False, traceback.format_exc(limit=5))
    finally:
        _pin(**old)
        (TR.BATCH_ROWS, TR.LOG_EVERY, TR.READING_PROMPTS, TR.READING_LIMITS) = saved


def _same_as_full(TR, cmd, full_out, cut_out, state):
    TR.main(cmd + ["--out", full_out])
    a, b = state(full_out), state(cut_out)
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


TESTS = dict(mask=t_mask, model=t_model, train=t_train)

if __name__ == "__main__":
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else list(TESTS)
    for name in only:
        TESTS[name]()
    bad = [n for n, ok in RESULTS if not ok]
    print("\n%d / %d gecti%s" % (len(RESULTS) - len(bad), len(RESULTS), "" if not bad else " | HATA: " + ", ".join(bad)))
    sys.exit(1 if bad else 0)
