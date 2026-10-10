"""gen_speed -- uretim hizi, her model (transformer, Model Z) ve her boy icin standart arac (belge 84, 91, 92, 95'in
bench'leri; kullanici, 10 Ekim: "üretim hız betiği hazırla zaten vardı sanırım onu da standart koda al").  Yalniz olcum;
bf16.  Standart paket madde 6 (2k / 8k, B 1 / 32; yalniz MIMARI degisince).

Kollar (--arm AD="train.py bayraklari", birden cok; ilki oran tabani): bayraksiz varsayilanlar + bayraklar, rastgele agirlik
(tohum 0): mimari kiyasi, ayni boyda Model Z ile transformer yan yana.  Bolumler (--parts):
  decode   StaticCache toplu yol (compile + CUDA graph): baglam x B izgarasi (--grid std: 2k / 8k x B 1 / 32; wide: 1k B
           1..256, 2k B 1..128, 8k / 32k B 1 / 8 / 32); ms / adim, token / sn, KV GB (yerel / glob), HESAP taban (agirlik +
           KV bayti / olculen kopya bant genisligi), karta sigan en buyuk B (HESAP).
  prefill  prefill_rows tek ileri gecis, T 256 / 1k / 2k x B 1 / 32, 8k / 32k B 1.
  e2e      Model Z generate (sinav istemleri, acik son cumle), batch_size 1 ve 32: sure, adim / sn, toplu = tek tek;
           --run verilirse egitilmis agirlikla (load_run), yoksa ilk Model Z kolu.
Cikti <out>/gen_speed.json + tablo; son satir GEN_SPEED_BITTI.

    python gen_speed.py --arm mz="--model model_z --d 1280 --layers 24 --heads 20"
                        --arm tf="--model transformer --d 1280 --layers 24 --heads 20 --glob_kv_heads 5" [--run <kosu>]
                        [--grid std|wide] [--parts decode,prefill,e2e] [--data <veri>] [--out /content/gen_speed]
    CPU kuru kosu (test): --device cpu --dtype fp32 --steps 4
"""
import torch  # noqa: I001

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np

V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p_ in ("common", "diag", "model_z", "transformer"):
    sys.path.insert(0, os.path.join(V2, p_))
import train as T  # noqa: E402
import data as D  # noqa: E402

GRIDS = dict(std={2048: (1, 32), 8192: (1, 32)},
             wide={1024: (1, 8, 32, 64, 128, 256), 2048: (1, 8, 32, 64, 128), 8192: (1, 8, 32), 32768: (1, 8, 32)})
PREFILL = ((256, 1), (256, 32), (1024, 1), (1024, 32), (2048, 1), (2048, 32), (8192, 1), (32768, 1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", action="append", default=[], help='AD="train.py bayraklari" (--model dahil); birden cok')
    ap.add_argument("--run", default="", help="egitilmis Model Z kosusu (agent.pt); yalniz e2e")
    ap.add_argument("--data", default="/content/drive/MyDrive/v2/fineweb_edu_s000")
    ap.add_argument("--stream", default=None, help="gpt2/ koku (varsayilan --data)")
    ap.add_argument("--grid", default="std", choices=sorted(GRIDS))
    ap.add_argument("--parts", default="decode,prefill,e2e")
    ap.add_argument("--steps", type=int, default=96, help="olculen decode adimi")
    ap.add_argument("--e2e_prompts", type=int, default=128)
    ap.add_argument("--e2e_sentences", type=int, default=6)
    ap.add_argument("--budget_s", type=float, default=1800)
    ap.add_argument("--min_free_gb", type=float, default=None, help="GPU tek basina mi (varsayilan toplam bellegin %%85'i)")
    ap.add_argument("--out", default="/content/gen_speed")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"))
    a = ap.parse_args(argv)
    assert a.arm, 'en az bir --arm AD="train.py bayraklari" gerek'
    arms_spec = {}
    for x in a.arm:
        name, _, flags = x.partition("=")
        assert name and "--model" in flags.split() and name not in arms_spec, '--arm AD="--model ...": %r' % x
        arms_spec[name] = flags.split()
    stream = a.stream or a.data
    dev = torch.device(a.device)
    cuda = dev.type == "cuda"
    DT = dict(bf16=torch.bfloat16, fp32=torch.float32)[a.dtype]
    t0 = time.time()
    log = lambda m: print("[gen_speed %6.1f sn] %s" % (time.time() - t0, m), flush=True)  # noqa: E731
    os.makedirs(a.out, exist_ok=True)
    if cuda:
        assert torch.cuda.is_available(), "GPU YOK"                       # kural 5
        free, tot = (x / 1e9 for x in torch.cuda.mem_get_info())
        log("GPU %s, bos %.1f / %.1f GB, torch %s" % (torch.cuda.get_device_name(0), free, tot, torch.__version__))
        need = a.min_free_gb if a.min_free_gb is not None else 0.85 * tot
        if free < need:
            sys.exit("DUR: GPU'da bos bellek %.1f GB < %.0f (olcum tek basina olmali)" % (free, need))
    elif os.name == "nt":
        log("EcoQoS kapali: %s" % T._no_power_throttling())
    head = os.popen('git -C "%s" log -1 --format="%%h %%s"' % V2).read()[:90].strip()
    res = dict(script="gen_speed.py", repo_head=head, torch=torch.__version__, device=torch.cuda.get_device_name(0) if cuda
               else "cpu", dtype=a.dtype, grid=a.grid, arms={}, decode=[], prefill=[], e2e=[])

    def sync():
        if cuda:
            torch.cuda.synchronize()

    def now():
        sync()
        return time.perf_counter()

    def over():
        return time.time() - t0 > a.budget_s

    def bandwidth():
        if not cuda:
            return 1e11
        x = torch.empty(1 << 30, dtype=torch.bfloat16, device=dev)
        y = torch.empty_like(x)
        for _ in range(3):
            y.copy_(x)
        t = now()
        for _ in range(10):
            y.copy_(x)
        bw = 2 * 2 * (1 << 30) * 10 / (now() - t)
        del x, y
        torch.cuda.empty_cache()
        return bw

    BW = bandwidth()
    res["bandwidth_GBs"] = BW / 1e9
    log("depo %s | bant genisligi (kopya) %.0f GB/s" % (head, BW / 1e9))
    valid = D.TokenStories(stream, a.data, "valid")
    ep = np.load(os.path.join(a.data, "exam_pack_plan.npz"))
    pool = np.concatenate([valid.sent[valid.story[s]:valid.story[s + 1]] for s in np.sort(ep["row_stories"].astype(np.int64))])
    lens = pool[:, 1] - pool[:, 0] + 1
    tmp = tempfile.mkdtemp(prefix="gen_speed_")

    def build(flags):
        """Bayraksiz varsayilanlar (train._args) + bayraklar, rastgele agirlik (tohum 0)."""
        args = T._args([*flags, "--out", tmp, "--data", a.data, "--stream", stream])
        m, _, _ = T._build(args, dev)
        if getattr(args, "g_keep_occurrences", 0):                      # deneme/g-window: pencereli G onbellegi
            m.g_window = T.g_window_table(a.data, args.g_keep_occurrences, args.vocab_rows, dev)
        keys = ("model", "d", "layers", "heads", "global_layers", "glob_kv_heads", "attn_gate", "ngram_embed", "g_nope",
                "mlp_ratio")
        return m.eval().to(DT), {k: getattr(args, k, None) for k in keys}

    def prompt_of(target):
        """target konuma sigan en uzun cumle dizisi (BOS + token'lar + Z'ler <= target)."""
        k = int(np.searchsorted(np.cumsum(lens) + 1, target, "right"))
        return [np.asarray(valid.stream[x:y]).astype(np.int64).tolist() for x, y in pool[:k]]

    def prompts_of(target, n):
        out, start = [], 0
        for _ in range(n):
            k = int(np.searchsorted(np.cumsum(lens[start:]) + 1, target, "right"))
            out.append([np.asarray(valid.stream[x:y]).astype(np.int64).tolist() for x, y in pool[start:start + k]])
            start = (start + max(k, 1)) % max(1, len(pool) - 4 * k)
        return out

    def tf_seq(p):
        return [D.EOS_ID] + [t for s in p for t in list(s) + [D.END_ID]]

    def weight_bytes(model):
        """Adim basina okunan agirlik: bloklarin 2-B matrisleri + E (cikis); n-gram tablosundan yalniz B satir okunur."""
        return sum(p.numel() * p.element_size() for n, p in model.named_parameters()
                   if n.startswith("blocks.") and p.dim() == 2) + model.E.weight.numel() * model.E.weight.element_size()

    def is_z(m):
        return hasattr(m, "global_layers")

    def make_cache(model, prompt, B, steps):
        """B satirli StaticCache: tek satir prefill edilir, satirlar kopyalanir (istemler ayni; prefill olculmez)."""
        mod = sys.modules[type(model).__module__]
        if is_z(model):
            c = mod.StaticCache(model, steps + 8, steps + 8, D.MAX_SENTENCE_TOKENS + 1)
            c.prefill_rows([prompt])
            c.state = tuple(t.repeat(B) for t in c.state)                 # sum_len, sen_len, tt, prev (bigram)
            c.n = [list(c.n[0]) for _ in range(B)]
            c.z = c.z.repeat(B)
            if getattr(c, "gw", None) is not None:
                c.gw[1:4] = [c.gw[1].repeat(B, 1), c.gw[2].repeat(B), c.gw[3].repeat(B)]
                c.gp = c.gp * B
        else:
            c = mod.StaticCache(model, steps + 8)
            c.prefill_rows([tf_seq(prompt)])
            c.n, c.t = c.n.repeat(B), c.t * B
        c.K = [k.repeat(B, 1, 1, 1) for k in c.K]
        c.V = [v.repeat(B, 1, 1, 1) for v in c.V]
        c.w, c.live = c.w.repeat(B), c.live.repeat(B)
        c.logits, c.out = c.logits.repeat(B, 1), c.out.repeat(B, 1)
        return c

    def decode(c, z, B, steps):
        """steps adim acgozlu (Model Z: END / EOS ya da cumle siniri -> Z adimi), graph replay; -> ms / adim (8 isinma)."""
        w = c.logits.float().argmax(-1).tolist()
        cur = [0] * B

        def one(w):
            if z:
                zz = [x in (D.END_ID, D.EOS_ID) or n >= D.MAX_SENTENCE_TOKENS for x, n in zip(w, cur)]
                for r in range(B):
                    cur[r] = 0 if zz[r] else cur[r] + 1
                return c.step_rows([D.END_ID if q else x for x, q in zip(w, zz)], zz, [True] * B).float().argmax(-1).tolist()
            return c.step_rows([D.END_ID if x == D.EOS_ID else x for x in w], [True] * B).float().argmax(-1).tolist()
        for _ in range(8):
            w = one(w)
        t = now()
        for _ in range(steps):
            w = one(w)
        return 1000 * (now() - t) / steps

    def kv_split(c):
        """-> (yerel KV bayti, glob KV bayti); transformer'da hepsi glob."""
        glob = getattr(c, "glob", [True] * len(c.K))
        loc = sum(2 * k.numel() * k.element_size() for k, g in zip(c.K, glob) if not g)   # g_window: G = kalici + halka
        return loc, sum(2 * k.numel() * k.element_size() for k in c.K) - loc

    def measure(tag, m, ctx, B, prompt):
        row = dict(arm=tag, ctx=ctx, B=B)
        try:
            c = make_cache(m, prompt, B, a.steps)
            loc, glob = kv_split(c)
            row.update(kv_gb=(loc + glob) / 1e9, kv_local_gb=loc / 1e9, kv_glob_gb=glob / 1e9)
            row["ms"] = decode(c, is_z(m), B, a.steps)
            row["tok_s"] = 1000 * B / row["ms"]
            row["floor_ms"] = 1000 * (weight_bytes(m) + loc + glob) / BW
            if cuda:
                free = torch.cuda.mem_get_info()[0] + loc + glob
                row["max_B"] = int(0.9 * free / ((loc + glob) / B))
            del c
        except Exception as e:  # noqa: BLE001
            row["skip"] = str(e).splitlines()[0][:150]
        if cuda:
            torch.cuda.empty_cache()
        log("%-10s ctx %5d B %3d: %s" % (tag, ctx, B, "ATLA " + row["skip"] if "skip" in row else
            "%.3f ms/adim, %.0f tok/sn, KV %.3f GB (yerel %.3f), taban %.3f ms (%.1fx), max B ~%s" % (
                row["ms"], row["tok_s"], row["kv_gb"], row["kv_local_gb"], row["floor_ms"], row["ms"] / row["floor_ms"],
                row.get("max_B"))))
        return row

    ARMS = []
    for tag, flags in arms_spec.items():
        m, spec = build(flags)
        ARMS.append((tag, m))
        res["arms"][tag] = dict(spec, flags=" ".join(flags), params_M=round(sum(p.numel() for p in m.parameters()) / 1e6, 1),
                                weight_read_MB=round(weight_bytes(m) / 1e6, 1))
    log("kollar: %s" % json.dumps(res["arms"]))
    parts = a.parts.split(",")
    if "decode" in parts:
        for ctx, Bs in (GRIDS[a.grid] if cuda else {512: (1, 2)}).items():
            prompt = prompt_of(ctx - a.steps - 24)                         # istem + decode kademeye sigsin
            for tag, m in ARMS:
                for B in Bs:
                    if over():
                        log("BUTCE: decode %s ctx %d B %d atlandi" % (tag, ctx, B))
                        continue
                    res["decode"].append(measure(tag, m, ctx, B, prompt))

    def timed(fn):
        t = now()
        fn()
        return 1000 * (now() - t)

    if "prefill" in parts:
        for T_, B in PREFILL if cuda else ((256, 1), (256, 3)):
            if over():
                log("BUTCE: prefill T %d B %d atlandi" % (T_, B))
                continue
            ps = prompts_of(T_, B)
            row = dict(T=T_, B=B)
            for tag, m in ARMS:
                mod = sys.modules[type(m).__module__]
                try:
                    for _ in range(2):                                    # ikinci tur olculur (isinma)
                        row[tag + "_ms"] = timed(lambda: mod.StaticCache(m, 8, 8, 129).prefill_rows(ps) if is_z(m) else
                                                 mod.StaticCache(m, 8).prefill_rows([tf_seq(p) for p in ps]))
                except Exception as e:  # noqa: BLE001
                    row[tag + "_skip"] = str(e).splitlines()[0][:120]
                if cuda:
                    torch.cuda.empty_cache()
            res["prefill"].append(row)
            log("prefill T %5d B %2d: %s" % (T_, B, ", ".join("%s %.1f ms" % (t, row[t + "_ms"]) for t, _ in ARMS
                                                              if t + "_ms" in row)))
    if "e2e" in parts and not over():
        mz = next((m for _, m in ARMS if is_z(m)), None)
        if a.run:
            from generate_readings import load_run
            mz, idt = load_run(a.run, a.data, dev)
            mz = mz.eval().to(DT)
            res["e2e_model"] = dict(run=a.run, identity={k: idt.get(k) for k in ("d", "layers", "global_layers")})
        else:
            res["e2e_model"] = "rastgele agirlik (ilk Model Z kolu)"
        if mz is None:
            log("e2e: Model Z yok, atlandi")
        else:
            ps = prompts_of(300, a.e2e_prompts if cuda else 3)
            outs = {}
            for bs in (1, 32):
                sync()
                t = time.perf_counter()
                outs[bs] = mz.generate(ps, a.e2e_sentences if cuda else 2, D.MAX_SENTENCE_TOKENS if cuda else 4,
                                       open_last=True, batch_size=bs)
                dt = time.perf_counter() - t
                ntok = sum(len(s) + 1 for g, _, _ in outs[bs] for s in g)
                res["e2e"].append(dict(batch_size=bs, prompts=len(ps), s=dt, tokens=ntok, tok_s=ntok / dt))
                log("e2e generate %d istem, batch_size %2d: %.2f s, %d adim, %.0f adim/sn" % (len(ps), bs, dt, ntok, ntok / dt))
            res["e2e_same"] = sum(x == y for x, y in zip(outs[1], outs[32]))
            log("e2e: toplu = tek tek %d / %d istem (%s; bf16'da cekirdek farki ayristirabilir)" % (
                res["e2e_same"], len(ps), a.dtype))
    json.dump(res, open(os.path.join(a.out, "gen_speed.json"), "w"), indent=1)
    first = ARMS[0][0]
    print("\n=== gen_speed decode (%s): ms / adim | token / sn | KV GB | taban kat | max B | %s'e gore kat ===" % (a.dtype, first))
    ref = {(r["ctx"], r["B"]): r for r in res["decode"] if r["arm"] == first and "skip" not in r}
    for r in res["decode"]:
        if "skip" in r:
            print("  %-10s ctx %5d B %3d  ATLA" % (r["arm"], r["ctx"], r["B"]))
            continue
        z = ref.get((r["ctx"], r["B"]))
        print("  %-10s ctx %5d B %3d  %8.3f  %9.0f  %6.3f  %5.1fx  %6s  %s" % (
            r["arm"], r["ctx"], r["B"], r["ms"], r["tok_s"], r["kv_gb"], r["ms"] / r["floor_ms"], r.get("max_B"),
            "%.2fx" % (r["ms"] / z["ms"]) if z and r["arm"] != first else ""))
    log("yazildi: %s" % os.path.join(a.out, "gen_speed.json"))
    print("GEN_SPEED_BITTI", flush=True)
    return res


if __name__ == "__main__":
    main()
