"""train_sentence -- SentenceTransformer egitimi (ajan: sentence.py; z: sentence_z.py).  Ulke hikayeleri: her cumle, onceki
cumlelerinin z'leri verilerek kelime kelime tahmin edilir (teacher forcing).  Ornek = tek cumle, kisa dizi
[BOS][z_1..z_k] kelimeler; butun ornekler bir kez tensor olarak kurulur.  Olcu (sinav hikayeleri, yalniz sonraki 1 cumle;
kullanici: "Safece 1 cümle sonrasına bakacak"; 2. cumleden itibaren):
    kayip        dogru z ve karisik z (her z ayni sirada baska ulkenin hikayesinden): model z'yi kullaniyor mu
    ulke adi     yalniz z'den gelebilen kelime: dogru z / karisik z / sifir z; cumle basinda ve cumle icinde ayri
    attention    (son epok) ulke adini tahmin eden satirlarin z'lere verdigi agirlik; son z'ye mi eskilere mi
    uretim       gercek onceki cumlelerin z'leriyle sonraki cumle acgozlu: gecerli (valid_next), bu ulke icin dogru olgu,
                 onceden soylenmis olgunun tekrari, END, dongu, birebir; son epok z kontrolu ve goz icin ornekler
z'ler ayri bir surecte bir kez hesaplanip CACHE'e yazilir (--prepare); egitim sureci dosyayi okur.
checkpoint.pt her CHECKPOINT_SECS'te ve epok sonunda (kaldigi yerden surdurur); sonda agent.pt ve results.json.

    python train_sentence.py [--epochs 4] [--d 64] [--layers 4] [--heads 4] [--batch 160] [--lr 1e-3] [--z 512]
                             [--device cpu|cuda] [--out klasor] [--resume 1]
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

from sentence import BOS, WORD, ZTOK, SentenceTransformer, rope

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_Z = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "grammar"))
from train_grammar import _no_power_throttling  # noqa: E402
RUNS = "G:/Drive'ım/model_z/runs/"
MEANING = RUNS + "meaning_table_countries_w5_d128_mix_sub_e4_20261004_203221/agent.pt"
GRAMMAR = RUNS + "grammar_countries_d64_cosine_lr0.003_b64_20261004_133746/agent.pt"
CACHE = os.path.join(MODEL_Z, "cache") + "/"         # z bankasi (git disi): ayri surecte bir kez hesaplanir
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
CLIP = 1.0
WARMUP = 0.05           # adimlarin bu kadari dogrusal isinma, sonra cosine
CHECKPOINT_SECS = 600
SHOW = 8


def load(args):
    """-> sozluk, hikayeler (split, ulke, cumle kimlikleri, valid_next kimlikleri), cumleler (kimlik listeleri)."""
    folder = os.path.join(MODEL_Z, "data", "countries")
    vocab = json.load(open(os.path.join(folder, "country_vocab.json"), encoding="utf-8"))
    ix = {w: i for i, w in enumerate(vocab)}
    ids = lambda s: [ix.get(w, 0) for w in s]  # noqa: E731
    stories, sents = [], []
    for line in open(os.path.join(folder, "country_stories.jsonl"), encoding="utf-8"):
        s = json.loads(line)
        first = len(sents)
        sents += [ids(x) for x in s["sentences"]]
        valid = [{tuple(ids(x)) for x in step} for step in s.get("valid_next", [])]
        stories.append(dict(split=s["split"], country=ids(s["country"].split()), sent=list(range(first, len(sents))),
                            valid=valid))
    return vocab, stories, sents


def pad_sentences(sents):
    L = max(len(s) for s in sents)
    ids = torch.zeros(len(sents), L, dtype=torch.long)
    mask = torch.zeros(len(sents), L, dtype=torch.bool)
    for i, s in enumerate(sents):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = True
    return ids, mask


def examples(stories, sents, END, donors=None):
    """Hikayeler -> her cumle bir ornek: [BOS][z_1..z_k] w_1..w_n (k onceki cumle).  Satir k (son ozet) w_1'i, satir
    k+i w_{i+1}'i (sonuncusu END) tahmin eder.  donors: hikaye -> z'leri alinacak baska hikaye (karisik z olcusu).
    -> tensorler + ulke adi hedef maskeleri (cumle basinda / icinde)."""
    rows = [(si, k) for si, st in enumerate(stories) for k in range(len(st["sent"]))]
    T = max(k + 1 + len(sents[stories[si]["sent"][k]]) for si, k in rows)
    n = len(rows)
    kind = torch.zeros(n, T, dtype=torch.long)
    tok = torch.zeros(n, T, dtype=torch.long)
    zidx = torch.full((n, T), -1)
    target = torch.full((n, T), -100)
    c_start = torch.zeros(n, T, dtype=torch.bool)
    c_in = torch.zeros(n, T, dtype=torch.bool)
    for r, (si, k) in enumerate(rows):
        st = stories[si]
        zsrc = stories[donors[si]]["sent"] if donors is not None else st["sent"]
        s = sents[st["sent"][k]]
        kind[r, 0] = BOS
        for j in range(k):
            kind[r, 1 + j], zidx[r, 1 + j] = ZTOK, zsrc[j % len(zsrc)]
        kind[r, 1 + k:1 + k + len(s)] = WORD
        tok[r, 1 + k:1 + k + len(s)] = torch.tensor(s)
        target[r, k:k + len(s) + 1] = torch.tensor(s + [END])
        name = st["country"]
        for i in range(len(s) - len(name) + 1):
            if s[i:i + len(name)] == name:
                for j in range(len(name)):
                    (c_start if i == 0 else c_in)[r, k + i + j] = True
    return dict(kind=kind, tok=tok, zidx=zidx, target=target, c_start=c_start, c_in=c_in,
                k=torch.tensor([k for _, k in rows]), story=torch.tensor([si for si, _ in rows]))


def batch(D, rows, zbank, device, zero=False):
    kind = D["kind"][rows]
    zvec = torch.zeros(len(rows), kind.shape[1], zbank.shape[1]) if zero else \
        zbank[D["zidx"][rows].clamp(min=0)] * (kind == ZTOK)[..., None]
    pos = torch.arange(kind.shape[1]).expand(len(rows), -1)
    return [x.to(device) for x in (kind, D["tok"][rows], zvec, pos)] + [None, D["target"][rows].to(device)]


@torch.no_grad()
def exam_scores(model, D, zbank, device, zero=False):
    """2. cumleden itibaren: kayip, ulke adi kaybi / dogrulugu (basta / icte), diger kelime dogrulugu."""
    rows = (D["k"] >= 1).nonzero().flatten()
    nll, hit, keep_all, cs, ci = [], [], [], [], []
    for c in range(0, len(rows), 1024):
        r = rows[c:c + 1024]
        kind, tok, zvec, pos, _, target = batch(D, r, zbank, device, zero)
        logits = (model.hidden(kind, tok, zvec, pos, None) @ model.E.weight.T).float()
        nll.append(-F.log_softmax(logits, -1).gather(-1, target.clamp(min=0)[..., None])[..., 0].cpu())
        hit.append((logits.argmax(-1) == target).cpu())
        keep_all.append((target >= 0).cpu())
        cs.append(D["c_start"][r])
        ci.append(D["c_in"][r])
    nll, hit, keep, cs, ci = (torch.cat(x) for x in (nll, hit, keep_all, cs, ci))
    country = cs | ci
    other = keep & ~country
    return dict(loss=float(nll[keep].mean()), loss_country=float(nll[country].mean()),
                acc_country=float(hit[country].float().mean()), acc_country_start=float(hit[cs].float().mean()),
                acc_country_in=float(hit[ci].float().mean()), acc_other=float(hit[other].float().mean()))


@torch.no_grad()
def attention_to_z(model, D, zbank, device):
    """Ulke adini tahmin eden satirlar (2+ onceki cumle): katman basina z'lere toplam agirlik ve bunun son z'ye payi."""
    rows = ((D["k"] >= 2) & (D["c_start"] | D["c_in"]).any(1)).nonzero().flatten()[:1024]
    kind, tok, zvec, pos, _, target = batch(D, rows, zbank, device)
    x = model.E(tok)
    zpos = kind == ZTOK
    x = x.index_put((zpos,), model.z_in(model.z_norm(zvec[zpos])))
    x = torch.where((kind == BOS)[..., None], model.bos.expand_as(x), x)
    B, T, d = x.shape
    causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
    country = (D["c_start"] | D["c_in"])[rows].to(x.device)
    last_z = (D["k"][rows] - 1 + 1).to(x.device)                  # son z'nin konumu: k
    out = []
    for blk in model.blocks:
        q, k_, v = blk.qkv(blk.n1(x)).view(B, T, 3, blk.heads, d // blk.heads).permute(2, 0, 3, 1, 4)
        q, k_ = rope(blk.q_norm(q), pos), rope(blk.k_norm(k_), pos)
        att = (q @ k_.transpose(-1, -2) / math.sqrt(d // blk.heads)).masked_fill(~causal, -1e9).softmax(-1).mean(1)
        to_z = (att * zpos[:, None, :]).sum(-1)                     # (B, T) butun z'lere
        to_last = att.gather(-1, last_z[:, None, None].expand(B, T, 1))[..., 0]
        out.append((float(to_z[country].mean()), float((to_last[country] / to_z[country].clamp_min(1e-9)).mean())))
        x = blk(x, pos, None)
    return out


@torch.no_grad()
def generation(model, exam, sents, zbank, device, show, keys=None):
    """Gercek onceki cumlelerin z'leriyle sonraki cumle -> olculer + ornekler."""
    vocab = model.vocab
    text = lambda s: " ".join(vocab[w] for w in s)  # noqa: E731
    longest = max(len(s) for s in sents)
    by_k = {}
    for si, st in enumerate(exam):
        for k in range(1, len(st["sent"])):                       # k onceki cumle -> st["sent"][k] tahmin
            by_k.setdefault(k, []).append(si)
    res = dict(n=0, valid=0, true_fact=0, repeat=0, has_country=0, ended=0, loop=0, exact=0, z_ok=0)
    shown, outs = [], []
    for k, items in sorted(by_k.items()):
        for c in range(0, len(items), 512):
            part = items[c:c + 512]
            zs = torch.stack([zbank[exam[si]["sent"][:k]] for si in part]).to(device)
            for si, o in zip(part, model.generate(zs, longest + 3)):
                st = exam[si]
                prev = {tuple(sents[g]) for g in st["sent"][:k]}
                facts = set().union(*st["valid"]) | {tuple(sents[g]) for g in st["sent"]}
                name = st["country"]
                ok = tuple(o) in st["valid"][k - 1]
                res["n"] += 1
                res["valid"] += ok
                res["true_fact"] += tuple(o) in facts
                res["repeat"] += tuple(o) in prev
                res["has_country"] += any(o[i:i + len(name)] == name for i in range(len(o)))
                res["ended"] += len(o) <= longest + 2
                res["loop"] += any(o[i:i + 3] == o[i + 3:i + 6] for i in range(max(0, len(o) - 5)))
                res["exact"] += o == sents[st["sent"][k]]
                if 0 < len(o) <= longest:                         # bitmemis / bos cumle z kontrolunde basarisiz
                    outs.append(o)
                if len(shown) < show and si % 37 == 0 and k >= 3:
                    tag = "GECERLI" if ok else ("TEKRAR" if tuple(o) in prev else
                                                ("DOGRU OLGU" if tuple(o) in facts else "-"))
                    shown.append(([text(sents[g]) for g in st["sent"][:k]], text(o), tag))
    if keys:
        from sentence_z import decode_z, encode_z
        for c in range(0, len(outs), 2048):                         # z kontrolu toplu (yalniz son olcumde)
            part = outs[c:c + 2048]
            back = decode_z(keys, encode_z(keys, *pad_sentences(part)))
            res["z_ok"] += sum(b == o for b, o in zip(back, part))
    n = res.pop("n")
    out = {k: round(v / n, 4) for k, v in res.items() if k != "z_ok" or keys}
    out["n"] = n
    for prev, o, tag in shown:
        print("   onceki : " + "\n            ".join(prev) + "\n   model  : %s   [%s]" % (o, tag), flush=True)
    return out


def z_cache_path(args):
    return CACHE + "sentence_z_countries_z%d.pt" % args.z


def prepare(args, sents):
    """z bankasi + cumlelerin %10'unda geri acma -> CACHE (bu surec grammar'i yukler; egitim sureci yuklemez)."""
    from sentence_z import build_keys, decode_z, encode_z
    keys = build_keys(args.meaning, args.grammar, args.z, max(len(s) for s in sents))
    zbank = torch.cat([encode_z(keys, *pad_sentences(sents[c:c + 2048])) for c in range(0, len(sents), 2048)])
    sample = list(range(0, len(sents), 10))
    back = decode_z(keys, zbank[sample])
    z_check = sum(back[i] == sents[g] for i, g in enumerate(sample)) / len(sample)
    os.makedirs(CACHE, exist_ok=True)
    torch.save(dict(zbank=zbank, z_check=z_check, meaning=args.meaning, grammar=args.grammar, z=args.z, n=len(sents)),
               z_cache_path(args))
    print("z bankasi yazildi: %s (%d cumle, geri acma %.3f)" % (z_cache_path(args), len(sents), z_check), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--meaning", default=MEANING)
    ap.add_argument("--grammar", default=GRAMMAR)
    ap.add_argument("--z", type=int, default=512)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=160, help="cumle (160 ~ 16 hikaye)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile", type=int, default=1, help="1: torch.compile (yalniz cuda)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", type=int, default=0)
    ap.add_argument("--prepare", action="store_true", help="yalniz z bankasini hesapla ve CACHE'e yaz")
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    cuda = args.device.startswith("cuda")
    if not cuda and os.name == "nt":
        print("guc kisitlamasi (EcoQoS) kapali:", _no_power_throttling(), flush=True)
    t0 = time.time()

    vocab, stories, sents = load(args)
    if args.prepare:
        return prepare(args, sents)
    if not os.path.exists(z_cache_path(args)):
        subprocess.run([sys.executable, os.path.abspath(__file__), "--prepare", "--z", str(args.z), "--meaning",
                        args.meaning, "--grammar", args.grammar], check=True)
    cache = torch.load(z_cache_path(args), weights_only=False)
    assert (cache["meaning"], cache["grammar"], cache["z"], cache["n"]) == (args.meaning, args.grammar, args.z,
                                                                            len(sents)), "z bankasi baska ayarla"
    zbank, z_check = cache["zbank"], cache["z_check"]
    train = [s for s in stories if s["split"] == "train"]
    exam = [s for s in stories if s["split"] == "exam"]
    END = len(vocab)
    TR, EX = examples(train, sents, END), examples(exam, sents, END)
    g = torch.Generator().manual_seed(args.seed + 1000)               # karisik z: baska ulkenin hikayesi
    donors = []
    for i, s in enumerate(exam):
        while True:
            j = int(torch.randint(len(exam), (1,), generator=g))
            if exam[j]["country"] != s["country"]:
                donors.append(j)
                break
    EX_SHUF = examples(exam, sents, END, donors)

    model = SentenceTransformer(vocab, args.z, args.d, args.layers, args.heads).to(args.device)
    if cuda:
        torch.set_float32_matmul_precision("high")
        if args.compile:
            for block in model.blocks:
                block.compile(dynamic=True)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=WEIGHT_DECAY), dict(params=no_decay, weight_decay=0.0)],
                            lr=args.lr, betas=BETAS)
    n = len(TR["kind"])
    steps_per = -(-n // args.batch)
    total_steps = steps_per * args.epochs
    first, start, ckpt = 1, 0, os.path.join(args.out, "checkpoint.pt") if args.out else None
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    if args.resume:
        pack = torch.load(ckpt, map_location=args.device, weights_only=False)
        keys_ = ("z", "d", "layers", "heads", "batch", "lr", "epochs", "seed", "meaning", "grammar")
        diff = {k: (pack["args"][k], vars(args)[k]) for k in keys_ if pack["args"][k] != vars(args)[k]}
        assert not diff, "surdurme ayari checkpoint'ten farkli (checkpoint, simdi): %s" % diff
        model.load_state_dict(pack["state"])
        opt.load_state_dict(pack["opt"])
        first, start = pack["epoch"], pack["next"]
        print("SURDURULDU: epok %d, ornek %d'den" % (first, start), flush=True)

    def save(epoch, nxt):
        if ckpt:
            torch.save(dict(vocab=vocab, state=model.state_dict(), opt=opt.state_dict(), epoch=epoch, next=nxt,
                            args=vars(args)), ckpt + ".part")
            os.replace(ckpt + ".part", ckpt)

    print("veri ulke: egitim %d hikaye (%d cumle ornegi), sinav %d hikaye | z %d (meaning+grammar+konum; cumlelerin %%10'unda "
          "geri acma %.3f) | d %d, katman %d, head %d, %d parametre | batch %d cumle, lr %g, %d adim | cihaz %s (%.0f sn "
          "hazirlik)" % (len(train), n, len(exam), args.z, z_check, args.d, args.layers, args.heads,
                         sum(p.numel() for p in model.parameters()), args.batch, args.lr, total_steps, args.device,
                         time.time() - t0), flush=True)
    history = []
    for epoch in range(first, args.epochs + 1):
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed + epoch))
        total, count, t_epoch, t_saved = 0.0, 0, time.time(), time.time()
        b0 = start if epoch == first else 0
        for b in range(b0, n, args.batch):
            if time.time() - t_saved > CHECKPOINT_SECS:
                t_saved = time.time()
                save(epoch, b)
            step = (epoch - 1) * steps_per + b // args.batch
            warm = max(1, int(WARMUP * total_steps))
            lr = args.lr * (step + 1) / warm if step < warm else \
                args.lr * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total_steps - warm)))
            for group in opt.param_groups:
                group["lr"] = lr
            with torch.autocast(args.device.split(":")[0], dtype=torch.bfloat16, enabled=cuda):
                loss = model(*batch(TR, perm[b:b + args.batch], zbank, args.device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            total += loss.detach()
            count += 1
        t_train = time.time() - t_epoch
        model.eval()
        last = epoch == args.epochs
        with torch.autocast(args.device.split(":")[0], dtype=torch.bfloat16, enabled=cuda):
            e = exam_scores(model, EX, zbank, args.device)
            s = exam_scores(model, EX_SHUF, zbank, args.device)
            z0 = exam_scores(model, EX, zbank, args.device, zero=True)
            keys = None
            if last:                                               # z kontrolu: grammar yalniz en sonda yuklenir
                from sentence_z import build_keys
                keys = build_keys(args.meaning, args.grammar, args.z, max(len(x) for x in sents))
            gen = generation(model, exam, sents, zbank, args.device, SHOW if last else 0, keys)
            att = attention_to_z(model, EX, zbank, args.device) if last else None
        model.train()
        print("epok %d  kayip %.3f | sinav %.3f (karisik z %.3f) | ulke adi dogru: z %.3f (basta %.3f, icte %.3f), "
              "karisik z %.3f, sifir z %.3f | diger kelime %.3f | uretim %s  (egitim %.0f sn, olcum %.0f sn)" % (
                  epoch, float(total) / count, e["loss"], s["loss"], e["acc_country"], e["acc_country_start"],
                  e["acc_country_in"], s["acc_country"], z0["acc_country"], e["acc_other"], gen, t_train,
                  time.time() - t_epoch - t_train), flush=True)
        if att:
            print("attention (ulke adini tahmin eden satirlar, 2+ onceki cumle): katman | butun z'lere | bunun son z'ye "
                  "payi\n" + "\n".join("   %d   %.3f   %.3f" % (i, a, b) for i, (a, b) in enumerate(att)), flush=True)
        history.append(dict(epoch=epoch, loss=float(total) / count, exam=e, exam_shuffled_z=s, exam_zero_z=z0,
                            generation=gen, attention=att))
        save(epoch + 1, 0)
    if args.out:
        torch.save(dict(vocab=vocab, state=model.state_dict(), args=vars(args)), os.path.join(args.out, "agent.pt"))
        json.dump(dict(args=vars(args), z_check=z_check, history=history), open(os.path.join(args.out, "results.json"),
                                                                                "w"), indent=1)
        print("kaydedildi:", args.out, flush=True)
    return model


if __name__ == "__main__":
    main()
