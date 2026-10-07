"""generate_readings -- kayitli bir V2 kosusundan okuma uretimi, egitim yok (adlar onayli, kullanici 6 Ekim).

Model train._build ile kurulur, agent.pt yuklenir (temizlik oncesi yolla egitilmis kosu durur: train._archived); okuma
train._readings ile (egitim sonundakiyle ayni uretim, ayni olculer).  Once kosunun kendi samples.json istemleri yeniden uretilir ve metinleri birebir
karsilastirilir (reproduced): yukleme ve kod egitimdekiyle ayni mi.  Sonra --prompts istemleri.
Cikti: <out>/samples_<ek>.txt, samples_<ek>.json (istem dosyasi reading_prompts_<ek>.json); agent.pt, results.json,
samples.txt / .json'a dokunmaz.

Ek istemler (common/reading_prompts_extra.json) extra_prompts() ile yazildi; tests_diag 'readings' Drive'dan yeniden
hesaplar.  Teshis araclari kosuyu load_run ile yukler (tek yer).

    python generate_readings.py --out <kosu> [--prompts reading_prompts_extra.json] [--data <v2/simplestories_gpt2>]
                                [--stream <simplestories>] [--device cuda]
"""
import torch  # noqa: I001  (Windows: torch once)

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(os.path.dirname(HERE), "common")
sys.path.insert(0, COMMON)
import data as D  # noqa: E402
import train as T  # noqa: E402

EXTRA_PROMPTS = os.path.join(COMMON, "reading_prompts_extra.json")
V1_POOL = (740, 3011, 9616, 10582, 15877, 17492, 20389, 20431)   # V1 istem havuzu (arsiv transformer_baseline/ss_prompts.json)
EXTRA_SEED, EXTRA_STORIES, MIN_SENTENCES = 2, 5, 8     # tohum 0 sinav alt kumesi, 1 V1 istemleri (ss_prompts.json)
EXTRA_FIRST_LABEL = 11


def extra_prompts(data_dir):
    """Sinav hikayelerinden ek okuma istemleri: aday = sinav alt kumesi (exam_pack_plan.npz) - V1 istem havuzu
    (ss_prompts.json) - sabit 10 (reading_prompts.json), en az MIN_SENTENCES cumle; secilen = np.sort(aday[default_rng(
    EXTRA_SEED).choice(len(aday), EXTRA_STORIES)]); hikaye basina 1 ve 3 cumle istem, greedy."""
    ep = np.load(os.path.join(data_dir, "exam_pack_plan.npz"))
    n_sent = np.diff(np.load(os.path.join(data_dir, "valid_story_offsets.npy")))
    fixed = json.load(open(T.READING_PROMPTS, encoding="utf-8"))["prompts"]
    exclude = sorted(set(V1_POOL) | {p["story"] for p in fixed})
    exam = np.sort(ep["row_stories"].astype(np.int64))
    cand = exam[~np.isin(exam, exclude) & (n_sent[exam] >= MIN_SENTENCES)]
    pick = np.sort(cand[np.random.default_rng(EXTRA_SEED).choice(len(cand), EXTRA_STORIES, replace=False)])
    prompts = [dict(label="okuma %d" % (EXTRA_FIRST_LABEL + 2 * i + j), story=int(s), sentences=k, decode="greedy")
               for i, s in enumerate(pick.tolist()) for j, k in enumerate((1, 3))]
    return dict(source="exam_pack_plan.npz (exam_set_sha256 %s): sinav alt kumesi, %d hikaye" % (
                    str(ep["exam_set_sha256"]), len(exam)),
                selection="generate_readings.extra_prompts: aday = sinav hikayeleri - V1 istem havuzu (transformer_baseline/"
                          "ss_prompts.json) - sabit 10 (reading_prompts.json), en az %d cumle (%d aday); secilen = "
                          "np.sort(aday[default_rng(%d).choice(len(aday), %d, replace=False)]); hikaye basina 1 ve 3 "
                          "cumle istem (ayni hikaye yan yana), hepsi greedy" % (MIN_SENTENCES, len(cand), EXTRA_SEED,
                                                                                EXTRA_STORIES),
                seed=EXTRA_SEED, min_sentences=MIN_SENTENCES, exclude=exclude,
                story_sentences={str(s): int(n_sent[s]) for s in pick.tolist()},
                story_index="sinav hikayesi indeksi = valid hikaye sirasi (reading_prompts.json ile ayni numara)",
                prompts=prompts)


def load_run(out, data_dir, dev):
    """<out>/agent.pt (train.py bicimi) -> (model, identity).  Model train._build ile; agirliklar strict.  Temizlik
    oncesi yolla egitilmis kosu (train._archived) DURUR: eski Model Z'nin sekilleri ayni, sessizce yanlis yuklenirdi."""
    pack = torch.load(os.path.join(out, "agent.pt"), map_location="cpu", weights_only=False)
    idt = pack["identity"]
    if T._archived(idt):
        sys.exit("DUR: %s: %s" % (out, T._archived(idt)))
    meta = json.load(open(os.path.join(data_dir, "train_boundaries.json"), encoding="utf-8"))
    longest = meta.get("max_sentence_tokens_all", meta["max_sentence_tokens"])      # = TokenStories.max_sentence_tokens
    assert longest == idt["longest"], "en uzun cumle %d, kosununki %d: baska veri" % (longest, idt["longest"])
    spec = SimpleNamespace(model=idt["model"], seed=idt["seed"], d=idt["d"], layers=idt["layers"], heads=idt["heads"],
                           global_layers=idt.get("global_layers", 0),          # global_layers'tan onceki: 0
                           layer_plan=idt.get("layer_plan"),
                           bag_k=idt.get("bag_k", 0),
                           bag_n_core=len(pack["state"]["bag.core"]) if idt.get("bag_k") else 0)   # C state_dict'ten
    model, _, layout = T._build(spec, dev)
    model.load_state_dict(pack["state"])
    return model, idt


def _args(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="kosu klasoru (agent.pt; cikti buraya)")
    ap.add_argument("--prompts", default=EXTRA_PROMPTS, help="reading_prompts_<ek>.json -> samples_<ek>.txt / .json")
    ap.add_argument("--data", default="/content/drive/MyDrive/v2/simplestories_gpt2", help="sinir dosyalari")
    ap.add_argument("--stream", default="/content/drive/MyDrive/simplestories", help="gpt2/valid.npy kok")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args(argv)


def main(argv=None):
    args = _args(argv)
    t0 = time.time()
    log = lambda msg: print("[%6.1f sn] %s" % (time.time() - t0, msg), flush=True)  # noqa: E731
    dev = torch.device(args.device)
    if dev.type == "cuda":                                               # GPU kapisi (kural 5)
        assert torch.cuda.is_available(), "GPU YOK"
    elif os.name == "nt":
        log("guc kisitlamasi (EcoQoS) kapali: %s" % T._no_power_throttling())
    base = os.path.basename(args.prompts)
    assert base.startswith("reading_prompts_") and base.endswith(".json") and len(base) > len("reading_prompts_.json"), \
        "istem dosyasi reading_prompts_<ek>.json olmali (samples.txt / .json uzerine yazilmaz): %s" % base
    name = "samples_" + base[len("reading_prompts_"):-len(".json")]
    spec = json.load(open(args.prompts, encoding="utf-8"))["prompts"]
    model, idt = load_run(args.out, args.data, dev)
    valid = D.TokenStories(args.stream, args.data, "valid")
    got = T._sha256(os.path.join(args.stream, "gpt2", "valid.npy"))
    assert got == valid.meta["stream_sha256"], "valid akisi sinir dosyasindakiyle ayni degil"
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(args.stream, "gpt2", "tokenizer.json"))
    log("%s | %s | istem %s (%d)" % (args.out, idt, args.prompts, len(spec)))
    reproduced = None
    old_path = os.path.join(args.out, "samples.json")
    if os.path.exists(old_path):
        old = json.load(open(old_path, encoding="utf-8"))["rows"]
        rows0, gen0 = T._readings(model, valid, tok, [{k: r[k] for k in ("label", "story", "sentences", "decode")}
                                                      for r in old])
        same = [r["label"] for r, o in zip(rows0, old) if (r["story_text"], r["eos"]) == (o["story_text"], o["eos"])]
        reproduced = dict(file="samples.json", same=same, different=[o["label"] for o in old if o["label"] not in same],
                          generation=gen0)
        log("samples.json yeniden uretimi: %d / %d istem birebir ayni%s" % (
            len(same), len(old), "" if len(same) == len(old) else " | FARKLI: %s" % reproduced["different"]))
    t = time.time()
    rows, gen = T._readings(model, valid, tok, spec)
    log("okuma uretimi %.0f sn: %s" % (time.time() - t, gen))
    open(os.path.join(args.out, name + ".txt"), "w", encoding="utf-8").write(T._samples_text(rows))
    json.dump(dict(generation=gen, rows=rows, reproduced=reproduced, source=dict(
        prompts=args.prompts, prompts_sha256=T._sha256(args.prompts), identity=idt,
        limits=T.READING_LIMITS, git=T._git(), device=str(dev))),
        open(os.path.join(args.out, name + ".json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    log("BITTI: %s/%s.txt | sentence_repeat %s | story_loop %s" % (
        args.out, name, {d: g["sentence_repeat"] for d, g in gen.items()},
        {d: (g["story_loop"], g["story_loop_prompts"]) for d, g in gen.items()}))
    return dict(generation=gen, rows=rows, reproduced=reproduced)


if __name__ == "__main__":
    main()
