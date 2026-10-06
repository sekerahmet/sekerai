"""metrics -- V2 ortak olculer (iki model ayni kod; belge 21 §7; adlar onayli, kullanici 6 Ekim).

Model sozlesmesi (iki model saglar):
    model.loss_per_target(batch) -> (nll (K,), pred (K,), target_kind (K,))   hedefli konumlar, satir sirasiyla
    model.generate(prompts, max_sentences, max_tokens, generator) -> istem basina (cumleler, END ile bitti mi, eos)
        prompts: istem basina cumle token listeleri (END yok); generator None acgozlu, yoksa CPU torch.Generator.
sentence_repeat (TEK tanim, kullanici onayli): yalniz END ile biten, bos olmayan uretilmis cumleler sayilir; pay, kelime
dizisi (normalize_words) o hikayede (istem + daha once uretilenler) birebir gecmis olanlar.  Kesik (no_end) ve bos
(empty) cumleler ayri oran.  bpb: EOS haric nll (END dahil) / ln 2 / hikaye baytlari; _eos: hepsi / (bayt + hikaye).
"""
import math

import numpy as np
import torch

from data import BATCH_ROWS, TargetKind, _WORD, build_batch


def normalize_words(token_ids, tokenizer):
    """Token kimlikleri -> V1 kelime listesi (GPT-2 decode -> V1 _WORD; bosluk ve satir sonu atilir)."""
    return _WORD.findall(tokenizer.decode([int(t) for t in token_ids]))


def _repeat_counts(prompt_words, gen_words, ended):
    """-> (tekrar eden, sayilan): yalniz END ile biten, bos olmayan uretilmis cumleler."""
    seen = {tuple(w) for w in prompt_words}
    hit = total = 0
    for w, e in zip(gen_words, ended):
        if e and w:
            total += 1
            hit += tuple(w) in seen
        seen.add(tuple(w))
    return hit, total


def sentence_repeat(prompt_sents, gen_sents, ended):
    """Kelime listeleri (normalize_words) -> oran (sayilan yoksa None)."""
    hit, total = _repeat_counts(prompt_sents, gen_sents, ended)
    return hit / total if total else None


def reading_view(sentences, tokenizer):
    """Cumle token listeleri -> V1 bicimi duz metin (kelimeler bosluklu, her cumle bir satir): kor okumada iki model ve V1
    ayni bicimde."""
    return "\n".join(" ".join(normalize_words(s, tokenizer)) for s in sentences)


@torch.no_grad()
def story_generation(model, prompts, decode, seed, max_sentences=80, max_tokens=128, tokenizer=None):
    """Kapali dongu: model.generate -> (istem basina dict(prompt, story metni reading_view), olculer).  decode greedy |
    sample (sicaklik 1, kesme yok, CPU Generator tohum seed).  Olculer: eos_rate, sentences (istem basina), sentence_repeat,
    loop (cumle icinde ardisik 3'lu kelime tekrari), no_end, empty (uretilen cumlelere oran)."""
    assert decode in ("greedy", "sample") and tokenizer is not None
    gen = torch.Generator().manual_seed(seed) if decode == "sample" else None
    outs = model.generate(prompts, max_sentences, max_tokens, gen)
    hit = total = n = eos = loop = no_end = empty = 0
    texts = []
    for p, (sents, ended, e) in zip(prompts, outs):
        pw = [normalize_words(s, tokenizer) for s in p]
        gw = [normalize_words(s, tokenizer) for s in sents]
        h, t = _repeat_counts(pw, gw, ended)
        hit, total, n, eos = hit + h, total + t, n + len(sents), eos + bool(e)
        loop += sum(any(w[i:i + 3] == w[i + 3:i + 6] for i in range(max(0, len(w) - 5))) for w in gw)
        no_end += sum(not x for x in ended)
        empty += sum(not w for w in gw)
        texts.append(dict(prompt=reading_view(p, tokenizer), story=reading_view(sents, tokenizer), eos=bool(e)))
    rate = lambda a: round(a / n, 4) if n else None  # noqa: E731
    return texts, dict(decode=decode, prompts=len(prompts), eos_rate=round(eos / len(prompts), 4),
                       sentences=round(n / len(prompts), 2), sentence_repeat=round(hit / total, 4) if total else None,
                       counted=total, loop=rate(loop), no_end=rate(no_end), empty=rate(empty))


@torch.no_grad()
def exam_scores(model, exam_stories, plan, story_bytes, layout, device="cpu", batch_rows=BATCH_ROWS):
    """Sinav: plan (row_offsets, row_stories) satirlariyla, egitimle ayni satir sekli -> loss, ppl, acc (butun hedefler),
    acc_token (FIRST + MID), end_ok, eos_ok, bits_per_byte (EOS haric), bits_per_byte_eos, by_kind (tur basina nll / acc /
    sayi).  story_bytes: hikaye kimligiyle indekslenen bayt dizisi (valid_bytes)."""
    ro, rs = plan
    rows = [rs[ro[r]:ro[r + 1]].tolist() for r in range(len(ro) - 1)]
    names = {TargetKind.FIRST: "first", TargetKind.MID: "mid", TargetKind.END: "end", TargetKind.EOS: "eos"}
    sums = {k: [0.0, 0, 0] for k in names}                               # nll, dogru, sayi
    for c in range(0, len(rows), batch_rows):
        batch = build_batch(exam_stories, rows[c:c + batch_rows], layout, device)
        nll, pred, tk = model.loss_per_target(batch)
        tgt = batch.target[batch.target >= 0]
        for k in names:
            m = tk == k
            sums[k][0] += float(nll[m].double().sum())
            sums[k][1] += int((pred[m] == tgt[m]).sum())
            sums[k][2] += int(m.sum())
    tot = [sum(v[i] for v in sums.values()) for i in range(3)]
    stories = np.asarray(rs)
    nbytes = int(np.asarray(story_bytes)[stories].sum())
    eos_nll = sums[TargetKind.EOS][0]
    nll = tot[0] / tot[2]
    tok = [sums[TargetKind.FIRST][i] + sums[TargetKind.MID][i] for i in range(3)]
    return dict(loss=round(nll, 4), ppl=round(math.exp(nll), 2), acc=round(tot[1] / tot[2], 4),
                acc_token=round(tok[1] / tok[2], 4), end_ok=round(sums[TargetKind.END][1] / sums[TargetKind.END][2], 4),
                eos_ok=round(sums[TargetKind.EOS][1] / sums[TargetKind.EOS][2], 4),
                bits_per_byte=round((tot[0] - eos_nll) / math.log(2) / nbytes, 4),
                bits_per_byte_eos=round(tot[0] / math.log(2) / (nbytes + len(stories)), 4),
                by_kind={names[k]: dict(nll=round(v[0] / v[2], 4) if v[2] else None, acc=round(v[1] / v[2], 4) if v[2]
                                        else None, n=v[2]) for k, v in sums.items()}, stories=len(stories), bytes=nbytes)
