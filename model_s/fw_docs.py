"""FineWeb belgeleri V2 bicimiyle (cumle sonu END token'lari dahil, data.build_batch): ham token'lamanin yerine (TERSINE s8a: ham
token'la p_0 4,40, V2 sinavi 3,10).  fw_docs(n, start) -> [(tokens, targets)] -- tokens modelin gordugu dizi, targets hedef (-1: yok)."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "deneme2", "v2", "common"))
import data as D  # noqa: E402

STREAM = r"G:/Drive'ım/v2/fineweb_edu_s000"
BOUND = r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--AI-NEW-MODEL/ff22c4d2-29d1-4095-9dc6-424e77bbd3a0/scratchpad/fw_valid"
_S = None


def fw_docs(n, start=0, maxlen=1024, minlen=32):
    global _S
    if _S is None:
        _S = D.TokenStories(STREAM, BOUND, "valid")
    out, i = [], start
    while len(out) < n and i < _S.n:
        try:
            b = D.build_batch(_S, [[i]], "transformer", "cpu")
        except AssertionError:                                          # satira sigmayan (> 2.048) belge: atla
            i += 1
            continue
        keep = b.doc[0] >= 0
        tok, tgt = b.tokens[0][keep][:maxlen], b.target[0][keep][:maxlen]
        if len(tok) >= minlen:
            out.append((tok.tolist(), tgt.tolist()))
        i += 1
    return out


if __name__ == "__main__":
    from sx_code_recall import RUN
    from baseline import BaselineTransformer
    pack = torch.load(os.path.join(RUN, "agent.pt"), map_location="cpu", weights_only=False)
    idt = pack["identity"]
    tf = BaselineTransformer(idt["d"], idt["layers"], idt["heads"], vocab_rows=pack["state"]["E.weight"].shape[0])
    tf.load_state_dict(pack["state"])
    tf.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for tok, tgt in fw_docs(40):
            x = torch.tensor([tok])
            lp = tf._logits(tf.hidden(x, torch.arange(len(tok))[None], None)).float().log_softmax(-1)[0]
            t = torch.tensor(tgt)
            ok = t >= 0
            tot += float(-lp[ok].gather(1, t[ok][:, None]).sum())
            cnt += int(ok.sum())
    print("V2 bicimi, 40 belge: kayip %.4f (%d hedef)" % (tot / cnt, cnt))
