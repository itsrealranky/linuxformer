"""Qwen's text behaviour in the merged model against stock Qwen3.5-4B, on MMLU (cais/mmlu "all"
test split, 14,042 questions; data/mmlu, from the HF hub).

Zero-shot, one forward per question: the prompt below, then the answer is the letter among
" A" " B" " C" " D" with the highest logit at the last position. Reported for both models:
accuracy, how often their answers agree, and the KL between their full next-token distributions
at that position; beside it the noise floor, the same KL between stock Qwen with sdpa and with
eager attention. (A comparison harness, not the official MMLU protocol.)

    python tests/eval_text.py [--limit N] [--floor N] [--out FILE]
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from check_merge import stock_qwen

from lx import core, merge

DATA = Path(__file__).resolve().parents[2] / "data" / "mmlu" / "all" / "test-00000-of-00001.parquet"


def prompt(q: dict) -> str:
    subj = q["subject"].replace("_", " ")
    lines = [f"The following is a multiple choice question about {subj}.", "", q["question"]]
    lines += [f"{l}. {c}" for l, c in zip("ABCD", q["choices"])]
    return "\n".join(lines) + "\nAnswer:"


def _kl(a, b):
    la, lb = F.log_softmax(a.float(), -1), F.log_softmax(b.float(), -1)
    return (la.exp() * (la - lb)).sum(-1).item()


@torch.no_grad()
def main():
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--floor", type=int, default=2000, help="questions for the noise floor")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    qs = pq.read_table(DATA).to_pylist()
    if a.limit:
        qs = qs[:a.limit]
    tok = AutoTokenizer.from_pretrained(merge.QWEN)
    letters = [tok.encode(f" {l}", add_special_tokens=False)[0] for l in "ABCD"]
    assert len(set(letters)) == 4
    stock = stock_qwen(merge.QWEN, "cuda")
    model, info = merge.build(core.build(merge.config()))
    tv = merge.text_view(model, info)
    n_ok = {"stock": 0, "merged": 0, "eager": 0}
    agree, agree_eager, kls, floor = 0, 0, [], []
    t0 = time.time()
    for i, q in enumerate(qs):
        ids = torch.tensor([tok(prompt(q))["input_ids"]], device="cuda")
        ls = stock(input_ids=ids, logits_to_keep=1).logits[0, -1]
        lm = tv(input_ids=ids, logits_to_keep=1).logits[0, -1]
        ps, pm = int(ls[letters].argmax()), int(lm[letters].argmax())
        n_ok["stock"] += ps == q["answer"]
        n_ok["merged"] += pm == q["answer"]
        agree += ps == pm
        kls.append(_kl(ls, lm))
        if i < a.floor:
            stock.config._attn_implementation = "eager"
            le = stock(input_ids=ids, logits_to_keep=1).logits[0, -1]
            stock.config._attn_implementation = "sdpa"
            floor.append(_kl(ls, le))
            pe = int(le[letters].argmax())
            n_ok["eager"] += pe == q["answer"]
            agree_eager += pe == ps
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}: stock {n_ok['stock'] / (i + 1):.4f} merged "
                  f"{n_ok['merged'] / (i + 1):.4f} agree {agree / (i + 1):.4f} "
                  f"KL {sum(kls) / len(kls):.2e} ({time.time() - t0:.0f}s)", flush=True)
    n, nf = len(qs), max(1, len(floor))
    res = {"benchmark": "MMLU (cais/mmlu all, test), zero-shot, answer-letter argmax",
           "questions": n, "accuracy_stock": n_ok["stock"] / n,
           "accuracy_merged": n_ok["merged"] / n, "answers_agree": agree / n,
           "mean_kl_last_position": sum(kls) / n, "max_kl": max(kls),
           "noise_floor_kl_sdpa_vs_eager": sum(floor) / nf,
           "noise_floor_questions": len(floor),
           # on the noise floor's questions: stock with eager attention against stock (sdpa),
           # and the merged model's KL on the same questions
           "accuracy_stock_eager": n_ok["eager"] / nf, "eager_answers_agree": agree_eager / nf,
           "mean_kl_merged_on_floor_questions": sum(kls[:len(floor)]) / nf,
           "seconds": round(time.time() - t0)}
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
