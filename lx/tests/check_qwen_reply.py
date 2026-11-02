"""Qwen's first reply to a question (greedy, before any shell output) with the merged model built
in memory, loaded from a merge.save() directory, and loaded with lx.fast: the three must agree.

    python tests/check_qwen_reply.py [--weights ../build/hf/model] [--out FILE] [QUESTION]
"""

import argparse
import gc
import json
from pathlib import Path

import torch

from lx.chat import Chat
from lx.linux import ROOT, Linux

FIB = ("Write a shell script that prints the first 20 Fibonacci numbers, save it as /tmp/fib.sh, "
       "run it on the machine, and tell me the 20th number.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", default=FIB)
    ap.add_argument("--weights", default=str(ROOT / "build" / "hf" / "model"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    replies = {}
    for name, kw in [("built", {"merged": True}), ("export", {"weights": Path(a.weights)}),
                     ("export, lx.fast", {"weights": Path(a.weights), "fast": True})]:
        lx = Linux(**kw)
        text, n, _ = Chat(lx)._generate([{"role": "user", "content": a.question}], 768)
        replies[name] = {"text": text, "tokens": n}
        print(name, repr(text), flush=True)
        del lx
        gc.collect()
        torch.cuda.empty_cache()
    same = len({r["text"] for r in replies.values()}) == 1
    res = {"question": a.question, "replies": replies, "all_same": same}
    print("OK" if same else "FAILED")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
