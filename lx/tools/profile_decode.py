"""Per-op GPU time of lx.fast's single-token decode, from the booted snapshot.

    python tools/profile_decode.py [--tokens 20]
"""

import argparse
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from lx.linux import ROOT, Linux
from lx.session import Session


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "build" / "hf" / "model"))
    ap.add_argument("--snapshot", default=str(ROOT / "build" / "snapshots" / "boot-merged.pt"))
    ap.add_argument("--tokens", type=int, default=20)
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args()
    lx = Linux(weights=Path(a.weights), fast=True)
    lx.fast.compile = a.compile
    s = Session.load(lx.c, lx.model, a.snapshot, lx.chunk, lx.offset, lx.fast)
    s.free(50)
    ids = torch.zeros((1, 1), dtype=torch.long, device=s.dev)
    for _ in range(3):
        lx.fast.head_argmax(lx.fast.decode(lx.model.model.embed_tokens(ids)[0], s.cache))
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(a.tokens):
            e = lx.model.model.embed_tokens(ids)
            lx.fast.head_argmax(lx.fast.decode(e[0], s.cache))
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30, max_name_column_width=70))


if __name__ == "__main__":
    main()
