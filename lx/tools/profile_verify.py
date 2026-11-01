"""Where the verify time goes: one chunk of the boot trace through the merged model's chip
layers and output head (the stock path, lx.session.verify), profiled by GPU kernel.

    python tools/profile_verify.py [--weights ../build/hf/model] [--chunks 3]
"""

import argparse
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from lx import hf
from lx import spec as S
from lx.linux import ROOT, Linux
from lx.session import Session


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "build" / "hf" / "model"))
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=16384)
    a = ap.parse_args()
    lx = Linux(chunk=a.chunk, weights=Path(a.weights))
    torch.cuda.empty_cache()
    print(f"model loaded: {torch.cuda.memory_allocated() / 1e9:.1f} GB allocated, "
          f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    s = Session(lx.c, lx.model, lx.reset_machine(), lx.chunk, lx.offset)
    toks = [S.tok("GO")]
    while len(toks) < a.chunks * lx.chunk:
        toks += s.propose(max_records=10_000)
    toks = toks[:a.chunks * lx.chunk]
    print(f"{len(toks)} tokens, chunk {lx.chunk}")
    C = lx.chunk
    r = s.verify(toks[:C])
    assert r["wrong"] == 0, r["first_wrong"]
    m, n_chip = lx.model, s.n_chip
    x = torch.tensor([toks[C:2 * C]], device=s.dev)
    torch.cuda.synchronize()
    t0 = time.time()
    e = m.model.embed_tokens(x + lx.offset)
    h = hf.chip_forward(m, e, s.cache, n_chip)[0]
    torch.cuda.synchronize()
    t1 = time.time()
    pred = hf.head_argmax(m, h, rows=s.rows)
    torch.cuda.synchronize()
    t2 = time.time()
    print(f"chip layers {t1 - t0:.2f}s, head {t2 - t1:.2f}s, {C / (t2 - t0):,.0f} tok/s")
    x = torch.tensor([toks[2 * C:3 * C]], device=s.dev)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        e = m.model.embed_tokens(x + lx.offset)
        h = hf.chip_forward(m, e, s.cache, n_chip)[0]
        pred = hf.head_argmax(m, h, rows=s.rows)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30, max_name_column_width=70))
    print(f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
