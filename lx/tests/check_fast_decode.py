"""lx.fast's decode (the pure path: the model's own predictions fed back, no reference) against
the stock decode, from the booted snapshot: type a command, then N tokens both ways. The two
streams, the two states after them and the reference's stream must all agree. Then the speed
of the fast decode over a longer run, still checked against the reference.

    python tests/check_fast_decode.py [--snapshot ../build/snapshots/boot-merged.pt]
                                      [--stock 4000] [--fast 200000] [--cmd "echo hi"]
"""

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from lx import spec as S
from lx.linux import ROOT, Linux
from lx.session import Session


def pure(s: Session, data: bytes, n: int):
    torch.cuda.synchronize()
    t0 = time.time()
    out = s.type(data)
    out += s.free(n - len(out))
    torch.cuda.synchronize()
    return out, time.time() - t0


def reference(ref, data: bytes, n: int) -> list:
    ref = copy.deepcopy(ref)
    want = []
    for b in data:
        want += ref.key(b)
    while len(want) < n:
        want += ref.step()
    return want[:n]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "build" / "hf" / "model"))
    ap.add_argument("--snapshot", default=str(ROOT / "build" / "snapshots" / "boot-merged.pt"))
    ap.add_argument("--cmd", default="echo hi")
    ap.add_argument("--stock", type=int, default=4000)
    ap.add_argument("--fast", type=int, default=200_000)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--idle-skip", action="store_true", help="lx.fast.Fast.idle_skip")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lx = Linux(weights=Path(a.weights), fast=True)
    lx.fast.compile, lx.fast.idle_skip = a.compile, a.idle_skip
    data = a.cmd.encode() + b"\n"
    ss = Session.load(lx.c, lx.model, a.snapshot, lx.chunk, lx.offset)
    sf = Session.load(lx.c, lx.model, a.snapshot, lx.chunk, lx.offset, lx.fast)
    n = a.stock
    st, t_s = pure(ss, data, n)
    ft, t_f1 = pure(sf, data, n)
    same_stream = st == ft
    diff = sum(int(x.ne(y).sum()) for (cs, rs), (cf, rf) in zip(ss._state(), sf._state())
               for x, y in ((cs, cf), (rs, rf)))
    t0 = time.time()
    ft += sf.free(a.fast - n)
    torch.cuda.synchronize()
    t_f = t_f1 + time.time() - t0
    want = reference(ss.ref, data, a.fast)
    first = next((i for i, (x, y) in enumerate(zip(ft, want)) if x != y), None)
    lo, nb = S.FAMILIES["BYTE"]
    res = {"cmd": a.cmd, "compile": a.compile, "idle_skip": a.idle_skip, "stock_tokens": n,
           "stock_s": round(t_s, 1),
           "stock_tok_per_s": round(n / t_s, 1), "fast_tokens": len(ft), "fast_s": round(t_f, 1),
           "fast_tok_per_s": round(len(ft) / t_f, 1),
           "fast_steady_tok_per_s": round((len(ft) - n) / (t_f - t_f1), 1),
           "streams_identical_first_n": same_stream, "state_elements_differ_after_n": diff,
           "fast_same_as_reference": first is None and len(ft) == len(want),
           "first_difference": first,
           "console": bytes(t - lo for t in ft if lo <= t < lo + nb).decode(errors="replace")}
    print(json.dumps(res))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
