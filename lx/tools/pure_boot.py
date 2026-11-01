"""The whole boot on the pure path: from reset to the shell prompt, the model is fed nothing but
its own predictions (one token at a time, no reference, no prefill). The proposer's stream is
computed beforehand only to compare with, chunk by chunk as the run goes (nothing from it is fed
to the model), so a wrong token stops the run within one chunk.

    python tools/pure_boot.py [--stock] [--every 262144] [--out results/pure_boot.json]
                              [--snapshot build/snapshots/boot-merged-pure.pt]
"""

import argparse
import json
import time
from pathlib import Path

import torch

from lx import spec as S
from lx.linux import PROMPT, ROOT, Linux
from lx.session import Session, console


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "build" / "hf" / "model"))
    ap.add_argument("--stock", action="store_true", help="the stock decode instead of lx.fast")
    ap.add_argument("--every", type=int, default=262144, help="tokens between comparisons")
    ap.add_argument("--max-tokens", type=int, default=0, help="stop after this many (0: all)")
    ap.add_argument("--out", default=str(ROOT / "lx" / "results" / "pure_boot.json"))
    ap.add_argument("--snapshot", default=str(ROOT / "build" / "snapshots" /
                                              "boot-merged-pure.pt"))
    ap.add_argument("--idle-skip", action="store_true", help="lx.fast.Fast.idle_skip")
    a = ap.parse_args()
    lx = Linux(weights=Path(a.weights), fast=not a.stock)
    if lx.fast:
        lx.fast.compile, lx.fast.idle_skip = True, a.idle_skip
    s = Session(lx.c, lx.model, lx.reset_machine(), lx.chunk, lx.offset, lx.fast)
    t0 = time.time()
    want = [S.tok("GO")] + s.propose(until=lambda m: m.out.endswith(PROMPT))
    prop = time.time() - t0
    n_all = min(len(want), a.max_tokens) if a.max_tokens else len(want)
    print(json.dumps({"reference_tokens": len(want), "running": n_all,
                      "propose_s": round(prop, 1)}), flush=True)
    b0, nb = S.FAMILIES["BYTE"]
    torch.cuda.synchronize()
    t0 = time.time()
    s.step(want[0])
    stream, first, nbytes = [want[0]], None, 0
    while len(stream) < n_all:
        a0 = len(stream)
        stream += s.free(min(a.every, n_all - a0))
        if stream[a0:] != want[a0:len(stream)]:
            first = next(i for i in range(a0, len(stream)) if stream[i] != want[i])
        nbytes += sum(b0 <= t < b0 + nb for t in stream[a0:])
        dt = time.time() - t0
        rate = (len(stream) - 1) / dt
        print(json.dumps({"tokens": len(stream), "of": n_all, "s": round(dt, 1),
                          "tok_per_s": round(rate, 1), "console_bytes": nbytes,
                          "eta_h": round((n_all - len(stream)) / rate / 3600, 2),
                          "first_difference": first}), flush=True)
        if first is not None:
            break
    torch.cuda.synchronize()
    dt = time.time() - t0
    out = bytes(console(stream))
    res = {"what": "pure boot", "device": torch.cuda.get_device_name(), "fast": bool(lx.fast),
           "idle_skip": a.idle_skip,
           "tokens": len(stream), "reference_tokens": len(want), "seconds": round(dt, 1),
           "tok_per_s": round((len(stream) - 1) / dt, 1), "propose_s": round(prop, 1),
           "same_as_reference": first is None and len(stream) == len(want),
           "first_difference": first, "first_difference_tokens":
               None if first is None else {"want": want[first], "got": stream[first]},
           "console_bytes": len(out),
           "console_matches_reference": out == bytes(s.ref.out)[:len(out)] and
               (len(stream) < len(want) or out == bytes(s.ref.out)),
           "console": out.decode(errors="replace"),
           "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
    if res["same_as_reference"] and a.snapshot:
        s.save(a.snapshot)
        res["saved"] = a.snapshot
    print(json.dumps({k: v for k, v in res.items() if k != "console"}), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
