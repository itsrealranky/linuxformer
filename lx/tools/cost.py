"""What shell commands cost the machine, on the reference alone (no GPU): from the booted
snapshot's proposer state, type each command and run until the prompt returns. Each command
starts from the booted state, not after the previous one.

    python tools/cost.py [--snapshot FILE] [--tok-per-s 9300] CMD ...
"""

import argparse
import copy
import time
from pathlib import Path

import torch

from lx.linux import ROOT, type_and_run


def run(m, cmd: str, max_records: int = 200_000_000) -> tuple[int, int, bytes]:
    """(tokens, instructions, console output) of typing cmd + newline on machine m."""
    o0, r0 = len(m.out), m.instret
    toks = type_and_run(m, cmd.encode().rstrip(b"\n") + b"\n", max_records)
    return len(toks), m.instret - r0, bytes(m.out[o0:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmds", nargs="+")
    ap.add_argument("--snapshot", default=str(ROOT / "build" / "snapshots" / "boot-merged.pt"))
    ap.add_argument("--tok-per-s", type=float, default=9300, help="verify speed, merged model")
    a = ap.parse_args()
    m0 = torch.load(a.snapshot, weights_only=False)["ref"]
    for cmd in a.cmds:
        m = copy.deepcopy(m0)
        t0 = time.time()
        toks, ins, out = run(m, cmd)
        out = out.decode(errors="replace").replace("\r\n", "\n")
        print(f"{toks:>11,} tokens {ins:>11,} instructions  ~{toks / a.tok_per_s / 60:5.1f} min verified"
              f"  ~{toks / 81 / 3600:6.1f} h pure  ({time.time() - t0:.0f}s on the reference)  {cmd!r}")
        for ln in out.splitlines()[1:-1][:12]:
            print(f"      | {ln}")


if __name__ == "__main__":
    main()
