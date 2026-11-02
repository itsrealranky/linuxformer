"""Model-alone shell commands (Linux.boot_alone / run_alone, lx.mega): from a booted snapshot,
type a command of several lines (more than one receive-FIFO batch), decode the model's own
predictions until its console shows the prompt, then compare with the reference typed at the
same places.

    python tests/check_alone.py [--snapshot ../build/snapshots/boot-merged-mega.pt] [--cmd CMD]
                                [--out FILE]
"""

import argparse
import json
import time
from pathlib import Path

import torch

from lx.linux import ROOT, Linux

CMD = "a=6\nb=7\necho product $((a*b)) of $a and $b, and this line is long enough to wrap\necho ok"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "build" / "hf" / "model"))
    ap.add_argument("--snapshot", default=str(ROOT / "build" / "snapshots" /
                                              "boot-merged-mega.pt"))
    ap.add_argument("--cmd", default=CMD)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    lx = Linux(weights=Path(a.weights), fast=True)
    load_s = time.time() - t0
    b = lx.boot_alone(Path(a.snapshot))
    r = lx.run(a.cmd)
    res = {"cmd": a.cmd, "load_s": round(load_s, 1),
           "boot": {k: v for k, v in b.items() if k != "out"},
           **{k: r[k] for k in ("tokens", "seconds", "tok_per_s", "same_as_reference",
                                "first_difference", "console_matches_reference",
                                "input_purity")},
           "launches": len(r["timeline"]), "output": r["output"].decode(errors="replace")}
    print(json.dumps(res, indent=1))
    ok = r["same_as_reference"] and r["input_purity"]["passed"] and (
        a.cmd != CMD or b"ok\n" in r["output"])
    print("OK" if ok else "FAILED")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
