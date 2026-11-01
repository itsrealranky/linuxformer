"""Two booted snapshots (lx.session.Session.save) bit for bit, and a boot's console against the
reference's: two boots of the same model at different prefill chunk sizes must end in the same
state, every conv window and recurrent state equal to the bit.

    python tests/check_snapshots.py A.pt B.pt [--boot results/X.json] [--out FILE]
"""

import argparse
import json
from pathlib import Path

import torch


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1).view({2: torch.int16, 4: torch.int32}[t.element_size()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--boot", default="", help="a lx.linux --out file whose boot console to check")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    A, B = (torch.load(p, weights_only=False) for p in (a.a, a.b))
    res = {"a": a.a, "b": a.b, "layers": len(A["state"]),
           "dtypes": [str(t.dtype) for t in A["state"][0]],
           "elements": 0, "elements_differ": 0, "differences": []}
    for L, (sa, sb) in enumerate(zip(A["state"], B["state"])):
        for kind, x, y in zip(("conv", "recurrent"), sa, sb):
            assert x.shape == y.shape and x.dtype == y.dtype
            res["elements"] += x.numel()
            ne = bits(x).ne(bits(y))
            res["elements_differ"] += int(ne.sum())
            for i in ne.nonzero()[:20 - len(res["differences"]), 0].tolist():
                at = list(torch.unravel_index(torch.tensor(i), x.shape))
                res["differences"].append({"layer": L, "state": kind,
                                           "index": [int(j) for j in at],
                                           "a": x.reshape(-1)[i].item(),
                                           "b": y.reshape(-1)[i].item()})
    res["host_state_same"] = all(A[k] == B[k] for k in ("next", "last", "n", "waiting"))
    res["reference_ram_same"] = bytes(A["ref"].ram) == bytes(B["ref"].ram)
    res["tokens"] = A["n"]
    ok = res["elements_differ"] == 0 and res["host_state_same"] and res["reference_ram_same"]
    if a.boot:
        boot = next(r for r in json.loads(Path(a.boot).read_text())["runs"] if r["what"] == "boot")
        ref = bytes(B["ref"].out).decode(errors="replace")
        res["boot"] = a.boot
        res["console_bytes"] = len(boot["out"])
        res["console_matches_reference"] = boot["out"] == ref
        ok = ok and res["console_matches_reference"]
    res["ok"] = ok
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
