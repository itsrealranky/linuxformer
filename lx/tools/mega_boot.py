"""The whole boot on the megakernel (csrc/mega.cu, lx/mega.py), the model alone: from reset to
the shell prompt, each step's input is the step before's argmax, fed back on the GPU (one kernel
launch a chunk; between chunks the host hands the last prediction to the next launch). The
proposer's stream is computed beforehand only to compare with, chunk by chunk as the run goes
(nothing from it is fed to the model), so a wrong token stops the run within one chunk. The
kernel records the token it embeds at each step: the run fails unless the first is GO and every
other is the model's own prediction for it (no keystrokes are typed during the boot).

    python tools/mega_boot.py [--every 262144] [--out results/mega_boot.json]
                              [--snapshot build/snapshots/boot-merged-mega.pt]
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch

from lx import spec as S
from lx.linux import PROMPT, ROOT, Linux, rel
from lx.mega import Mega
from lx.session import Session, console


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "build" / "hf" / "model"))
    ap.add_argument("--every", type=int, default=262144, help="tokens a launch (and comparison)")
    ap.add_argument("--max-tokens", type=int, default=0, help="stop after this many (0: all)")
    ap.add_argument("--out", default=str(ROOT / "lx" / "results" / "mega_boot.json"))
    ap.add_argument("--snapshot", default=str(ROOT / "build" / "snapshots" /
                                              "boot-merged-mega.pt"))
    a = ap.parse_args()
    lx = Linux(weights=Path(a.weights), fast=True)
    s0 = Session(lx.c, lx.model, lx.reset_machine(), lx.chunk, lx.offset, lx.fast)
    t0 = time.time()
    want = [S.tok("GO")] + s0.propose(until=lambda m: m.out.endswith(PROMPT))
    prop = time.time() - t0
    reset = "/tmp/mega_boot_reset.pt"
    s0.save(reset)  # the reset state (and the reference, run to the prompt)
    del s0
    torch.cuda.empty_cache()
    s = Session.load(lx.c, lx.model, reset, lx.chunk, lx.offset)
    t0 = time.time()
    mg = Mega(lx.fast, lx.model, s.cache, lx.offset)
    build_s = time.time() - t0
    n_all = min(len(want), a.max_tokens) if a.max_tokens else len(want)
    print(json.dumps({"reference_tokens": len(want), "running": n_all, "propose_s": round(prop, 1),
                      "mega_build_s": round(build_s, 1), "mega": mg.stats}), flush=True)
    b0, nb = S.FAMILIES["BYTE"]
    torch.cuda.synchronize()
    t0 = time.time()
    stream, first, nbytes, nxt = [], None, 0, want[0]
    inp = torch.zeros(a.every, dtype=torch.int32, device=mg.dev)
    pure = {"inputs_checked": 0, "forced_by_host": {"GO": 1, "keystrokes": 0}, "chunks": 0,
            "chunk_handoffs": 0}
    while len(stream) < n_all:
        a0, m = len(stream), min(a.every, n_all - len(stream))
        preds = mg.run([nxt], m, inputs=inp)
        got = inp[:m].tolist()
        # step 0: GO (the first chunk) or the last chunk's last prediction; the rest: each step
        # before's prediction
        assert got[0] == nxt and (a0 > 0 or nxt == S.tok("GO")), (a0, got[0], nxt)
        assert got[1:] == preds[:-1], next(i for i in range(1, m) if got[i] != preds[i - 1])
        pure["inputs_checked"] += m
        pure["chunks"] += 1
        pure["chunk_handoffs"] += a0 > 0
        stream += got
        nxt = preds[-1]
        if stream[a0:] != want[a0:len(stream)]:
            first = next(i for i in range(a0, len(stream)) if stream[i] != want[i])
        nbytes += sum(b0 <= t < b0 + nb for t in stream[a0:])
        dt = time.time() - t0
        rate = len(stream) / dt
        print(json.dumps({"tokens": len(stream), "of": n_all, "s": round(dt, 1),
                          "tok_per_s": round(rate, 1), "console_bytes": nbytes,
                          "eta_min": round((n_all - len(stream)) / rate / 60, 1),
                          "first_difference": first}), flush=True)
        if first is not None:
            break
    torch.cuda.synchronize()
    dt = time.time() - t0
    s.next, s.last, s.n = nxt, stream[-1], len(stream)
    s._count(stream)
    out = bytes(console(stream))
    git = lambda *x: subprocess.run(["git", *x], capture_output=True, text=True,  # noqa: E731
                                    cwd=ROOT).stdout.strip()
    pure["passed"] = pure["inputs_checked"] == len(stream)
    pure["rule"] = ("the kernel's embedded input at step 0 is GO; at every other step, the "
                    "model's own prediction from the step before (across a chunk boundary, the "
                    "last chunk's last prediction, handed over by the host)")
    res = {"what": "megakernel boot (model alone)", "device": torch.cuda.get_device_name(),
           "commit": git("rev-parse", "HEAD"), "clean_checkout": git("status", "--porcelain") == "",
           "weights": rel(a.weights), "ctas": mg.ncta, "tokens_a_launch": a.every,
           "input_purity": pure,
           "stream_sha256": hashlib.sha256(json.dumps(stream).encode()).hexdigest(),
           "reference_sha256": hashlib.sha256(json.dumps(want).encode()).hexdigest(),
           "tokens": len(stream), "reference_tokens": len(want), "seconds": round(dt, 1),
           "minutes": round(dt / 60, 2), "tok_per_s": round(len(stream) / dt, 1),
           "propose_s": round(prop, 1), "mega_build_s": round(build_s, 1),
           "same_as_reference": first is None and len(stream) == len(want),
           "first_difference": first, "first_difference_tokens":
               None if first is None else {"want": want[first], "got": stream[first]},
           "console_bytes": len(out),
           "console_matches_reference": out == bytes(s.ref.out)[:len(out)] and
               (len(stream) < len(want) or out == bytes(s.ref.out)),
           "console": out.decode(errors="replace"),
           "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1), "mega": mg.stats}
    if res["same_as_reference"] and a.snapshot:
        s.save(a.snapshot)
        res["saved"] = rel(a.snapshot)
    print(json.dumps({k: v for k, v in res.items() if k != "console"}), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
