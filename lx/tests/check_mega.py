"""lx.mega's decode (one persistent kernel, the model's own predictions fed back on the GPU)
against the stock decode, from a point of the boot: N tokens both ways, the two streams and the
two states after them must agree bit for bit; then a longer run checked against the reference.

    python tests/check_mega.py --at 0                 # from reset (the boot's first token)
    python tests/check_mega.py --at 3000000           # mid-boot (verified prefill up to there)
    python tests/check_mega.py --snapshot ../build/snapshots/boot-merged.pt --cmd "echo hi"
"""

import argparse
import json
import time
from pathlib import Path

import torch

from lx import spec as S
from lx.linux import PROMPT, ROOT, Linux
from lx.mega import Mega
from lx.session import Session
from tests.check_fast_decode import pure, reference


def typed_tokens(s: Session, data: bytes) -> list[int]:
    """The tokens Session.type(data) feeds, without running them."""
    w, out = s.waiting, []
    for b in data:
        assert w < S.RX_SLOTS
        out.append(S.tok("KEY0" if w == 0 else "KEY", b))
        w += 1
    return out


def state_diff(a: Session, b: Session) -> int:
    return sum(int(x.ne(y).sum()) for (cs, rs), (cf, rf) in zip(a._state(), b._state())
               for x, y in ((cs, cf), (rs, rf)))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "build" / "hf" / "model"))
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--at", type=int, default=None, help="boot token to start from (prefill)")
    ap.add_argument("--cmd", default="")
    ap.add_argument("--stock", type=int, default=2000)
    ap.add_argument("--long", type=int, default=50000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lx = Linux(weights=Path(a.weights), fast=True)
    lx.fast.compile, lx.fast.idle_skip = True, True
    data = (a.cmd.encode() + b"\n") if a.cmd else b""
    if a.snapshot:
        snap = a.snapshot
        want = None
    else:
        s0 = Session(lx.c, lx.model, lx.reset_machine(), lx.chunk, lx.offset, lx.fast)
        want = [S.tok("GO")] + s0.propose(until=lambda m: m.out.endswith(PROMPT))
        at = a.at or 0
        if at:
            r = s0.verify(want[:at])
            assert r["wrong"] == 0, r["first_wrong"]
        snap = f"/tmp/mega_at_{at}.pt"
        s0.save(snap)
        del s0
        torch.cuda.empty_cache()
    ss = Session.load(lx.c, lx.model, snap, lx.chunk, lx.offset)
    sm = Session.load(lx.c, lx.model, snap, lx.chunk, lx.offset)
    if want is not None:
        at = a.at or 0
        ref = want[at:]
        if at == 0:  # nothing run yet: the first input is GO
            ss.next = sm.next = want[0]
            ss.last = sm.last = want[0]
    else:
        ref = reference(ss.ref, data, a.long)
    t0 = time.time()
    mg = Mega(lx.fast, lx.model, sm.cache, lx.offset)
    build_s = time.time() - t0
    n = a.stock
    # the stock decode
    st, t_s = pure(ss, data, n)
    # the megakernel: the same n tokens, then the states
    forced = typed_tokens(sm, data) if data else [sm.next]
    torch.cuda.synchronize()
    t0 = time.time()
    p1 = mg.run(forced, n)
    torch.cuda.synchronize()
    t1 = time.time() - t0
    mt = forced + p1[len(forced) - 1:n - 1]
    same = mt == st
    first_vs_stock = None if same else next(i for i, (x, y) in enumerate(zip(mt, st)) if x != y)
    diff = state_diff(ss, sm)
    # longer: continue from the last prediction, checked against the reference
    m2 = max(0, a.long - n)
    torch.cuda.synchronize()
    t0 = time.time()
    p2 = mg.run([p1[-1]], m2) if m2 else []
    torch.cuda.synchronize()
    t2 = time.time() - t0
    stream = mt + ([p1[-1]] + p2[:-1] if m2 else [])
    want_n = ref[:len(stream)]
    first = next((i for i, (x, y) in enumerate(zip(stream, want_n)) if x != y), None)
    lo, nb = S.FAMILIES["BYTE"]
    res = {"start": a.snapshot or f"boot token {a.at or 0}", "cmd": a.cmd, "mega": mg.stats,
           "build_s": round(build_s, 1), "stock_tokens": n, "stock_tok_per_s": round(n / t_s, 1),
           "mega_first_tok_per_s": round(n / t1, 1),
           "streams_identical_first_n": same, "first_difference_vs_stock": first_vs_stock,
           "state_elements_differ_after_n": diff,
           "mega_tokens": len(stream), "mega_long_tok_per_s": round(m2 / t2, 1) if m2 else None,
           "mega_same_as_reference": first is None and len(stream) <= len(ref),
           "first_difference_vs_reference": first,
           "console": bytes(t - lo for t in stream if lo <= t < lo + nb).decode(errors="replace")}
    if first_vs_stock is not None:
        i = first_vs_stock
        res["at_first_difference"] = {"stock": st[i], "mega": mt[i], "context": st[max(0, i - 5):i]}
    print(json.dumps({k: v for k, v in res.items() if k != "console"}, indent=1))
    print("console:", res["console"][-300:])
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
