"""lx.session on tests/uarttest.S under host input (check_core.UART_SCHEDULES), on a GPU:

  verify   the reference's trace run as prefill in random pieces: 0 wrong predictions, and the
           bytes the model predicts are the program's output
  snapshot the session saved halfway and loaded into a fresh one, which finishes the run
  pure     the model alone (its argmax fed back, one CUDA graph per token; the host types at
           the same record boundaries, counted from the stream): the stream must equal the
           reference's trace token for token

    python tests/check_session.py [--text 0x40] [NAME ...]
"""

import argparse
import random
import sys
import tempfile
import time

import torch
from check_core import RV_RAM, UART_SCHEDULES, uart_trace

from lx import core, hf
from lx import spec as S
from lx.session import Session, _final_ids, _is_typed, console


def pure_run(c, model, m0, sched: dict, max_tokens: int = 20000):
    """The pure path from reset: (stream, seconds)."""
    s = Session(c, model, m0)
    fin = _final_ids()
    stream = [S.tok("GO")]
    s.step(stream[0])
    records = 0
    t0 = time.time()
    while len(stream) < max_tokens:
        if records in sched:
            stream += s.type(sched.pop(records))
        t = s.next
        s.step(t)
        stream.append(t)
        if fin[t] and not _is_typed(t):
            records += 1
        if t == S.tok("HALT"):
            break
    torch.cuda.synchronize()
    return stream, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--text", type=lambda x: int(x, 0), default=0x40)
    a = ap.parse_args()
    te = a.text
    c = core.build(core.Config(ram_size=RV_RAM, text_end=te))
    model, lay = hf.build_model(c)
    print(f"T={te:#x}: {c.stats()}, hidden {lay.hidden}", flush=True)
    rng = random.Random(0)
    bad = 0
    for name, sched in UART_SCHEDULES.items():
        if a.names and name not in a.names:
            continue
        m0, toks, m = uart_trace(te, sched)
        seq = [S.tok("GO")] + toks
        # verify, in random pieces, with a save / load halfway
        cuts = sorted(rng.sample(range(1, len(seq)), 5))
        half = cuts[2]
        s = Session(c, model, m0, chunk=rng.choice([64, 256, 4096]))
        wrong, out = 0, b""
        with tempfile.TemporaryDirectory() as d:
            for i, (lo, hi) in enumerate(zip([0] + cuts, cuts + [len(seq)])):
                if lo == half:
                    s.save(f"{d}/s.pt")
                    s = Session.load(c, model, f"{d}/s.pt", chunk=s.chunk)
                r = s.verify(seq[lo:hi])
                wrong += r["wrong"]
                out += r["out"]
                if r["wrong"]:
                    print(f"    wrong in [{lo}, {hi}): {r['first_wrong']}")
        ok_v = wrong == 0 and out == bytes(m.out)
        # the pure path
        stream, dt = pure_run(c, model, m0, dict(sched))
        n_same = next((i for i, (x, y) in enumerate(zip(stream, seq)) if x != y),
                      min(len(stream), len(seq)))
        ok_p = stream == seq
        print(f"  {name:12s} {len(seq):5d} tokens  verify {'ok' if ok_v else 'FAILED'} "
              f"(cuts {cuts}, save/load at {half}, out {out[:16]!r})  pure "
              f"{'ok' if ok_p else f'FAILED at {n_same}'} ({len(stream) / dt:.0f} tok/s, "
              f"console {console(stream)[:16]!r})", flush=True)
        if not ok_p:
            print(f"    pure {stream[max(0, n_same - 3):n_same + 3]} vs trace "
                  f"{seq[max(0, n_same - 3):n_same + 3]}")
        bad += (not ok_v) + (not ok_p)
    print("all ok" if not bad else f"{bad} FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
