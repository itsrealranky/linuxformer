"""How far a pure decode could go with self-speculation: drafts guessed from the stream's own past
tokens only (no reference, nothing about the CPU), each round checked by one prefill call that
keeps the drafted prefix the model agrees with, plus the model's own next token. Simulated on
the boot's token stream (which the model reproduces exactly): rounds needed per drafter.

    python tools/draft_sim.py [--draft 64]
"""

import argparse
import json
import time

import numpy as np

from lx import merge
from lx import ref as R
from lx.linux import CMDLINE, IMAGE, PROMPT

CACHE = IMAGE.parent / "boot_stream.npy"


def boot_stream() -> np.ndarray:
    if CACHE.exists():
        return np.load(CACHE)
    cfg = merge.config()
    ram, regs = R.linux_ram(IMAGE.read_bytes(), cfg.ram_size, CMDLINE)
    m = R.Machine(ram, cfg.text_end, regs)
    toks = []
    while not m.out.endswith(PROMPT):
        toks += m.step()
    t = np.array(toks, dtype=np.int64)
    np.save(CACHE, t)
    return t


def simulate(t: np.ndarray, D: int, how: str, n: int = 6) -> dict:
    """Rounds to produce t with drafts of up to D tokens. how: "copy" (continue after the latest
    earlier occurrence of the last n tokens), "period" (copy from p back, p = distance to the
    latest earlier occurrence of the last token), "delta" (period, then 2 t[i-p] - t[i-2p])."""
    N = len(t)
    tl = t.tolist()
    seen = {}  # copy: n-gram ending at q -> q + 1; else: token -> its latest position q
    i, rounds, hist = 1, 0, np.zeros(D + 2, dtype=np.int64)
    q = 0  # positions < q are indexed; only those before the current suffix are

    while i < N:
        while q < i - 1:
            if how == "copy":
                if q + 1 >= n:
                    seen[tuple(tl[q + 1 - n:q + 1])] = q + 1
            else:
                seen[tl[q]] = q
            q += 1
        if how == "copy":
            j = seen.get(tuple(tl[i - n:i])) if i >= n else None
        else:
            j = seen.get(tl[i - 1])
            j = None if j is None else j + 1  # period p = (i - 1) - previous occurrence
        a = 0
        if j is not None:
            p = i - j
            while a < D and i + a < N:  # an accepted draft token equals the stream's token
                b = i + a - p
                x = tl[b]
                if how == "delta" and b - p >= 0:
                    x = 2 * x - tl[b - p]
                if x != tl[i + a]:
                    break
                a += 1
        hist[a] += 1
        rounds += 1
        i += a + 1
    return {"how": how, "draft": D, "rounds": rounds, "tokens_per_round": round(N / rounds, 2),
            "rounds_accepting_none": int(hist[0]),
            "tokens_in_rounds_accepting_8_plus": int(sum((k + 1) * hist[k] for k in range(8, D + 1)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", type=int, nargs="+", default=[16, 64, 256])
    ap.add_argument("--how", nargs="+", default=["copy", "period", "delta"])
    a = ap.parse_args()
    t0 = time.time()
    t = boot_stream()
    print(f"stream: {len(t):,} tokens ({time.time() - t0:.0f} s)", flush=True)
    for how in a.how:
        for D in a.draft:
            t0 = time.time()
            r = simulate(t, D, how)
            r["s"] = round(time.time() - t0)
            print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
