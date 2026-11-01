"""The LX circuit (lx/core.py) teacher-forced against the reference machine (lx/ref.py): every
position's prediction must be the trace's next token, with the IR's exactness checks on.

    python tests/check_core.py stats                 # build the circuit, print its shape
    python tests/check_core.py rvtests [NAME ...]    # riscv-tests (RAM 64 KB), two fetch modes
    python tests/check_core.py linux N [SKIP]        # N records of the Linux boot, after SKIP
    python tests/check_core.py uart                  # UART input (tests/uarttest.S + KEY tokens)
"""

import copy
import sys
import time
from pathlib import Path

from lx import core, ir
from lx import ref as R
from lx import spec as S

ROOT = Path(__file__).resolve().parents[2]
RV_RAM = 0x10000
RV_TEXT = (0, 0x1BC)  # every riscv-test's code fetched from RAM, or its first 111 words from ROM
LINUX_IMAGE = ROOT / "kernel" / "Image"
LINUX_CMD = "console=ttyS0 rdinit=/bin/sh"


def rv_trace(path: Path, text_end: int, max_tokens: int = 30000):
    """(machine at reset, tokens up to the tohost store) of a riscv-test, or None if it uses
    something the circuit does not implement."""
    tohost = int(path.with_suffix(".tohost").read_text(), 16) - S.RAM_BASE
    ram, regs = R.linux_ram(path.read_bytes(), RV_RAM, None)
    m = R.Machine(ram, text_end, regs)
    m0 = copy.deepcopy(m)
    toks = []
    try:
        while len(toks) < max_tokens:
            m.last_store = None
            toks += m.step()
            if m.last_store == tohost:
                break
    except R.Unsupported:
        return None
    return m0, toks


# tests/uarttest.S under host input: {record index: bytes typed before that record}
UART_SCHEDULES = {
    "one-by-one": {20: b"a", 60: b"b", 100: b"\xff", 140: b"\x80", 180: b"\n"},
    "burst": {10: b"hello x!\n"},
    "drop-empty": {10: b"ab", 13: b"cd", 50: b"x", 90: b"yz", 200: b"\n"},
    "refill": {10: b"abc", 16: b"def", 30: b"g", 120: b"hij", 124: b"k\n"},
    "full": {5: bytes(range(0x21, 0x21 + 63)) + b"\n"},
}


def uart_trace(text_end: int, schedule: dict, max_records: int = 20000):
    """(machine at reset, tokens with the host's KEY0 / KEY tokens, final machine)."""
    img = (ROOT / "build" / "uarttest" / "uarttest.bin").read_bytes()
    ram, regs = R.linux_ram(img, RV_RAM, None)
    m = R.Machine(ram, text_end, regs)
    m0 = copy.deepcopy(m)
    toks = []
    for rec in range(max_records):
        for b in schedule.get(rec, b""):
            toks += m.key(b)
        if m.halted:
            break
        toks += m.step()
    return m0, toks, m


def linux_machine(cfg: core.Config) -> R.Machine:
    ram, regs = R.linux_ram(LINUX_IMAGE.read_bytes(), cfg.ram_size, LINUX_CMD)
    return R.Machine(ram, cfg.text_end, regs)


def linux_boot(cfg: core.Config, prompt: bytes = b"/ # "):
    """(machine at reset, every token until the shell prompt is printed, the final machine, the
    number of records)."""
    m = linux_machine(cfg)
    m0 = copy.deepcopy(m)
    toks, n = [], 0
    while not m.out.endswith(prompt):
        toks += m.step()
        n += 1
    return m0, toks, m, n


def linux_trace(cfg: core.Config, n: int, skip: int = 0):
    """(machine after ``skip`` records of the boot, the next ``n`` records' tokens)."""
    m = linux_machine(cfg)
    for _ in range(skip):
        m.step()
    m0 = copy.deepcopy(m)
    toks = []
    for _ in range(n):
        toks += m.step()
    return m0, toks


def check(c, m0, toks, kernel=False):
    """Mismatches [(position, expected, predicted)] of a teacher-forced run from m0."""
    core.snapshot(c, m0)
    seq = [S.tok("GO")] + toks
    mat = ir.evaluate(c, seq, check=True, kernel=kernel)
    pred = core.predictions(c, mat)
    bad = [(i, seq[i + 1], pred[i]) for i in range(len(seq) - 1)
           if pred[i] != seq[i + 1] and S.family(seq[i + 1])[0] not in S.INPUT]
    return bad, mat, seq


def show(bad, seq, n=3):
    for i, want, got in bad[:n]:
        ctx = " ".join(S.name(t) for t in seq[max(0, i - 5):i + 1])
        print(f"    at {i}: ... {ctx} -> want {S.name(want)}, got "
              f"{S.name(got) if got >= 0 else got}")


def main():
    cmd = sys.argv[1]
    if cmd == "stats":
        t0 = time.time()
        cfg = core.Config()
        c = core.build(cfg)
        print(f"build {time.time() - t0:.1f}s", c.stats())
        heads = sorted(c.heads.items())
        print("heads per attention sublayer:", {(s + 1) // 2: n for s, n in heads})
        print("ram blocks", len(c.ram_blocks))
    elif cmd == "rvtests":
        names = sys.argv[2:]
        paths = sorted((ROOT / "build" / "rvtests").glob("*.bin"))
        if names:
            paths = [p for p in paths if any(n in p.stem for n in names)]
        tot = {"ok": 0, "bad": 0, "unsup": 0}
        for te in RV_TEXT:
            t0 = time.time()
            c = core.build(core.Config(ram_size=RV_RAM, text_end=te))
            print(f"T={te:#x}: built in {time.time() - t0:.1f}s, {c.stats()}")
            for p in paths:
                tr = rv_trace(p, te)
                if tr is None:
                    tot["unsup"] += 1
                    continue
                m0, toks = tr
                t1 = time.time()
                try:
                    bad, _, seq = check(c, m0, toks)
                except ir.ExactnessError as e:
                    print(f"  {p.stem:24s} EXACTNESS {e}")
                    tot["bad"] += 1
                    continue
                tot["ok" if not bad else "bad"] += 1
                print(f"  {p.stem:24s} {len(toks):6d} tokens {time.time() - t1:5.1f}s "
                      f"{'ok' if not bad else f'{len(bad)} MISMATCHES'}")
                if bad:
                    show(bad, seq)
        print(f"ok {tot['ok']}, bad {tot['bad']}, unsupported {tot['unsup']}")
    elif cmd == "uart":
        bad_runs = 0
        for te in (0, 0x40):
            c = core.build(core.Config(ram_size=RV_RAM, text_end=te))
            for name, sched in UART_SCHEDULES.items():
                m0, toks, m = uart_trace(te, sched)
                typed = b"".join(sched.values())
                bad, _, seq = check(c, m0, toks)
                n_in = sum(S.family(t)[0] in S.INPUT for t in toks)
                print(f"  T={te:#x} {name:12s} {len(toks):5d} tokens ({n_in} typed) "
                      f"out={bytes(m.out)!r} halted={m.halted} "
                      f"{'ok' if not bad else f'{len(bad)} MISMATCHES'}", flush=True)
                bad_runs += bool(bad) or not m.halted or not typed
                if bad:
                    show(bad, seq)
        print("all ok" if not bad_runs else f"{bad_runs} FAILED")
    elif cmd == "linux":
        n, skip = int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 0
        cfg = core.Config()
        t0 = time.time()
        c = core.build(cfg)
        print(f"built in {time.time() - t0:.1f}s, {c.stats()}", flush=True)
        t0 = time.time()
        m0, toks = linux_trace(cfg, n, skip)
        print(f"records {skip}..{skip + n}: {len(toks)} tokens ({time.time() - t0:.1f}s)",
              flush=True)
        t0 = time.time()
        bad, _, seq = check(c, m0, toks)
        print(f"{len(toks)} tokens {time.time() - t0:.1f}s "
              f"{'ok' if not bad else f'{len(bad)} MISMATCHES'}")
        if bad:
            show(bad, seq)


if __name__ == "__main__":
    main()
