"""lx/ref.py against ref/lxtrace (the unmodified mini-rv32ima.h core): the same tokens and the same
final state, on riscv-tests or on the Linux boot.

    python tests/check_ref.py rvtests            # every built riscv-test, both fetch modes
    python tests/check_ref.py linux N            # the first N records of the Linux boot
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from lx import ref as R
from lx import spec as S

ROOT = Path(__file__).resolve().parents[2]
LXTRACE = ROOT / "ref" / "lxtrace"


def c_trace(args: list[str]) -> tuple[list[int], np.ndarray, bytes, str]:
    with tempfile.TemporaryDirectory() as d:
        tf, sf = f"{d}/t.bin", f"{d}/s.bin"
        out = subprocess.run([str(LXTRACE), *args, "-o", tf, "-S", sf], capture_output=True,
                             text=True).stdout
        toks = np.fromfile(tf, dtype="<u4").tolist() if Path(tf).exists() else []
        if not Path(sf).exists():
            return toks, None, b"", out
        raw = Path(sf).read_bytes()
    return toks, np.frombuffer(raw[:192], dtype="<u4"), raw[192:], out


def py_state(m: R.Machine) -> np.ndarray:
    c = m.csr
    vals = m.x + [m.pc, c["mstatus"], c["mie"], c["mtvec"], c["mscratch"], c["mepc"],
                  c["mcause"], c["mtval"], (c["mip"] & ~0x80) | (m.mtip(m.T) << 7), m.priv,
                  m.res, m.T, m.mlo, m.mhi, m.anchor, m.n]
    return np.array([v & R.M32 for v in vals], dtype=np.uint32)


STATE_NAMES = [f"x{i}" for i in range(32)] + ["pc", "mstatus", "mie", "mtvec", "mscratch", "mepc",
                                               "mcause", "mtval", "mip", "priv", "res", "time",
                                               "mtimecmp", "mtimecmph", "anchor", "n"]


def compare(name: str, ctoks, cstate, cram, m: R.Machine, ptoks) -> bool:
    ok = True
    if ctoks != ptoks:
        i = next((i for i, (a, b) in enumerate(zip(ctoks, ptoks)) if a != b),
                 min(len(ctoks), len(ptoks)))
        lo = max(0, i - 6)
        print(f"{name}: tokens differ at {i} (c {len(ctoks)}, py {len(ptoks)})")
        print("   c:", " ".join(S.name(t) for t in ctoks[lo:i + 4]))
        print("  py:", " ".join(S.name(t) for t in ptoks[lo:i + 4]))
        ok = False
    ps = py_state(m)
    # mip: the C core stores the bit the latest step set; compare the stored bits apart from MTIP
    cs = cstate.copy()
    k = STATE_NAMES.index("mip")
    cs[k] &= ~np.uint32(0x80)
    ps[k] &= ~np.uint32(0x80)
    bad = [STATE_NAMES[i] for i in np.nonzero(cs != ps)[0]]
    if bad:
        print(f"{name}: state differs: " + ", ".join(f"{b} c={cs[STATE_NAMES.index(b)]:#x} "
                                                    f"py={ps[STATE_NAMES.index(b)]:#x}" for b in bad))
        ok = False
    # RAM: the C image also holds the core's own state at the top (STATE_BYTES)
    top = m.size - R.STATE_BYTES
    if cram[:top] != bytes(m.ram[:top]):
        diff = next(i for i in range(top) if cram[i] != m.ram[i])
        print(f"{name}: RAM differs at offset {diff:#x}")
        ok = False
    return ok


def run_test(path: Path, text_end: int, ram_size: int = 0x10000) -> bool:
    tohost = int(path.with_suffix(".tohost").read_text(), 16)
    args = ["-f", str(path), "-m", hex(ram_size), "-T", hex(text_end), "-H", hex(tohost),
            "-c", "200000"]
    ctoks, cstate, cram, out = c_trace(args)
    if "UNSUPPORTED" in out:
        return None
    ram, regs = R.linux_ram(path.read_bytes(), ram_size, None)
    m = R.Machine(ram, text_end, regs)
    toks = []
    th = tohost - S.RAM_BASE
    while len(toks) < len(ctoks) + 64:
        m.last_store = None
        toks += m.step()
        if m.last_store == th:
            break
    name = f"{path.stem} T={text_end:#x}"
    return compare(name, ctoks, cstate, cram, m, toks), out.split()[1] if out.startswith(
        "TOHOST") else "?"


def main():
    if sys.argv[1] == "rvtests":
        d = ROOT / "build" / "rvtests"
        n_ok = n_bad = n_uns = 0
        for p in sorted(d.glob("*.bin")):
            for te in (int(p.with_suffix(".textend").read_text(), 16), 0):
                r = run_test(p, te)
                if r is None:
                    n_uns += 1
                    continue
                ok, th = r
                n_ok += ok
                n_bad += not ok
                print(f"{p.stem:24s} T={te:#06x} tohost={th:3s} {'same' if ok else 'DIFF'}")
        print(f"same {n_ok}, different {n_bad}, unsupported {n_uns}")
    elif sys.argv[1] == "linux":
        n = int(sys.argv[2])
        img = ROOT / "kernel" / "Image"
        ram_size, te, cmd = 0x480000, 0xF65C0, "console=ttyS0 rdinit=/bin/sh"
        stop = ["-c", str(n)] if n else ["-x", "/ # "]
        ctoks, cstate, cram, out = c_trace(["-f", str(img), "-m", hex(ram_size), "-T", hex(te),
                                            "-k", cmd, *stop])
        print(out.strip())
        n = n or int(out.split()[1])
        ram, regs = R.linux_ram(img.read_bytes(), ram_size, cmd)
        m = R.Machine(ram, te, regs)
        toks = []
        for _ in range(n):
            toks += m.step()
        ok = compare(f"linux {n}", ctoks, cstate, cram, m, toks)
        print("same" if ok else "DIFFERENT", len(toks), "tokens", len(m.out), "uart bytes")


if __name__ == "__main__":
    main()
