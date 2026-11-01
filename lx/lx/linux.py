"""Linux in the model: boot it (or load a booted snapshot) and run shell commands.

    python -m lx.linux [--merged] [--snapshot FILE] [--out FILE] [CMD ...]   # or a prompt: -i

Every byte shown is one the model predicted (BYTE tokens), never the reference's: the
reference only proposes the trace (speculative execution), and a single wrong prediction stops
the run. A command is typed as KEY0 / KEY tokens (at most RX_SLOTS bytes at a time: the rest
once the kernel has read those), then the machine runs until the shell prints its prompt again.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from lx import core, hf
from lx import ref as R
from lx import spec as S
from lx.session import Session

ROOT = Path(__file__).resolve().parents[2]
IMAGE = ROOT / "kernel" / "Image"
CMDLINE = "console=ttyS0 rdinit=/bin/sh"
PROMPT = b"/ # "


class Wrong(RuntimeError):
    pass


def type_and_run(m: R.Machine, data: bytes, max_records: int = 30_000_000) -> list[int]:
    """The proposer's tokens for typing ``data`` into machine m (RX_SLOTS bytes at a time, each
    once the receive FIFO is empty) and running until the shell waits for input again: the
    prompt printed after the input, the FIFO empty, and the CPU idle (a WFI since the latest
    byte printed). The idle test matters for multi-line input: the kernel's tty buffer can hold
    lines the shell has not read yet while the FIFO is already empty and a prompt is showing."""
    o0, toks, n = len(m.out), [], 0
    wfi = S.tok("WFI")
    b0, nb = S.FAMILIES["BYTE"]
    for a in range(0, len(data), S.RX_SLOTS):
        for b in data[a:a + S.RX_SLOTS]:
            toks += m.key(b)
        last = a + S.RX_SLOTS >= len(data)
        idle = False
        while not m.halted and n < max_records:
            if m.rx_waiting == 0 and (not last or (idle and m.out.endswith(PROMPT)
                                                   and len(m.out) > o0)):
                break
            rec = m.step()
            toks += rec
            n += 1
            if rec[-1] == wfi:
                idle = True
            elif any(b0 <= t < b0 + nb for t in rec):
                idle = False
    return toks


class Linux:
    def __init__(self, merged: bool = False, chunk: int | None = None,
                 weights: Path | None = None, fast: bool = False, lean: bool = False) -> None:
        """weights: a merge.save() directory (the merged model) instead of building it. fast:
        verify with lx.fast (the same weights, skipping their zeros) instead of the stock
        forward. lean (with weights; implies fast): the chip layers' projections stay in host
        memory, lx.fast holding their nonzero parts on the GPU."""
        # prefill memory is mostly the 128 GDN heads' q/k/v per token (~0.65 MB): 64k tokens
        # fit beside the 28 GB chip-only model on an 80 GB card, 32k beside the 32 GB merged one
        merged = merged or weights is not None
        self.weights = weights
        # lean: 15.8 GB of weights on the GPU (Qwen's 12.2, the embedding 3.6), 8k beside them
        chunk = chunk or (8192 if lean else 32768 if merged else 65536)
        from lx import merge

        self.cfg = merge.config() if merged else core.Config()
        t0 = time.time()
        self.c = core.build(self.cfg)
        if lean:
            assert weights is not None, "lean loads a saved checkpoint"
            self.model, self.info = merge.load(weights, self.c, device="cpu")
            self.offset = self.info["chip_token_offset"]
        elif merged:
            self.model, self.info = merge.load(weights, self.c) if weights else \
                merge.build(self.c)
            self.offset = self.info["chip_token_offset"]
        else:
            self.model, lay = hf.build_model(self.c)
            self.info = {"hidden": lay.hidden, "circuit": self.c.stats()}
            self.offset = 0
        self.info["params_B"] = round(sum(p.numel() for p in self.model.parameters()) / 1e9, 2)
        self.fast = None
        if fast or lean:
            from lx.fast import Fast, place_lean

            rows = slice(self.offset, self.offset + S.VOCAB) if self.offset else None
            self.fast = Fast(self.model, self.c.n_layers(), rows)
            if lean:
                place_lean(self.model, self.c.n_layers(), "cuda")
                self.fast.to("cuda").lean = True
                self.fast.head_block = 2048
        self.info["fast"], self.info["lean"] = fast or lean, lean
        self.build_s = time.time() - t0
        self.chunk = chunk
        self.s = None
        self.log = []

    def reset_machine(self) -> R.Machine:
        ram, regs = R.linux_ram(IMAGE.read_bytes(), self.cfg.ram_size, CMDLINE)
        return R.Machine(ram, self.cfg.text_end, regs)

    def _verify(self, toks, what: str) -> dict:
        torch.cuda.synchronize()
        t0 = time.time()
        r = self.s.verify(toks)
        torch.cuda.synchronize()
        dt = time.time() - t0
        if r["wrong"]:
            raise Wrong(f"{what}: {r['wrong']} wrong predictions, first {r['first_wrong']}")
        return {"tokens": len(toks), "seconds": round(dt, 2), "tok_per_s": round(len(toks) / dt),
                "out": r["out"]}

    def boot(self, snapshot: Path | None = None) -> dict:
        """From reset to the shell prompt (or from a saved snapshot of that)."""
        if snapshot and Path(snapshot).exists():
            t0 = time.time()
            self.s = Session.load(self.c, self.model, snapshot, self.chunk, self.offset,
                                  self.fast)
            rec = {"what": "boot (snapshot)", "path": str(snapshot), "tokens": self.s.n,
                   "seconds": round(time.time() - t0, 2), "out": b""}
        else:
            self.s = Session(self.c, self.model, self.reset_machine(), self.chunk, self.offset,
                             self.fast)
            t0 = time.time()
            toks = [S.tok("GO")] + self.s.propose(until=lambda m: m.out.endswith(PROMPT))
            prop = time.time() - t0
            # verified chunk by chunk, for the timeline: (seconds, tokens, console bytes so far)
            timeline, out, dt = [], b"", 0.0
            for a in range(0, len(toks), self.chunk):
                r = self._verify(toks[a:a + self.chunk], "boot")
                dt += r["seconds"]
                out += r["out"]
                timeline.append((round(dt, 2), a + r["tokens"], len(out)))
            rec = {"what": "boot", "tokens": len(toks), "seconds": round(dt, 1),
                   "tok_per_s": round(len(toks) / dt), "out": out, "propose_s": round(prop, 1),
                   "timeline": timeline}
            if snapshot:
                self.s.save(snapshot)
                rec["saved"] = str(snapshot)
        self.log.append(rec)
        return rec

    def run(self, cmd: str | bytes, max_records: int = 30_000_000, rehearse: bool = False,
            progress=None) -> dict:
        """Type cmd + newline and run until the prompt comes back. Returns the record, with
        "out" the console bytes (as the model predicted them) and "output" the command's
        output (without the echoed command line and the prompt). ``rehearse``: the reference
        alone, nothing verified (the model's state is then behind: for previewing commands).
        ``progress(seconds, tokens verified, tokens, console bytes so far)`` after each chunk."""
        data = (cmd.encode() if isinstance(cmd, str) else cmd).rstrip(b"\n") + b"\n"
        o0 = len(self.s.ref.out)
        t0 = time.time()
        toks = type_and_run(self.s.ref, data, max_records)
        prop = time.time() - t0
        rec = {"what": "run", "cmd": data.decode(errors="replace").rstrip("\n"),
               "propose_s": round(prop, 1)}
        if rehearse:
            rec.update(tokens=len(toks), seconds=0.0, tok_per_s=0, rehearsed=True,
                       out=bytes(self.s.ref.out[o0:]), timeline=[])
        else:
            # verified chunk by chunk, for the timeline: (seconds, tokens, console bytes so far)
            timeline, out, dt = [], b"", 0.0
            for a in range(0, len(toks), self.chunk):
                r = self._verify(toks[a:a + self.chunk], repr(data))
                dt += r["seconds"]
                out += r["out"]
                timeline.append((round(dt, 2), a + r["tokens"], len(out)))
                if progress:
                    progress(dt, a + r["tokens"], len(toks), out)
            rec.update(tokens=len(toks), seconds=round(dt, 2), tok_per_s=round(len(toks) / dt),
                       out=out, timeline=timeline)
        text = rec["out"].replace(b"\r\n", b"\n")
        body = text[:-len(PROMPT)] if text.endswith(PROMPT) else text
        if data.count(b"\n") == 1:  # one line: drop its echo (multi-line: the whole transcript)
            rest = _after_echo(body, data[:-1])
            body = rest if rest is not None else body
        rec["output"] = body
        self.log.append(rec)
        return rec


def _after_echo(text: bytes, line: bytes) -> bytes | None:
    """text after the shell's echo of line and its newline; the line editor breaks a long line
    with newlines (and carriage returns) where it wraps."""
    i = j = 0
    while j < len(line) and i < len(text):
        if text[i] == line[j]:
            i, j = i + 1, j + 1
        elif text[i] in b"\r\n":
            i += 1
        else:
            return None
    if j < len(line):
        return None
    return text[i + 1:] if text[i:i + 1] == b"\n" else text[i:]


def run_pure(lx: Linux, cmd: str | bytes, max_tokens: int = 2_000_000) -> dict:
    """The pure path from the current state: type cmd + newline, then feed the model its own
    predictions until the bytes it predicts end with the prompt. Only afterwards is the stream
    compared with the reference's (from a copy of the proposer's machine, typed at the same
    place)."""
    import copy

    from lx.session import console

    data = (cmd.encode() if isinstance(cmd, str) else cmd).rstrip(b"\n") + b"\n"
    assert len(data) <= S.RX_SLOTS
    s = lx.s
    ref = copy.deepcopy(s.ref)
    b0, nb = S.FAMILIES["BYTE"]
    out = bytearray()

    def stop(toks):
        t = toks[-1]
        if b0 <= t < b0 + nb:
            out.append(t - b0)
            return out.endswith(PROMPT) and len(out) > len(data)
        return False

    torch.cuda.synchronize()
    t0 = time.time()
    stream = s.type(data)
    stream += s.free(max_tokens, stop)
    torch.cuda.synchronize()
    dt = time.time() - t0
    want = []
    for b in data:
        want += ref.key(b)
    while len(want) < len(stream):
        want += ref.step()
    first = next((i for i, (x, y) in enumerate(zip(stream, want)) if x != y), None)
    rec = {"what": "pure", "cmd": data.decode().rstrip("\n"), "tokens": len(stream),
           "seconds": round(dt, 1), "tok_per_s": round(len(stream) / dt, 1),
           "out": bytes(console(stream)), "same_as_reference": first is None,
           "first_difference": first}
    lx.log.append(rec)
    return rec


def _jsonable(rec: dict) -> dict:
    return {k: (v.decode(errors="replace") if isinstance(v, bytes) else v) for k, v in rec.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmds", nargs="*")
    ap.add_argument("--merged", action="store_true")
    ap.add_argument("--weights", default="", help="a merge.save() directory (implies --merged)")
    ap.add_argument("--snapshot", default=str(ROOT / "build" / "snapshots" / "boot.pt"))
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument("--pure", default="", help="run this command on the pure path (no reference)")
    ap.add_argument("--fast", action="store_true", help="verify with lx.fast (zeros skipped)")
    ap.add_argument("--compile", action="store_true", help="lx.fast with its compiled steps")
    ap.add_argument("--lean", action="store_true",
                    help="lx.fast, the chip projections kept in host memory (fits a 24 GB card)")
    ap.add_argument("--max-gib", type=float, default=None,
                    help="cap this process's GPU memory (PyTorch's allocator) at this many GiB")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    a.merged = a.merged or bool(a.weights)
    if a.merged and a.snapshot.endswith("boot.pt"):
        a.snapshot = a.snapshot.replace("boot.pt", "boot-merged.pt")
    if a.max_gib:
        torch.cuda.set_per_process_memory_fraction(
            a.max_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    lx = Linux(a.merged, a.chunk, Path(a.weights) if a.weights else None, a.fast or a.compile,
               a.lean)
    if a.compile:
        lx.fast.compile = lx.info["compile"] = True
    print(f"{torch.cuda.get_device_name()}: {lx.info['circuit']}, hidden {lx.info['hidden']}, "
          f"{lx.info['params_B']}B parameters ({lx.build_s:.0f}s to build)", flush=True)
    b = lx.boot(Path(a.snapshot))
    sys.stdout.write(b["out"].decode(errors="replace"))
    print(f"\n[{b['what']}: {b['tokens']} tokens in {b['seconds']}s]", flush=True)
    cmds = list(a.cmds)
    while True:
        if cmds:
            cmd = cmds.pop(0)
        elif a.interactive:
            try:
                cmd = input(PROMPT.decode())
            except EOFError:
                break
        else:
            break
        r = lx.run(cmd)
        sys.stdout.write(r["out"].decode(errors="replace"))
        print(f"\n[{r['tokens']} tokens, {r['seconds']}s at {r['tok_per_s']} tok/s, "
              f"proposed in {r['propose_s']}s]", flush=True)
    if a.pure:
        r = run_pure(lx, a.pure)
        sys.stdout.write(r["out"].decode(errors="replace"))
        print(f"\n[pure path: {r['tokens']} tokens, {r['seconds']}s at {r['tok_per_s']} tok/s, "
              f"same as the reference: {r['same_as_reference']}]", flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        res = {"device": torch.cuda.get_device_name(), "merged": a.merged, **lx.info,
               "build_s": round(lx.build_s), "chunk": lx.chunk,
               "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
               "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 1),
               "max_gib": a.max_gib, "runs": [_jsonable(r) for r in lx.log]}
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
