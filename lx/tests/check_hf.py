"""The LX circuit on a GPU in stock HF modules (bf16, fla kernels when installed): chunked prefill
and token-by-token decode from an injected snapshot, against the IR evaluator and the trace.

    python tests/check_hf.py rvtests [--share] [--decode N] [--chunk N] [NAME ...]
    python tests/check_hf.py linux N [--skip K] [--share] [--decode N] [--chunk N]
    python tests/check_hf.py boot [--share] [--chunk N] [--out FILE]   # reset to the shell prompt
    python tests/check_hf.py ramtest [--share] [--chunk N] [--out FILE]  # every RAM word, r/w
    python tests/check_hf.py uart [--share] [--decode N]          # typed input (KEY tokens)

A run must reproduce the IR's residual bit for bit (every IR dim; with --share, where dims share
residual slots as in the merged model, the output code) and the output head's argmax must be
the trace's next token at every position. --decode N also decodes the first N positions token
by token. Exits 1 on any difference.
"""

import argparse
import copy
import sys
import time

import torch
from check_core import (RV_RAM, RV_TEXT, ROOT, UART_SCHEDULES, linux_boot, linux_trace, rv_trace,
                        show, uart_trace)

from lx import core, hf, ir
from lx import ref as R
from lx import spec as S


def wrong_tokens(pred: torch.Tensor, seq: list[int], first: int = 8) -> tuple[int, list]:
    """(number of positions whose prediction is not the next token, the first few of them);
    positions followed by host input (KEY0 / KEY) predict nothing."""
    n = min(len(pred), len(seq)) - 1
    target = torch.tensor(seq[1:n + 1], device=pred.device)
    typed = torch.zeros_like(target, dtype=torch.bool)
    for f in S.INPUT:
        lo, size = S.FAMILIES[f]
        typed |= (target >= lo) & (target < lo + size)
    bad = ((pred[:n] != target) & ~typed).nonzero()[:, 0].tolist()
    return len(bad), [(i, seq[i + 1], int(pred[i])) for i in bad[:first]]


def run_one(c, model, m0, toks, decode: int, chunk: int | None, view: str = "model",
            tokens_only: bool = False):
    """[(mode, seconds, positions, residual diffs, wrong-token count, first wrong tokens)] and
    the sequence."""
    core.snapshot(c, m0)
    seq = [S.tok("GO")] + toks
    want = None if tokens_only else ir.evaluate(c, seq, check=False)
    out = []
    for mode, n in [("prefill", len(seq))] + ([("decode", min(decode, len(seq)))] if decode else []):
        torch.cuda.synchronize()
        t = time.time()
        h, pred = hf.run(c, model, seq[:n], mode, chunk, view, keep_h=not tokens_only)
        torch.cuda.synchronize()
        dt = time.time() - t
        diff = [] if tokens_only else hf.compare(c, model, h, want[:n])
        out.append((mode, dt, n, diff, *wrong_tokens(pred, seq[:n])))
    return out, seq


def report(name, results, seq) -> bool:
    ok = True
    for mode, dt, n, diff, n_wrong, wrong in results:
        good = not diff and not n_wrong
        ok &= good
        print(f"  {name:24s} {mode:7s} {n:7d} pos {dt:6.1f}s {n / dt:8.0f} tok/s "
              f"{'ok' if good else f'{len(diff)} DIMS DIFFER, {n_wrong} WRONG TOKENS'}",
              flush=True)
        for d in diff[:4]:
            print(f"    dim {d[0]} first differs at {d[1]}: got {d[2]!r}, want {d[3]!r}")
        if wrong:
            show(wrong, seq)
    return ok


def ram_trace(cfg: core.Config):
    """tests/ramtest.S (built by tests/build_ramtest.sh) from reset to its power-off."""
    img = (ROOT / "build" / "ramtest" / "ramtest.bin").read_bytes()
    ram, regs = R.linux_ram(img, cfg.ram_size, None)
    m = R.Machine(ram, cfg.text_end, regs)
    m0 = copy.deepcopy(m)
    toks, n = [], 0
    while not m.halted:
        toks += m.step()
        n += 1
    return m0, toks, m, n


def long_run(a, name: str, trace) -> bool:
    """A whole run of the full machine (``trace(cfg)`` -> machine at reset, tokens, final
    machine, records), verified by the model in one chunked prefill (chip view): every
    position's argmax must be the next token, and the BYTE tokens the model predicts must spell
    the reference's console output."""
    import json

    cfg = core.Config()
    t0 = time.time()
    c = core.build(cfg)
    model, lay = hf.build_model(c, share=a.share)
    n_par = sum(p.numel() for p in model.parameters())
    build_s = time.time() - t0
    print(f"{c.stats()}, hidden {lay.hidden}, {n_par / 1e9:.1f}B parameters "
          f"({build_s:.0f}s to build)", flush=True)
    t0 = time.time()
    m0, toks, m, n_rec = trace(cfg)
    ref_s = time.time() - t0
    seq = [S.tok("GO")] + toks
    print(f"{name}: {n_rec} records, {len(seq)} tokens, {len(m.out)} console bytes "
          f"(reference {ref_s:.0f}s)", flush=True)
    chunk = a.chunk or 65536
    core.snapshot(c, m0)
    for L in sorted({chunk, len(seq) % chunk or chunk}):  # fla autotunes each new shape
        hf.run(c, model, seq[:min(L, len(seq))], "prefill", chunk, "chip", keep_h=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    _, pred = hf.run(c, model, seq, "prefill", chunk, "chip", keep_h=False)
    torch.cuda.synchronize()
    dt = time.time() - t0
    n_wrong, wrong = wrong_tokens(pred, seq)
    b0, nb = S.FAMILIES["BYTE"]
    p = pred[:-1]
    console = bytes((p[(p >= b0) & (p < b0 + nb)] - b0).cpu().tolist())
    ok = n_wrong == 0 and console == bytes(m.out)
    res = {"run": name, "device": torch.cuda.get_device_name(), "kernels": "fla", "dtype": "bf16",
           "view": "chip", "share_slots": a.share, "chunk": chunk, "circuit": c.stats(),
           "hidden": lay.hidden, "params_B": round(n_par / 1e9, 2), "ram_bytes": cfg.ram_size,
           "rom_bytes": cfg.text_end, "records": n_rec, "tokens": len(seq),
           "positions_checked": len(seq) - 1, "wrong_tokens": n_wrong,
           "first_wrong": wrong[:8], "console_bytes": len(m.out),
           "console_matches_reference": console == bytes(m.out), "seconds": round(dt, 1),
           "tok_per_s": round(len(seq) / dt), "records_per_s": round(n_rec / dt),
           "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
           "build_s": round(build_s), "reference_s": round(ref_s)}
    print(json.dumps(res), flush=True)
    print(console.decode(errors="replace")[-600:])
    if wrong:
        show(wrong, seq)
    if a.out:
        from pathlib import Path

        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
        Path(a.out).with_suffix(".console.txt").write_bytes(console)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["rvtests", "linux", "boot", "ramtest", "uart"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--decode", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--view", choices=["model", "chip"], default="model")
    ap.add_argument("--tokens-only", action="store_true",
                    help="skip the IR (long runs): check the output head's tokens only")
    ap.add_argument("--out", default="")
    a = ap.parse_intermixed_args()
    try:
        import fla  # noqa: F401

        kernels = "fla"
    except ImportError:
        kernels = "torch fallbacks"
    print(f"{torch.cuda.get_device_name()}, bf16, kernels: {kernels}", flush=True)
    bad = 0
    if a.cmd == "rvtests":
        paths = sorted((ROOT / "build" / "rvtests").glob("*.bin"))
        if a.args:
            paths = [p for p in paths if any(n in p.stem for n in a.args)]
        for te in RV_TEXT:
            t0 = time.time()
            c = core.build(core.Config(ram_size=RV_RAM, text_end=te))
            model, lay = hf.build_model(c, share=a.share)
            print(f"T={te:#x}: {c.stats()}, hidden {lay.hidden} "
                  f"({time.time() - t0:.0f}s to build)", flush=True)
            for p in paths:
                tr = rv_trace(p, te)
                if tr is None:
                    continue
                results, seq = run_one(c, model, *tr, a.decode, a.chunk, a.view)
                bad += not report(p.stem, results, seq)
            del model
            torch.cuda.empty_cache()
    elif a.cmd == "uart":
        for te in (0, 0x40):
            c = core.build(core.Config(ram_size=RV_RAM, text_end=te))
            model, lay = hf.build_model(c, share=a.share)
            print(f"T={te:#x}: {c.stats()}, hidden {lay.hidden}", flush=True)
            for name, sched in UART_SCHEDULES.items():
                m0, toks, m = uart_trace(te, sched)
                results, seq = run_one(c, model, m0, toks, a.decode, a.chunk, a.view)
                bad += not report(f"uart {name} -> {bytes(m.out)[:12]!r}", results, seq)
            del model
            torch.cuda.empty_cache()
    elif a.cmd == "boot":
        bad += not long_run(a, "boot", linux_boot)
    elif a.cmd == "ramtest":
        bad += not long_run(a, "ramtest", ram_trace)
    else:
        n = int(a.args[0])
        cfg = core.Config()
        t0 = time.time()
        c = core.build(cfg)
        model, lay = hf.build_model(c, share=a.share)
        n_par = sum(p.numel() for p in model.parameters())
        print(f"{c.stats()}, hidden {lay.hidden}, {n_par / 1e9:.1f}B parameters "
              f"({time.time() - t0:.0f}s to build)", flush=True)
        t0 = time.time()
        m0, toks = linux_trace(cfg, n, a.skip)
        print(f"records {a.skip}..{a.skip + n}: {len(toks)} tokens ({time.time() - t0:.0f}s)",
              flush=True)
        results, seq = run_one(c, model, m0, toks, a.decode, a.chunk, a.view, a.tokens_only)
        bad += not report(f"linux {a.skip}+{n}", results, seq)
    print("all ok" if not bad else f"{bad} FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
