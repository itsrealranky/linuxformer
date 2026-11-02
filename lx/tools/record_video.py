"""Recordings for the hero video: the model alone on the megakernel (as tools/mega_boot.py and
lx.chat --alone), keeping what those runs only hash or summarize: the whole token stream, a
per-launch timeline, and snapshots of the model's state (with the reference machine at the same
token) at chosen record boundaries.

  boot      from reset to the shell prompt. --snap N...: the first record boundary at or after
            each N tokens (stream[0] is GO), and always the prompt
  session   Qwen's Fibonacci session (lx.chat --alone) from a booted snapshot; per shell call
            the stream and timeline, and snapshots at the boundaries before the first byte of
            each --mark string in the console and after its last line, and at the end of the call

The reference only proposes beforehand (boot) or runs afterwards (session) to compare with;
nothing from it is fed to the model. Launch boundaries move to the snapshot points, which
changes no input: each step's input is still the step before's prediction.

    python tools/record_video.py boot --out results/video_boot.json [--snap 1000000 ...]
    python tools/record_video.py session --out results/video_fib.json
"""

import argparse
import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from lx import spec as S
from lx.linux import PROMPT, ROOT, Linux, rel
from lx.mega import Mega
from lx.session import Session, console

DATA = ROOT / "build" / "video_data"
FIB_Q = ("Write a shell script that prints the first 20 Fibonacci numbers, save it as "
         "/tmp/fib.sh, run it on the machine, and tell me the 20th number.")


def git(*x) -> str:
    return subprocess.run(["git", *x], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def sha(stream) -> str:
    """As tools/mega_boot.py hashes a stream (its JSON list)."""
    return hashlib.sha256(json.dumps(list(map(int, stream))).encode()).hexdigest()


@torch.no_grad()
def boot(a, lx: Linux) -> dict:
    m = lx.reset_machine()
    t0 = time.time()
    want, refs, targets = [S.tok("GO")], {}, sorted(a.snap)
    while not m.out.endswith(PROMPT):
        want += m.step()
        while targets and len(want) >= targets[0]:
            refs[len(want)] = copy.deepcopy(m)
            targets.pop(0)
    refs[len(want)] = copy.deepcopy(m)
    prop = time.time() - t0
    s0 = Session(lx.c, lx.model, lx.reset_machine(), lx.chunk, lx.offset, lx.fast)
    reset = "/tmp/record_video_reset.pt"
    s0.save(reset)
    del s0
    torch.cuda.empty_cache()
    s = Session.load(lx.c, lx.model, reset, lx.chunk, lx.offset)
    mg = Mega(lx.fast, lx.model, s.cache, lx.offset)
    N = len(want)
    ends = sorted(set(range(a.every, N, a.every)) | set(refs) | {N})
    inp = torch.zeros(max(e - b for b, e in zip([0] + ends, ends)), dtype=torch.int32,
                      device=mg.dev)
    stream, nxt, timeline, snaps = [], want[0], [], []
    pure = {"inputs_checked": 0, "forced_by_host": {"GO": 1, "keystrokes": 0}, "launches": 0}
    dec, save_s = 0.0, 0.0
    torch.cuda.synchronize()
    t_loop = time.time()
    for e in ends:
        a0, n = len(stream), e - len(stream)
        torch.cuda.synchronize()
        t1 = time.time()
        preds = mg.run([nxt], n, inputs=inp)
        dec += time.time() - t1
        got = inp[:n].tolist()
        assert got[0] == nxt and (a0 > 0 or nxt == S.tok("GO"))
        assert got[1:] == preds[:-1]
        pure["inputs_checked"] += n
        pure["launches"] += 1
        stream += got
        nxt = preds[-1]
        timeline.append((round(dec, 3), len(stream)))
        if stream[a0:] != want[a0:len(stream)]:
            first = next(i for i in range(a0, len(stream)) if stream[i] != want[i])
            raise SystemExit(f"the model's stream differs from the reference at token {first}")
        if e in refs:
            t2 = time.time()
            s.ref, s.next, s.last, s.n, s.waiting = refs[e], nxt, stream[-1], e, 0
            path = DATA / f"boot-{e}.pt"
            s.save(path)
            save_s += time.time() - t2
            snaps.append({"tokens": e, "path": str(path.relative_to(ROOT)),
                          "console_bytes": len(console(stream))})
        print(json.dumps({"tokens": len(stream), "of": N, "decode_s": round(dec, 1),
                          "tok_per_s": round(len(stream) / dec, 1)}), flush=True)
    loop_s = time.time() - t_loop - save_s  # as mega_boot.py times it (host checks included)
    out = bytes(console(stream))
    DATA.mkdir(parents=True, exist_ok=True)
    np.save(DATA / "boot_stream.npy", np.asarray(stream, dtype=np.int32))
    pure["passed"] = pure["inputs_checked"] == len(stream)
    return {"what": "model-alone boot on the megakernel, stream and snapshots kept",
            "tokens": len(stream), "reference_tokens": N, "same_as_reference": stream == want,
            "stream_sha256": sha(stream), "reference_sha256": sha(want),
            "decode_seconds": round(dec, 1), "seconds": round(loop_s, 1),
            "minutes": round(loop_s / 60, 2), "tok_per_s": round(len(stream) / loop_s, 1),
            "snapshot_save_s": round(save_s, 1), "propose_s": round(prop, 1),
            "console_bytes": len(out), "console_matches_reference": out == bytes(m.out),
            "console": out.decode(errors="replace"), "input_purity": pure,
            "tokens_a_launch": a.every, "timeline": timeline, "snapshots": snaps,
            "stream": str((DATA / "boot_stream.npy").relative_to(ROOT))}


class Recorder:
    """lx.linux.Linux.run_alone's loop, keeping the stream and splitting launches at given
    stream positions (to save snapshots there) without changing when it types or stops."""

    def __init__(self, lx: Linux, tag: str):
        self.lx, self.tag = lx, tag
        self.calls = []

    def run_alone(self, data: bytes, splits: dict, max_tokens: int = 60_000_000) -> dict:
        from lx.session import _final_ids

        lx = self.lx
        s, mg = lx.s, lx.mg
        final = _final_ids().numpy()
        b0, nb = S.FAMILIES["BYTE"]
        rx0, nrx = S.FAMILIES["RX"]
        typed = np.zeros(final.shape[0], dtype=bool)
        for f in S.INPUT:
            typed[S.FAMILIES[f][0]:S.FAMILIES[f][0] + S.FAMILIES[f][1]] = True
        wfi = S.tok("WFI")
        stream, typed_at, out, idle = [], [], bytearray(), False
        inp = torch.zeros((1 << 16) + S.RX_SLOTS, dtype=torch.int32, device=mg.dev)
        pure = {"inputs_checked": 0, "keystrokes": 0, "launches": 0}
        timeline, saved, dec = [], [], 0.0
        for a in range(0, len(data), S.RX_SLOTS):
            batch = data[a:a + S.RX_SLOTS]
            last = a + S.RX_SLOTS >= len(data)
            assert final[s.last] and s.waiting == 0, (s.last, s.waiting)
            forced = [S.tok("KEY0" if i == 0 else "KEY", b) for i, b in enumerate(batch)]
            typed_at.append(len(stream))
            m, keys = 4096, len(forced)
            while len(stream) < max_tokens:
                n = len(forced) + m - 1
                # the launch, in pieces at the split points inside it
                cut = sorted(p - len(stream) for p in splits if 0 < p - len(stream) < n)
                assert not cut or cut[0] >= len(forced), "a split inside the keystrokes"
                got, preds, b = [], [], 0
                self._base = (len(stream), s.n)
                for c in cut + [n]:
                    f_ = forced[b:] if b < len(forced) else [preds[-1]]
                    torch.cuda.synchronize()
                    t1 = time.time()
                    pr = mg.run(f_, c - b, inputs=inp)
                    dec += time.time() - t1
                    g = inp[:c - b].tolist()
                    assert g[:min(len(f_), c - b)] == f_[:c - b]
                    got += g
                    preds += pr
                    b = c
                    if c < n:
                        self._snapshot(stream + got, preds[-1], splits, saved, out, typed)
                assert got[:len(forced)] == forced and \
                    got[len(forced):] == preds[len(forced) - 1:n - 1]
                pure["inputs_checked"] += n
                pure["keystrokes"] += keys
                keys = 0
                pure["launches"] += 1
                g = np.asarray(got)
                s.waiting += int(typed[g].sum()) - int(((g >= rx0) & (g < rx0 + nrx)).sum())
                isb = (g >= b0) & (g < b0 + nb)
                out += bytes((g[isb] - b0).astype(np.uint8))
                iw = np.nonzero(g == wfi)[0]
                ib = np.nonzero(isb)[0]
                if iw.size or ib.size:
                    idle = (iw[-1] if iw.size else -1) > (ib[-1] if ib.size else -1)
                stream += got
                s.next, s.last, s.n = preds[-1], got[-1], s.n + n
                forced = [s.next]
                timeline.append((round(dec, 3), len(stream), len(out)))
                done = s.waiting == 0 and (not last or (idle and out.endswith(PROMPT)
                                                        and len(out) > 0))
                if done and final[s.last]:
                    break
                m = 1 if done else min(2 * m, 1 << 16)
        ref, want = lx.alone_ref, []
        for i, at in enumerate(typed_at):
            while len(want) < at:
                want += ref.step()
            for bb in data[i * S.RX_SLOTS:(i + 1) * S.RX_SLOTS]:
                want += ref.key(bb)
        while len(want) < len(stream):
            want += ref.step()
        first = next((i for i, (x, y) in enumerate(zip(stream, want)) if x != y), None)
        if first is None and len(want) != len(stream):
            first = min(len(want), len(stream))
        pure["passed"] = pure["inputs_checked"] == len(stream) and \
            pure["keystrokes"] == len(data)
        k = len(self.calls)
        path = DATA / f"{self.tag}_call{k}_stream.npy"
        np.save(path, np.asarray(stream, dtype=np.int32))
        rec = {"tokens": len(stream), "seconds": round(dec, 2),
               "tok_per_s": round(len(stream) / dec), "out": bytes(out),
               "timeline": timeline, "typed_at": typed_at,
               "same_as_reference": first is None, "first_difference": first,
               "wrong": 0 if first is None else len(stream) - first,
               "console_matches_reference": bytes(out) == bytes(ref.out[-len(out):]) if out
               else True, "input_purity": pure, "stream": str(path.relative_to(ROOT)),
               "stream_sha256": sha(stream), "reference_sha256": sha(want),
               "snapshots": saved}
        # the end of the call (the reference typed at the same places, for the machine)
        e = len(stream)
        p = DATA / f"{self.tag}_call{k}_end-{e}.pt"
        ref_saved, s.ref = s.ref, copy.deepcopy(ref)
        s.save(p)
        s.ref = ref_saved
        saved.append({"tokens": e, "why": "end of the call", "path": str(p.relative_to(ROOT))})
        self.calls.append(rec)
        return rec

    def _snapshot(self, stream, nxt, splits, saved, out, typed) -> None:
        s = self.lx.s
        e = len(stream)
        ref = copy.deepcopy(self.lx.alone_ref)
        ref_toks = []
        # the reference to the same point: typed where the model's stream has the keystrokes
        key0, nk = S.FAMILIES["KEY0"]
        key1 = S.FAMILIES["KEY"][0]
        i = 0
        while i < e:
            t = stream[i]
            if key0 <= t < key0 + nk or key1 <= t < key1 + nk:
                ref_toks += ref.key(t - (key0 if t < key0 + nk else key1))
                i += 1
                continue
            r = ref.step()
            ref_toks += r
            i += len(r)
        assert i == e and ref_toks == stream[:e], "the split point is not a record boundary"
        k = len(self.calls)
        p = DATA / f"{self.tag}_call{k}_{splits[e]}-{e}.pt"
        old = (s.ref, s.next, s.last, s.n)
        s.ref, s.next, s.last = ref, nxt, stream[-1]
        s.n = self._base[1] + e - self._base[0]
        s.save(p)
        s.ref, s.next, s.last, s.n = old
        saved.append({"tokens": e, "why": splits[e], "path": str(p.relative_to(ROOT))})


def plan_splits(lx: Linux, data: bytes, marks: list[str]) -> dict:
    """The reference, typed as run_alone types (its launch sizes, from the reference's stream):
    the stream positions before the first console byte of each mark and after the line that
    ends the marked block (the next prompt-less line not of the kernel's report)."""
    from lx.session import _final_ids

    ref = copy.deepcopy(lx.alone_ref)
    final = _final_ids().numpy()
    typed = np.zeros(final.shape[0], dtype=bool)
    for f in S.INPUT:
        typed[S.FAMILIES[f][0]:S.FAMILIES[f][0] + S.FAMILIES[f][1]] = True
    b0, nb = S.FAMILIES["BYTE"]
    rx0, nrx = S.FAMILIES["RX"]
    wfi = S.tok("WFI")
    stream, out, idle, waiting, last_t = [], bytearray(), False, lx.s.waiting, lx.s.last
    bounds = []  # record boundaries (stream positions after a record or a keystroke)
    pending = []

    def produce(k):
        nonlocal pending
        a0 = len(stream)
        while len(stream) - a0 < k:
            if not pending:
                pending = ref.step()
            take = min(k - (len(stream) - a0), len(pending))
            stream.extend(pending[:take])
            pending = pending[take:]
            if not pending:
                bounds.append(len(stream))
        return stream[a0:]

    for a in range(0, len(data), S.RX_SLOTS):
        batch = data[a:a + S.RX_SLOTS]
        last = a + S.RX_SLOTS >= len(data)
        assert final[last_t] and waiting == 0 and not pending
        a0 = len(stream)
        for b in batch:
            stream.extend(ref.key(b))
            bounds.append(len(stream))
        m = 4096
        got = stream[a0:] + produce(m - 1)
        while True:
            g = np.asarray(got)
            waiting += int(typed[g].sum()) - int(((g >= rx0) & (g < rx0 + nrx)).sum())
            isb = (g >= b0) & (g < b0 + nb)
            out += bytes((g[isb] - b0).astype(np.uint8))
            iw = np.nonzero(g == wfi)[0]
            ib = np.nonzero(isb)[0]
            if iw.size or ib.size:
                idle = (iw[-1] if iw.size else -1) > (ib[-1] if ib.size else -1)
            last_t = got[-1]
            done = waiting == 0 and (not last or (idle and out.endswith(PROMPT) and len(out)))
            if done and final[last_t]:
                break
            m = 1 if done else min(2 * m, 1 << 16)
            got = produce(m)
    # console byte positions
    pos = [i for i, t in enumerate(stream) if b0 <= t < b0 + nb]
    text = bytes(stream[i] - b0 for i in pos)
    bset = sorted(set(bounds))
    import bisect
    splits = {}
    for mk in marks:
        j = text.find(mk.encode())
        if j < 0:
            continue
        before = bset[bisect.bisect_right(bset, pos[j]) - 1]  # the boundary before the byte
        splits[before] = "before"
        # after: the end of the line that names binfmt_flat's failure (the kernel's last line)
        k = text.find(b"binfmt_flat:", j)
        k = text.find(b"\n", k if k >= 0 else j)
        after = bset[bisect.bisect_left(bset, pos[k] + 1)]
        splits[after] = "after"
    return {"splits": splits, "tokens": len(stream), "console": bytes(out)}


@torch.no_grad()
def session(a, lx: Linux) -> dict:
    from lx.chat import Chat

    b = lx.boot_alone(Path(a.snapshot))
    rec = Recorder(lx, "fib")
    chat = Chat(lx)
    orig = lx.run_alone
    plans = []

    def run_alone(data, max_tokens=60_000_000, progress=None):
        t0 = time.time()
        plan = plan_splits(lx, data, a.mark)
        plans.append({"splits": {str(k): v for k, v in plan["splits"].items()},
                      "reference_tokens": plan["tokens"], "plan_s": round(time.time() - t0, 1)})
        print(json.dumps(plans[-1]), flush=True)
        r = rec.run_alone(data, plan["splits"], max_tokens)
        r["planned_tokens_match"] = plan["tokens"] == r["tokens"]
        return r

    lx.run_alone = run_alone
    t0 = time.time()
    res = chat.ask(FIB_Q)
    lx.run_alone = orig
    for ev, call in zip([e for e in res["events"] if "machine_tokens" in e], rec.calls):
        ev.update({k: call[k] for k in ("stream", "stream_sha256", "reference_sha256",
                                        "snapshots", "typed_at", "planned_tokens_match",
                                        "console_matches_reference", "same_as_reference",
                                        "wrong") if k in call})
    return {"what": "Qwen's Fibonacci session, model alone, stream and snapshots kept",
            "mode": "model alone", "boot": {k: v for k, v in b.items() if k != "out"},
            "question": FIB_Q, "chat": res, "plans": plans,
            "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["boot", "session"])
    ap.add_argument("--weights", default=str(ROOT / "build" / "hf" / "model"))
    ap.add_argument("--every", type=int, default=65536, help="boot: tokens a launch")
    ap.add_argument("--snap", type=int, nargs="*", default=[])
    ap.add_argument("--snapshot", default=str(ROOT / "build" / "snapshots" /
                                              "boot-merged-mega.pt"))
    ap.add_argument("--mark", nargs="*", default=["seq: page allocation failure"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    lx = Linux(weights=Path(a.weights), fast=True)
    t0 = time.time()
    res = boot(a, lx) if a.what == "boot" else session(a, lx)
    res.update(device=torch.cuda.get_device_name(), commit=git("rev-parse", "HEAD"),
               clean_checkout=git("status", "--porcelain") == "", weights=rel(a.weights),
               peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1),
               total_s=round(time.time() - t0, 1))
    Path(a.out).write_text(json.dumps(res, indent=1, default=lambda o: o.decode(
        errors="replace") if isinstance(o, bytes) else str(o)))
    print(json.dumps({k: v for k, v in res.items() if k not in ("console", "timeline", "chat")},
                     default=str)[:3000], flush=True)


if __name__ == "__main__":
    main()
