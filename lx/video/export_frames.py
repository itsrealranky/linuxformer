"""Every frame of the hero video as data, for lx/video/player.html, and its fact sheet.

From the recordings (results/video_boot.json, results/video_fib.json: the model alone on the
megakernel, each stream identical to the reference), their events (tools/video_events.py: the
streams replayed through the reference ISA decoder, which checks every token), the address map
(tools/address_map.py) and snapshots of the model's state (the zoom, the crash). A page square
lights for a RAM store into it, a register cell for a write to it, the heartbeat for a timer
interrupt; the console is the BYTE tokens; video time follows the runs' per-launch decode-time
timelines (linear inside a launch), at one speed per piece of the storyboard.

    python video/export_frames.py [--story hero|loop] [--out ../build/video/frames.json]
                                  [--facts video/video_facts.json]

The streams and snapshots are in build/video_data (made by tools/record_video.py); events are
cached there. Every number a caption shows comes from the fact sheet written here.
"""

import argparse
import base64
import bisect
import hashlib
import json
from pathlib import Path

import numpy as np

from lx import spec as S
from lx.linux import ROOT
from tools import address_map as AM
from tools import video_events as VE

RES = ROOT / "lx" / "results"
DATA = ROOT / "build" / "video_data"
FPS = 60
N_CHIP, N_QWEN = 37, 32
REG_LAYER, PRIV_LAYER = 4, 5  # regs.rs1 / regs.rs2, csr.priv (results/address_map.json)
COLS = {"16x9": 108, "9x16": 58}  # console columns at the player's font sizes
ROWS = {"16x9": 3, "9x16": 4}
LOG_TEXT = 0x80178314  # __log_buf + 4: the first record's text ("Linux version ...")
# bytes the 8250 driver writes to THR's address while LCR bit 7 selects the divisor latch:
# a real 8250 does not transmit them (mini-rv32ima's UART prints them; see the fact sheet)
SHOW_DIVISOR_BYTES = False


def load(p):
    return json.loads(Path(p).read_text())


def rel(p) -> str:
    return str(Path(p).resolve().relative_to(ROOT))


def sha_stream(stream) -> str:
    return hashlib.sha256(json.dumps(list(map(int, stream))).encode()).hexdigest()


def commas(n) -> str:
    return f"{int(n):,}"


def speed_text(x: float) -> str:
    return f"{x:.0f}x" if x >= 9.95 else f"{x:.1f}x"


# --- runs --------------------------------------------------------------------------------------

class Run:
    """A stream, its events, and its clock (tokens <-> decode seconds, per launch)."""

    def __init__(self, tag, stream, ev, tl_tokens, tl_seconds):
        self.tag, self.stream, self.ev = tag, stream, ev
        self.tt = np.asarray([0] + list(tl_tokens), dtype=np.float64)
        self.ts = np.asarray([0.0] + list(tl_seconds), dtype=np.float64)
        assert (np.diff(self.tt) > 0).all() and (np.diff(self.ts) >= 0).all()
        assert self.tt[-1] == len(stream)

    def secs(self, t):
        return float(np.interp(t, self.tt, self.ts))

    def token_at(self, s):
        return int(np.floor(np.interp(s, self.ts, self.tt)))


def events_cached(name, what, streams, machine=None):
    p = DATA / f"{name}_events.npz"
    key = hashlib.sha256(b"".join(s.tobytes() for s in streams)).hexdigest()
    if p.exists():
        ev = dict(np.load(p))
        if str(ev.get("key", "")) == key:
            return ev
    ev = VE.events(what, streams, machine=machine)
    ev["key"] = np.array(key)
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **ev)
    return ev


def booted(boot_stream):
    """The reference machine at the end of the boot stream."""
    m = VE.reset_machine()
    i = 1
    while i < len(boot_stream):
        i += len(m.step())
    assert i == len(boot_stream)
    return m


def stream_or_regenerate(path, sha, make):
    """A recorded stream; if absent, the reference's (which the recording equals: its sha256 is
    in the results), regenerated and checked."""
    p = ROOT / path
    if p.exists():
        s = np.load(p)
    else:
        s = np.asarray(make(), dtype=np.int32)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, s)
    assert sha_stream(s) == sha, path
    return s


def boot_reference(n_tokens: int) -> list:
    """The reference's boot stream: GO, then its tokens from reset."""
    m, toks = VE.reset_machine(), [S.tok("GO")]
    while len(toks) < n_tokens:
        toks += m.step()
    return toks


def session_reference(calls: list, m0) -> list:
    """The reference's stream for each call, from the booted machine, with each call's command
    typed where the model's stream has its keystrokes (typed_at)."""
    import copy
    m, out = copy.deepcopy(m0), []
    for c in calls:
        data = c["command"].encode().rstrip(b"\n") + b"\n"
        toks = []
        for i, at in enumerate(c["typed_at"]):
            while len(toks) < at:
                toks += m.step()
            assert len(toks) == at, (len(toks), at)
            for b in data[i * S.RX_SLOTS:(i + 1) * S.RX_SLOTS]:
                toks += m.key(b)
        while len(toks) < c["machine_tokens"]:
            toks += m.step()
        out.append(np.asarray(toks, dtype=np.int32))
    return out


def boot_run(boot: dict) -> Run:
    assert boot["same_as_reference"] and boot["stream_sha256"] == boot["reference_sha256"]
    stream = stream_or_regenerate(boot["stream"], boot["stream_sha256"],
                                  lambda: boot_reference(boot["tokens"]))
    assert len(stream) == boot["tokens"]
    ev = events_cached("boot", "boot", [stream])
    tl = boot["timeline"]
    return Run(0, stream, ev, [t for _, t in tl], [s for s, _ in tl])


def session_run(fib: dict, boot_stream) -> tuple[Run, list]:
    calls = [e for e in fib["chat"]["events"] if "machine_tokens" in e]
    m0 = booted(boot_stream)
    for c in calls:
        assert c["same_as_reference"] and c["stream_sha256"] == c["reference_sha256"]
    if not all((ROOT / c["stream"]).exists() for c in calls):
        for c, s in zip(calls, session_reference(calls, m0)):
            (ROOT / c["stream"]).parent.mkdir(parents=True, exist_ok=True)
            np.save(ROOT / c["stream"], s)
    streams = [stream_or_regenerate(c["stream"], c["stream_sha256"], None) for c in calls]
    for c, s in zip(calls, streams):
        assert len(s) == c["machine_tokens"]
    ev = events_cached("fib", "session", streams, machine=m0)
    tt, ts, a0, s0 = [], [], 0, 0.0
    for c in calls:
        for sec, tok, *_ in c["machine_timeline"]:
            tt.append(a0 + tok)
            ts.append(s0 + sec)
        a0 += c["machine_tokens"]
        s0 += c["machine_seconds"]
    return Run(1, np.concatenate(streams), ev, tt, ts), calls


# --- the console -------------------------------------------------------------------------------

def terminal(data: bytes, cols: int) -> list[str]:
    """Display rows of a terminal wrapping at cols (\\r \\n \\b \\t; the runs print no escapes)."""
    rows, cur, x = [], [], 0
    for b in data:
        if b == 10:
            rows.append("".join(cur))
            cur, x = [], 0
        elif b == 13:
            x = 0
        elif b == 8:
            x = max(0, x - 1)
        elif b == 9:
            x = min(cols, (x // 8 + 1) * 8)
        elif 32 <= b < 127:
            if x >= cols:
                rows.append("".join(cur))
                cur, x = [], 0
            cur += [" "] * (x - len(cur))
            if x < len(cur):
                cur[x] = chr(b)
            else:
                cur.append(chr(b))
            x += 1
    return rows + ["".join(cur)]


class Console:
    """The console up to (run, token), from the runs' UART bytes (BYTE tokens)."""

    def __init__(self, runs):
        self.keys, b = [], []
        self.hidden = []
        for r in runs:
            dl = r.ev["uart_dlab"].astype(bool)
            for t, x, d in zip(r.ev["uart_t"], r.ev["uart_b"], dl):
                if d and not SHOW_DIVISOR_BYTES:
                    self.hidden.append({"run": r.tag, "token": int(t), "byte": int(x)})
                    continue
                self.keys.append((r.tag, int(t)))
                b.append(int(x))
        self.data = bytes(b)
        self.states = {k: [] for k in COLS}
        self.memo = {k: {} for k in COLS}

    def count(self, tag: int, tok: int) -> int:
        return bisect.bisect_right(self.keys, (tag, tok))

    def state(self, n: int, layout: str) -> int:
        memo = self.memo[layout]
        if n not in memo:
            rows = terminal(self.data[:n], COLS[layout])
            if len(rows) > 1 and rows[-1] == "":
                rows = rows[:-1]  # an empty cursor row would push text out of the strip
            rows = rows[-ROWS[layout]:]
            st = self.states[layout]
            if not st or st[-1] != rows:
                st.append(rows)
            memo[n] = len(st) - 1
        return memo[n]


# --- what a frame shows of a run ---------------------------------------------------------------

class Events:
    def __init__(self, run: Run, amap: dict):
        ev = run.ev
        self.st_t = ev["store_t"]
        a = ev["store_a"].astype(np.int64)  # offsets into memory (stores to MMIO are not here)
        assert (a >= 0).all() and (a < amap["ram_size"]).all()
        self.st_p = a // 4096
        blk = a // (4 * amap["ram_block_words"])
        lut = np.full(amap["ram_size"] // (4 * amap["ram_block_words"]), -1, dtype=np.int64)
        for t in amap["ram"]:
            lut[t["block"]] = t["layer"]
        self.st_l = lut[blk]
        assert (self.st_l >= 0).all(), "a store outside the RAM layers"
        self.rg_t, self.rg_r = ev["regw_t"], ev["regw_r"]
        self.irq_t = ev["trap_t"][ev["trap_c"] == S.TRAP_INT]
        self.pv_t, self.pv_p = ev["priv_t"], ev["priv_p"]
        self.pg_t, self.pages = ev["pages_t"], ev["pages"]

    def window(self, a: int, b: int) -> dict:
        """What committed at tokens (a, b]."""
        s0, s1 = np.searchsorted(self.st_t, [a, b], side="right")
        r0, r1 = np.searchsorted(self.rg_t, [a, b], side="right")
        q0, q1 = np.searchsorted(self.irq_t, [a, b], side="right")
        p0, p1 = np.searchsorted(self.pv_t, [a, b], side="right")
        layers = set(np.unique(self.st_l[s0:s1]).tolist())
        regs = 0
        for r in np.unique(self.rg_r[r0:r1]).tolist():
            regs |= 1 << int(r)
        if r1 > r0:
            layers.add(REG_LAYER)
        if p1 > p0:
            layers.add(PRIV_LAYER)
        return {"pages": np.unique(self.st_p[s0:s1]).tolist(), "regs": regs,
                "irq": int(q1 - q0), "layers": sorted(int(x) for x in layers)}

    def priv(self, b: int, start: int) -> int:
        k = np.searchsorted(self.pv_t, b, side="right")
        return int(self.pv_p[k - 1]) if k else start

    def pagemap(self, b: int):
        k = np.searchsorted(self.pg_t, b, side="right")
        return self.pages[k - 1] if k else None


# --- the zoom and the crash, from snapshots ----------------------------------------------------

def torch_load(p):
    import torch
    return torch.load(ROOT / p, weights_only=False, map_location="cpu")


def from_snapshot(name: str, snap: str, make):
    """Data read from a snapshot (GBs, not published), kept as results/video_{name}.json and
    read from there when the snapshot is not here."""
    p = RES / f"video_{name}.json"
    if (ROOT / snap).exists():
        d = make()
        p.write_text(json.dumps(d, separators=(",", ":")))
        return d
    return load(p)


def pack2(cls: np.ndarray) -> str:
    f = cls.reshape(-1).astype(np.uint8)
    return base64.b64encode((f[0::4] | f[1::4] << 2 | f[2::4] << 4 | f[3::4] << 6).tobytes()
                            ).decode()


def zoom_data(snap: str, amap: dict, addr: int, word: bytes) -> dict:
    """The key head holding addr: its 8 value heads' state as cell classes (0, A_BIT, ballast),
    the row's raw values, and what they decode to."""
    d = torch_load(snap)
    cell = AM.locate(amap, addr)[-1]
    assert cell["region"] == "RAM"
    L, j, row = cell["layer"], cell["key_head"], cell["row"]
    rs = d["state"][L][1][0, 8 * j:8 * j + 8].double().numpy()  # (8, 256, 128)
    cls = np.zeros(rs.shape, dtype=np.uint8)
    cls[rs == AM.A_BIT] = 1
    cls[rs == AM.A_REF] = 2
    assert ((rs == 0) | (rs == AM.A_BIT) | (rs == AM.A_REF)).all()
    raw = rs[:, row, :]
    words = [int(sum(1 << i for i in range(32) if raw[x, 2 + i] == AM.A_BIT)) for x in range(8)]
    text = b"".join(w.to_bytes(4, "little") for w in words)
    ofs = addr - amap["ram_base"]
    row0 = ofs & ~31
    want = bytes(d["ref"].ram[row0:row0 + 32])
    k = ofs - row0
    assert text == want and text[k:k + len(word)] == word, (text, want)
    page = ofs // 4096
    pc = [c for c in AM.page_cells(amap, page) if c["region"] == "RAM"][0]
    return {"snapshot": snap, "tokens": int(d["n"]), "address": addr, "page": page,
            "page_rows": pc["rows"], "layer": L, "key_head": j, "row": row,
            "value_heads": list(range(8 * j, 8 * j + 8)),
            "word_value_heads": [8 * j + (k // 4) + i for i in range((k % 4 + len(word) + 3) // 4)],
            "row_address": amap["ram_base"] + row0, "row_bytes_hex": text.hex(),
            "offset_in_row": k, "word": word.decode(), "raw": raw[:, :34].astype(np.int64).tolist(),
            "words": words, "classes": pack2(cls), "classes_shape": list(cls.shape),
            "a_bit": AM.A_BIT, "a_ref": AM.A_REF, "decodes_to_word": True}


def crash_data(snap: str, amap: dict, n_pages: int, log: str) -> dict:
    d = torch_load(snap)
    mem, bad = AM.decode(amap, d["state"])
    assert bytes(mem) == bytes(d["ref"].ram) and not any(bad.values())
    pm = VE.page_map(mem, n_pages)
    blocks = VE.free_blocks(mem, n_pages)
    hist = {}
    for _, o in blocks:
        hist[o] = hist.get(o, 0) + 1
    best = cur = 0
    for p in range(n_pages):
        cur = cur + 1 if pm[p] == VE.P_FREE else 0
        best = max(best, cur)
    import re
    line = re.search(r"Normal: (.*?) = (\d+)kB", log)
    kern = {int(kb): int(n) for n, kb in re.findall(r"(\d+)\*(\d+)kB", line.group(1))}
    ours = {4 << o: n for o, n in hist.items()}
    order = re.search(r"order:(\d+)", log)
    nommu = re.search(r"Allocation of length (\d+)", log)
    return {"snapshot": snap, "tokens": int(d["n"]),
            "free_pages_struct_page": int((pm == VE.P_FREE).sum()),
            "free_blocks": [[int(p), int(o)] for p, o in blocks],
            "histogram_kb_struct_page": {str(k): v for k, v in sorted(ours.items())},
            "histogram_kb_kernel_log": {str(k): v for k, v in sorted(kern.items()) if v},
            "kernel_log_total_kb": int(line.group(2)),
            "histograms_agree": {k: v for k, v in kern.items() if v} == ours,
            "largest_free_block_kb": max(ours), "longest_free_run_pages": best,
            "order": int(order.group(1)), "request_bytes": int(nommu.group(1)),
            "order7_windows": [int((pm[k * 128:(k + 1) * 128] == VE.P_FREE).sum())
                               for k in range(-(-n_pages // 128))],
            "pagemap": base64.b64encode(pm.astype(np.uint8).tobytes()).decode(),
            "decode_exactness": bad}


# --- the storyboard ----------------------------------------------------------------------------

def hero_story(B: Run, F: Run, calls, boot, con: Console):
    """Pieces: (segment, kind, args, seconds). Kinds: run (tokens a -> b of a run at one speed),
    pause (a run frozen at a token), qwen (Qwen's k-th reply streaming), hold, card."""
    ev = B.ev
    up = ev["uart_b"]
    banner = bytes(up.astype(np.uint8)).find(b"Linux version")
    t_L = int(ev["uart_t"][banner])
    nl = bytes(up.astype(np.uint8)).find(b"\n", banner)
    freeze = int(ev["uart_t"][nl]) + 1
    snaps = {s["tokens"]: s["path"] for s in boot["snapshots"]}
    assert freeze in snaps, (freeze, sorted(snaps))
    s1 = B.token_at(B.secs(t_L) - 0.6)
    end = len(B.stream)
    marks = {"freeze": freeze, "freeze_snapshot": snaps[freeze], "slow_from": s1,
             "banner_L": t_L}
    boot_part = [
        ("cold", "run", (B, 0, s1), 1.4),
        ("cold", "run", (B, s1, freeze), 1.6),
        ("zoom", "pause", (B, freeze), 10.0),
        ("ff", "run", (B, freeze, end), 7.0),
    ]
    if F is None:  # preview: the boot only
        return boot_part + [("card", "card", (), 4.0)], marks
    c0 = calls[0]["machine_tokens"]
    snap0 = {s["why"]: s["tokens"] for s in calls[0]["snapshots"]}
    before, after = snap0["before"], snap0["after"]
    marks.update(before=before, after=after)
    return boot_part + [
        ("qwen", "qwen", (0,), 1.5),
        ("qwen", "run", (F, 0, before), 4.5),
        ("crash", "pause", (F, before), 2.5),
        ("crash", "run", (F, before, after), 0.8),
        ("crash", "pause", (F, after), 1.2),
        ("crash", "run", (F, after, c0), 0.5),
        ("fix", "qwen", (1,), 1.5),
        ("fix", "run", (F, c0, len(F.stream)), 6.5),
        ("answer", "qwen", (2,), 1.0),
        ("answer", "hold", (), 2.0),
        ("card", "card", (), 4.0),
    ], marks


def loop_story(B: Run, F, calls, boot, con):
    return [("loop", "run", (B, 0, len(B.stream)), 10.0)], {}


# --- the fact sheet ----------------------------------------------------------------------------

class Facts:
    def __init__(self):
        self.d = {}

    def add(self, key, value, source, text=None):
        if text is None:
            text = str(value).lower() if isinstance(value, bool) else \
                commas(value) if isinstance(value, int) else str(value)
        self.d[key] = {"value": value, "text": text,
                       "source": source}
        return self.d[key]["text"]

def build(story_name: str, out: Path, facts_path: Path):
    boot = load(RES / "video_boot.json")
    fib = load(RES / "video_fib.json") if story_name == "hero" else None
    amap = load(RES / "address_map.json")
    checks = load(RES / "address_map_check.json")
    B = boot_run(boot)
    F, calls = session_run(fib, B.stream) if fib else (None, [])
    con = Console([B] + ([F] if F else []))
    EB, EF = Events(B, amap), Events(F, amap) if F else None
    n_pages = amap["ram_size"] // 4096
    story = {"hero": hero_story, "preview": hero_story, "loop": loop_story}[story_name]
    pieces, marks = story(B, F, calls, boot, con)
    fx = Facts()

    # facts: the runs
    fx.add("boot_tokens", boot["tokens"], "results/video_boot.json: tokens")
    fx.add("boot_wrong", 0 if boot["same_as_reference"] else None,
           "results/video_boot.json: same_as_reference (stream sha256 = reference sha256)")
    fx.add("boot_minutes", round(boot["decode_seconds"] / 60, 1),
           "results/video_boot.json: decode_seconds / 60", f"{boot['decode_seconds'] / 60:.1f}")
    fx.add("boot_tok_per_s", round(boot["tok_per_s"]), "results/video_boot.json: tok_per_s")
    fx.add("gpu", boot["device"], "results/video_boot.json: device",
           "H100" if "H100" in boot["device"] else boot["device"])
    fx.add("instructions", int(B.ev["end_instret"]), "tools/video_events.py on the boot stream")
    fx.add("uart_bytes", int(len(B.ev["uart_t"])), "tools/video_events.py on the boot stream")
    fx.add("timer_interrupts", int((B.ev["trap_c"] == S.TRAP_INT).sum()),
           "tools/video_events.py on the boot stream")
    qwen = []
    if fib:
        sess_tokens = sum(c["machine_tokens"] for c in calls)
        fx.add("session_tokens", sess_tokens, "results/video_fib.json: sum of machine_tokens")
        fx.add("total_tokens", boot["tokens"] + sess_tokens, "boot_tokens + session_tokens")
        for k, c in enumerate(calls):
            fx.add(f"call{k}_tokens", c["machine_tokens"], f"results/video_fib.json: call {k} "
                   "machine_tokens")
            fx.add(f"call{k}_keys", c["input_purity"]["keystrokes"],
                   f"results/video_fib.json: call {k} input_purity.keystrokes")
            assert c["same_as_reference"] and c["wrong"] == 0
            fx.add(f"call{k}_wrong", c["wrong"], f"results/video_fib.json: call {k} wrong "
                   "(stream sha256 = reference sha256)")
        fx.add("total_wrong", fx.d["boot_wrong"]["value"] + sum(c["wrong"] for c in calls),
               "boot_wrong + the calls' wrong")
        c0, c1 = calls[0]["command"], calls[1]["command"]
        assert "seq " in c0 and "while " in c1 and "seq" not in c1
        fx.add("call1_while_no_seq", True, "results/video_fib.json: call 0's command uses "
               "$(seq 1 20); call 1's uses a while loop and no seq")
        qwen = list(fib["chat"]["events"])
        import re
        nth = int(re.search(r"first (\d+) Fibonacci", fib["question"]).group(1))
        fx.add("fib_n", nth, "results/video_fib.json: question ('the first N Fibonacci numbers "
               "... the Nth number')", f"{nth}th")
        ans = re.findall(r"\*\*(\d+)\*\*", qwen[-1]["qwen"])
        fx.add("answer", int(ans[-1]), "results/video_fib.json: Qwen's final reply", ans[-1])
        fibs = [0, 1]
        while len(fibs) < nth:
            fibs.append(fibs[-1] + fibs[-2])
        assert fibs[nth - 1] == int(ans[-1]), "Qwen's answer is not the Nth Fibonacci number"
        assert calls[-1]["machine_out"].replace("\r", "").split("\n")[-2] == ans[-1]
    import re
    v = re.search(rb"Linux version (\d+)\.(\d+)\.\d+-(rc\d+)", con.data)
    fx.add("linux_version", f"{int(v[1])}.{int(v[2])}-{v[3].decode()}",
           "boot console: 'Linux version 6.8.0-rc1mini-rv32ima ...' (kernel/linux-6.8-rc1)")
    merge_check = load(RES / "m6_merge_check.json")
    fx.add("qwen_tensors_differ", len(merge_check["qwen_tensors_not_identical"]),
           "results/m6_merge_check.json: qwen_tensors_not_identical (README: 0 of 426 tensors "
           "differ)")
    fx.add("no_training", True, "README 'What is in the weights': the compiler (lx/lx/core.py, "
           "emit.py) computes the 37 chip layers' weights from the CPU circuit; Qwen's are the "
           "release's", "No training")
    fx.add("license", "MIT", "LICENSE (third-party components: NOTICE)")
    fx.add("chip_layers", N_CHIP, "build/hf/model/lx_info.json: chip_layers")
    fx.add("qwen_layers", N_QWEN, "build/hf/model/lx_info.json: qwen_layers")
    fx.add("ram_squares", n_pages, "results/address_map.json: ram_size / 4096")
    fx.add("rom_pages", -(-amap["text_end"] // 4096), "results/address_map.json: text_end")

    # facts: the zoom
    z = cr = None
    if story_name in ("hero", "preview"):
        zs = marks["freeze_snapshot"]
        z = from_snapshot("zoom", zs, lambda: zoom_data(zs, amap, LOG_TEXT, b"Linux"))
        o = LOG_TEXT - amap["ram_base"]
        sa = B.ev["store_a"].astype(np.int64)
        hit = (sa < o + len(z["word"])) & (sa + B.ev["store_s"] > o) & (B.ev["store_t"] < z["tokens"])
        z["written_at_tokens"] = sorted(set(int(x) for x in B.ev["store_t"][hit]))
        assert z["tokens"] == marks["freeze"]
        fx.add("zoom_address", LOG_TEXT, "System.map: __log_buf + 4", f"0x{LOG_TEXT:08X}")
        fx.add("zoom_layer", z["layer"], "results/address_map.json: locate(zoom_address)",
               f"model.layers.{z['layer']}")
        fx.add("zoom_key_head", z["key_head"], "results/address_map.json")
        fx.add("zoom_row", z["row"], "results/address_map.json")
        fx.add("zoom_value_heads", z["word_value_heads"], "results/address_map.json",
               "–".join(map(str, (z["word_value_heads"][0], z["word_value_heads"][-1]))))
        fx.add("zoom_written_at", z["written_at_tokens"][0],
               "tools/video_events.py: first store to zoom_address")
        fx.add("zoom_page", z["page"], "zoom_address // 4096")
    if fib:
        cs = [s for s in calls[0]["snapshots"] if s["why"] == "before"][0]["path"]
        cr = from_snapshot("crash", cs, lambda: crash_data(cs, amap, n_pages,
                                                            calls[0]["machine_out"]))
        assert cr["tokens"] == fib["boot"]["tokens"] + marks["before"], (cr["tokens"], marks)
        assert cr["histograms_agree"], cr
        fx.add("crash_need_kb", (4 << cr["order"]), "kernel log: order:7 (4 KB << 7)",
               f"{4 << cr['order']} KB")
        fx.add("crash_order", cr["order"], "kernel log: 'page allocation failure: order:N'")
        fx.add("crash_request_bytes", cr["request_bytes"], "kernel log: nommu 'Allocation of "
               "length N'")
        fx.add("crash_free_kb", cr["kernel_log_total_kb"], "kernel log: 'Normal: ... = N kB'",
               f"{commas(cr['kernel_log_total_kb'])} KB")
        fx.add("crash_largest_kb", cr["largest_free_block_kb"], "kernel log free-block line, "
               "= struct page buddy blocks in the snapshot", f"{cr['largest_free_block_kb']} KB")
        need = 4 << cr["order"]
        frag = (cr["histograms_agree"] and cr["kernel_log_total_kb"] >= need
                and cr["largest_free_block_kb"] < need)
        assert frag and max(cr["order7_windows"]) < 128
        fx.add("crash_fragmented", frag, "crash_free_kb >= crash_need_kb > crash_largest_kb, "
               "kernel log histogram = struct page free blocks in the snapshot")
        fx.add("crash_best_aligned_free_pages", max(cr["order7_windows"]), "struct page: free "
               "pages in the fullest 512 KB-aligned block (order7_windows); an order-7 block "
               "must be aligned")
        fx.add("crash_longest_free_run_kb", 4 * cr["longest_free_run_pages"], "struct page: "
               "the longest run of free pages; it crosses a 512 KB boundary, so it is several "
               "free blocks, not one", f"{4 * cr['longest_free_run_pages']} KB")

    # frames
    cols = {k: [] for k in ("tok", "qtok", "seg", "piece", "prog", "speed", "priv", "pm",
                            "con16x9", "con9x16", "pages", "regs", "irq", "layers", "cpu",
                            "qwen", "chat")}
    pms, pm_index = [], {}
    speeds = []
    segs = []
    replies = [e for e in qwen]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ROOT / "models" / "Qwen3.5-4B")  # scripts/setup.sh
    reply_cuts = []
    for e in replies:
        ids = tok.encode(e["qwen"], add_special_tokens=False)
        assert tok.decode(ids) == e["qwen"]
        cuts = [len(tok.decode(ids[:k])) for k in range(len(ids) + 1)]
        reply_cuts.append({"text": e["qwen"], "cuts": cuts, "tokens": e["tokens"],
                           "seconds": e["seconds"], "command": e.get("command")})
    state = {"run": B, "tok": 0, "priv": 3, "chat": [0, 0], "qtok": 0}
    piece_meta = []
    for pi, (seg, kind, args, dur) in enumerate(pieces):
        nf = int(round(dur * FPS))
        if not segs or segs[-1]["id"] != seg:
            segs.append({"id": seg, "start": len(cols["tok"])})
        pm_ = {"segment": seg, "kind": kind, "start": len(cols["tok"]), "frames": nf}
        if kind == "run":
            R, a, b = args
            sa, sb = R.secs(a), R.secs(b)
            sp = (sb - sa) / dur
            pm_.update(run=R.tag, tokens=[a, b], real_seconds=round(sb - sa, 2), speed=sp)
            label = "speed: " + speed_text(sp)
            if state["run"] is not R:
                state.update(run=R, tok=a)
        elif kind == "pause":
            R, a = args
            state.update(run=R, tok=a)
            label = "paused"
            pm_.update(run=R.tag, token=a)
        elif kind == "qwen":
            q = reply_cuts[args[0]]
            sp = q["seconds"] / dur
            label = "speed: " + speed_text(sp)
            pm_.update(reply=args[0], real_seconds=q["seconds"], speed=sp)
        else:
            label = "" if kind == "card" else "paused"
        if label not in speeds:
            speeds.append(label)
        piece_meta.append(pm_)
        R = state["run"]
        E = EB if R is B else EF
        priv0 = 3 if R is B else int(B.ev["end_priv"])
        for i in range(nf):
            # t = tokens decoded so far; an event at stream index x shows once t > x
            prev = state["tok"]
            if kind == "run":
                t = b if i == nf - 1 else min(b, max(a, R.token_at(sa + (sb - sa) * (i + 1) / nf)))
            else:
                t = prev
            w = E.window(prev - 1, t - 1) if t > prev else \
                {"pages": [], "regs": 0, "irq": 0, "layers": []}
            state["tok"] = t
            if kind == "qwen":
                q = reply_cuts[args[0]]
                state["qtok"] = int(round(q["tokens"] * (i + 1) / nf))
                state["chat"] = [args[0], q["cuts"][int(round((len(q["cuts"]) - 1) * (i + 1)
                                                              / nf))]]
            pm = E.pagemap(t - 1)
            if pm is not None:
                key = pm.tobytes()
                if key not in pm_index:
                    pm_index[key] = len(pms)
                    pms.append(base64.b64encode(key).decode())
                pmi = pm_index[key]
            else:
                pmi = -1
            n_con = con.count(R.tag, t - 1)
            cols["tok"].append(int(t) + (boot["tokens"] if R is F else 0))
            cols["qtok"].append(state["qtok"])
            cols["seg"].append(len(segs) - 1)
            cols["piece"].append(pi)
            cols["prog"].append(round((i + 0.5) / nf, 5))
            cols["speed"].append(speeds.index(label))
            cols["priv"].append(E.priv(t - 1, priv0))
            cols["pm"].append(pmi)
            cols["con16x9"].append(con.state(n_con, "16x9"))
            cols["con9x16"].append(con.state(n_con, "9x16"))
            cols["pages"].append(w["pages"])
            cols["regs"].append(w["regs"])
            cols["irq"].append(w["irq"])
            cols["layers"].append(w["layers"])
            cols["cpu"].append(int(kind == "run"))
            cols["qwen"].append(int(kind == "qwen"))
            cols["chat"].append(list(state["chat"]))
        if kind == "qwen":
            state["chat"] = [args[0], len(reply_cuts[args[0]]["text"])]
            state["qtok"] = reply_cuts[args[0]]["tokens"]

    # the speed each segment's caption names
    seg_speed = {}
    for p in piece_meta:
        if p["kind"] == "run":
            seg_speed.setdefault(p["segment"], []).append(p["speed"])
    if "ff" in seg_speed:
        fx.add("ff_speed", round(seg_speed["ff"][0]), "frames: real decode seconds / video "
               "seconds of the fast-forward", speed_text(seg_speed["ff"][0]))
    captions = hero_captions(fx) if story_name != "loop" else {"loop": ("", "", [])}
    meta = {"fps": FPS, "frames": len(cols["tok"]), "story": story_name,
            "segments": segs, "pieces": piece_meta, "speeds": speeds,
            "captions": {k: [b, s] for k, (b, s, _) in captions.items()}, "marks": marks,
            "chip_layers": N_CHIP, "qwen_layers": N_QWEN,
            "qwen_attention_layers": [N_CHIP + i for i in range(3, N_QWEN, 4)],
            "n_pages": n_pages, "rom_pages": -(-amap["text_end"] // 4096),
            "first_ram_page": amap["first_ram_block"] * amap["ram_block_words"] * 4 // 4096,
            "console": {k: con.states[k] for k in COLS}, "console_cols": COLS,
            "replies": [{"text": q["text"], "tokens": q["tokens"]} for q in reply_cuts],
            "question": fib["question"] if fib else "", "zoom": z,
            "crash": cr, "facts": {k: v["text"] for k, v in fx.d.items()}}
    frames = {"meta": meta, "pagemaps": pms, "cols": cols}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(frames, separators=(",", ":")))
    if story_name == "hero":
        write_facts(facts_path, fx, captions, boot, fib, calls, checks, z, cr, con, piece_meta,
                    marks)
    print(json.dumps({"frames": meta["frames"], "pagemaps": len(pms),
                      "bytes": out.stat().st_size, "speeds": speeds}))


# {fact} is the fact's text; the third item names facts a caption states in words
CAPTIONS = {
    "cold": ("Linux is booting inside an AI model.",
             "Qwen3.5-4B · RISC-V CPU compiled into {chip_layers} layers · "
             "RAM = Gated DeltaNet state · model alone", []),
    "zoom": ("This word lives inside the model.",
             "kernel log at {zoom_address} · {zoom_layer} · key head {zoom_key_head} · "
             "value heads {zoom_value_heads} · row {zoom_row}", []),
    "ff": ("No emulator. The model computes every step.",
           "{boot_tokens} tokens · {boot_wrong} wrong vs reference · {boot_minutes} min on one "
           "{gpu} · sped up {ff_speed}", []),
    "qwen": ("Now Qwen uses its own computer.",
             "Qwen's {qwen_layers} layers write a shell script · its keystrokes run through the "
             "{chip_layers} CPU layers", []),
    "crash": ("Not enough room in one piece to load the program.",
              "needs {crash_need_kb} in one piece (order {crash_order}) · {crash_free_kb} free · "
              "largest piece {crash_largest_kb} · fragmented, not out of memory",
              ["crash_fragmented", "crash_best_aligned_free_pages"]),
    "fix": ("Qwen reads the error and fixes its own code.",
            "new script: a while loop, no seq · {call1_tokens} machine tokens · {call1_wrong} "
            "wrong vs reference", ["call1_while_no_seq"]),
    "answer": ("{fib_n} Fibonacci number: {answer} ✓",
               "{total_tokens} machine tokens since power-on · {total_wrong} wrong · model alone",
               []),
    "card": ("{no_training} · Qwen's weights bit-identical · Linux {linux_version} · open source",
             "every frame is drawn from the recorded runs (lx/results/video_*.json)",
             ["qwen_tensors_differ", "license"]),
}


def hero_captions(fx: Facts) -> dict:
    """{segment: (big, small, facts used)}; a fact not available renders as '?'."""
    import re

    class Texts(dict):
        def __missing__(self, k):
            return "?"
    texts = Texts({k: v["text"] for k, v in fx.d.items()})
    out = {}
    for k, (big, small, said) in CAPTIONS.items():
        keys = re.findall(r"\{(\w+)\}", big + small)
        out[k] = (big.format_map(texts), small.format_map(texts), keys + said)
    if "qwen_tensors_differ" in fx.d:
        assert fx.d["qwen_tensors_differ"]["value"] == 0 and fx.d["license"]["value"] == "MIT"
    return out


def write_facts(path, fx, captions, boot, fib, calls, checks, z, cr, con, pieces, marks):
    doc = {
        "what": "Every number, label and caption in the hero video, with its source. Written "
                "by lx/video/export_frames.py from the recorded runs.",
        "facts": fx.d,
        "captions": {k: {"big": b, "small": s, "facts": used} for k, (b, s, used) in
                     captions.items()},
        "segments": [{k: v for k, v in p.items() if k != "speed"} |
                     ({"speed": round(p["speed"], 2), "speed_label": speed_text(p["speed"])}
                      if "speed" in p else {}) for p in pieces],
        "marks": marks,
        "checks": {
            "boot_same_as_reference": boot["same_as_reference"],
            "boot_stream_sha256": boot["stream_sha256"],
            "boot_console_matches_reference": boot["console_matches_reference"],
            "session_calls_same_as_reference": [c["same_as_reference"] for c in calls],
            "session_console_matches_reference": [c.get("console_matches_reference")
                                                  for c in calls],
            "address_map_all_identical": checks["all_identical"],
            "zoom_decodes": z and z["decodes_to_word"],
            "crash_histograms_agree": cr and cr["histograms_agree"],
        },
        "labels": {
            "mode": "model alone: the megakernel feeds the model its own predictions; the "
                    "reference only checks the stream afterwards (results/video_*.json). In the "
                    "session, Qwen's replies are its own greedy tokens; the chat harness types "
                    "the command from Qwen's tool call as keystrokes at record boundaries "
                    "(call*_keys), the session's only inputs; the boot's only input is GO",
            "wrong": "tokens that differ from the reference machine's stream so far: 0 for "
                     "every run (each stream's sha256 equals the reference's)",
            "speed": "real decode seconds per video second, per piece (segments)",
            "page square": "1 square = 4 KB of the machine's 4.5 MB of memory; lit = a store "
                           "into it in this frame (fades over 0.5 s); free / in use / reserved "
                           "from the kernel's struct page array, read from the machine's RAM",
            "register cell": "lit = the register was written in this frame",
            "heartbeat": "a pulse per timer interrupt (TRAP token, mcause 0x80000007)",
            "layer bar": "model.layers 0-36 (the CPU) glow while machine tokens decode, "
                         "37-68 (Qwen) while Qwen writes; a CPU layer flashes when the state "
                         "it holds is written (registers: 4, privilege: 5, RAM: by address)",
            "zoom transitions": "dashed outlines are the zoom between levels (illustrative); "
                                "every value shown is read from the snapshot",
            "crash block": "the sliding 512 KB outline is illustrative: each aligned 512 KB "
                           "block shows its real count of free pages from struct page",
        },
        "console_divisor_bytes_hidden": con.hidden,
        "zoom": z and {k: v for k, v in z.items() if k not in ("classes", "raw")},
        "crash": cr and {k: v for k, v in cr.items() if k != "pagemap"},
    }
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--story", choices=["hero", "preview", "loop"], default="hero")
    ap.add_argument("--out", default=str(ROOT / "build" / "video" / "frames.json"))
    ap.add_argument("--facts", default=str(ROOT / "lx" / "video" / "video_facts.json"))
    ap.add_argument("--check-regenerate", action="store_true",
                    help="regenerate the streams from the reference (what a checkout without "
                         "build/video_data does) and compare their sha256 with the results")
    a = ap.parse_args()
    if a.check_regenerate:
        boot, fib = load(RES / "video_boot.json"), load(RES / "video_fib.json")
        bs = np.asarray(boot_reference(boot["tokens"]), dtype=np.int32)
        calls = [e for e in fib["chat"]["events"] if "machine_tokens" in e]
        ss = session_reference(calls, booted(bs))
        r = {"boot": sha_stream(bs) == boot["stream_sha256"],
             "calls": [sha_stream(s) == c["stream_sha256"] for s, c in zip(ss, calls)]}
        print(json.dumps(r))
        raise SystemExit(0 if r["boot"] and all(r["calls"]) else 1)
    build(a.story, Path(a.out), Path(a.facts))


if __name__ == "__main__":
    main()
