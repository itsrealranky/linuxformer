"""Launch videos from the runs' own logs, phone-sized (1080x1350) with large type:

  hero       the model-alone boot from mid-boot to the prompt, then Qwen's Fibonacci session,
             captions, a live token counter, the mode and the playback speed always on screen,
             an end card
  timelapse  the whole model-alone boot at one constant speed
  dmesg      the kernel's boot messages from the model-alone boot, readable

Boot: the model-alone stream is identical to the reference (mega_boot.json, same_as_reference),
so each console byte is placed at the token where the reference (tools/draft_sim.boot_stream)
has it; times come from the run's log (one line per 262,144-token launch, interpolated
linearly inside a launch). Chat: lx.chat's per-launch timelines (seconds, tokens, console bytes).

    python tools/render_launch.py hero OUT.mp4 [--chat results/m7_chat_fib_alone.json]
    python tools/render_launch.py timelapse OUT.mp4 [--speed 20]   (assets/: --speed 40)
    python tools/render_launch.py dmesg OUT.mp4 [--speed 20]       (assets/: --speed 4)
"""

import argparse
import bisect
import json
import re
import subprocess
import textwrap

from PIL import Image, ImageDraw, ImageFont

from lx import spec as S

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
SANS_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BG, PANEL, FG, DIM = (12, 14, 18), (24, 27, 35), (226, 231, 238), (140, 150, 160)
USER, QWEN, SHELL, ACC, RED = ((240, 200, 110), (120, 200, 255), (120, 220, 140),
                               (120, 220, 140), (255, 110, 100))
W, H, FPS, M = 1080, 1350, 30, 44
# escape sequences, and the boot console's two stray remnants (the reference prints them too)
JUNK = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|``m(?=Freeing)|^m(?=Linux version)")

cap_f, sub_f = ImageFont.truetype(SANS_BOLD, 50), ImageFont.truetype(SANS, 34)
big_f, lab_f = ImageFont.truetype(BOLD, 100), ImageFont.truetype(SANS, 32)
mode_f = ImageFont.truetype(SANS_BOLD, 30)


def term(size: int):
    f, b = ImageFont.truetype(FONT, size), ImageFont.truetype(BOLD, size)
    return f, b, int((W - 2 * M) // f.getlength("M")), int(size * 1.3)


def clean(text: str) -> str:
    return JUNK.sub("", text)


def term_lines(text: str, cols: int) -> list[str]:
    lines = []
    for raw in clean(text).replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        raw = "".join(c if c.isprintable() else " " for c in raw)
        while len(raw) > cols:
            lines.append(raw[:cols])
            raw = raw[cols:]
        lines.append(raw)
    return lines


def wrap_px(text: str, font, width: int) -> list[str]:
    out = []
    for para in text.split("\n"):
        line = ""
        for word in para.split(" "):
            t = f"{line} {word}".strip()
            if font.getlength(t) <= width or not line:
                line = t
            else:
                out.append(line)
                line = word
        out.append(line)
    return out


class Frame:
    def __init__(self):
        self.img = Image.new("RGB", (W, H), BG)
        self.d = ImageDraw.Draw(self.img)

    def caption(self, title: str, sub: str = "") -> int:
        y = 36
        for ln in wrap_px(title, cap_f, W - 2 * M):
            self.d.text((M, y), ln, font=cap_f, fill=FG)
            y += 62
        for ln in wrap_px(sub, sub_f, W - 2 * M) if sub else []:
            self.d.text((M, y + 4), ln, font=sub_f, fill=DIM)
            y += 44
        return y

    def panel(self, y0: int, rows: int, lines, size: int, fill, hl: str = "") -> int:
        """`rows` lines of (text, color, bold) from y0; a line containing hl is highlighted."""
        f, b, _, lh = term(size)
        y1 = y0 + rows * lh + 24
        self.d.rectangle((M - 18, y0, W - M + 18, y1), fill=fill)
        for i, (t, c, bold) in enumerate(lines[-rows:]):
            y = y0 + 12 + i * lh
            if hl and hl in t:
                self.d.rectangle((M - 10, y - 4, W - M + 10, y + lh - 4), fill=(90, 30, 30))
                c, bold = (255, 235, 230), True
            self.d.text((M, y), t, font=b if bold else f, fill=c)
        return y1

    def counter(self, y: int, n: int, total: int, label: str, clock: str) -> int:
        self.d.text((M, y), f"{n:,}", font=big_f, fill=ACC)
        self.d.text((M, y + 116), label, font=lab_f, fill=FG)
        bar = int((W - 2 * M) * min(n, total) / max(total, 1))
        self.d.rectangle((M, y + 166, W - M, y + 180), outline=DIM)
        self.d.rectangle((M, y + 166, M + bar, y + 180), fill=ACC)
        self.d.text((M, y + 194), clock, font=lab_f, fill=DIM)
        return y + 240

    def mode(self, text: str, speed: str) -> None:
        y = H - 64
        self.d.rectangle((0, y - 12, W, H), fill=(30, 34, 44))
        self.d.text((M, y), text, font=mode_f, fill=(255, 214, 120))
        self.d.text((W - M - mode_f.getlength(speed), y), speed, font=mode_f, fill=FG)


def speed_label(rate: float | None) -> str:
    if rate is None:
        return "speed: paused"
    return f"speed: {rate:,.0f}x" if rate >= 2 else f"speed: {rate:.1f}x"


# --- the model-alone boot ------------------------------------------------------------------

class Boot:
    def __init__(self, res="results/mega_boot.json", log="results/mega_boot.log"):
        from tools.draft_sim import boot_stream

        mk = json.load(open(res))
        assert mk["same_as_reference"]
        self.console, self.total, self.seconds = mk["console"], mk["tokens"], mk["seconds"]
        stream = [S.tok("GO")] + boot_stream().tolist()
        assert len(stream) == self.total
        b0, nb = S.FAMILIES["BYTE"]
        self.byte_at = [i for i, t in enumerate(stream) if b0 <= t < b0 + nb]  # token index
        assert len(self.byte_at) == len(self.console.encode("latin-1", "replace")), \
            (len(self.byte_at), len(self.console))
        self.pts = [(0.0, 0)]
        for ln in open(log):
            if ln.startswith('{"tokens"'):
                r = json.loads(ln)
                self.pts.append((r["s"], r["tokens"]))
        self.pts[-1] = (self.seconds, self.total)

    def tokens_at(self, sec: float) -> int:
        for (s0, n0), (s1, n1) in zip(self.pts, self.pts[1:]):
            if sec <= s1:
                return int(n0 + (n1 - n0) * (sec - s0) / max(s1 - s0, 1e-9))
        return self.total

    def time_of(self, tok: int) -> float:
        for (s0, n0), (s1, n1) in zip(self.pts, self.pts[1:]):
            if tok <= n1:
                return s0 + (s1 - s0) * (tok - n0) / max(n1 - n0, 1)
        return self.seconds

    def text_at(self, sec: float) -> str:
        return self.console[:bisect.bisect_right(self.byte_at, self.tokens_at(sec))]


def clock(sec: float, total: float | None = None) -> str:
    m, s = divmod(int(sec), 60)
    t = f"{m:02d}:{s:02d}"
    if total is not None:
        tm, ts = divmod(int(total), 60)
        t += f" / {tm:02d}:{ts:02d}"
    return t


BOOT_MODE = "Model alone · video sped up"


def boot_segments(b: Boot, start_byte: int, scroll_s: float, ff_s: float, hold_s: float):
    """(real start, real end, video seconds) pieces: the first burst of kernel messages from
    start_byte, the quiet rest of the boot fast-forwarded, the prompt held."""
    t0 = b.time_of(b.byte_at[start_byte])
    burst_end = b.time_of(b.byte_at[2095])
    return [(t0, burst_end, scroll_s), (burst_end, b.seconds, ff_s), (b.seconds, b.seconds, hold_s)]


def boot_frames(b: Boot, segs, size=34, rows=12):
    _, _, cols, _ = term(size)
    for r0, r1, dv in segs:
        n = int(dv * FPS)
        for f in range(n):
            k = f / max(n - 1, 1)
            sec = r0 + (r1 - r0) * k
            hold = r1 == r0
            fr = Frame()
            if not hold:
                y = fr.caption("Linux booting inside the weights of Qwen3.5-4B",
                               "Model alone, no reference, sped up")
            else:
                y = fr.caption(f"{b.total:,} tokens, 0 wrong, {b.seconds / 60:.0f} min on one H100",
                               "Booted to the BusyBox shell. Model load not included.")
            lines = [(t, FG, False) for t in term_lines(b.text_at(sec), cols)]
            y = fr.panel(max(y, 230) + 10, rows, lines, size, (0, 0, 0))
            toks = b.tokens_at(sec)
            fr.counter(y + 26, toks, b.total, "tokens predicted by the model · wrong: 0",
                       f"decode time {clock(sec)} on one H100")
            fr.mode(BOOT_MODE, speed_label(None if hold else (r1 - r0) / dv))
            yield fr.img


# --- Qwen's session ---------------------------------------------------------------------------

def chat_wrap(prefix: str, text: str, color, cols: int, bold=False) -> list:
    out = []
    for i, ln in enumerate(text.splitlines() or [""]):
        for j, piece in enumerate(textwrap.wrap(ln, cols - len(prefix), drop_whitespace=False)
                                  or [""]):
            out.append(((prefix if i == 0 and j == 0 else " " * len(prefix)) + piece, color, bold))
    return out


def machine_clock(e: dict, target: float, ratio: float = 8.0) -> list[tuple[float, float]]:
    """(seconds into the command, video seconds): stretches that print play at speed p,
    the rest at ratio * p, with p chosen so the command takes `target` video seconds."""
    tl = e["machine_timeline"]
    tp = ts = 0.0
    s0, b0 = 0.0, 0
    for sec, _, nb in tl:
        if nb > b0:
            tp += sec - s0
        else:
            ts += sec - s0
        s0, b0 = sec, nb
    p = (tp + ts / ratio) / target
    pts, v, s0, b0 = [(0.0, 0.0)], 0.0, 0.0, 0
    for sec, _, nb in tl:
        v += (sec - s0) / (p if nb > b0 else p * ratio)
        pts.append((sec, v))
        s0, b0 = sec, nb
    return pts


def at(pts, tv):
    for (s0, v0), (s1, v1) in zip(pts, pts[1:]):
        if tv <= v1:
            k = 0.0 if v1 <= v0 else (tv - v0) / (v1 - v0)
            return s0 + k * (s1 - s0), (s1 - s0) / max(v1 - v0, 1e-9)
    return pts[-1][0], None


def tl_at(e: dict, sec: float) -> tuple[int, int]:
    toks = nbytes = 0
    for s, tk, nb in e["machine_timeline"]:
        if s <= sec:
            toks, nbytes = tk, nb
    return toks, nbytes


CRASH = "Unable to allocate RAM"


def chat_frames(run: dict, console0: str, qwen_speed=3.0, machine_s=(5.0, 4.5), hold=2.5,
                size=32):
    chat = run["chats"][0]
    alone = run.get("mode") == "model alone"
    mode = BOOT_MODE if alone else "Verified mode · video sped up"
    label = ("tokens predicted by the model · wrong: 0" if alone
             else "tokens checked by the model · wrong: 0")
    _, _, cols, _ = term(size)
    ev = chat["events"]
    total = sum(e.get("machine_tokens", 0) for e in ev)
    shots = []  # (kind, video seconds, data)
    shots.append(("user", 2.0, chat["question"]))
    ncmd = 0
    for e in ev:
        shots.append(("qwen", e["seconds"] / qwen_speed, e))
        if "command" in e:
            e["_clock"] = machine_clock(e, machine_s[min(ncmd, 1)])
            shots.append(("machine", e["_clock"][-1][1], e))
            ncmd += 1
    shots.append(("final", hold, None))
    lines, console, done_toks, real = [], console0, 0, 0.0
    crash_shown, ncmd = False, -1
    cap = ("Now Qwen uses it", "Qwen3.5-4B, the chat model in the same weights, gets a shell "
           "tool.")
    for kind, dv, data in shots:
        n = max(1, int(dv * FPS))
        if kind == "qwen":
            ncmd += "command" in data
        if kind == "qwen" and "command" in data:
            cap = (("Qwen writes a script and runs it on the computer inside itself", "")
                   if ncmd == 0 else
                   ("Qwen reads the kernel error and rewrites its script", ""))
        elif kind in ("qwen", "final"):
            cap = ("20th Fibonacci number: 4181", "")
        base_lines = list(lines)
        for f in range(n):
            frac = (f + 1) / n
            rate, hl, show_console, zoom = qwen_speed, "", console, None
            cur = list(base_lines)
            if kind == "user":
                cur += chat_wrap("you  ", data, USER, cols, True)
                rate = None
            elif kind == "qwen":
                text = data["qwen"].replace("**", "")
                if "command" in data:
                    before = text.split("<tool_call>")[0].strip()
                    body = (before + "\n" if before else "") + "$ " + data["command"]
                else:
                    body = text
                cur += chat_wrap("qwen ", body[:int(len(body) * frac)], QWEN, cols, True)
                real_now = real + data["seconds"] * frac
            elif kind == "machine":
                sec, rate = at(data["_clock"], frac * dv)
                toks, nb = tl_at(data, sec)
                show_console = console + data["machine_out"][:nb]
                done_now = done_toks + toks
                real_now = real + sec
                if CRASH in data["machine_out"][:nb] and not crash_shown:
                    zoom = True
            elif kind == "final":
                rate = None
                hl = "4181"
            fr = Frame()
            y = fr.caption(*cap)
            y = fr.panel(max(y, 200) + 6, 7, cur, size, PANEL)
            tlines = [(t, FG, False) for t in term_lines(show_console, cols)]
            y = fr.panel(y + 14, 8, tlines, size, (0, 0, 0), hl=hl)
            shown = done_now if kind == "machine" else done_toks
            fr.counter(y + 22, shown, total, label,
                       f"real time {clock(real_now if kind in ('qwen', 'machine') else real)}")
            fr.mode(mode, speed_label(rate))
            if zoom:
                yield from crash_frames(mode, show_console, hold, done_now, total, label,
                                        f"real time {clock(real_now)}")
                crash_shown = True
            yield fr.img
        if kind == "user":
            lines += chat_wrap("you  ", data, USER, cols, True) + [("", FG, False)]
        elif kind == "qwen":
            lines += cur[len(base_lines):] + [("", FG, False)]
            real += data["seconds"]
        elif kind == "machine":
            console += data["machine_out"]
            done_toks += data.get("machine_tokens", 0)
            real += data.get("machine_seconds", 0.0)
            lines += chat_wrap("  ", "[output in the terminal below]", SHELL, cols) + \
                [("", FG, False)]


def crash_frames(mode: str, console: str, hold: float, toks: int, total: int, label: str,
                 clk: str):
    """The terminal zoomed onto the kernel's allocation error, held."""
    _, _, cols, _ = term(46)
    lines = term_lines(console, cols)
    first = next(j for j, t in enumerate(lines) if "page allocation failure" in t)
    i = next(j for j, t in enumerate(lines) if "allocate RAM" in t)
    ctx = lines[first:first + 2] + ["  ..."] + lines[i:i + 2]
    for _ in range(int(hold * FPS)):
        z = Frame()
        y = z.caption("Crash: the inner computer ran out of memory (4.5 MB RAM)", "")
        y = z.panel(max(y, 200) + 20, len(ctx), [(t, DIM, False) for t in ctx], 46, (0, 0, 0),
                    hl="allocate RAM")
        z.counter(y + 40, toks, total, label, clk)
        z.mode(mode, "speed: paused")
        yield z.img


head_f, body_f = ImageFont.truetype(SANS_BOLD, 58), ImageFont.truetype(SANS, 44)
CARD = [("No training.", head_f, ACC), ("Qwen's weights", head_f, FG),
        ("bit-identical.", head_f, FG), ("", body_f, FG), ("37 compiled CPU layers.", body_f, FG),
        ("Linux 6.8 on RISC-V.", body_f, FG), ("", body_f, FG), ("linuxformer", body_f, DIM)]


def card(mode: str) -> Image.Image:
    fr = Frame()
    rows = [(t, f, c, 76 if f is head_f else 60) for t, f, c in CARD]
    assert all(f.getlength(t) <= W - 2 * M - 40 for t, f, _, _ in rows)
    y = (H - sum(r[3] for r in rows)) // 2
    for ln, font, color, lh in rows:
        fr.d.text((M + 20, y), ln, font=font, fill=color)
        y += lh
    fr.mode(mode, "speed: —")
    return fr.img


# --- outputs ---------------------------------------------------------------------------------

def timelapse_frames(b: Boot, speed: float, size=30, rows=16):
    _, _, cols, _ = term(size)
    n = int(b.seconds / speed * FPS) + 1
    for f in range(n + int(3 * FPS)):
        sec = min(b.seconds, f / FPS * speed)
        fr = Frame()
        y = fr.caption(f"Linux boot, model alone, no reference, {b.seconds / 60:.0f} min",
                       "Qwen3.5-4B + 37 compiled layers, one H100. Every character below is a "
                       "token the model predicted.")
        lines = [(t, FG, False) for t in term_lines(b.text_at(sec), cols)]
        y = fr.panel(max(y, 250) + 10, rows, lines, size, (0, 0, 0))
        fr.counter(y + 24, b.tokens_at(sec), b.total, "tokens predicted · wrong: 0",
                   f"decode time {clock(sec, b.seconds)}")
        fr.mode(BOOT_MODE, f"speed: {speed:g}x" if sec < b.seconds else "speed: paused")
        yield fr.img


def dmesg_frames(b: Boot, speed: float, size=36, rows=15):
    _, _, cols, _ = term(size)
    t0 = b.time_of(b.byte_at[0]) - 1
    t1 = b.time_of(b.byte_at[2095])
    n = int((t1 - t0) / speed * FPS)
    for f in range(n + int(2 * FPS)):
        sec = min(t1, t0 + f / FPS * speed)
        fr = Frame()
        y = fr.caption("The kernel's boot log, model alone",
                       "Linux 6.8 (no MMU) inside Qwen3.5-4B's weights")
        lines = [(t, FG, False) for t in term_lines(b.text_at(sec), cols)]
        y = fr.panel(max(y, 220) + 10, rows, lines, size, (0, 0, 0))
        fr.d.text((M, y + 24), f"decode time {clock(sec)}   ·   token {b.tokens_at(sec):,}",
                  font=lab_f, fill=DIM)
        fr.mode(BOOT_MODE, f"speed: {speed:g}x" if sec < t1 else "speed: paused")
        yield fr.img


def write(out: str, gens) -> dict:
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt",
                           "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v",
                           "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "slow",
                           "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    n = 0
    for gen in gens:
        for img in gen:
            ff.stdin.write(img.tobytes())
            n += 1
    ff.stdin.close()
    ff.wait()
    return {"out": out, "frames": n, "seconds": round(n / FPS, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["hero", "timelapse", "dmesg"])
    ap.add_argument("out")
    ap.add_argument("--boot", default="results/mega_boot.json")
    ap.add_argument("--log", default="results/mega_boot.log")
    ap.add_argument("--chat", default="results/m7_chat_fib_alone.json")
    ap.add_argument("--speed", type=float, default=20.0)
    a = ap.parse_args()
    b = Boot(a.boot, a.log)
    if a.what == "hero":
        run = json.load(open(a.chat))
        mode = BOOT_MODE if run.get("mode") == "model alone" else \
            "Verified mode · video sped up"
        gens = [boot_frames(b, boot_segments(b, 700, 5.0, 2.0, 3.0)),
                chat_frames(run, b.console),
                (card(mode) for _ in range(3 * FPS))]
    elif a.what == "timelapse":
        gens = [timelapse_frames(b, a.speed)]
    else:
        gens = [dmesg_frames(b, a.speed)]
    print(json.dumps(write(a.out, gens)))


if __name__ == "__main__":
    main()
