"""Render lx.chat's results (questions, Qwen's tool calls, the machine's output, Qwen's answers)
as an image.

    python tools/render_chat.py results/m6_chat.json out.png
"""

import argparse
import json
import textwrap

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
COLS = 96
BG, FG, DIM = (12, 14, 18), (215, 222, 230), (120, 130, 140)
USER, QWEN, SHELL = (240, 200, 110), (120, 200, 255), (120, 220, 140)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("out")
    a = ap.parse_args()
    d = json.loads(open(a.results).read())
    lines = [("Qwen3.5-4B using the shell of the Linux machine in its own weights", QWEN, True),
             (f"{d['params_B']}B parameters, one H100; every machine byte predicted and verified", DIM,
              False), ("", FG, False)]

    def add(prefix, text, color, bold=False):
        for i, ln in enumerate(text.strip().splitlines() or [""]):
            for j, piece in enumerate(textwrap.wrap(ln, COLS - len(prefix)) or [""]):
                lines.append(((prefix if i == 0 and j == 0 else " " * len(prefix)) + piece, color,
                              bold))

    for c in d["chats"]:
        add("user  ", c["question"], USER, True)
        for e in c["events"]:
            if "command" in e:
                add("qwen  ", f"shell: {e['command']}", QWEN)
                add("  $   ", e["response"], SHELL)
                add("      ", f"[{e['machine_tokens']:,} machine tokens, {e['machine_seconds']:.0f} s, "
                    f"{e['wrong']} wrong]", DIM)
            else:
                add("qwen  ", e["qwen"], QWEN, True)
        lines.append(("", FG, False))
    font, bold = ImageFont.truetype(FONT, 16), ImageFont.truetype(BOLD, 16)
    cw, lh = font.getbbox("M")[2], 22
    img = Image.new("RGB", (cw * COLS + 48, lh * len(lines) + 40), BG)
    dr = ImageDraw.Draw(img)
    for i, (text, color, b) in enumerate(lines):
        dr.text((24, 20 + i * lh), text, font=bold if b else font, fill=color)
    img.save(a.out)


if __name__ == "__main__":
    main()
