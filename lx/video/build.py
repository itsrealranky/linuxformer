"""A HyperFrames composition from the player and a frames file: player.html with the layout,
the size, the frames and the fonts filled in, as one self-contained index.html (no network).

    python video/build.py FRAMES.json OUT_DIR --layout 16x9|9x16|loop [--first N] [--count N]
"""

import argparse
import base64
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIZES = {"16x9": (1920, 1080), "9x16": (1080, 1920), "loop": (1100, 760)}
FONTS = [("LXInter", w, f"Inter-{n}.woff2") for w, n in
         [(400, "Regular"), (500, "Medium"), (600, "SemiBold"), (700, "Bold"), (800, "ExtraBold")]] + \
        [("LXMono", w, f"JetBrainsMono-{n}.woff2") for w, n in
         [(400, "Regular"), (500, "Medium"), (700, "Bold")]]
HF_CONFIG = {"$schema": "https://hyperframes.heygen.com/schema/hyperframes.json",
             "paths": {"blocks": "compositions", "components": "compositions/components",
                       "assets": "assets"}, "media": {"autoProxy": True}}


def font_faces() -> str:
    out = []
    for fam, w, f in FONTS:
        b = base64.b64encode((HERE / "fonts" / f).read_bytes()).decode()
        out.append(f"@font-face {{ font-family: '{fam}'; font-weight: {w}; font-style: normal; "
                   f"font-display: block; src: url(data:font/woff2;base64,{b}) format('woff2'); }}")
    return "\n".join(out)


def build(frames: Path, out: Path, layout: str, first: int = 0, count: int | None = None):
    data = json.loads(frames.read_text())
    n = data["meta"]["frames"]
    count = n - first if count is None else count
    assert 0 <= first and first + count <= n, (first, count, n)
    w, h = SIZES[layout]
    html = (HERE / "player.html").read_text()
    for k, v in {"__W__": w, "__H__": h, "__LAYOUT__": layout, "__OFFSET__": first,
                 "__COUNT__": count, "__DURATION__": f"{count / data['meta']['fps']:.6f}"}.items():
        html = html.replace(k, str(v))
    html = html.replace("/*__FONTS__*/", font_faces())
    html = html.replace("/*__FRAMES__*/", "window.FRAMES = " + frames.read_text() + ";")
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(html)
    (out / "hyperframes.json").write_text(json.dumps(HF_CONFIG, indent=1))
    return {"dir": str(out), "frames": count, "seconds": count / data["meta"]["fps"],
            "size": [w, h], "bytes": len(html)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("out")
    ap.add_argument("--layout", choices=list(SIZES), required=True)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    a = ap.parse_args()
    print(json.dumps(build(Path(a.frames), Path(a.out), a.layout, a.first, a.count)))


if __name__ == "__main__":
    main()
