"""The hero video, its cut-downs and their checks, from the published results (`make video`).

    python video/render.py [--skip-export] [--no-repeat]

1. video/export_frames.py: every frame as data (hero, loop) and video/video_facts.json.
2. video/build.py: one self-contained HyperFrames composition per output (player.html, the
   frames and the fonts inlined; nothing is fetched).
3. HyperFrames, pinned in video/package.json (Node >= 22.20; telemetry off; Chrome in software
   mode, so frames do not depend on a GPU): each frame is player.html's renderFrame(n), encoded
   H.264 High, yuv420p, 60 fps, faststart.
4. Checks, written to build/video/out/qa.json (exit 1 if any fails): X's upload limits, no blank
   frames, the encoded frames match the player's own frames, a second render of the hero is
   frame-for-frame identical, every caption number comes from the fact sheet, no banned words.

Outputs in build/video/out: hero_16x9.mp4, hero_9x16.mp4, zoom_clip.mp4, boot_loop.gif,
thumb_{1,2,3}_*.png, video_facts.json, qa.json. Nothing is uploaded anywhere.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LX = HERE.parent
ROOT = LX.parent
FRAMES = ROOT / "build" / "video"
PROJ = FRAMES / "hf"
OUT = FRAMES / "out"
FACTS = HERE / "video_facts.json"
sys.path.insert(0, str(HERE))
import build as B  # noqa: E402

FPS = 60
X_LIMITS = {"seconds": 140, "bytes": 512 << 20, "landscape": (1920, 1200),
            "portrait": (1200, 1920), "fps": 60}
BANNED = r"\b(neurons?|thinks?|thinking|understands?|understanding|brains?)\b"
# product names whose digits are not facts
NAMES = ["Qwen3.5-4B", "RISC-V", "video_*.json"]


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(list(map(str, cmd)), check=True, **kw)


def node_env() -> dict:
    env = dict(os.environ, HYPERFRAMES_NO_TELEMETRY="1", DO_NOT_TRACK="1")
    if env.get("NODE_BIN"):
        env["PATH"] = env["NODE_BIN"] + os.pathsep + env["PATH"]
    v = subprocess.run(["node", "--version"], env=env, capture_output=True, text=True).stdout
    major, minor = (int(x) for x in v.strip().lstrip("v").split(".")[:2])
    if (major, minor) < (22, 20):
        sys.exit(f"HyperFrames needs Node >= 22.20 (found {v.strip()}); set NODE_BIN to its bin dir")
    if not (HERE / "node_modules" / ".bin" / "hyperframes").exists():
        run(["npm", "ci", "--no-audit", "--no-fund"], cwd=HERE, env=env)
    return env


def hf(env, *args, **kw):
    return run([HERE / "node_modules" / ".bin" / "hyperframes", *args], env=env, **kw)


def export(story: str) -> Path:
    out = FRAMES / f"frames_{story}.json"
    run([sys.executable, HERE / "export_frames.py", "--story", story, "--out", out,
         "--facts", FACTS], cwd=LX, env=dict(os.environ, PYTHONPATH="."))
    return out


def render(env, proj: Path, out: Path, fmt="mp4"):
    args = ["render", proj, "--no-browser-gpu", "-o", out]
    args += ["--fps", FPS, "--quality", "delivery"] if fmt == "mp4" else \
        ["--format", "gif", "--fps", 15, "--gif-loop", 0]
    hf(env, *args, stdout=subprocess.DEVNULL)
    return out


def snapshots(env, proj: Path, seconds: list[float]) -> list[Path]:
    d = proj / ".hyperframes" / "artifacts" / "snapshots"
    shutil.rmtree(d, ignore_errors=True)
    hf(env, "snapshot", proj, "--at", ",".join(f"{s:.4f}" for s in seconds), "--no-end",
       "--no-browser-gpu", stdout=subprocess.DEVNULL)
    pngs = sorted(d.glob("*/frame-*.png"))
    assert len(pngs) == len(seconds), pngs
    return pngs


# --- checks --------------------------------------------------------------------------------------

def probe(path: Path) -> dict:
    j = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
        capture_output=True, text=True, check=True).stdout)
    v = [s for s in j["streams"] if s["codec_type"] == "video"][0]
    w, h = v["width"], v["height"]
    lim = X_LIMITS["landscape"] if w >= h else X_LIMITS["portrait"]
    num, den = map(int, v["r_frame_rate"].split("/"))
    atoms, data, i = [], path.read_bytes()[:1 << 20], 0
    while i + 8 <= len(data):
        size = int.from_bytes(data[i:i + 4], "big")
        atoms.append(data[i + 4:i + 8].decode("latin1"))
        if size < 8:
            break
        i += size
    r = {"codec": v["codec_name"], "profile": v.get("profile"), "pix_fmt": v["pix_fmt"],
         "size": [w, h], "fps": num / den, "frames": int(v.get("nb_frames", 0)),
         "seconds": float(j["format"]["duration"]), "bytes": int(j["format"]["size"]),
         "color": [v.get("color_space"), v.get("color_primaries"), v.get("color_transfer")],
         "audio": any(s["codec_type"] == "audio" for s in j["streams"]),
         "top_atoms": atoms}
    r["faststart"] = "moov" in atoms and ("mdat" not in atoms or atoms.index("moov") <
                                          atoms.index("mdat"))
    r["x_ok"] = (r["codec"] == "h264" and r["profile"] == "High" and r["pix_fmt"] == "yuv420p"
                 and r["fps"] <= X_LIMITS["fps"] and w <= lim[0] and h <= lim[1]
                 and r["seconds"] <= X_LIMITS["seconds"] and r["bytes"] <= X_LIMITS["bytes"]
                 and r["faststart"])
    return r


def frame_stats(path: Path) -> dict:
    """Per-frame brightest luma: a frame with no text or lit square in it would be blank."""
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf",
                        "signalstats,metadata=print:key=lavfi.signalstats.YMAX:file=-",
                        "-f", "null", "-"], capture_output=True, text=True, check=True).stdout
    ymax = [int(float(x)) for x in re.findall(r"YMAX=([\d.]+)", p)]
    blank = [i for i, y in enumerate(ymax) if y < 128]
    return {"frames": len(ymax), "min_ymax": min(ymax), "blank_frames": blank[:20],
            "n_blank": len(blank)}


def frame_md5(path: Path) -> list[str]:
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "framemd5", "-"],
                       capture_output=True, text=True, check=True).stdout
    return [l.split(",")[-1].strip() for l in p.splitlines() if l and not l.startswith("#")]


def psnr(png: Path, mp4: Path, n: int) -> float:
    p = subprocess.run(["ffmpeg", "-v", "info", "-i", png, "-i", mp4, "-lavfi",
                        f"[1:v]select=eq(n\\,{n}),setpts=PTS-STARTPTS[b];[0:v]format=yuv420p[a];"
                        "[a][b]psnr", "-frames:v", "1", "-f", "null", "-"],
                       capture_output=True, text=True).stderr
    m = re.search(r"average:([\d.]+|inf)", p)
    return float("inf") if m.group(1) == "inf" else float(m.group(1))


def matches_player(env, proj: Path, mp4: Path, frames: list[int]) -> list[dict]:
    """The player's own frame n (a snapshot) against the encoded frame n and its neighbours:
    the encode must hold frame n at n (closest by PSNR), not a frame before or after."""
    pngs = snapshots(env, proj, [n / FPS for n in frames])
    out = []
    for n, png in zip(frames, pngs):
        s = {d: psnr(png, mp4, n + d) for d in (-1, 0, 1)}
        out.append({"frame": n, "psnr_db": {str(k): round(v, 2) for k, v in s.items()},
                    "ok": s[0] >= 30 and s[0] > max(s[-1], s[1])})
    return out


def text_checks(frames: Path) -> dict:
    meta = json.loads(frames.read_text())["meta"]
    facts = json.loads(FACTS.read_text())
    fact_texts = {v["text"] for v in facts["facts"].values()}
    caps = meta["captions"]
    same = all(facts["captions"][k]["big"] == b and facts["captions"][k]["small"] == s
               for k, (b, s) in caps.items())
    stray = {}
    for k, (b, s) in caps.items():
        t = f"{b} {s}"
        for nm in NAMES:
            t = t.replace(nm, " ")
        for num in re.findall(r"\d[\d,.]*\d|\d", t):
            if not any(num in ft for ft in fact_texts):
                stray.setdefault(k, []).append(num)
    words = "\n".join(f"{b}\n{s}" for b, s in caps.values()) + (HERE / "player.html").read_text()
    banned = sorted(set(m.lower() for m in re.findall(BANNED, words, re.I)))
    missing = [k for k, (b, s) in caps.items() if "?" in b + s]
    cols = json.loads(frames.read_text())["cols"]
    card = [i for i, sg in enumerate(meta["segments"]) if sg["id"] == "card"]
    no_speed = [n for n, (sg, sp) in enumerate(zip(cols["seg"], cols["speed"]))
                if sg not in card and not meta["speeds"][sp]]
    return {"captions_equal_fact_sheet": same, "caption_numbers_not_in_facts": stray,
            "captions_missing_facts": missing, "banned_words": banned,
            "frames_without_speed_label": len(no_speed)}


# --- main ----------------------------------------------------------------------------------------

def segment(meta, sid):
    s = meta["segments"]
    i = [k for k, x in enumerate(s) if x["id"] == sid][0]
    end = s[i + 1]["start"] if i + 1 < len(s) else meta["frames"]
    return s[i]["start"], end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-export", action="store_true", help="reuse build/video/frames_*.json")
    ap.add_argument("--no-repeat", action="store_true", help="skip the second hero render")
    a = ap.parse_args()
    env = node_env()
    OUT.mkdir(parents=True, exist_ok=True)

    hero = FRAMES / "frames_hero.json" if a.skip_export else export("hero")
    loop = FRAMES / "frames_loop.json" if a.skip_export else export("loop")
    meta = json.loads(hero.read_text())["meta"]
    z0, z1 = segment(meta, "zoom")
    zoom_first, zoom_count = z0 - 90, (z1 - z0) + 90 + 60  # 1.5 s before, 1 s after
    projects = {
        "hero_16x9": B.build(hero, PROJ / "hero_16x9", "16x9"),
        "hero_9x16": B.build(hero, PROJ / "hero_9x16", "9x16"),
        "zoom_clip": B.build(hero, PROJ / "zoom_clip", "16x9", zoom_first, zoom_count),
        "boot_loop": B.build(loop, PROJ / "boot_loop", "loop"),
    }
    for name in ("hero_16x9", "hero_9x16", "zoom_clip"):
        render(env, PROJ / name, OUT / f"{name}.mp4")
    render(env, PROJ / "boot_loop", OUT / "boot_loop.gif", fmt="gif")

    # thumbnails: the decoded word, the failure, the answer
    c0, c1 = segment(meta, "crash")
    a0, a1 = segment(meta, "answer")
    thumbs = {"thumb_1_zoom": z0 + int(8.9 * FPS), "thumb_2_crash": c0 + int(1.6 * FPS),
              "thumb_3_answer": a1 - 30}
    for (name, n), png in zip(thumbs.items(), snapshots(env, PROJ / "hero_16x9",
                                                        [n / FPS for n in thumbs.values()])):
        shutil.copy(png, OUT / f"{name}.png")
    shutil.copy(FACTS, OUT / "video_facts.json")

    qa = {"outputs": {}, "projects": projects}
    for name in ("hero_16x9", "hero_9x16", "zoom_clip"):
        p = OUT / f"{name}.mp4"
        qa["outputs"][name] = probe(p) | {"stats": frame_stats(p),
                                          "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    gif = OUT / "boot_loop.gif"
    qa["outputs"]["boot_loop"] = {"bytes": gif.stat().st_size, "under_15mb": gif.stat().st_size
                                  < 15 << 20}
    # frames in running pieces, where every frame differs from its neighbours
    spots = [segment(meta, "cold")[0] + 30, segment(meta, "ff")[0] + 200,
             segment(meta, "fix")[0] + 200]
    qa["matches_player"] = matches_player(env, PROJ / "hero_16x9", OUT / "hero_16x9.mp4", spots)
    if not a.no_repeat:
        again = FRAMES / "repeat_hero_16x9.mp4"
        render(env, PROJ / "hero_16x9", again)
        m1, m2 = frame_md5(OUT / "hero_16x9.mp4"), frame_md5(again)
        qa["repeat_render"] = {
            "frames": len(m1), "identical_frames": sum(x == y for x, y in zip(m1, m2)),
            "all_identical": m1 == m2,
            "file_identical": hashlib.sha256(again.read_bytes()).hexdigest()
            == qa["outputs"]["hero_16x9"]["sha256"],
            "spot_check": {str(n): [m1[n], m2[n]] for n in spots}}
        again.unlink()
    qa["text"] = text_checks(hero)
    t = qa["text"]
    qa["passed"] = bool(
        all(o["x_ok"] and o["stats"]["n_blank"] == 0 for k, o in qa["outputs"].items()
            if k != "boot_loop") and qa["outputs"]["boot_loop"]["under_15mb"]
        and all(m["ok"] for m in qa["matches_player"])
        and (a.no_repeat or qa["repeat_render"]["all_identical"])
        and t["captions_equal_fact_sheet"] and not t["caption_numbers_not_in_facts"]
        and not t["captions_missing_facts"] and not t["banned_words"]
        and t["frames_without_speed_label"] == 0)
    (OUT / "qa.json").write_text(json.dumps(qa, indent=1, ensure_ascii=False))
    print(json.dumps({"passed": qa["passed"], "out": str(OUT)}))
    sys.exit(0 if qa["passed"] else 1)


if __name__ == "__main__":
    main()
