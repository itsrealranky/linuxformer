"""Charts from the results files: the boot from reset to the shell prompt in each mode, and the
shell commands' latencies, verified mode and model alone.

    python tools/plot_results.py [--shell results/m5_shell_merged.json]
        [--pure results/m5_pure_merged.json] OUT.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--shell", default="results/m5_shell_merged.json")
    ap.add_argument("--pure", default="results/m5_pure_merged.json")
    a = ap.parse_args()
    runs = [r for r in json.loads(Path(a.shell).read_text())["runs"] if r["what"] == "run"]
    pure = [r for r in json.loads(Path(a.pure).read_text())["runs"] if r["what"] == "pure"]

    res = Path(a.shell).parent
    boots = [  # (label, results file, the run's seconds and tok/s)
        ("verified mode, stock kernels", "m4_boot_merged.json"),
        ("verified mode, lx.fast", "fast_boot_compiled.json"),
        ("model alone, megakernel", "mega_boot.json")]
    rows, tokens = [], set()
    for label, name in boots:
        d = json.loads((res / name).read_text())
        r = next(r for r in d["runs"] if r["what"] == "boot") if "runs" in d else d
        assert r.get("same_as_reference", True), name
        tokens.add(r["tokens"])
        rows.append((label, r["seconds"], r["tok_per_s"], d["device"], d.get("params_B")))

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
    y = range(len(rows))
    ax1.barh(y, [r[1] for r in rows], color=["#2a7"] * 2 + ["#c63"] * (len(rows) - 2))
    for i, (_, sec, tps, _, _) in enumerate(rows):
        t = f"{sec / 3600:.2f} h" if sec > 7200 else f"{sec / 60:.1f} min"
        ax1.text(sec, i, f" {t} ({tps:,.0f} tok/s)", va="center", fontsize=8)
    ax1.set_yticks(list(y), [r[0] for r in rows], fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlim(0, max(r[1] for r in rows) * 1.45)
    ax1.set_xlabel("seconds")
    assert len(tokens) == 1, tokens
    ax1.set_title(f"Boot from reset to the shell prompt: {tokens.pop():,} tokens, 0 wrong\n"
                  f"{rows[0][4]}B merged model, {rows[0][3]}", fontsize=10)

    labels = [r["cmd"] for r in runs]
    y = range(len(runs))
    ax3.barh(y, [r["seconds"] for r in runs], color="#2a7", label="verified mode")
    for i, r in enumerate(runs):
        ax3.text(r["seconds"], i, f" {r['seconds']:.0f}s  ({r['tokens'] / 1e6:.2f}M tokens)",
                 va="center", fontsize=8)
    for p in pure:
        if p["cmd"] in labels:
            i = labels.index(p["cmd"])
            ax3.barh(i + 0.4, p["seconds"], height=0.3, color="#c63", label="model alone (stock decode)")
            ax3.text(p["seconds"], i + 0.4, f" {p['seconds'] / 60:.0f} min ({p['tok_per_s']:.1f} tok/s)",
                     va="center", fontsize=8)
    ax3.set_yticks(list(y), labels, fontsize=8)
    ax3.invert_yaxis()
    ax3.set_xscale("log")
    ax3.set_xlabel("seconds (log)")
    ax3.set_title("Shell commands, from the booted snapshot", fontsize=10)
    ax3.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)


if __name__ == "__main__":
    main()
