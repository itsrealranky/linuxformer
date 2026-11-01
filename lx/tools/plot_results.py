"""Charts from the results files: the boot's progress (lx.linux's timeline) and the shell
commands' latencies, verified (speculative) and pure.

    python tools/plot_results.py [--boot results/m4_boot_merged.json]
        [--shell results/m5_shell_merged.json] [--pure results/m5_pure_merged.json] OUT.png
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
    ap.add_argument("--boot", default="results/m4_boot_merged.json")
    ap.add_argument("--shell", default="results/m5_shell_merged.json")
    ap.add_argument("--pure", default="results/m5_pure_merged.json")
    a = ap.parse_args()
    boot_res = json.loads(Path(a.boot).read_text())
    boot = next(r for r in boot_res["runs"] if r["what"] == "boot")
    runs = [r for r in json.loads(Path(a.shell).read_text())["runs"] if r["what"] == "run"]
    pure = [r for r in json.loads(Path(a.pure).read_text())["runs"] if r["what"] == "pure"]

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
    s, toks, nbytes = zip(*([(0, 0, 0)] + boot["timeline"]))
    m = [t / 60 for t in s]
    ax1.plot(m, [t / 1e6 for t in toks], color="#2a7", lw=2, label="tokens verified")
    ax1.set_xlabel("minutes")
    ax1.set_ylabel("tokens verified (millions)", color="#2a7")
    ax1.set_ylim(0, None)
    ax2 = ax1.twinx()
    ax2.plot(m, nbytes, color="#36c", lw=1.5, label="console bytes")
    ax2.set_ylabel("console bytes predicted", color="#36c")
    ax2.set_ylim(0, None)
    ax1.set_title(f"Boot to shell: {boot['tokens']:,} tokens, {boot['seconds'] / 60:.1f} min "
                  f"({boot['tok_per_s']:,} tok/s), 0 wrong\n"
                  f"{boot_res['params_B']}B merged model, {boot_res['device']}", fontsize=10)

    labels = [r["cmd"] for r in runs]
    y = range(len(runs))
    ax3.barh(y, [r["seconds"] for r in runs], color="#2a7", label="verified (speculative)")
    for i, r in enumerate(runs):
        ax3.text(r["seconds"], i, f" {r['seconds']:.0f}s  ({r['tokens'] / 1e6:.2f}M tokens)",
                 va="center", fontsize=8)
    for p in pure:
        if p["cmd"] in labels:
            i = labels.index(p["cmd"])
            ax3.barh(i + 0.4, p["seconds"], height=0.3, color="#c63", label="pure (model alone)")
            ax3.text(p["seconds"], i + 0.4, f" {p['seconds'] / 60:.0f} min ({p['tok_per_s']:.0f} tok/s)",
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
