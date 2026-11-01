"""Export the merged model as a Hugging Face directory (lx.merge.save) with the booted snapshot
as safetensors, then check both: reload the model (lx.merge.load) and compare every tensor with
the fresh build, bit for bit, and reload the snapshot and compare every field.

    python tools/export_hf.py OUT [--snapshot ../build/snapshots/boot-merged.pt]
        [--card ../docs/launch/model_card.md] [--json results/m8_export.json]
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import torch

from lx import core, merge
from lx.linux import ROOT
from lx.session import load_safe, save_safe


def tensors_differ(a: dict, b: dict) -> list[str]:
    bad = sorted(set(a) ^ set(b))
    for k in set(a) & set(b):
        x, y = a[k], b[k]
        if x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x, y):
            bad.append(k)
    return bad


def snapshot_differ(d: dict, e: dict) -> list[str]:
    bad = [k for k in ("next", "last", "n", "waiting") if d[k] != e[k]]
    if len(d["state"]) != len(e["state"]):
        return bad + ["state"]
    for L, ((a, b), (c, f)) in enumerate(zip(d["state"], e["state"])):
        if tensors_differ({"c": a, "r": b}, {"c": c, "r": f}):
            bad.append(f"state.{L}")
    va, vb = vars(d["ref"]), vars(e["ref"])
    bad += [f"ref.{k}" for k in sorted(set(va) | set(vb))
            if k not in va or k not in vb or va[k] != vb[k] or type(va[k]) is not type(vb[k])]
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--snapshot", default=str(ROOT / "build" / "snapshots" / "boot-merged.pt"))
    ap.add_argument("--card", default=str(ROOT / "docs" / "launch" / "model_card.md"))
    ap.add_argument("--json", default="results/m8_export.json")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)

    t0 = time.time()
    c = core.build(merge.config())
    model, info = merge.build(c)
    build_s = time.time() - t0
    info["params_B"] = round(sum(p.numel() for p in model.parameters()) / 1e9, 2)
    t0 = time.time()
    shards = merge.save(model, info, out)
    save_s = time.time() - t0
    t0 = time.time()
    loaded, _ = merge.load(out, c)
    load_s = time.time() - t0
    bad = tensors_differ(model.state_dict(), loaded.state_dict())
    ca, cb = model.config.to_dict(), loaded.config.to_dict()
    cfg_bad = sorted(k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k))
    dtypes = {}
    for p in loaded.state_dict().values():
        dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + 1
    n_tensors = len(loaded.state_dict())
    del model, loaded
    torch.cuda.empty_cache()

    d = torch.load(a.snapshot, weights_only=False)
    snap = out / "boot-snapshot.safetensors"
    save_safe(d, snap)
    snap_bad = snapshot_differ(d, load_safe(snap))
    if Path(a.card).exists():
        shutil.copy(a.card, out / "README.md")

    weights_bytes = sum(f.stat().st_size for f in shards)
    total_bytes = sum(f.stat().st_size for f in out.iterdir() if f.is_file())
    res = {"device": torch.cuda.get_device_name(), "out": str(out), "params_B": info["params_B"],
           "shards": [f.name for f in shards], "weights_bytes": weights_bytes,
           "weights_GB": round(weights_bytes / 1e9, 2), "snapshot_bytes": snap.stat().st_size,
           "total_bytes": total_bytes, "tensors": n_tensors, "dtypes": dtypes,
           "tensors_differ": bad, "config_keys_differ": cfg_bad, "snapshot_fields_differ": snap_bad,
           "build_s": round(build_s, 1), "save_s": round(save_s, 1), "load_s": round(load_s, 1)}
    Path(a.json).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
