"""lx.fast against the stock forward, bit for bit: from the same state, a chunk of the boot trace
through every chip layer both ways, comparing the residual stream, the conv states and the
recurrent states after each layer, then the output head's argmax. Then the speed of both.

    python tests/check_fast.py [--weights ../build/hf/model] [--skip 2] [--chunk 32768]
"""

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from lx import hf
from lx import spec as S
from lx.fast import Fast
from lx.linux import ROOT, Linux
from lx.session import Session


def diff(a, b):
    if torch.equal(a, b):
        return None
    d = (a.float() - b.float()).abs()
    return {"n": int((d != 0).sum()) if d.isfinite().all() else -1, "max": float(d.max()),
            "first": [int(i) for i in (d != 0).nonzero()[0]]}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "build" / "hf" / "model"))
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--skip", type=int, default=2, help="chunks run (stock) before the compared one")
    ap.add_argument("--out", default=None)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--compile", action="store_true", help="lx.fast's compiled steps")
    a = ap.parse_args()
    lx = Linux(chunk=a.chunk, weights=Path(a.weights))
    s = Session(lx.c, lx.model, lx.reset_machine(), lx.chunk, lx.offset)
    C, n = lx.chunk, a.skip + 3
    toks = [S.tok("GO")]
    while len(toks) < n * C:
        toks += s.propose(max_records=10_000)
    for i in range(a.skip):
        r = s.verify(toks[i * C:(i + 1) * C])
        assert r["wrong"] == 0
    t0 = time.time()
    f = Fast(lx.model, s.n_chip, s.rows)
    f.compile = a.compile
    prep = time.time() - t0
    m, dev = lx.model, s.dev
    x = torch.tensor([toks[a.skip * C:(a.skip + 1) * C]], device=dev) + lx.offset
    cache_f = copy.deepcopy(s.cache)
    hs = m.model.embed_tokens(x)
    hf_ = hs[0].clone()
    bad = []
    for L in range(s.n_chip):
        hs = m.model.layers[L](hs, position_embeddings=None, attention_mask=None,
                               position_ids=None, past_key_values=s.cache)
        f.layer(L, hf_, cache_f)
        cs, cf = s.cache.layers[L], cache_f.layers[L]
        d = {"h": diff(hs[0], hf_), "conv": diff(cs.conv_states[0], cf.conv_states[0]),
             "state": diff(cs.recurrent_states[0], cf.recurrent_states[0])}
        d = {k: v for k, v in d.items() if v}
        if d:
            bad.append((L, d))
            print(f"layer {L}: {d}")
    ps = hf.head_argmax(m, hs[0], rows=s.rows)
    pf = f.head_argmax(hf_)
    head_same = torch.equal(ps, pf)
    print(f"layers differing: {len(bad)} of {s.n_chip}; head argmax identical: {head_same}")
    # speed, on the next chunk from the (now equal or not) states
    x = torch.tensor([toks[(a.skip + 1) * C:(a.skip + 2) * C]], device=dev) + lx.offset
    sc, fc = copy.deepcopy(s.cache), copy.deepcopy(cache_f)
    times = {}
    for name, fn in [("stock", lambda: hf.head_argmax(m, hf.chip_forward(
            m, m.model.embed_tokens(x), sc, s.n_chip)[0], rows=s.rows)),
                     ("fast", lambda: f.head_argmax(f.forward(m.model.embed_tokens(x), fc)[0]))]:
        torch.cuda.synchronize()
        t0 = time.time()
        p = fn()
        torch.cuda.synchronize()
        times[name] = time.time() - t0
        times[name + "_pred"] = p
    same_next = torch.equal(times.pop("stock_pred"), times.pop("fast_pred"))
    if a.profile:
        from torch.profiler import ProfilerActivity, profile

        x = torch.tensor([toks[(a.skip + 2) * C:(a.skip + 3) * C]], device=dev) + lx.offset
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            f.head_argmax(f.forward(m.model.embed_tokens(x), fc)[0])
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25,
                                        max_name_column_width=60))
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=12,
                                        max_name_column_width=60))
    res = {"chunk": C, "skip": a.skip, "prep_s": round(prep, 1), "layers_differing": len(bad),
           "differences": bad, "head_identical": head_same, "next_chunk_preds_identical": same_next,
           "stock_s": round(times["stock"], 3), "fast_s": round(times["fast"], 3),
           "stock_tok_per_s": round(C / times["stock"]), "fast_tok_per_s": round(C / times["fast"]),
           "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
    print(json.dumps({k: v for k, v in res.items() if k != "differences"}))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
