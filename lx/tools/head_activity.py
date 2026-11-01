"""How much GDN work lx.fast could still skip: per chip layer, how many live value heads have a
nonzero write gate (beta), change their state, or give a nonzero output, token by token in the
decode; and how many (key head, 64-token block) pairs the prefill runs.

    python tools/head_activity.py [--tokens 2000] [--cmd "echo hi"] [--out results/head_activity.json]
"""

import argparse
import copy
import json
from pathlib import Path

import torch

from lx.linux import ROOT, Linux
from lx.session import Session


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "build" / "hf" / "model"))
    ap.add_argument("--snapshot", default=str(ROOT / "build" / "snapshots" / "boot-merged.pt"))
    ap.add_argument("--cmd", default="echo hi")
    ap.add_argument("--tokens", type=int, default=2000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    lx = Linux(weights=Path(a.weights), fast=True)
    f, m = lx.fast, lx.model
    s = Session.load(lx.c, m, a.snapshot, lx.chunk, lx.offset, f)
    ref = copy.deepcopy(s.ref)
    toks = []
    for b in a.cmd.encode() + b"\n":
        toks += ref.key(b)
    while len(toks) < a.tokens:
        toks += ref.step()
    toks = toks[:a.tokens]

    # the decode, teacher-forced on the reference's tokens, with each layer's heads inspected
    f.dec = f.dec or [f._dec(L) for L in range(f.n_chip)]
    names = ("live", "beta_nonzero", "state_changed", "output_nonzero", "beta0_state_changed")
    per = {L: dict.fromkeys(names, 0) for L, d in enumerate(f.dec) if d.gdn and d.nkh}
    for t in toks:
        x = m.model.embed_tokens(torch.tensor([[t + lx.offset]], device=s.dev))[0].clone()
        for L, d in enumerate(f.dec):
            if d.gdn:
                cl = s.cache.layers[L]
                qkv = f._gdn_pre(d, x, cl.conv_states[0])
                if qkv is not None:
                    q, k, v, beta, g, z = qkv
                    rs = cl.recurrent_states[0]
                    nhv = v.shape[2]
                    before = rs[0, d.hv].clone().view(nhv, -1).view(torch.int32)
                    o = f._gdn_rec(d, q, k, v, beta, g, rs)
                    changed = (rs[0, d.hv].reshape(nhv, -1).view(torch.int32) != before).any(1)
                    norm = m.model.layers[L].linear_attn.norm
                    out = (norm(o.view(z.shape), z) != 0).any(1)
                    b0 = beta.view(-1) == 0
                    st = per[L]
                    st["live"] += nhv
                    st["beta_nonzero"] += int((~b0).sum())
                    st["state_changed"] += int(changed.sum())
                    st["output_nonzero"] += int(out.sum())
                    st["beta0_state_changed"] += int((b0 & changed).sum())
                    f._gdn_post(norm, d, x, o, z)
            if d.mlp:
                f._mlp_step(d, x)
    n = len(toks)
    dec = {L: {k: round(v / n, 2) for k, v in st.items()} for L, st in per.items()}
    tot = {k: round(sum(st[k] for st in per.values()) / n, 1) for k in names}

    # the prefill (verify): active (key head, 64-token block) pairs, for the boot from reset and
    # for the command from the snapshot
    layer_of = {id(g): L for L, (g, _) in enumerate(f.layers)}
    pre = {}
    for what, run in (("boot", lambda: lx.boot(None)),
                      ("cmd", lambda: (lx.boot(Path(a.snapshot)), lx.run(a.cmd))[1])):
        f.stats = []
        rec = run()
        by = {}
        for g, nkh, nb, p in f.stats:
            t = by.setdefault(layer_of[id(g)], [0, 0])
            t[0], t[1] = t[0] + nkh * nb, t[1] + p
        tp, ap_ = sum(t[0] for t in by.values()), sum(t[1] for t in by.values())
        pre[what] = {"tokens": rec["tokens"], "pairs_total": tp, "pairs_active": ap_,
                     "active_fraction": round(ap_ / max(tp, 1), 4),
                     "by_layer": {L: round(t[1] / max(t[0], 1), 4) for L, t in sorted(by.items())}}
    f.stats = None
    res = {"cmd": a.cmd, "decode_tokens": n, "decode_per_token_total": tot,
           "decode_per_token_by_layer": dec, "prefill": pre}
    print(json.dumps({k: v for k, v in res.items() if k != "decode_per_token_by_layer"}, indent=1))
    for L, st in dec.items():
        print(L, st)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
