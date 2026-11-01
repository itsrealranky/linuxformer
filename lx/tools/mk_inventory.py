"""What one decode step of the chip layers touches, for planning a single-kernel decode: per layer,
the packed weights' sizes and nonzeros (lx.fast's decode tables), and over a teacher-forced run
of N tokens, the active value heads (nonzero query, key or decay), the nonzeros of each live key
head's query and key, the decaying heads, the MLP units with nonzero activation. And which sums
could depend on their order: a sum is order-free when sum |terms| < 2^24 * (the finest bit any
term has), since then every partial sum is exact in fp32; the others are counted.

    python tools/mk_inventory.py [--tokens 300] [--cmd "echo hi"] [--out results/mk_inventory.json]
"""

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from lx.fast import _norm_cols
from lx.linux import ROOT, Linux
from lx.session import Session


def finest(t: torch.Tensor) -> torch.Tensor:
    """The value of each fp32 element's lowest set bit (inf for zeros)."""
    t = t.float()
    b = t.view(torch.int32)
    e = (b >> 23) & 255
    m = (b & 0x7FFFFF) | torch.where(e > 0, 0x800000, 0)
    tz = (m & -m).float().log2()
    g = torch.exp2(torch.where(e > 0, e - 127 - 23, -126 - 23).float() + tz)
    return torch.where(t == 0, torch.inf, g)


def order_free(terms: torch.Tensor) -> torch.Tensor:
    """terms (..., n) fp32: whether every summation order gives the same fp32 sum."""
    s = terms.double().abs().sum(-1)
    g = finest(terms).amin(-1).double()
    return (s == 0) | (s < 2.0 ** 24 * g)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "build" / "hf" / "model"))
    ap.add_argument("--snapshot", default=str(ROOT / "build" / "snapshots" / "boot-merged.pt"))
    ap.add_argument("--cmd", default="echo hi")
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--out", default="results/mk_inventory.json")
    a = ap.parse_args()
    lx = Linux(weights=Path(a.weights), fast=True)
    f, m = lx.fast, lx.model
    f.idle_skip = True
    f.dec = f.dec or [f._dec(L) for L in range(f.n_chip)]
    kd, vd, r = f.kd, f.vd, f.r

    static, tot_nnz, tot_dense = {}, 0, 0
    for L, d in enumerate(f.dec):
        st = {"gdn": d.gdn, "mlp": d.mlp}
        if d.gdn:
            n_z = d.w_in.shape[0] - d.n_p - d.n_b - d.n_a
            st.update(n_p=d.n_p, n_b=d.n_b, n_a=d.n_a, n_z=n_z, cu=len(d.cu),
                      w_in=list(d.w_in.shape), w_in_nnz=int((d.w_in != 0).sum()),
                      nkh=d.nkh, nhv=d.nkh * r, decays=d.decays,
                      hv=[d.hv.start, d.hv.stop] if isinstance(d.hv, slice) else "indices",
                      w_out=list(d.w_out.shape), w_out_nnz=int((d.w_out != 0).sum()),
                      conv_nnz=int((d.w_conv != 0).sum()))
            tot_nnz += st["w_in_nnz"] + st["w_out_nnz"]
            tot_dense += d.w_in.numel() + d.w_out.numel()
        if d.mlp:
            st.update(n_u=d.n_u, m_cin=len(d.m_cin), w_gu_nnz=int((d.w_gu != 0).sum()),
                      w_gu=list(d.w_gu.shape), w_d=list(d.w_d.shape),
                      w_d_nnz=int((d.w_d != 0).sum()))
            tot_nnz += st["w_gu_nnz"] + st["w_d_nnz"]
            tot_dense += d.w_gu.numel() + d.w_d.numel()
        static[L] = st
    head = {"w_head": list(f.w_head.shape), "nnz": int((f.w_head != 0).sum()),
            "row_nnz_max": int((f.w_head != 0).sum(1).max())}

    s = Session.load(lx.c, m, a.snapshot, lx.chunk, lx.offset, f)
    ref = copy.deepcopy(s.ref)
    toks = []
    for b in a.cmd.encode() + b"\n":
        toks += ref.key(b)
    while len(toks) < a.tokens:
        toks += ref.step()
    toks = toks[:a.tokens]

    keys = ("active_heads", "decaying_heads", "q_nnz_max", "k_nnz_max", "q_nnz_mean",
            "k_nnz_mean", "p_nonzero", "mlp_units_active", "norm_gdn_not_order_free",
            "norm_mlp_not_order_free", "proj_rows_not_order_free", "out_rows_not_order_free",
            "gu_rows_not_order_free", "down_rows_not_order_free", "qk_l2_not_order_free",
            "head_rows_not_order_free")
    dyn = {L: dict.fromkeys(keys, 0) for L in range(f.n_chip)}
    head_margin = []
    for t in toks:
        x = m.model.embed_tokens(torch.tensor([[t + lx.offset]], device=s.dev))[0].clone()
        for L, d in enumerate(f.dec):
            st = dyn[L]
            if d.gdn:
                cl = s.cache.layers[L]
                xf = x.float()
                st["norm_gdn_not_order_free"] += int(~order_free(xf * xf))
                xn = _norm_cols(x, d.eps, d.w1u, d.cu)
                prod = xn.float()[0][None, :] * d.w_in.float()
                st["proj_rows_not_order_free"] += int((~order_free(prod)).sum())
                qkv = f._gdn_pre(d, x, cl.conv_states[0])
                if qkv is not None:
                    q, k, v, beta, g, z = qkv
                    P = (xn @ d.w_in.T)[0]
                    st["p_nonzero"] += int((P != 0).sum())
                    qn, kn = (q[0, 0] != 0).sum(-1), (k[0, 0] != 0).sum(-1)
                    st["q_nnz_max"] = max(st["q_nnz_max"], int(qn.max()))
                    st["k_nnz_max"] = max(st["k_nnz_max"], int(kn.max()))
                    st["q_nnz_mean"] += float(qn.float().mean())
                    st["k_nnz_mean"] += float(kn.float().mean())
                    qf, kf = q[0, 0].float(), k[0, 0].float()
                    st["qk_l2_not_order_free"] += int((~order_free(qf * qf)).sum() +
                                                      (~order_free(kf * kf)).sum())
                    gz = torch.zeros(d.nkh * r, device=s.dev, dtype=torch.bool) if g is None \
                        else (g.view(-1) != 0)
                    live_kh = (qn > 0) | (kn > 0)
                    act = live_kh.repeat_interleave(r) | gz
                    st["active_heads"] += int(act.sum())
                    st["decaying_heads"] += int(gz.sum())
                    o = f._gdn_rec(d, q, k, v, beta, g, cl.recurrent_states[0])
                    norm = m.model.layers[L].linear_attn.norm
                    on = norm(o.view(z.shape), z).view(1, -1)
                    if len(d.out_rows):
                        prod = on.float()[0][None, :] * d.w_out.float()
                        st["out_rows_not_order_free"] += int((~order_free(prod)).sum())
                    f._gdn_post(norm, d, x, o, z)
            if d.mlp:
                xf = x.float()
                st["norm_mlp_not_order_free"] += int(~order_free(xf * xf))
                xn = _norm_cols(x, d.eps_m, d.w1m, d.m_cin)
                prod = xn.float()[0][None, :] * d.w_gu.float()
                st["gu_rows_not_order_free"] += int((~order_free(prod)).sum())
                gu = xn @ d.w_gu.T
                act = F.silu(gu[:, :d.n_u]) * gu[:, d.n_u:]
                st["mlp_units_active"] += int((act != 0).sum())
                prod = act.float()[0][None, :] * d.w_d.float()
                st["down_rows_not_order_free"] += int((~order_free(prod)).sum())
                f._mlp_step(d, x)
        hn = _norm_cols(x, m.model.norm.eps, (1.0 + m.model.norm.weight.float())[f.head_cols],
                        f.head_cols)
        logits = (hn @ f.w_head.T)[0].float()
        top = logits.topk(2)
        head_margin.append(float(top.values[0] - top.values[1]))
        if len(head_margin) <= 5:
            for blk in f.w_head.split(32768):
                prod = hn.float()[0][None, :] * blk.float()
                dyn[0]["head_rows_not_order_free"] += int((~order_free(prod)).sum())
    n = len(toks)
    for L, st in dyn.items():
        for k_ in ("active_heads", "decaying_heads", "q_nnz_mean", "k_nnz_mean", "p_nonzero",
                   "mlp_units_active"):
            st[k_] = round(st[k_] / n, 2)
    res = {"cmd": a.cmd, "tokens": n, "weights_nnz": tot_nnz, "weights_packed_elements": tot_dense,
           "head": head, "head_margin_min": min(head_margin),
           "per_token_active_heads": round(sum(st["active_heads"] for st in dyn.values()), 1),
           "per_token_decaying_heads": round(sum(st["decaying_heads"] for st in dyn.values()), 1),
           "not_order_free_total": {k_: sum(st[k_] for st in dyn.values())
                                    for k_ in keys if k_.endswith("not_order_free")},
           "static": static, "dynamic": dyn}
    print(json.dumps({k_: v for k_, v in res.items() if k_ not in ("static", "dynamic")},
                     indent=1))
    for L in range(f.n_chip):
        print(L, static[L], dyn[L])
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
