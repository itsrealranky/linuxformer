"""The megakernel's exact arithmetic (csrc/mk_math.cuh) against torch's and fla's own kernels,
bit for bit, on random inputs: the RMSNorm mean of squares (rows of 2560) and its rsqrt, the
gated norm (rows of 128, any head count), silu, sigmoid, softplus, exp, and fla's recurrent step
(l2 norms, decay, the delta rule, the output) from sparse and dense queries and keys.

    python tests/check_mk_math.py
"""

import ctypes
import json

import torch
import torch.nn.functional as F

from lx import mk_build

P = ctypes.c_void_p


def ptr(t):
    return P(t.data_ptr())


def stream():
    return P(torch.cuda.current_stream().cuda_stream)


def bits(t):
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


def ndiff(a, b):
    return int((bits(a) != bits(b)).sum())


def ndiff0(a, b):
    """Bits that differ, -0 counted as +0: a sum of exact zeros carries the sign its order
    gives it, and no later op of the decode can tell the two zeros apart (they only enter
    products and sums whose all-zero result is +0)."""
    return ndiff(a.float() + 0.0, b.float() + 0.0)


def rand_bf16(shape, scale=1.0, nz=1.0, dev="cuda"):
    x = (torch.randn(shape, device=dev) * scale).to(torch.bfloat16)
    if nz < 1.0:
        x = torch.where(torch.rand(shape, device=dev) < nz, x, torch.zeros_like(x))
    return x


@torch.no_grad()
def main():
    torch.manual_seed(0)
    lib = mk_build.load("mk_test")
    dev = "cuda"
    res = {}

    # RMSNorm over 2560: x.float().pow(2).mean(-1) + eps, rsqrt; one row per torch call (the
    # decode's shape: one output)
    rows = 3000
    xs = []
    for i in range(rows):
        kind = i % 3
        if kind == 0:
            x = rand_bf16(2560, 10.0 ** float(torch.empty(1).uniform_(-3, 3)))
        elif kind == 1:  # sparse, wide range (as the residual stream)
            x = rand_bf16(2560, 30.0, nz=0.02) * (2.0 ** torch.randint(-8, 12, (2560,), device=dev)).to(torch.bfloat16)
        else:
            x = rand_bf16(2560, 1.0) * (2.0 ** torch.randint(-20, 20, (2560,), device=dev)).to(torch.bfloat16)
        xs.append(x)
    X = torch.stack(xs).float().contiguous()
    eps = 1e-6
    mean = torch.empty(rows, device=dev)
    r = torch.empty(rows, device=dev)
    lib.t_norm2560(ptr(X), ptr(mean), ptr(r), ctypes.c_float(eps), rows, stream())
    want_m = torch.cat([X[i:i + 1].pow(2).mean(-1) for i in range(rows)])
    want_r = torch.rsqrt(want_m + eps)
    res["norm2560_mean_diff"] = ndiff(mean, want_m)
    res["norm2560_rsqrt_diff"] = ndiff(r, want_r)

    # the gated norm, rows of 128, as the module computes it, for several head counts
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNormGated

    gd = 0
    for nhv in (1, 8, 16, 24, 32, 48, 64, 80, 128):
        for trial in range(8):
            norm = Qwen3_5RMSNormGated(128, eps=1e-6).to(dev, torch.bfloat16)
            norm.weight.copy_(rand_bf16(128, 1.0) + 1.0)
            o = rand_bf16((nhv, 128), 10.0 ** (trial - 4), nz=0.3 + 0.1 * trial)
            z = rand_bf16((nhv, 128), 3.0)
            want = norm(o, z)
            out = torch.empty_like(o)
            lib.t_gated(ptr(o), ptr(z), ptr(norm.weight), ptr(out), ctypes.c_float(1e-6), nhv,
                        stream())
            gd += ndiff(out, want)
    res["gated_norm_diff"] = gd

    # elementwise: torch's fp32 ops
    n = 1 << 22
    x = torch.cat([torch.randn(n // 4, device=dev) * 30, torch.randn(n // 4, device=dev) * 3,
                   torch.randn(n // 4, device=dev) * 0.01,
                   (torch.rand(n // 4, device=dev) * 200 - 100)])
    xp = x.abs() + 1e-7
    for which, name, fn, inp in ((0, "silu", F.silu, x), (1, "sigmoid", torch.sigmoid, x),
                                  (2, "softplus", F.softplus, x), (3, "rsqrt", torch.rsqrt, xp),
                                  (4, "expf", torch.exp, x.clamp(-80, 80))):
        out = torch.empty_like(inp)
        lib.t_elem(ptr(inp), ptr(out), n, which, stream())
        res[f"{name}_diff"] = ndiff(out, fn(inp))
    # bf16-input silu and sigmoid (as the model runs them)
    xb = x.to(torch.bfloat16)
    out = torch.empty(n, device=dev)
    lib.t_elem(ptr(xb.float().contiguous()), ptr(out), n, 0, stream())
    res["silu_bf16_diff"] = ndiff(out.to(torch.bfloat16), F.silu(xb))
    lib.t_elem(ptr(xb.float().contiguous()), ptr(out), n, 1, stream())
    res["sigmoid_bf16_diff"] = ndiff(out.to(torch.bfloat16), torch.sigmoid(xb))

    # fla's step: one value head per (fla) head, H = HV
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

    # variants: 0 the emulation, 3 the other in-lane pairing (the same for bf16 inputs: their
    # squares are exact), 1 and 2 wrong orders on purpose (l2 sums, dot products): they must fail
    names = {0: "rec", 3: "rec_fma_even", 1: "rec_wrong_l2_order", 2: "rec_wrong_dot_order"}
    for variant in (0, 3, 1, 2):
        torch.manual_seed(1)
        tot = {"o": 0, "h": 0, "cases": 0, "dense_cases_wrong": 0}
        for case in range(48):
            nh = 64
            dense = case % 4 == 3
            nz = 1.0 if dense else 2.0 / 256
            q = rand_bf16((nh, 256), 2.0, nz=nz)
            k = rand_bf16((nh, 256), 2.0, nz=nz if dense else 1.0 / 256)
            if case % 6 == 1:  # a key of one nonzero, the RAM's write
                k = torch.zeros_like(k)
                k[torch.arange(nh), torch.randint(0, 256, (nh,))] = rand_bf16(nh, 2.0)
            # silu(-0) is -0: queries and keys can hold either zero
            for x in (q, k):
                x[(x == 0) & (torch.rand(x.shape, device=dev) < 0.5)] = -0.0
            v = rand_bf16((nh, 128), 3.0)
            beta = torch.rand(nh, device=dev).to(torch.bfloat16)
            g = None if case % 2 == 0 else -torch.rand(nh, device=dev) * 4
            h0 = torch.randn(nh, 256, 128, device=dev) * (10.0 ** (case % 5 - 2))
            if case % 3 == 2:  # a sparse state (a few rows)
                h0 = h0 * (torch.rand(nh, 256, 1, device=dev) < 0.03)
            h0 = h0 + 0.0  # the state holds no -0 (it starts +0; fla's load adds +0)
            o_w, h_w = fused_recurrent_gated_delta_rule(
                q[None, None], k[None, None], v[None, None], g=None if g is None else g[None, None],
                beta=beta[None, None], initial_state=h0[None], output_final_state=True,
                use_qk_l2norm_in_kernel=True)
            h = h0.clone()
            o = torch.empty(nh, 128, dtype=torch.bfloat16, device=dev)
            lib.t_rec(ptr(q), ptr(k), ptr(v), ptr(beta), None if g is None else ptr(g), ptr(h),
                      ptr(o), ctypes.c_float(256 ** -0.5), nh, variant, stream())
            do, dh = ndiff0(o, o_w[0, 0]), ndiff(h, h_w[0])
            tot["o"] += do
            tot["h"] += dh
            tot["cases"] += 1
            tot["dense_cases_wrong"] += int(dense and (do + dh) > 0)
        res[names[variant]] = tot
    torch.cuda.synchronize()
    assert lib.t_last_error() == 0
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
