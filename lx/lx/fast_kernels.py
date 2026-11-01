"""fla's fused recurrent gated-delta-rule step (fla.ops.gated_delta_rule.fused_recurrent, one
token, head-wise beta, the state K x V) that leaves out value heads with nothing to do: if the
key head's query and key are all zero and the decay is 0, the step changes neither the state
nor the output (k = 0 adds nothing to it, exp(0) = 1 scales by one, q = 0 reads 0), so such a
program writes its zero output and never loads or stores its state. Active heads run fla's
code, unchanged, at fla's block sizes."""

import triton
import triton.language as tl
from fla.ops.utils.op import exp


@triton.heuristics({"USE_G": lambda args: args["g"] is not None})
@triton.jit
def recurrent_step(q, k, v, g, beta, o, h, scale, H: tl.constexpr, HV: tl.constexpr,
                   K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                   USE_G: tl.constexpr):
    i_v, i_hv = tl.program_id(0), tl.program_id(1)
    i_h = i_hv // (HV // H)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    p_o = o + i_hv * V + o_v

    b_q = tl.load(q + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(k + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
    live = (tl.sum((b_q != 0).to(tl.int32)) + tl.sum((b_k != 0).to(tl.int32))) > 0
    if USE_G:
        b_g = tl.load(g + i_hv).to(tl.float32)
        live = live | (b_g != 0)
    if live:
        p_h = h + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
        b_h = tl.zeros([BK, BV], dtype=tl.float32)
        b_h += tl.load(p_h, mask=mask_h, other=0).to(tl.float32)
        b_v = tl.load(v + i_hv * V + o_v, mask=mask_v, other=0).to(tl.float32)
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        b_beta = tl.load(beta + i_hv).to(tl.float32)
        if USE_G:
            b_h *= exp(b_g)
        b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
        b_h += b_k[:, None] * b_v
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mask_h)
    else:
        tl.store(p_o, tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty), mask=mask_v)
