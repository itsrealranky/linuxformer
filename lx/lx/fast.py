"""The chip layers' forward without the zeros: the same weights, the same cache tensors and the
same kernels as the stock forward (hf.chip_forward), on only the parts that are not zero.

    f = Fast(model, n_chip, rows)
    h = f.forward(h, cache)          # as hf.chip_forward(model, h, cache, n_chip), T > 1
    pred = f.head_argmax(h)          # as hf.head_argmax(model, h, rows=rows)

What is skipped, and why that changes nothing:

- weight rows and columns that are all zero: a projection's dead outputs are exactly 0, and its
  dead inputs contribute exactly 0;
- GDN work on a key head over a 64-token block where its queries, keys and decays are all zero:
  with g = 0 (exp(g) = 1) and k = 0 the head's state stays as it is, and with q = 0 its output
  there is exactly 0. Each key head's active blocks run as one sequence of fla's
  chunk_gated_delta_rule (varlen), from the head's state, in fla's own 64-token blocks.

Sums over fewer terms are the same numbers because the circuit's values are exact in fp32;
tests/check_fast.py checks every layer's output and state against the stock forward bit for
bit, and lx.linux --fast verifies the whole boot.
"""

from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F

BT = 64  # fla's block (chunk) size
# the chip layers' projections, which lx.fast reads once (their nonzero parts) and then never
PROJ = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj", "gate_proj",
        "up_proj", "down_proj")


def place_lean(model, n_chip: int, dev) -> None:
    """The model on dev except the chip layers' projections (half its weights, 0.08% nonzero),
    which stay where they are: build Fast from the model first, then Fast.to(dev)."""
    held = []
    for layer in model.model.layers[:n_chip]:
        for m in layer.modules():
            for name in PROJ:
                lin = getattr(m, name, None)
                if isinstance(lin, torch.nn.Linear):
                    held.append((lin, lin.weight))
                    lin.weight = None
    model.to(dev)
    for lin, w in held:
        lin.weight = w


def _nz_rows(w):
    return (w != 0).any(1).nonzero()[:, 0]


def _nz_cols(w):
    return (w != 0).any(0).nonzero()[:, 0]


class _Norm:
    """Qwen3_5RMSNorm(h)[:, cols] for any cols, computed as the module computes it."""

    def __init__(self, norm, h):
        self.x = h.float()
        self.r = torch.rsqrt(self.x.pow(2).mean(-1, keepdim=True) + norm.eps)
        self.w1 = 1.0 + norm.weight.float()
        self.dtype = h.dtype

    def __call__(self, cols):
        return ((self.x[:, cols] * self.r) * self.w1[cols]).to(self.dtype)


@dataclass
class _GDN:
    kh: torch.Tensor  # live key heads (a live query or key)
    pc: torch.Tensor  # projected channels with any weight (their conv state is kept)
    cin: torch.Tensor  # in_proj_qkv's input columns
    w_p: torch.Tensor  # in_proj_qkv[pc][:, cin]
    w_conv: torch.Tensor  # conv1d.weight[pc] (n, 1, 4)
    q_src: torch.Tensor  # projected columns that are queries of live key heads ...
    q_dst: torch.Tensor  # ... and their place in (n_kh * KD)
    k_src: torch.Tensor
    k_dst: torch.Tensor
    v_src: torch.Tensor  # values of the live key heads' value heads, placed in (n_kh * R * VD)
    v_dst: torch.Tensor
    qkv_idx: torch.Tensor  # the same as one gather: q, k, v from (conv output, 0)
    cin_b: torch.Tensor
    w_b: torch.Tensor  # in_proj_b, the live key heads' value heads
    decays: bool  # any of those value heads' g can be nonzero
    cin_a: torch.Tensor
    w_a: torch.Tensor  # in_proj_a, as w_b
    a_neg: torch.Tensor  # -exp(A_log), fp32, those value heads
    dt_bias: torch.Tensor
    cin_z: torch.Tensor
    out: list  # per live key head: (in_proj_z rows, out_proj cols [fp32]) or None (no output)
    out_rows: torch.Tensor  # residual dims out_proj writes


@dataclass
class _Dec:
    """One chip layer's single-token step (Fast.decode), packed."""
    gdn: bool
    mlp: bool
    eps: float = 0.0
    cu: torch.Tensor = None  # input_layernorm columns the GDN's projections read
    w1u: torch.Tensor = None  # (1 + input_layernorm.weight)[cu], fp32
    w_in: torch.Tensor = None  # in_proj_qkv[pc], in_proj_b, in_proj_a, in_proj_z (live), over cu
    n_p: int = 0
    n_b: int = 0
    n_a: int = 0  # 0 unless the layer decays
    pc: torch.Tensor = None
    w_conv: torch.Tensor = None
    qkv_idx: torch.Tensor = None  # the live heads' q, k, v, as indices into (conv output, 0)
    nkh: int = 0
    hv: object = None  # the live value heads in the recurrent state: a slice, or indices
    decays: bool = False
    a_neg: torch.Tensor = None
    dt_bias: torch.Tensor = None
    w_out: torch.Tensor = None  # out_proj[out_rows][:, live value dims]
    out_rows: torch.Tensor = None
    zero: torch.Tensor = None
    eps_m: float = 0.0
    m_cin: torch.Tensor = None
    w1m: torch.Tensor = None
    w_gu: torch.Tensor = None  # gate_proj and up_proj, stacked
    n_u: int = 0
    w_d: torch.Tensor = None
    m_rows: torch.Tensor = None


def _conv_qkv(p, past, w, idx, nkh: int, kd: int):
    """The prefill's causal conv and silu as the stock forward computes them (F.conv1d over the
    cached window, then the new inputs: per output +0, then weight * input for taps 0..3 added
    in fp32, rounded to bf16), in (token, channel) layout; then the live heads' q, k, v gathered
    by idx (index n: zero). p (T, n), past (4, n), oldest first; w (n, 4). Returns (qkv, the new
    window, whether each token's query or key is nonzero, by key head (T, nkh))."""
    T = p.shape[0]
    full = torch.cat([past, p])
    wf = w.float()
    acc = full[1:T + 1].float() * wf[:, 0]
    acc = torch.where(acc == 0, 0.0, acc)  # +0 + (-0) is +0
    for j in range(1, 4):
        acc = acc + full[1 + j:T + 1 + j].float() * wf[:, j]
    y = F.silu(acc.to(p.dtype))
    qkv = torch.cat([y, y.new_zeros(T, 1)], 1)[:, idx]
    act = (qkv[:, :2 * nkh * kd].view(T, 2, nkh, kd) != 0).any(3).any(1)
    return qkv, full[-past.shape[0]:], act


def _compile(f, **kw):
    import torch._dynamo.config
    import torch._inductor.config
    # round to bf16 wherever the eager ops round (else inductor keeps fp32 between)
    torch._inductor.config.emulate_precision_casts = True
    torch._dynamo.config.recompile_limit = 256
    return torch.compile(f, **kw)


def _norm_cols(h, eps, w1, cols):
    """Qwen3_5RMSNorm(h)[:, cols] as the module computes it, given w1 = (1 + weight)[cols]."""
    x = h.float()
    r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return ((x[:, cols] * r) * w1).to(h.dtype)


@dataclass
class _MLP:
    cin: torch.Tensor
    w_g: torch.Tensor
    w_u: torch.Tensor
    w_d: torch.Tensor  # down_proj[rows][:, units]
    rows: torch.Tensor


class Fast:
    def __init__(self, model, n_chip: int, rows: slice | None = None) -> None:
        cfg = model.config
        self.model, self.n_chip = model, n_chip
        self.nk, self.nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        self.kd, self.vd = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        self.r = self.nv // self.nk
        self.layers = []
        for L in range(n_chip):
            layer = model.model.layers[L]
            self.layers.append((self._gdn(layer.linear_attn), self._mlp(layer.mlp)))
        w = model.lm_head.weight if rows is None else model.lm_head.weight[rows]
        self.head_cols = _nz_cols(w)
        self.w_head = w[:, self.head_cols].contiguous()

    def _gdn(self, ga) -> _GDN:
        kd, vd, nk, r = self.kd, self.vd, self.nk, self.r
        w, conv = ga.in_proj_qkv.weight, ga.conv1d.weight
        dev = w.device
        pc = _nz_rows(w)
        live = torch.zeros(w.shape[0], dtype=torch.bool, device=dev)
        live[pc] = True
        live &= (conv[:, 0] != 0).any(1)
        nq = nk * kd
        kh = (live[:nq].view(nk, kd).any(1) | live[nq:2 * nq].view(nk, kd).any(1)).nonzero()[:, 0]
        pos = torch.full((nk,), -1, dtype=torch.long, device=dev)
        pos[kh] = torch.arange(len(kh), device=dev)

        def take(lo, hi, dst_of):
            src = (live[pc] & (pc >= lo) & (pc < hi)).nonzero()[:, 0]
            dst = dst_of(pc[src] - lo)
            return src[dst >= 0], dst[dst >= 0]

        def qk(c):
            return torch.where(pos[c // kd] >= 0, pos[c // kd] * kd + c % kd, -1)

        def vv(c):
            p = pos[c // vd // r]
            return torch.where(p >= 0, (p * r + c // vd % r) * vd + c % vd, -1)

        q_src, q_dst = take(0, nq, qk)
        k_src, k_dst = take(nq, 2 * nq, qk)
        v_src, v_dst = take(2 * nq, 2 * nq + self.nv * vd, vv)
        vh = (kh[:, None] * r + torch.arange(r, device=dev)).reshape(-1)
        wb, wa = ga.in_proj_b.weight[vh], ga.in_proj_a.weight[vh]
        cin_b, cin_a = _nz_cols(wb), _nz_cols(wa)
        a_neg = -ga.A_log.float()[vh].exp()
        g0 = a_neg * F.softplus(ga.dt_bias[vh].float())  # g where a = 0
        decays = bool(len(cin_a)) or bool((g0 != 0).any())
        wo, wz = ga.out_proj.weight, ga.in_proj_z.weight
        out_rows = _nz_rows(wo)
        dims = (vh[:, None] * vd + torch.arange(vd, device=dev)).reshape(-1)
        cin_z = _nz_cols(wz[dims])
        out = []
        for i in range(len(kh)):
            d = dims[i * r * vd:(i + 1) * r * vd]
            wo_k = wo[out_rows][:, d]
            out.append(None if not (wo_k != 0).any() else
                       (wz[d][:, cin_z].contiguous(), wo_k.float().contiguous()))
        cin = _nz_cols(w[pc])
        nq = len(kh) * kd
        idx = torch.full((2 * nq + len(kh) * r * vd,), len(pc), dtype=torch.long, device=dev)
        idx[q_dst], idx[nq + k_dst], idx[2 * nq + v_dst] = q_src, k_src, v_src
        return _GDN(kh=kh, pc=pc, cin=cin, w_p=w[pc][:, cin].contiguous(),
                    w_conv=conv[pc].contiguous(), q_src=q_src, q_dst=q_dst, k_src=k_src,
                    k_dst=k_dst, v_src=v_src, v_dst=v_dst, qkv_idx=idx, cin_b=cin_b,
                    w_b=wb[:, cin_b].contiguous(), decays=decays, cin_a=cin_a,
                    w_a=wa[:, cin_a].contiguous(), a_neg=a_neg, dt_bias=ga.dt_bias[vh],
                    cin_z=cin_z, out=out, out_rows=out_rows)

    def _mlp(self, mlp) -> _MLP:
        g, u, d = mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight
        units = ((g != 0).any(1) & (u != 0).any(1) & (d != 0).any(0)).nonzero()[:, 0]
        cin = _nz_cols(torch.cat([g[units], u[units]]))
        rows = _nz_rows(d[:, units])
        return _MLP(cin=cin, w_g=g[units][:, cin].contiguous(), w_u=u[units][:, cin].contiguous(),
                    w_d=d[rows][:, units].contiguous(), rows=rows)

    # ------------------------------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, h: torch.Tensor, cache) -> torch.Tensor:
        """h (1, T, hidden) -> (1, T, hidden), updating the cache's conv and recurrent states."""
        x = h[0].clone()
        for L in range(self.n_chip):
            self.layer(L, x, cache)
        return x[None]

    def layer(self, L: int, h: torch.Tensor, cache) -> None:
        """One chip layer on h (T, hidden), in place."""
        layer = self.model.model.layers[L]
        g, m = self.layers[L]
        if len(g.pc):
            self._gdn_forward(layer.linear_attn, g, _Norm(layer.input_layernorm, h), h,
                              cache.layers[L])
        if len(m.rows):
            xm = _Norm(layer.post_attention_layernorm, h)(m.cin)
            act = F.silu(xm @ m.w_g.T) * (xm @ m.w_u.T)
            h[:, m.rows] = h[:, m.rows] + act @ m.w_d.T

    def _gdn_forward(self, ga, g: _GDN, xn: _Norm, h: torch.Tensor, cl) -> None:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        T, dev = h.shape[0], h.device
        kd, vd, r, nkh = self.kd, self.vd, self.r, len(g.kh)
        # in_proj_qkv, then the conv and silu as causal_conv1d_fn does them, after the cached
        # left context (the conv state: each channel's latest inputs)
        cs = cl.conv_states[0]
        nb = -(-T // BT)

        def nonzero_blocks(t, d):
            t = F.pad(t.view(T, nkh, d) != 0, (0, 0, 0, 0, 0, nb * BT - T))
            return t.view(nb, BT, nkh, d).any(3).any(1)

        if self.compile and nkh:
            if self._conv_c is None:
                self._conv_c = _compile(_conv_qkv, dynamic=True)
            qkv, win, act_tok = self._conv_c(xn(g.cin) @ g.w_p.T, cs[0, g.pc].T,
                                             g.w_conv[:, 0], g.qkv_idx, nkh, kd)
            cs[0, g.pc] = win.T
            q, k, v = qkv[:, :nkh * kd], qkv[:, nkh * kd:2 * nkh * kd], qkv[:, 2 * nkh * kd:]
            act = F.pad(act_tok, (0, 0, 0, nb * BT - T)).view(nb, BT, nkh).any(1)
        else:
            p = (xn(g.cin) @ g.w_p.T).T[None]  # (1, n, T)
            full = torch.cat([cs[:, g.pc], p], dim=-1)
            cs[:, g.pc] = full[..., -cs.shape[-1]:]
            y = F.conv1d(full.to(g.w_conv.dtype), g.w_conv, None,
                         padding=g.w_conv.shape[-1] - 1, groups=len(g.pc))[:, :, :full.shape[-1]]
            y = F.silu(y[:, :, -T:])[0].T  # (T, n)
            if nkh == 0:
                return
            q = torch.zeros(T, nkh * kd, dtype=h.dtype, device=dev)
            k = torch.zeros(T, nkh * kd, dtype=h.dtype, device=dev)
            v = torch.zeros(T, nkh * r * vd, dtype=h.dtype, device=dev)
            q[:, g.q_dst] = y[:, g.q_src]
            k[:, g.k_dst] = y[:, g.k_src]
            v[:, g.v_dst] = y[:, g.v_src]
            act = nonzero_blocks(q, kd) | nonzero_blocks(k, kd)
        # the decay, as the stock forward computes it: -exp(A_log) * softplus(a + dt_bias)
        if g.decays:
            a = (xn(g.cin_a) @ g.w_a.T) if len(g.cin_a) else \
                torch.zeros(T, nkh * r, dtype=h.dtype, device=dev)
            gd = (g.a_neg * F.softplus(a.float() + g.dt_bias)).view(T, nkh, r)
        # the (key head, block) pairs with any nonzero query, key or decay
        if g.decays:
            act |= nonzero_blocks(gd, r)
        pairs = act.T.nonzero()  # by key head, then block
        if self.stats is not None:  # (layer's _GDN, live key heads, blocks, active pairs)
            self.stats.append((g, nkh, nb, len(pairs)))
        if len(pairs) == 0:
            return
        s_of, b_of = pairs[:, 0], pairs[:, 1]
        lens = torch.where(b_of == nb - 1, T - b_of * BT, BT)
        n_tok = int(lens.sum())
        starts = torch.cumsum(lens, 0) - lens
        tok = (b_of * BT - starts).repeat_interleave(lens, output_size=n_tok) + \
            torch.arange(n_tok, device=dev)
        s_tok = s_of.repeat_interleave(lens, output_size=n_tok)
        per = torch.zeros(nkh, dtype=torch.long, device=dev).index_add_(0, s_of, lens)
        seqs = (per > 0).nonzero()[:, 0]
        cu = F.pad(torch.cumsum(per[seqs], 0), (1, 0))
        # one sequence per key head, its R value heads as fla's heads (query and key repeated,
        # as the stock forward's repeat_interleave does)
        qs = q.unflatten(1, (nkh, kd))[tok, s_tok][:, None].expand(n_tok, r, kd).contiguous()
        ks = k.unflatten(1, (nkh, kd))[tok, s_tok][:, None].expand(n_tok, r, kd).contiguous()
        vs = v.unflatten(1, (nkh, r, vd))[tok, s_tok]
        beta = (xn(g.cin_b) @ g.w_b.T).sigmoid().view(T, nkh, r)[tok, s_tok]
        gs = gd[tok, s_tok] if g.decays else \
            torch.zeros(n_tok, r, dtype=torch.float32, device=dev)
        rs = cl.recurrent_states[0]
        vh = (g.kh[seqs][:, None] * r + torch.arange(r, device=dev)).reshape(-1)
        o, fin = chunk_gated_delta_rule(
            qs[None], ks[None], vs[None], g=gs[None], beta=beta[None],
            initial_state=rs[0, vh].view(len(seqs), r, kd, vd), output_final_state=True,
            use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
        rs[0, vh] = fin.reshape(-1, kd, vd).to(rs.dtype)
        # the gated norm and out_proj on the active rows, summed in fp32 and rounded once (as
        # one bf16 matmul over all value heads rounds once)
        o = o[0]
        out32 = torch.zeros(T, len(g.out_rows), dtype=torch.float32, device=dev)
        xz = xn(g.cin_z)
        cu_l = cu.tolist()
        for j, s in enumerate(seqs.tolist()):
            if g.out[s] is None:
                continue
            wz, wo = g.out[s]
            a, b = cu_l[j], cu_l[j + 1]
            t = tok[a:b]
            on = ga.norm(o[a:b].reshape(-1, vd), (xz[t] @ wz.T).view(-1, vd))
            out32.index_add_(0, t, on.view(b - a, r * vd).float() @ wo.T)
        h[:, g.out_rows] = h[:, g.out_rows] + out32.to(h.dtype)

    # ------------------------------------------------------------------------------------------
    @torch.no_grad()
    def decode(self, h: torch.Tensor, cache) -> torch.Tensor:
        """One token: h (1, hidden) -> (1, hidden), in fixed shapes (no host sync, so it can be a
        CUDA graph). Every live key head runs: at a single token all of them are active. Each
        norm is taken once, on the columns its projections read, and those projections are one
        matmul; fla's recurrent kernel takes the live heads' queries and keys unrepeated (its
        grouped value heads: value head i reads key head i // R, as repeat_interleave gives)."""
        if self.dec is None:
            self.dec = [self._dec(L) for L in range(self.n_chip)]
        if self._steps is None:
            self._steps = (self._gdn_pre, self._gdn_post, self._mlp_step)
            if self.compile:
                self._steps = tuple(_compile(f, dynamic=False) for f in self._steps)
        pre, post, mlp = self._steps
        x = h.clone()
        for L, d in enumerate(self.dec):
            if d.gdn:
                cl = cache.layers[L]
                qkv = pre(d, x, cl.conv_states[0])
                if qkv is not None:
                    o = self._gdn_rec(d, *qkv[:5], cl.recurrent_states[0])
                    post(self.model.model.layers[L].linear_attn.norm, d, x, o, qkv[5])
            if d.mlp:
                mlp(d, x)
        return x

    dec = None
    compile = False  # torch.compile the steps between fla's kernels (checked bit for bit)
    lean = False  # the chip projections are not on the device (place_lean): no stock forward
    head_block = 8192  # tokens per head matmul: 8192 x 459,982 bf16 logits is 7 GiB
    stats = None  # a list: the prefill's GDN work per layer call is appended to it
    idle_skip = False  # decode: skip value heads whose query, key and decay are all zero
    _conv_c = _steps = None

    @torch.no_grad()
    def to(self, dev) -> "Fast":
        """Every tensor Fast built (from weights in host memory, say) moved to dev."""
        if self.dec is None:
            self.dec = [self._dec(L) for L in range(self.n_chip)]

        def mv(o):
            if isinstance(o, torch.Tensor):
                return o.to(dev)
            return type(o)(mv(v) for v in o) if isinstance(o, (list, tuple)) else o

        for obj in [*(x for pair in self.layers for x in pair), *self.dec]:
            for f in fields(obj):
                setattr(obj, f.name, mv(getattr(obj, f.name)))
        self.head_cols, self.w_head = self.head_cols.to(dev), self.w_head.to(dev)
        # exp on dev, as the stock forward takes it there (another device's exp may round apart)
        for L, (g, _) in enumerate(self.layers):
            vh = (g.kh[:, None] * self.r + torch.arange(self.r, device=dev)).reshape(-1)
            g.a_neg = -self.model.model.layers[L].linear_attn.A_log.to(dev).float()[vh].exp()
            self.dec[L].a_neg = g.a_neg
        return self

    def init_cache(self, cache, dev) -> None:
        """The chip layers' states as the stock forward of one zero token leaves them, which
        Session starts from: all zero (a zero hidden state gives zero inputs everywhere), conv
        windows bf16, recurrent states fp32."""
        conv = 2 * self.nk * self.kd + self.nv * self.vd
        k = self.model.config.linear_conv_kernel_dim
        for L in range(self.n_chip):
            cache.update_conv_state(torch.zeros(1, conv, 1, dtype=torch.bfloat16, device=dev), L,
                                    conv_kernel_size=k)
            cache.layers[L].update_recurrent_state(
                torch.zeros(1, self.nv, self.kd, self.vd, dtype=torch.float32, device=dev))

    def _dec(self, L: int) -> _Dec:
        layer = self.model.model.layers[L]
        ga, (g, m) = layer.linear_attn, self.layers[L]
        vd, r, nkh = self.vd, self.r, len(g.kh)
        d = _Dec(gdn=bool(len(g.pc)), mlp=bool(len(m.rows)))
        if d.gdn:
            dev = g.pc.device
            norm = layer.input_layernorm
            vh = (g.kh[:, None] * r + torch.arange(r, device=dev)).reshape(-1)
            dims = (vh[:, None] * vd + torch.arange(vd, device=dev)).reshape(-1)
            cu = torch.unique(torch.cat([g.cin, g.cin_b, g.cin_a, g.cin_z]))

            def over_cu(w, cin):
                out = torch.zeros(w.shape[0], len(cu), dtype=w.dtype, device=dev)
                out[:, torch.searchsorted(cu, cin)] = w
                return out

            parts = [over_cu(g.w_p, g.cin), over_cu(g.w_b, g.cin_b)]
            if g.decays:
                parts.append(over_cu(g.w_a, g.cin_a))
            parts.append(over_cu(ga.in_proj_z.weight[dims][:, g.cin_z], g.cin_z))
            n_p = len(g.pc)
            contig = nkh and bool((vh == torch.arange(int(vh[0]), int(vh[0]) + len(vh),
                                                      device=dev)).all())
            d.eps, d.cu, d.w1u = norm.eps, cu, (1.0 + norm.weight.float())[cu]
            d.w_in, d.n_p, d.n_b = torch.cat(parts).contiguous(), n_p, nkh * r
            d.n_a = nkh * r if g.decays else 0
            d.pc, d.w_conv, d.qkv_idx, d.nkh = g.pc, g.w_conv, g.qkv_idx, nkh
            d.hv = slice(int(vh[0]), int(vh[0]) + len(vh)) if contig else vh
            d.decays, d.a_neg, d.dt_bias = g.decays, g.a_neg, g.dt_bias
            d.w_out = ga.out_proj.weight[g.out_rows][:, dims].contiguous()
            d.out_rows, d.zero = g.out_rows, torch.zeros(1, dtype=g.w_p.dtype, device=dev)
        if d.mlp:
            norm = layer.post_attention_layernorm
            d.eps_m, d.m_cin, d.w1m = norm.eps, m.cin, (1.0 + norm.weight.float())[m.cin]
            d.w_gu, d.n_u = torch.cat([m.w_g, m.w_u]).contiguous(), m.w_g.shape[0]
            d.w_d, d.m_rows = m.w_d, m.rows
        return d

    def _gdn_pre(self, d: _Dec, x: torch.Tensor, cs: torch.Tensor):
        """Up to fla: the projections, the conv (updating the conv state cs), then q, k, v,
        beta, g and in_proj_z's output, or None if no key head is live."""
        kd, vd, nkh, nhv = self.kd, self.vd, d.nkh, d.nkh * self.r
        n_p, n_b, n_a = d.n_p, d.n_b, d.n_a
        P = (_norm_cols(x, d.eps, d.w1u, d.cu) @ d.w_in.T)[0]
        # in_proj_qkv, then the conv and silu as causal_conv1d_update does them
        full = torch.cat([cs[:, d.pc], P[None, :n_p, None]], dim=-1)
        cs[:, d.pc] = full[..., -cs.shape[-1]:]
        y = F.silu(F.conv1d(full, d.w_conv, None, padding=0, groups=n_p)[0, :, -1])
        if not nkh:
            return None
        qkv = torch.cat([y, d.zero])[d.qkv_idx]
        q = qkv[:nkh * kd].view(1, 1, nkh, kd)
        k = qkv[nkh * kd:2 * nkh * kd].view(1, 1, nkh, kd)
        v = qkv[2 * nkh * kd:].view(1, 1, nhv, vd)
        beta = P[n_p:n_p + n_b].sigmoid().view(1, 1, nhv)
        # g = 0 multiplies the state by exp(0) = 1: leaving it out is the same
        g = (d.a_neg * F.softplus(P[n_p + n_b:n_p + n_b + n_a].float() + d.dt_bias)).view(
            1, 1, nhv) if d.decays else None
        return q, k, v, beta, g, P[n_p + n_b + n_a:].view(nhv, vd)

    def _gdn_rec(self, d: _Dec, q, k, v, beta, g, rs: torch.Tensor) -> torch.Tensor:
        """fla's fused recurrent step, as fused_recurrent_gated_delta_rule launches it, but on
        the live heads' part of the recurrent state in place: each program loads its own tile of
        the state, then stores it back (no two programs share one). With idle_skip, value heads
        whose query, key and decay are all zero leave their state untouched (lx.fast_kernels)."""
        import triton
        from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
        from fla.ops.gated_delta_rule.fused_recurrent import \
            fused_recurrent_gated_delta_rule_fwd_kernel as kernel

        if not isinstance(d.hv, slice):
            o, ht = fused_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta,
                                                     initial_state=rs[:, d.hv],
                                                     output_final_state=True,
                                                     use_qk_l2norm_in_kernel=True)
            rs[:, d.hv] = ht
            return o
        h = rs[:, d.hv]
        assert h.is_contiguous() and q.is_contiguous() and k.is_contiguous() and \
            v.is_contiguous() and beta.is_contiguous() and (g is None or g.is_contiguous())
        H, K, HV, V = q.shape[2], q.shape[3], v.shape[2], v.shape[3]
        BK, BV = triton.next_power_of_2(K), min(8, triton.next_power_of_2(V))
        o = torch.empty_like(v)
        if self.idle_skip:
            from lx.fast_kernels import recurrent_step

            recurrent_step[(triton.cdiv(V, BV), HV)](
                q, k, v, g, beta, o, h, K ** -0.5, H=H, HV=HV, K=K, V=V, BK=BK, BV=BV,
                num_warps=1, num_stages=3)
            return o
        kernel[(triton.cdiv(V, BV), HV)](
            q=q, k=k, v=v, g=g, gk=None, gv=None, beta=beta, A_log=None, dt_bias=None, o=o,
            h0=h, ht=h, cu_seqlens=None, scale=K ** -0.5, T=1, H=H, HV=HV, K=K, V=V, BK=BK,
            BV=BV, IS_BETA_HEADWISE=True, USE_QK_L2NORM_IN_KERNEL=True, APPLY_BETA_SIGMOID=False,
            ALLOW_NEG_EIGVAL=False, STATE_V_FIRST=False, num_warps=1, num_stages=3)
        return o

    @staticmethod
    def _gdn_post(norm, d: _Dec, x: torch.Tensor, o: torch.Tensor, z: torch.Tensor) -> None:
        if len(d.out_rows):
            on = norm(o.view(z.shape), z)
            x.index_add_(1, d.out_rows, on.view(1, -1) @ d.w_out.T)

    @staticmethod
    def _mlp_step(d: _Dec, x: torch.Tensor) -> None:
        gu = _norm_cols(x, d.eps_m, d.w1m, d.m_cin) @ d.w_gu.T
        act = F.silu(gu[:, :d.n_u]) * gu[:, d.n_u:]
        x.index_add_(1, d.m_rows, act @ d.w_d.T)

    # ------------------------------------------------------------------------------------------
    @torch.no_grad()
    def head_argmax(self, h: torch.Tensor, block: int | None = None) -> torch.Tensor:
        """The output head's argmax over the rows given at construction, for h (T, hidden)."""
        out, block = [], block or self.head_block
        for a in range(0, h.shape[0], block):
            hn = _Norm(self.model.model.norm, h[a:a + block])(self.head_cols)
            out.append((hn @ self.w_head.T).argmax(-1))
        return torch.cat(out)
