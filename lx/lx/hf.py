"""The LX core as a stock Qwen3_5ForCausalLM, with a machine snapshot injected into the cache.

    model, lay = build_model(c)                    # chip layers only (+ one zero attention layer)
    h, pred = run(c, model, seq)                   # residual after the chip layers, argmax tokens

Layer shapes are the widened Qwen3.5-4B layers (core.T4B: 16 key heads of KD = 256 dims, 128
value heads of 128 dims, MLP width 9216). Token id t is chip token t (lx.spec); id PAD = S.VOCAB
is a zero row. A run feeds PAD, writes the snapshot (the heads' ``init``) into the layers'
recurrent states (state row = key dim, columns = value dims), then runs the trace from GO, as
chunked prefill (fla's chunk_gated_delta_rule from that initial state) or token by token. The
output head is tied to the embedding: a row's code dims hold CODE_AMP * (2 * bit - 1) of its
token's code and the exit leaves the predicted code there, so the argmax over the whole
vocabulary is the token whose code bits all agree.
"""

import math

import numpy as np
import torch

from lx import spec as S
from lx.emit import Layout, allocate, check_bf16, emit_layers
from lx.ir import KD, VD, Head

BASE = {  # Qwen3.5-4B text config, minus the sizes we pick
    "hidden_act": "silu",
    "linear_conv_kernel_dim": 4,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "rms_norm_eps": 1e-6,
    "attn_output_gate": True,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000000,
                        "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10],
                        "mrope_interleaved": True},
    "max_position_embeddings": 262144,
    "tie_word_embeddings": True,
}
CODE_GAIN = 2.0**20  # final-norm gain on the output code (the exit leaves KAPPA * +-1 there)
PAD = S.VOCAB
N_VOCAB = -(-(S.VOCAB + 1) // 64) * 64


def pick_layout(n_dims: int, hidden: int | None = None) -> Layout:
    """Smallest hidden = m * 4^k (a multiple of 64) that fits m ballast dims and n_dims; or,
    given ``hidden``, the m (with hidden = m * 4^k) for it."""
    if hidden is not None:
        for m in range(3, 16):
            k = round(math.log(hidden / m, 4))
            if m * 4**k == hidden and m + n_dims <= hidden:
                return Layout(hidden=hidden, m=m)
        raise ValueError(f"no ballast count m with {hidden} = m * 4^k fits {n_dims} dims")
    best = None
    for m in range(3, 16):
        k = 0
        while m * 4**k < m + n_dims or (m * 4**k) % 64:
            k += 1
        if best is None or m * 4**k < best.hidden:
            best = Layout(hidden=m * 4**k, m=m)
    return best


def embedding_matrix(c, lay: Layout, n_vocab: int = N_VOCAB) -> torch.Tensor:
    """Rows of every chip token, PAD (zero) and padding (zero). A big family's features are
    linear in its payload bits (checked on a sample); small families are computed token by
    token."""
    feats = sorted(c.token_fns)
    cols = [lay.dim(d) for d in feats]

    def vec(t):
        return np.array([float(c.feature_value(d, t)) for d in feats])

    e = np.zeros((n_vocab, lay.hidden), dtype=np.float32)
    rng = np.random.default_rng(0)
    for name, (base, size) in S.FAMILIES.items():
        if size <= 4096:
            rows = np.stack([vec(base + i) for i in range(size)])
        else:
            nb = (size - 1).bit_length()
            v0 = vec(base)
            deltas = np.stack([vec(base + (1 << i)) - v0 for i in range(nb)])
            bits = (np.arange(size)[:, None] >> np.arange(nb)[None, :]) & 1
            rows = v0 + bits @ deltas
            for p in [size - 1, *rng.integers(0, size, 24)]:
                assert (vec(base + int(p)) == rows[p]).all(), (name, p)
        e[base:base + size][:, cols] = rows
        e[base:base + size, lay.off:lay.off + lay.m] = lay.ballast
    for t in S.CONTROLS.values():
        e[t, cols] = vec(t)
        e[t, lay.off:lay.off + lay.m] = lay.ballast
    e = torch.from_numpy(e)
    check_bf16("embedding", e)
    return e


def build_model(c, dtype=torch.bfloat16, share: bool = False, device: str = "cuda"):
    """(model, layout) for circuit c. With ``share``, dims share residual slots (allocate) and
    the exit clears everything but the output code, as in the merged model; without, every IR
    dim keeps its own residual dim to the end (so all of them can be compared)."""
    from transformers import GenerationConfig, Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q

    keep = set(c.keep)
    alloc = None
    if share:
        alloc = allocate(c, None, keep, c.target.intermediate)
        lay = pick_layout(alloc.n_slots + 1)
        lay.slots = alloc.slots
    else:
        lay = pick_layout(len(c.dims))
    lay.scale  # noqa: B018 - asserts m * B^2 / hidden is a power of 4
    n_chip = c.n_layers()
    tc = dict(BASE, hidden_size=lay.hidden, vocab_size=N_VOCAB, num_hidden_layers=n_chip + 1,
              layer_types=["linear_attention"] * n_chip + ["full_attention"],
              linear_key_head_dim=KD, linear_value_head_dim=VD,
              linear_num_key_heads=c.target.key_heads, linear_num_value_heads=c.target.value_heads,
              intermediate_size=c.target.intermediate, eos_token_id=S.tok("HALT"),
              pad_token_id=PAD, bos_token_id=PAD)
    sd = emit_layers(c, lay, tc, 0, "model.", alloc, keep if share else None, dtype=dtype)
    sd["model.embed_tokens.weight"] = embedding_matrix(c, lay).to(dtype)
    norm = torch.full((lay.hidden,), -1.0)  # (1 + w) = 0: the output head sees only the code
    for d in keep:
        norm[lay.dim(d)] = CODE_GAIN
    check_bf16("final norm", norm)
    sd["model.norm.weight"] = norm.to(dtype)
    config = Qwen3_5TextConfig(**tc)
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
    for k, p in model.state_dict().items():  # the zero attention layer (and the tied head)
        if k not in sd and k != "lm_head.weight":
            assert k.startswith(f"model.layers.{n_chip}."), k
            sd[k] = torch.zeros(p.shape, dtype=dtype)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    assert not unexpected and set(missing) <= {"lm_head.weight"}, (missing[:5], unexpected[:5])
    del sd
    model.tie_weights()
    model.model.rotary_emb = q.Qwen3_5TextRotaryEmbedding(config=config)  # non-persistent buffers
    model.generation_config = GenerationConfig(eos_token_id=S.tok("HALT"), pad_token_id=PAD,
                                               do_sample=False)
    model.layout, model.alloc = lay, alloc
    return model.to(device=device, dtype=dtype).eval(), lay


def head_slots(c) -> dict:
    """IR node index -> (layer, key head) in the emitted model (emit_layers' order)."""
    out = {}
    for L in range(c.n_layers()):
        heads = [i for i, n in enumerate(c.nodes) if isinstance(n, Head) and n.level == 2 * L + 1]
        for j, i in enumerate(heads):
            out[i] = (L, j)
    return out


def inject(c, cache) -> None:
    """Write the heads' initial states (the snapshot) into the cache's recurrent states."""
    r = c.target.value_heads // c.target.key_heads
    for ni, (L, j) in head_slots(c).items():
        st = cache.layers[L].recurrent_states[0]
        init = c.nodes[ni].init or []
        for x in range(len(c.nodes[ni].vheads())):
            st[0, j * r + x].zero_()
            rows = init[x] if x < len(init) else {}
            if rows:
                idx = torch.tensor(list(rows), device=st.device)
                st[0, j * r + x, idx] = torch.tensor(np.stack(list(rows.values())),
                                                     dtype=st.dtype, device=st.device)


def chip_forward(model, h: torch.Tensor, cache, n_chip: int) -> torch.Tensor:
    """The chip layers on residual rows h (1, T, hidden), carrying the cache: the "chip view"
    (no positions, no masks, and not the zero attention layer, whose cache would grow with the
    trace)."""
    for layer in model.model.layers[:n_chip]:
        h = layer(h, position_embeddings=None, attention_mask=None, position_ids=None,
                  past_key_values=cache)
    return h


def head_argmax(model, h: torch.Tensor, block: int = 2048, rows: slice | None = None):
    """The output head's argmax (final norm, tied embedding) for residual rows h (T, hidden);
    with ``rows``, over those vocabulary rows only (ids relative to rows.start)."""
    w = model.lm_head.weight if rows is None else model.lm_head.weight[rows]
    return torch.cat([torch.nn.functional.linear(model.model.norm(h[a:a + block]), w).argmax(-1)
                      for a in range(0, h.shape[0], block)])


@torch.no_grad()
def run(c, model, seq: list[int], mode: str = "prefill", chunk: int | None = None,
        view: str = "model", keep_h: bool = True):
    """Run seq (from GO) after PAD and the injected snapshot. Returns (h, pred): the residual
    after the last chip layer ([len(seq), hidden] on the device; None without ``keep_h``) and
    the output head's argmax at every position (a tensor on the device).

    view "model": the stock forward (model(input_ids, past_key_values)), the zero attention
    layer included; "chip": chip_forward, chunk by chunk.
    """
    n_chip = c.n_layers()
    dev = next(model.parameters()).device
    x = torch.tensor([seq], device=dev)
    step = 1 if mode == "decode" else (chunk or len(seq))
    hs, preds = [], []
    if view == "model":
        caught = []
        out = model(input_ids=torch.tensor([[PAD]], device=dev), use_cache=True,
                    logits_to_keep=1)
        cache = out.past_key_values
        inject(c, cache)
        hook = model.model.layers[n_chip - 1].register_forward_hook(
            lambda _m, _i, o: caught.append((o[0] if isinstance(o, tuple) else o)[0]))
        try:
            for a in range(0, len(seq), step):
                model(input_ids=x[:, a:a + step], past_key_values=cache, use_cache=True,
                      logits_to_keep=1)
                h = torch.cat(caught)
                caught.clear()
                preds.append(head_argmax(model, h))
                if keep_h:
                    hs.append(h)
        finally:
            hook.remove()
    else:
        from transformers import DynamicCache

        cache = DynamicCache(config=model.config)
        zero = torch.zeros(1, 1, model.config.hidden_size, dtype=model.dtype, device=dev)
        chip_forward(model, zero, cache, n_chip)
        inject(c, cache)
        for a in range(0, len(seq), step):
            h = chip_forward(model, model.model.embed_tokens(x[:, a:a + step]), cache, n_chip)[0]
            preds.append(head_argmax(model, h))
            if keep_h:
                hs.append(h)
    return (torch.cat(hs) if keep_h else None), torch.cat(preds)


def compare(c, model, h: torch.Tensor, want: np.ndarray) -> list:
    """IR dims whose residual differs from the IR evaluator's anywhere: (name, first position,
    got, want). Without slot sharing every dim is compared; with it, the output code."""
    lay = model.layout
    dims = sorted(c.keep) if model.alloc else list(range(1, len(c.dims)))
    got = h[:, [lay.dim(d) for d in dims]].double().cpu().numpy()
    w = want[:, dims]
    bad = np.nonzero((got != w).any(0))[0]
    out = []
    for i in bad[:8]:
        t = int(np.argmax(got[:, i] != w[:, i]))
        out.append((c.dims[dims[i]].name, t, float(got[t, i]), float(w[t, i])))
    return out
