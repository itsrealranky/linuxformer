"""The merged model: the LX chip layers, then stock Qwen3.5-4B's 32 layers, in one
Qwen3_5ForCausalLM at Qwen's own hidden size (2560).

    model, info = build(c)                       # c = core.build(core.Config()), the Linux machine
    text = text_view(model, info)                # Qwen3.5-4B: a stock Qwen3_5ForCausalLM

Layers 0..36 are the circuit (lx.emit, residual slots shared, the exit clearing all but the
output code); layers 37..68 are Qwen's, in Qwen's order. Ids below QV are Qwen's tokens (its
embedding rows, bit-identical); chip token t is id QV + t (the circuit's rows). Every weight of
Qwen's is copied unchanged, in its stored dtype (bf16; A_log and the gated norms' weights
fp32); the one change is the shape of its 24 Gated DeltaNet layers, which
must match the circuit's (one config for all layers): key heads of 256 dims instead of 128
(zero-padded) and 128 value heads instead of 32 (Qwen's two per key head, then six zero
heads). The padding leaves every projection, the conv, the decay and the L2-normalized keys
as they were, but the kernel's query scale 1/sqrt(d_k) halves the head output's square. The
gated RMSNorm after it absorbs that up to its epsilon, which matters: Qwen's head outputs have
a mean square of 1e-8..1e-6, the size of eps = 1e-6. So the merged config's rms_norm_eps is
halved too (5e-7), which cancels it exactly (o / sqrt(2) normed with eps / 2 is o normed with
eps); in every other norm eps is far below the activations (measured: no change in the logits).

Tokens are routed by kind, as Qwenputer's RvChat does: chip tokens run the chip layers only (the
machine's state is their cache) and the head over the chip rows; text tokens run Qwen's layers
only, through text_view, which is a stock model whose layers are the merged model's.
"""

import json
from pathlib import Path

import torch

from lx import core, hf
from lx import spec as S
from lx.emit import allocate, emit_layers

QWEN = Path(__file__).resolve().parents[2] / "models" / "Qwen3.5-4B"
QV = 248320  # Qwen3.5-4B's vocabulary (embedding rows)
QPRE = "model.language_model."


def config() -> core.Config:
    """The Linux machine's circuit for the merged model: its exit leaves the output code at
    +-1 (the head is Qwen's final norm, whose eps would swamp Qwenputer's 2^-20)."""
    return core.Config(kappa=1.0)


@torch.no_grad()
def check_head(model, info, block: int = 8192) -> list[int]:
    """Chip tokens the head gets wrong when the residual holds exactly their code (+-kappa in
    the code dims, zero elsewhere: what the exit leaves): Qwen's final norm, then the tied
    embedding's chip rows. Every token is tried."""
    rows = chip_rows(info)
    E = model.lm_head.weight[rows]
    code = torch.tensor(info["code_dims"], device=E.device)
    bad = []
    for a in range(0, E.shape[0], block):
        h = torch.zeros(min(block, E.shape[0] - a), E.shape[1], dtype=E.dtype, device=E.device)
        h[:, code] = (E[a:a + block][:, code].float().sign() * info["kappa"]).to(E.dtype)
        p = hf.head_argmax(model, h, rows=rows)
        want = torch.arange(a, a + h.shape[0], device=E.device)
        bad += (want[p != want]).tolist()
    return bad


def _qwen_text_config(qdir: Path) -> dict:
    cfg = json.loads((qdir / "config.json").read_text())
    return cfg.get("text_config", cfg)


def _qwen_weights(qdir: Path) -> dict:
    from safetensors import safe_open

    idx = json.loads((qdir / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for f in sorted(set(idx.values())):
        with safe_open(qdir / f, "pt") as fh:
            for k in fh.keys():
                if k.startswith(QPRE):
                    out[k[len(QPRE):]] = fh.get_tensor(k)
    return out


def pad_gdn(w: dict, q: dict, nk: int, dk: int, nv: int, dv: int) -> dict:
    """Qwen's Gated DeltaNet weights (``w``: names under ``linear_attn.``) in the circuit's
    shape: key dim q dk -> dk, value heads q nv -> nv (Qwen's value head h, of key head h // r,
    is value head (h // r) * R + h % r)."""
    qnk, qdk, qnv, qdv = (q["linear_num_key_heads"], q["linear_key_head_dim"],
                          q["linear_num_value_heads"], q["linear_value_head_dim"])
    assert qnk == nk and qdv == dv and qdk <= dk and nv % nk == 0 and qnv % qnk == 0
    r, R = qnv // qnk, nv // nk
    vmap = [(h // r) * R + h % r for h in range(qnv)]
    # channel maps of the conv / qkv rows: q, k (head-major), then v
    qk = [j * dk + i for j in range(nk) for i in range(qdk)]
    rows = qk + [nk * dk + x for x in qk] + [2 * nk * dk + vh * dv + i for vh in vmap
                                               for i in range(dv)]
    rows = torch.tensor(rows)
    vrows = torch.tensor([vh * dv + i for vh in vmap for i in range(dv)])
    heads = torch.tensor(vmap)
    conv_dim = 2 * nk * dk + nv * dv

    def put(shape, idx, src, dim=0):
        t = torch.zeros(shape, dtype=src.dtype)
        t.index_copy_(dim, idx, src)
        return t

    hdim = w["in_proj_qkv.weight"].shape[1]
    out = {
        "in_proj_qkv.weight": put((conv_dim, hdim), rows, w["in_proj_qkv.weight"]),
        "conv1d.weight": put((conv_dim,) + tuple(w["conv1d.weight"].shape[1:]), rows,
                             w["conv1d.weight"]),
        "in_proj_z.weight": put((nv * dv, hdim), vrows, w["in_proj_z.weight"]),
        "in_proj_b.weight": put((nv, hdim), heads, w["in_proj_b.weight"]),
        "in_proj_a.weight": put((nv, hdim), heads, w["in_proj_a.weight"]),
        "dt_bias": put((nv,), heads, w["dt_bias"]),
        "A_log": put((nv,), heads, w["A_log"]),
        "norm.weight": w["norm.weight"],
        "out_proj.weight": put((hdim, nv * dv), vrows, w["out_proj.weight"], dim=1),
    }
    for k in ("in_proj_qkv.weight", "conv1d.weight"):
        assert torch.equal(out[k].index_select(0, rows), w[k])
    return out


def build(c, qdir: Path = QWEN, dtype=torch.bfloat16, device: str = "cuda"):
    """(model, info) for circuit c and Qwen3.5-4B in qdir."""
    from transformers import GenerationConfig, Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q

    qc = _qwen_text_config(qdir)
    hidden = qc["hidden_size"]
    assert c.cfg.kappa == 1.0, "build the circuit with merge.config()"
    keep = set(c.keep)
    alloc = allocate(c, None, keep, c.target.intermediate)
    lay = hf.pick_layout(alloc.n_slots + 1, hidden)
    lay.slots = alloc.slots
    lay.scale  # noqa: B018 - asserts m * B^2 / hidden is a power of 4
    n_chip = c.n_layers()
    assert c.target.intermediate == qc["intermediate_size"]
    assert c.target.key_heads == qc["linear_num_key_heads"]
    n_vocab = QV + hf.N_VOCAB
    tc = dict(hf.BASE, hidden_size=hidden, vocab_size=n_vocab,
              num_hidden_layers=n_chip + qc["num_hidden_layers"],
              layer_types=["linear_attention"] * n_chip + list(qc["layer_types"]),
              linear_key_head_dim=hf.KD, linear_value_head_dim=hf.VD,
              linear_num_key_heads=c.target.key_heads,
              linear_num_value_heads=c.target.value_heads,
              intermediate_size=qc["intermediate_size"], eos_token_id=qc["eos_token_id"],
              pad_token_id=qc.get("pad_token_id"), bos_token_id=qc.get("bos_token_id"),
              rms_norm_eps=qc["rms_norm_eps"] * qc["linear_key_head_dim"] / hf.KD)
    for k in ("num_attention_heads", "num_key_value_heads", "head_dim",
              "attn_output_gate", "rope_parameters", "linear_conv_kernel_dim", "hidden_act",
              "tie_word_embeddings", "max_position_embeddings"):
        assert tc[k] == qc[k], (k, tc[k], qc[k])
    sd = emit_layers(c, lay, tc, 0, "model.", alloc, keep, dtype=dtype)
    qw = _qwen_weights(qdir)
    emb = qw.pop("embed_tokens.weight")
    assert emb.shape == (QV, hidden) and emb.dtype == dtype
    sd["model.embed_tokens.weight"] = torch.cat([emb, hf.embedding_matrix(c, lay).to(dtype)])
    sd["model.norm.weight"] = qw.pop("norm.weight")
    keep_dtype = {}  # Qwen's tensors stored in another dtype (A_log, the gated norms: fp32)
    for i in range(qc["num_hidden_layers"]):
        p = f"layers.{i}."
        mine = {k[len(p):]: qw.pop(k) for k in list(qw) if k.startswith(p)}
        if qc["layer_types"][i] == "linear_attention":
            gdn = {k[len("linear_attn."):]: mine.pop(k) for k in list(mine)
                   if k.startswith("linear_attn.")}
            mine.update({"linear_attn." + k: v for k, v in
                         pad_gdn(gdn, qc, tc["linear_num_key_heads"], hf.KD,
                                 tc["linear_num_value_heads"], hf.VD).items()})
        for k, v in mine.items():
            sd[f"model.layers.{n_chip + i}.{k}"] = v
            if v.dtype != dtype:
                keep_dtype[f"model.layers.{n_chip + i}.{k}"] = v
    assert not qw, list(qw)[:5]
    config = Qwen3_5TextConfig(**tc)
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    assert not unexpected and set(missing) <= {"lm_head.weight"}, (missing[:5], unexpected[:5])
    del sd
    model.tie_weights()
    model.model.rotary_emb = q.Qwen3_5TextRotaryEmbedding(config=config)
    model.generation_config = GenerationConfig(eos_token_id=[248046, 248044], do_sample=False)
    model.layout, model.alloc = lay, alloc
    info = {"chip_layers": n_chip, "qwen_layers": qc["num_hidden_layers"], "qwen_vocab": QV,
            "chip_token_offset": QV, "chip_vocab": S.VOCAB, "hidden": hidden,
            "ballast_dims": lay.m, "chip_slots": alloc.n_slots,
            "code_dims": [lay.dim(d) for d in c.keep], "kappa": c.cfg.kappa,
            "circuit": c.stats()}
    model = model.to(device=device, dtype=dtype)
    # inv_freq stays fp32, as in a stock load (.to(dtype) casts buffers too)
    model.model.rotary_emb = q.Qwen3_5TextRotaryEmbedding(config=config).to(device)
    for name, v in keep_dtype.items():
        mod, attr = name.rsplit(".", 1)
        setattr(model.get_submodule(mod), attr,
                torch.nn.Parameter(v.to(device), requires_grad=False))
    return model.eval(), info


TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                   "chat_template.jinja")


def save(model, info, out: Path, qdir: Path = QWEN) -> list[Path]:
    """The merged checkpoint as a Hugging Face directory: safetensors shards (every tensor in its
    stored dtype), config, generation config, Qwen's tokenizer, and lx_info.json. Returns the
    shards."""
    import shutil

    out = Path(out)
    model.save_pretrained(out, max_shard_size="5GB")
    for f in TOKENIZER_FILES:
        shutil.copy(qdir / f, out / f)
    (out / "lx_info.json").write_text(json.dumps(info, indent=1) + "\n")
    return sorted(out.glob("model*.safetensors"))


def load(path: Path, c, device: str = "cuda"):
    """(model, info) from a save() directory, for circuit c (= core.build(config())): the
    tensors as stored, so the model is build(c)'s bit for bit. (from_pretrained with a dtype
    would cast Qwen's fp32 tensors.)"""
    from safetensors.torch import load_file
    from transformers import GenerationConfig, Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q

    path = Path(path)
    config = Qwen3_5TextConfig.from_pretrained(path)
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
    idx = path / "model.safetensors.index.json"
    shards = sorted(set(json.loads(idx.read_text())["weight_map"].values())) if idx.exists() \
        else ["model.safetensors"]
    sd = {}
    for f in shards:
        sd.update(load_file(path / f, device=device))
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    assert not unexpected and set(missing) <= {"lm_head.weight"}, (missing[:5], unexpected[:5])
    del sd
    model.tie_weights()
    model.model.rotary_emb = q.Qwen3_5TextRotaryEmbedding(config=config).to(device)
    model.generation_config = GenerationConfig.from_pretrained(path)
    keep = set(c.keep)
    alloc = allocate(c, None, keep, c.target.intermediate)
    lay = hf.pick_layout(alloc.n_slots + 1, config.hidden_size)
    lay.slots = alloc.slots
    model.layout, model.alloc = lay, alloc
    info = json.loads((path / "lx_info.json").read_text())
    assert info["circuit"] == c.stats(), "the checkpoint was built from another circuit"
    return model.eval(), info


def text_view(model, info):
    """Qwen3.5-4B as a stock Qwen3_5ForCausalLM whose modules are the merged model's (its 32
    layers, embedding, final norm; the head is the embedding's first QV rows): generate() on
    it is Qwen's text path. Its layers are numbered 0..31 for its own cache."""
    import copy

    from transformers import Qwen3_5ForCausalLM

    n = info["chip_layers"]
    cfg = copy.deepcopy(model.config)
    cfg.num_hidden_layers = info["qwen_layers"]
    cfg.layer_types = list(model.config.layer_types[n:])
    cfg.vocab_size = info["qwen_vocab"]
    with torch.device("meta"):
        tv = Qwen3_5ForCausalLM(cfg)
    tv.model.embed_tokens = model.model.embed_tokens  # ids < QV only
    tv.model.layers = torch.nn.ModuleList(model.model.layers[n:])
    for i, layer in enumerate(tv.model.layers):
        for m in layer.modules():
            if hasattr(m, "layer_idx"):
                m.layer_idx = i
    tv.model.norm = model.model.norm
    tv.model.rotary_emb = model.model.rotary_emb
    tv.lm_head = torch.nn.Linear(cfg.hidden_size, info["qwen_vocab"], bias=False,
                                 device="meta")
    tv.lm_head.weight = torch.nn.Parameter(model.model.embed_tokens.weight[:info["qwen_vocab"]],
                                           requires_grad=False)
    tv.config.tie_word_embeddings = False
    tv.generation_config = model.generation_config
    return tv.eval()


def chip_rows(info) -> slice:
    return slice(info["chip_token_offset"], info["chip_token_offset"] + info["chip_vocab"])
