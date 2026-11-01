"""The merged model (lx.merge) against its parts.

  weights  every tensor of Qwen3.5-4B's text model is in the merged model bit for bit (the
           Gated DeltaNet ones as the padded blocks lx.merge.pad_gdn maps them to; the rest of
           those blocks is zero)
  text     the text view against stock Qwen3.5-4B (the same safetensors in a stock
           Qwen3_5ForCausalLM): next-token KL and top-1 agreement over prompts, and greedy
           continuations
  chip     the chip layers (ids offset by Qwen's vocabulary, the head over the chip rows) on
           windows of the Linux boot: every prediction must be the trace's next token
  boot     the same for the whole boot, reset to the shell prompt (--boot)

    python tests/check_merge.py [--device cuda] [--boot] [--out FILE]
"""

import argparse
import json
import sys
import time

import torch
import torch.nn.functional as F
from check_core import linux_trace

from lx import core, merge
from lx import spec as S
from lx.session import Session

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Linux is a family of open-source Unix-like operating systems based on",
    "Question: What is 17 * 23?\nAnswer:",
    "In 1905, Albert Einstein published four papers that",
    "The mitochondria is the powerhouse of the cell. It produces",
    "Translate to French: 'The weather is nice today.'\nFrench:",
    "<|im_start|>user\nWrite a haiku about the sea.<|im_end|>\n<|im_start|>assistant\n",
]


def stock_qwen(qdir, device):
    """Qwen3.5-4B's text model in a stock Qwen3_5ForCausalLM, every tensor as stored."""
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    qc = merge._qwen_text_config(qdir)
    cfg = Qwen3_5TextConfig(**{k: v for k, v in qc.items() if k not in ("dtype",)})
    with torch.device("meta"):
        m = Qwen3_5ForCausalLM(cfg)
    sd = {"model." + k: v for k, v in merge._qwen_weights(qdir).items()}
    missing, unexpected = m.load_state_dict(sd, strict=False, assign=True)
    assert set(missing) <= {"lm_head.weight"} and not unexpected, (missing[:4], unexpected[:4])
    m.tie_weights()
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q

    m.model.rotary_emb = q.Qwen3_5TextRotaryEmbedding(config=cfg)
    return m.to(device=device).eval()


def check_weights(model, info, qdir) -> list:
    """Names of Qwen tensors that are not in the merged model bit for bit."""
    qc = merge._qwen_text_config(qdir)
    qw = merge._qwen_weights(qdir)
    msd = model.state_dict()
    n = info["chip_layers"]
    bad = []
    nk, nv = model.config.linear_num_key_heads, model.config.linear_num_value_heads
    for k, v in qw.items():
        if k == "embed_tokens.weight":
            got = msd["model.embed_tokens.weight"][:merge.QV]
        elif k == "norm.weight":
            got = msd["model.norm.weight"]
        else:
            _, i, rest = k.split(".", 2)
            got = msd[f"model.layers.{n + int(i)}.{rest}"]
        got = got.cpu()
        if ".linear_attn." in k:
            name = k.split(".linear_attn.")[1]
            others = got.clone()
            _zero_block(name, qc, nk, nv, others)
            ok = torch.equal(_gdn_block(name, qc, nk, nv)(got), v) and not others.any()
        else:
            ok = torch.equal(got, v)
        if not ok:
            bad.append(k)
    return bad


def _maps(qc, nk, nv, dk=256, dv=128):
    qnk, qdk, qnv = qc["linear_num_key_heads"], qc["linear_key_head_dim"], qc["linear_num_value_heads"]
    r, R = qnv // qnk, nv // nk
    vmap = [(h // r) * R + h % r for h in range(qnv)]
    qk = [j * dk + i for j in range(nk) for i in range(qdk)]
    rows = qk + [nk * dk + x for x in qk] + [2 * nk * dk + vh * dv + i for vh in vmap
                                               for i in range(dv)]
    vrows = [vh * dv + i for vh in vmap for i in range(dv)]
    return torch.tensor(rows), torch.tensor(vrows), torch.tensor(vmap)


def _gdn_block(name, qc, nk, nv):
    rows, vrows, heads = _maps(qc, nk, nv)
    return {"in_proj_qkv.weight": lambda t: t[rows], "conv1d.weight": lambda t: t[rows],
            "in_proj_z.weight": lambda t: t[vrows], "in_proj_b.weight": lambda t: t[heads],
            "in_proj_a.weight": lambda t: t[heads], "dt_bias": lambda t: t[heads],
            "A_log": lambda t: t[heads], "norm.weight": lambda t: t,
            "out_proj.weight": lambda t: t[:, vrows]}[name]


def _zero_block(name, qc, nk, nv, t):
    rows, vrows, heads = _maps(qc, nk, nv)
    if name in ("in_proj_qkv.weight", "conv1d.weight"):
        t[rows] = 0
    elif name == "in_proj_z.weight":
        t[vrows] = 0
    elif name in ("in_proj_b.weight", "in_proj_a.weight", "dt_bias", "A_log"):
        t[heads] = 0
    elif name == "out_proj.weight":
        t[:, vrows] = 0
    else:
        t.zero_()


def _kl(a, b):
    la, lb = F.log_softmax(a.float(), -1), F.log_softmax(b.float(), -1)
    return (la.exp() * (la - lb)).sum(-1).mean().item()


@torch.no_grad()
def check_text(tv, stock, tok, device, n_new: int = 48) -> dict:
    """KL(stock || merged) per prompt, beside the noise floor KL(stock sdpa || stock eager
    attention), top-1 agreement, and greedy continuations."""
    kls, floor, agree, total, same_gen = [], [], 0, 0, 0
    gens = []
    for p in PROMPTS:
        ids = torch.tensor([tok(p)["input_ids"]], device=device)
        a = stock(input_ids=ids).logits.float()
        b = tv(input_ids=ids).logits.float()
        stock.config._attn_implementation = "eager"
        a_eager = stock(input_ids=ids).logits.float()
        stock.config._attn_implementation = "sdpa"
        kls.append(_kl(a, b))
        floor.append(_kl(a, a_eager))
        agree += (a.argmax(-1) == b.argmax(-1)).sum().item()
        total += ids.shape[1]
        ga = stock.generate(ids, max_new_tokens=n_new, do_sample=False)[0, ids.shape[1]:]
        gb = tv.generate(ids, max_new_tokens=n_new, do_sample=False)[0, ids.shape[1]:]
        same_gen += torch.equal(ga, gb)
        gens.append({"prompt": p, "stock": tok.decode(ga), "merged": tok.decode(gb)})
    return {"prompts": len(PROMPTS), "mean_kl": sum(kls) / len(kls), "max_kl": max(kls),
            "noise_floor_mean_kl": sum(floor) / len(floor), "top1_agree": agree / total,
            "greedy_identical": same_gen, "generations": gens}


def check_chip(lx_c, model, info, cfg, windows) -> list:
    out = []
    for n, skip in windows:
        m0, toks = linux_trace(cfg, n, skip)
        s = Session(lx_c, model, m0, offset=info["chip_token_offset"])
        t0 = time.time()
        r = s.verify([S.tok("GO")] + toks)
        out.append({"records": [skip, skip + n], "tokens": len(toks) + 1, "wrong": r["wrong"],
                    "first_wrong": r["first_wrong"], "seconds": round(time.time() - t0, 1)})
        print(f"  chip: records {skip}..{skip + n}: {len(toks) + 1} tokens, "
              f"{r['wrong']} wrong", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--boot", action="store_true")
    ap.add_argument("--no-text", action="store_true")
    ap.add_argument("--no-chip", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    t0 = time.time()
    cfg = merge.config()
    c = core.build(cfg)
    model, info = merge.build(c, device=a.device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"merged: {info['circuit']}, hidden {info['hidden']}, {info['chip_layers']} + "
          f"{info['qwen_layers']} layers, {n_par / 1e9:.2f}B parameters, chip slots "
          f"{info['chip_slots']} ({time.time() - t0:.0f}s)", flush=True)
    res = {"device": a.device if a.device == "cpu" else torch.cuda.get_device_name(),
           **info, "params_B": round(n_par / 1e9, 2)}
    bad = check_weights(model, info, merge.QWEN)
    res["qwen_tensors_not_identical"] = bad
    print(f"weights: {len(bad)} Qwen tensors differ {bad[:4]}", flush=True)
    ok = not bad
    hb = merge.check_head(model, info)
    res["head_wrong_tokens"] = hb[:16]
    res["head_tokens_checked"] = S.VOCAB
    print(f"head: {len(hb)} of {S.VOCAB} chip tokens not recovered from their code {hb[:8]}",
          flush=True)
    ok &= not hb
    if not a.no_text:
        tok = AutoTokenizer.from_pretrained(merge.QWEN)
        tv = merge.text_view(model, info)
        stock = stock_qwen(merge.QWEN, a.device)
        r = check_text(tv, stock, tok, a.device)
        res["text"] = r
        print(f"text: mean KL {r['mean_kl']:.2e} (max {r['max_kl']:.2e}; noise floor, sdpa vs "
              f"eager: {r['noise_floor_mean_kl']:.2e}), top-1 agreement {r['top1_agree']:.4f}, "
              f"greedy identical {r['greedy_identical']}/{r['prompts']}", flush=True)
        ok &= r["mean_kl"] < 2 * r["noise_floor_mean_kl"]
        for g in r["generations"][:3]:
            print(f"    {g['prompt'][:40]!r}: stock {g['stock'][:60]!r} | merged "
                  f"{g['merged'][:60]!r}")
        del stock
        torch.cuda.empty_cache()
    if not a.no_chip:
        res["chip"] = check_chip(c, model, info, cfg,
                                 [(2000, 0), (620, 4970260), (120, 1240200)])
        ok &= all(w["wrong"] == 0 for w in res["chip"])
    if a.boot:
        from lx.linux import Linux

        lx = Linux.__new__(Linux)
        lx.cfg, lx.c, lx.model, lx.info, lx.offset = cfg, c, model, info, info["chip_token_offset"]
        lx.chunk, lx.s, lx.log = 65536, None, []
        b = lx.boot(None)
        res["boot"] = {k: v for k, v in b.items() if k != "out"}
        res["boot"]["console"] = b["out"].decode(errors="replace")[-200:]
        print(f"boot: {b['tokens']} tokens, 0 wrong, {b['seconds']}s ({b['tok_per_s']} tok/s)",
              flush=True)
    if a.out:
        from pathlib import Path

        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    print("all ok" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
