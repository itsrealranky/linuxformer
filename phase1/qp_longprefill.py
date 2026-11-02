"""Phase 1.2: a multi-million-token RV32 trace through the released Qwenputer checkpoint
(divagr1925/Qwen3.5-4B-Qwenputer) as one chunked prefill, checking the model's greedy next token
at every position.

    python qp_longprefill.py MODEL_DIR --prog c-sudoku --chunk 16384 [--max-tokens N]

The chip layers (the first ``chip_layers`` of the checkpoint, stock HF Qwen3.5 GatedDeltaNet
modules, bf16, fla chunk kernels) run the reference machine's trace from GO with the program's
snapshot injected into the recurrent states (as chipformer.rv.hf.check / RvChat do). Chunks are
consecutive forwards that carry the HF cache (conv windows and recurrent states).

At every position two predictions of the next token are compared with the trace:
  head  argmax of the tied lm_head over the whole vocabulary on the final-normed residual
        (Qwen's layers skipped, as RvChat does for chip tokens)
  code  the circuit's output code read from the residual after the chip layers (RvChat's decode)
Prints positions, mismatches of each, and tokens/s of the chunked prefill alone and with the head.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

QP = Path(__file__).resolve().parents[1] / "qwenputer/experimental/riscv"
sys.path.insert(0, str(QP))

from chipformer.q35.ir import Target  # noqa: E402
from chipformer.rv import bridge as BR  # noqa: E402
from chipformer.rv import core, elf, hf, ref  # noqa: E402
from chipformer.rv import spec as S  # noqa: E402
from chipformer.rv.measure import DATA  # noqa: E402


def load_chip_model(path: Path, dev: str):
    """The checkpoint's embeddings, chip layers and final norm as a Qwen3_5ForCausalLM with no
    attention layer (the chip layers are driven one by one, see chip_forward)."""
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    cfg = json.loads((path / "config.json").read_text())
    meta = json.loads((path / "chipformer.json").read_text())
    n_chip = meta["chip_layers"]
    tc = {k: v for k, v in cfg.items() if k not in ("architectures", "model_type", "dtype")}
    tc.update(num_hidden_layers=n_chip, layer_types=["linear_attention"] * n_chip)
    torch.set_default_dtype(torch.bfloat16)
    with torch.device(dev):
        model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**tc))
    torch.set_default_dtype(torch.float32)
    for p in model.parameters():
        p.data.zero_()
    want = {"model.embed_tokens.weight", "model.norm.weight"}
    want |= {k for k in model.state_dict() if k.startswith("model.layers.")
             and int(k.split(".")[2]) < n_chip}
    idx = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    sd = model.state_dict()
    for f in sorted({idx[k] for k in want}):
        with safe_open(path / f, "pt", device=dev) as fh:
            for k in fh.keys():  # noqa: SIM118
                if k in want:
                    sd[k].copy_(fh.get_tensor(k))
                    want.discard(k)
    assert not want, sorted(want)[:5]
    model.tie_weights()
    return model.eval(), meta, cfg


def chip_forward(model, h, cache):
    """The chip layers on residual rows h (1, T, hidden), carrying the cache (as RvChat's chip
    view: no positions, no masks; GatedDeltaNet layers only)."""
    for layer in model.model.layers:
        h = layer(h, position_embeddings=None, attention_mask=None, position_ids=None,
                  past_key_values=cache)
    return h


def new_cache(model, c, dev):
    """A cache after one zero-embedding prefix token, with the snapshot injected."""
    from transformers import DynamicCache

    cache = DynamicCache(config=model.config)
    h = model.config.hidden_size
    chip_forward(model, torch.zeros(1, 1, h, dtype=torch.bfloat16, device=dev), cache)
    hf.inject(c, cache)
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--prog", default="c-sudoku")
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--head-block", type=int, default=4096)
    ap.add_argument("--no-head", action="store_true", help="skip the full lm_head check")
    ap.add_argument("--mem-gb", type=float, default=0, help="cap this process's CUDA memory")
    ap.add_argument("--out", default="")
    ap.add_argument("--echo", action="store_true",
                    help="insert the bridge's echo text token after every putchar (BYTE), as the "
                         "merged model's stream has it")
    args = ap.parse_args()
    dev = "cuda"
    if args.mem_gb:
        tot = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(args.mem_gb / tot)
    path = Path(args.model)

    t0 = time.time()
    model, meta, cfg = load_chip_model(path, dev)
    n_chip, v0 = meta["chip_layers"], meta["chip_token_offset"]
    T = BR.Text(path / "tokenizer.json")
    c = core.build(Target(cfg["linear_num_key_heads"], cfg["linear_num_value_heads"],
                          cfg["intermediate_size"]), T)
    print(f"model + circuit {time.time() - t0:.0f} s, {torch.cuda.memory_allocated() / 2**30:.1f}"
          f" GB", flush=True)

    t0 = time.time()
    prog = elf.program(DATA / f"{args.prog}.elf")
    toks = ref.Machine.from_program(prog).run(10**9)
    if args.max_tokens:
        toks = toks[: args.max_tokens]
    n = len(toks)
    n_instr = sum(1 for t in toks if S.is_final(t) and S.family(t)[0] != "R")
    print(f"{args.prog}: {n} tokens, {n_instr} instructions ({n / n_instr:.3f} tokens/instr), "
          f"reference {time.time() - t0:.0f} s", flush=True)
    core.snapshot(c, ref.Machine.from_program(prog))

    stream = []
    for i, t in enumerate(toks):
        stream.append(v0 + t)
        fam, p = S.family(t)
        if args.echo and fam == "BYTE" and i + 1 < len(toks):
            stream.append(T.printable[p])  # the merged model echoes a putchar as Qwen text
    n = len(stream)
    if args.echo:
        print(f"stream with echoes: {n} tokens", flush=True)
    ids = torch.tensor(stream, device=dev)
    target = ids[1:]
    code_dims = torch.tensor(meta["code_dims"], device=dev)
    nfam = meta["nfam"]
    fam_w = (2 ** torch.arange(nfam, device=dev))
    pay_w = (2 ** torch.arange(len(meta["code_dims"]) - nfam, device=dev))
    # chip id of every (family id, payload) pair the decoder can return
    lut = {}

    def code_ids(hc: torch.Tensor) -> torch.Tensor:
        bits = (hc[:, code_dims].float() > 0).long()
        fid = (bits[:, :nfam] * fam_w).sum(1)
        pay = (bits[:, nfam:] * pay_w).sum(1)
        key = (fid << 16 | pay).cpu().numpy()
        out = np.empty(len(key), dtype=np.int64)
        for i, kk in enumerate(key):
            v = lut.get(int(kk))
            if v is None:
                t = core.decode_code(int(kk) >> 16, int(kk) & 0xFFFF, T)
                v = -1 if t < 0 else (t - S.TEXT_TOKEN if S.is_text(t) else v0 + t)
                lut[int(kk)] = v
            out[i] = v
        return torch.from_numpy(out).to(dev)

    from transformers import AutoTokenizer

    T_tok = AutoTokenizer.from_pretrained(path)
    mism = []

    def name_of(i: int) -> str:
        return S.name(i - v0) if i >= v0 else repr(T_tok.decode([i])) if i >= 0 else str(i)

    def run(cache, a, b, check):
        hc = chip_forward(model, model.model.embed_tokens(ids[a:b].view(1, -1)), cache)[0]
        if not check:
            return 0, 0
        hn = model.model.norm(hc)
        m = min(b, n - 1) - a  # positions with a next token
        cpred = code_ids(hc[:m])
        bad_code = int((cpred != target[a:a + m]).sum())
        bad_head = 0
        if not args.no_head:
            for s in range(0, m, args.head_block):
                e = min(m, s + args.head_block)
                pred = model.lm_head(hn[s:e]).argmax(-1)
                wrong = (pred != target[a + s:a + e]).nonzero()[:, 0]
                bad_head += len(wrong)
                for w in wrong.tolist()[: max(0, 12 - len(mism))]:
                    p, want = a + s + w, int(target[a + s + w])
                    got = int(pred[w])
                    mism.append({"pos": p, "want": name_of(want), "head": name_of(got),
                                 "code": name_of(int(cpred[s + w])),
                                 "prev": [name_of(int(x)) for x in ids[max(0, p - 3):p + 1]]})
        return bad_head, bad_code

    # warm-up (fla's autotuner runs each new shape several times; it is kept off the real cache)
    with torch.no_grad():
        shapes = {args.chunk, n % args.chunk or args.chunk}
        for L in shapes:
            wc = new_cache(model, c, dev)
            run(wc, 0, min(L, n), True)
            del wc
        torch.cuda.synchronize()

        res = {}
        for mode in ("prefill", "prefill+check"):
            cache = new_cache(model, c, dev)
            torch.cuda.synchronize()
            t0 = time.time()
            bad_head = bad_code = 0
            first_bad = None
            for a in range(0, n, args.chunk):
                b = min(n, a + args.chunk)
                bh, bc = run(cache, a, b, mode != "prefill")
                if (bh or bc) and first_bad is None:
                    first_bad = a
                bad_head += bh
                bad_code += bc
            torch.cuda.synchronize()
            dt = time.time() - t0
            res[mode] = {"seconds": round(dt, 2), "tok_per_s": round(n / dt),
                         "instr_per_s": round(n_instr / dt)}
            if mode != "prefill":
                res[mode].update(positions=n - 1, head_mismatches=bad_head,
                                 code_mismatches=bad_code, first_bad_chunk=first_bad,
                                 first_mismatches=mism[:12])
            print(mode, res[mode], flush=True)
            del cache
    out = {"model": str(path), "prog": args.prog, "tokens": n, "instructions": n_instr,
           "tokens_per_instr": n / n_instr, "chunk": args.chunk,
           "device": torch.cuda.get_device_name(), "peak_gb":
           round(torch.cuda.max_memory_allocated() / 2**30, 2), **res}
    print(json.dumps(out), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
