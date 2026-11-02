"""Phase 1.3: chunked-prefill throughput of candidate chip-layer shapes for the Linux machine.

    python bench_gdn_config.py --layers 32 --dk 256 --nv 128 [--chunk 16384] [--tokens 262144]

Random bf16 weights in stock HF Qwen3.5 GatedDeltaNet decoder layers (the fla chunk kernels),
driven like the chip view (no attention layers, cache carried across chunks). Measures speed
only; exactness is checked on the real circuit (qp_longprefill.py).
"""

import argparse
import json
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--dk", type=int, default=256)
    ap.add_argument("--nk", type=int, default=16)
    ap.add_argument("--nv", type=int, default=128)
    ap.add_argument("--inter", type=int, default=9216)
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--tokens", type=int, default=262144)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    from transformers import DynamicCache, Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q

    cfg = Qwen3_5TextConfig(hidden_size=args.hidden, num_hidden_layers=args.layers,
                            layer_types=["linear_attention"] * args.layers,
                            linear_key_head_dim=args.dk, linear_num_key_heads=args.nk,
                            linear_num_value_heads=args.nv, linear_value_head_dim=128,
                            intermediate_size=args.inter, vocab_size=1024)
    torch.manual_seed(0)
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        layers = torch.nn.ModuleList([q.Qwen3_5DecoderLayer(cfg, i) for i in range(args.layers)])
    torch.set_default_dtype(torch.float32)
    for p in layers.parameters():
        p.data.normal_(0, 0.02)
    for L in layers:
        L.linear_attn.A_log.data.fill_(0.0)
        L.linear_attn.dt_bias.data.fill_(-4.0)
    n_par = sum(p.numel() for p in layers.parameters())
    layers.eval()

    def fwd(h, cache):
        for L in layers:
            h = L(h, position_embeddings=None, attention_mask=None, position_ids=None,
                  past_key_values=cache)
        return h

    x = torch.randn(1, args.tokens, args.hidden, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        fwd(x[:, : args.chunk], DynamicCache(config=cfg))  # autotune
        torch.cuda.synchronize()
        cache = DynamicCache(config=cfg)
        t0 = time.time()
        for a in range(0, args.tokens, args.chunk):
            fwd(x[:, a:a + args.chunk], cache)
        torch.cuda.synchronize()
        dt = time.time() - t0
    state_mb = args.layers * args.nv * args.dk * 128 * 4 / 2**20
    out = {**vars(args), "params_B": round(n_par / 1e9, 2), "tok_per_s": round(args.tokens / dt),
           "ram_words_per_layer": args.nv * args.dk, "state_fp32_MB": round(state_mb),
           "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
           "device": torch.cuda.get_device_name()}
    print(json.dumps(out), flush=True)
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
