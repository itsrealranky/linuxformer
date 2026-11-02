"""Where the time goes in one chunked-prefill step of a GatedDeltaNet decoder layer.

    python profile_layer.py --dk 256 --nv 128 --chunk 16384
"""

import argparse

import torch
from torch.profiler import ProfilerActivity, profile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--dk", type=int, default=256)
    ap.add_argument("--nk", type=int, default=16)
    ap.add_argument("--nv", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=16384)
    args = ap.parse_args()
    from transformers import DynamicCache, Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q

    cfg = Qwen3_5TextConfig(hidden_size=args.hidden, num_hidden_layers=1,
                            layer_types=["linear_attention"], linear_key_head_dim=args.dk,
                            linear_num_key_heads=args.nk, linear_num_value_heads=args.nv,
                            linear_value_head_dim=128, intermediate_size=9216, vocab_size=1024)
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        layer = q.Qwen3_5DecoderLayer(cfg, 0)
    torch.set_default_dtype(torch.float32)
    for p in layer.parameters():
        p.data.normal_(0, 0.02)
    x = torch.randn(1, args.chunk, args.hidden, device="cuda", dtype=torch.bfloat16)

    def step(cache):
        return layer(x, position_embeddings=None, attention_mask=None, position_ids=None,
                     past_key_values=cache)

    with torch.no_grad():
        cache = DynamicCache(config=cfg)
        for _ in range(3):
            step(cache)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                step(cache)
            torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18, max_name_column_width=70))


if __name__ == "__main__":
    main()
