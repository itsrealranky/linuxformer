---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
tags:
- qwen3.5
- compiled-weights
- risc-v
- linux
---

# linuxformer: Qwen3.5-4B with a Linux-capable CPU in its weights

*Every number here is from `lx/results/summary.md` in the [code repository](https://github.com/itsrealranky/linuxformer).*

This checkpoint is stock Qwen3.5-4B (all 426 tensors bit-identical) with 37 added layers that
implement an RV32IMA + Zicsr CPU. Those layers are computed by a compiler, not trained. The
CPU's 4.5 MB of RAM and its kernel-text ROM are Gated DeltaNet recurrent state. It boots no-MMU
Linux 6.8 to a BusyBox shell, and Qwen can call that shell as a tool.

| | |
|---|---|
| architecture | `Qwen3_5ForCausalLM`, 69 layers (37 chip + 32 Qwen), hidden 2560, 15.99B parameters, bf16 |
| vocabulary | Qwen's 248,320 tokens, then the chip's 459,982 tokens (id 248,320 + t) |
| config change | `rms_norm_eps` 5e-7 (Qwen: 1e-6), which cancels the zero-padding of Qwen's GDN heads exactly |
| model alone | plain decoding, no reference: from reset to the shell prompt, all 7,331,612 tokens identical to the reference, in 20.3 min (6,017 tok/s) on one H100 80GB with the megakernel (`lx/csrc/mega.cu`); the stock decode runs at 80.8 tok/s. Model load (25 s from these files) is not included |
| verified mode | mini-rv32ima proposes the trace and the model checks every token in chunked prefill: 0 wrong, 2.1 min with `lx.fast`, 13.0 min with the stock kernels, ending in the same state to the bit |
| MMLU (zero-shot, 14,042) | 72.61%, vs 72.72% for stock Qwen3.5-4B; same answer on 98.55% |

## How to run it

Use the drivers in the [code repository](https://github.com/itsrealranky/linuxformer) (one 80 GB GPU; add `--fast` for `lx.fast`, or `--lean`
to stay within 22 GiB of GPU memory, see Limitations):

```bash
git clone https://github.com/itsrealranky/linuxformer && cd linuxformer
pip install -r requirements.txt                    # Python 3.11, CUDA 13
hf download itsrealranky/linuxformer --local-dir model
cd lx
python -m lx.linux --weights ../model --snapshot ../model/boot-snapshot.safetensors -i   # a shell
python -m lx.chat  --weights ../model --snapshot ../model/boot-snapshot.safetensors \
    "Compute 123456 * 789 using shell arithmetic on the machine, and tell me the result."
python -m lx.chat --alone --weights ../model --snapshot ../model/boot-snapshot.safetensors \
    "Compute 123456 * 789 using shell arithmetic on the machine, and tell me the result."   # model alone (H100)
```

The shell is live about 30 s after start: 25 s to rebuild the circuit's layout and load the
weights, then 1 s for the snapshot. `lx.merge.load` reads the shards with every tensor in its
stored dtype. Qwen's `A_log` and gated-norm weights are fp32. `from_pretrained` with a dtype
would cast them, so it is not bit-identical to the build. On this H100, `tools/export_hf.py`
checks that the loaded model equals a fresh build, all 945 tensors bit for bit, and
`echo hi` from the exported files verified 182,928 tokens in 19.6 s.

`--fast` runs the chip layers with `lx/fast.py`: the same tensors, with their zero rows,
columns and idle GDN heads skipped (the chip weights are 0.08% nonzero), on the stock kernels.
Its boot ends in the stock boot's state to the bit, 6.3x faster; its decode is bit-identical
to the stock decode, 5.4x faster.

Tokens are routed by kind, as in Qwenputer's RvChat. Chip tokens run layers 0–36 and the head
over the chip's rows. Text tokens run layers 37–68 through `lx.merge.text_view`, a stock
Qwen3.5-4B sharing those modules. A plain `generate()` over the whole checkpoint does not run
either the machine or Qwen correctly.

Files:
- `model-0000{1..7}-of-00007.safetensors`: the checkpoint, 31.98 GB (897 bf16 tensors, 48 fp32).
- `config.json`, `generation_config.json`, Qwen's tokenizer files, and `lx_info.json` (the
  circuit's shape and the chip's token offset).
- `boot-snapshot.safetensors` (634 MB): the machine after boot, at the shell prompt. It holds
  every chip layer's conv window and recurrent state plus the reference machine's registers and
  memory. No pickle.
- `gpl-source/`: the corresponding source of the GPL and LGPL programs in that snapshot: the
  Linux, BusyBox and uClibc-ng tarballs with their patches, configs and build scripts
  (`gpl-source/README.txt`, `SHA256SUMS`).

## Limitations

- **Speed.** The model alone reaches about 6,000 tokens/s only with the megakernel, which is
  written for the H100 and drives 262,144-token chunks from the host (between chunks the host
  passes only the model's own last prediction, and types GO and keystrokes). The stock decode
  is 80.8 tokens/s. A shell command costs 0.18–1.06M tokens (20–113 s in verified mode with
  the stock kernels).
- **Text behavior.** It is close to stock but not identical. It gives the same MMLU answer as
  stock on 98.55% of questions; stock with eager attention instead of sdpa does so on 98.50%.
  The mean KL is 5.8e-4, against 4.5e-4 for that kernel swap.
- **Machine size.** The machine has 4.5 MB of RAM, no MMU and no disk.
- **Hardware.** With the stock forward, 32 GB of weights plus prefill activations: one 80 GB
  GPU. With `--lean`, `lx.fast` keeps the chip layers' projections (16 GB, 0.08% nonzero) in
  host memory and only their nonzero parts on the GPU. The boot from reset (0 wrong, final state
  identical to the stock boot's, 215.6 s) and a Qwen chat both ran with PyTorch's allocator
  capped at 22 GiB, peaking at 20.9 GB. That was on the H100; on a real 24 GB card it is
  untested, and it will be slower. Loading needs about 32 GB of host RAM.

## License

The checkpoint is Apache-2.0: Qwen3.5-4B's weights plus compiled layers. The booted snapshot
holds a memory image of Linux and BusyBox (GPL-2.0), with uClibc-ng (LGPL-2.1) linked into
BusyBox. Their complete corresponding source,
the kernel, BusyBox and uClibc-ng tarballs with their patches and configs, is in `gpl-source/`
here; the code repository builds them (`NOTICE`, `scripts/setup.sh`).

## Credits

Built on [Qwenputer](https://github.com/divagr18/qwenputer) (Divyansh Agrawal, MIT): the circuit
compiler, the gadgets, the divider and the multiplier. The machine follows
[mini-rv32ima](https://github.com/cnlohr/mini-rv32ima) (CNLohr, MIT). Built on
[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (Qwen team, Apache-2.0).
