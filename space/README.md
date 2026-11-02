---
title: Linux inside Qwen3.5-4B
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
suggested_hardware: a100-large
license: apache-2.0
short_description: A shell on the Linux computer compiled into Qwen's weights
---

A shell on the Linux machine inside the weights of Qwen3.5-4B, and Qwen using that shell as a
tool. Every machine token is verified against the reference. See the model card and the code
repository for how it works.

Needs one 80 GB GPU (`LX_FAST=1` uses the zero-skipping `lx.fast`; `LX_LEAN=1` also keeps the
chip projections in host memory, within 22 GiB of GPU memory and about 32 GB of host RAM to
load). Set `LX_REPO` to the model repository. `deploy.sh` copies the `lx` package
from the code repository next to `app.py`.
