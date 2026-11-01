"""Build lx's CUDA sources (lx/csrc/*.cu) with nvcc into shared libraries, called through ctypes
on torch's tensors and stream. Linked to torch's own CUDA runtime, so they share its context.

    lib = mk_build.load("mk_test")      # build/mk/mk_test-<hash>.so, rebuilt when a source changes
"""

import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "csrc"
OUT = Path(__file__).resolve().parents[2] / "build" / "mk"
# nvcc's own contractions are kept (as PyTorch's build has them): the exact paths write every
# rounding step as an intrinsic (__fmul_rn, __fadd_rn, __fmaf_rn), which nvcc never fuses
FLAGS = ["-O3", "-std=c++17", "-arch=sm_90a", "-lineinfo", "-shared", "-Xcompiler", "-fPIC"]


def cuda_home() -> Path:
    import torch

    pip_nvcc = Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13"  # nvidia-cuda-nvcc
    for c in (os.environ.get("LX_CUDA_HOME"), os.environ.get("CUDA_HOME"), pip_nvcc,
              "/usr/local/cuda"):
        if c and (Path(c) / "bin" / "nvcc").exists():
            return Path(c)
    raise RuntimeError("nvcc not found: set LX_CUDA_HOME to a CUDA 13 toolkit")


def cudart_dir() -> Path:
    import torch

    d = Path(torch.__file__).resolve().parents[1] / "nvidia" / "cu13" / "lib"
    if not (d / "libcudart.so.13").exists():
        raise RuntimeError(f"torch's libcudart.so.13 not found in {d}")
    return d


def load(name: str, extra: list[str] = ()) -> ctypes.CDLL:
    src = SRC / f"{name}.cu"
    h = hashlib.sha256()
    for p in sorted(SRC.glob("*.cu*")):
        h.update(p.name.encode() + p.read_bytes())
    h.update(" ".join(FLAGS + list(extra)).encode())
    so = OUT / f"{name}-{h.hexdigest()[:12]}.so"
    if not so.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        home, rt = cuda_home(), cudart_dir()
        tmp = so.with_suffix(".tmp.so")
        cmd = [str(home / "bin" / "nvcc"), *FLAGS, *extra, "-I", str(SRC), "-o", str(tmp),
               str(src), "-L", str(rt), "-l:libcudart.so.13", "-Xlinker", f"-rpath={rt}"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"nvcc failed:\n{' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
        tmp.rename(so)
    return ctypes.CDLL(str(so))
