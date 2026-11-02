#!/bin/bash
# Fetch every upstream piece at the version the results were produced with, and build what the
# repository does not ship:
#   mini-rv32ima (the reference core) and its buildroot (the RV32 no-MMU uClibc toolchain)
#   the kernel tree (kernel.org 6.8-rc1 + kernel/patches) and busybox (1.36.1 + userland/patches)
#   ref/lxtrace, riscv-tests, the test programs, Qwen3.5-4B, MMLU
# kernel/Image, the image every result booted, is in the repository; KERNEL=1 rebuilds it from
# these sources, bit for bit (the build scripts fix every timestamp, see NOTICE).
#
#   scripts/setup.sh    with requirements.txt's environment active; needs git, gcc, make,
#                       riscv64-unknown-elf-gcc and the hf CLI
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
cd "$R"

MINI_RV32IMA=84858f58cb41899705e2ff2d6ee3b2d5c0795bfe
BUILDROOT=851edd247f5e2e8e10f943c52218bc54c657688f
RISCV_TESTS=3df703cd52d5f0d8224e3c50bf03c6cb70efff98
QWEN=Qwen/Qwen3.5-4B
QWEN_REV=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
LINUX_SHA256=327c0c2dcbe348ab9902dc30511f5cfb239cfff8a8d285b29779c4b5bd7c8e42
BUSYBOX_SHA256=b8cc24c9574d809e7279c3be349795c5d5ceb6fdf19ca709f80cde50e47de314

fetch() {  # url dir commit
    [ -d "$2/.git" ] || git init -q "$2"
    git -C "$2" fetch -q --depth 1 "$1" "$3"
    git -C "$2" checkout -q FETCH_HEAD
}

fetch https://github.com/cnlohr/mini-rv32ima mini-rv32ima $MINI_RV32IMA
fetch https://github.com/cnlohr/buildroot mini-rv32ima/buildroot $BUILDROOT
fetch https://github.com/riscv-software-src/riscv-tests riscv-tests $RISCV_TESTS
git -C riscv-tests submodule update -q --init env

# the toolchain, as mini-rv32ima's "toolchain" target (it also builds buildroot's own kernel and
# rootfs, and downloads the kernel and busybox tarballs into buildroot/dl)
if [ ! -x mini-rv32ima/buildroot/output/host/bin/riscv32-buildroot-linux-uclibc-gcc ]; then
    (cd mini-rv32ima &&
     cp -a configs/custom_kernel_config buildroot/kernel_config &&
     cp -a configs/buildroot_config buildroot/.config &&
     cp -a configs/busybox_config buildroot/busybox_config &&
     cp -a configs/uclibc_config buildroot/uclibc_config &&
     cp -a configs/uclibc_config buildroot/uclibc_config_extra &&
     make -C buildroot)
fi
DL=mini-rv32ima/buildroot/dl
echo "$LINUX_SHA256  $DL/linux/linux-6.8-rc1.tar.gz" | sha256sum -c -
echo "$BUSYBOX_SHA256  $DL/busybox/busybox-1.36.1.tar.bz2" | sha256sum -c -

if [ ! -d kernel/linux-6.8-rc1 ]; then
    tar -C kernel -xzf $DL/linux/linux-6.8-rc1.tar.gz
    for p in kernel/patches/*.patch; do patch -d kernel/linux-6.8-rc1 -p1 -s < "$p"; done
fi
if [ ! -d userland/busybox-1.36.1 ]; then
    tar -C userland -xjf $DL/busybox/busybox-1.36.1.tar.bz2
    for p in userland/patches/*.patch; do patch -d userland/busybox-1.36.1 -p1 -s < "$p"; done
fi
userland/build_busybox.sh userland/busybox-1.36.1
cmp <(grep -v '^#' userland/busybox-1.36.1/.config) <(grep -v '^#' userland/busybox.config)
if [ "${KERNEL:-0}" = 1 ]; then
    kernel/build.sh
fi

(cd lx && PYTHONPATH=. python3 -m lx.cgen)
(cd ref && gcc -O2 -fno-strict-aliasing -Wall -Wno-unused -o lxtrace lxtrace.c &&
           gcc -O2 -fno-strict-aliasing -Wall -Wno-unused -o rvtrace rvtrace.c)
lx/tests/build_riscv_tests.sh
lx/tests/build_ramtest.sh
lx/tests/build_uarttest.sh

hf download $QWEN --revision $QWEN_REV --local-dir models/Qwen3.5-4B
hf download cais/mmlu all/test-00000-of-00001.parquet --repo-type dataset --local-dir data/mmlu
sha256sum -c scripts/SHA256SUMS
