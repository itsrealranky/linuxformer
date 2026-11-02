#!/bin/bash
# Build the kernel Image with the initramfs from userland/busybox.
set -euo pipefail
K=$(cd "$(dirname "$0")" && pwd)
BBD=${BBD:-$K/../userland/busybox-1.36.1}
export PATH=$K/../mini-rv32ima/buildroot/output/host/bin:$PATH
# a reproducible build: the banner's user, host, time and build number are fixed, and with the
# time set, gen_initramfs stamps every initramfs entry with it
export KBUILD_BUILD_USER=linuxformer KBUILD_BUILD_HOST=localhost KBUILD_BUILD_VERSION=8 \
       KBUILD_BUILD_TIMESTAMP="Sat Oct 31 19:54:04 UTC 2026"
"$K/mklist.sh" "$BBD/busybox" "$BBD/busybox.links" > "$K/initramfs.list"
"$K/mkconfig.sh" "$K/linux-6.8-rc1" "$K/initramfs.list"
make -C "$K/linux-6.8-rc1" ARCH=riscv CROSS_COMPILE=riscv32-buildroot-linux-uclibc- -j32 Image > "$K/build.log" 2>&1
cp "$K/linux-6.8-rc1/arch/riscv/boot/Image" "$K/Image"
sha256sum "$K/Image"
