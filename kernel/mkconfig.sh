#!/bin/bash
# Trimmed no-MMU RV32IMA kernel config for the in-weights machine.
# Starts from mini-rv32ima's buildroot kernel config (config.orig) and removes everything a
# serial-console busybox shell does not need. Usage: mkconfig.sh <kernel-src> <initramfs-list>
set -euo pipefail
SRC=$1
LIST=$2
HERE=$(cd "$(dirname "$0")" && pwd)
cp "$HERE/config.orig" "$SRC/.config"
cd "$SRC"
C=scripts/config
# no runtime code patching (kernel text is read-only after load)
$C -d JUMP_LABEL -e RISCV_SKIP_UNALIGNED_PROBE -e RISCV_KEEP_INIT_TEXT -d RISCV_EFFICIENT_UNALIGNED_ACCESS
# fewer instructions per call / smaller text
$C -d STACKPROTECTOR -d STACKPROTECTOR_STRONG -d FRAME_POINTER -d DEBUG_KERNEL \
   -d DEBUG_BUGVERBOSE -d SYMBOLIC_ERRNAME -d KALLSYMS
# printk: keep messages, drop timestamps and shrink the ring buffer
$C -d PRINTK_TIME --set-val LOG_BUF_SHIFT 12
# the boot-time BLAKE2s self-test runs unless CRYPTO_MANAGER_DISABLE_TESTS is set (needs CRYPTO)
$C -e CRYPTO -e CRYPTO_MANAGER_DISABLE_TESTS
# devices: only the 8250 UART, CLINT timer, syscon poweroff
$C -d VT -d VT_CONSOLE -d INPUT -d HVC_DRIVER -d HVC_RISCV_MINIRV32 -d DUMMY_CONSOLE \
   -d VIRTIO_MENU -d VIRTIO -d VIRTIO_MMIO -d SIFIVE_PLIC -d SERIAL_8250_16550A_VARIANTS \
   -d UNIX98_PTYS -d LEGACY_TIOCSTI -d DEVTMPFS -d DEVTMPFS_MOUNT
# syscalls / features a single-threaded shell does not use
$C -d FUTEX -d FUTEX_PI -d RSEQ -d MEMBARRIER -d CACHESTAT_SYSCALL -d ELFCORE -d COREDUMP \
   -d BINFMT_ELF_FDPIC -d FILE_LOCKING -d SYSCTL -d PROC_SYSCTL -d SYSCTL_EXCEPTION_TRACE \
   -d SYSFS -d EPOLL -d SIGNALFD -d TIMERFD -d EVENTFD -d AIO -d IO_URING -d ADVISE_SYSCALLS \
   -d KCMP -d CRC32_SLICEBY8 -e CRC32_SARWATE
# as in the build of kernel/Image (no plugin is selected)
$C -e GCC_PLUGINS
# initramfs from a gen_init_cpio list (no root needed for /dev/console)
$C --set-str INITRAMFS_SOURCE "$(realpath --relative-to="$SRC" "$LIST")"
$C --set-str CMDLINE "console=ttyS0 rdinit=/bin/sh"
make ARCH=riscv CROSS_COMPILE=${CROSS_COMPILE:-riscv32-buildroot-linux-uclibc-} olddefconfig >/dev/null
