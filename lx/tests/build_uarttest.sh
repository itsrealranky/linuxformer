#!/bin/bash
# Build tests/uarttest.S as a flat binary at 0x80000000: build/uarttest/uarttest.{elf,bin}.
set -e
R=$(cd "$(dirname "$0")/../.." && pwd)
O=$R/build/uarttest
mkdir -p "$O"
riscv64-unknown-elf-gcc -march=rv32ima_zicsr -mabi=ilp32 -nostdlib -nostartfiles -Ttext=0x80000000 \
    "$R/lx/tests/uarttest.S" -o "$O/uarttest.elf"
riscv64-unknown-elf-objcopy -O binary "$O/uarttest.elf" "$O/uarttest.bin"
echo "uarttest: $(stat -c %s "$O/uarttest.bin") bytes"
