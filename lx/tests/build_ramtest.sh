#!/bin/bash
# Build tests/ramtest.S for the full machine (lx.core.Config): it tests every RAM word above the
# ROM (text_end) up to the top of RAM. Output: build/ramtest/ramtest.{elf,bin}.
set -e
R=$(cd "$(dirname "$0")/../.." && pwd)
O=$R/build/ramtest
mkdir -p "$O"
LO=$(cd "$R/lx" && python3 -c "from lx.core import Config; print(hex(0x80000000 + Config().text_end))")
HI=$(cd "$R/lx" && python3 -c "from lx.core import Config; print(hex(0x80000000 + Config().ram_size))")
riscv64-unknown-elf-gcc -march=rv32ima_zicsr -mabi=ilp32 -nostdlib -nostartfiles -Ttext=0x80000000 \
    -DRAM_LO="$LO" -DRAM_HI="$HI" "$R/lx/tests/ramtest.S" -o "$O/ramtest.elf"
riscv64-unknown-elf-objcopy -O binary "$O/ramtest.elf" "$O/ramtest.bin"
echo "ramtest: $LO .. $HI, $(stat -c %s "$O/ramtest.bin") bytes"
