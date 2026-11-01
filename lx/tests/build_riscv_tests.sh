#!/bin/bash
# Build riscv-tests (p environment) as flat binaries at 0x80000000, with their tohost address
# and text end, into build/rvtests/<suite>-<name>.{bin,tohost,textend}.
set -e
R=$(cd "$(dirname "$0")/../.." && pwd)
T=$R/riscv-tests
O=$R/build/rvtests
mkdir -p "$O"
CC=riscv64-unknown-elf-gcc
CFLAGS="-march=rv32ima_zicsr_zifencei -mabi=ilp32 -static -mcmodel=medany -fvisibility=hidden -nostdlib -nostartfiles"
for suite in rv32ui rv32um rv32ua rv32mi; do
    for src in "$T"/isa/$suite/*.S; do
        n=$(basename "$src" .S)
        out=$O/$suite-$n
        $CC $CFLAGS -I"$T/env/p" -I"$T/isa/macros/scalar" -T"$T/env/p/link.ld" "$src" -o "$out.elf" 2>/dev/null || { echo "skip $suite-$n"; continue; }
        riscv64-unknown-elf-objcopy -O binary "$out.elf" "$out.bin"
        riscv64-unknown-elf-nm "$out.elf" | awk '$3 == "tohost" {print "0x" $1}' > "$out.tohost"
        riscv64-unknown-elf-objdump -h "$out.elf" | awk '$2 == ".text.init" {printf "0x%x\n", strtonum("0x" $4) + strtonum("0x" $3) - 0x80000000}' > "$out.textend"
    done
done
ls "$O"/*.bin | wc -l
