#!/bin/bash
# Phase 1.1: instructions from reset to a shell prompt, before and after shrinking the boot.
# Every count is mini-rv32ima.h's own execution (ref/rvtrace: one instruction per step,
# time = cycle / divisor). Writes results/boot_counts.jsonl.
set -euo pipefail
cd "$(dirname "$0")"
R=../ref/rvtrace
OUT=results/boot_counts.jsonl
mkdir -p results
: > $OUT

run() {  # name ram divisor image cmdline-or-- stop [script trigger response]
    local name=$1 ram=$2 t=$3 img=$4 cmd=$5 stop=$6; shift 6
    local args=(-f "$img" -m "$ram" -t "$t" -c 400000000 -x "$stop" -o "results/$name.console.txt")
    [ "$cmd" != "--" ] && args+=(-k "$cmd")
    [ $# -ge 2 ] && args+=(-s "$1" "$2")
    local s; s=$($R "${args[@]}")
    local instr; instr=$(echo "$s" | awk '/^instructions/{print $2}')
    local trig; trig=$(echo "$s" | awk '/^first-trigger/{print $4}')
    local tok; tok=$(echo "$s" | awk '/^token model/{print $3}')
    local ok=false; tail -c 64 "results/$name.console.txt" | grep -q -- "$stop" && ok=true
    printf '{"name":"%s","image":"%s","ram":"%s","time_divisor":%s,"cmdline":"%s","stop":"%s","reached":%s,"instructions":%s,"first_trigger_instr":%s,"model_tokens":%s}\n' \
        "$name" "$img" "$ram" "$t" "$cmd" "$stop" "$ok" "$instr" "${trig:-0}" "${tok:-0}" | tee -a $OUT
}

PRE=../mini-rv32ima/mini-rv32ima/DownloadedImage           # mini-rv32ima's prebuilt Linux 6.1.14
BR=../mini-rv32ima/buildroot/output/images/Image           # mini-rv32ima's buildroot config, 6.8-rc1
NEW=../kernel/Image                                        # kernel/build.sh (trimmed)

run prebuilt-64M 0x4000000 1 $PRE -- "# " "login: " $'root\n'
run buildroot-64M 0x4000000 1 $BR "console=ttyS0 rdinit=/bin/sh" "~ # "
run buildroot-8M 0x800000 1 $BR "console=ttyS0 rdinit=/bin/sh" "~ # "
run trimmed-4.5M-t1 0x480000 1 $NEW "console=ttyS0 rdinit=/bin/sh" "/ # "
run trimmed-4.5M-t64 0x480000 64 $NEW "console=ttyS0 rdinit=/bin/sh" "/ # "
