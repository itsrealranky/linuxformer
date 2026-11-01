# UART input: poll the LSR, read a byte from the RBR, echo it; 'x' drops the next byte (a load
# into x0, which still takes it from the FIFO); a newline powers off.
#
#   riscv64-unknown-elf-gcc -march=rv32ima_zicsr -mabi=ilp32 -nostdlib -nostartfiles \
#       -Ttext=0x80000000 uarttest.S -o uarttest.elf

    .section .text
    .globl _start
_start:
    li   a4, 0x10000000
poll:
    lbu  t0, 5(a4)          # LSR: bit 0 = a byte is waiting
    andi t0, t0, 1
    beqz t0, poll
    lb   t1, 0(a4)          # RBR (MMIO loads are not sign-extended)
    li   t2, 'x'
    bne  t1, t2, echo
    lw   x0, 0(a4)          # drop the next byte (if there is one yet)
    j    poll
echo:
    sb   t1, 0(a4)
    li   t2, 10
    bne  t1, t2, poll
    li   a4, 0x11100000
    li   t0, 0x5555
    sw   t0, 0(a4)
1:
    j    1b
