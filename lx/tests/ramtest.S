# Every RAM word from RAM_LO to RAM_HI is written with a value of its own address, then all are
# read back and compared. Prints "RAM OK\n" or "RAM BAD\n" on the UART and powers off.
#
#   riscv64-unknown-elf-gcc -march=rv32ima_zicsr -mabi=ilp32 -nostdlib -nostartfiles \
#       -Ttext=0x80000000 -DRAM_LO=... -DRAM_HI=... ramtest.S -o ramtest.elf
#
# value(A) = A + C: unique per word, so a word read from the wrong place, or a lost write,
# differs. Unrolled 8 words per loop; the check subtracts the address and compares with C + k.

#define C 0x5A5A1234

    .section .text
    .globl _start
_start:
    li   a0, RAM_LO
    li   a1, RAM_HI
    li   s10, RAM_LO + C
    mv   a2, a0
1:
    .irp k, 0, 4, 8, 12, 16, 20, 24, 28
    addi t0, s10, \k
    sw   t0, \k(a2)
    .endr
    addi a2, a2, 32
    addi s10, s10, 32
    bltu a2, a1, 1b

    li   s2, C
    addi s3, s2, 4
    addi s4, s2, 8
    addi s5, s2, 12
    addi s6, s2, 16
    addi s7, s2, 20
    addi s8, s2, 24
    addi s9, s2, 28
    mv   a2, a0
2:
    .irp r, 2, 3, 4, 5, 6, 7, 8, 9
    lw   t0, (4 * (\r - 2))(a2)
    sub  t0, t0, a2
    bne  t0, s\r, fail
    .endr
    addi a2, a2, 32
    bltu a2, a1, 2b

    li   a4, 0x10000000
    .irp ch, 82, 65, 77, 32, 79, 75, 10          # "RAM OK\n" (code below the ROM end is
    li   t0, \ch                                 # not readable as data: no strings)
    sb   t0, 0(a4)
    .endr
    j    done
fail:
    li   a4, 0x10000000
    .irp ch, 82, 65, 77, 32, 66, 65, 68, 10      # "RAM BAD\n"
    li   t0, \ch
    sb   t0, 0(a4)
    .endr
done:
    li   a4, 0x11100000
    li   t0, 0x5555
    sw   t0, 0(a4)
4:
    j    4b
