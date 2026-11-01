// lxtrace: the LX token trace (lx/spec.py) of a run of the unmodified mini-rv32ima.h core.
//
// The platform is rvtrace.c's (one instruction per MiniRV32IMAStep call, timer = cycle / 64,
// the WFI fast-forward, the default DTB with the RAM size and cmdline patched in), except that
// a WFI instruction takes one cycle: rvtrace, like the demo wrapper, adds a second one when the
// step returns 1. Every record is read off the state the core leaves behind (registers, CSRs,
// the program counter, the old word of a store, the trap cause from MINIRV32_POSTEXEC), so it
// does not depend on a second implementation of the ISA; lx/ref.py must produce the same
// tokens. Cases the LX circuit does not implement stop the run with "UNSUPPORTED".
//
//   lxtrace -f IMAGE [-m RAM] [-k CMDLINE] [-T TEXT_END] [-o TOKENS] [-S STATE] [-c MAX]
//           [-x STOP_STRING] [-H TOHOST] [-l CONSOLE_LOG]

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lxtokens.h"

static uint32_t ram_amt = 64 * 1024 * 1024;
static uint8_t *ram_image;
static struct MiniRV32IMAState *core;

static void unsup(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    printf("UNSUPPORTED: ");
    vprintf(fmt, ap);
    printf("\n");
    va_end(ap);
    exit(2);
}

// --- hooks: the last store of a step (byte offset, the word before it) and its trap ----------
static uint32_t st_ofs, st_old;
static int st_n;
static uint32_t trap_code;
static inline void st_log(uint32_t ofs) {
    st_n++;
    st_ofs = ofs;
    st_old = *(uint32_t *)(ram_image + (ofs & ~3u));
}

#define MINIRV32_CUSTOM_MEMORY_BUS
#define MINIRV32_STORE4(ofs, val) (st_log(ofs), *(uint32_t *)(image + ofs) = val)
#define MINIRV32_STORE2(ofs, val) (st_log(ofs), *(uint16_t *)(image + ofs) = val)
#define MINIRV32_STORE1(ofs, val) (st_log(ofs), *(uint8_t *)(image + ofs) = val)
#define MINIRV32_LOAD4(ofs) (*(uint32_t *)(image + ofs))
#define MINIRV32_LOAD2(ofs) (*(uint16_t *)(image + ofs))
#define MINIRV32_LOAD1(ofs) (*(uint8_t *)(image + ofs))
#define MINIRV32_LOAD2_SIGNED(ofs) (*(int16_t *)(image + ofs))
#define MINIRV32_LOAD1_SIGNED(ofs) (*(int8_t *)(image + ofs))

static uint32_t HandleControlStore(uint32_t addy, uint32_t val);
static uint32_t HandleControlLoad(uint32_t addy);
static void HandleOtherCSRWrite(uint8_t *image, uint16_t csrno, uint32_t value);
static int32_t HandleOtherCSRRead(uint8_t *image, uint16_t csrno);

#define MINIRV32WARN(x...) fprintf(stderr, x);
#define MINI_RV32_RAM_SIZE ram_amt
#define MINIRV32_IMPLEMENTATION
#define MINIRV32_POSTEXEC(pc, ir, retval) { if (retval > 0) trap_code = retval; }
#define MINIRV32_HANDLE_MEM_STORE_CONTROL(addy, val) if (HandleControlStore(addy, val)) return val;
#define MINIRV32_HANDLE_MEM_LOAD_CONTROL(addy, rval) rval = HandleControlLoad(addy);
#define MINIRV32_OTHERCSR_WRITE(csrno, value) HandleOtherCSRWrite(image, csrno, value);
#define MINIRV32_OTHERCSR_READ(csrno, value) value = HandleOtherCSRRead(image, csrno);

#include "../mini-rv32ima/mini-rv32ima/mini-rv32ima.h"
#include "../mini-rv32ima/mini-rv32ima/default64mbdtc.h"

// --- platform -----------------------------------------------------------------------------------
#define OUTBUF 4096
static char outtail[OUTBUF];
static int outlen;
static FILE *console_log;
static const char *stop_after;
static int stopped;

static void out_byte(int c) {
    if (console_log) fputc(c, console_log);
    memmove(outtail, outtail + 1, OUTBUF - 2);
    outtail[OUTBUF - 2] = (char)c;
    if (outlen < OUTBUF - 1) outlen++;
}

static int ends_with(const char *s) {
    size_t n = strlen(s);
    return n <= (size_t)outlen && memcmp(outtail + OUTBUF - 1 - n, s, n) == 0;
}

static uint32_t HandleControlStore(uint32_t addy, uint32_t val) {
    if (addy == 0x10000000) out_byte(val & 0xff);
    else if (addy == 0x11004004) core->timermatchh = val;
    else if (addy == 0x11004000) core->timermatchl = val;
    else if (addy == 0x11100000) { core->pc = core->pc + 4; return val; }
    return 0;
}

static uint32_t HandleControlLoad(uint32_t addy) {
    if (addy == 0x10000005) return 0x60;  // no UART input (kbhit = 0)
    else if (addy == 0x1100bffc) return core->timerh;
    else if (addy == 0x1100bff8) return core->timerl;
    return 0;
}

static void HandleOtherCSRWrite(uint8_t *image, uint16_t csrno, uint32_t value) {
    (void)image; (void)csrno; (void)value;
}

static int32_t HandleOtherCSRRead(uint8_t *image, uint16_t csrno) {
    (void)image;
    if (csrno == 0x140) unsup("read of CSR 0x140");
    return 0;
}

// --- tokens ------------------------------------------------------------------------------------
static FILE *tok_out;
static uint64_t n_tokens, n_records;
static uint32_t anchor, nn, last_T, text_end;

static void emit(uint32_t t) {
    if (tok_out) fwrite(&t, 4, 1, tok_out);
    n_tokens++;
}

static void jump_to(uint32_t w, int r) {
    if ((w >> 16) != (anchor >> 16)) emit(T_JH(w >> 16));
    emit(r ? T_R(w & 0xffff) : T_J(w & 0xffff));
    anchor = w, nn = 0;
}

static uint32_t word_of(uint32_t addr, const char *what) {
    uint32_t o = addr - MINIRV32_RAM_IMAGE_OFFSET;
    if ((addr & 3) || o >= ram_amt) unsup("%s target %08x", what, addr);
    return o >> 2;
}

static void value(uint32_t v) {
    if (v < 0x8000 || v >= 0xffff8000u) emit(T_F(v & 0xffff));
    else emit(T_V(v & 0xffff)), emit(T_F(v >> 16));
}

static uint32_t data_ofs(uint32_t addr, uint32_t size, uint32_t pc) {
    uint32_t o = addr - MINIRV32_RAM_IMAGE_OFFSET;
    if (o < text_end) unsup("data access to kernel text %08x at %08x", addr, pc);
    if (o & (size - 1)) unsup("misaligned %u-byte access %08x at %08x", size, addr, pc);
    return o;
}

static int nibbles_used(uint32_t v) {
    int n = 0;
    for (int i = 0; i < 8; i++) n += (v >> (4 * i)) != 0;
    return n;
}

// QS / QD of a DIV-class instruction (as lx/ref.py div_parts)
static void div_digits(uint32_t f3, uint32_t a, uint32_t b) {
    int sgn = f3 == 4 || f3 == 6;
    int sa = sgn && (a >> 31), sb = sgn && (b >> 31);
    uint32_t ua = sa ? -a : a, ub = sb ? -b : b;
    uint32_t q = ub ? ua / ub : 0xffffffffu;
    int da = nibbles_used(a), db = nibbles_used(sb ? ~b : b);
    int k0 = b == 0 ? 0 : 7 - da + db;
    if (k0 < 0) k0 = 0;
    if (k0 > 8) k0 = 8;
    emit(T_QS(k0));
    for (int k = k0; k < 8; k++) emit(T_QD(k << 4 | ((q >> (4 * (7 - k))) & 15)));
}

static const uint16_t stored_csrs[8] = {0x300, 0x304, 0x305, 0x340, 0x341, 0x342, 0x343, 0x344};

static uint32_t csr_value(uint32_t csrno) {
    switch (csrno) {
    case 0x300: return core->mstatus;
    case 0x304: return core->mie;
    case 0x305: return core->mtvec;
    case 0x340: return core->mscratch;
    case 0x341: return core->mepc;
    case 0x342: return core->mcause;
    case 0x343: return core->mtval;
    default: return core->mip;
    }
}

// one executed instruction's record; pre: the registers and pc before the step
static void instruction_record(uint32_t ir, uint32_t pc, const uint32_t *regs, int ret) {
    uint32_t opc = ir & 0x7f, rd = (ir >> 7) & 31, f3 = (ir >> 12) & 7;
    uint32_t r1i = (ir >> 15) & 31, a = regs[r1i], b = regs[(ir >> 20) & 31];
    int32_t imm_i = (int32_t)ir >> 20;
    int jumped = 0;
    if (trap_code) {
        uint32_t cause = trap_code - 1;
        if (core->mcause != cause) unsup("trap bookkeeping");
        emit(T_TRAP(cause));
        emit(T_D(core->mstatus | 3u << 13));
        jump_to(word_of(core->pc, "trap"), 1);
        return;
    }
    switch (opc) {
    case 0x37: case 0x17: case 0x13:
        if (rd) value(core->regs[rd]); else emit(T_NT);
        break;
    case 0x33:
        if ((ir >> 25 & 1) && f3 >= 4) {
            if (rd) div_digits(f3, a, b), value(core->regs[rd]); else emit(T_NT);
        } else if (rd) value(core->regs[rd]); else emit(T_NT);
        break;
    case 0x6f: case 0x67:
        jump_to(word_of(core->pc, "jump"), 0);
        jumped = 1;
        break;
    case 0x63: {
        int32_t s1 = (int32_t)a, s2 = (int32_t)b;
        int t = f3 == 0 ? a == b : f3 == 1 ? a != b : f3 == 4 ? s1 < s2 : f3 == 5 ? s1 >= s2
              : f3 == 6 ? a < b : a >= b;
        if (t) jump_to(word_of(core->pc, "branch"), 0), jumped = 1;
        else emit(T_NT);
        break;
    }
    case 0x03: {
        uint32_t addr = a + imm_i, o = addr - MINIRV32_RAM_IMAGE_OFFSET;
        if (o < ram_amt - 3) data_ofs(addr, f3 == 2 ? 4 : (f3 & 3) == 1 ? 2 : 1, pc);
        if (rd) value(core->regs[rd]); else emit(T_NT);
        break;
    }
    case 0x23: {
        uint32_t addr = a + (uint32_t)((((int32_t)ir >> 25) << 5) | ((ir >> 7) & 31));
        uint32_t o = addr - MINIRV32_RAM_IMAGE_OFFSET;
        if (o >= ram_amt - 3) {  // MMIO (SYSCON stops the run before it gets here)
            if (addr == 0x10000000) emit(T_BYTE(b & 0xff));
            else if (addr == 0x11004004) emit(T_MTH);
            else if (addr == 0x11004000) emit(T_MTL);
            else emit(T_NT);
            break;
        }
        data_ofs(addr, f3 == 2 ? 4 : f3 == 1 ? 2 : 1, pc);
        if (st_n != 1 || st_ofs != o) unsup("store bookkeeping at %08x", pc);
        if (f3 != 2) emit(T_D(st_old & 0xffff)), emit(T_D(st_old >> 16));
        emit(T_ST0 + ((o >> 2) & 7));
        break;
    }
    case 0x0f:
        emit(T_NT);
        break;
    case 0x73: {
        uint32_t csrno = ir >> 20;
        if (f3 & 3) {
            if (csrno == 0xC00) unsup("read of the cycle CSR at %08x", pc);
            int stored = 0;
            for (int i = 0; i < 8; i++) stored |= stored_csrs[i] == csrno;
            if (stored && ((f3 & 3) == 1 || r1i)) {
                uint32_t v = csr_value(csrno);
                emit(T_D(v & 0xffff)), emit(T_D(v >> 16));
            }
            if (rd) value(core->regs[rd]); else emit(T_NT);
        } else if ((csrno & 0xff) == 0x02) {  // MRET
            emit(T_D(core->mstatus | (core->extraflags & 3) << 13));
            jump_to(word_of(core->pc, "mret"), 1);
            jumped = 1;
        } else if (csrno == 0x105 && ret == 1) {  // WFI
            if (core->mstatus >> 16) unsup("WFI with mstatus >= 2^16");
            emit(T_D(core->mstatus));
            emit(T_WFI);
        } else unsup("system op %08x at %08x", ir, pc);
        break;
    }
    case 0x2f: {
        uint32_t f5 = ir >> 27, o = data_ofs(a, 4, pc);
        if (f5 == 2) {
            if (rd) value(core->regs[rd]); else emit(T_NT);
        } else if (f5 == 3) {
            if (st_n) emit(T_ST0 + ((o >> 2) & 7));
            else if (rd) value(core->regs[rd]); else emit(T_NT);
        } else {
            if (st_n != 1 || st_ofs != o) unsup("amo bookkeeping at %08x", pc);
            emit(T_D(st_old & 0xffff)), emit(T_D(st_old >> 16));
            emit(T_ST0 + ((o >> 2) & 7));
        }
        break;
    }
    default:
        unsup("opcode %08x at %08x", ir, pc);
    }
    if (!jumped) nn++;
}

static void tick(uint32_t t) {
    if ((t >> 16) != (last_T >> 16)) emit(T_D(t >> 16));
    emit(T_TK(t & 0xffff));
    last_T = t;
    n_records++;
}

int main(int argc, char **argv) {
    const char *image_file = 0, *cmdline = 0, *tok_file = 0, *state_file = 0, *log_file = 0;
    uint64_t max_records = ~0ull, time_divisor = 64;
    uint32_t tohost = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-f")) image_file = argv[++i];
        else if (!strcmp(a, "-k")) cmdline = argv[++i];
        else if (!strcmp(a, "-m")) ram_amt = (uint32_t)strtoul(argv[++i], 0, 0);
        else if (!strcmp(a, "-T")) text_end = (uint32_t)strtoul(argv[++i], 0, 0);
        else if (!strcmp(a, "-o")) tok_file = argv[++i];
        else if (!strcmp(a, "-S")) state_file = argv[++i];
        else if (!strcmp(a, "-c")) max_records = strtoull(argv[++i], 0, 0);
        else if (!strcmp(a, "-x")) stop_after = argv[++i];
        else if (!strcmp(a, "-H")) tohost = (uint32_t)strtoul(argv[++i], 0, 0);
        else if (!strcmp(a, "-l")) log_file = argv[++i];
        else { fprintf(stderr, "bad arg %s\n", a); return 1; }
    }
    if (log_file) console_log = fopen(log_file, "wb");
    if (tok_file) tok_out = fopen(tok_file, "wb");
    ram_image = calloc(ram_amt, 1);
    FILE *f = fopen(image_file, "rb");
    if (!f) { fprintf(stderr, "no image\n"); return 1; }
    fseek(f, 0, SEEK_END);
    long flen = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fread(ram_image, flen, 1, f) != 1) return 1;
    fclose(f);
    int dtb_ptr = ram_amt - sizeof(default64mbdtb) - sizeof(struct MiniRV32IMAState);
    memcpy(ram_image + dtb_ptr, default64mbdtb, sizeof(default64mbdtb));
    if (cmdline) strncpy((char *)(ram_image + dtb_ptr + 0xc0), cmdline, 54);
    uint32_t *dtb = (uint32_t *)(ram_image + dtb_ptr);
    if (dtb[0x13c / 4] == 0x00c0ff03) {
        uint32_t v = dtb_ptr;
        dtb[0x13c / 4] = (v >> 24) | (((v >> 16) & 0xff) << 8) | (((v >> 8) & 0xff) << 16) | ((v & 0xff) << 24);
    }
    core = (struct MiniRV32IMAState *)(ram_image + ram_amt - sizeof(struct MiniRV32IMAState));
    core->pc = MINIRV32_RAM_IMAGE_OFFSET;
    core->regs[10] = 0;
    core->regs[11] = dtb_ptr + MINIRV32_RAM_IMAGE_OFFSET;
    core->extraflags |= 3;

    uint64_t last_time = 0, instr = 0, interrupts = 0, halt = 0;
    uint64_t *ccount = (uint64_t *)&core->cyclel;
    while (n_records < max_records && !stopped) {
        if (core->extraflags & 4) {
            uint64_t match = ((uint64_t)core->timermatchh << 32) | core->timermatchl;
            if (!match || core->timermatchh) unsup("WFI with mtimecmp unarmed or >= 2^32");
            uint64_t wake = (match + 1) * time_divisor;
            if (wake > *ccount + 1) {
                *ccount = wake - 1;
                uint64_t t = *ccount / time_divisor;
                uint64_t timer = ((uint64_t)core->timerh << 32) | core->timerl;
                timer += t - last_time;
                last_time = t;
                core->timerl = (uint32_t)timer;
                core->timerh = (uint32_t)(timer >> 32);
            }
        }
        uint32_t elapsed = (uint32_t)(*ccount / time_divisor - last_time);
        last_time += elapsed;
        uint32_t pc = core->pc, regs[32];
        memcpy(regs, core->regs, sizeof regs);
        uint32_t pofs = pc - MINIRV32_RAM_IMAGE_OFFSET;
        if (pofs >= ram_amt - 3 || (pc & 3)) unsup("fetch at %08x", pc);
        uint32_t ir = *(uint32_t *)(ram_image + pofs);
        uint64_t c0 = *ccount;
        st_n = 0, trap_code = 0;
        int ret = MiniRV32IMAStep(core, ram_image, 0, elapsed, 1);
        int executed = *ccount != c0;
        if (ret == 1 && !executed) { *ccount += 1; continue; }  // asleep in WFI
        if (core->timerh) unsup("time >= 2^32");
        if (core->timerl != last_T) tick(core->timerl);
        if (ret != 0 && ret != 1) {  // SYSCON: the step returned the stored value
            if (ret != 0x5555 && ret != 0x7777) unsup("SYSCON %x", ret);
            if (nn == LX_N_MAX) jump_to(anchor + nn, 1), n_records++;
            if ((pofs >> 2) >= (text_end >> 2)) emit(T_I(ir & 0xffff)), emit(T_I(ir >> 16));
            emit(T_HALT);
            n_records++, instr++, halt = ret;
            break;
        }
        if (!executed) {  // the timer interrupt
            if (core->mcause != 0x80000007u) unsup("interrupt bookkeeping");
            emit(T_TRAP(LX_TRAP_INT));
            emit(T_D(core->mstatus | 3u << 13));
            jump_to(word_of(core->pc, "interrupt"), 1);
            n_records++, interrupts++;
            continue;
        }
        if (nn == LX_N_MAX) jump_to(anchor + nn, 1), n_records++;
        if ((pofs >> 2) != anchor + nn) unsup("pc bookkeeping at %08x", pc);
        if ((pofs >> 2) >= (text_end >> 2)) emit(T_I(ir & 0xffff)), emit(T_I(ir >> 16));
        instruction_record(ir, pc, regs, ret);
        n_records++, instr++;
        if (tohost && st_n && st_ofs + MINIRV32_RAM_IMAGE_OFFSET == tohost) {
            printf("TOHOST %u\n", *(uint32_t *)(ram_image + st_ofs));
            break;
        }
        if (stop_after && ends_with(stop_after)) stopped = 1;
    }
    printf("records %llu tokens %llu instructions %llu interrupts %llu cycles %llu time %u%s\n",
           (unsigned long long)n_records, (unsigned long long)n_tokens, (unsigned long long)instr,
           (unsigned long long)interrupts, (unsigned long long)*ccount, core->timerl,
           halt ? " HALT" : "");
    if (state_file) {
        // regs, pc, mstatus mie mtvec mscratch mepc mcause mtval mip, priv, reservation, time,
        // mtimecmp lo hi, anchor, n, then RAM
        uint32_t st[32 + 16];
        memcpy(st, core->regs, 128);
        uint32_t extra[16] = {core->pc, core->mstatus, core->mie, core->mtvec, core->mscratch,
                              core->mepc, core->mcause, core->mtval, core->mip,
                              core->extraflags & 3, core->extraflags >> 3, core->timerl,
                              core->timermatchl, core->timermatchh, anchor, nn};
        memcpy(st + 32, extra, sizeof extra);
        FILE *s = fopen(state_file, "wb");
        fwrite(st, 4, 48, s);
        fwrite(ram_image, 1, ram_amt, s);
        fclose(s);
    }
    if (tok_out) fclose(tok_out);
    if (console_log) fclose(console_log);
    return 0;
}
