// Deterministic reference harness around the unmodified mini-rv32ima.h core.
//
// Differences from mini-rv32ima.c (the demo wrapper), all of which only affect the platform:
//   * one instruction per MiniRV32IMAStep call (the demo runs 1024), so the timer and the timer
//     interrupt are evaluated before every instruction;
//   * time is locked to the cycle counter (the demo's -l), timer = cycle / time_divisor;
//   * UART input comes from a script (trigger string -> bytes), not from the terminal;
//   * WFI idles cycle by cycle exactly like the demo with a 1-instruction flip.
// The core semantics are mini-rv32ima.h's own. Memory macros only add logging.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t ram_amt = 64 * 1024 * 1024;
static uint8_t *ram_image;
static struct MiniRV32IMAState *core;

// --- logging hooks ---------------------------------------------------------------------------
static uint64_t n_store1, n_store2, n_store4, n_load;
static uint8_t *written;  // per word: written since reset
static uint8_t *fetched;  // per word: fetched
static uint8_t *code_written;
static uint32_t *pc_hist;  // per word: executions
static uint32_t caller_pc[8]; static int n_caller; static uint32_t *ra_hist;  // per word: written then fetched later (runtime code)

static inline void mark_write(uint32_t ofs) { written[ofs >> 2] = 1; if (fetched[ofs >> 2]) code_written[ofs >> 2] |= 2; }

#define MINIRV32_CUSTOM_MEMORY_BUS
static uint64_t n_misaligned;
static void log_misaligned(uint32_t pc, uint32_t ofs, int m) {
    if (n_misaligned++ < 64 && getenv("RVMIS"))
        fprintf(stderr, "misaligned %d-byte access at pc %08x addr %08x\n", m + 1, pc, ofs + 0x80000000);
}
#define MIS(ofs, m) (((ofs) & (m)) ? log_misaligned(pc, ofs, m) : (void)0)
// data accesses below -T (the kernel's static text); an instruction fetch is the LOAD4 at pc
static uint32_t text_hi = 0;
static uint64_t n_text_load, n_text_store;
static uint32_t first_text_load_pc, first_text_load_addr, first_text_store_pc, first_text_store_addr;
static inline void text_ld(uint32_t pc, uint32_t ofs) {
    if (ofs < text_hi && ofs != pc - 0x80000000u) {
        if (!n_text_load++) first_text_load_pc = pc, first_text_load_addr = ofs;
    }
}
static inline void text_st(uint32_t pc, uint32_t ofs) {
    if (ofs < text_hi) { if (!n_text_store++) first_text_store_pc = pc, first_text_store_addr = ofs; }
}
#define MINIRV32_STORE4(ofs, val) (n_store4++, MIS(ofs, 3), text_st(pc, ofs), mark_write(ofs), mark_write((ofs) + 3), *(uint32_t *)(image + ofs) = val)
#define MINIRV32_STORE2(ofs, val) (n_store2++, MIS(ofs, 1), text_st(pc, ofs), mark_write(ofs), mark_write((ofs) + 1), *(uint16_t *)(image + ofs) = val)
#define MINIRV32_STORE1(ofs, val) (n_store1++, text_st(pc, ofs), mark_write(ofs), *(uint8_t *)(image + ofs) = val)
#define MINIRV32_LOAD4(ofs) (n_load++, MIS(ofs, 3), text_ld(pc, ofs), *(uint32_t *)(image + ofs))
#define MINIRV32_LOAD2(ofs) (n_load++, MIS(ofs, 1), text_ld(pc, ofs), *(uint16_t *)(image + ofs))
#define MINIRV32_LOAD1(ofs) (n_load++, text_ld(pc, ofs), *(uint8_t *)(image + ofs))
#define MINIRV32_LOAD2_SIGNED(ofs) (n_load++, MIS(ofs, 1), text_ld(pc, ofs), *(int16_t *)(image + ofs))
#define MINIRV32_LOAD1_SIGNED(ofs) (n_load++, text_ld(pc, ofs), *(int8_t *)(image + ofs))

static uint32_t HandleControlStore(uint32_t addy, uint32_t val);
static uint32_t HandleControlLoad(uint32_t addy);
static void HandleOtherCSRWrite(uint8_t *image, uint16_t csrno, uint32_t value);
static int32_t HandleOtherCSRRead(uint8_t *image, uint16_t csrno);

static int fail_on_all_faults = 0;
static uint64_t n_traps;
#define MINIRV32WARN(x...) fprintf(stderr, x);
#define MINI_RV32_RAM_SIZE ram_amt
#define MINIRV32_IMPLEMENTATION
#define MINIRV32_POSTEXEC(pc, ir, retval) { if (retval > 0) { n_traps++; if (fail_on_all_faults) { printf("FAULT\n"); return 3; } } }
#define MINIRV32_HANDLE_MEM_STORE_CONTROL(addy, val) if (HandleControlStore(addy, val)) return val;
#define MINIRV32_HANDLE_MEM_LOAD_CONTROL(addy, rval) rval = HandleControlLoad(addy);
#define MINIRV32_OTHERCSR_WRITE(csrno, value) HandleOtherCSRWrite(image, csrno, value);
#define MINIRV32_OTHERCSR_READ(csrno, value) value = HandleOtherCSRRead(image, csrno);

#include "../mini-rv32ima/mini-rv32ima/mini-rv32ima.h"
#include "../mini-rv32ima/mini-rv32ima/default64mbdtc.h"

// --- UART input script -------------------------------------------------------------------------
#define OUTBUF 4096
static char outtail[OUTBUF];  // last bytes printed
static int outlen;
static FILE *console_log;
static uint64_t n_out_bytes;
static const char *triggers[16], *responses[16];
static int n_script, script_pos;
static const char *pending;  // bytes currently available to the guest
static uint64_t prompt_cycle, prompt_instr;  // when the first trigger was seen
static const char *stop_after;  // stop once this string is printed after the script finished
static int stopped;

static void out_byte(int c) {
    n_out_bytes++;
    if (console_log) fputc(c, console_log);
    memmove(outtail, outtail + 1, OUTBUF - 2);
    outtail[OUTBUF - 2] = (char)c;
    outtail[OUTBUF - 1] = 0;
    if (outlen < OUTBUF - 1) outlen++;
}

static int ends_with(const char *s) {
    size_t n = strlen(s);
    return n <= (size_t)outlen && memcmp(outtail + OUTBUF - 1 - n, s, n) == 0;
}

static int kbhit(void) { return pending && *pending; }

static uint32_t HandleControlStore(uint32_t addy, uint32_t val) {
    if (addy == 0x10000000) out_byte(val & 0xff);
    else if (addy == 0x11004004) core->timermatchh = val;
    else if (addy == 0x11004000) core->timermatchl = val;
    else if (addy == 0x11100000) { core->pc = core->pc + 4; return val; }
    return 0;
}

static uint32_t HandleControlLoad(uint32_t addy) {
    if (addy == 0x10000005) return 0x60 | kbhit();
    else if (addy == 0x10000000 && kbhit()) return (uint8_t)*pending++;
    else if (addy == 0x1100bffc) return core->timerh;
    else if (addy == 0x1100bff8) return core->timerl;
    return 0;
}

static void HandleOtherCSRWrite(uint8_t *image, uint16_t csrno, uint32_t value) {
    (void)image; (void)csrno; (void)value;
}

static int32_t HandleOtherCSRRead(uint8_t *image, uint16_t csrno) {
    (void)image;
    if (csrno == 0x140) { if (!kbhit()) return -1; return (uint8_t)*pending++; }
    return 0;
}

// --- statistics ---------------------------------------------------------------------------------
static uint64_t op_count[128];
static uint64_t rd_writes, rd_fit[33];  // rd_fit[b]: values that fit in b-bit sign-extended
static uint64_t ptr_vals;  // values in [RAM base, RAM end)
static uint64_t taken, jal, jalr, branches, wfi_idle, interrupts;
static uint64_t near_jump[16];  // taken jumps whose target is within 2^k words of the pc

static uint64_t instr_from_written, instr_above_text, page_cross[4], jump_to_ram;
static uint64_t csr_count[4096][8], amo_count[32], mmio_n[64];
static uint32_t mmio_addr[64], mmio_kind[64]; static int n_mmio;
static uint64_t tokens, tok_value, tok_jump, tok_plain, tok_refresh, tok_subword, tok_div, tok_trap,
    tok_amo, tok_wfi, run_len;

static int fits_signed(uint32_t v, int b) {
    if (b >= 32) return 1;
    int32_t s = (int32_t)v;
    return s >= -(1 << (b - 1)) && s < (1 << (b - 1));
}

int main(int argc, char **argv) {
    const char *image_file = 0, *dtb_file = 0, *cmdline = 0, *log_file = 0;
    uint64_t max_instr = 2000000000ull, time_divisor = 1;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-f")) image_file = argv[++i];
        else if (!strcmp(a, "-b")) dtb_file = argv[++i];
        else if (!strcmp(a, "-k")) cmdline = argv[++i];
        else if (!strcmp(a, "-m")) ram_amt = (uint32_t)strtoul(argv[++i], 0, 0);
        else if (!strcmp(a, "-t")) time_divisor = strtoull(argv[++i], 0, 0);
        else if (!strcmp(a, "-c")) max_instr = strtoull(argv[++i], 0, 0);
        else if (!strcmp(a, "-o")) log_file = argv[++i];
        else if (!strcmp(a, "-x")) stop_after = argv[++i];
        else if (!strcmp(a, "-d")) fail_on_all_faults = 1;
        else if (!strcmp(a, "-T")) text_hi = (uint32_t)strtoul(argv[++i], 0, 0);
        else if (!strcmp(a, "-s")) {  // -s TRIGGER RESPONSE (repeatable)
            triggers[n_script] = argv[++i];
            responses[n_script++] = argv[++i];
        } else { fprintf(stderr, "bad arg %s\n", a); return 1; }
    }
    if (log_file) console_log = fopen(log_file, "wb");
    ram_image = calloc(ram_amt, 1);
    written = calloc(ram_amt / 4, 1);
    fetched = calloc(ram_amt / 4, 1);
    code_written = calloc(ram_amt / 4, 1);
    pc_hist = calloc(ram_amt / 4, 4);
    ra_hist = calloc(ram_amt / 4, 4);
    if (getenv("RVCALLER")) { char *p = getenv("RVCALLER"); while (*p && n_caller < 8) { caller_pc[n_caller++] = strtoul(p, &p, 16); if (*p == ',') p++; } }
    FILE *f = fopen(image_file, "rb");
    if (!f) { fprintf(stderr, "no image\n"); return 1; }
    fseek(f, 0, SEEK_END);
    long flen = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fread(ram_image, flen, 1, f) != 1) return 1;
    fclose(f);
    int dtb_ptr;
    if (dtb_file) {
        f = fopen(dtb_file, "rb");
        fseek(f, 0, SEEK_END);
        long dl = ftell(f);
        fseek(f, 0, SEEK_SET);
        dtb_ptr = ram_amt - dl - sizeof(struct MiniRV32IMAState);
        if (fread(ram_image + dtb_ptr, dl, 1, f) != 1) return 1;
        fclose(f);
    } else {
        dtb_ptr = ram_amt - sizeof(default64mbdtb) - sizeof(struct MiniRV32IMAState);
        memcpy(ram_image + dtb_ptr, default64mbdtb, sizeof(default64mbdtb));
        if (cmdline) strncpy((char *)(ram_image + dtb_ptr + 0xc0), cmdline, 54);
    }
    core = (struct MiniRV32IMAState *)(ram_image + ram_amt - sizeof(struct MiniRV32IMAState));
    core->pc = MINIRV32_RAM_IMAGE_OFFSET;
    core->regs[10] = 0;
    core->regs[11] = dtb_ptr + MINIRV32_RAM_IMAGE_OFFSET;
    core->extraflags |= 3;
    if (!dtb_file) {
        uint32_t *dtb = (uint32_t *)(ram_image + dtb_ptr);
        if (dtb[0x13c / 4] == 0x00c0ff03) {
            uint32_t v = dtb_ptr;
            dtb[0x13c / 4] = (v >> 24) | (((v >> 16) & 0xff) << 8) | (((v >> 8) & 0xff) << 16) | ((v & 0xff) << 24);
        }
    }

    uint64_t instr = 0, last_time = 0;
    const uint32_t ram_base = MINIRV32_RAM_IMAGE_OFFSET;
    uint64_t iters = 0;
    while (instr < max_instr && !stopped) {
        if ((++iters & 0xFFFFF) == 0 && getenv("RVDBG")) fprintf(stderr, "it %llu instr %llu pc %08x cyc %llu flags %x\n", (unsigned long long)iters, (unsigned long long)instr, core->pc, (unsigned long long)*(uint64_t *)&core->cyclel, core->extraflags);
        uint64_t *ccount = (uint64_t *)&core->cyclel;
        if ((core->extraflags & 4) && !kbhit()) {
            // Sleeping in WFI: each idle call adds one cycle and nothing else happens until
            // timer > timermatch, so jump to the first such cycle (same state as iterating).
            uint64_t match = ((uint64_t)core->timermatchh << 32) | core->timermatchl;
            if (!match) { printf("WFI with no timer armed\n"); break; }
            uint64_t wake = (match + 1) * time_divisor;
            if (wake > *ccount + 1) {
                uint64_t skip = wake - 1 - *ccount;
                // the idle calls in between only advance the timer; replay their effect
                *ccount += skip;
                wfi_idle += skip;
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
        uint32_t pc = core->pc, regs_before[32];
        memcpy(regs_before, core->regs, sizeof regs_before);
        uint32_t mcause_before = core->mcause, mepc_before = core->mepc;
        uint64_t c0 = *ccount;
        int ret = MiniRV32IMAStep(core, ram_image, 0, elapsed, 1);
        int executed = (*ccount != c0);  // cycle advances only when an instruction ran
        if (ret == 1) { *ccount += 1; wfi_idle++; }
        else if (ret == 0x5555) { printf("POWEROFF\n"); break; }
        else if (ret == 0x7777) { printf("RESTART\n"); break; }
        if (!executed && ret != 1) {
            if (core->mcause != mcause_before || core->mepc != mepc_before || (core->mcause & 0x80000000)) interrupts++;
            continue;
        }
        if (!executed) continue;
        instr++;
        uint32_t ofs = pc - ram_base;
        for (int q = 0; q < n_caller; q++) if (pc == caller_pc[q]) { uint32_t r = regs_before[1] - ram_base; if (r < ram_amt) ra_hist[r >> 2]++; }
        if (ofs < ram_amt) pc_hist[ofs >> 2]++, fetched[ofs >> 2] = 1, code_written[ofs >> 2] |= (written[ofs >> 2] ? 1 : 0);
        if (ofs < ram_amt && written[ofs >> 2]) instr_from_written++;
        if (ofs >= text_hi) instr_above_text++;
        uint32_t ir = (ofs < ram_amt - 3) ? *(uint32_t *)(ram_image + ofs) : 0;
        uint32_t opc = ir & 0x7f;
        op_count[opc]++;
        uint32_t rd = (ir >> 7) & 31;
        int writes_rd = opc == 0x37 || opc == 0x17 || opc == 0x6f || opc == 0x67 || opc == 0x03 ||
                        opc == 0x13 || opc == 0x33 || opc == 0x2f || (opc == 0x73 && ((ir >> 12) & 3));
        int trapped = core->mcause != mcause_before || core->mepc != mepc_before;
        uint32_t npc = core->pc;
        // token model: value records of 1 token (16-bit signed) or 2 (16 + 16 bits); jumps carry
        // the target (1 token within +-2^15 words, else 2); a PC refresh every 64 sequential
        // records; SB/SH carry the merged word; DIV/REM one token per hex quotient digit.
        uint32_t f3 = (ir >> 12) & 7, f7 = ir >> 25;
        int is_jump = npc != pc + 4 && !trapped;
        int is_link = opc == 0x6f || opc == 0x67;
        uint64_t tk = 0;
        if (trapped) tk += 2, tok_trap += 2;
        else {
            if (writes_rd && rd && !is_link) {
                int vt = fits_signed(core->regs[rd], 16) ? 1 : 2;
                tk += vt, tok_value += vt;
                if (opc == 0x2f && (ir >> 27) != 2 && (ir >> 27) != 3) tk += 2, tok_amo += 2;
            }
            if (opc == 0x33 && f7 == 1 && f3 >= 4) tk += 8, tok_div += 8;
            if (opc == 0x23 && f3 != 2) tk += 2, tok_subword += 2;
            if (is_jump) {
                int32_t d = ((int32_t)(npc - pc)) / 4;
                int jt = (d > -(1 << 15) && d < (1 << 15)) ? 1 : 2;
                tk += jt, tok_jump += jt;
                run_len = 0;
            } else if (!(writes_rd && rd && !is_link) && !(opc == 0x33 && f7 == 1 && f3 >= 4)) {
                tk += 1, tok_plain += 1;  // NT / ST / fence / CSR write / rd = x0
            }
            if (!is_jump && ++run_len == 63) tk += 1, tok_refresh += 1, run_len = 0;
            if (ir == 0x10500073) tk += 1, tok_wfi += 1;
        }
        tokens += tk;
        if (writes_rd && rd && !trapped) {
            uint32_t v = core->regs[rd];
            rd_writes++;
            for (int b = 1; b <= 32; b++) if (fits_signed(v, b)) rd_fit[b]++;
            if (v - ram_base < ram_amt) ptr_vals++;
        }
        (void)regs_before;
        if (opc == 0x73) csr_count[ir >> 20 & 0xfff][(ir >> 12) & 7]++;
        if (opc == 0x2f) amo_count[ir >> 27]++;
        if (opc == 0x03 || opc == 0x23) {
            uint32_t a = regs_before[(ir >> 15) & 31] + (uint32_t)(opc == 0x03 ? ((int32_t)ir >> 20) : (((int32_t)ir >> 25) << 5 | ((ir >> 7) & 31)));
            if (a - ram_base >= ram_amt) {
                int k = 0;
                while (k < n_mmio && !(mmio_addr[k] == a && mmio_kind[k] == opc)) k++;
                if (k == n_mmio && n_mmio < 64) mmio_addr[n_mmio++] = a, mmio_kind[k] = opc;
                if (k < 64) mmio_n[k]++;
            }
        }
        if (opc == 0x63) branches++;
        if (opc == 0x6f) jal++;
        if (opc == 0x67) jalr++;
        if (npc != pc + 4) {
            uint32_t ws = (pc - ram_base) >> 2, wt = (npc - ram_base) >> 2;
            for (int b = 0; b < 4; b++) if ((ws >> (15 + b)) != (wt >> (15 + b))) page_cross[b]++;
            if (wt >= (text_hi >> 2)) jump_to_ram++;
            taken++;
            int32_t d = ((int32_t)(npc - pc)) / 4;
            if (d < 0) d = -d;
            for (int k = 0; k < 16; k++) if (d < (1 << k)) near_jump[k]++;
        }
        // script
        if (script_pos < n_script && !(pending && *pending) && ends_with(triggers[script_pos])) {
            if (script_pos == 0) { prompt_cycle = *ccount; prompt_instr = instr; }
            pending = responses[script_pos++];
        }
        if (script_pos >= n_script && stop_after && !(pending && *pending) && ends_with(stop_after)) stopped = 1;
    }
    uint64_t *ccount = (uint64_t *)&core->cyclel;
    printf("\n=== rvtrace summary ===\n");
    printf("instructions %llu  cycles %llu  wfi_idle_cycles %llu  interrupts %llu  traps %llu\n",
           (unsigned long long)instr, (unsigned long long)*ccount, (unsigned long long)wfi_idle,
           (unsigned long long)interrupts, (unsigned long long)n_traps);
    printf("first-trigger at instr %llu cycle %llu\n", (unsigned long long)prompt_instr, (unsigned long long)prompt_cycle);
    printf("uart bytes out %llu\n", (unsigned long long)n_out_bytes);
    printf("loads %llu stores sb %llu sh %llu sw %llu misaligned %llu\n", (unsigned long long)n_load,
           (unsigned long long)n_store1, (unsigned long long)n_store2, (unsigned long long)n_store4,
           (unsigned long long)n_misaligned);
    printf("branches %llu jal %llu jalr %llu taken(pc!=pc+4) %llu\n", (unsigned long long)branches,
           (unsigned long long)jal, (unsigned long long)jalr, (unsigned long long)taken);
    printf("near jumps (|dw| < 2^k):");
    for (int k = 4; k < 16; k += 2) printf(" k%d=%.3f", k, near_jump[k] / (double)(taken ? taken : 1));
    printf("\nrd writes %llu; fit signed bits:", (unsigned long long)rd_writes);
    for (int b = 8; b <= 24; b += 2) printf(" %d:%.3f", b, rd_fit[b] / (double)(rd_writes ? rd_writes : 1));
    printf(" ramptr:%.3f\n", ptr_vals / (double)(rd_writes ? rd_writes : 1));
    uint64_t nw = 0, nf = 0, ncw = 0, ncw2 = 0;
    for (uint32_t i = 0; i < ram_amt / 4; i++) { nw += written[i]; nf += fetched[i]; ncw += (code_written[i] & 1); ncw2 += (code_written[i] & 2) ? 1 : 0; }
    printf("words written %llu (%.1f KB)  words fetched %llu (%.1f KB)  fetched-after-write %llu  written-after-fetch %llu\n",
           (unsigned long long)nw, nw * 4 / 1024.0, (unsigned long long)nf, nf * 4 / 1024.0,
           (unsigned long long)ncw, (unsigned long long)ncw2);
    printf("instructions fetched from runtime-written words %llu  above -T text end %llu\n",
           (unsigned long long)instr_from_written, (unsigned long long)instr_above_text);
    printf("taken jumps %llu: page crossings for 2^15..2^18-word pages %llu %llu %llu %llu; into runtime code %llu\n",
           (unsigned long long)taken, (unsigned long long)page_cross[0], (unsigned long long)page_cross[1],
           (unsigned long long)page_cross[2], (unsigned long long)page_cross[3], (unsigned long long)jump_to_ram);
    printf("data accesses below -T: loads %llu (first pc %08x addr %08x)  stores %llu (first pc %08x addr %08x)\n",
           (unsigned long long)n_text_load, first_text_load_pc, first_text_load_addr + 0x80000000u,
           (unsigned long long)n_text_store, first_text_store_pc, first_text_store_addr + 0x80000000u);
    printf("token model: %llu tokens, %.3f per instruction (value %llu jump %llu plain %llu refresh "
           "%llu subword %llu div %llu trap %llu amo %llu wfi %llu)\n", (unsigned long long)tokens,
           tokens / (double)(instr ? instr : 1), (unsigned long long)tok_value,
           (unsigned long long)tok_jump, (unsigned long long)tok_plain, (unsigned long long)tok_refresh,
           (unsigned long long)tok_subword, (unsigned long long)tok_div, (unsigned long long)tok_trap,
           (unsigned long long)tok_amo, (unsigned long long)tok_wfi);
    printf("system/csr (csr:funct3=count):");
    for (int c = 0; c < 4096; c++) for (int m = 0; m < 8; m++) if (csr_count[c][m]) printf(" %03x:%d=%llu", c, m, (unsigned long long)csr_count[c][m]);
    printf("\namo (funct5=count):");
    for (int c = 0; c < 32; c++) if (amo_count[c]) printf(" %d=%llu", c, (unsigned long long)amo_count[c]);
    printf("\nmmio (addr kind=count):");
    for (int k = 0; k < n_mmio; k++) printf(" %08x %s=%llu", mmio_addr[k], mmio_kind[k] == 3 ? "ld" : "st", (unsigned long long)mmio_n[k]);
    printf("\nopcodes:");
    for (int i = 0; i < 128; i++) if (op_count[i]) printf(" %02x:%.3f", i, op_count[i] / (double)instr);
    printf("\n");
    if (getenv("RVRAHIST")) { FILE *h = fopen(getenv("RVRAHIST"), "wb"); fwrite(ra_hist, 4, ram_amt / 4, h); fclose(h); }
    if (getenv("RVHIST")) { FILE *h = fopen(getenv("RVHIST"), "wb"); fwrite(pc_hist, 4, ram_amt / 4, h); fclose(h); }
    if (console_log) fclose(console_log);
    return 0;
}
