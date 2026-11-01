"""Tokens, records and the address map of the LX machine (RV32IMA + Zicsr, M/U modes, CLINT
timer, UART), the CPU that boots Linux inside the model. It executes exactly what
mini-rv32ima.h executes, under the platform of ref/rvtrace.c (see ref.py).

A trace is a sequence of records. Machine state changes only at a record's last (final) token,
its commit. The PC is implied: PC = RAM_BASE + 4 * W with the word index W = anchor + n, where
the anchor (21 bits) is set by J tokens (low 16 bits; a JH token right before a J also sets the
high 5 bits) and n counts the commits since then that advance the PC.

Token families (payload bits):

  V     16  not final; the low half of a 2-token value
  F     16  final; after V: the value's high half; else the whole value, sign-extended
  D     16  not final; data the commit needs that the circuit cannot have in time: the new
            value of a CSR (low, high), the new mstatus of a trap / MRET / WFI (with the new
            privilege in bits 13..14 for a trap / MRET), the old word of a read-modify-write
            (low, high), the new time's high half before a TK
  I     16  not final; the instruction word of code in RAM (low, then high)
  J     16  final; a jump: anchor[15:0] = payload (after a JH also anchor[20:16]), n = 0
  R     16  final; the same for a trap, an MRET or an anchor refresh, which never write a
            register (the register file cannot know in time that an instruction faults)
  JH     5  not final; the high bits of the next J / R word index (when they change)
  TK    16  final; a timer tick: time[15:0] = payload (after a D also time[31:16])
  TRAP  16  not final; a trap: payload = mcause (0..11), or 15 for the timer interrupt
  QS    16  not final; DIV-class: k0 (as Qwenputer's RV core)
  QD   128  not final; DIV-class: quotient digit k << 4 | q
  BYTE 256  final; a byte written to the UART
  RX   256  final; a load from the UART RBR that takes a byte from the receive FIFO (rd = it)
  KEY0 256  the host's input into an empty receive FIFO: a record boundary that pushes the
            byte (and starts the FIFO's slots over); nothing else changes
  KEY  256  the same into a FIFO that is not empty (at most RX_SLOTS bytes since a KEY0)
  controls  NT ST0..ST7 MTL MTH WFI HALT (final); GO starts a trace (a record boundary that
            changes nothing)

Records (instruction records of code in RAM start with I(lo) I(hi)):

  rd-writing instruction (ALU, MUL, load, LR, LUI, AUIPC)   F | V F   (NT when rd = x0)
  DIV / DIVU / REM / REMU                                   QS QD.. F | V F   (NT: rd = x0)
  taken branch, JAL, JALR                                   [JH] J    (the link is in-circuit)
  not-taken branch, FENCE                                   NT
  SW (RAM), SC.W that succeeds                              ST<x>     (x = the word's value
                                                                       head, address bits 4:2)
  SB, SH (RAM)                                              D D ST<x> (the old word)
  AMO<op>.W                                                 D D ST<x> (the old word = rd)
  SC.W that fails                                           F(1) | NT
  load from the UART RBR with input waiting                 RX(byte)  (any width, any rd;
                                                            with none: 0, as any MMIO load)
  load from the UART LSR                                    F(0x60 | input waiting)
  store to the UART THR                                     BYTE
  store to mtimecmp (low / high word)                       MTL / MTH
  store to SYSCON                                           HALT
  other MMIO store                                          NT
  CSR op                                                    [D D] F | V F | NT
                                                            (D D: the CSR's new value, when it
                                                            is a stored CSR and the op writes)
  WFI                                                       D WFI     (mstatus | MIE)
  MRET                                                      D [JH] R  (target mepc)
  an exception (ECALL, EBREAK, illegal, access fault)       TRAP D [JH] R  (target mtvec)

Records that are not instructions, at a record boundary, in this order of priority:

  the time advances (64 commits since the last tick, or a WFI sleeps until the timer)
                                                            [D] TK
  the timer interrupt is taken                              TRAP(15) D [JH] R
  n == N_MAX (the anchor refresh)                           [JH] R

Time: T = cycle / 64 (rvtrace -t 64), held in the circuit as two halves. Every instruction is
one cycle (WFI too, in this platform), a timer interrupt entry none; a WFI that finds
T <= mtimecmp sleeps until T = mtimecmp + 1. The timer interrupt is pending when
T > mtimecmp != 0 (mtimecmp < 2^32).

Address map (mini-rv32ima): RAM at RAM_BASE; the kernel's text (below TEXT_END, never written
after load) is fetched from a read-only copy (ROM); the rest of RAM is in the RAM layers. MMIO:
UART 0x10000000 (THR / RBR at +0, LSR at +5), CLINT 0x11000000 (mtimecmp +0x4000, mtime
+0xbff8), SYSCON 0x11100000.
"""

FAMILY_SIZES = [("V", 1 << 16), ("F", 1 << 16), ("D", 1 << 16), ("I", 1 << 16), ("J", 1 << 16),
                ("R", 1 << 16), ("TK", 1 << 16), ("JH", 32), ("TRAP", 16), ("QS", 16),
                ("QD", 128), ("BYTE", 256), ("RX", 256), ("KEY0", 256), ("KEY", 256)]
FAMILIES = {}
_base = 0
for _n, _s in FAMILY_SIZES:
    FAMILIES[_n] = (_base, _s)
    _base += _s
CTRL = _base
CONTROL_NAMES = ["NT", *(f"ST{x}" for x in range(8)), "MTL", "MTH", "WFI", "HALT", "GO"]
CONTROLS = {n: CTRL + i for i, n in enumerate(CONTROL_NAMES)}
VOCAB = CTRL + len(CONTROL_NAMES)
FINAL = {"F", "J", "R", "TK", "BYTE", "RX", "NT", *(f"ST{x}" for x in range(8)), "MTL", "MTH",
         "WFI", "HALT"}
INPUT = {"KEY0", "KEY"}  # the host's tokens: record boundaries the circuit never predicts
RX_SLOTS = 64  # receive FIFO: bytes pushed since its latest KEY0
PAYLOAD_BITS = {n: (s - 1).bit_length() for n, s in FAMILY_SIZES}

RAM_BASE = 0x80000000
N_MAX = 63  # the anchor refresh comes first when n == N_MAX
TICK = 64  # commits per time step
TRAP_INT = 15  # TRAP payload of the timer interrupt (mcause 0x80000007)
MCAUSE_INT = 0x80000007
MISA = 0x40401101
MVENDORID = 0xFF0FF0FF
# the stored CSRs (each its own head); misa / mvendorid read as constants, others as 0
CSRS = {"mstatus": 0x300, "mie": 0x304, "mtvec": 0x305, "mscratch": 0x340, "mepc": 0x341,
        "mcause": 0x342, "mtval": 0x343, "mip": 0x344}
CSR_CONST = {0x301: MISA, 0xF11: MVENDORID}
UART_THR, UART_LSR = 0x10000000, 0x10000005
MTIMECMP_LO, MTIMECMP_HI = 0x11004000, 0x11004004
MTIME_LO, MTIME_HI = 0x1100BFF8, 0x1100BFFC
SYSCON = 0x11100000
MMIO_LO, MMIO_HI = 0x10000000, 0x12000000


def family(t: int) -> tuple[str, int]:
    """(family or control name, payload) of a token."""
    if t >= CTRL:
        if t >= VOCAB:
            raise ValueError(f"not an lx token: {t}")
        return CONTROL_NAMES[t - CTRL], 0
    for nm, (base, size) in FAMILIES.items():
        if base <= t < base + size:
            return nm, t - base
    raise ValueError(f"not an lx token: {t}")


def tok(nm: str, payload: int = 0) -> int:
    if nm in CONTROLS:
        assert payload == 0, nm
        return CONTROLS[nm]
    base, size = FAMILIES[nm]
    assert 0 <= payload < size, (nm, payload)
    return base + payload


def is_final(t: int) -> bool:
    return family(t)[0] in FINAL


def fits16(v: int) -> bool:
    v &= 0xFFFFFFFF
    return v < 0x8000 or v >= 0xFFFF8000


def value_record(v: int) -> list[int]:
    """Tokens of a 32-bit value: 1 when it fits 16 bits signed, else 2."""
    v &= 0xFFFFFFFF
    if fits16(v):
        return [tok("F", v & 0xFFFF)]
    return [tok("V", v & 0xFFFF), tok("F", v >> 16)]


def name(t: int) -> str:
    fam, p = family(t)
    return fam if fam in CONTROLS else f"{fam}({p:#x})" if fam in ("V", "F", "D", "I", "J", "R", "TK") \
        else f"{fam}({p})"
