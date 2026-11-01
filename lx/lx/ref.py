"""The LX reference machine: mini-rv32ima.h's semantics under ref/rvtrace.c's platform, record
by record, with the tokens the circuit must produce (see spec.py).

Cases the circuit does not implement raise ``Unsupported`` (the harness checks that none occurs
in a trace): data accesses into the kernel's text, misaligned data accesses, a jump or trap to a
misaligned or out-of-RAM target, reads of the cycle CSR or of CSR 0x140, a WFI with mtimecmp
unarmed or >= 2^32, and mstatus >= 2^16 at a WFI.

UART input is the host's: ``key(b)`` at a record boundary (its KEY0 / KEY token goes into the
trace there) puts b in the receive FIFO, which mini-rv32ima's IsKBHit / ReadKBByte read.
"""

import re
from pathlib import Path

from lx import spec as S

M32 = 0xFFFFFFFF


class Unsupported(Exception):
    pass


def sx(v: int, b: int) -> int:
    v &= (1 << b) - 1
    return v - (1 << b) if v >> (b - 1) else v


def nibbles_used(v: int) -> int:
    return sum(1 for i in range(8) if v >> 4 * i)


def div_parts(f3: int, a: int, b: int) -> tuple[int, list[int], int]:
    """(k0, the quotient digits k0..7 of |A| / |B|, the result) of DIV / DIVU / REM / REMU
    (funct3 4..7), as Qwenputer's RV core computes them (rv/ref.py)."""
    signed = f3 in (4, 6)
    sa, sb = signed and a >> 31 == 1, signed and b >> 31 == 1
    ua, ub = (-a) & M32 if sa else a, (-b) & M32 if sb else b
    q = ua // ub if ub else M32
    da = nibbles_used(a)
    db = nibbles_used((~b & M32) if sb else b)
    k0 = 0 if b == 0 else min(max(7 - da + db, 0), 8)
    assert (q >> 4 * (8 - k0) == 0) if k0 < 8 else q == 0
    digits = [q >> 4 * (7 - k) & 15 for k in range(k0, 8)]
    qs = (-q) & M32 if sa != sb and b else q
    res = qs if f3 in (4, 5) else (a - qs * b) & M32
    return k0, digits, res


def alu_m(f3: int, a: int, b: int) -> int:
    """RV32M as mini-rv32ima.h computes it."""
    sa, sb = sx(a, 32), sx(b, 32)
    if f3 == 0:
        return (a * b) & M32
    if f3 == 1:
        return ((sa * sb) >> 32) & M32
    if f3 == 2:
        return ((sa * b) >> 32) & M32
    if f3 == 3:
        return ((a * b) >> 32) & M32
    if f3 == 4:
        if b == 0:
            return M32
        if sa == -(1 << 31) and sb == -1:
            return a
        q = abs(sa) // abs(sb)
        return (q if (sa < 0) == (sb < 0) else -q) & M32
    if f3 == 5:
        return M32 if b == 0 else a // b
    if f3 == 6:
        if b == 0:
            return a
        if sa == -(1 << 31) and sb == -1:
            return 0
        r = abs(sa) % abs(sb)
        return (-r if sa < 0 else r) & M32
    return a if b == 0 else a % b


def amo_op(f5: int, old: int, rs2: int) -> int | None:
    s2, so = sx(rs2, 32), sx(old, 32)
    return {1: rs2, 0: (rs2 + old) & M32, 4: rs2 ^ old, 12: rs2 & old, 8: rs2 | old,
            16: rs2 if s2 < so else old, 20: rs2 if s2 > so else old,
            24: rs2 if rs2 < old else old, 28: rs2 if rs2 > old else old}.get(f5)


class Machine:
    """State as the circuit holds it, and ``step``: one record (its tokens), committed."""

    def __init__(self, ram: bytes | bytearray, text_end: int = 0, regs: dict | None = None):
        self.size = len(ram)
        assert self.size % 4 == 0 and self.size <= 1 << 23
        self.ram = bytearray(ram)
        self.text_end = text_end  # byte offset; code below it is fetched from the ROM
        self.text_words = text_end // 4
        self.rom = bytes(self.ram[:text_end])  # the read-only copy (kernel text)
        self.x = [0] * 32
        for r, v in (regs or {}).items():
            self.x[r] = v & M32
        self.priv = 3
        self.csr = {k: 0 for k in S.CSRS}
        self.T = 0  # time (cycle / 64)
        self.nt = 0  # commits since the latest tick
        self.mlo = self.mhi = 0  # mtimecmp
        self.res = 0  # the LR reservation (mini-rv32ima's extraflags >> 3)
        self.anchor = 0  # PC word index = anchor + n
        self.n = 0
        self.post_wfi = False  # the latest final token was WFI
        self.halted = False
        self.out = bytearray()  # UART output
        # UART input, as the circuit holds it: slot k = the k-th byte since the latest KEY0;
        # rx_w = KEY tokens since then, rx_r = bytes read (empty: rx_r > rx_w)
        self.rx_buf = bytearray(S.RX_SLOTS)
        self.rx_w, self.rx_r = 0, 1
        self.cycles = 0  # mini-rv32ima's cycle counter
        self.instret = 0
        self.last_store = None  # byte offset of the latest record's RAM store

    # --- helpers ------------------------------------------------------------------------------

    @property
    def pcw(self) -> int:
        return self.anchor + self.n

    @property
    def pc(self) -> int:
        return S.RAM_BASE + 4 * self.pcw

    def word(self, w: int) -> int:
        return int.from_bytes(self.ram[4 * w:4 * w + 4], "little")

    def mtip(self, t: int) -> bool:
        return self.mhi == 0 and self.mlo != 0 and t > self.mlo

    def jump(self, w: int, fam: str = "J") -> list[int]:
        """[JH] J (or R) to word w; the anchor becomes w."""
        assert 0 <= w < 1 << 21
        toks = [S.tok("JH", w >> 16)] if w >> 16 != self.anchor >> 16 else []
        toks.append(S.tok(fam, w & 0xFFFF))
        self.anchor, self.n = w, 0
        return toks

    def target(self, addr: int) -> int:
        ofs = (addr - S.RAM_BASE) & M32
        if addr & 3 or ofs >= self.size:
            raise Unsupported(f"jump target {addr:#x}")
        return ofs >> 2

    def ram_ofs(self, addr: int, size: int, pc: int) -> int:
        ofs = (addr - S.RAM_BASE) & M32
        if ofs < self.text_end:
            raise Unsupported(f"data access to kernel text {addr:#x} at {pc:#x}")
        if ofs & (size - 1):
            raise Unsupported(f"misaligned {size}-byte access {addr:#x} at {pc:#x}")
        return ofs

    @property
    def rx_waiting(self) -> int:
        return self.rx_w + 1 - self.rx_r

    def key(self, b: int) -> list[int]:
        """The host types byte b (at a record boundary): its KEY0 / KEY token."""
        assert 0 <= b < 256
        if not self.rx_waiting:
            self.rx_w, self.rx_r = 0, 0
            self.rx_buf[0] = b
            return [S.tok("KEY0", b)]
        if self.rx_w + 1 >= S.RX_SLOTS:
            raise ValueError("receive FIFO full: let the machine read it first")
        self.rx_w += 1
        self.rx_buf[self.rx_w] = b
        return [S.tok("KEY", b)]

    # --- records ------------------------------------------------------------------------------

    def step(self) -> list[int]:
        assert not self.halted
        if self.post_wfi:
            self.post_wfi = False
            t1 = self.T + (self.nt == S.TICK)
            if not self.mtip(t1):
                if self.mhi or not self.mlo:
                    raise Unsupported("WFI with mtimecmp unarmed or >= 2^32")
                return self.tick(self.mlo + 1)
        if self.nt == S.TICK:
            return self.tick(self.T + 1)
        if self.mtip(self.T) and self.csr["mie"] >> 7 & 1 and self.csr["mstatus"] >> 3 & 1:
            return self.trap(S.TRAP_INT, S.MCAUSE_INT, 0, self.pc)
        if self.n == S.N_MAX:
            return self.jump(self.pcw, "R")
        return self.instruction()

    def tick(self, t: int) -> list[int]:
        if t >> 32:
            raise Unsupported("time >= 2^32")
        toks = [S.tok("D", t >> 16)] if t >> 16 != self.T >> 16 else []
        toks.append(S.tok("TK", t & 0xFFFF))
        self.T, self.nt, self.cycles = t, 0, 64 * t
        return toks

    def trap(self, payload: int, mcause: int, mtval: int, pc: int) -> list[int]:
        ms = self.csr["mstatus"]
        new = ((ms & 8) << 4) | (self.priv << 11)
        toks = [S.tok("TRAP", payload), S.tok("D", new | 3 << 13)]
        toks += self.jump(self.target(self.csr["mtvec"]), "R")
        self.csr.update(mepc=pc, mcause=mcause, mtval=mtval, mstatus=new)
        self.priv = 3
        return toks

    def instruction(self) -> list[int]:
        w = self.pcw
        if w >= self.size // 4:
            raise Unsupported(f"fetch outside RAM at {self.pc:#x}")
        ir = self.word(w)
        pre = [S.tok("I", ir & 0xFFFF), S.tok("I", ir >> 16)] if w >= self.text_words else []
        self.cycles += 1
        self.instret += 1
        self.nt += 1
        self.last_store = None
        return pre + self.execute(ir, self.pc)

    def execute(self, ir: int, pc: int) -> list[int]:
        x = self.x
        opc, rd, f3 = ir & 0x7F, ir >> 7 & 31, ir >> 12 & 7
        r1, r2 = x[ir >> 15 & 31], x[ir >> 20 & 31]
        imm_i = sx(ir >> 20, 12)
        rval = None  # the value written to rd (None: no write)
        trap = None  # (cause, mtval)

        def advance(toks):
            self.n += 1
            return toks

        if opc == 0x37:
            rval = ir & 0xFFFFF000
        elif opc == 0x17:
            rval = (pc + (ir & 0xFFFFF000)) & M32
        elif opc in (0x6F, 0x67):
            if opc == 0x6F:
                off = sx(((ir >> 31) << 20) | (ir >> 21 & 0x3FF) << 1 | (ir >> 20 & 1) << 11
                         | (ir >> 12 & 0xFF) << 12, 21)
                tgt = (pc + off) & M32
            else:
                tgt = (r1 + imm_i) & ~1 & M32
            tw = self.target(tgt)
            if rd:
                x[rd] = (pc + 4) & M32
            return self.jump(tw)
        elif opc == 0x63:
            off = sx((ir >> 31) << 12 | (ir >> 7 & 1) << 11 | (ir >> 25 & 0x3F) << 5
                     | (ir >> 8 & 15) << 1, 13)
            s1, s2 = sx(r1, 32), sx(r2, 32)
            cond = {0: r1 == r2, 1: r1 != r2, 4: s1 < s2, 5: s1 >= s2, 6: r1 < r2,
                    7: r1 >= r2}.get(f3)
            if cond is None:
                trap = (2, pc)
            elif cond:
                return self.jump(self.target((pc + off) & M32))
            else:
                return advance([S.tok("NT")])
        elif opc == 0x03:
            addr = (r1 + imm_i) & M32
            ofs = (addr - S.RAM_BASE) & M32
            if ofs >= self.size - 3:
                if addr == S.UART_THR and self.rx_waiting:
                    b = self.rx_buf[self.rx_r]
                    self.rx_r += 1
                    if rd:
                        x[rd] = b
                    return advance([S.tok("RX", b)])
                if S.MMIO_LO <= addr < S.MMIO_HI:
                    rval = self.mmio_load(addr)
                else:
                    trap = (5, addr)
            elif f3 in (0, 1, 2, 4, 5):
                size = (1, 2, 4, 0, 1, 2)[f3]
                ofs = self.ram_ofs(addr, size, pc)
                v = int.from_bytes(self.ram[ofs:ofs + size], "little")
                rval = sx(v, 8 * size) & M32 if f3 < 4 else v
            else:
                trap = (2, pc)
        elif opc == 0x23:
            addr = (r1 + sx((ir >> 25) << 5 | (ir >> 7 & 31), 12)) & M32
            ofs = (addr - S.RAM_BASE) & M32
            if ofs >= self.size - 3:
                if S.MMIO_LO <= addr < S.MMIO_HI:
                    return advance(self.mmio_store(addr, r2))
                trap = (7, addr)
            elif f3 in (0, 1, 2):
                size = (1, 2, 4)[f3]
                ofs = self.ram_ofs(addr, size, pc)
                wo = ofs & ~3
                old = int.from_bytes(self.ram[wo:wo + 4], "little")
                self.ram[ofs:ofs + size] = (r2 & ((1 << 8 * size) - 1)).to_bytes(size, "little")
                self.last_store = ofs
                st = S.tok(f"ST{wo >> 2 & 7}")
                if f3 == 2:
                    return advance([st])
                return advance([S.tok("D", old & 0xFFFF), S.tok("D", old >> 16), st])
            else:
                trap = (2, pc)
        elif opc in (0x13, 0x33):
            is_reg = opc == 0x33
            b = r2 if is_reg else imm_i & M32
            if is_reg and ir >> 25 & 1:
                if f3 >= 4:
                    k0, digits, res = div_parts(f3, r1, b)
                    assert res == alu_m(f3, r1, b)
                    if not rd:
                        return advance([S.tok("NT")])
                    x[rd] = res
                    return advance([S.tok("QS", k0)]
                                   + [S.tok("QD", (k0 + i) << 4 | q) for i, q in enumerate(digits)]
                                   + S.value_record(res))
                rval = alu_m(f3, r1, b)
            else:
                sh = b & 31
                rval = {0: (r1 - b) if is_reg and ir >> 30 & 1 else (r1 + b),
                        1: r1 << sh, 2: int(sx(r1, 32) < sx(b, 32)), 3: int(r1 < b),
                        4: r1 ^ b, 5: (sx(r1, 32) >> sh) if ir >> 30 & 1 else (r1 >> sh),
                        6: r1 | b, 7: r1 & b}[f3] & M32
        elif opc == 0x0F:
            return advance([S.tok("NT")])
        elif opc == 0x73:
            csrno = ir >> 20
            if f3 & 3:
                rs1i = ir >> 15 & 31
                src = rs1i if f3 >= 4 else r1
                old = self.csr_read(csrno)
                new = {1: src, 2: old | src, 3: old & ~src & M32}[f3 & 3]
                toks = []
                name = next((k for k, v in S.CSRS.items() if v == csrno), None)
                if name and ((f3 & 3) == 1 or rs1i):
                    self.csr[name] = new
                    toks = [S.tok("D", new & 0xFFFF), S.tok("D", new >> 16)]
                if rd:
                    x[rd] = old
                    return advance(toks + S.value_record(old))
                return advance(toks + [S.tok("NT")])
            if f3 == 0:
                if csrno & 0xFF == 0x02:  # MRET (also URET / SRET, as mini-rv32ima)
                    ms = self.csr["mstatus"]
                    new = ((ms & 0x80) >> 4) | (self.priv << 11) | 0x80
                    npriv = ms >> 11 & 3
                    toks = [S.tok("D", new | npriv << 13)]
                    toks += self.jump(self.target(self.csr["mepc"]), "R")
                    self.csr["mstatus"], self.priv = new, npriv
                    return toks
                if csrno == 0:
                    trap = (11 if self.priv else 8, pc)
                elif csrno == 1:
                    trap = (3, pc)
                elif csrno == 0x105:
                    new = self.csr["mstatus"] | 8
                    if new >> 16:
                        raise Unsupported("WFI with mstatus >= 2^16")
                    self.csr["mstatus"] = new
                    self.post_wfi = True
                    return advance([S.tok("D", new), S.tok("WFI")])
                else:
                    trap = (2, pc)
            else:
                trap = (2, pc)
        elif opc == 0x2F:
            f5 = ir >> 27
            ofs = (r1 - S.RAM_BASE) & M32
            if ofs >= self.size - 3:
                trap = (7, r1)
            else:
                ofs = self.ram_ofs(r1, 4, pc)
                old = int.from_bytes(self.ram[ofs:ofs + 4], "little")
                st = S.tok(f"ST{ofs >> 2 & 7}")
                if f5 == 2:  # LR.W
                    self.res = ofs & 0x1FFFFFFF
                    rval = old
                elif f5 == 3:  # SC.W
                    if self.res == ofs & 0x1FFFFFFF:
                        self.ram[ofs:ofs + 4] = r2.to_bytes(4, "little")
                        self.last_store = ofs
                        if rd:
                            x[rd] = 0
                        return advance([st])
                    rval = 1
                else:
                    new = amo_op(f5, old, r2)
                    if new is None:
                        trap = (2, pc)
                    else:
                        self.ram[ofs:ofs + 4] = new.to_bytes(4, "little")
                        self.last_store = ofs
                        if rd:
                            x[rd] = old
                        return advance([S.tok("D", old & 0xFFFF), S.tok("D", old >> 16), st])
        else:
            trap = (2, pc)

        if trap is not None:
            cause, tval = trap
            return self.trap(cause, cause, tval, pc)
        if rd and rval is not None:
            x[rd] = rval
            return advance(S.value_record(rval))
        return advance([S.tok("NT")])

    def csr_read(self, csrno: int) -> int:
        if csrno in (0xC00, 0x140):
            raise Unsupported(f"read of CSR {csrno:#x}")
        if csrno == S.CSRS["mip"]:
            return (self.csr["mip"] & ~0x80) | (self.mtip(self.T) << 7)
        for k, v in S.CSRS.items():
            if v == csrno:
                return self.csr[k]
        return S.CSR_CONST.get(csrno, 0)

    def mmio_load(self, addr: int) -> int:
        if addr == S.UART_LSR:
            return 0x60 | (self.rx_waiting > 0)
        if addr == S.MTIME_LO:
            return self.T & M32
        return 0  # RBR without input, mtime high (time < 2^32), the rest

    def mmio_store(self, addr: int, v: int) -> list[int]:
        if addr == S.UART_THR:
            self.out.append(v & 0xFF)
            return [S.tok("BYTE", v & 0xFF)]
        if addr == S.MTIMECMP_HI:
            self.mhi = v
            return [S.tok("MTH")]
        if addr == S.MTIMECMP_LO:
            self.mlo = v
            return [S.tok("MTL")]
        if addr == S.SYSCON and v:
            if v not in (0x5555, 0x7777):
                raise Unsupported(f"SYSCON {v:#x}")
            self.halted = True
            return [S.tok("HALT")]
        return [S.tok("NT")]

    # --- running ------------------------------------------------------------------------------

    def run(self, max_records: int, stop=None) -> list[int]:
        toks = []
        for _ in range(max_records):
            if self.halted or (stop and stop(self)):
                break
            toks += self.step()
        return toks


# --- the Linux boot image (as rvtrace lays it out) -------------------------------------------

_DTB_H = Path(__file__).resolve().parents[2] / "mini-rv32ima/mini-rv32ima/default64mbdtc.h"
STATE_BYTES = 48 * 4  # sizeof(struct MiniRV32IMAState) at the top of RAM


def default_dtb() -> bytes:
    src = _DTB_H.read_text()
    body = src[src.index("{") + 1:src.rindex("}")]
    return bytes(int(v, 16) for v in re.findall(r"0x[0-9a-fA-F]+", body))


def linux_ram(image: bytes, ram_size: int, cmdline: str | None) -> tuple[bytearray, dict]:
    """RAM and registers at reset, as rvtrace sets them up (default DTB, cmdline, RAM size)."""
    ram = bytearray(ram_size)
    ram[:len(image)] = image
    dtb = default_dtb()
    ptr = ram_size - len(dtb) - STATE_BYTES
    ram[ptr:ptr + len(dtb)] = dtb
    if cmdline is not None:
        ram[ptr + 0xC0:ptr + 0xC0 + 54] = cmdline.encode()[:54].ljust(54, b"\0")  # strncpy
    if int.from_bytes(ram[ptr + 0x13C:ptr + 0x140], "little") == 0x00C0FF03:
        ram[ptr + 0x13C:ptr + 0x140] = ptr.to_bytes(4, "big")
    return ram, {10: 0, 11: ptr + S.RAM_BASE}
