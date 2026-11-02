"""Events of a recorded token stream, for the hero video (tools/render_hero2.py).

The stream is replayed through the ISA decoder of the reference machine (lx/ref.py), record by
record: each record's tokens must equal the recorded ones, or the replay stops. The stream
carries values (what a register gets, the bytes printed) but not register numbers or store
addresses, which follow from the instruction at the PC; the decoder supplies those. The
machine's keyboard input is read from the stream itself (its KEY0 / KEY tokens).

Per event, the index of the stream token that commits it (stream[0] is GO for a boot):

  regw    register writes (rd)             store   RAM stores (byte offset, size)
  uart    UART bytes (byte, the RAM offset it was loaded from, the token that stored it there)
  trap    traps (payload: mcause, 15 = timer interrupt)    priv   privilege after a change
  key     keyboard bytes typed by the host  fetch   instructions fetched from RAM (I tokens)
  pages   every --every tokens: the kernel's page map, read from RAM through its own
          struct page array (mem_map): free (buddy) / reserved / slab / other, per 4 KB page

    python tools/video_events.py boot --stream STREAM.npy --out OUT.npz
    python tools/video_events.py session --stream CALL0.npy CALL1.npy --snapshot BOOTED.pt \
        --out OUT.npz
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from lx import merge
from lx import ref as R
from lx import spec as S
from lx.linux import CMDLINE, IMAGE

# kernel/Image's symbols (its System.map, checked by kernel_symbols() when the build is present)
SYMS = {"mem_map": 0x8017536C, "linux_banner": 0x801644E0, "__log_buf": 0x80178310,
        "_stext": 0x80001000, "_etext": 0x800E4240, "__init_end": 0x8014C020,
        "__bss_start": 0x80175000, "_end": 0x8017C4B0}
SYSTEM_MAP = IMAGE.parent / "linux-6.8-rc1" / "System.map"
PAGE = 4096
PFN0 = S.RAM_BASE // PAGE  # ARCH_PFN_OFFSET: mem_map[0] is the page at RAM_BASE
PAGE_STRUCT = 32  # sizeof(struct page) in this config (no MEMCG, KMSAN, page virtual)
PG_SLAB, PG_RESERVED = 1 << 11, 1 << 14
PAGE_TYPE_BASE, PG_BUDDY = 0xF0000000, 0x80
# page map codes
P_NONE, P_FREE, P_RESERVED, P_SLAB, P_USED = 0, 1, 2, 3, 4


def kernel_symbols() -> dict:
    if SYSTEM_MAP.exists():
        have = {}
        for ln in SYSTEM_MAP.read_text().splitlines():
            a, _, nm = ln.split()
            if nm in SYMS:
                have[nm] = int(a, 16)
        assert have == SYMS, (have, SYMS)
    return SYMS


def page_map(ram, n_pages: int) -> np.ndarray | None:
    """Per 4 KB page: P_FREE (in a buddy free block), P_RESERVED, P_SLAB, P_USED (any other
    page the kernel manages), P_NONE (no struct page: before mem_map is set up, or beyond the
    kernel's memory node). Read from the kernel's own struct page array."""
    mm = int.from_bytes(ram[SYMS["mem_map"] - S.RAM_BASE:][:4], "little")
    if not mm:
        return None
    o = mm - S.RAM_BASE
    nmax = (len(ram) - o) // PAGE_STRUCT
    n = min(n_pages, nmax)
    sp = np.frombuffer(bytes(ram[o:o + n * PAGE_STRUCT]), dtype="<u4").reshape(n, 8)
    flags, private, ptype = sp[:, 0], sp[:, 5], sp[:, 6]
    out = np.full(n_pages, P_NONE, dtype=np.uint8)
    out[:n] = P_USED
    out[:n][(flags & PG_SLAB) != 0] = P_SLAB
    out[:n][(flags & PG_RESERVED) != 0] = P_RESERVED
    buddy = (ptype & (PAGE_TYPE_BASE | PG_BUDDY)) == PAGE_TYPE_BASE
    for p in np.nonzero(buddy)[0]:
        order = int(private[p])
        if order <= 10:
            out[p:p + (1 << order)] = P_FREE
    return out


def free_blocks(ram, n_pages: int) -> list[tuple[int, int]]:
    """The buddy allocator's free blocks: (first page, order), from struct page."""
    mm = int.from_bytes(ram[SYMS["mem_map"] - S.RAM_BASE:][:4], "little")
    o = mm - S.RAM_BASE
    n = min(n_pages, (len(ram) - o) // PAGE_STRUCT)
    sp = np.frombuffer(bytes(ram[o:o + n * PAGE_STRUCT]), dtype="<u4").reshape(n, 8)
    buddy = (sp[:, 6] & (PAGE_TYPE_BASE | PG_BUDDY)) == PAGE_TYPE_BASE
    return [(int(p), int(sp[p, 5])) for p in np.nonzero(buddy)[0]]


def reset_machine() -> R.Machine:
    cfg = merge.config()
    ram, regs = R.linux_ram(IMAGE.read_bytes(), cfg.ram_size, CMDLINE)
    return R.Machine(ram, cfg.text_end, regs)


CHAIN = 8  # copies followed back from a UART byte
# 8250 line control register: with its bit 7 (DLAB) set, a store to THR's address writes the
# baud divisor on a real 8250; mini-rv32ima's UART has no divisor latch and prints it
UART_LCR = S.UART_THR + 3
# instructions that write rd (when rd != 0 and they do not trap)
_RD_OPS = {0x37, 0x17, 0x6F, 0x67, 0x03, 0x13, 0x33, 0x2F}


def replay(m: R.Machine, stream: np.ndarray, i0: int, every: int = 4096) -> dict:
    """Step m through stream[i0:] (m at the record boundary before stream[i0]). Asserts that
    every record's tokens are the stream's. Returns the event arrays."""
    toks = stream.tolist()
    N = len(toks)
    n_pages = m.size // PAGE
    key0, nk = S.FAMILIES["KEY0"]
    key1 = S.FAMILIES["KEY"][0]
    b0, nb = S.FAMILIES["BYTE"]
    t0_, ntr = S.FAMILIES["TRAP"]
    regw_t, regw_r = [], []
    st_t, st_a, st_s = [], [], []
    ua_t, ua_b, ua_src, ua_wr, ua_dlab = [], [], [], [], []
    lcr = 0  # the 8250's line control register as the kernel last wrote it
    tr_t, tr_c = [], []
    pv_t, pv_p = [], []
    ky_t, ky_b = [], []
    fe_t = []
    pg_t, pg_map, pg_instret = [], [], []
    rec_end = []  # final token of each record (and of each typed key)
    writer = np.full(m.size, -1, dtype=np.int64)  # token that last stored each RAM byte
    # copies: register -> (RAM offset, size) it was loaded from (through mv / andi 0xff), and
    # per RAM byte the offset it was copied from by its latest store (-1: computed, not copied)
    src = [None] * 32
    parent = np.full(m.size, -1, dtype=np.int64)
    i = i0
    priv = m.priv
    next_sample = i0
    words = m.size // 4
    tw = m.text_words
    ram = m.ram
    x = m.x
    t_start = time.time()
    while i < N:
        if i >= next_sample:
            pm = page_map(ram, n_pages)
            pg_t.append(i)
            pg_instret.append(m.instret)
            pg_map.append(pm if pm is not None else np.zeros(n_pages, dtype=np.uint8))
            next_sample = i + every
        t = toks[i]
        if key0 <= t < key0 + nk or key1 <= t < key1 + nk:
            got = m.key(t - (key0 if t < key0 + nk else key1))
            assert got == [t], (i, got, t)
            ky_t.append(i)
            ky_b.append(got[0] - (key0 if t < key0 + nk else key1))
            rec_end.append(i)
            i += 1
            continue
        ir = None
        pcw = m.anchor + m.n
        if pcw < words:
            ir = int.from_bytes(ram[4 * pcw:4 * pcw + 4], "little")
            v1 = x[ir >> 15 & 31]
            v2 = x[ir >> 20 & 31]
        ni = m.instret
        rec = m.step()
        n = len(rec)
        if toks[i:i + n] != rec:
            j = next(k for k in range(n) if i + k >= N or toks[i + k] != rec[k])
            raise AssertionError(f"stream differs from the decoder at token {i + j}: "
                                 f"{S.name(rec[j])} vs "
                                 f"{S.name(toks[i + j]) if i + j < N else 'end'}")
        e = i + n - 1  # the record's final token: its commit
        rec_end.append(e)
        trapped = any(t0_ <= r < t0_ + ntr for r in rec)
        if trapped:
            tr_t.append(e)
            tr_c.append(next(r - t0_ for r in rec if t0_ <= r < t0_ + ntr))
        if m.instret != ni:
            if pcw >= tw:
                fe_t.append(e)
            opc, rd, f3 = ir & 0x7F, ir >> 7 & 31, ir >> 12 & 7
            if opc == 0x23:
                s_ = src[ir >> 20 & 31]
                if m.last_store is not None:
                    size = (1, 2, 4)[f3]
                    a = m.last_store
                    st_t.append(e)
                    st_a.append(a)
                    st_s.append(size)
                    writer[a:a + size] = e
                    parent[a:a + size] = -1
                    if s_ is not None:
                        k = min(size, s_[1])
                        parent[a:a + k] = np.arange(s_[0], s_[0] + k)
                else:
                    addr = (v1 + R.sx((ir >> 25) << 5 | (ir >> 7 & 31), 12)) & R.M32
                    if addr == UART_LCR:
                        lcr = v2 & 0xFF
                    if addr == S.UART_THR and b0 <= rec[-1] < b0 + nb:
                        ua_t.append(e)
                        ua_b.append(rec[-1] - b0)
                        ua_dlab.append(lcr >> 7 & 1)
                        # the copies back from the byte's register: (offset, its writer)
                        ch, p = [], -1 if s_ is None else s_[0]
                        while p >= 0 and len(ch) < CHAIN:
                            ch.append((p, int(writer[p])))
                            p = int(parent[p])
                        ch += [(-1, -1)] * (CHAIN - len(ch))
                        ua_src.append([c_[0] for c_ in ch])
                        ua_wr.append([c_[1] for c_ in ch])
            elif opc == 0x2F and m.last_store is not None:
                st_t.append(e)
                st_a.append(m.last_store)
                st_s.append(4)
                writer[m.last_store:m.last_store + 4] = e
                parent[m.last_store:m.last_store + 4] = -1
            if rd and not trapped:
                if opc in _RD_OPS or (opc == 0x73 and f3 & 3):
                    if not (opc == 0x33 and ir >> 25 & 1 and f3 >= 4 and rec[-1] ==
                            S.tok("NT")):
                        regw_t.append(e)
                        regw_r.append(rd)
                    rs1, imm = ir >> 15 & 31, ir >> 20
                    if opc == 0x03:
                        ofs = (v1 + R.sx(imm, 12) - S.RAM_BASE) & R.M32
                        src[rd] = (ofs, (1, 2, 4, 0, 1, 2, 0, 0)[f3]) if ofs < m.size else None
                    elif opc == 0x13 and f3 == 0 and imm == 0:  # mv
                        src[rd] = src[rs1]
                    elif opc == 0x13 and f3 == 7 and imm == 0xFF and src[rs1] is not None:
                        src[rd] = (src[rs1][0], 1)  # the low byte
                    else:
                        src[rd] = None
        if m.priv != priv:
            priv = m.priv
            pv_t.append(e)
            pv_p.append(priv)
        i += n
        if len(rec_end) % 1_000_000 == 0:
            print(json.dumps({"tokens": i, "of": N, "s": round(time.time() - t_start)}),
                  flush=True)
    a = lambda v, d=np.int64: np.asarray(v, dtype=d)  # noqa: E731
    return {"regw_t": a(regw_t), "regw_r": a(regw_r, np.uint8), "store_t": a(st_t),
            "store_a": a(st_a), "store_s": a(st_s, np.uint8), "uart_t": a(ua_t),
            "uart_b": a(ua_b, np.uint8), "uart_src": a(ua_src).reshape(-1, CHAIN),
            "uart_writer": a(ua_wr).reshape(-1, CHAIN), "uart_dlab": a(ua_dlab, np.uint8),
            "trap_t": a(tr_t), "trap_c": a(tr_c, np.uint8), "priv_t": a(pv_t),
            "priv_p": a(pv_p, np.uint8), "key_t": a(ky_t), "key_b": a(ky_b, np.uint8),
            "fetch_t": a(fe_t), "rec_end": a(rec_end), "pages_t": a(pg_t),
            "pages_instret": a(pg_instret), "pages": np.stack(pg_map),
            "end_instret": np.int64(m.instret), "end_priv": np.int64(m.priv),
            "start_token": np.int64(i0), "tokens": np.int64(N)}


def events(what: str, streams: list, snapshot: str = "", every: int = 4096,
           machine=None) -> dict:
    """boot: one stream from reset (stream[0] = GO). session: the calls' streams one after
    another (the machine waits at the prompt between them), from a booted reference machine
    (machine, or a snapshot's)."""
    import copy
    kernel_symbols()
    t0 = time.time()
    stream = np.concatenate(streams)
    if what == "boot":
        assert stream[0] == S.tok("GO")
        ev = replay(reset_machine(), stream, 1, every)
    else:
        if machine is None:
            import torch

            from lx.session import load_safe
            machine = (load_safe(snapshot) if snapshot.endswith(".safetensors") else
                       torch.load(snapshot, weights_only=False, map_location="cpu"))["ref"]
        m = copy.deepcopy(machine)
        start = m.instret
        ev = replay(m, stream, 0, every)
        ev["start_instret"] = np.int64(start)
    ev["call_starts"] = np.cumsum([0] + [len(s) for s in streams[:-1]])
    ev["seconds"] = np.float64(time.time() - t0)
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["boot", "session"])
    ap.add_argument("--stream", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--snapshot", default="", help="session: the booted snapshot it starts from")
    ap.add_argument("--every", type=int, default=4096)
    a = ap.parse_args()
    t0 = time.time()
    ev = events(a.what, [np.load(p) for p in a.stream], a.snapshot, a.every)
    np.savez_compressed(a.out, **ev)
    print(json.dumps({"out": a.out, "seconds": round(time.time() - t0, 1),
                      "instructions": int(ev["end_instret"]), "uart_bytes": len(ev["uart_t"]),
                      "traps": len(ev["trap_t"]),
                      "timer_interrupts": int((ev["trap_c"] == S.TRAP_INT).sum()),
                      "stores": len(ev["store_t"]), "register_writes": len(ev["regw_t"])}))


if __name__ == "__main__":
    main()
