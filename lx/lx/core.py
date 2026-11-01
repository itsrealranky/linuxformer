"""The LX machine (spec.py) as a circuit of Gated DeltaNet heads and SwiGLU units (ir.py).

Every position computes the record that the next token belongs to from the state after its own
commit, so at a record's last token the circuit already works on the next record and inside a
record it recomputes the current one (nothing changes until the record's last token). A write
takes its key from t-1 (a conv tap) or from dims at t, its write gate beta from the token at t,
and its value from token payloads (directly at lags 0..3, or through the lag head) or from dims
the circuit computed at t-1 below the writing layer.

Layout (Qwen3.5 linear-attention layers: 16 key heads of 256 rows, 8 value heads each):

  L1   attn  anchor (low 16 bits), the counters n (commits since the anchor), j (tokens since the
             record boundary), v (DIV), time (low half), the WFI flag, the IR head (an
             instruction fetched from RAM), the lag head (token features at lags 1..3)
       mlp   thermometers; token patterns
  L2   attn  anchor (high 5 bits), time (high half), the tick counter (commits since a tick)
       mlp   PC word W = anchor + n, W + 1, time + 1
  L3   attn  ROM: the kernel's text, 8 nibbles per word (one sublayer, up to 2^18 words)
       mlp   nibbles to bits; W in RAM?
  L4   mlp   decode (as mini-rv32ima.h does)
  L5   attn  registers (2 read ports), quotient digits
  L5-7 mlp   adder, logic, shifter, multiplier, store / AMO data, address mux, MMIO decode
  L8.. attn  CSRs, privilege, mtimecmp, the LR reservation, then RAM (2048 words per key head);
       mlp   multiplier, DIV, compares, faults, the interrupt and tick conditions
  last mlp   the output code (the exit)

The DIV / REM digit recurrence and the multiplier are Qwenputer's (chipformer/rv/core.py, MIT,
(c) 2026 Divyansh Agrawal), as are the counter, thermometer and adder gadgets.
"""

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from lx import spec as S
from lx.graph import Expr
from lx.ir import A_BIT, A_REF, REF_COLS, VD, Chan, K, Target
from lx.prims import Circ, chan, from_bits, sel, thead

T4B = Target(key_heads=16, value_heads=128, intermediate=9216)  # the widened Qwen3.5-4B layers
OUT_NAMES = list(S.FAMILIES) + S.CONTROL_NAMES
FAM_ID = {n: i for i, n in enumerate(OUT_NAMES)}
NFAM = (len(OUT_NAMES) - 1).bit_length()
NPAY = 16
OUT_W = 2.0**7  # head output weight of a one-hot read (as thead)
JMAX = 12  # longest record after its boundary: DIV (QS, 8 QD, V, F)
CODE_AMP = 128
KAPPA = 2.0**-20
PAYLOAD = {"V": 16, "F": 16, "D": 16, "I": 16, "J": 16, "R": 16, "TK": 16, "JH": 5, "QD": 7,
           "RX": 8, "KEY0": 8, "KEY": 8}
ROM_HEAD_WORDS = 1 << 14  # one key head of the ROM: 8 value heads x 256 rows x 8 words
RAM_HEAD_WORDS = 1 << 11  # one key head of RAM: 8 value heads x 256 rows


@dataclass
class Config:
    ram_size: int = 0x480000  # bytes
    text_end: int = 0xF65C0  # bytes below this are fetched from the ROM
    # the exit's output code: +-kappa per bit (small, as Qwenputer's, where text tokens run the
    # chip layers too; lx.merge's head is Qwen's final norm, whose eps needs it near 1)
    kappa: float = KAPPA

    @property
    def ram_words(self) -> int:
        return self.ram_size // 4

    @property
    def text_words(self) -> int:
        return self.text_end // 4


def code_bits(fam: str, p: int) -> list[int]:
    fid = FAM_ID[fam]
    return [fid >> i & 1 for i in range(NFAM)] + [p >> i & 1 for i in range(NPAY)]


def lits_of(bits, v):
    """0/1 literals that are all 1 exactly when the bits spell v."""
    return [b if v >> i & 1 else 1 - b for i, b in enumerate(bits)]


def andpre(lits) -> Expr:
    """sum(lits) - (n - 1): 1 when all literals are 1, <= 0 otherwise."""
    return sum(lits, Expr()) - (len(lits) - 1)


def col_offsets(bounds: list[tuple[int, int]], top: int = 255) -> list[int]:
    reach, hist = {0: None}, []
    for pos, neg in bounds:
        nxt = {}
        for gp in reach:
            for g in range(64):
                if neg <= 16 * g - gp <= top - pos and g not in nxt:
                    nxt[g] = gp
        assert nxt, "no column offsets"
        hist.append(nxt)
        reach = nxt
    g, offs = min(reach), []
    for i in range(len(bounds) - 1, -1, -1):
        gp = hist[i][g]
        offs.append(16 * g - gp)
        g = gp
    return offs[::-1]


class LxCore(Circ):
    floor = 0

    def __init__(self, cfg: Config, target: Target | None = None) -> None:
        super().__init__(target or T4B)
        self.cfg = cfg
        self.tables = {}

    def ffn(self, terms, name="ffn", into=None, min_level=0):
        return super().ffn(terms, name, into, max(min_level, self.floor))

    @contextmanager
    def at_least(self, sub: int):
        old, self.floor = self.floor, max(self.floor, sub - 1)
        try:
            yield
        finally:
            self.floor = old

    def fn_bits(self, z, lo, hi, f, n, name, lits=()):
        terms = [[] for _ in range(n)]
        for b in range(n):
            if f(lo) >> b & 1:
                terms[b] += self.t_and(list(lits)) if lits else self.t_const(1)
            for theta in range(lo + 1, hi + 1):
                dd = (f(theta) >> b & 1) - (f(theta - 1) >> b & 1)
                if dd:
                    terms[b] += self.t_ge(z, theta, lo, hi, dd, lits=list(lits))
        return terms

    def zeros(self, n, name):
        return [self.token_feature(f"{name}[{i}]", lambda t: 0, full=True) for i in range(n)]

    def table(self, name, queries, keys, vheads, levels="bit", tiny=0.0):
        first, rest = vheads[0], vheads[1:]
        outs = self.head(name, {s: sel(self, e) for s, e in queries.items()},
                         {s: sel(self, e) for s, e in keys.items()}, first["v"], first.get("a", 0),
                         first["z"], first["outs"], tiny=tiny, beta=first.get("beta"),
                         into=first.get("into"),
                         vx=[{"v": x["v"], "a": x.get("a", 0), "z": x["z"], "outs": x["outs"],
                              "beta": x.get("beta"), "into": x.get("into")} for x in rest])
        node = self.nodes[-1]
        node.levels = levels
        self.tables[name] = node
        return outs

    def int_bits(self, z, hi, name, n):
        out = []
        for j in range(n):
            terms = []
            for theta in range(1, hi + 1):
                d = (theta >> j & 1) - ((theta - 1) >> j & 1)
                if d:
                    terms += self.t_ge(z, theta, 0, hi, d)
            out.append(self.node(terms, f"{name}.b{j}"))
        return out

    def counter(self, name, reset, step):
        one = self.one
        ((x,),) = thead(self, name, {0: reset, 1: one - reset}, {0: one - reset, 1: one - reset},
                        [{"data": {2: reset}, "ref": one - reset,
                          "a": 110 * step + 238 * reset}], tiny=2.0**-90)
        self.tables[name] = self.nodes[-1]
        self.nodes[-1].levels = "decay"
        return x

    def thermo(self, x, reset, ms, name):
        """[k > m] for m in ms, from x = 2^-k (0 at reset tokens)."""
        one = self.one
        return [self.node([(1, one - reset, 7 * one - 8 * 2**m * x),
                           (-1, one - reset, 6 * one - 8 * 2**m * x)], f"{name}>{m}")
                for m in ms]

    def split(self, z, lo, hi, name):
        terms = self.t_const(lo // 16) if lo // 16 else []
        for t in range(lo // 16 + 1, hi // 16 + 1):
            terms += self.t_ge(z, 16 * t, lo, hi)
        return self.node(terms, name)

    def reg_table(self, name, keys, queries, data, beta, a=0):
        """A head of 0/1 rows with explicit value Chans: data = [(column Chan list, out dim)];
        every Chan is a source of the out dim (exactly one is live at a write)."""
        v, z, outs, into, col = {}, {}, [], [], 2
        for chans, dim in data:
            for ch in chans:
                v[col] = chan(ch, A_BIT)
                z[col] = K * self.one
                outs.append((col, OUT_W))
                into.append(dim)
                col += 1
        assert col <= VD, (name, col)
        for rc in REF_COLS:
            v[rc] = chan(self.one, A_REF)
        return {"v": v, "z": z, "outs": outs, "into": into, "beta": beta, "a": a}


def adder_stage2(c, g, p_, d, n_nib, name):
    one = c.one

    def carry_sources(k):
        return [[g[j]] + p_[j + 1:k] for j in range(k)]

    r = []
    for k in range(n_nib):
        for b in range(4):
            terms = c.t_and([d[k][b]])
            for src in carry_sources(k):
                low = d[k][:b]
                terms += c.t_and(src + low + [one - d[k][b]])
                terms += c.t_and(src + low + [d[k][b]], -1)
            r.append(c.node(terms, f"{name}.r{4 * k + b}"))
    cout = c.node([t for src in carry_sources(n_nib) for t in c.t_and(src)], f"{name}.cout")
    return r, cout


def stage1_digits(c, sums, name, hi):
    g, p_, d = [], [], []
    for k, forms in enumerate(sums):
        gt, pt, dt = [], [], [[] for _ in range(4)]
        for s, lits in forms:
            gt += c.t_ge(s, 16, 0, hi, lits=lits)
            pt += c.t_eq(s, 15, 0, hi, lits=lits)
            for b in range(4):
                for theta in range(1, hi + 1):
                    dd = (theta % 16 >> b & 1) - ((theta - 1) % 16 >> b & 1)
                    if dd:
                        dt[b] += c.t_ge(s, theta, 0, hi, dd, lits=lits)
        g.append(c.node(gt, f"{name}.g{k}"))
        p_.append(c.node(pt, f"{name}.p{k}"))
        d.append([c.node(dt[b], f"{name}.d{k}.{b}") for b in range(4)])
    return g, p_, d


def nib(bits, k, inv=False):
    x = from_bits(bits[4 * k:4 * k + 4])
    return 15 - x if inv else x


def add_nibbles(c, a_bits, b_bits, n_nib, lits, name, inv_b=False, cin=0):
    """A carry-lookahead add of two bit vectors (n_nib nibbles), gated by literals; with inv_b
    and cin = 1 a subtraction. Returns (result bits, carry out)."""
    sums = [[(nib(a_bits, k) + nib(b_bits, k, inv_b) + (cin if k == 0 else 0), lits)]
            for k in range(n_nib)]
    g, p_, d = stage1_digits(c, sums, name, 31)
    return adder_stage2(c, g, p_, d, n_nib, name)


def eq_bits(c, a, b, name, lits=()):
    """[a == b] for two bit vectors (two MLP stages: xnor, then and)."""
    xn = [c.node(c.t_and([ai, bi]) + c.t_and([1 - ai, 1 - bi]), f"{name}.x{i}")
          for i, (ai, bi) in enumerate(zip(a, b))]
    return c.and_(xn + list(lits), name)


def ge_const(c, bits, k, name):
    """[value(bits) >= k] for a constant k (one MLP node of mutually exclusive ANDs)."""
    n = len(bits)
    if k <= 0:
        return c.one
    if k >= 1 << n:
        return Expr()
    terms = []
    for i in range(n - 1, -1, -1):  # the highest bit where value and k differ: value 1, k 0
        if not k >> i & 1:
            lits = [bits[i]] + [bits[j] if k >> j & 1 else 1 - bits[j] for j in range(i + 1, n)]
            terms += c.t_and(lits)
    terms += c.t_and(lits_of(bits, k))  # value == k
    return c.node(terms, name)


def lt_terms(c, bits, k, lits=()):
    """Mutually exclusive AND terms, one true iff value(bits) < k (and the literals)."""
    n, terms = len(bits), []
    for i in range(n - 1, -1, -1):  # the highest bit where they differ: value 0, k 1
        if k >> i & 1:
            terms += c.t_and(list(lits) + [1 - bits[i]]
                             + [bits[j] if k >> j & 1 else 1 - bits[j] for j in range(i + 1, n)])
    return terms


def inc_bits(c, bits, name):
    """Bits of value(bits) + 1 (mod 2^n)."""
    out = []
    for i, b in enumerate(bits):
        low = list(bits[:i])
        if low:
            terms = c.t_and([b]) + c.t_and(low + [1 - b]) + c.t_and(low + [b], -1)
        else:
            terms = c.t_and([1 - b])
        out.append(c.node(terms, f"{name}.b{i}"))
    return out


def sel_terms(c, cases, i):
    """Terms of bit i of sum(cond * value) over exclusive cases [(lits, bit list or int)]."""
    terms = []
    for lits, val in cases:
        if isinstance(val, int):
            if val >> i & 1:
                terms += c.t_and(list(lits))
        elif i < len(val):
            b = val[i]
            if isinstance(b, int):
                if b:
                    terms += c.t_and(list(lits))
            elif b.terms:
                terms += c.t_and([*lits, b])
    return terms


CSR_OPS = ("CSRRW", "CSRRS", "CSRRC", "CSRRWI", "CSRRSI", "CSRRCI")
AMO_F5 = {"AMOADD": 0, "AMOSWAP": 1, "LR": 2, "SC": 3, "AMOXOR": 4, "AMOOR": 8, "AMOAND": 12,
          "AMOMIN": 16, "AMOMAX": 20, "AMOMINU": 24, "AMOMAXU": 28}
LAG1 = (0.0, 1.0, 0.0, 0.0)


def taps(*lags):
    return tuple(1.0 if k in lags else 0.0 for k in range(4))


def build(cfg: Config | None = None, target: Target | None = None) -> LxCore:
    cfg = cfg or Config()
    c = LxCore(cfg, target)
    f = c.token_feature
    one, zero = c.one, Expr()

    def famof(t):
        return S.family(t) if 0 <= t < S.VOCAB else (None, 0)

    fam = {n: f(f"is{n}", lambda t, n=n: int(famof(t)[0] == n), full=True) for n in OUT_NAMES}
    pay = {n: [f(f"{n}.b{i}", lambda t, n=n, i=i: famof(t)[1] >> i & 1 if famof(t)[0] == n
                 else 0, full=True) for i in range(w)] for n, w in PAYLOAD.items()}

    def trapf(name, fn):
        return f(name, lambda t: int(famof(t)[0] == "TRAP" and bool(fn(famof(t)[1]))), full=True)

    isTI = trapf("isTI", lambda p: p == S.TRAP_INT)

    def mcv(p):
        return S.MCAUSE_INT if p == S.TRAP_INT else p

    mcb = {b: trapf(f"mc.b{b}", lambda p, b=b: mcv(p) >> b & 1) for b in (0, 1, 2, 3, 31)}

    def next_digit(k):
        def fn(t):
            nm, p = famof(t)
            return int(nm == "QS" and p == k or nm == "QD" and p >> 4 == k - 1)
        return fn

    kk = [f(f"digit=={k}", next_digit(k), full=True) for k in range(8)]
    final = sum((fam[n] for n in S.FINAL), Expr())
    c.code = [f(f"code{i}", lambda t, i=i: CODE_AMP * (2 * code_bits(*famof(t))[i] - 1)
                if famof(t)[0] else 0, full=True) for i in range(NFAM + NPAY)]
    dp = sum(kk, Expr())  # the next token is a quotient digit
    isI, isTK, isR, isD, isJH = fam["I"], fam["TK"], fam["R"], fam["D"], fam["JH"]
    isF, isV, isNT, isWFI = fam["F"], fam["V"], fam["NT"], fam["WFI"]
    isJR = fam["J"] + isR
    isST = sum((fam[f"ST{x}"] for x in range(8)), Expr())
    isK0, isKEY, isRX = fam["KEY0"], fam["KEY"], fam["RX"]
    bnd = final + fam["GO"] + isK0 + isKEY  # a record boundary
    FNT = isF + isNT

    # ---------------------------------------------------------------- L1: state, lags
    (alo,) = thead(c, "anchor.lo", {0: one}, {0: one},
                   [{"data": {2 + i: pay["J"][i] + pay["R"][i] for i in range(16)}, "ref": one,
                     "beta": isJR}])
    c.tables["anchor.lo"] = c.nodes[-1]
    xn = c.counter("count", isJR, final - isJR - isTK)
    rec = bnd + isI  # j restarts at a boundary and at each I token
    xj = c.counter("j", rec, one - rec)
    v_reset = rec + fam["QS"] + fam["QD"]
    xv = c.counter("vcnt", v_reset, one - v_reset)
    # the UART's receive FIFO pointers: KEYs and RX reads since the latest KEY0
    xw = c.counter("rx.w", isK0, isKEY)
    xr = c.counter("rx.r", isK0, isRX)
    (tlo,) = thead(c, "time.lo", {0: one}, {0: one},
                   [{"data": {2 + i: pay["TK"][i] for i in range(16)}, "ref": one,
                     "beta": isTK}])
    c.tables["time.lo"] = c.nodes[-1]
    ((pw,),) = thead(c, "wfi", {0: one}, {0: one},
                     [{"data": {2: isWFI}, "ref": one, "beta": final}])
    c.tables["wfi"] = c.nodes[-1]
    # IR: an instruction fetched from RAM (I lo, I hi), cleared at every boundary
    irdata = {2 + i: Chan(pay["I"][i], LAG1) for i in range(16)}
    irdata.update({18 + i: pay["I"][i] for i in range(16)})
    (ir,) = thead(c, "ir", {0: one}, {0: one},
                  [{"data": irdata, "ref": one, "beta": isI, "a": 238 * bnd}])
    c.tables["ir"] = c.nodes[-1]
    # the lag head: token features of t-1, t-2, t-3 as dims at t
    lag_feats = [("I", isI, 1), ("V", isV, 1), ("JH", isJH, 1), ("D", isD, 1),
                 ("TRAP", fam["TRAP"], 1), ("TI", isTI, 1), ("D", isD, 2),
                 ("TRAP", fam["TRAP"], 2), ("TI", isTI, 2), ("JH", isJH, 2), ("D", isD, 3),
                 ("TI", isTI, 3)]
    data, col = {}, 2
    for _, e, lg in lag_feats:
        data[col] = Chan(e, taps(lg))
        col += 1
    for lg in (1, 2, 3):
        for i in range(16):
            data[col] = Chan(pay["D"][i], taps(lg))
            col += 1
    (lagd,) = thead(c, "lag", {0: one}, {0: one}, [{"data": data, "ref": one}])
    LG = {(nm, lg): lagd[k] for k, (nm, _, lg) in enumerate(lag_feats)}
    n0 = len(lag_feats)
    Dp = {lg: lagd[n0 + 16 * (lg - 1):n0 + 16 * lg] for lg in (1, 2, 3)}
    I1, V1, JH1, D1, TR1, TI1 = (LG[n, 1] for n in ("I", "V", "JH", "D", "TRAP", "TI"))
    D2, TR2, TI2, JH2 = (LG[n, 2] for n in ("D", "TRAP", "TI", "JH"))
    D3, TI3 = LG["D", 3], LG["TI", 3]

    # L1 mlp: token patterns
    PIL = c.and_([isI, 1 - I1], "pos.Ilo")  # predicts I(hi)
    PIH = c.and_([isI, I1], "pos.Ihi")  # the instruction's first token comes next
    tkD = c.and_([isTK, D1], "tkD")
    jhJR = c.and_([isJR, JH1], "jhJR")
    refR = c.node(c.t_and([isR, 1 - D1, 1 - JH1]) + c.t_and([isR, JH1, 1 - D2]), "refreshR")
    intR = c.node(c.t_and([isR, D1, TI2]) + c.t_and([isR, JH1, D2, TI3]), "intR")
    # at t-1 of a trap's R: the record is a trap (trapw) / an exception (trapx)
    trapw = c.node(c.t_and([isD, TR1]) + c.t_and([isJH, TR2]), "trapw")
    trapx = c.node(c.t_and([isD, TR1, 1 - TI1]) + c.t_and([isJH, TR2, 1 - TI2]), "trapx")

    # ---------------------------------------------------------------- L2: state
    (ahi,) = thead(c, "anchor.hi", {0: one}, {0: one},
                   [{"data": {2 + i: Chan(pay["JH"][i], LAG1) for i in range(5)}, "ref": one,
                     "beta": jhJR}])
    c.tables["anchor.hi"] = c.nodes[-1]
    (thi,) = thead(c, "time.hi", {0: one}, {0: one},
                   [{"data": {2 + i: Dp[1][i] for i in range(16)}, "ref": one, "beta": tkD}])
    c.tables["time.hi"] = c.nodes[-1]
    xt = c.counter("ntick", isTK, final - isTK - refR - intR)

    cnt = c.thermo(xn, isJR, range(S.N_MAX), "n")
    n_int = sum(cnt, Expr())
    rdue = cnt[S.N_MAX - 1]  # n == N_MAX: the refresh comes first
    jt = c.thermo(xj, rec, range(JMAX), "j")
    vt = c.thermo(xv, v_reset, range(3), "v")
    (nt64,) = c.thermo(xt, isTK, [S.TICK - 1], "nt")

    def jeq(v):
        return (one if v == 0 else jt[v - 1]) - (jt[v] if v < JMAX else 0)

    j1, j2, j3 = jeq(1), jeq(2), jeq(3)
    v0, v1 = one - vt[0], vt[0] - vt[1]
    inrec = one - bnd - isI  # a position after a record's first token

    # the receive FIFO (slot k: the k-th byte since the latest KEY0); a byte is waiting when
    # the reads have not caught up with the pushes (r <= w)
    rxw = c.thermo(xw, isK0, range(S.RX_SLOTS - 1), "rx.w")
    rxr = c.thermo(xr, isK0, range(S.RX_SLOTS), "rx.r")

    def slot(th, k):
        return andpre(([th[k - 1]] if k else []) + ([1 - th[k]] if k < len(th) else []))

    rxb = c.table("rx", {k: slot(rxr, k) for k in range(S.RX_SLOTS)},
                  {k: slot(rxw, k) for k in range(S.RX_SLOTS)},
                  [{"v": {**{2 + i: chan(Chan(pay["KEY0"][i] + pay["KEY"][i]), A_BIT)
                             for i in range(8)}, **{rc: chan(one, A_REF) for rc in REF_COLS}},
                    "z": {2 + i: K * one for i in range(8)},
                    "outs": [(2 + i, OUT_W) for i in range(8)], "beta": isK0 + isKEY}])
    rx_wait = c.node(c.t_ge(sum(rxw, Expr()) - sum(rxr, Expr()), 0, -S.RX_SLOTS,
                            S.RX_SLOTS - 1), "rx.waiting")

    # L2 mlp: W = anchor + n, W + 1, T + 1
    abits = alo + ahi

    def add_small(extra, name):
        Lv = from_bits(abits[:6]) + n_int + extra
        hi_ = 63 + 63 + extra
        bits = []
        for jb in range(6):
            terms = []
            for theta in range(1, hi_ + 1):
                d = (theta >> jb & 1) - ((theta - 1) >> jb & 1)
                if d:
                    terms += c.t_ge(Lv, theta, 0, hi_, d)
            bits.append(c.node(terms, f"{name}.b{jb}"))
        for jb in range(6, 21):
            below = abits[6:jb]
            terms = (c.t_and([abits[jb]])
                     + c.t_ge(Lv, 64, 0, hi_, 1, lits=below + [1 - abits[jb]])
                     + c.t_ge(Lv, 64, 0, hi_, -1, lits=below + [abits[jb]]))
            bits.append(c.node(terms, f"{name}.b{jb}"))
        return bits

    W = add_small(0, "w")
    W1 = add_small(1, "w+1")
    T = tlo + thi
    T1 = inc_bits(c, T, "t+1")
    pc_bits = [zero, zero] + W + [zero] * 8 + [one]  # PC = RAM_BASE + 4 W
    link_bits = [zero, zero] + W1 + [zero] * 8 + [one]
    pcT = [c.and_([trapw, b], f"pcT.b{i}") if b.terms and i < 31 else
           (trapw if i == 31 else zero) for i, b in enumerate(pc_bits)]

    # ---------------------------------------------------------------- L3: ROM (kernel text)
    # word W: key head W[17:14], value head W[13:11], row W[10:3], word in row W[2:0]
    TW = cfg.text_words
    nib_d = c.zeros(8, "rom.n")
    n_rom = -(-TW // ROM_HEAD_WORDS)
    assert n_rom <= 16, "text larger than one sublayer of ROM"
    for h in range(n_rom):
        hl = lits_of(W[14:18], h) + [1 - b for b in W[18:21]]
        q = {r: andpre(lits_of(W[3:11], r) + hl) for r in range(256)}
        vhs = []
        for x in range(8):
            z, outs, into = {}, [], []
            for k in range(8):
                ze = K * andpre(lits_of(W[11:14], x) + lits_of(W[0:3], k))
                for nb in range(8):
                    col = 2 + 8 * k + nb
                    z[col] = ze
                    outs.append((col, OUT_W))
                    into.append(nib_d[nb])
            vhs.append({"v": {}, "z": z, "outs": outs, "into": into})
        c.table(f"rom{h}", q, {}, vhs, levels="byte")
    rom_bits = []
    for k in range(8):
        rom_bits += c.int_bits(128 * nib_d[k], 15, f"rom.n{k}", 4)
    ib = [rom_bits[i] + ir[i] for i in range(32)]  # the instruction word (ROM or IR)
    ramcode = ge_const(c, W, TW, "ramcode")
    fm = c.node(c.t_and([PIL]) + c.t_and([bnd, ramcode]), "fm")  # RAM reads at W (a fetch)

    # ---------------------------------------------------------------- L4: decode
    opc, f3b, rdb, rs1b, rs2b = ib[0:7], ib[12:15], ib[7:12], ib[15:20], ib[20:25]
    b25, b30, f5b, csrb = ib[25], ib[30], ib[27:32], ib[20:32]
    U = {}

    def op(v):
        return lits_of(opc, v)

    def f3(v):
        return lits_of(f3b, v)

    def uop(name, lits):
        U[name] = c.and_(lits, f"uop.{name}")

    G = {v: c.and_(op(v), f"op{v:#04x}")
         for v in (0x37, 0x17, 0x6F, 0x67, 0x63, 0x03, 0x23, 0x13, 0x33, 0x0F, 0x73, 0x2F)}
    U["LUI"], U["AUIPC"], U["JAL"], U["JALR"], U["FENCE"] = (G[v] for v in
                                                            (0x37, 0x17, 0x6F, 0x67, 0x0F))
    for nm, v in (("BEQ", 0), ("BNE", 1), ("BLT", 4), ("BGE", 5), ("BLTU", 6), ("BGEU", 7)):
        uop(nm, op(0x63) + f3(v))
    uop("BRILL", op(0x63) + [1 - f3b[2], f3b[1]])
    for nm, v in (("LB", 0), ("LH", 1), ("LW", 2), ("LBU", 4), ("LHU", 5)):
        uop(nm, op(0x03) + f3(v))
    U["LDBAD"] = c.node(c.t_and(op(0x03) + f3(3)) + c.t_and(op(0x03) + [f3b[2], f3b[1]]),
                        "uop.LDBAD")
    for nm, v in (("SB", 0), ("SH", 1), ("SW", 2)):
        uop(nm, op(0x23) + f3(v))
    U["STBAD"] = c.node(c.t_and(op(0x23) + [f3b[2]]) + c.t_and(op(0x23) + f3(3)), "uop.STBAD")
    for nm, v in (("ADDI", 0), ("SLLI", 1), ("SLTI", 2), ("SLTIU", 3), ("XORI", 4), ("ORI", 6),
                  ("ANDI", 7)):
        uop(nm, op(0x13) + f3(v))
    uop("SRLI", op(0x13) + f3(5) + [1 - b30])
    uop("SRAI", op(0x13) + f3(5) + [b30])
    for nm, v in (("SLL", 1), ("SLT", 2), ("SLTU", 3), ("XOR", 4), ("OR", 6), ("AND", 7)):
        uop(nm, op(0x33) + f3(v) + [1 - b25])
    uop("ADD", op(0x33) + f3(0) + [1 - b25, 1 - b30])
    uop("SUB", op(0x33) + f3(0) + [1 - b25, b30])
    uop("SRL", op(0x33) + f3(5) + [1 - b25, 1 - b30])
    uop("SRA", op(0x33) + f3(5) + [1 - b25, b30])
    for v, nm in enumerate(("MUL", "MULH", "MULHSU", "MULHU", "DIV", "DIVU", "REM", "REMU")):
        uop(nm, op(0x33) + f3(v) + [b25])
    for nm, v in zip(CSR_OPS, (1, 2, 3, 5, 6, 7)):
        uop(nm, op(0x73) + f3(v))
    sys0 = op(0x73) + f3(0)
    uop("MRET", sys0 + lits_of(csrb[0:8], 0x02))
    uop("ECALL", sys0 + lits_of(csrb, 0))
    uop("EBREAK", sys0 + lits_of(csrb, 1))
    uop("WFI", sys0 + lits_of(csrb, 0x105))
    SYS0 = c.and_(sys0, "uop.SYS0")
    uop("SYS4", op(0x73) + f3(4))
    for nm, v in AMO_F5.items():
        uop(nm, op(0x2F) + lits_of(f5b, v))

    def any_of(*names):
        return sum((U[n] for n in names), Expr())

    LOADV = any_of("LB", "LH", "LW", "LBU", "LHU")
    BRANCH = any_of("BEQ", "BNE", "BLT", "BGE", "BLTU", "BGEU")
    csrop = any_of(*CSR_OPS)
    amo_ops = [n for n in AMO_F5 if n not in ("LR", "SC")]
    AMOBAD = G[0x2F] - any_of(*AMO_F5)
    ILL_ST = (one - sum(G.values(), Expr()) + U["BRILL"] + U["SYS4"]
              + SYS0 - any_of("MRET", "ECALL", "EBREAK", "WFI"))  # illegal at any address
    rd_oh = {r: c.and_(lits_of(rdb, r), f"rd=={r}") for r in range(1, 32)}
    rs1_oh = {r: c.and_(lits_of(rs1b, r), f"rs1=={r}") for r in range(1, 32)}
    rs2_oh = {r: c.and_(lits_of(rs2b, r), f"rs2=={r}") for r in range(1, 32)}
    rd_nz = sum(rd_oh.values(), Expr())
    rs1_z = c.and_([1 - b for b in rs1b], "rs1==0")
    csr_oh = {k: c.and_(lits_of(csrb, v), f"csr=={k}") for k, v in S.CSRS.items()}
    csr_const = {v: c.and_(lits_of(csrb, a), f"csr=={a:#x}") for a, v in S.CSR_CONST.items()}
    wr = c.node(c.t_and([csrop]) + c.t_and([csrop, f3b[1], rs1_z], -1), "csr.wr")
    imm_i = ib[20:32] + [ib[31]] * 20
    imm_s = ib[7:12] + ib[25:32] + [ib[31]] * 20
    imm_u = [zero] * 12 + ib[12:32]
    offb = [ib[9], ib[10], ib[11]] + ib[25:31] + [ib[7]] + [ib[31]] * 14  # imm_b >> 2
    offj = ib[22:31] + [ib[20]] + ib[12:20] + [ib[31]] * 6  # imm_j >> 2
    jl = [c.node(c.t_and(op(0x6F) + [b]) + c.t_and(op(0x67) + [b]), f"link.b{i}")
          if b.terms else zero for i, b in enumerate(link_bits)]
    writes_rd = (G[0x37] + G[0x17] + G[0x6F] + G[0x67] + G[0x13] + G[0x33] + G[0x03] + csrop
                 + G[0x2F])

    # ---------------------------------------------------------------- L5: registers
    Fb, Vb, Db = pay["F"], pay["V"], pay["D"]
    src = {}
    for i in range(32):
        if i < 16:
            cs = [Chan(Fb[i] - V1), Chan(Vb[i], LAG1), Chan(Dp[2][i] + isST + D1 - 2)]
            if i < 8:
                cs.append(Chan(pay["RX"][i]))
        else:
            cs = [Chan(Fb[i - 16] + V1 - 1), Chan(Fb[15] - V1), Chan(Dp[1][i - 16] + isST - 1)]
        if jl[i].terms:
            cs.append(Chan(jl[i], LAG1))  # the link, computed at t-1
        src[i] = cs
    reg_beta = isF + fam["J"] + isST + isRX
    reg_keys = {r: Chan(rd_oh[r] + writes_rd - 1, LAG1) for r in range(1, 32)}
    c.reg_cols = {}

    def reg_port(pname, rq):
        dims = c.zeros(32, f"x.{pname}")
        vhs = []
        for half in range(2):
            v, z, outs, into, col = {}, {}, [], [], 2
            for i in range(16 * half, 16 * half + 16):
                for ch in src[i]:
                    v[col] = chan(ch, A_BIT)
                    z[col] = K * one
                    outs.append((col, OUT_W))
                    into.append(dims[i])
                    c.reg_cols.setdefault(i, col)
                    col += 1
            for rc in REF_COLS:
                v[rc] = chan(one, A_REF)
            vhs.append({"v": v, "z": z, "outs": outs, "into": into, "beta": reg_beta})
        c.table(f"regs.{pname}", rq, reg_keys, vhs)
        return dims

    A = reg_port("rs1", rs1_oh)
    B = reg_port("rs2", rs2_oh)

    # the quotient head: value head d holds the record's d-th digit, written at its QD token
    # (j == d + 2), in one copy per alignment nd' > d; reads gated by the current nd give the
    # digits so far right-aligned
    nd_int = sum(jt, Expr()) - 1 - sum(vt, Expr())
    nd_is = {v: c.node(c.t_eq(nd_int, v, -4, JMAX), f"nd=={v}") for v in range(1, 9)}
    qal = c.zeros(32, "quo")
    vhs = []
    for d in range(8):
        v, z, outs, into, col = {}, {}, [], [], 2
        for ndp in range(d + 1, 9):
            for b in range(4):
                v[col] = chan(Chan(pay["QD"][b]), A_BIT)
                z[col] = K * nd_is[ndp]
                outs.append((col, OUT_W))
                into.append(qal[4 * (ndp - 1 - d) + b])
                col += 1
        for rc in REF_COLS:
            v[rc] = chan(one, A_REF)
        vhs.append({"v": v, "z": z, "outs": outs, "into": into,
                    "beta": fam["QD"] + jeq(d + 2) - 1})
    c.table("quo", {0: one}, {0: fam["QD"]}, vhs)

    # ---------------------------------------------------------------- L4-5: target adder
    # word-level W + offset: branch / JAL targets (bits 0..20) and AUIPC (PC + imm_u is
    # 4 * (W + 2^29 + imm_u / 4) mod 2^32), gated by opcode bits so it runs beside the decode
    Wfull = W + [zero] * 8 + [one] + [zero] * 2
    toffs = [(op(0x63), offb + [ib[31]] * 8), (op(0x6F), offj + [ib[31]] * 8),
             (op(0x17), [zero] * 10 + ib[12:32] + [zero] * 2)]
    tsums = [[(nib(Wfull, k) + nib(ob, k), ol) for ol, ob in toffs] for k in range(8)]
    tg, tp, td = stage1_digits(c, tsums, "tgt", 30)
    bt, _ = adder_stage2(c, tg, tp, td, 8, "tgt")
    auipc = [zero, zero] + bt[0:30]

    # ---------------------------------------------------------------- L5-6: adder
    # three operand modes: A + B, A - B, A + X (X: the immediate muxed at decode, ~imm + 1 for
    # SLTI / SLTIU, 0 for AMOs)
    sub_i = c.node(c.t_and(op(0x13) + [1 - f3b[2], f3b[1]]), "imm.sub")
    immx = []
    for i in range(32):
        terms = (c.t_and(op(0x03) + [imm_i[i]]) + c.t_and(op(0x67) + [imm_i[i]])
                 + c.t_and(op(0x13) + f3(0) + [imm_i[i]]) + c.t_and(op(0x23) + [imm_s[i]])
                 + c.t_and(op(0x13) + [1 - f3b[2], f3b[1], 1 - imm_i[i]]))
        immx.append(c.node(terms, f"imm.x{i}"))
    md = {"ADD": U["ADD"], "SUB": any_of("SUB", "SLT", "SLTU") + BRANCH,
          "IMM": G[0x03] + G[0x23] + G[0x2F] + any_of("ADDI", "JALR", "SLTI", "SLTIU")}
    modes = {"ADD": (B, False, 0), "SUB": (B, True, 1), "IMM": (immx, False, sub_i)}
    sums = [[(nib(A, k) + nib(b, k, inv) + (ci if k == 0 else 0), [md[m]])
             for m, (b, inv, ci) in modes.items()] for k in range(8)]
    g, p_, d = stage1_digits(c, sums, "add", 31)
    r, c32 = adder_stage2(c, g, p_, d, 8, "add")
    EA = r
    md["SUBI"] = any_of("SLTI", "SLTIU")

    # ---------------------------------------------------------------- L5-7: logic, shifter
    with c.at_least(12):
        logic = []
        for i in range(32):
            terms = []
            for opn, bb in (("AND", B), ("ANDI", imm_i)):
                terms += c.t_and([A[i], bb[i], U[opn]])
            for opn, bb in (("OR", B), ("ORI", imm_i)):
                terms += c.t_and([U[opn]]) + c.t_and([1 - A[i], 1 - bb[i], U[opn]], -1)
            for opn, bb in (("XOR", B), ("XORI", imm_i)):
                terms += (c.t_and([A[i], 1 - bb[i], U[opn]])
                          + c.t_and([1 - A[i], bb[i], U[opn]]))
            logic.append(c.node(terms, f"logic.b{i}"))
        sh_ops = {"L": ("SLL", "SLLI"), "R": ("SRL", "SRLI"), "A": ("SRA", "SRAI")}

        def src_bit(dirn, i):
            if 0 <= i < 32:
                return A[i]
            return A[31] if dirn == "A" and i >= 32 else None

        y = {}
        for dirn, (rop, iop) in sh_ops.items():
            y[dirn] = []
            for i in range(32):
                terms = []
                for opn, sb_ in ((rop, B), (iop, imm_i)):
                    for k_ in range(8):
                        jj = i - 4 * k_ if dirn == "L" else i + 4 * k_
                        bit = src_bit(dirn, jj)
                        if bit is None:
                            continue
                        terms += c.t_and([U[opn], bit] + lits_of(sb_[2:5], k_))
                y[dirn].append(c.node(terms, f"sh{dirn}1.b{i}"))
        shift = []
        for i in range(32):
            terms = []
            for dirn, (rop, iop) in sh_ops.items():
                yy = y[dirn]
                for opn, sb_ in ((rop, B), (iop, imm_i)):
                    for k_ in range(4):
                        jj = i - k_ if dirn == "L" else i + k_
                        bit = (yy[jj] if 0 <= jj < 32 else
                               (yy[31] if dirn == "A" and jj >= 32 else None))
                        if bit is None:
                            continue
                        terms += c.t_and([U[opn], bit] + lits_of(sb_[0:2], k_))
            shift.append(c.node(terms, f"shift.b{i}"))

    # ---------------------------------------------------------------- L5-9: multiplier
    # (Qwenputer's) 64-bit product of nibbles; REM: R = A - sigma Q B mod 2^32
    divc = any_of("DIV", "DIVU", "REM", "REMU")
    signed = any_of("DIV", "REM")
    mh_a = any_of("MULH", "MULHSU")
    mh_b = U["MULH"]
    mul_any = any_of("MUL", "MULH", "MULHSU", "MULHU")
    remc = any_of("REM", "REMU")
    prod = {}
    for i in range(8):
        for jn in range(8):
            terms = []
            for v in range(1, 16):
                terms += c.t_and(lits_of(A[4 * i:4 * i + 4], v) + [1 - divc], v, a=nib(B, jn))
                terms += c.t_and(lits_of(qal[4 * i:4 * i + 4], v) + [divc], v, a=nib(B, jn))
            prod[i, jn] = c.node(terms, f"mul.p{i}{jn}")
    corr_a = [c.node(c.t_and([A[31], mh_a], a=nib(B, jn, inv=True)), f"mul.ca{jn}")
              for jn in range(8)]
    corr_b = [c.node(c.t_and([B[31], mh_b], a=nib(A, jn, inv=True)), f"mul.cb{jn}")
              for jn in range(8)]
    plus1 = c.node(c.t_and([A[31], mh_a]) + c.t_and([B[31], mh_b]), "mul.plus1")
    lo, hi = {}, {}
    for (i, jn), pij in prod.items():
        hi[i, jn] = c.node([t for m_ in range(1, 15) for t in c.t_ge(pij, 16 * m_, 0, 225)],
                           f"mul.hi{i}{jn}")
        lo[i, jn] = pij - 16 * hi[i, jn]
    sa = c.and_([signed, A[31]], "div.sa")
    sbn = c.and_([signed, B[31]], "div.sb")
    sigma = c.node(c.t_and([signed, A[31], 1 - B[31]])
                   + c.t_and([signed, A[31]] + [1 - b for b in B], -1)
                   + c.t_and([signed, 1 - A[31], B[31]]), "div.sigma")
    cols = []
    rem_p, rem_m = [], []
    for k in range(16):
        s = sum((lo[i, k - i] for i in range(8) if 0 <= k - i < 8), Expr())
        s += sum((hi[i, k - 1 - i] for i in range(8) if 0 <= k - 1 - i < 8), Expr())
        if k < 8:
            nlo = sum(1 for i in range(8) if 0 <= k - i < 8)
            nhi = sum(1 for i in range(8) if 0 <= k - 1 - i < 8)
            rem_p.append((nib(A, k) + s, 15 + 15 * nlo + 14 * nhi))
            rem_m.append((nib(A, k) - s, (15, 15 * nlo + 14 * nhi)))
        if k >= 8:
            s += corr_a[k - 8] + corr_b[k - 8]
        if k == 8:
            s += plus1
        cols.append(s)
    rm_off = col_offsets([b for _, b in rem_m])
    mmodes = [("mul", cols, [0] * 16, [255] * 16, [mul_any]),
              ("remp", [e for e, _ in rem_p], [0] * 8, [hp for _, hp in rem_p], [remc, sigma]),
              ("remm", [e + o for (e, _), o in zip(rem_m, rm_off)],
               [o - b[1] for (_, b), o in zip(rem_m, rm_off)], [o + 15 for o in rm_off],
               [remc, 1 - sigma])]
    tsum = [[] for _ in range(16)]
    for mname, mcols, los, his, lits in mmodes:
        car = [c.split(e, lo_, hi_, f"mul.{mname}.c{k}")
               for k, (e, lo_, hi_) in enumerate(zip(mcols, los, his))]
        for k, e in enumerate(mcols):
            tsum[k].append((e - 16 * car[k] + (car[k - 1] if k else 0), lits))
    mg, mp, mdg = stage1_digits(c, tsum, "mul", 30)
    mres, _ = adder_stage2(c, mg, mp, mdg, 16, "mul")
    mul_lo, mul_hi = mres[:32], mres[32:]

    # ---------------------------------------------------------------- L5-10: DIV digits
    # (Qwenputer's radix-16 recurrence, one quotient digit per QD token)
    bp = [c.node(c.t_and([B[i]]) + c.t_and([B[i], signed, B[31]], -1)
                 + c.t_and([1 - B[i], signed, B[31]]), f"div.bp{i}") for i in range(32)]
    bm = [nib(bp, 0) + sbn] + [nib(bp, jn) for jn in range(1, 8)]
    nzA = [c.node(c.t_const(1) + c.t_and([1 - b for b in A[4 * i:]], -1), f"div.nza{i}")
           for i in range(8)]
    dahi = []
    for i in range(32):
        terms = []
        for k in range(8):
            jj = i + 4 * (7 - k)
            if jj < 32:
                terms += c.t_and([kk[k], A[jj], 1 - sa]) + c.t_and([kk[k], 1 - A[jj], sa])
        dahi.append(c.node(terms, f"div.ahi{i}"))
    ca = c.node([t for k in range(8)
                 for t in c.t_and([kk[k], sa] + [1 - A[jj] for jj in range(4 * (7 - k))])],
                "div.ca")
    nzB = [c.node(c.t_const(1) + c.t_and([1 - b for b in bp[4 * i:]], -1), f"div.nzb{i}")
           for i in range(8)]
    bz = c.and_([1 - b for b in B], "div.bz")
    kv = 7 - sum(nzA, Expr()) + sum(nzB, Expr())
    k0b = []
    for bit in range(4):
        terms = []
        for v in range(1, 8):
            if v >> bit & 1:
                terms += c.t_eq(kv, v, -1, 15, lits=[1 - bz])
        if bit == 3:
            terms += c.t_ge(kv, 8, -1, 15, lits=[1 - bz])
        k0b.append(c.node(terms, f"div.k0.b{bit}"))
    prod2, hi2, lo2 = {}, {}, {}
    for a_ in range(7):
        for b_ in range(8):
            top = 15 * (16 if b_ == 0 else 15)
            terms = []
            for v in range(1, 16):
                terms += c.t_and(lits_of(qal[4 * a_:4 * a_ + 4], v), v, a=bm[b_])
            prod2[a_, b_] = c.node(terms, f"div.p{a_}{b_}")
            hi2[a_, b_] = c.split(prod2[a_, b_], 0, top, f"div.hi{a_}{b_}")
            lo2[a_, b_] = prod2[a_, b_] - 16 * hi2[a_, b_]
    him, lom = {}, {}
    for m_ in range(1, 16):
        for b_ in range(8):
            top = 16 if b_ == 0 else 15
            terms = []
            for x_ in range(1, top + 1):
                dd = m_ * x_ // 16 - m_ * (x_ - 1) // 16
                if dd:
                    terms += c.t_ge(bm[b_], x_, 0, top, dd)
            him[m_, b_] = c.node(terms, f"div.mh{m_}.{b_}")
            lom[m_, b_] = m_ * bm[b_] - 16 * him[m_, b_]
    bounds, base = [], []
    for i in range(10):
        e = (nib(dahi, i) if i < 8 else Expr()) + (ca if i == 0 else 0)
        pos = (15 if i < 8 else 0) + (1 if i == 0 else 0)
        neg = 0
        for a_ in range(7):
            b_ = i - 1 - a_
            if 0 <= b_ < 8:
                e -= lo2[a_, b_]
                neg += 15
            b_ = i - 2 - a_
            if 0 <= b_ < 8:
                e -= hi2[a_, b_]
                neg += 15 if b_ == 0 else 14
        base.append(e)
        neg += (15 if i < 8 else 0) + ((15 if i == 1 else 14) if 1 <= i <= 8 else 0)
        bounds.append((pos, neg))
    coff = col_offsets(bounds)
    cmp_ge = []
    for m_ in range(1, 16):
        dig, car = [], []
        for i in range(10):
            e = base[i] + coff[i]
            if i < 8:
                e -= lom[m_, i]
            if 1 <= i <= 8:
                e -= him[m_, i - 1]
            lo_, hi_ = coff[i] - bounds[i][1], coff[i] + bounds[i][0]
            cc = c.split(e, lo_, hi_, f"div.c{m_}.{i}")
            car.append((cc, hi_ // 16))
            dig.append(e - 16 * cc)
        g_, p2_ = [], []
        for i in range(9):
            s = dig[i] + (car[i - 1][0] if i else 0)
            shi = 15 + (car[i - 1][1] if i else 0)
            g_.append(c.node(c.t_ge(s, 16, 0, shi), f"div.g{m_}.{i}"))
            p2_.append(c.node(c.t_eq(s, 15, 0, shi), f"div.pp{m_}.{i}"))
        s9 = dig[9] + car[8][0]
        shi = 15 + car[8][1]
        u = []
        for inc in (0, 1):
            terms = []
            for theta in range(1, shi + 1):
                dd = int((theta + inc) % 16 >= 8) - int((theta - 1 + inc) % 16 >= 8)
                if dd:
                    terms += c.t_ge(s9, theta, 0, shi, dd)
            u.append(c.node(terms, f"div.u{inc}.{m_}"))
        terms = c.t_and([u[0]])
        for jn in range(9):
            srcl = [g_[jn]] + p2_[jn + 1:9]
            terms += c.t_and(srcl + [u[1]]) + c.t_and(srcl + [u[0]], -1)
        cmp_ge.append(one - c.node(terms, f"div.neg{m_}"))
    qbits = [sum(((m_ >> bit & 1) - ((m_ - 1) >> bit & 1)) * cmp_ge[m_ - 1]
                 for m_ in range(1, 16)) for bit in range(4)]
    kbits = [sum((kk[k] for k in range(8) if k >> bit & 1), Expr()) for bit in range(3)]
    qs = []
    for i in range(32):
        zero_below = [1 - qal[jn] for jn in range(i)]
        qs.append(c.node(c.t_and([1 - sigma, qal[i]]) + c.t_and([sigma, qal[i]] + zero_below)
                         + c.t_and([sigma, 1 - qal[i]])
                         + c.t_and([sigma, 1 - qal[i]] + zero_below, -1), f"div.qs{i}"))

    # ---------------------------------------------------------------- L5-7: AMO / store data
    # at a read-modify-write's D(hi) position the old word is D(lo) at t-1 and D(hi) at t
    old = Dp[1] + Db
    radd, _ = add_nibbles(c, old, B, 8, [U["AMOADD"]], "amo.add")
    mm = any_of("AMOMIN", "AMOMAX", "AMOMINU", "AMOMAXU")
    rsub, csub = add_nibbles(c, B, old, 8, [mm], "amo.sub", inv_b=True, cin=1)
    # rs2 < old, signed (exclusive literal sets) and unsigned
    lts = [[B[31], 1 - old[31]], [B[31], old[31], rsub[31]], [1 - B[31], 1 - old[31], rsub[31]]]
    ltu = 1 - csub
    lane = EA[0:2]
    wv = []
    for i in range(32):
        terms = c.t_and([any_of("SW", "SC", "AMOSWAP"), B[i]])
        for k in range(4):
            sb = B[i - 8 * k] if 8 * k <= i < 8 * k + 8 else old[i]
            terms += c.t_and([U["SB"], sb] + lits_of(lane, k))
        for hh in range(2):
            sb = B[i - 16 * hh] if 16 * hh <= i < 16 * hh + 16 else old[i]
            terms += c.t_and([U["SH"], sb] + lits_of(lane, 2 * hh))
        terms += c.t_and([U["AMOADD"], radd[i]])
        terms += c.t_and([U["AMOXOR"], old[i], 1 - B[i]]) + c.t_and([U["AMOXOR"], 1 - old[i], B[i]])
        terms += c.t_and([U["AMOAND"], old[i], B[i]])
        terms += c.t_and([U["AMOOR"]]) + c.t_and([U["AMOOR"], 1 - old[i], 1 - B[i]], -1)
        for opn, take_lt in (("AMOMIN", True), ("AMOMAX", False)):
            # take rs2 when (rs2 < old) == take_lt, else old
            terms += c.t_and([U[opn], B[i] if not take_lt else old[i]])
            for fl in lts:
                terms += c.t_and([U[opn], B[i]] + fl, 1 if take_lt else -1)
                terms += c.t_and([U[opn], old[i]] + fl, -1 if take_lt else 1)
        terms += c.t_and([U["AMOMINU"], ltu, B[i]]) + c.t_and([U["AMOMINU"], 1 - ltu, old[i]])
        terms += c.t_and([U["AMOMAXU"], 1 - ltu, B[i]]) + c.t_and([U["AMOMAXU"], ltu, old[i]])
        wv.append(c.node(terms, f"wv.b{i}"))

    # ---------------------------------------------------------------- L7: jump targets
    tgt = [c.node(c.t_and([U["JALR"], EA[i + 2]]) + c.t_and([BRANCH + U["JAL"], bt[i]]),
                  f"tgt.b{i}") for i in range(21)]

    # ---------------------------------------------------------------- L7: address decode
    aw = [c.node(c.t_and([1 - fm, EA[i + 2]]) + c.t_and([fm, W[i]]), f"aw.b{i}")
          for i in range(21)]
    in_ram = c.node(lt_terms(c, EA[0:23], cfg.ram_size - 3, [EA[31]] + [1 - b for b in EA[23:31]]),
                    "in_ram")
    mmio = c.and_(lits_of(EA[25:32], S.MMIO_LO >> 25), "mmio")
    at = {a: c.and_(lits_of(EA, a), f"ea=={a:#x}") for a in
          (S.UART_THR, S.UART_LSR, S.MTIMECMP_LO, S.MTIMECMP_HI, S.MTIME_LO, S.SYSCON)}

    # ---------------------------------------------------------------- L6-9: CSRs
    # one key head each, row 0; keys from t-1 (a CSR op that writes, or a trap / MRET / WFI
    # record), beta at F / NT (CSR ops), R (traps, MRET), WFI. Value sources (exactly one live
    # at a write): the new value from the record's D tokens (D D F|NT, or D D V F), the new
    # mstatus / privilege from a trap / MRET / WFI's D, and in-circuit values from t-1.
    vlags = {"c1": [], "c2": []}
    for i in range(32):
        if i < 16:
            vlags["c1"].append(Chan(Dp[2][i] + FNT + D1 - 2))
            vlags["c2"].append(Chan(Dp[3][i] + isF + V1 + D2 - 3))
        else:
            vlags["c1"].append(Chan(Dp[1][i - 16] + FNT + D1 - 2))
            vlags["c2"].append(Chan(Dp[2][i - 16] + isF + V1 + D2 - 3))
    csr_beta = FNT + isR + isWFI
    ok_lits = [1 - rdue, 1 - TR1, 1 - TR2]  # t-1 belongs to an instruction record
    c.csr_cols = {}

    def csr_head(name, key, nbits=32, extra=None, std=True, beta=csr_beta, vheads=1):
        dims = c.zeros(nbits, f"csr.{name}")
        srcs = {i: ([vlags["c1"][i], vlags["c2"][i]] if std else []) for i in range(nbits)}
        for i, chs in (extra or {}).items():
            srcs[i] = srcs[i] + chs
        v, z, outs, into, col, cols = {}, {}, [], [], 2, {}
        for i in range(nbits):
            for ch in srcs[i]:
                v[col] = chan(ch, A_BIT)
                z[col] = K * one
                outs.append((col, OUT_W))
                into.append(dims[i])
                cols.setdefault(i, col)
                col += 1
        assert col <= VD, (name, col)
        for rc in REF_COLS:
            v[rc] = chan(one, A_REF)
        c.csr_cols[name] = cols
        c.table(f"csr.{name}", {0: one}, {0: Chan(key, LAG1)},
                [{"v": v, "z": z, "outs": outs, "into": into, "beta": beta}])
        return dims

    def key_node(name, csr=None, trap=False, mret=False, wfi=False):
        terms = c.t_and([wr, csr_oh[csr]] + ok_lits) if csr else []
        if trap:
            terms += c.t_and([trapw])
        if mret:
            terms += c.t_and([U["MRET"]] + ok_lits)
        if wfi:
            terms += c.t_and([U["WFI"]] + ok_lits)
        return c.node(terms, f"key.{name}")

    ms_extra = {}
    for i in range(16):  # rw1: D at t-1 (R or WFI); rw2: JH at t-1, D at t-2 (R)
        ms_extra[i] = [Chan(Dp[1][i] + (isR if i < 13 else 0) + isWFI - 1)]
        if i < 13:
            ms_extra[i].append(Chan(Dp[2][i] + isR + JH1 - 2))
    ms = csr_head("mstatus", key_node("mstatus", "mstatus", True, True, True), extra=ms_extra)
    mie = csr_head("mie", key_node("mie", "mie"))
    mtvec = csr_head("mtvec", key_node("mtvec", "mtvec"))
    mscratch = csr_head("mscratch", key_node("mscratch", "mscratch"))
    mip = csr_head("mip", key_node("mip", "mip"))
    mepc = csr_head("mepc", key_node("mepc", "mepc", True),
                    extra={i: [Chan(pcT[i], LAG1)] for i in range(32) if pcT[i].terms})
    mcause = csr_head("mcause", key_node("mcause", "mcause", True),
                      extra={b: [Chan(e, taps(2, 3))] for b, e in mcb.items()})
    priv = csr_head("priv", key_node("priv", trap=True, mret=True), nbits=2, std=False,
                    beta=isR, extra={b: [Chan(Dp[1][13 + b] + isR - 1),
                                         Chan(Dp[2][13 + b] + isR + JH1 - 2)] for b in range(2)})
    # mtimecmp: key at any store (beta MTL / MTH picks the half), value rs2 from t-1
    mt = c.table("match", {0: one}, {0: Chan(G[0x23], LAG1)},
                 [{"v": {**{2 + i: chan(Chan(B[i], LAG1), A_BIT) for i in range(32)},
                         **{rc: chan(one, A_REF) for rc in REF_COLS}},
                   "z": {2 + i: K * one for i in range(32)},
                   "outs": [(2 + i, OUT_W) for i in range(32)], "beta": fam[nm]}
                  for nm in ("MTL", "MTH")])
    mlo, mhi = mt[:32], mt[32:]
    # mtval: an access fault's address (a load / store / AMO outside RAM and MMIO), else the PC
    vmem = G[0x03] + G[0x23] + G[0x2F]
    tval = []
    for i in range(32):
        terms = c.t_and([trapx, vmem, 1 - in_ram, EA[i]])
        if pc_bits[i].terms:
            terms += c.t_and([trapx, pc_bits[i]]) + c.t_and([trapx, vmem, 1 - in_ram,
                                                              pc_bits[i]], -1)
        tval.append(c.node(terms, f"tval.b{i}"))
    mtval = csr_head("mtval", key_node("mtval", "mtval", True),
                     extra={i: [Chan(tval[i], LAG1)] for i in range(32)})
    # the LR reservation (word index), written at an LR's F / NT
    res = c.table("res", {0: one}, {0: Chan(U["LR"], LAG1)},
                  [{"v": {**{2 + i: chan(Chan(EA[i + 2], LAG1), A_BIT) for i in range(21)},
                          **{rc: chan(one, A_REF) for rc in REF_COLS}},
                    "z": {2 + i: K * one for i in range(21)},
                    "outs": [(2 + i, OUT_W) for i in range(21)], "beta": FNT}])

    # ---------------------------------------------------------------- the timer, interrupts
    tsub, tge = add_nibbles(c, T, mlo, 8, [], "mtip.sub", inv_b=True, cin=1)  # [T >= mlo]
    eqm = eq_bits(c, T, mlo, "mtip.eq")
    mh0 = c.and_([1 - b for b in mhi], "mh0")
    mlz = c.and_([1 - b for b in mlo], "mlz")
    mtip0 = c.and_([mh0, 1 - mlz, tge, 1 - eqm], "mtip0")  # T > mtimecmp != 0
    mtip1 = c.and_([mh0, 1 - mlz, tge], "mtip1")  # T + 1 > mtimecmp != 0
    sleep = c.node(c.t_and([pw, nt64, 1 - mtip1]) + c.t_and([pw, 1 - nt64, 1 - mtip0]), "sleep")
    tickc = c.node(c.t_and([nt64]) + c.t_and([sleep, 1 - nt64]), "tick")
    intrc = c.and_([mtip0, mie[7], ms[3], 1 - tickc], "intr")
    refrc = c.and_([rdue, 1 - tickc, 1 - intrc], "refresh")
    instr = one - tickc - intrc - refrc
    mp1 = inc_bits(c, mlo, "m+1")
    newT = [c.node(c.t_and([sleep, mp1[i]]) + c.t_and([1 - sleep, T1[i]]), f"newT.b{i}")
            for i in range(32)]
    eqhi = eq_bits(c, mp1[16:32], thi, "m+1.hi")
    needDT = c.node(c.t_and([1 - sleep] + tlo) + c.t_and([sleep, 1 - eqhi]), "needDT")

    # ---------------------------------------------------------------- L8..: RAM
    # word w (21 bits): key head = 2048-word block w[20:11], value head w[2:0], row w[10:3];
    # writes: key from t-1 (any store / AMO: beta ST<x> only follows one to RAM), value wv
    mem = c.zeros(32, "mem")
    ramw = G[0x23] + G[0x2F]
    first_blk = TW // RAM_HEAD_WORDS
    n_blk = -(-cfg.ram_words // RAM_HEAD_WORDS)
    c.ram_blocks = list(range(first_blk, n_blk))
    for blk in c.ram_blocks:
        hl = lits_of(aw[11:21], blk)
        keys, q = {}, {}
        for row in range(256):
            rl = lits_of(aw[3:11], row)
            keys[row] = Chan(andpre(rl + hl + [ramw]), LAG1)
            q[row] = andpre(rl + hl)
        vhs = []
        for x in range(8):
            v = {2 + i: chan(Chan(wv[i], LAG1), A_BIT) for i in range(32)}
            for rc in REF_COLS:
                v[rc] = chan(one, A_REF)
            zx = K * andpre(lits_of(aw[0:3], x))
            vhs.append({"v": v, "z": {2 + i: zx for i in range(32)},
                        "outs": [(2 + i, OUT_W) for i in range(32)], "into": mem,
                        "beta": fam[f"ST{x}"]})
        c.table(f"ram{blk}", q, keys, vhs)

    # ---------------------------------------------------------------- compares, faults
    b31 = {"SUB": B[31], "SUBI": imm_i[31]}
    slt = c.node([t for m_ in b31 for t in c.t_and([md[m_], A[31], 1 - b31[m_]])
                  + c.t_and([md[m_], A[31], b31[m_], r[31]])
                  + c.t_and([md[m_], 1 - A[31], 1 - b31[m_], r[31]])], "slt")
    sltu = c.node(c.t_and([md["SUB"] + md["SUBI"], 1 - c32]), "sltu")
    eq = c.and_([1 - b for b in r], "eq")
    taken = c.node(c.t_and([U["BEQ"], eq]) + c.t_and([U["BNE"], 1 - eq])
                   + c.t_and([U["BLT"], slt]) + c.t_and([U["BGE"], 1 - slt])
                   + c.t_and([U["BLTU"], sltu]) + c.t_and([U["BGEU"], 1 - sltu]), "taken")
    jmp = c.node(c.t_and([U["JAL"] + U["JALR"]]) + c.t_and([BRANCH, taken]), "jmp")
    resm = eq_bits(c, res, EA[2:23], "sc.match")
    sc_ok = c.and_([U["SC"], in_ram, resm], "sc.ok")
    pz = c.and_([1 - priv[0], 1 - priv[1]], "priv==0")
    nr = 1 - in_ram
    causes = {2: [[ILL_ST], [U["LDBAD"], in_ram], [U["STBAD"], in_ram], [AMOBAD, in_ram]],
              3: [[U["EBREAK"]]], 5: [[G[0x03], nr, 1 - mmio]],
              7: [[G[0x23], nr, 1 - mmio], [G[0x2F], nr]], 8: [[U["ECALL"], pz]],
              11: [[U["ECALL"], 1 - pz]]}
    exc = c.node([t for ls in causes.values() for lits in ls for t in c.t_and(lits)], "exc")
    cause = [c.node([t for cv, ls in causes.items() if cv >> b & 1 for lits in ls
                     for t in c.t_and(lits)], f"cause.b{b}") for b in range(4)]
    dtrap = [zero] * 16
    dtrap[7], dtrap[11], dtrap[12], dtrap[13], dtrap[14] = ms[3], priv[0], priv[1], one, one
    dmret = [zero] * 16
    dmret[3], dmret[7], dmret[11], dmret[12], dmret[13], dmret[14] = (ms[7], one, priv[0],
                                                                       priv[1], ms[11], ms[12])
    dwfi = list(ms[0:16])
    dwfi[3] = one
    mtvw, mepcw = mtvec[2:23], mepc[2:23]

    def needjh(bits, name):  # a target's high word bits differ from the anchor's
        return 1 - eq_bits(c, bits, ahi, name)

    nj_tv, nj_mepc = needjh(mtvw[16:21], "nj.mtvec"), needjh(mepcw[16:21], "nj.mepc")
    nj_W, nj_t = needjh(W[16:21], "nj.w"), needjh(tgt[16:21], "nj.tgt")

    # ---------------------------------------------------------------- CSR read / write
    csr_dims = {"mstatus": ms, "mie": mie, "mtvec": mtvec, "mscratch": mscratch, "mepc": mepc,
                "mcause": mcause, "mtval": mtval, "mip": mip}
    csr_old = []
    for i in range(32):
        terms = []
        for k, dims in csr_dims.items():
            terms += c.t_and([csr_oh[k], mtip0 if k == "mip" and i == 7 else dims[i]])
        for v, e in csr_const.items():
            if v >> i & 1:
                terms += c.t_and([e])
        csr_old.append(c.node(terms, f"csr.old.b{i}"))
    zimm = rs1b + [zero] * 27
    csr_new = []
    for i in range(32):
        o_ = csr_old[i]
        terms = c.t_and([U["CSRRW"], A[i]]) + c.t_and([U["CSRRS"], o_]) + c.t_and(
            [U["CSRRS"], 1 - o_, A[i]]) + c.t_and([U["CSRRC"], o_, 1 - A[i]])
        terms += c.t_and([U["CSRRSI"], o_]) + c.t_and([U["CSRRCI"], o_])
        if i < 5:
            terms += (c.t_and([U["CSRRWI"], zimm[i]]) + c.t_and([U["CSRRSI"], 1 - o_, zimm[i]])
                      + c.t_and([U["CSRRCI"], o_, zimm[i]], -1))
        csr_new.append(c.node(terms, f"csr.new.b{i}"))
    csrwr = c.and_([wr, sum(csr_oh.values(), Expr())], "csr.wr.stored")  # D D comes first

    # ---------------------------------------------------------------- the rd value
    addish = any_of("ADD", "ADDI", "SUB")
    sltish, sltuish = any_of("SLT", "SLTI"), any_of("SLTU", "SLTIU")
    logicops = any_of("AND", "ANDI", "OR", "ORI", "XOR", "XORI")
    shiftops = any_of("SLL", "SLLI", "SRL", "SRLI", "SRA", "SRAI")
    mulhs = any_of("MULH", "MULHSU", "MULHU")
    ldm = [G[0x03], nr, mmio]  # an MMIO load (any width)
    srcs = [([addish], r), ([U["AUIPC"]], auipc), ([U["LUI"]], imm_u), ([logicops], logic),
            ([shiftops], shift),
            ([U["MUL"]], mul_lo), ([mulhs], mul_hi), ([any_of("DIV", "DIVU")], qs),
            ([remc], mul_lo), ([csrop], csr_old), ([U["LW"], in_ram] + lits_of(lane, 0), mem),
            ([U["LR"], in_ram], mem), (ldm + [at[S.UART_LSR]], [rx_wait] + [zero] * 4 + [one, one]),
            (ldm + [at[S.MTIME_LO]], T)]
    for k in range(4):
        lk = lits_of(lane, k)
        byte = mem[8 * k:8 * k + 8]
        srcs.append(([U["LBU"], in_ram] + lk, byte))
        srcs.append(([U["LB"], in_ram] + lk, byte + [byte[7]] * 24))
        if k % 2 == 0:
            half = mem[8 * k:8 * k + 16]
            srcs.append(([U["LHU"], in_ram] + lk, half))
            srcs.append(([U["LH"], in_ram] + lk, half + [half[15]] * 16))
    small = [[sltish], [sltuish], [U["SC"], in_ram, 1 - resm]]  # 0 / 1 values
    val, fit_t = [], []
    for i in range(32):
        terms = []
        for gate, bits in srcs:
            if i < len(bits) and not isinstance(bits[i], int) and bits[i].terms:
                terms += c.t_and(gate + [bits[i]])
        if i == 0:
            terms += (c.t_and([slt, sltish]) + c.t_and([sltu, sltuish])
                      + c.t_and(small[2]))
        val.append(c.node(terms, f"val.b{i}"))
    for gate, bits in srcs:  # fits16: bits 15..31 all equal
        hb = [bits[i] if i < len(bits) else zero for i in range(15, 32)]
        if all(b.terms for b in hb):
            fit_t += c.t_and(gate + hb)
        fit_t += c.t_and(gate + [1 - b for b in hb if b.terms])
    for g_ in small:
        fit_t += c.t_and(g_)
    # the other MMIO loads (mtime high, the UART's other registers, an empty RBR) read 0
    fit_t += (c.t_and(ldm) + c.t_and(ldm + [at[S.UART_LSR]], -1)
              + c.t_and(ldm + [at[S.MTIME_LO]], -1))
    fits = c.node(fit_t, "fits16")
    valx = c.node(c.t_and([addish + U["AUIPC"] + U["LUI"] + logicops + shiftops + sltish
                           + sltuish + mul_any])
                  + c.t_and([LOADV, in_ram]) + c.t_and(ldm)
                  + c.t_and(ldm + [at[S.UART_THR], rx_wait], -1) + c.t_and([U["LR"], in_ram])
                  + c.t_and(small[2]) + c.t_and([csrop, 1 - csrwr]), "valx")
    rmw = c.node(c.t_and([U["SB"] + U["SH"], in_ram]) + c.t_and([any_of(*amo_ops), in_ram]),
                 "rmw")
    stm = c.and_([G[0x23], nr, mmio], "st.mmio")

    # ---------------------------------------------------------------- the next token
    es = c.node(c.t_and([bnd, instr, 1 - ramcode]) + c.t_and([PIH]), "es")  # exec starts
    fetch = c.and_([bnd, instr, ramcode], "fetch")
    okd = c.and_([instr, 1 - exc], "okd")
    E = [es, 1 - exc]
    cases = []

    def case(lits, fam_name, payload=0):
        cases.append((lits, fam_name, payload))

    case([bnd, tickc, needDT], "D", newT[16:32])
    case([bnd, tickc, 1 - needDT], "TK", newT[0:16])
    case([j1, tickc], "TK", newT[0:16])
    case([bnd, intrc], "TRAP", S.TRAP_INT)
    for lits in ([j1, intrc], [j1, instr, exc]):
        case(lits, "D", dtrap)
    for jx, jy, ctx in ((j2, j3, [intrc]), (j2, j3, [instr, exc])):
        case([jx, nj_tv] + ctx, "JH", mtvw[16:21])
        case([jx, 1 - nj_tv] + ctx, "R", mtvw[0:16])
        case([jy] + ctx, "R", mtvw[0:16])
    case([bnd, refrc, nj_W], "JH", W[16:21])
    case([bnd, refrc, 1 - nj_W], "R", W[0:16])
    case([j1, refrc], "R", W[0:16])
    case([fetch], "I", mem[0:16])
    case([PIL], "I", mem[16:32])
    case([es, exc], "TRAP", cause)
    case(E + [valx, rd_nz, fits], "F", val[0:16])
    case(E + [valx, rd_nz, 1 - fits], "V", val[0:16])
    case([j1, okd, valx, rd_nz], "F", val[16:32])
    case(E + [valx, 1 - rd_nz], "NT")
    case(E + ldm + [at[S.UART_THR], rx_wait], "RX", rxb)
    case(E + [csrwr], "D", csr_new[0:16])
    case([j1, okd, csrwr], "D", csr_new[16:32])
    case([j2, okd, csrwr, rd_nz, fits], "F", val[0:16])
    case([j2, okd, csrwr, rd_nz, 1 - fits], "V", val[0:16])
    case([j3, okd, csrwr, rd_nz], "F", val[16:32])
    case([j2, okd, csrwr, 1 - rd_nz], "NT")
    case(E + [divc, rd_nz], "QS", k0b)
    case(E + [divc, 1 - rd_nz], "NT")
    case([okd, divc, dp], "QD", qbits + kbits)
    dv = [okd, divc, 1 - dp, inrec]
    case(dv + [v0, fits], "F", val[0:16])
    case(dv + [v0, 1 - fits], "V", val[0:16])
    case(dv + [v1], "F", val[16:32])
    case(E + [jmp, nj_t], "JH", tgt[16:21])
    case(E + [jmp, 1 - nj_t], "J", tgt[0:16])
    case([j1, okd, jmp], "J", tgt[0:16])
    case(E + [BRANCH, 1 - taken], "NT")
    case(E + [U["FENCE"]], "NT")
    case(E + [rmw], "D", mem[0:16])
    case([j1, okd, rmw], "D", mem[16:32])
    for x in range(8):
        xl = lits_of(EA[2:5], x)
        case(E + [U["SW"], in_ram] + xl, f"ST{x}")
        case(E + [sc_ok] + xl, f"ST{x}")
        case([j2, okd, rmw] + xl, f"ST{x}")
    case(E + [stm, at[S.UART_THR]], "BYTE", B[0:8])
    case(E + [stm, at[S.MTIMECMP_LO]], "MTL")
    case(E + [stm, at[S.MTIMECMP_HI]], "MTH")
    case(E + [stm, at[S.SYSCON], 1 - bz], "HALT")
    case(E + [stm, at[S.SYSCON], bz], "NT")
    case(E + [stm, 1 - at[S.UART_THR] - at[S.MTIMECMP_LO] - at[S.MTIMECMP_HI]
              - at[S.SYSCON]], "NT")
    case(E + [U["MRET"]], "D", dmret)
    case([j1, okd, U["MRET"], nj_mepc], "JH", mepcw[16:21])
    case([j1, okd, U["MRET"], 1 - nj_mepc], "R", mepcw[0:16])
    case([j2, okd, U["MRET"]], "R", mepcw[0:16])
    case(E + [U["WFI"]], "D", dwfi)
    case([j1, okd, U["WFI"]], "WFI")

    fam_cases = [(l_, FAM_ID[fn]) for l_, fn, _ in cases]
    pay_cases = [(l_, pl) for l_, _, pl in cases]
    c.cases = cases
    fam_bits = [c.node(sel_terms(c, fam_cases, i), f"exit.fam{i}") for i in range(NFAM)]
    pay_bits = [c.node(sel_terms(c, pay_cases, i), f"exit.pay{i}") for i in range(NPAY)]
    # a coverage dim: exactly 1 wherever the next token is the circuit's (0 means no case)
    c.cover = c.node([t for l_, _, _ in cases for t in c.t_and(l_)], "exit.cover")
    c.out_fam, c.out_pay = fam_bits, pay_bits
    exit_min = max(c.level(b) for b in fam_bits + pay_bits) - 1
    for e in c.code:
        c.ffn([(-1, one, e), (1, one, -1 * e)], "code.clear", into=e)
    for i, e in enumerate(c.code):
        cs, bit = (fam_cases, i) if i < NFAM else (pay_cases, i - NFAM)
        terms = c.t_const(-cfg.kappa)
        for t in sel_terms(c, cs, bit):
            terms.append((t[0] * 2 * cfg.kappa, t[1], t[2]))
        c.ffn(terms, f"exit.code{i}", into=e, min_level=exit_min)
    c.keep = [next(iter(e.terms)) for e in c.code]
    c.probe = {"W": W, "anchor": abits, "T": T, "ib": ib, "A": A, "B": B, "EA": EA, "aw": aw,
               "mem": mem, "val": val, "fits": fits, "exc": exc, "cause": cause, "tick": tickc,
               "intr": intrc, "refresh": refrc, "es": es, "ms": ms, "mie": mie, "mtvec": mtvec,
               "mepc": mepc, "mcause": mcause, "mtval": mtval, "mip": mip, "priv": priv,
               "mlo": mlo, "mhi": mhi, "res": res, "wv": wv, "csr_old": csr_old,
               "csr_new": csr_new, "tgt": tgt, "taken": taken, "qal": qal, "pw": pw,
               "nt64": nt64, "mtip0": mtip0, "sleep": sleep, "newT": newT, "valx": valx,
               "rmw": rmw, "in_ram": in_ram, "mmio": mmio, "ramcode": ramcode, "fm": fm,
               "lag": lagd, "uops": U}
    return c


# --- snapshots (the state the host injects) ---------------------------------------------------


def _row(cols_bits: dict) -> np.ndarray:
    """A state row: the ballast plus A_BIT in the given columns."""
    row = np.zeros(VD)
    row[list(REF_COLS)] = A_REF
    for col in cols_bits:
        row[col] = A_BIT
    return row


def _bits_row(value: int, n: int, col0: int = 2) -> np.ndarray:
    return _row([col0 + i for i in range(n) if value >> i & 1])


def counter_rows(k: int) -> dict:
    r0 = np.zeros(VD)
    r0[2] = A_BIT * 2.0**-k
    return {0: r0, 1: _row([])}


_ROM_CACHE: dict = {}


def rom_rows(c: LxCore, rom: bytes) -> list:
    """Per ROM key head, the 8 value heads' rows (nibbles of 8 words per row)."""
    key = (hash(rom), c.cfg.text_words)
    if _ROM_CACHE.get("key") == key:
        return _ROM_CACHE["rows"]
    words = np.frombuffer(rom[:4 * c.cfg.text_words], dtype="<u4").astype(np.int64)
    out = []
    for h in range(-(-len(words) // ROM_HEAD_WORDS)):
        vh = []
        for x in range(8):
            base = h * ROM_HEAD_WORDS + x * RAM_HEAD_WORDS
            blk = words[base:base + RAM_HEAD_WORDS]
            rows = {}
            for r in range(-(-len(blk) // 8)):
                row = np.zeros(VD)
                row[list(REF_COLS)] = A_REF
                for k in range(8):
                    if 8 * r + k < len(blk):
                        wd = int(blk[8 * r + k])
                        for nb in range(8):
                            row[2 + 8 * k + nb] = wd >> 4 * nb & 15
                rows[r] = row
            vh.append(rows)
        out.append(vh)
    _ROM_CACHE.update(key=key, rows=out)
    return out


def snapshot(c: LxCore, m) -> None:
    """Load a reference machine's state (lx.ref.Machine, at a record boundary) into the
    circuit's tables. The trace from here starts with GO."""
    t = c.tables
    t["anchor.lo"].init = [{0: _bits_row(m.anchor & 0xFFFF, 16)}]
    t["anchor.hi"].init = [{0: _bits_row(m.anchor >> 16, 5)}]
    t["count"].init = [counter_rows(m.n)]
    t["j"].init = [counter_rows(0)]
    t["vcnt"].init = [counter_rows(0)]
    t["ntick"].init = [counter_rows(m.nt)]
    t["time.lo"].init = [{0: _bits_row(m.T & 0xFFFF, 16)}]
    t["time.hi"].init = [{0: _bits_row(m.T >> 16, 16)}]
    t["wfi"].init = [{0: _bits_row(int(m.post_wfi), 1)}]
    t["ir"].init = None
    regs = [{}, {}]
    for rr in range(1, 32):
        for half in range(2):
            regs[half][rr] = _row([c.reg_cols[i] for i in range(16 * half, 16 * half + 16)
                                   if m.x[rr] >> i & 1])
    t["regs.rs1"].init = regs
    t["regs.rs2"].init = regs
    for k in S.CSRS:
        cols = c.csr_cols[k]
        t[f"csr.{k}"].init = [{0: _row([cols[i] for i in range(32) if m.csr[k] >> i & 1])}]
    t["csr.priv"].init = [{0: _row([c.csr_cols["priv"][i] for i in range(2)
                                     if m.priv >> i & 1])}]
    t["match"].init = [{0: _bits_row(m.mlo, 32)}, {0: _bits_row(m.mhi, 32)}]
    t["res"].init = [{0: _bits_row(m.res >> 2 & 0x1FFFFF, 21)}]
    t["rx.w"].init = [counter_rows(m.rx_w)]
    t["rx.r"].init = [counter_rows(m.rx_r)]
    t["rx"].init = [{k: _bits_row(m.rx_buf[k], 8) for k in range(m.rx_w + 1)}]
    for h, vh in enumerate(rom_rows(c, bytes(m.rom))):
        t[f"rom{h}"].init = vh
    words = np.frombuffer(bytes(m.ram), dtype="<u4")
    for blk in c.ram_blocks:
        vh = [{} for _ in range(8)]
        seg = words[blk * RAM_HEAD_WORDS:(blk + 1) * RAM_HEAD_WORDS]
        for wi in np.nonzero(seg)[0]:
            wd = int(seg[wi])
            vh[wi & 7][wi >> 3] = _bits_row(wd, 32)
        t[f"ram{blk}"].init = vh


def decode_code(fid: int, p: int) -> int:
    if not 0 <= fid < len(OUT_NAMES):
        return -1
    nm = OUT_NAMES[fid]
    if nm in S.CONTROLS:
        return S.tok(nm)
    return S.tok(nm, p) if p < S.FAMILIES[nm][1] else -1


def _ev(e: Expr, mat: np.ndarray) -> np.ndarray:
    if not e.terms:
        return np.zeros(mat.shape[0])
    return sum(mat[:, d] * float(w_) for d, w_ in e.terms.items())


def predictions(c: LxCore, mat: np.ndarray) -> list[int]:
    """The predicted next token at every position (from the exit's select dims)."""
    fb = [np.rint(_ev(e, mat)).astype(int) for e in c.out_fam]
    pb = [np.rint(_ev(e, mat)).astype(int) for e in c.out_pay]
    out = []
    for t in range(mat.shape[0]):
        fid = sum(int(fb[i][t]) << i for i in range(NFAM))
        p = sum(int(pb[i][t]) << i for i in range(NPAY))
        out.append(decode_code(fid, p))
    return out


def probe(c: LxCore, mat: np.ndarray, name: str, t: int) -> int:
    """The integer a probe's bit list holds at position t (debugging)."""
    bits = c.probe[name]
    if isinstance(bits, Expr):
        return int(round(_ev(bits, mat)[t]))
    return sum(int(round(_ev(b, mat)[t])) << i for i, b in enumerate(bits))
