"""The chip layers' decode as one persistent CUDA kernel (csrc/mega.cu): every token's 37 layers
and the output head in one launch, the argmax fed back as the next input on the GPU. The same
weights as lx.fast's decode (from its packed layers, which are the stock layers' nonzero parts),
the same cache tensors (updated in place), the same rounding.

    m = Mega(fast, model, cache, offset)
    preds = m.run(forced, n)   # n steps: inputs forced[0..], then each step's own prediction
"""

import ctypes
import functools
from dataclasses import dataclass

import numpy as np
import torch

from lx import mk_build

NT = 512
KD, VD, R = 256, 128, 8
HEAD_ROWS = 3584
MAXCU = 2048

I32 = ctypes.c_int
F32 = ctypes.c_float
PTR = ctypes.c_void_p


class Phase(ctypes.Structure):
    _fields_ = [("kind", I32), ("layer", I32), ("eps", F32), ("cu_off", I32), ("n_cu", I32),
                ("upd_off", I32), ("n_upd", I32), ("item_off", I32), ("n_items", I32),
                ("n_units", I32), ("row_off", I32), ("d_off", I32), ("f64", I32),
                ("grp_n", I32), ("pair", I32)]


class Item(ctypes.Structure):
    _fields_ = [("idx", I32), ("ch_off", I32), ("n_ch", I32), ("nvh", I32), ("kh", I32),
                ("beta_row", I32), ("a_row", I32), ("z_row", I32), ("hv_off", I32),
                ("dim_off", I32), ("bm_off", I32), ("gw_off", I32), ("geps", F32),
                ("n_cu", I32), ("nv", I32)]


class Params(ctypes.Structure):
    _fields_ = [("phases", PTR), ("n_phases", I32), ("chans", PTR), ("blob", PTR),
                ("res_off", PTR), ("res_len", PTR), ("sched", PTR),
                ("rowptr", PTR), ("nz", PTR), ("cols", PTR), ("w1", PTR), ("upd_rows", PTR),
                ("dptr", PTR), ("dnz", PTR), ("a_neg", PTR), ("dt_bias", PTR), ("gnorm_w", PTR),
                ("bitmaps", PTR), ("rs", PTR), ("cs", PTR), ("n_layers", I32),
                ("n_dense", I32), ("n_sparse", I32),
                ("hdense", PTR), ("hsparse", PTR), ("htab", PTR), ("hptr", PTR), ("hnz", PTR),
                ("head_rows", I32), ("n_vocab", I32), ("embed", PTR), ("offset", I32),
                ("forced", PTR), ("n_forced", I32), ("n_tokens", I32), ("preds", PTR),
                ("acc", PTR), ("bar", PTR), ("part_v", PTR), ("part_i", PTR), ("dbg", PTR),
                ("dbg_tok", I32), ("dbg_items", PTR), ("prof", PTR), ("wsave", PTR),
                ("accs", PTR), ("inputs", PTR)]


def _bits(t: torch.Tensor) -> torch.Tensor:
    """bf16 values as their 16 bits, in int64."""
    return t.contiguous().view(torch.int16).to(torch.int64) & 0xffff


def _i32(v: torch.Tensor) -> torch.Tensor:
    """Values in [0, 2^32) (int64) as the int32 with their bits."""
    return torch.where(v >= 1 << 31, v - (1 << 32), v).to(torch.int32)


def _pack(idx: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
    """(index < 2^16, bf16 value) as one uint32 (stored in int32)."""
    assert idx.numel() == 0 or int(idx.max()) < 1 << 16
    return _i32(idx.to(torch.int64) | (_bits(val) << 16))


@dataclass
class _Rows:
    """A growing CSR (rows of (column, bf16 weight)), built in host memory."""
    ptr: list
    parts: list
    n: int = 0

    def add(self, w: torch.Tensor) -> int:
        """Append the rows of the dense bf16 matrix w; returns the first row's id."""
        first = len(self.ptr) - 1
        nzi = w.nonzero()
        counts = torch.bincount(nzi[:, 0], minlength=w.shape[0]) if len(nzi) else \
            torch.zeros(w.shape[0], dtype=torch.long, device=w.device)
        cum = torch.cumsum(counts, 0) + self.n
        self.ptr.extend(cum.tolist())
        self.parts.append(_pack(nzi[:, 1], w[nzi[:, 0], nzi[:, 1]]).cpu())
        self.n += len(nzi)
        return first


IT_VTAB, IT_META, NVTAB, TZERO = 64, 320, 64, 4095
UPR = 4
# GDN layers whose step seldom adds to the residual (measured): their MLP runs alongside
PAIR_LAYERS = (3, 7, 8, 9, 10, 11, 12, 13)
MAXDENSE, MAXSPARSE, MAXPH = 32, 256, 64


def _pad16(b: bytes) -> bytes:
    return b + bytes(-len(b) % 16)


def _normed(cols: np.ndarray, w1: np.ndarray, body: bytes) -> bytes:
    """A resident blob: its norm's columns (residual rows, weights), then the body."""
    c = _pad16(np.asarray(cols).astype(np.int16).tobytes())
    w = _pad16(np.asarray(w1).astype(np.float32).tobytes())
    assert len(cols) <= MAXCU
    return np.array([len(cols), 16 + len(c) + len(w), 0, 0], np.int32).tobytes() + c + w + body


def _ranges(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """The concatenation of arange(s, s + c) over the pairs (s, c)."""
    tot = int(counts.sum())
    if tot == 0:
        return np.zeros(0, np.int64)
    first = np.repeat(np.cumsum(counts) - counts, counts)
    return np.repeat(starts, counts) + np.arange(tot) - first


def _sparse_np(w: torch.Tensor):
    """Row-major nonzeros of a 2-D bf16 tensor: columns, float values, bf16 bits; per-row counts
    and first entries."""
    nzi = w.nonzero()
    cols = nzi[:, 1].cpu().numpy()
    v = w[nzi[:, 0], nzi[:, 1]]
    counts = np.bincount(nzi[:, 0].cpu().numpy(), minlength=w.shape[0])
    return (cols, v.float().cpu().numpy(), _bits(v).cpu().numpy(), counts,
            np.cumsum(counts) - counts)


def _item_blob(it: Item, cols, vals, counts, starts, taps, slot, ch) -> bytes:
    """A GDN item's resident blob (csrc/mega.cu): per channel (by count, descending) its
    projection row (the layer's entries starts..starts + counts), conv taps (n, 4), slot (-1 none),
    conv channel. Sets it.nv."""
    n = len(counts)
    assert n <= 2 * NT and np.all(np.diff(counts) <= 0)
    idx = _ranges(starts, counts)
    ev, ec = vals[idx], cols[idx]
    nzt = taps != 0
    ntap = nzt.sum(1)
    table = np.unique(np.concatenate([ev, taps[nzt]]).astype(np.float32))
    assert len(table) <= NVTAB, len(table)
    it.nv = len(table)
    code = np.searchsorted(table, ev) * it.n_cu + ec.astype(np.int64)  # code-major: banks by column
    assert it.n_cu * it.nv <= TZERO
    assert counts.max(initial=0) < 32 and ch.max(initial=0) < 1 << 15
    # thread t's channels: t (round 0) and 2 NT - 1 - t (round 1); block (round, warp) of entries
    c = np.arange(n)
    thr = np.where(c < NT, c, 2 * NT - 1 - c)
    blk, lane = (c >= NT) * 16 + thr // 32, thr % 32
    mx = np.zeros(32, np.int64)
    np.maximum.at(mx, blk, counts)
    mx = (mx + 3) // 4 * 4
    wbase = np.concatenate([[0], np.cumsum(mx * 32)[:-1]])
    e_ch = np.repeat(c, counts)
    e_k = np.arange(len(idx)) - np.repeat(np.cumsum(counts) - counts, counts)
    ent = np.full(int(mx.sum()) * 32, TZERO, np.uint16)
    ent[wbase[blk[e_ch]] + e_k * 32 + lane[e_ch]] = code
    assert np.all(mx[16:] <= mx[:16])
    single = ntap == 1
    ti = np.where(single, nzt.argmax(1), 0)
    tc = np.where(single, np.searchsorted(table, taps[np.arange(n), ti]), 0)
    u = lambda x: np.asarray(x).astype(np.uint64)  # noqa: E731
    meta = (u(counts) | (u(np.where(slot < 0, 0x7ff, slot)) << u(19))
            | (u(ti) << u(30)) | (u(tc) << u(32)) | (u(~single) << u(38)) | (u(ch) << u(39)))
    vt = np.zeros(NVTAB, np.float32)
    vt[:len(table)] = table
    head = bytes(it).ljust(IT_VTAB, b"\0")
    assert len(head) == IT_VTAB and IT_VTAB + 4 * NVTAB == IT_META
    return _pad16(head + vt.tobytes() + meta.tobytes() + wbase.astype(np.int32).tobytes()
                  + mx.astype(np.int32).tobytes() + ent.tobytes())


def _mlp_blob(a: int, b: int, gu, n_u: int, drow: np.ndarray, dbits: np.ndarray,
              cin: np.ndarray, w1: np.ndarray) -> bytes:
    """An MLP part's resident blob (csrc/mega.cu): units a..b-1, each its gate row then up row;
    its norm over the columns these rows use (cin, w1: the layer's)."""
    cols, _, bits, counts, starts = gu
    us = np.arange(a, b)
    gc, uc = counts[us], counts[n_u + us]
    idx = _ranges(np.stack([starts[us], starts[n_u + us]], 1).reshape(-1),
                  np.stack([gc, uc], 1).reshape(-1))
    used = np.unique(cols[idx])
    local = np.searchsorted(used, cols[idx])
    ent = local.astype(np.uint32) | (bits[idx].astype(np.uint32) << np.uint32(16))
    e0 = np.cumsum(gc + uc) - (gc + uc)
    assert gc.max(initial=0) < 256 and uc.max(initial=0) < 256 and len(idx) < 1 << 16
    assert cols.max(initial=0) < 1 << 16 and drow.max(initial=0) < 1 << 16
    u = lambda x: np.asarray(x).astype(np.uint64)  # noqa: E731
    meta = (u(e0) | (u(gc) << u(16)) | (u(uc) << u(24)) | (u(drow[us]) << u(32))
            | (u(dbits[us]) << u(48)))
    head = np.array([a, b, len(idx), 0], np.int32)
    return _normed(cin[used], w1[used], _pad16(head.tobytes() + meta.tobytes() + ent.tobytes()))


class Mega:
    def __init__(self, fast, model, cache, offset: int, ncta: int | None = None,
                 subprof: bool = False, spec: bool = True,
                 pair_layers: tuple[int, ...] = PAIR_LAYERS) -> None:
        self.lib = mk_build.load("mega", ["-DMK_SUBPROF"] if subprof else [])
        dev = fast.w_head.device
        self.dev, self.fast, self.model, self.cache, self.offset = dev, fast, model, cache, offset
        self.ncta = ncta or torch.cuda.get_device_properties(dev).multi_processor_count
        assert self.ncta >= 1
        gdn_blobs, mlp_blobs = [], []  # (phase, item or cta, bytes)
        mlp_later = []  # paired MLPs: (phase, its GDN phase, units, blob maker)
        if fast.dec is None:
            fast.dec = [fast._dec(L) for L in range(fast.n_chip)]
        rows = _Rows(ptr=[0], parts=[])
        dcols = _Rows(ptr=[0], parts=[])  # CSC by out_proj dim / MLP unit: (residual row, w)
        cols, w1, upd, phases, items, chans = [], [], [], [], [], []
        a_neg, dt_bias, gnw = [], [], []
        self.bm_heads = []  # (layer, first value head, count, bitmap offset) of decaying layers
        n_bm = 0
        ncta = self.ncta

        def add_cols(c, w):
            off = sum(len(x) for x in cols)
            cols.append(c.to(torch.int16).cpu())
            w1.append(w.float().cpu())
            assert len(c) <= MAXCU
            return off, len(c)

        def add_upd(r):
            off = sum(len(x) for x in upd)
            upd.append(r.to(torch.int16).cpu())
            return off, len(r)

        for L in range(fast.n_chip):
            d = fast.dec[L]
            layer = model.model.layers[L]
            if d.gdn:
                g = fast.layers[L][0]
                nkh, n_p, n_b, n_a = d.nkh, d.n_p, d.n_b, d.n_a
                cu_off, n_cu = add_cols(d.cu, d.w1u)
                row0 = rows.add(d.w_in)
                # each pc channel's place in its key head's q | k | v
                idx = d.qkv_idx
                S = len(idx)
                src = idx[idx < n_p]
                assert len(src) == 0 or int(torch.bincount(src, minlength=n_p).max()) <= 1
                slot_of = torch.full((n_p,), -1, dtype=torch.long, device=idx.device)
                s_all = torch.arange(S, device=idx.device)
                slot_of[idx[idx < n_p]] = s_all[idx < n_p]
                nq = nkh * KD
                s = slot_of
                item_of = torch.where(s < 0, -1, torch.where(s < nq, s // KD, torch.where(
                    s < 2 * nq, (s - nq) // KD, (s - 2 * nq) // (R * VD))))
                local = torch.where(s < 0, -1, torch.where(s < nq, s % KD, torch.where(
                    s < 2 * nq, KD + (s - nq) % KD, 2 * KD + (s - 2 * nq) % (R * VD))))
                orphan = (item_of < 0).nonzero()[:, 0]
                n_items = max(nkh, 1)
                if nkh == 0:  # conv windows only, spread over a few CTAs
                    n_items = max(1, min(ncta, -(-n_p // 512)))
                item_of[orphan] = torch.arange(len(orphan), device=idx.device) % n_items
                taps = _bits(d.w_conv[:, 0])  # (n_p, 4)
                w01 = _i32(taps[:, 0] | (taps[:, 1] << 16))
                w23 = _i32(taps[:, 2] | (taps[:, 3] << 16))
                assert d.w_conv.shape[-1] == 4
                bm_layer = None
                if d.decays:
                    bm_layer = n_bm
                    self.bm_heads.append((L, nkh * R, n_bm))
                    n_bm += nkh * R * 8
                dims0 = len(dcols.ptr) - 1
                if nkh:
                    w_out = d.w_out  # (n_out, nhv * VD)
                    # CSC by dim: entries (residual row, weight)
                    wt = w_out.T.contiguous()
                    nzi = wt.nonzero()
                    counts = torch.bincount(nzi[:, 0], minlength=wt.shape[0])
                    dcols.ptr.extend((torch.cumsum(counts, 0) + dcols.n).tolist())
                    dcols.parts.append(_pack(d.out_rows[nzi[:, 1]], wt[nzi[:, 0], nzi[:, 1]]).cpu())
                    dcols.n += len(nzi)
                    a_off = sum(len(x) for x in a_neg)
                    if d.decays:
                        a_neg.append(d.a_neg.float().cpu())
                        dt_bias.append(d.dt_bias.cpu())
                    gw_off = sum(len(x) for x in gnw)
                    gnw.append(layer.linear_attn.norm.weight.detach().to(torch.bfloat16).cpu())
                    geps = float(layer.linear_attn.norm.variance_epsilon)
                item_off = len(items)
                p_rows = torch.arange(n_p, device=idx.device) + row0
                cv_cols, cv_vals, _, cv_cnt, cv_st = _sparse_np(d.w_in[:n_p])
                taps_np = d.w_conv[:, 0].float().cpu().numpy()
                item_np, local_np = item_of.cpu().numpy(), local.cpu().numpy()
                pc_np = d.pc.cpu().numpy()
                cu_np, w1u_np = d.cu.cpu().numpy(), d.w1u.float().cpu().numpy()
                for i in range(n_items):
                    mn = np.nonzero(item_np == i)[0]
                    mn = mn[np.argsort(-cv_cnt[mn], kind="stable")]  # by count, descending
                    mine = torch.from_numpy(mn).to(idx.device)
                    ch_off = sum(len(c) for c in chans)
                    chans.append(torch.stack([
                        p_rows[mine].to(torch.int32), d.pc[mine].to(torch.int32),
                        local[mine].to(torch.int32), w01[mine], w23[mine]], 1).cpu())
                    it = dict(idx=len(items), ch_off=ch_off, n_ch=len(mine), n_cu=len(d.cu),
                              nvh=0, kh=0, beta_row=0, a_row=-1, z_row=0, hv_off=0, dim_off=0,
                              bm_off=-1, gw_off=0, geps=0.0)
                    if nkh:
                        it.update(nvh=R, kh=int(g.kh[i]), beta_row=row0 + n_p + i * R,
                                  a_row=row0 + n_p + n_b + i * R if d.decays else -1,
                                  z_row=row0 + n_p + n_b + n_a + i * R * VD,
                                  hv_off=a_off + i * R, dim_off=dims0 + i * R * VD,
                                  bm_off=bm_layer + i * R * 8 if d.decays else -1,
                                  gw_off=gw_off, geps=geps)
                    items.append(Item(**it))
                    gdn_blobs.append((len(phases), len(items) - 1, _normed(cu_np, w1u_np, _item_blob(
                        items[-1], cv_cols, cv_vals, cv_cnt[mn], cv_st[mn], taps_np[mn],
                        local_np[mn], pc_np[mn]))))
                upd_off, n_upd = add_upd(d.out_rows if nkh else d.out_rows[:0])
                phases.append(Phase(kind=0, layer=L, eps=float(d.eps), cu_off=cu_off, n_cu=n_cu,
                                    upd_off=upd_off, n_upd=n_upd, item_off=item_off,
                                    n_items=n_items))
            if d.mlp:
                cu_off, n_cu = add_cols(d.m_cin, d.w1m)
                row0 = rows.add(d.w_gu)
                d0 = len(dcols.ptr) - 1
                wt = d.w_d.T.contiguous()  # (n_u, n_rows)
                nzi = wt.nonzero()
                counts = torch.bincount(nzi[:, 0], minlength=wt.shape[0])
                dcols.ptr.extend((torch.cumsum(counts, 0) + dcols.n).tolist())
                dcols.parts.append(_pack(d.m_rows[nzi[:, 1]], wt[nzi[:, 0], nzi[:, 1]]).cpu())
                dcols.n += len(nzi)
                upd_off, n_upd = add_upd(d.m_rows)
                assert int(counts.min()) == 1 and int(counts.max()) == 1  # one down weight a unit
                drow = d.m_rows[nzi[:, 1]].cpu().numpy()
                dbits = _bits(wt[nzi[:, 0], nzi[:, 1]]).cpu().numpy()
                gu = _sparse_np(d.w_gu)
                cin_np, w1m_np = d.m_cin.cpu().numpy(), d.w1m.float().cpu().numpy()
                mk = functools.partial(_mlp_blob, gu=gu, n_u=d.n_u, drow=drow, dbits=dbits,
                                       cin=cin_np, w1=w1m_np)
                if spec and d.gdn and L in pair_layers and phases[-1].layer == L:
                    phases[-1].pair = 1  # its units over the CTAs without the GDN's items
                    mlp_later.append((len(phases), len(phases) - 1, d.n_u, mk))
                else:
                    for c in range(ncta):
                        a, b = d.n_u * c // ncta, d.n_u * (c + 1) // ncta
                        if a < b:
                            mlp_blobs.append((len(phases), c, mk(a, b)))
                phases.append(Phase(kind=1, layer=L, eps=float(d.eps_m), cu_off=cu_off,
                                    n_cu=n_cu, upd_off=upd_off, n_upd=n_upd, n_units=d.n_u,
                                    row_off=row0, d_off=d0, f64=1))
        # the output head: rows split over the CTAs, each CTA's entries by column
        norm = model.model.norm
        hc = fast.head_cols
        cu_off, n_cu = add_cols(hc, (1.0 + norm.weight.float())[hc])
        wh = fast.w_head
        V = wh.shape[0]
        hr = -(-V // ncta)
        assert hr <= HEAD_ROWS
        # columns with no zero and few values: 2-bit codes, resident; the others by column
        dense = ((wh != 0).sum(0) == V).nonzero()[:, 0]
        htab = torch.unique(wh[:, dense].float()) if len(dense) else torch.zeros(0, device=dev)
        if len(htab) > 4 or len(dense) > MAXDENSE:
            dense = dense[:0]
            htab = htab[:0]
        sparse = torch.tensor([c for c in range(n_cu) if c not in set(dense.tolist())],
                              dtype=torch.long, device=dev)
        assert len(sparse) <= MAXSPARSE
        W = -(-hr // 16)
        head_b = [b""] * ncta
        hnorm = (hc.cpu().numpy(), (1.0 + norm.weight.float())[hc].cpu().numpy())
        if len(dense):
            nd = len(dense)
            c0 = torch.searchsorted(htab.contiguous(), wh[:, dense].float().contiguous())
            c0 = torch.cat([c0, c0.new_zeros(ncta * hr - V, nd)]).view(ncta, hr, nd)
            codes = torch.cat([c0, c0.new_zeros(ncta, W * 16 - hr, nd)], 1)  # rows padded
            codes = codes.reshape(ncta, W, 16, nd).to(torch.int64)
            words = (codes << (2 * torch.arange(16, device=dev))[None, None, :, None]).sum(2)
            words = _i32(words.permute(0, 2, 1).contiguous()).cpu().numpy()  # (ncta, nd, W)
            head_b = [_pad16(words[c].tobytes()) for c in range(ncta)]
        head_b = [_normed(*hnorm, b) for b in head_b]
        self.htab = torch.zeros(4, dtype=torch.float32, device=dev)
        self.htab[:len(htab)] = htab
        self.hdense = dense.to(torch.int16).contiguous() if len(dense) else \
            torch.zeros(1, dtype=torch.int16, device=dev)
        self.hsparse = sparse.to(torch.int16).contiguous() if len(sparse) else \
            torch.zeros(1, dtype=torch.int16, device=dev)
        self.n_dense, self.n_sparse = len(dense), len(sparse)
        ws = wh[:, sparse]
        ns = len(sparse)
        nzi = ws.nonzero()  # by row, then column
        cta = nzi[:, 0] // hr
        key = (cta * ns + nzi[:, 1]) * hr + (nzi[:, 0] - cta * hr)
        order = torch.argsort(key)
        nzi, cta = nzi[order], cta[order]
        cnt = torch.bincount(cta * ns + nzi[:, 1], minlength=ncta * ns).view(ncta, ns)
        hptr = torch.zeros(ncta, ns + 1, dtype=torch.long, device=dev)
        hptr[:, 1:] = torch.cumsum(cnt.flatten(), 0).view(ncta, ns)
        hptr[1:, 0] = hptr[:-1, -1]
        self.hptr = hptr.to(torch.int32).contiguous()
        self.hnz = _pack(nzi[:, 0] - cta * hr, ws[nzi[:, 0], nzi[:, 1]])
        del nzi, order, cta, key, ws
        head_phase = len(phases)
        phases.append(Phase(kind=2, layer=-1, eps=float(norm.eps), cu_off=cu_off, n_cu=n_cu))
        self.head_rows, self.n_vocab = hr, V
        # runs of GDN phases (no MLP between) run speculatively together (csrc/mega.cu)
        grp_of = {}
        if spec:
            k = 0
            while k < len(phases):
                k1 = k
                while k1 < len(phases) and phases[k1].kind == 0:
                    k1 += 1
                if k1 - k >= 3:
                    phases[k].grp_n = k1 - k
                    grp_of.update({x: k for x in range(k, k1)})
                k = max(k1, k + 1)
        # each CTA's resident data: the head's codes, its MLP parts, its GDN items (no two of a
        # phase on one CTA; a group's spread evenly; the least loaded CTAs first)
        assert len(phases) <= MAXPH
        assert max(ph.n_upd for ph in phases) <= UPR * NT
        pieces = [[(head_phase, head_b[c])] for c in range(ncta)]
        load = [len(head_b[c]) for c in range(ncta)]
        for k, c, b in mlp_blobs:
            pieces[c].append((k, b))
            load[c] += len(b)
        by_phase = {}
        for k, i, b in gdn_blobs:
            by_phase.setdefault(k, []).append(b)
        n_grp = {}  # (group, cta): its items
        gdn_ctas = {}  # phase: the CTAs with its items
        for k, bl in by_phase.items():
            assert len(bl) <= ncta
            used = set()
            gk = grp_of.get(k, -1)
            for b in sorted(bl, key=len, reverse=True):
                c = min((x for x in range(ncta) if x not in used),
                        key=lambda x: (n_grp.get((gk, x), 0) if gk >= 0 else 0, load[x]))
                n_grp[gk, c] = n_grp.get((gk, c), 0) + 1
                gdn_ctas.setdefault(k, set()).add(c)
                used.add(c)
                pieces[c].append((k, b))
                load[c] += len(b)
        for k, kg, n_u, mk in mlp_later:
            cs = sorted((c for c in range(ncta) if c not in gdn_ctas.get(kg, set())),
                        key=lambda c: load[c])
            for x, c in enumerate(cs):
                a, b = n_u * x // len(cs), n_u * (x + 1) // len(cs)
                if a < b:
                    blob = mk(a, b)
                    pieces[c].append((k, blob))
                    load[c] += len(blob)
        sched = np.full((len(phases), ncta), -1, np.int32)
        res, res_off, res_len, pos = [], [], [], 0
        for c in range(ncta):
            off = 0
            for k, b in pieces[c]:
                sched[k, c] = off
                off += len(b)
            res.append(b"".join(b for _, b in pieces[c]))
            res_off.append(pos)
            res_len.append(off)
            pos += off
        self.blob = torch.frombuffer(bytearray(b"".join(res) + bytes(16)),
                                     dtype=torch.uint8).to(dev)
        self.res_off = torch.tensor(res_off, dtype=torch.int32, device=dev)
        self.res_len = torch.tensor(res_len, dtype=torch.int32, device=dev)
        self.sched = torch.from_numpy(sched).flatten().to(dev)
        self.smem = int(self.lib.mk_smem_base()) + max(res_len)
        assert self.smem <= int(self.lib.mk_max_smem()), (self.smem, max(res_len))
        assert int(self.lib.mk_max_ctas(self.smem)) >= ncta

        def t(x, dt):
            return torch.cat(x).to(dt).to(dev).contiguous() if x else \
                torch.zeros(1, dtype=dt, device=dev)

        self.rowptr = torch.tensor(rows.ptr, dtype=torch.int32, device=dev)
        self.nz = t(rows.parts, torch.int32)
        self.dptr = torch.tensor(dcols.ptr, dtype=torch.int32, device=dev)
        self.dnz = t(dcols.parts, torch.int32)
        self.cols, self.w1 = t(cols, torch.int16), t(w1, torch.float32)
        self.upd_rows = t(upd, torch.int16)
        self.a_neg, self.dt_bias = t(a_neg, torch.float32), t(dt_bias, torch.bfloat16)
        self.gnorm_w = t(gnw, torch.bfloat16)
        self.chans = t(chans, torch.int32)
        self.bitmaps = torch.zeros(max(n_bm, 1), dtype=torch.int32, device=dev)
        self.item_list = items
        self.phases = (Phase * len(phases))(*phases)
        self.phases_d = torch.frombuffer(bytearray(self.phases), dtype=torch.uint8).to(dev)
        self.n_phases = len(phases)
        self.embed = model.model.embed_tokens.weight
        assert self.embed.dtype == torch.bfloat16 and self.embed.is_contiguous()
        self.acc = torch.zeros(3 * 2560, dtype=torch.float32, device=dev)
        # three barrier counters, then three group masks, 128 B apart
        self.bar = torch.zeros(112, dtype=torch.int64, device=dev)
        self.accs = torch.zeros(3 * 2560, dtype=torch.float32, device=dev)
        self.wsave = torch.zeros(max(int(self.chans.shape[0]), 1) * 2, dtype=torch.int32,
                                 device=dev)
        self.part_v = torch.zeros(ncta, dtype=torch.float32, device=dev)
        self.part_i = torch.zeros(ncta, dtype=torch.int32, device=dev)
        self.stats = {"phases": self.n_phases, "items": len(items), "ctas": ncta,
                      "csr_nnz": rows.n, "csc_nnz": dcols.n, "head_nnz": int(self.hnz.numel()),
                      "chans": int(self.chans.shape[0]) if self.chans.dim() > 1 else 0,
                      "head_dense": self.n_dense, "head_sparse": self.n_sparse,
                      "res_max": max(res_len), "res_min": min(res_len), "smem": self.smem,
                      "pairs": [ph.layer for ph in phases if ph.pair],
                      "groups": [(k, phases[k].grp_n) for k in range(len(phases))
                                 if phases[k].grp_n],
                      "grp_items_max": max((v for (g, _), v in n_grp.items() if g >= 0),
                                           default=0)}

    def _state_ptrs(self):
        rs, cs = [], []
        for L in range(self.fast.n_chip):
            cl = self.cache.layers[L]
            r, c = cl.recurrent_states[0], cl.conv_states[0]
            assert r.is_contiguous() and c.is_contiguous() and r.dtype == torch.float32
            assert c.dtype == torch.bfloat16 and c.shape[-1] == 4
            rs.append(r.data_ptr())
            cs.append(c.data_ptr())
        return (torch.tensor(rs, dtype=torch.int64, device=self.dev),
                torch.tensor(cs, dtype=torch.int64, device=self.dev))

    def sync_bitmaps(self) -> None:
        """The decaying layers' nonzero-row bitmaps, from the state as it is now."""
        for L, nhv, off in self.bm_heads:
            g = self.fast.layers[L][0]
            vh = (g.kh[:, None] * R + torch.arange(R, device=self.dev)).reshape(-1)
            nzr = (self.cache.layers[L].recurrent_states[0][0, vh] != 0).any(-1)  # (nhv, 256)
            w = nzr.view(nhv, 8, 32).to(torch.int64)
            words = (w << torch.arange(32, device=self.dev)).sum(-1)  # (nhv, 8)
            self.bitmaps[off:off + nhv * 8] = _i32(words.flatten())

    @torch.no_grad()
    def run(self, forced: list[int], n: int, dbg: torch.Tensor | None = None,
            dbg_tok: int = 0, dbg_items: torch.Tensor | None = None,
            prof: torch.Tensor | None = None, inputs: torch.Tensor | None = None) -> list[int]:
        """n steps; step i's input is forced[i] (i < len(forced)) or step i - 1's prediction.
        Returns the n predictions (chip token ids). dbg (n_phases, 2560) fp32: the residual
        stream at the start of each phase of step dbg_tok. inputs (n,) int32: each step's input
        as the kernel embedded it."""
        assert len(forced) >= 1
        self.sync_bitmaps()
        rs, cs = self._state_ptrs()
        f = torch.tensor(forced, dtype=torch.int32, device=self.dev)
        preds = torch.zeros(n, dtype=torch.int32, device=self.dev)
        self.acc.zero_()
        self.accs.zero_()
        self.bar.zero_()
        p = Params(phases=self.phases_d.data_ptr(), n_phases=self.n_phases,
                   chans=self.chans.data_ptr(), blob=self.blob.data_ptr(),
                   res_off=self.res_off.data_ptr(), res_len=self.res_len.data_ptr(),
                   sched=self.sched.data_ptr(), rowptr=self.rowptr.data_ptr(), nz=self.nz.data_ptr(),
                   cols=self.cols.data_ptr(), w1=self.w1.data_ptr(),
                   upd_rows=self.upd_rows.data_ptr(), dptr=self.dptr.data_ptr(),
                   dnz=self.dnz.data_ptr(), a_neg=self.a_neg.data_ptr(),
                   dt_bias=self.dt_bias.data_ptr(), gnorm_w=self.gnorm_w.data_ptr(),
                   bitmaps=self.bitmaps.data_ptr(), rs=rs.data_ptr(), cs=cs.data_ptr(),
                   n_layers=self.fast.n_chip,
                   n_dense=self.n_dense, n_sparse=self.n_sparse, hdense=self.hdense.data_ptr(),
                   hsparse=self.hsparse.data_ptr(), htab=self.htab.data_ptr(),
                   hptr=self.hptr.data_ptr(), hnz=self.hnz.data_ptr(), head_rows=self.head_rows,
                   n_vocab=self.n_vocab, embed=self.embed.data_ptr(), offset=self.offset,
                   forced=f.data_ptr(), n_forced=len(forced), n_tokens=n,
                   preds=preds.data_ptr(), acc=self.acc.data_ptr(), bar=self.bar.data_ptr(),
                   part_v=self.part_v.data_ptr(), part_i=self.part_i.data_ptr(),
                   dbg=dbg.data_ptr() if dbg is not None else None, dbg_tok=dbg_tok,
                   dbg_items=dbg_items.data_ptr() if dbg_items is not None else None,
                   prof=prof.data_ptr() if prof is not None else None,
                   wsave=self.wsave.data_ptr(), accs=self.accs.data_ptr(),
                   inputs=inputs.data_ptr() if inputs is not None else None)
        st = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        err = self.lib.mk_run(ctypes.byref(p), self.ncta, self.smem, st)
        if err:
            raise RuntimeError(f"mega launch failed: cudaError {err}")
        out = preds.tolist()  # syncs
        err = self.lib.mk_last_error()
        if err:
            raise RuntimeError(f"mega failed: cudaError {err}")
        return out
