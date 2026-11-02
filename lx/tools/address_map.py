"""Where the machine's memory is in the model's state, from the compiler (lx/core.py, placed by
lx/hf.head_slots), and a check that it is all there: every byte of RAM and ROM decoded from a
snapshot's state tensors alone, with this map, against the reference machine saved with it.

Recurrent state of chip layer L (model.layers.L, snapshot key recurrent.L): [1, 128 value heads,
256 rows, 128 columns]. A key head j owns value heads 8j..8j+7. Columns 0 and 1 are the ballast
(2^24 in a row that was ever written); data starts at column 2.

  ROM  the kernel's text, bytes [0, text_end): word W = (addr - RAM_BASE) / 4 in layer 2, key
       head W >> 14, value head 8 (W >> 14) + (W >> 11 & 7), row W >> 3 & 255; its 8 nibbles
       in columns 2 + 8 (W & 7) + i, as integers 0..15 (nibble i = bits 4i..4i+3)
  RAM  bytes [8192 * first_block, ram_size): word w in the table of 2048-word block b = w >> 11,
       placed at (layer, key head j); value head 8j + (w & 7), row w >> 3 & 255; bit i of the
       word in column 2 + i, as 0 or A_BIT = 128

    python tools/address_map.py --out results/address_map.json [--check SNAPSHOT ...]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lx import core, hf, merge
from lx import spec as S
from lx.ir import A_BIT, A_REF, REF_COLS

PAGE = 4096


def build_map(c=None) -> dict:
    c = c or core.build(merge.config())
    cfg = c.cfg
    r = c.target.value_heads // c.target.key_heads
    slots = {c.nodes[ni].name: (L, j) for ni, (L, j) in hf.head_slots(c).items()}
    rom = [{"table": f"rom{h}", "layer": slots[f"rom{h}"][0], "key_head": slots[f"rom{h}"][1],
            "value_heads": [slots[f"rom{h}"][1] * r + x for x in range(r)],
            "words": [h * core.ROM_HEAD_WORDS,
                      min((h + 1) * core.ROM_HEAD_WORDS, cfg.text_words)]}
           for h in range(-(-cfg.text_words // core.ROM_HEAD_WORDS))]
    ram = [{"table": f"ram{b}", "block": b, "layer": slots[f"ram{b}"][0],
            "key_head": slots[f"ram{b}"][1],
            "bytes": [b * core.RAM_HEAD_WORDS * 4, (b + 1) * core.RAM_HEAD_WORDS * 4]}
           for b in c.ram_blocks]
    other = {k: {"layer": v[0], "key_head": v[1]} for k, v in slots.items()
             if not (k.startswith("rom") or k.startswith("ram"))}
    return {"ram_base": S.RAM_BASE, "ram_size": cfg.ram_size, "text_end": cfg.text_end,
            "first_ram_block": c.ram_blocks[0], "rom_head_words": core.ROM_HEAD_WORDS,
            "ram_block_words": core.RAM_HEAD_WORDS, "value_heads_per_key_head": r,
            "a_bit": A_BIT, "a_ref": A_REF, "ballast_columns": list(REF_COLS),
            "rom_encoding": "word W: column 2 + 8*(W & 7) + i holds nibble i (0..15) of the word",
            "ram_encoding": "word w: column 2 + i holds A_BIT (128) when bit i of the word is 1",
            "rom": rom, "ram": ram, "other_tables": other}


def locate(amap: dict, addr: int) -> list[dict]:
    """The state cells that hold the byte at physical address addr (ROM and / or RAM)."""
    ofs = addr - amap["ram_base"]
    assert 0 <= ofs < amap["ram_size"], hex(addr)
    w, byte = ofs >> 2, ofs & 3
    out = []
    if ofs < amap["text_end"]:
        t = amap["rom"][w // amap["rom_head_words"]]
        col = 2 + 8 * (w & 7) + 2 * byte
        out.append({"region": "ROM", "table": t["table"], "layer": t["layer"],
                    "key_head": t["key_head"], "value_head": t["key_head"] * 8 + (w >> 11 & 7),
                    "row": w >> 3 & 255, "columns": [col, col + 1],
                    "encoding": "two nibbles (low, high), integers 0..15"})
    b = w // amap["ram_block_words"]
    if b >= amap["first_ram_block"]:
        t = amap["ram"][b - amap["first_ram_block"]]
        col = 2 + 8 * byte
        out.append({"region": "RAM", "table": t["table"], "layer": t["layer"],
                    "key_head": t["key_head"], "value_head": t["key_head"] * 8 + (w & 7),
                    "row": w >> 3 & 255, "columns": list(range(col, col + 8)),
                    "encoding": "8 bits (bit 0 first), each 0 or 128"})
    return out


def page_cells(amap: dict, page: int) -> list[dict]:
    """Per 4 KB page: (region, layer, key head, value heads, rows) holding it."""
    out = []
    w0, w1 = page * PAGE // 4, (page + 1) * PAGE // 4
    rh, bw = amap["rom_head_words"], amap["ram_block_words"]
    tw = amap["text_end"] // 4
    if w0 < tw:
        t = amap["rom"][w0 // rh]
        vh = sorted({t["key_head"] * 8 + (w >> 11 & 7) for w in (w0, min(w1, tw) - 1)})
        out.append({"region": "ROM", "layer": t["layer"], "key_head": t["key_head"],
                    "value_heads": vh, "rows": [w0 >> 3 & 255, (min(w1, tw) - 1) >> 3 & 255],
                    "bytes": min(w1, tw) * 4 - w0 * 4})
    b = w0 // bw
    if b >= amap["first_ram_block"]:
        t = amap["ram"][b - amap["first_ram_block"]]
        out.append({"region": "RAM", "layer": t["layer"], "key_head": t["key_head"],
                    "value_heads": [t["key_head"] * 8 + x for x in range(8)],
                    "rows": [w0 >> 3 & 255, (w1 - 1) >> 3 & 255], "bytes": PAGE})
    return out


def decode(amap: dict, state) -> tuple[bytearray, dict]:
    """All of memory from the recurrent states alone (state: per chip layer (conv, recurrent)).
    Returns (bytes, exactness counts: cells outside the encoding's value set)."""
    size, tw = amap["ram_size"], amap["text_end"] // 4
    words = np.zeros(size // 4, dtype=np.uint64)
    rams = np.zeros(size // 4, dtype=np.uint64)
    bad = {"rom_data_not_integer_0_15": 0, "ram_data_not_0_or_128": 0,
           "ballast_not_0_or_2^24": 0, "unused_columns_nonzero": 0}
    nib_w = (np.uint64(1) << (4 * np.arange(8, dtype=np.uint64)))
    bit_w = (np.uint64(1) << np.arange(32, dtype=np.uint64))
    rec = {L: state[L][1][0] for L in {t["layer"] for t in amap["rom"] + amap["ram"]}}

    def ballast(blk):
        b = blk[..., :2]
        bad["ballast_not_0_or_2^24"] += int(((b != 0) & (b != A_REF)).sum())

    for t in amap["rom"]:
        j = t["key_head"]
        blk = rec[t["layer"]][8 * j:8 * j + 8].double().numpy()  # (8 value heads, 256, 128)
        ballast(blk)
        bad["unused_columns_nonzero"] += int((blk[..., 66:] != 0).sum())
        nib = blk[..., 2:66].reshape(8, 256, 8, 8)  # (x, row, k, nibble)
        bad["rom_data_not_integer_0_15"] += int(((nib != np.round(nib)) | (nib < 0)
                                                 | (nib > 15)).sum())
        wv = (np.round(nib).astype(np.uint64) * nib_w).sum(-1)  # (x, row, k)
        w0 = t["words"][0]
        n = t["words"][1] - w0
        words[w0:w0 + n] = wv.reshape(-1)[:n]  # W = h*16384 + x*2048 + row*8 + k
    for t in amap["ram"]:
        j = t["key_head"]
        blk = rec[t["layer"]][8 * j:8 * j + 8].double().numpy()
        ballast(blk)
        bad["unused_columns_nonzero"] += int((blk[..., 34:] != 0).sum())
        bits = blk[..., 2:34]  # (x, row, 32)
        bad["ram_data_not_0_or_128"] += int(((bits != 0) & (bits != A_BIT)).sum())
        wv = ((bits == A_BIT).astype(np.uint64) * bit_w).sum(-1)  # (x, row)
        w0 = t["block"] * amap["ram_block_words"]
        rams[w0:w0 + amap["ram_block_words"]] = wv.T.reshape(-1)  # w = b*2048 + row*8 + x
    mem = words.copy()
    lo = amap["first_ram_block"] * amap["ram_block_words"]
    mem[tw:] = rams[tw:]
    # the words below text_end in the first RAM block are in both
    bad["rom_ram_overlap_words_differ"] = int((rams[lo:tw] != words[lo:tw]).sum())
    return bytearray(mem.astype("<u4").tobytes()), bad


def check(amap: dict, path: str) -> dict:
    t0 = time.time()
    d = torch.load(path, weights_only=False, map_location="cpu")
    mem, bad = decode(amap, d["state"])
    ref = d["ref"]
    want = bytes(ref.ram)
    diff = np.nonzero(np.frombuffer(bytes(mem), np.uint8) != np.frombuffer(want, np.uint8))[0]
    tw = amap["text_end"]
    # negative control: one stored bit flipped in the state must show up as one wrong byte
    cell = locate(amap, amap["ram_base"] + tw + 0x1000)[-1]
    st = [(cs, rs.clone() if L == cell["layer"] else rs) for L, (cs, rs) in
          enumerate(d["state"])]
    v = st[cell["layer"]][1][0, cell["value_head"], cell["row"], cell["columns"][0]]
    st[cell["layer"]][1][0, cell["value_head"], cell["row"], cell["columns"][0]] = \
        A_BIT if v == 0 else 0
    mem2, _ = decode(amap, st)
    ctrl = int((np.frombuffer(bytes(mem2), np.uint8) != np.frombuffer(bytes(mem), np.uint8)).sum())
    return {"snapshot": path, "tokens": d["n"], "bytes_compared": len(want),
            "rom_bytes": tw, "ram_layer_bytes": amap["ram_size"] - amap["first_ram_block"]
            * amap["ram_block_words"] * 4, "bytes_differing": int(diff.size),
            "first_differences": [hex(int(i) + amap["ram_base"]) for i in diff[:8]],
            "rom_equals_reference_rom": bytes(mem[:tw]) == bytes(ref.rom),
            "identical": diff.size == 0 and bytes(mem[:tw]) == bytes(ref.rom),
            "exactness": bad, "negative_control_bytes_differing": ctrl,
            "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--check", nargs="*", default=[])
    ap.add_argument("--check-out", default="")
    a = ap.parse_args()
    amap = build_map()
    if a.out:
        Path(a.out).write_text(json.dumps(amap, indent=1))
    res = []
    for p in a.check:
        r = check(amap, p)
        print(json.dumps(r), flush=True)
        res.append(r)
    if a.check_out:
        Path(a.check_out).write_text(json.dumps({"checks": res, "all_identical":
                                                 all(r["identical"] for r in res)}, indent=1))


if __name__ == "__main__":
    main()
