"""A machine running in the model. The chip layers' cache is the machine's whole state; the host
only types (KEY0 / KEY tokens) and reads the bytes the model predicts (BYTE tokens).

    s = Session(c, model, machine)               # a machine snapshot injected (hf.inject)
    s.verify(toks)                               # speculative: a proposed trace, checked
    s.free(n)                                    # the pure path: the model's own tokens
    s.save(path); s = Session.load(c, model, path)

verify() is speculative execution: a proposer (the reference, lx.ref) writes the trace ahead
and the model runs it as one chunked prefill; at every position the output head's argmax must
be the next token (positions before host input predict nothing). free() is the pure path: each
argmax is fed back as the next input, one CUDA graph per token, with no reference anywhere;
the host's typing needs only the token stream (a byte is waiting until its RX comes by).
"""

import copy
import json
from pathlib import Path

import torch

from lx import core, hf
from lx import ref as R
from lx import spec as S

_FINAL_IDS = None


def _final_ids() -> torch.Tensor:
    """bool[VOCAB]: the token ends a record (a place where the host may type)."""
    global _FINAL_IDS
    if _FINAL_IDS is None:
        f = torch.zeros(hf.N_VOCAB, dtype=torch.bool)
        for name, (base, size) in S.FAMILIES.items():
            f[base:base + size] = name in S.FINAL or name in S.INPUT
        for name, t in S.CONTROLS.items():
            f[t] = name in S.FINAL or name == "GO"
        _FINAL_IDS = f
    return _FINAL_IDS


def _typed(t: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(t, dtype=torch.bool)
    for f in S.INPUT:
        lo, size = S.FAMILIES[f]
        out |= (t >= lo) & (t < lo + size)
    return out


def console(tokens) -> bytes:
    b0, nb = S.FAMILIES["BYTE"]
    return bytes(t - b0 for t in tokens if b0 <= t < b0 + nb)


class Session:
    def __init__(self, c, model, machine: R.Machine | None, chunk: int = 65536,
                 offset: int = 0) -> None:
        """``offset``: chip token t is model id offset + t (lx.merge: Qwen's vocabulary first);
        the head's argmax is then over the chip's rows only."""
        from transformers import DynamicCache

        self.c, self.model, self.chunk, self.offset = c, model, chunk, offset
        self.rows = slice(offset, offset + S.VOCAB) if offset else None
        self.n_chip = c.n_layers()
        self.dev = next(model.parameters()).device
        self.cache = DynamicCache(config=model.config)
        zero = torch.zeros(1, 1, model.config.hidden_size, dtype=model.dtype, device=self.dev)
        hf.chip_forward(model, zero, self.cache, self.n_chip)
        if machine is not None:
            core.snapshot(c, machine)
            hf.inject(c, self.cache)
        self.ref = copy.deepcopy(machine)  # the proposer's machine (verify), or None
        self.next = None  # the model's prediction for the next token (None before GO)
        self.last = None  # the last token run
        self.n = 0  # tokens run
        self.waiting = 0  # typed bytes not yet read (counted from the stream: KEY*, RX)
        self.graph = None

    # --- speculative execution ------------------------------------------------------------
    @torch.no_grad()
    def verify(self, toks: list[int]) -> dict:
        """Run the proposed tokens as chunked prefill. Returns {"wrong": positions whose
        prediction is not the next token, "first_wrong": [(index, want, got)], "out": the bytes
        the model predicted (BYTE tokens), "predicted": predictions [tensor]}."""
        if self.n == 0 and toks[:1] != [S.tok("GO")]:
            toks = [S.tok("GO")] + list(toks)
        x = torch.tensor([toks], device=self.dev)
        preds = []
        for a in range(0, len(toks), self.chunk):
            e = self.model.model.embed_tokens(x[:, a:a + self.chunk] + self.offset)
            h = hf.chip_forward(self.model, e, self.cache, self.n_chip)[0]
            preds.append(hf.head_argmax(self.model, h, rows=self.rows))
        pred = torch.cat(preds)
        # prediction i is for token i + 1; the one before toks[0] came from the last call
        prev = torch.tensor([-1 if self.next is None else self.next], device=self.dev)
        want_pred = torch.cat([prev, pred[:-1]])
        target = x[0]
        bad = (want_pred != target) & ~_typed(target)
        if self.next is None:
            bad[0] = False
        idx = bad.nonzero()[:, 0].tolist()
        self.next, self.last = int(pred[-1]), toks[-1]
        self.n += len(toks)
        self._count(toks)
        got = want_pred.tolist()
        return {"wrong": len(idx), "first_wrong": [(i, toks[i], got[i]) for i in idx[:8]],
                "out": console(got[i] for i in range(len(toks)) if not _is_typed(toks[i])),
                "predicted": pred}

    def _count(self, toks) -> None:
        rx0, nrx = S.FAMILIES["RX"]
        for t in toks:
            if _is_typed(t):
                self.waiting += 1
            elif rx0 <= t < rx0 + nrx:
                self.waiting -= 1

    def propose(self, keys: bytes = b"", until=None, max_records: int = 50_000_000) -> list:
        """The proposer's tokens: ``keys`` typed now, then records until ``until(machine)``."""
        toks = []
        for b in keys:
            toks += self.ref.key(b)
        n = 0
        while not self.ref.halted and not (until and until(self.ref)):
            toks += self.ref.step()
            n += 1
            if n >= max_records:
                break
        return toks

    # --- the pure path --------------------------------------------------------------------
    def _body(self) -> None:
        h = hf.chip_forward(self.model, self.model.model.embed_tokens(self.g_ids), self.cache,
                            self.n_chip)[0]
        self.g_pred.copy_(hf.head_argmax(self.model, h, rows=self.rows))

    @torch.no_grad()
    def _capture(self) -> None:
        """One CUDA graph of a single-token step (HF updates the cache in place, so the graph's
        buffers are the cache's). Warm-up runs change the state: it is saved and put back."""
        self.g_ids = torch.zeros((1, 1), dtype=torch.long, device=self.dev)
        self.g_pred = torch.zeros(1, dtype=torch.long, device=self.dev)
        saved = self._state()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._body()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._body()
        self._set_state(saved)

    @torch.no_grad()
    def step(self, t: int) -> int:
        """Run token t alone (a CUDA graph replay); returns the model's next prediction."""
        if self.graph is None:
            self._capture()
        self.g_ids.fill_(t + self.offset)
        self.graph.replay()
        self.next, self.last = int(self.g_pred), t
        self.n += 1
        self._count([t])
        return self.next

    def type(self, data: bytes) -> list[int]:
        """Pure path: type bytes now (only at a record boundary), as KEY0 / KEY tokens."""
        assert _final_ids()[self.last], "type only at a record boundary"
        out = []
        for b in data:
            assert self.waiting < S.RX_SLOTS
            t = S.tok("KEY0" if self.waiting == 0 else "KEY", b)
            self.step(t)
            out.append(t)
        return out

    def free(self, n: int, stop=None) -> list[int]:
        """Feed the model its own predictions for n tokens (or until ``stop(tokens)``)."""
        out = []
        for _ in range(n):
            t = self.next
            self.step(t)
            out.append(t)
            if stop and stop(out):
                break
        return out

    # --- snapshots ------------------------------------------------------------------------
    def _state(self):
        return [(self.cache.layers[L].conv_states[0].clone(),
                 self.cache.layers[L].recurrent_states[0].clone()) for L in range(self.n_chip)]

    def _set_state(self, st) -> None:
        for L, (cs, rs) in enumerate(st):
            self.cache.layers[L].conv_states[0].copy_(cs)
            self.cache.layers[L].recurrent_states[0].copy_(rs)

    def save(self, path) -> None:
        """The model's state (every chip layer's conv window and recurrent state), the last
        prediction, and the proposer's machine. A .safetensors path stores it without pickle."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        d = {"state": [(cs.cpu(), rs.cpu()) for cs, rs in self._state()],
             "next": self.next, "last": self.last, "n": self.n, "waiting": self.waiting,
             "ref": self.ref}
        if Path(path).suffix == ".safetensors":
            save_safe(d, path)
        else:
            torch.save(d, path)

    @classmethod
    def load(cls, c, model, path, chunk: int = 65536, offset: int = 0) -> "Session":
        d = load_safe(path) if Path(path).suffix == ".safetensors" else \
            torch.load(path, weights_only=False)
        s = cls(c, model, None, chunk, offset)
        s._set_state([(cs.to(s.dev), rs.to(s.dev)) for cs, rs in d["state"]])
        s.next, s.last, s.n, s.waiting, s.ref = d["next"], d["last"], d["n"], d["waiting"], d["ref"]
        return s


def save_safe(d: dict, path) -> None:
    """A Session.save dict as safetensors: the state tensors and the machine's byte fields as
    tensors, every other field (ints, lists, the CSR dict) as JSON in the metadata."""
    from safetensors.torch import save_file

    t, ref, kinds = {}, {}, {}
    for L, (cs, rs) in enumerate(d["state"]):
        t[f"conv.{L}"], t[f"recurrent.{L}"] = cs.contiguous(), rs.contiguous()
    for k, v in vars(d["ref"]).items():
        if isinstance(v, (bytes, bytearray)):
            t[f"ref.{k}"] = torch.frombuffer(bytearray(v), dtype=torch.uint8).clone() if v else \
                torch.zeros(0, dtype=torch.uint8)
            kinds[k] = type(v).__name__
        else:
            ref[k] = v
    meta = {k: d[k] for k in ("next", "last", "n", "waiting")}
    meta.update(ref=ref, bytes=kinds)
    save_file(t, str(path), metadata={"lx.session": json.dumps(meta)})


def load_safe(path) -> dict:
    from safetensors import safe_open

    with safe_open(str(path), "pt") as f:
        meta = json.loads(f.metadata()["lx.session"])
        t = {k: f.get_tensor(k) for k in f.keys()}
    m = R.Machine.__new__(R.Machine)
    vars(m).update(meta.pop("ref"))
    for k, kind in meta.pop("bytes").items():
        b = t[f"ref.{k}"].numpy().tobytes()
        setattr(m, k, bytearray(b) if kind == "bytearray" else b)
    n = sum(k.startswith("conv.") for k in t)
    return dict(meta, ref=m, state=[(t[f"conv.{L}"], t[f"recurrent.{L}"]) for L in range(n)])


def _is_typed(t: int) -> bool:
    return any(S.FAMILIES[f][0] <= t < S.FAMILIES[f][0] + S.FAMILIES[f][1] for f in S.INPUT)
