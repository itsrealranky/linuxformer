"""Qwen3.5-4B using the Linux machine in its own weights (lx.merge): a ``shell`` tool. A call's
command is typed into the machine's UART (KEY tokens through the chip layers) and the tool's
response is what the machine printed (the BYTE tokens the chip layers predicted), until the
shell's prompt comes back. Qwen's text runs on its own layers (merge.text_view, stock generate,
greedy).

    python -m lx.chat [--snapshot FILE] [--out FILE] [--rehearse] "question" ...

--rehearse runs the commands on the reference machine only (nothing verified, for previewing
what Qwen does and what it costs).
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch

from lx import merge
from lx.linux import ROOT, Linux

MAX_LINE = 250  # the shell's line editor holds 256 bytes (CONFIG_FEATURE_EDITING_MAX_LEN)
SHELL_TOOL = {"type": "function", "function": {
    "name": "shell",
    "description": "Run a command in the BusyBox shell (hush) of a small RISC-V Linux 6.8 "
                   "machine (RV32IMA, no MMU, 4.5 MB of RAM, initramfs root, no /proc mounted) "
                   "and return what the terminal shows. The machine runs inside this model's own "
                   "weights; a command takes from seconds to a few minutes. hush is a POSIX "
                   "shell: no brace expansion, no arrays, no (( )) loops; $((...)) arithmetic "
                   "is 64-bit. Shell builtins (echo, printf, test, while/for/if) are fastest; "
                   "external programs can fail when memory is short. Each line must be under "
                   f"{MAX_LINE} characters; several lines run one after another.",
    "parameters": {"type": "object", "properties": {
        "command": {"type": "string", "description": "e.g. uname -a, or ls /bin"}},
        "required": ["command"]}}}

CALL = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def parse_call(text: str):
    """(text before the call, function name, {parameter: value}) or None."""
    m = CALL.search(text)
    if not m:
        return None
    return text[:m.start()].strip(), m.group(1).strip(), dict(PARAM.findall(m.group(2)))


class Chat:
    def __init__(self, lx: Linux, rehearse: bool = False) -> None:
        from transformers import AutoTokenizer

        self.lx = lx
        self.rehearse = rehearse
        self.tv = merge.text_view(lx.model, lx.info)
        self.tok = AutoTokenizer.from_pretrained(lx.weights or merge.QWEN)
        self.dev = next(lx.model.parameters()).device

    @torch.no_grad()
    def _generate(self, messages, max_new: int):
        ids = self.tok.apply_chat_template(messages, tools=[SHELL_TOOL], add_generation_prompt=True,
                                           enable_thinking=False, tokenize=True)
        ids = ids if isinstance(ids, list) else ids["input_ids"]
        x = torch.tensor([ids], device=self.dev)
        torch.cuda.synchronize()
        t0 = time.time()
        out = self.tv.generate(x, max_new_tokens=max_new, do_sample=False,
                               stop_strings=["</tool_call>"], tokenizer=self.tok)
        torch.cuda.synchronize()
        new = out[0, len(ids):]
        return self.tok.decode(new, skip_special_tokens=True), len(new), time.time() - t0

    def ask(self, question: str, max_calls: int = 4, max_new: int = 768, echo=print,
            progress=None, max_records: int = 30_000_000, on_event=None) -> dict:
        """progress, max_records: as in Linux.run, for each command. on_event(event) whenever
        an event is added or gets its result (a call is passed before it runs)."""
        on_event = on_event or (lambda ev: None)
        messages = [{"role": "user", "content": question}]
        events = []
        for _ in range(max_calls + 1):
            text, n, dt = self._generate(messages, max_new)
            call = parse_call(text)
            if call is None:
                events.append({"qwen": text, "tokens": n, "seconds": round(dt, 1)})
                on_event(events[-1])
                echo(f"qwen: {text}")
                messages.append({"role": "assistant", "content": text})
                break
            before, name, args = call
            cmd = args.get("command", "")
            echo(f"qwen: {before}\n  -> shell: {cmd!r}" if before else f"  -> shell: {cmd!r}")
            ev = {"qwen": text, "tokens": n, "seconds": round(dt, 1), "call": name,
                  "command": cmd}
            on_event(ev)
            if name != "shell" or not cmd:
                result = f"error: unknown tool {name!r} or empty command"
            elif max(len(ln) for ln in cmd.splitlines()) >= MAX_LINE:
                result = (f"error: a line is longer than the shell's {MAX_LINE}-character limit; "
                          "split it into several lines (not run)")
            else:
                r = self.lx.run(cmd, max_records, rehearse=self.rehearse, progress=progress)
                result = r["output"].decode(errors="replace")
                ev.update(machine_tokens=r["tokens"], machine_seconds=r["seconds"],
                          machine_tok_per_s=r["tok_per_s"], machine_timeline=r["timeline"],
                          machine_out=r["out"].decode(errors="replace"))
                if self.rehearse:
                    ev["rehearsed"] = True
                elif "same_as_reference" in r:  # model alone: compared with the reference after
                    ev.update(wrong=r["wrong"], same_as_reference=r["same_as_reference"],
                              first_difference=r["first_difference"],
                              input_purity=r["input_purity"])
                else:
                    ev["wrong"] = 0  # lx.linux.run raises on any wrong prediction
                echo(f"  <- linux ({r['tokens']} tokens, {r['seconds']}s): {result!r}")
            ev["response"] = result
            on_event(ev)
            events.append(ev)
            messages.append({"role": "assistant", "content": before, "tool_calls": [
                {"type": "function", "function": {"name": name, "arguments": args}}]})
            messages.append({"role": "tool", "content": result})
        return {"question": question, "events": events}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", nargs="*")
    ap.add_argument("--snapshot", default=str(ROOT / "build" / "snapshots" / "boot-merged.pt"))
    ap.add_argument("--weights", default="", help="a merge.save() directory (else built)")
    ap.add_argument("--rehearse", action="store_true")
    ap.add_argument("--fast", action="store_true", help="verify with lx.fast (zeros skipped)")
    ap.add_argument("--alone", action="store_true",
                    help="model alone: decode with lx.mega from the snapshot, no reference in "
                         "the loop (compared with afterwards)")
    ap.add_argument("--lean", action="store_true",
                    help="lx.fast, the chip projections kept in host memory (fits a 24 GB card)")
    ap.add_argument("--max-gib", type=float, default=None,
                    help="cap this process's GPU memory (PyTorch's allocator) at this many GiB")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.max_gib:
        torch.cuda.set_per_process_memory_fraction(
            a.max_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    lx = Linux(merged=True, weights=Path(a.weights) if a.weights else None,
               fast=a.fast or a.alone, lean=a.lean)
    boot = lx.boot_alone if a.alone else lx.boot
    b = boot(Path(a.snapshot))
    print(f"[{b['what']}: {b['tokens']} tokens in {b['seconds']}s]", flush=True)
    chat = Chat(lx, a.rehearse)
    qs = a.questions or ["What Linux kernel version is your shell machine running, and on what "
                         "CPU architecture? Check with the shell, then answer in one sentence."]
    res = {"device": torch.cuda.get_device_name(), **lx.info, "rehearsed": a.rehearse,
           "mode": "model alone" if a.alone else "verified", "boot": {
        k: v for k, v in b.items() if k != "out"}, "chats": []}
    for i, q in enumerate(qs):
        if i:  # every question starts from the booted machine
            boot(Path(a.snapshot))
        print(f"\nuser: {q}", flush=True)
        res["chats"].append(chat.ask(q))
    res["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
    res["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 1)
    res["max_gib"] = a.max_gib
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
