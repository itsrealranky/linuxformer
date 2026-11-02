"""The demo Space: a shell on the Linux machine inside Qwen3.5-4B's weights, and Qwen using that
shell as a tool. Every machine token is verified: the reference (mini-rv32ima's machine)
proposes the trace and the model checks each token, stopping at the first wrong one. Every
request starts from the booted snapshot. Needs one 80 GB GPU and serves one request at a time.

    python space/app.py --weights build/hf/model     # local, from tools/export_hf.py's output
    LX_REPO=user/name python app.py                   # on a Space: downloads the model repo
"""

import argparse
import os
import queue
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lx" if (HERE.parent / "lx" / "lx").exists() else HERE))

import gradio as gr  # noqa: E402
import torch  # noqa: E402

from lx.chat import MAX_LINE, Chat  # noqa: E402
from lx.linux import PROMPT, Linux  # noqa: E402

MAX_RECORDS = 3_000_000  # per command: about 4M tokens, 7 min verified
SHELL_EXAMPLES = ["echo hi", "echo $((123456 * 789))", "uname -a", "ls /", "cat /proc/version"]
QWEN_EXAMPLES = [
    "Compute 123456 * 789 using shell arithmetic on the machine, and tell me the result.",
    "What Linux kernel version is your shell machine running, and on what CPU architecture? "
    "Check with the shell, then answer in one sentence.",
    "Use a while loop in the shell to add up the numbers 1 to 10, and tell me the sum.",
]
INTRO = """\
# Linux inside the weights of Qwen3.5-4B

The checkpoint is stock Qwen3.5-4B, bit for bit, plus 37 layers that a compiler turned into a
RISC-V computer. There was no training. The computer's 4.5 MB of RAM lives in the layers' Gated
DeltaNet state. Linux 6.8 booted on it, and this page starts from that booted state.

**Every machine token is verified.** mini-rv32ima's machine proposes the trace, and the model
checks every next-token prediction in chunked prefill. One wrong token would stop the run. The
model itself does the work: fed its own outputs, it produces the same stream. Without the
reference, the stock decode runs at 80.8 tokens/s (an H100 megakernel in the repository: about
6,000), so this demo uses verified mode at about 9,000 tokens/s. A builtin like `echo hi` takes
about 20 s. A command that runs a program takes about 1.5 to 2 minutes. Each request starts from
the freshly booted machine.
"""

LX: Linux = None
CHAT: Chat = None
SNAPSHOT: Path = None
LOCK = threading.Lock()


def stream(job):
    """Run job(push) in a thread and yield whatever it pushes, until it returns."""
    q, err = queue.Queue(), []

    def target():
        try:
            job(q.put)
        except Exception as e:  # noqa: BLE001 - shown to the user
            err.append(e)
        finally:
            q.put(None)

    threading.Thread(target=target, daemon=True).start()
    while (item := q.get()) is not None:
        yield item
    if err:
        raise gr.Error(f"{type(err[0]).__name__}: {err[0]}")


def console(out: bytes) -> str:
    return PROMPT.decode() + out.decode(errors="replace").replace("\r\n", "\n")


def stats(done: int, total: int, seconds: float, finished: bool) -> str:
    s = (f"**{done:,} / {total:,}** machine tokens verified, **0 wrong**, {seconds:.0f} s on "
         f"{torch.cuda.get_device_name()}")
    return s + (" (done)" if finished else " (running)")


def check(text: str) -> None:
    if not text.strip():
        raise gr.Error("Type something first.")
    if max(len(ln) for ln in text.splitlines()) >= MAX_LINE:
        raise gr.Error(f"The machine's shell takes lines under {MAX_LINE} characters.")


def run_shell(cmd: str):
    check(cmd)

    def job(push):
        with LOCK:
            LX.boot(SNAPSHOT)
            push((console(b""), "proposing the trace..."))
            r = LX.run(cmd.strip(), MAX_RECORDS, progress=lambda sec, done, total, out: push(
                (console(out), stats(done, total, sec, False))))
            push((console(r["out"]), stats(r["tokens"], r["tokens"], r["seconds"], True)))

    yield from stream(job)


def ask_qwen(question: str):
    check(question)

    def job(push):
        with LOCK:
            LX.boot(SNAPSHOT)
            msgs = [{"role": "user", "content": question}]
            term = {"base": console(b""), "out": console(b""), "stats": ""}

            def show():
                push((list(msgs), term["out"], term["stats"]))

            def on_event(ev):
                if "command" not in ev:
                    msgs.append({"role": "assistant", "content": ev["qwen"]})
                elif "response" not in ev:
                    before = ev["qwen"].split("<tool_call>")[0].strip()
                    if before:
                        msgs.append({"role": "assistant", "content": before})
                    msgs.append({"role": "assistant", "content": f"```sh\n{ev['command']}\n```",
                                 "metadata": {"title": "shell (running on the machine)"}})
                else:
                    if "machine_out" in ev:
                        term["base"] += ev["machine_out"].replace("\r\n", "\n")
                        term["out"] = term["base"]
                    took = (f"{ev['machine_tokens']:,} machine tokens, {ev['machine_seconds']:.0f}"
                            " s, 0 wrong" if "machine_tokens" in ev else "not run")
                    msgs[-1] = {"role": "assistant", "content": f"```sh\n{ev['command']}\n```\n"
                                f"```\n{ev['response'].rstrip() or '(no output)'}\n```",
                                "metadata": {"title": f"shell ({took})"}}
                show()

            def progress(sec, done, total, out):
                term["out"] = term["base"] + out.decode(errors="replace").replace("\r\n", "\n")
                term["stats"] = stats(done, total, sec, False)
                show()

            show()
            CHAT.ask(question, max_calls=3, echo=lambda s: None, progress=progress,
                     max_records=MAX_RECORDS, on_event=on_event)
            term["stats"] = term["stats"].replace("(running)", "(done)")
            show()

    yield from stream(job)


def ui() -> gr.Blocks:
    with gr.Blocks(title="Linux inside Qwen3.5-4B") as demo:
        gr.Markdown(INTRO)
        with gr.Tab("Qwen uses its own Linux"):
            with gr.Row():
                with gr.Column(scale=5):
                    chat = gr.Chatbot(height=560, label="Qwen3.5-4B, with a shell tool")
                    q = gr.Textbox(label="Ask Qwen", placeholder=QWEN_EXAMPLES[0])
                    go = gr.Button("Ask", variant="primary")
                    gr.Examples(QWEN_EXAMPLES, q)
                with gr.Column(scale=6):
                    term_q = gr.Code(label="the machine's console", language=None, lines=26,
                                     interactive=False)
                    stat_q = gr.Markdown()
            go.click(ask_qwen, q, [chat, term_q, stat_q])
            q.submit(ask_qwen, q, [chat, term_q, stat_q])
        with gr.Tab("Shell"):
            cmd = gr.Textbox(label="Command", placeholder="echo hi")
            run = gr.Button("Run", variant="primary")
            gr.Examples(SHELL_EXAMPLES, cmd)
            term = gr.Code(label="the machine's console", language=None, lines=20,
                           interactive=False)
            stat = gr.Markdown()
            run.click(run_shell, cmd, [term, stat])
            cmd.submit(run_shell, cmd, [term, stat])
    return demo


def main():
    global LX, CHAT, SNAPSHOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.environ.get("LX_WEIGHTS", ""))
    ap.add_argument("--repo", default=os.environ.get("LX_REPO", ""))
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--fast", action="store_true", default=bool(os.environ.get("LX_FAST")),
                    help="verify with lx.fast (the same weights, zeros skipped)")
    ap.add_argument("--lean", action="store_true", default=bool(os.environ.get("LX_LEAN")),
                    help="lx.fast, the chip projections kept in host memory (a 24 GB GPU)")
    a = ap.parse_args()
    if a.weights:
        weights = Path(a.weights)
    else:
        from huggingface_hub import snapshot_download

        weights = Path(snapshot_download(a.repo))
    LX = Linux(weights=weights, fast=a.fast, lean=a.lean)
    SNAPSHOT = weights / "boot-snapshot.safetensors"
    LX.boot(SNAPSHOT)
    CHAT = Chat(LX)
    ui().queue(default_concurrency_limit=1).launch(server_port=a.port)


if __name__ == "__main__":
    main()
