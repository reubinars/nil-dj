"""The nil plugin process: handshake, lifecycle and stop handling.

nil starts `nil-dj plugin` with the plugin folder as working directory and talks JSON-RPC over
stdio (PLUGIN-API.md). stdout is the protocol channel and nothing else may write to it.
"""

from __future__ import annotations

import io
import os
import sys
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from nil_dj import __version__
from nil_dj.config import Config, load_config
from nil_dj.embedder import Embedder
from nil_dj.host import METHOD_NOT_FOUND, Connection, NilError

STOP_BUDGET_S = 2.0  # all stop callbacks together; nil ends the process after 3 s

granted: list[str] = []  # permissions nil approved, set by `initialize`
_stop_callbacks: list[Callable[[], None]] = []


def on_stop(fn: Callable[[], None]) -> None:
    """Register `fn` to run (with a deadline) when nil shuts the plugin down or stdin closes."""
    _stop_callbacks.append(fn)


def guard_stdout() -> TextIO:
    """Return the protocol stream; everything else that writes to stdout goes to stderr."""
    proto = os.fdopen(os.dup(1), "w", encoding="utf-8", newline="\n")
    os.dup2(2, 1)  # anything else writing to fd 1 (uvicorn, prints, C libs) lands in stderr
    sys.stdout = sys.stderr
    return proto


def default_embedder(cfg: Config) -> Embedder:
    import argparse

    from nil_dj.cli import make_embedder

    return make_embedder(cfg, argparse.Namespace(precision=None, device_id=None, cpu=False))


def _start(
    conn: Connection,
    cfg: Config,
    make_embedder: Callable[[Config], Embedder],
    start_mcp: bool,
) -> None:
    """Build the app once nil says `initialized`. Runs on the notification worker thread."""
    conn.call("panel.set", {"panel": {"elements": [{"type": "heading", "text": "nil-dj"}]}})


class _ReplyWatch:
    """Writer wrapper that signals once the reply to `shutdown` has been flushed."""

    def __init__(self, writer: TextIO):
        self._writer = writer
        self.shutting_down = threading.Event()
        self.replied = threading.Event()
        self._is_reply = False

    def write(self, text: str) -> int:
        self._is_reply = self.shutting_down.is_set() and '"result"' in text
        return self._writer.write(text)

    def flush(self) -> None:
        self._writer.flush()
        if self._is_reply:
            self.replied.set()


def _run_stop_callbacks() -> None:
    deadline = time.monotonic() + STOP_BUDGET_S

    def guarded(fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception:  # noqa: BLE001 - one bad callback must not block the others
            traceback.print_exc(file=sys.stderr)

    for fn in _stop_callbacks.copy():
        t = threading.Thread(target=guarded, args=(fn,), daemon=True, name="nil-dj-stop")
        t.start()
        t.join(max(0.0, deadline - time.monotonic()))


def run(
    reader: TextIO,
    writer: TextIO,
    plugin_dir: Path,
    make_embedder: Callable[[Config], Embedder],
    start_mcp: bool = True,
) -> int:
    """Serve nil until `shutdown` or EOF on `reader`. Returns the exit code (always 0)."""
    granted.clear()
    _stop_callbacks.clear()
    out = _ReplyWatch(writer)
    done = threading.Event()
    state: dict[str, Config] = {}

    def on_request(method: str, params: dict):
        if method == "initialize":
            data_dir = params.get("data_dir")
            state["cfg"] = load_config(plugin_dir, Path(data_dir) if data_dir else None)
            granted[:] = [p for p in params.get("granted", []) if isinstance(p, str)]
            return {"name": "nil-dj", "version": __version__}
        if method == "shutdown":
            out.shutting_down.set()
            done.set()  # the reply goes out when this returns; the main thread waits for it
            return {}
        raise NilError(METHOD_NOT_FOUND, f"unknown method {method}")

    def on_notification(method: str, params: dict) -> None:
        if method == "initialized" and "cfg" in state:
            _start(conn, state["cfg"], make_embedder, start_mcp)

    conn = Connection(reader, out, on_request, on_notification)  # type: ignore[arg-type]

    def serve() -> None:
        try:
            conn.serve()
        finally:
            done.set()

    threading.Thread(target=serve, daemon=True, name="nil-dj-reader").start()
    done.wait()
    if out.shutting_down.is_set():
        out.replied.wait(1.0)
    _run_stop_callbacks()
    return 0


def main() -> int:
    proto = guard_stdout()
    stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace", newline="\n")
    code = run(stdin, proto, Path.cwd(), default_embedder)
    sys.stderr.flush()
    os._exit(code)  # worker threads (ffmpeg pool, uvicorn) must not hold the exit past 3 s
