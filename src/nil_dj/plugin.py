"""The nil plugin process: handshake, lifecycle and stop handling.

nil starts `nil-dj plugin` with the plugin folder as working directory and talks JSON-RPC over
stdio (PLUGIN-API.md). stdout is the protocol channel and nothing else may write to it.
"""

from __future__ import annotations

import io
import json
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
from nil_dj.dj import DJ, default_embedder
from nil_dj.embedder import Embedder
from nil_dj.host import METHOD_NOT_FOUND, Connection, NilError
from nil_dj.panel import MCP_NOT_RUNNING, Panel

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


def _start(
    conn: Connection,
    cfg: Config,
    make_embedder: Callable[[Config], Embedder],
    start_mcp: bool,
    state: dict,
    mcp_status: Callable[[], str] = lambda: MCP_NOT_RUNNING,
) -> DJ:
    """Build the app once nil says `initialized`; the panel goes in `state["panel"]`. Runs on
    the notification worker thread, so nothing slow here: the model loads on its own thread."""
    dj = DJ(cfg, conn, lambda: make_embedder(cfg))
    on_stop(dj.stop)
    conn.call("panel.set", {"panel": {"elements": [{"type": "heading", "text": "nil-dj"}]}})
    panel = state["panel"] = Panel(dj, conn.call, mcp_status, problems=cfg.problems)
    on_stop(panel.close)
    try:
        dj.refresh_library()
    except NilError as e:  # e.g. no library permission: pending stays unknown
        print(f"nil-dj: can't read nil's library: {e.message}", file=sys.stderr)
    panel.refresh(force=True)
    dj.preload()
    return dj


class _ReplyWatch:
    """Wraps the streams to learn the id of nil's `shutdown` request and to signal once the
    reply with that id has been flushed."""

    def __init__(self, reader: TextIO, writer: TextIO):
        self._reader = reader
        self._writer = writer
        self.shutdown_id: object = None
        self.replied = threading.Event()
        self._is_reply = False

    def __getattr__(self, name: str):  # reconfigure() and friends go to the real reader
        return getattr(self._reader, name)

    def readline(self, size: int = -1) -> str:
        line = self._reader.readline(size)
        if '"shutdown"' in line:
            try:
                msg = json.loads(line)
            except ValueError:
                return line
            if isinstance(msg, dict) and msg.get("method") == "shutdown":
                self.shutdown_id = msg.get("id")
        return line

    def write(self, text: str) -> int:
        self._is_reply = False
        if self.shutdown_id is not None:
            try:
                msg = json.loads(text)
            except ValueError:
                msg = None
            self._is_reply = (
                isinstance(msg, dict) and "method" not in msg and msg.get("id") == self.shutdown_id
            )
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
    out = _ReplyWatch(reader, writer)
    done = threading.Event()
    state: dict = {}  # "cfg" from initialize, "dj" once initialized

    def on_request(method: str, params: dict):
        if method == "initialize":
            data_dir = params.get("data_dir")
            state["cfg"] = load_config(plugin_dir, Path(data_dir) if data_dir else None)
            granted[:] = [p for p in params.get("granted", []) if isinstance(p, str)]
            return {"name": "nil-dj", "version": __version__}
        if method == "shutdown":
            done.set()  # the reply goes out when this returns; the main thread waits for it
            return {}
        raise NilError(METHOD_NOT_FOUND, f"unknown method {method}")

    def on_notification(method: str, params: dict) -> None:
        if method == "initialized" and "cfg" in state:
            state["dj"] = _start(conn, state["cfg"], make_embedder, start_mcp, state)
        elif method == "player.changed" and "dj" in state:
            state["dj"].radio.on_player_changed(params)
        elif method == "panel.event" and "panel" in state:
            state["panel"].on_event(params)

    conn = Connection(out, out, on_request, on_notification)  # type: ignore[arg-type]

    def serve() -> None:
        try:
            conn.serve()
        finally:
            done.set()

    try:
        threading.Thread(target=serve, daemon=True, name="nil-dj-reader").start()
        done.wait()
        if out.shutdown_id is not None:
            deadline = time.monotonic() + 1.0
            while not out.replied.wait(0.01) and not conn.closed.is_set():
                if time.monotonic() > deadline:
                    break
        _run_stop_callbacks()
        return 0
    finally:
        granted.clear()
        _stop_callbacks.clear()


def main() -> int:
    code = 1
    try:
        proto = guard_stdout()
        stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace", newline="\n")
        code = run(stdin, proto, Path.cwd(), default_embedder)
    except BaseException:  # noqa: BLE001 - report it, then still leave promptly
        traceback.print_exc(file=sys.stderr)
    finally:
        sys.stderr.flush()
        os._exit(code)  # worker threads (ffmpeg pool, uvicorn) must not hold the exit past 3 s
