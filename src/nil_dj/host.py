"""JSON-RPC 2.0 over stdio, the way nil's plugin protocol (PLUGIN-API.md) defines it.

One message per line, UTF-8, at most MAX_LINE bytes. Errors and diagnostics go to stderr only;
in production stdout is the protocol channel.
"""

import json
import queue
import sys
import threading
import traceback
from collections.abc import Callable
from typing import Any, TextIO

MAX_LINE = 1_048_576

PERMISSION = -32001
NOT_FOUND = -32002
BUSY = -32003
STALE_QUEUE = -32004  # player.remove / move: the queue changed, or the track has started
INTERNAL = -32603
INVALID_PARAMS = -32602
METHOD_NOT_FOUND = -32601

_CHUNK = 65536


class NilError(Exception):
    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


class MessageTooLarge(ValueError):
    pass


def encode(msg: dict) -> str:
    text = json.dumps(msg, ensure_ascii=False, separators=(",", ":"))
    if len(text.encode("utf-8")) > MAX_LINE:
        raise MessageTooLarge(f"message is over {MAX_LINE} bytes")
    return text + "\n"


def _log(text: str) -> None:
    print(f"nil-dj host: {text}", file=sys.stderr, flush=True)


class Connection:
    """A JSON-RPC connection to nil.

    Threads:
    - `serve()` is the reader thread. `on_request` runs on it, so a request handler must
      not call `call()`: the reply could only be read by the thread that is waiting for it,
      and it would deadlock. Keep request handlers short.
    - `on_notification` runs on a separate worker thread, in order, and may call `call()`.
    - `call()` and replies may be written from any thread; writes are serialised.

    Bad input never ends the connection. If `reader` supports `reconfigure` (a
    `TextIOWrapper`, such as a wrapped `sys.stdin`), it is switched to `errors="replace"`
    so invalid UTF-8 becomes U+FFFD and the line is then ignored or parsed as usual. A reader
    that can't be reconfigured must not raise on decoding; a stray `UnicodeDecodeError` is
    logged and reading continues.
    """

    def __init__(
        self,
        reader: TextIO,
        writer: TextIO,
        on_request: Callable[[str, dict], Any],
        on_notification: Callable[[str, dict], None],
        timeout_s: float = 10.0,
    ):
        try:
            reader.reconfigure(errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError, OSError):
            pass
        self._reader = reader
        self._writer = writer
        self._on_request = on_request
        self._on_notification = on_notification
        self.timeout_s = timeout_s
        self.closed = threading.Event()
        self._write_lock = threading.Lock()
        self._lock = threading.Lock()  # guards _next_id, _pending and closed transitions
        self._next_id = 0
        self._pending: dict[int, dict] = {}
        self._notes: queue.Queue = queue.Queue()

    # -- sending ---------------------------------------------------------------------------

    def _send(self, msg: dict) -> None:
        line = encode(msg)
        with self._write_lock:
            try:
                self._writer.write(line)
                self._writer.flush()
            except (OSError, ValueError) as e:
                raise NilError(INTERNAL, "nil has closed the connection") from e

    def call(self, method: str, params: dict | None = None) -> Any:
        slot = {"event": threading.Event(), "reply": None}
        with self._lock:
            if self.closed.is_set():
                raise NilError(INTERNAL, "nil has closed the connection")
            self._next_id += 1
            msg_id = self._next_id
            self._pending[msg_id] = slot
        try:
            self._send(
                {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params or {}}
            )
            if not slot["event"].wait(self.timeout_s):
                raise NilError(INTERNAL, "nil didn't answer")
        finally:
            with self._lock:
                self._pending.pop(msg_id, None)
        reply = slot["reply"]
        if reply is None:
            raise NilError(INTERNAL, "nil has closed the connection")
        if "error" in reply:
            err = reply["error"]
            if not isinstance(err, dict):
                raise NilError(INTERNAL, "malformed error reply from nil")
            raise NilError(
                err.get("code", INTERNAL), str(err.get("message", "")), err.get("data")
            )
        return reply.get("result")

    def notify(self, method: str, params: dict | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    # -- receiving -------------------------------------------------------------------------

    def _read_line(self) -> str | None:
        """Next line, "" for one that was too long (already skipped), None at EOF."""
        parts: list[str] = []
        size = 0
        too_long = False
        while True:
            chunk = self._reader.readline(_CHUNK)
            if not chunk:
                return None if not parts and not too_long else "".join(parts)
            if not too_long:
                parts.append(chunk)
                size += len(chunk)
                # A character is at least one byte, so this never rejects a legal line.
                if size > MAX_LINE + 2:
                    too_long = True
                    parts = []
            if chunk.endswith("\n"):
                break
        line = "" if too_long else "".join(parts)
        if too_long or len(line.rstrip("\r\n").encode("utf-8")) > MAX_LINE:
            _log("ignoring a line that is too long")
            return ""
        return line

    def serve(self) -> None:
        worker = threading.Thread(target=self._work, daemon=True, name="nil-dj-notifications")
        worker.start()
        try:
            while True:
                try:
                    line = self._read_line()
                except UnicodeDecodeError:
                    _log("ignoring a line that is not valid UTF-8")
                    continue
                except (OSError, ValueError):
                    break
                if line is None:
                    break
                line = line.strip()
                if line:
                    self._dispatch(line)
        finally:
            with self._lock:
                self.closed.set()
                pending = list(self._pending.values())
            for slot in pending:
                slot["event"].set()  # reply stays None: "closed"
            self._notes.put(None)

    def _dispatch(self, line: str) -> None:
        try:
            msg = json.loads(line)
        except ValueError:
            _log(f"ignoring a line that is not JSON: {line[:80]!r}")
            return
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            _log(f"ignoring a line that is not JSON-RPC 2.0: {line[:80]!r}")
            return
        method = msg.get("method")
        if method is None:
            self._on_reply(msg)
            return
        if not isinstance(method, str):
            _log("ignoring a message whose method is not a string")
            return
        params = msg.get("params")
        if not isinstance(params, dict):
            params = {}
        # PLUGIN-API only defines `"id": null` for requests sent to nil, so on this side a
        # null id is treated like no id: a notification.
        if msg.get("id") is None:
            self._notes.put((method, params))
        else:
            self._on_incoming_request(msg["id"], method, params)

    def _on_reply(self, msg: dict) -> None:
        msg_id = msg.get("id")
        with self._lock:
            slot = self._pending.get(msg_id) if isinstance(msg_id, int) else None
            if slot is not None:
                slot["reply"] = msg
        if slot is None:
            _log(f"dropping a reply with unknown id {msg_id!r}")
        else:
            slot["event"].set()

    def _on_incoming_request(self, msg_id: Any, method: str, params: dict) -> None:
        try:
            result = self._on_request(method, params)
            reply: dict = {"jsonrpc": "2.0", "id": msg_id, "result": result}
        except NilError as e:
            reply = self._error_reply(msg_id, e.code, e.message, e.data)
        except Exception as e:  # noqa: BLE001 - any handler failure becomes an error reply
            reply = self._error_reply(msg_id, INTERNAL, str(e) or type(e).__name__)
        try:
            try:
                self._send(reply)
            except MessageTooLarge:
                self._send(self._error_reply(msg_id, INTERNAL, "reply is too large"))
            except NilError:
                raise
            except Exception as e:  # noqa: BLE001 - e.g. a result JSON can't encode
                self._send(self._error_reply(msg_id, INTERNAL, str(e) or type(e).__name__))
        except NilError:
            pass  # the connection is gone; nothing to tell

    @staticmethod
    def _error_reply(msg_id: Any, code: int, message: str, data: Any = None) -> dict:
        error: dict = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        return {"jsonrpc": "2.0", "id": msg_id, "error": error}

    def _work(self) -> None:
        while True:
            item = self._notes.get()
            if item is None:
                return
            method, params = item
            try:
                self._on_notification(method, params)
            except Exception:  # noqa: BLE001 - a bad handler must not stop the worker
                traceback.print_exc(file=sys.stderr)
