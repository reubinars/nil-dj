"""The nil-dj panel: what nil shows in the plugin's tab, and what its controls do.

`build` turns a `View` into the `panel.set` payload and has no side effects. `Panel` keeps the
view's state (the last prompt, result and message), sends it, and handles `panel.event`.

Threads: `refresh` is called from anywhere (DJ listeners run on the thread that changed
something) and never blocks on the throttle: a refresh inside the interval arms one timer that
sends the latest state when the interval is over. `on_event` runs on the notification worker and
only hands the event to the panel's own thread (`nil-dj-panel`), one event at a time and in
order, so a search waiting for the model never holds up radio's `player.changed`. Never call
either from the reader thread.
"""

from __future__ import annotations

import sys
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from nil_dj.dj import DJ, IndexStatus, NotInNil, Playlist
from nil_dj.host import NilError

MCP_NOT_RUNNING = "MCP: not running"
EMPTY_PROMPT = "Type what you'd like to hear."
PLAYLIST_MINUTES = 45
UNEXPECTED = "Something went wrong: {}. Details are in nil's plugin log."


@dataclass
class View:
    status: IndexStatus
    model_state: str
    prompt: str = ""
    result: Playlist | None = None
    message: str | None = None
    radio_on: bool = False
    mcp: str = MCP_NOT_RUNNING
    problems: list[str] = field(default_factory=list)


def _text(text: str, muted: bool = False) -> dict:
    el = {"type": "text", "text": text}
    if muted:
        el["muted"] = True
    return el


def _index_elements(s: IndexStatus) -> list[dict]:
    if s.running:
        value = s.done / s.total if s.total else None
        return [{"type": "progress", "value": value,
                 "label": f"Indexing {s.done:,} of {s.total:,}"}]  # fmt: skip
    if s.problem:
        return [_text(s.problem, muted=True)]
    line = f"{s.indexed:,} tracks indexed"
    if s.pending:
        line += f" · {s.pending:,} not indexed yet"
    if s.failed:
        line += f" · {s.failed:,} failed"
    return [_text(line, muted=True)]


def _model_elements(state: str) -> list[dict]:
    if state == "downloading":
        label = "Downloading the sound model (once, about 620 MB)…"
        return [{"type": "progress", "value": None, "label": label}]
    if state == "loading":
        return [{"type": "progress", "value": None, "label": "Loading the sound model…"}]
    if state.startswith("failed"):
        return [_text(f"Sound model {state}")]
    if state not in ("not loaded", "ready"):  # a failed download says it all itself
        return [_text(state)]
    return []


def _index_buttons(s: IndexStatus) -> list[dict]:
    if s.running:
        return []
    buttons = []
    if s.rebuild_needed:  # pending is the whole library then; updating can't help
        buttons.append({"type": "button", "id": "rebuild", "label": "Rebuild index"})
    elif s.indexed == 0:
        buttons.append({"type": "button", "id": "index", "label": "Index library"})
    else:  # even with nothing pending: nil's library may have changed since it was read
        buttons.append({"type": "button", "id": "index", "label": "Update index"})
    return [{"type": "row", "children": buttons}] if buttons else []


def build(view: View) -> dict:
    els: list[dict] = [{"type": "heading", "text": "nil-dj"}]
    els += [_text(p, muted=True) for p in view.problems]
    els += _index_elements(view.status)
    if view.message != view.model_state:  # the same sentence twice reads as two problems
        els += _model_elements(view.model_state)
    els += _index_buttons(view.status)
    els.append({"type": "input", "id": "prompt", "placeholder": "Describe a playlist…",
                "submit_label": "Make", "value": view.prompt})  # fmt: skip
    els.append({"type": "row", "children": [
        {"type": "button", "id": "more", "label": "More like this"},
        {"type": "toggle", "id": "radio", "label": "Radio", "value": view.radio_on},
    ]})  # fmt: skip
    if view.message:
        els.append(_text(view.message))
    if view.result is not None:
        els.append({"type": "heading", "text": view.result.title})
        if view.result.tracks:
            els.append({"type": "tracks", "items": [t.item() for t in view.result.tracks]})
        else:
            els.append(_text("No matching tracks."))
    els.append(_text(view.mcp, muted=True))
    return {"elements": els}


class Panel:
    def __init__(
        self,
        dj: DJ,
        call: Callable[..., Any],
        mcp_status: Callable[[], str],
        min_interval_s: float = 0.5,
        problems: list[str] | None = None,
    ):
        self.dj = dj
        self.call = call
        self.mcp_status = mcp_status
        self.min_interval_s = min_interval_s
        self.problems = list(problems or [])
        self.prompt = ""
        self.result: Playlist | None = None
        self.message: str | None = None
        self._lock = threading.Lock()  # the state above and the throttle below
        self._send_lock = threading.Lock()  # one panel.set at a time, newest view last
        self._last = float("-inf")
        self._timer: threading.Timer | None = None
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nil-dj-panel")
        dj.listeners.append(self.refresh)
        previous = dj.radio.on_change

        def radio_changed() -> None:
            self._set_message(None)  # so the radio's own message can show
            previous()

        dj.radio.on_change = radio_changed

    # -- sending ---------------------------------------------------------------------------

    def refresh(self, force: bool = False) -> None:
        """Send the panel now (`force`, or when the interval is over), else once at its end."""
        with self._lock:
            if self._closed:
                return
            now = time.monotonic()
            due = self._last + self.min_interval_s
            if not force and now < due:
                if self._timer is None:
                    self._timer = threading.Timer(due - now, self._trailing)
                    self._timer.daemon = True
                    self._timer.start()
                return
            self._last = now
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        self._send()

    def _trailing(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._timer = None
            self._last = time.monotonic()
        self._send()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        # A search may still be waiting for the model: don't wait for it.
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _send(self) -> None:
        with self._send_lock:  # the view is built inside, so a later send is never staler
            try:
                self.call("panel.set", {"panel": build(self._view())})
            except Exception as e:  # noqa: BLE001 - callers (timer, events, startup) rely on this
                if not self.dj._closed():  # at shutdown nobody is left to tell
                    print(f"nil-dj: panel.set failed: {e}", file=sys.stderr)

    def _view(self) -> View:
        status = self.dj.index_status()  # opens the index: once per send
        with self._lock:
            prompt, result, message = self.prompt, self.result, self.message
        return View(
            status=status,
            model_state=self.dj.model_state,
            prompt=prompt,
            result=result,
            message=message or self.dj.radio.message,
            radio_on=self.dj.radio.on,
            mcp=self.mcp_status(),
            problems=self.problems,
        )

    def _set_message(self, message: str | None) -> None:
        with self._lock:
            self.message = message

    # -- events ----------------------------------------------------------------------------

    def on_event(self, params: dict) -> None:
        """Queue the event for the panel's thread and return at once."""
        try:
            self._executor.submit(self._event, params)
        except RuntimeError:  # closed: nil is shutting the plugin down
            pass

    def _event(self, params: dict) -> None:
        id, value = params.get("id"), params.get("value")
        try:
            self.dj.refresh_library_if_stale()  # nil rescans its folders without telling us
            self._handle(id, value)
        except (ValueError, NilError, NotInNil) as e:
            self._set_message(str(e))
        except Exception as e:  # noqa: BLE001 - the panel must say something and stay usable
            traceback.print_exc(file=sys.stderr)
            self._set_message(UNEXPECTED.format(type(e).__name__))
        finally:
            self.refresh(force=True)

    def _handle(self, id: object, value: object) -> None:
        if id == "prompt":
            text = value.strip() if isinstance(value, str) else ""
            if not text:
                self._set_message(EMPTY_PROMPT)
                return
            with self._lock:
                self.prompt, self.message = value, None
            self.refresh(force=True)  # shows the model loading while the search waits for it
            result = self.dj.build_playlist(text, PLAYLIST_MINUTES)
            with self._lock:
                self.result = result
        elif id == "more":
            self._set_message(None)
            result = self.dj.more_like_playing()
            with self._lock:
                self.result = result
        elif id == "radio":
            self._set_message(None)
            self.dj.set_radio(value is True)
        elif id == "index":
            self._set_message(None)
            self.dj.reindex()
        elif id == "rebuild":
            self._set_message(None)
            self.dj.reindex(rebuild=True)
