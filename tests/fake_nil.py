# ruff: noqa: TRY004, SIM115
"""A fake of nil's side of the plugin protocol (PLUGIN-API.md), for plugin tests."""

import os
import queue
import re
import threading
import time

from nil_dj.host import (
    INTERNAL,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    NOT_FOUND,
    PERMISSION,
    STALE_QUEUE,
    Connection,
    NilError,
)

_NEEDS = {
    "player.state": "playback",
    "player.play": "playback",
    "player.append": "playback",
    "player.remove": "playback",
    "player.move": "playback",
    "player.clear_upcoming": "playback",
    "library.folders": "library",
    "library.tracks": "library",
    "playlists.list": "playlists",
    "playlists.read": "playlists",
    "playlists.save": "playlists",
}
_MAX_ROW_DEPTH = 4
_TRANSPORT = ("nil has closed the connection", "nil didn't answer")  # Connection.call failures


def _key(path: str) -> str:
    return path.replace("\\", "/").lower()


def _is_num(v) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


_SETTINGS_TYPES = ("heading", "text", "progress", "button", "toggle", "row")


def validate_panel(panel, place: str = "tab") -> None:
    """Raise ValueError unless `panel` follows the Panels table of PLUGIN-API.md (for `place`
    "settings": only the elements nil 0.2.0 allows in a plugin's Settings entry)."""
    if not isinstance(panel, dict) or not isinstance(panel.get("elements"), list):
        raise ValueError("panel needs an elements list")
    ids: set[str] = set()

    def need(el: dict, field: str, kind: type) -> None:
        if not isinstance(el.get(field), kind):
            raise ValueError(f"{el.get('type')}: '{field}' must be {kind.__name__}")

    def optional(el: dict, field: str, kind: type) -> None:
        if field in el and not isinstance(el[field], kind):
            raise ValueError(f"{el.get('type')}: '{field}' must be {kind.__name__}")

    def check(el, depth: int) -> None:
        if not isinstance(el, dict) or not isinstance(el.get("type"), str):
            raise ValueError("every element needs a type")
        kind = el["type"]
        if place == "settings" and kind not in _SETTINGS_TYPES:
            raise ValueError(f"{kind} is not allowed in settings")
        if "id" in el:
            if not isinstance(el["id"], str):
                raise ValueError("id must be a string")
            if el["id"] in ids:
                raise ValueError(f"duplicate id {el['id']!r}")
            ids.add(el["id"])
        if kind in ("heading", "text"):
            need(el, "text", str)
            optional(el, "muted", bool)
        elif kind == "input":
            for f in ("id", "placeholder", "submit_label"):
                need(el, f, str)
            optional(el, "value", str)
        elif kind == "button":
            need(el, "id", str)
            need(el, "label", str)
            optional(el, "enabled", bool)
        elif kind == "toggle":
            need(el, "id", str)
            need(el, "label", str)
            need(el, "value", bool)
        elif kind == "progress":
            if "value" not in el or not (el["value"] is None or _is_num(el["value"])):
                raise ValueError("progress: 'value' must be a number or null")
            optional(el, "label", str)
        elif kind == "tracks":
            need(el, "items", list)
            optional(el, "label", str)
            for item in el["items"]:
                if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                    raise ValueError("tracks: every item needs a path")
                optional(item, "title", str)
                optional(item, "artist", str)
        elif kind == "row":
            if depth > _MAX_ROW_DEPTH:
                raise ValueError("rows nest at most 4 deep")
            need(el, "children", list)
            for child in el["children"]:
                check(child, depth + 1)

    for el in panel["elements"]:
        check(el, 1)


class _CountingWriter:
    """Counts flushed lines, so FakeNil can tell when a reply has been written."""

    def __init__(self, writer):
        self._writer = writer
        self.flushes = 0

    def write(self, text):
        return self._writer.write(text)

    def flush(self):
        self._writer.flush()
        self.flushes += 1

    def close(self):
        self._writer.close()


class FakeNil:
    """nil as a plugin sees it: answers requests, sends notifications, records panels."""

    def __init__(self, tracks: list[dict], folders: list[str] = ["E:\\Music"]):  # noqa: B006
        self.tracks = tracks
        self.folders = folders
        self.state = "stopped"
        self.queue: list[str] = []
        self.index: int | None = None
        self.next_starting = False  # the decoder has taken the track after the current one
        self.playlists: dict[str, list[str]] = {}
        self.panels: list[dict] = []  # the tab's
        self.settings_panels: list[dict] = []  # `place: "settings"`: the Settings entry's
        self.calls: list[tuple[str, dict]] = []
        self.fail: dict[str, tuple] = {}  # method -> (code, message[, data])
        self.granted: list[str] = []
        self.init_result: dict | None = None
        self.run_thread: threading.Thread | None = None  # set by the running_plugin fixture
        self.exit_code: int | None = None
        self._errors: list[str] = []
        self._lock = threading.Lock()
        # fd pairs: nil -> plugin, plugin -> nil
        r1, w1 = os.pipe()
        r2, w2 = os.pipe()
        self._plugin_reader = open(r1, encoding="utf-8", newline="\n")
        self._nil_writer = _CountingWriter(open(w1, "w", encoding="utf-8", newline="\n"))
        self._nil_reader = open(r2, encoding="utf-8", newline="\n")
        self._plugin_writer = open(w2, "w", encoding="utf-8", newline="\n")
        self.conn = Connection(
            self._nil_reader, self._nil_writer, self._on_request, lambda m, p: None, 5.0
        )
        self._announced = ("stopped", None, 0)
        self._frames: queue.Queue = queue.Queue()
        self._known = {_key(t["path"]) for t in tracks}

    # -- wiring ----------------------------------------------------------------------------

    def streams(self):
        """(plugin_reader, plugin_writer): what `plugin.run` is given."""
        return self._plugin_reader, self._plugin_writer

    def start(self) -> None:
        threading.Thread(target=self.conn.serve, daemon=True, name="fake-nil").start()
        threading.Thread(target=self._frame_loop, daemon=True, name="fake-nil-frames").start()

    def handshake(
        self, granted=("playback", "library", "playlists"), data_dir="data", nil_version="0.2.0"
    ) -> dict:
        """`nil_version` None leaves the key out, as a nil that doesn't send it would."""
        self.granted = list(granted)
        params = {"api_version": 1, "granted": list(granted), "data_dir": str(data_dir)}
        if nil_version is not None:
            params["nil_version"] = nil_version
        reply = self.request("initialize", params)
        assert "result" in reply, reply
        self.init_result = reply["result"]
        self.notify("initialized", {})
        return self.init_result

    def notify(self, method: str, params: dict) -> None:
        self.conn.notify(method, params)

    def request(self, method: str, params: dict) -> dict:
        """The reply as `{"result": ...}` or `{"error": {"code", "message", "data"?}}`."""
        try:
            return {"result": self.conn.call(method, params)}
        except NilError as e:
            if e.code == INTERNAL and e.message in _TRANSPORT:
                raise  # connection trouble, not an error reply
            error = {"code": e.code, "message": e.message}
            if e.data is not None:
                error["data"] = e.data
            return {"error": error}

    def advance(self, index: int) -> None:
        self.index = index
        self.state = "playing"
        self._frames.put(None)

    def close(self) -> None:
        """Close every stream (the plugin sees EOF on stdin), and raise anything the panel checks found."""
        self._frames.put("stop")
        for f in (self._nil_writer, self._plugin_writer, self._plugin_reader, self._nil_reader):
            try:
                f.close()
            except (OSError, ValueError):
                pass
        self.raise_errors()

    def raise_errors(self) -> None:
        with self._lock:
            errors, self._errors = self._errors, []
        if errors:
            raise AssertionError("; ".join(errors))

    def wait_for(self, pred, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            self.raise_errors()
            if pred():
                return True
            if time.monotonic() > deadline:
                raise AssertionError("timed out waiting for the condition")
            time.sleep(0.01)

    def element(self, id: str) -> dict:
        """The element `id` in the last tab panel."""
        return self._find(self.panels, id)

    def settings_element(self, id: str) -> dict:
        """The element `id` in the last settings panel."""
        return self._find(self.settings_panels, id)

    @staticmethod
    def _find(panels: list[dict], id: str) -> dict:
        def find(els):
            for el in els:
                if el.get("id") == id:
                    return el
                if el.get("type") == "row" and (hit := find(el["children"])):
                    return hit
            return None

        hit = find(panels[-1]["elements"]) if panels else None
        if hit is None:
            raise KeyError(id)
        return hit

    # -- nil's behaviour -------------------------------------------------------------------

    def _frame_loop(self) -> None:
        """Like nil's UI frame: after the commands of a frame are handled (and their replies
        written), announce the player once, and only if (state, index, queue_len) changed."""
        while True:
            flushes = self._frames.get()
            if flushes == "stop":
                return
            deadline = time.monotonic() + 1.0
            while (
                flushes is not None
                and self._nil_writer.flushes < flushes
                and time.monotonic() < deadline
            ):
                time.sleep(0.001)
            while not self._frames.empty():  # several commands in one frame: one notification
                if self._frames.get() == "stop":
                    return
            self._announce()

    def _announce(self) -> None:
        if "playback" not in self.granted:
            return
        index = self.index if self.state != "stopped" else None
        key = (self.state, index, len(self.queue))
        if key == self._announced:
            return
        self._announced = key
        params: dict = {"state": self.state, "queue_len": len(self.queue)}
        if index is not None:
            params["index"] = index
            params["path"] = self.queue[index]
        try:
            self.conn.notify("player.changed", params)
        except NilError:
            pass

    def _check_paths(self, paths) -> None:
        unknown = [p for p in paths if _key(p) not in self._known]
        if unknown:
            raise NilError(
                NOT_FOUND, f"not in nil's library: {len(unknown)} path(s)", {"paths": unknown}
            )

    def _paths_param(self, params: dict) -> list[str]:
        paths = params.get("paths")
        if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
            raise NilError(INVALID_PARAMS, "paths must be an array of strings")
        return paths

    def _queue_edit(self, params: dict, with_to: bool = False) -> set[int]:
        """nil's checks for player.remove / player.move, in nil's order; returns the indices."""
        tracks, to = params.get("tracks"), params.get("to")
        ok = isinstance(tracks, list) and bool(tracks)
        ok = ok and all(
            isinstance(t, dict)
            and isinstance(t.get("index"), int)
            and not isinstance(t["index"], bool)
            and t["index"] >= 0  # nil's index is unsigned
            and isinstance(t.get("path"), str)
            for t in tracks
        )
        indices = [t["index"] for t in tracks] if ok else []
        ok = ok and len(set(indices)) == len(indices)
        if with_to:
            ok = ok and isinstance(to, int) and not isinstance(to, bool)
            ok = ok and 0 <= to <= len(self.queue)
        if not ok:
            raise NilError(INVALID_PARAMS, "bad queue edit params")
        if self.state == "stopped" or self.index is None:
            raise NilError(STALE_QUEUE, "queue changed", {"reason": "stopped"})
        for t in tracks:
            i = t["index"]
            if not 0 <= i < len(self.queue) or _key(self.queue[i]) != _key(t["path"]):
                raise NilError(STALE_QUEUE, "queue changed", {"reason": "changed"})
        taken = self.index + (1 if self.next_starting else 0)  # the last index nil has taken
        if any(i <= taken for i in indices) or (with_to and to <= taken):
            raise NilError(STALE_QUEUE, "queue changed", {"reason": "started"})
        return set(indices)

    def _on_request(self, method: str, params: dict):
        self.calls.append((method, params))
        need = _NEEDS.get(method)
        if need and need not in self.granted:
            raise NilError(PERMISSION, f"needs the {need} permission")
        if method in self.fail:
            code, message, *data = self.fail[method]
            raise NilError(code, message, *data)
        if method == "panel.set":
            panel, place = params.get("panel"), params.get("place", "tab")
            try:
                if place not in ("tab", "settings"):
                    raise ValueError(f"unknown place {place!r}")
                validate_panel(panel, place)
            except ValueError as e:
                with self._lock:
                    self._errors.append(f"bad panel: {e}")
                raise NilError(INVALID_PARAMS, str(e)) from e
            (self.settings_panels if place == "settings" else self.panels).append(panel)
            return {}
        if method == "player.state":
            result: dict = {"state": self.state, "queue": list(self.queue)}
            if self.state != "stopped" and self.index is not None:
                result["index"] = self.index
                result["path"] = self.queue[self.index]
                result["position_s"] = 0.0
            return result
        if method == "player.play":
            paths = self._paths_param(params)
            start = params.get("start", 0)
            if not paths or not isinstance(start, int) or not 0 <= start < len(paths):
                raise NilError(INVALID_PARAMS, "paths must be non-empty and start in range")
            self._check_paths(paths)
            self.queue, self.index, self.state = list(paths), start, "playing"
            self._frames.put(self._nil_writer.flushes + 1)
            return {}
        if method == "player.append":
            paths = self._paths_param(params)
            self._check_paths(paths)
            if paths:
                first_new = len(self.queue)
                self.queue.extend(paths)
                if self.state == "stopped":
                    self.index, self.state = first_new, "playing"
                self._frames.put(self._nil_writer.flushes + 1)
            return {}
        if method == "player.remove":
            indices = self._queue_edit(params)
            self.queue = [p for i, p in enumerate(self.queue) if i not in indices]
            self._frames.put(self._nil_writer.flushes + 1)
            return {}
        if method == "player.move":
            indices = self._queue_edit(params, with_to=True)
            to = params["to"]
            moved = [self.queue[t["index"]] for t in params["tracks"]]
            rest = [(i, p) for i, p in enumerate(self.queue) if i not in indices]
            before = [p for i, p in rest if i < to]
            self.queue = before + moved + [p for i, p in rest if i >= to]
            self._frames.put(self._nil_writer.flushes + 1)
            return {}
        if method == "player.clear_upcoming":
            if self.state == "stopped" or self.index is None:
                raise NilError(STALE_QUEUE, "queue changed", {"reason": "stopped"})
            keep = self.index + (2 if self.next_starting else 1)
            removed = max(0, len(self.queue) - keep)
            del self.queue[keep:]
            self._frames.put(self._nil_writer.flushes + 1)
            return {"removed": removed}
        if method == "library.folders":
            return list(self.folders)
        if method == "library.tracks":
            return [dict(t) for t in self.tracks]
        if method == "playlists.list":
            return sorted(self.playlists)
        if method == "playlists.read":
            name = params.get("name")
            if not isinstance(name, str):
                raise NilError(INVALID_PARAMS, "name must be a string")
            if name not in self.playlists:
                raise NilError(NOT_FOUND, f"no playlist {name!r}")
            by_key = {_key(t["path"]): t for t in self.tracks}
            rows = []
            for path in self.playlists[name]:
                t = by_key.get(_key(path))
                rows.append(
                    {
                        "path": path,
                        "title": t.get("title") if t else None,
                        "artist": t.get("artist") if t else None,
                        "duration_s": t.get("duration_s") if t else None,
                        "missing": t is None,
                    }
                )
            return rows
        if method == "playlists.save":
            name = params.get("name")
            if not isinstance(name, str):
                raise NilError(INVALID_PARAMS, "name must be a string")
            paths = self._paths_param(params)
            self._check_paths(paths)
            stored = re.sub(r'[<>:"/\\|?*]', "_", name)
            self.playlists[stored] = list(paths)
            return {"name": stored}
        raise NilError(METHOD_NOT_FOUND, f"unknown method {method}")
