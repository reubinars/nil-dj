"""The MCP server: the DJ's tools for an AI client (spec §8).

Inside nil it serves streamable HTTP on 127.0.0.1 from a daemon thread (`McpHttp`). `nil-dj serve
--stdio` bridges a stdio client to that server when it is up, else serves a standalone DJ.
"""

from __future__ import annotations

import functools
import socket
import sys
import threading
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server import create_proxy

from nil_dj.config import Config
from nil_dj.dj import DJ, NotInNil, default_embedder
from nil_dj.host import NilError
from nil_dj.whisper import default_detector

HOST = "127.0.0.1"
INSTRUCTIONS = (
    "An AI DJ over the user's own music library in the nil player. Tracks are identified by "
    "their file path: pass the `path` values the search tools return to play, append and "
    "save_playlist. Describe sound qualities (instrumental, vocals, acoustic) in the query "
    "text; filters only take min_minutes, max_minutes, year_from, year_to, artists, "
    "exclude_artists, formats, hires_only and languages (codes such as ja, en, id, "
    "instrumental). remove_from_queue, move_in_queue and clear_upcoming edit nil's queue "
    "(they need nil 0.3.0); take positions from now_playing's `upcoming` and pass its `path` "
    "values as `paths`, so an edit fails rather than hitting a queue that has moved on."
)


def _port_message(port: int) -> str:
    return f"MCP unavailable: port {port} in use"


def _tool_errors(fn: Callable[..., Any]) -> Callable[..., Any]:
    """The DJ's errors as `ToolError`s, so the client sees their text."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except NotInNil as e:
            raise ToolError(str(e)) from e
        except NilError as e:
            raise ToolError(f"nil: {e.message}") from e
        except ValueError as e:  # the caller's mistake, IndexProblem among them
            raise ToolError(str(e)) from e

    return wrapper


def build_server(dj: DJ) -> FastMCP:
    mcp = FastMCP("nil-dj", instructions=INSTRUCTIONS)

    def tool(fn: Callable[..., Any]) -> Callable[..., Any]:
        mcp.tool(_tool_errors(fn))
        return fn

    @tool
    def search_tracks(query: str, limit: int = 20, filters: dict | None = None) -> list[dict]:
        """Find tracks by how they sound, from a description ("rainy-day piano"). Returns path,
        artist, title, album, duration_s, format, language and score, best match first. Language
        words in the description ("japanese city pop") filter by language; so does
        filters.languages (codes such as "ja", "en", "id", "instrumental")."""
        return [t.as_dict() for t in dj.search_tracks(query, limit, filters)]

    @tool
    def find_tracks(
        artist: str | None = None, album: str | None = None, title: str | None = None,
        limit: int = 50,
    ) -> list[dict]:  # fmt: skip
        """Find tracks by their tags: case-insensitive substring match on artist, album and
        title."""
        return [t.as_dict() for t in dj.find_tracks(artist, album, title, limit)]

    @tool
    def similar_tracks(
        seed: str | dict, limit: int = 20, filters: dict | None = None
    ) -> list[dict]:
        """Tracks that sound like a seed: a track path, {"album": ..., "artist": ...} (artist
        optional), or "now_playing" for the track nil is playing. filters.languages limits the
        result to languages (codes such as "ja", "en", "id", "instrumental")."""
        return [t.as_dict() for t in dj.similar_tracks(seed, limit, filters)]

    @tool
    def build_playlist(
        prompt: str,
        minutes: float = 45,
        filters: dict | None = None,
        seeds: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> dict:
        """An ordered playlist for the prompt, about `minutes` long. `seeds` (paths) steer the
        sound; `exclude` (paths) are left out. Language words in the prompt, or filters.languages
        (codes such as "ja", "en", "id", "instrumental"), limit the languages.
        Does not play: pass the paths to play."""
        pl = dj.build_playlist(prompt, minutes, filters, seeds, exclude)
        return {
            "tracks": [t.as_dict() for t in pl.tracks],
            "total_minutes": round(pl.total_s / 60, 1),
        }

    @tool
    def play(paths: list[str], radio: bool = False) -> dict:
        """Play these tracks in nil, replacing the queue. With radio=true, radio keeps the queue
        topped up with similar tracks."""
        dj.play(paths, radio)
        return {"playing": len(paths), "radio": dj.radio_on}

    @tool
    def append(paths: list[str]) -> dict:
        """Add these tracks to the end of nil's queue; they play gaplessly."""
        dj.append(paths)
        return {"appended": len(paths)}

    @tool
    def now_playing() -> dict:
        """nil's player: state, path, index, position_s, queue_len and the playing track's
        tags (`track`), plus whether radio is on. `upcoming`: the next 20 tracks with `position`
        (1 = next), path, title and artist."""
        return {**dj.now_playing(), "radio": dj.radio_on}

    @tool
    def remove_from_queue(positions: list[int], paths: list[str] | None = None) -> dict:
        """Remove upcoming tracks from nil's queue. Positions are 1-based within `upcoming` from
        now_playing (1 = the next track). Pass `paths`: the `path` values from now_playing's
        `upcoming` for those positions, in the same order; the edit fails if the queue has
        moved on since."""
        dj.remove_from_queue(positions, paths)
        return {}

    @tool
    def move_in_queue(positions: list[int], to: int, paths: list[str] | None = None) -> dict:
        """Move upcoming tracks so they start at position `to`, in the order given. to=1 plays
        them next; a position past the last puts them at the end. Pass `paths`: the `path`
        values from now_playing's `upcoming` for `positions`, in the same order; the edit fails
        if the queue has moved on since."""
        dj.move_in_queue(positions, to, paths)
        return {}

    @tool
    def clear_upcoming() -> dict:
        """Remove every upcoming track; the playing one keeps playing."""
        return {"removed": dj.clear_upcoming()}

    @tool
    def set_radio(on: bool) -> dict:
        """Turn radio on or off. Radio tops up the queue from what was played."""
        dj.set_radio(on)
        return {"radio": dj.radio_on}

    @tool
    def save_playlist(name: str, paths: list[str]) -> dict:
        """Save these tracks as a playlist in nil. Returns the name nil stored it under."""
        return {"name": dj.save_playlist(name, paths)}

    @tool
    def list_playlists() -> list[str]:
        """The names of nil's playlists."""
        return dj.list_playlists()

    @tool
    def play_playlist(name: str) -> dict:
        """Play one of nil's playlists (the tracks still on disk)."""
        return {"playing": dj.play_playlist(name)}

    @tool
    def index_status() -> dict:
        """The sound index: indexed, pending (in nil's library, not indexed yet), failed and
        their errors, whether indexing is running and its progress (done/total), and any
        problem."""
        s = dj.index_status()
        d = asdict(s)
        d["errors"] = [{"path": p, "error": e} for p, e in s.errors]
        return d

    @tool
    def reindex() -> dict:
        """Bring the index up to date with nil's library, in the background. `started` is
        false when indexing is already running; follow it with index_status."""
        return {"started": dj.reindex()}

    return mcp


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((HOST, port))
        except OSError:
            return False
    return True


class McpHttp:
    """The server over HTTP on a daemon thread; `status` is the panel's MCP line."""

    def __init__(self, server: FastMCP, port: int, on_change: Callable[[], None] = lambda: None):
        self.server = server
        self.port = port
        self.on_change = on_change
        self.status = f"MCP: http://{HOST}:{port}/mcp"

    def start(self) -> None:
        if not port_free(self.port):
            self.status = _port_message(self.port)
            return
        threading.Thread(target=self._run, daemon=True, name="nil-dj-mcp").start()

    def _run(self) -> None:
        try:
            self.server.run(
                transport="http", host=HOST, port=self.port, show_banner=False,
                uvicorn_config={"access_log": False},
            )  # fmt: skip
        except SystemExit:  # uvicorn couldn't bind: taken since port_free looked
            self.status = _port_message(self.port)
        except Exception as e:  # noqa: BLE001 - the panel says why; the plugin goes on
            self.status = f"MCP unavailable: {e}"
        else:
            return
        self.on_change()


def _listening(port: int) -> bool:
    try:
        with socket.create_connection((HOST, port), timeout=0.5):
            return True
    except OSError:
        return False


def serve_stdio(cfg: Config) -> int:
    """MCP over stdio: a proxy to the plugin's server when nil runs it, else a standalone DJ."""
    if _listening(cfg.mcp_port):
        create_proxy(f"http://{HOST}:{cfg.mcp_port}/mcp").run(transport="stdio", show_banner=False)
    else:
        dj = DJ(cfg, None, lambda: default_embedder(cfg), detector=default_detector(cfg))
        build_server(dj).run(transport="stdio", show_banner=False)
    return 0


def serve_http(cfg: Config) -> int:
    """`nil-dj serve`: a standalone DJ over HTTP on `cfg.mcp_port`, until interrupted."""
    if not port_free(cfg.mcp_port):
        print(_port_message(cfg.mcp_port), file=sys.stderr)
        return 1
    dj = DJ(cfg, None, lambda: default_embedder(cfg), detector=default_detector(cfg))
    build_server(dj).run(
        transport="http", host=HOST, port=cfg.mcp_port, show_banner=False,
        uvicorn_config={"access_log": False},
    )  # fmt: skip
    return 0
