"""The DJ: search, playlists, playback through nil, and indexing. The nil panel and the MCP
tools both call it, from different threads.

Threads: `_lock` guards the shared state (model, library paths, radio flag; the index run has
its own in `IndexRunner`) and is never held across a nil call, a model load or a listener.
sqlite3 connections stay on the thread that made them, so every method opens its own
short-lived `Index`; the index thread has its own too.
"""

from __future__ import annotations

import math
import sqlite3
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterable
from contextlib import closing
from dataclasses import asdict, dataclass, field
from pathlib import PureWindowsPath

import numpy as np

from nil_dj.config import Config
from nil_dj.embedder import Embedder, models_present
from nil_dj.host import Connection
from nil_dj.index import Index, IndexFormatError, Row
from nil_dj.indexer import walk
from nil_dj.order import Candidate, candidate, order
from nil_dj.reindex import IndexRunner
from nil_dj.search import (
    Hit,
    candidate_pool_size,
    describe_format,
    filters_from,
    find_by_tags,
    rank,
    search_text,
    similar_to_album,
    similar_to_paths,
)

NOT_IN_NIL = "nil-dj isn't running inside nil — enable it in nil's Settings → Plugins."
NO_INDEX = "No index yet — press Index library."
NOTHING_PLAYING = "Nothing is playing."
PLAYING_NOT_INDEXED = "The playing track isn't in the index yet — press Update index."
TYPICAL_TRACK_S = 180  # sizes the candidate pool for a playlist length


class NotInNil(Exception):
    def __init__(self) -> None:
        super().__init__(NOT_IN_NIL)


class IndexProblem(ValueError):
    """The index can't be used; `rebuild_needed` when it comes from an `IndexFormatError`."""

    def __init__(self, message: str, rebuild_needed: bool = False):
        super().__init__(message)
        self.rebuild_needed = rebuild_needed


def default_embedder(cfg: Config, use_gpu: bool = True) -> Embedder:
    from nil_dj.embedder import OnnxClapEmbedder

    return OnnxClapEmbedder(
        cfg.precision, cfg.gpu_device_id, use_gpu=use_gpu, models_dir=cfg.models_dir
    )


@dataclass
class Track:
    path: str
    title: str
    artist: str | None
    album: str | None
    duration_s: float
    format: str
    score: float | None = None

    @classmethod
    def from_row(cls, row: Row, score: float | None = None) -> Track:
        i = row.info
        return cls(
            path=row.path,
            # PureWindowsPath splits on both separators: nil's paths are Windows paths.
            title=i.title or PureWindowsPath(row.path).stem,
            artist=i.album_artist or i.artist,
            album=i.album,
            duration_s=i.duration_s,
            format=describe_format(i),
            score=score,
        )

    @classmethod
    def from_hit(cls, hit: Hit) -> Track:
        return cls.from_row(hit.row, hit.score)

    def item(self) -> dict:
        """A row for the panel's `tracks` element (fields that are unknown are left out)."""
        d = {"path": self.path, "title": self.title, "artist": self.artist,
             "duration_s": self.duration_s}  # fmt: skip
        return {k: v for k, v in d.items() if v is not None}

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Playlist:
    tracks: list[Track]
    total_s: float
    title: str


@dataclass
class IndexStatus:
    indexed: int = 0
    pending: int | None = None
    failed: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)
    running: bool = False
    done: int = 0
    total: int = 0
    problem: str | None = None
    rebuild_needed: bool = False  # the problem comes from an IndexFormatError


def _normalise(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def _clock(seconds: float) -> str:
    s = round(seconds)
    return f"{s // 60:d}:{s % 60:02d}"


class DJ:
    def __init__(
        self,
        cfg: Config,
        nil: Connection | None,
        make_embedder: Callable[[], Embedder],
        clock: Callable[[], float] = time.monotonic,
    ):
        self.cfg = cfg
        self.nil = nil
        self.clock = clock
        # Called after every state change, on the thread that made it, with no lock held. On
        # the loading thread, `embedder()` raises rather than wait for its own load.
        self.listeners: list[Callable[[], None]] = []
        self._make_embedder = make_embedder
        self._lock = threading.Lock()
        self._loaded = threading.Condition(self._lock)  # signalled when a model load ends
        self._loader: int | None = None  # the thread loading the model, one at a time
        self._embedder: Embedder | None = None
        self._model_state = "not loaded"
        self._library: set[str] | None = None  # paths from the last library.tracks
        self._radio = False
        self._runner = IndexRunner(cfg, self._paths_to_index, self.embedder, self.changed)

    # -- plumbing --------------------------------------------------------------------------

    def changed(self) -> None:
        for fn in list(self.listeners):
            try:
                fn()
            except Exception:  # noqa: BLE001 - a broken listener must not break the DJ
                traceback.print_exc(file=sys.stderr)

    def _nil(self) -> Connection:
        if self.nil is None:
            raise NotInNil()
        return self.nil

    def _open(self) -> Index:
        """A new connection for the calling thread. Raises IndexProblem."""
        path = self.cfg.index_path
        if not path.exists():
            raise IndexProblem(NO_INDEX)
        try:
            return Index.open(path, None)
        except IndexFormatError as e:
            raise IndexProblem(str(e), rebuild_needed=True) from e
        except sqlite3.DatabaseError as e:  # not an index at all
            raise IndexProblem(f"the index file can't be read ({e})", rebuild_needed=True) from e

    @staticmethod
    def _check_model(index: Index, model_id: str) -> None:
        built = index.get_meta("model_id")
        if built != model_id:
            raise IndexProblem(f"index built with {built}; rebuild it for {model_id}", True)

    def _open_for_text(self) -> tuple[Index, Embedder]:
        self._open().close()  # say there's no index before loading the model for nothing
        emb = self.embedder()
        index = self._open()
        try:
            self._check_model(index, emb.model_id)
        except IndexProblem:
            index.close()
            raise
        return index, emb

    # -- the sound model -------------------------------------------------------------------

    @property
    def model_state(self) -> str:
        with self._lock:
            return self._model_state

    def embedder(self) -> Embedder:
        """The model, loaded on first use. One thread loads; the others wait for it. No lock is
        held while it loads or while listeners hear about it."""
        present = models_present(self.cfg.models_dir, self.cfg.precision)
        with self._loaded:
            while self._loader is not None:
                if self._loader == threading.get_ident():
                    raise RuntimeError(
                        "embedder() was called from a listener while the model loads"
                    )
                self._loaded.wait()
            if self._embedder is not None:
                return self._embedder
            self._loader = threading.get_ident()
            self._model_state = "loading" if present else "downloading"
        self.changed()
        try:
            emb = self._make_embedder()
        except BaseException as e:  # even an interrupt must free the waiters
            self._finish_load(None, f"failed: {str(e) or type(e).__name__}")
            raise
        self._finish_load(emb, "ready")
        return emb

    def _finish_load(self, emb: Embedder | None, state: str) -> None:
        with self._loaded:
            self._embedder = emb
            self._model_state = state
            self._loader = None
            self._loaded.notify_all()
        self.changed()

    def preload(self) -> None:
        """Load the model in the background, if there is an index to search."""
        if self.cfg.index_path.exists():
            threading.Thread(target=self._preload, daemon=True, name="nil-dj-model").start()

    def _preload(self) -> None:
        try:
            self.embedder()
        except Exception:  # noqa: BLE001 - model_state says it failed
            traceback.print_exc(file=sys.stderr)

    # -- search ----------------------------------------------------------------------------

    def search_tracks(
        self, query: str, limit: int = 20, filters: dict | None = None
    ) -> list[Track]:
        f = filters_from(filters)
        index, emb = self._open_for_text()
        with closing(index):
            hits = search_text(index, emb, query, limit, f, self.cfg.prompt_template)
        return [Track.from_hit(h) for h in hits]

    def find_tracks(
        self, artist: str | None = None, album: str | None = None, title: str | None = None,
        limit: int = 50,
    ) -> list[Track]:  # fmt: skip
        with closing(self._open()) as index:
            rows = find_by_tags(index, artist, album, title, limit)
        return [Track.from_row(r) for r in rows]

    def similar_tracks(
        self, seed: str | dict, limit: int = 20, filters: dict | None = None
    ) -> list[Track]:
        """Tracks like a path, an album (`{"album", "artist"?}`) or `"now_playing"`."""
        f = filters_from(filters)
        playing = seed == "now_playing"
        if playing:
            seed = self._playing_path()
        with closing(self._open()) as index:
            if isinstance(seed, dict):
                album, artist = seed.get("album"), seed.get("artist")
                if not isinstance(album, str):
                    # ValueError: the panel and MCP report those as the user's mistake.
                    raise ValueError("an album seed needs an 'album' name")  # noqa: TRY004
                try:
                    hits = similar_to_album(index, album, artist, limit, f)
                except KeyError as e:
                    raise ValueError(f"No album {e.args[0]!r} in the index.") from e
            else:
                try:
                    hits = similar_to_paths(index, [seed], limit, f)
                except KeyError as e:
                    raise ValueError(
                        PLAYING_NOT_INDEXED if playing else f"{seed} isn't in the index."
                    ) from e
        return [Track.from_hit(h) for h in hits]

    def _playlist(
        self, name: str, pool: list[Hit], target_s: float, history: Iterable[Candidate] = ()
    ) -> Playlist:
        cands = [candidate(h.row, h.vector) for h in pool]
        match = np.array([h.score for h in pool], np.float32)
        chosen = order(
            cands, match, weights=self.cfg.weights, target_s=target_s, history=list(history)
        )
        by_path = {h.row.path: h for h in pool}
        tracks = [Track.from_hit(by_path[c.path]) for c in chosen]
        total = sum(t.duration_s for t in tracks)
        return Playlist(tracks, total, f"{name} — {len(tracks)} tracks, {_clock(total)}")

    @staticmethod
    def _pool_size(minutes: float) -> int:
        return candidate_pool_size(math.ceil(minutes * 60 / TYPICAL_TRACK_S))

    def build_playlist(
        self,
        prompt: str,
        minutes: float = 45,
        filters: dict | None = None,
        seeds: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> Playlist:
        """A playlist for the prompt, about `minutes` long. Seeds steer the sound; they are not
        forced in."""
        f = filters_from(filters)
        index, emb = self._open_for_text()
        with closing(index):
            q = emb.embed_text(self.cfg.prompt_template.format(prompt))
            if seeds:
                wanted = list(dict.fromkeys(seeds))
                rows, matrix = index.load(f"path IN ({', '.join('?' * len(wanted))})", wanted)
                found = {r.path for r in rows}
                if missing := [p for p in wanted if p not in found]:
                    raise ValueError(f"{missing[0]} isn't in the index.")
                q = _normalise(q + _normalise(matrix.mean(axis=0)))
            pool = rank(index, q, self._pool_size(minutes), f, frozenset(exclude or ()))
        return self._playlist(prompt, pool, minutes * 60)

    def more_like_playing(self, minutes: float = 45) -> Playlist:
        path = self._playing_path()
        with closing(self._open()) as index:
            rows, matrix = index.load("path = ?", [path])
            if not rows:
                raise ValueError(PLAYING_NOT_INDEXED)
            pool = rank(index, matrix[0], self._pool_size(minutes), exclude=frozenset({path}))
        seed = candidate(rows[0], matrix[0])
        name = f"More like {Track.from_row(rows[0]).title}"
        return self._playlist(name, pool, minutes * 60, history=[seed])

    # -- playback --------------------------------------------------------------------------

    def _playing_path(self) -> str:
        state = self._nil().call("player.state") or {}
        path = state.get("path")
        if state.get("state") == "stopped" or not isinstance(path, str):
            raise ValueError(NOTHING_PLAYING)
        return path

    def play(self, paths: list[str], radio: bool = False) -> None:
        self._nil().call("player.play", {"paths": list(paths)})
        if radio:
            self.set_radio(True)

    def append(self, paths: list[str]) -> None:
        self._nil().call("player.append", {"paths": list(paths)})

    def now_playing(self) -> dict:
        """nil's player state with `queue_len` instead of the queue, and `track` (the index's
        tags, or None) when something is playing or paused."""
        state = dict(self._nil().call("player.state") or {})
        state["queue_len"] = len(state.pop("queue", None) or [])
        if isinstance(path := state.get("path"), str):
            try:
                with closing(self._open()) as index:
                    rows, _ = index.load("path = ?", [path])
            except IndexProblem:
                rows = []
            state["track"] = Track.from_row(rows[0]).as_dict() if rows else None
        return state

    @property
    def radio_on(self) -> bool:
        with self._lock:
            return self._radio

    def set_radio(self, on: bool) -> None:
        self._nil()
        with self._lock:
            self._radio = bool(on)
        self.changed()

    # -- nil's playlists -------------------------------------------------------------------

    def save_playlist(self, name: str, paths: list[str]) -> str:
        """Returns the name nil stored it under."""
        return self._nil().call("playlists.save", {"name": name, "paths": list(paths)})["name"]

    def list_playlists(self) -> list[str]:
        return list(self._nil().call("playlists.list"))

    def play_playlist(self, name: str) -> int:
        """Play the tracks that are still on disk; returns how many."""
        rows = self._nil().call("playlists.read", {"name": name})
        paths = [r["path"] for r in rows if not r.get("missing")]
        if not paths:
            raise ValueError(f"Playlist {name!r} has no playable tracks.")
        self.play(paths)
        return len(paths)

    # -- indexing --------------------------------------------------------------------------

    def refresh_library(self) -> list[str]:
        """nil's library paths (`library.tracks`), cached for `index_status().pending`."""
        tracks = self._nil().call("library.tracks") or []
        paths = [
            t["path"] for t in tracks if isinstance(t, dict) and isinstance(t.get("path"), str)
        ]
        with self._lock:
            self._library = set(paths)
        self.changed()
        return paths

    def _paths_to_index(self) -> list[str]:
        # Standalone walks nil's folders here rather than in run_index, so an unplugged drive
        # is refused as an empty library instead of deleting every row.
        return self.refresh_library() if self.nil is not None else walk(self.cfg.folders)

    def index_status(self) -> IndexStatus:
        run = self._runner.state()
        status = IndexStatus(running=run.running, done=run.done, total=run.total)
        with self._lock:
            library = self._library
            model_id = self._embedder.model_id if self._embedder is not None else None
        indexed: set[str] = set()
        try:
            with closing(self._open()) as index:
                indexed = set(index.stamps())
                status.errors = index.errors()
                if model_id is not None:
                    self._check_model(index, model_id)
        except IndexProblem as e:
            status.problem, status.rebuild_needed = str(e), e.rebuild_needed
        status.indexed = len(indexed)
        status.failed = len(status.errors)
        if library is not None:
            status.pending = len(library - indexed - {p for p, _ in status.errors})
        if run.problem is not None:  # the last run's problem is the news
            status.problem, status.rebuild_needed = run.problem, run.rebuild_needed
        return status

    def reindex(self, rebuild: bool = False) -> bool:
        """Start indexing in the background; False if a run is already going."""
        return self._runner.start(rebuild)

    def stop(self) -> None:
        self._runner.stop()
