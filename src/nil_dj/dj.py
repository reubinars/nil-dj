"""The DJ: search, playlists, playback through nil, and indexing. The nil panel and the MCP
tools both call it, from different threads.

Threads: `_lock` guards the shared state (model, library paths; the index run has
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
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path, PureWindowsPath

import numpy as np
from filelock import FileLock, Timeout

from nil_dj.config import Config
from nil_dj.embedder import Embedder, ModelDownloadFailed, models_present
from nil_dj.host import INVALID_PARAMS, METHOD_NOT_FOUND, STALE_QUEUE, Connection, NilError
from nil_dj.index import INDEX_BUSY, Index, IndexFormatError, IndexProblem, Row, path_key
from nil_dj.indexer import walk
from nil_dj.languages import load_overrides, overrides_changed, parse_prompt, resolve_index
from nil_dj.order import Candidate, candidate, order
from nil_dj.radio import Radio
from nil_dj.reindex import IndexRunner, overrides_path
from nil_dj.search import (
    Allowed,
    Filters,
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
from nil_dj.whisper import DetectorFactory

NOT_IN_NIL = "nil-dj isn't running inside nil — enable it in nil's Settings → Plugins."
NO_INDEX = "No index yet — press Index library."
NOTHING_PLAYING = "Nothing is playing."
QUEUE_CHANGED = (
    "The queue changed (or the next track is already starting). Call now_playing and try again."
)
NEEDS_NIL_030 = "Queue editing needs nil 0.3.0 or newer."
UPCOMING_SHOWN = 20  # how many upcoming tracks now_playing lists
PLAYING_NOT_INDEXED = "The playing track isn't in the index yet — press Update index."
LANGUAGES_NOT_READY = "Language data isn't ready yet — run Update index in the DJ tab."
TYPICAL_TRACK_S = 180  # sizes the candidate pool for a playlist length
LIBRARY_MAX_AGE_S = 30.0  # how often a panel event may re-read nil's library


class NotInNil(Exception):
    def __init__(self) -> None:
        super().__init__(NOT_IN_NIL)


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
    language: str | None = None
    language_alt: str | None = None

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
            language=row.language,
            language_alt=row.language_alt,
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
        d = asdict(self)
        if d["language_alt"] is None:
            del d["language_alt"]
        return d


@dataclass
class Playlist:
    tracks: list[Track]
    total_s: float
    title: str
    name: str = ""  # what was asked for: the prompt, or "More like …"


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
    languages_pending: int = 0
    language_problems: list[str] = field(default_factory=list)
    phase: str = "index"


def _normalise(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def _clock(seconds: float) -> str:
    s = round(seconds)
    return f"{s // 60:d}:{s % 60:02d}"


def minutes_text(seconds: float) -> str:
    """A playlist's length for people: "38 min", "1 h 52 min" (to the nearest minute)."""
    m = int(seconds / 60 + 0.5)
    return f"{m // 60} h {m % 60} min" if m >= 60 else f"{m} min"


class DJ:
    def __init__(
        self,
        cfg: Config,
        nil: Connection | None,
        make_embedder: Callable[[], Embedder],
        clock: Callable[[], float] = time.monotonic,
        detector: DetectorFactory | None = None,
    ):
        self.cfg = cfg
        self._detector = detector
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
        self._library: set[str] | None = None  # path_keys from the last library.tracks
        self._library_at: float | None = None  # when it was last asked for, by `clock`
        self._runner = IndexRunner(cfg, self._paths_to_index, self.embedder, self.changed, detector)
        self.radio = Radio(
            self._open, self._call_nil, cfg.weights, on_change=self.changed, closed=self._closed,
            allowed=self._allowed,
        )  # fmt: skip

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

    def _closed(self) -> bool:
        return self.nil is not None and self.nil.closed.is_set()

    def _call_nil(self, method: str, params: dict | None = None):
        return self._nil().call(method, params)

    def _allowed(self) -> Allowed:
        """In nil, only what nil's library has may be chosen: the index can hold tracks nil no
        longer has. Before the library is read, and standalone, anything may."""
        with self._lock:
            library = self._library
        if self.nil is None or library is None:
            return None
        return lambda path: path_key(path) in library

    def _open(self) -> Index:
        """A new connection for the calling thread. Raises IndexProblem."""
        path = self.cfg.index_path
        if not path.exists():
            raise IndexProblem(NO_INDEX)
        try:
            return Index.open(path, None)
        except IndexFormatError as e:
            raise IndexProblem(str(e), rebuild_needed=True) from e
        except sqlite3.OperationalError as e:  # locked or can't be opened: a rebuild won't help
            raise IndexProblem(INDEX_BUSY) from e
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
            # A failed download is already a sentence for the panel; anything else isn't.
            failed = isinstance(e, ModelDownloadFailed)
            self._finish_load(None, str(e) if failed else f"failed: {str(e) or type(e).__name__}")
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

    def _resolve_languages(self) -> None:
        """Apply a changed languages.toml; a running index run resolves by itself."""
        path = overrides_path(self.cfg)
        lock_path = Path(f"{self.cfg.index_path}.lock")
        try:
            with FileLock(lock_path, timeout=0), closing(self._open()) as index:
                if overrides_changed(index, path) or index.unresolved_languages():
                    resolve_index(index, path)
        except Timeout:
            pass
        except Exception:  # noqa: BLE001 - the model preload goes on
            traceback.print_exc(file=sys.stderr)

    def _preload(self) -> None:
        self._resolve_languages()
        try:
            self.embedder()
        except Exception:  # noqa: BLE001 - model_state says it failed
            traceback.print_exc(file=sys.stderr)

    # -- search ----------------------------------------------------------------------------

    @staticmethod
    def _with_prompt_languages(
        query: str, filters: dict | None
    ) -> tuple[Filters | None, str]:
        """The filters, with the languages asked for in the prompt added unless the filters name
        some, and the prompt without those words."""
        f = filters_from(filters)
        codes, text = parse_prompt(query)
        if codes and not (f and f.languages):
            f = f or Filters()
            f.languages = codes
        return f, text

    @staticmethod
    def _check_languages(index: Index, f: Filters | None) -> None:
        if f and f.languages and not index.has_languages():
            raise ValueError(LANGUAGES_NOT_READY)

    def search_tracks(
        self, query: str, limit: int = 20, filters: dict | None = None
    ) -> list[Track]:
        f, text = self._with_prompt_languages(query, filters)
        index, emb = self._open_for_text()
        with closing(index):
            self._check_languages(index, f)
            hits = search_text(
                index, emb, text, limit, f, self.cfg.prompt_template, self._allowed()
            )
        return [Track.from_hit(h) for h in hits]

    def find_tracks(
        self, artist: str | None = None, album: str | None = None, title: str | None = None,
        limit: int = 50,
    ) -> list[Track]:  # fmt: skip
        with closing(self._open()) as index:
            rows = find_by_tags(index, artist, album, title, limit, self._allowed())
        return [Track.from_row(r) for r in rows]

    def similar_tracks(
        self, seed: str | dict, limit: int = 20, filters: dict | None = None
    ) -> list[Track]:
        """Tracks like a path, an album (`{"album", "artist"?}`) or `"now_playing"`."""
        f = filters_from(filters)
        playing = seed == "now_playing"
        if playing:
            seed = self._playing_path()
        allowed = self._allowed()
        with closing(self._open()) as index:
            self._check_languages(index, f)
            if isinstance(seed, dict):
                album, artist = seed.get("album"), seed.get("artist")
                if not isinstance(album, str):
                    # ValueError: the panel and MCP report those as the user's mistake.
                    raise ValueError("an album seed needs an 'album' name")  # noqa: TRY004
                try:
                    hits = similar_to_album(index, album, artist, limit, f, allowed)
                except KeyError as e:
                    raise ValueError(f"No album {e.args[0]!r} in the index.") from e
            else:
                try:
                    hits = similar_to_paths(index, [seed], limit, f, allowed)
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
            cands, match, weights=replace(self.cfg.weights, language=0.0), target_s=target_s,
            history=list(history),
        )  # fmt: skip
        by_path = {h.row.path: h for h in pool}
        tracks = [Track.from_hit(by_path[c.path]) for c in chosen]
        total = sum(t.duration_s for t in tracks)
        title = f"{name} — {len(tracks)} tracks, {_clock(total)}"
        return Playlist(tracks, total, title, name)

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
        f, text = self._with_prompt_languages(prompt, filters)
        allowed = self._allowed()
        index, emb = self._open_for_text()
        with closing(index):
            self._check_languages(index, f)
            q = emb.embed_text(self.cfg.prompt_template.format(text))
            if seeds:
                rows, matrix = index.load_paths(seeds)
                found = {path_key(r.path) for r in rows}
                if missing := [p for p in seeds if path_key(p) not in found]:
                    raise ValueError(f"{missing[0]} isn't in the index.")
                q = _normalise(q + _normalise(matrix.mean(axis=0)))
            pool = rank(index, q, self._pool_size(minutes), f, frozenset(exclude or ()), allowed)
        return self._playlist(prompt, pool, minutes * 60)

    def more_like_playing(self, minutes: float = 45) -> Playlist:
        path = self._playing_path()
        allowed = self._allowed()
        with closing(self._open()) as index:
            rows, matrix = index.load_paths([path])
            if not rows:
                raise ValueError(PLAYING_NOT_INDEXED)
            pool = rank(
                index, matrix[0], self._pool_size(minutes), exclude=frozenset({path}),
                allowed=allowed,
            )  # fmt: skip
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
        """nil's player state with `queue_len` instead of the queue, `track` (the index's tags,
        or None) when something is playing or paused, and `upcoming`: the next
        UPCOMING_SHOWN queue entries (1-based `position`, path, title, artist; None for a path
        the index lacks)."""
        state = dict(self._nil().call("player.state") or {})
        queue = state.pop("queue", None) or []
        state["queue_len"] = len(queue)
        index = state.get("index")
        upcoming: list[str] = []
        if isinstance(index, int) and state.get("state") != "stopped":
            upcoming = queue[index + 1 : index + 1 + UPCOMING_SHOWN]
        path = state.get("path")
        wanted = ([path] if isinstance(path, str) else []) + upcoming
        by_key: dict[str, Track] = {}
        if wanted:
            try:
                with closing(self._open()) as idx:
                    rows, _ = idx.load_paths(wanted)
            except IndexProblem:
                rows = []
            by_key = {path_key(r.path): Track.from_row(r) for r in rows}
        if isinstance(path, str):
            hit = by_key.get(path_key(path))
            state["track"] = hit.as_dict() if hit else None
        state["upcoming"] = [
            {
                "position": n,
                "path": p,
                "title": hit.title if (hit := by_key.get(path_key(p))) else None,
                "artist": hit.artist if hit else None,
            }
            for n, p in enumerate(upcoming, 1)
        ]
        return state

    # -- editing nil's queue ---------------------------------------------------------------

    def _upcoming_state(self) -> tuple[int, list[str]]:
        """(index of the playing track, the queue)."""
        state = self._nil().call("player.state") or {}
        index = state.get("index")
        if state.get("state") == "stopped" or not isinstance(index, int):
            raise ValueError(NOTHING_PLAYING)
        return index, list(state.get("queue") or [])

    def _edit(self, method: str, params: dict):
        try:
            return self._nil().call(method, params)
        except NilError as e:
            if e.code == STALE_QUEUE:
                reason = e.data.get("reason") if isinstance(e.data, dict) else None
                raise ValueError(NOTHING_PLAYING if reason == "stopped" else QUEUE_CHANGED) from e
            if e.code == INVALID_PARAMS and method == "player.move":
                # the queue shrank after the DJ read it, so `to` is past the end
                raise ValueError(QUEUE_CHANGED) from e
            if e.code == METHOD_NOT_FOUND:
                raise ValueError(NEEDS_NIL_030) from e
            raise

    @staticmethod
    def _check_positions(positions: list[int], upcoming: int) -> None:
        if not positions:
            raise ValueError("Name at least one position.")
        for p in positions:
            if not isinstance(p, int) or isinstance(p, bool):
                raise ValueError("Positions are whole numbers starting at 1.")  # noqa: TRY004  (the tool turns ValueErrors into messages)
            if p < 1:
                raise ValueError("Positions start at 1 (the next track).")
            if p > upcoming:
                have = "is 1 upcoming track" if upcoming == 1 else f"are {upcoming} upcoming tracks"
                raise ValueError(f"No upcoming track at position {p}; there {have}.")
        for p in positions:
            if positions.count(p) > 1:
                raise ValueError(f"Position {p} is listed twice.")

    @staticmethod
    def _check_seen(positions: list[int], paths: list[str] | None, current: int, queue: list[str]):
        """The paths the agent saw at `positions` must still be there (same file, to nil)."""
        if paths is None:
            return
        if len(paths) != len(positions):
            raise ValueError("Give one path per position, in the same order.")
        if any(path_key(p) != path_key(queue[current + n]) for n, p in zip(positions, paths)):
            raise ValueError(QUEUE_CHANGED)

    def remove_from_queue(self, positions: list[int], paths: list[str] | None = None) -> None:
        """Remove upcoming tracks by 1-based position (1 = the next track). `paths`, if given,
        are the paths the caller saw at those positions."""
        current, queue = self._upcoming_state()
        self._check_positions(positions, len(queue) - current - 1)
        self._check_seen(positions, paths, current, queue)
        tracks = [{"index": current + p, "path": queue[current + p]} for p in positions]
        self._edit("player.remove", {"tracks": tracks})

    def move_in_queue(self, positions: list[int], to: int, paths: list[str] | None = None) -> None:
        """Move upcoming tracks so they start at position `to` of the upcoming tracks, in the
        order given; a `to` past the last goes to the end. `paths`, if given, are the paths the
        caller saw at `positions`."""
        current, queue = self._upcoming_state()
        self._check_positions(positions, len(queue) - current - 1)
        if not isinstance(to, int) or isinstance(to, bool) or to < 1:
            raise ValueError("to must be 1 or more.")
        self._check_seen(positions, paths, current, queue)
        tracks = [{"index": current + p, "path": queue[current + p]} for p in positions]
        # nil puts the tracks in front of the queue entry at `to`: the `to`-th upcoming track
        # that isn't moving, so they start at position `to` in the result.
        staying = [i for i in range(current + 1, len(queue)) if i - current not in positions]
        self._edit(
            "player.move",
            {"tracks": tracks, "to": staying[to - 1] if to <= len(staying) else len(queue)},
        )

    def clear_upcoming(self) -> int:
        """Remove every upcoming track; returns how many nil removed."""
        self._upcoming_state()
        result = self._edit("player.clear_upcoming", {})
        return int((result or {}).get("removed", 0))

    @property
    def radio_on(self) -> bool:
        return self.radio.on

    def set_radio(self, on: bool) -> None:
        self._nil()
        self.radio.set(on)  # tells the listeners itself

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
        """nil's library paths (`library.tracks`), cached by `path_key` for
        `index_status().pending` and for what may be chosen."""
        tracks = self._nil().call("library.tracks") or []
        paths = [
            t["path"] for t in tracks if isinstance(t, dict) and isinstance(t.get("path"), str)
        ]
        with self._lock:
            self._library = {path_key(p) for p in paths}
            self._library_at = self.clock()
        self.changed()
        return paths

    def refresh_library_if_stale(self) -> None:
        """`refresh_library` when the last read is over 30 s old, since nil rescans its folders
        by itself and never says so. A failed read is logged and the old paths are kept."""
        if self.nil is None:
            return
        now = self.clock()
        with self._lock:
            if self._library_at is not None and now - self._library_at < LIBRARY_MAX_AGE_S:
                return
            self._library_at = now  # a failing read is retried after 30 s too, not every event
        try:
            self.refresh_library()
        except NilError as e:
            print(f"nil-dj: can't read nil's library: {e.message}", file=sys.stderr)

    def _paths_to_index(self) -> list[str]:
        # Standalone walks nil's folders here rather than in run_index, so an unplugged drive
        # is refused as an empty library instead of deleting every row.
        return self.refresh_library() if self.nil is not None else walk(self.cfg.folders)

    def index_status(self) -> IndexStatus:
        run = self._runner.state()
        status = IndexStatus(running=run.running, done=run.done, total=run.total, phase=run.phase)
        status.language_problems = load_overrides(overrides_path(self.cfg))[1]
        with self._lock:
            library = self._library
            model_id = self._embedder.model_id if self._embedder is not None else None
        indexed: set[str] = set()
        try:
            with closing(self._open()) as index:
                indexed = set(index.stamps())
                status.errors = index.errors()
                if self._detector is not None:
                    status.languages_pending = index.languages_pending(self._detector.model_id)
                if model_id is not None:
                    self._check_model(index, model_id)
        except IndexProblem as e:
            status.problem, status.rebuild_needed = str(e), e.rebuild_needed
        status.indexed = len(indexed)
        status.failed = len(status.errors)
        if library is not None:
            known = {path_key(p) for p in indexed} | {path_key(p) for p, _ in status.errors}
            status.pending = len(library - known)
        if run.running:  # a rebuild may be fixing the problem: the run reports its own
            status.problem, status.rebuild_needed = None, False
        elif run.problem is not None:  # the last run's problem is the news
            status.problem, status.rebuild_needed = run.problem, run.rebuild_needed
        return status

    def reindex(self, rebuild: bool = False) -> bool:
        """Start indexing in the background; False if a run is already going."""
        return self._runner.start(rebuild)

    def stop(self) -> None:
        self._runner.stop()
