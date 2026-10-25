"""Background index runs for the DJ: one at a time, stoppable, and never losing an error."""

from __future__ import annotations

import sqlite3
import sys
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock, Timeout

from nil_dj.config import Config
from nil_dj.decode import FfmpegMissing, require_ffmpeg
from nil_dj.embedder import MODEL_DOWNLOAD_FAILED, Embedder, ModelDownloadFailed
from nil_dj.index import INDEX_BUSY, Index, IndexFormatError
from nil_dj.indexer import AlreadyRunning, EmptyLibrary, Progress, run_index_paths

EMPTY_LIBRARY = "nil's library is empty — add a music folder in nil first."
INDEX_NEEDS_MODEL = (
    "Indexing needs the sound model, which couldn't be downloaded. "
    "Check the internet connection, then press {} again."
)
ALREADY_INDEXING = "Another nil-dj index is running."
REBUILD_IN_USE = "The index is in use; try Rebuild again in a moment."
STOP_JOIN_S = 1.5  # nil gives all stop callbacks 2 s


@dataclass(frozen=True)
class RunState:
    running: bool = False
    done: int = 0
    total: int = 0
    problem: str | None = None  # from the last run
    rebuild_needed: bool = False  # the problem comes from an IndexFormatError


class IndexRunner:
    """Runs `run_index_paths` on a daemon thread.

    `paths()` gives the tracks to index (nil's library, or a walk of the folders) and
    `embedder()` the model; both are called on the run's thread. `on_change` is called after
    every state change, outside the lock.
    """

    def __init__(
        self,
        cfg: Config,
        paths: Callable[[], list[str]],
        embedder: Callable[[], Embedder],
        on_change: Callable[[], None],
    ):
        self._cfg = cfg
        self._paths = paths
        self._embedder = embedder
        self._on_change = on_change
        self._lock = threading.Lock()
        self._state = RunState()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def state(self) -> RunState:
        with self._lock:
            return self._state

    def start(self, rebuild: bool = False) -> bool:
        """Start a run; False if one is already going."""
        with self._lock:
            if self._state.running:
                return False
            self._state = RunState(running=True)
            self._thread = thread = threading.Thread(
                target=self._thread_main, args=(rebuild,), daemon=True, name="nil-dj-index"
            )
            thread.start()  # inside the lock, so stop() never joins a thread not yet started
        self._on_change()
        return True

    def stop(self) -> None:
        """Ask the run to stop and wait a little for it; quick enough for nil's shutdown."""
        self._stop.set()
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(STOP_JOIN_S)

    def _thread_main(self, rebuild: bool) -> None:
        problem = None
        try:
            problem = self._run(rebuild)
        except Exception as e:  # noqa: BLE001 - reported in the state, never lost
            traceback.print_exc(file=sys.stderr)
            problem = (f"Indexing failed: {str(e) or type(e).__name__}", False)
        finally:
            with self._lock:
                s = self._state
                text, rebuild_needed = problem or (None, False)
                self._state = RunState(False, s.done, s.total, text, rebuild_needed)
            self._on_change()

    def _progress(self, p: Progress) -> None:
        with self._lock:
            self._state = RunState(True, p.done, p.total)
        self._on_change()

    def _run(self, rebuild: bool) -> tuple[str, bool] | None:
        """One run; returns (problem, rebuild_needed) or None."""
        paths = self._paths()
        if not paths:  # refused before a rebuild could remove anything
            return EMPTY_LIBRARY, False
        try:
            require_ffmpeg()  # before the model: a first run would download it, then stop
        except FfmpegMissing as e:
            return str(e), False
        index_path = self._cfg.index_path
        try:
            emb = self._embedder()
        except ModelDownloadFailed as e:
            traceback.print_exc(file=sys.stderr)  # the cause, for nil's plugin log
            if str(e) != MODEL_DOWNLOAD_FAILED:  # the disk, not the network
                return str(e), False
            label = "Update index" if index_path.exists() else "Index library"
            return INDEX_NEEDS_MODEL.format(label), False
        if self._stop.is_set():
            return None
        lock_path = Path(f"{index_path}.lock")
        try:
            if rebuild:
                with FileLock(lock_path, timeout=0):
                    try:
                        Index.remove(index_path)
                    except OSError:  # Windows: another process has the file open
                        traceback.print_exc(file=sys.stderr)
                        return REBUILD_IN_USE, True
            index = Index.open(index_path, emb.model_id)
        except Timeout:
            return ALREADY_INDEXING, False
        except IndexFormatError as e:
            return str(e), True
        except sqlite3.OperationalError:  # locked or can't be opened: a rebuild won't help
            traceback.print_exc(file=sys.stderr)
            return INDEX_BUSY, False
        except sqlite3.DatabaseError as e:  # not an index at all: only a rebuild helps
            return f"the index file can't be read ({e})", True
        try:
            index.set_meta("precision", self._cfg.precision)
            run_index_paths(paths, index, emb, lock_path, self._progress, stop=self._stop)
        except AlreadyRunning:
            return ALREADY_INDEXING, False
        except EmptyLibrary:
            return EMPTY_LIBRARY, False
        except FfmpegMissing as e:
            return str(e), False
        finally:
            index.close()
        return None
