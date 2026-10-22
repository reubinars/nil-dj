"""Builds and refreshes the index: incremental by size + mtime, one commit per track (so an
interrupted run keeps its progress), and one indexer at a time via a lock file."""

from __future__ import annotations

import itertools
import os
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from filelock import FileLock, Timeout

from nil_dj.decode import DecodeError, decode_segment, integrated_lufs, require_ffmpeg
from nil_dj.embedder import Embedder
from nil_dj.index import Index
from nil_dj.tags import TagError, TrackInfo, read_info
from nil_dj.tempo import estimate_bpm

# The same set nil scans (apps/bp-gui/src/backend/fs_library.rs).
AUDIO_EXTENSIONS = frozenset({"flac", "wav", "aif", "aiff", "m4a", "dsf", "dff"})
CLIP_S = 10.0
TEMPO_S = 60.0


@dataclass
class Progress:
    done: int
    total: int
    path: str
    error: str | None


@dataclass
class RunResult:
    added: int = 0
    updated: int = 0
    removed: int = 0
    failed: int = 0
    unchanged: int = 0


class AlreadyRunning(Exception):
    pass


def _walk_dir(folder: str, out: list[str]) -> None:
    try:
        entries = list(os.scandir(folder))
    except OSError:
        return
    for entry in entries:
        try:
            # nil (Rust's file_type()) sees symlinks and junctions as neither file nor dir.
            if entry.is_symlink() or entry.is_junction():
                continue
            if entry.is_dir(follow_symlinks=False):
                _walk_dir(entry.path, out)
            elif entry.is_file(follow_symlinks=False):
                ext = os.path.splitext(entry.name)[1][1:].lower()
                if ext in AUDIO_EXTENSIONS:
                    out.append(str(Path(entry.path)))
        except OSError:
            continue


def walk(folders: list[Path]) -> list[str]:
    """Audio files under the folders, as nil sees them: paths joined with the OS separator and
    never normalised, symlinks and unreadable directories skipped."""
    out: list[str] = []
    for folder in folders:
        _walk_dir(str(folder), out)
    return sorted(out)


def clip_windows(duration_s: float) -> list[tuple[float, float]]:
    """(start, length) of the clips to embed: the whole track up to 10 s, one centred clip up to
    30 s, else three clips centred at 20 %, 50 % and 80 %."""
    d = duration_s
    if d <= CLIP_S:
        return [(0, d)]
    if d < 30:
        return [(d / 2 - CLIP_S / 2, CLIP_S)]
    return [(min(max(c * d - CLIP_S / 2, 0), d - CLIP_S), CLIP_S) for c in (0.2, 0.5, 0.8)]


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) // 2)


@dataclass
class _Analysed:
    info: TrackInfo
    clips: list[np.ndarray]
    bpm: float | None
    lufs: float | None


def _analyse(path: str) -> _Analysed:
    """Everything per track except the model; runs on a worker thread (mostly ffmpeg)."""
    p = Path(path)
    info = read_info(p)
    d = info.duration_s
    clips = [decode_segment(p, start, length) for start, length in clip_windows(d)]
    bpm = estimate_bpm(decode_segment(p, max(0.0, d / 2 - TEMPO_S / 2), TEMPO_S))
    return _Analysed(info, clips, bpm, integrated_lufs(p))


def _wait(future: Future) -> _Analysed:
    # Poll so Ctrl+C reaches the main thread; a blocking wait isn't interruptible on Windows.
    while True:
        try:
            return future.result(timeout=0.5)
        except TimeoutError:
            continue


def run_index(
    folders: list[Path],
    index: Index,
    embedder: Embedder,
    lock_path: Path,
    on_progress: Callable[[Progress], None] = lambda p: None,
    workers: int | None = None,
) -> RunResult:
    lock = FileLock(lock_path, timeout=0)
    try:
        lock.acquire()
    except Timeout as e:
        raise AlreadyRunning(str(lock_path)) from e
    try:
        require_ffmpeg()
        return _run(folders, index, embedder, on_progress, workers or default_workers())
    finally:
        lock.release()


def _run(folders, index, embedder, on_progress, workers: int) -> RunResult:
    result = RunResult()
    on_disk = walk(folders)
    present = set(on_disk)
    stamps = index.stamps()

    gone_rows = [p for p in stamps if p not in present]
    gone_errors = [p for p, _ in index.errors() if p not in present]
    index.delete(gone_rows + gone_errors)
    result.removed = len(gone_rows)

    work: list[tuple[str, int, int, bool]] = []
    for path in on_disk:
        try:
            st = os.stat(path)
        except OSError as e:
            index.record_error(path, f"can't stat: {e}")
            result.failed += 1
            continue
        old = stamps.get(path)
        if old == (st.st_size, st.st_mtime_ns):
            result.unchanged += 1
        else:
            work.append((path, st.st_size, st.st_mtime_ns, old is not None))

    # Workers analyse ahead; this thread embeds (DirectML allows one Run at a time per session)
    # and writes (one SQLite writer), in walk order so progress and resumption stay predictable.
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="nil-dj-analyse")
    pending: deque[tuple[tuple[str, int, int, bool], Future]] = deque()
    queue = iter(work)
    window = 4 * workers
    try:
        for item in itertools.islice(queue, window):
            pending.append((item, pool.submit(_analyse, item[0])))
        done = 0
        while pending:
            (path, size, mtime_ns, existed), future = pending.popleft()
            for item in itertools.islice(queue, 1):
                pending.append((item, pool.submit(_analyse, item[0])))
            done += 1
            error = None
            try:
                a = _wait(future)
                embedding = embedder.embed_audio(a.clips)
                index.upsert(path, size, mtime_ns, a.info, a.bpm, a.lufs, embedding)
            except (DecodeError, TagError, OSError) as e:
                error = str(e) or type(e).__name__
                index.record_error(path, error)
                result.failed += 1
            else:
                if existed:
                    result.updated += 1
                else:
                    result.added += 1
            on_progress(Progress(done, len(work), path, error))
    finally:
        # On Ctrl+C, drop queued tracks; ones already in ffmpeg finish on their own.
        pool.shutdown(wait=False, cancel_futures=True)
    return result
