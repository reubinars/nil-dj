"""Builds and refreshes the index: incremental by size + mtime, one commit per track (so an
interrupted run keeps its progress), and one indexer at a time via a lock file."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock, Timeout

from nil_dj.decode import DecodeError, decode_segment, integrated_lufs, require_ffmpeg
from nil_dj.embedder import Embedder
from nil_dj.index import Index
from nil_dj.tags import TagError, read_info
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


def _index_one(path: str, size: int, mtime_ns: int, index: Index, embedder: Embedder) -> None:
    p = Path(path)
    info = read_info(p)
    d = info.duration_s
    clips = [decode_segment(p, start, length) for start, length in clip_windows(d)]
    embedding = embedder.embed_audio(clips)
    bpm = estimate_bpm(decode_segment(p, max(0.0, d / 2 - TEMPO_S / 2), TEMPO_S))
    lufs = integrated_lufs(p)
    index.upsert(path, size, mtime_ns, info, bpm, lufs, embedding)


def run_index(
    folders: list[Path],
    index: Index,
    embedder: Embedder,
    lock_path: Path,
    on_progress: Callable[[Progress], None] = lambda p: None,
) -> RunResult:
    lock = FileLock(lock_path, timeout=0)
    try:
        lock.acquire()
    except Timeout as e:
        raise AlreadyRunning(str(lock_path)) from e
    try:
        require_ffmpeg()
        return _run(folders, index, embedder, on_progress)
    finally:
        lock.release()


def _run(folders, index, embedder, on_progress) -> RunResult:
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

    for done, (path, size, mtime_ns, existed) in enumerate(work, 1):
        error = None
        try:
            _index_one(path, size, mtime_ns, index, embedder)
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
    return result
