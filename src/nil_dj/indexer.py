"""Builds and refreshes the index: incremental by size + mtime, one commit per track (so an
interrupted run keeps its progress), and one indexer at a time via a lock file."""

from __future__ import annotations

import itertools
import os
import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from filelock import FileLock, Timeout

from nil_dj import embedder as clap
from nil_dj import whisper
from nil_dj.decode import (
    SAMPLE_RATE,
    DecodeError,
    decode_segment,
    integrated_lufs,
    require_ffmpeg,
)
from nil_dj.embedder import Embedder, ModelDownloadFailed
from nil_dj.index import Index, path_key
from nil_dj.languages import Window, encode_failed, encode_raw
from nil_dj.tags import TagError, TrackInfo, read_info
from nil_dj.tempo import estimate_bpm
from nil_dj.whisper import DetectorFactory, LanguageDetector

# The same set nil scans (apps/bp-gui/src/backend/fs_library.rs).
AUDIO_EXTENSIONS = frozenset({"flac", "wav", "aif", "aiff", "m4a", "dsf", "dff"})
CLIP_S = 10.0
TEMPO_S = 60.0
WINDOW_S = 30.0

LANGUAGE_MODEL_FAILED = (
    "Language detection needs the language model, which couldn't be downloaded. "
    "Check the internet connection, then press Update index again."
)
# The download failures that mean "no network" (the others, such as a full disk, keep their text).
_NETWORK_FAILURES = frozenset({clap.MODEL_DOWNLOAD_FAILED, whisper.MODEL_DOWNLOAD_FAILED})


@dataclass
class Progress:
    done: int
    total: int
    path: str
    error: str | None
    phase: str = "index"  # "languages" during the language catch-up


@dataclass
class RunResult:
    added: int = 0
    updated: int = 0
    removed: int = 0
    failed: int = 0
    unchanged: int = 0
    stopped: bool = False
    detected: int = 0
    language_failed: int = 0
    language_problem: str | None = None


class AlreadyRunning(Exception):
    pass


class EmptyLibrary(Exception):
    pass


NO_TRACKS_FOUND = "no tracks found in the music folders (is the drive plugged in?)"


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


def language_windows(duration_s: float) -> list[tuple[float, float]]:
    """(start, length) of the windows decoded per track: the whole track up to 30 s, else three
    30-s windows centred at 20 %, 50 % and 80 %, clamped inside it. Every CLAP clip lies inside
    one of them."""
    d = duration_s
    if d <= WINDOW_S:
        return [(0, d)]
    return [(min(max(c * d - WINDOW_S / 2, 0), d - WINDOW_S), WINDOW_S) for c in (0.2, 0.5, 0.8)]


def _samples(t: float) -> int:
    """Sample offset ffmpeg seeks to for `t` seconds: decode_segment passes times to the
    millisecond, which is a whole number of samples at 48 kHz."""
    return round(float(f"{t:.3f}") * SAMPLE_RATE)


def _slice(audio: np.ndarray, window_start: float, start: float, length: float) -> np.ndarray:
    """The clip (start, length) out of a window decoded from `window_start`: the samples a
    separate decode of the clip would give for a 48 kHz source. Zero-padded when the decode came
    back short (tags that overstate the duration)."""
    offset = _samples(start) - _samples(window_start)
    n = _samples(length)
    out = audio[offset : offset + n]
    if out.size < n:
        out = np.concatenate([out, np.zeros(n - out.size, np.float32)])
    return out


def _clips(decodes: list[np.ndarray], windows, duration_s: float) -> list[np.ndarray]:
    """The CLAP clips (clip_windows) cut out of the decoded language windows."""
    clips = []
    for clip in clip_windows(duration_s):
        start, length = clip
        i, window = next(
            (i, w) for i, w in enumerate(windows)
            if w[0] - 1e-9 <= start and start + length <= w[0] + w[1] + 1e-9
        )  # fmt: skip
        if clip == window:  # a track of up to 10 s: the very same decode as before
            clips.append(decodes[i])
        else:
            clips.append(_slice(decodes[i], window[0], start, length))
    return clips


def _mels(decodes: list[np.ndarray]) -> list[np.ndarray] | str:
    """The Whisper mels of the windows, or why they couldn't be made: a mel failure fails this
    track's language detection, not the track or the run."""
    try:
        return [whisper.features(whisper.to_16k(x)) for x in decodes]
    except Exception as e:  # noqa: BLE001 - e.g. numpy on a degenerate decode
        return str(e) or type(e).__name__


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) // 2)


@dataclass
class _Analysed:
    info: TrackInfo
    clips: list[np.ndarray]
    bpm: float | None
    lufs: float | None
    mels: list[np.ndarray] | str | None = None  # str: why the mels failed


def _analyse(path: str, want_mels: bool = False) -> _Analysed:
    """Everything per track except the models; runs on a worker thread (mostly ffmpeg). Each
    language window is decoded once; the CLAP clips are slices of it."""
    p = Path(path)
    info = read_info(p)
    d = info.duration_s
    windows = language_windows(d)
    decodes = [decode_segment(p, start, length) for start, length in windows]
    clips = _clips(decodes, windows, d)
    bpm = estimate_bpm(decode_segment(p, max(0.0, d / 2 - TEMPO_S / 2), TEMPO_S))
    mels = _mels(decodes) if want_mels else None
    return _Analysed(info, clips, bpm, integrated_lufs(p), mels)


def _analyse_language(path: str, duration_s: float) -> list[np.ndarray] | str:
    """Whisper mels of an indexed track's language windows; runs on a worker thread."""
    p = Path(path)
    return _mels([decode_segment(p, s, n) for s, n in language_windows(duration_s)])


def _wait[T](future: Future[T], stop: threading.Event | None = None) -> T | None:
    """The future's result, or None if stop was set first."""
    # Poll so Ctrl+C reaches the main thread; a blocking wait isn't interruptible on Windows.
    while True:
        if stop is not None and stop.is_set():
            return None
        try:
            return future.result(timeout=0.5)
        except TimeoutError:
            continue


def _locked(lock_path: Path, run: Callable[[], RunResult]) -> RunResult:
    lock = FileLock(lock_path, timeout=0)
    try:
        lock.acquire()
    except Timeout as e:
        raise AlreadyRunning(str(lock_path)) from e
    try:
        require_ffmpeg()
        return run()
    finally:
        lock.release()


def run_index(
    folders: list[Path],
    index: Index,
    embedder: Embedder,
    lock_path: Path,
    on_progress: Callable[[Progress], None] = lambda p: None,
    workers: int | None = None,
    stop: threading.Event | None = None,
    *,
    detector: DetectorFactory | None = None,
) -> RunResult:
    """Index the folders. With a `detector`, also detect languages (new tracks, then the
    catch-up of rows without a detection); None means no language work at all."""
    on_disk = walk(folders)
    if not on_disk:  # an unplugged drive looks like this; never delete every row for it
        raise EmptyLibrary(NO_TRACKS_FOUND)
    n = workers or default_workers()
    return _locked(
        lock_path, lambda: _run(on_disk, index, embedder, on_progress, n, stop, detector)
    )


def run_index_paths(
    paths: list[str],
    index: Index,
    embedder: Embedder,
    lock_path: Path,
    on_progress: Callable[[Progress], None] = lambda p: None,
    workers: int | None = None,
    stop: threading.Event | None = None,
    *,
    detector: DetectorFactory | None = None,
) -> RunResult:
    """Index exactly these paths (nil's library.tracks) without walking. Rows and errors for
    other paths are deleted; an empty list is refused rather than wiping the index."""
    if not paths:
        raise EmptyLibrary("no tracks to index")
    on_disk = sorted(set(paths))
    n = workers or default_workers()
    return _locked(
        lock_path, lambda: _run(on_disk, index, embedder, on_progress, n, stop, detector)
    )


def _load_problem(e: Exception) -> str:
    if isinstance(e, ModelDownloadFailed) and str(e) in _NETWORK_FAILURES:
        return LANGUAGE_MODEL_FAILED
    return str(e) or type(e).__name__


def _detect(det: LanguageDetector, mels: list[np.ndarray] | str) -> tuple[str, bool]:
    """(raw language record of one track, whether detection worked). A mel or detection error
    is stored as a failed record, never in index_errors: the track is still indexed."""
    if isinstance(mels, str):
        return encode_failed(mels), False
    try:
        windows = [Window(*det.detect(m)) for m in mels]
    except Exception as e:  # noqa: BLE001 - any ONNX error fails this track only
        return encode_failed(str(e) or type(e).__name__), False
    return encode_raw(windows), True


def _count(result: RunResult, ok: bool | None) -> None:
    if ok is not None:
        if ok:
            result.detected += 1
        else:
            result.language_failed += 1


def _run(on_disk, index, embedder, on_progress, workers: int, stop=None, detector=None):
    result = RunResult()

    def stopping() -> bool:
        if stop is not None and stop.is_set():
            result.stopped = True
        return result.stopped

    if stopping():
        return result
    if detector is not None:  # a different model invalidates every detection
        index.use_language_model(detector.model_id)
    # nil may spell a path differently from the row (case, slashes): compare by path_key, and
    # keep updating a row under the path it was stored as.
    present = {path_key(p) for p in on_disk}
    stamps = index.stamps()
    stored = {path_key(p): p for p in stamps}

    # Stat everything first so a stop during the walk leaves the index untouched.
    work: list[tuple[str, int, int, str | None]] = []  # path, size, mtime_ns, its row's path
    stat_errors: list[tuple[str, str]] = []
    for path in on_disk:
        if stopping():
            return result
        try:
            st = os.stat(path)
        except OSError as e:
            stat_errors.append((path, f"can't stat: {e}"))
            continue
        row = stored.get(path_key(path))
        if row is not None and stamps[row] == (st.st_size, st.st_mtime_ns):
            result.unchanged += 1
        else:
            work.append((path, st.st_size, st.st_mtime_ns, row))
    if stopping():
        return result

    gone_rows = [p for p in stamps if path_key(p) not in present]
    gone_errors = [p for p, _ in index.errors() if path_key(p) not in present]
    index.delete(gone_rows + gone_errors)
    result.removed = len(gone_rows)
    for path, error in stat_errors:
        index.record_error(path, error)
    result.failed += len(stat_errors)

    # Language-only work: rows without a detection that are on disk, unchanged and statable.
    catch_up: list[tuple[str, float]] = []
    if detector is not None:
        skip = {path_key(p) for p, *_ in work} | {path_key(p) for p, _ in stat_errors}
        catch_up = [
            (p, d) for p, d in index.language_work()
            if path_key(p) in present and path_key(p) not in skip
        ]  # fmt: skip

    # The detector loads the first time a mel needs it; a failed load turns detection off.
    det: LanguageDetector | None = None
    detecting = detector is not None

    def loaded() -> LanguageDetector | None:
        nonlocal det, detecting
        if det is None and detecting:
            try:
                det = detector.load()
            except Exception as e:  # noqa: BLE001 - CLAP work goes on without languages
                result.language_problem = _load_problem(e)
                detecting = False
        return det if detecting else None

    def submit(item) -> Future | None:
        phase, args = item
        if phase == "index":
            return pool.submit(_analyse, args[0], detecting)
        return pool.submit(_analyse_language, *args) if detecting else None

    # Workers analyse ahead; this thread runs the models (DirectML allows one Run at a time per
    # session) and writes (one SQLite writer), in walk order so progress and resumption stay
    # predictable. New and changed tracks come first, then the language catch-up, through the
    # same look-ahead.
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="nil-dj-analyse")
    pending: deque[tuple[tuple[str, tuple], Future | None]] = deque()
    queue = itertools.chain((("index", w) for w in work), (("languages", c) for c in catch_up))
    window = 4 * workers
    try:
        for item in itertools.islice(queue, window):
            pending.append((item, submit(item)))
        done = 0
        while pending:
            if stopping():
                break
            (phase, args), future = pending.popleft()
            for item in itertools.islice(queue, 1):
                pending.append((item, submit(item)))
            if phase == "languages":
                if done == len(work) and catch_up and detecting:  # the first catch-up item
                    on_progress(Progress(0, len(catch_up), "", None, "languages"))
                if future is None or loaded() is None:
                    break
                path, _ = args
                done += 1
                try:
                    mels = _wait(future, stop)
                    if mels is None or stopping():
                        break
                    raw, ok = _detect(det, mels)
                except (DecodeError, OSError) as e:
                    raw, ok = encode_failed(str(e) or type(e).__name__), False
                if stopping():
                    break
                index.set_language_raw(path, raw)
                _count(result, ok)
                on_progress(Progress(done - len(work), len(catch_up), path, None, "languages"))
                continue

            path, size, mtime_ns, row = args
            done += 1
            error = None
            try:
                a = _wait(future, stop)
                if a is None or stopping():
                    break
                embedding = embedder.embed_audio(a.clips)
                raw = ok = None
                if a.mels is not None and loaded() is not None:
                    raw, ok = _detect(det, a.mels)
                if stopping():
                    break
                index.upsert(
                    row or path, size, mtime_ns, a.info, a.bpm, a.lufs, embedding, language_raw=raw
                )
            except (DecodeError, TagError, OSError) as e:
                error = str(e) or type(e).__name__
                index.record_error(path, error)
                result.failed += 1
            else:
                _count(result, ok)
                if row is not None:
                    result.updated += 1
                else:
                    result.added += 1
            on_progress(Progress(done, len(work), path, error))
    finally:
        # On Ctrl+C or stop, drop queued tracks; ones already in ffmpeg finish on their own.
        pool.shutdown(wait=False, cancel_futures=True)
        det = None  # release the model with the run
    return result
