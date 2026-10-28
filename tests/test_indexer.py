import os
import sys
import threading
from dataclasses import replace

import numpy as np
import pytest
from fakes import FakeDetector, FakeEmbedder, fake_detector_factory
from filelock import FileLock

from nil_dj import embedder, indexer, whisper
from nil_dj.decode import DecodeError, decode_segment
from nil_dj.embedder import ModelDownloadFailed
from nil_dj.features import log_mel
from nil_dj.index import Index
from nil_dj.indexer import (
    LANGUAGE_MODEL_FAILED,
    WINDOW_S,
    AlreadyRunning,
    EmptyLibrary,
    Progress,
    _analyse,
    clip_windows,
    language_windows,
    run_index,
    run_index_paths,
    walk,
)
from nil_dj.languages import Window, decode_raw, encode_failed
from nil_dj.tags import read_info


@pytest.fixture
def lib(tmp_path):
    d = tmp_path / "Music"
    d.mkdir()
    return d


@pytest.fixture
def idx(tmp_path):
    i = Index.open(tmp_path / "index.sqlite", "fake")
    yield i
    i.close()


@pytest.fixture
def lock(tmp_path):
    return tmp_path / "index.lock"


def by_path(index):
    rows, _ = index.load()
    return {r.path: r for r in rows}


def test_clip_windows():
    assert clip_windows(5) == [(0, 5)]
    assert clip_windows(10) == [(0, 10)]
    assert clip_windows(20) == [(5, 10)]
    assert clip_windows(100) == [(15, 10), (45, 10), (75, 10)]
    w = clip_windows(31)
    assert [s for s, _ in w] == pytest.approx([1.2, 10.5, 19.8])
    for d in (5, 10, 10.5, 20, 29.9, 30, 31, 100, 7200):
        for start, length in clip_windows(d):
            assert 0 <= start and start + length <= d + 1e-9


def _nil_tree(lib) -> list[str]:
    (lib / "a.FLAC").write_bytes(b"")
    (lib / "sub" / "深い").mkdir(parents=True)
    (lib / "sub" / "b.m4a").write_bytes(b"")
    (lib / "sub" / "深い" / "c.dsf").write_bytes(b"")
    (lib / "notes.txt").write_text("x")
    (lib / ".flac").write_bytes(b"")  # no extension, as in nil (Rust's Path::extension)
    return sorted(
        [str(lib / "a.FLAC"), str(lib / "sub" / "b.m4a"), str(lib / "sub" / "深い" / "c.dsf")]
    )


def test_walk_matches_nil_rules(lib):
    expected = _nil_tree(lib)
    assert walk([lib]) == expected


def test_walk_skips_symlinks(lib):
    expected = _nil_tree(lib)
    try:
        os.symlink(lib / "a.FLAC", lib / "link.flac")
        os.symlink(lib / "sub", lib / "linked-dir", target_is_directory=True)
    except OSError as e:  # Windows without admin or Developer Mode
        pytest.skip(f"can't create symlinks here: {e}")
    assert walk([lib]) == expected


@pytest.mark.skipif(sys.platform != "win32", reason="junctions are Windows-only")
def test_walk_skips_junctions(lib):
    import _winapi

    expected = _nil_tree(lib)
    _winapi.CreateJunction(str(lib / "sub"), str(lib / "junction"))
    assert (lib / "junction" / "b.m4a").exists()
    assert walk([lib]) == expected


def test_walk_skips_missing_folder(tmp_path):
    assert walk([tmp_path / "nope"]) == []


def test_index_then_incremental(make_audio, lib, idx, lock):
    paths = [make_audio(f"{n}.flac", 12, folder=lib) for n in "abc"]
    e = FakeEmbedder()
    assert run_index([lib], idx, e, lock).added == 3
    again = run_index([lib], idx, e, lock)
    assert (again.added, again.updated, again.unchanged) == (0, 0, 3)

    st = paths[0].stat()
    os.utime(paths[0], ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    assert run_index([lib], idx, e, lock).updated == 1

    paths[1].unlink()
    r = run_index([lib], idx, e, lock)
    assert (r.removed, r.unchanged) == (1, 2)
    assert idx.count() == 2


def test_rows_carry_tags_and_measurements(make_audio, lib, idx, lock):
    p = make_audio("clicks.flac", 40, kind="clicks", bpm=120, tags={"title": "雨", "artist": "X"},
                   folder=lib)  # fmt: skip
    run_index([lib], idx, FakeEmbedder(), lock)
    row = by_path(idx)[str(p)]
    assert (row.info.title, row.info.artist, row.info.codec) == ("雨", "X", "flac")
    assert row.size == p.stat().st_size and row.mtime_ns == p.stat().st_mtime_ns
    assert row.bpm is not None and min(abs(row.bpm - 120), abs(row.bpm - 240), abs(row.bpm - 60)) < 3
    assert row.lufs is not None


def test_short_and_silent_tracks(make_audio, lib, idx, lock):
    short = make_audio("short.flac", 2, folder=lib)
    quiet = make_audio("quiet.flac", 40, kind="silence", folder=lib)
    r = run_index([lib], idx, FakeEmbedder(), lock)
    assert (r.added, r.failed) == (2, 0)
    rows = by_path(idx)
    assert rows[str(quiet)].lufs is None and rows[str(quiet)].bpm is None
    assert rows[str(short)].bpm is None


def test_bad_files_recorded_and_cleared(make_audio, lib, idx, lock):
    make_audio("good.flac", 12, folder=lib)
    (lib / "x.flac").write_bytes(b"")
    (lib / "y.flac").write_text("not audio at all")
    r = run_index([lib], idx, FakeEmbedder(), lock)
    assert (r.added, r.failed) == (1, 2)
    assert [p for p, _ in idx.errors()] == [str(lib / "x.flac"), str(lib / "y.flac")]

    (lib / "y.flac").unlink()
    make_audio("y.flac", 12, folder=lib)
    r = run_index([lib], idx, FakeEmbedder(), lock)
    assert (r.added, r.failed, r.unchanged) == (1, 1, 1)
    assert [p for p, _ in idx.errors()] == [str(lib / "x.flac")]


def test_error_for_vanished_file_is_dropped(make_audio, lib, idx, lock):
    make_audio("keep.flac", 12, folder=lib)  # an empty library is refused
    (lib / "x.flac").write_bytes(b"")
    run_index([lib], idx, FakeEmbedder(), lock)
    (lib / "x.flac").unlink()
    run_index([lib], idx, FakeEmbedder(), lock)
    assert idx.errors() == []


class InterruptingEmbedder(FakeEmbedder):
    def __init__(self, at_call: int):
        self.calls = 0
        self.at_call = at_call

    def embed_audio(self, clips):
        self.calls += 1
        if self.calls == self.at_call:
            raise KeyboardInterrupt
        return super().embed_audio(clips)


def test_interrupt_keeps_progress(make_audio, lib, idx, lock):
    for n in "abcde":
        make_audio(f"{n}.flac", 12, folder=lib)
    with pytest.raises(KeyboardInterrupt):
        run_index([lib], idx, InterruptingEmbedder(at_call=3), lock)
    assert idx.count() == 2
    with FileLock(lock, timeout=0):
        pass
    assert run_index([lib], idx, FakeEmbedder(), lock).added == 3


def test_second_indexer_refused(make_audio, lib, idx, lock):
    make_audio("a.flac", 12, folder=lib)
    with FileLock(lock), pytest.raises(AlreadyRunning):
        run_index([lib], idx, FakeEmbedder(), lock)


class ThreadRecordingEmbedder(FakeEmbedder):
    def __init__(self):
        self.threads = set()

    def embed_audio(self, clips):
        self.threads.add(threading.get_ident())
        return super().embed_audio(clips)


def test_parallel_matches_serial(make_audio, lib, tmp_path, lock):
    for i in range(6):
        make_audio(f"{i}.flac", 12 + i, freq=300 + 100 * i, folder=lib)
    (lib / "bad.flac").write_bytes(b"")
    results = []
    for workers in (1, 4):
        index = Index.open(tmp_path / f"w{workers}.sqlite", "fake")
        r = run_index([lib], index, FakeEmbedder(), lock, workers=workers)
        rows, m = index.load()
        results.append(((r.added, r.failed), [(x.path, x.info, x.bpm, x.lufs) for x in rows],
                        m.tobytes(), index.errors()))  # fmt: skip
        index.close()
    assert results[0] == results[1]
    assert results[0][0] == (6, 1)


def test_parallel_embeds_on_calling_thread(make_audio, lib, idx, lock):
    for i in range(4):
        make_audio(f"{i}.flac", 12, folder=lib)
    e = ThreadRecordingEmbedder()
    run_index([lib], idx, e, lock, workers=4)
    assert e.threads == {threading.get_ident()}


def test_parallel_interrupt_keeps_progress(make_audio, lib, idx, lock):
    for n in "abcdef":
        make_audio(f"{n}.flac", 12, folder=lib)
    with pytest.raises(KeyboardInterrupt):
        run_index([lib], idx, InterruptingEmbedder(at_call=3), lock, workers=4)
    assert idx.count() == 2
    with FileLock(lock, timeout=0):
        pass
    assert run_index([lib], idx, FakeEmbedder(), lock, workers=4).added == 4


def test_parallel_progress_in_walk_order(make_audio, lib, idx, lock):
    make_audio("a.flac", 40, folder=lib)  # slowest, first in walk order
    for n in "bcd":
        make_audio(f"{n}.flac", 2, folder=lib)
    seen = []
    run_index([lib], idx, FakeEmbedder(), lock, on_progress=seen.append, workers=4)
    assert [os.path.basename(p.path) for p in seen] == ["a.flac", "b.flac", "c.flac", "d.flac"]
    assert [p.done for p in seen] == [1, 2, 3, 4]


def test_progress_reported_per_file(make_audio, lib, idx, lock):
    make_audio("a.flac", 12, folder=lib)
    (lib / "x.flac").write_bytes(b"")
    seen = []
    run_index([lib], idx, FakeEmbedder(), lock, on_progress=seen.append)
    assert [(p.done, p.total) for p in seen] == [(1, 2), (2, 2)]
    assert seen[0].error is None and seen[1].error is not None


def test_index_paths_only_those(make_audio, lib, idx, lock):
    paths = [str(make_audio(f"{n}.flac", 12, folder=lib)) for n in "abc"]
    r = run_index_paths(paths[:2], idx, FakeEmbedder(), lock)
    assert r.added == 2
    assert set(by_path(idx)) == set(paths[:2])


def test_index_paths_removes_rows_not_in_library(make_audio, lib, idx, lock):
    paths = [str(make_audio(f"{n}.flac", 12, folder=lib)) for n in "abc"]
    e = FakeEmbedder()
    run_index_paths(paths, idx, e, lock)
    r = run_index_paths(paths[:2], idx, e, lock)
    assert (r.removed, r.unchanged) == (1, 2)
    assert idx.count() == 2


def test_index_paths_empty_refuses(make_audio, lib, idx, lock):
    paths = [str(make_audio(f"{n}.flac", 12, folder=lib)) for n in "ab"]
    run_index_paths(paths, idx, FakeEmbedder(), lock)
    with pytest.raises(EmptyLibrary):
        run_index_paths([], idx, FakeEmbedder(), lock)
    assert idx.count() == 2


def test_index_paths_missing_file_recorded(lib, idx, lock):
    missing = str(lib / "nope.flac")
    r = run_index_paths([missing], idx, FakeEmbedder(), lock)
    assert r.failed == 1
    [(path, error)] = idx.errors()
    assert path == missing and error.startswith("can't stat")


def test_stop_flag(make_audio, lib, idx, lock):
    paths = [str(make_audio(f"{n}.flac", 12, folder=lib)) for n in "abc"]
    stop = threading.Event()
    e = FakeEmbedder()
    r = run_index_paths(paths, idx, e, lock, on_progress=lambda p: stop.set(), workers=1, stop=stop)
    assert r.stopped and r.added == 1
    assert idx.count() == 1
    r2 = run_index_paths(paths, idx, e, lock)
    assert (r2.added, r2.unchanged, r2.stopped) == (2, 1, False)
    assert idx.count() == 3


def test_japanese_path(make_audio, lib, idx, lock):
    p = str(make_audio("曲/深い.flac", 12, folder=lib))
    run_index_paths([p], idx, FakeEmbedder(), lock)
    assert list(by_path(idx)) == [p]


def test_preset_stop_writes_nothing(make_audio, lib, idx, lock):
    paths = [str(make_audio(f"{n}.flac", 12, folder=lib)) for n in "abc"]
    run_index_paths(paths[:2], idx, FakeEmbedder(), lock)
    idx.record_error("/gone.flac", "old")
    before = (set(by_path(idx)), idx.errors())
    stop = threading.Event()
    stop.set()
    missing = str(lib / "nope.flac")
    r = run_index_paths([paths[0], missing], idx, FakeEmbedder(), lock, stop=stop)
    assert r.stopped and (r.added, r.removed, r.failed) == (0, 0, 0)
    assert (set(by_path(idx)), idx.errors()) == before


def test_stop_set_during_embed_writes_nothing(make_audio, lib, idx, lock):
    path = str(make_audio("a.flac", 12, folder=lib))
    stop = threading.Event()

    class StoppingEmbedder(FakeEmbedder):
        def embed_audio(self, clips):
            stop.set()
            return super().embed_audio(clips)

    r = run_index_paths([path], idx, StoppingEmbedder(), lock, stop=stop)
    assert r.stopped and r.added == 0
    assert idx.count() == 0


def test_run_index_empty_folders_refuses(make_audio, lib, idx, lock, tmp_path):
    make_audio("a.flac", 12, folder=lib)
    run_index([lib], idx, FakeEmbedder(), lock)
    empty = tmp_path / "unplugged"
    empty.mkdir()
    with pytest.raises(EmptyLibrary):
        run_index([empty], idx, FakeEmbedder(), lock)
    assert idx.count() == 1


def _stored_as(idx, path, stored):
    """Make the row look as if it had been stored under another spelling of its path."""
    with idx._db:
        idx._db.execute("UPDATE tracks SET path = ? WHERE path = ?", (stored, path))


def test_index_paths_matches_rows_by_key(make_audio, lib, idx, lock):
    # nil can report a path in another case or with other slashes than the row was stored
    # under (on Windows both name the same file): the row is kept, not deleted and re-added.
    real = str(make_audio("a.flac", 12, folder=lib))
    run_index_paths([real], idx, FakeEmbedder(), lock)
    stored = real.upper().replace("/", "\\")
    _stored_as(idx, real, stored)
    r = run_index_paths([real], idx, FakeEmbedder(), lock)
    assert (r.removed, r.unchanged, r.added, r.updated) == (0, 1, 0, 0)
    assert set(by_path(idx)) == {stored}

    st = os.stat(real)
    os.utime(real, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    r = run_index_paths([real], idx, FakeEmbedder(), lock)
    assert (r.removed, r.added, r.updated) == (0, 0, 1)
    assert set(by_path(idx)) == {stored}  # updated under the path it was stored as


# --- languages (spec §4, §5.2) ---


def raw_of(index, path):
    sql = "SELECT language_raw FROM tracks WHERE path = ?"
    return index._db.execute(sql, (path,)).fetchone()[0]


def raws(index):
    return dict(index._db.execute("SELECT path, language_raw FROM tracks ORDER BY path"))


class NoEmbedder(FakeEmbedder):
    def embed_audio(self, clips):
        raise AssertionError("embed_audio called for language-only work")


class FailingLoad:
    """A DetectorFactory whose load raises `error` and counts the attempts."""

    def __init__(self, error, model_id="fake-whisper"):
        self.model_id = model_id
        self.error = error
        self.calls = 0

    def load(self):
        self.calls += 1
        raise self.error


def test_language_windows():
    assert language_windows(4.0) == [(0, 4.0)]
    assert language_windows(12.0) == [(0, 12.0)]
    assert language_windows(30.0) == [(0, 30.0)]
    assert language_windows(200.0) == [(25.0, 30.0), (85.0, 30.0), (145.0, 30.0)]
    assert language_windows(31.0) == pytest.approx([(0, 30.0), (0.5, 30.0), (1.0, 30.0)])
    for d in (0.5, 4, 29.9, 30, 30.1, 31, 45, 200, 7200):
        for start, length in language_windows(d):
            assert 0 <= start and start + length <= d + 1e-9
    assert WINDOW_S == 30.0


@pytest.mark.parametrize("d", [4.0, 10.0, 12.0, 29.9, 30.0, 31.0, 45.0, 200.0, 7200.0])
def test_every_clap_clip_lies_inside_a_language_window(d):
    windows = language_windows(d)
    for start, length in clip_windows(d):
        assert any(
            ws - 1e-9 <= start and start + length <= ws + wl + 1e-9 for ws, wl in windows
        ), (d, start, windows)


def _separate_decodes(path):
    d = read_info(path).duration_s
    return [decode_segment(path, s, n) for s, n in clip_windows(d)]


# Odd durations: clip and window starts that aren't whole milliseconds.
@pytest.mark.parametrize("seconds", [5, 7.3457, 12, 23.4567, 31, 37.123, 201.3333])
def test_sliced_clips_equal_separate_decodes_at_48k(make_audio, seconds):
    p = make_audio(f"s{seconds}.flac", seconds, rate=48000, freq=437.3)
    old = _separate_decodes(p)
    new = _analyse(str(p), want_mels=False).clips
    assert len(new) == len(old)
    for a, b in zip(new, old, strict=True):
        assert a.dtype == np.float32 and a.shape == b.shape
        assert np.array_equal(a, b)


# Odd durations: clip and window starts that aren't whole milliseconds.
@pytest.mark.parametrize("seconds", [5, 7.3457, 12, 23.4567, 31, 37.123, 201.3333])
def test_sliced_clips_close_to_separate_decodes_at_44k(make_audio, seconds):
    p = make_audio(f"s{seconds}.flac", seconds, rate=44100, freq=437.3)
    old = _separate_decodes(p)
    new = _analyse(str(p), want_mels=False).clips
    assert len(new) == len(old)
    for a, b in zip(new, old, strict=True):
        assert a.shape == b.shape
        diff = float(np.mean(np.abs(log_mel(a) - log_mel(b))))
        assert diff <= 0.5, diff


def test_short_decode_is_padded_not_crashing(make_audio, lib, idx, lock, monkeypatch):
    p = make_audio("a.flac", 20, rate=48000, folder=lib)
    real_info = indexer.read_info

    def claims(seconds):
        monkeypatch.setattr(
            indexer, "read_info", lambda path: replace(real_info(path), duration_s=seconds)
        )

    # Tags say 30 s: one language window decodes only 20 s; the last CLAP clip (19-29 s) is
    # 1 s of audio, then zeros.
    claims(30.0)
    a = _analyse(str(p), want_mels=True)
    assert [c.size for c in a.clips] == [480000] * 3
    last = a.clips[2]
    assert np.abs(last[:47000]).max() > 0.1 and not last[49000:].any()
    assert len(a.mels) == 1 and a.mels[0].shape == (80, 3000)

    # Tags say 60 s: the 15-45 s window decodes 5 s, so the 25-35 s clip is all padding; the
    # 33-63 s window has no audio at all, which is a decode error (as the old 43-s clip was).
    claims(60.0)
    det = FakeDetector()
    r = run_index([lib], idx, FakeEmbedder(), lock, detector=fake_detector_factory(det))
    assert (r.added, r.failed) == (0, 1)
    [(path, error)] = idx.errors()
    assert path == str(p) and error

    # And when every window decodes something but less than asked, nothing is out of range.
    monkeypatch.setattr(indexer, "read_info", real_info)
    real_decode = indexer.decode_segment
    monkeypatch.setattr(
        indexer, "decode_segment", lambda path, s, n: real_decode(path, s, n)[:1000]
    )
    a = _analyse(str(p), want_mels=True)
    assert [c.size for c in a.clips] == [480000]
    assert not a.clips[0].any()  # the clip starts at 5 s, past the 1000 decoded samples


def test_new_tracks_get_raw_language(make_audio, lib, idx, lock):
    p = str(make_audio("a.flac", 40, folder=lib))
    q = str(make_audio("b.flac", 12, folder=lib))
    make_audio("c.flac", 4, folder=lib)  # a jingle
    make_audio("d.flac", 30, folder=lib)
    seen = []

    def answer(feats):
        seen.append((feats.shape, feats.dtype))
        return {"ja": 0.9, "en": 0.1, "ko": 0.00001}, 0.2

    det = FakeDetector(answer=answer)
    progress = []
    r = run_index([lib], idx, FakeEmbedder(), lock, on_progress=progress.append,
                  detector=fake_detector_factory(det))  # fmt: skip
    assert (r.added, r.detected, r.language_failed, r.language_problem) == (4, 4, 0, None)
    windows = decode_raw(raw_of(idx, p))
    assert windows == [Window({"ja": 0.9, "en": 0.1}, 0.2)] * 3
    assert decode_raw(raw_of(idx, q)) == [Window({"ja": 0.9, "en": 0.1}, 0.2)]
    assert det.calls == 3 + 1 + 1 + 1
    assert seen == [((80, 3000), np.float32)] * 6
    assert {x.phase for x in progress} == {"index"}  # nothing left for a catch-up
    assert idx.get_meta("language_model") == "fake-whisper"


def _index_without_languages(make_audio, lib, idx, lock, names="abc", seconds=12):
    paths = [str(make_audio(f"{n}.flac", seconds, folder=lib)) for n in names]
    run_index([lib], idx, FakeEmbedder(), lock)
    assert all(v is None for v in raws(idx).values())
    return paths


def test_catch_up_detects_only_undetected_and_never_embeds(make_audio, lib, idx, lock):
    paths = _index_without_languages(make_audio, lib, idx, lock)
    idx.set_language_raw(paths[1], encode_failed("earlier"))  # already done: left alone
    det = FakeDetector(answer=lambda f: ({"fr": 1.0}, 0.0))
    progress = []
    r = run_index([lib], idx, NoEmbedder(), lock, on_progress=progress.append,
                  detector=fake_detector_factory(det))  # fmt: skip
    assert (r.added, r.updated, r.unchanged, r.detected) == (0, 0, 3, 2)
    assert det.calls == 2
    assert raw_of(idx, paths[1]) == encode_failed("earlier")
    for p in (paths[0], paths[2]):
        assert decode_raw(raw_of(idx, p)) == [Window({"fr": 1.0}, 0.0)]
    assert progress[0] == Progress(0, 2, "", None, "languages")
    assert [(x.done, x.total, x.path, x.phase) for x in progress[1:]] == [
        (1, 2, paths[0], "languages"), (2, 2, paths[2], "languages"),
    ]  # fmt: skip
    assert idx.language_work() == []


def test_catch_up_after_new_tracks(make_audio, lib, idx, lock):
    old = _index_without_languages(make_audio, lib, idx, lock, names="ab")
    new = str(make_audio("c.flac", 12, folder=lib))
    progress = []
    r = run_index([lib], idx, FakeEmbedder(), lock, on_progress=progress.append, workers=2,
                  detector=fake_detector_factory())  # fmt: skip
    assert (r.added, r.unchanged, r.detected) == (1, 2, 3)
    assert [(x.done, x.total, x.path, x.phase) for x in progress] == [
        (1, 1, new, "index"),
        (0, 2, "", "languages"), (1, 2, old[0], "languages"), (2, 2, old[1], "languages"),
    ]  # fmt: skip
    assert all(v is not None for v in raws(idx).values())


def test_catch_up_resumes_after_stop(make_audio, lib, idx, lock):
    paths = _index_without_languages(make_audio, lib, idx, lock)
    stop = threading.Event()

    def on_progress(p):
        if p.phase == "languages" and p.done == 1:
            stop.set()

    det = FakeDetector()
    r = run_index([lib], idx, NoEmbedder(), lock, on_progress=on_progress, workers=2, stop=stop,
                  detector=fake_detector_factory(det))  # fmt: skip
    assert r.stopped and r.detected == 1
    assert raw_of(idx, paths[0]) is not None
    assert [raw_of(idx, p) for p in paths[1:]] == [None, None]
    assert idx.languages_pending("fake-whisper") == 2
    with FileLock(lock, timeout=0):
        pass

    det2 = FakeDetector()
    progress = []
    r = run_index([lib], idx, NoEmbedder(), lock, on_progress=progress.append,
                  detector=fake_detector_factory(det2))  # fmt: skip
    assert (r.stopped, r.detected, det2.calls) == (False, 2, 2)
    assert [x.path for x in progress[1:]] == paths[1:]
    assert idx.languages_pending("fake-whisper") == 0


def test_stop_during_detection_writes_nothing(make_audio, lib, idx, lock):
    paths = _index_without_languages(make_audio, lib, idx, lock, names="a")
    stop = threading.Event()

    def answer(feats):
        stop.set()
        return {"en": 1.0}, 0.0

    r = run_index([lib], idx, NoEmbedder(), lock, stop=stop,
                  detector=fake_detector_factory(FakeDetector(answer=answer)))  # fmt: skip
    assert r.stopped and r.detected == 0
    assert raw_of(idx, paths[0]) is None


@pytest.mark.parametrize("message", [embedder.MODEL_DOWNLOAD_FAILED, whisper.MODEL_DOWNLOAD_FAILED])
def test_detector_load_failure_keeps_clap_work(make_audio, lib, idx, lock, message):
    old = _index_without_languages(make_audio, lib, idx, lock, names="a")
    new = [str(make_audio(f"{n}.flac", 12, folder=lib)) for n in "bc"]
    factory = FailingLoad(ModelDownloadFailed(message))
    progress = []
    r = run_index([lib], idx, FakeEmbedder(), lock, on_progress=progress.append,
                  detector=factory)  # fmt: skip
    assert (r.added, r.failed, r.unchanged, r.detected) == (2, 0, 1, 0)
    assert r.language_problem == LANGUAGE_MODEL_FAILED
    assert factory.calls == 1  # tried once, then off for the rest of the run
    assert set(by_path(idx)) == {*old, *new}
    assert all(v is None for v in raws(idx).values())
    assert idx.errors() == []
    assert {x.phase for x in progress} == {"index"}  # no catch-up without a detector
    assert idx.languages_pending("fake-whisper") == 3  # the next run retries

    r = run_index([lib], idx, NoEmbedder(), lock, detector=fake_detector_factory())
    assert (r.detected, r.language_problem) == (3, None)


def test_detector_load_failure_other_reason(make_audio, lib, idx, lock):
    _index_without_languages(make_audio, lib, idx, lock, names="ab")
    for error, text in [
        (ModelDownloadFailed("Couldn't save the language model: disk full. Free..."),
         "Couldn't save the language model: disk full. Free..."),
        (RuntimeError("expected 99 language tokens"), "expected 99 language tokens"),
    ]:  # fmt: skip
        factory = FailingLoad(error)
        r = run_index([lib], idx, NoEmbedder(), lock, detector=factory)
        assert (r.language_problem, r.detected, factory.calls) == (text, 0, 1)
        assert all(v is None for v in raws(idx).values())


def test_detect_error_on_one_track_records_failed_raw(make_audio, lib, idx, lock):
    paths = [str(make_audio(f"{n}.flac", 12, folder=lib)) for n in "abc"]

    class Flaky(FakeDetector):
        def detect(self, feats):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("onnx went wrong")
            return {"en": 1.0}, 0.0

    r = run_index([lib], idx, FakeEmbedder(), lock, detector=fake_detector_factory(Flaky()))
    assert (r.added, r.failed, r.detected, r.language_failed) == (3, 0, 2, 1)
    assert raw_of(idx, paths[1]) == encode_failed("onnx went wrong")
    assert decode_raw(raw_of(idx, paths[1])) is None
    assert decode_raw(raw_of(idx, paths[0])) == [Window({"en": 1.0}, 0.0)]
    assert idx.errors() == []
    assert idx.language_work() == []  # a failure is a result: not retried every run


def test_decode_error_in_catch_up_records_failed_raw(make_audio, lib, idx, lock, monkeypatch):
    paths = _index_without_languages(make_audio, lib, idx, lock, names="ab")
    real_decode = indexer.decode_segment

    def decode(path, start, length):
        if str(path) == paths[0]:
            raise DecodeError("Invalid data found when processing input")
        return real_decode(path, start, length)

    monkeypatch.setattr(indexer, "decode_segment", decode)
    r = run_index([lib], idx, NoEmbedder(), lock, detector=fake_detector_factory())
    assert (r.detected, r.language_failed, r.failed) == (1, 1, 0)
    assert raw_of(idx, paths[0]) == encode_failed("Invalid data found when processing input")
    assert decode_raw(raw_of(idx, paths[1])) is not None
    assert idx.errors() == []


def test_no_detector_means_no_language_work(make_audio, lib, idx, lock):
    p = str(make_audio("a.flac", 12, folder=lib))
    progress = []
    r = run_index([lib], idx, FakeEmbedder(), lock, on_progress=progress.append)
    assert (r.added, r.detected, r.language_failed, r.language_problem) == (1, 0, 0, None)
    assert raw_of(idx, p) is None
    assert idx.get_meta("language_model") is None
    assert [x.phase for x in progress] == ["index"]
    r = run_index_paths([p], idx, NoEmbedder(), lock)
    assert (r.unchanged, r.detected) == (1, 0)
    assert raw_of(idx, p) is None


def test_model_change_triggers_catch_up(make_audio, lib, idx, lock):
    paths = [str(make_audio(f"{n}.flac", 12, folder=lib)) for n in "ab"]
    run_index([lib], idx, FakeEmbedder(), lock,
              detector=fake_detector_factory(FakeDetector(model_id="w1")))  # fmt: skip
    assert idx.language_work() == []

    det = FakeDetector(model_id="w2", answer=lambda f: ({"de": 1.0}, 0.5))
    r = run_index_paths(paths, idx, NoEmbedder(), lock, detector=fake_detector_factory(det))
    assert (r.unchanged, r.detected, det.calls) == (2, 2, 2)
    assert idx.get_meta("language_model") == "w2"
    assert all(decode_raw(raw_of(idx, p)) == [Window({"de": 1.0}, 0.5)] for p in paths)


def test_detector_loaded_only_when_needed(make_audio, lib, idx, lock):
    make_audio("a.flac", 12, folder=lib)
    run_index([lib], idx, FakeEmbedder(), lock, detector=fake_detector_factory())
    factory = FailingLoad(AssertionError("loaded with nothing to detect"))
    r = run_index([lib], idx, NoEmbedder(), lock, detector=factory)
    assert (factory.calls, r.unchanged, r.language_problem) == (0, 1, None)


def test_catch_up_skips_tracks_that_vanished_or_failed_to_stat(make_audio, lib, idx, lock):
    paths = _index_without_languages(make_audio, lib, idx, lock, names="abcd")
    os.unlink(paths[0])  # still in nil's list: a stat error, its row is kept
    det = FakeDetector()
    r = run_index_paths(paths[:3], idx, NoEmbedder(), lock, detector=fake_detector_factory(det))
    assert (r.failed, r.removed, r.detected, det.calls) == (1, 1, 2, 2)  # d.flac was removed
    assert set(raws(idx)) == set(paths[:3])
    assert raw_of(idx, paths[0]) is None
    assert idx.errors()[0][0] == paths[0]


def _features_failing_for(monkeypatch, n_ok):
    """whisper.features raises ValueError after `n_ok` successful calls (12-s tracks: one each)."""
    real = whisper.features
    calls = []

    def features(x16):
        calls.append(1)
        if len(calls) == n_ok + 1:
            raise ValueError("degenerate decode")
        return real(x16)

    monkeypatch.setattr(whisper, "features", features)


def test_mel_error_fails_one_track_in_full_work(make_audio, lib, idx, lock, monkeypatch):
    paths = [str(make_audio(f"{n}.flac", 12, folder=lib)) for n in "abc"]
    _features_failing_for(monkeypatch, 1)
    r = run_index([lib], idx, FakeEmbedder(), lock, workers=1, detector=fake_detector_factory())
    assert (r.added, r.failed, r.detected, r.language_failed) == (3, 0, 2, 1)
    assert set(by_path(idx)) == set(paths)  # still indexed with CLAP
    assert raw_of(idx, paths[1]) == encode_failed("degenerate decode")
    assert decode_raw(raw_of(idx, paths[2])) is not None
    assert idx.errors() == []


def test_mel_error_fails_one_track_in_catch_up(make_audio, lib, idx, lock, monkeypatch):
    paths = _index_without_languages(make_audio, lib, idx, lock)
    _features_failing_for(monkeypatch, 1)
    r = run_index([lib], idx, NoEmbedder(), lock, workers=1, detector=fake_detector_factory())
    assert (r.stopped, r.detected, r.language_failed) == (False, 2, 1)
    assert raw_of(idx, paths[1]) == encode_failed("degenerate decode")
    assert decode_raw(raw_of(idx, paths[0])) is not None
    assert decode_raw(raw_of(idx, paths[2])) is not None
    assert idx.errors() == []
