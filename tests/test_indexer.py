import os
import sys
import threading

import pytest
from fakes import FakeEmbedder
from filelock import FileLock

from nil_dj.index import Index
from nil_dj.indexer import (
    AlreadyRunning,
    EmptyLibrary,
    clip_windows,
    run_index,
    run_index_paths,
    walk,
)


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
