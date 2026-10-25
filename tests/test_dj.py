import itertools
import threading
import time

import pytest
from fake_nil import FakeNil
from fakes import ChosenText, FakeEmbedder, add_track, unit

from nil_dj.config import Config
from nil_dj.decode import FfmpegMissing
from nil_dj.dj import DJ, NOT_IN_NIL, NotInNil
from nil_dj.embedder import REVISION, model_files
from nil_dj.host import Connection, NilError
from nil_dj.index import Index

P = [f"E:\\Music\\{a}\\0{n}.flac" for n, a in enumerate("ABCDEF")]
J1 = "E:\\Music\\雨\\01 雨音.flac"
J2 = "E:\\Music\\宇多田ヒカル\\02 First Love.flac"
NEW = "E:\\Music\\New\\01.flac"  # in nil's library, not in the index
TRACKS = [  # path, artist, vector
    (P[0], "A", unit(1, 0, 0)),
    (P[1], "B", unit(1, 0.2, 0)),  # p0's nearest
    (P[2], "C", unit(1, 0.5, 0)),
    (P[3], "D", unit(0, 1, 0)),
    (P[4], "E", unit(0, 1, 0.3)),  # p3's neighbour
    (P[5], "F", unit(0, 0, 1)),
    (J1, "雨", unit(0.5, 0, 1)),
    (J2, "宇多田ヒカル", unit(-1, 0, 0)),
]
NO_INDEX = "No index yet — press Index library."
EMPTY = "nil's library is empty — add a music folder in nil first."


def library(paths):
    return [
        {"path": p, "title": None, "artist": None, "album": None, "duration_s": 240.0}
        for p in paths
    ]


def connect(fake: FakeNil) -> Connection:
    fake.granted = ["playback", "library", "playlists"]
    reader, writer = fake.streams()
    conn = Connection(reader, writer, lambda m, p: {}, lambda m, p: None, 5.0)
    threading.Thread(target=conn.serve, daemon=True).start()
    fake.start()
    return conn


@pytest.fixture
def cfg(tmp_path):
    return Config(index_path=tmp_path / "data" / "index.sqlite", folders=[])


@pytest.fixture
def fake():
    f = FakeNil(library([t[0] for t in TRACKS] + [NEW]))
    yield f
    f.close()


@pytest.fixture
def text():
    return ChosenText(unit(1, 0, 0))


@pytest.fixture
def dj(cfg, fake, text):
    index = Index.open(cfg.index_path, "fake")
    for path, artist, v in TRACKS:
        add_track(index, path, artist=artist, album=f"{artist} album", vector=v)
    index.close()
    d = DJ(cfg, connect(fake), lambda: text)
    yield d
    d.stop()


def paths(tracks):
    return [t.path for t in tracks]


def until(pred, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def test_tracks_carry_tags(dj):
    (t,) = dj.find_tracks(artist="宇多田")
    assert t.path == J2
    assert t.title == "02 First Love"  # no title tag: the file stem
    assert t.artist == "宇多田ヒカル"
    assert t.format == "flac 16-bit/44.1 kHz"
    assert t.item() == {"path": J2, "title": "02 First Love", "artist": "宇多田ヒカル",
                        "duration_s": 240.0}  # fmt: skip
    assert t.as_dict()["score"] is None


def test_find_by_tags(dj):
    assert paths(dj.find_tracks(artist="雨")) == [J1]
    assert paths(dj.find_tracks(album="b ALBUM")) == [P[1]]


def test_search_tracks(dj, text):
    found = dj.search_tracks("bright", limit=2)
    assert paths(found) == [P[0], P[1]]
    assert found[0].score == pytest.approx(1.0)
    assert text.seen == ["bright"]


def test_build_playlist_orders_and_fits(dj):
    pl = dj.build_playlist("bright", minutes=12)
    assert len(pl.tracks) == 3
    artists = [t.artist for t in pl.tracks]
    assert all(a != b for a, b in itertools.pairwise(artists))
    assert pl.total_s == 720
    assert pl.title == "bright — 3 tracks, 12:00"
    assert pl.tracks[0].path == P[0]


def test_build_playlist_uses_template_and_exclude(cfg, fake, text, dj):
    cfg.prompt_template = "This is a sound of {}"
    pl = dj.build_playlist("rain", minutes=4, exclude=[P[0]])
    assert text.seen == ["This is a sound of rain"]
    assert paths(pl.tracks) == [P[1]]


def test_build_playlist_filters(dj):
    pl = dj.build_playlist("bright", filters={"formats": ["dsd"]})
    assert pl.tracks == []
    assert pl.total_s == 0
    assert pl.title == "bright — 0 tracks, 0:00"


def test_unknown_filter(dj):
    with pytest.raises(ValueError, match="bpm"):
        dj.build_playlist("x", filters={"bpm": 3})
    with pytest.raises(ValueError, match="artists"):
        dj.search_tracks("x", filters={"artists": "Björk"})


def test_seeds_steer(dj):
    def score_of_neighbour(**kw):
        pl = dj.build_playlist("bright", minutes=40, **kw)
        return next(t.score for t in pl.tracks if t.path == P[4])

    assert score_of_neighbour(seeds=[P[3]]) > score_of_neighbour() + 0.3
    with pytest.raises(ValueError, match="isn't in the index"):
        dj.build_playlist("bright", seeds=[NEW])


def test_similar_tracks(dj, fake):
    assert paths(dj.similar_tracks(P[0], limit=2)) == [P[1], P[2]]
    assert paths(dj.similar_tracks({"album": "d album", "artist": "d"}, limit=1)) == [P[4]]
    fake.queue, fake.index, fake.state = [P[3]], 0, "playing"
    assert paths(dj.similar_tracks("now_playing", limit=1)) == [P[4]]
    with pytest.raises(ValueError, match="isn't in the index"):
        dj.similar_tracks(NEW)
    with pytest.raises(ValueError, match="No album"):
        dj.similar_tracks({"album": "nope"})


def test_more_like_playing(dj, fake):
    fake.queue, fake.index, fake.state = [P[0]], 0, "playing"
    pl = dj.more_like_playing(minutes=12)
    assert P[0] not in paths(pl.tracks)
    assert pl.tracks[0].path == P[1]
    assert pl.title == "More like 00 — 3 tracks, 12:00"

    fake.state, fake.index = "stopped", None
    with pytest.raises(ValueError) as e:
        dj.more_like_playing()
    assert str(e.value) == "Nothing is playing."

    fake.queue, fake.index, fake.state = [NEW], 0, "playing"
    with pytest.raises(ValueError) as e:
        dj.more_like_playing()
    assert str(e.value) == "The playing track isn't in the index yet — press Update index."


def test_play_and_append_reach_nil(dj, fake):
    dj.play([J1, P[0]])
    assert fake.queue == [J1, P[0]]
    assert [p.encode("utf-8") for p in fake.queue] == [J1.encode("utf-8"), P[0].encode("utf-8")]
    dj.append([J2])
    assert fake.queue == [J1, P[0], J2]
    assert fake.queue[2].encode("utf-8") == J2.encode("utf-8")
    now = dj.now_playing()
    assert now["state"] == "playing" and now["path"] == J1 and now["queue_len"] == 3
    assert now["track"]["artist"] == "雨"


def test_play_with_radio_turns_radio_on(dj, fake):
    assert not dj.radio_on
    dj.play([P[0]], radio=True)
    assert fake.queue[0] == P[0]
    assert len(fake.queue) == 6  # a lone track is at the end: radio adds five
    assert dj.radio_on
    dj.set_radio(False)
    assert not dj.radio_on


def test_nil_error_passes_through(dj, fake):
    fake.fail["player.play"] = (-32003, "no audio device")
    with pytest.raises(NilError) as e:
        dj.play([P[0]])
    assert (e.value.code, e.value.message) == (-32003, "no audio device")


def test_not_in_nil(dj, cfg, text):
    alone = DJ(cfg, None, lambda: text)
    for call in (
        lambda: alone.play([P[0]]),
        lambda: alone.append([P[0]]),
        alone.now_playing,
        lambda: alone.set_radio(True),
        alone.list_playlists,
        lambda: alone.more_like_playing(),
    ):
        with pytest.raises(NotInNil) as e:
            call()
        assert str(e.value) == NOT_IN_NIL
    assert paths(alone.search_tracks("bright", limit=1)) == [P[0]]
    assert alone.index_status().pending is None


def test_playlists(dj, fake):
    assert dj.save_playlist("a/b", [J1, P[1]]) == "a_b"
    assert "a_b" in dj.list_playlists()
    assert dj.play_playlist("a_b") == 2
    assert fake.queue == [J1, P[1]]
    fake.playlists["gone"] = ["E:\\Music\\gone.flac"]
    with pytest.raises(ValueError) as e:
        dj.play_playlist("gone")
    assert str(e.value) == "Playlist 'gone' has no playable tracks."


def test_listener_errors_do_not_break(dj, capsys):
    seen = []
    dj.listeners.append(lambda: 1 / 0)
    dj.listeners.append(lambda: seen.append(1))
    dj.changed()
    assert seen == [1]
    assert "ZeroDivisionError" in capsys.readouterr().err


# -- index file problems -------------------------------------------------------------------


def test_no_index(cfg, text):
    alone = DJ(cfg, None, lambda: text)
    status = alone.index_status()
    assert status.problem == NO_INDEX and not status.rebuild_needed
    with pytest.raises(ValueError) as e:
        alone.search_tracks("x")
    assert str(e.value) == NO_INDEX


def newer_schema(cfg):
    index = Index.open(cfg.index_path, "fake")
    index.set_meta("schema_version", "99")
    index.close()


def test_index_format_error_needs_rebuild(cfg, text):
    newer_schema(cfg)
    alone = DJ(cfg, None, lambda: text)
    status = alone.index_status()
    assert status.problem == "index schema '99' is newer than this nil-dj understands"
    assert status.rebuild_needed
    with pytest.raises(ValueError, match="newer than this nil-dj"):
        alone.search_tracks("x")


def test_model_mismatch_needs_rebuild(cfg, text):
    Index.open(cfg.index_path, "other model").close()
    alone = DJ(cfg, None, lambda: text)
    assert alone.index_status().problem is None  # the model isn't known until it loads
    with pytest.raises(ValueError, match="index built with other model"):
        alone.search_tracks("x")
    status = alone.index_status()
    assert "index built with other model" in status.problem and status.rebuild_needed


# -- indexing ------------------------------------------------------------------------------


@pytest.fixture
def audio_lib(make_audio, tmp_path):
    return [str(make_audio(f"{n}.flac", 3, freq=f, folder=tmp_path / "Music"))
            for n, f in (("a", 300), ("雨", 500), ("c", 700))]  # fmt: skip


class Gate:
    """An embedder factory that waits until the test opens the gate."""

    def __init__(self):
        self.open = threading.Event()
        self.calls = 0

    def __call__(self):
        self.calls += 1
        assert self.open.wait(10)
        return FakeEmbedder()


def test_index_status_and_reindex(cfg, audio_lib):
    fake = FakeNil(library(audio_lib))
    try:
        gate = Gate()
        d = DJ(cfg, connect(fake), gate)
        d.refresh_library()
        before = d.index_status()
        assert before.problem == NO_INDEX
        assert before.pending == 3 and not before.running

        changes = []
        d.listeners.append(lambda: changes.append(d.index_status().done))
        assert d.reindex() is True
        until(lambda: d.index_status().running)
        assert d.reindex() is False
        gate.open.set()
        until(lambda: not d.index_status().running)
        after = d.index_status()
        assert (after.indexed, after.pending, after.failed, after.problem) == (3, 0, 0, None)
        assert (after.done, after.total) == (3, 3)
        assert 3 in changes

        fake.tracks = []
        assert d.reindex(rebuild=True) is True
        until(lambda: not d.index_status().running)
        status = d.index_status()
        assert status.problem == EMPTY and not status.rebuild_needed
        assert status.indexed == 3  # nothing deleted, and rebuild didn't remove the file
    finally:
        fake.close()


def test_reindex_standalone_walks_folders(cfg, audio_lib, tmp_path):
    cfg.folders = [tmp_path / "Music"]
    d = DJ(cfg, None, FakeEmbedder)
    assert d.reindex()
    until(lambda: not d.index_status().running)
    assert d.index_status().indexed == 3

    cfg.folders = [tmp_path / "unplugged"]
    assert d.reindex()
    until(lambda: not d.index_status().running)
    assert d.index_status().problem == EMPTY
    assert d.index_status().indexed == 3


def test_reindex_problems_are_reported(cfg, audio_lib, monkeypatch, capsys):
    fake = FakeNil(library(audio_lib))
    try:
        d = DJ(cfg, connect(fake), FakeEmbedder)
        changed = threading.Event()
        d.listeners.append(changed.set)

        def missing():
            raise FfmpegMissing("ffmpeg not found on PATH.")

        monkeypatch.setattr("nil_dj.reindex.require_ffmpeg", missing)
        d.reindex()
        until(lambda: not d.index_status().running)
        assert d.index_status().problem == "ffmpeg not found on PATH."
        assert changed.is_set()
        monkeypatch.undo()

        def broken(*a, **kw):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr("nil_dj.reindex.run_index_paths", broken)
        d.reindex()
        until(lambda: not d.index_status().running)
        assert "disk on fire" in d.index_status().problem
        assert "RuntimeError: disk on fire" in capsys.readouterr().err
    finally:
        fake.close()


def test_rebuild_clears_a_format_error(cfg, audio_lib):
    newer_schema(cfg)
    fake = FakeNil(library(audio_lib))
    try:
        d = DJ(cfg, connect(fake), FakeEmbedder)
        d.reindex()  # without rebuild the file is kept and the problem stays
        until(lambda: not d.index_status().running)
        assert d.index_status().rebuild_needed
        d.reindex(rebuild=True)
        until(lambda: not d.index_status().running)
        status = d.index_status()
        assert (status.problem, status.rebuild_needed, status.indexed) == (None, False, 3)
    finally:
        fake.close()


def test_stop_returns_quickly_and_ends_the_run(cfg, audio_lib):
    fake = FakeNil(library(audio_lib))
    try:
        gate = Gate()
        d = DJ(cfg, connect(fake), gate)
        d.reindex()
        until(lambda: gate.calls)
        t0 = time.monotonic()
        d.stop()
        assert time.monotonic() - t0 < 1.8
        gate.open.set()
        until(lambda: not d.index_status().running)
        assert d.index_status().indexed == 0
    finally:
        fake.close()


# -- the sound model -----------------------------------------------------------------------


def test_model_state_transitions(cfg, tmp_path):
    cfg.models_dir = tmp_path / "models"
    for f in model_files(cfg.precision):
        (cfg.models_dir / REVISION / f).parent.mkdir(parents=True, exist_ok=True)
        (cfg.models_dir / REVISION / f).write_bytes(b"")
    gate = Gate()
    d = DJ(cfg, None, gate)
    states = []
    d.listeners.append(lambda: states.append(d.model_state))
    assert d.model_state == "not loaded"
    loader = threading.Thread(target=d.embedder)
    loader.start()
    until(lambda: d.model_state == "loading")
    gate.open.set()
    loader.join(5)
    assert d.model_state == "ready"
    assert states == ["loading", "ready"]
    assert d.embedder() is d.embedder()
    assert gate.calls == 1

    def boom():
        raise RuntimeError("boom")

    failing = DJ(cfg, None, boom)
    with pytest.raises(RuntimeError):
        failing.embedder()
    assert failing.model_state == "failed: boom"


def test_model_downloading_when_files_absent(cfg, tmp_path):
    cfg.models_dir = tmp_path / "empty"
    gate = Gate()
    d = DJ(cfg, None, gate)
    loader = threading.Thread(target=d.embedder)
    loader.start()
    until(lambda: d.model_state == "downloading")
    gate.open.set()
    loader.join(5)
    assert d.model_state == "ready"


def test_preload_only_with_an_index(cfg, tmp_path):
    gate = Gate()
    gate.open.set()
    d = DJ(cfg, None, gate)
    d.preload()
    time.sleep(0.1)
    assert gate.calls == 0 and d.model_state == "not loaded"
    Index.open(cfg.index_path, "fake").close()
    d.preload()
    until(lambda: d.model_state == "ready")


# -- fix round 1 ---------------------------------------------------------------------------


def test_more_like_playing_skips_the_seeds_artist(dj, fake, cfg):
    twin = "E:\\Music\\A\\99.flac"  # the seed's artist and album, nearer than anything else
    index = Index.open(cfg.index_path, None)
    add_track(index, twin, artist="A", album="A album", vector=unit(1, 0.05, 0))
    index.close()
    fake.queue, fake.index, fake.state = [P[0]], 0, "playing"
    pl = dj.more_like_playing(minutes=12)
    assert pl.tracks[0].path == P[1]  # the history rule passes over the seed's artist


def test_seeded_query_is_normalised(dj):
    pl = dj.build_playlist("bright", minutes=40, seeds=[P[3]])
    assert pl.tracks
    assert all(t.score <= 1 + 1e-6 for t in pl.tracks)


def test_pool_size_follows_the_minutes(dj, fake, monkeypatch):
    import nil_dj.dj as dj_module

    limits = []
    real = dj_module.rank

    def spy(index, q, limit, *a, **kw):
        limits.append(limit)
        return real(index, q, limit, *a, **kw)

    monkeypatch.setattr("nil_dj.dj.rank", spy)
    dj.build_playlist("bright", minutes=601)  # ceil(601 * 60 / 180) = 201 tracks -> 603
    dj.build_playlist("bright", minutes=45)  # 15 tracks -> the floor of 60
    fake.queue, fake.index, fake.state = [P[0]], 0, "playing"
    dj.more_like_playing(minutes=601)
    assert limits == [603, 60, 603]


def test_pending_leaves_out_failed_paths(dj):
    index = Index.open(dj.cfg.index_path, None)
    index.record_error(NEW, "can't decode")
    index.close()
    dj.refresh_library()
    status = dj.index_status()
    assert (status.pending, status.failed) == (0, 1)


def test_reindex_on_a_corrupt_file_needs_rebuild(cfg, audio_lib):
    cfg.index_path.parent.mkdir(parents=True)
    cfg.index_path.write_bytes(b"this is not a database" * 100)
    fake = FakeNil(library(audio_lib))
    try:
        d = DJ(cfg, connect(fake), FakeEmbedder)
        assert d.reindex()
        until(lambda: not d.index_status().running)
        status = d.index_status()
        assert status.rebuild_needed
        assert "can't be read" in status.problem
    finally:
        fake.close()


def test_a_file_without_meta_needs_rebuild_and_is_left_alone(cfg, text):
    cfg.index_path.parent.mkdir(parents=True)
    cfg.index_path.write_bytes(b"")  # an empty file: sqlite would happily make tables in it
    status = DJ(cfg, None, lambda: text).index_status()
    assert status.rebuild_needed
    assert status.problem == f"{cfg.index_path} is not a nil-dj index"
    assert cfg.index_path.read_bytes() == b""


def test_listeners_run_outside_the_load(cfg, tmp_path):
    cfg.models_dir = tmp_path / "empty"
    d = DJ(cfg, None, FakeEmbedder)
    seen = []

    def listener():
        if d.model_state == "downloading":
            try:
                d.embedder()  # on the loading thread: must refuse, not deadlock
            except RuntimeError as e:
                seen.append(str(e))

    d.listeners.append(listener)
    loader = threading.Thread(target=d.embedder, daemon=True)
    loader.start()
    loader.join(5)
    assert not loader.is_alive(), "the listener deadlocked against the model load"
    assert d.model_state == "ready"
    assert seen == ["embedder() was called from a listener while the model loads"]


# -- nil's library decides what can be chosen ------------------------------------------------


def test_tracks_gone_from_nil_are_never_chosen(dj, fake):
    # P[1] is still in the index, but nil no longer has it (the file moved, the folder went).
    fake.tracks = library([t[0] for t in TRACKS if t[0] != P[1]])
    dj.refresh_library()
    assert P[1] not in paths(dj.search_tracks("bright", limit=20))
    assert P[1] not in paths(dj.build_playlist("bright", minutes=40).tracks)
    assert P[1] not in paths(dj.similar_tracks(P[0], limit=20))
    assert P[1] not in paths(dj.similar_tracks({"album": "a album"}, limit=20))
    assert dj.find_tracks(artist="B") == []
    fake.queue, fake.index, fake.state = [P[0]], 0, "playing"
    assert P[1] not in paths(dj.more_like_playing(minutes=40).tracks)


def test_before_nil_is_asked_and_standalone_nothing_is_dropped(dj, cfg, text):
    assert paths(dj.find_tracks(artist="B")) == [P[1]]  # library not read yet
    alone = DJ(cfg, None, lambda: text)
    assert paths(alone.find_tracks(artist="B")) == [P[1]]


def test_paths_differing_only_in_case_or_slashes_are_one_track(cfg, text):
    stored = ["E:\\Music\\A.flac", "E:\\Music\\B\\01.flac"]
    index = Index.open(cfg.index_path, "fake")
    add_track(index, stored[0], artist="A", album="Al", vector=unit(1, 0, 0))
    add_track(index, stored[1], artist="B", album="Bl", vector=unit(1, 0.2, 0))
    index.close()
    reported = ["e:/music/a.flac", "E:/MUSIC/b/01.FLAC"]  # as nil might send them
    fake = FakeNil(library(reported))
    d = DJ(cfg, connect(fake), lambda: text)
    try:
        d.refresh_library()
        assert d.index_status().pending == 0
        assert paths(d.search_tracks("bright")) == stored  # still candidates
        assert paths(d.similar_tracks(reported[0])) == [stored[1]]
        fake.queue, fake.index, fake.state = [reported[0]], 0, "playing"
        assert paths(d.more_like_playing(minutes=4).tracks) == [stored[1]]
        assert d.now_playing()["track"]["path"] == stored[0]
        pl = d.build_playlist("bright", minutes=4, seeds=[reported[1]])  # found: no ValueError
        assert len(pl.tracks) == 1
    finally:
        d.stop()
        fake.close()


# -- a busy or locked index ------------------------------------------------------------------

BUSY = "The index is busy (another nil-dj is using it). Try again in a moment."


def locked(*args, **kwargs):
    import sqlite3

    raise sqlite3.OperationalError("database is locked")


def test_a_busy_index_is_not_a_broken_one(dj, monkeypatch):
    monkeypatch.setattr(Index, "open", locked)
    status = dj.index_status()
    assert (status.problem, status.rebuild_needed) == (BUSY, False)
    with pytest.raises(ValueError) as e:
        dj.find_tracks(artist="A")
    assert str(e.value) == BUSY


def test_indexing_into_a_busy_index_says_so(cfg, audio_lib, monkeypatch):
    fake = FakeNil(library(audio_lib))
    try:
        d = DJ(cfg, connect(fake), FakeEmbedder)
        monkeypatch.setattr(Index, "open", locked)
        assert d.reindex()
        until(lambda: not d.index_status().running)
        status = d.index_status()
        assert (status.problem, status.rebuild_needed) == (BUSY, False)
    finally:
        fake.close()


def test_rebuild_with_the_file_in_use_says_try_again(cfg, audio_lib, monkeypatch):
    def in_use(path):
        raise PermissionError(13, "The process cannot access the file", str(path))

    fake = FakeNil(library(audio_lib))
    try:
        d = DJ(cfg, connect(fake), FakeEmbedder)
        monkeypatch.setattr(Index, "remove", in_use)
        assert d.reindex(rebuild=True)
        until(lambda: not d.index_status().running)
        status = d.index_status()
        assert status.problem == "The index is in use; try Rebuild again in a moment."
        assert status.rebuild_needed
    finally:
        fake.close()


# -- the run's own state ---------------------------------------------------------------------


def test_a_running_rebuild_hides_the_problem_it_fixes(cfg, audio_lib):
    newer_schema(cfg)
    fake = FakeNil(library(audio_lib))
    gate = Gate()
    try:
        d = DJ(cfg, connect(fake), gate)
        assert d.index_status().rebuild_needed
        assert d.reindex(rebuild=True)
        until(lambda: d.index_status().running)
        status = d.index_status()
        assert (status.problem, status.rebuild_needed) == (None, False)
        gate.open.set()
        until(lambda: not d.index_status().running)
        assert d.index_status().problem is None
    finally:
        gate.open.set()
        fake.close()


def test_a_run_whose_thread_cannot_start_is_not_left_running(cfg, text, monkeypatch):
    def no_thread(self):
        raise RuntimeError("can't start new thread")

    d = DJ(cfg, None, lambda: text)
    monkeypatch.setattr(threading.Thread, "start", no_thread)
    with pytest.raises(RuntimeError):
        d.reindex()
    monkeypatch.undo()
    assert not d.index_status().running
