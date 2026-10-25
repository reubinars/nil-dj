import random
import threading
import time

import pytest
from fake_nil import FakeNil
from fakes import ChosenText, add_track, unit

from nil_dj.config import Config, Weights
from nil_dj.dj import DJ
from nil_dj.host import Connection, NilError
from nil_dj.index import Index, IndexProblem
from nil_dj.radio import NO_SEED, Radio

# Twelve artists whose vectors fan out from p0, so "like p0" has a clear order.
P = [f"E:\\Music\\{n:02d}\\01.flac" for n in range(12)]
NEW = "E:\\Music\\New\\01.flac"  # in nil's library, not in the index
A, B, C = P[0], P[1], P[2]


@pytest.fixture
def now():
    return [0.0]


@pytest.fixture
def fake():
    tracks = [{"path": p, "title": None, "artist": None, "album": None, "duration_s": 240.0}
              for p in [*P, NEW]]  # fmt: skip
    f = FakeNil(tracks)
    yield f
    f.close()


@pytest.fixture
def dj(tmp_path, fake, now):
    cfg = Config(index_path=tmp_path / "data" / "index.sqlite", folders=[])
    index = Index.open(cfg.index_path, "fake")
    for n, p in enumerate(P):
        add_track(index, p, artist=f"artist {n}", album=f"album {n}", vector=unit(1, 0.1 * n))
    index.close()
    fake.granted = ["playback", "library", "playlists"]
    reader, writer = fake.streams()
    holder: dict = {}

    def on_notification(method, params):
        if method == "player.changed":
            holder["dj"].radio.on_player_changed(params)

    conn = Connection(reader, writer, lambda m, p: {}, on_notification, 5.0)
    threading.Thread(target=conn.serve, daemon=True).start()
    fake.start()
    d = DJ(cfg, conn, lambda: ChosenText(unit(1)))
    d.radio.rng = random.Random(7)
    d.radio.clock = lambda: now[0]
    holder["dj"] = d
    yield d
    d.stop()


def appends(fake):
    return [p["paths"] for m, p in fake.calls if m == "player.append"]


def settle(seconds=0.3):
    time.sleep(seconds)  # lets a wrong second append show up


def test_tops_up_at_second_to_last(dj, fake):
    dj.play([A, B, C], radio=True)
    settle()
    assert appends(fake) == []  # index 0 of 3
    fake.advance(1)
    fake.wait_for(lambda: len(appends(fake)) == 1)
    (added,) = appends(fake)
    assert len(added) == 5
    assert not {A, B, C} & set(added)
    fake.advance(2)
    settle()
    assert len(appends(fake)) == 1  # the queue grew to 8, index 2 is not near its end


def test_tops_up_again_when_the_longer_queue_runs_down(dj, fake):
    dj.play([A, B, C], radio=True)
    fake.advance(1)
    fake.wait_for(lambda: len(appends(fake)) == 1)
    fake.advance(6)  # second to last of 8
    fake.wait_for(lambda: len(appends(fake)) == 2)
    assert not set(appends(fake)[0]) & set(appends(fake)[1])  # the queue is excluded


def test_single_track_with_radio_tops_up_at_once(dj, fake):
    dj.play([A], radio=True)
    fake.wait_for(lambda: len(appends(fake)) == 1)
    settle()
    assert len(appends(fake)) == 1  # play's own player.changed did not add a second batch
    assert A not in appends(fake)[0]


def test_off_does_nothing_but_records_history(dj, fake, now):
    now[0] = 50.0
    dj.play([A, B])
    fake.wait_for(lambda: dj.radio.history)
    fake.advance(1)
    fake.wait_for(lambda: len(dj.radio.history) == 2)
    settle()
    assert dj.radio.history == [(A, 50.0), (B, 50.0)]
    assert appends(fake) == []
    assert not dj.radio.on


def test_set_on_while_at_end_tops_up(dj, fake):
    dj.play([A, B])
    fake.wait_for(lambda: dj.radio.history)
    dj.set_radio(True)
    assert len(appends(fake)) == 1


def test_excludes_recent_two_hours(dj, now):
    dj.radio.history = [(A, 0.0)]
    now[0] = 3600.0
    for seed in range(10):
        dj.radio.rng = random.Random(seed)
        assert A not in dj.radio.pick([])
    now[0] = 7300.0
    picks = []
    for seed in range(10):
        dj.radio.rng = random.Random(seed)
        picks += dj.radio.pick([])
    assert A in picks  # the closest track to itself, no longer excluded


def test_pick_skips_the_queue(dj):
    dj.radio.history = [(A, 0.0)]
    assert not {B, C} & set(dj.radio.pick([B, C]))


def test_seed_skips_unindexed(dj, fake):
    dj.play([A, NEW])
    fake.wait_for(lambda: dj.radio.history)
    fake.advance(1)
    fake.wait_for(lambda: len(dj.radio.history) == 2)
    dj.set_radio(True)
    (added,) = appends(fake)
    assert len(added) == 5
    assert dj.radio.on
    assert dj.radio.message is None


def test_no_indexed_history_turns_off(dj, fake):
    changes = []
    dj.listeners.append(lambda: changes.append((dj.radio.on, dj.radio.message)))
    dj.play([NEW])
    fake.wait_for(lambda: dj.radio.history)
    dj.set_radio(True)
    assert not dj.radio.on
    assert dj.radio.message == "Radio stopped: none of the recently played tracks are in the index."
    assert NO_SEED == dj.radio.message
    assert appends(fake) == []
    assert changes[-1] == (False, NO_SEED)


def test_append_error_kept_on(dj, fake):
    fake.fail["player.append"] = (-32003, "no audio device")
    dj.play([A], radio=True)
    fake.wait_for(lambda: dj.radio.message is not None)
    assert dj.radio.message == "Radio: no audio device"
    assert dj.radio.on


def test_deterministic_with_seed(dj):
    dj.radio.history = [(A, 0.0), (B, 1.0)]
    dj.radio.rng = random.Random(7)
    first = dj.radio.pick([])
    dj.radio.rng = random.Random(7)
    assert dj.radio.pick([]) == first
    assert len(first) == 5


def test_play_with_radio_and_set_race_appends_once(dj, fake):
    dj.play([A])  # off: only history
    fake.wait_for(lambda: dj.radio.history)
    dj.radio.on = True
    stale = {"state": "playing", "queue_len": 1, "index": 0, "path": A}
    barrier = threading.Barrier(2)

    def run(fn):
        barrier.wait()
        fn()

    threads = [
        threading.Thread(target=run, args=(lambda: dj.radio.set(True),)),
        threading.Thread(target=run, args=(lambda: dj.radio.on_player_changed(stale),)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    settle()
    assert len(appends(fake)) == 1
    dj.radio.on_player_changed(stale)  # a late, stale notification adds nothing either
    assert len(appends(fake)) == 1


def test_a_stopped_notification_without_index_or_path_is_fine(dj, fake):
    dj.radio.on = True
    dj.radio.on_player_changed({"state": "stopped", "queue_len": 0})
    dj.radio.on_player_changed({"state": "playing", "queue_len": 3})
    assert appends(fake) == []
    assert dj.radio.history == []


def test_an_unexpected_error_releases_the_lock():
    def boom():
        raise RuntimeError("index gone")

    state = {"state": "playing", "queue": ["x"], "index": 0}
    radio = Radio(boom, lambda m, p=None: state, Weights())
    radio.on = True
    with pytest.raises(RuntimeError):
        radio.on_player_changed({"state": "playing", "queue_len": 1, "index": 0, "path": "x"})
    assert radio._lock.acquire(blocking=False)
    radio._lock.release()


def test_nil_error_from_state_is_reported():
    def call(method, params=None):
        raise NilError(-32002, "no permission")

    radio = Radio(lambda: None, call, Weights())
    radio.set(True)
    assert radio.on
    assert radio.message == "Radio: no permission"


def test_a_replacement_queue_of_the_same_length_tops_up_again(dj, fake):
    dj.play([A], radio=True)
    fake.wait_for(lambda: len(appends(fake)) == 1)
    settle()
    dj.play([B])  # queue_len 1, index 0 again, but a different queue
    fake.wait_for(lambda: len(appends(fake)) == 2)
    settle()
    assert len(appends(fake)) == 2


def _no_index_radio():
    def no_index():
        raise IndexProblem("no index yet")

    state = {"state": "playing", "queue": ["x"], "index": 0, "path": "x"}
    changes = []
    radio = Radio(
        no_index, lambda m, p=None: state, Weights(), on_change=lambda: changes.append(1)
    )
    return radio, changes


def test_set_on_without_an_index_turns_radio_off():
    radio, changes = _no_index_radio()
    radio.set(True)
    assert not radio.on
    assert radio.message == "no index yet"
    assert changes == [1]


def test_a_notification_without_an_index_turns_radio_off():
    radio, changes = _no_index_radio()
    radio.on = True
    radio.on_player_changed({"state": "playing", "queue_len": 1, "index": 0, "path": "x"})
    assert not radio.on
    assert radio.message == "no index yet"
    assert changes == [1]


def test_a_closed_connection_is_not_reported():
    def call(method, params=None):
        raise NilError(-32603, "nil has closed the connection")

    changes = []
    radio = Radio(
        lambda: None, call, Weights(), on_change=lambda: changes.append(1), closed=lambda: True
    )
    radio.on = True
    radio.on_player_changed({"state": "playing", "queue_len": 1, "index": 0, "path": "x"})
    assert radio.message is None
    assert changes == []


def test_recent_is_a_copy(dj, fake):
    dj.play([A])
    fake.wait_for(lambda: dj.radio.history)
    recent = dj.radio.recent()
    assert recent == [A]
    recent.append("x")
    assert dj.radio.recent() == [A]


def test_radio_never_adds_tracks_gone_from_nil(dj, fake):
    gone = set(P[3:8])
    fake.tracks = [t for t in fake.tracks if t["path"] not in gone]
    dj.refresh_library()
    dj.play([A], radio=True)
    fake.wait_for(lambda: len(appends(fake)) == 1)
    (added,) = appends(fake)
    assert len(added) == 5
    assert not gone & set(added)


def test_radio_seeds_from_a_path_spelt_differently(dj):
    played = A.replace("\\", "/").lower()  # nil's spelling of the row stored as A
    dj.radio.history = [(played, 0.0)]
    picked = dj.radio.pick([played])
    assert len(picked) == 5
    assert A not in picked  # just played: skipped even though the spelling differs
    assert dj.radio.message is None


def test_failed_append_is_not_retried_for_the_same_queue(dj, fake):
    fake.fail["player.append"] = (-32003, "no audio device")
    dj.play([A], radio=True)
    fake.wait_for(lambda: dj.radio.message is not None)
    fake.notify("player.changed", {"state": "playing", "index": 0, "queue_len": 1, "path": A})
    settle()
    assert sum(1 for m, _ in fake.calls if m == "player.append") == 1


def test_switching_radio_back_on_retries_after_a_failed_append(dj, fake):
    fake.fail["player.append"] = (-32003, "no audio device")
    dj.play([A], radio=True)
    fake.wait_for(lambda: dj.radio.message is not None)
    del fake.fail["player.append"]  # the device is back
    before = len(appends(fake))
    dj.set_radio(False)
    dj.set_radio(True)
    fake.wait_for(lambda: len(appends(fake)) == before + 1)
    assert dj.radio.on and dj.radio.message is None
