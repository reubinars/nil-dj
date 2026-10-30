"""FakeNil's behaviour as the plugin sees it on the pipes, and its permission/panel/fail rules."""

import json
import queue
import threading

import pytest
from test_fake_nil import _fake

from nil_dj.host import NilError

A = "E:\\Music\\a.flac"
B = "E:\\Music\\b.flac"


class Wire:
    """Raw lines the plugin would read, pumped on a thread so a bug can't hang the test."""

    def __init__(self, fake):
        self.fake = fake
        self.lines = queue.Queue()
        reader, self.writer = fake.streams()
        fake.start()
        threading.Thread(target=self._pump, args=(reader,), daemon=True).start()

    def _pump(self, reader):
        for line in reader:
            self.lines.put(json.loads(line))

    def send(self, msg):
        self.writer.write(json.dumps({"jsonrpc": "2.0", **msg}) + "\n")
        self.writer.flush()

    def next(self):
        return self.lines.get(timeout=3)

    def expect_quiet(self):
        """A marker request is answered next, so nothing else was sent before it."""
        self.send({"id": 99, "method": "library.folders", "params": {}})
        assert self.next() == {"jsonrpc": "2.0", "id": 99, "result": ["E:\\Music"]}


@pytest.fixture
def wire():
    f = _fake()
    yield Wire(f)
    f.close()


def test_reply_arrives_before_player_changed(wire):
    wire.send({"id": 1, "method": "player.play", "params": {"paths": [A]}})
    assert wire.next()["id"] == 1
    changed = wire.next()
    assert changed["method"] == "player.changed"
    assert changed["params"] == {"state": "playing", "queue_len": 1, "index": 0, "path": A}


def test_unchanged_key_sends_nothing(wire):
    wire.send({"id": 1, "method": "player.play", "params": {"paths": [A]}})
    wire.next()
    wire.next()
    wire.send({"id": 2, "method": "player.append", "params": {"paths": []}})
    assert wire.next()["id"] == 2
    wire.send({"id": 3, "method": "player.play", "params": {"paths": [A]}})
    assert wire.next()["id"] == 3
    wire.expect_quiet()


def test_remove_and_clear_announce_the_new_queue_len(wire):
    wire.send({"id": 1, "method": "player.play", "params": {"paths": [A, B, A]}})
    wire.next()
    wire.next()
    wire.send({"id": 2, "method": "player.remove", "params": {"tracks": [{"index": 2, "path": A}]}})
    assert wire.next()["id"] == 2
    assert wire.next()["params"]["queue_len"] == 2
    wire.send({"id": 3, "method": "player.clear_upcoming", "params": {}})
    assert wire.next()["id"] == 3
    assert wire.next()["params"]["queue_len"] == 1


def test_advance_announces_the_new_index(wire):
    wire.fake.queue = [A, B]
    wire.fake.advance(1)
    assert wire.next()["params"]["index"] == 1


def test_no_player_changed_without_the_playback_grant(wire):
    wire.fake.granted = ["library"]
    wire.fake.queue = [A]
    wire.fake.advance(0)
    wire.expect_quiet()


def test_missing_permission_is_refused():
    f = _fake()
    f.granted = ["library"]
    with pytest.raises(NilError) as e:
        f._on_request("player.state", {})
    assert e.value.code == -32001


def test_bad_panel_fails_the_next_wait_for():
    f = _fake()
    with pytest.raises(NilError) as e:
        f._on_request("panel.set", {"panel": {"elements": [{"type": "heading"}]}})
    assert e.value.code == -32602
    assert f.panels == []
    with pytest.raises(AssertionError, match="bad panel"):
        f.wait_for(lambda: True)
    f.wait_for(lambda: True)  # reported once


def test_fail_returns_the_configured_error():
    f = _fake()
    f.fail["player.play"] = (-32003, "no audio device")
    with pytest.raises(NilError) as e:
        f._on_request("player.play", {"paths": [A]})
    assert (e.value.code, e.value.message) == (-32003, "no audio device")
    assert f.queue == []
