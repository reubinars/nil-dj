import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest
from fake_nil import validate_panel
from fakes import FakeEmbedder, add_track, unit

from nil_dj.dj import NO_INDEX, IndexStatus, Playlist, Track
from nil_dj.index import Index
from nil_dj.panel import MCP_NOT_RUNNING, Panel, View, build

MCP_UP = "MCP: http://127.0.0.1:47263/mcp"


def tracks(n):
    return [
        Track(f"E:\\Music\\A\\0{i}.flac", f"Song {i}", "A", "Al", 200.0 + i, "FLAC 16/44.1")
        for i in range(n)
    ]


def view(**kw):
    base = {"status": IndexStatus(indexed=2765), "model_state": "ready", "mcp": MCP_UP}
    return View(**{**base, **kw})


def built(**kw):
    panel = build(view(**kw))
    validate_panel(panel)
    return panel["elements"]


def button_ids(els):
    return [c["id"] for e in els if e["type"] == "row" for c in e["children"] if "id" in c]


def test_no_index():
    els = built(status=IndexStatus(problem=NO_INDEX))
    assert els == [
        {"type": "heading", "text": "nil-dj"},
        {"type": "text", "text": NO_INDEX, "muted": True},
        {"type": "row", "children": [{"type": "button", "id": "index", "label": "Index library"}]},
        {"type": "input", "id": "prompt", "placeholder": "Describe a playlist…",
         "submit_label": "Make", "value": ""},
        {"type": "row", "children": [
            {"type": "button", "id": "more", "label": "More like this"},
            {"type": "toggle", "id": "radio", "label": "Radio", "value": False},
        ]},
        {"type": "text", "text": MCP_UP, "muted": True},
    ]  # fmt: skip


def test_indexing_shows_progress_and_no_buttons():
    els = built(status=IndexStatus(running=True, done=120, total=2765, indexed=10))
    assert els[1] == {"type": "progress", "value": 120 / 2765, "label": "Indexing 120 of 2,765"}
    assert button_ids(els) == ["more", "radio"]  # no index buttons


def test_pending_and_failed():
    els = built(status=IndexStatus(indexed=2765, pending=12, failed=3))
    assert els[1] == {
        "type": "text",
        "text": "2,765 tracks indexed · 12 not indexed yet · 3 failed",
        "muted": True,
    }
    assert els[2]["children"] == [{"type": "button", "id": "index", "label": "Update index"}]


def test_up_to_date_still_offers_update_index():
    # nil rescans its folders by itself, so 0 pending may be out of date: an update is cheap.
    for pending in (0, None):
        els = built(status=IndexStatus(indexed=5, pending=pending))
        assert els[1]["text"] == "5 tracks indexed"
        assert els[2]["children"] == [{"type": "button", "id": "index", "label": "Update index"}]


def test_rebuild_needed_offers_only_rebuild():
    # index_status reports the whole library as pending when it can't read the index
    status = IndexStatus(problem="index schema is newer", rebuild_needed=True, pending=2765)
    els = built(status=status)
    assert button_ids(els) == ["rebuild", "more", "radio"]


def test_rebuild_button_only_when_needed():
    els = built(status=IndexStatus(problem="index schema is newer", rebuild_needed=True))
    assert els[2]["children"] == [{"type": "button", "id": "rebuild", "label": "Rebuild index"}]
    els = built(status=IndexStatus(indexed=3, problem="something else", rebuild_needed=False))
    assert "rebuild" not in button_ids(els)


@pytest.mark.parametrize(
    ("state", "label"),
    [
        ("downloading", "Downloading the sound model (once, about 620 MB)…"),
        ("loading", "Loading the sound model…"),
    ],
)
def test_model_progress(state, label):
    assert {"type": "progress", "value": None, "label": label} in built(model_state=state)


def test_model_failed_and_ready():
    failed = built(model_state="failed: no network")
    assert {"type": "text", "text": "Sound model failed: no network"} in failed
    assert not [e for e in built(model_state="ready") if e["type"] == "progress"]


def test_result_with_tracks():
    ts = tracks(2)
    els = built(result=Playlist(ts, 401.0, "rain — 2 tracks, 6:41"), prompt="rain", radio_on=True)
    assert els[3]["value"] == "rain"  # after the status line and Update index
    assert els[4]["children"][1]["value"] is True
    assert els[-3:-1] == [
        {"type": "heading", "text": "rain — 2 tracks, 6:41"},
        {"type": "tracks", "items": [t.item() for t in ts]},
    ]


def test_empty_result():
    els = built(result=Playlist([], 0.0, "zzz — 0 tracks, 0:00"))
    assert els[-3:-1] == [
        {"type": "heading", "text": "zzz — 0 tracks, 0:00"},
        {"type": "text", "text": "No matching tracks."},
    ]
    assert not [e for e in els if e["type"] == "tracks"]


def test_mcp_unavailable_and_message():
    els = built(mcp=MCP_NOT_RUNNING, message="Nothing is playing.")
    assert els[-2:] == [
        {"type": "text", "text": "Nothing is playing."},
        {"type": "text", "text": "MCP: not running", "muted": True},
    ]


def test_config_problem():
    els = built(problems=["config.toml: unknown key 'x'"])
    assert els[1] == {"type": "text", "text": "config.toml: unknown key 'x'", "muted": True}


# -- through the plugin ----------------------------------------------------------------------


def seed_index(plugin_dir, n=3):
    index = Index.open(plugin_dir / "data" / "index.sqlite", "fake")
    for i in range(n):
        add_track(index, f"E:\\Music\\A\\0{i}.flac", artist="A", album="Al", vector=unit(1, i, 0))
    index.close()


def settle(nil, pred, timeout=5.0):
    nil.wait_for(lambda: bool(nil.panels) and pred(nil.panels[-1]["elements"]), timeout)


def has_text(els, text):
    return any(e.get("text") == text for e in els)


def test_prompt_makes_a_playlist(make_running_plugin, plugin_dir):
    seed_index(plugin_dir)
    nil = make_running_plugin()
    nil.wait_for(lambda: nil.panels)
    nil.notify("panel.event", {"id": "prompt", "value": "rain"})
    settle(nil, lambda els: any(e["type"] == "tracks" for e in els))
    els = nil.panels[-1]["elements"]
    (shown,) = [e for e in els if e["type"] == "tracks"]
    assert len(shown["items"]) >= 1
    assert any(e["type"] == "heading" and e["text"].startswith("rain — ") for e in els)
    assert nil.element("prompt")["value"] == "rain"


def test_blank_prompt(running_plugin):
    running_plugin.notify("panel.event", {"id": "prompt", "value": "   "})
    settle(running_plugin, lambda els: has_text(els, "Type what you'd like to hear."))


def test_more_while_stopped(make_running_plugin, plugin_dir):
    seed_index(plugin_dir)
    nil = make_running_plugin()
    nil.notify("panel.event", {"id": "more"})
    settle(nil, lambda els: has_text(els, "Nothing is playing."))


def test_search_without_an_index_shows_the_problem(running_plugin):
    running_plugin.notify("panel.event", {"id": "prompt", "value": "rain"})
    settle(running_plugin, lambda els: has_text(els, NO_INDEX))


def test_radio_toggle(running_plugin):
    nil = running_plugin
    nil.wait_for(lambda: "radio" in button_ids(nil.panels[-1]["elements"]) if nil.panels else 0)
    assert nil.element("radio")["value"] is False
    nil.notify("panel.event", {"id": "radio", "value": True})
    nil.wait_for(lambda: nil.element("radio")["value"] is True)


def test_unknown_events_are_ignored(running_plugin):
    nil = running_plugin
    nil.wait_for(lambda: ("library.tracks", {}) in nil.calls)
    before = [c for c in nil.calls if c[0] != "panel.set"]
    nil.notify("panel.event", {"id": "nope"})
    nil.notify("panel.event", {})
    nil.notify("panel.event", {"id": "prompt", "value": "   "})
    settle(nil, lambda els: has_text(els, "Type what you'd like to hear."))
    assert [c for c in nil.calls if c[0] != "panel.set"] == before


def test_index_button_indexes(make_running_plugin, fake_nil, make_audio, tmp_path, monkeypatch):
    gate = threading.Event()
    embed = FakeEmbedder.embed_audio

    def held(self, clips):  # keeps the run going until the test has seen its progress
        assert gate.wait(10)
        return embed(self, clips)

    monkeypatch.setattr(FakeEmbedder, "embed_audio", held)
    files = [make_audio(f"{n}.flac", 3, freq=f, folder=tmp_path / "Music")
             for n, f in (("a", 300), ("b", 500))]  # fmt: skip
    fake_nil.tracks = [
        {"path": str(p), "title": None, "artist": None, "album": None, "duration_s": 3.0}
        for p in files
    ]
    nil = make_running_plugin()
    nil.wait_for(lambda: nil.panels and "index" in button_ids(nil.panels[-1]["elements"]))
    nil.notify("panel.event", {"id": "index"})
    settle(nil, lambda els: any(
        e["type"] == "progress" and e["label"].startswith("Indexing ") for e in els))  # fmt: skip
    assert "index" not in button_ids(nil.panels[-1]["elements"])
    gate.set()
    settle(nil, lambda els: has_text(els, "2 tracks indexed"))


def test_bad_config_shows_and_plugin_runs(make_running_plugin, plugin_dir):
    (plugin_dir / "config.toml").write_text('mcp_port = "x"\n', encoding="utf-8")
    nil = make_running_plugin()
    msg = "config.toml: mcp_port must be a whole number from 1 to 65535"
    settle(nil, lambda els: has_text(els, msg))
    nil.notify("panel.event", {"id": "prompt", "value": " "})
    settle(nil, lambda els: has_text(els, "Type what you'd like to hear."))


# -- the throttle ----------------------------------------------------------------------------


class StubDJ:
    def __init__(self):
        self.listeners = []
        self.radio = SimpleNamespace(on=False, message=None, on_change=lambda: None)
        self.model_state = "ready"
        self.calls = []

    def index_status(self):
        return IndexStatus(indexed=1)

    def build_playlist(self, *a):
        self.calls.append(("build_playlist", a))

    def more_like_playing(self):
        self.calls.append(("more_like_playing",))

    def set_radio(self, on):
        self.calls.append(("set_radio", on))

    def reindex(self, rebuild=False):
        self.calls.append(("reindex", rebuild))

    def refresh_library_if_stale(self):
        self.calls.append(("refresh_library_if_stale",))

    def _closed(self):
        return False


def drained(panel):
    """Wait until the panel has handled every event sent so far."""
    panel._executor.submit(lambda: None).result(5)


def test_throttle():
    sent = []
    panel = Panel(StubDJ(), lambda m, p: sent.append(p["panel"]), lambda: MCP_UP, 0.2)
    for i in range(50):
        panel.message = f"step {i}"
        panel.refresh()
    deadline = time.monotonic() + 3
    while len(sent) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.4)
    panel.close()
    assert 2 <= len(sent) <= 3
    assert {"type": "text", "text": "step 49"} in sent[-1]["elements"]


def test_forced_refresh_sends_at_once():
    sent = []
    panel = Panel(StubDJ(), lambda m, p: sent.append(p), lambda: MCP_UP, 60)
    panel.refresh()
    panel.refresh()
    assert len(sent) == 1
    panel.refresh(force=True)
    assert len(sent) == 2
    panel.close()


def test_a_radio_event_clears_the_panels_message():
    dj = StubDJ()
    panel = Panel(dj, lambda m, p: None, lambda: MCP_UP, 60)
    panel.message = "old"
    dj.radio.on_change()
    assert panel.message is None
    panel.close()


def sent_panel(panel):
    got = []
    panel.call = lambda m, p: got.append(p["panel"]["elements"])
    panel.refresh(force=True)
    return got[-1]


def test_radio_message_shows_unless_the_panel_has_its_own():
    dj = StubDJ()
    dj.radio.message = "Radio stopped: nothing indexed."
    panel = Panel(dj, None, lambda: MCP_UP, 60)
    assert has_text(sent_panel(panel), "Radio stopped: nothing indexed.")
    panel.message = "Nothing is playing."
    els = sent_panel(panel)
    assert has_text(els, "Nothing is playing.")
    assert not has_text(els, "Radio stopped: nothing indexed.")
    panel.close()


def test_unknown_ids_do_nothing():
    dj = StubDJ()
    panel = Panel(dj, lambda m, p: None, lambda: MCP_UP, 60)
    panel.on_event({"id": "nope", "value": "x"})
    panel.on_event({})
    drained(panel)
    assert dj.calls == [("refresh_library_if_stale",)] * 2
    assert (panel.prompt, panel.result, panel.message) == ("", None, None)
    panel.close()


def test_radio_value_must_be_a_true_boolean():
    dj = StubDJ()
    panel = Panel(dj, lambda m, p: None, lambda: MCP_UP, 60)
    panel.on_event({"id": "radio", "value": "false"})
    panel.on_event({"id": "radio", "value": True})
    drained(panel)
    radio = [c for c in dj.calls if c[0] == "set_radio"]
    assert radio == [("set_radio", False), ("set_radio", True)]
    panel.close()


def test_sends_never_raise(capsys):
    from nil_dj.host import MessageTooLarge

    dj = StubDJ()
    calls = []

    def call(method, params):
        calls.append(method)
        raise MessageTooLarge("too big")

    panel = Panel(dj, call, lambda: MCP_UP, 60)
    panel.refresh(force=True)  # call fails
    assert calls == ["panel.set"]

    def broken_mcp():
        raise RuntimeError("mcp")

    panel.mcp_status = broken_mcp
    panel.refresh(force=True)

    def broken_status():
        raise sqlite3.OperationalError("locked")

    dj.index_status = broken_status
    panel.mcp_status = lambda: MCP_UP
    panel.refresh(force=True)
    panel.on_event({"id": "more"})  # the forced refresh after an event doesn't raise either
    drained(panel)
    panel.close()
    err = capsys.readouterr().err
    assert "too big" in err and "mcp" in err and "locked" in err
    assert "Traceback" not in err


def test_the_panel_closes_before_the_dj_stops(running_plugin):
    from nil_dj import plugin

    running_plugin.wait_for(lambda: ("library.tracks", {}) in running_plugin.calls)
    owners = [type(getattr(fn, "__self__", None)).__name__ for fn in plugin._stop_callbacks]
    assert owners.index("Panel") < owners.index("DJ")


# -- events off the notification worker ------------------------------------------------------


def test_a_slow_search_does_not_hold_up_radio(make_running_plugin, fake_nil, plugin_dir,
                                              monkeypatch):  # fmt: skip
    seed_index(plugin_dir)
    paths = [f"E:\\Music\\A\\0{i}.flac" for i in range(3)]
    fake_nil.tracks = [{"path": p, "title": None, "artist": None, "album": None,
                        "duration_s": 240.0} for p in paths]  # fmt: skip
    fake_nil._known = {p.replace("\\", "/").lower() for p in paths}
    gate = threading.Event()
    searching = threading.Event()
    embed = FakeEmbedder.embed_text

    def held(self, text):  # the first run's model download, as far as Make can tell
        searching.set()
        assert gate.wait(10)
        return embed(self, text)

    monkeypatch.setattr(FakeEmbedder, "embed_text", held)
    nil = make_running_plugin()
    try:
        nil.notify("panel.event", {"id": "radio", "value": True})
        nil.wait_for(lambda: nil.panels and nil.element("radio")["value"] is True)
        nil.notify("panel.event", {"id": "prompt", "value": "rain"})
        assert searching.wait(5)
        nil.queue, nil.index, nil.state = [paths[0]], 0, "playing"
        nil.notify("player.changed",
                   {"state": "playing", "index": 0, "queue_len": 1, "path": paths[0]})  # fmt: skip
        nil.wait_for(lambda: any(m == "player.append" for m, _ in nil.calls), timeout=3)
    finally:
        gate.set()
    settle(nil, lambda els: any(e["type"] == "tracks" for e in els))


def test_an_unexpected_error_is_reported_and_the_panel_refreshed(capsys):
    dj = StubDJ()

    def broken():
        raise sqlite3.OperationalError("disk I/O error")

    dj.more_like_playing = broken
    sent = []
    panel = Panel(dj, lambda m, p: sent.append(p["panel"]["elements"]), lambda: MCP_UP, 60)
    panel.on_event({"id": "more"})
    drained(panel)
    panel.close()
    message = "Something went wrong: OperationalError. Details are in nil's plugin log."
    assert panel.message == message
    assert has_text(sent[-1], message)
    err = capsys.readouterr().err
    assert "Traceback" in err and "disk I/O error" in err


def test_close_does_not_wait_for_a_running_action():
    dj = StubDJ()
    gate = threading.Event()
    dj.more_like_playing = lambda: gate.wait(10)
    panel = Panel(dj, lambda m, p: None, lambda: MCP_UP, 60)
    panel.on_event({"id": "more"})
    t0 = time.monotonic()
    panel.close()
    assert time.monotonic() - t0 < 0.5
    panel.on_event({"id": "more"})  # after close: dropped, no error
    gate.set()


# -- nil's library, re-read ------------------------------------------------------------------


def test_new_music_in_nil_shows_as_pending_after_a_while(tmp_path, capsys):
    from fake_nil import FakeNil
    from test_dj import connect, library

    from nil_dj.config import Config
    from nil_dj.dj import DJ

    cfg = Config(index_path=tmp_path / "data" / "index.sqlite", folders=[])
    old = ["E:\\Music\\A\\01.flac", "E:\\Music\\A\\02.flac"]
    index = Index.open(cfg.index_path, "fake")
    for i, p in enumerate(old):
        add_track(index, p, artist="A", album="Al", vector=unit(1, i))
    index.close()
    fake = FakeNil(library(old))
    now = [1000.0]
    dj = DJ(cfg, connect(fake), FakeEmbedder, clock=lambda: now[0])
    sent = []
    panel = Panel(dj, lambda m, p: sent.append(p["panel"]["elements"]), lambda: MCP_UP, 0)

    def reads():
        return sum(1 for m, _ in fake.calls if m == "library.tracks")

    try:
        dj.refresh_library()
        assert dj.index_status().pending == 0

        fake.tracks = library([*old, "E:\\Music\\B\\01.flac"])  # nil rescanned by itself
        now[0] += 10
        panel.on_event({"id": "radio", "value": False})
        drained(panel)
        assert reads() == 1  # read under 30 s ago
        now[0] += 21
        panel.on_event({"id": "radio", "value": False})
        drained(panel)
        assert reads() == 2
        assert dj.index_status().pending == 1
        assert has_text(sent[-1], "2 tracks indexed · 1 not indexed yet")
        assert "index" in button_ids(sent[-1])

        fake.fail["library.tracks"] = (-32603, "library busy")
        now[0] += 31
        panel.on_event({"id": "prompt", "value": "  "})
        drained(panel)
        assert has_text(sent[-1], "Type what you'd like to hear.")  # the action still ran
        assert dj.index_status().pending == 1  # the last list is kept
        assert "can't read nil's library: library busy" in capsys.readouterr().err
    finally:
        panel.close()
        dj.stop()
        fake.close()
