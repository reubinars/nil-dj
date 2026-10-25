import time
from types import SimpleNamespace

import pytest
from fake_nil import validate_panel
from fakes import add_track, unit

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


def test_up_to_date_has_no_index_buttons():
    els = built(status=IndexStatus(indexed=5, pending=0))
    assert els[1]["text"] == "5 tracks indexed"
    assert [e["type"] for e in els[2:]] == ["input", "row", "text"]


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
    assert els[2]["value"] == "rain"
    assert els[3]["children"][1]["value"] is True
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
    nil.wait_for(lambda: nil.panels)
    nil.notify("panel.event", {"id": "nope"})
    nil.notify("panel.event", {})
    nil.notify("panel.event", {"id": "prompt", "value": "   "})
    settle(nil, lambda els: has_text(els, "Type what you'd like to hear."))


def test_index_button_indexes(make_running_plugin, fake_nil, make_audio, tmp_path):
    files = [make_audio(f"{n}.flac", 3, freq=f, folder=tmp_path / "Music")
             for n, f in (("a", 300), ("b", 500))]  # fmt: skip
    fake_nil.tracks = [
        {"path": str(p), "title": None, "artist": None, "album": None, "duration_s": 3.0}
        for p in files
    ]
    nil = make_running_plugin()
    nil.wait_for(lambda: nil.panels and "index" in button_ids(nil.panels[-1]["elements"]))
    nil.notify("panel.event", {"id": "index"})
    settle(nil, lambda els: has_text(els, "2 tracks indexed"))
    assert any(e["type"] == "progress" for p in nil.panels for e in p["elements"])


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

    def index_status(self):
        return IndexStatus(indexed=1)

    def _closed(self):
        return False


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
