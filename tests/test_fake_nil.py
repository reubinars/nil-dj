import pytest
from fake_nil import FakeNil, validate_panel

from nil_dj.host import NilError


def _track(path):
    return {"path": path, "title": "t", "artist": "a", "album": "al", "duration_s": 1.0}


def _fake():
    f = FakeNil([_track("E:\\Music\\a.flac"), _track("E:\\Music\\b.flac")])
    f.granted = ["playback", "library", "playlists"]
    return f


def _nested_rows(depth):
    row = {"type": "row", "children": []}
    for _ in range(depth - 1):
        row = {"type": "row", "children": [row]}
    return row


def test_play_unknown_path_lists_every_unknown_path():
    f = _fake()
    with pytest.raises(NilError) as e:
        f._on_request("player.play", {"paths": ["E:/music/A.flac", "x", "y"]})
    assert e.value.code == -32002
    assert e.value.data == {"paths": ["x", "y"]}


def test_play_empty_is_invalid_and_append_starts_when_stopped():
    f = _fake()
    with pytest.raises(NilError) as e:
        f._on_request("player.play", {"paths": []})
    assert e.value.code == -32602
    f._on_request("player.append", {"paths": ["E:\\Music\\b.flac"]})
    assert (f.state, f.index, f.queue) == ("playing", 0, ["E:\\Music\\b.flac"])


def test_save_sanitises_name():
    f = _fake()
    result = f._on_request("playlists.save", {"name": 'a<b>:"/\\|?*', "paths": []})
    assert result == {"name": "a_b" + "_" * 8}


@pytest.mark.parametrize(
    "panel",
    [
        {"elements": [{"type": "heading"}]},
        {"elements": [{"type": "button", "id": "a", "label": "x"}] * 2},
        {"elements": [{"type": "progress"}]},
        {"elements": [{"type": "tracks", "items": [{"title": "x"}]}]},
        {"elements": [_nested_rows(5)]},
    ],
)
def test_bad_panels_are_rejected(panel):
    with pytest.raises(ValueError):
        validate_panel(panel)


def test_good_panel_and_unknown_type_pass():
    validate_panel(
        {
            "elements": [
                _nested_rows(4),
                {"type": "row", "children": [{"type": "button", "id": "a", "label": "x"}]},
                {"type": "progress", "value": None},
                {"type": "future"},
            ]
        }
    )
