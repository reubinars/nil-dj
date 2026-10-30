import pytest
from fake_nil import FakeNil, validate_panel

from nil_dj.host import NilError


def _track(path):
    return {"path": path, "title": "t", "artist": "a", "album": "al", "duration_s": 1.0}


A, B, C, D = (f"E:\\Music\\{n}.flac" for n in "abcd")


def _fake():
    f = FakeNil([_track(A), _track(B), _track(C), _track(D)])
    f.granted = ["playback", "library", "playlists"]
    return f


def _playing():
    f = _fake()
    f.queue, f.index, f.state = [A, B, C, D], 0, "playing"
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


def test_remove_drops_the_named_tracks():
    f = _playing()
    assert f._on_request("player.remove", {"tracks": [{"index": 2, "path": C}]}) == {}
    assert f.queue == [A, B, D]


def test_move_puts_tracks_in_front_of_to_in_the_order_given():
    f = _playing()
    tracks = [{"index": 3, "path": D}, {"index": 2, "path": C}]
    f._on_request("player.move", {"tracks": tracks, "to": 1})
    assert f.queue == [A, D, C, B]
    f = _playing()
    f._on_request("player.move", {"tracks": [{"index": 1, "path": B}], "to": 4})
    assert f.queue == [A, C, D, B]


def test_clear_upcoming_counts_and_respects_next_starting():
    f = _playing()
    f.next_starting = True
    assert f._on_request("player.clear_upcoming", {}) == {"removed": 2}
    assert f.queue == [A, B]


@pytest.mark.parametrize(
    ("tracks", "reason"),
    [
        ([{"index": 1, "path": C}], "changed"),
        ([{"index": 9, "path": C}], "changed"),
        ([{"index": 0, "path": A}], "started"),
    ],
)
def test_queue_changed_reasons(tracks, reason):
    f = _playing()
    with pytest.raises(NilError) as e:
        f._on_request("player.remove", {"tracks": tracks})
    assert (e.value.code, e.value.data) == (-32004, {"reason": reason})
    assert f.queue == [A, B, C, D]


def test_next_starting_locks_the_next_track():
    f = _playing()
    f.next_starting = True
    with pytest.raises(NilError) as e:
        f._on_request("player.remove", {"tracks": [{"index": 1, "path": B}]})
    assert (e.value.code, e.value.data) == (-32004, {"reason": "started"})
    with pytest.raises(NilError) as e:
        f._on_request("player.move", {"tracks": [{"index": 3, "path": D}], "to": 1})
    assert e.value.data == {"reason": "started"}
    f._on_request("player.remove", {"tracks": [{"index": 2, "path": C}]})
    assert f.queue == [A, B, D]


def test_stopped_is_stopped():
    f = _fake()
    f.queue = [A, B]
    for method, params in (
        ("player.remove", {"tracks": [{"index": 1, "path": B}]}),
        ("player.move", {"tracks": [{"index": 1, "path": B}], "to": 0}),
        ("player.clear_upcoming", {}),
    ):
        with pytest.raises(NilError) as e:
            f._on_request(method, params)
        assert (e.value.code, e.value.data) == (-32004, {"reason": "stopped"})


@pytest.mark.parametrize(
    ("method", "params"),
    [
        ("player.remove", {"tracks": []}),
        ("player.remove", {}),
        ("player.remove", {"tracks": [{"index": 1, "path": B}, {"index": 1, "path": B}]}),
        ("player.move", {"tracks": [{"index": 1, "path": B}], "to": 5}),
        ("player.move", {"tracks": [{"index": 1, "path": B}], "to": "x"}),
        ("player.move", {"tracks": [{"index": 1, "path": B}]}),
    ],
)
def test_bad_queue_params(method, params):
    f = _playing()
    with pytest.raises(NilError) as e:
        f._on_request(method, params)
    assert e.value.code == -32602
