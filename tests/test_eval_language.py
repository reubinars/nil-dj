import sys
from pathlib import Path

import pytest
from fakes import add_track, unit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

import language as ev
import radio_language as rl

from nil_dj.config import Weights
from nil_dj.index import Index


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("ja", (frozenset({"ja"}), False)),
        (" EN ", (frozenset({"en"}), False)),
        ("jw", (frozenset({"id"}), False)),
        ("ms", (frozenset({"id"}), False)),
        ("su", (frozenset({"id"}), False)),
        ("instrumental? (la)", (frozenset({"instrumental"}), False)),
        ("instrumental", (frozenset({"instrumental"}), False)),
        ("mixed ja en", (frozenset({"ja", "en"}), False)),
        ("mixed", (frozenset(), True)),
        ("?", None),
        ("", None),
        (None, None),
        ("klingon", None),
    ],
)
def test_parse_label(text, label):
    assert ev.parse_label(text) == label


def test_is_correct_rules():
    assert ev.is_correct(ev.parse_label("ja"), "ja", None)
    assert not ev.is_correct(ev.parse_label("ja"), "en", None)
    assert ev.is_correct(ev.parse_label("jw"), "id", None)
    assert ev.is_correct(ev.parse_label("instrumental? (la)"), "instrumental", None)
    assert ev.is_correct(ev.parse_label("mixed ja en"), "en", None)
    assert not ev.is_correct(ev.parse_label("mixed ja en"), "fr", None)
    assert ev.is_correct(ev.parse_label("mixed"), "ja", "en")
    assert not ev.is_correct(ev.parse_label("mixed"), "ja", None)
    assert not ev.is_correct(ev.parse_label("ja"), None, None)


def test_expected_label_falls_back_to_the_guess():
    assert ev.expected_label("ja", "") == "ja"
    assert ev.expected_label("ja", "mixed") == "mixed"
    assert ev.expected_label(None, None) == ""


def test_read_labels_with_bom(tmp_path):
    f = tmp_path / "l.csv"
    f.write_text(
        "artist,title,whisper_guess,confidence,no_speech,your_label\n"
        "A,T,ja,0.9,0.1,\nB,U,en,0.9,0.1,fr\n",
        encoding="utf-8-sig",
    )
    assert ev.read_labels(f) == [("A", "T", "ja"), ("B", "U", "fr")]


def test_evaluate_counts(tmp_path):
    index = Index.open(tmp_path / "i.sqlite", "fake")
    add_track(index, "a", artist="Ann", album="x", title="One", vector=unit(1), language="ja")
    add_track(index, "b", artist="Bo", album="y", title="Two", vector=unit(1, 1), language="en")
    labels = [("ANN", "one", "ja"), ("Bo", "Two", "fr"), ("Zed", "None", "en"), ("Ann", "One", "?")]
    r = ev.evaluate(index, labels)
    assert (r["correct"], r["total"], r["skipped"], r["missing"]) == (1, 2, 1, 1)
    assert r["confusion"][("fr", "en")] == 1
    index.close()


def test_radio_simulation_runs(tmp_path):
    path = tmp_path / "i.sqlite"
    index = Index.open(path, "fake")
    for n in range(120):
        add_track(index, f"E:\\M\\{n}.flac", artist=f"a{n}", album=f"b{n}", duration_s=240,
                  vector=unit(1, 0.01 * n), language="ja" if n % 2 else "en")  # fmt: skip
    index.close()
    out = rl.evaluate(lambda: Index.open(path, None), Weights(), 1, 1)
    assert set(out) == {"ja", "en"}
    n, res = out["ja"]
    assert n == 60 and res["with"][2] > 0
    assert res["with"][0] >= res["without"][0]


def test_radio_simulation_survives_missing_durations(tmp_path):
    path = tmp_path / "i.sqlite"
    index = Index.open(path, "fake")
    for n in range(120):
        dur = 0 if n == 0 else 240
        add_track(index, f"E:\\M\\{n}.flac", artist=f"a{n}", album=f"b{n}", duration_s=dur,
                  vector=unit(1, 0.01 * n), language="en")  # fmt: skip
    index.close()
    assert rl._seconds(None) == rl._seconds(0) == 240.0
    played = rl.simulate(lambda: Index.open(path, None), Weights(), "E:\\M\\0.flac", 1)
    assert len(played) > 1
