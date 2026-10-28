import json
import sqlite3
from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest

from nil_dj import index as index_mod
from nil_dj.index import Final, Index, IndexFormatError
from nil_dj.tags import TrackInfo

SPEC_COLUMNS = [
    "path", "size", "mtime_ns", "title", "artist", "album_artist", "album", "genre", "year",
    "disc", "number", "duration_s", "codec", "sample_rate", "bits", "bpm", "lufs", "embedding",
    "indexed_at",
    "language_raw", "language", "language_alt", "language_conf", "vocal_score", "language_source",
]  # fmt: skip


BASE = TrackInfo(
    title="t", artist="a", album_artist=None, album="al", genre=None, year=2001, disc=1,
    number=1, duration_s=200.0, codec="flac", sample_rate=44100, bits=16,
)  # fmt: skip


def info(**kw) -> TrackInfo:
    return replace(BASE, **kw)


def vec(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def idx(tmp_path):
    i = Index.open(tmp_path / "sub" / "index.sqlite", "m")
    yield i
    i.close()


def test_schema_matches_spec(idx, tmp_path):
    db = sqlite3.connect(tmp_path / "sub" / "index.sqlite")
    assert [r[1] for r in db.execute("PRAGMA table_info(tracks)")] == SPEC_COLUMNS
    assert [r[1] for r in db.execute("PRAGMA table_info(index_errors)")] == ["path", "error", "at"]
    assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    meta = dict(db.execute("SELECT key, value FROM meta"))
    assert meta == {"schema_version": "1.1", "model_id": "m", "dim": "512"}


def test_upsert_roundtrip(idx):
    vs = {f"E:\\Music\\{c}.flac": vec(i) for i, c in enumerate("cab")}
    for p, v in vs.items():
        idx.upsert(p, 10, 20, info(title=p), 120.0, -9.5, v)
    rows, m = idx.load()
    assert [r.path for r in rows] == sorted(vs)
    assert m.shape == (3, 512) and m.dtype == np.float32
    for r, row_vec in zip(rows, m, strict=True):
        assert np.array_equal(row_vec, vs[r.path])
        assert (r.size, r.mtime_ns, r.bpm, r.lufs, r.info.title) == (10, 20, 120.0, -9.5, r.path)
    assert idx.stamps() == {p: (10, 20) for p in vs}


def test_upsert_replaces_and_clears_error(idx):
    idx.record_error("p", "bad")
    assert idx.errors() == [("p", "bad")]
    idx.upsert("p", 1, 1, info(), None, None, vec(0))
    idx.upsert("p", 2, 2, info(), None, None, vec(1))
    assert idx.errors() == []
    assert idx.count() == 1
    assert idx.stamps() == {"p": (2, 2)}


def test_load_where(idx):
    idx.upsert("old", 1, 1, info(year=1990), None, None, vec(0))
    idx.upsert("new", 1, 1, info(year=2010), None, None, vec(1))
    rows, m = idx.load("year >= ?", [2000])
    assert [r.path for r in rows] == ["new"] and m.shape == (1, 512)


def test_load_empty(idx):
    rows, m = idx.load()
    assert rows == [] and m.shape == (0, 512)


def test_model_mismatch_raises(tmp_path):
    Index.open(tmp_path / "i.sqlite", "old-model").close()
    with pytest.raises(
        IndexFormatError, match="index built with old-model; run nil-dj index --rebuild"
    ):
        Index.open(tmp_path / "i.sqlite", "new-model")


def test_open_without_model_check(tmp_path):
    Index.open(tmp_path / "i.sqlite", "some-model").close()
    i = Index.open(tmp_path / "i.sqlite", None)
    assert i.get_meta("model_id") == "some-model"
    i.close()
    with pytest.raises(IndexFormatError):
        Index.open(tmp_path / "new.sqlite", None)


def test_newer_schema_raises(tmp_path):
    Index.open(tmp_path / "i.sqlite", "m").close()
    db = sqlite3.connect(tmp_path / "i.sqlite")
    db.execute("UPDATE meta SET value = '2' WHERE key = 'schema_version'")
    db.commit()
    db.close()
    with pytest.raises(IndexFormatError):
        Index.open(tmp_path / "i.sqlite", "m")


def test_delete_removes_rows_and_errors(idx):
    idx.upsert("a", 1, 1, info(), None, None, vec(0))
    idx.upsert("b", 1, 1, info(), None, None, vec(1))
    idx.record_error("c", "bad")
    idx.delete(["a", "c"])
    assert list(idx.stamps()) == ["b"]
    assert idx.errors() == []


def test_meta_get_set(idx):
    assert idx.get_meta("precision") is None
    idx.set_meta("precision", "fp16")
    assert idx.get_meta("precision") == "fp16"


def test_remove_deletes_db_files(tmp_path):
    p = tmp_path / "i.sqlite"
    i = Index.open(p, "m")
    i.close()
    Index.remove(p)
    assert list(tmp_path.iterdir()) == []
    Index.remove(p)  # missing files are fine


def test_read_only_open_never_writes_schema(tmp_path):
    path = tmp_path / "other.sqlite"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE notes (x)")
    db.commit()
    db.close()
    with pytest.raises(IndexFormatError, match="is not a nil-dj index"):
        Index.open(path, None)
    db = sqlite3.connect(path)
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    db.close()
    assert tables == {"notes"}


ROW_PATH = "E:\\Music\\a.flac"


def make_v1_index(p):
    v1 = (
        index_mod.SCHEMA.split("indexed_at    INTEGER NOT NULL,")[0]
        + "indexed_at INTEGER NOT NULL\n);"
    )
    v1 += "\nCREATE TABLE index_errors (path TEXT PRIMARY KEY, error TEXT NOT NULL, at INTEGER NOT NULL);"
    v1 = v1.replace("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);\n", "", 1)
    db = sqlite3.connect(p)
    db.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    db.executescript(v1)
    db.executemany(
        "INSERT INTO meta VALUES (?, ?)",
        [("schema_version", "1"), ("model_id", "fake"), ("dim", "512")],
    )
    db.execute(
        "INSERT INTO tracks (path, size, mtime_ns, title, duration_s, codec, sample_rate, "
        "embedding, indexed_at) VALUES (?, 1, 1, 't', 240.0, 'flac', 44100, ?, 1)",
        (ROW_PATH, vec(0).astype("<f4").tobytes()),
    )
    db.commit()
    db.close()


def test_version_1_index_migrates_in_place(tmp_path):
    p = tmp_path / "i.sqlite"
    make_v1_index(p)
    with closing(Index.open(p, "fake")) as ix:
        assert ix.get_meta("schema_version") == "1.1"
        assert ix.language_work() == [(ROW_PATH, 240.0)]
        rows, m = ix.load()
        assert rows[0].language is None and m.shape == (1, 512)
        assert np.array_equal(m[0], vec(0))
    with closing(Index.open(p, None)) as ix:  # reopening is a no-op
        assert ix.get_meta("schema_version") == "1.1"


def test_old_check_accepts_1_1(tmp_path, monkeypatch):
    p = tmp_path / "i.sqlite"
    Index.open(p, "m").close()
    monkeypatch.setattr(index_mod, "SCHEMA_VERSION", "1")
    Index.open(p, "m").close()  # an older nil-dj: major-only check, no error


def test_upsert_then_raw_then_final(idx):
    idx.upsert("a", 1, 1, info(), None, None, vec(0))
    assert idx.language_work() == [("a", 200.0)]
    idx.set_language_raw("a", '{"windows":[]}')
    assert idx.language_work() == []
    idx.set_final({"a": Final("ja", "en", 0.9, 0.8, "whisper")})
    rows, _ = idx.load()
    assert (rows[0].language, rows[0].language_alt) == ("ja", "en")
    idx.upsert("a", 2, 2, info(), None, None, vec(0))  # file changed
    rows, _ = idx.load()
    assert (rows[0].language, rows[0].language_alt) == (None, None)
    assert idx.language_work() == [("a", 200.0)]
    idx.upsert("b", 1, 1, info(), None, None, vec(1), language_raw='{"windows":[]}')
    assert [p for p, _ in idx.language_work()] == ["a"]


def test_model_change_clears_raw_only(idx):
    idx.upsert("a", 1, 1, info(), None, None, vec(0))
    idx.upsert("b", 1, 1, info(), None, None, vec(1))
    idx.use_language_model("w1")
    for p in "ab":
        idx.set_language_raw(p, '{"windows":[]}')
    idx.set_final({"a": Final("ja", None, 0.9, 0.8, "whisper")})
    assert idx.languages_pending("w1") == 0
    assert idx.languages_pending("w2") == 2  # before use_language_model
    idx.use_language_model("w2")
    assert [p for p, _ in idx.language_work()] == ["a", "b"]
    assert idx.languages_pending("w2") == 2
    assert idx.get_meta("language_model") == "w2"
    rows, m = idx.load()
    assert rows[0].language is None and m.shape == (2, 512)
    assert idx.stamps() == {"a": (1, 1), "b": (1, 1)}
    db = idx._db
    assert (
        db.execute("SELECT language_conf, vocal_score, language_source FROM tracks").fetchall()[0]
        == (None,) * 3
    )


def test_has_languages_ignores_unknown(idx):
    assert not idx.has_languages()
    idx.upsert("a", 1, 1, info(), None, None, vec(0))
    assert not idx.has_languages()
    idx.set_final({"a": Final("unknown", None, 0.0, 0.0, "whisper")})
    assert not idx.has_languages()
    idx.set_final({"a": Final("instrumental", None, 0.9, 0.0, "whisper")})
    assert idx.has_languages()


def test_language_rows_counts_unsure_failed(idx):
    for i, (artist, title) in enumerate([("x", "t0"), ("y", "t1"), ("z", "t2"), ("w", "t3")]):
        idx.upsert(f"p{i}", 1, 1, info(artist=artist, title=title), None, None, vec(i))
    idx.set_language_raw("p0", '{"windows":[]}')
    idx.set_language_raw("p3", json.dumps({"failed": "decode error"}))
    assert [(r.path, r.artist, r.raw) for r in idx.language_rows()][:2] == [
        ("p0", "x", '{"windows":[]}'),
        ("p1", "y", None),
    ]
    idx.set_final(
        {
            "p0": Final("en", None, 0.4, 0.9, "whisper"),
            "p1": Final("en", None, 0.95, 0.9, "whisper"),
            "p2": Final("instrumental", None, 0.3, 0.0, "whisper"),
            "p3": Final("unknown", None, 0.2, 0.0, "whisper"),
        }
    )
    assert idx.language_counts() == [("en", 2), ("instrumental", 1), ("unknown", 1)]
    assert idx.unsure_tracks(0.6) == [("p0", "x", "t0", "en", 0.4, "whisper")]
    assert idx.failed_languages() == [("p3", "decode error")]
    idx.upsert("q", 1, 1, info(), None, None, vec(9))
    assert ("not detected", 1) in idx.language_counts()
