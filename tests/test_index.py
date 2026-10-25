import sqlite3
from dataclasses import replace

import numpy as np
import pytest

from nil_dj.index import Index, IndexFormatError
from nil_dj.tags import TrackInfo

SPEC_COLUMNS = [
    "path", "size", "mtime_ns", "title", "artist", "album_artist", "album", "genre", "year",
    "disc", "number", "duration_s", "codec", "sample_rate", "bits", "bpm", "lufs", "embedding",
    "indexed_at",
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
    assert meta == {"schema_version": "1", "model_id": "m", "dim": "512"}


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
    with pytest.raises(IndexFormatError, match="index built with old-model; run nil-dj index --rebuild"):
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
