"""The index file: the contract with nil (spec §6). SQLite in WAL mode so nil can read while the
indexer writes."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nil_dj.tags import TrackInfo

SCHEMA_VERSION = "1"
DIM = 512

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE tracks (
  path          TEXT PRIMARY KEY,
  size          INTEGER NOT NULL,
  mtime_ns      INTEGER NOT NULL,
  title         TEXT, artist TEXT, album_artist TEXT, album TEXT, genre TEXT,
  year          INTEGER, disc INTEGER, number INTEGER,
  duration_s    REAL NOT NULL,
  codec         TEXT NOT NULL,
  sample_rate   INTEGER NOT NULL,
  bits          INTEGER,
  bpm           REAL,
  lufs          REAL,
  embedding     BLOB NOT NULL,
  indexed_at    INTEGER NOT NULL
);

CREATE TABLE index_errors (path TEXT PRIMARY KEY, error TEXT NOT NULL, at INTEGER NOT NULL);
"""

_INFO_COLUMNS = (
    "title", "artist", "album_artist", "album", "genre", "year", "disc", "number",
    "duration_s", "codec", "sample_rate", "bits",
)  # fmt: skip
_ROW_COLUMNS = ("path", "size", "mtime_ns", *_INFO_COLUMNS, "bpm", "lufs")


class IndexFormatError(Exception):
    pass


@dataclass(frozen=True)
class Row:
    path: str
    size: int
    mtime_ns: int
    info: TrackInfo
    bpm: float | None
    lufs: float | None


class Index:
    def __init__(self, db: sqlite3.Connection):
        self._db = db

    @classmethod
    def open(cls, path: Path, model_id: str) -> Index:
        path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(path)
        try:
            db.execute("PRAGMA journal_mode=WAL")
            exists = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta'"
            ).fetchone()
            if not exists:
                with db:
                    db.executescript(SCHEMA)
                    db.executemany(
                        "INSERT INTO meta (key, value) VALUES (?, ?)",
                        [("schema_version", SCHEMA_VERSION), ("model_id", model_id), ("dim", "512")],
                    )
            else:
                meta = dict(db.execute("SELECT key, value FROM meta"))
                version = meta.get("schema_version", "")
                major = version.split(".")[0]
                if not major.isdigit() or int(major) > int(SCHEMA_VERSION):
                    raise IndexFormatError(
                        f"index schema {version!r} is newer than this nil-dj understands"
                    )
                if meta.get("model_id") != model_id:
                    raise IndexFormatError(
                        f"index built with {meta.get('model_id')}; run nil-dj index --rebuild"
                    )
        except BaseException:
            db.close()
            raise
        return cls(db)

    def close(self) -> None:
        self._db.close()

    @staticmethod
    def remove(path: Path) -> None:
        for p in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
            p.unlink(missing_ok=True)

    def get_meta(self, key: str) -> str | None:
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def stamps(self) -> dict[str, tuple[int, int]]:
        return {p: (s, m) for p, s, m in self._db.execute("SELECT path, size, mtime_ns FROM tracks")}

    def upsert(
        self,
        path: str,
        size: int,
        mtime_ns: int,
        info: TrackInfo,
        bpm: float | None,
        lufs: float | None,
        embedding: np.ndarray,
    ) -> None:
        blob = np.asarray(embedding).astype("<f4").tobytes()
        if len(blob) != DIM * 4:
            raise ValueError(f"embedding must have {DIM} values")
        values = (
            path, size, mtime_ns, *(getattr(info, c) for c in _INFO_COLUMNS), bpm, lufs, blob,
            int(time.time()),
        )  # fmt: skip
        columns = (*_ROW_COLUMNS, "embedding", "indexed_at")
        with self._db:
            self._db.execute(
                f"INSERT OR REPLACE INTO tracks ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' * len(columns))})",
                values,
            )
            self._db.execute("DELETE FROM index_errors WHERE path = ?", (path,))

    def record_error(self, path: str, error: str) -> None:
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO index_errors (path, error, at) VALUES (?, ?, ?)",
                (path, error, int(time.time())),
            )

    def errors(self) -> list[tuple[str, str]]:
        return list(self._db.execute("SELECT path, error FROM index_errors ORDER BY path"))

    def delete(self, paths: Iterable[str]) -> None:
        params = [(p,) for p in paths]
        with self._db:
            self._db.executemany("DELETE FROM tracks WHERE path = ?", params)
            self._db.executemany("DELETE FROM index_errors WHERE path = ?", params)

    def count(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]

    def load(self, where: str = "", params: Sequence = ()) -> tuple[list[Row], np.ndarray]:
        """Rows ordered by path and their (n, 512) float32 embedding matrix."""
        sql = f"SELECT {', '.join(_ROW_COLUMNS)}, embedding FROM tracks"
        if where:
            sql += f" WHERE {where}"
        sql += " ORDER BY path"
        rows, blobs = [], []
        n_info = len(_INFO_COLUMNS)
        for r in self._db.execute(sql, tuple(params)):
            info = TrackInfo(*r[3 : 3 + n_info])
            rows.append(Row(r[0], r[1], r[2], info, r[3 + n_info], r[4 + n_info]))
            blobs.append(r[-1])
        matrix = np.frombuffer(b"".join(blobs), "<f4").reshape(len(blobs), DIM).astype(np.float32)
        return rows, matrix
