"""The index file (spec §6): private to nil-dj, not shared with nil. SQLite in WAL mode so search
can read while the indexer writes."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nil_dj.tags import TrackInfo

SCHEMA_VERSION = "1.1"
# sqlite3.OperationalError on open (locked, can't open): nothing is wrong with the file.
INDEX_BUSY = "The index is busy (another nil-dj is using it). Try again in a moment."
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
  indexed_at    INTEGER NOT NULL,
  language_raw    TEXT,
  language        TEXT,
  language_alt    TEXT,
  language_conf   REAL,
  vocal_score     REAL,
  language_source TEXT
);

CREATE TABLE index_errors (path TEXT PRIMARY KEY, error TEXT NOT NULL, at INTEGER NOT NULL);
"""

# Added to `tracks` by the 1.0 -> 1.1 migration, in the same order as in SCHEMA.
_LANGUAGE_COLUMNS = (
    ("language_raw", "TEXT"), ("language", "TEXT"), ("language_alt", "TEXT"),
    ("language_conf", "REAL"), ("vocal_score", "REAL"), ("language_source", "TEXT"),
)  # fmt: skip
_FINAL_COLUMNS = ("language", "language_alt", "language_conf", "vocal_score", "language_source")
# These mirror languages.INSTRUMENTAL / languages.UNKNOWN (index.py must not import languages).
_NOT_A_LANGUAGE = "('instrumental', 'unknown')"

_INFO_COLUMNS = (
    "title", "artist", "album_artist", "album", "genre", "year", "disc", "number",
    "duration_s", "codec", "sample_rate", "bits",
)  # fmt: skip
_ROW_COLUMNS = ("path", "size", "mtime_ns", *_INFO_COLUMNS, "bpm", "lufs")


def _casefold(s: str | None) -> str | None:
    return s.casefold() if isinstance(s, str) else s


def path_key(path: str) -> str:
    """What makes two paths the same track: nil's paths are Windows paths, where case and the
    kind of slash don't matter. The index stores paths as given; compare them by this."""
    return path.replace("\\", "/").casefold()


def _path_key(path: str | None) -> str | None:
    return path_key(path) if isinstance(path, str) else path


class IndexFormatError(Exception):
    pass


class IndexProblem(ValueError):
    """The index can't be used; `rebuild_needed` when it comes from an `IndexFormatError`."""

    def __init__(self, message: str, rebuild_needed: bool = False):
        super().__init__(message)
        self.rebuild_needed = rebuild_needed


@dataclass(frozen=True)
class Row:
    path: str
    size: int
    mtime_ns: int
    info: TrackInfo
    bpm: float | None
    lufs: float | None
    language: str | None = None
    language_alt: str | None = None


@dataclass(frozen=True)
class LangRow:
    path: str
    artist: str | None
    album_artist: str | None
    album: str | None
    raw: str | None


@dataclass(frozen=True)
class Final:
    language: str | None
    alt: str | None
    conf: float | None
    vocal_score: float | None
    source: str | None


class Index:
    def __init__(self, db: sqlite3.Connection):
        self._db = db

    @classmethod
    def open(cls, path: Path, model_id: str | None) -> Index:
        """Open or create the index. `model_id=None` opens an existing index for reading
        without checking which model built it, and never writes to a file that isn't one.
        Opening a 1.0 index migrates it in place, even on a read-only open: that is a write, so a
        concurrent writer surfaces as the usual sqlite3.OperationalError (INDEX_BUSY)."""
        if model_id is None and not path.exists():
            raise IndexFormatError(f"no index at {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(path)
        # SQLite's lower() only folds ASCII; filters need "Björk" == "björk".
        db.create_function("casefold", 1, _casefold, deterministic=True)
        db.create_function("path_key", 1, _path_key, deterministic=True)
        try:
            exists = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta'"
            ).fetchone()
            if not exists and model_id is None:
                raise IndexFormatError(f"{path} is not a nil-dj index")
            db.execute("PRAGMA journal_mode=WAL")
            if not exists:
                with db:
                    db.executescript(SCHEMA)
                    db.executemany(
                        "INSERT INTO meta (key, value) VALUES (?, ?)",
                        [
                            ("schema_version", SCHEMA_VERSION),
                            ("model_id", model_id),
                            ("dim", "512"),
                        ],
                    )
            else:
                meta = dict(db.execute("SELECT key, value FROM meta"))
                version = meta.get("schema_version", "")
                major = version.split(".")[0]
                if not major.isdigit() or int(major) > int(SCHEMA_VERSION.split(".")[0]):
                    raise IndexFormatError(
                        f"index schema {version!r} is newer than this nil-dj understands"
                    )
                if model_id is not None and meta.get("model_id") != model_id:
                    raise IndexFormatError(
                        f"index built with {meta.get('model_id')}; run nil-dj index --rebuild"
                    )
                have = {r[1] for r in db.execute("PRAGMA table_info(tracks)")}
                if any(name not in have for name, _ in _LANGUAGE_COLUMNS):
                    # Python's sqlite3 autocommits DDL, so `with db:` would not make this atomic.
                    db.execute("BEGIN IMMEDIATE")
                    try:
                        # Re-read under the write lock: another process may have migrated.
                        have = {r[1] for r in db.execute("PRAGMA table_info(tracks)")}
                        for name, kind in _LANGUAGE_COLUMNS:
                            if name not in have:
                                db.execute(f"ALTER TABLE tracks ADD COLUMN {name} {kind}")
                        db.execute(
                            "UPDATE meta SET value = ? WHERE key = 'schema_version'",
                            (SCHEMA_VERSION,),
                        )
                        db.commit()
                    except BaseException:
                        db.rollback()
                        raise
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
        return {
            p: (s, m) for p, s, m in self._db.execute("SELECT path, size, mtime_ns FROM tracks")
        }

    def upsert(
        self,
        path: str,
        size: int,
        mtime_ns: int,
        info: TrackInfo,
        bpm: float | None,
        lufs: float | None,
        embedding: np.ndarray,
        language_raw: str | None = None,
    ) -> None:
        blob = np.asarray(embedding).astype("<f4").tobytes()
        if len(blob) != DIM * 4:
            raise ValueError(f"embedding must have {DIM} values")
        values = (
            path, size, mtime_ns, *(getattr(info, c) for c in _INFO_COLUMNS), bpm, lufs, blob,
            int(time.time()), language_raw,
        )  # fmt: skip
        columns = (*_ROW_COLUMNS, "embedding", "indexed_at", "language_raw")
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

    def load_paths(self, paths: Iterable[str]) -> tuple[list[Row], np.ndarray]:
        """`load` for these paths, matched by `path_key`, so nil's spelling finds the row."""
        keys = sorted({path_key(p) for p in paths})
        return self.load(f"path_key(path) IN ({', '.join('?' * len(keys))})", keys)

    def load(self, where: str = "", params: Sequence = ()) -> tuple[list[Row], np.ndarray]:
        """Rows ordered by path and their (n, 512) float32 embedding matrix."""
        sql = f"SELECT {', '.join(_ROW_COLUMNS)}, language, language_alt, embedding FROM tracks"
        if where:
            sql += f" WHERE {where}"
        sql += " ORDER BY path"
        rows, blobs = [], []
        n_info = len(_INFO_COLUMNS)
        for r in self._db.execute(sql, tuple(params)):
            info = TrackInfo(*r[3 : 3 + n_info])
            rows.append(
                Row(
                    r[0],
                    r[1],
                    r[2],
                    info,
                    r[3 + n_info],
                    r[4 + n_info],
                    r[5 + n_info],
                    r[6 + n_info],
                )
            )
            blobs.append(r[-1])
        matrix = np.frombuffer(b"".join(blobs), "<f4").reshape(len(blobs), DIM).astype(np.float32)
        return rows, matrix

    # --- languages (spec §5, §6) ---

    def set_language_raw(self, path: str, raw: str) -> None:
        with self._db:
            self._db.execute("UPDATE tracks SET language_raw = ? WHERE path = ?", (raw, path))

    def use_language_model(self, model_id: str) -> None:
        """Record the detector in use; a different one invalidates every detection."""
        with self._db:
            old = self.get_meta("language_model")
            if old is not None and old != model_id:
                self._db.execute(
                    "UPDATE tracks SET language_raw = NULL, language = NULL, language_alt = NULL, "
                    "language_conf = NULL, vocal_score = NULL, language_source = NULL"
                )
            self._db.execute(
                "INSERT INTO meta (key, value) VALUES ('language_model', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (model_id,),
            )

    def language_work(self) -> list[tuple[str, float]]:
        return list(
            self._db.execute(
                "SELECT path, duration_s FROM tracks WHERE language_raw IS NULL ORDER BY path"
            )
        )

    def languages_pending(self, model_id: str) -> int:
        old = self.get_meta("language_model")
        where = "" if old is not None and old != model_id else " WHERE language_raw IS NULL"
        return self._db.execute(f"SELECT COUNT(*) FROM tracks{where}").fetchone()[0]

    def language_rows(self) -> list[LangRow]:
        return [
            LangRow(*r)
            for r in self._db.execute(
                "SELECT path, artist, album_artist, album, language_raw FROM tracks ORDER BY path"
            )
        ]

    def unresolved_languages(self) -> bool:
        """True when a detection has no final language yet (a run was cut short)."""
        row = self._db.execute(
            "SELECT 1 FROM tracks WHERE language_raw IS NOT NULL AND language IS NULL LIMIT 1"
        ).fetchone()
        return row is not None

    def set_final(self, finals: dict[str, Final]) -> None:
        sets = ", ".join(f"{c} = ?" for c in _FINAL_COLUMNS)
        with self._db:
            self._db.executemany(
                f"UPDATE tracks SET {sets} WHERE path = ?",
                [
                    (f.language, f.alt, f.conf, f.vocal_score, f.source, path)
                    for path, f in finals.items()
                ],
            )

    def has_languages(self) -> bool:
        return bool(
            self._db.execute(
                "SELECT 1 FROM tracks WHERE language IS NOT NULL AND language != 'unknown' LIMIT 1"
            ).fetchone()
        )

    def language_counts(self) -> list[tuple[str, int]]:
        return list(
            self._db.execute(
                "SELECT COALESCE(language, 'not detected') AS l, COUNT(*) AS n FROM tracks "
                "GROUP BY l ORDER BY n DESC, l"
            )
        )

    def unsure_tracks(
        self, conf_max: float
    ) -> list[tuple[str, str | None, str | None, str, float, str]]:
        """Sung tracks whose final language has conf below `conf_max` (languages.CONF_MIN)."""
        return list(
            self._db.execute(
                "SELECT path, artist, title, language, language_conf, language_source FROM tracks "
                f"WHERE language_conf < ? AND language NOT IN {_NOT_A_LANGUAGE} "
                "ORDER BY path",
                (conf_max,),
            )
        )

    def failed_languages(self) -> list[tuple[str, str]]:
        out = []
        for path, raw in self._db.execute(
            "SELECT path, language_raw FROM tracks WHERE language_raw LIKE '%\"failed\"%' "
            "ORDER BY path"
        ):
            try:
                reason = json.loads(raw).get("failed")
            except (ValueError, AttributeError):
                continue
            if reason is not None:
                out.append((path, str(reason)))
        return out
