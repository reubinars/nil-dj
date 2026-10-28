"""Sound search: brute-force cosine over the rows left after SQL filters (spec §7). At library
scale (thousands of tracks) a full scan takes milliseconds, so there is no ANN index."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from nil_dj.embedder import Embedder
from nil_dj.index import Index, Row, _casefold, path_key
from nil_dj.languages import normalise_code
from nil_dj.tags import TrackInfo

CODECS = ("flac", "alac", "aac", "wav", "aiff", "dsd")
_ARTIST = "casefold(COALESCE(album_artist, artist, ''))"
_TRACK_ARTIST = "casefold(COALESCE(artist, ''))"
_ALBUM_ARTIST = "casefold(COALESCE(album_artist, ''))"

# Which rows may be chosen, by path: nil's library in plugin mode, None (all) standalone.
Allowed = Callable[[str], bool] | None


def _placeholders(n: int) -> str:
    return ", ".join("?" * n)


@dataclass
class Filters:
    min_minutes: float | None = None
    max_minutes: float | None = None
    year_from: int | None = None
    year_to: int | None = None
    artists: list[str] | None = None
    exclude_artists: list[str] | None = None
    formats: list[str] | None = None
    hires_only: bool = False
    languages: list[str] | None = None

    def to_sql(self) -> tuple[str, list]:
        clauses: list[str] = []
        params: list = []
        if self.min_minutes is not None:
            clauses.append("duration_s >= ?")
            params.append(self.min_minutes * 60)
        if self.max_minutes is not None:
            clauses.append("duration_s <= ?")
            params.append(self.max_minutes * 60)
        if self.year_from is not None:
            clauses.append("year >= ?")
            params.append(self.year_from)
        if self.year_to is not None:
            clauses.append("year <= ?")
            params.append(self.year_to)
        for names, negate in ((self.artists, False), (self.exclude_artists, True)):
            if names:
                folded = [n.casefold() for n in names]
                ph = _placeholders(len(folded))
                match = f"({_ARTIST} IN ({ph}) OR {_TRACK_ARTIST} IN ({ph}))"
                clauses.append(f"NOT {match}" if negate else match)
                params += folded * 2
        if self.formats:
            unknown = sorted(set(self.formats) - set(CODECS))
            if unknown:
                raise ValueError(f"unknown format {', '.join(unknown)}; use {', '.join(CODECS)}")
            clauses.append(f"codec IN ({_placeholders(len(self.formats))})")
            params += list(self.formats)
        if self.languages:
            ph = _placeholders(len(self.languages))
            clauses.append(f"(language IN ({ph}) OR language_alt IN ({ph}))")
            params += list(self.languages) * 2
        if self.hires_only:
            clauses.append("(bits > 16 OR sample_rate > 48000 OR codec = 'dsd')")
        return " AND ".join(clauses), params


_FILTER_NAMES = [f.name for f in fields(Filters)]


def _check_filter(key: str, value: Any) -> None:
    if value is None:
        return
    if key in ("min_minutes", "max_minutes"):
        ok, kind = isinstance(value, int | float) and not isinstance(value, bool), "a number"
    elif key in ("year_from", "year_to"):
        ok, kind = isinstance(value, int) and not isinstance(value, bool), "a whole number"
    elif key == "hires_only":
        ok, kind = isinstance(value, bool), "true or false"
    else:
        ok = isinstance(value, list) and all(isinstance(v, str) for v in value)
        kind = "a list of names"
    if not ok:
        raise ValueError(f"filter '{key}' must be {kind}")


def filters_from(d: dict | None) -> Filters | None:
    """Filters from a JSON-style dict (MCP arguments); raises ValueError on a bad key or value."""
    if not d:
        return None
    for key, value in d.items():
        if key not in _FILTER_NAMES:
            raise ValueError(f"unknown filter '{key}'; use {', '.join(_FILTER_NAMES)}")
        _check_filter(key, value)
    f = Filters(**d)
    if f.languages is not None:
        f.languages = [normalise_code(c) for c in f.languages]  # ValueError names a bad code
    return f


@dataclass
class Hit:
    row: Row
    score: float
    vector: np.ndarray


def candidate_pool_size(needed: int) -> int:
    return max(60, 3 * needed)


def _where(filters: Filters | None) -> tuple[str, list]:
    return filters.to_sql() if filters else ("", [])


def rank(
    index: Index,
    q: np.ndarray,
    limit: int,
    filters: Filters | None = None,
    exclude: frozenset[str] = frozenset(),
    allowed: Allowed = None,
) -> list[Hit]:
    """The `limit` rows most like `q` (unit vector) after the filters, best first. `exclude`
    is compared by `path_key`."""
    rows, matrix = index.load(*_where(filters))
    skip = {path_key(p) for p in exclude}
    keep = [
        i for i, r in enumerate(rows)
        if path_key(r.path) not in skip and (allowed is None or allowed(r.path))
    ]  # fmt: skip
    if not keep:
        return []
    rows = [rows[i] for i in keep]
    matrix = matrix[keep]
    scores = matrix @ q
    # Descending score; ties by path (code-point order = UTF-8 byte order), so results are exact.
    order = np.lexsort((np.array([r.path for r in rows]), -scores))[:limit]
    return [Hit(rows[i], float(scores[i]), matrix[i]) for i in order]


def search_text(
    index: Index,
    embedder: Embedder,
    query: str,
    limit: int = 20,
    filters: Filters | None = None,
    template: str = "{}",
    allowed: Allowed = None,
) -> list[Hit]:
    q = embedder.embed_text(template.format(query))
    return rank(index, q, limit, filters, allowed=allowed)


def _similar(index: Index, seed_vectors: np.ndarray, seeds: set[str], limit: int,
             filters: Filters | None, allowed: Allowed) -> list[Hit]:  # fmt: skip
    mean = seed_vectors.mean(axis=0)
    q = (mean / np.linalg.norm(mean)).astype(np.float32)
    return rank(index, q, limit, filters, frozenset(seeds), allowed)


def similar_to_paths(
    index: Index, seeds: list[str], limit: int = 20, filters: Filters | None = None,
    allowed: Allowed = None,
) -> list[Hit]:  # fmt: skip
    """Seeds are matched by `path_key`; a seed that isn't in the index raises KeyError."""
    rows, matrix = index.load_paths(seeds)
    found = {path_key(r.path) for r in rows}
    missing = [p for p in seeds if path_key(p) not in found]
    if missing:
        raise KeyError(min(missing))
    return _similar(index, matrix, {r.path for r in rows}, limit, filters, allowed)


def similar_to_album(
    index: Index, album: str, artist: str | None, limit: int = 20, filters: Filters | None = None,
    allowed: Allowed = None,
) -> list[Hit]:  # fmt: skip
    where, params = "casefold(album) = ?", [album.casefold()]
    if artist is not None:
        where += f" AND ({_ARTIST} = ? OR {_TRACK_ARTIST} = ?)"
        params += [artist.casefold()] * 2
    rows, matrix = index.load(where, params)
    if not rows:
        raise KeyError(album if artist is None else f"{album} by {artist}")
    return _similar(index, matrix, {r.path for r in rows}, limit, filters, allowed)


def find_by_tags(
    index: Index,
    artist: str | None = None,
    album: str | None = None,
    title: str | None = None,
    limit: int = 50,
    allowed: Allowed = None,
) -> list[Row]:
    """Rows whose tags contain each given text (case-insensitive), in album order."""
    clauses: list[str] = []
    params: list = []
    if artist is not None:
        clauses.append(f"(instr({_TRACK_ARTIST}, ?) OR instr({_ALBUM_ARTIST}, ?))")
        params += [artist.casefold()] * 2
    for column, text in (("album", album), ("title", title)):
        if text is not None:
            clauses.append(f"instr(casefold(COALESCE({column}, '')), ?)")
            params.append(text.casefold())
    rows, _ = index.load(" AND ".join(clauses), params)
    if allowed is not None:
        rows = [r for r in rows if allowed(r.path)]

    def key(r: Row):
        i = r.info
        return (
            _casefold(i.album_artist or i.artist) or "", _casefold(i.album) or "",
            i.disc or 0, i.number or 0, r.path,
        )  # fmt: skip

    return sorted(rows, key=key)[:limit]


def describe_format(info: TrackInfo) -> str:
    """For example "flac 24-bit/96 kHz", "aac 44.1 kHz" or "dsd 2.8 MHz"."""
    if info.codec == "dsd":
        # Truncated, so the rates read as people name them: 2.8, 5.6, 11.2 MHz.
        return f"dsd {math.floor(info.sample_rate / 100_000) / 10:g} MHz"
    rate = f"{info.sample_rate / 1000:g} kHz"
    return f"{info.codec} {info.bits}-bit/{rate}" if info.bits else f"{info.codec} {rate}"
