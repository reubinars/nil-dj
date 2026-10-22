"""Sound search: brute-force cosine over the rows left after SQL filters (spec §7). At library
scale (thousands of tracks) a full scan takes milliseconds, so there is no ANN index."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nil_dj.embedder import Embedder
from nil_dj.index import Index, Row

CODECS = ("flac", "alac", "aac", "wav", "aiff", "dsd")
_ARTIST = "casefold(COALESCE(album_artist, artist, ''))"
_TRACK_ARTIST = "casefold(COALESCE(artist, ''))"


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
        if self.hires_only:
            clauses.append("(bits > 16 OR sample_rate > 48000 OR codec = 'dsd')")
        return " AND ".join(clauses), params


@dataclass
class Hit:
    row: Row
    score: float


def candidate_pool_size(needed: int) -> int:
    return max(60, 3 * needed)


def _rank(rows: list[Row], matrix: np.ndarray, q: np.ndarray, limit: int,
          exclude: frozenset[str] = frozenset()) -> list[Hit]:  # fmt: skip
    keep = [i for i, r in enumerate(rows) if r.path not in exclude]
    if not keep:
        return []
    rows = [rows[i] for i in keep]
    scores = matrix[keep] @ q
    # Descending score; ties by path (code-point order = UTF-8 byte order), so results are exact.
    order = np.lexsort((np.array([r.path for r in rows]), -scores))[:limit]
    return [Hit(rows[i], float(scores[i])) for i in order]


def _where(filters: Filters | None) -> tuple[str, list]:
    return filters.to_sql() if filters else ("", [])


def search_text(
    index: Index,
    embedder: Embedder,
    query: str,
    limit: int = 20,
    filters: Filters | None = None,
    template: str = "{}",
) -> list[Hit]:
    q = embedder.embed_text(template.format(query))
    rows, matrix = index.load(*_where(filters))
    return _rank(rows, matrix, q, limit)


def _similar(index: Index, seed_vectors: np.ndarray, seeds: set[str], limit: int,
             filters: Filters | None) -> list[Hit]:  # fmt: skip
    mean = seed_vectors.mean(axis=0)
    q = (mean / np.linalg.norm(mean)).astype(np.float32)
    rows, matrix = index.load(*_where(filters))
    return _rank(rows, matrix, q, limit, frozenset(seeds))


def similar_to_paths(
    index: Index, seeds: list[str], limit: int = 20, filters: Filters | None = None
) -> list[Hit]:
    wanted = set(seeds)
    rows, matrix = index.load(f"path IN ({_placeholders(len(wanted))})", sorted(wanted))
    missing = wanted - {r.path for r in rows}
    if missing:
        raise KeyError(min(missing))
    return _similar(index, matrix, wanted, limit, filters)


def similar_to_album(
    index: Index, album: str, artist: str | None, limit: int = 20, filters: Filters | None = None
) -> list[Hit]:
    where, params = "casefold(album) = ?", [album.casefold()]
    if artist is not None:
        where += f" AND ({_ARTIST} = ? OR {_TRACK_ARTIST} = ?)"
        params += [artist.casefold()] * 2
    rows, matrix = index.load(where, params)
    if not rows:
        raise KeyError(album if artist is None else f"{album} by {artist}")
    return _similar(index, matrix, {r.path for r in rows}, limit, filters)
