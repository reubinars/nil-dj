"""Greedy playlist ordering (spec §7): walk the pool one track at a time, preferring tracks that
sound like the previous one and match the request, without repeating artists or albums."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nil_dj.config import Weights
from nil_dj.index import Row, _casefold
from nil_dj.languages import INSTRUMENTAL, UNKNOWN

LENGTH_SLACK = 1.05


@dataclass(frozen=True)
class Candidate:
    path: str
    vector: np.ndarray
    duration_s: float
    bpm: float | None
    lufs: float | None
    artist: str | None
    album: str | None  # prefixed by the artist, so two artists' "Greatest Hits" differ
    languages: frozenset[str] = frozenset()  # empty: neutral (instrumental, unknown, undetected)


def candidate(row: Row, vector: np.ndarray) -> Candidate:
    info = row.info
    artist = _casefold(info.album_artist or info.artist)
    album = _casefold(info.album)
    langs: frozenset[str] = frozenset()
    if row.language not in (None, INSTRUMENTAL, UNKNOWN):
        langs = frozenset(x for x in (row.language, row.language_alt) if x)
    return Candidate(
        path=row.path,
        vector=vector,
        duration_s=info.duration_s,
        bpm=row.bpm,
        lufs=row.lufs,
        artist=artist,
        album=f"{artist or ''}\x00{album}" if album else None,
        languages=langs,
    )


def tempo_jump(a: float | None, b: float | None) -> float:
    """Tempo difference in 20-BPM units; half and double time count as no change."""
    if a is None or b is None:
        return 0.0
    return min(abs(a - b), abs(2 * a - b), abs(a - 2 * b)) / 20


def loud_jump(a: float | None, b: float | None) -> float:
    if a is None or b is None:
        return 0.0
    return abs(a - b) / 6


def anchor_of(played: Sequence[Candidate]) -> frozenset[str] | None:
    """The language set of the newest played track that is not neutral."""
    return next((c.languages for c in reversed(played) if c.languages), None)


def _score(
    w: Weights, prev: Candidate | None, t: Candidate, match: float,
    anchor: frozenset[str] | None = None,
) -> float:  # fmt: skip
    if prev is None:
        return match
    bonus = w.language if anchor and (not t.languages or t.languages & anchor) else 0.0
    return (
        bonus +
        w.similarity * float(prev.vector @ t.vector)
        + w.match * match
        - w.tempo * tempo_jump(prev.bpm, t.bpm)
        - w.loudness * loud_jump(prev.lufs, t.lufs)
    )


def order(
    cands: Sequence[Candidate],
    match: np.ndarray,
    *,
    weights: Weights,
    target_s: float | None = None,
    count: int | None = None,
    history: Sequence[Candidate] = (),
    top_k: int = 1,
    rng: random.Random | None = None,
) -> list[Candidate]:
    if (target_s is None) == (count is None):
        raise ValueError("give exactly one of target_s or count")
    if top_k > 1 and rng is None:
        rng = random.Random()
    w = weights
    unused = list(range(len(cands)))
    chosen: list[Candidate] = []
    total = 0.0
    while unused:
        if count is not None and len(chosen) >= count:
            break
        if target_s is not None and total >= target_s:
            break
        played = [*history, *chosen]
        prev = played[-1] if played else None
        anchor = anchor_of(played)
        pool = unused
        if target_s is not None:
            pool = [i for i in pool if total + cands[i].duration_s <= LENGTH_SLACK * target_s]
        if not pool:
            break
        recent = {c.album for c in played[-2:] if c.album is not None}
        fresh_artist = [
            i for i in pool if prev is None or prev.artist is None or cands[i].artist != prev.artist
        ]
        eligible = [i for i in fresh_artist if cands[i].album not in recent] or fresh_artist or pool
        ranked = sorted(
            eligible,
            key=lambda i: (-_score(w, prev, cands[i], float(match[i]), anchor), cands[i].path),
        )
        pick = rng.choice(ranked[:top_k]) if top_k > 1 else ranked[0]
        chosen.append(cands[pick])
        total += cands[pick].duration_s
        unused.remove(pick)
    return chosen
