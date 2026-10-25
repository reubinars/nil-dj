import random
from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest
from fakes import BASE, unit

from nil_dj.config import Weights
from nil_dj.index import Row
from nil_dj.order import Candidate, candidate, loud_jump, order, tempo_jump

W = Weights(0.6, 0.4, 0, 0)


def cand(path, v=(1, 0), artist=None, album=None, dur=240.0, bpm=None, lufs=None):
    return Candidate(
        path=path, vector=unit(*v), duration_s=dur, bpm=bpm, lufs=lufs,
        artist=artist, album=album,
    )  # fmt: skip


def names(cs):
    return [c.path for c in cs]


def test_jumps():
    assert tempo_jump(120, 60) == 0
    assert tempo_jump(120, 130) == 0.5
    assert tempo_jump(None, 90) == 0
    assert loud_jump(-10, -16) == 1.0
    assert loud_jump(None, -8) == 0


def test_candidate_from_row():
    info = replace(BASE, artist="BjÖrk", album_artist=None, album="Al")
    c = candidate(Row("p", 1, 1, info, 100.0, -9.0), unit(1, 0))
    assert (c.path, c.artist, c.album, c.duration_s, c.bpm, c.lufs) == (
        "p", "björk", "björk\x00al", 240.0, 100.0, -9.0,
    )  # fmt: skip
    info = replace(BASE, artist="A", album_artist="Va", album=None)
    c = candidate(Row("p", 1, 1, info, None, None), unit(1, 0))
    assert (c.artist, c.album) == ("va", None)


def test_starts_with_best_match_then_follows_similarity():
    a = cand("A", (1, 0), "a")
    b = cand("B", (1, 0.1), "b")
    c = cand("C", (0, 1), "c")
    got = order([c, b, a], np.array([0.5, 0.1, 0.9]), weights=W, count=3)
    assert names(got) == ["A", "B", "C"]


def test_never_same_artist_twice_in_a_row():
    cs = [cand(f"t{i}", (1, i * 0.01), "xy"[i // 2], f"al{i}") for i in range(4)]
    got = order(cs, np.full(4, 0.5), weights=W, count=4)
    assert len(got) == 4
    assert all(p.artist != q.artist for p, q in pairwise(got))


def test_album_at_most_once_in_three():
    cs = [cand(f"x{i}", (1, 0), "x", "x\x00al") for i in range(5)]
    cs += [cand(f"y{i}", (1, 0), "y", f"y{i}") for i in range(2)]
    cs += [cand(f"z{i}", (1, 0), "z", f"z{i}") for i in range(2)]
    got = order(cs, np.full(len(cs), 0.5), weights=W, count=6)
    for i in range(len(got) - 2):
        albums = [c.album for c in got[i : i + 3]]
        assert len(set(albums)) == 3


def test_relaxes_album_then_artist():
    cs = [cand(f"t{i}", (1, 0), "x", "x\x00al") for i in range(4)]
    got = order(cs, np.full(4, 0.5), weights=W, count=4)
    assert sorted(names(got)) == ["t0", "t1", "t2", "t3"]


def test_no_repeats():
    cs = [cand(f"t{i}", (1, i * 0.1), f"a{i}") for i in range(6)]
    got = order(cs, np.full(6, 0.5), weights=W, count=6)
    assert len(set(names(got))) == 6


def test_length_target():
    cs = [cand(f"t{i}", (1, 0), f"a{i}", dur=d) for i, d in enumerate([250, 250, 250, 100])]
    got = order(cs, np.full(4, 0.5), weights=W, target_s=600)
    assert sum(c.duration_s for c in got) == 600
    cs = [cand(f"t{i}", (1, 0), f"a{i}", dur=200) for i in range(5)]
    assert len(order(cs, np.full(5, 0.5), weights=W, target_s=600)) == 3


def test_exactly_one_of_target_and_count():
    cs = [cand("a")]
    with pytest.raises(ValueError):
        order(cs, np.array([0.5]), weights=W)
    with pytest.raises(ValueError):
        order(cs, np.array([0.5]), weights=W, target_s=10, count=1)


def test_ties_break_by_path():
    cs = [cand("b", (1, 0), "x"), cand("a", (1, 0), "y")]
    assert names(order(cs, np.array([0.5, 0.5]), weights=W, count=2)) == ["a", "b"]


def _cands():
    return [cand(f"t{i}", (1, i * 0.3), f"a{i}", f"al{i}") for i in range(8)]


def test_top_k_seeded():
    cs, m = _cands(), np.linspace(0.9, 0.2, 8)
    first = order(cs, m, weights=W, count=5, top_k=3, rng=random.Random(1))
    again = order(cs, m, weights=W, count=5, top_k=3, rng=random.Random(1))
    assert names(first) == names(again)
    # each pick is within the top 3 of its step
    idx = {c.path: i for i, c in enumerate(cs)}
    used: set[str] = set()
    for n, pick in enumerate(first):
        prev = first[n - 1] if n else None
        scored = []
        for c in cs:
            if c.path in used or (prev and c.artist == prev.artist):
                continue
            s = m[idx[c.path]] * 0.4
            if prev:
                s = 0.6 * float(prev.vector @ c.vector) + s
            else:
                s = m[idx[c.path]]
            scored.append((-s, c.path))
        assert pick.path in [p for _, p in sorted(scored)[:3]]
        used.add(pick.path)
    orders = {
        tuple(names(order(cs, m, weights=W, count=5, top_k=3, rng=random.Random(s))))
        for s in range(1, 21)
    }
    assert len(orders) >= 2


def test_history_counts_for_rules():
    hist = [cand("h", (1, 0), "x", "x\x00al")]
    cs = [cand("x1", (1, 0), "x", "x\x00b"), cand("y1", (0, 1), "y", "y\x00b")]
    got = order(cs, np.array([0.9, 0.1]), weights=W, count=1, history=hist)
    assert names(got) == ["y1"]
