"""Does radio keep the language, and what does it cost in similarity? Prints counts only.

    uv run python eval/radio_language.py [--index PATH] [--sessions 10] [--seed 1]

For each language with at least 50 sung tracks: simulated 2-hour radio sessions from random seed
tracks of that language, with Weights.language at its configured value and at 0.

- share: the share of sung picks that are in the seed's language.
- drift: 1 minus the mean cosine of the picks in the second hour to the seed track's vector.
"""

from __future__ import annotations

import argparse
import random
import sys
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import numpy as np

from nil_dj.config import Weights, load_config
from nil_dj.index import Index
from nil_dj.languages import INSTRUMENTAL, UNKNOWN
from nil_dj.radio import Radio

SESSION_S = 7200.0
MIN_TRACKS = 50


def sung(language: str | None) -> bool:
    return language not in (None, INSTRUMENTAL, UNKNOWN)


def simulate(
    open_index, weights: Weights, seed_path: str, rng_seed: int
) -> list[tuple[str, float]]:
    """(path, start time) of the seed and every pick of a 2-hour session."""
    now = [0.0]
    radio = Radio(open_index, lambda *_: {}, weights, clock=lambda: now[0],
                  rng=random.Random(rng_seed))  # fmt: skip
    with closing(open_index()) as index:
        rows, _ = index.load()
    length = {r.path: r.info.duration_s for r in rows}
    played: list[tuple[str, float]] = []

    def play(path: str) -> None:
        radio.history.append((path, now[0]))
        played.append((path, now[0]))
        now[0] += length[path]

    play(seed_path)
    while now[0] < SESSION_S:
        picks = radio.pick([])
        if not picks:
            break
        for path in picks:
            if now[0] >= SESSION_S:
                break
            play(path)
    return played


def evaluate(open_index, weights: Weights, sessions: int, rng_seed: int) -> dict:
    with closing(open_index()) as index:
        rows, matrix = index.load()
    at = {r.path: i for i, r in enumerate(rows)}
    by_lang: dict[str, list[str]] = {}
    for r in rows:
        if sung(r.language):
            by_lang.setdefault(r.language, []).append(r.path)
    out = {}
    for lang, paths in sorted(by_lang.items()):
        if len(paths) < MIN_TRACKS:
            continue
        rng = random.Random(rng_seed)
        seeds = [rng.choice(paths) for _ in range(sessions)]
        res = {}
        for name, w in (("with", weights), ("without", replace(weights, language=0.0))):
            same = total = 0
            drifts = []
            for n, seed in enumerate(seeds):
                played = simulate(open_index, w, seed, rng_seed * 1000 + n)
                vec = matrix[at[seed]]
                for path, t in played[1:]:
                    r = rows[at[path]]
                    if sung(r.language):
                        total += 1
                        same += lang in (r.language, r.language_alt)
                late = [float(matrix[at[p]] @ vec) for p, t in played[1:] if t >= SESSION_S / 2]
                if late:
                    drifts.append(1 - float(np.mean(late)))
            res[name] = (same / total if total else 0.0, float(np.mean(drifts)) if drifts else 0.0,
                         total)  # fmt: skip
        out[lang] = (len(paths), res)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--index", type=Path, default=None)
    p.add_argument("--sessions", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args(argv)
    cfg = load_config()
    path = args.index or cfg.index_path

    def open_index() -> Index:
        return Index.open(path, None)

    results = evaluate(open_index, cfg.weights, args.sessions, args.seed)
    print(f"language weight {cfg.weights.language:.2f} vs 0; {args.sessions} sessions each")
    if not results:
        print(f"no language has {MIN_TRACKS} sung tracks")
        return 1
    print("lang  tracks  sung picks  share(with)  share(0)  drift(with)  drift(0)  drift worse")
    shares, worse = [], []
    for lang, (n, r) in results.items():
        (sw, dw, picks), (s0, d0, _) = r["with"], r["without"]
        rel = (dw - d0) / d0 if d0 else 0.0
        shares.append(sw)
        worse.append(rel)
        print(f"{lang:<5} {n:>6}  {picks:>10}  {sw:>10.1%}  {s0:>8.1%}  {dw:>11.4f}"
              f"  {d0:>8.4f}  {rel:>+10.1%}")  # fmt: skip
    print(f"mean share {np.mean(shares):.1%}; mean drift change {np.mean(worse):+.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
