"""Radio: while on, keep nil's queue topped up with tracks like what was recently played.

Threads: `on_player_changed` runs on the notification worker; `set` runs on MCP or panel
threads. `_lock` serialises the trigger check, the pick and the append, so one queue state is
never topped up twice. It is held across nil calls, which is fine because replies are read on
the reader thread; never call a Radio method from the reader thread. `on_change` runs with the
lock released.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from contextlib import closing
from threading import Lock
from typing import Any

import numpy as np

from nil_dj.config import Weights
from nil_dj.host import NilError
from nil_dj.index import Index, IndexProblem, path_key
from nil_dj.order import anchor_of, candidate, order

TOP_UP_COUNT = 5
POOL_SIZE = 40
SEED_WEIGHTS = (1.0, 0.8, 0.6, 0.4, 0.2)  # newest played first
EXCLUDE_S = 7200.0
HISTORY_LIMIT = 200
NO_SEED = "Radio stopped: none of the recently played tracks are in the index."


class Radio:
    def __init__(
        self,
        open_index: Callable[[], Index],
        call: Callable[[str, dict], Any],
        weights: Weights,
        clock: Callable[[], float] = time.time,
        rng: random.Random | None = None,
        on_change: Callable[[], None] = lambda: None,
        closed: Callable[[], bool] = lambda: False,
        allowed: Callable[[], Callable[[str], bool] | None] = lambda: None,
    ):
        self.closed = closed
        self.allowed = allowed  # what may be added, by path (nil's library); None: anything
        self.open_index = open_index
        self.call = call
        self.weights = weights
        self.clock = clock
        self.rng = rng or random.Random()
        self.on_change = on_change
        self.on = False
        self.message: str | None = None
        self.history: list[tuple[str, float]] = []  # (path, started_at), newest last
        self._lock = Lock()
        self._topped_up_for: tuple[str, ...] | None = None  # the queue at the last top-up

    # -- entry points ----------------------------------------------------------------------

    def set(self, on: bool) -> None:
        with self._lock:
            before = (self.on, self.message)
            self.on = bool(on)
            self.message = None
            if self.on:
                self._topped_up_for = None  # switching on again retries a queue that failed
                self._guarded(self._set_state)
            changed = (self.on, self.message) != before
        if changed:
            self.on_change()

    def on_player_changed(self, params: dict) -> None:
        with self._lock:
            before = (self.on, self.message)
            self._record(params)
            if self.on and self._near_end(params):
                # The notification may be stale (a top-up since); decide on nil's answer.
                self._guarded(lambda: self._top_up(self.call("player.state", {}) or {}))
            changed = (self.on, self.message) != before
        if changed:
            self.on_change()

    def recent(self) -> list[str]:
        """The played paths, oldest first; a copy, safe to read from any thread."""
        with self._lock:
            return [p for p, _ in self.history]

    def _set_state(self) -> None:
        state = self.call("player.state", {}) or {}
        self._record(state)
        self._top_up(state)

    def _guarded(self, fn: Callable[[], None]) -> None:
        """With `_lock` held. Turns problems into a message instead of raising."""
        try:
            fn()
        except IndexProblem as e:
            self.on = False
            self.message = str(e)
        except NilError as e:
            if not self.closed():  # at shutdown nobody is left to tell
                self.message = f"Radio: {e}"

    # -- history and trigger ---------------------------------------------------------------

    def _record(self, state: dict) -> None:
        path = state.get("path")
        if state.get("state") != "playing" or not isinstance(path, str):
            return
        if self.history and self.history[-1][0] == path:
            return
        self.history.append((path, self.clock()))
        del self.history[:-HISTORY_LIMIT]

    @staticmethod
    def _near_end(state: dict) -> bool:
        index, n = state.get("index"), state.get("queue_len")
        if state.get("state") != "playing" or not isinstance(index, int):
            return False
        return isinstance(n, int) and index >= n - 2

    def _top_up(self, state: dict) -> None:
        """With `_lock` held. `state` is a `player.state` result."""
        queue = state.get("queue") or []
        key = tuple(queue)
        if not self._near_end({**state, "queue_len": len(key)}) or key == self._topped_up_for:
            return
        try:
            paths = self.pick(queue)
            if self.on and paths:
                self.call("player.append", {"paths": paths})
                self.message = None
        finally:
            # Even when nothing is left to add, or nil refused the append, don't search again
            # for this queue: the next player.changed would only fail the same way.
            self._topped_up_for = key

    # -- choosing --------------------------------------------------------------------------

    def pick(self, queue: list[str]) -> list[str]:
        """Up to 5 paths like what was played, none played in the last two hours or queued.
        Turns radio off with a message when nothing played is in the index."""
        with closing(self.open_index()) as index:
            rows, matrix = index.load()  # one snapshot for the seed, the pool and the history
        # History and queue paths are nil's spelling; rows are the index's: match by key.
        at = {path_key(r.path): i for i, r in enumerate(rows)}
        recent = [path_key(p) for p, _ in reversed(self.history[-len(SEED_WEIGHTS) :])]
        seeded = [(at[p], w) for p, w in zip(recent, SEED_WEIGHTS, strict=False) if p in at]
        history = [path_key(p) for p, _ in self.history]
        if not seeded:
            newest = next((at[p] for p in reversed(history) if p in at), None)
            if newest is None:
                self.on = False
                self.message = NO_SEED
                return []
            seeded = [(newest, 1.0)]
        mean = sum(w * matrix[i] for i, w in seeded)
        seed = (mean / np.linalg.norm(mean)).astype(np.float32)
        sims = matrix @ seed

        cutoff = self.clock() - EXCLUDE_S
        skip = {path_key(p) for p, t in self.history if t > cutoff} | {path_key(p) for p in queue}
        in_library = self.allowed()
        allowed = [
            i for i, r in enumerate(rows)
            if path_key(r.path) not in skip and (in_library is None or in_library(r.path))
        ]  # fmt: skip
        allowed.sort(key=lambda i: -float(sims[i]))  # stable: ties stay in path order
        hist = [candidate(rows[at[p]], matrix[at[p]]) for p in history if p in at]
        keep = set(allowed[:POOL_SIZE])
        anchor = anchor_of(hist)
        if anchor:  # a language rare among the nearest tracks can still be chosen
            same = [i for i in allowed if set(candidate(rows[i], matrix[i]).languages) & anchor]
            keep.update(same[:POOL_SIZE])
        top = [i for i in allowed if i in keep]  # similarity order
        if not top:
            return []
        cands = [candidate(rows[i], matrix[i]) for i in top]
        match = np.array([sims[i] for i in top], np.float32)
        chosen = order(
            cands, match, weights=self.weights, count=TOP_UP_COUNT, history=hist, top_k=3,
            rng=self.rng,
        )  # fmt: skip
        return [c.path for c in chosen]
