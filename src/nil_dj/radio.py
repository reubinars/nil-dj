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
from nil_dj.index import Index
from nil_dj.order import candidate, order

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
    ):
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
        self._topped_up_at: int | None = None  # the queue length at the last top-up

    # -- entry points ----------------------------------------------------------------------

    def set(self, on: bool) -> None:
        with self._lock:
            before = (self.on, self.message)
            self.on = bool(on)
            self.message = None
            if self.on:
                try:
                    state = self.call("player.state", {}) or {}
                    self._record(state)
                    self._top_up(state)
                except NilError as e:
                    self.message = f"Radio: {e}"
            changed = (self.on, self.message) != before
        if changed:
            self.on_change()

    def on_player_changed(self, params: dict) -> None:
        with self._lock:
            before = (self.on, self.message)
            self._record(params)
            if self.on and self._near_end(params):
                try:
                    # The notification may be stale (a top-up since); decide on nil's answer.
                    self._top_up(self.call("player.state", {}) or {})
                except NilError as e:
                    self.message = f"Radio: {e}"
            changed = (self.on, self.message) != before
        if changed:
            self.on_change()

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
        n = len(queue)
        if not self._near_end({**state, "queue_len": n}) or n == self._topped_up_at:
            return
        paths = self.pick(queue)
        if self.on and paths:
            self.call("player.append", {"paths": paths})
            self.message = None
        # Even when nothing is left to add, don't search again for this queue length.
        self._topped_up_at = n

    # -- choosing --------------------------------------------------------------------------

    def pick(self, queue: list[str]) -> list[str]:
        """Up to 5 paths like what was played, none played in the last two hours or queued.
        Turns radio off with a message when nothing played is in the index."""
        with closing(self.open_index()) as index:
            rows, matrix = index.load()  # one snapshot for the seed, the pool and the history
        at = {r.path: i for i, r in enumerate(rows)}
        recent = [p for p, _ in reversed(self.history[-len(SEED_WEIGHTS) :])]
        seeded = [(at[p], w) for p, w in zip(recent, SEED_WEIGHTS, strict=False) if p in at]
        if not seeded:
            newest = next((at[p] for p, _ in reversed(self.history) if p in at), None)
            if newest is None:
                self.on = False
                self.message = NO_SEED
                return []
            seeded = [(newest, 1.0)]
        mean = sum(w * matrix[i] for i, w in seeded)
        seed = (mean / np.linalg.norm(mean)).astype(np.float32)
        sims = matrix @ seed

        cutoff = self.clock() - EXCLUDE_S
        skip = {p for p, t in self.history if t > cutoff} | set(queue)
        allowed = [i for i, r in enumerate(rows) if r.path not in skip]
        allowed.sort(key=lambda i: -float(sims[i]))  # stable: ties stay in path order
        top = allowed[:POOL_SIZE]
        if not top:
            return []
        cands = [candidate(rows[i], matrix[i]) for i in top]
        match = np.array([sims[i] for i in top], np.float32)
        hist = [candidate(rows[at[p]], matrix[at[p]]) for p, _ in self.history if p in at]
        chosen = order(
            cands, match, weights=self.weights, count=TOP_UP_COUNT, history=hist, top_k=3,
            rng=self.rng,
        )  # fmt: skip
        return [c.path for c in chosen]
