# Queue Editing Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** MCP tools `remove_from_queue`, `move_in_queue` and `clear_upcoming`, and `upcoming` in
`now_playing`, on top of nil 0.3.0's `player.remove`, `player.move` and
`player.clear_upcoming`.

**Architecture:** The DJ reads `player.state`, turns the agent's 1-based upcoming positions into
nil's `{index, path}` pairs, calls nil once, and turns nil's `-32004` / `-32601` into plain
messages raised as `ValueError`, which the server's `_tool_errors` already passes to the client
as text. FakeNil gets the three methods with nil's rules so every test runs without nil.

**Tech Stack:** Python 3.12, FastMCP, pytest, the existing `FakeNil` (`tests/fake_nil.py`).

**Spec:** `docs/superpowers/specs/2026-10-30-queue-editing-design.md` (approved by the owner
2026-10-30). One correction, fixed in the spec by Task 3: `move_in_queue`'s tracks must *start
at* position `to` in the result, so `to` maps to the index of the `to`-th upcoming track that
is **not** being moved (or `queue_len` past the last), not to `current + to`. With upcoming
`[A, B, C, D]`, moving `[1]` to `3` gives `[B, C, A, D]`.

## Global Constraints

- Positions are 1-based within the upcoming tracks: position 1 is `queue[index + 1]`.
- Messages, exactly:
  - `-32004` with `data.reason` `"changed"` or `"started"`: `The queue changed (or the next track is already starting). Call now_playing and try again.`
  - `-32004` `"stopped"`, or nothing playing when the tool reads the state: `Nothing is playing.` (the existing `NOTHING_PLAYING`)
  - `-32601`: `Queue editing needs nil 0.3.0 or newer.`
- No retry inside a tool.
- `upcoming` holds at most 20 entries `{ "position", "path", "title", "artist" }`; `title` and `artist` are `null` for a path the index lacks; `[]` when stopped.
- The panel is unchanged.
- Commits use the repo's noreply email (already the repo-local `user.email`); no attribution lines. Nothing from audio-rea's `docs/` is copied here.
- Before each commit: `pytest` and `ruff check . && ruff format --check .` clean.

## Review Focus

1. **A position given twice, or as 0** (`remove_from_queue([2, 2])`, `[0]`): refused before nil is called, naming the bad position. Test: Task 2 `test_bad_positions_never_reach_nil`.
2. **Moving tracks to a position after themselves** (`move_in_queue([1], 3)`): the tracks start at position 3 of the result. Test: Task 2 `test_move_lands_at_the_position_given`.
3. **`to` far past the end** (`move_in_queue([1], 99)`): the tracks go to the end, not an error. Test: Task 2 `test_move_past_the_end_goes_last`.
4. **Older nil** (a nil 0.2.0 answers `-32601`): the agent reads the upgrade message, not "nil: method not found". Test: Task 2 `test_old_nil_says_upgrade`.
5. **The next track starting while the agent removes it**: the agent gets the "queue changed" message and the queue is unchanged. Test: Task 2 `test_started_and_changed_say_try_again`.

---

### Task 1: FakeNil learns `player.remove`, `player.move`, `player.clear_upcoming`

**Files:**
- Modify: `tests/fake_nil.py` (`_NEEDS`, `__init__`, `_on_request`)
- Test: `tests/test_fake_nil.py`

**Interfaces:**
- Produces, for Tasks 2 and 3:
  - `FakeNil.next_starting: bool = False`: when true the track after the current one counts as taken by the decoder (nil's last ~2 s).
  - `player.remove {tracks: [{index, path}]}` → `{}`; `player.move {tracks, to}` → `{}`; `player.clear_upcoming` → `{"removed": n}`. All need `playback`.
  - nil's rules (audio-rea spec, Plugin API), in this order: `-32602` (`tracks` not a non-empty list of `{index: int, path: str}`, an index repeated, `to` not an int in `0..=len(queue)`); `-32004 "stopped"` when stopped; `"changed"` when an index ≥ `len(queue)` or its path differs (compare with `path_key`); `"started"` when an index ≤ current, a `to` ≤ current, or an index (or `to`) is the next track while `next_starting`. `NilError(-32004, "queue changed", {"reason": r})`.
  - Move: the named tracks, in the order given, go in front of the track now at `to` (at the end for `to == len(queue)`). Clear removes everything after the taken track(s) (`current + 1`, or `current + 2` while `next_starting`) and returns the count.
  - Remove and clear put a frame (`self._frames.put(self._nil_writer.flushes + 1)`), so `player.changed` follows a length change, as for append.

- [ ] **Step 1: Write the failing tests** (with `_fake()`, queue of 4 known paths `a, b, c, d` playing at index 0; add `c.flac` and `d.flac` to `_fake`'s tracks)

```python
def test_remove_drops_the_named_tracks():
    f = _playing()
    assert f._on_request("player.remove", {"tracks": [{"index": 2, "path": C}]}) == {}
    assert f.queue == [A, B, D]

def test_move_puts_tracks_in_front_of_to_in_the_order_given():
    f = _playing()
    f._on_request("player.move", {"tracks": [{"index": 3, "path": D}, {"index": 2, "path": C}], "to": 1})
    assert f.queue == [A, D, C, B]
    f = _playing()
    f._on_request("player.move", {"tracks": [{"index": 1, "path": B}], "to": 4})
    assert f.queue == [A, C, D, B]

def test_clear_upcoming_counts_and_respects_next_starting():
    f = _playing()
    f.next_starting = True
    assert f._on_request("player.clear_upcoming", {}) == {"removed": 2}
    assert f.queue == [A, B]

@pytest.mark.parametrize(("tracks", "reason"), [
    ([{"index": 1, "path": C}], "changed"),
    ([{"index": 9, "path": C}], "changed"),
    ([{"index": 0, "path": A}], "started"),
])
def test_queue_changed_reasons(tracks, reason):
    f = _playing()
    with pytest.raises(NilError) as e:
        f._on_request("player.remove", {"tracks": tracks})
    assert (e.value.code, e.value.data) == (-32004, {"reason": reason})
    assert f.queue == [A, B, C, D]

def test_next_starting_locks_the_next_track():  # remove index 1 while next_starting -> "started"
def test_stopped_is_stopped():                  # state "stopped" -> remove / move / clear: "stopped"
def test_bad_queue_params():                    # tracks [] / missing / index repeated / to 5 / to "x" -> -32602
```

`_playing()` is a new helper: `_fake()` with `queue = [A, B, C, D]`, `index = 0`, `state = "playing"`.

- [ ] **Step 2: Run them to see them fail**

Run: `pytest tests/test_fake_nil.py -q`
Expected: the new tests FAIL (`unknown method` / no `next_starting`).

- [ ] **Step 3: Implement the three methods in `tests/fake_nil.py`**

- [ ] **Step 4: Run the whole suite**

Run: `pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/fake_nil.py tests/test_fake_nil.py
git commit -m "test: the fake nil edits the queue like nil 0.3.0"
```

---

### Task 2: DJ: `upcoming`, `remove_from_queue`, `move_in_queue`, `clear_upcoming`

**Files:**
- Modify: `src/nil_dj/dj.py` (constants, `now_playing`, new methods)
- Test: `tests/test_dj.py`

**Interfaces:**
- Consumes: Task 1's FakeNil methods and `next_starting`.
- Produces, in `dj.py`:
  ```python
  QUEUE_CHANGED = "The queue changed (or the next track is already starting). Call now_playing and try again."
  NEEDS_NIL_030 = "Queue editing needs nil 0.3.0 or newer."
  UPCOMING_SHOWN = 20

  class DJ:
      def now_playing(self) -> dict: ...  # gains "upcoming"
      def remove_from_queue(self, positions: list[int]) -> None: ...
      def move_in_queue(self, positions: list[int], to: int) -> None: ...
      def clear_upcoming(self) -> int: ...  # nil's "removed"
  ```
  - `now_playing`: `upcoming` from `queue[index + 1 : index + 1 + UPCOMING_SHOWN]` before `queue` is dropped; titles and artists from one `index.load_paths` over the playing path and the upcoming paths together (rows matched back by `path_key`), via `Track.from_row`; `None` for paths without a row and when the index can't be opened (`IndexProblem`). Stopped or no `index` → `[]`.
  - A private `_upcoming_state() -> tuple[int, list[str]]`: reads `player.state`, returns `(current, queue)`; raises `ValueError(NOTHING_PLAYING)` when stopped or `index` is missing.
  - Positions are checked before any queue call: non-empty (`"Name at least one position."`), each an `int` in `1..=len(upcoming)` (`f"No upcoming track at position {p}; there are {n} upcoming tracks."`, naming the first bad one), no repeats (`f"Position {p} is listed twice."`). `to` must be ≥ 1 (`"to must be 1 or more."`).
  - `move_in_queue`'s nil `to`: the queue index of the `to`-th upcoming track not in `positions`, or `len(queue)` if there are fewer.
  - A private `_edit(method, params) -> Any` wraps the call: `NilError` with code `-32004` → `ValueError(NOTHING_PLAYING)` for reason `"stopped"`, else `ValueError(QUEUE_CHANGED)`; code `METHOD_NOT_FOUND` (`-32601`) → `ValueError(NEEDS_NIL_030)`; any other `NilError` passes through.

- [ ] **Step 1: Write the failing tests** (the `dj` and `fake` fixtures; play `[P[0], P[1], P[2], P[3], P[4]]` first)

```python
def test_now_playing_lists_upcoming(dj, fake):
    dj.play(P[:5])
    up = dj.now_playing()["upcoming"]
    assert [u["position"] for u in up] == [1, 2, 3, 4]
    assert [u["path"] for u in up] == P[1:5]
    assert up[0]["title"] and "artist" in up[0]

def test_upcoming_is_capped_and_marks_unindexed_paths(dj, fake):
    # play P[0] then append 25 copies of P[1] and NEW (in nil's library, not in the index)
    # -> 20 entries; the NEW entry has title None and artist None

def test_upcoming_is_empty_when_stopped(dj, fake):  # fresh fake -> now_playing()["upcoming"] == []

def test_remove_maps_positions_to_index_and_path(dj, fake):
    dj.play(P[:5])
    dj.remove_from_queue([2, 4])
    assert fake.calls[-1] == ("player.remove", {"tracks": [{"index": 2, "path": P[2]}, {"index": 4, "path": P[4]}]})
    assert fake.queue == [P[0], P[1], P[3]]

def test_move_lands_at_the_position_given(dj, fake):
    dj.play(P[:5])
    dj.move_in_queue([1], 3)            # upcoming [P1, P2, P3, P4] -> [P2, P3, P1, P4]
    assert fake.queue == [P[0], P[2], P[3], P[1], P[4]]
    dj.move_in_queue([3, 1], 1)         # play them next, in the order given
    assert fake.queue[1:3] == [P[1], P[2]]

def test_move_past_the_end_goes_last(dj, fake):  # move_in_queue([1], 99) -> P[1] last; nil got to == 5

def test_clear_upcoming_returns_the_count(dj, fake):  # play P[:5] -> clear_upcoming() == 4; queue [P[0]]

@pytest.mark.parametrize("positions", [[], [0], [5], [2, 2], ["1"]])
def test_bad_positions_never_reach_nil(dj, fake, positions):
    dj.play(P[:5])
    before = len(fake.calls)
    with pytest.raises(ValueError):
        dj.remove_from_queue(positions)
    assert [m for m, _ in fake.calls[before:]] == ["player.state"]

def test_started_and_changed_say_try_again(dj, fake):
    dj.play(P[:5])
    fake.next_starting = True
    with pytest.raises(ValueError) as e:
        dj.remove_from_queue([1])
    assert str(e.value) == QUEUE_CHANGED
    fake.fail["player.remove"] = (-32004, "queue changed")  # FakeNil's fail sends no data
    # -> still QUEUE_CHANGED (a -32004 without a reason counts as changed)

def test_nothing_playing(dj, fake):  # stopped -> each of the three raises ValueError(NOTHING_PLAYING)

def test_old_nil_says_upgrade(dj, fake):
    dj.play(P[:5])
    for method in ("player.remove", "player.move", "player.clear_upcoming"):
        fake.fail[method] = (-32601, "method not found")
    # each tool raises ValueError(NEEDS_NIL_030)
```

Add the three methods to `test_not_in_nil`'s list (they raise `NotInNil`).

- [ ] **Step 2: Run them to see them fail**

Run: `pytest tests/test_dj.py -q`
Expected: the new tests FAIL.

- [ ] **Step 3: Implement**

- [ ] **Step 4: Run the whole suite**

Run: `pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nil_dj/dj.py tests/test_dj.py
git commit -m "feat: the DJ removes, moves and clears upcoming tracks"
```

---

### Task 3: MCP tools, docs, version

**Files:**
- Modify: `src/nil_dj/server.py` (three tools, `now_playing` docstring), `src/nil_dj/__init__.py` and `pyproject.toml` (version `0.3.0`), `docs/validation-log.md`, `docs/superpowers/specs/2026-10-30-queue-editing-design.md`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: Task 2's DJ methods.
- Tools (docstrings are what the agent reads; keep them to these facts):
  ```python
  @tool
  def remove_from_queue(positions: list[int]) -> dict:
      """Remove upcoming tracks from nil's queue. Positions are 1-based within `upcoming` from
      now_playing (1 = the next track)."""
      dj.remove_from_queue(positions); return {}

  @tool
  def move_in_queue(positions: list[int], to: int) -> dict:
      """Move upcoming tracks so they start at position `to`, in the order given. to=1 plays
      them next; a position past the last puts them at the end."""

  @tool
  def clear_upcoming() -> dict:
      """Remove every upcoming track; the playing one keeps playing."""
      return {"removed": dj.clear_upcoming()}
  ```
  `now_playing`'s docstring adds: "`upcoming`: the next 20 tracks with `position` (1 = next),
  path, title and artist."

- [ ] **Step 1: Write the failing tests**

```python
TOOLS = {..., "remove_from_queue", "move_in_queue", "clear_upcoming"}  # extend the set

def test_queue_tools_reach_nil(dj, fake):
    server = build_server(dj)
    call(server, "play", {"paths": P[:4]})
    assert [u["position"] for u in call(server, "now_playing")["upcoming"]] == [1, 2, 3]
    assert call(server, "move_in_queue", {"positions": [3], "to": 1}) == {}
    assert fake.queue == [P[0], P[3], P[1], P[2]]
    assert call(server, "remove_from_queue", {"positions": [1]}) == {}
    assert call(server, "clear_upcoming") == {"removed": 2}

def test_queue_tool_errors_are_plain_text(dj, fake):
    server = build_server(dj)
    assert tool_error(server, "clear_upcoming") == "Nothing is playing."
    call(server, "play", {"paths": P[:3]})
    fake.next_starting = True
    assert tool_error(server, "remove_from_queue", {"positions": [1]}) == QUEUE_CHANGED
```

- [ ] **Step 2: Run them to see them fail**

Run: `pytest tests/test_server.py -q`
Expected: FAIL (`test_tools_listed` and the two new tests).

- [ ] **Step 3: Add the tools**

- [ ] **Step 4: Version and docs**
  - `__version__` and `pyproject.toml` `version` → `0.3.0`.
  - `docs/validation-log.md`: a "Queue editing tools" section: "Needs nil 0.3.0. Tested here only against the fake nil." and the owner's Windows check from the spec (ask the agent to remove, move and clear while playing, with radio on and off).
  - Spec: in MCP tools, replace "and `to` into `current + to` (capped at the queue length)" with "and `to` into the index of the `to`-th upcoming track not being moved (the queue length past the last)", and add the `[A, B, C, D]` example; Status → "approved 2026-10-30; implemented".

- [ ] **Step 5: Full verification**

Run: `pytest -q && ruff check . && ruff format --check .`
Expected: all PASS, nothing to fix.

- [ ] **Step 6: Commit**

```bash
git add src/nil_dj tests/test_server.py pyproject.toml docs
git commit -m "feat: MCP tools to edit nil's queue; nil-dj 0.3.0"
```
