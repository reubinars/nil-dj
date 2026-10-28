# DJ Panel Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The DJ tab shows only the playlist tools and queue-style results; index status, the index button and the MCP address move to the DJ's entry in Settings → Plugins; failed tracks are written to nil's plugin log.

**Architecture:** `panel.py` builds two pure panels (`build_tab`, `build_settings`) plus the old single panel (`build_single`) for nil older than 0.2.0. `Panel` sends each place only when it changed and routes events by `place`. The indexer and runner print failures to stderr, which nil writes to the plugin log.

**Tech Stack:** Python 3.12, pytest, the fake nil in `tests/fake_nil.py`.

**Spec:** `docs/superpowers/specs/2026-10-28-panel-redesign-design.md`. Needs nil 0.2.0 (audio-rea plan `docs/superpowers/plans/2026-10-28-plugin-panels-look.md`, private).

## Global Constraints

- Settings panel may contain only heading, text, progress, button, toggle, row.
- Tab ids: `prompt`, `more`, `radio`. Settings ids: `index`, `rebuild`. An id from the other place is ignored.
- Exact copy: "Index your library first: Settings → Plugins → DJ." / "Indexing your library… progress is in Settings → Plugins." / "No matching tracks." / "MCP server: <address>" / "MCP server: not running" / "<n> failed (see Open log)".
- Log lines: `nil-dj: index failed: <path>: <reason>`, `nil-dj: language detection failed: <path>: <reason>`, `nil-dj: <n> tracks failed to index:`, `nil-dj: <n> language detections failed:`, entries as `  <path>: <reason>`.
- `nil_version` below 0.2.0, missing or unparseable → single old panel, never send `place`.
- Nothing copied from audio-rea `docs/`. No Claude attribution in commits.
- `uv run pytest -q` and `uv run ruff check` / `uv run ruff format --check` clean before each commit.

## Review Focus

- A real nil 0.1.1 (what the owner runs today) must keep a working tab — pinned by Task 3's `test_old_nil_gets_one_panel` with `"0.1.1"`.
- A send that fails (nil busy, rejected) must not leave the panel stale forever — pinned by Task 3's `test_a_failed_send_is_retried`.
- stderr closed or broken while indexing (nil stopping) must not crash the run — pinned by Task 4's `test_log_write_errors_are_ignored`.
- A playlist under one minute or over ten hours formats sensibly — pinned by Task 1's `test_minutes_format` cases.
- The first index on a fresh install, with the tab open: the hint, not an empty page — pinned by Task 2's `test_tab_hint_while_first_index_runs`.

---

### Task 1: Playlist name and length text

**Files:**
- Modify: `src/nil_dj/dj.py:116-121,386,429-430`
- Test: `tests/test_dj.py`

**Interfaces:**
- Produces: `Playlist.name: str` (new field, last, default `""`), set to the `name` passed to `_playlist`; `title` unchanged (MCP still uses it). `dj.minutes_text(seconds: float) -> str` in `dj.py`.

- [ ] **Step 1: Failing tests.**
  - `test_minutes_format` (parametrized): `0 → "0 min"`, `29 → "0 min"`, `31 → "1 min"`, `2280 → "38 min"`, `3600 → "1 h 0 min"`, `6720 → "1 h 52 min"`, `37800 → "10 h 30 min"`. Rounds to the nearest minute.
  - `test_playlist_has_its_name`: `build_playlist("city pop", 45)` on the existing fixture → `result.name == "city pop"` (the prompt is the name `_playlist` receives); `more_like_playing()` → name starts with `"More like "`.
- [ ] **Step 2: Run** `uv run pytest tests/test_dj.py -q -k "minutes or name"` — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q` — PASS.
- [ ] **Step 5: Commit** `feat: playlists keep their name; minutes text`.

### Task 2: `build_tab`, `build_settings`, `build_single`

**Files:**
- Modify: `src/nil_dj/panel.py:28,46-123`
- Modify: `tests/fake_nil.py:42-100` (`validate_panel(panel, place="tab")`: `label` must be a string when present; for `"settings"` refuse `tracks` and `input` anywhere)
- Test: `tests/test_panel.py:17-170`

**Interfaces:**
- Consumes: `Playlist.name`, `minutes_text` (Task 1).
- Produces: `build_tab(view: View) -> dict`, `build_settings(view: View) -> dict`, `build_single(view: View) -> dict` (today's `build`, renamed, unchanged output). `MCP_NOT_RUNNING` stays the value `"MCP: not running"` used by `mcp_status` callers; `build_settings` shows it as `"MCP server: not running"` and a running status `"MCP: <addr>"` as `"MCP server: <addr>"` — `McpHttp.status` is `"MCP: http://…/mcp"`, a port message or `"MCP unavailable: …"`: replace a leading `"MCP: "` with `"MCP server: "` and show anything else unchanged.

`build_tab`: no heading; hint (only when `status.indexed == 0` and no `status.problem`: running → the "Indexing your library…" line, else the "Index your library first…" line, both muted); `_model_elements` (unless equal to message, as today); input `prompt`; row `more` + toggle `radio`; message; result as `{"type":"tracks","label": f"{name} · {n} tracks · {minutes_text(total_s)}","items":[...]}` (`name` falls back to `title` when empty; "1 track" singular), or text "No matching tracks.".

`build_settings`: `view.problems` (muted), `_index_elements` with the failed part as `"{n:,} failed (see Open log)"`, the MCP line (muted), `_index_buttons`.

- [ ] **Step 1: Failing tests.** Convert the existing `build`-based tests in `test_panel.py` to the panel each now belongs to (index/button/MCP/problem tests → `build_settings`; model/result/message tests → `build_tab`) and add:
  - `test_tab_has_no_heading_and_no_index_or_mcp`: a view with indexed tracks → no element of type heading; no text containing "indexed" or "MCP".
  - `test_tab_hint_when_nothing_indexed` / `test_tab_hint_while_first_index_runs`: exact copy above.
  - `test_result_label`: 2 tracks totalling 6720 s named "city pop" → label `"city pop · 2 tracks · 1 h 52 min"`; no heading for the result.
  - `test_failed_points_to_the_log`: `failed=3` → the index line contains `"3 failed (see Open log)"`.
  - `test_settings_mcp_line`: `"MCP: http://127.0.0.1:47263/mcp"` → `"MCP server: http://127.0.0.1:47263/mcp"`; `MCP_NOT_RUNNING` → `"MCP server: not running"`; `"MCP unavailable: boom"` unchanged.
  - `test_settings_panel_is_valid_for_settings`: every `build_settings` view in the file passes `validate_panel(p, "settings")`; `build_tab` outputs pass `validate_panel(p)`.
  - `test_single_panel_is_unchanged`: `build_single(view)` equals the old `build` output for one rich view (inline the expected element list).
- [ ] **Step 2: Run** `uv run pytest tests/test_panel.py -q` — FAIL.
- [ ] **Step 3: Implement.** `Panel._send` temporarily calls `build_single` so integration tests stay green until Task 3.
- [ ] **Step 4: Run** `uv run pytest -q` — PASS.
- [ ] **Step 5: Commit** `feat: DJ tab and settings panels`.

### Task 3: Two places: sending, events, old-nil fallback

**Files:**
- Modify: `src/nil_dj/panel.py` (`Panel.__init__`, `_send`, `on_event`, `_event`, `_handle`), `src/nil_dj/plugin.py:57-71,154-171`
- Modify: `tests/fake_nil.py` (`handshake(..., nil_version="0.2.0")`; record `self.panels` for the tab and `self.settings_panels` for `place: "settings"`, validating each with its place; `element(id)` searches the last tab panel, new `settings_element(id)` the last settings panel; unknown `place` → recorded error)
- Test: `tests/test_panel.py`, `tests/test_plugin.py`

**Interfaces:**
- Produces: `panel.supports_places(nil_version: object) -> bool` (True iff a string whose first two dot-separated integer parts are ≥ (0, 2); "0.2.0", "0.10.1", "1.0" True; "0.1.1", None, "", "garbage", 2 False). `Panel(dj, call, mcp_status, min_interval_s=0.5, problems=None, places=False)`. `plugin.py` stores `params.get("nil_version")` from `initialize` in `state["nil_version"]` and passes `places=supports_places(...)`.
- `_start`'s first placeholder `panel.set` sends `{"elements": []}` when `places`, else today's heading.

Sending: with `places`, `_send` builds both and sends each (`panel.set {"panel": p}` and `{"panel": p, "place": "settings"}`) only if it differs from the last sent payload of that place; a send that raises clears that place's last payload. Without `places`, sends `build_single` as today (also deduplicated the same way).

Events: `on_event(params)` reads `params.get("place")` (`"settings"` or tab). `_handle(id, value, place)`: `prompt`/`more`/`radio` only from tab; `index`/`rebuild` only from settings when `places`, only from tab when not.

- [ ] **Step 1: Failing tests.**
  - `test_supports_places` (parametrized with the values above).
  - `test_unchanged_panels_are_not_resent` (StubDJ, `places=True`): two forced refreshes with no change → 2 `panel.set` calls total (one per place) not 4.
  - `test_radio_change_sends_only_the_tab`: toggle radio state on the stub → next forced refresh sends one call without `place`.
  - `test_a_failed_send_is_retried`: `call` raises on the first settings send → next refresh sends the settings panel again.
  - `test_events_from_the_wrong_place_are_ignored`: `{"id":"index"}` (tab) with `places=True` → no reindex; `{"id":"prompt","value":"x","place":"settings"}` → no search.
  - `test_old_nil_gets_one_panel` (fake nil, plugin run) for `"0.1.1"`, `None` (key absent) and `"garbage"`: no `panel.set` carries `place`; the last tab panel has an element `id == "index"`.
  - `test_new_nil_gets_two_panels`: `"0.2.0"` → `fake_nil.settings_panels` non-empty, `settings_element("index")` exists, tab has no `index`.
  - Update existing integration tests that click `index` (`test_index_button_indexes`) to send `{"id":"index","place":"settings"}` and read state via `settings_panels`.
- [ ] **Step 2: Run** `uv run pytest tests/test_panel.py tests/test_plugin.py -q` — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q` — PASS.
- [ ] **Step 5: Commit** `feat: DJ settings panel in nil 0.2.0, single panel on older nil`.

### Task 4: Failure log

**Files:**
- Modify: `src/nil_dj/indexer.py` (where `record_error` is called, ~343 and ~432; where a detection ends with `ok is False`, the `_detect` / catch-up paths ~299 and ~404-411)
- Modify: `src/nil_dj/reindex.py:124-183`
- Test: `tests/test_indexer.py`, `tests/test_dj.py` (or a runner test file already used for `IndexRunner`)

**Interfaces:**
- Produces: `indexer.log_line(text: str) -> None` (prints `text` to `sys.stderr` with flush; swallows `OSError` and `ValueError`); `reindex.log_failures(index: Index) -> None` (uses `index.errors()` and `index.failed_languages()`, prints the summary lines from Global Constraints; skips an empty section).
- The language failure's reason is the same text passed to `encode_failed`.

`IndexRunner._run` calls `log_failures(index)` after `_resolve` on the normal and stopped paths (inside the `try` before `index.close()`), and not when the run failed to open the index.

- [ ] **Step 1: Failing tests.**
  - `test_index_failures_are_logged` (`capsys`): index a folder with one undecodable file (the existing bad-file fixture) → stderr contains `"nil-dj: index failed: "` + its path + `": "`.
  - `test_language_failures_are_logged`: a detector stub that raises → stderr contains `"nil-dj: language detection failed: "` + path.
  - `test_run_end_lists_all_failures`: an index with one prior `record_error` and one failed language record, run with nothing new to do → stderr contains `"nil-dj: 1 tracks failed to index:"`, `"  <path>: <reason>"`, `"nil-dj: 1 language detections failed:"`.
  - `test_run_end_skips_empty_sections`: no failures → neither summary line.
  - `test_log_write_errors_are_ignored`: monkeypatch `sys.stderr` with an object whose `write` raises `OSError` → `log_line("x")` returns None.
- [ ] **Step 2: Run** `uv run pytest tests/test_indexer.py -q -k log` — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q` — PASS.
- [ ] **Step 5: Commit** `feat: failed tracks go to nil's plugin log`.

### Task 5: Docs and validation entry

**Files:**
- Modify: `README.md` (where the panel is described: index and MCP now under Settings → Plugins → DJ; failures in Open log; needs nil 0.2.0 for the new layout), `docs/validation-log.md` ("Still to do": the four Windows checks from the spec's "Windows check" section)

- [ ] **Step 1:** Edit both files.
- [ ] **Step 2:** `uv run pytest -q` PASS.
- [ ] **Step 3: Commit** `docs: panel redesign`.
