# Plan 3: nil-dj as a nil plugin — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** nil starts nil-dj as a plugin. nil-dj shows a panel in nil (prompt playlist, more like this, radio, indexing), runs radio, and serves MCP on `127.0.0.1:47263`, so any agent can drive nil.

**Architecture:**
- One Python process is started by nil with the plugin folder as its working directory.
- `host` speaks line-delimited JSON-RPC with nil:
  - a reader thread resolves replies;
  - a worker thread handles notifications, so handlers may call nil.
- A `DJ` service holds all behaviour: search, ordering, playback through nil, playlists, indexing and radio. The panel and the MCP tools are thin layers over it.
- The MCP server runs FastMCP's HTTP transport on a daemon thread in the same process.

**Tech Stack:**
- Python 3.12, numpy, ONNX Runtime (Plan 1).
- `fastmcp>=4.1,<5` (probed 4.1.3 on 2026-10-25: sync tools run on AnyIO worker threads; `run(transport="http", host, port, show_banner=False, uvicorn_config=...)`; `fastmcp.server.server.create_proxy(url)`; `fastmcp.Client(server)` in memory; `fastmcp.exceptions.ToolError`).
- `uv`, pytest, ruff.

**Spec:**
- `docs/superpowers/specs/2026-10-22-nil-dj-design.md` (§4–§11).
- The protocol as built is `~/projects/audio-rea/PLUGIN-API.md`. Read it; it wins over nil's spec where they differ. Do not copy anything else from audio-rea `docs/` into this repo.

## Global Constraints

- Python ≥ 3.12; line length 100; `ruff check .` clean; existing 107 tests stay green.
- Default `pytest` run needs no network, no model and no GPU. Real-CLAP tests are `@pytest.mark.slow`; Windows-only ones are `@pytest.mark.windows` + `skipif`.
- **stdout carries the protocol and nothing else.** One JSON object per line, UTF-8 (`ensure_ascii=False`), `\n`, flushed per message, ≤ 1,048,576 bytes.
- Plugin identity: `name = "nil-dj"`, `api_version = 1`, `permissions = ["playback", "library", "playlists"]`.
- MCP: `127.0.0.1` only, default port **47263**. Overrides: `mcp_port` in `config.toml`, then env `NIL_DJ_MCP_PORT`. Never fall back to another port.
- Plugin folder layout (spec §4):
  - `plugin.toml`, `config.toml`;
  - `app\.venv\`;
  - `data\index.sqlite`, `data\index.sqlite.lock`;
  - `models\<REVISION>\`.
- Exact user-facing copy:
  - `NOT_IN_NIL = "nil-dj isn't running inside nil — enable it in nil's Settings → Plugins."`
  - `"MCP unavailable: port {port} in use"`
- Ordering defaults: similarity 0.60, match 0.40, tempo 0.15, loudness 0.10.
- Radio constants: 5 tracks per top-up, top 40 candidates, random among the top 3, history weights `1.0, 0.8, 0.6, 0.4, 0.2`, 2 h exclusion.
- Never a Tidal/Qobuz downloader or store integration.
- Commits: no `Co-Authored-By`/"Generated with" lines. Author email is the GitHub noreply address already configured.

## Review Focus

1. **Japanese and other non-ASCII paths and titles** must survive the round trip: nil → plugin (`library.tracks`, `player.changed`) and plugin → nil (`panel.set`, `player.play`), byte-exact. Covered in Task 5 (codec), Task 7 (play through FakeNil) and Task 8 (radio history).
2. **nil's library comes back empty** (drive unplugged, first scan not finished). "Index library" must refuse with a message, not delete every row. Covered in Task 4.
3. **The playing track isn't indexed** (new album, failed decode):
   - "More like this" says so;
   - radio falls back to older history, or turns off with a reason.
   - Neither crashes the worker. Covered in Tasks 7 and 8.
4. **A broken `config.toml`** (bad TOML, wrong types, unknown keys, port out of range): the plugin still starts with defaults and the panel shows the problem. Covered in Task 1 and Task 9.
5. **`shutdown` or stdin closing while indexing or while the model loads**: the process exits within 3 s, and the index stays usable (one commit per track). Covered in Task 6 (subprocess) and Task 4 (stop flag).

---

## File Structure

| File | Responsibility |
|---|---|
| `src/nil_dj/__main__.py` (new) | `python -m nil_dj` → `cli.main()` |
| `src/nil_dj/config.py` (rewrite) | Plugin folder discovery, `config.toml`, index/models paths, music folders |
| `src/nil_dj/embedder.py` (modify) | Model files into `models\<REVISION>\` |
| `src/nil_dj/order.py` (new) | Greedy ordering (spec §7), pure |
| `src/nil_dj/indexer.py` (modify) | Index an explicit path list; stop flag |
| `src/nil_dj/search.py` (modify) | `Hit.vector`, `rank()`, `find_by_tags()`, `describe_format()` |
| `src/nil_dj/host.py` (new) | JSON-RPC line codec and thread-safe `Connection` |
| `src/nil_dj/dj.py` (new) | The `DJ` service used by panel, radio and MCP |
| `src/nil_dj/radio.py` (new) | Radio state, history, top-ups |
| `src/nil_dj/panel.py` (new) | Panel elements from state; panel events |
| `src/nil_dj/server.py` (new) | FastMCP tools, HTTP thread, stdio bridge |
| `src/nil_dj/plugin.py` (new) | Plugin entry: fd guard, handshake, wiring, shutdown |
| `src/nil_dj/install.py` (new) | `nil-dj install --nil <home>` |
| `src/nil_dj/cli.py` (modify) | New commands `plugin`, `install`, `serve`; new config |
| `tests/fakes.py` (extend) | `unit()`, `BASE`, `ChosenText`, `add_track()` shared helpers |
| `tests/fake_nil.py` (new) | `FakeNil`: nil's side of the protocol over pipes |

---

### Task 1: Plugin folder and `config.toml`

**Files:**
- Rewrite: `src/nil_dj/config.py`
- Create: `src/nil_dj/__main__.py`
- Modify: `src/nil_dj/cli.py` (`load_config()` call sites only; drop `NIL_CONFIG_DIR` use)
- Test: `tests/test_config.py` (rewrite), `tests/test_cli.py` (remove the `NIL_CONFIG_DIR` env line)

**Interfaces:**
- Produces:
  - `DEFAULT_MCP_PORT = 47263`
  - `Weights(similarity=0.60, match=0.40, tempo=0.15, loudness=0.10)`: frozen dataclass.
  - `Config`: dataclass with fields
    - `index_path: Path`, `folders: list[Path]`;
    - `plugin_dir: Path | None = None`, `models_dir: Path | None = None`;
    - `precision="fp32"`, `gpu_device_id: int | None = None`, `prompt_template="{}"`;
    - `mcp_port=DEFAULT_MCP_PORT`, `weights=Weights()`;
    - `problems: list[str]` (default empty).
  - `find_plugin_dir() -> Path | None`
  - `load_config(plugin_dir: Path | None = None, data_dir: Path | None = None) -> Config`
  - `music_folders(nil_home: Path) -> list[Path]`
  - `legacy_index_path() -> Path` (the Plan 1 `default_index_path`, renamed; used by install).

Rules for `load_config` (write them as the docstring):
- **`plugin_dir`:** the argument, else `find_plugin_dir()`. `find_plugin_dir`:
  1. `Path(sys.prefix)` is `<p>/app/.venv` and `<p>/plugin.toml` exists → `<p>`.
  2. Else env `NIL_HOME` → `<NIL_HOME>/plugins/nil-dj`.
  3. Else `None`.
- **`index_path`:**
  1. env `NIL_DJ_INDEX`;
  2. else `data_dir/index.sqlite`;
  3. else `plugin_dir/data/index.sqlite`;
  4. else `legacy_index_path()`.
- **`models_dir`:** `plugin_dir/models`. `None` without a plugin folder (the Hugging Face cache, for dev checkouts).
- **`folders`:** `music_folders(plugin_dir.parent.parent)`, reading `<home>/data/library.toml` key `folders`. `[]` without a plugin folder.
- **`config.toml`** sits in `plugin_dir`. Keys: `mcp_port`, `precision`, `gpu_device_id`, `prompt_template`, table `[order]` with `similarity`, `match`, `tempo`, `loudness`. Env `NIL_DJ_MCP_PORT` overrides `mcp_port`.
  - Any problem appends one line to `problems`, keeps the default for that key, and never raises. Exact copy:
    - `"config.toml: <TOMLDecodeError text>; using defaults"`
    - `"config.toml: unknown key 'x'"`
    - `"config.toml: mcp_port must be a whole number from 1 to 65535"`
    - `"config.toml: precision must be fp32, fp16 or int8"`
    - `"config.toml: order.tempo must be a number"`
  - The env variable uses the same port message, with prefix `NIL_DJ_MCP_PORT:` instead of `config.toml:`.

- [ ] **Step 1: Write the failing tests** in `tests/test_config.py`:
  - `test_music_folders_reads_library_toml`: `<home>/data/library.toml` with `folders = ['E:\Music', 'D:\More']` → those two strings. Keep the missing-file and missing-key cases → `[]`.
  - `test_plugin_dir_from_venv_prefix`: monkeypatch `sys.prefix` to `<tmp>/plugins/nil-dj/app/.venv` with a `plugin.toml` → that folder. Without `plugin.toml` → falls to `NIL_HOME`/`None`.
  - `test_plugin_dir_from_nil_home`
  - `test_paths_in_plugin_mode`: `load_config(p, data_dir=p/"data")` →
    - `index_path == p/"data"/"index.sqlite"`;
    - `models_dir == p/"models"`;
    - folders come from `p.parent.parent/"data"/"library.toml"`.
  - `test_nil_dj_index_env_wins`
  - `test_config_toml_values`: every key set → `Config` fields and `Weights` match.
  - `test_config_toml_problems` (parametrised): `"mcp_port = 0"`, `"mcp_port = 'x'"`, `"precision = 'fp8'"`, `"[order]\ntempo = 'fast'"`, `"bogus = 1"`, `"mcp_port = "` (invalid TOML).
    - Each → the exact message above is in `problems`.
    - The affected value stays default.
  - `test_mcp_port_env`: `NIL_DJ_MCP_PORT=47300` → 47300. `"abc"` → default plus the env problem.
  - `test_legacy_index_path_non_windows`: the old POSIX default.

- [ ] **Step 2: Run** `uv run pytest tests/test_config.py -v`. Expected: FAIL (import errors).
- [ ] **Step 3: Implement** `config.py` as above. Add `__main__.py`:
  ```python
  from nil_dj.cli import main
  raise SystemExit(main())
  ```
  Update `cli.py` for the new `load_config()`. Delete `nil_config_dir` and `NIL_CONFIG_DIR`.
- [ ] **Step 4: Run** `uv run pytest -q && uv run ruff check .`. Expected: all pass.
- [ ] **Step 5: Commit** `feat: plugin folder discovery and config.toml`

---

### Task 2: Model files in `models\<revision>\`

**Files:**
- Modify: `src/nil_dj/embedder.py`, `src/nil_dj/cli.py:23-27`
- Test: `tests/test_embedder.py`

**Interfaces:**
- Produces:
  - `model_files(precision: str) -> list[str]`: the repo files that precision needs. For `int8`, `onnx/audio_model_quantized.onnx` + `onnx/text_model_fp16.onnx` + `tokenizer.json`.
  - `models_present(models_dir: Path | None, precision: str) -> bool`: every `model_files` entry exists under `models_dir/REVISION`. Always `False` for `None`.
  - `OnnxClapEmbedder(precision="fp32", device_id=None, use_gpu=True, models_dir: Path | None = None)`
- `_fetch(filename, models_dir)`:
  - with `models_dir`: `hf_hub_download(REPO, filename, revision=REVISION, local_dir=models_dir / REVISION)`;
  - without: the Hugging Face cache, as today.

- [ ] **Step 1: Write the failing tests:**
  - `test_model_files`: exact lists for fp32/fp16/int8.
  - `test_fetch_uses_local_dir`: monkeypatch `huggingface_hub.hf_hub_download` to record kwargs → `local_dir == tmp/REVISION`, `revision == REVISION`.
  - `test_models_present`: create the fp16 files empty → True for fp16, False for fp32 and for `None`.
- [ ] **Step 2: Run** `uv run pytest tests/test_embedder.py -v`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement**, building the embedder's paths from `model_files()`. In `cli.make_embedder`, pass `models_dir=cfg.models_dir`.
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: keep the CLAP files in the plugin's models folder`

---

### Task 3: Ordering

**Files:**
- Create: `src/nil_dj/order.py`
- Test: `tests/test_order.py`

**Interfaces:**
- Consumes: `Weights` (Task 1), `Row` (`index.py`).
- Produces:
  - `Candidate`: frozen dataclass with `path: str`, `vector: np.ndarray`, `duration_s: float`, `bpm: float | None`, `lufs: float | None`, `artist: str | None`, `album: str | None`.
    - `artist` = casefold(album_artist or artist).
    - `album` = casefold(album) prefixed by the artist key, e.g. `"a\x00al"`. `None` when there is no album tag.
  - `candidate(row: Row, vector: np.ndarray) -> Candidate`
  - `tempo_jump(a: float | None, b: float | None) -> float`: `min(|a−b|, |2a−b|, |a−2b|) / 20`, 0 if either is None.
  - `loud_jump(a, b) -> float`: `|a−b| / 6`, 0 if either is None.
  - `order(cands, match, *, weights, target_s=None, count=None, history=(), top_k=1, rng=None) -> list[Candidate]`:
    - `cands: Sequence[Candidate]`; `match: np.ndarray` aligned with `cands`; `weights: Weights`; `history: Sequence[Candidate]` is the tracks already played, oldest first; `rng: random.Random | None`.
    - Exactly one of `target_s`/`count` must be given, else `ValueError`.

Algorithm (spec §7, plus these rulings):
- **Each step scores** every unused candidate:
  - `prev` = the last of `history + chosen`.
  - Score = `w.similarity·(prev·t) + w.match·match[t] − w.tempo·tempo_jump − w.loudness·loud_jump`.
  - With no `prev`, the score is `match[t]` alone.
- **Eligibility:**
  - the artist ≠ prev's artist;
  - the album is not among the last 2 tracks' albums;
  - `None` keys never conflict.
  - If none are eligible, drop the album rule; if still none, drop the artist rule.
- **Length:**
  - with `target_s`, a candidate is eligible only if `total + d ≤ 1.05·target_s`;
  - stop when `total ≥ target_s` or nothing fits;
  - with `count`, stop after `count` tracks.
- **Choice:**
  - sort eligible candidates by (−score, path);
  - take the first, or, with `top_k > 1`, `rng.choice` among the first `top_k`.

- [ ] **Step 1: Write the failing tests**, using 2-D unit vectors padded to 512 (`unit()` from `tests/fakes.py`; move `unit`, `BASE` and `ChosenText` there from `test_search.py` and import them back):
  - `test_jumps`:
    - `tempo_jump(120, 60) == 0`; `tempo_jump(120, 130) == 0.5`; `tempo_jump(None, 90) == 0`;
    - `loud_jump(-10, -16) == 1.0`; `loud_jump(None, -8) == 0`.
  - `test_starts_with_best_match_then_follows_similarity`: A matches best. B is near A; C is far from A but matches second best. Weights `(0.6, 0.4, 0, 0)`, `count=3` → `[A, B, C]`.
  - `test_never_same_artist_twice_in_a_row`: artists `x, x, y, y` (all on distinct albums), `count=4` → no two adjacent equal.
  - `test_album_at_most_once_in_three`: five tracks of one album by artist x, plus tracks by y and z → no album repeats inside any window of 3 while alternatives exist.
  - `test_relaxes_album_then_artist`: four tracks, same artist and album → all four returned, no exception.
  - `test_no_repeats`
  - `test_length_target`:
    - durations `[250, 250, 250, 100]`, all else equal, `target_s=600` → total 600 (the third 250 would reach 750 > 630, so the 100 is taken);
    - `[200]*5`, `target_s=600` → 3 tracks.
  - `test_ties_break_by_path`: identical candidates with paths `b`, `a` → `a` first.
  - `test_top_k_seeded`:
    - `random.Random(1)` → the same list twice;
    - each pick is among the top 3 of its step (check by recomputing);
    - seeds 1..20 give at least 2 different orders.
  - `test_history_counts_for_rules`: `history=[track by x]` → the first pick is not by x.
- [ ] **Step 2: Run** `uv run pytest tests/test_order.py -v`. Expected: FAIL (no module).
- [ ] **Step 3: Implement** `order.py` as specified.
- [ ] **Step 4: Run** `uv run pytest tests/test_order.py tests/test_search.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: greedy playlist ordering`

---

### Task 4: Indexing an explicit path list

**Files:**
- Modify: `src/nil_dj/indexer.py`
- Test: `tests/test_indexer.py`

**Interfaces:**
- Produces:
  - `run_index_paths(paths: list[str], index, embedder, lock_path, on_progress=..., workers=None, stop: threading.Event | None = None) -> RunResult`: indexes exactly `paths` (nil's `library.tracks`), sorted, without walking.
    - Rows and errors whose path is not in `paths` are deleted.
    - Empty `paths` raises `EmptyLibrary` and changes nothing.
  - `run_index(folders, …, stop=None)` keeps its signature and gains `stop`.
  - `RunResult.stopped: bool = False`
  - `class EmptyLibrary(Exception)`
- Refactor `_run` to take the on-disk list. `stop` is checked before writing each track. When it is set, submit nothing more, write nothing more, and return with `stopped=True`.

- [ ] **Step 1: Write the failing tests:**
  - `test_index_paths_only_those`: three files on disk, two passed → two rows.
  - `test_index_paths_removes_rows_not_in_library`: index three, then pass two → `removed == 1`.
  - `test_index_paths_empty_refuses`: index two, call with `[]` → `EmptyLibrary`, and still two rows (Review Focus 2).
  - `test_index_paths_missing_file_recorded`: a path that doesn't exist → `failed == 1`, an `index_errors` row starting `can't stat`.
  - `test_stop_flag`: set the event from `on_progress` after the first track of three → `stopped` True, one row. A second run indexes the remaining two (Review Focus 5).
  - `test_japanese_path` with `曲/深い.flac` → row path equals the input string exactly.
- [ ] **Step 2: Run** `uv run pytest tests/test_indexer.py -v`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: index the tracks nil lists, with a stop flag`

---

### Task 5: JSON-RPC connection (`host`)

**Files:**
- Create: `src/nil_dj/host.py`
- Test: `tests/test_host.py`

**Interfaces:**
- Produces:
  - `MAX_LINE = 1_048_576`
  - `class NilError(Exception)` with `code: int`, `message: str`, `data: Any = None`. `str()` is `message`.
  - Codes as constants: `PERMISSION = -32001`, `NOT_FOUND = -32002`, `BUSY = -32003`, `INTERNAL = -32603`, `INVALID_PARAMS = -32602`, `METHOD_NOT_FOUND = -32601`.
  - `class MessageTooLarge(ValueError)`
  - `encode(msg: dict) -> str`: `json.dumps(ensure_ascii=False, separators=(",", ":"))` + `"\n"`. Raises `MessageTooLarge` above `MAX_LINE` UTF-8 bytes, newline excluded.
  - `class Connection`:
    - `Connection(reader: TextIO, writer: TextIO, on_request: Callable[[str, dict], Any], on_notification: Callable[[str, dict], None], timeout_s: float = 10.0)`
    - `call(method: str, params: dict | None = None) -> Any`: thread-safe; ids are increasing ints. Raises `NilError` on an error reply, `NilError(INTERNAL, "nil didn't answer")` after `timeout_s`, and `NilError(INTERNAL, "nil has closed the connection")` once closed.
    - `serve() -> None`: blocks reading lines until EOF, then sets `closed` and fails pending calls.
      - Replies resolve pending calls.
      - Requests from nil run `on_request` on the reader thread. A raised `NilError` becomes an error reply; any other exception becomes `INTERNAL` with its text.
      - Notifications are queued to one worker thread that calls `on_notification` in order. Handler exceptions are printed to stderr with a traceback and never stop the worker.
      - Malformed lines: one stderr line each, ignored.
    - `closed: threading.Event`

- [ ] **Step 1: Write the failing tests**, connecting two `Connection`s back to back over `os.pipe()` pairs (wrapped with `open(fd, encoding="utf-8", newline="\n")`):
  - `test_encode_utf8_and_limit`:
    - `encode({"t": "深い"})` contains the raw `深い`;
    - a 1,048,577-byte payload → `MessageTooLarge`;
    - exactly at the limit passes.
  - `test_call_roundtrip_japanese`: the peer echoes params → the result equals `{"path": "E:\\音楽\\曲.flac"}`.
  - `test_error_reply_raises_nil_error`: the peer raises `NilError(-32002, "not in nil's library: 1 path(s)", {"paths": ["x"]})` → the caller sees the same code, message and data.
  - `test_concurrent_calls_from_threads`: 8 threads × 20 calls each → every result matches its request.
  - `test_notification_handler_can_call_back`: the peer's notification handler calls `call()` on its own connection without deadlock.
  - `test_handler_exception_does_not_stop_worker`: the first notification raises; the second is still handled.
  - `test_malformed_line_ignored`: write `"not json\n"` raw, then a valid request → answered.
  - `test_timeout`: the peer never answers; `timeout_s=0.2` → `NilError` "nil didn't answer".
  - `test_eof_fails_pending_and_sets_closed`
- [ ] **Step 2: Run** `uv run pytest tests/test_host.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest tests/test_host.py -v`. Expected: PASS, with no thread left hanging (the test suite finishes).
- [ ] **Step 5: Commit** `feat: JSON-RPC connection for the nil plugin protocol`

---

### Task 6: `FakeNil`, plugin entry and lifecycle

**Files:**
- Create: `tests/fake_nil.py`, `src/nil_dj/plugin.py`
- Modify: `src/nil_dj/cli.py` (add `plugin` subcommand → `plugin.main()`), `tests/conftest.py` (fixtures)
- Test: `tests/test_plugin.py`

**Interfaces:**
- Consumes: `Connection`, `NilError` (Task 5); `load_config` (Task 1).
- Produces:
  - `plugin.run(reader: TextIO, writer: TextIO, plugin_dir: Path, make_embedder: Callable[[Config], Embedder], start_mcp: bool = True) -> int`
    - Returns 0 after `shutdown` or EOF.
    - The `initialize` reply is `{"name": "nil-dj", "version": __version__}`.
    - Config is loaded with `data_dir` from `initialize`; `granted` is kept as `plugin.granted`.
    - On `initialized`, it builds the app via `plugin._start(conn, cfg, make_embedder, start_mcp)`. In this task that only sends a placeholder panel `{"elements": [{"type": "heading", "text": "nil-dj"}]}`; Tasks 7–10 extend `_start`.
    - On `shutdown`, it replies `{}` and calls each registered stop callback (`plugin.on_stop(fn)`), with 2 s in total. `serve()` returns.
  - `plugin.guard_stdout() -> TextIO`: returns the protocol stream and points everything else at stderr:
    ```python
    proto = os.fdopen(os.dup(1), "w", encoding="utf-8", newline="\n")
    os.dup2(2, 1)          # anything else writing to fd 1 (uvicorn, prints, C libs) lands in stderr
    sys.stdout = sys.stderr
    ```
  - `plugin.main() -> int`. It calls `guard_stdout()`.
    - stdin is reopened as UTF-8.
    - It runs `run(..., plugin_dir=Path.cwd(), make_embedder=default_embedder)` (`dj.default_embedder`, Task 7; until then, a local equivalent), then flushes stderr and calls `os._exit(code)` so worker threads (ffmpeg pool, uvicorn) can't hold the exit past nil's 3 s.
  - `FakeNil(tracks: list[dict], folders: list[str] = ["E:\\Music"])`: nil's side, with state
    - `state`, `queue`, `index`, `playlists: dict[str, list[str]]`;
    - `panels: list[dict]` (every `panel.set`), `calls: list[tuple[str, dict]]`;
    - `fail: dict[str, tuple[int, str]]` (method → error to return).
  - FakeNil methods:
    - `streams() -> (plugin_reader, plugin_writer)`, `start()`;
    - `handshake(granted=("playback", "library", "playlists"), data_dir=...) -> dict`;
    - `notify(method, params)`, `request(method, params) -> dict` (the raw reply);
    - `advance(index)`: sets `index`, sends `player.changed`;
    - `close()`: closes the plugin's stdin;
    - `wait_for(pred, timeout=5.0)`;
    - `element(id) -> dict` (from the last panel).
  - FakeNil behaviour matches PLUGIN-API.md:
    - `player.play`/`append` semantics, with `-32002` + `data.paths` for paths not in `tracks`;
    - `-32602` for empty `play`;
    - a `player.changed` after each play/append/advance;
    - `playlists.save` sanitises `<>:"/\|?*` to `_`.
    - Every `panel.set` is validated against the Panels table (types, required fields, unique ids, row depth ≤ 4); a bad panel fails the test via `AssertionError` raised in the test thread on the next `wait_for`.
- Fixtures in `conftest.py`:
  - `plugin_dir(tmp_path)`: `<tmp>/nil/plugins/nil-dj` with `data/`.
  - `fake_nil`: a `FakeNil` over a library of tracks made by `add_track` (see Task 7) or empty.
  - `running_plugin(fake_nil, plugin_dir)`: starts `plugin.run(..., make_embedder=lambda cfg: FakeEmbedder(), start_mcp=False)` in a thread, does the handshake, and yields `fake_nil`. On teardown it sends `shutdown` and joins (≤ 3 s). `plugin.run` builds the DJ with `lambda: make_embedder(cfg)`. From Task 7 on, tests that need an index first create `plugin_dir/data/index.sqlite` (model_id `"fake"`) with `add_track`.

- [ ] **Step 1: Write the failing tests** in `tests/test_plugin.py`:
  - `test_handshake_and_first_panel`:
    - the reply is `{"name": "nil-dj", "version": __version__}`;
    - after `initialized`, `fake_nil.panels[0]` has heading `nil-dj`.
  - `test_shutdown_returns_zero`: `request("shutdown", {})` → result `{}`; `run` returns 0 within 3 s.
  - `test_stdin_close_returns`: `fake_nil.close()` → `run` returns 0.
  - `test_guard_stdout` (subprocess: `python -c` calls `guard_stdout()`, then `print("stray")`, `os.write(1, b"raw\n")`, and writes `{}` plus a newline to the returned stream):
    - the child's stdout is exactly `"{}\n"`;
    - stderr contains `stray` and `raw`.
  - `test_subprocess_end_to_end` (real process, `sys.executable -m nil_dj plugin`, cwd = `plugin_dir`, env `NIL_DJ_MCP_PORT` = a free port):
    - handshake → `initialized` → the first `panel.set` arrives;
    - every stdout line parses as JSON-RPC;
    - `shutdown` → the process exits with 0 within 3 s.
  - `test_subprocess_exits_on_stdin_close`: close stdin → exit within 3 s (Review Focus 5).
- [ ] **Step 2: Run** `uv run pytest tests/test_plugin.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `plugin.py`, `FakeNil` and the fixtures. Add the `plugin` subcommand to `cli.py` (no flags; it calls `plugin.main()`).
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: PASS. No model is loaded in the subprocess tests, because there's no index.
- [ ] **Step 5: Commit** `feat: nil plugin entry point, handshake and shutdown`

---

### Task 7: The `DJ` service

**Files:**
- Create: `src/nil_dj/dj.py`
- Modify: `src/nil_dj/search.py` (`Hit.vector`, `rank`, `find_by_tags`, `describe_format`), `tests/fakes.py` (`add_track`)
- Test: `tests/test_dj.py`, `tests/test_search.py`

**Interfaces:**
- Consumes: `Config`, `Weights`, `order`, `candidate`, `run_index_paths`, `run_index`, `EmptyLibrary`, `Connection`, `NilError`, `models_present`.
- Produces in `search.py`:
  - `Hit(row, score, vector)`
  - `rank(index, q, limit, filters=None, exclude=frozenset()) -> list[Hit]`: `search_text` and the `similar_*` functions use it.
  - `find_by_tags(index, artist=None, album=None, title=None, limit=50) -> list[Row]`:
    - casefolded substring (`instr`);
    - `artist` matches `artist` or `album_artist`;
    - ordered by album artist, album, disc, number, path.
  - `describe_format(info: TrackInfo) -> str`: `"flac 24-bit/96 kHz"`, `"flac 16-bit/44.1 kHz"`, `"aac 44.1 kHz"`, `"dsd 2.8 MHz"` (sample_rate 2822400), `"dsd 5.6 MHz"`.
- Produces in `dj.py`:
  - `NOT_IN_NIL` (copy from Global Constraints); `class NotInNil(Exception)`.
  - `default_embedder(cfg: Config) -> Embedder`: `OnnxClapEmbedder(cfg.precision, cfg.gpu_device_id, models_dir=cfg.models_dir)`. Used by `plugin.main`, `serve` and `serve --stdio`.
  - `Track`: dataclass with `path, title, artist, album, duration_s, format, score: float | None`. `artist` = album_artist or artist; `title` falls back to the file stem.
    - `.item() -> dict` gives `{path, title, artist, duration_s}` for the panel.
    - `.as_dict()` gives all fields for MCP.
  - `Playlist`: `tracks: list[Track]`, `total_s: float`, `title: str`.
  - `IndexStatus`: `indexed: int`, `pending: int | None`, `failed: int`, `errors: list[tuple[str, str]]`, `running: bool`, `done: int`, `total: int`, `problem: str | None`.
  - `class DJ`, `DJ(cfg: Config, nil: Connection | None, make_embedder: Callable[[], Embedder], clock=time.monotonic)`, with members:
    - `model_state -> str`: one of `"not loaded"`, `"downloading"`, `"loading"`, `"ready"`, `"failed: <reason>"`. `"downloading"` when `models_present` is false at load start.
    - `embedder() -> Embedder`: lazy; one load at a time under a lock; calls `changed()` on each state change.
    - `preload() -> None`: loads on a daemon thread if the index exists.
    - `listeners: list[Callable[[], None]]` and `changed()`: notifies each listener; the panel subscribes.
    - `search_tracks(query, limit=20, filters: dict | None = None) -> list[Track]`
    - `find_tracks(artist=None, album=None, title=None, limit=50) -> list[Track]`
    - `similar_tracks(seed: str | dict, limit=20, filters=None) -> list[Track]`: `seed` is a path, `{"album", "artist"?}` or `"now_playing"`.
    - `build_playlist(prompt, minutes=45, filters=None, seeds=None, exclude=None) -> Playlist`
    - `more_like_playing(minutes=45) -> Playlist`
    - `play(paths, radio=False)`, `append(paths)`, `now_playing() -> dict`, `set_radio(on: bool)`
    - `save_playlist(name, paths) -> str`, `list_playlists() -> list[str]`, `play_playlist(name) -> int`
    - `index_status() -> IndexStatus`, `reindex(rebuild=False) -> bool`, `stop() -> None`
  - `filters_from(d: dict | None) -> Filters | None`: raises `ValueError("unknown filter 'x'; use …")` for unknown keys.

Rulings:
- **No index file:** `problem = "No index yet — press Index library."`. `IndexFormatError` → `problem = str(e)`. Search methods raise `ValueError(problem)`.
- **`build_playlist`:**
  - `q = embed_text(cfg.prompt_template.format(prompt))`;
  - with `seeds`, `q = normalise(q + normalise(mean(seed vectors)))`. Seeds steer only; they aren't forced in;
  - pool = `rank(..., candidate_pool_size(ceil(minutes*60/180)), filters, exclude)`;
  - then `order(..., target_s=minutes*60, weights=cfg.weights)` with `match` = the hit scores;
  - `title = f"{prompt} — {n} tracks, {m:d}:{s:02d}"`.
- **`more_like_playing`:**
  - `player.state` → `path`;
  - nothing playing → `ValueError("Nothing is playing.")`;
  - not indexed → `ValueError("The playing track isn't in the index yet — press Update index.")` (Review Focus 3);
  - the pool is similar to that track, with `match` = similarity to it and `history=[that track]`;
  - `title = f"More like {track title} — …"`.
- **Every nil-backed method** goes through `_nil()`, which raises `NotInNil` when `nil is None`. `NilError` propagates unchanged.
- **`play(paths, radio=True)`** calls `player.play`, then `set_radio(True)`. `set_radio` delegates to `self.radio` (Task 8). Until then, store the flag.
- **`play_playlist`** skips `missing` entries. With none left: `ValueError(f"Playlist {name!r} has no playable tracks.")`.
- **`index_status().pending`:**
  - in nil, `len(library paths − indexed − errored)`, where library paths are cached from the last `library.tracks` (refreshed by `reindex` and at start);
  - standalone, `None`.
- **`reindex`:**
  - returns False if running;
  - otherwise starts a daemon thread:
    - in nil, `library.tracks` → `run_index_paths`;
    - standalone, `run_index(cfg.folders)`;
  - updates `done/total` from progress and calls `changed()` (the panel throttles);
  - `EmptyLibrary` → `problem = "nil's library is empty — add a music folder in nil first."`;
  - `FfmpegMissing` → `problem = str(e)`;
  - with `rebuild`, first `Index.remove` under the lock;
  - `stop()` sets the stop event.

- [ ] **Step 1: Write the failing tests.** `add_track(index, path, *, artist, album, vector, duration_s=240, bpm=None, lufs=None, codec="flac", bits=16, sample_rate=44100, title=None)` in `fakes.py` upserts a row from `BASE`. `test_dj.py` uses a `dj` fixture: `DJ` over an index of 8 synthetic tracks (two Japanese paths), a `Connection` to a started `FakeNil` whose library holds the same paths, and `ChosenText` as the embedder.
  - `test_describe_format` (in `test_search.py`): the five strings above.
  - `test_find_by_tags`: substring, case-insensitive, `artist` matches album_artist.
  - `test_build_playlist_orders_and_fits`: `minutes=12` with 240 s tracks → 3 tracks; adjacent artists differ; total 720.
  - `test_build_playlist_filters`: `{"formats": ["dsd"]}` with no DSD rows → empty playlist, `total_s == 0`.
  - `test_unknown_filter`: `{"bpm": 3}` → `ValueError` naming `bpm`.
  - `test_seeds_steer`: the same prompt with `seeds=[p]` ranks p's neighbour higher than without.
  - `test_more_like_playing`:
    - FakeNil playing `p0` → result excludes `p0`, and the first track is `p0`'s nearest;
    - nothing playing → "Nothing is playing.";
    - an unindexed playing path → the exact "isn't in the index yet" message.
  - `test_play_and_append_reach_nil`: `fake.queue` equals the paths; Japanese paths are byte-identical (Review Focus 1).
  - `test_nil_error_passes_through`: `fake.fail["player.play"] = (-32003, "no audio device")` → `NilError` with that code and message.
  - `test_not_in_nil`: `DJ(cfg, None, …).play([...])` → `NotInNil` with `NOT_IN_NIL`. `search_tracks` still works.
  - `test_playlists`:
    - `save_playlist("a/b", paths)` → `"a_b"`;
    - `list_playlists()` contains it;
    - `play_playlist("a_b")` → plays them;
    - all missing → `ValueError`.
  - `test_index_status_and_reindex` (with real files from `make_audio` and `FakeEmbedder`):
    - before: `problem` says no index; `pending` == the library size;
    - `reindex()` → True; a second call while running → False;
    - after completion: `indexed` == n, `pending == 0`;
    - FakeNil with an empty library → `problem` is the exact empty-library message, and the rows are untouched.
  - `test_model_state_transitions`: a slow fake factory → `"loading"`, then `"ready"`; a raising factory → `"failed: boom"`.
- [ ] **Step 2: Run** `uv run pytest tests/test_dj.py tests/test_search.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** In `plugin._start`, create the `DJ`, refresh the library paths, and call `dj.preload()`. Register `dj.stop` with `on_stop`.
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: DJ service: search, playlists, playback through nil, indexing`

---

### Task 8: Radio

**Files:**
- Create: `src/nil_dj/radio.py`
- Modify: `src/nil_dj/dj.py` (`self.radio`, `set_radio`, `play(radio=True)`), `src/nil_dj/plugin.py` (route `player.changed` to `dj.radio.on_player_changed`)
- Test: `tests/test_radio.py`

**Interfaces:**
- Consumes: `order`, `candidate`, `Weights`, `Index.load`, `Connection.call`.
- Produces:
  - `Radio(open_index: Callable[[], Index], call: Callable[[str, dict], Any], weights: Weights, clock=time.time, rng: random.Random | None = None, on_change: Callable[[], None] = lambda: None)`
  - `on: bool`, `message: str | None`
  - `history: list[tuple[str, float]]` (path, started_at), newest last.
  - `set(on: bool) -> None`: on → `message = None`; calls `player.state` and tops up if the trigger holds.
  - `on_player_changed(params: dict) -> None`: always records history; tops up when on.
  - `pick(queue: list[str]) -> list[str]`: up to 5 paths. Pure apart from the index and the clock.

Rulings:
- **History** records `path` when `state == "playing"` and `path` differs from the last entry (pause/resume doesn't duplicate). It is kept even while radio is off, so turning radio on uses what was played.
- **Trigger:**
  - `state == "playing"`, `index is not None`, `index >= queue_len − 2`, and `queue_len != self._topped_up_at`;
  - after a successful append, set `_topped_up_at = queue_len + appended`.
  - This deviates from the spec's `queue_len − 1`: PLUGIN-API.md says append at the second-to-last track for gapless playback. Task 12 folds it into the spec.
- **Seed:**
  - weighted mean of the newest 5 history paths that are indexed, with weights 1.0, 0.8, 0.6, 0.4, 0.2 applied by position among the newest 5 played, skipping unindexed ones;
  - if none of the 5 is indexed, use the newest indexed path in all history;
  - if there is none, `on = False` and `message = "Radio stopped: none of the recently played tracks are in the index."` (Review Focus 3), then `on_change()`.
- **Exclusions:** paths played within 7200 s per `clock`, plus `queue`.
- **Pick:**
  - top 40 by seed similarity;
  - `order(count=5, history=<indexed history candidates, newest last>, top_k=3, rng)` with `match` = seed similarity;
  - fewer than 5 → whatever there is; none → no append.
- A `NilError` from `player.append` → `message = f"Radio: {e}"`, radio stays on, `on_change()`.
- `DJ.set_radio(on)` raises `NotInNil` outside nil.

- [ ] **Step 1: Write the failing tests** (FakeNil + synthetic index; `rng=random.Random(7)`; `clock` a mutable fake):
  - `test_tops_up_at_second_to_last`:
    - `dj.play([a, b, c], radio=True)` → no append while index 0;
    - `fake.advance(1)` → exactly one `player.append` of 5 paths, with none of a, b, c among them;
    - `advance(2)` → no second append (`_topped_up_at`).
  - `test_single_track_with_radio_tops_up_at_once`
  - `test_off_does_nothing_but_records_history`
  - `test_set_on_while_at_end_tops_up`
  - `test_excludes_recent_two_hours`: a track played at t=0 is not picked at t=3600; it can be at t=7300.
  - `test_seed_skips_unindexed`: newest played is not indexed → the seed comes from the others, and the append still happens.
  - `test_no_indexed_history_turns_off`: the exact message; `on` False; no append.
  - `test_append_error_kept_on`: `fake.fail["player.append"] = (-32003, "no audio device")` → message `"Radio: no audio device"`, `on` stays True.
  - `test_deterministic_with_seed`: the same rng seed → the same 5 paths.
- [ ] **Step 2: Run** `uv run pytest tests/test_radio.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** and wire it into `DJ` and `plugin._start`.
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: radio tops up the queue from what was played`

---

### Task 9: The panel

**Files:**
- Create: `src/nil_dj/panel.py`
- Modify: `src/nil_dj/plugin.py` (create a `Panel`, subscribe it to `dj.listeners` and `radio.on_change`, route `panel.event`)
- Test: `tests/test_panel.py`

**Interfaces:**
- Consumes: `DJ`, `IndexStatus`, `Playlist`, `Connection.call`.
- Produces:
  - `build(view: View) -> dict` (`{"elements": [...]}`), pure.
  - `View`: dataclass with `status: IndexStatus`, `model_state: str`, `prompt: str`, `result: Playlist | None`, `message: str | None`, `radio_on: bool`, `mcp: str`, `problems: list[str]`.
  - `Panel(dj: DJ, call: Callable, mcp_status: Callable[[], str], min_interval_s: float = 0.5)`
    - `refresh(force: bool = False)`: throttles non-forced refreshes to one per `min_interval_s`, sending the latest view after the interval.
    - `on_event(params: dict)`

Layout, top to bottom; ids and copy are exact:
1. `heading` `"nil-dj"`.
2. Each line of `problems` (config, Task 1) as `text` muted.
3. Index line:
   - while indexing, `progress` `{value: done/total, label: f"Indexing {done:,} of {total:,}"}`;
   - else `status.problem` or `f"{indexed:,} tracks indexed"` + `f" · {pending:,} not indexed yet"` if pending + `f" · {failed:,} failed"` if failed, as `text` muted.
4. Model line, only for downloading or loading:
   - `progress` `{value: null, label: "Downloading the sound model (once, about 620 MB)…"}` or `"Loading the sound model…"`;
   - failed → `text` `f"Sound model {model_state}"`.
5. Index buttons, in a `row`, not while running:
   - `button` `index`: `"Index library"` when no index, `"Update index"` when `pending > 0` or `failed > 0`;
   - `button` `rebuild` `"Rebuild index"` only when `problem` comes from `IndexFormatError`.
6. `input` `prompt`, placeholder `"Describe a playlist…"`, `submit_label` `"Make"`, `value` = the last prompt.
7. `row`: `button` `more` `"More like this"`; `toggle` `radio` `"Radio"` `value=radio_on`.
8. `message` (or `radio.message`) as `text`.
9. With a result: `heading` `result.title`, then `tracks` `[t.item() for t in result.tracks]`. An empty result → `text` `"No matching tracks."`.
10. `text` muted: `f"MCP: http://127.0.0.1:{port}/mcp"` or the unavailability reason (from `mcp_status()`).

Events (run on the notification worker):
- `prompt`:
  - an empty or whitespace value → message `"Type what you'd like to hear."`;
  - otherwise force a refresh (shows model loading), then `dj.build_playlist(value, 45)`.
- `more` → `dj.more_like_playing()`.
- `radio` → `dj.set_radio(value)`.
- `index` → `dj.reindex()`; `rebuild` → `dj.reindex(rebuild=True)`.
- `ValueError`, `NilError` and `NotInNil` become `message` (`str(e)`), never exceptions. The panel always refreshes after an event.
- Unknown ids are ignored.

- [ ] **Step 1: Write the failing tests:**
  - `build()` snapshots as dicts for:
    - no index;
    - indexing at 120/2765 (label `"Indexing 120 of 2,765"`);
    - 2,765 indexed with 12 pending and 3 failed;
    - model downloading;
    - a result of 2 tracks;
    - an empty result;
    - MCP unavailable;
    - a config problem.
  - Each built panel also passes FakeNil's validator (`validate_panel` exported from `fake_nil.py`).
  - Through `running_plugin`:
    - `notify("panel.event", {"id": "prompt", "value": "rain"})` → the last panel has a `tracks` element with ≥ 1 item and heading starting `rain — `;
    - whitespace prompt → the exact message;
    - `{"id": "more"}` while stopped → `"Nothing is playing."`;
    - `{"id": "radio", "value": true}` → the toggle value is true in the next panel;
    - `{"id": "index"}` → progress appears, then `"N tracks indexed"`;
    - a bad `config.toml` in `plugin_dir` → its problem line shows and the plugin still runs (Review Focus 4).
  - `test_throttle`: 50 non-forced refreshes in a tight loop → ≤ 3 `panel.set` calls, the last one carrying the final state.
- [ ] **Step 2: Run** `uv run pytest tests/test_panel.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** and wire it into `plugin._start`.
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: nil-dj panel in nil`

---

### Task 10: MCP server

**Files:**
- Create: `src/nil_dj/server.py`
- Modify: `pyproject.toml` (add `"fastmcp>=4.1,<5"`; `uv lock`), `src/nil_dj/plugin.py` (start the HTTP server when `start_mcp`), `src/nil_dj/cli.py` (`serve [--stdio]`)
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `DJ` and all its methods, `NOT_IN_NIL`, `NilError`.
- Produces:
  - `build_server(dj: DJ) -> FastMCP`, named `"nil-dj"` with `instructions` saying that tracks are identified by path. Tools, exactly the spec §8 table:
    - `search_tracks`, `find_tracks`, `similar_tracks`, `build_playlist`;
    - `play`, `append`, `now_playing`, `set_radio`;
    - `save_playlist`, `list_playlists`, `play_playlist`;
    - `index_status`, `reindex`.
  - Each tool is a sync function with a docstring (the client sees it), returning plain dicts or lists. Tracks use `Track.as_dict()`.
    - `build_playlist` → `{"tracks": [...], "total_minutes": round(total_s/60, 1)}`.
    - `reindex` → `{"started": bool}`.
  - Errors become `ToolError`:
    - `NotInNil` → `NOT_IN_NIL`;
    - `NilError` → `f"nil: {e.message}"`;
    - `ValueError` → its text.
  - `class McpHttp`, `McpHttp(server: FastMCP, port: int)`:
    - `start()`: if `port_free(port)` is false, `status = f"MCP unavailable: port {port} in use"` and it doesn't start. Otherwise it runs `server.run(transport="http", host="127.0.0.1", port=port, show_banner=False, uvicorn_config={"access_log": False})` on a daemon thread. A `SystemExit` or exception in that thread sets the same port message (or `f"MCP unavailable: {e}"`) and calls `on_change`.
    - `status: str`, default `f"MCP: http://127.0.0.1:{port}/mcp"`.
    - `on_change` callable.
  - `port_free(port) -> bool`: tries to bind `127.0.0.1:port` with a plain socket (no `SO_REUSEADDR`).
  - `serve_stdio(cfg: Config) -> int`:
    - if a TCP connect to `127.0.0.1:cfg.mcp_port` succeeds within 0.5 s, `create_proxy(f"http://127.0.0.1:{port}/mcp").run(transport="stdio", show_banner=False)`;
    - else `build_server(DJ(cfg, None, lambda: default_embedder(cfg))).run(transport="stdio", show_banner=False)`.
  - `nil-dj serve`: standalone HTTP on `cfg.mcp_port` with `DJ(cfg, None, …)`; blocks.

- [ ] **Step 1: Write the failing tests** (`asyncio.run` around `fastmcp.Client(build_server(dj))`; `dj` fixture from Task 7):
  - `test_tools_listed`: names equal the 13 above.
  - `test_build_playlist_tool`: returns tracks and `total_minutes`; filters pass through.
  - `test_play_tool_reaches_nil`: `fake.queue` updated; `radio=True` turns radio on.
  - `test_errors_reach_client`:
    - with `DJ(cfg, None, …)`, `play` → `ToolError` text `NOT_IN_NIL`;
    - FakeNil `-32002` → `"nil: not in nil's library: 1 path(s)"`;
    - unknown filter → the `ValueError` text.
  - `test_busy_port_message`: hold a bound socket on a port, `McpHttp(...).start()` → status `"MCP unavailable: port N in use"`; nothing raised.
  - `test_http_roundtrip`: on a free port, `McpHttp.start()`, then `Client(f"http://127.0.0.1:{port}/mcp")` calls `list_playlists` → the FakeNil playlists.
  - `test_mcp_port_env`: `NIL_DJ_MCP_PORT` set → the server status shows that port (through `load_config`).
  - `test_panel_shows_mcp_status` (in `running_plugin` with `start_mcp=True` on a busy port): the last panel's final text is the unavailable message.
  - `test_subprocess_stdout_clean_with_http` (extends Task 6's subprocess test): after `initialized`, make one HTTP request to the plugin's MCP port (uvicorn logs it) → every stdout line still parses as JSON-RPC.
- [ ] **Step 2: Run** `uv run pytest tests/test_server.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement**; `uv add "fastmcp>=4.1,<5"`; wire `McpHttp` into `plugin._start` with `on_change=panel.refresh`.
- [ ] **Step 4: Run** `uv run pytest -q && uv run ruff check .`. Expected: PASS. The Task 6 subprocess test still sees only JSON-RPC on stdout with uvicorn running.
- [ ] **Step 5: Commit** `feat: MCP server in the plugin, stdio bridge and standalone serve`

---

### Task 11: `nil-dj install`

**Files:**
- Create: `src/nil_dj/install.py`
- Modify: `src/nil_dj/cli.py` (`install --nil PATH`)
- Test: `tests/test_install.py`

**Interfaces:**
- Consumes: `legacy_index_path()` (Task 1), `__version__`.
- Produces:
  - `class InstallError(Exception)`
  - `install(nil_home: Path, repo_root: Path, run: Callable[[list[str], Path], None] = run_checked, legacy_index: Path | None = None, windows: bool = sys.platform == "win32") -> Path`: returns the plugin folder.
  - `run_checked(cmd, cwd)`: `subprocess.run(check=True)`; a `CalledProcessError` becomes `InstallError` with the command and its stderr tail.
  - `plugin_toml(version: str, windows: bool) -> str`
  - `DEFAULT_CONFIG_TOML: str`: every key from Task 1, commented, with defaults.
  - `repo_root() -> Path`: `Path(__file__).parents[2]` if `pyproject.toml` there names `nil-dj`, else `InstallError("Run nil-dj install from a nil-dj checkout (uv run nil-dj install --nil …).")`.

Steps `install` performs, in order:
1. Check that `nil_home/"nil.exe"` exists (on non-Windows, `nil_home` exists), else `InstallError(f"{nil_home} has no nil.exe")`. Check `uv` on PATH, else `InstallError("uv is needed: winget install astral-sh.uv")`.
2. `p = nil_home/"plugins"/"nil-dj"`; create `p/"data"`.
3. If `p/"app"` exists, rename it to `p/"app.old"` (an existing `app.old` is removed first).
   - `PermissionError`/`OSError` → `InstallError("nil-dj is running in nil: switch it off in Settings → Plugins, then run install again.")`.
   - Then `shutil.rmtree(app.old)`.
4. Run, with `cwd=repo_root`:
   - `uv export --frozen --no-dev --no-emit-project --no-hashes -o <p>/app/requirements.txt`;
   - `uv venv --python 3.12 <p>/app/.venv`;
   - `uv pip install --python <py> -r <p>/app/requirements.txt`;
   - `uv pip install --python <py> --no-deps <repo_root>`.
   - `<py>` is `app\.venv\Scripts\python.exe` on Windows and `app/.venv/bin/python` elsewhere.
5. Write `plugin.toml` (always):
   - `name = "nil-dj"`, `version`, `description = "AI DJ: playlists from a description, more like this, radio"`, `api_version = 1`, `permissions = ["playback", "library", "playlists"]`;
   - `command = ["app\\.venv\\Scripts\\python.exe", "-m", "nil_dj", "plugin"]`, or `["app/.venv/bin/python", "-m", "nil_dj", "plugin"]` when not `windows`.
6. Write `config.toml` only if absent.
7. If `legacy_index` exists and `p/"data"/"index.sqlite"` does not, run a WAL checkpoint on the legacy file (`sqlite3` `PRAGMA wal_checkpoint(TRUNCATE)`), then move `index.sqlite` into `data\`, deleting the `-wal`/`-shm`/`.lock` siblings. If both exist, leave the legacy one and print a note.
8. Print the next steps: `f"Installed nil-dj {version} in {p}. In nil, open Settings → Plugins and switch nil-dj on."` If `ffmpeg` isn't on PATH, add `"ffmpeg isn't on PATH; indexing needs it: winget install Gyan.FFmpeg"`.

- [ ] **Step 1: Write the failing tests** (a `run` fake records commands and, for the `uv venv` command, creates the python path):
  - `test_fresh_install`:
    - the command list equals the four commands above, in order;
    - `plugin.toml` parses with `tomllib` and has the exact fields (both `windows=True` and `False` variants);
    - `config.toml` written; `data/` exists.
  - `test_reinstall_keeps_data_models_config`: pre-create `data/index.sqlite`, `models/x`, a custom `config.toml` and `app/old.txt` → the first three are unchanged; `app/old.txt` is gone.
  - `test_app_in_use`: monkeypatch `Path.rename` for `app` to raise `PermissionError` → the exact `InstallError`; `app/` is untouched.
  - `test_moves_legacy_index`: a legacy index with a row in WAL mode → after install, `data/index.sqlite` opens with that row; the legacy file and its siblings are gone.
  - `test_legacy_kept_when_data_exists`
  - `test_no_nil_exe`: the exact message.
  - `test_cli_install` (`main(["install", "--nil", str(home)])` with `install.run_checked` monkeypatched) → 0, and the plugin folder exists.
  - `@pytest.mark.slow test_real_install`: a real `uv` into a tmp home → `<py> -m nil_dj --version` prints the version.
- [ ] **Step 2: Run** `uv run pytest tests/test_install.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q`, then `uv run pytest -m slow tests/test_install.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: nil-dj install puts the plugin next to nil`

---

### Task 12: Docs, version, and validation on Windows

**Files:**
- Modify:
  - `src/nil_dj/__init__.py` (`__version__ = "0.2.0"`); `tests/test_cli.py` (the version string);
  - `src/nil_dj/index.py:1-2` (docstring: the index is private to nil-dj, not shared with nil);
  - `docs/superpowers/specs/2026-10-22-nil-dj-design.md`;
  - `README.md`; `docs/validation-log.md`.

- [ ] **Step 1: Fold the plan's rulings into the spec:**
  - §7 radio trigger = `index ≥ queue_len − 2` (gapless, per PLUGIN-API.md).
  - §7 length rule: overshooting candidates are skipped for that step.
  - `build_playlist` seeds steer the query vector.
  - The pool assumes 180 s per track.
  - Indexing refuses an empty nil library.
  - `host.call` 10 s timeout.
  - `install` uses `uv export` + `uv pip install` from the lock, and the `app.old` rename check.
  - `stdout` guard (fd 1 → stderr).
- [ ] **Step 2: Update the README.** Change "Work in progress" to:
  - an install section: build nil; `uv run nil-dj install --nil E:\Projects\nil`; switch it on in nil;
  - MCP client config snippets for Claude Code (`claude mcp add --transport http nil-dj http://127.0.0.1:47263/mcp`), Claude Desktop and Cursor (stdio `nil-dj serve --stdio`).
  - Keep it short; Plan 4 writes the showcase README.
- [ ] **Step 3: Run** `uv run pytest -q && uv run ruff check .`. Expected: PASS. **Commit** `docs: nil-dj as a plugin: spec rulings, install, MCP setup`
- [ ] **Step 4: Validate on Windows by hand.** Ask the owner before each outward step.
  1. Ask the owner to package the current nil main into `E:\Projects\nil` (audio-rea `package-nil.ps1`) and start it once, so `data\` is migrated.
  2. `scripts/sync-to-windows.sh`; on Windows in `E:\projects\nil-dj`: `uv run pytest -q`.
  3. `uv run nil-dj install --nil E:\Projects\nil`. Check that the old `%LOCALAPPDATA%\nil-dj\index.sqlite` moved to `plugins\nil-dj\data\`.
  4. Owner: switch nil-dj on in Settings → Plugins, open its tab:
     - the status shows 2,8xx tracks indexed;
     - the first prompt downloads the model to `models\<rev>\`;
     - a prompt gives a playlist; Play all plays bit-perfect;
     - Save as playlist;
     - More like this;
     - radio past the end of a playlist (gapless append);
     - Update index after adding an album.
  5. From WSL or Windows, `claude mcp add --transport http nil-dj http://127.0.0.1:47263/mcp` → ask for "20 minutes of something calm" → nil plays it.
  6. Kill `nil.exe` in Task Manager → no stray `python.exe`.
  7. Record the results in `docs/validation-log.md` (date, timings, owner verdicts verbatim), and ask the owner to delete the old Hugging Face cache copy of the model.
- [ ] **Step 5: Commit** `docs: Windows validation of the nil-dj plugin`
