# nil-dj — design

An AI DJ for the **nil** bit-perfect music player. You ask any MCP-capable agent for music in plain
language ("45 minutes of mellow instrumental for coding at night"); nil-dj finds matching tracks in
your own library **by how they sound**, orders them so they flow, and plays them in nil. When the
playlist ends, **radio mode** keeps going on its own, with no AI involved.

nil-dj is a **plugin** of nil: plain nil stays only a player, and nil-dj is an optional add-on that
nil starts, supervises and shows a panel for. nil's side (plugin host, protocol, panel, playlists,
`Append`, portable data next to `nil.exe`) is specified in the nil repo:
`audio-rea/docs/superpowers/specs/2026-10-23-plugins-and-playlists-design.md` (§ references to it
below are written "nil §n").

Agreed in brainstorming on 2026-10-22; revised 2026-10-23 for the plugin design (supersedes the
HTTP control API, radio in nil and the shared ordering fixture).

## 1. Goals

| Goal | Meaning |
|---|---|
| **Daily driver** | The owner uses it every day with their own library, so indexing, re-indexing and playback must be robust. |
| **Showcase** | A public portfolio project for job-seeking: clear README, architecture diagram, measured evaluation, demo video. |

**Success criteria**

- A plain-language request from an MCP client starts a fitting, well-ordered playlist in nil, played bit-perfect.
- Radio continues playback with no AI client involved, as long as the nil-dj plugin is enabled.
- From nil's own panel, without any MCP client: a prompt playlist, more like this, radio.
- Works with any MCP client (Claude Code, Claude Desktop, Cursor, VS Code, …).
- An evaluation report shows that sound search beats tag-only and random baselines, and that ordering beats random order.

## 2. Scope

**In v1**

1. Build and play a playlist from a prompt.
2. "More like this": similar to the playing track, a track or an album.
3. Smooth ordering (sound similarity, tempo and loudness transitions, artist spacing).
4. Saved playlists (`.m3u8`), stored by nil.
5. Radio mode in the plugin: offline, and seeded by the AI (play a playlist with radio on).
6. A panel inside nil (prompt box, more like this, radio toggle, results).

**Out of v1 (later)**

- AI-steered radio ("drift it toward jazz" while it plays).
- Library questions that need play history ("most played"); nil records no history yet.
- Learning taste from plays and skips.
- Any downloader or store integration. **Never** a Tidal/Qobuz downloader (terms of service,
  anti-circumvention, and a liability for a public portfolio).

## 3. Context: the owner's setup (measured 2026-10-22)

| Item | Value |
|---|---|
| OS | Windows (nil is Windows-only, WASAPI exclusive); development happens in WSL |
| GPU | NVIDIA RTX 3060 Laptop, 4 GB VRAM; Ryzen 5 5600H; 15 GB RAM |
| Library | `E:\Music`, 2,765 tracks: 2,520 FLAC, 134 M4A, 97 DSF, 13 WAV, 1 DFF |
| nil config dir | `%APPDATA%\bitperfect\` (`library.toml` holds `folders = ['E:\Music']`); becomes `E:\Projects\nil\data\`, next to `nil.exe` (nil §3) |
| nil library | In-memory `LibraryIndex` built from a folder scan, cached as `library-cache.json`; no database |
| nil playback | `bp_engine::Command::Play { tracks, start }` replaces the queue; there is no append |

## 4. Architecture

nil starts nil-dj as a child process and talks to it with JSON-RPC over stdio (nil §6). nil-dj
serves MCP to agents from the same process.

```
MCP clients ──HTTP 127.0.0.1:47263 (or stdio)──► nil-dj plugin process (Python)
                                                  ├─ indexer ── CLAP ──► data\index.sqlite
                                                  ├─ search / order
                                                  ├─ radio
                                                  ├─ panel
                                                  └─ host ◄──JSON-RPC over stdio──► nil (Rust)
                                                                                    ├─ plugin host
                                                                                    ├─ panel drawing
                                                                                    ├─ playlists (.m3u8)
                                                                                    └─ engine (Append)
```

- **nil-dj** owns everything ML and DJ: decoding for analysis, CLAP embeddings, search, ordering,
  radio, the panel's contents, MCP.
- **nil** owns playback, playlists, the plugin lifecycle and drawing the panel. It has no DJ logic
  and never reads nil-dj's index.
- Both run natively on Windows. Code lives in WSL at `~/projects/nil-dj` and is synced to
  `E:\projects\nil-dj` with `rsync --delete` for Windows runs.

### Files (`<nil home>\plugins\nil-dj\`, e.g. `E:\Projects\nil\plugins\nil-dj\`)

```
plugin.toml        manifest (nil §6.1): permissions playback, library, playlists
config.toml        mcp_port, precision, gpu_device_id, ordering weights
app\               nil-dj and its Python environment (app\.venv)
data\index.sqlite  the index (§6); `data_dir` from initialize
models\<revision>\ the CLAP files for the chosen precision only
```

Deleting the folder removes nil-dj completely. `nil-dj install` (§9) rewrites `plugin.toml` and
`app\` and never touches `data\`, `models\` or `config.toml`. The model and index therefore
live on the same drive as nil.

**Finding the plugin folder:** as a plugin, the working directory nil sets plus `data_dir` from
`initialize`. Run by hand from the installed environment (`app\.venv\Scripts\nil-dj.exe index`),
the folder above `app\`. From a development checkout, `NIL_HOME` (nil's home, as for nil itself)
→ `<NIL_HOME>\plugins\nil-dj\`. `NIL_DJ_INDEX` still overrides the index path for tests.

### Data flow

- **Panel:** prompt typed in nil → `panel.event` → search → order → `panel.set` with a `tracks`
  list; the owner presses Play all (nil acts directly).
- **Chat:** agent → MCP tool → search (`index.sqlite`) → order → `player.play` → nil
  `PlayerCommand::Play` → WASAPI (bit-perfect).
- **Radio:** `player.changed` shows the last queued track started → pick 5 tracks →
  `player.append`.

## 5. nil-dj components

| Module | Responsibility |
|---|---|
| `config` | Plugin folder, `config.toml`, index path, model path; env overrides. Music folders come from nil (`library.folders`) when running as a plugin, or from `<nil home>\data\library.toml` for the standalone CLI. |
| `embedder` | Interface `embed_audio(clips) -> vec512`, `embed_text(str) -> vec512`. Backend: **ONNX Runtime**, no PyTorch at runtime (§5.1). |
| `indexer` | As a plugin, indexes exactly the paths from `library.tracks`; standalone, walks the folders; decodes with ffmpeg to 48 kHz mono float (DSF/DFF included); takes three 10 s clips centred at 20 %, 50 % and 80 % of the track (one clip centred in the middle for 10–30 s tracks; the whole track, repeat-padded, for tracks up to 10 s), decoding only those segments with ffmpeg seeks so long tracks never sit in memory; embeds each, averages, L2-normalises. Measures tempo (BPM, from the middle 60 s) and loudness (integrated LUFS, ffmpeg's `ebur128` filter over the whole track, streamed). Walks the same files nil does: extensions `flac wav aif aiff m4a dsf dff` (case-insensitive), symlinks skipped. Reads tags with mutagen. Incremental by size + mtime; commits after each track (resumable); deletes rows for files that no longer exist; refuses an empty library (nil's, or the folders walked standalone, so an unplugged drive never wipes the index); single instance via a lock file. Standalone (`nil-dj index`, or the MCP `reindex` without nil) walks the music folders itself and refuses an empty or unplugged library, also with `--rebuild`, before anything is deleted: it prints "Nothing was changed: no tracks found in the music folders (is the drive plugged in?)" to stderr and exits 5. |
| `index` | Schema, migrations, reads and writes of `index.sqlite`. |
| `search` | Text → CLAP text vector → SQL filters → brute-force cosine over the remaining rows (numpy). Also similarity to a track, an album (mean vector) or the playing track. |
| `order` | The ordering algorithm in §7. |
| `radio` | Radio state and selection (§7). |
| `host` | JSON-RPC over stdio with nil: handshake, requests (`player.*`, `library.*`, `playlists.*`, `panel.set`), notifications in, `shutdown`. A request times out after 10 s. |
| `panel` | Builds the panel (nil §7) from the current state; handles `panel.event`. |
| `server` | FastMCP server, Streamable HTTP on `127.0.0.1:47263/mcp` by default (`mcp_port` in `config.toml` or `NIL_DJ_MCP_PORT`); `nil-dj serve --stdio` for stdio-only clients is a thin bridge to the plugin's HTTP endpoint, so those clients control the same nil; if the plugin isn't running, it serves search-only tools itself (§10). It binds 127.0.0.1 only. If the port is taken it does not move to another port; the panel says "MCP unavailable: port N in use" and nothing crashes. `stdout` is the protocol stream when run by nil, so the plugin points fd 1 at stderr and keeps the protocol on a private copy of the original stdout; stray prints, uvicorn and C libraries cannot corrupt it. Ports chosen to avoid Windows' ephemeral range (≥ 49152), the owner's reserved ranges and common dev-tool ports. |
| `eval` | Prompt set, rating CLI, metrics, report generation (§12). |
| `cli` | `nil-dj plugin` (what nil runs), `nil-dj install`, `nil-dj index`, `nil-dj serve`, `nil-dj search "<text>"`, `nil-dj eval …`. |

Tooling: Python 3.12, `uv` (needed by `nil-dj install`; not at run time), `pytest`, `ruff`. Licence: MIT.

### 5.1 Embedding backend (ONNX Runtime)

- **Model files:** the ONNX export of `laion/clap-htsat-unfused` published as
  `Xenova/clap-htsat-unfused` (`onnx/audio_model*.onnx`, `onnx/text_model*.onnx`, tokenizer and
  preprocessor configs), downloaded with `huggingface_hub` on first use into `models\<revision>\`
  (`local_dir`, not the shared Hugging Face cache), pinned to a revision, only the files the chosen
  precision needs.
  Sizes: audio 118 MB, text 502 MB at fp32; fp16 about half.
- **Precision:** `fp32` by default; `fp16` and `int8` selectable (`--precision`). The eval reports
  the quality cost of each (§12).
- **Execution provider:** `onnxruntime-directml` (version pinned; DirectML is in maintenance mode)
  on Windows, so any DirectX 12 GPU works (NVIDIA, AMD, Intel). `gpu_device_id` is a config value;
  by default nil-dj enumerates the DXGI adapters and picks the one with the most dedicated video
  memory (adapter 0 on a hybrid laptop is often the integrated GPU). The indexer prints the
  adapter it uses, because the owner's laptop has both an NVIDIA and an AMD GPU. If DirectML is unavailable or fails to create a session, fall back to the CPU
  provider with a warning. `int8` runs on CPU by default, since DirectML's support for dynamically quantised ops is
patchy; step 1 checks whether DirectML can run it.
- **Preprocessing without PyTorch:** `transformers` 5.x makes `ClapFeatureExtractor` require
  PyTorch, so nil-dj computes the log-mel input itself in numpy (`features` module), matching the
  extractor's non-fusion path: Slaney mel scale and norm, 64 bins, 50–14,000 Hz, n_fft 1024,
  hop 480, Hann window, power 2, dB, giving a (1001, 64) array; clips shorter than 10 s are
  repeat-padded. Text uses the `tokenizers` library on `tokenizer.json`. The ONNX text model takes
  only `input_ids` (no attention mask), so prompts are embedded one at a time, unpadded, truncated
  to 512 tokens. Clips are exactly 10 s, so preprocessing is deterministic.
- **Measured 2026-10-22 (WSL, CPU):** numpy features vs the PyTorch extractor differ by at most
  4e-6; ONNX vs PyTorch `ClapModel` embeddings have cosine 1.0000 (fp32), 1.0000 (fp16), 0.996
  (int8 audio) and 0.995 (int8 text); about 110 ms per 10 s clip on CPU.
- **Install size:** about 650 MB (fp32) or 350 MB (fp16) including the model, versus about 3 GB
  for a PyTorch + CUDA install.
- **Faithfulness:** PyTorch is a **dev-only** dependency, used by a parity test (§11) that compares
  ONNX vectors with the original PyTorch `ClapModel`.

## 6. The index (`plugins\nil-dj\data\index.sqlite`)

Private to nil-dj. SQLite in **WAL mode** so search can read while the indexer writes.

```sql
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
-- keys: schema_version ("1"), model_id ("laion/clap-htsat-unfused"), dim ("512")

CREATE TABLE tracks (
  path          TEXT PRIMARY KEY,   -- Windows path exactly as nil sees it, e.g. E:\Music\...
  size          INTEGER NOT NULL,
  mtime_ns      INTEGER NOT NULL,
  title         TEXT, artist TEXT, album_artist TEXT, album TEXT, genre TEXT,
  year          INTEGER, disc INTEGER, number INTEGER,
  duration_s    REAL NOT NULL,
  codec         TEXT NOT NULL,      -- flac | alac | aac | wav | aiff | dsd
  sample_rate   INTEGER NOT NULL,
  bits          INTEGER,            -- NULL for DSD and lossy
  bpm           REAL,               -- NULL when no stable tempo was found
  lufs          REAL,
  embedding     BLOB NOT NULL,      -- 512 little-endian float32, L2-normalised
  indexed_at    INTEGER NOT NULL    -- unix seconds
);

CREATE TABLE index_errors (path TEXT PRIMARY KEY, error TEXT NOT NULL, at INTEGER NOT NULL);
```

- **Hi-res** means `bits > 16` or `sample_rate > 48000`, or `codec = 'dsd'`.
- Paths are stored as given, but compared by a key: `\` becomes `/`, then casefold. A row and
  nil's path that differ only in case or slashes are the same track: it is not pending, an
  update keeps the row (and updates it under its stored path), and seeds and radio find it.
- A different `schema_version` major number, or a file that cannot be read, means a full re-index. The panel then offers only **Rebuild index**; **Update index** is hidden, since updating cannot help.
- Brute-force search is deliberate: 2,765 × 512 float32 ≈ 5.7 MB, scanned in milliseconds. No ANN
  index or vector database is needed at this scale; the README documents the measurement.

## 7. Selection and ordering

### Search

1. `q = embed_text(prompt)`; whether to wrap the prompt as `"This is a sound of {prompt}"` is
   decided by the eval (§12).
2. Apply filters in SQL.
3. Score = cosine(q, track). Keep a candidate pool of the top `max(60, 3 × tracks needed)`, assuming 180 s per track to size it. Use
   rank, never a fixed score threshold (CLAP text-audio scores are small, ≈ 0.1–0.4).

### Ordering (shared by playlists and radio)

Greedy. Start with the best-matching candidate, then repeatedly append the candidate `t` with the
highest

```
score(t) = 0.60 · sim(prev, t)
         + 0.40 · match(t)
         − 0.15 · tempo_jump(prev, t)
         − 0.10 · loud_jump(prev, t)
```

- `sim` = cosine of track vectors. `match` = cosine to the prompt vector (playlists) or to the
  seed vector (radio).
- `tempo_jump = min(|a − b|, |2a − b|, |a − 2b|) / 20` (half and double tempo count as close);
  0 when either BPM is NULL.
- `loud_jump = |ΔLUFS| / 6`; 0 when either is NULL.
- **Hard rules:** never the same artist (album artist, else artist) twice in a row; the same album
  at most once in any 3 consecutive tracks; no repeats. If no candidate satisfies the rules, relax
  the album rule first, then the artist rule.
- **Ties** break by path (byte order of the UTF-8 string), so results are exact.
- **Length:** stop when the total reaches the target; accept the next track only if the total stays
  within target + 5 %; a candidate that would overshoot is skipped for that step, and a shorter one may follow. The weights are config values with these defaults, tuned by the eval.

### Radio (in the plugin)

- **On/off:** the panel's Radio toggle, MCP `set_radio`, or `play(paths, radio=true)`. Radio runs
  while the plugin runs; it is off after nil restarts.
- **Trigger:** radio is on and `player.changed` reports `index ≥ queue_len − 2`, so the append lands
  before the last track starts and playback stays gapless (PLUGIN-API.md). nil-dj asks nil for
  `player.state`, picks 5 tracks and sends `player.append`. Radio tops up at most once per queue
  state: it remembers the queue contents from before the append, so the longer queue triggers
  again as it runs down, and a new queue of the same length still tops up. With no index, radio
  switches itself off and shows the index problem. When the connection to nil closes, radio sets
  no message.
- **Seed vector:** weighted mean of the last 5 played tracks (weights 1.0, 0.8, 0.6, 0.4, 0.2,
  newest first), L2-normalised. Tracks missing from the index are skipped; if none of the last 5
  are indexed, use the most recent indexed track played; if there is none, radio turns off and says
  why.
- **Excluded:** tracks played in the last 2 hours (in-memory history for the session) and tracks
  already in the queue. Tracks the current device cannot play are skipped by nil's engine, as for
  any queue.
- **Choice:** top 40 by similarity to the seed, then the ordering above with `match` = seed
  similarity. Each step picks uniformly at random among the **top 3** scores so radio varies;
  tests fix the RNG seed. Fewer than 5 candidates: append what there is; none: let playback end.

## 8. Interfaces

### MCP tools (nil-dj)

Tracks are identified by their file path.

| Tool | Behaviour |
|---|---|
| `search_tracks(query, limit=20, filters?)` | Sound search. Returns path, artist, title, album, duration, format, score. |
| `find_tracks(artist?, album?, title?, limit=50)` | Case-insensitive substring match on tags. |
| `similar_tracks(seed, limit=20, filters?)` | `seed`: a track path, `{"album": ..., "artist": ...}`, or `"now_playing"`. |
| `build_playlist(prompt, minutes, filters?, seeds?, exclude?)` | Search + ordering; returns the ordered list and total length. Seeds steer the query vector (the prompt vector plus the seeds' mean, normalised); they are not forced into the result. Does not play. |
| `play(paths, radio=false)` | Plays in nil; with `radio=true` radio is switched on (AI-seeded radio). |
| `append(paths)` | Appends to what is playing, gaplessly. |
| `now_playing()` | Track, state, queue, radio state (from `player.state`). |
| `set_radio(on)` | Radio on or off. |
| `save_playlist(name, paths)`, `list_playlists()`, `play_playlist(name)` | Through nil (`playlists.*`, then `player.play`). |
| `index_status()`, `reindex()` | Counts, unindexed files, errors, progress; `reindex` runs in the background. |

**Filters:** `min_minutes`, `max_minutes`, `year_from`, `year_to`, `artists`, `exclude_artists`,
`formats` (`flac`, `alac`, `aac`, `wav`, `aiff`, `dsd`), `hires_only`. Sound qualities
(instrumental, vocals, acoustic) belong in the query text, not in filters.

### nil plugin API

nil-dj uses `player.state`, `player.play`, `player.append`, `library.folders`, `library.tracks`,
`playlists.list`, `playlists.read`, `playlists.save`, `panel.set`, and receives `player.changed`,
`panel.event` and `shutdown` (nil §6.4).

### Panel

- Index status line, or a `progress` bar while indexing; **Index library** when there is no index; **Update index** whenever an index exists and no run is going, even with nothing pending (nil rescans its folders without telling plugins, so nil-dj re-reads `library.tracks` at most every 30 s, on a panel event, and the pending count can lag); only **Rebuild index** when the index format is unreadable.
- Panel actions run on their own thread, one at a time, so a search waiting for the model never delays radio. An unexpected error shows "Something went wrong: <type>. Details are in nil's plugin log."
- `input` "Describe a playlist…" → **Make** (45 min by default).
- **More like this** (seeded from the playing track) and a **Radio** toggle.
- The result as a `tracks` list (nil supplies Play all / Queue all / Save as playlist).
- The MCP address, `http://127.0.0.1:<port>/mcp`, or why MCP is unavailable.

## 9. Install

`nil-dj install --nil E:\Projects\nil` (run once from the repo): checks that the folder holds
`nil.exe`, creates `plugins\nil-dj\` there, builds `app\.venv` with `uv` (`uv export` from the lock file, then `uv pip install` of those pinned requirements and nil-dj itself), writes
`plugin.toml` with `command = ["app\\.venv\\Scripts\\python.exe", "-m", "nil_dj", "plugin"]`, and keeps
`data\`, `models\` and `config.toml` if present. The owner then turns the plugin on in nil's
Settings → Plugins. ffmpeg must be on `PATH`, as for the indexer today. A frozen `.exe` (no uv
needed) is later work.

Re-running install renames the existing `app\` to `app.old` first, which fails while nil has nil-dj running (Windows holds the files), so a running plugin is reported with "switch it off" rather than half-replaced; `app.old` is then deleted.

The current standalone data (`%LOCALAPPDATA%\nil-dj\index.sqlite`) is moved into `data\` by
`nil-dj install`, but only after a clean WAL checkpoint (busy = 0). On any failure, or when `data\index.sqlite` already exists, the old index is left in place with a note and the install still succeeds. The old Hugging Face cache copy of the model is left for the owner to delete.

## 10. Errors

| Situation | Behaviour |
|---|---|
| nil-dj started outside nil (`nil-dj serve`) | Search and `build_playlist` work; playback and playlist tools return "nil-dj isn't running inside nil — enable it in nil's Settings → Plugins." |
| nil returns `-32001` / `-32002` / `-32003` | Passed to the MCP client and the panel with nil's message. |
| Empty library (nil has no tracks, or the walked folders hold none) | Indexing is refused and no rows are deleted. Panel: "nil's library is empty — add a music folder in nil first."; `nil-dj index`: the message above, exit 5. |
| Index unreadable or from another schema | The panel shows the problem and offers only **Rebuild index** (no **Update index**). |
| MCP port in use | Panel shows "MCP unavailable: port N in use"; nothing crashes, panel and radio keep working. |
| stdin closes (nil gone) | Exit at once. |
| ffmpeg missing | The indexer stops at start with an install hint. |
| A file fails to decode | Recorded in `index_errors`; indexing continues; `index_status` lists them. |
| No usable DirectML GPU | CPU fallback with a warning. |
| First run | The ONNX model files (≈ 620 MB fp32, ≈ 310 MB fp16) download once to `models\`; offline afterwards. |
| Files moved or deleted | Rows removed on the next index run; nil rejects unknown paths (`-32002`). Until then, in nil, a row whose path isn't in nil's library is never chosen: playlists, search, find, similar tracks and radio skip it. Standalone there is no library to check against. |
| nil refuses radio's `player.append` | Radio shows "Radio: <nil's message>" and stays on; it doesn't try again until the queue changes. |
| Second indexer | Refused by the lock file. |
| Index locked or can't be opened (SQLite `OperationalError`) | "The index is busy (another nil-dj is using it). Try again in a moment." **Rebuild index** is not offered for it. |
| Rebuild while another process holds the index file | "The index is in use; try Rebuild again in a moment."; **Rebuild index** stays. |

## 11. Testing

**nil-dj (pytest)**

- `order` on small synthetic cases (vectors, BPM, LUFS, artists, albums, target length, RNG seed).
- Filters and schema against a temporary SQLite file.
- `config`: `library.toml`, `config.toml`, env overrides, the plugin folder layout.
- **`FakeNil`:** a test double speaking the plugin protocol over pipes. Drives the handshake,
  `panel.event` → `panel.set`, radio reacting to `player.changed` with `player.append`, error codes
  passed through, exit on stdin close.
- MCP tools through FastMCP's in-memory client with `FakeNil`; busy port → panel message;
  `NIL_DJ_MCP_PORT` override.
- `install` into a temporary folder: keeps `data\`, `models\`, `config.toml`; moves the old index.
- `indexer` on generated tones and noise with a fake embedder (incremental, resume, deletes,
  decode errors).
- Real-CLAP tests marked `slow`, run on demand.
- **Parity test** (marked `parity`, dev-only, needs PyTorch): the same clips and prompts through
  the PyTorch `ClapModel` and through ONNX. Cosine between the two vectors must be ≥ 0.999 at fp32,
  ≥ 0.99 at fp16 and ≥ 0.97 at int8. It runs in WSL in a dev-only `uv` dependency group with
  CPU PyTorch ≥ 2.6 (the checkpoint is a `.bin` file, which `transformers` refuses to load with
  older torch, so the owner's `torch-env` with 2.5.1 cannot be used) and saves the PyTorch
  reference features and vectors to `tests/fixtures/clap-reference.npz`; on Windows, a test without
  PyTorch compares DirectML output against that file with the same thresholds, so the GPU path is
  checked too.
- GitHub Actions: `ruff` + `pytest` on CPU, no model download.

nil's tests are in nil §9.

## 12. Evaluation (`eval/`)

- **Search quality:** ≈ 30 prompts (mood, instruments, genre, tempo, combined with filters). The
  owner rates the top 10 per prompt 0/1/2 with `nil-dj eval rate` (≈ 20 min). Metrics nDCG@10 and
  P@10 for four systems: random, tags only (text match on tags), CLAP, CLAP + prompt template.
  The best CLAP system is also run at fp32, fp16 and int8 to report the quality cost of each
  precision, along with indexing time and download size.
- **Playlist quality:** mean adjacent similarity and BPM jumps vs random order; length error;
  filter compliance (must be 100 %).
- **Radio:** unique artists per hour and drift (similarity of each hour to the seed) over a
  simulated 2-hour session.
- Output: `eval/REPORT.md` with charts. Titles and scores only, never audio. Ratings are stored in
  the repo so the report is reproducible.

## 13. Showcase

README: what it is, demo video/GIF (prompt → nil playing with the bit-perfect indicator), install
(`nil-dj install`, then enable in nil), client config snippets (Claude Code, Claude Desktop, Cursor, VS Code), architecture
diagram, **design decisions** (ONNX Runtime + DirectML instead of PyTorch, brute force vs vector
DB, a plugin over stdio instead of an HTTP control API, permissions enforced by the host, one data
folder, local analysis for privacy), a short description of nil's plugin API (nil's `docs/` is
private; the public copy is nil's `PLUGIN-API.md`), evaluation results.

## 14. Build order

1. Done (Plan 1): `embedder`, `indexer`, `index`, `search` + CLI, validated on Windows.
2. nil (Plan 2, in audio-rea): data folder, `Append`, playlists, plugin host, panel, Plugins page,
   example plugin.
3. nil-dj (Plan 3): plugin folder and `install`, `host`, `order`, `radio`, `panel`, MCP server.
4. Eval, README, demo (Plan 4).
