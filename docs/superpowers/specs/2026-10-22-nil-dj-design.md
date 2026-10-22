# nil-dj — design

An AI DJ for the **nil** bit-perfect music player. You ask any MCP-capable agent for music in plain
language ("45 minutes of mellow instrumental for coding at night"); nil-dj finds matching tracks in
your own library **by how they sound**, orders them so they flow, and plays them in nil. When the
playlist ends, nil's **radio mode** keeps going on its own, with no AI involved.

Agreed in brainstorming on 2026-10-22.

## 1. Goals

| Goal | Meaning |
|---|---|
| **Daily driver** | The owner uses it every day with their own library, so indexing, re-indexing and playback must be robust. |
| **Showcase** | A public portfolio project for job-seeking: clear README, architecture diagram, measured evaluation, demo video. |

**Success criteria**

- A plain-language request from an MCP client starts a fitting, well-ordered playlist in nil, played bit-perfect.
- Radio mode continues playback from nil alone when nil-dj is not running.
- Works with any MCP client (Claude Code, Claude Desktop, Cursor, VS Code, …).
- An evaluation report shows that sound search beats tag-only and random baselines, and that ordering beats random order.

## 2. Scope

**In v1**

1. Build and play a playlist from a prompt.
2. "More like this": similar to the playing track, a track or an album.
3. Smooth ordering (sound similarity, tempo and loudness transitions, artist spacing).
4. Saved playlists (`.m3u8`) in nil.
5. Radio mode in nil: offline, and seeded by the AI (play a playlist with radio on).

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
| nil config dir | `%APPDATA%\bitperfect\` (`library.toml` holds `folders = ['E:\Music']`) |
| nil library | In-memory `LibraryIndex` built from a folder scan, cached as `library-cache.json`; no database |
| nil playback | `bp_engine::Command::Play { tracks, start }` replaces the queue; there is no append |

## 4. Architecture

Two halves with a narrow, documented contract between them: the **SQLite index file** and a small
**localhost HTTP API**.

```
MCP clients ──HTTP (or stdio)──► nil-dj  (Python, new public repo)
                                  ├─ indexer ── CLAP ──► index.sqlite ◄── read-only ─┐
                                  ├─ search / order                                 │
                                  └─ nil_client ──HTTP 127.0.0.1:7340──► nil (Rust, existing)
                                                                          ├─ control API
                                                                          ├─ playlists (.m3u8)
                                                                          └─ radio ──────┘
```

- **nil-dj** owns everything ML: decoding for analysis, CLAP embeddings, search, ordering, MCP.
- **nil** owns playback, playlists and radio. Radio reads precomputed vectors only, so it needs no
  model and no Python.
- Both run natively on Windows. Code lives in WSL at `~/projects/nil-dj` and is synced to
  `E:\projects\nil-dj` with `rsync --delete` for Windows runs.

### Data flow

- **Chat:** agent → nil-dj tool → search (`index.sqlite`) → order → `nil_client` → nil `/v1/play`
  → `PlayerCommand::Play` → WASAPI (bit-perfect).
- **Radio:** nil sees the last playlist track start → reads vectors → picks 5 tracks → `Append`.

## 5. nil-dj components

| Module | Responsibility |
|---|---|
| `config` | Reads nil's `library.toml` for music folders; nil API address; index path; token path. |
| `embedder` | Interface `embed_audio(clips) -> vec512`, `embed_text(str) -> vec512`. Backend: **ONNX Runtime**, no PyTorch at runtime (§5.1). |
| `indexer` | Walks the folders; decodes with ffmpeg to 48 kHz mono float (DSF/DFF included); takes three 10 s clips at 20 %, 50 % and 80 % of the track (the whole track if shorter than 30 s); embeds each, averages, L2-normalises. Measures tempo (BPM) and loudness (integrated LUFS). Reads tags with mutagen. Incremental by size + mtime; commits after each track (resumable); deletes rows for files that no longer exist; single instance via a lock file. |
| `index` | Schema, migrations, reads and writes of `index.sqlite`. |
| `search` | Text → CLAP text vector → SQL filters → brute-force cosine over the remaining rows (numpy). Also similarity to a track, an album (mean vector) or the playing track. |
| `order` | The ordering algorithm in §7, shared with nil through a fixture. |
| `nil_client` | HTTP client for nil's API; reads the token; re-reads it and retries once on 401. |
| `server` | FastMCP server, Streamable HTTP on `127.0.0.1:7341/mcp` by default; `--stdio` for stdio-only clients. |
| `eval` | Prompt set, rating CLI, metrics, report generation (§12). |
| `cli` | `nil-dj index`, `nil-dj serve`, `nil-dj search "<text>"`, `nil-dj eval …`. |

Tooling: Python 3.12, `uv` (install with `uvx nil-dj`), `pytest`, `ruff`. Licence: MIT.

### 5.1 Embedding backend (ONNX Runtime)

- **Model files:** the ONNX export of `laion/clap-htsat-unfused` published as
  `Xenova/clap-htsat-unfused` (`onnx/audio_model*.onnx`, `onnx/text_model*.onnx`, tokenizer and
  preprocessor configs), downloaded with `huggingface_hub` on first use and pinned to a revision.
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

## 6. The index (`%LOCALAPPDATA%\nil-dj\index.sqlite`)

The contract with nil. SQLite in **WAL mode** so nil can read while the indexer writes.

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
- A change to `schema_version`'s major number means nil must not read the file (radio turns off
  with a message).
- Brute-force search is deliberate: 2,765 × 512 float32 ≈ 5.7 MB, scanned in milliseconds. No ANN
  index or vector database is needed at this scale; the README documents the measurement.

## 7. Selection and ordering

### Search

1. `q = embed_text(prompt)`; whether to wrap the prompt as `"This is a sound of {prompt}"` is
   decided by the eval (§12).
2. Apply filters in SQL.
3. Score = cosine(q, track). Keep a candidate pool of the top `max(60, 3 × tracks needed)`. Use
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
  within target + 5 %. The weights are config values with these defaults, tuned by the eval.

### Radio (in nil)

- **Trigger:** radio is on and the **last** track of the playlist starts (`TrackStarted` with the
  last index). nil picks 5 tracks and sends `Append`.
- **Seed vector:** weighted mean of the last 5 played tracks (weights 1.0, 0.8, 0.6, 0.4, 0.2,
  newest first), L2-normalised. Tracks missing from the index are skipped; if none of the last 5
  are indexed, use the most recent indexed track played; if there is none, radio turns off and says
  why.
- **Excluded:** tracks played in the last 2 hours (in-memory history for the session), tracks
  already in the playlist, and tracks the current device cannot play (nil's existing capability
  check, e.g. DSD256 on the TP35).
- **Choice:** top 40 by similarity to the seed, then the ordering above with `match` = seed
  similarity. Each step picks uniformly at random among the **top 3** scores so radio varies;
  tests fix the RNG seed. Fewer than 5 candidates: append what there is; none: let playback end.

### One algorithm, two languages

`fixtures/order-cases.json` in nil-dj holds small synthetic cases (vectors, BPM, LUFS, artists,
albums, target length, RNG seed) and the expected order. nil-dj's tests and nil's tests (a copy in
the nil repo) both run every case and must match exactly.

## 8. Interfaces

### MCP tools (nil-dj)

Tracks are identified by their file path.

| Tool | Behaviour |
|---|---|
| `search_tracks(query, limit=20, filters?)` | Sound search. Returns path, artist, title, album, duration, format, score. |
| `find_tracks(artist?, album?, title?, limit=50)` | Case-insensitive substring match on tags. |
| `similar_tracks(seed, limit=20, filters?)` | `seed`: a track path, `{"album": ..., "artist": ...}`, or `"now_playing"`. |
| `build_playlist(prompt, minutes, filters?, seeds?, exclude?)` | Search + ordering; returns the ordered list and total length. Does not play. |
| `play(paths, radio=false)` | Plays in nil; with `radio=true` radio is switched on (AI-seeded radio). |
| `append(paths)` | Appends to what is playing, gaplessly. |
| `now_playing()` | Track, position, playlist, radio state, device, bit-perfect path. |
| `set_radio(on)` | Radio on or off. |
| `save_playlist(name, paths)`, `list_playlists()`, `play_playlist(name)` | Through nil. |
| `index_status()`, `reindex()` | Counts, unindexed files, errors, progress; `reindex` runs in the background. |

**Filters:** `min_minutes`, `max_minutes`, `year_from`, `year_to`, `artists`, `exclude_artists`,
`formats` (`flac`, `alac`, `aac`, `wav`, `aiff`, `dsd`), `hires_only`. Sound qualities
(instrumental, vocals, acoustic) belong in the query text, not in filters.

### nil control API (`127.0.0.1:7340`, JSON)

| Route | Effect |
|---|---|
| `GET /v1/status` | Now-playing state, playlist, radio, device, playback path (bit-perfect or not) |
| `POST /v1/play {paths, start, radio}` | `PlayerCommand::Play`; sets radio |
| `POST /v1/append {paths}` | New `PlayerCommand::Append` |
| `POST /v1/transport {action}` | `pause` · `resume` · `next` · `previous` · `stop` |
| `POST /v1/radio {on}` | Radio toggle |
| `GET /v1/playlists` | Names |
| `POST /v1/playlists {name, paths}` | Save (overwrite allowed) |
| `POST /v1/playlists/{name}/play` | Load and play |

- **Binding:** `127.0.0.1` only.
- **Token:** at start nil writes 32 random bytes (hex) to `%APPDATA%\bitperfect\api-token`; every
  request needs `Authorization: Bearer <token>`. Missing or wrong → 401.
- **Host check:** `Host` must be `127.0.0.1:7340` or `localhost:7340` (blocks DNS rebinding) → else 403.
- **Validation:** paths not in nil's library → 400 listing them. Unsupported-on-device tracks are
  skipped and reported by the engine as today.
- Runs on its own thread in the GUI process and forwards to the same `PlayerCommand` channel the
  UI uses; it never blocks the UI or the audio thread.

## 9. Changes to nil

| Change | Where |
|---|---|
| `Command::Append { tracks }`: extend the current track list; gapless when the format matches, exactly as within a `Play` list. When stopped or finished, behaves like `Play` from the first appended track. | `bp-engine` |
| `PlayerCommand::Append { items }` | `bp-gui` backend |
| Control API server (§8) | `bp-gui`, new `api` module; small HTTP crate without an async runtime (`tiny_http`) |
| Playlists: read/write `.m3u8` (UTF-8, `#EXTM3U`, `#EXTINF`, absolute paths), stored in `%APPDATA%\bitperfect\playlists\`; a minimal list and play view in the GUI | `bp-gui` |
| Radio: index reader (`rusqlite`, read-only, WAL), selection (§7), toggle in the now-playing bar, disabled with tooltip "Run `nil-dj index` first" when no index | `bp-gui`, new `radio` module |

The product brief listed playlists and a play queue as out of scope for nil v1; this project adds
them deliberately and in this minimal form.

## 10. Errors

| Situation | Behaviour |
|---|---|
| nil not running | Search and `build_playlist` work; playback tools return "nil isn't running — start nil and try again." |
| 401 from nil | Re-read the token file, retry once. |
| ffmpeg missing | The indexer stops at start with an install hint. |
| A file fails to decode | Recorded in `index_errors`; indexing continues; `index_status` lists them. |
| No usable DirectML GPU | CPU fallback with a warning. |
| First run | The ONNX model files (≈ 620 MB fp32, ≈ 310 MB fp16) download once to the Hugging Face cache; offline afterwards. |
| Files moved or deleted | Rows removed on the next index run; nil rejects unknown paths. |
| Second indexer | Refused by the lock file. |
| Index schema newer than nil knows | Radio off, with a message. |

## 11. Testing

**nil-dj (pytest)**

- `order` against `fixtures/order-cases.json`.
- Filters and schema against a temporary SQLite file.
- `config` parsing of `library.toml`.
- `nil_client` against a fake HTTP server (token, 401 retry, nil down).
- MCP tools through FastMCP's in-memory client with a fake nil.
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

**nil (cargo test)**

- Control API against the existing `MockPlayer`: each route, token, Host check, path validation.
- `.m3u8` round trip, including Japanese filenames.
- Radio selection against the copied fixture.
- `Append` gapless across the boundary, using the engine's transport test rig.

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
(`uvx nil-dj`), client config snippets (Claude Code, Claude Desktop, Cursor, VS Code), architecture
diagram, **design decisions** (ONNX Runtime + DirectML instead of PyTorch, brute force vs vector
DB, SQLite as the contract, token + Host check, one algorithm in two languages, local analysis for
privacy), evaluation results.

## 14. Build order

1. nil-dj: `embedder` (ONNX + parity test first), `indexer`, `index`, `search` + `nil-dj search`
   CLI. Validate on the real library on Windows with DirectML first; this carries the most risk.
2. nil: `Append`, control API, playlists.
3. nil-dj: `order`, `nil_client`, MCP server.
4. nil: radio.
5. Eval, README, demo.
