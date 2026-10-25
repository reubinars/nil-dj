# Validation log — Plan 1 (index and search) on the real library

Windows laptop: Ryzen 5 5600H, RTX 3060 Laptop + AMD Radeon iGPU, library `E:\Music` on 2026-10-22.
Tools: uv 0.13.5, ffmpeg 9.0.2 (gyan.dev full build), onnxruntime-directml 1.24.4, Python 3.12.

## Tests on Windows

- Fast suite: 102 passed, 3 skipped (symlink creation needs admin or Developer Mode; two
  Linux-only tests). The first run found a test bug (symlinks), fixed by splitting the walk tests
  and adding a Windows junction test, which fails if `walk` stops skipping junctions.
- Parity (`-m slow`): fp32, fp16 and int8 all pass the thresholds (0.999 / 0.99 / 0.97).
  The first run found a Windows-only ONNX Runtime 1.24 bug: the fp16 models fail to load on the
  CPU provider at `ORT_ENABLE_ALL` (`SimplifiedLayerNormFusion` references a cast it inserted).
  CPU sessions for fp16 models now use `ORT_ENABLE_EXTENDED`; this affected int8 (fp16 text model
  on CPU) and `--cpu --precision fp16`.

## GPU choice

| DXGI index | Adapter | Dedicated VRAM | fp32 model run, 3 clips |
|---|---|---|---|
| 0 | NVIDIA GeForce RTX 3060 Laptop GPU | 5.9 GiB | 203 ms |
| 1 | AMD Radeon(TM) Graphics | 0.5 GiB | 277 ms |
| 2 | Microsoft Basic Render Driver | 0 (software) | — |
| — | CPU provider | — | 427 ms |

`pick_adapter` picks 0, and DirectML `device_id` follows DXGI order (device 0 is the faster one).
Log-mel features take 25 ms per clip in numpy. Embedding (≈ 0.2 s per track) is not the bottleneck;
ffmpeg analysis is.

## Full index

`uv run nil-dj index` (fp32, DirectML on the RTX 3060, 6 analysis workers):

```
Embedding on DmlExecutionProvider (NVIDIA GeForce RTX 3060 Laptop GPU), precision fp32; analysing with 6 workers
added 2790, updated 0, removed 0, failed 3, unchanged 0 in 39:23
```

- 2,793 files, 39 min 23 s, 71 tracks/min. Before parallel analysis, single-worker cost
  projected ≈ 4.5 h (ffmpeg decode + ebur128 loudness; DSF ≈ 25 s per track).
- 41 files were added to the library while it ran; the next run picked them up:
  `added 41, updated 0, removed 0, failed 3, unchanged 2790 in 0:24`.

Failures (all three are damaged files, not nil-dj bugs, except where noted):

| File | Error | Cause |
|---|---|---|
| `11 - 雪の降る街.flac` | can't read tags: file said 13806176 bytes, read 274037 bytes | Corrupt embedded cover art (PNG block); the audio decodes. mutagen rejects the whole file. |
| `HEART STATION.flac` | ffmpeg: overread / Invalid data found when processing input | Truncated: exactly 54.0 MiB, about half the size a 4:37 24/96 FLAC needs. |
| `Kahitna - Titik Nadir.flac` | not a valid FLAC file | The file is a ZIP archive named `.flac`. |

## Path agreement with nil

`uv run python scripts/compare_with_nil.py` after the catch-up run:

```
nil 2834, indexed 2831, index_errors 3
in nil not in index: 3            (all 3 in index_errors)
  ...of which unexplained: 0
in index not in nil: 0
in index_errors not in nil: 0
```

The indexed path set equals nil's minus the decode errors (Review Focus 1).

## Incremental run

`added 0, updated 0, removed 0, failed 3, unchanged 2831 in 0:01` (8 s including start-up and
model load).

## Search sanity

Top 5 per prompt (`nil-dj search "<prompt>" --limit 5`, template `{}`). Verdicts are the owner's.

### "mellow piano"

```
0.638  JAN GUNNAR HOFF — MOON RIVER  [HOME]  3:41  dsd
0.630  KANAE IZUTSU — LOVE THEME FROM 'SPARTACUS ~ SUPERSTAR'  [ANOTHER ANSWER]  10:35  dsd
0.616  TSUYOSHI YAMAMOTO TRIO — MISTY  [MISTY FOR DIRECT CUTTING]  5:52  dsd
0.604  Kobasolo — Outro  [Clumsy]  1:18  flac
0.596  JAZZ PARADISE — スタンド・バイ・ミー(STAND BY ME)  [夏の午後に聴くAFTERNOON JAZZ]  3:27  flac
```

Verdict: good (owner, for all five: "That looks good").

### "energetic electronic dance"

```
0.406  Caramella Girls — Megamix  [Supergott (Speedy Mixes)]  3:59  aac
0.402  Yanaginagi — point at infinity  [yanaginagi Best Album -MUSEUM-]  4:59  flac
0.395  Caramella Girls — Ett & Två  [Supergott (Speedy Mixes)]  2:51  aac
0.394  Sakuzyo — Fairythm  [Fairythm]  5:45  flac
0.394  Misekai — { New World  [Artrium]  1:37  flac
```

Verdict: good (owner, for all five: "That looks good").

### "female vocals, acoustic guitar"

```
0.565  Mamerico — tricolore  [minuscule]  4:02  aac
0.552  AIMER — TWINKLE TWINKLE LITTLE STAR　-PROLOGUE-  [SLEEPLESS NIGHTS]  1:28  flac
0.552  KANAE IZUTSU — TAKEDA'S LULLABY  [ANOTHER ANSWER]  4:07  dsd
0.549  Billie Eilish — WILDFLOWER  [HIT ME HARD AND SOFT]  4:21  flac
0.548  Mamerico — oyasumi  [minuscule]  3:47  aac
```

Verdict: good (owner, for all five: "That looks good").

### "heavy distorted guitars"

```
0.552  Kelompok Penerbang Roket — Ekspedisi 69  [Galaksi Palapa]  5:51  flac
0.542  GALNERYUS — THE TIME BEFORE DAWN  [UNDER THE FORCE OF COURAGE]  2:58  flac
0.536  UNDEAD CORPORATION DOUJIN WORKS — Haul me up -Guitar Instrumental-  [The Tyrant]  3:57  flac
0.529  UNDEAD CORPORATION DOUJIN WORKS — What breaks the gloomy sky isn't the sunlight -Guitar Instrumental-  [The Tyrant]  3:26  flac
0.527  UNDEAD CORPORATION DOUJIN WORKS — Evil But Innocent -Guitar Instrumental-  [The Tyrant]  3:56  flac
```

Verdict: good (owner, for all five: "That looks good").

### similar: DECO*27 — ネオネオン

```
0.965  Akbar Chalay — Astaga Bercanda  [Astaga Bercanda]  2:47  flac
0.958  Akbar Chalay — Emang Boleh?  [Emang Boleh?]  2:49  flac
0.957  ReGLOSS — Midsummer Citrus Snapshot ver.  [Snapshot]  3:40  flac
0.956  猫又おかゆ — ネコカブリーナ  [ぺるそにゃ～りすぺくと]  3:10  flac
0.954  猫又おかゆ — 大嫌いがもっともっと積もるまで  [ぺるそにゃ～りすぺくと]  2:52  flac
```

Verdict: good (owner, for all five: "That looks good").

# Validation log — Plan 3 (nil-dj as a nil plugin)

## Tested from WSL on 2026-10-25 (branch `feat/nil-dj-plugin`)

- Fast suite: 325 passed, 1 skipped, ruff clean.
- `-m slow`: 4 passed, including the real `uv` install into a scratch nil folder and the
  real-model embedder tests. 1 skipped: the GPU test needs DXGI, so Windows only.
- End-to-end smoke with the real model: the plugin ran over JSON-RPC against the fake nil from
  the tests, with real CLAP on CPU, real ffmpeg decoding and six generated 40 s FLACs (three
  noise colours, three tones).
  - Index library: 6 tracks indexed in 15.5 s, including the model load. The model files landed
    in `plugins/nil-dj/models/<rev>/`.
  - The prompt "rain falling on a roof" ranked brown noise, pink noise, white noise, then the
    three tones. Sensible.
  - `shutdown`: reply `{}`, exit code 0, no thread left behind.
- Model download failing on the first run (offline): the panel shows "Indexing failed: …" and
  "Sound model failed: …", and nil-dj keeps running. The text is Hugging Face's raw error, which
  is long; it gets friendlier copy in the final fix round.

## Not tested yet: needs you on Windows

This session couldn't run anything on Windows: the worktree guard refuses every
`powershell.exe` call. These steps are yours. Each one says what "good" looks like.

1. **Package nil.** Package the current nil `main` into `E:\Projects\nil` with audio-rea's
   `package-nil.ps1`, then start nil once so its `data\` migrates. Quit it.
2. **Sync and run the tests on Windows.** `scripts/sync-to-windows.sh` mirrors the *main*
   checkout, so run it after the merge. Before the merge, sync the worktree from WSL instead:
   `rsync -a --delete --exclude .git --exclude .venv --exclude __pycache__ --exclude .pytest_cache --exclude .ruff_cache --exclude .superpowers --exclude .claude ~/projects/nil-dj/.claude/worktrees/nil-dj-plugin/ /mnt/e/projects/nil-dj/`
   Then, in `E:\projects\nil-dj`: `uv run pytest -q` (expect about 355 passed and a few
   Windows skips), and `uv run pytest -q -m slow` (the GPU test should run, not skip).
3. **Install.** `uv run nil-dj install --nil E:\Projects\nil`. Check:
   - `E:\Projects\nil\plugins\nil-dj\` exists with `app\`, `plugin.toml` and `data\`;
   - `%LOCALAPPDATA%\nil-dj\index.sqlite` has moved to `plugins\nil-dj\data\index.sqlite`, so
     your 2,7xx-track index carries over and nothing gets re-indexed. If install says "The old
     index is in use, so it was left at …", close anything using it and run install again.
4. **In nil:** Settings → Plugins, switch nil-dj on, then open its tab.
   - The status line shows your track count indexed, with nothing pending (or only new files).
   - Because an index exists, the model starts downloading as soon as nil-dj starts, to
     `plugins\nil-dj\models\<rev>\`. That's about 600 MB, once. The panel says it is
     downloading, then loading; a prompt (try "mellow piano") waits for it.
   - A playlist appears. Play all plays it, and the output stays bit-perfect (check nil's
     signal-path indicator).
   - Save as playlist: it shows up in nil's playlists.
   - More like this, while a track plays, gives a new list.
   - Radio on, play a short playlist: when it nears the end, more tracks are appended and
     playback continues gaplessly.
   - Before pressing Update index the first time, check it says nothing (or only your new
     files) pending; if every track shows as pending, stop and tell me, because the paths don't
     match.
   - Add an album to `E:\Music`, then Update index: only the new tracks are indexed.
5. **MCP.** With nil running and nil-dj on, the panel should show `MCP: http://127.0.0.1:47263/mcp`.
   Then: `claude mcp add --transport http nil-dj http://127.0.0.1:47263/mcp`. Ask Claude for
   "20 minutes of something calm": nil should start playing it.
6. **No stray processes.** Kill `nil.exe` in Task Manager. No `python.exe` from nil-dj should
   be left behind (check in Task Manager after a few seconds).
7. **Clean up.** Once step 4 works, delete the old model copy in the Hugging Face cache
   (`%USERPROFILE%\.cache\huggingface\hub\models--Xenova--clap-htsat-unfused`).

Write down anything that looks off (copy, timings, errors) under this section, and I'll pick it
up from here.
