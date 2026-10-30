# nil-dj

An AI DJ for the nil bit-perfect music player. Ask any MCP-capable agent for music in plain
language ("45 minutes of mellow instrumental for coding at night") and nil-dj finds matching tracks
in your own library by how they sound (CLAP embeddings, run locally with ONNX Runtime), orders them
so they flow, and plays them in nil.

## Install

nil-dj runs as a plugin inside nil (see `PLUGIN-API.md` for the protocol). You need
[uv](https://docs.astral.sh/uv/) and ffmpeg on `PATH`.

1. Build nil.
2. From this repo: `uv run nil-dj install --nil E:\Projects\nil`
3. In nil, open Settings → Plugins and switch nil-dj on, then press Index library in its entry
   there (in the DJ tab on nil before 0.2.0). The sound model downloads on first use.

## In nil

With nil 0.2.0 or later, the DJ tab is for making playlists: describe one and press Make, or
press More like this, or switch Radio on. Results show like nil's queue. The housekeeping lives
in the DJ's entry in Settings → Plugins: the index line with its progress bar, the Index library
/ Update index / Rebuild index button, and the MCP server address. Tracks that failed to index or
to get a language are listed, with reasons, in nil's plugin log (Open log in the same entry).

Older nil gets everything in the tab, as before.

## Connect an agent

nil-dj serves MCP on `http://127.0.0.1:47263/mcp` while nil is running (`NIL_DJ_MCP_PORT` changes
the port).

Claude Code:

```
claude mcp add --transport http nil-dj http://127.0.0.1:47263/mcp
```

Claude Desktop and Cursor (stdio; add to the MCP servers config):

```json
{
  "mcpServers": {
    "nil-dj": {
      "command": "uv",
      "args": ["run", "--directory", "<path to nil-dj>", "nil-dj", "serve", "--stdio"]
    }
  }
}
```

Without nil-dj running in nil, `serve --stdio` is search-only: play, queue and playlist tools
answer "nil-dj isn't running inside nil".

Besides search, play, append and playlists, the agent can edit nil's queue: `remove_from_queue`,
`move_in_queue` and `clear_upcoming`. These need nil 0.3.0 or newer. Positions come from the
`upcoming` list in `now_playing` (1 = the next track); pass its `path` values as `paths` and an
edit fails, rather than hitting the wrong track, if the queue has moved on.

## Languages

nil-dj detects the sung language of every track, so you can ask for one and radio can stay in it.
Detection runs Whisper (`onnx-community/whisper-small`, fp32, pinned revision) locally on a few
30-second windows per track. Nothing leaves your machine. Search and radio never load Whisper:
they read the result from the index.

- **Prompts and filters.** Name a language in a prompt ("Japanese city pop", "French and Italian
  jazz", "instrumental") or pass `filters.languages` over MCP. The language words are taken out of
  the text sent to the sound model and applied as a filter. Genre names that contain a language
  ("J-pop", "K-pop", "French house", "Latin") are left alone. Results carry a `language` field.
- **Radio.** Radio keeps the language of what is playing: tracks in that language, and tracks with
  no language (instrumental or unknown), get a bonus. The weight is `language` in the `[order]`
  table of `config.toml` (default 0.30; 0 turns it off). Prompt playlists do not use it.
- **First run.** The first Update index after upgrading downloads whisper-small (about 970 MB:
  encoder 353 MB, decoder 615 MB, tokenizer 2.5 MB) and detects languages for the whole library
  once. That catch-up takes a while and is safe to stop: tracks already done stay done, and the
  next run continues. Settings → Plugins → DJ shows "Detecting languages done / total". New tracks are detected
  as they are indexed.
- **`languages.toml`.** Put a `languages.toml` next to `config.toml` in the plugin folder (beside
  the index when there is no plugin folder) to correct a verdict. Use ISO 639-1 codes (`ja`,
  `en`, `pt`), `a+b` for a second language, and `instrumental`. Artist names match without
  regard to case or accents; paths in `[track]` without regard to case or slashes.

  ```toml
  [artist]
  "Some Artist" = "ja"

  [album."Some Artist"]
  "Some Album" = "ja+en"

  [track]
  'E:\Music\Some Artist\Some Album\01 Song.flac' = "instrumental"
  ```

- **`nil-dj languages`** lists the tracks per language, the unsure ones (confidence below 0.6)
  and any where detection failed. `nil-dj languages --resolve` re-applies `languages.toml`.
  `nil-dj index --no-languages` skips detection.
- **Evaluation.** `eval/language.py --labels labels.csv` compares the index to hand labels
  (counts only). `eval/radio_language.py` simulates 2-hour radio sessions per language and reports
  how often radio stays in the language and what it costs in similarity drift. Run both with
  `uv run python eval/<script>.py`.
