# DJ panel redesign: design

Date: 2026-10-28. Status: approved in conversation, awaiting spec review.

## Why

The DJ tab in nil shows only text, with the index status, index buttons and the MCP address
listed in it. The owner wants it to match nil's look: a tab for making playlists whose results
look like nil's queue, with the housekeeping moved into the plugin's entry in Settings →
Plugins. Failed tracks should also be visible somewhere: today the panel shows only a count.

## Depends on nil 0.2.0

nil 0.2.0 draws the manifest `title` ("DJ") as the tab's page title, draws `tracks` as
queue-style rows with album art and an optional uppercase `label`, draws `toggle` as a switch,
and accepts `panel.set {panel, "place": "settings"}` for a second panel inside the plugin's
Settings entry (allowed there: heading, text, progress, button, toggle, row). Events from it
arrive as `panel.event {id, value?, "place": "settings"}`; tab events have no `place`.

## The two panels

`panel.py`'s `build(view)` splits into two pure functions of `View`:

**`build_tab(view)`**, sent without `place`, top to bottom:

1. No heading (nil draws "DJ").
2. Only when nothing is indexed: one muted line, "Index your library first: Settings →
   Plugins → DJ." While the first index runs: "Indexing your library… progress is in
   Settings → Plugins."
3. Model progress ("Downloading the sound model (once, about 620 MB)…", "Loading the sound
   model…") and model failures, as today: they are why Make waits.
4. The `prompt` input ("Describe a playlist…", submit "Make"), then a row with the `more`
   button ("More like this") and the `radio` toggle ("Radio").
5. The message line, as today.
6. The result: a `tracks` element whose `label` is "<title> · <n> tracks · <duration>"
   (duration as "1 h 52 min", or "38 min" under an hour); nil uppercases it. No separate result
   heading. An empty result is the text "No matching tracks."

**`build_settings(view)`**, sent with `place: "settings"`:

- The index line, as `_index_elements` builds it today, with the failed part reading
  "3 failed (see Open log)"; the progress bar instead while a run is going.
- Problems: the index problem, `languages.toml` problems and startup problems.
- "MCP server: <address>" or "MCP server: not running", as muted text.
- One button in a row: Index library (nothing indexed), Update index (normally) or Rebuild
  index (rebuild needed); none while a run is going.

## Sending

`refresh` keeps its throttle and sends both panels inside the send lock. Each is sent only if
its payload differs from the last one sent of that place, so a radio change re-sends only the
tab. A failed send forgets the last payload of that place so the next refresh retries it.

## Events

`plugin.py` passes `place` through to `Panel.on_event`. Tab ids: `prompt`, `more`, `radio`.
Settings ids: `index`, `rebuild`. An id arriving from the other place is ignored.

## Older nil

`initialize` gives `nil_version`. If it parses as a version below 0.2.0, is missing, or cannot
be parsed, the DJ sends today's single tab panel (heading, index, buttons, MCP line, all of it)
and never sends `place`. The old layout is kept as that one fallback function, not maintained
as a second design.

## Failure log

nil writes the plugin's stderr to `logs/plugins/nil-dj.log`, which Settings' **Open log**
opens.

- As failures happen, the indexer prints one line each to stderr:
  `nil-dj: index failed: <path>: <reason>` and
  `nil-dj: language detection failed: <path>: <reason>`.
- When a run ends (finished or stopped), the runner in `reindex.py` prints
  `nil-dj: <n> tracks failed to index:` followed by every row in `index_errors`, then
  `nil-dj: <n> language detections failed:` followed by every failed detection, each as an
  indented `<path>: <reason>` line. A section with no failures is left out. This includes
  failures from earlier runs, so the log always has the full list.
- A failing write to stderr is ignored; indexing carries on.
- The CLI prints the same per-track lines to the terminal.

Paths go only to the owner's local log, as they already do. Nothing about eval reports or the
repo changes.

## Testing

- `build_tab` / `build_settings`, one test per state: nothing indexed, first index running,
  model downloading / loading / failed, a result (label text and duration formats), an empty
  result; index counts with failed and pending, a problem, each button choice and none while
  running, MCP up and down.
- Events: each id reaches its handler from its place; ids from the wrong place are ignored.
- Sending: an unchanged panel is not re-sent; a radio change sends only the tab; a failed
  send is retried.
- Fallback: `nil_version` "0.1.1", missing, and "garbage" give the single old panel and no
  `place`; "0.2.0" gives two panels.
- Failure log: per-track lines for index and language failures; the end-of-run summary lists
  earlier failures too and leaves out empty sections.

## Windows check

After building nil 0.2.0 and installing the plugin, recorded in `docs/validation-log.md`:

1. DJ tab: "DJ" title, art rows, uppercase label, Radio switch; double-click and the context
   menu still work.
2. Settings → Plugins → DJ: index line and MCP address show; Update index runs with its
   progress bar in the entry.
3. Open log lists the failed tracks with reasons.
4. Now Playing's queue looks as before.
