# nil-dj — language index

nil-dj learns the **sung language** of every track in the library. Radio then stays in the language
of what is playing, and a request can name a language ("Japanese city pop", "instrumental for
focus", or an MCP `languages` filter).

Agreed in brainstorming on 2026-10-28. Extends the main design,
`2026-10-22-nil-dj-design.md` (written "main §n" below). nil itself does not change.

## 1. Decisions

| Question | Decision |
|---|---|
| Purpose | Both: radio keeps the language of what is playing, and the owner can ask for a language. |
| Detection | Listen to the vocals with Whisper (not tags, not titles). Index time only. |
| Radio rule | Strong preference, no toggle: a large bonus for the same language; radio may cross over when nothing in the language is left. |
| Asking | Language words in the prompt, plus a `languages` filter in MCP. |
| Instrumentals | Neutral in radio (they always count as matching); left out of a language request; `instrumental` can itself be asked for. |
| Model | `onnx-community/whisper-small`, fp32, ONNX Runtime + DirectML, like CLAP. No torch or transformers at runtime. |
| Where | The nil-dj plugin only. nil core stays a lean Rust player. |

## 2. Evidence (throwaway probe, 2026-10-28)

124 hand-picked tracks from the owner's library: Japanese (incl. romaji-tagged artists),
English, Indonesian, Javanese, Spanish, Russian, French, instrumental scores, mixed-language songs.
Three 30-s windows per track, one decoder step each.

- **whisper-small: 124/124** correct as labelled by the owner, with the rules in §4.
  whisper-base: 117/124 (three sung tracks called instrumental; wrong languages on four).
- Whisper reports Indonesian as `jw` (Javanese) consistently, hence the Indonesian group.
- `no_speech` alone cannot find instrumentals: sung pop and rap score 0.5–0.75.
- Cost per 30-s window on the owner's laptop: decode 0.12 s, numpy mel 0.10 s, whisper-small 0.16 s
  (base 0.06 s). A 10-s decode costs 0.07 s.

The owner's labels are kept outside the repo at `E:\projects\nil-dj-eval\language-labels.csv`
(columns `artist, title, whisper_guess, confidence, no_speech, your_label`) for §11.

## 3. Components

| Unit | Does | Depends on |
|---|---|---|
| `languages.py` (new, pure) | Language codes and names, the Indonesian group, the window → verdict rule (§4), prompt word parsing (§8), the override file (§6), resolve (§6). | stdlib, numpy |
| `whisper.py` (new) | Loads whisper-small; `features(audio_16k) -> (80, 3000)` log-mel; `detect(features) -> (probs, no_speech)`; 48 → 16 kHz resampling. | onnxruntime, numpy, `gpu`, `huggingface_hub` |
| `indexer.py` | Decodes 30-s windows once; feeds CLAP and Whisper; the language catch-up (§5). | the above |
| `index.py` | Schema 1.1 migration, language columns, `languages_pending`. | sqlite3 |
| `search.py` | `Filters.languages`. | |
| `order.py`, `radio.py` | The language bonus and the anchor; the radio pool (§7). | |
| `dj.py`, `server.py`, `panel.py`, `cli.py` | Prompt parsing, MCP filter and result field, panel progress, `nil-dj languages`. | |

### 3.1 Whisper backend (`whisper.py`)

- Repo `onnx-community/whisper-small`, revision `36050c46d777d46dc4b5f43f6d90574fc38f8732`, files
  `onnx/encoder_model.onnx`, `onnx/decoder_model.onnx` (plus `.onnx_data` files if present) and
  `tokenizer.json`, downloaded into the models folder on first use with a pinned revision, as for
  CLAP. Always fp32; the `precision` setting applies to CLAP only.
- Providers: DirectML on the adapter `gpu.pick_adapter` chooses (or `gpu_device_id`), then CPU.
- Token ids come from `tokenizer.json` `added_tokens`: `<|startoftranscript|>`, `<|nospeech|>`
  (else `<|nocaptions|>`), and the 99 language tokens that follow `<|startoftranscript|>`. No text
  is tokenized or decoded.
- `detect`: run the encoder, then one decoder step with `input_ids = [[sot]]`. `no_speech` = softmax
  over the full vocabulary at `<|nospeech|>`; language probabilities = softmax over the language
  tokens only.
- Log-mel (numpy port of `WhisperFeatureExtractor`): 16 kHz, pad or cut to 30 s, reflect-pad
  200, Hann 400, hop 160, power spectrum, 80 Slaney mel filters 0–8 kHz, drop the last frame,
  `log10(max(x, 1e-10))`, clamp to `max − 8`, `(x + 4) / 4`, float32. Target ≤ 0.03 s per window
  (vectorised float32 rfft); the probe's version is the baseline.
- Resampling 48 → 16 kHz: Kaiser-windowed sinc low-pass (257 taps, cutoff 7.6 kHz, β 8), keep every
  third sample.
- The model is loaded only when there is language work and released when the index run ends.
  Search and radio never load it.

## 4. From windows to a verdict

Windows: three 30-s windows centred at 20 / 50 / 80 % of the track, clamped inside it; one window
of the whole track when it is ≤ 30 s.

For each window `w`: language probabilities `p_w` and `ns_w` (no-speech). Before anything else, the
probabilities of `jw`, `ms` and `su` are added into `id` (**the Indonesian group**, stored as `id`).

- Weights `v_w = max(0.001, 1 − ns_w)`; aggregate `P = Σ v_w p_w / Σ v_w`.
- `top` = argmax `P`, `conf = P[top]`, `vocal_score = 1 − mean(ns_w)`.
- **Instrumental** when `conf < 0.6` and `mean(ns_w) > 0.7`.
- Otherwise the language is `top`. It is **unsure** when `conf < 0.6`.
- **Mixed:** a window is *confident* when its own top language has probability ≥ 0.6. If the
  confident windows name two or more languages, `language_alt` = the one of them, other than `top`,
  with the highest `P`.

The thresholds are module constants in `languages.py`. They are applied in resolve (§6), not at
detection time, so they can be re-tuned without running Whisper again.

## 5. The index

### 5.1 Schema 1.1

`SCHEMA_VERSION` becomes `"1.1"`. Opening an index whose `tracks` table lacks the columns adds them
with `ALTER TABLE` inside one transaction and sets `schema_version` to `1.1`. Existing rows and
embeddings are untouched. The version check compares the major number only (unchanged code), so an
older nil-dj still opens a 1.1 index. New indexes are created with the columns.

```sql
-- added to tracks
language_raw    TEXT,  -- Whisper output as JSON (below); NULL = not detected yet
language        TEXT,  -- final: a code ('ja', 'en', 'id', …), 'instrumental' or 'unknown'
language_alt    TEXT,  -- final second language of a mixed track, else NULL
language_conf   REAL,  -- conf from §4, NULL for overrides
vocal_score     REAL,  -- 1 − mean no-speech
language_source TEXT   -- 'whisper' | 'artist' | 'override'
-- meta keys: language_model ("onnx-community/whisper-small@36050c4…"),
--            overrides_mtime_ns (of languages.toml when resolve last ran)
```

`language_raw` holds everything §4 needs: `{"windows": [{"p": {code: prob, …}, "ns": float}, …]}`,
with each window's probabilities rounded to 4 decimals and only codes ≥ 0.001 kept (in practice
under 1 kB). A failed detection stores `{"failed": "<reason>"}`.

`language` and the other final columns are written only by resolve (§6).

### 5.2 Indexing and catch-up

- **New or changed tracks:** the worker decodes the three 30-s windows once. CLAP gets, from each
  window, exactly the 10-s clip `clip_windows` gives today (same start and length, sliced by sample
  offset), so embeddings from older runs stay valid and comparable. Tempo and loudness use the same
  clips as today. The worker also resamples and computes the Whisper mel, so the main thread only
  runs the models and writes.
- **Language catch-up:** rows with `language_raw IS NULL` (all 2,890 existing rows, once) need only
  decode + mel + Whisper; CLAP is not loaded when nothing else is pending. It is part of every index
  run (`nil-dj index`, the panel's **Update index**, MCP `reindex`).
- Each track's result is written as soon as it is done, so a stopped run resumes where it stopped.
- **Model change:** when `meta.language_model` differs from the pinned model, all `language_raw`
  are set to NULL (re-detect only; no CLAP rebuild).
- `IndexStatus` gains `languages_pending`. Update is offered while it is > 0; the panel shows
  "Detecting languages 120 / 2,890"; MCP `index_status` returns it.
- Expected one-time catch-up for 2,890 tracks: ≈ 30–40 min (model ≈ 0.5 s per track on the main
  thread; decode and mel in the workers overlap with it).

## 6. Resolve and corrections

**Resolve** recomputes the final columns of every row that has `language_raw`, from `language_raw`,
the override file and the artist counts. It needs no audio (≈ 1 s for the library). It runs at the
end of every index run, at startup when `languages.toml`'s mtime differs from
`meta.overrides_mtime_ns`, and from `nil-dj languages --resolve`.

First match wins:

1. **Track override** (by path, compared with `path_key`).
2. **Album override** (album artist, else artist + album).
3. **Artist override** (matches the album artist or the artist).
4. **Artist majority**, only for an *unsure* sung track (§4): take the tracks of the same artist
   (album artist, else artist) that are confident, sung and from Whisper. If there are ≥ 3 and
   ≥ 70 % share one language, use it (`language_source = 'artist'`). Instrumentals are never
   relabelled.
5. **Whisper's verdict** (§4). A failed detection gives `unknown`.

Names compare with `casefold`.

**Override file** `<plugin dir>\languages.toml` (next to `config.toml`), optional:

```toml
[artist]
"Guyon Waton" = "id"

[album."Michael Giacchino"]
"Up" = "instrumental"

[track]
'E:\Music\Artist\Album\01 Song.flac' = "ja+en"   # mixed: language + alt
```

A value is a language code (§8), `instrumental`, or `code+code`. A bad file (TOML error, unknown
code, wrong type) is reported as a problem with the key that caused it, like `config.toml`
problems, and resolve then ignores the whole file until it is fixed.

## 7. Radio

- **Candidate languages:** a track's language set is `{language, language_alt}` minus NULLs. A track
  is **neutral** when `language` is `instrumental`, `unknown` or NULL (not detected yet).
- **Anchor:** the language set of the most recent played track (radio history, newest first, then
  tracks chosen earlier in the same top-up) that is not neutral. No such track: no anchor, and no
  bonus for anyone.
- **Bonus:** in `order._score`, `+ w.language` when `t` is neutral or its language set meets the
  anchor. New weight `Weights.language`, default **0.30**, settable in `config.toml` `[order]`.
- **Pool:** today radio takes the top 40 by seed similarity. With an anchor it takes the top 40
  overall plus the top 40 among tracks whose language set meets the anchor (same exclusions), so a
  language that is rare among the nearest tracks can still be chosen. The bonus is large next to
  typical similarity differences, so radio stays in the language until its tracks run out (e.g.
  all played in the last 2 hours); then it crosses over.
- The bonus is used by radio only. Prompt playlists and `build_playlist` pass `language = 0`
  (a prompt that wants a language filters for it, §8).
- 0.30 is a starting value; §11 measures it and the default is changed if the measurement says so.

## 8. Asking for a language

**Codes:** Whisper's language codes, with `id` standing for the Indonesian group (`jw`, `ms`, `su`
are accepted as aliases and mean `id`), plus `instrumental`.

**`Filters.languages`:** a list of codes. SQL: `language IN (…) OR language_alt IN (…)`.
`instrumental` selects `language = 'instrumental'`. Unsure tracks match their language; `unknown`
and undetected tracks never match. Accepted in the `filters` dict of `search_tracks`,
`similar_tracks` and `build_playlist`; the tool descriptions list it. An unknown code is a tool
error naming it.

**Prompt words** (`languages.parse_prompt(text) -> (languages, rest)`), used for the panel prompt and
for MCP `search_tracks` and `build_playlist` queries:

- Matches, case-insensitively and on word boundaries: the English name of every Whisper language
  ("Japanese", "Indonesian", "Korean", …), common endonyms and aliases (日本語, 한국어, 中文,
  "bahasa", "bahasa indonesia", "español", "français", "deutsch", "русский", …) and instrumental
  words ("instrumental", "instrumentals", "no vocals", "without vocals", "no singing").
- **Genre names stay with CLAP** and are not language requests: J-pop, J-rock, K-pop, C-pop,
  Mandopop, Cantopop, French house, Latin, Spanish guitar, Italo disco. They are matched first and
  protected.
- Matched words are removed, together with a connecting "or" / "and" / "," between two of them, and
  the whitespace is tidied. "Japanese or English city pop" → `["ja", "en"]` + "city pop".
  If nothing is left, CLAP gets "music".
- An explicit `filters.languages` wins over prompt words; the words are still removed from the
  CLAP text.

**Not ready:** a language request while no row has a final language gives the message
"Language data isn't ready yet — run Update index in the DJ tab." instead of an empty result.

**Results:** search results gain `language` (and `language_alt` when set).

## 9. CLI

`nil-dj languages`: counts per language (final), then the unsure tracks (`conf < 0.6`, sung) with
artist, title, guess, confidence and source, so the owner can add overrides.
`nil-dj languages --resolve` runs resolve and prints the counts.

## 10. Errors

| Situation | Behaviour |
|---|---|
| Whisper download or load fails | The index run continues without languages: CLAP work completes, `language_raw` stays NULL, the problem shows in the panel and `index_status`; the next run retries. |
| Detection fails on one track (decode error, ONNX error) | The track is still indexed if CLAP succeeded; `language_raw = {"failed": …}`, the error goes in `index_errors`, final `language = 'unknown'`. `--rebuild` retries it. |
| `languages.toml` invalid | Reported as a problem; overrides ignored until fixed; everything else works. |
| Language asked, no language data | The "not ready" message (§8). |
| Index from a newer major schema | Unchanged (main §6). |

## 11. Testing and evaluation

**Unit tests** (fake Whisper in tests, as CLAP is faked today):

- resampling: a 1 kHz tone keeps its level; content above 8 kHz is attenuated ≥ 60 dB
- §4 rules on made-up window probabilities: instrumental, unsure, the Indonesian group, mixed,
  the single short window
- resolve order, artist majority (thresholds, instrumentals untouched), override parsing and errors
- `parse_prompt`, including the protected genre names and connectors
- `Filters.languages` SQL; the "not ready" message
- radio: the anchor skips neutral tracks; neutral candidates get the bonus; the extra pool; no
  anchor → no bonus; playlists pass `language = 0`
- the 1 → 1.1 migration on a version 1 index file; a 1.1 index opens with the version 1 check
- catch-up: only undetected rows; resumes after a stop; CLAP not loaded for language-only work;
  CLAP clips are byte-identical to the old 10-s clips
- a model change clears `language_raw` only

**Parity** (dev `parity` group, like the CLAP mel): numpy log-mel vs transformers'
`WhisperFeatureExtractor` on noise, a tone and a real clip: max abs diff ≤ 1e-3.

**Owner's library (Windows):**

- `eval/language.py --labels E:\projects\nil-dj-eval\language-labels.csv` compares the index's final
  languages with the owner's labels (Indonesian group counted as `id`; mixed tracks count as correct
  when the label is either language). Target ≥ 123 / 124. The labels file stays out of the repo; the
  report states only the counts.
- Radio language check: simulated 2-hour radio sessions seeded from tracks of each language with
  ≥ 50 sung tracks; report the share of sung tracks in the anchor language. Target ≥ 90 %; adjust
  `language` until met, keeping similarity drift (main §12) no worse than 10 % above today's.
- Catch-up time for the full library is recorded in `docs/validation-log.md`.

## 12. Documentation

- Main spec §6: point to this spec for the new columns.
- README: a short "Languages" section (what it does, the prompt words, `languages.toml`,
  `nil-dj languages`, the one-time catch-up and model download size).
- MCP tool descriptions: `filters.languages` and the `language` result field.

## 13. Out of scope

Lyrics or transcription; editing overrides from the panel or MCP; language in nil's own UI or
playlists; smart playlists; a Rust port of nil-dj.
