# Language index — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** nil-dj learns the sung language of every track with whisper-small. Radio stays in the
language of what is playing, and a prompt or MCP filter can ask for a language.

**Architecture:**
- A pure `languages.py` holds codes, the window → verdict rule, prompt parsing, overrides and resolve.
- `whisper.py` holds the ONNX model, the resampler and the log-mel.
- The indexer decodes three 30-s windows per track once. CLAP gets its 10-s clips sliced out of
  those windows; Whisper gets all of each window.
- Raw Whisper output is stored per row (`language_raw`). The final columns are written only by
  resolve, which needs no audio.
- Radio adds a language bonus in `order._score` and widens its pool.

**Tech Stack:**
- Python ≥ 3.12, numpy, ONNX Runtime + DirectML, `huggingface_hub`, SQLite (WAL).
- `tomllib` for `languages.toml`.
- Dev only: pytest, ruff, and the `parity` group (torch + transformers) to make the reference
  fixture.

**Spec:** `docs/superpowers/specs/2026-10-28-language-index-design.md`, written "spec §n" below.
It extends `docs/superpowers/specs/2026-10-22-nil-dj-design.md`.

## Global Constraints

- Python ≥ 3.12. Line length 100. `uv run ruff check .` clean. All existing tests stay green.
- A default `uv run pytest` needs no network, no model and no GPU.
  - Real-model tests are `@pytest.mark.slow`.
  - Reference-making needs the `parity` group.
  - Windows-only checks are `@pytest.mark.windows` + `skipif`.
- Whisper:
  - repo `onnx-community/whisper-small`, revision `36050c46d777d46dc4b5f43f6d90574fc38f8732`
  - `MODEL_ID = "onnx-community/whisper-small@36050c46d777d46dc4b5f43f6d90574fc38f8732"`
  - always fp32
  - files under `<models_dir>/whisper-small/<REVISION>/`
- No torch or transformers at runtime. Search and radio never load Whisper.
- Verdict constants (spec §4): `CONF_MIN = 0.6`, `INSTRUMENTAL_NS = 0.7`, `WINDOW_CONF = 0.6`,
  `MIN_WEIGHT = 0.001`. Artist majority: `MAJORITY_MIN = 3` tracks, `MAJORITY_SHARE = 0.7`.
- `language_raw` rounds probabilities to 4 decimals and keeps only codes ≥ 0.001.
- Radio: `Weights.language` default **0.30**, settable in `config.toml` `[order]`. Prompt
  playlists use `language = 0`.
- Exact copy:
  - `LANGUAGES_NOT_READY = "Language data isn't ready yet — run Update index in the DJ tab."`
  - panel progress label `f"Detecting languages {done:,} / {total:,}"`
- The owner's labels (`E:\projects\nil-dj-eval\language-labels.csv`) never enter the repo. Eval
  reports print counts only, never paths or titles.
- Never copy anything from audio-rea `docs/`. Never a Tidal/Qobuz downloader.
- Commits: no `Co-Authored-By` or "Generated with" lines. The author email is the configured noreply
  address.

## Rulings made while planning (the spec differs; owner please confirm)

1. **CLAP clips are not byte-identical to the old 10-s decodes** (spec §11 asked for that).
   - Measured: slicing a 30-s ffmpeg decode is bit-exact for 48 kHz sources.
   - For 44.1 kHz sources the seek lands at a different sub-sample phase. The CLAP log-mel then
     differs by a mean of 0.000004–0.27 dB.
   - Rule: the tests require bit-exact clips for 48 kHz and a mean log-mel diff ≤ 0.5 dB for
     44.1 kHz. Task 11 measures cosine ≥ 0.99 between stored and new embeddings on the real
     library.
2. **A per-track detection failure does not go in `index_errors`** (spec §10 said it did).
   - `index_errors` means "not indexed": rows in it count as *failed* in the panel and in
     pending.
   - Instead the reason stays in `language_raw = {"failed": …}`, the final language is `unknown`,
     and `nil-dj languages` lists these tracks.
3. **CLAP still loads for every index run** in the plugin and the CLI, as today. The search box
   needs it anyway.
   - The spec's "CLAP is not loaded for language-only work" becomes: `embed_audio` is never
     called for language-only rows.

## Review Focus

1. **The owner presses Update while the one-time catch-up runs, or quits nil mid-way.**
   - Detected rows must stay detected. The next run starts with the undetected rows only.
   - Final columns must still be resolved for what was done.
   - Task 6 (stop mid-catch-up, then resume) and Task 7 (resolve after a stopped run).
2. **The Whisper download fails (offline first run) during an ordinary Update.**
   - New tracks still get indexed with CLAP.
   - `language_raw` stays NULL, the panel says why, and the next run retries.
   - Task 6 (factory raises) and Task 7 (problem text in status).
3. **Prompt words that look like languages but are genres or titles**, e.g. "J-pop for a
   rainy day" or "French house":
   - CLAP must get the full text and no filter is applied.
   - Task 2 (protected genres, word boundaries such as "Englishman" and "Latina").
4. **Very short tracks and odd durations**: a 4-s jingle, a 12-s interlude, a 30.0-s track,
   and a track whose tags report a longer duration than the audio has (decode returns fewer
   samples).
   - Task 6 (window and slice tests, short decode padded with zeros, never an IndexError).
5. **Mixed tags in overrides and filters**: an artist in `languages.toml` spelled with
   different case or diacritics ("BLACKPINK" vs "Blackpink"), album artist vs track artist, and
   Windows path case/slashes in `[track]` keys.
   - Task 5 (casefold, `path_key`).

---

## File Structure

| File | Responsibility |
|---|---|
| `src/nil_dj/languages.py` (new) | Codes and names, aliases, `Window`/`Verdict`, `verdict`, raw JSON, `parse_prompt`, overrides, `resolve`, `resolve_index`. No ONNX. |
| `src/nil_dj/whisper.py` (new) | `to_16k`, `features`, `LanguageDetector` protocol, `WhisperDetector`, `DetectorFactory`, `whisper_present`. |
| `src/nil_dj/index.py` | Schema 1.1, migration, language columns and queries. |
| `src/nil_dj/indexer.py` | 30-s windows, CLAP slicing, detection, catch-up, `Progress.phase`. |
| `src/nil_dj/reindex.py`, `dj.py`, `panel.py`, `plugin.py`, `server.py`, `cli.py` | Wiring, status, progress, CLI `languages`. |
| `src/nil_dj/search.py`, `order.py`, `radio.py`, `config.py` | `Filters.languages`, `Weights.language`, anchor, bonus, pool. |
| `scripts/make_whisper_reference.py` (new) | Writes `tests/fixtures/whisper-reference.npz` (parity group). |
| `eval/language.py`, `eval/radio_language.py` (new) | Windows evaluation against the owner's labels and radio sessions. |
| `tests/test_languages.py`, `tests/test_whisper.py` (new) | Unit tests for the new modules. |
| `tests/fakes.py` | `FakeDetector`, `fake_detector_factory`; `add_track(..., language=, language_alt=)`. |

---

### Task 1: Codes and the verdict rule (`languages.py`)

**Files:**
- Create: `src/nil_dj/languages.py`
- Test: `tests/test_languages.py`

**Interfaces:**
- Produces:
  - Constants: `INSTRUMENTAL = "instrumental"`, `UNKNOWN = "unknown"`, `GROUP = "id"`,
    `GROUP_ALIASES = ("jw", "ms", "su")`, plus the Global Constraints numbers.
  - `NAMES: dict[str, str]`: Whisper's 99 codes → English name, without `jw`/`ms`/`su`;
    `"id": "Indonesian"`.
  - `normalise_code(code: str) -> str`: strips and casefolds, maps aliases to `id`, and accepts
    `instrumental`. Otherwise it raises `ValueError(f"unknown language '{code}'")`.
  - `@dataclass(frozen=True) Window(p: dict[str, float], ns: float)`
  - `@dataclass(frozen=True) Verdict(language: str, alt: str | None, conf: float, vocal_score: float, unsure: bool)`:
    `language` is a code or `INSTRUMENTAL`.
  - `verdict(windows: list[Window]) -> Verdict`
  - `encode_raw(windows: list[Window]) -> str`, `encode_failed(reason: str) -> str`
  - `decode_raw(text: str) -> list[Window] | None`: None for a failed record.

- [ ] **Step 1: Write the failing tests**

```python
def w(ns, **p): return Window(p, ns)

def test_group_is_merged_before_anything():
    v = verdict([w(0.2, jw=0.5, ms=0.2, en=0.3)])
    assert (v.language, round(v.conf, 3)) == ("id", 0.7)

def test_instrumental_needs_low_conf_and_high_no_speech():
    assert verdict([w(0.8, ja=0.4, en=0.3, ko=0.3)]).language == "instrumental"
    assert verdict([w(0.8, ja=0.7, en=0.3)]).language == "ja"          # confident: sung
    v = verdict([w(0.5, ja=0.4, en=0.3, ko=0.3)])                       # sung but unsure
    assert (v.language, v.unsure) == ("ja", True)

def test_weights_favour_vocal_windows():
    v = verdict([w(0.99, en=1.0), w(0.1, ja=1.0), w(0.1, ja=1.0)])
    assert v.language == "ja" and v.conf > 0.99 and round(v.vocal_score, 3) == round(1 - 1.19 / 3, 3)

def test_no_speech_one_still_counts_with_min_weight():
    assert verdict([w(1.0, fr=1.0)]).language == "fr"   # conf 1.0: sung, however quiet

def test_mixed_names_alt_from_confident_windows():
    v = verdict([w(0.2, ja=0.9, en=0.1), w(0.2, en=0.8, ja=0.2), w(0.2, ja=0.9, en=0.1)])
    assert (v.language, v.alt) == ("ja", "en")
    assert verdict([w(0.2, ja=0.9, en=0.1), w(0.2, en=0.55, ja=0.45)]).alt is None

def test_single_window():
    assert verdict([w(0.1, es=0.95)]) == Verdict("es", None, 0.95, 0.9, False)

def test_raw_round_trip_rounds_and_drops_tiny():
    text = encode_raw([w(0.123456, ja=0.987654, en=0.0004)])
    assert decode_raw(text) == [w(0.1235, ja=0.9877)]
    assert decode_raw(encode_failed("decode error")) is None

def test_normalise_code():
    assert normalise_code(" JW ") == "id" and normalise_code("instrumental") == "instrumental"
    with pytest.raises(ValueError, match="unknown language 'xx'"):
        normalise_code("xx")
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/test_languages.py -v`. Expected: ImportError.

- [ ] **Step 3: Implement `languages.py` as in Interfaces**
  - `NAMES` lists Whisper's language tokens in tokenizer order, as `{code: English name}`.
    whisper-small has 99 (`en` … `su`, `yue` is not in v2-small).
  - Write `verdict` exactly in the order of spec §4.

- [ ] **Step 4: Run them and see them pass.** `uv run pytest tests/test_languages.py -v`

- [ ] **Step 5: Commit.** `git commit -m "feat: language codes and the window verdict rule"`

---

### Task 2: Prompt words (`languages.parse_prompt`)

**Files:**
- Modify: `src/nil_dj/languages.py`
- Test: `tests/test_languages.py`

**Interfaces:**
- Produces: `parse_prompt(text: str) -> tuple[list[str], str]`. The list holds codes in the order
  they appear, without duplicates. The rest is the text without the matched words; it is
  `"music"` when nothing else is left.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("text, langs, rest", [
    ("Japanese or English city pop", ["ja", "en"], "city pop"),
    ("japanese, korean and chinese ballads", ["ja", "ko", "zh"], "ballads"),
    ("instrumental for focus", ["instrumental"], "for focus"),
    ("songs with no vocals", ["instrumental"], "songs with"),
    ("lagu bahasa indonesia", ["id"], "lagu"),
    ("日本語 rock", ["ja"], "rock"),
    ("español", ["es"], "music"),
    ("J-pop for a rainy day", [], "J-pop for a rainy day"),
    ("French house bangers", [], "French house bangers"),
    ("Latin jazz", [], "Latin jazz"),
    ("Spanish guitar", [], "Spanish guitar"),
    ("K-pop and Korean ballads", ["ko"], "K-pop and ballads"),
    ("Englishman in New York", [], "Englishman in New York"),
    ("Latina summer", [], "Latina summer"),
    ("rock and roll", [], "rock and roll"),
])
def test_parse_prompt(text, langs, rest):
    assert parse_prompt(text) == (langs, rest)
```

- [ ] **Step 2: Run them and see them fail.**

- [ ] **Step 3: Implement `parse_prompt`**
  - Build one regex alternation, longest phrases first, matched case-insensitively. The pieces:
    - every `NAMES` value;
    - endonyms and aliases:
      - 日本語 → `ja`; 한국어 → `ko`; 中文 / 汉语 / 華語 → `zh`
      - bahasa / bahasa indonesia / indonesia → `id`
      - español → `es`; français → `fr`; deutsch → `de`; русский → `ru`
      - português → `pt`; italiano → `it`
      - "Chinese", "Mandarin" and "Cantonese" → `zh`
    - the instrumental words in spec §8.
  - Boundaries are lookarounds on `\w` and `-`, so "J-pop" and "Englishman" don't match.
    CJK endonyms match without word boundaries.
  - Replace the protected genres (spec §8) with placeholders first and restore them after.
  - A connector ("or", "and", ",") is removed only when it sits between two removed matches.
  - Collapse whitespace and strip it, along with any stray leading or trailing comma.

- [ ] **Step 4: Run them and see them pass.**

- [ ] **Step 5: Commit.** `git commit -m "feat: language words in prompts"`

---

### Task 3: Whisper backend (`whisper.py`)

**Files:**
- Create: `src/nil_dj/whisper.py`, `scripts/make_whisper_reference.py`,
  `tests/fixtures/whisper-reference.npz`, `tests/test_whisper.py`
- Modify: `tests/fakes.py`, `tests/reference_signals.py` (add a 16-kHz voice-like signal if useful)

**Interfaces:**
- Consumes: `languages.NAMES`, `embedder.ModelDownloadFailed`, `gpu.list_adapters` and
  `gpu.pick_adapter` (as `OnnxClapEmbedder` uses them).
- Produces:
  - Constants: `REPO`, `REVISION`, `MODEL_ID` (Global Constraints) and
    `FILES = ("tokenizer.json", "onnx/encoder_model.onnx", "onnx/decoder_model.onnx")`.
  - `to_16k(x48: np.ndarray) -> np.ndarray` (float32)
  - `features(x16: np.ndarray) -> np.ndarray`: shape (80, 3000), float32
  - `class LanguageDetector(Protocol)`: `model_id: str`;
    `detect(self, feats: np.ndarray) -> tuple[dict[str, float], float]`. It returns probabilities
    over Whisper codes (aliases not merged) and the no-speech value.
  - `class WhisperDetector`: `__init__(self, models_dir: Path | None, gpu_device_id: int | None = None, use_gpu: bool = True)`
  - `@dataclass(frozen=True) DetectorFactory(model_id: str, load: Callable[[], LanguageDetector])`
  - `whisper_present(models_dir: Path | None) -> bool`
  - `default_detector(cfg: Config, use_gpu: bool = True) -> DetectorFactory`
  - `tests/fakes.py`:
    - `FakeDetector(model_id="fake-whisper", answer=lambda feats: ({"en": 1.0}, 0.1))`.
      It counts `calls` and can raise.
    - `fake_detector_factory(det=None) -> DetectorFactory`.

- [ ] **Step 1: Write the failing tests** (`tests/test_whisper.py`)

```python
def test_to_16k_keeps_a_1khz_tone():
    t = np.arange(48000 * 2) / 48000
    y = to_16k(np.sin(2 * np.pi * 1000 * t).astype(np.float32))
    assert y.size == 32000 and abs(rms(y[2000:-2000]) - 1 / np.sqrt(2)) < 0.01

def test_to_16k_attenuates_above_8khz_by_60db():
    t = np.arange(48000 * 2) / 48000
    y = to_16k(np.sin(2 * np.pi * 9000 * t).astype(np.float32))
    assert 20 * np.log10(rms(y[2000:-2000]) / (1 / np.sqrt(2))) <= -60

def test_features_shape_and_short_input_is_padded():
    assert features(np.zeros(16000, np.float32)).shape == (80, 3000)
    assert features(np.ones(16000 * 40, np.float32)).dtype == np.float32   # cut to 30 s

def test_features_match_transformers_reference():
    ref = np.load(FIXTURES / "whisper-reference.npz")
    for name in ("noise", "tone", "voice"):
        assert np.abs(features(ref[f"in_{name}"]) - ref[f"feat_{name}"]).max() <= 1e-3

@pytest.mark.slow
def test_whisper_detects_english_speech_and_silence(tmp_path):
    det = WhisperDetector(tmp_path)
    probs, ns = det.detect(features(np.zeros(16000 * 30, np.float32)))
    assert len(probs) == 99 and abs(sum(probs.values()) - 1) < 1e-3 and ns > 0.5
```

The `-60` limit is the 257-tap β 8 design. If it falls short by under 3 dB, use the
measured value and note it in the commit. Do not change the filter.

- [ ] **Step 2: Write `scripts/make_whisper_reference.py`**
  - Follow `scripts/make_reference.py`.
  - Use `WhisperFeatureExtractor.from_pretrained("openai/whisper-small")`.
  - Inputs, all at 16 kHz and stored as `in_<name>`:
    - `to_16k(noise_10s())`
    - `to_16k(tone_4s())`
    - "voice": 8 s of an amplitude-modulated harmonic series at 140 Hz from `reference_signals`
  - Store `feat_<name>`.
  - Run `uv run --group parity python scripts/make_whisper_reference.py`. Commit the `.npz`.

- [ ] **Step 3: Run the tests and see them fail.** `uv run pytest tests/test_whisper.py -v`

- [ ] **Step 4: Implement `whisper.py`**
  - The mel and resampler are the probe's (spec §3.1). Frame in float32 and use `np.fft.rfft`
    on float32 frames.
  - `WhisperDetector`:
    - Download with `hf_hub_download(REPO, f, revision=REVISION, local_dir=models_dir / "whisper-small" / REVISION)`.
      Also try `onnx/encoder_model.onnx_data` and `onnx/decoder_model.onnx_data`; a missing one
      is fine (`EntryNotFoundError`).
    - A network or disk failure raises `ModelDownloadFailed`, as `embedder._fetch` does.
    - Token ids come from `tokenizer.json` `added_tokens` (spec §3.1). Assert there are 99
      language tokens.
    - `detect` is the probe's `Whisper.detect` (encoder, then one decoder step from `[[sot]]`).
  - `whisper_present` checks that `FILES` exist under the revision folder.

- [ ] **Step 5: Run the tests and see them pass, slow test included.**
  `uv run pytest tests/test_whisper.py -v -m "not parity"` passes. The slow test downloads about
  1 GB once.

- [ ] **Step 6: Commit.** `git commit -m "feat: whisper-small language detector"`

---

### Task 4: Index schema 1.1

**Files:**
- Modify: `src/nil_dj/index.py`, `tests/fakes.py` (`add_track` language kwargs)
- Test: `tests/test_index.py`

**Interfaces:**
- Consumes: `languages.Verdict`-shaped finals (Task 5 writes them through `set_final`).
- Produces:
  - `SCHEMA_VERSION = "1.1"` and the six columns of spec §5.1, created for new indexes and added
    by migration.
  - `Row` gains `language: str | None = None` and `language_alt: str | None = None`. `load` and
    `load_paths` fill them.
  - `upsert(..., embedding, language_raw: str | None = None)`. INSERT OR REPLACE leaves the final
    columns NULL until resolve.
  - `set_language_raw(path: str, raw: str) -> None`: one commit.
  - `use_language_model(model_id: str) -> None`. If meta `language_model` differs, it sets every
    `language_raw` and final column to NULL, then writes the meta, in one transaction.
  - `language_work() -> list[tuple[str, float]]`: `(path, duration_s)` where `language_raw IS NULL`,
    ordered by path.
  - `languages_pending(model_id: str) -> int`: all rows if meta `language_model` is set and
    differs, else the NULL count.
  - `@dataclass(frozen=True) LangRow(path, artist, album_artist, album, raw: str | None)`
  - `language_rows() -> list[LangRow]`
  - `@dataclass(frozen=True) Final(language: str | None, alt: str | None, conf: float | None, vocal_score: float | None, source: str | None)`
  - `set_final(finals: dict[str, Final]) -> None`: one transaction.
  - `has_languages() -> bool`: some row has a non-NULL `language` other than `unknown`.
  - `language_counts() -> list[tuple[str, int]]`: by count descending, NULL shown as
    `"not detected"`.
  - `unsure_tracks() -> list[tuple[str, str | None, str | None, str, float, str]]`:
    `(path, artist, title, language, conf, source)` where `language_conf < CONF_MIN` and the
    language is not instrumental or unknown.
  - `failed_languages() -> list[tuple[str, str]]`: `(path, reason)`.
  - `add_track(..., language=None, language_alt=None)` in fakes writes the finals through
    `set_final` with source `override`.

- [ ] **Step 1: Write the failing tests**

```python
def test_version_1_index_migrates_in_place(tmp_path):
    p = tmp_path / "i.sqlite"
    make_v1_index(p)                       # helper: the old SCHEMA text + one row, schema "1"
    with closing(Index.open(p, "fake")) as ix:
        assert ix.get_meta("schema_version") == "1.1"
        assert ix.language_work() == [(ROW_PATH, 240.0)]
        rows, m = ix.load()
        assert rows[0].language is None and m.shape == (1, 512)

def test_old_check_accepts_1_1():         # major-only check is unchanged
    ...open a 1.1 index with the version check monkeypatched to SCHEMA_VERSION = "1"; no error

def test_upsert_then_raw_then_final(tmp_path): ...
    # upsert without raw → in language_work; set_language_raw → not; set_final → load() has it;
    # upsert again (file changed) → finals NULL and back in language_work

def test_model_change_clears_raw_only(tmp_path): ...
    # embeddings and stamps unchanged; language_work() == all rows; pending counts all rows
    # before use_language_model is called

def test_has_languages_ignores_unknown(tmp_path): ...
```

- [ ] **Step 2: Run them and see them fail.**

- [ ] **Step 3: Implement**
  - Do the migration in `Index.open` after the version check. If `PRAGMA table_info(tracks)` lacks
    `language_raw`, run `ALTER TABLE` for each column and update `schema_version` inside one
    `with db:`.
  - The CREATE schema gains the same six columns.

- [ ] **Step 4: Run all tests (`uv run pytest`) and see them pass.** Existing index tests are
  unchanged.

- [ ] **Step 5: Commit.** `git commit -m "feat: index schema 1.1 with language columns"`

---

### Task 5: Overrides and resolve

**Files:**
- Modify: `src/nil_dj/languages.py`
- Test: `tests/test_languages.py`

**Interfaces:**
- Consumes: `Index.language_rows`, `Index.set_final`, `LangRow`, `Final`, `index.path_key`.
- Produces:
  - `@dataclass(frozen=True) Overrides(tracks: dict[str, tuple[str, str | None]], albums: dict[tuple[str, str], tuple[str, str | None]], artists: dict[str, tuple[str, str | None]])`.
    Track keys are `path_key`; names are casefolded.
  - `load_overrides(path: Path) -> tuple[Overrides, list[str]]`.
    - A missing file gives empty overrides and no problems.
    - Any problem gives empty overrides plus messages in the form
      `languages.toml: [artist] "X": unknown language 'xx'`, or `languages.toml: <TOML error>`.
  - `resolve(rows: list[LangRow], overrides: Overrides) -> dict[str, Final]`: covers rows with
    raw only.
  - `resolve_index(index: Index, overrides_path: Path) -> list[str]`. It writes the finals and
    meta `overrides_mtime_ns` (`"0"` when the file is missing), and returns the problems.
  - `overrides_changed(index: Index, overrides_path: Path) -> bool`

- [ ] **Step 1: Write the failing tests**

```python
def test_resolve_order(): ...
    # track override > album override > artist override > artist majority > whisper,
    # each with source 'override' / 'override' / 'override' / 'artist' / 'whisper'
def test_track_override_matches_path_key():   # 'E:\\Music\\A\\x.flac' vs 'e:/music/a/X.FLAC'
def test_album_override_uses_album_artist_else_artist():
def test_names_compare_casefolded():          # "BLACKPINK" override hits artist "Blackpink"
def test_artist_majority_thresholds():
    # unsure sung track + 3 confident 'id' whisper tracks of the same artist → 'id', 'artist'
    # only 2 confident → stays whisper; 3 'id' + 2 'en' (60 %) → stays whisper
def test_artist_majority_never_touches_instrumentals():
def test_failed_raw_gives_unknown():
def test_mixed_override_value():              # "ja+en" → ('ja', 'en')
@pytest.mark.parametrize("toml, problem", [
    ("[artist]\nX = 'xx'", "unknown language 'xx'"),
    ("[artist]\nX = 3", "must be a language code"),
    ("[album]\nX = 'ja'", "[album] needs an artist table"),
    ("[bogus]\n", "unknown section"),
    ("[artist\n", "languages.toml:"),
])
def test_bad_override_file_is_ignored_whole(tmp_path, toml, problem): ...
```

- [ ] **Step 2: Run them and see them fail.**

- [ ] **Step 3: Implement as spec §6.**
  - `resolve` counts artist majorities once, over rows whose own verdict is confident (not
    unsure), sung, and not overridden.
  - Finals use `verdict(...)` for `conf` and `vocal_score`. Override finals keep `vocal_score`
    from raw when there is one and set `conf = None`.

- [ ] **Step 4: Run them and see them pass.**

- [ ] **Step 5: Commit.** `git commit -m "feat: language overrides and resolve"`

---

### Task 6: Indexer: windows, slicing, detection, catch-up

**Files:**
- Modify: `src/nil_dj/indexer.py`
- Test: `tests/test_indexer.py`

**Interfaces:**
- Consumes:
  - `whisper.to_16k`, `whisper.features`, `whisper.DetectorFactory`;
  - `languages.Window`, `encode_raw`, `encode_failed`;
  - `Index.upsert(language_raw=)`, `set_language_raw`, `use_language_model`, `language_work`.
- Produces:
  - `WINDOW_S = 30.0`
  - `language_windows(duration_s: float) -> list[tuple[float, float]]`: `[(0, d)]` when
    d ≤ 30; else three windows at 0.2, 0.5 and 0.8, clamped as in spec §4.
  - `Progress` gains `phase: str = "index"`; the catch-up uses `"languages"`.
  - `RunResult` gains `detected: int = 0`, `language_failed: int = 0` and
    `language_problem: str | None = None`.
  - `run_index` and `run_index_paths` gain a keyword-only `detector: DetectorFactory | None = None`.
    None means no language work at all, as today.
  - `LANGUAGE_MODEL_FAILED = "Language detection needs the language model, which couldn't be downloaded. Check the internet connection, then press Update index again."`

- [ ] **Step 1: Write the failing tests**

```python
def test_language_windows():
    assert language_windows(12.0) == [(0, 12.0)]
    assert language_windows(30.0) == [(0, 30.0)]
    assert language_windows(200.0) == [(25.0, 30.0), (85.0, 30.0), (145.0, 30.0)]

@pytest.mark.parametrize("d", [4.0, 12.0, 31.0, 200.0])
def test_every_clap_clip_lies_inside_a_language_window(d): ...

def test_sliced_clips_equal_separate_decodes_at_48k(make_audio):      # bit-exact
def test_sliced_clips_close_to_separate_decodes_at_44k(make_audio):   # mean log-mel diff ≤ 0.5 dB
def test_short_decode_is_padded_not_crashing(make_audio, monkeypatch): # tags say 60 s, audio 20 s

def test_new_tracks_get_raw_language(tmp_path, make_audio):
    det = FakeDetector(answer=lambda f: ({"ja": 0.9, "en": 0.1}, 0.2))
    run_index([...], index, FakeEmbedder(), lock, detector=fake_detector_factory(det))
    assert decode_raw(raw_of(index, path))[0].p == {"ja": 0.9, "en": 0.1}
    assert det.calls == len(language_windows(duration))

def test_catch_up_detects_only_undetected_and_never_embeds(...):
    # index once without a detector; then run with one: embed_audio not called (a FakeEmbedder
    # subclass that raises), every row now has raw, phase "languages" progress reaches total
def test_catch_up_resumes_after_stop(...):   # stop after the 1st language track; 2nd run does the rest
def test_detector_load_failure_keeps_clap_work(...):
    # factory.load raises ModelDownloadFailed(MODEL_DOWNLOAD_FAILED): new tracks indexed,
    # raw NULL, result.language_problem == LANGUAGE_MODEL_FAILED
def test_detect_error_on_one_track_records_failed_raw(...):  # not in index.errors()
def test_no_detector_means_no_language_work(...):            # existing behaviour, raw NULL
def test_model_change_triggers_catch_up(...):                # model_id differs → all rows again
def test_detector_loaded_only_when_needed(...):              # nothing pending → load not called
```

- [ ] **Step 2: Run them and see them fail.**

- [ ] **Step 3: Implement**
  - `_analyse(path, want_mels: bool)`:
    - Decode each language window once.
    - Slice each CLAP clip at `round((clip_start − win_start) * 48000)` for `round(length * 48000)`
      samples, padding with zeros if the decode was short.
    - Tempo and loudness stay as today.
    - With `want_mels`, also return `[features(to_16k(w)) for w in windows]`.
  - `_analyse_language(path, duration_s) -> list[np.ndarray]` runs on a worker.
  - `_run`:
    - If `detector` is given, call `index.use_language_model(detector.model_id)` before the
      work lists are computed.
    - Language-only work is `language_work()` minus the paths already in the full work list and
      minus deleted rows.
    - Load the detector lazily, the first time a mel needs it. A load failure sets
      `language_problem` (use `LANGUAGE_MODEL_FAILED` for the network message, else
      `str(e)`) and turns detection off for the rest of the run.
    - Process the full work first (phase `"index"`), then the catch-up (phase `"languages"`).
      Use the same look-ahead pool for both. Before the first catch-up item, emit
      `Progress(0, n, "", None, "languages")`.
    - A `DecodeError` or `OSError` in `_analyse_language`, or any `Exception` from `detect`,
      stores `encode_failed(reason)` and counts `language_failed`.
    - Drop the detector reference when `_run` returns.

- [ ] **Step 4: Run all tests and see them pass.**

- [ ] **Step 5: Commit.** `git commit -m "feat: detect languages while indexing, with catch-up"`

---

### Task 7: Runner, DJ status and panel

**Files:**
- Modify: `src/nil_dj/reindex.py`, `src/nil_dj/dj.py`, `src/nil_dj/panel.py`, `src/nil_dj/plugin.py`,
  `src/nil_dj/server.py`
- Test: `tests/test_dj.py`, `tests/test_panel.py`, `tests/test_plugin.py`

**Interfaces:**
- Consumes: Task 6's `detector=`, `RunResult.language_problem` and `Progress.phase`;
  `resolve_index` and `overrides_changed`; `load_overrides`; `Index.languages_pending`;
  `whisper.MODEL_ID` and `default_detector`.
- Produces:
  - `IndexRunner(cfg, paths, embedder, on_change, detector: DetectorFactory | None = None)`.
    After `run_index_paths` it calls
    `resolve_index(index, cfg.plugin_dir / "languages.toml")`; it does so after a stopped run too.
  - `language_problem` becomes the run's problem (rebuild_needed False).
  - `RunState` gains `phase: str = "index"`.
  - `DJ(cfg, nil, make_embedder, clock=..., detector: DetectorFactory | None = None)`
  - `IndexStatus` gains `languages_pending: int = 0`, `language_problems: list[str]` and
    `phase: str = "index"`.
    - `languages_pending` is 0 when `detector` is None.
    - `language_problems` comes from `load_overrides`.
  - `DJ.preload`'s thread first runs `resolve_index` when `overrides_changed`. It skips this when
    the index lock is held, because the run will resolve.
  - Panel:
    - progress label `f"Detecting languages {done:,} / {total:,}"` when `phase == "languages"`;
    - the idle line gains `" · {n:,} languages not detected yet"`;
    - `language_problems` show as muted lines.
  - `plugin._start`, `serve_stdio` and `serve_http` pass `detector=default_detector(cfg)`.

- [ ] **Step 1: Write the failing tests**
  - `test_status_counts_languages_pending` (fake detector factory, rows without raw)
  - `test_runner_resolves_after_run` (finals set; `languages.toml` override applied)
  - `test_runner_reports_language_model_failure` (status.problem == LANGUAGE_MODEL_FAILED, indexed > 0)
  - `test_startup_resolves_when_overrides_changed`
  - `test_bad_overrides_show_in_status`
  - panel: `test_progress_label_for_languages`, `test_idle_line_mentions_undetected_languages`,
    `test_update_offered_while_languages_pending`
  - `test_plugin_start_passes_a_detector` (monkeypatch `default_detector`)

- [ ] **Step 2: Run them and see them fail.**

- [ ] **Step 3: Implement as in Interfaces.**

- [ ] **Step 4: Run all tests and see them pass.**

- [ ] **Step 5: Commit.** `git commit -m "feat: language progress and status in the DJ tab"`

---

### Task 8: CLI

**Files:**
- Modify: `src/nil_dj/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `default_detector`, `resolve_index`, `Index.language_counts`, `unsure_tracks`,
  `failed_languages`.
- Produces:
  - `nil-dj index`:
    - passes `detector=default_detector(cfg, use_gpu=not args.cpu)`, or the equivalent flag
      the command already has;
    - resolves after the run;
    - prints `Languages: {detected} detected, {failed} failed` when there was language work;
    - prints a language problem on stderr and still exits 0;
    - `--no-languages` skips detection.
  - `nil-dj languages [--resolve]`:
    - prints counts as `ja  1,204` lines;
    - then a section for unsure tracks: `artist — title  guess conf source`;
    - then one for failed tracks;
    - `--resolve` runs resolve first and prints any problems;
    - exits 1 with the not-ready message when there is no index.

- [ ] **Step 1: Write the failing tests** `test_index_detects_languages` (fake factory
  monkeypatched), `test_index_no_languages_flag`, `test_languages_lists_counts_and_unsure`,
  `test_languages_resolve_reports_bad_toml`.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them and see them pass.**
- [ ] **Step 5: Commit.** `git commit -m "feat: nil-dj languages"`

---

### Task 9: Asking for a language

**Files:**
- Modify: `src/nil_dj/search.py`, `src/nil_dj/dj.py`, `src/nil_dj/server.py`
- Test: `tests/test_search.py`, `tests/test_dj.py`, `tests/test_server.py`

**Interfaces:**
- Consumes: `normalise_code`, `parse_prompt`, `Index.has_languages`, `Row.language` and
  `Row.language_alt`.
- Produces:
  - `Filters.languages: list[str] | None = None`. `filters_from` normalises the codes, and an
    unknown code is a `ValueError` naming it.
  - SQL: `(language IN (…) OR language_alt IN (…))`. An `instrumental` code adds
    `language = 'instrumental'` through the same IN list.
  - `Track` gains `language: str | None` and `language_alt: str | None`. `as_dict` includes
    `language` always and `language_alt` only when set.
  - `DJ.search_tracks` and `DJ.build_playlist` run `parse_prompt` on the query.
    - Explicit `filters["languages"]` wins over the prompt words, but the words are still
      removed.
    - When a language is requested and `not index.has_languages()`, they raise
      `ValueError(LANGUAGES_NOT_READY)`.
    - The playlist name stays the user's full prompt.
  - MCP:
    - the `search_tracks`, `similar_tracks` and `build_playlist` descriptions name
      `filters.languages` (codes such as `ja`, `en`, `id`, `instrumental`);
    - results carry `language`.

- [ ] **Step 1: Write the failing tests**

```python
def test_languages_filter_matches_language_or_alt(tmp_path): ...  # ja, en+ja, en, NULL rows
def test_instrumental_filter(tmp_path): ...
def test_unknown_language_code_is_an_error(): ...
def test_prompt_words_filter_and_are_removed(dj_with_rows):
    # FakeEmbedder sees "city pop" (record embed_text args); only ja/en rows come back
def test_explicit_filter_wins_over_prompt_words(dj_with_rows): ...
def test_language_request_before_detection_says_not_ready(dj_with_rows): ...
def test_genre_prompt_is_not_a_language_request(dj_with_rows): ...  # "J-pop" → no filter
def test_results_carry_language(dj_with_rows): ...
async def test_mcp_filter_and_field(server): ...
```

- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run all tests and see them pass.**
- [ ] **Step 5: Commit.** `git commit -m "feat: ask for a language in prompts and filters"`

---

### Task 10: Radio stays in the language

**Files:**
- Modify: `src/nil_dj/config.py`, `src/nil_dj/order.py`, `src/nil_dj/radio.py`, `src/nil_dj/dj.py`
- Test: `tests/test_order.py`, `tests/test_radio.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: `Row.language`, `Row.language_alt`, `INSTRUMENTAL`, `UNKNOWN`.
- Produces:
  - `Weights.language: float = 0.30`. `[order] language = 0.5` in `config.toml` works through
    the existing field validation.
  - `Candidate.languages: frozenset[str] = frozenset()`; empty means neutral. `candidate()`
    fills it from the row.
  - `anchor_of(played: Sequence[Candidate]) -> frozenset[str] | None`: the newest non-empty
    language set.
  - `_score(w, prev, t, match, anchor)` adds `w.language` when an anchor exists and either
    `t.languages` is empty or it meets the anchor. `order` computes the anchor from
    `[*history, *chosen]` on each step.
  - `Radio.pick`: the pool is the top `POOL_SIZE` overall ∪ the top `POOL_SIZE` among the allowed
    tracks that meet the anchor of `hist`, kept in similarity order.
  - `DJ._playlist` passes `replace(self.cfg.weights, language=0.0)`.

- [ ] **Step 1: Write the failing tests**
  - `test_anchor_skips_neutral_tracks`
  - `test_neutral_candidate_gets_bonus`
  - `test_no_anchor_no_bonus`
  - `test_same_language_beats_slightly_more_similar_other_language`
  - `test_crosses_over_when_language_runs_out`
  - `test_radio_pool_includes_rare_anchor_language` (41st most similar is the only `ja`;
    anchor `ja` → chosen)
  - `test_playlists_ignore_language` (a `_playlist` spy on the weights)
  - `test_config_order_language`

- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run all tests and see them pass.**
- [ ] **Step 5: Commit.** `git commit -m "feat: radio keeps the language of what is playing"`

---

### Task 11: Evaluation, docs, Windows validation

**Files:**
- Create: `eval/language.py`, `eval/radio_language.py`
- Modify: `README.md`, `docs/superpowers/specs/2026-10-22-nil-dj-design.md` (§6 pointer),
  `docs/validation-log.md`
- Test: `tests/test_eval_language.py` (label parsing only)

**Interfaces:**
- Consumes: the index (`Index.open(path, None)`), `config.load_config`, `order`, `Radio` internals
  via `Radio.pick` with a fake clock.
- Produces:
  - `eval/language.py --labels CSV [--index PATH]`
    - Matches rows by casefolded (artist, title).
    - The expected label is `your_label`, else `whisper_guess`:
      - `instrumental? (xx)` → instrumental; `jw`/`ms`/`su` → id;
      - `mixed a b` → correct if the language is a or b;
      - plain `mixed` → correct if `language_alt` is set;
      - `?` → skipped.
    - Prints `correct/total`, the confusion counts by code, and how many labels had no row.
      No titles.
  - `eval/radio_language.py`:
    - For each language with ≥ 50 sung tracks, run 10 simulated 2-hour sessions from random seeds
      of that language.
    - Print the share of sung picks in the anchor language, and the mean similarity drift
      (main spec §12) with language 0.30 vs 0.

- [ ] **Step 1: Write the failing tests** for the label rules (`parse_label`, `is_correct`).
- [ ] **Step 2: Implement the scripts. Run the tests and see them pass.**
- [ ] **Step 3: Docs**
  - README "Languages" section (spec §12). Include the measured download size of whisper-small
    from `models/whisper-small`.
  - The main spec §6 gets a pointer.
- [ ] **Step 4: Windows validation.** Sync with `scripts/sync-to-windows.sh`, then in the plugin
  app venv:
  1. `nil-dj index`: the catch-up time, the windows per second and the mel time per window go
     in `docs/validation-log.md`. The mel target is ≤ 0.03 s per window (spec §3.1).
  2. `eval/language.py --labels E:\projects\nil-dj-eval\language-labels.csv`: needs ≥ 123/124.
  3. `eval/radio_language.py`: needs ≥ 90 % and drift ≤ 10 % worse. Otherwise tune
     `Weights.language`, re-run, and record the value.
  4. For 20 random tracks indexed before this change, re-embed with the new slicing.
     Cosine ≥ 0.99 against the stored embedding (Ruling 1).
  5. The owner's panel check: progress label, a "Japanese city pop" prompt, radio on a Japanese
     track.
- [ ] **Step 5: Commit.** `git commit -m "docs: languages README, evaluation scripts, validation"`
