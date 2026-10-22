# nil-dj Plan 1 — Index and search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `nil-dj index` builds `index.sqlite` from the owner's library with CLAP (ONNX Runtime, DirectML on Windows), and `nil-dj search "<text>"` / `nil-dj similar <path>` return ranked tracks from it.

**Architecture:** A `src/`-layout Python package. Small modules with one job each (`config`, `features`, `embedder`, `gpu`, `decode`, `tempo`, `tags`, `index`, `indexer`, `search`, `cli`). The embedder sits behind a protocol so tests use a fake. Everything that touches real audio uses ffmpeg in a subprocess. Plan 1 of 5 from the spec's build order (§14); plans 2–5 (nil control API, MCP server, radio, eval) come after this works on the real library.

**Tech Stack:** Python 3.12, uv, numpy, onnxruntime / onnxruntime-directml, huggingface_hub, tokenizers, mutagen, filelock, ffmpeg (external), pytest, ruff. Dev-only parity group: torch (CPU) + transformers 4.x.

**Spec:** `docs/superpowers/specs/2026-10-22-nil-dj-design.md` (§3, §5, §5.1, §6, §7 Search, §10, §11 nil-dj part, §14 step 1).

## Global Constraints

- Repo: `~/projects/nil-dj` (WSL). Windows copy: `E:\projects\nil-dj`. Package name `nil-dj`, import name `nil_dj`, console script `nil-dj`. Licence MIT.
- `requires-python = ">=3.12"`. Dependencies: `onnxruntime>=1.24 ; sys_platform != "win32"`, `onnxruntime-directml==1.24.4 ; sys_platform == "win32"` (never both on one platform), `numpy`, `huggingface_hub`, `tokenizers`, `mutagen`, `filelock`, `brotli>=1.2` (huggingface_hub's HTTP stack failed with `TypeError: process() takes no keyword arguments` against brotli 1.0.9).
- No PyTorch and no `transformers` in runtime dependencies.
- Model: repo `Xenova/clap-htsat-unfused`, revision `c28f2883575e590e04d3146ff0713c2448d691ba`; files `onnx/audio_model{,_fp16,_quantized}.onnx`, `onnx/text_model{,_fp16,_quantized}.onnx`, `tokenizer.json`. Precision names: `fp32` → suffix `""`, `fp16` → `"_fp16"`, `int8` → `"_quantized"`. ONNX I/O: audio `input_features` float32 `(batch, 1, 1001, 64)` → `audio_embeds (batch, 512)`; text `input_ids` int64 `(batch, seq)` → `text_embeds (batch, 512)`. Outputs are **not** normalised.
- `model_id` stored in the index: `"laion/clap-htsat-unfused"` (all precisions share it). `schema_version` `"1"`, `dim` `"512"`.
- Index path: `%LOCALAPPDATA%\nil-dj\index.sqlite` on Windows; `$XDG_DATA_HOME/nil-dj/index.sqlite` (default `~/.local/share/nil-dj/`) elsewhere; env `NIL_DJ_INDEX` overrides. nil config dir: `%APPDATA%\bitperfect\` on Windows; `~/.config/bitperfect/` elsewhere; env `NIL_CONFIG_DIR` overrides.
- Paths are stored exactly as produced by walking the configured folder with the OS separator (on Windows `E:\Music\…`), never normalised, case-folded or resolved, so they equal nil's paths.
- Audio extensions: `flac wav aif aiff m4a dsf dff`, compared lower-cased. Symlinks (files and directories) are skipped.
- Tests never download the model or need a GPU unless marked `slow`; `parity` tests need the parity group. `pytest` default run: `-m "not slow and not parity"`.
- Commits: no Co-Authored-By or "Generated with" lines.

## Review Focus

1. **Paths differ from nil's** (Japanese names, nested folders, mixed-case extensions): the indexed path set must equal nil's `library-cache.json` path set minus decode errors. Pinned in Task 12 (Windows validation) and by `test_walk_matches_nil_rules` in Task 9.
2. **Very short or silent tracks** (a 2 s interlude, a silent hidden track): indexing must succeed with `bpm` and `lufs` NULL, not crash or store `-inf`. Pinned by `test_short_and_silent_tracks` in Task 9.
3. **A bad file** (zero bytes, a text file renamed `.flac`, a file deleted mid-run): recorded in `index_errors`, the run continues, and a later successful index clears its error. Pinned by `test_bad_files_recorded_and_cleared` in Task 9.
4. **Interrupted run** (Ctrl+C halfway): rows already committed stay, the lock is released, and the next run only does the rest. Pinned by `test_interrupt_keeps_progress` in Task 9.
5. **Search that matches nothing** (empty index, or filters that exclude everything): returns an empty list, and the CLI prints a clear message with exit code 0. Pinned by `test_empty_results` in Task 10 and `test_cli_search_no_index` in Task 11.

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `README.md` (one paragraph + "work in progress"), `LICENSE` (MIT, "reubinars"), `.gitignore` (`.venv/ __pycache__/ .pytest_cache/ .ruff_cache/ dist/`), `src/nil_dj/__init__.py` (`__version__ = "0.1.0"`), `src/nil_dj/cli.py`, `tests/test_cli.py`, `.github/workflows/ci.yml`, `scripts/sync-to-windows.sh`

**Interfaces:**
- Produces: `nil_dj.cli.main(argv: list[str] | None = None) -> int`, argparse with subcommands added by later tasks; `--version` prints `nil-dj 0.1.0`.

- [ ] **Step 1: Write `pyproject.toml`** with hatchling build backend, the dependencies from Global Constraints, `[project.scripts] nil-dj = "nil_dj.cli:main"`, `[dependency-groups] dev = ["pytest", "ruff"]` and `parity = ["torch>=2.6", "transformers>=4.57,<5"]`, a `[[tool.uv.index]]` named `pytorch-cpu` at `https://download.pytorch.org/whl/cpu` with `explicit = true` and `[tool.uv.sources] torch = { index = "pytorch-cpu" }`, `[tool.pytest.ini_options]` with `addopts = "-m 'not slow and not parity'"` and markers `slow`, `parity`, `windows`; `[tool.ruff] line-length = 100`.
- [ ] **Step 2: Write the failing test**

```python
def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert capsys.readouterr().out.strip() == "nil-dj 0.1.0"
```

- [ ] **Step 3: Run** `uv run pytest tests/test_cli.py -v` → FAIL (no `main`).
- [ ] **Step 4: Implement `main`** in `src/nil_dj/cli.py` (argparse, `version=f"nil-dj {__version__}"`).
- [ ] **Step 5: Run** `uv run pytest -v && uv run ruff check .` → PASS, no lint errors.
- [ ] **Step 6: Write `.github/workflows/ci.yml`**: ubuntu-latest, `apt-get install -y ffmpeg`, `astral-sh/setup-uv`, `uv sync`, `uv run ruff check .`, `uv run pytest`.
- [ ] **Step 7: Write `scripts/sync-to-windows.sh`**: `rsync -a --delete --exclude .git --exclude .venv --exclude __pycache__ --exclude .pytest_cache --exclude .ruff_cache ~/projects/nil-dj/ /mnt/e/projects/nil-dj/` (excluded paths are never deleted, so the Windows `.venv` survives). `chmod +x`.
- [ ] **Step 8: Commit** `git add -A && git commit -m "build: project scaffold, CLI entry point, CI, Windows sync script"`

### Task 2: Config

**Files:**
- Create: `src/nil_dj/config.py`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `nil_config_dir() -> Path`, `default_index_path() -> Path` (rules in Global Constraints).
  - `music_folders(config_dir: Path | None = None) -> list[Path]`: reads `library.toml` key `folders` with `tomllib`; missing file or key → `[]`.
  - `@dataclass Config: index_path: Path; folders: list[Path]; precision: str = "fp32"; gpu_device_id: int | None = None; prompt_template: str = "{}"`
  - `load_config() -> Config`.

- [ ] **Step 1: Write failing tests**
  - `test_music_folders_reads_nil_library_toml`: `tmp_path/"library.toml"` containing `folders = ['E:\Music', 'D:\More']` (TOML literal strings, as nil writes them) → `[Path("E:\\Music"), Path("D:\\More")]` compared as strings `["E:\\Music", "D:\\More"]`.
  - `test_music_folders_missing_file` → `[]`.
  - `test_env_overrides(monkeypatch)`: `NIL_DJ_INDEX=/x/i.sqlite`, `NIL_CONFIG_DIR=/y` → `default_index_path() == Path("/x/i.sqlite")`, `nil_config_dir() == Path("/y")`.
  - `test_defaults_non_windows(monkeypatch)` (skip on win32): `HOME=/h`, no `XDG_DATA_HOME` → `Path("/h/.local/share/nil-dj/index.sqlite")`, `Path("/h/.config/bitperfect")`.
- [ ] **Step 2: Run** `uv run pytest tests/test_config.py -v` → FAIL.
- [ ] **Step 3: Implement** in `src/nil_dj/config.py`. Keep folder strings as read; build with `Path(s)` only at the call site that walks them.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: config (nil folders, index path, env overrides)`

### Task 3: Log-mel features and the PyTorch reference fixture

**Files:**
- Create: `src/nil_dj/features.py`, `tests/reference_signals.py`, `scripts/make_reference.py`, `tests/fixtures/clap-reference.npz` (generated), `tests/test_features.py`

**Interfaces:**
- Produces:
  - `SAMPLE_RATE = 48000`, `CLIP_SAMPLES = 480000`
  - `log_mel(clip: np.ndarray) -> np.ndarray`: float32 mono at 48 kHz, length ≤ `CLIP_SAMPLES` → float32 `(1001, 64)`.
  - `tests/reference_signals.py`: `noise_10s() -> np.ndarray` (`np.random.default_rng(0).standard_normal(480000) * 0.1`, float32), `tone_4s() -> np.ndarray` (4 s of a 440 Hz sine at 0.3 plus `default_rng(1)` noise × 0.01, float32), `PROMPTS = ["This is a sound of rain", "mellow jazz piano trio", "aggressive electronic dance music", "雨の夜の静かなピアノ"]`.
  - Fixture keys: `feat_noise`, `feat_tone` (1001, 64); `audio_noise`, `audio_tone` (512,); `text` (4, 512). Embeddings unnormalised, straight from PyTorch `get_audio_features` / `get_text_features`.

- [ ] **Step 1: Write `scripts/make_reference.py`.** Inserts `tests/` on `sys.path`, imports `reference_signals`, loads `ClapModel`, `ClapFeatureExtractor`, `AutoTokenizer` from `laion/clap-htsat-unfused`, computes features with `fe(x, sampling_rate=48000, return_tensors="pt")` and embeddings under `torch.no_grad()`, then writes the npz with the keys above.
- [ ] **Step 2: Generate the fixture**: `uv run --group parity python scripts/make_reference.py`. Expected: `tests/fixtures/clap-reference.npz` around 1 MB.
- [ ] **Step 3: Write failing tests**
  - `test_log_mel_matches_reference_noise`: `np.abs(log_mel(noise_10s()) - fx["feat_noise"]).max() < 1e-4`.
  - `test_log_mel_repeat_pads_short_clip`: same against `feat_tone` with `tone_4s()`.
  - `test_log_mel_silence_is_finite`: `np.isfinite(log_mel(np.zeros(480000, np.float32))).all()`.
- [ ] **Step 4: Run** `uv run pytest tests/test_features.py -v` → FAIL.
- [ ] **Step 5: Implement `log_mel`** as a numpy port of `transformers.audio_utils` (attribute it in a comment, Apache-2.0):
  - Work in float64.
  - Repeat-pad: if `n < 480000`, `np.tile(x, 480000 // n)`, then zero-pad on the right to 480000.
  - Slaney mel filter bank: 513 frequency bins, 64 filters, 50–14000 Hz, `norm="slaney"`, `mel_scale="slaney"`.
  - Periodic Hann window of 1024.
  - Frames: reflect-pad 512 on both sides, hop 480. Power spectrum `|rfft|²`, then `mel @ power`, then `10·log10(max(v, 1e-10))`.
  - Transpose to `(1001, 64)`, cast to float32.
- [ ] **Step 6: Run** → PASS.
- [ ] **Step 7: Commit** `feat: numpy CLAP log-mel features with PyTorch reference fixture`

### Task 4: Embedder (ONNX) and GPU choice

**Files:**
- Create: `src/nil_dj/embedder.py`, `src/nil_dj/gpu.py`, `tests/test_embedder.py`, `tests/test_gpu.py`, `tests/fakes.py`

**Interfaces:**
- Consumes: `features.log_mel`, `features.CLIP_SAMPLES`.
- Produces:
  - `class Embedder(Protocol): model_id: str; def embed_audio(self, clips: list[np.ndarray]) -> np.ndarray; def embed_text(self, text: str) -> np.ndarray`. Both return float32 `(512,)` with L2 norm 1. `embed_audio` L2-normalises each clip's embedding, takes the mean, and L2-normalises again.
  - `class OnnxClapEmbedder(Embedder)`: `__init__(self, precision: str = "fp32", device_id: int | None = None, use_gpu: bool = True)`. Attribute `provider: str` (`"DmlExecutionProvider"` or `"CPUExecutionProvider"`), `adapter_name: str | None`. Batches clips in one `run` call.
  - `gpu.py`: `@dataclass Adapter: index: int; name: str; dedicated_vram: int; software: bool`; `list_adapters() -> list[Adapter]` (Windows only; `[]` elsewhere); `pick_adapter(adapters: list[Adapter]) -> Adapter | None` (most `dedicated_vram` among non-software; ties → lowest index).
  - `tests/fakes.py`: `class FakeEmbedder` whose `embed_audio` returns a normalised vector seeded from a hash of the clips' bytes (deterministic) and whose `embed_text` is seeded from the text. `model_id = "fake"`.

- [ ] **Step 1: Write failing tests**
  - `test_pick_adapter_prefers_most_vram`: `[Adapter(0,"AMD Radeon(TM) Graphics",512<<20,False), Adapter(1,"NVIDIA GeForce RTX 3060 Laptop GPU",4<<30,False), Adapter(2,"Microsoft Basic Render Driver",0,True)]` → index 1.
  - `test_pick_adapter_none_when_only_software` → `None`.
  - `test_cpu_fallback_when_no_dml(monkeypatch)`: patch `onnxruntime.get_available_providers` to `["CPUExecutionProvider"]`, and patch the internal session factory so no model is loaded → `provider == "CPUExecutionProvider"`.
  - `test_unknown_precision_rejected` → `ValueError` mentioning `fp32, fp16, int8`.
  - `@pytest.mark.slow test_onnx_matches_reference[fp32|fp16|int8]` (parametrised with thresholds 0.999 / 0.99 / 0.97): cosine of `embed_audio([noise_10s()])` with `fx["audio_noise"]`, the same for the tone, and `embed_text(p)` with each `fx["text"][i]`, each at least the threshold. Runs on whichever provider is available, so on Windows it checks DirectML.
  - `@pytest.mark.windows @pytest.mark.slow test_list_adapters_finds_hardware` (skip unless win32): at least one non-software adapter is returned.
- [ ] **Step 2: Run** `uv run pytest tests/test_gpu.py tests/test_embedder.py -v` → FAIL.
- [ ] **Step 3: Implement `gpu.list_adapters`** with ctypes, without extra dependencies:
  - `dxgi.CreateDXGIFactory1` with IID_IDXGIFactory1 `{770aae78-f26f-4dba-a829-253c83d1b387}`.
  - Vtable slots (zero-based): `IDXGIFactory1::EnumAdapters1` = 12; `IDXGIAdapter1::GetDesc1` = 10; `IUnknown::Release` = 2.
  - Loop until `DXGI_ERROR_NOT_FOUND` (`0x887A0002`).
  - `DXGI_ADAPTER_DESC1` = `WCHAR Description[128]; UINT VendorId, DeviceId, SubSysId, Revision; SIZE_T DedicatedVideoMemory, DedicatedSystemMemory, SharedSystemMemory; LUID AdapterLuid; UINT Flags`. Software flag = `2`.
- [ ] **Step 4: Implement `OnnxClapEmbedder`**:
  - Download the two ONNX files plus `tokenizer.json` with `hf_hub_download(repo, file, revision=REVISION)`.
  - Providers: when `use_gpu` and `"DmlExecutionProvider"` is available and the precision is not `int8`, use `[("DmlExecutionProvider", {"device_id": id}), "CPUExecutionProvider"]`. `id` = `device_id`, else `pick_adapter(list_adapters()).index`, else 0. DirectML sessions need `sess_options.enable_mem_pattern = False` and `execution_mode = ORT_SEQUENTIAL`.
  - If creating the session raises, warn on stderr and retry on CPU.
  - Text: `Tokenizer.from_file`, `enable_truncation(512)`, `encode(text).ids` as int64 `(1, n)`.
  - Audio: stack `log_mel` per clip into `(k, 1, 1001, 64)`.
- [ ] **Step 5: Run** the fast tests → PASS. Then `uv run pytest -m slow tests/test_embedder.py -v` (downloads the model once) → PASS for all three precisions.
- [ ] **Step 6: Commit** `feat: ONNX CLAP embedder with DirectML adapter choice and CPU fallback`

### Task 5: Decoding and loudness (ffmpeg)

**Files:**
- Create: `src/nil_dj/decode.py`, `tests/conftest.py`, `tests/test_decode.py`

**Interfaces:**
- Produces:
  - `class DecodeError(Exception)`; `class FfmpegMissing(DecodeError)`.
  - `require_ffmpeg() -> str`: path to ffmpeg, or raises `FfmpegMissing("ffmpeg not found on PATH. Install it: winget install Gyan.FFmpeg (Windows) or apt install ffmpeg")`.
  - `decode_segment(path: Path, start_s: float, length_s: float) -> np.ndarray`: float32 mono 48 kHz, via `ffmpeg -nostdin -v error -ss {start} -t {length} -i {path} -vn -ac 1 -ar 48000 -f f32le -`. Fewer samples than asked near the end is fine. Non-zero exit or 0 samples → `DecodeError(stderr tail)`. Timeout 120 s.
  - `integrated_lufs(path: Path) -> float | None`: `ffmpeg -nostdin -hide_banner -nostats -i {path} -vn -af ebur128 -f null -`. Parse the last `I:\s+(-?\d+(\.\d+)?) LUFS`. `-70.0` or lower (ebur128's floor for silence) → `None`. Non-zero exit → `DecodeError`. Timeout 600 s.
  - `tests/conftest.py` fixture `make_audio(tmp_path)`: a factory `make_audio(name, seconds, kind="sine"|"silence"|"clicks", bpm=120, rate=44100, codec="flac", tags: dict | None = None) -> Path`. It renders with ffmpeg lavfi: `sine=f=440`, `anullsrc`, and for clicks `aevalsrc` with a 10 ms burst every `60/bpm` s. Tags are written with mutagen after encoding.

- [ ] **Step 1: Write failing tests**
  - `test_decode_segment_length`: 30 s sine FLAC at 44.1 kHz, `decode_segment(p, 10, 10)` → `len == 480000 ± 480`, dtype float32.
  - `test_decode_segment_near_end`: 12 s file, `decode_segment(p, 8, 10)` → about 4 s of samples.
  - `test_decode_error_on_garbage`: a `.flac` file containing `b"not audio"` → `DecodeError`.
  - `test_lufs_of_sine`: a -20 dBFS stereo 1 kHz sine → `-24 < lufs < -16`.
  - `test_lufs_silence_is_none`: a silence file → `None`.
  - `test_ffmpeg_missing(monkeypatch)`: `shutil.which` → `None` → `FfmpegMissing`.
- [ ] **Step 2: Run** `uv run pytest tests/test_decode.py -v` → FAIL.
- [ ] **Step 3: Implement** in `src/nil_dj/decode.py` (`subprocess.run`, `capture_output=True`, `np.frombuffer(out, "<f4")`).
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: ffmpeg segment decoding and ebur128 loudness`

### Task 6: Tempo

**Files:**
- Create: `src/nil_dj/tempo.py`, `tests/test_tempo.py`

**Interfaces:**
- Produces: `estimate_bpm(x: np.ndarray, sr: int = 48000) -> float | None`.

- [ ] **Step 1: Write failing tests** on synthetic signals generated in numpy (no files):
  - `test_click_tracks[90|128|170]`: 60 s of 10 ms noise bursts every `60/bpm` s plus a little noise. The result `r` satisfies `min(abs(r-b), abs(r-2*b), abs(r-b/2)) <= 0.02*b`.
  - `test_noise_has_no_tempo`: 60 s of white noise → `None`.
  - `test_silence_has_no_tempo`: zeros → `None`.
  - `test_too_short`: 3 s of clicks → `None`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** with this algorithm:

```
if len(x) < 8 s: return None
STFT: n_fft 2048, hop 512, Hann; mag = log1p(1000·|X|)
onset[t] = Σ_f max(0, mag[f,t] − mag[f,t−1]); onset −= moving_mean(onset, 1 s); clip at 0
if onset.std() == 0: return None
ac = autocorrelation(onset − mean), normalised by ac[0]
lags for 60–200 BPM; fps = sr/512; bpm(lag) = 60·fps/lag
weight = exp(−0.5·(log2(bpm/120))²)            # log-normal prior, 1 octave σ
best = argmax(ac[lag]·weight); refine lag with a parabola through its neighbours
confidence = ac[best]; return None if confidence < 0.1 else round(bpm, 1)
```

  If the 0.1 threshold doesn't separate the noise test from the click tests, adjust only that constant, and record the value in a comment.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: onset-autocorrelation tempo estimate`

### Task 7: Tags and stream info

**Files:**
- Create: `src/nil_dj/tags.py`, `tests/test_tags.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) TrackInfo: title: str | None; artist: str | None; album_artist: str | None; album: str | None; genre: str | None; year: int | None; disc: int | None; number: int | None; duration_s: float; codec: str; sample_rate: int; bits: int | None`
  - `class TagError(Exception)`; `read_info(path: Path) -> TrackInfo` (mutagen; `TagError` when mutagen can't open the file or reports no duration).
  - `codec` mapping: FLAC → `flac`; MP4 with an `alac` codec → `alac`, otherwise `aac`; WAVE → `wav`; AIFF → `aiff`; DSF and DSDIFF → `dsd`. `bits`: from stream info for flac/alac/wav/aiff; `None` for aac and dsd. `year`: the first 4 digits of the date tag. `disc`/`number`: the integer before any `/`.

- [ ] **Step 1: Write failing tests** using `make_audio`:
  - `test_flac_tags_and_info`: a 24-bit, 96 kHz FLAC with tags `title="ネオネオン", artist="DECO*27", albumartist="DECO*27", album="アンデッドアリス", genre="J-Pop", date="2020-05-01", discnumber="1/2", tracknumber="3/12"` → all fields equal, `year == 2020`, `disc == 1`, `number == 3`, `codec == "flac"`, `bits == 24`, `sample_rate == 96000`, `duration_s ≈ seconds ± 0.05`.
  - `test_m4a_aac_and_alac`: ffmpeg `-c:a aac` → `codec == "aac"`, `bits is None`; `-c:a alac` → `codec == "alac"`, `bits == 16`.
  - `test_wav_and_aiff`: codecs `wav` and `aiff`.
  - `test_missing_tags_are_none`: a FLAC with no tags → title etc. `None`; info still read.
  - `test_garbage_raises_tag_error`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** with `mutagen.File(path)` plus type checks (`mutagen.flac.FLAC`, `mutagen.mp4.MP4`, `mutagen.wave.WAVE`, `mutagen.aiff.AIFF`, `mutagen.dsf.DSF`, `mutagen.dsdiff.DSDIFF`). Read tags through each format's native keys (Vorbis comments, MP4 atoms `©nam ©ART aART ©alb ©gen ©day disk trkn`, ID3 frames `TIT2 TPE1 TPE2 TALB TCON TDRC TPOS TRCK` for wav/aiff/dsf).
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: tag and stream info reading`

### Task 8: Index (SQLite)

**Files:**
- Create: `src/nil_dj/index.py`, `tests/test_index.py`

**Interfaces:**
- Consumes: `tags.TrackInfo`.
- Produces:
  - `class IndexFormatError(Exception)`.
  - `@dataclass(frozen=True) Row: path: str; size: int; mtime_ns: int; info: TrackInfo; bpm: float | None; lufs: float | None`.
  - `class Index`:
    - `open(path: Path, model_id: str) -> Index` (classmethod): creates the parent dir, applies the schema from spec §6 exactly, sets `PRAGMA journal_mode=WAL`, and writes `meta` (`schema_version` `"1"`, `model_id`, `dim` `"512"`) on creation. Opening an existing file whose `model_id` differs raises `IndexFormatError("index built with <old>; run nil-dj index --rebuild")`. A `schema_version` major above 1 raises `IndexFormatError`.
    - `stamps() -> dict[str, tuple[int, int]]`: path → (size, mtime_ns).
    - `upsert(path: str, size: int, mtime_ns: int, info: TrackInfo, bpm: float | None, lufs: float | None, embedding: np.ndarray) -> None`: stores the embedding as `embedding.astype("<f4").tobytes()`, deletes any `index_errors` row for the path, and commits.
    - `record_error(path: str, error: str) -> None` (commits); `errors() -> list[tuple[str, str]]` sorted by path.
    - `delete(paths: Iterable[str]) -> None` (rows and errors; commits).
    - `count() -> int`.
    - `load(where: str = "", params: Sequence = ()) -> tuple[list[Row], np.ndarray]`: rows ordered by path and an `(n, 512)` float32 matrix. `where` is a SQL fragment without `WHERE`.
    - `close()`.
    - `@staticmethod remove(path: Path) -> None`: deletes the db and its `-wal`/`-shm` files (for `--rebuild`).

- [ ] **Step 1: Write failing tests**
  - `test_schema_matches_spec`: `PRAGMA table_info(tracks)` column names equal the spec list in order; `journal_mode == "wal"`; `meta` values.
  - `test_upsert_roundtrip`: insert 3 rows; `load()` returns them by path order, and the matrix row equals the inserted vector exactly.
  - `test_upsert_replaces_and_clears_error`: `record_error(p)`, then `upsert(p)` → `errors() == []`, `count() == 1`.
  - `test_load_where`: `load("year >= ?", [2000])` filters.
  - `test_model_mismatch_raises`; `test_newer_schema_raises` (set `schema_version` to `"2"` directly).
  - `test_delete_removes_rows_and_errors`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** with `sqlite3` (stdlib).
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: SQLite index per spec schema v1`

### Task 9: Indexer

**Files:**
- Create: `src/nil_dj/indexer.py`, `tests/test_indexer.py`

**Interfaces:**
- Consumes: `Embedder`, `decode.decode_segment`, `decode.integrated_lufs`, `decode.require_ffmpeg`, `tempo.estimate_bpm`, `tags.read_info`, `Index`.
- Produces:
  - `walk(folders: list[Path]) -> list[str]`: recursive `os.scandir`, `follow_symlinks=False`, skips symlinks and unreadable dirs, keeps the 7 extensions case-insensitively, returns `str(Path(entry.path))` sorted.
  - `clip_windows(duration_s: float) -> list[tuple[float, float]]`: `(start, length)` pairs. `d ≤ 10` → `[(0, d)]`. `10 < d < 30` → `[(d/2 − 5, 10)]`. Otherwise for `c` in `(0.2, 0.5, 0.8)`: `(clamp(c·d − 5, 0, d − 10), 10)`.
  - `@dataclass Progress: done: int; total: int; path: str; error: str | None`
  - `class AlreadyRunning(Exception)`
  - `run_index(folders: list[Path], index: Index, embedder: Embedder, lock_path: Path, on_progress: Callable[[Progress], None] = lambda p: None) -> RunResult`, where `@dataclass RunResult: added: int; updated: int; removed: int; failed: int; unchanged: int`.
  - Order of work: take `FileLock(lock_path, timeout=0)` (on `Timeout` raise `AlreadyRunning`) → `require_ffmpeg()` → walk → delete rows for paths no longer on disk → for each new or changed path (size or mtime differs): `read_info`, decode the windows, `embed_audio`, BPM from `decode_segment(path, max(0, d/2 − 30), 60)`, `integrated_lufs` → `upsert`. Any `DecodeError`/`TagError`/`OSError` for a path → `record_error` and continue. `KeyboardInterrupt` propagates after the lock is released; rows already committed stay.

- [ ] **Step 1: Write failing tests** (all with `FakeEmbedder` and `make_audio`):
  - `test_clip_windows`: `5 → [(0,5)]`; `20 → [(5,10)]`; `100 → [(15,10),(45,10),(75,10)]`; `31 → [(1.2,10),(10.5,10),(19.8,10)]` (approx), every window inside `[0, d]`.
  - `test_walk_matches_nil_rules`: a tree with `a.FLAC`, `sub/b.m4a`, `sub/深い/c.dsf` (empty file is fine for walk), `notes.txt`, and a symlink to `a.FLAC` → exactly the 3 real audio paths, sorted.
  - `test_index_then_incremental`: 3 files → `added == 3`. A second run → `unchanged == 3`. Touch one (new mtime) → `updated == 1`. Delete one → `removed == 1`, `count() == 2`.
  - `test_short_and_silent_tracks`: a 2 s sine and a 40 s silence → both indexed; for the silence `lufs is None` and `bpm is None`; the 2 s sine has `bpm is None`.
  - `test_bad_files_recorded_and_cleared`: a zero-byte `x.flac` and a text file `y.flac` → `failed == 2`, `errors()` lists both, other files are indexed. Replace `y.flac` with real audio → the next run indexes it and `errors()` has only `x.flac`.
  - `test_interrupt_keeps_progress`: an embedder whose 3rd call raises `KeyboardInterrupt` → the exception propagates, `count() == 2`, and the lock file can be acquired again. The next run → `added == 3`.
  - `test_second_indexer_refused`: hold `FileLock(lock_path)` in the test → `AlreadyRunning`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** in `src/nil_dj/indexer.py`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: incremental, resumable indexer`

### Task 10: Search

**Files:**
- Create: `src/nil_dj/search.py`, `tests/test_search.py`

**Interfaces:**
- Consumes: `Index.load`, `Embedder.embed_text`, `Row`.
- Produces:
  - `@dataclass Filters: min_minutes: float | None = None; max_minutes: float | None = None; year_from: int | None = None; year_to: int | None = None; artists: list[str] | None = None; exclude_artists: list[str] | None = None; formats: list[str] | None = None; hires_only: bool = False`, and `to_sql() -> tuple[str, list]`.
    - Artist matching is case-insensitive equality on `COALESCE(album_artist, artist)` or `artist`.
    - `formats` values are codecs (`flac alac aac wav aiff dsd`); unknown values raise `ValueError`.
    - `hires_only` = `(bits > 16 OR sample_rate > 48000 OR codec = 'dsd')`.
  - `@dataclass Hit: row: Row; score: float`
  - `search_text(index: Index, embedder: Embedder, query: str, limit: int = 20, filters: Filters | None = None, template: str = "{}") -> list[Hit]`
  - `similar_to_paths(index: Index, seeds: list[str], limit: int = 20, filters: Filters | None = None) -> list[Hit]`: the mean of the seed vectors, normalised; the seeds themselves are excluded from the results. An unknown seed path raises `KeyError(path)`.
  - `similar_to_album(index: Index, album: str, artist: str | None, limit: int = 20, filters: Filters | None = None) -> list[Hit]`: case-insensitive album match (and artist when given), then the same as `similar_to_paths`. No match → `KeyError`.
  - `candidate_pool_size(needed: int) -> int = max(60, 3 * needed)` (used by Plan 3's `build_playlist`).
  - Ranking: cosine (dot product of unit vectors), descending; ties broken by path, ascending.

- [ ] **Step 1: Write failing tests** with an index filled from hand-made unit vectors (no audio) and a `FakeEmbedder` whose `embed_text` returns a chosen vector:
  - `test_rank_by_cosine_with_path_tiebreak`.
  - `test_filters_each` (one assertion per filter field, including `hires_only` with a 16/44.1 FLAC, a 24/96 FLAC and a DSD row).
  - `test_template_applied`: `template="This is a sound of {}"` → the fake receives `"This is a sound of rain"`.
  - `test_similar_excludes_seeds`; `test_similar_album_mean`; `test_unknown_seed_keyerror`.
  - `test_empty_results`: an empty index → `[]`; filters that exclude everything → `[]`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**: `index.load(*filters.to_sql())`, then `matrix @ q`, then `np.lexsort` on (path, −score).
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: brute-force cosine search with SQL filters`

### Task 11: CLI commands

**Files:**
- Modify: `src/nil_dj/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: everything above. `cli` builds a `OnnxClapEmbedder` through a module-level factory `make_embedder(cfg: Config, args) -> Embedder`, which tests monkeypatch to return `FakeEmbedder`.
- Produces the commands:
  - `nil-dj index [--rebuild] [--precision fp32|fp16|int8] [--device-id N] [--cpu] [--folder PATH ...]`. Folders default to `music_folders()`; with none configured it exits 2 with `No music folders: add one in nil, or pass --folder`.
    - Prints `Embedding on <provider> (<adapter name>), precision <p>`.
    - Shows a one-line progress `done/total, tracks/min, ETA` (rewritten with `\r` on a TTY, one line every 100 tracks otherwise).
    - Ends with `added N, updated N, removed N, failed N, unchanged N in <m:ss>`.
    - Exit codes: `AlreadyRunning` → 3 with `Another nil-dj index is running`; `FfmpegMissing` → 4 with its message; Ctrl+C → 130 with `Stopped; progress is saved. Run nil-dj index again to continue.`
  - `nil-dj search "<text>" [--limit N] [--min-minutes … --max-minutes … --year-from … --year-to … --artist A ... --exclude-artist A ... --format F ... --hires-only] [--template T]`: prints `score  artist — title  [album]  m:ss  codec`. No index file → exit 1 with `No index yet. Run: nil-dj index`. No results → `No matching tracks.`, exit 0.
  - `nil-dj similar <path> [--limit N] [filters…]`: same output.
  - `nil-dj status`: index path, track count, model and precision, error count, then the first 20 errors.

- [ ] **Step 1: Write failing tests** (monkeypatched factory, `NIL_DJ_INDEX` and `NIL_CONFIG_DIR` pointing into `tmp_path`, audio from `make_audio`):
  - `test_cli_index_then_search(capsys)`: `main(["index","--folder",d]) == 0`, its output contains `added 2`; `main(["search","rain"]) == 0` prints 2 result lines.
  - `test_cli_search_no_index` → exit 1 and the exact message; `test_cli_search_no_results` → exit 0, `No matching tracks.`
  - `test_cli_index_no_folders` → exit 2 and the message.
  - `test_cli_status` lists the count and errors.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** in `cli.py`. Store `precision` in `meta` as key `precision` on each index run, so `status` can show it.
- [ ] **Step 4: Run** `uv run pytest -v && uv run ruff check .` → PASS.
- [ ] **Step 5: Commit** `feat: index, search, similar and status commands`

### Task 12: Windows validation on the real library

No new product code unless a step finds a bug; fix bugs with a failing test first, in the task that owns the code. Results go in `docs/validation-log.md`.

**Files:**
- Create: `docs/validation-log.md`, `scripts/compare_with_nil.py` (reads nil's `%APPDATA%\bitperfect\library-cache.json` `files[].path` and the index `tracks.path` and `index_errors.path`, then prints the counts in each set difference).

- [ ] **Step 1: Ask the owner, then install the tools on Windows**: `winget install astral-sh.uv` and `winget install Gyan.FFmpeg`. Open a new PowerShell session so the PATH change is picked up. Verify that `uv --version` and `ffmpeg -version` both print versions.
- [ ] **Step 2: Sync and set up**: `scripts/sync-to-windows.sh`, then in PowerShell `cd E:\projects\nil-dj; uv sync`. Expected: the venv contains `onnxruntime-directml 1.24.4` and no `onnxruntime`.
- [ ] **Step 3: Run the fast tests on Windows**: `uv run pytest`. Expected: all pass. Paths are Windows paths here, so this exercises `walk` with backslashes.
- [ ] **Step 4: Check DirectML parity and the adapter**: `uv run pytest -m "slow" -v`. Expected: `test_onnx_matches_reference` passes for fp32, fp16 and int8 (int8 runs on CPU), and `test_list_adapters_finds_hardware` passes. Then `uv run python -c "from nil_dj.gpu import *; print(list_adapters(), pick_adapter(list_adapters()))"` should show the picked adapter is the RTX 3060.
  - Confirm that DirectML `device_id` follows DXGI order: time 50 `embed_audio` calls on device 0 and on device 1. The NVIDIA id should be clearly faster. If the order differs, map by trying each id and recording the result, then fix `pick_adapter`'s use with a test.
- [ ] **Step 5: Full index**: `uv run nil-dj index`. Record the elapsed time, tracks per minute, the failed count and the error messages in the log. Expected: about 2,765 tracks, and the GPU is named in the first line.
- [ ] **Step 6: Path agreement**: `uv run python scripts/compare_with_nil.py`. Expected: `in nil not in index` equals exactly the files listed in `index_errors`, and `in index not in nil` is 0. Any other difference is a bug in `walk` and gets fixed (Review Focus 1).
- [ ] **Step 7: Incremental check**: run `uv run nil-dj index` again. Expected: `unchanged ≈ 2765`, finishing in seconds.
- [ ] **Step 8: Search sanity**: run `nil-dj search` for `"mellow piano"`, `"energetic electronic dance"`, `"female vocals, acoustic guitar"`, `"heavy distorted guitars"`, and `nil-dj similar` on one known track. Paste the top 5 for each into the log, with the owner's one-line verdict on each.
- [ ] **Step 9: Commit** `docs: Windows validation of indexing and search on the real library`

---

## Self-review notes

- Spec coverage for Plan 1 (§14 step 1):
  - §5 `config`, `embedder`, `indexer`, `index`, `search`, `cli`: Tasks 2–11.
  - §5.1: Tasks 3, 4 and 12.
  - §6: Task 8.
  - §7 Search: Task 10. The prompt template is a parameter whose default is `"{}"` until the eval decides.
  - §10 nil-dj rows (ffmpeg missing, decode errors, no GPU, first-run download, moved or deleted files, second indexer): Tasks 4, 5, 9 and 11.
  - §11 nil-dj unit tests and the parity test: Tasks 3, 4 and 9.
  - Out of this plan by design: `order`, `nil_client`, the MCP server and `find_tracks` (Plan 3); everything in nil (Plans 2 and 4); eval (Plan 5).
- Type names used across tasks: `TrackInfo`, `Row`, `Index`, `Embedder`, `OnnxClapEmbedder`, `FakeEmbedder`, `Filters`, `Hit`, `Progress`, `RunResult`, `Adapter`, `DecodeError`, `FfmpegMissing`, `TagError`, `IndexFormatError`, `AlreadyRunning`.
