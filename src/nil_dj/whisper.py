"""Whisper-small language ID through ONNX Runtime: resampler, log-mel, encoder + one decoder step.

No torch or transformers at runtime. The mel and the resampler are numpy ports, checked against
transformers' WhisperFeatureExtractor by tests/fixtures/whisper-reference.npz.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime as ort

from nil_dj.config import Config
from nil_dj.embedder import CPU, DML, ModelDownloadFailed
from nil_dj.gpu import list_adapters, pick_adapter

REPO = "onnx-community/whisper-small"
REVISION = "36050c46d777d46dc4b5f43f6d90574fc38f8732"
MODEL_ID = f"{REPO}@{REVISION}"
FILES = ("tokenizer.json", "onnx/encoder_model.onnx", "onnx/decoder_model.onnx")
DATA_FILES = ("onnx/encoder_model.onnx_data", "onnx/decoder_model.onnx_data")
N_LANGUAGES = 99

MODEL_DOWNLOAD_FAILED = (
    "Couldn't download the language model (about 1 GB, needed once). "
    "Check the internet connection, then try again."
)

SR = 16000
N_FFT, HOP, N_MELS = 400, 160, 80
WINDOW_S = 30


def _hz_to_mel(f):
    f = np.asarray(f, dtype=np.float64)
    m = 3.0 * f / 200.0
    return np.where(
        f >= 1000.0, 15.0 + np.log(np.maximum(f, 1e-12) / 1000.0) * 27.0 / np.log(6.4), m
    )


def _mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    return np.where(m >= 15.0, 1000.0 * np.exp(np.log(6.4) / 27.0 * (m - 15.0)), 200.0 * m / 3.0)


def _mel_filters() -> np.ndarray:
    """Slaney-scale triangular filters, area-normalised: (201, 80), as transformers builds them."""
    n_bins = N_FFT // 2 + 1
    edges = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(8000.0), N_MELS + 2))
    bins = np.linspace(0, SR // 2, n_bins)
    slopes = edges[None, :] - bins[:, None]
    diff = np.diff(edges)
    f = np.maximum(0.0, np.minimum(-slopes[:, :-2] / diff[:-1], slopes[:, 2:] / diff[1:]))
    return f * (2.0 / (edges[2:] - edges[:N_MELS]))[None, :]


_MEL = _mel_filters().astype(np.float32)
_HANN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)

# 48 kHz -> 16 kHz: 257-tap Kaiser (beta 8) windowed-sinc low-pass at 7.6 kHz, keep every third.
_TAPS = np.arange(-128, 129)
_LOWPASS = np.sinc(_TAPS * (7600 / 24000)) * (7600 / 24000) * np.kaiser(_TAPS.size, 8.0)
_LOWPASS /= _LOWPASS.sum()


def to_16k(x48: np.ndarray) -> np.ndarray:
    """Resample 48 kHz mono to 16 kHz: float32."""
    return np.convolve(x48, _LOWPASS, mode="same")[::3].astype(np.float32)


def features(x16: np.ndarray) -> np.ndarray:
    """Whisper log-mel of the first 30 s (zero-padded if shorter): float32 (80, 3000)."""
    x = np.zeros(SR * WINDOW_S, np.float32)
    n = min(x16.size, x.size)
    x[:n] = x16[:n]
    x = np.pad(x, (N_FFT // 2, N_FFT // 2), mode="reflect")
    frames = 1 + (x.size - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(frames)[:, None]
    power = np.abs(np.fft.rfft(x[idx] * _HANN, axis=1)) ** 2
    mel = (power.astype(np.float32) @ _MEL)[:-1].T  # (80, 3000)
    log = np.log10(np.maximum(mel, 1e-10))
    log = np.maximum(log, log.max() - 8.0)
    return ((log + 4.0) / 4.0).astype(np.float32)


class LanguageDetector(Protocol):
    model_id: str

    def detect(self, feats: np.ndarray) -> tuple[dict[str, float], float]:
        """(probability per Whisper language code, no-speech probability) for one window."""
        ...


def _fetch(filename: str, models_dir: Path | None, optional: bool = False) -> str | None:
    """Local path of a model file under models_dir/whisper-small/REVISION, else the HF cache."""
    import requests
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import (
        EntryNotFoundError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
    )

    kwargs = {}
    if models_dir is not None:
        kwargs["local_dir"] = models_dir / "whisper-small" / REVISION
    # LocalEntryNotFoundError and the requests errors are OSErrors too: they go first.
    try:
        return hf_hub_download(REPO, filename, revision=REVISION, **kwargs)
    except EntryNotFoundError:
        if optional:
            return None
        raise ModelDownloadFailed(MODEL_DOWNLOAD_FAILED) from None
    except (
        LocalEntryNotFoundError, HfHubHTTPError, requests.RequestException, ConnectionError,
        TimeoutError,
    ) as e:  # fmt: skip
        raise ModelDownloadFailed(MODEL_DOWNLOAD_FAILED) from e
    except OSError as e:  # disk full, permissions
        raise ModelDownloadFailed(
            f"Couldn't save the language model: {e.strerror or e}. "
            "Free some space on the drive nil is on, then try again."
        ) from e


def _tokens(added_tokens: list[dict]) -> tuple[int, int | None, dict[str, int]]:
    """(start-of-transcript id, no-speech id, {language code: token id}) from tokenizer.json."""
    ids = {t["content"]: t["id"] for t in added_tokens}
    sot = ids["<|startoftranscript|>"]
    nospeech = ids.get("<|nospeech|>", ids.get("<|nocaptions|>"))
    langs = {}
    for content, i in ids.items():
        code = content[2:-2]
        if (
            content.startswith("<|") and content.endswith("|>") and 2 <= len(code) <= 3
            and code.isalpha() and code.islower() and sot < i < sot + 101
        ):  # fmt: skip
            langs[code] = i
    return sot, nospeech, langs


class WhisperDetector:
    def __init__(
        self, models_dir: Path | None, gpu_device_id: int | None = None, use_gpu: bool = True
    ):
        self.model_id = MODEL_ID
        self.adapter_name: str | None = None
        tokenizer_path, encoder_path, decoder_path = (_fetch(f, models_dir) for f in FILES)
        for f in DATA_FILES:  # external weights, if the export has any; must sit beside the graph
            _fetch(f, models_dir, optional=True)

        tok = json.loads(Path(tokenizer_path).read_text(encoding="utf-8"))
        self._sot, self._nospeech, langs = _tokens(tok["added_tokens"])
        assert len(langs) == N_LANGUAGES, (
            f"expected {N_LANGUAGES} language tokens, got {len(langs)}"
        )
        self._codes = list(langs)
        self._lang_ids = np.array([langs[c] for c in self._codes])

        self._enc = self._dec = None
        self.provider = CPU
        if use_gpu and DML in ort.get_available_providers():
            adapters = list_adapters()
            if gpu_device_id is None:
                picked = pick_adapter(adapters)
                gpu_device_id = picked.index if picked else 0
            self.adapter_name = next((a.name for a in adapters if a.index == gpu_device_id), None)
            try:
                options = ort.SessionOptions()
                options.enable_mem_pattern = False
                options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                providers = [(DML, {"device_id": gpu_device_id}), CPU]
                self._enc = ort.InferenceSession(encoder_path, options, providers=providers)
                self._dec = ort.InferenceSession(decoder_path, options, providers=providers)
                self.provider = DML
            except Exception as e:  # noqa: BLE001 - any session failure means "use the CPU"
                print(f"warning: DirectML unavailable ({e}); detecting on CPU", file=sys.stderr)
                self.adapter_name = None
        if self._dec is None:
            options = ort.SessionOptions()
            self._enc = ort.InferenceSession(encoder_path, options, providers=[CPU])
            self._dec = ort.InferenceSession(decoder_path, options, providers=[CPU])
        self._dec_inputs = {i.name for i in self._dec.get_inputs()}

    def detect(self, feats: np.ndarray) -> tuple[dict[str, float], float]:
        (hidden,) = self._enc.run(None, {"input_features": feats[None]})[:1]
        feeds = {
            "input_ids": np.array([[self._sot]], dtype=np.int64),
            "encoder_hidden_states": hidden,
        }
        feeds = {k: v for k, v in feeds.items() if k in self._dec_inputs}
        z = self._dec.run(["logits"], feeds)[0][0, -1].astype(np.float64)
        full = np.exp(z - z.max())
        full /= full.sum()
        nospeech = float(full[self._nospeech]) if self._nospeech is not None else 0.0
        lz = z[self._lang_ids]
        lp = np.exp(lz - lz.max())
        lp /= lp.sum()
        return {c: float(p) for c, p in zip(self._codes, lp)}, nospeech


@dataclass(frozen=True)
class DetectorFactory:
    model_id: str
    load: Callable[[], LanguageDetector]


def whisper_present(models_dir: Path | None) -> bool:
    """True when every model file is already in the models folder."""
    if models_dir is None:
        return False
    return all((models_dir / "whisper-small" / REVISION / f).exists() for f in FILES)


def default_detector(cfg: Config, use_gpu: bool = True) -> DetectorFactory:
    return DetectorFactory(
        MODEL_ID,
        lambda: WhisperDetector(cfg.models_dir, cfg.gpu_device_id, use_gpu=use_gpu),
    )
