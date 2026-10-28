"""Whisper backend: resampler, log-mel (parity with transformers), detector, download errors."""

import json
from pathlib import Path

import numpy as np
import pytest
import requests
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

from nil_dj import whisper as wh
from nil_dj.config import Config
from nil_dj.embedder import ModelDownloadFailed
from nil_dj.languages import NAMES
from nil_dj.whisper import WhisperDetector, features, to_16k, whisper_present

FIXTURES = Path(__file__).parent / "fixtures"


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def test_to_16k_keeps_a_1khz_tone():
    t = np.arange(48000 * 2) / 48000
    y = to_16k(np.sin(2 * np.pi * 1000 * t).astype(np.float32))
    assert y.size == 32000 and y.dtype == np.float32
    assert abs(rms(y[2000:-2000]) - 1 / np.sqrt(2)) < 0.01


def test_to_16k_attenuates_above_8khz_by_60db():
    t = np.arange(48000 * 2) / 48000
    y = to_16k(np.sin(2 * np.pi * 9000 * t).astype(np.float32))
    assert 20 * np.log10(rms(y[2000:-2000]) / (1 / np.sqrt(2))) <= -60


def test_features_shape_and_short_input_is_padded():
    assert features(np.zeros(16000, np.float32)).shape == (80, 3000)
    long = features(np.ones(16000 * 40, np.float32))  # cut to 30 s
    assert long.shape == (80, 3000) and long.dtype == np.float32


def test_features_match_transformers_reference():
    ref = np.load(FIXTURES / "whisper-reference.npz")
    for name in ("noise", "tone", "voice"):
        assert np.abs(features(ref[f"in_{name}"]) - ref[f"feat_{name}"]).max() <= 1e-3


def test_whisper_present_needs_every_file(tmp_path):
    assert not whisper_present(None)
    assert not whisper_present(tmp_path)
    root = tmp_path / "whisper-small" / wh.REVISION
    for f in wh.FILES:
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_bytes(b"x")
    assert whisper_present(tmp_path)


def test_model_id_pins_the_revision():
    assert wh.MODEL_ID == f"{wh.REPO}@{wh.REVISION}"


def test_language_tokens_come_from_added_tokens_only():
    added = [{"id": 10, "content": "<|startoftranscript|>"}]
    added += [{"id": 11 + i, "content": f"<|{c}|>"} for i, c in enumerate(["en", "jw", "su"])]
    added += [{"id": 50, "content": "<|nospeech|>"}, {"id": 51, "content": "<|notimestamps|>"}]
    sot, nospeech, langs = wh._tokens(added)
    assert (sot, nospeech) == (10, 50)
    assert langs == {"en": 11, "jw": 12, "su": 13}


@pytest.mark.parametrize(
    "error",
    [LocalEntryNotFoundError("offline"), requests.ConnectionError("x"), TimeoutError()],
)
def test_download_errors_become_model_download_failed(monkeypatch, tmp_path, error):
    def boom(*a, **k):
        raise error

    monkeypatch.setattr("huggingface_hub.hf_hub_download", boom)
    with pytest.raises(ModelDownloadFailed):
        wh._fetch("tokenizer.json", tmp_path)


def test_a_full_disk_is_reported(monkeypatch, tmp_path):
    import errno

    def boom(*a, **k):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", boom)
    with pytest.raises(ModelDownloadFailed, match="No space left"):
        wh._fetch("tokenizer.json", tmp_path)


def test_optional_data_files_may_be_missing(monkeypatch, tmp_path):
    def fake(repo, f, **k):
        if f.endswith("_data"):
            raise EntryNotFoundError("none")
        return str(tmp_path / f)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake)
    assert wh._fetch("onnx/encoder_model.onnx_data", tmp_path, optional=True) is None


def test_default_detector_reads_the_config(tmp_path):
    factory = wh.default_detector(
        Config(index_path=tmp_path / "i.sqlite", folders=[], models_dir=tmp_path)
    )
    assert factory.model_id == wh.MODEL_ID and callable(factory.load)


@pytest.mark.slow
def test_whisper_detects_english_speech_and_silence(tmp_path):
    det = WhisperDetector(tmp_path)
    probs, ns = det.detect(features(np.zeros(16000 * 30, np.float32)))
    assert len(probs) == 99 and abs(sum(probs.values()) - 1) < 1e-3 and ns > 0.5
    assert set(NAMES) <= set(probs) and {"jw", "ms", "su"} <= set(probs)
    assert whisper_present(tmp_path)
    tok = json.loads(
        (tmp_path / "whisper-small" / wh.REVISION / "tokenizer.json").read_text(encoding="utf-8")
    )
    assert tok["added_tokens"]
