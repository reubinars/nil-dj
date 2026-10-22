from pathlib import Path

import numpy as np
import onnxruntime
import pytest
from fakes import FakeEmbedder
from reference_signals import PROMPTS, noise_10s, tone_4s

from nil_dj import embedder as emb
from nil_dj.embedder import OnnxClapEmbedder


class _FakeSession:
    def __init__(self, path, providers):
        self.providers = providers


@pytest.fixture
def no_model(monkeypatch):
    """Session factory and downloads patched out: nothing is fetched or loaded."""
    calls = []

    def session(path, providers, options):
        calls.append(providers)
        return _FakeSession(path, providers)

    monkeypatch.setattr(emb, "_fetch", lambda filename: f"/nowhere/{filename}")
    monkeypatch.setattr(emb, "_session", session)
    return calls


def test_cpu_fallback_when_no_dml(monkeypatch, no_model):
    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: ["CPUExecutionProvider"])
    e = OnnxClapEmbedder()
    assert e.provider == "CPUExecutionProvider"
    assert e.adapter_name is None


def test_dml_used_with_picked_adapter(monkeypatch, no_model):
    from nil_dj.gpu import Adapter

    monkeypatch.setattr(
        onnxruntime,
        "get_available_providers",
        lambda: ["DmlExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr(
        emb,
        "list_adapters",
        lambda: [Adapter(0, "iGPU", 1 << 29, False), Adapter(1, "RTX", 4 << 30, False)],
    )
    e = OnnxClapEmbedder()
    assert e.provider == "DmlExecutionProvider"
    assert e.adapter_name == "RTX"
    assert no_model[0][0] == ("DmlExecutionProvider", {"device_id": 1})


def test_session_failure_retries_on_cpu(monkeypatch, capsys):
    monkeypatch.setattr(
        onnxruntime,
        "get_available_providers",
        lambda: ["DmlExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr(emb, "list_adapters", list)
    monkeypatch.setattr(emb, "_fetch", lambda filename: f"/nowhere/{filename}")

    def session(path, providers, options):
        if providers[0] != "CPUExecutionProvider":
            raise RuntimeError("DML device lost")
        return _FakeSession(path, providers)

    monkeypatch.setattr(emb, "_session", session)
    e = OnnxClapEmbedder()
    assert e.provider == "CPUExecutionProvider"
    assert "DML device lost" in capsys.readouterr().err


def test_int8_stays_on_cpu(monkeypatch, no_model):
    monkeypatch.setattr(
        onnxruntime,
        "get_available_providers",
        lambda: ["DmlExecutionProvider", "CPUExecutionProvider"],
    )
    assert OnnxClapEmbedder("int8").provider == "CPUExecutionProvider"


@pytest.mark.parametrize(
    ("precision", "audio", "text"),
    [
        ("fp32", "onnx/audio_model.onnx", "onnx/text_model.onnx"),
        ("fp16", "onnx/audio_model_fp16.onnx", "onnx/text_model_fp16.onnx"),
        # The quantised text model scores 0.91 cosine on a Japanese prompt (spec bar: 0.97), so
        # int8 quantises only the audio tower; text runs once per query in fp16.
        ("int8", "onnx/audio_model_quantized.onnx", "onnx/text_model_fp16.onnx"),
    ],
)
def test_model_files_per_precision(monkeypatch, precision, audio, text):
    fetched = []
    monkeypatch.setattr(emb, "_fetch", lambda f: fetched.append(f) or f"/nowhere/{f}")
    monkeypatch.setattr(emb, "_session", lambda path, providers, options: _FakeSession(path, []))
    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: ["CPUExecutionProvider"])
    OnnxClapEmbedder(precision)
    assert fetched == [audio, text, "tokenizer.json"]


def test_unknown_precision_rejected(no_model):
    with pytest.raises(ValueError, match="fp32, fp16, int8"):
        OnnxClapEmbedder("bf16")


def test_fake_embedder_is_deterministic_and_normalised():
    f = FakeEmbedder()
    a = f.embed_audio([noise_10s()])
    assert a.shape == (512,) and a.dtype == np.float32
    assert abs(np.linalg.norm(a) - 1) < 1e-6
    assert np.array_equal(a, f.embed_audio([noise_10s()]))
    assert not np.array_equal(a, f.embed_audio([tone_4s()]))
    assert np.array_equal(f.embed_text("rain"), f.embed_text("rain"))


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.mark.slow
@pytest.mark.parametrize(("precision", "threshold"), [("fp32", 0.999), ("fp16", 0.99), ("int8", 0.97)])
def test_onnx_matches_reference(precision, threshold):
    fx = np.load(Path(__file__).parent / "fixtures" / "clap-reference.npz")
    e = OnnxClapEmbedder(precision)
    assert _cos(e.embed_audio([noise_10s()]), fx["audio_noise"]) >= threshold
    assert _cos(e.embed_audio([tone_4s()]), fx["audio_tone"]) >= threshold
    for i, p in enumerate(PROMPTS):
        assert _cos(e.embed_text(p), fx["text"][i]) >= threshold, p
