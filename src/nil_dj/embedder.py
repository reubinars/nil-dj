"""CLAP embeddings through ONNX Runtime: DirectML on Windows when a GPU is usable, else CPU."""

from __future__ import annotations

import sys
from typing import Protocol

import numpy as np
import onnxruntime as ort

from nil_dj.features import log_mel
from nil_dj.gpu import list_adapters, pick_adapter

REPO = "Xenova/clap-htsat-unfused"
REVISION = "c28f2883575e590e04d3146ff0713c2448d691ba"
MODEL_ID = "laion/clap-htsat-unfused"
PRECISIONS = {"fp32": "", "fp16": "_fp16", "int8": "_quantized"}

DML = "DmlExecutionProvider"
CPU = "CPUExecutionProvider"


class Embedder(Protocol):
    model_id: str

    def embed_audio(self, clips: list[np.ndarray]) -> np.ndarray: ...

    def embed_text(self, text: str) -> np.ndarray: ...


def _fetch(filename: str) -> str:
    """Local path of a model file, downloading it once into the Hugging Face cache."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(REPO, filename, revision=REVISION)


def _session(path: str, providers: list, options: ort.SessionOptions) -> ort.InferenceSession:
    return ort.InferenceSession(path, sess_options=options, providers=providers)


def _normalise(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


class OnnxClapEmbedder:
    def __init__(self, precision: str = "fp32", device_id: int | None = None, use_gpu: bool = True):
        if precision not in PRECISIONS:
            raise ValueError(f"unknown precision {precision!r}; choose one of fp32, fp16, int8")
        self.model_id = MODEL_ID
        self.precision = precision
        self.adapter_name: str | None = None

        suffix = PRECISIONS[precision]
        # Xenova's quantised text model drifts to 0.91 cosine on some prompts, below the 0.97
        # parity bar, so int8 quantises only the audio tower (the indexing cost); text is fp16.
        text_suffix = PRECISIONS["fp16"] if precision == "int8" else suffix
        audio_path = _fetch(f"onnx/audio_model{suffix}.onnx")
        text_path = _fetch(f"onnx/text_model{text_suffix}.onnx")
        self._tokenizer_path = _fetch("tokenizer.json")
        self._tokenizer = None

        # DirectML's support for dynamically quantised ops is patchy, so int8 stays on CPU.
        if use_gpu and precision != "int8" and DML in ort.get_available_providers():
            adapters = list_adapters()
            if device_id is None:
                picked = pick_adapter(adapters)
                device_id = picked.index if picked else 0
            self.adapter_name = next((a.name for a in adapters if a.index == device_id), None)
            try:
                options = ort.SessionOptions()
                options.enable_mem_pattern = False
                options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                providers = [(DML, {"device_id": device_id}), CPU]
                self._audio = _session(audio_path, providers, options)
                self._text = _session(text_path, providers, options)
                self.provider = DML
                return
            except Exception as e:  # noqa: BLE001 - any session failure means "use the CPU"
                print(f"warning: DirectML unavailable ({e}); embedding on CPU", file=sys.stderr)
                self.adapter_name = None

        options = ort.SessionOptions()
        self._audio = _session(audio_path, [CPU], options)
        self._text = _session(text_path, [CPU], options)
        self.provider = CPU

    def embed_audio(self, clips: list[np.ndarray]) -> np.ndarray:
        """Mean of the clips' unit vectors, normalised again: float32 (512,)."""
        feats = np.stack([log_mel(c) for c in clips])[:, None, :, :].astype(np.float32)
        (embeds,) = self._audio.run(["audio_embeds"], {"input_features": feats})
        return _normalise(_normalise(embeds).mean(axis=0)).astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        if self._tokenizer is None:
            from tokenizers import Tokenizer

            self._tokenizer = Tokenizer.from_file(self._tokenizer_path)
            self._tokenizer.enable_truncation(512)
        ids = np.array([self._tokenizer.encode(text).ids], dtype=np.int64)
        (embeds,) = self._text.run(["text_embeds"], {"input_ids": ids})
        return _normalise(embeds[0]).astype(np.float32)
