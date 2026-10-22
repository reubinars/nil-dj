"""Test doubles."""

import hashlib

import numpy as np


def _unit(seed_bytes: bytes) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "little")
    v = np.random.default_rng(seed).standard_normal(512)
    return (v / np.linalg.norm(v)).astype(np.float32)


class FakeEmbedder:
    """Deterministic vectors seeded from the input; no model."""

    model_id = "fake"
    provider = "CPUExecutionProvider"
    adapter_name = None

    def embed_audio(self, clips: list[np.ndarray]) -> np.ndarray:
        return _unit(b"".join(np.asarray(c, np.float32).tobytes() for c in clips))

    def embed_text(self, text: str) -> np.ndarray:
        return _unit(text.encode("utf-8"))
