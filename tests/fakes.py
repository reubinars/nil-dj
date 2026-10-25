"""Test doubles."""

import hashlib

import numpy as np

from nil_dj.tags import TrackInfo


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


BASE = TrackInfo(
    title="t", artist="A", album_artist=None, album="Al", genre=None, year=2000, disc=1,
    number=1, duration_s=240.0, codec="flac", sample_rate=44100, bits=16,
)  # fmt: skip


def unit(*xs) -> np.ndarray:
    v = np.zeros(512, np.float32)
    v[: len(xs)] = xs
    return v / np.linalg.norm(v)


class ChosenText:
    """Fake embedder whose text vector is chosen by the test."""

    model_id = "fake"

    def __init__(self, q: np.ndarray):
        self.q = q
        self.seen: list[str] = []

    def embed_text(self, text: str) -> np.ndarray:
        self.seen.append(text)
        return self.q

    def embed_audio(self, clips):
        raise AssertionError("not used")
