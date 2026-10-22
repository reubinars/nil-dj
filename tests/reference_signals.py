"""Deterministic signals and prompts shared by the reference generator and the tests."""

import numpy as np

SR = 48000

PROMPTS = [
    "This is a sound of rain",
    "mellow jazz piano trio",
    "aggressive electronic dance music",
    "雨の夜の静かなピアノ",
]


def noise_10s() -> np.ndarray:
    return (np.random.default_rng(0).standard_normal(480000) * 0.1).astype(np.float32)


def tone_4s() -> np.ndarray:
    t = np.arange(4 * SR) / SR
    x = 0.3 * np.sin(2 * np.pi * 440 * t) + np.random.default_rng(1).standard_normal(t.size) * 0.01
    return x.astype(np.float32)
