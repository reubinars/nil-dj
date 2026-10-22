import numpy as np
import pytest

from nil_dj.tempo import estimate_bpm

SR = 48000


def clicks(bpm: float, seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(seconds * SR)) * 0.005
    burst = int(0.01 * SR)
    period = 60 / bpm
    for t in np.arange(0, seconds - 0.01, period):
        i = int(t * SR)
        x[i : i + burst] += rng.standard_normal(burst) * 0.5
    return x.astype(np.float32)


@pytest.mark.parametrize("bpm", [90, 128, 170])
def test_click_tracks(bpm):
    r = estimate_bpm(clicks(bpm, 60))
    assert r is not None
    assert min(abs(r - bpm), abs(r - 2 * bpm), abs(r - bpm / 2)) <= 0.02 * bpm


@pytest.mark.parametrize("seed", range(10))
def test_noise_has_no_tempo(seed):
    x = (np.random.default_rng(seed).standard_normal(60 * SR) * 0.1).astype(np.float32)
    assert estimate_bpm(x) is None


def test_silence_has_no_tempo():
    assert estimate_bpm(np.zeros(60 * SR, np.float32)) is None


def test_too_short():
    assert estimate_bpm(clicks(120, 3)) is None
