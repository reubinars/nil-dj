from pathlib import Path

import numpy as np
import pytest
from reference_signals import noise_10s, tone_4s

from nil_dj.features import CLIP_SAMPLES, log_mel


@pytest.fixture(scope="module")
def fx():
    return np.load(Path(__file__).parent / "fixtures" / "clap-reference.npz")


def test_log_mel_matches_reference_noise(fx):
    feat = log_mel(noise_10s())
    assert feat.shape == (1001, 64) and feat.dtype == np.float32
    assert np.abs(feat - fx["feat_noise"]).max() < 1e-4


def test_log_mel_repeat_pads_short_clip(fx):
    assert np.abs(log_mel(tone_4s()) - fx["feat_tone"]).max() < 1e-4


def test_log_mel_silence_is_finite():
    assert np.isfinite(log_mel(np.zeros(CLIP_SAMPLES, np.float32))).all()
