"""CLAP's log-mel input features in numpy, so no PyTorch is needed at runtime.

A port of the non-fusion path of `transformers`' ClapFeatureExtractor and the helpers it uses in
`transformers.audio_utils` (Copyright The HuggingFace Team, Apache License 2.0).
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 48000
CLIP_SAMPLES = 480000  # 10 s

N_FFT = 1024
HOP = 480
N_MELS = 64
F_MIN = 50.0
F_MAX = 14000.0
MEL_FLOOR = 1e-10


def _hertz_to_mel(f: np.ndarray | float) -> np.ndarray:
    # Slaney scale: linear below 1 kHz, logarithmic above.
    f = np.asarray(f, dtype=np.float64)
    mels = 3.0 * f / 200.0
    logstep = 27.0 / np.log(6.4)
    log_region = f >= 1000.0
    mels = np.where(log_region, 15.0 + np.log(np.maximum(f, 1e-12) / 1000.0) * logstep, mels)
    return mels


def _mel_to_hertz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    freq = 200.0 * m / 3.0
    logstep = np.log(6.4) / 27.0
    return np.where(m >= 15.0, 1000.0 * np.exp(logstep * (m - 15.0)), freq)


def _mel_filters() -> np.ndarray:
    """(513, 64) Slaney-normalised triangular filters on the Slaney mel scale."""
    n_bins = N_FFT // 2 + 1
    mel_freqs = np.linspace(_hertz_to_mel(F_MIN), _hertz_to_mel(F_MAX), N_MELS + 2)
    filter_freqs = _mel_to_hertz(mel_freqs)
    fft_freqs = np.linspace(0, SAMPLE_RATE // 2, n_bins)

    diff = np.diff(filter_freqs)
    slopes = filter_freqs[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / diff[:-1]
    up = slopes[:, 2:] / diff[1:]
    filters = np.maximum(0.0, np.minimum(down, up))

    enorm = 2.0 / (filter_freqs[2 : N_MELS + 2] - filter_freqs[:N_MELS])
    return filters * enorm[None, :]


_MEL = _mel_filters()
# Periodic Hann window.
_WINDOW = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)


def log_mel(clip: np.ndarray) -> np.ndarray:
    """Mono 48 kHz clip of at most 10 s -> float32 (1001, 64) log-mel features."""
    x = np.asarray(clip, dtype=np.float64)
    n = x.size
    if n == 0:
        raise ValueError("empty clip")
    if n > CLIP_SAMPLES:
        raise ValueError(f"clip longer than {CLIP_SAMPLES} samples")
    if n < CLIP_SAMPLES:
        x = np.tile(x, CLIP_SAMPLES // n)
        x = np.pad(x, (0, CLIP_SAMPLES - x.size))

    x = np.pad(x, (N_FFT // 2, N_FFT // 2), mode="reflect")
    n_frames = 1 + (x.size - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = x[idx] * _WINDOW
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2  # (frames, 513)
    mel = power @ _MEL  # (frames, 64)
    return (10.0 * np.log10(np.maximum(mel, MEL_FLOOR))).astype(np.float32)
