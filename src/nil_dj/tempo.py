"""Tempo from the autocorrelation of a spectral-flux onset envelope."""

from __future__ import annotations

import numpy as np

N_FFT = 2048
HOP = 512
MIN_SECONDS = 8
MIN_BPM, MAX_BPM = 60, 200
# Peak autocorrelation below this means no stable beat. Measured: click tracks 0.77-0.99, white
# noise 0.07-0.11 over 10 seeds (the plan's starting value 0.1 let one noise seed through).
MIN_CONFIDENCE = 0.2


def _onset_envelope(x: np.ndarray, sr: int) -> np.ndarray:
    window = np.hanning(N_FFT + 1)[:-1].astype(np.float32)
    n_frames = 1 + (x.size - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    mag = np.log1p(1000.0 * np.abs(np.fft.rfft(x[idx] * window, axis=1)))
    onset = np.maximum(0.0, np.diff(mag, axis=0)).sum(axis=1)
    width = max(1, round(sr / HOP))  # 1 s
    onset = onset - np.convolve(onset, np.ones(width) / width, mode="same")
    return np.maximum(onset, 0.0)


def estimate_bpm(x: np.ndarray, sr: int = 48000) -> float | None:
    x = np.asarray(x, dtype=np.float32)
    if x.size < MIN_SECONDS * sr:
        return None
    onset = _onset_envelope(x, sr)
    if onset.std() == 0:
        return None

    o = onset - onset.mean()
    n = o.size
    spec = np.fft.rfft(o, 2 * n)
    ac = np.fft.irfft(spec * np.conj(spec))[:n]
    if ac[0] <= 0:
        return None
    ac = ac / ac[0]

    fps = sr / HOP
    lags = np.arange(int(np.ceil(60 * fps / MAX_BPM)), int(60 * fps / MIN_BPM) + 1)
    bpms = 60 * fps / lags
    weight = np.exp(-0.5 * np.log2(bpms / 120) ** 2)  # log-normal prior around 120, σ 1 octave
    best = lags[np.argmax(ac[lags] * weight)]
    confidence = ac[best]
    if confidence < MIN_CONFIDENCE:
        return None

    a, b, c = ac[best - 1], ac[best], ac[best + 1]
    denom = a - 2 * b + c
    lag = best + (0.5 * (a - c) / denom if denom != 0 else 0.0)
    return round(float(60 * fps / lag), 1)
