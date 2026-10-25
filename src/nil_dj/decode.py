"""Audio decoding and loudness through ffmpeg in a subprocess (DSF/DFF included)."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 48000
SEGMENT_TIMEOUT_S = 120
LUFS_TIMEOUT_S = 600
# ebur128 reports -70 LUFS (its gate) for silence.
LUFS_FLOOR = -70.0

_LUFS = re.compile(r"I:\s+(-?\d+(?:\.\d+)?) LUFS")


class DecodeError(Exception):
    pass


class FfmpegMissing(DecodeError):
    pass


def require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise FfmpegMissing(
            "ffmpeg not found on PATH. Install it: winget install Gyan.FFmpeg (Windows) "
            "or apt install ffmpeg"
        )
    return path


def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    try:
        # As a plugin our stdin is nil's JSON-RPC stream: ffmpeg must not inherit it.
        return subprocess.run(
            cmd, check=False, capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL
        )
    except subprocess.TimeoutExpired as e:
        raise DecodeError(f"ffmpeg timed out after {timeout} s") from e


def _tail(stderr: bytes) -> str:
    text = stderr.decode("utf-8", "replace").strip()
    return text[-500:] or "ffmpeg failed"


def decode_segment(path: Path, start_s: float, length_s: float) -> np.ndarray:
    """`length_s` seconds from `start_s` as float32 mono 48 kHz (shorter near the end)."""
    cmd = [
        require_ffmpeg(), "-nostdin", "-v", "error",
        "-ss", f"{start_s:.3f}", "-t", f"{length_s:.3f}", "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-",
    ]  # fmt: skip
    proc = _run(cmd, SEGMENT_TIMEOUT_S)
    if proc.returncode != 0:
        raise DecodeError(_tail(proc.stderr))
    x = np.frombuffer(proc.stdout, "<f4").astype(np.float32)
    if x.size == 0:
        raise DecodeError(_tail(proc.stderr) if proc.stderr else "no audio decoded")
    return x


def integrated_lufs(path: Path) -> float | None:
    """Integrated loudness of the whole file (streamed); None for silence."""
    cmd = [
        require_ffmpeg(), "-nostdin", "-hide_banner", "-nostats", "-i", str(path),
        "-vn", "-af", "ebur128", "-f", "null", "-",
    ]  # fmt: skip
    proc = _run(cmd, LUFS_TIMEOUT_S)
    if proc.returncode != 0:
        raise DecodeError(_tail(proc.stderr))
    matches = _LUFS.findall(proc.stderr.decode("utf-8", "replace"))
    if not matches:
        raise DecodeError("ebur128 reported no loudness")
    lufs = float(matches[-1])
    return None if lufs <= LUFS_FLOOR else lufs
