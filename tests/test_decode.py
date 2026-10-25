import shutil

import numpy as np
import pytest

from nil_dj import decode
from nil_dj.decode import (
    DecodeError,
    FfmpegMissing,
    decode_segment,
    integrated_lufs,
    require_ffmpeg,
)


def test_decode_segment_length(make_audio):
    p = make_audio("a.flac", 30, rate=44100)
    x = decode_segment(p, 10, 10)
    assert x.dtype == np.float32
    assert abs(len(x) - 480000) <= 480


def test_decode_segment_near_end(make_audio):
    p = make_audio("a.flac", 12)
    x = decode_segment(p, 8, 10)
    assert abs(len(x) - 4 * 48000) <= 4800


def test_decode_error_on_garbage(tmp_path):
    p = tmp_path / "bad.flac"
    p.write_bytes(b"not audio")
    with pytest.raises(DecodeError):
        decode_segment(p, 0, 10)


def test_lufs_of_sine(make_audio):
    p = make_audio("s.flac", 10, freq=1000, amp_db=-20, channels=2)
    lufs = integrated_lufs(p)
    assert -24 < lufs < -16


def test_lufs_silence_is_none(make_audio):
    assert integrated_lufs(make_audio("q.flac", 10, kind="silence")) is None


def test_lufs_error_on_garbage(tmp_path):
    p = tmp_path / "bad.flac"
    p.write_bytes(b"not audio")
    with pytest.raises(DecodeError):
        integrated_lufs(p)


def test_ffmpeg_missing(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(FfmpegMissing, match="winget install Gyan.FFmpeg"):
        require_ffmpeg()
    assert issubclass(FfmpegMissing, decode.DecodeError)


def test_ffmpeg_never_reads_our_stdin(monkeypatch):
    # As a plugin, stdin is nil's JSON-RPC stream: a child must not inherit it.
    import subprocess

    from nil_dj import decode

    seen = {}

    def run(cmd, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(decode.subprocess, "run", run)
    decode._run(["ffmpeg", "-version"], 5)
    assert seen["stdin"] is subprocess.DEVNULL
