import subprocess
from pathlib import Path

import mutagen
import pytest

# codec name -> ffmpeg encoder arguments (bits=None means the encoder's default, 16-bit).
_ENCODERS = {
    "flac": lambda bits: ["-c:a", "flac", "-sample_fmt", "s32" if bits == 24 else "s16"]
    + (["-bits_per_raw_sample", "24"] if bits == 24 else []),
    "aac": lambda bits: ["-c:a", "aac", "-b:a", "128k"],
    "alac": lambda bits: ["-c:a", "alac", "-sample_fmt", "s32p" if bits == 24 else "s16p"],
    "wav": lambda bits: ["-c:a", "pcm_s24le" if bits == 24 else "pcm_s16le"],
    "aiff": lambda bits: ["-c:a", "pcm_s24be" if bits == 24 else "pcm_s16be"],
}


def _source(kind: str, seconds: float, rate: int, bpm: float, freq: float, amp: float, ch: int):
    layout = "stereo" if ch == 2 else "mono"
    if kind == "silence":
        return f"anullsrc=r={rate}:cl={layout}:d={seconds}"
    if kind == "sine":
        expr = f"{amp}*sin(2*PI*{freq}*t)"
    elif kind == "clicks":
        expr = f"if(lt(mod(t\\,{60 / bpm})\\,0.01)\\,{amp}*(2*random(0)-1)\\,0)"
    else:
        raise ValueError(kind)
    return f"aevalsrc={'|'.join([expr] * ch)}:s={rate}:c={layout}:d={seconds}"


@pytest.fixture
def make_audio(tmp_path):
    """Render a test file with ffmpeg: make_audio("a.flac", 30) -> Path."""

    def make(
        name: str,
        seconds: float,
        kind: str = "sine",
        bpm: float = 120,
        rate: int = 44100,
        codec: str = "flac",
        tags: dict | None = None,
        *,
        bits: int | None = None,
        freq: float = 440,
        amp_db: float = -10,
        channels: int = 1,
        folder: Path | None = None,
    ) -> Path:
        path = (folder or tmp_path) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        amp = 10 ** (amp_db / 20)
        src = _source(kind, seconds, rate, bpm, freq, amp, channels)
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", src]
        cmd += _ENCODERS[codec](bits)
        if path.suffix.lower() == ".m4a":
            cmd += ["-f", "mp4"]
        subprocess.run([*cmd, str(path)], check=True, capture_output=True)
        if tags:
            f = mutagen.File(path, easy=True)
            for k, v in tags.items():
                f[k] = v
            f.save()
        return path

    return make
