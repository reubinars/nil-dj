import subprocess
import threading
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


@pytest.fixture
def plugin_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("NIL_DJ_INDEX", raising=False)
    monkeypatch.delenv("NIL_DJ_MCP_PORT", raising=False)
    d = tmp_path / "nil" / "plugins" / "nil-dj"
    (d / "data").mkdir(parents=True)
    return d


@pytest.fixture
def fake_nil():
    from fake_nil import FakeNil

    return FakeNil([])


@pytest.fixture
def make_running_plugin(fake_nil, plugin_dir, monkeypatch):
    """Factory: start `plugin.run` against `fake_nil`, hand-shake, return `fake_nil`."""
    from fakes import FakeEmbedder

    from nil_dj import plugin

    started = []

    def make(start_mcp=False, env=None):
        for key, value in (env or {}).items():
            monkeypatch.setenv(key, value)
        reader, writer = fake_nil.streams()

        def target():
            fake_nil.exit_code = plugin.run(
                reader, writer, plugin_dir, lambda cfg: FakeEmbedder(), start_mcp
            )

        fake_nil.start()
        fake_nil.run_thread = threading.Thread(target=target, daemon=True, name="plugin-run")
        fake_nil.run_thread.start()
        started.append(fake_nil)
        fake_nil.handshake(data_dir=plugin_dir / "data")
        return fake_nil

    yield make
    try:
        thread = fake_nil.run_thread
        if thread is not None and thread.is_alive():
            try:
                fake_nil.request("shutdown", {})
            except Exception:  # noqa: BLE001, S110 - teardown must go on
                pass
            thread.join(3)
    finally:
        fake_nil.close()
    if thread is not None:
        assert not thread.is_alive(), "plugin.run did not return"


@pytest.fixture
def running_plugin(make_running_plugin):
    return make_running_plugin()
