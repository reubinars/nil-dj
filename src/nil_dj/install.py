"""`nil-dj install`: put the plugin into a nil home as `<nil_home>/plugins/nil-dj/`."""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from nil_dj import __version__

DEFAULT_CONFIG_TOML = """\
# nil-dj settings. Remove the leading # from a line to change it; restart the plugin after.

# Port for the MCP server (1-65535).
# mcp_port = 47263

# Model precision: "fp32", "fp16" or "int8".
# precision = "fp32"

# DirectML adapter index; unset lets the system pick.
# gpu_device_id = 0

# Text wrapped around your description before it is embedded. Must contain {}.
# prompt_template = "{}"

# How "More like this" and the radio order tracks.
# [order]
# similarity = 0.6
# match = 0.4
# tempo = 0.15
# loudness = 0.1
"""


class InstallError(Exception):
    pass


def run_checked(cmd: list[str], cwd: Path) -> None:
    try:
        subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or "").strip()[-800:]
        raise InstallError(f"{' '.join(cmd)} failed:\n{tail}") from e


def repo_root() -> Path:
    root = Path(__file__).parents[2]
    try:
        text = (root / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        text = ""
    if 'name = "nil-dj"' not in text:
        raise InstallError(
            "Run nil-dj install from a nil-dj checkout (uv run nil-dj install --nil …)."
        )
    return root


def plugin_toml(version: str, windows: bool) -> str:
    py = "app\\\\.venv\\\\Scripts\\\\python.exe" if windows else "app/.venv/bin/python"
    return (
        'name = "nil-dj"\n'
        f'version = "{version}"\n'
        'description = "AI DJ: playlists from a description, more like this, radio"\n'
        "api_version = 1\n"
        f'command = ["{py}", "-m", "nil_dj", "plugin"]\n'
        'permissions = ["playback", "library", "playlists"]\n'
    )


def _venv_python(app: Path, windows: bool) -> Path:
    if windows:
        return app / ".venv" / "Scripts" / "python.exe"
    return app / ".venv" / "bin" / "python"


def _move_legacy_index(legacy: Path, dest: Path) -> bool:
    """Move the legacy index to `dest`; False (legacy untouched) if it is in use or won't move."""
    try:
        con = sqlite3.connect(legacy)
        try:
            busy = con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]
        finally:
            con.close()
        if busy:
            return False
        try:
            shutil.move(legacy, dest)
        except OSError:
            dest.unlink(missing_ok=True)  # a half-copied file must not pass for the index
            raise
        for suffix in ("-wal", "-shm", ".lock"):
            Path(f"{legacy}{suffix}").unlink(missing_ok=True)
    except (sqlite3.Error, OSError):
        return False
    return True


def install(
    nil_home: Path,
    repo_root: Path,
    run: Callable[[list[str], Path], None] = run_checked,
    legacy_index: Path | None = None,
    windows: bool = sys.platform == "win32",
) -> Path:
    """Install into `nil_home`; returns the plugin folder."""
    nil_home = nil_home.resolve()  # uv runs in repo_root, so relative paths would land there
    if windows:
        if not (nil_home / "nil.exe").exists():
            raise InstallError(f"{nil_home} has no nil.exe")
    elif not nil_home.exists():
        raise InstallError(f"{nil_home} has no nil.exe")
    if shutil.which("uv") is None:
        raise InstallError("uv is needed: winget install astral-sh.uv")

    p = nil_home / "plugins" / "nil-dj"
    (p / "data").mkdir(parents=True, exist_ok=True)

    app = p / "app"
    if app.exists():
        old = p / "app.old"
        if old.exists():
            try:
                shutil.rmtree(old)
            except OSError as e:
                raise InstallError(
                    f"Couldn't remove {old}: delete it by hand, then run install again."
                ) from e
        try:
            app.rename(old)
        except OSError as e:
            raise InstallError(
                "nil-dj is running in nil: switch it off in Settings → Plugins, "
                "then run install again."
            ) from e
        try:
            shutil.rmtree(old)
        except OSError:
            print(f"Couldn't remove {old}; delete it by hand.")

    py = _venv_python(app, windows)
    reqs = app / "requirements.txt"
    run(
        ["uv", "export", "--frozen", "--no-dev", "--no-emit-project", "--no-hashes",
         "-o", str(reqs)],
        repo_root,
    )  # fmt: skip
    run(["uv", "venv", "--python", "3.12", str(app / ".venv")], repo_root)
    run(["uv", "pip", "install", "--python", str(py), "-r", str(reqs)], repo_root)
    run(["uv", "pip", "install", "--python", str(py), "--no-deps", str(repo_root)], repo_root)

    (p / "plugin.toml").write_text(plugin_toml(__version__, windows), encoding="utf-8")
    config = p / "config.toml"
    if not config.exists():
        config.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")

    if legacy_index is not None and legacy_index.exists():
        dest = p / "data" / "index.sqlite"
        if dest.exists():
            print(f"Left the old index at {legacy_index}; {dest} already exists.")
        else:
            if _move_legacy_index(legacy_index, dest):
                print(f"Moved the index from {legacy_index} to {dest}.")
            else:
                print(
                    f"The old index is in use, so it was left at {legacy_index}: "
                    "stop nil-dj, then run install again to move it."
                )

    print(
        f"Installed nil-dj {__version__} in {p}. "
        "In nil, open Settings → Plugins and switch nil-dj on."
    )
    if shutil.which("ffmpeg") is None:
        print("ffmpeg isn't on PATH; indexing needs it: winget install Gyan.FFmpeg")
    return p
