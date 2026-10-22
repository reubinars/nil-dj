"""Where things live: nil's music folders, the index file, and runtime options."""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path


def nil_config_dir() -> Path:
    """nil's config dir: %APPDATA%\\bitperfect on Windows, ~/.config/bitperfect elsewhere."""
    if override := os.environ.get("NIL_CONFIG_DIR"):
        return Path(override)
    if sys.platform == "win32":
        return Path(os.environ["APPDATA"]) / "bitperfect"
    return Path(os.environ["HOME"]) / ".config" / "bitperfect"


def default_index_path() -> Path:
    if override := os.environ.get("NIL_DJ_INDEX"):
        return Path(override)
    if sys.platform == "win32":
        base = Path(os.environ["LOCALAPPDATA"])
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        base = Path(xdg) if xdg else Path(os.environ["HOME"]) / ".local" / "share"
    return base / "nil-dj" / "index.sqlite"


def music_folders(config_dir: Path | None = None) -> list[Path]:
    """The folders nil scans, from its `library.toml`. Missing file or key -> []."""
    path = (config_dir or nil_config_dir()) / "library.toml"
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        return []
    # nil writes literal strings like 'E:\Music'; keep them exactly as written.
    return [Path(s) for s in data.get("folders", [])]


@dataclass
class Config:
    index_path: Path
    folders: list[Path]
    precision: str = "fp32"
    gpu_device_id: int | None = None
    prompt_template: str = "{}"


def load_config() -> Config:
    return Config(index_path=default_index_path(), folders=music_folders())
