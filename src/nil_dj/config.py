"""Where things live: the plugin folder, nil's music folders, the index, and runtime options."""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MCP_PORT = 47263
PRECISIONS = ("fp32", "fp16", "int8")


@dataclass(frozen=True)
class Weights:
    similarity: float = 0.60
    match: float = 0.40
    tempo: float = 0.15
    loudness: float = 0.10


@dataclass
class Config:
    index_path: Path
    folders: list[Path]
    plugin_dir: Path | None = None
    models_dir: Path | None = None
    precision: str = "fp32"
    gpu_device_id: int | None = None
    prompt_template: str = "{}"
    mcp_port: int = DEFAULT_MCP_PORT
    weights: Weights = field(default_factory=Weights)
    problems: list[str] = field(default_factory=list)


def legacy_index_path() -> Path:
    """The Plan 1 index location, outside any plugin folder (used by install)."""
    if sys.platform == "win32":
        base = Path(os.environ["LOCALAPPDATA"])
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        base = Path(xdg) if xdg else Path(os.environ["HOME"]) / ".local" / "share"
    return base / "nil-dj" / "index.sqlite"


def find_plugin_dir() -> Path | None:
    """The folder nil installed us into, or None for a dev checkout.

    Inside nil the interpreter is `<p>/app/.venv` with `<p>/plugin.toml`; otherwise
    `NIL_HOME` points at nil's folder and the plugin lives in `<NIL_HOME>/plugins/nil-dj`.
    """
    venv = Path(sys.prefix)
    if venv.name == ".venv" and venv.parent.name == "app":
        root = venv.parent.parent
        if (root / "plugin.toml").is_file():
            return root
    if home := os.environ.get("NIL_HOME"):
        return Path(home) / "plugins" / "nil-dj"
    return None


def _read_folders(nil_home: Path) -> tuple[list[Path], str | None]:
    """(folders, problem). Never raises; a problem means no folders."""
    path = nil_home / "data" / "library.toml"
    try:
        with path.open("rb") as f:
            data = tomllib.loads(f.read().decode("utf-8-sig"))
    except FileNotFoundError:
        return [], None
    except (tomllib.TOMLDecodeError, OSError, UnicodeDecodeError) as err:
        return [], f"library.toml: {err}; no music folders"
    folders = data.get("folders", [])
    if not isinstance(folders, list):
        return [], "library.toml: folders must be a list; no music folders"
    # nil writes literal strings like 'E:\Music'; keep them exactly as written.
    return [Path(s) for s in folders if isinstance(s, str)], None


def music_folders(nil_home: Path) -> list[Path]:
    """The folders nil scans, from `<nil_home>/data/library.toml`. Never raises; problems -> []."""
    return _read_folders(nil_home)[0]


_PORT_MSG = "mcp_port must be a whole number from 1 to 65535"


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _valid_port(value: object) -> bool:
    return _is_int(value) and 1 <= value <= 65535  # type: ignore[operator]


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _apply_config_toml(path: Path, cfg: Config) -> None:
    try:
        with path.open("rb") as f:
            data = tomllib.loads(f.read().decode("utf-8-sig"))
    except FileNotFoundError:
        return
    except (tomllib.TOMLDecodeError, OSError, UnicodeDecodeError) as err:
        cfg.problems.append(f"config.toml: {err}; using defaults")
        return
    bad = cfg.problems.append
    known = {"mcp_port", "precision", "gpu_device_id", "prompt_template", "order"}
    for key in data:
        if key not in known:
            bad(f"config.toml: unknown key '{key}'")
    if "mcp_port" in data:
        if _valid_port(data["mcp_port"]):
            cfg.mcp_port = data["mcp_port"]
        else:
            bad(f"config.toml: {_PORT_MSG}")
    if "precision" in data:
        if data["precision"] in PRECISIONS:
            cfg.precision = data["precision"]
        else:
            bad("config.toml: precision must be fp32, fp16 or int8")
    if "gpu_device_id" in data:
        if _is_int(data["gpu_device_id"]):
            cfg.gpu_device_id = data["gpu_device_id"]
        else:
            bad("config.toml: gpu_device_id must be a whole number")
    if "prompt_template" in data:
        tpl = data["prompt_template"]
        if isinstance(tpl, str) and "{}" in tpl:
            cfg.prompt_template = tpl
        else:
            bad("config.toml: prompt_template must be text containing {}")
    if "order" in data:
        order = data["order"]
        if not isinstance(order, dict):
            bad("config.toml: order must be a table")
            return
        values: dict[str, float] = {}
        for key, value in order.items():
            if key not in Weights.__dataclass_fields__:
                bad(f"config.toml: unknown key 'order.{key}'")
            elif _is_number(value):
                values[key] = float(value)
            else:
                bad(f"config.toml: order.{key} must be a number")
        cfg.weights = Weights(**values)


def load_config(plugin_dir: Path | None = None, data_dir: Path | None = None) -> Config:
    """Work out paths and options. Never raises on a bad `config.toml`.

    - plugin_dir: the argument, else `find_plugin_dir()`.
    - index_path: env NIL_DJ_INDEX, else `data_dir/index.sqlite`, else
      `plugin_dir/data/index.sqlite`, else `legacy_index_path()`.
    - models_dir: `plugin_dir/models`; None without a plugin folder (the Hugging Face cache).
    - folders: `music_folders(plugin_dir.parent.parent)`; [] without a plugin folder.
    - `plugin_dir/config.toml` keys: mcp_port, precision, gpu_device_id, prompt_template and an
      `[order]` table of similarity/match/tempo/loudness weights. Env NIL_DJ_MCP_PORT overrides
      mcp_port.
    - Every problem adds one line to `problems` and keeps the default for that key.
    """
    plugin_dir = plugin_dir or find_plugin_dir()
    if override := os.environ.get("NIL_DJ_INDEX"):
        index_path = Path(override)
    elif data_dir:
        index_path = data_dir / "index.sqlite"
    elif plugin_dir:
        index_path = plugin_dir / "data" / "index.sqlite"
    else:
        index_path = legacy_index_path()
    folders, folders_problem = _read_folders(plugin_dir.parent.parent) if plugin_dir else ([], None)
    cfg = Config(
        index_path=index_path,
        folders=folders,
        plugin_dir=plugin_dir,
        models_dir=plugin_dir / "models" if plugin_dir else None,
    )
    if folders_problem:
        cfg.problems.append(folders_problem)
    if plugin_dir:
        _apply_config_toml(plugin_dir / "config.toml", cfg)
    if (raw := os.environ.get("NIL_DJ_MCP_PORT")) is not None:
        try:
            port = int(raw)
        except ValueError:
            port = 0
        if _valid_port(port):
            cfg.mcp_port = port
        else:
            cfg.problems.append(f"NIL_DJ_MCP_PORT: {_PORT_MSG}")
    return cfg
