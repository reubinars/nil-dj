import sys
from pathlib import Path

import pytest

from nil_dj.config import (
    DEFAULT_MCP_PORT,
    Weights,
    find_plugin_dir,
    legacy_index_path,
    load_config,
    music_folders,
)

PORT_MSG = "config.toml: mcp_port must be a whole number from 1 to 65535"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("NIL_HOME", "NIL_DJ_INDEX", "NIL_DJ_MCP_PORT"):
        monkeypatch.delenv(var, raising=False)


def write_library(home: Path, text: str) -> None:
    (home / "data").mkdir(parents=True, exist_ok=True)
    (home / "data" / "library.toml").write_text(text, "utf-8")


def test_music_folders_reads_library_toml(tmp_path):
    write_library(tmp_path, "folders = ['E:\\Music', 'D:\\More']\n")
    assert [str(p) for p in music_folders(tmp_path)] == ["E:\\Music", "D:\\More"]


def test_music_folders_missing_file(tmp_path):
    assert music_folders(tmp_path) == []


def test_music_folders_missing_key(tmp_path):
    write_library(tmp_path, "other = 1\n")
    assert music_folders(tmp_path) == []


def test_plugin_dir_from_venv_prefix(tmp_path, monkeypatch):
    plugin = tmp_path / "plugins" / "nil-dj"
    (plugin / "app" / ".venv").mkdir(parents=True)
    (plugin / "plugin.toml").write_text("", "utf-8")
    monkeypatch.setattr(sys, "prefix", str(plugin / "app" / ".venv"))
    assert find_plugin_dir() == plugin


def test_plugin_dir_needs_plugin_toml(tmp_path, monkeypatch):
    plugin = tmp_path / "plugins" / "nil-dj"
    (plugin / "app" / ".venv").mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(plugin / "app" / ".venv"))
    assert find_plugin_dir() is None
    monkeypatch.setenv("NIL_HOME", str(tmp_path))
    assert find_plugin_dir() == tmp_path / "plugins" / "nil-dj"


def test_plugin_dir_from_nil_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NIL_HOME", str(tmp_path))
    assert find_plugin_dir() == tmp_path / "plugins" / "nil-dj"


def test_plugin_dir_none_for_dev_checkout():
    assert find_plugin_dir() is None


def test_paths_in_plugin_mode(tmp_path):
    p = tmp_path / "plugins" / "nil-dj"
    p.mkdir(parents=True)
    write_library(tmp_path, "folders = ['E:\\Music']\n")
    cfg = load_config(p, data_dir=p / "data")
    assert cfg.index_path == p / "data" / "index.sqlite"
    assert cfg.models_dir == p / "models"
    assert cfg.plugin_dir == p
    assert [str(f) for f in cfg.folders] == ["E:\\Music"]
    assert cfg.problems == []


def test_index_defaults_to_plugin_data_dir(tmp_path):
    p = tmp_path / "plugins" / "nil-dj"
    assert load_config(p).index_path == p / "data" / "index.sqlite"


def test_dev_mode_has_no_plugin_paths():
    cfg = load_config()
    assert cfg.plugin_dir is None
    assert cfg.models_dir is None
    assert cfg.folders == []
    assert cfg.index_path == legacy_index_path()


def test_nil_dj_index_env_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("NIL_DJ_INDEX", "/x/i.sqlite")
    assert load_config(tmp_path, data_dir=tmp_path / "data").index_path == Path("/x/i.sqlite")


def test_config_toml_values(tmp_path):
    (tmp_path / "config.toml").write_text(
        "mcp_port = 47300\nprecision = 'fp16'\ngpu_device_id = 1\n"
        "prompt_template = 'music: {}'\n"
        "[order]\nsimilarity = 0.5\nmatch = 0.3\ntempo = 1\nloudness = 0.2\n",
        "utf-8",
    )
    cfg = load_config(tmp_path)
    assert cfg.mcp_port == 47300
    assert cfg.precision == "fp16"
    assert cfg.gpu_device_id == 1
    assert cfg.prompt_template == "music: {}"
    assert cfg.weights == Weights(similarity=0.5, match=0.3, tempo=1.0, loudness=0.2)
    assert cfg.problems == []


def test_defaults_without_config_toml(tmp_path):
    cfg = load_config(tmp_path)
    assert cfg.mcp_port == DEFAULT_MCP_PORT == 47263
    assert cfg.weights == Weights(0.60, 0.40, 0.15, 0.10)
    assert cfg.problems == []


@pytest.mark.parametrize(
    ("text", "message", "attr", "default"),
    [
        ("mcp_port = 0", PORT_MSG, "mcp_port", DEFAULT_MCP_PORT),
        ("mcp_port = 'x'", PORT_MSG, "mcp_port", DEFAULT_MCP_PORT),
        ("mcp_port = true", PORT_MSG, "mcp_port", DEFAULT_MCP_PORT),
        (
            "precision = 'fp8'",
            "config.toml: precision must be fp32, fp16 or int8",
            "precision",
            "fp32",
        ),
        (
            "[order]\ntempo = 'fast'",
            "config.toml: order.tempo must be a number",
            "weights",
            Weights(),
        ),
        ("bogus = 1", "config.toml: unknown key 'bogus'", "mcp_port", DEFAULT_MCP_PORT),
        ("[order]\nloud = 1", "config.toml: unknown key 'order.loud'", "weights", Weights()),
        (
            "gpu_device_id = 'a'",
            "config.toml: gpu_device_id must be a whole number",
            "gpu_device_id",
            None,
        ),
        (
            "prompt_template = 'no braces'",
            "config.toml: prompt_template must be text containing {}",
            "prompt_template",
            "{}",
        ),
    ],
)
def test_config_toml_problems(tmp_path, text, message, attr, default):
    (tmp_path / "config.toml").write_text(text, "utf-8")
    cfg = load_config(tmp_path)
    assert message in cfg.problems
    assert getattr(cfg, attr) == default


def test_config_toml_invalid_toml(tmp_path):
    (tmp_path / "config.toml").write_text("mcp_port = ", "utf-8")
    cfg = load_config(tmp_path)
    assert len(cfg.problems) == 1
    assert cfg.problems[0].startswith("config.toml: ")
    assert cfg.problems[0].endswith("; using defaults")
    assert cfg.mcp_port == DEFAULT_MCP_PORT


def test_one_bad_key_keeps_the_good_ones(tmp_path):
    (tmp_path / "config.toml").write_text("mcp_port = 0\nprecision = 'int8'\n", "utf-8")
    cfg = load_config(tmp_path)
    assert cfg.precision == "int8"
    assert len(cfg.problems) == 1


def test_mcp_port_env(tmp_path, monkeypatch):
    (tmp_path / "config.toml").write_text("mcp_port = 47301\n", "utf-8")
    monkeypatch.setenv("NIL_DJ_MCP_PORT", "47300")
    assert load_config(tmp_path).mcp_port == 47300
    monkeypatch.setenv("NIL_DJ_MCP_PORT", "abc")
    cfg = load_config(tmp_path)
    assert cfg.mcp_port == 47301
    assert cfg.problems == ["NIL_DJ_MCP_PORT: mcp_port must be a whole number from 1 to 65535"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX defaults")
def test_legacy_index_path_non_windows(monkeypatch):
    monkeypatch.setenv("HOME", "/h")
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    assert legacy_index_path() == Path("/h/.local/share/nil-dj/index.sqlite")
