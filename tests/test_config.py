import sys
from pathlib import Path

import pytest

from nil_dj.config import default_index_path, music_folders, nil_config_dir


def test_music_folders_reads_nil_library_toml(tmp_path):
    (tmp_path / "library.toml").write_text("folders = ['E:\\Music', 'D:\\More']\n", "utf-8")
    assert [str(p) for p in music_folders(tmp_path)] == ["E:\\Music", "D:\\More"]


def test_music_folders_missing_file(tmp_path):
    assert music_folders(tmp_path) == []


def test_music_folders_missing_key(tmp_path):
    (tmp_path / "library.toml").write_text("other = 1\n", "utf-8")
    assert music_folders(tmp_path) == []


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("NIL_DJ_INDEX", "/x/i.sqlite")
    monkeypatch.setenv("NIL_CONFIG_DIR", "/y")
    assert default_index_path() == Path("/x/i.sqlite")
    assert nil_config_dir() == Path("/y")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX defaults")
def test_defaults_non_windows(monkeypatch):
    monkeypatch.setenv("HOME", "/h")
    for var in ("XDG_DATA_HOME", "NIL_DJ_INDEX", "NIL_CONFIG_DIR"):
        monkeypatch.delenv(var, raising=False)
    assert default_index_path() == Path("/h/.local/share/nil-dj/index.sqlite")
    assert nil_config_dir() == Path("/h/.config/bitperfect")
