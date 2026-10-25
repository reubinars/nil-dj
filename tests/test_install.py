import sqlite3
import subprocess
import tomllib
from pathlib import Path

import pytest

from nil_dj import __version__
from nil_dj import install as installer
from nil_dj.cli import main
from nil_dj.config import DEFAULT_MCP_PORT, Weights, load_config
from nil_dj.install import DEFAULT_CONFIG_TOML, InstallError, install, plugin_toml


class Runner:
    """Records commands; `uv venv` creates the python the later steps expect."""

    def __init__(self):
        self.calls: list[tuple[list[str], Path]] = []

    def __call__(self, cmd, cwd):
        self.calls.append((cmd, cwd))
        if cmd[:2] == ["uv", "venv"]:
            venv = Path(cmd[-1])
            for sub in ("bin/python", "Scripts/python.exe"):
                (venv / sub).parent.mkdir(parents=True, exist_ok=True)
                (venv / sub).touch()


@pytest.fixture
def home(tmp_path):
    h = tmp_path / "nil"
    h.mkdir()
    return h


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    return r


@pytest.mark.parametrize("windows", [True, False])
def test_fresh_install(home, repo, windows):
    if windows:
        (home / "nil.exe").touch()
    run = Runner()
    p = install(home, repo, run=run, windows=windows)

    assert p == home / "plugins" / "nil-dj"
    app = p / "app"
    py = app / ".venv" / ("Scripts/python.exe" if windows else "bin/python")
    assert [c for c, _ in run.calls] == [
        ["uv", "export", "--frozen", "--no-dev", "--no-emit-project", "--no-hashes",
         "-o", str(app / "requirements.txt")],
        ["uv", "venv", "--python", "3.12", str(app / ".venv")],
        ["uv", "pip", "install", "--python", str(py), "-r", str(app / "requirements.txt")],
        ["uv", "pip", "install", "--python", str(py), "--no-deps", str(repo)],
    ]  # fmt: skip
    assert all(cwd == repo for _, cwd in run.calls)

    manifest = tomllib.loads((p / "plugin.toml").read_text(encoding="utf-8"))
    command = (
        ["app\\.venv\\Scripts\\python.exe", "-m", "nil_dj", "plugin"]
        if windows
        else ["app/.venv/bin/python", "-m", "nil_dj", "plugin"]
    )
    assert manifest == {
        "name": "nil-dj",
        "version": __version__,
        "description": "AI DJ: playlists from a description, more like this, radio",
        "api_version": 1,
        "command": command,
        "permissions": ["playback", "library", "playlists"],
    }
    assert plugin_toml(__version__, windows) == (p / "plugin.toml").read_text(encoding="utf-8")
    assert (p / "config.toml").read_text(encoding="utf-8") == DEFAULT_CONFIG_TOML
    assert (p / "data").is_dir()


def test_default_config_lists_every_key_and_matches_defaults(tmp_path):
    assert tomllib.loads(DEFAULT_CONFIG_TOML) == {}
    uncommented = "\n".join(
        ln[2:] if ln.startswith("# ") and ("=" in ln or ln == "# [order]") else ln
        for ln in DEFAULT_CONFIG_TOML.splitlines()
        if not ln.startswith("# ") or "=" in ln or ln == "# [order]"
    )
    parsed = tomllib.loads(uncommented)
    assert set(parsed) == {"mcp_port", "precision", "gpu_device_id", "prompt_template", "order"}
    plugin = tmp_path / "plugins" / "nil-dj"
    plugin.mkdir(parents=True)
    (plugin / "config.toml").write_text(uncommented, encoding="utf-8")
    cfg = load_config(plugin_dir=plugin)
    assert cfg.problems == []
    assert cfg.mcp_port == DEFAULT_MCP_PORT
    assert cfg.precision == "fp32"
    assert cfg.prompt_template == "{}"
    assert cfg.weights == Weights()


def test_reinstall_keeps_data_models_config(home, repo):
    p = home / "plugins" / "nil-dj"
    (p / "data").mkdir(parents=True)
    (p / "data" / "index.sqlite").write_bytes(b"index")
    (p / "models").mkdir()
    (p / "models" / "x").write_bytes(b"model")
    (p / "config.toml").write_text("precision = 'fp16'\n", encoding="utf-8")
    (p / "app").mkdir()
    (p / "app" / "old.txt").write_text("old", encoding="utf-8")

    install(home, repo, run=Runner(), windows=False)

    assert (p / "data" / "index.sqlite").read_bytes() == b"index"
    assert (p / "models" / "x").read_bytes() == b"model"
    assert (p / "config.toml").read_text(encoding="utf-8") == "precision = 'fp16'\n"
    assert not (p / "app" / "old.txt").exists()
    assert not (p / "app.old").exists()


def test_app_in_use(home, repo, monkeypatch):
    p = home / "plugins" / "nil-dj"
    (p / "app").mkdir(parents=True)
    (p / "app" / "keep.txt").write_text("keep", encoding="utf-8")
    real_rename = Path.rename

    def rename(self, target):
        if self.name == "app":
            raise PermissionError("in use")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", rename)
    run = Runner()
    with pytest.raises(InstallError) as e:
        install(home, repo, run=run, windows=False)
    assert str(e.value) == (
        "nil-dj is running in nil: switch it off in Settings → Plugins, "
        "then run install again."
    )
    assert (p / "app" / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert run.calls == []


def _legacy(tmp_path):
    legacy = tmp_path / "legacy" / "index.sqlite"
    legacy.parent.mkdir()
    con = sqlite3.connect(legacy)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE t (v TEXT)")
    con.execute("INSERT INTO t VALUES ('hello')")
    con.commit()
    return con, legacy


def test_moves_legacy_index(home, repo, tmp_path):
    con, legacy = _legacy(tmp_path)  # connection stays open so the -wal file exists
    for suffix in ("-shm", ".lock"):
        Path(f"{legacy}{suffix}").touch()
    try:
        p = install(home, repo, run=Runner(), legacy_index=legacy, windows=False)
    finally:
        con.close()

    moved = sqlite3.connect(p / "data" / "index.sqlite")
    try:
        assert moved.execute("SELECT v FROM t").fetchall() == [("hello",)]
    finally:
        moved.close()
    assert list(legacy.parent.iterdir()) == []


def test_legacy_kept_when_data_exists(home, repo, tmp_path, capsys):
    con, legacy = _legacy(tmp_path)
    con.close()
    dest = home / "plugins" / "nil-dj" / "data" / "index.sqlite"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"mine")

    install(home, repo, run=Runner(), legacy_index=legacy, windows=False)

    assert legacy.exists()
    assert dest.read_bytes() == b"mine"
    assert "Left the old index" in capsys.readouterr().out


def test_no_nil_exe(tmp_path, repo):
    home = tmp_path / "empty"
    home.mkdir()
    with pytest.raises(InstallError) as e:
        install(home, repo, run=Runner(), windows=True)
    assert str(e.value) == f"{home} has no nil.exe"


def test_no_uv(home, repo, monkeypatch):
    monkeypatch.setattr(installer.shutil, "which", lambda name: None)
    with pytest.raises(InstallError) as e:
        install(home, repo, run=Runner(), windows=False)
    assert str(e.value) == "uv is needed: winget install astral-sh.uv"


def test_next_steps_and_ffmpeg_hint(home, repo, monkeypatch, capsys):
    real_which = installer.shutil.which
    monkeypatch.setattr(
        installer.shutil, "which", lambda n: None if n == "ffmpeg" else real_which(n)
    )
    p = install(home, repo, run=Runner(), windows=False)
    out = capsys.readouterr().out
    assert f"Installed nil-dj {__version__} in {p}. In nil, open Settings → Plugins" in out
    assert "ffmpeg isn't on PATH; indexing needs it: winget install Gyan.FFmpeg" in out


def test_run_checked_reports_stderr(tmp_path):
    cmd = ["python3", "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"]
    with pytest.raises(InstallError) as e:
        installer.run_checked(cmd, tmp_path)
    assert "boom" in str(e.value)
    assert "python3" in str(e.value)


def test_repo_root_is_this_checkout():
    assert (installer.repo_root() / "pyproject.toml").is_file()


def test_cli_install(home, tmp_path, monkeypatch):
    run = Runner()
    monkeypatch.setattr(installer, "run_checked", run)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path / "homedir"))
    assert main(["install", "--nil", str(home)]) == 0
    assert (home / "plugins" / "nil-dj" / "plugin.toml").is_file()
    assert len(run.calls) == 4


def test_cli_install_reports_error(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert main(["install", "--nil", str(tmp_path / "missing")]) == 1
    assert "has no nil.exe" in capsys.readouterr().err


def test_relative_nil_home_is_resolved(tmp_path, repo, monkeypatch):
    (tmp_path / "work" / "nilrel").mkdir(parents=True)
    monkeypatch.chdir(tmp_path / "work")
    run = Runner()
    p = install(Path("nilrel"), repo, run=run, windows=False)
    assert p == (tmp_path / "work" / "nilrel").resolve() / "plugins" / "nil-dj"
    paths = [a for cmd, _ in run.calls for a in cmd if "/" in a]
    assert paths
    for a in paths:
        assert Path(a).is_absolute()
        assert Path(a).is_relative_to(p) or Path(a) == repo


class _FakeCon:
    """Wraps a sqlite3 connection; the checkpoint reports busy or raises."""

    def __init__(self, real, mode):
        self.real, self.mode = real, mode

    def execute(self, sql, *a):
        if "wal_checkpoint" in sql:
            if self.mode == "busy":
                return type("R", (), {"fetchone": lambda s: (1, -1, -1)})()
            raise sqlite3.OperationalError("database is locked")
        return self.real.execute(sql, *a)

    def close(self):
        self.real.close()


@pytest.mark.parametrize("mode", ["busy", "error"])
def test_legacy_left_when_checkpoint_fails(home, repo, tmp_path, monkeypatch, capsys, mode):
    con, legacy = _legacy(tmp_path)
    wal = Path(f"{legacy}-wal")
    assert wal.exists()
    real_connect = sqlite3.connect
    monkeypatch.setattr(
        installer.sqlite3, "connect", lambda *a, **k: _FakeCon(real_connect(*a, **k), mode)
    )
    try:
        p = install(home, repo, run=Runner(), legacy_index=legacy, windows=False)
        wal_kept = wal.exists()  # closing our own connection would delete it
    finally:
        con.close()
    assert p == home.resolve() / "plugins" / "nil-dj"
    assert legacy.exists()
    assert wal_kept
    assert not (p / "data" / "index.sqlite").exists()
    assert (
        f"The old index is in use, so it was left at {legacy}: "
        "stop nil-dj, then run install again to move it."
    ) in capsys.readouterr().out


def test_legacy_left_when_move_fails(home, repo, tmp_path, monkeypatch, capsys):
    con, legacy = _legacy(tmp_path)
    con.close()

    def boom(src, dst):
        raise OSError("denied")

    monkeypatch.setattr(installer.shutil, "move", boom)
    p = install(home, repo, run=Runner(), legacy_index=legacy, windows=False)
    assert legacy.exists()
    assert not (p / "data" / "index.sqlite").exists()
    assert "The old index is in use" in capsys.readouterr().out


def test_stale_app_old_cannot_be_removed(home, repo, monkeypatch):
    p = home / "plugins" / "nil-dj"
    (p / "app").mkdir(parents=True)
    (p / "app.old").mkdir()

    def rmtree(path, *a, **k):
        raise OSError("busy")

    monkeypatch.setattr(installer.shutil, "rmtree", rmtree)
    with pytest.raises(InstallError) as e:
        install(home, repo, run=Runner(), windows=False)
    old = home.resolve() / "plugins" / "nil-dj" / "app.old"
    assert str(e.value) == f"Couldn't remove {old}: delete it by hand, then run install again."
    assert (p / "app").is_dir()


def test_app_old_cleanup_failure_does_not_fail(home, repo, monkeypatch, capsys):
    p = home / "plugins" / "nil-dj"
    (p / "app").mkdir(parents=True)

    def rmtree(path, *a, **k):
        raise OSError("busy")

    monkeypatch.setattr(installer.shutil, "rmtree", rmtree)
    got = install(home, repo, run=Runner(), windows=False)
    assert got == home.resolve() / "plugins" / "nil-dj"
    assert "app.old" in capsys.readouterr().out


@pytest.mark.slow
def test_real_install(home):
    p = install(home, installer.repo_root(), windows=False, legacy_index=None)
    py = p / "app" / ".venv" / "bin" / "python"
    out = subprocess.run(
        [str(py), "-m", "nil_dj", "--version"], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == f"nil-dj {__version__}"
