import shutil

import pytest
from fakes import FakeEmbedder
from filelock import FileLock

from nil_dj import cli
from nil_dj.cli import main


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("NIL_DJ_INDEX", str(tmp_path / "data" / "index.sqlite"))
    monkeypatch.setenv("NIL_HOME", str(tmp_path / "nil"))
    monkeypatch.setattr(cli, "make_embedder", lambda cfg, args: FakeEmbedder())
    lib = tmp_path / "Music"
    lib.mkdir()
    return lib


def out_err(capsys) -> str:
    c = capsys.readouterr()
    return c.out + c.err


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert capsys.readouterr().out.strip() == "nil-dj 0.2.0"


def test_cli_index_then_search(env, make_audio, capsys):
    make_audio("a.flac", 12, tags={"title": "Rain", "artist": "Me", "album": "Wet"}, folder=env)
    make_audio("b.flac", 12, freq=880, folder=env)
    assert main(["index", "--folder", str(env)]) == 0
    text = out_err(capsys)
    assert "Embedding on CPUExecutionProvider, precision fp32" in text
    assert "added 2, updated 0, removed 0, failed 0, unchanged 0 in " in text

    assert main(["search", "rain"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    assert any("Me — Rain  [Wet]  0:12  flac" in line for line in lines)

    assert main(["search", "rain", "--artist", "me"]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 1


def test_cli_index_uses_nil_folders(env, make_audio, capsys, tmp_path):
    (tmp_path / "nil" / "data").mkdir(parents=True)
    (tmp_path / "nil" / "data" / "library.toml").write_text(f"folders = ['{env}']\n", "utf-8")
    make_audio("a.flac", 12, folder=env)
    assert main(["index"]) == 0
    assert "added 1" in out_err(capsys)


def test_cli_search_no_index(env, capsys):
    assert main(["search", "rain"]) == 1
    assert "No index yet. Run: nil-dj index" in out_err(capsys)


def test_cli_search_no_results(env, make_audio, capsys):
    make_audio("a.flac", 12, folder=env)
    main(["index", "--folder", str(env)])
    capsys.readouterr()
    assert main(["search", "rain", "--year-from", "2100"]) == 0
    assert capsys.readouterr().out.strip() == "No matching tracks."


def test_cli_similar(env, make_audio, capsys):
    a = make_audio("a.flac", 12, folder=env)
    make_audio("b.flac", 12, freq=880, folder=env)
    main(["index", "--folder", str(env)])
    capsys.readouterr()
    assert main(["similar", str(a)]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1 and "b" in lines[0]
    assert main(["similar", str(env / "nope.flac")]) == 1
    assert "Not in the index" in out_err(capsys)


def test_cli_index_no_folders(env, capsys):
    assert main(["index"]) == 2
    assert "No music folders: add one in nil, or pass --folder" in out_err(capsys)


def test_cli_status(env, make_audio, capsys):
    make_audio("a.flac", 12, folder=env)
    (env / "bad.flac").write_bytes(b"")
    main(["index", "--folder", str(env), "--precision", "fp16"])
    capsys.readouterr()
    assert main(["status"]) == 0
    text = capsys.readouterr().out
    assert "tracks: 1" in text
    assert "precision fp16" in text
    assert "errors: 1" in text
    assert "bad.flac" in text


def test_cli_status_no_index(env, capsys):
    assert main(["status"]) == 1
    assert "No index yet" in out_err(capsys)


def test_cli_rebuild(env, make_audio, capsys):
    make_audio("a.flac", 12, folder=env)
    main(["index", "--folder", str(env)])
    capsys.readouterr()
    assert main(["index", "--folder", str(env), "--rebuild"]) == 0
    assert "added 1" in out_err(capsys)


def test_cli_rebuild_first_run(env, make_audio, capsys):
    make_audio("a.flac", 12, folder=env)
    assert main(["index", "--folder", str(env), "--rebuild"]) == 0
    assert "added 1" in out_err(capsys)


def test_cli_already_running(env, make_audio, capsys, tmp_path):
    make_audio("a.flac", 12, folder=env)
    with FileLock(str(tmp_path / "data" / "index.sqlite") + ".lock"):
        assert main(["index", "--folder", str(env)]) == 3
    assert "Another nil-dj index is running" in out_err(capsys)


def test_cli_ffmpeg_missing(env, capsys, monkeypatch):
    (env / "a.flac").touch()  # a track, so the empty-library refusal doesn't come first
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert main(["index", "--folder", str(env)]) == 4
    assert "ffmpeg not found" in out_err(capsys)


def test_cli_ffmpeg_checked_before_model_load(env, capsys, monkeypatch):
    def no_model(cfg, args):
        raise AssertionError("model loaded (and maybe downloaded) before the ffmpeg check")

    monkeypatch.setattr(cli, "make_embedder", no_model)
    (env / "a.flac").touch()  # a track, so the empty-library refusal doesn't come first
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert main(["index", "--folder", str(env)]) == 4


def test_cli_output_survives_non_utf8_stdout(env, capsys, monkeypatch):
    import io
    import sys

    (env / "雨.flac").write_bytes(b"")
    main(["index", "--folder", str(env)])
    capsys.readouterr()
    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="cp1252"))
    assert main(["status"]) == 0
    sys.stdout.flush()
    assert "errors: 1" in raw.getvalue().decode("cp1252")


def test_cli_jobs(env, make_audio, capsys, monkeypatch):
    make_audio("a.flac", 12, folder=env)
    seen = []
    real = cli.run_index

    def spy(*a, **kw):
        seen.append(kw.get("workers"))
        return real(*a, **kw)

    monkeypatch.setattr(cli, "run_index", spy)
    assert main(["index", "--folder", str(env), "--jobs", "3"]) == 0
    assert main(["index", "--folder", str(env)]) == 0
    assert seen == [3, None]  # None: the indexer's default (half the logical cores)
    assert "with 3 workers" in out_err(capsys)


def test_cli_ctrl_c(env, make_audio, capsys, monkeypatch):
    make_audio("a.flac", 12, folder=env)

    def boom(*a, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_index", boom)
    assert main(["index", "--folder", str(env)]) == 130
    assert "Stopped; progress is saved. Run nil-dj index again to continue." in out_err(capsys)


def test_cli_index_empty_folder_refuses_and_keeps_rows(env, make_audio, capsys, tmp_path):
    make_audio("a.flac", 12, folder=env)
    assert main(["index", "--folder", str(env)]) == 0
    empty = tmp_path / "Unplugged"
    empty.mkdir()
    capsys.readouterr()
    assert main(["index", "--folder", str(empty)]) == 5
    assert "no tracks found" in capsys.readouterr().err
    assert main(["search", "tone"]) == 0
    assert capsys.readouterr().out.strip()


def test_cli_rebuild_empty_folder_refuses_and_keeps_index(env, make_audio, capsys, tmp_path):
    make_audio("a.flac", 12, folder=env)
    main(["index", "--folder", str(env)])
    empty = tmp_path / "Unplugged"
    empty.mkdir()
    capsys.readouterr()
    assert main(["index", "--folder", str(empty), "--rebuild"]) == 5
    assert "no tracks found" in capsys.readouterr().err
    assert main(["search", "tone"]) == 0


def test_cli_index_refuses_an_empty_folder_before_loading_the_model(env, capsys, monkeypatch):
    def no_model(cfg, args):
        raise AssertionError("model loaded (and maybe downloaded) for an empty library")

    monkeypatch.setattr(cli, "make_embedder", no_model)
    assert main(["index", "--folder", str(env)]) == 5
    assert "no tracks found" in capsys.readouterr().err
