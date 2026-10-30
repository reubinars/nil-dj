import shutil

import pytest
from fakes import FakeDetector, FakeEmbedder, fake_detector_factory
from filelock import FileLock

from nil_dj import cli
from nil_dj.cli import main
from nil_dj.index import Final, Index


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("NIL_DJ_INDEX", str(tmp_path / "data" / "index.sqlite"))
    monkeypatch.setenv("NIL_HOME", str(tmp_path / "nil"))
    monkeypatch.setattr(cli, "make_embedder", lambda cfg, args: FakeEmbedder())
    monkeypatch.setattr(
        cli, "default_detector", lambda cfg, use_gpu=True: fake_detector_factory()
    )  # never the real model
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
    assert capsys.readouterr().out.strip() == "nil-dj 0.3.0"


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


def test_index_detects_languages(env, make_audio, capsys, monkeypatch):
    det = FakeDetector()
    monkeypatch.setattr(
        cli, "default_detector", lambda cfg, use_gpu=True: fake_detector_factory(det)
    )
    make_audio("a.flac", 12, folder=env)
    assert main(["index", "--folder", str(env)]) == 0
    assert "Languages: 1 detected, 0 failed" in capsys.readouterr().out
    assert det.calls > 0
    assert main(["languages"]) == 0
    assert "en  1" in capsys.readouterr().out


def test_index_language_load_problem_exits_zero(env, make_audio, capsys, monkeypatch):
    from nil_dj.whisper import DetectorFactory

    def boom():
        raise RuntimeError("offline")

    monkeypatch.setattr(
        cli, "default_detector", lambda cfg, use_gpu=True: DetectorFactory("m", boom)
    )
    make_audio("a.flac", 12, folder=env)
    assert main(["index", "--folder", str(env)]) == 0
    c = capsys.readouterr()
    assert "added 1" in c.out
    assert "offline" in c.err


def test_index_no_languages_flag(env, make_audio, capsys, monkeypatch):
    det = FakeDetector()
    monkeypatch.setattr(
        cli, "default_detector", lambda cfg, use_gpu=True: fake_detector_factory(det)
    )
    make_audio("a.flac", 12, folder=env)
    assert main(["index", "--folder", str(env), "--no-languages"]) == 0
    assert "Languages:" not in out_err(capsys)
    assert det.calls == 0


def _seed(env, make_audio, names):
    for n in names:
        make_audio(n, 12, folder=env)
    assert main(["index", "--folder", str(env), "--no-languages"]) == 0


def test_languages_lists_counts_and_unsure(env, make_audio, capsys):
    _seed(env, make_audio, ["a.flac", "b.flac", "c.flac"])
    idx = Index.open(cli.load_config().index_path, None)
    paths = [r[0] for r in idx._db.execute("SELECT path FROM tracks ORDER BY path")]
    idx._db.execute("UPDATE tracks SET artist = 'Me', title = 'Rain' WHERE path = ?", (paths[0],))
    idx.set_final(
        {
            paths[0]: Final("ja", None, 0.4, 1.0, "audio"),
            paths[1]: Final("ja", None, 0.9, 1.0, "audio"),
            paths[2]: Final("en", None, 0.3, 1.0, "audio"),
        }
    )
    idx._db.commit()
    idx.close()
    capsys.readouterr()
    assert main(["languages"]) == 0
    out = capsys.readouterr().out
    assert "ja  2" in out and "en  1" in out
    assert "Me — Rain  ja 0.40 audio" in out
    assert "c.flac  en 0.30 audio" in out  # no tags: the file name, not the path
    assert "b.flac" not in out


def test_languages_lists_failed_tracks(env, make_audio, capsys):
    _seed(env, make_audio, ["a.flac"])
    idx = Index.open(cli.load_config().index_path, None)
    path = idx._db.execute("SELECT path FROM tracks").fetchone()[0]
    idx.set_language_raw(path, '{"failed": "decode error"}')
    idx.close()
    capsys.readouterr()
    assert main(["languages"]) == 0
    assert "a.flac — decode error" in capsys.readouterr().out


def test_languages_resolve_reports_bad_toml(env, make_audio, capsys):
    from nil_dj.reindex import overrides_path

    _seed(env, make_audio, ["a.flac"])
    path = overrides_path(cli.load_config())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("this is [not toml", "utf-8")
    capsys.readouterr()
    assert main(["languages", "--resolve"]) == 0
    c = capsys.readouterr()
    assert c.err.strip()


def test_languages_no_index(env, capsys):
    assert main(["languages"]) == 1
    assert "No index yet" in out_err(capsys)


def _seed_detected(env, make_audio):
    _seed(env, make_audio, ["a.flac"])
    idx = Index.open(cli.load_config().index_path, None)
    path = idx._db.execute("SELECT path FROM tracks").fetchone()[0]
    idx.set_language_raw(path, '{"windows": [{"p": {"en": 0.9}, "ns": 0.1}]}')
    idx.close()


def _languages(idx_path):
    idx = Index.open(idx_path, None)
    try:
        return dict(idx.language_counts())
    finally:
        idx.close()


def test_cli_ctrl_c_resolves_what_was_detected(env, make_audio, capsys, monkeypatch):
    _seed_detected(env, make_audio)
    assert _languages(cli.load_config().index_path) == {"not detected": 1}

    def boom(*a, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_index", boom)
    assert main(["index", "--folder", str(env)]) == 130
    assert _languages(cli.load_config().index_path) == {"en": 1}


def test_cli_ctrl_c_survives_a_failing_resolve(env, make_audio, capsys, monkeypatch):
    _seed_detected(env, make_audio)

    def boom(*a, **kw):
        raise KeyboardInterrupt

    def bad(*a, **kw):
        raise RuntimeError("disk")

    monkeypatch.setattr(cli, "run_index", boom)
    monkeypatch.setattr(cli, "resolve_index", bad)
    assert main(["index", "--folder", str(env)]) == 130
    assert "Stopped; progress is saved" in out_err(capsys)


def test_cli_model_download_failure_says_run_index_again(env, make_audio, capsys, monkeypatch):
    from nil_dj.embedder import ModelDownloadFailed
    from nil_dj.indexer import LANGUAGE_MODEL_FAILED
    from nil_dj.whisper import MODEL_DOWNLOAD_FAILED, DetectorFactory

    def load():
        raise ModelDownloadFailed(MODEL_DOWNLOAD_FAILED)

    monkeypatch.setattr(
        cli, "default_detector", lambda cfg, use_gpu=True: DetectorFactory("fake", load)
    )
    make_audio("a.flac", 12, folder=env)
    assert main(["index", "--folder", str(env)]) == 0
    text = out_err(capsys)
    assert "run nil-dj index again" in text
    assert "Update index" not in text
    assert LANGUAGE_MODEL_FAILED not in text
