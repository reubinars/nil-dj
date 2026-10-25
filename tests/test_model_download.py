"""A failed model download reads as one plain sentence everywhere: never Hugging Face's text."""

import errno

import pytest
import requests
from fake_nil import FakeNil
from fakes import add_track, unit
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from test_dj import connect, library, until
from test_server import tool_error

from nil_dj import embedder as emb
from nil_dj.config import Config
from nil_dj.dj import DJ
from nil_dj.embedder import MODEL_DOWNLOAD_FAILED, ModelDownloadFailed, OnnxClapEmbedder
from nil_dj.index import Index
from nil_dj.panel import Panel, build
from nil_dj.reindex import INDEX_NEEDS_MODEL
from nil_dj.server import build_server

HUB = (
    "An error happened while trying to locate the file on the Hub and we cannot find the "
    "requested files in the local cache. Please check your connection and try again or make "
    "sure your Internet connection is on."
)
PATHS = ["E:\\Music\\A\\01.flac", "E:\\Music\\B\\01.flac"]


def raising(error):
    def download(*args, **kwargs):
        raise error

    return download


@pytest.mark.parametrize(
    "error",
    [
        LocalEntryNotFoundError(HUB),
        HfHubHTTPError("503 Server Error: Service Unavailable for url: https://huggingface.co"),
        requests.ConnectionError("Max retries exceeded with url: https://huggingface.co"),
        requests.Timeout("Read timed out. (read timeout=10)"),
    ],
)
def test_fetch_turns_download_errors_into_one_sentence(monkeypatch, tmp_path, error):
    monkeypatch.setattr("huggingface_hub.hf_hub_download", raising(error))
    with pytest.raises(ModelDownloadFailed) as e:
        emb._fetch("tokenizer.json", tmp_path)
    assert str(e.value) == MODEL_DOWNLOAD_FAILED
    assert isinstance(e.value, ValueError)
    assert e.value.__cause__ is error  # the detail stays in the traceback for nil's log


def test_fetch_reports_a_full_disk(monkeypatch, tmp_path):
    full = OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", raising(full))
    with pytest.raises(ModelDownloadFailed) as e:
        emb._fetch("tokenizer.json", tmp_path)
    assert str(e.value) == (
        "Couldn't save the sound model: No space left on device. "
        "Free some space on the drive nil is on, then try again."
    )


@pytest.fixture
def offline(monkeypatch):
    """Every download fails the way it does with the Wi-Fi down; counts the attempts."""
    attempts = []

    def download(*args, **kwargs):
        attempts.append(args)
        raise LocalEntryNotFoundError(HUB)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    return attempts


@pytest.fixture
def cfg(tmp_path):
    return Config(
        index_path=tmp_path / "data" / "index.sqlite", folders=[], models_dir=tmp_path / "models"
    )


@pytest.fixture
def fake():
    f = FakeNil(library(PATHS))
    yield f
    f.close()


def seed(cfg):
    index = Index.open(cfg.index_path, emb.MODEL_ID)
    for n, p in enumerate(PATHS):
        add_track(index, p, artist="A", album="Al", vector=unit(1, n))
    index.close()


def offline_dj(cfg, fake):
    return DJ(cfg, connect(fake), lambda: OnnxClapEmbedder(models_dir=cfg.models_dir))


def texts(panel):
    return [e.get("text") or e.get("label") or "" for e in panel["elements"]]


def test_the_panel_says_it_once_and_never_mentions_the_hub(cfg, fake, offline):
    seed(cfg)
    dj = offline_dj(cfg, fake)
    sent = []
    panel = Panel(dj, lambda m, p: sent.append(p["panel"]), lambda: "MCP: not running", 0)
    try:
        panel.on_event({"id": "prompt", "value": "rain"})
        until(lambda: sent and MODEL_DOWNLOAD_FAILED in texts(sent[-1]))
        shown = texts(sent[-1])
        assert shown.count(MODEL_DOWNLOAD_FAILED) == 1
        assert not [t for t in shown if "Hub" in t or "Sound model" in t]
        assert dj.model_state == MODEL_DOWNLOAD_FAILED

        tries = len(offline)
        with pytest.raises(ModelDownloadFailed):
            dj.embedder()  # nothing cached: the next use downloads again
        assert len(offline) > tries
    finally:
        panel.close()
        dj.stop()


def test_the_model_line_alone_is_shown_verbatim(cfg):
    from nil_dj.dj import IndexStatus
    from nil_dj.panel import View

    els = build(View(IndexStatus(indexed=3), MODEL_DOWNLOAD_FAILED))["elements"]
    assert {"type": "text", "text": MODEL_DOWNLOAD_FAILED} in els


@pytest.mark.parametrize(("has_index", "label"), [(False, "Index library"), (True, "Update index")])
def test_indexing_says_it_needs_the_model(cfg, fake, offline, has_index, label):
    if has_index:
        seed(cfg)
    dj = offline_dj(cfg, fake)
    try:
        assert dj.reindex()
        until(lambda: not dj.index_status().running)
        problem = dj.index_status().problem
        assert problem == INDEX_NEEDS_MODEL.format(label)
        assert problem == (
            "Indexing needs the sound model, which couldn't be downloaded. "
            f"Check the internet connection, then press {label} again."
        )
        assert "Hub" not in problem and "Indexing failed" not in problem
        assert not dj.index_status().rebuild_needed
    finally:
        dj.stop()


def test_mcp_tools_get_the_same_sentence(cfg, fake, offline):
    seed(cfg)
    dj = offline_dj(cfg, fake)
    try:
        error = tool_error(build_server(dj), "search_tracks", {"query": "rain"})
        assert error == MODEL_DOWNLOAD_FAILED
        assert "Hub" not in error
    finally:
        dj.stop()
