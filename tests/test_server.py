import asyncio
import socket
import threading
import time
from contextlib import closing

import pytest
import uvicorn
from fake_nil import FakeNil
from fakes import ChosenText, add_track, unit
from fastmcp import Client
from fastmcp.exceptions import ToolError
from test_dj import TRACKS, P, connect, library

from nil_dj.config import Config, load_config
from nil_dj.dj import DJ, NOT_IN_NIL
from nil_dj.index import Final, Index
from nil_dj.search import filters_from
from nil_dj.server import McpHttp, build_server, port_free

TOOLS = {
    "search_tracks", "find_tracks", "similar_tracks", "build_playlist",
    "play", "append", "now_playing", "set_radio",
    "save_playlist", "list_playlists", "play_playlist",
    "index_status", "reindex",
}  # fmt: skip


@pytest.fixture
def cfg(tmp_path):
    return Config(index_path=tmp_path / "data" / "index.sqlite", folders=[])


@pytest.fixture
def fake():
    f = FakeNil(library([t[0] for t in TRACKS]))
    yield f
    f.close()


@pytest.fixture
def text():
    return ChosenText(unit(1, 0, 0))


@pytest.fixture
def dj(cfg, fake, text):  # as in test_dj
    index = Index.open(cfg.index_path, "fake")
    for path, artist, v in TRACKS:
        add_track(index, path, artist=artist, album=f"{artist} album", vector=v)
    index.close()
    d = DJ(cfg, connect(fake), lambda: text)
    yield d
    d.stop()


def call(target, tool: str, args: dict | None = None):
    """`tool`'s result data, through a real MCP client."""

    async def go():
        async with Client(target) as client:
            return (await client.call_tool(tool, args or {})).data

    return asyncio.run(go())


def tool_error(target, tool: str, args: dict | None = None) -> str:
    with pytest.raises(ToolError) as e:
        call(target, tool, args)
    return str(e.value)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def uvicorn_servers(monkeypatch):
    """Records the uvicorn servers started during the test; stops them and joins their
    threads afterwards, so none outlives the test."""
    servers: list[uvicorn.Server] = []
    serve = uvicorn.Server.serve

    async def recorded(self, *args, **kwargs):
        servers.append(self)
        return await serve(self, *args, **kwargs)

    monkeypatch.setattr(uvicorn.Server, "serve", recorded)
    yield servers
    for s in servers:
        s.should_exit = True
    for t in threading.enumerate():
        if t.name == "nil-dj-mcp":
            t.join(5)
            assert not t.is_alive(), "the MCP server did not stop"


def until(pred, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


def test_tools_listed(dj):
    async def go():
        async with Client(build_server(dj)) as client:
            return await client.list_tools()

    tools = asyncio.run(go())
    assert {t.name for t in tools} == TOOLS
    assert all(t.description for t in tools)


def test_build_playlist_tool(dj):
    server = build_server(dj)
    out = call(server, "build_playlist", {"prompt": "bright", "minutes": 12})
    assert len(out["tracks"]) == 3
    assert out["tracks"][0]["path"] == P[0]
    assert out["total_minutes"] == 12.0
    assert set(out["tracks"][0]) == {"path", "title", "artist", "album", "duration_s", "format",
                                     "score",
                                     "language"}  # fmt: skip
    only = call(server, "build_playlist",
                {"prompt": "bright", "minutes": 12, "filters": {"artists": ["D"]}})  # fmt: skip
    assert [t["artist"] for t in only["tracks"]] == ["D"]


def test_play_tool_reaches_nil(dj, fake):
    server = build_server(dj)
    call(server, "play", {"paths": [P[1], P[2]]})
    assert fake.queue == [P[1], P[2]]
    assert not dj.radio_on
    call(server, "play", {"paths": [P[0]], "radio": True})
    assert fake.queue[0] == P[0]
    assert dj.radio_on
    assert call(server, "now_playing")["path"] == P[0]


def test_errors_reach_client(dj, cfg, fake, text):
    alone = build_server(DJ(cfg, None, lambda: text))
    assert tool_error(alone, "play", {"paths": [P[0]]}) == NOT_IN_NIL
    server = build_server(dj)
    unknown = {"paths": ["E:\\Music\\nowhere.flac"]}  # FakeNil answers -32002
    assert tool_error(server, "play", unknown) == "nil: not in nil's library: 1 path(s)"
    with pytest.raises(ValueError) as e:
        filters_from({"mood": "sad"})
    args = {"query": "bright", "filters": {"mood": "sad"}}
    assert tool_error(server, "search_tracks", args) == str(e.value)


def test_busy_port_message(dj):
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        held.listen()
        port = held.getsockname()[1]
        assert not port_free(port)
        mcp = McpHttp(build_server(dj), port)
        mcp.start()
        assert mcp.status == f"MCP unavailable: port {port} in use"


def test_bind_failure_in_thread(dj, monkeypatch):
    server = build_server(dj)
    changed = threading.Event()

    def taken(**kwargs):
        raise SystemExit(3)  # what uvicorn does when the bind fails

    monkeypatch.setattr(server, "run", taken)
    port = free_port()
    mcp = McpHttp(server, port, on_change=changed.set)
    mcp.start()
    assert changed.wait(5)
    assert mcp.status == f"MCP unavailable: port {port} in use"


def test_http_roundtrip(dj, fake, uvicorn_servers):
    fake.playlists = {"Evening": [P[0]], "日本語": [P[1]]}
    port = free_port()
    mcp = McpHttp(build_server(dj), port)
    assert mcp.status == f"MCP: http://127.0.0.1:{port}/mcp"
    mcp.start()
    until(lambda: not port_free(port))
    assert call(f"http://127.0.0.1:{port}/mcp", "list_playlists") == ["Evening", "日本語"]
    assert len(uvicorn_servers) == 1  # the fixture stops it
    assert mcp.status == f"MCP: http://127.0.0.1:{port}/mcp"


def test_mcp_port_env(dj, plugin_dir, monkeypatch):
    monkeypatch.setenv("NIL_DJ_MCP_PORT", "50123")
    cfg = load_config(plugin_dir)
    assert McpHttp(build_server(dj), cfg.mcp_port).status == "MCP: http://127.0.0.1:50123/mcp"


def test_panel_shows_mcp_status(make_running_plugin):
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        held.listen()
        port = held.getsockname()[1]
        nil = make_running_plugin(start_mcp=True, env={"NIL_DJ_MCP_PORT": str(port)})
        unavailable = f"MCP unavailable: port {port} in use"
        nil.wait_for(lambda: nil.settings_panels)
        assert {"type": "text", "text": unavailable, "muted": True} in (
            nil.settings_panels[-1]["elements"]
        )


def test_mcp_filter_and_field(dj, cfg):
    with closing(Index.open(cfg.index_path, None)) as index:
        index.set_final({
            P[0]: Final("ja", None, None, None, "override"),
            P[1]: Final("en", "ja", None, None, "override"),
        })  # fmt: skip
    args = {"query": "x", "filters": {"languages": ["ja"]}}
    found = call(build_server(dj), "search_tracks", args)
    assert sorted(t["path"] for t in found) == sorted([P[0], P[1]])
    by_path = {t["path"]: t for t in found}
    assert by_path[P[0]]["language"] == "ja"
    assert by_path[P[1]]["language_alt"] == "ja"
    assert "language_alt" not in by_path[P[0]]
    assert "unknown language 'xx'" in tool_error(
        build_server(dj), "search_tracks", {"query": "x", "filters": {"languages": ["xx"]}}
    )
