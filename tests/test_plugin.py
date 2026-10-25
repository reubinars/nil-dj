import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time

import pytest

from nil_dj import __version__


def test_handshake_and_first_panel(running_plugin):
    assert running_plugin.init_result == {"name": "nil-dj", "version": __version__}
    running_plugin.wait_for(lambda: running_plugin.panels)
    assert running_plugin.panels[0] == {"elements": [{"type": "heading", "text": "nil-dj"}]}


def test_shutdown_returns_zero(running_plugin):
    assert running_plugin.request("shutdown", {}) == {"result": {}}
    running_plugin.run_thread.join(3)
    assert not running_plugin.run_thread.is_alive()
    assert running_plugin.exit_code == 0


def test_stdin_close_returns(running_plugin):
    running_plugin.close()
    running_plugin.run_thread.join(3)
    assert not running_plugin.run_thread.is_alive()
    assert running_plugin.exit_code == 0


def test_stop_callbacks_run_after_the_reply_and_a_hanging_one_is_cut_off(running_plugin):
    from nil_dj import plugin

    ran = threading.Event()
    plugin.on_stop(lambda: threading.Event().wait(30))  # hangs
    plugin.on_stop(ran.set)
    assert running_plugin.request("shutdown", {}) == {"result": {}}
    running_plugin.run_thread.join(4)
    assert not running_plugin.run_thread.is_alive()
    assert ran.is_set()


def test_guard_stdout(tmp_path):
    code = (
        "import os, sys\n"
        "from nil_dj.plugin import guard_stdout\n"
        "proto = guard_stdout()\n"
        "print('stray')\n"
        "os.write(1, b'raw\\n')\n"
        "proto.write('{}\\n')\n"
        "proto.flush()\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=tmp_path,
    )
    assert done.stdout == "{}\n"
    assert "stray" in done.stderr
    assert "raw" in done.stderr


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Proc:
    """`nil-dj plugin` as a real child process, driven by hand."""

    def __init__(self, plugin_dir):
        env = {k: v for k, v in os.environ.items() if k != "NIL_DJ_INDEX"}
        env["NIL_DJ_MCP_PORT"] = str(_free_port())
        self.errfile = open(plugin_dir / "stderr.txt", "wb")  # noqa: SIM115
        self.p = subprocess.Popen(
            [sys.executable, "-m", "nil_dj", "plugin"],
            cwd=plugin_dir,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.errfile,
        )
        self.lines: queue.Queue = queue.Queue()
        self.seen: list[dict] = []
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for raw in self.p.stdout:
            self.lines.put(raw.decode("utf-8"))
        self.lines.put(None)

    def send(self, msg: dict) -> None:
        self.p.stdin.write((json.dumps({"jsonrpc": "2.0", **msg}) + "\n").encode("utf-8"))
        self.p.stdin.flush()

    def read_until(self, pred, timeout=20.0) -> dict:
        while True:
            line = self.lines.get(timeout=timeout)
            assert line is not None, "plugin closed stdout"
            msg = json.loads(line)
            assert msg["jsonrpc"] == "2.0", line
            self.seen.append(msg)
            if pred(msg):
                return msg

    def drain(self) -> list[str]:
        """Every remaining stdout line up to EOF."""
        out = []
        while (line := self.lines.get(timeout=5)) is not None:
            out.append(line)
        return out

    def stderr_text(self) -> str:
        self.errfile.flush()
        return self.errfile.name and open(self.errfile.name, encoding="utf-8").read()

    def kill(self) -> None:
        if self.p.poll() is None:
            self.p.kill()
        self.p.wait()
        for f in (self.p.stdin, self.p.stdout, self.errfile):
            try:
                f.close()
            except OSError:
                pass


@pytest.fixture
def proc(plugin_dir):
    p = Proc(plugin_dir)
    try:
        yield p
    finally:
        p.kill()


def _handshake(proc, plugin_dir) -> dict:
    proc.send(
        {
            "id": 1,
            "method": "initialize",
            "params": {
                "api_version": 1,
                "nil_version": "0.2.0",
                "granted": ["playback", "library", "playlists"],
                "data_dir": str(plugin_dir / "data"),
            },
        }
    )
    reply = proc.read_until(lambda m: m.get("id") == 1 and "method" not in m)
    assert reply["result"] == {"name": "nil-dj", "version": __version__}
    proc.send({"method": "initialized", "params": {}})
    panel = proc.read_until(lambda m: m.get("method") == "panel.set")
    proc.send({"id": panel["id"], "result": {}})
    return panel


def test_subprocess_end_to_end(proc, plugin_dir):
    panel = _handshake(proc, plugin_dir)
    assert panel["params"]["panel"]["elements"][0]["text"] == "nil-dj"
    proc.send({"id": 2, "method": "shutdown", "params": {}})
    assert proc.read_until(lambda m: m.get("id") == 2)["result"] == {}
    assert proc.p.wait(3) == 0, proc.stderr_text()
    for line in proc.drain():  # nothing but JSON-RPC until EOF
        assert json.loads(line)["jsonrpc"] == "2.0", proc.stderr_text()


def test_subprocess_exits_on_stdin_close(proc, plugin_dir):
    _handshake(proc, plugin_dir)
    proc.p.stdin.close()
    assert proc.p.wait(3) == 0, proc.stderr_text()


class _SlowWriter:
    """Forwards to `writer`, recording each line only once it has really been written."""

    def __init__(self, writer):
        self._writer = writer
        self.written = []

    def write(self, text):
        time.sleep(0.2)
        self.written.append(text)
        return self._writer.write(text)

    def flush(self):
        self._writer.flush()


def test_stop_callbacks_run_only_after_the_shutdown_reply_is_written(fake_nil, plugin_dir):
    from fakes import FakeEmbedder

    from nil_dj import plugin

    reader, raw_writer = fake_nil.streams()
    writer = _SlowWriter(raw_writer)
    seen = []
    done = threading.Event()

    def callback():
        seen.append(any('"result":{}' in line and '"id":2' in line for line in writer.written))
        done.set()

    thread = threading.Thread(
        target=lambda: plugin.run(reader, writer, plugin_dir, lambda cfg: FakeEmbedder(), False),
        daemon=True,
    )
    fake_nil.start()
    thread.start()
    try:
        fake_nil.handshake(data_dir=plugin_dir / "data")  # initialize is id 1, shutdown id 2
        plugin.on_stop(callback)
        assert fake_nil.request("shutdown", {}) == {"result": {}}
        assert done.wait(3)
        assert seen == [True]
    finally:
        thread.join(3)
        fake_nil.close()
