import json
import os
import threading
import time

import pytest

from nil_dj import host
from nil_dj.host import MAX_LINE, Connection, MessageTooLarge, NilError, encode


def _pipe():
    r, w = os.pipe()
    return (
        open(r, encoding="utf-8", newline="\n"),
        open(w, "w", encoding="utf-8", newline="\n"),
    )


class Pair:
    """Two connections wired back to back: `a` is the plugin, `b` plays nil."""

    def __init__(self, b_request=None, b_notification=None, timeout_s=10.0):
        self.a_notes = []
        self.b_notes = []
        self.a_in, self.b_out = _pipe()
        self.b_in, self.a_out = _pipe()
        self.a = Connection(
            self.a_in, self.a_out, lambda m, p: p, lambda m, p: self.a_notes.append((m, p)),
            timeout_s,
        )
        self.b = Connection(
            self.b_in,
            self.b_out,
            b_request or (lambda m, p: p),
            b_notification or (lambda m, p: self.b_notes.append((m, p))),
        )
        self.threads = [threading.Thread(target=c.serve, daemon=True) for c in (self.a, self.b)]
        for t in self.threads:
            t.start()

    def close(self):
        for f in (self.a_out, self.b_out):
            try:
                f.close()
            except OSError:
                pass
        for t in self.threads:
            t.join(2)
        for f in (self.a_in, self.b_in):
            f.close()
        assert not any(t.is_alive() for t in self.threads)


@pytest.fixture
def make_pair():
    pairs = []

    def make(**kw):
        p = Pair(**kw)
        pairs.append(p)
        return p

    yield make
    for p in pairs:
        p.close()
    workers = [t for t in threading.enumerate() if t.name == "nil-dj-notifications"]
    for t in workers:
        t.join(2)
    assert not any(t.is_alive() for t in workers)


def wait_for(cond, secs=2.0):
    end = time.time() + secs
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def test_encode_utf8_and_limit():
    assert "深い" in encode({"t": "深い"})
    assert encode({"t": "x"}).endswith("\n")
    overhead = len(encode({"t": ""}).encode()) - 1
    assert len(encode({"t": "x" * (MAX_LINE - overhead)}).encode()) - 1 == MAX_LINE
    with pytest.raises(MessageTooLarge):
        encode({"t": "x" * (MAX_LINE - overhead + 1)})


def test_messages_are_single_lines():
    line = encode({"a": "x\ny"})
    assert line.count("\n") == 1
    assert json.loads(line)["a"] == "x\ny"


def test_call_roundtrip_japanese(make_pair):
    p = make_pair()
    params = {"path": "E:\\音楽\\曲.flac"}
    assert p.a.call("echo", params) == params


def test_error_reply_raises_nil_error(make_pair):
    def refuse(method, params):
        raise NilError(-32002, "not in nil's library: 1 path(s)", {"paths": ["x"]})

    p = make_pair(b_request=refuse)
    with pytest.raises(NilError) as ei:
        p.a.call("player.play", {"paths": ["x"]})
    assert ei.value.code == -32002
    assert ei.value.message == "not in nil's library: 1 path(s)"
    assert str(ei.value) == ei.value.message
    assert ei.value.data == {"paths": ["x"]}


def test_other_exception_becomes_internal(make_pair):
    def boom(method, params):
        raise RuntimeError("kaput")

    p = make_pair(b_request=boom)
    with pytest.raises(NilError) as ei:
        p.a.call("x")
    assert ei.value.code == host.INTERNAL
    assert "kaput" in ei.value.message


def test_concurrent_calls_from_threads(make_pair):
    p = make_pair()
    bad = []

    def work(t):
        for i in range(20):
            if p.a.call("echo", {"t": t, "i": i}) != {"t": t, "i": i}:
                bad.append((t, i))

    threads = [threading.Thread(target=work, args=(t,), daemon=True) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert not any(t.is_alive() for t in threads)
    assert bad == []


def test_notification_handler_can_call_back(make_pair):
    got = []
    holder = {}

    def on_note(method, params):
        got.append(holder["p"].b.call("echo", {"n": params["n"]}))

    p = make_pair(b_notification=on_note)
    holder["p"] = p
    p.a.notify("hello", {"n": 1})
    assert wait_for(lambda: got == [{"n": 1}])


def test_handler_exception_does_not_stop_worker(make_pair, capsys):
    seen = []

    def on_note(method, params):
        if params["n"] == 1:
            raise RuntimeError("first fails")
        seen.append(params["n"])

    p = make_pair(b_notification=on_note)
    p.a.notify("n", {"n": 1})
    p.a.notify("n", {"n": 2})
    assert wait_for(lambda: seen == [2])
    assert "first fails" in capsys.readouterr().err


def test_malformed_line_ignored(make_pair, capsys):
    p = make_pair()
    p.a_out.write("not json\n")
    p.a_out.write("[1,2]\n")
    p.a_out.flush()
    assert p.a.call("echo", {"ok": True}) == {"ok": True}
    assert "ignoring" in capsys.readouterr().err


def test_overlong_line_skipped(make_pair, capsys):
    p = make_pair()
    p.a_out.write("x" * (MAX_LINE + 5000) + "\n")
    p.a_out.flush()
    assert p.a.call("echo", {"ok": 1}) == {"ok": 1}
    assert "too long" in capsys.readouterr().err


def test_timeout(make_pair, capsys):
    never = threading.Event()
    p = make_pair(b_request=lambda m, params: never.wait(5), timeout_s=0.2)
    try:
        with pytest.raises(NilError, match="nil didn't answer"):
            p.a.call("slow")
    finally:
        never.set()


def test_late_reply_is_dropped(make_pair, capsys):
    gate = threading.Event()
    p = make_pair(b_request=lambda m, params: (gate.wait(5), params)[1], timeout_s=0.1)
    with pytest.raises(NilError):
        p.a.call("slow", {"v": 1})
    gate.set()
    p.a.timeout_s = 5
    assert p.a.call("echo", {"v": 2}) == {"v": 2}
    assert "unknown id" in capsys.readouterr().err


def test_null_id_request_treated_as_notification(make_pair):
    p = make_pair()
    p.a_out.write('{"jsonrpc":"2.0","id":null,"method":"odd","params":{}}\n')
    p.a_out.flush()
    assert wait_for(lambda: p.b_notes == [("odd", {})])


def test_oversized_outgoing_call_raises(make_pair):
    p = make_pair()
    with pytest.raises(MessageTooLarge):
        p.a.call("echo", {"t": "x" * (MAX_LINE + 1)})
    assert p.a.call("echo", {"ok": 1}) == {"ok": 1}


def test_oversized_reply_becomes_internal_error(make_pair):
    p = make_pair(b_request=lambda m, params: {"t": "x" * (MAX_LINE + 1)})
    with pytest.raises(NilError) as ei:
        p.a.call("big")
    assert ei.value.code == host.INTERNAL


def test_eof_fails_pending_and_sets_closed(make_pair):
    gate = threading.Event()
    p = make_pair(b_request=lambda m, params: gate.wait(5))
    errors = []

    def caller():
        try:
            p.a.call("slow")
        except NilError as e:
            errors.append(e)

    t = threading.Thread(target=caller, daemon=True)
    t.start()
    time.sleep(0.1)
    p.b_out.close()
    t.join(3)
    gate.set()
    assert p.a.closed.wait(2)
    assert [e.message for e in errors] == ["nil has closed the connection"]
    with pytest.raises(NilError, match="nil has closed the connection"):
        p.a.call("again")


def test_invalid_utf8_line_skipped_connection_survives(make_pair, capsys):
    p = make_pair()
    p.a_out.flush()
    p.a_out.buffer.write(b"\xff\xfe bad\n")
    p.a_out.buffer.flush()
    p.a.notify("after", {"n": 1})
    assert wait_for(lambda: ("after", {"n": 1}) in p.b_notes)
    assert not p.b.closed.is_set()
    assert p.b.call("echo", {"ok": 1}) == {"ok": 1}
    assert "ignoring" in capsys.readouterr().err


def test_unserialisable_result_becomes_internal_error(make_pair):
    circular: list = []
    circular.append(circular)
    p = make_pair(b_request=lambda m, params: circular if m == "bad" else params)
    with pytest.raises(NilError) as ei:
        p.a.call("bad")
    assert ei.value.code == host.INTERNAL
    assert p.a.call("echo", {"ok": 1}) == {"ok": 1}
    assert not p.b.closed.is_set()


def test_notifications_handled_in_order(make_pair):
    p = make_pair()
    for i in range(50):
        p.a.notify("n", {"i": i})
    assert wait_for(lambda: len(p.b_notes) == 50)
    assert [params["i"] for _, params in p.b_notes] == list(range(50))


class _ChunkyWriter:
    """A writer that writes in small slices and yields between them, as a slow pipe does."""

    def __init__(self):
        self.parts = []

    def write(self, text):
        for i in range(0, len(text), 1000):
            self.parts.append(text[i : i + 1000])
            time.sleep(0)

    def flush(self):
        pass


def test_concurrent_writes_do_not_interleave():
    w = _ChunkyWriter()
    with open(os.devnull, encoding="utf-8") as r:
        _run_interleave(r, w)


def _run_interleave(r, w):
    conn = Connection(r, w, lambda m, p: p, lambda m, p: None)
    blob = "あ" * 20_000

    def work():
        for _ in range(5):
            conn.notify("big", {"blob": blob})

    threads = [threading.Thread(target=work, daemon=True) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    lines = "".join(w.parts).split("\n")
    assert lines.pop() == ""
    assert len(lines) == 20
    assert all(json.loads(line)["params"] == {"blob": blob} for line in lines)
