"""Tests for the benchmark against an in-process stub /generate server."""
import json
import socket
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from servekit.bench import CORRECTNESS_PROMPTS, BenchConfig, run_benchmark, wait_for_ready


def _make_handler(unready_posts: int):
    """Stub /generate that 503s for its first `unready_posts` requests -- a
    server that's listening but can't serve yet. GET returns the request count.
    """
    state = {"posts": 0}
    lock = threading.Lock()

    class _Handler(BaseHTTPRequestHandler):
        def _reply(self, code, payload):
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            with lock:
                self._reply(200, {"posts": state["posts"]})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n))
            with lock:
                state["posts"] += 1
                seen = state["posts"]
            if seen <= unready_posts:
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self._reply(200, {"text": "OUT:" + body["text"][:12], "meta_info": {"completion_tokens": 7}})

        def log_message(self, *a):  # silence access logs
            pass

    return _Handler


def _start_server(unready_posts: int = 0):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(unready_posts))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _request_count(url: str) -> int:
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read())["posts"]


def _dead_url():
    """A URL nothing is listening on."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}"


def test_correctness_and_throughput():
    srv, url = _start_server()
    try:
        cfg = BenchConfig(requests=12, input_len=8, output_len=5, concurrency=4, seed=1)
        rep = run_benchmark(url, cfg)
        assert not rep.errors, rep.errors
        assert rep.correctness and len(rep.correctness["results"]) == len(CORRECTNESS_PROMPTS)
        assert all(r["output"] for r in rep.correctness["results"])
        t = rep.throughput
        assert t["completed"] == 12 and t["errors"] == 0
        assert t["output_tokens"] == 12 * 7
        assert t["output_tok_per_s"] > 0
        assert t["latency_s"]["p50"] >= 0
    finally:
        srv.shutdown()


def test_correctness_captures_all_prompts():
    srv, url = _start_server()
    try:
        rep = run_benchmark(url, BenchConfig(requests=1, correctness=True))
        outs = [r["output"] for r in rep.correctness["results"]]
        assert len(outs) == len(CORRECTNESS_PROMPTS) and all(outs)
    finally:
        srv.shutdown()


def test_wait_for_ready_polls_through_503():
    srv, url = _start_server(unready_posts=3)
    try:
        waited = wait_for_ready(url, timeout_s=10.0, interval_s=0.05)
        assert waited > 0
        assert wait_for_ready(url, timeout_s=10.0, interval_s=0.05) >= 0
    finally:
        srv.shutdown()


def test_wait_for_ready_times_out():
    try:
        wait_for_ready(_dead_url(), timeout_s=0.3, interval_s=0.05)
    except TimeoutError as e:
        assert "not serving" in str(e)
    else:
        raise AssertionError("expected TimeoutError against a dead port")


def test_run_benchmark_waits_then_benchmarks():
    srv, url = _start_server(unready_posts=2)
    try:
        cfg = BenchConfig(requests=4, input_len=8, output_len=5, concurrency=2, wait_ready_s=10.0)
        rep = run_benchmark(url, cfg)
        assert not rep.errors, rep.errors
        assert rep.ready_wait_s is not None and rep.ready_wait_s >= 0
        assert rep.throughput["completed"] == 4
    finally:
        srv.shutdown()


def test_run_benchmark_gives_up_when_never_ready():
    cfg = BenchConfig(requests=4, wait_ready_s=0.3)
    rep = run_benchmark(_dead_url(), cfg)
    assert len(rep.errors) == 1 and "not serving" in rep.errors[0]
    assert rep.throughput is None and rep.correctness is None
