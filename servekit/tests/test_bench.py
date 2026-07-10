"""Tests for the post-ready benchmark against an in-process stub /generate server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from servekit.bench import CORRECTNESS_PROMPTS, BenchConfig, run_benchmark


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        # Deterministic echo so greedy correctness output/hash is stable per prompt.
        resp = {"text": "OUT:" + body["text"][:12], "meta_info": {"completion_tokens": 7}}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):  # silence access logs
        pass


def _start_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


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
        assert t["output_tokens"] == 12 * 7          # meta_info.completion_tokens honored
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


if __name__ == "__main__":
    test_correctness_and_throughput()
    test_correctness_captures_all_prompts()
    print("ok")
