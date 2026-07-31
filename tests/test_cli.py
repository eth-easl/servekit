"""Tests for the CLI split: `profile` and `bench` are joined by the report file."""
import json
import tempfile
import threading
import time
from pathlib import Path

from servekit.bench import wait_for_ready
from servekit.cli import main

from test_bench import _dead_url, _request_count, _start_server

# What `servekit profile` leaves on disk, minus the benchmark key that
# `servekit bench --into` is supposed to fill in.
PROFILE_REPORT = {
    "command": "python -m sglang.launch_server",
    "started_at": 1.0,
    "ready_at": 60.0,
    "success": True,
    "total_duration_s": 59.0,
    "phases": [{"name": "weight_loading", "duration_s": 30.0, "source": "engine_reported"}],
    "benchmark": None,
}

BENCH_ARGS = ["--requests", "4", "--concurrency", "2", "--input-len", "8", "--output-len", "5"]


def test_bench_merges_into_profile_report():
    srv, url = _start_server()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "run.json"
        path.write_text(json.dumps(PROFILE_REPORT))
        try:
            rc = main(["bench", "--url", url, "--into", str(path)] + BENCH_ARGS)
        finally:
            srv.shutdown()
        assert rc == 0
        merged = json.loads(path.read_text())
        for key in ("command", "started_at", "ready_at", "success", "phases"):
            assert merged[key] == PROFILE_REPORT[key]
        assert merged["benchmark"]["throughput"]["completed"] == 4
        assert merged["benchmark"]["base_url"] == url


def test_bench_writes_standalone_report():
    """The restore case: no profile exists, so the bench stands on its own."""
    srv, url = _start_server()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "bench.json"
        try:
            rc = main(["bench", "--url", url, "--out", str(path), "--wait-ready", "10"] + BENCH_ARGS)
        finally:
            srv.shutdown()
        assert rc == 0
        report = json.loads(path.read_text())
        assert report["ready_wait_s"] is not None
        assert report["throughput"]["completed"] == 4


def test_bench_waits_for_the_profile_report_before_loading():
    """--into sequences the two commands: no load until profiling is done.

    The report file, not the socket, is the go signal -- a real engine serves
    traffic before it announces readiness, so an HTTP probe alone would let
    bench start hammering a server the profiler is still measuring.
    """
    srv, url = _start_server()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "run.json"
        served_before_report = []

        def write_report_late():
            # Server already answering: the window bench must NOT be loading.
            wait_for_ready(url, timeout_s=10.0, interval_s=0.05)
            time.sleep(0.5)
            served_before_report.append(_request_count(url))
            path.write_text(json.dumps(PROFILE_REPORT))

        t = threading.Thread(target=write_report_late)
        t.start()
        try:
            rc = main(["bench", "--url", url, "--into", str(path), "--wait-ready", "10"] + BENCH_ARGS)
        finally:
            t.join()
            srv.shutdown()
        assert rc == 0
        assert json.loads(path.read_text())["benchmark"]["throughput"]["completed"] == 4
        assert served_before_report[0] <= 1  # only the probe from write_report_late


def test_bench_gives_up_if_the_profile_report_never_appears():
    srv, url = _start_server()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "run.json"
        try:
            rc = main(["bench", "--url", url, "--into", str(path), "--wait-ready", "0.3"] + BENCH_ARGS)
        finally:
            srv.shutdown()
        assert rc == 1
        assert not (Path(d) / "run.bench.json").exists()


def test_bench_salvages_results_when_into_never_appears():
    srv, url = _start_server()
    with tempfile.TemporaryDirectory() as d:
        missing = Path(d) / "nope.json"
        try:
            rc = main(["bench", "--url", url, "--into", str(missing)] + BENCH_ARGS)
        finally:
            srv.shutdown()
        assert rc == 1
        salvaged = json.loads((Path(d) / "nope.bench.json").read_text())
        assert salvaged["throughput"]["completed"] == 4


def test_bench_rejects_a_nonexistent_directory():
    assert main(["bench", "--url", _dead_url(), "--into", "/no/such/dir/run.json"]) == 2


def test_bench_requires_an_output_destination():
    assert main(["bench", "--url", _dead_url()]) == 2


def test_unknown_subcommand():
    assert main([]) == 2
    assert main(["frobnicate"]) == 2


def test_bench_reports_failure_when_server_never_serves():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "bench.json"
        rc = main(["bench", "--url", _dead_url(), "--out", str(path), "--wait-ready", "0.3"])
        assert rc == 1
        assert json.loads(path.read_text())["errors"]
