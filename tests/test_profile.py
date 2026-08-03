import re
from datetime import datetime
from pathlib import Path

import pytest

from servekit.profile import SGLANG, VLLM, _process_stream, detect_framework

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE = FIXTURES / "apertus-8b-sglang.log"
VLLM_FIXTURE = FIXTURES / "llama70b-vllm-ngc.log"
TIMESTAMP_PATTERN = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
# vLLM stamps mid-line and omits the year: "INFO 07-28 17:04:47 [monitor.py:53]".
VLLM_TIMESTAMP_PATTERN = re.compile(r"\b(\d{2}-\d{2} \d{2}:\d{2}:\d{2})\b")


def _timestamp(line):
    match = TIMESTAMP_PATTERN.match(line)
    if match:
        return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
    match = VLLM_TIMESTAMP_PATTERN.search(line)
    if match:
        return datetime.strptime("2026-" + match.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
    return None


def _fake_clock_stream(lines):
    """Replay `lines` against a clock driven by the log's own embedded
    timestamps, so gap sizes match real production runs."""
    state = {"t": 0.0, "base": None}

    def clock():
        return state["t"]

    def stream():
        for line in lines:
            ts = _timestamp(line)
            if ts is not None:
                if state["base"] is None:
                    state["base"] = ts
                state["t"] = ts - state["base"]
            else:
                state["t"] += 0.001
            yield line

    return stream(), clock


def test_detects_framework_from_the_launch_command():
    assert detect_framework(["python", "-m", "sglang.launch_server", "--model-path", "m"]) is SGLANG
    assert detect_framework(["vllm", "serve", "/models/llama"]) is VLLM
    # No default: an unattributable command must fail loudly rather than emit an
    # empty phase table that still looks like a measurement.
    with pytest.raises(ValueError):
        detect_framework(["python", "-m", "http.server"])


def test_parses_real_sglang_log():
    lines = FIXTURE.read_text().splitlines(keepends=True)
    stream, clock = _fake_clock_stream(lines)

    report = _process_stream(stream, spawn_time=0.0, ready_timeout=3600.0, spec=SGLANG, clock=clock, echo=False)

    assert report.success
    assert report.ready_at is not None

    by_name = {p.name: p for p in report.phases}
    assert by_name["process_startup"].source == "wall_clock"
    assert by_name["torch_distributed_init"].duration_s == 5.64
    # Max over TP ranks (fixture's four ranks report 101.61/101.65/101.66/101.63).
    assert by_name["weight_loading"].duration_s == 101.66
    assert by_name["cuda_graph_capture"].duration_s == 17.13
    assert by_name["piecewise_cuda_graph_capture"].duration_s == 21.42

    # Diagnosed gaps: marker lines for each hypothesis are present, so they
    # should be named rather than left as "unknown".
    assert by_name["tp_worker_spawn"].source == "wall_clock"
    assert by_name["kv_cache_alloc"].source == "wall_clock"

    # HTTP bind vs. the first, JIT-heavy warmup request, split at "Uvicorn running".
    assert by_name["http_bind"].source == "wall_clock"
    assert 1.0 <= by_name["http_bind"].duration_s <= 4.0
    assert by_name["warmup_request(JIT)"].source == "wall_clock"
    assert 8.0 <= by_name["warmup_request(JIT)"].duration_s <= 12.0

    # An undiagnosed gap (torch_distributed_init -> weight_loading) stays "unknown".
    assert "unknown" in by_name


def test_parses_real_vllm_log():
    """Job 2918061: Llama-3.1-70B, TP=4, GH200, nvcr.io#nvidia/vllm:26.07-py3."""
    lines = VLLM_FIXTURE.read_text().splitlines(keepends=True)
    stream, clock = _fake_clock_stream(lines)

    report = _process_stream(stream, spawn_time=0.0, ready_timeout=3600.0, spec=VLLM, clock=clock, echo=False)

    assert report.success
    by_name = {p.name: p for p in report.phases}
    # Max over the four ranks (232.88 / 233.14 / 233.14 / 233.24).
    assert by_name["weight_loading"].duration_s == 233.24
    assert by_name["torch_compile"].duration_s == 35.70
    assert by_name["cuda_graph_capture"].duration_s == 11.00

    assert by_name["worker_spawn+dist_init"].source == "wall_clock"
    assert by_name["kv_cache_alloc"].source == "wall_clock"
    # Everything between graph capture and ready is the API server coming up.
    assert by_name["api_server_startup"].source == "wall_clock"

    # vLLM emits no warmup request of its own, so nothing may be attributed to
    # one -- the tail before ready stays an honest unnamed gap.
    assert "warmup_request(JIT)" not in by_name
    assert "http_bind" not in by_name


def _replay(lines):
    stream, clock = _fake_clock_stream(lines)
    report = _process_stream(stream, spawn_time=0.0, ready_timeout=3600.0, spec=SGLANG, clock=clock, echo=False)
    return {p.name: p for p in report.phases}


def test_tp_parallel_phase_takes_max_over_ranks():
    """A TP-parallel phase ends with the SLOWEST rank, not the first to report.

    Regression test: taking the first line took the *fastest* rank, which
    understated a 6.19 s /dev/shm load as 3.72 s (-40%). The bias is worst
    exactly where the phase is fast and per-rank spread is widest.
    """
    lines = [
        "[2026-01-01 00:00:00] Init torch distributed ends. elapsed=1.00 s\n",
        "[2026-01-01 00:00:10 TP1] Load weight end. elapsed=3.72 s\n",
        "[2026-01-01 00:00:10 TP2] Load weight end. elapsed=4.06 s\n",
        "[2026-01-01 00:00:12 TP3] Load weight end. elapsed=5.81 s\n",
        "[2026-01-01 00:00:12 TP0] Load weight end. elapsed=6.19 s\n",
        "[2026-01-01 00:00:20] Capture cuda graph end. Time elapsed: 2.00 s\n",
        "[2026-01-01 00:00:30] The server is fired up and ready to roll!\n",
    ]
    by_name = _replay(lines)
    assert by_name["weight_loading"].duration_s == 6.19


def test_out_of_order_ranks_still_take_max():
    """The slowest rank reporting FIRST must not be overwritten by a faster one."""
    lines = [
        "[2026-01-01 00:00:10 TP0] Load weight end. elapsed=6.19 s\n",
        "[2026-01-01 00:00:10 TP1] Load weight end. elapsed=3.72 s\n",
        "[2026-01-01 00:00:20] The server is fired up and ready to roll!\n",
    ]
    assert _replay(lines)["weight_loading"].duration_s == 6.19


def test_later_phase_is_not_absorbed_into_the_open_one():
    """Only the SAME phase name extends an open phase; a different one opens its own."""
    lines = [
        "[2026-01-01 00:00:10 TP0] Load weight end. elapsed=5.00 s\n",
        "[2026-01-01 00:00:10 TP1] Load weight end. elapsed=5.50 s\n",
        "[2026-01-01 00:00:20 TP0] Capture cuda graph end. Time elapsed: 2.00 s\n",
        "[2026-01-01 00:00:20 TP1] Capture cuda graph end. Time elapsed: 2.75 s\n",
        "[2026-01-01 00:00:30] The server is fired up and ready to roll!\n",
    ]
    by_name = _replay(lines)
    assert by_name["weight_loading"].duration_s == 5.50
    assert by_name["cuda_graph_capture"].duration_s == 2.75


# --- a worker node's log ----------------------------------------------------

WORKER_FIXTURE = FIXTURES / "llama70b-sglang-worker-node.log"


def test_a_worker_nodes_log_resolves_without_a_ready_line():
    """Node 1 of the 2-node TP=8 run: real output, and it never says "ready".

    It says "Dummy health check server started" instead, which SGLang logs once
    every scheduler on the node is constructed.
    """
    lines = WORKER_FIXTURE.read_text().splitlines(keepends=True)
    stream, clock = _fake_clock_stream(lines)

    report = _process_stream(
        stream, spawn_time=0.0, ready_timeout=3600.0, spec=SGLANG, clock=clock, echo=False, head=False,
    )

    assert not SGLANG.ready_pattern.search(WORKER_FIXTURE.read_text())
    assert report.success and report.ready_at == 92.0

    by_name = {p.name: p for p in report.phases}
    # Ranks 4-7's own numbers, which node 0's report cannot see.
    assert by_name["torch_distributed_init"].duration_s == 14.63
    assert by_name["weight_loading"].duration_s == 3.23
    assert by_name["cuda_graph_capture"].duration_s == 21.03
    assert by_name["piecewise_cuda_graph_capture"].duration_s == 39.87


def test_the_worker_signal_lands_after_the_read_that_follows_capture():
    """Job 2990070: freeing at capture end killed all eight ranks in get_tokenizer."""
    lines = WORKER_FIXTURE.read_text().splitlines(keepends=True)
    last_capture = max(i for i, line in enumerate(lines) if "Capture piecewise CUDA graph end" in line)
    worker_ready = next(i for i, line in enumerate(lines) if SGLANG.worker_ready_pattern.search(line))
    assert worker_ready > last_capture


def test_a_worker_pattern_is_not_used_on_the_head():
    """The head's own ready line is later and stricter; a worker never prints it."""
    lines = FIXTURE.read_text().splitlines(keepends=True)
    stream, clock = _fake_clock_stream(lines)

    report = _process_stream(
        stream, spawn_time=0.0, ready_timeout=3600.0, spec=SGLANG, clock=clock, echo=False, head=True,
    )

    assert report.success
    assert SGLANG.ready_pattern.search(FIXTURE.read_text())
