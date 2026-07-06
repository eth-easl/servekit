import re
from datetime import datetime
from pathlib import Path

from servekit.profile import _process_stream

FIXTURE = Path(__file__).parent / "fixtures" / "apertus-8b-sglang.log"
TIMESTAMP_PATTERN = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def _fake_clock_stream(lines):
    """Replay `lines` against a clock driven by the log's own embedded
    timestamps (second resolution), so gap sizes match real production runs
    instead of an arbitrary per-line step."""
    state = {"t": 0.0, "base": None}

    def clock():
        return state["t"]

    def stream():
        for line in lines:
            match = TIMESTAMP_PATTERN.match(line)
            if match:
                ts = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
                if state["base"] is None:
                    state["base"] = ts
                state["t"] = ts - state["base"]
            else:
                state["t"] += 0.001
            yield line

    return stream(), clock


def test_parses_real_sglang_log():
    lines = FIXTURE.read_text().splitlines(keepends=True)
    stream, clock = _fake_clock_stream(lines)

    report = _process_stream(stream, spawn_time=0.0, ready_timeout=3600.0, clock=clock, echo=False)

    assert report.success
    assert report.ready_at is not None

    by_name = {p.name: p for p in report.phases}
    assert by_name["process_startup"].source == "wall_clock"
    assert by_name["torch_distributed_init"].duration_s == 5.64
    assert by_name["weight_loading"].duration_s == 101.61
    assert by_name["cuda_graph_capture"].duration_s == 17.13
    assert by_name["piecewise_cuda_graph_capture"].duration_s == 21.42

    # Diagnosed gaps: marker lines for each hypothesis are present in the
    # fixture inside the expected window, so they should be named rather than
    # left as "unknown".
    assert by_name["tp_worker_spawn"].source == "wall_clock"  # chat template -> first TP-tagged line
    assert by_name["kv_cache_alloc"].source == "wall_clock"  # KV Cache is allocated -> Memory pool end

    # The old lumped "http_bind_and_warmup" gap is now split at the "Uvicorn
    # running" milestone: HTTP bind (piecewise graph end -> Uvicorn up) vs. the
    # first, JIT-heavy warmup request (Uvicorn up -> POST /generate 200).
    assert by_name["http_bind"].source == "wall_clock"
    assert 1.0 <= by_name["http_bind"].duration_s <= 4.0
    assert by_name["warmup_request(JIT)"].source == "wall_clock"
    assert 8.0 <= by_name["warmup_request(JIT)"].duration_s <= 12.0

    # Gaps we haven't diagnosed a hypothesis for yet (e.g. between
    # torch_distributed_init and weight_loading, where nothing is logged at
    # all) must stay "unknown" rather than being guessed.
    assert "unknown" in by_name
