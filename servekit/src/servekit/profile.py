"""Cold-start profiling for LLM inference engines, by parsing their own stdout/stderr."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from .bench import BenchConfig, run_benchmark

# (phase name, regex whose sole capture group is the engine-reported elapsed seconds)
SGLANG_PHASE_PATTERNS = [
    ("torch_distributed_init", re.compile(r"Init torch distributed ends\. elapsed=([\d.]+) s")),
    ("weight_loading", re.compile(r"Load weight end\. elapsed=([\d.]+) s")),
    ("cuda_graph_capture", re.compile(r"Capture cuda graph end\. Time elapsed: ([\d.]+) s")),
    ("piecewise_cuda_graph_capture", re.compile(r"Capture piecewise CUDA graph end\. Time elapsed: ([\d.]+) s")),
]
READY_PATTERN = re.compile(r"The server is fired up and ready to roll!")

# Wall-clock milestones from the tail of startup, where the engine emits no
# `elapsed=` timing of its own. Each milestone, the first time it is seen,
# closes the interval since the previous marker as a phase whose duration
# servekit measures itself. This is what splits the old lumped
# "http_bind_and_warmup" gap into the HTTP bind vs. the first (JIT-heavy)
# warmup request: the warmup POST on an already-loaded, already-captured model
# is dominated by first-call lazy init (FlashInfer JIT/autotune, sampling
# backend, first kernel loads), not by serving work.
MILESTONE_PATTERNS = [
    ("http_bind", re.compile(r"Uvicorn running on")),
    ("warmup_request(JIT)", re.compile(r'"POST /generate HTTP/1\.1"\s+200')),
]

GAP_THRESHOLD_S = 0.5

# A gap between two phases is only named if *all* marker regexes for that
# hypothesis were seen among the lines inside the gap -- otherwise it's
# labeled "unknown" rather than assumed. Add entries here as gaps get
# diagnosed one at a time; unmatched gaps stay "unknown" until then.
GAP_HYPOTHESES = [
    (
        "tp_worker_spawn",
        [
            re.compile(r"Using default HuggingFace chat template"),
            re.compile(r"TP\d+\]"),
        ],
    ),
    (
        "kv_cache_alloc",
        [
            re.compile(r"KV Cache is allocated"),
            re.compile(r"Memory pool end\."),
        ],
    ),
]


@dataclass
class Phase:
    name: str
    duration_s: float
    source: str  # "wall_clock" (measured by servekit) or "engine_reported" (parsed from the log)


@dataclass
class ProfileReport:
    command: str
    started_at: float
    ready_at: Optional[float]
    success: bool
    phases: List[Phase] = field(default_factory=list)
    benchmark: Optional[dict] = None  # post-ready bench results, if --bench was used

    @property
    def total_duration_s(self) -> float:
        if self.ready_at is None:
            return 0.0
        return round(self.ready_at - self.started_at, 2)

    def to_dict(self) -> dict:
        return {
            "command": self.command,
            "started_at": self.started_at,
            "ready_at": self.ready_at,
            "success": self.success,
            "total_duration_s": self.total_duration_s,
            "phases": [asdict(p) for p in self.phases],
            "benchmark": self.benchmark,
        }


def _process_stream(
    lines: Iterable[str],
    spawn_time: float,
    ready_timeout: float,
    clock: Callable[[], float] = time.time,
    echo: bool = True,
    stop_at_ready: bool = False,
    on_ready: Optional[Callable[["ProfileReport"], None]] = None,
) -> ProfileReport:
    """Consume `lines`, extracting phase timings until ready (or timeout).

    Once ready is detected, matching stops but iteration continues (unless
    `stop_at_ready`) so a live server's output keeps streaming through --
    servekit only stops *measuring*, it never stops the process from serving.
    """
    seen_phases = set()
    seen_milestones = set()
    phases: List[Phase] = []
    last_marker_time = spawn_time
    ready_at: Optional[float] = None
    timed_out = False

    gap_evidence = {name: set() for name, _ in GAP_HYPOTHESES}

    def gap_name() -> str:
        for name, markers in GAP_HYPOTHESES:
            if len(gap_evidence[name]) == len(markers):
                return name
        return "unknown"

    def record_gap(gap: float) -> None:
        if gap > GAP_THRESHOLD_S:
            phases.append(Phase(gap_name(), round(gap, 2), "wall_clock"))
        for evidence in gap_evidence.values():
            evidence.clear()

    # blocks this thread when a new line is not available yet and the pipe is not closed.
    for line in lines:
        if echo:
            sys.stdout.write(line)
        now = clock()

        if not phases:
            phases.append(Phase("process_startup", round(now - spawn_time, 2), "wall_clock"))
            last_marker_time = now

        if ready_at is not None:
            continue

        for name, markers in GAP_HYPOTHESES:
            for idx, marker in enumerate(markers):
                if marker.search(line):
                    gap_evidence[name].add(idx)

        milestone_hit = False
        for name, pattern in MILESTONE_PATTERNS:
            if name in seen_milestones:
                continue
            if pattern.search(line):
                seen_milestones.add(name)
                gap = now - last_marker_time
                if gap > GAP_THRESHOLD_S:
                    phases.append(Phase(name, round(gap, 2), "wall_clock"))
                last_marker_time = now
                for evidence in gap_evidence.values():
                    evidence.clear()
                milestone_hit = True
                break
        if milestone_hit:
            continue

        for name, pattern in SGLANG_PHASE_PATTERNS:
            if name in seen_phases:
                continue
            match = pattern.search(line)
            if not match:
                continue
            seen_phases.add(name)
            duration = float(match.group(1))
            record_gap(now - last_marker_time - duration)
            phases.append(Phase(name, round(duration, 2), "engine_reported"))
            last_marker_time = now
            break

        if READY_PATTERN.search(line):
            ready_at = now
            record_gap(now - last_marker_time)
            if on_ready is not None:
                on_ready(ProfileReport(command="", started_at=spawn_time, ready_at=ready_at, success=True, phases=list(phases)))
            if stop_at_ready:
                break
            continue

        if now - spawn_time > ready_timeout:
            timed_out = True
            break

    return ProfileReport(
        command="",
        started_at=spawn_time,
        ready_at=ready_at,
        success=ready_at is not None and not timed_out,
        phases=phases,
    )


def run_profile(
    command: List[str],
    ready_timeout: float = 1800.0,
    on_ready: Optional[Callable[[ProfileReport], None]] = None,
) -> ProfileReport:
    """Run `command`, profiling it until it reports ready.

    The wrapped process (e.g. the SGLang server) keeps running after ready is
    detected; only measurement stops, and `on_ready` (if given) fires once so
    a caller can emit the report without waiting for the server to exit. If
    ready is never seen within `ready_timeout`, the process is terminated and
    the run is marked failed.
    """
    spawn_time = time.time()
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    report = _process_stream(proc.stdout, spawn_time, ready_timeout, on_ready=on_ready)
    if not report.success:
        proc.terminate()
    proc.wait()
    report.command = " ".join(command)
    return report


def run_profile_with_bench(
    command: List[str],
    base_url: str,
    bench_config: BenchConfig,
    ready_timeout: float = 1800.0,
) -> ProfileReport:
    """Profile cold start, then benchmark the live server before tearing it down.

    The server's stdout is drained in a background thread the whole time. This
    matters: under benchmark load the engine logs heavily, and if we stopped
    reading stdout the OS pipe buffer would fill and the server would block on
    write -> deadlock. So the drain thread keeps measuring/echoing while the
    main thread hits the server with requests. Once the benchmark finishes,
    the server is terminated and the combined report is returned.
    """
    spawn_time = time.time()
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    ready_event = threading.Event()
    holder: dict = {}

    def _on_ready(_: ProfileReport) -> None:
        ready_event.set()

    def _drain() -> None:
        holder["report"] = _process_stream(
            proc.stdout, spawn_time, ready_timeout, on_ready=_on_ready
        )

    drain = threading.Thread(target=_drain, daemon=True)
    drain.start()

    # Wait until the server is ready, or the drain thread ends first (failure/timeout).
    while not ready_event.is_set() and drain.is_alive():
        drain.join(timeout=0.5)

    bench_result = None
    if ready_event.is_set():
        print("\n[SERVEKIT] server ready -> running post-ready benchmark", flush=True)
        bench_result = run_benchmark(base_url, bench_config).to_dict()

    # Measurement done: stop the server and let the drain thread finish.
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    drain.join(timeout=10)

    report = holder.get("report") or ProfileReport(
        command="", started_at=spawn_time, ready_at=None, success=False
    )
    report.command = " ".join(command)
    report.benchmark = bench_result
    return report


def render_table(report: ProfileReport) -> str:
    rows = [(p.name, p.duration_s, p.source) for p in report.phases]
    name_w = max([len(r[0]) for r in rows] + [len("phase")])
    header = f"{'phase':<{name_w}}  {'duration_s':>10}  source"
    sep = "-" * len(header)
    lines = ["[SERVEKIT] Cold-start profile", sep, header, sep]
    for name, dur, source in rows:
        lines.append(f"{name:<{name_w}}  {dur:>10.2f}  {source}")
    lines.append(sep)
    status = "ready" if report.success else "FAILED (no ready signal)"
    lines.append(f"{'total':<{name_w}}  {report.total_duration_s:>10.2f}  {status}")
    return "\n".join(lines)


def save_json(report: ProfileReport, path: Path) -> None:
    path.write_text(json.dumps(report.to_dict(), indent=2))
