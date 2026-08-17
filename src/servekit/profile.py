"""Cold-start profiling for LLM inference engines, by parsing their own stdout/stderr."""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

GAP_THRESHOLD_S = 0.5


@dataclass(frozen=True)
class FrameworkSpec:
    """Everything framework-specific about reading an engine's startup log.

    Phase names are shared across frameworks where a counterpart exists, so two
    reports from different engines stay directly diffable.
    """

    name: str
    # (phase name, regex whose capture group is the engine-reported elapsed seconds)
    phase_patterns: List[tuple]
    ready_pattern: "re.Pattern"
    # Markers with no engine timing of their own; each closes the interval since
    # the previous marker as a phase servekit times itself.
    milestone_patterns: List[tuple]
    # A gap is only named if *all* marker regexes for that hypothesis were seen
    # inside it -- otherwise it stays "unknown" rather than assumed.
    gap_hypotheses: List[tuple]
    command_markers: List["re.Pattern"]
    # Flags whose value is the model directory, most-specific first. Empty means
    # `servekit launch` cannot rewrite this engine's command (see engine_args).
    model_flags: List[str] = field(default_factory=list)
    # manifest field -> the flags that set it, defaulting to 1 when absent.
    parallel_flags: Dict[str, List[str]] = field(default_factory=dict)
    # Topology field -> the flags that set it. Empty means servekit cannot tell
    # a multi-node command from a single-node one, and assumes single-node.
    dist_flags: Dict[str, List[str]] = field(default_factory=dict)
    # What a worker node prints instead of ready_pattern, which only the head
    # ever prints. None means servekit cannot wrap a worker for this engine.
    worker_ready_pattern: Optional["re.Pattern"] = None


SGLANG = FrameworkSpec(
    name="sglang",
    phase_patterns=[
        ("torch_distributed_init", re.compile(r"Init torch distributed ends\. elapsed=([\d.]+) s")),
        ("weight_loading", re.compile(r"Load weight end\. elapsed=([\d.]+) s")),
        # Two spellings. SGLang renamed these between v0.5.10 and the 2026-07
        # nightlies ("Capture cuda graph end. Time elapsed: X s" ->
        # "Capture target decode CUDA graph end. elapsed=X s"), which silently
        # dropped ~16 s of a 121 s cold start into the unaccounted gap on job
        # 2922770. Both forms are kept so older profiles still parse.
        ("cuda_graph_capture", re.compile(r"Capture cuda graph end\. Time elapsed: ([\d.]+) s")),
        ("cuda_graph_capture", re.compile(r"Capture target decode CUDA graph end\. elapsed=([\d.]+) s")),
        ("prefill_cuda_graph_capture", re.compile(r"Capture target prefill CUDA graph end\. elapsed=([\d.]+) s")),
        ("piecewise_cuda_graph_capture", re.compile(r"Capture piecewise CUDA graph end\. Time elapsed: ([\d.]+) s")),
    ],
    ready_pattern=re.compile(r"The server is fired up and ready to roll!"),
    # Splits HTTP bind from the first warmup request, which is dominated by
    # first-call lazy init rather than serving work.
    milestone_patterns=[
        ("http_bind", re.compile(r"Uvicorn running on")),
        ("warmup_request(JIT)", re.compile(r'"POST /generate HTTP/1\.1"\s+200')),
    ],
    gap_hypotheses=[
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
    ],
    command_markers=[re.compile(r"\bsglang\b")],
    model_flags=["--model-path", "--model_path", "--model"],
    parallel_flags={
        "tp_size": ["--tensor-parallel-size", "--tp-size", "--tp"],
        "pp_size": ["--pipeline-parallel-size", "--pp-size", "--pp"],
        "dp_size": ["--data-parallel-size", "--dp-size", "--dp"],
    },
    dist_flags={
        "nnodes": ["--nnodes"],
        "node_rank": ["--node-rank"],
        "dist_init_addr": ["--dist-init-addr", "--nccl-init-addr"],
    },
    # A worker runs no tokenizer process and binds no API server, so it never
    # prints ready_pattern
    worker_ready_pattern=re.compile(r"Dummy health check server started"),
)

# The ready boundary is NOT the same event as SGLang's: vLLM never self-issues a
# warmup request, so its total excludes the first-call JIT that SGLang's
# includes. Read it together with bench's ready_wait_s.
VLLM = FrameworkSpec(
    name="vllm",
    phase_patterns=[
        ("weight_loading", re.compile(r"Model loading took .*? and ([\d.]+) seconds")),
        ("torch_compile", re.compile(r"torch\.compile took ([\d.]+) s in total")),
        ("cuda_graph_capture", re.compile(r"[Gg]raph capturing finished in ([\d.]+) secs")),
    ],
    ready_pattern=re.compile(r"Application startup complete"),
    # vLLM's uvicorn never logs a distinct bind line, and ready is the last
    # line of startup, so a milestone here could never fire (job 2918061).
    milestone_patterns=[],
    gap_hypotheses=[
        # vLLM reports no elapsed time for distributed init, and it falls in the
        # same gap as the worker spawn, so the two cannot be separated here.
        (
            "worker_spawn+dist_init",
            [
                re.compile(r"\(Worker(_TP\d+)? pid=\d+\)"),
                re.compile(r"distributed_init_method="),
            ],
        ),
        (
            "kv_cache_alloc",
            [
                re.compile(r"GPU KV cache size"),
                re.compile(r"Available KV cache memory"),
            ],
        ),
        # The tail after the engine is ready: FastAPI app construction and route
        # registration, which is all that separates graph capture from ready.
        (
            "api_server_startup",
            [
                re.compile(r"Starting vLLM server on"),
                re.compile(r"Available routes are:"),
            ],
        ),
    ],
    command_markers=[re.compile(r"\bvllm\b")],
)

FRAMEWORKS = [SGLANG, VLLM]


def detect_framework(command: List[str]) -> FrameworkSpec:
    """Pick a spec from the launch command, or raise.

    Deliberately no default: a guessed framework yields a half-parsed phase
    table that still looks like a measurement.
    """
    joined = " ".join(command)
    for spec in FRAMEWORKS:
        if any(m.search(joined) for m in spec.command_markers):
            return spec
    known = ", ".join(s.name for s in FRAMEWORKS)
    raise ValueError(f"cannot tell which framework launches {joined!r} (known: {known})")


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
    framework: str
    phases: List[Phase] = field(default_factory=list)
    benchmark: Optional[dict] = None  # filled in later by `servekit bench --into <report>`
    node_rank: Optional[int] = None
    nnodes: Optional[int] = None

    @property
    def total_duration_s(self) -> float:
        if self.ready_at is None:
            return 0.0
        return round(self.ready_at - self.started_at, 2)

    def to_dict(self) -> dict:
        return {
            "command": self.command,
            "framework": self.framework,
            "started_at": self.started_at,
            "ready_at": self.ready_at,
            "success": self.success,
            "total_duration_s": self.total_duration_s,
            "phases": [asdict(p) for p in self.phases],
            "benchmark": self.benchmark,
            "node_rank": self.node_rank,
            "nnodes": self.nnodes,
        }


def _process_stream(
    lines: Iterable[str],
    spawn_time: float,
    ready_timeout: float,
    spec: FrameworkSpec,
    clock: Callable[[], float] = time.time,
    echo: bool = True,
    stop_at_ready: bool = False,
    on_ready: Optional[Callable[["ProfileReport"], None]] = None,
    head: bool = True,
) -> ProfileReport:
    """Consume `lines`, extracting phase timings until ready (or timeout).

    Once ready is detected, matching stops but iteration continues (unless
    `stop_at_ready`) so a live server's output keeps streaming through --
    servekit only stops *measuring*, it never stops the process from serving.

    `head=False` resolves on the worker's own readiness line instead, since a
    worker never prints the head's.
    """
    seen_phases = set()
    seen_milestones = set()
    phases: List[Phase] = []
    phase_index: dict = {}
    # The engine phase most recently opened, kept open so other TP ranks can
    # still report into it (see the max-over-ranks note below).
    pending_phase: Optional[str] = None
    last_marker_time = spawn_time
    ready_at: Optional[float] = None
    timed_out = False

    gap_evidence = {name: set() for name, _ in spec.gap_hypotheses}
    ready_pattern = spec.ready_pattern if head else (spec.worker_ready_pattern or spec.ready_pattern)

    def gap_name() -> str:
        for name, markers in spec.gap_hypotheses:
            if len(gap_evidence[name]) == len(markers):
                return name
        return "unknown"

    def record_gap(gap: float) -> None:
        if gap > GAP_THRESHOLD_S:
            phases.append(Phase(gap_name(), round(gap, 2), "wall_clock"))
        for evidence in gap_evidence.values():
            evidence.clear()

    def resolve(at: float) -> bool:
        """Close the report at `at`. True if the caller should stop iterating."""
        nonlocal pending_phase
        pending_phase = None
        record_gap(at - last_marker_time)
        if on_ready is not None:
            on_ready(
                ProfileReport(
                    command="",
                    started_at=spawn_time,
                    ready_at=at,
                    success=True,
                    framework=spec.name,
                    phases=list(phases),
                )
            )
        return stop_at_ready

    for line in lines:
        if echo:
            # Flushed per line: our stdout is block-buffered when it is a file,
            # so without this the engine's log reaches the job output in 8 KB
            # bursts and is lost entirely if the wrapper is killed -- neither of
            # which happens when the engine runs unwrapped.
            sys.stdout.write(line)
            sys.stdout.flush()
        now = clock()

        if not phases:
            phases.append(Phase("process_startup", round(now - spawn_time, 2), "wall_clock"))
            last_marker_time = now

        if ready_at is not None:
            continue

        for name, markers in spec.gap_hypotheses:
            for idx, marker in enumerate(markers):
                if marker.search(line):
                    gap_evidence[name].add(idx)

        milestone_hit = False
        for name, pattern in spec.milestone_patterns:
            if name in seen_milestones:
                continue
            if pattern.search(line):
                seen_milestones.add(name)
                pending_phase = None  # a milestone closes any open engine phase
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

        # Each of these phases is TP-parallel (one `elapsed=` line per rank), and
        # is not over until the SLOWEST rank reports, so we keep it open and take
        # the MAX rather than locking on the first line seen. Taking the first
        # line took the *fastest* rank, which was badly wrong exactly when it
        # mattered: a fast /dev/shm load reported 3.72 s against a true 6.19 s.
        for name, pattern in spec.phase_patterns:
            match = pattern.search(line)
            if not match:
                continue
            duration = float(match.group(1))
            if name == pending_phase:
                # Another rank reporting into the still-open phase: extend, don't re-open.
                idx = phase_index[name]
                if duration > phases[idx].duration_s:
                    phases[idx].duration_s = round(duration, 2)
                last_marker_time = now
                break
            if name in seen_phases:
                break  # late straggler after another phase already started
            seen_phases.add(name)
            record_gap(now - last_marker_time - duration)
            phase_index[name] = len(phases)
            phases.append(Phase(name, round(duration, 2), "engine_reported"))
            last_marker_time = now
            pending_phase = name
            break

        if ready_pattern.search(line):
            ready_at = now
            if resolve(now):
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
        framework=spec.name,
        phases=phases,
    )


def run_profile(
    command: List[str],
    ready_timeout: float = 1800.0,
    on_ready: Optional[Callable[[ProfileReport], None]] = None,
    head: bool = True,
    env: Optional[Dict[str, str]] = None,
) -> ProfileReport:
    """Run `command`, profiling it until it reports ready.

    The wrapped process keeps running after ready is detected; only measurement
    stops, and `on_ready` (if given) fires once so a caller can emit the report
    without waiting for the server to exit. If ready is never seen within
    `ready_timeout`, the process is terminated and the run is marked failed.

    `head=False` says this node is a worker, which never prints the ready line;
    its own readiness line closes the report instead.

    stdout keeps draining for the process's whole life rather than detaching
    once measured: we own the read end of the pipe, so exiting would SIGPIPE the
    server on its next log line, and an unread pipe eventually fills and blocks
    the server on write.

    SIGTERM/SIGINT are forwarded to the child so a caller that backgrounded us
    (`servekit profile ... & ...; kill $!`) tears the server down too.
    """
    spec = detect_framework(command)
    spawn_time = time.time()
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, **env} if env else None,
    )

    def _forward(signum, _frame):  # noqa: ANN001 - signal handler signature
        print(f"\n[SERVEKIT] got signal {signum}, terminating the server", flush=True)
        proc.terminate()

    previous = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous[sig] = signal.signal(sig, _forward)
    try:
        report = _process_stream(
            proc.stdout,
            spawn_time,
            ready_timeout,
            spec=spec,
            on_ready=on_ready,
            head=head,
        )
        if not report.success:
            proc.terminate()
        proc.wait()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    report.command = " ".join(command)
    return report


def render_table(report: ProfileReport) -> str:
    rows = [(p.name, p.duration_s, p.source) for p in report.phases]
    name_w = max([len(r[0]) for r in rows] + [len("phase")])
    header = f"{'phase':<{name_w}}  {'duration_s':>10}  source"
    sep = "-" * len(header)
    title = f"[SERVEKIT] Cold-start profile (framework={report.framework})"
    if report.node_rank is not None:
        title += f" (node {report.node_rank}/{report.nnodes})"
    lines = [title, sep, header, sep]
    for name, dur, source in rows:
        lines.append(f"{name:<{name_w}}  {dur:>10.2f}  {source}")
    lines.append(sep)
    status = "ready" if report.success else "FAILED (no ready signal)"
    lines.append(f"{'total':<{name_w}}  {report.total_duration_s:>10.2f}  {status}")
    return "\n".join(lines)


def save_json(report: ProfileReport, path: Path) -> None:
    path.write_text(json.dumps(report.to_dict(), indent=2))
