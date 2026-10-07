"""Replay a saved vLLM log through Servekit's live profiling parser."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from .profile import FrameworkSpec, ProfileReport, VLLM, _process_stream

SPECS = Path(__file__).parent / "profile_specs" / "vllm"
DOCKER_TIME = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:\d\d))\s")
VLLM_TIME = re.compile(r"\b(\d\d-\d\d \d\d:\d\d:\d\d)\b")
BANNER = re.compile(r"\bversion (\d+\.\d+\.\d+[0-9A-Za-z.+_-]*)")
VERSION = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:rc(\d+))?(?:\.dev(\d+))?")


def _version_key(version: str) -> Tuple[int, ...]:
    match = VERSION.match(version)
    if not match:
        raise ValueError(f"cannot parse vLLM version {version!r}")
    major, minor, patch, rc, dev = match.groups()
    return (int(major), int(minor), int(patch),
            int(rc is None), int(rc or 0), int(dev is None), int(dev or 0))


def _select_spec(version: str) -> Tuple[Path, bool]:
    paths = list(SPECS.glob("*.json"))
    exact = next((path for path in paths if path.stem == version), None)
    if exact:
        return exact, False
    older = [path for path in paths if _version_key(path.stem) <= _version_key(version)]
    if not older:
        raise ValueError(f"no vLLM patterns at or below version {version}")
    return max(older, key=lambda path: _version_key(path.stem)), True


def _load_spec(path: Path) -> FrameworkSpec:
    """A release file supplies patterns, never a separate parsing algorithm."""
    data = json.loads(path.read_text())
    if data["version"] != path.stem or data["engine"] != "vllm":
        raise ValueError(f"{path.name}: expected vLLM patterns for version {path.stem}")
    phases = []
    milestones = []
    gaps = [("server_startup", [VLLM.ready_pattern])]
    for event in data["events"]:
        name = event["phase"]
        line = re.compile(event["line"])
        if "took" in event:
            phases.append((name, re.compile(event["took"])))
            if "before" in event:
                gaps.append((event["before"], [line]))
        else:
            milestones.append((name, line))
    return FrameworkSpec(
        name="vllm", phase_patterns=phases, ready_pattern=VLLM.ready_pattern,
        milestone_patterns=milestones, gap_hypotheses=gaps,
        command_markers=VLLM.command_markers,
        repeatable_phases=data.get("repeatable_phases", []),
    )


def _parse_time(line: str, year: int) -> Optional[float]:
    docker = DOCKER_TIME.match(line)
    if docker:
        return datetime.fromisoformat(docker.group(1).replace("Z", "+00:00")).timestamp()
    own = VLLM_TIME.search(line)
    if own:
        return datetime.strptime(f"{year}-{own.group(1)}", "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc).timestamp()
    return None


def profile_log(path: Path, launch_time: Optional[str] = None) -> Tuple[ProfileReport, bool]:
    """Return the usual Servekit report and whether its patterns are approximate."""
    lines: List[str] = path.read_text(errors="replace").splitlines(keepends=True)
    version = next((match.group(1) for line in lines[:400]
                    if (match := BANNER.search(line))), None)
    if version is None:
        raise ValueError(f"{path}: no vLLM version banner in the first 400 lines")
    spec_path, approximate = _select_spec(version)
    spec = _load_spec(spec_path)

    start = None
    if launch_time:
        stamp = datetime.fromisoformat(launch_time.replace("Z", "+00:00"))
        start = (stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)).timestamp()
    year = datetime.fromtimestamp(start, timezone.utc).year if start is not None else datetime.now(timezone.utc).year
    timed = []
    previous = None
    for line in lines:
        at = _parse_time(line, year)
        if at is not None:
            previous = at
        if previous is not None:
            timed.append((line, previous))
    if not timed:
        raise ValueError(f"{path}: no Docker or vLLM timestamps found")
    first = timed[0][1]
    if start is None:
        start = first
    elif start > first:
        raise ValueError(f"--launch-time is after the first log line in {path}")

    current = first

    def replay():
        nonlocal current
        for line, at in timed:
            current = at
            yield line

    report = _process_stream(replay(), spawn_time=start, ready_timeout=float("inf"),
                             spec=spec, clock=lambda: current, echo=False, stop_at_ready=True)
    if not report.success:
        raise ValueError(f"{path}: no vLLM ready line found")
    report.framework_version = version
    report.pattern_version = spec_path.stem
    return report, approximate
