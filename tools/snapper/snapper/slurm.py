from __future__ import annotations

import re
import shlex
import time

from . import rcc

_JOBID_RE = re.compile(r"Submitted batch job (\d+)")
_RANK_RE = re.compile(r"^===RANK (\d+)===$")
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_jobid(sbatch_stdout: str) -> int:
    match = _JOBID_RE.search(sbatch_stdout)
    if not match:
        raise ValueError(f"could not parse jobid from sbatch output: {sbatch_stdout!r}")
    return int(match.group(1))


def parse_squeue_state(stdout: str) -> str | None:
    state = stdout.strip()
    return state or None


def parse_rank_dump(text: str) -> dict[int, str]:
    logs: dict[int, str] = {}
    current: int | None = None
    buf: list[str] = []

    for line in text.splitlines():
        match = _RANK_RE.match(line)
        if match:
            if current is not None:
                logs[current] = "\n".join(buf)
            current = int(match.group(1))
            buf = []
            continue
        if current is not None:
            buf.append(line)

    if current is not None:
        logs[current] = "\n".join(buf)
    return logs


def count_ready_ranks(logs_by_rank: dict[int, str], markers: list[str]) -> int:
    return sum(
        1
        for text in logs_by_rank.values()
        if all(marker in text for marker in markers)
    )


def _env_prefix(env: dict[str, str]) -> str:
    parts = []
    for key, value in env.items():
        if not _ENV_KEY_RE.match(key):
            raise ValueError(f"invalid environment variable name: {key!r}")
        parts.append(f"{key}={shlex.quote(str(value))}")
    return " ".join(parts) + (" " if parts else "")


def submit(dep, *, nodes: int, serve_script: str, env: dict[str, str], rcc_mod=rcc) -> int:
    cmd = f"{_env_prefix(env)}sbatch --nodes={nodes} {dep.rel_dir}/{serve_script}"
    res = rcc_mod.run(dep.profile, cmd)
    if res.returncode != 0:
        raise RuntimeError(f"sbatch failed: {res.stderr or res.stdout}")
    return parse_jobid(res.stdout)


def probe(dep, *, rcc_mod=rcc) -> int:
    if not dep.probe_script:
        raise ValueError(f"{dep.name}: no probe script declared")
    res = rcc_mod.run(dep.profile, f"sbatch {dep.rel_dir}/{dep.probe_script}")
    if res.returncode != 0:
        raise RuntimeError(f"probe sbatch failed: {res.stderr or res.stdout}")
    return parse_jobid(res.stdout)


def wait_in_queue(
    profile: str,
    jobid: int,
    *,
    poll: float = 5.0,
    timeout: float = 1800,
    rcc_mod=rcc,
    sleep=time.sleep,
) -> str:
    """Block until the job leaves the queue; return its last observed state."""
    waited = 0.0
    last = "PENDING"
    while True:
        res = rcc_mod.run(profile, f"squeue -j {jobid} -h -o %T")
        if res.returncode != 0:
            raise RuntimeError(f"squeue failed: {res.stderr or res.stdout}")
        state = parse_squeue_state(res.stdout)
        if state is None:
            return ""
        last = state
        if waited >= timeout:
            return last
        sleep(poll)
        waited += poll


def _fetch_rank_logs(dep, jobid: int, nodes: int, rcc_mod=rcc) -> dict[int, str]:
    ranks = " ".join(str(rank) for rank in range(nodes))
    pattern = dep.log_pattern.format(job=jobid, rank="$r")
    cmd = f'for r in {ranks}; do echo "===RANK $r==="; cat {pattern} 2>/dev/null; done'
    res = rcc_mod.run(dep.profile, cmd)
    if res.returncode != 0:
        raise RuntimeError(f"log fetch failed: {res.stderr or res.stdout}")
    return parse_rank_dump(res.stdout)


def wait_for_ready(
    dep,
    jobid: int,
    nodes: int,
    *,
    poll: float = 15.0,
    timeout: float = 1800,
    rcc_mod=rcc,
    sleep=time.sleep,
) -> bool:
    waited = 0.0
    while True:
        logs = _fetch_rank_logs(dep, jobid, nodes, rcc_mod=rcc_mod)
        if count_ready_ranks(logs, dep.ready_markers) >= nodes:
            return True
        if waited >= timeout:
            return False
        sleep(poll)
        waited += poll
