from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str


def _run_argv(argv: list[str], *, stream: bool = False) -> RunResult:
    """The single shell-out boundary. Tests monkeypatch this function."""
    if stream:
        proc = subprocess.run(argv)
        return RunResult(proc.returncode, "", "")
    proc = subprocess.run(argv, capture_output=True, text=True)
    return RunResult(proc.returncode, proc.stdout, proc.stderr)


def push(profile: str) -> RunResult:
    return _run_argv(["rcc", "--profile", profile, "push"])


def run(profile: str, command: str, *, stream: bool = False) -> RunResult:
    return _run_argv(
        ["rcc", "--profile", profile, "run", "bash", "-lc", command],
        stream=stream,
    )
