import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Mapping, Optional

SCRIPTSDIR = Path(__file__).resolve().parent / "scripts"

# The scripts' #SBATCH directives (partition, account, cpus-per-task) are
# tuned for CSCS Clariden. SLURM applies sbatch CLI flags over #SBATCH
# directives, so a cluster with different accounting or node shapes (e.g.
# Bristen: account infra02/partition normal/128 cpus per node, vs Clariden's
# a-infra02/debug/288) can run the same scripts unmodified by setting this,
# e.g. SERVEKIT_E2E_SBATCH_ARGS="--partition=normal --account=infra02 --cpus-per-task=128".
SBATCH_ARGS = shlex.split(os.environ.get("SERVEKIT_E2E_SBATCH_ARGS", ""))


def sbatch_wait(
    script: Path,
    timeout: float = 600,
    poll_interval: float = 5.0,
    env: Optional[Mapping[str, str]] = None,
) -> str:
    """Submit `script`, poll until it finishes, return its job id.

    Cancels the job and raises if it's still queued/running after `timeout`
    seconds, so a hung job (e.g. a stuck weight load) fails the test instead
    of running until the SLURM allocation's own time limit.

    `env` is passed per call rather than set on os.environ, so one test's knobs
    cannot leak into another's job in the same session.
    """
    submit = subprocess.run(
        ["sbatch", *SBATCH_ARGS, script.name],
        cwd=script.parent,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, **(env or {})},
    )
    job_id = re.search(r"Submitted batch job (\d+)", submit.stdout).group(1)

    deadline = time.monotonic() + timeout
    while True:
        status = subprocess.run(["squeue", "-h", "-j", job_id], capture_output=True, text=True)
        if status.returncode != 0:
            raise RuntimeError(f"squeue failed while polling job {job_id}: {status.stderr.strip()}")
        if not status.stdout.strip():
            break
        if time.monotonic() >= deadline:
            subprocess.run(["scancel", job_id])
            raise TimeoutError(f"job {job_id} ({script.name}) did not finish within {timeout:g}s; cancelled")
        time.sleep(poll_interval)

    return job_id
