import re
import subprocess
import time
from pathlib import Path

SCRIPTSDIR = Path(__file__).resolve().parent / "scripts"
EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def sbatch_wait(script: Path, timeout: float = 600, poll_interval: float = 5.0) -> str:
    """Submit `script`, poll until it finishes, return its job id.

    Cancels the job and raises if it's still queued/running after `timeout`
    seconds, so a hung job (e.g. a stuck weight load) fails the test instead
    of running until the SLURM allocation's own time limit.
    """
    submit = subprocess.run(
        ["sbatch", script.name],
        cwd=script.parent,
        capture_output=True,
        text=True,
        check=True,
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
