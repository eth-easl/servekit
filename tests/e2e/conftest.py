import re
import subprocess
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def sbatch_wait(script: Path) -> str:
    """Submit `script`, block until it finishes, return its job id."""
    proc = subprocess.run(
        ["sbatch", "--wait", script.name],
        cwd=script.parent,
        capture_output=True,
        text=True,
        check=True,
    )
    return re.search(r"Submitted batch job (\d+)", proc.stdout).group(1)
