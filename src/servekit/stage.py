"""Running the sliced stager and reading back what it did.

`_stage/stage_to_shm_sliced.sh` is the experiment's script byte-for-byte, not a
reimplementation: the code path that measured 11.9 GB/s on Bristen and 17.0 GB/s
on Clariden is the code path that ships, and stays diffable against
`experiments/lustre-loading-exp/scripts/phase4_shm/`.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

STAGER = Path(__file__).parent / "_stage" / "stage_to_shm_sliced.sh"

# 64 slices per file over every file at once: the measured fan-out on both a
# 128-core and a 288-core node. Not derived from cpu_count -- no measurement
# backs a scaling rule, and this one is in the critical path.
DEFAULT_SLICES = 64

_SUMMARY = re.compile(r"staged_bytes=(\d+).*stage_wall_s=([\d.]+) stage_GBps=([\d.]+)")


@dataclass
class StageResult:
    dest: Path
    wall_s: float
    gbps: float
    bytes: int


def stage(
    src: Path,
    dest: Path,
    slices: int = DEFAULT_SLICES,
    file_pattern: str = "*",
    files: Optional[Sequence[str]] = None,
) -> StageResult:
    """Copy `src` into `dest`, returning the stager's own timing.

    `files` names an explicit set and overrides `file_pattern`; a presharded
    node's file set is a union of its ranks' reads, which no glob describes.

    Raises on a non-zero exit, which covers the free-space pre-flight and the
    size-parity gate the script already does.
    """
    # The stager forks one `bash -c` per slice. NVIDIA's NGC images set
    # BASH_ENV=/etc/bash.bashrc, so each of those thousands of shells would run
    # nvidia-smi before its dd: 0.41 GB/s instead of 16.89, and the engine
    # starved alongside it (clariden-loading-exp, results/vllm/llama-3.1-70b).
    env = {k: v for k, v in os.environ.items() if k != "BASH_ENV"}
    env["FILE_PATTERN"] = file_pattern
    with tempfile.TemporaryDirectory() as tmp:
        if files is not None:
            listing = Path(tmp) / "files.txt"
            listing.write_text("\n".join(files) + "\n")
            env["FILE_LIST"] = str(listing)
        proc = subprocess.run(
            ["bash", str(STAGER), str(src), str(dest), str(slices)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    sys.stdout.write(proc.stdout)
    sys.stdout.flush()
    if proc.returncode != 0:
        raise RuntimeError(f"stager failed (rc={proc.returncode}) staging {src} -> {dest}")
    match = _SUMMARY.search(proc.stdout)
    if match is None:
        raise RuntimeError(f"stager exited 0 but printed no summary line; refusing to trust {dest}")
    return StageResult(dest=dest, bytes=int(match.group(1)), wall_s=float(match.group(2)), gbps=float(match.group(3)))
