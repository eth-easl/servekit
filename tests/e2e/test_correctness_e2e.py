"""
Compares logprobs between the baseline loader (no servekit) and servekit fast weight load.
The logprobs are expected to be equal within a tolerance.

Dev workflow:
(1) MODE=baseline sbatch tests/e2e/scripts/correctness-llama70b.sbatch
(2) cp tests/e2e/scripts/logs/baseline-<id>.json tests/e2e/fixtures/<run-name>.json
(3) pytest tests/e2e/test_correctness_e2e.py -m e2e (runs MODE=fast; the sbatch job itself
    runs `servekit verify --reference` against the fixture and writes the result)
"""

import json
import os
from pathlib import Path

import pytest

from conftest import SCRIPTSDIR, sbatch_wait

os.environ.setdefault("MODE", "fast")

pytestmark = pytest.mark.e2e

FIXTURE = Path(__file__).parent / "fixtures" / "llama70b-tp4-bf16.json"


def _result(job_id: str) -> dict:
    path = SCRIPTSDIR / "logs" / f"fast-{job_id}.json"
    assert path.is_file(), f"the fast arm wrote no verify result at {path}"
    return json.loads(path.read_text())


def test_fast_weight_load_matches_the_lustre_baseline():
    if not FIXTURE.is_file():
        pytest.skip(f"no baseline capture at {FIXTURE}; run: MODE=baseline sbatch tests/e2e/scripts/correctness-llama70b.sbatch")

    result = _result(sbatch_wait(SCRIPTSDIR / "correctness-llama70b.sbatch"))
    assert result["passed"], "\n".join(result["failures"])
