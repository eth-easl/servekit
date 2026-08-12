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
from dataclasses import dataclass
from pathlib import Path

import pytest

from conftest import SCRIPTSDIR, sbatch_wait

os.environ.setdefault("MODE", "fast")

pytestmark = pytest.mark.e2e

FIXTURES = Path(__file__).parent / "fixtures"


def _result(job_id: str) -> dict:
    path = SCRIPTSDIR / "logs" / f"fast-{job_id}.json"
    assert path.is_file(), f"the fast arm wrote no verify result at {path}"
    return json.loads(path.read_text())


@dataclass
class Case:
    script: str
    fixture: str
    id: str
    skip: bool = False


CASES = [
    Case(
        script="llama70b.sbatch",
        fixture="llama70b-tp4-bf16.json",
        id="llama70b",
    ),
    Case(
        script="llama70b-multinode-tp8.sbatch",
        fixture="llama70b-tp8-bf16.json",
        id="llama70b_multinode_tp8",
    ),
    # PP
    Case(
        script="llama70b-pp.sbatch",
        fixture="llama70b-tp4pp2-bf16.json",
        id="llama70b_multinode_pp",
    ),
    # redundant
    Case(
        script="apertus8b.sbatch",
        fixture="apertus8b-tp4-bf16.json",
        id="apertus8b_tp4",
        skip=True,
    ),
    # moe, bf16
    Case(
        script="qwen3coder.sbatch",
        fixture="qwen3-coder-30b-a3b-tp4-bf16.json",
        id="qwen3_coder_30b_a3b_tp4",
    ),
]


@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=c.id, marks=pytest.mark.skip if c.skip else ()) for c in CASES],
)
def test_fast_weight_load_matches_the_lustre_baseline(case):
    fixture_path = FIXTURES / case.fixture
    if not fixture_path.is_file():
        pytest.skip(
            f"no baseline capture at {fixture_path}; run: MODE=baseline sbatch tests/e2e/scripts/correctness-{case.script}"
        )

    job_id = sbatch_wait(SCRIPTSDIR / f"correctness-{case.script}")
    result = _result(job_id)
    assert result["passed"], "\n".join(result["failures"])

