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
FIXTURE_TP8 = Path(__file__).parent / "fixtures" / "llama70b-tp8-bf16.json"
FIXTURE_APERTUS8B_TP4 = Path(__file__).parent / "fixtures" / "apertus8b-tp4-bf16.json"
FIXTURE_QWEN3_CODER_TP4 = Path(__file__).parent / "fixtures" / "qwen3-coder-30b-a3b-tp4-bf16.json"


def _result(job_id: str, prefix: str = "fast") -> dict:
    path = SCRIPTSDIR / "logs" / f"{prefix}-{job_id}.json"
    assert path.is_file(), f"the fast arm wrote no verify result at {path}"
    return json.loads(path.read_text())


def test_fast_weight_load_matches_the_lustre_baseline_llama70b():
    if not FIXTURE.is_file():
        pytest.skip(f"no baseline capture at {FIXTURE}; run: MODE=baseline sbatch tests/e2e/scripts/correctness-llama70b.sbatch")

    result = _result(sbatch_wait(SCRIPTSDIR / "correctness-llama70b.sbatch"))
    assert result["passed"], "\n".join(result["failures"])


def test_fast_weight_load_matches_the_lustre_baseline_llama70b_multinode_tp8():
    if not FIXTURE_TP8.is_file():
        pytest.skip(
            f"no baseline capture at {FIXTURE_TP8}; run: "
            f"MODE=baseline sbatch tests/e2e/scripts/correctness-llama70b-multinode-tp8.sbatch"
        )

    job_id = sbatch_wait(SCRIPTSDIR / "correctness-llama70b-multinode-tp8.sbatch")
    result = _result(job_id, prefix="fast-multinode-tp8")
    assert result["passed"], "\n".join(result["failures"])

# one more test, redundant with the above
@pytest.mark.skip
def test_fast_weight_load_matches_the_lustre_baseline_apertus8b_tp4():
    if not FIXTURE_APERTUS8B_TP4.is_file():
        pytest.skip(
            f"no baseline capture at {FIXTURE_APERTUS8B_TP4}; run: "
            f"MODE=baseline sbatch tests/e2e/scripts/correctness-apertus8b.sbatch"
        )

    result = _result(sbatch_wait(SCRIPTSDIR / "correctness-apertus8b.sbatch"))
    assert result["passed"], "\n".join(result["failures"])

# moe, bf16
def test_fast_weight_load_matches_the_lustre_baseline_qwen3_coder_30b_a3b_tp4():
    if not FIXTURE_QWEN3_CODER_TP4.is_file():
        pytest.skip(
            f"no baseline capture at {FIXTURE_QWEN3_CODER_TP4}; run: "
            f"MODE=baseline sbatch tests/e2e/scripts/correctness-qwen3coder.sbatch"
        )

    result = _result(sbatch_wait(SCRIPTSDIR / "correctness-qwen3coder.sbatch"))
    assert result["passed"], "\n".join(result["failures"])

