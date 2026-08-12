"""The whole pipeline in one job: `servekit prepare` shards Apertus-8B for TP4,
`servekit launch` stages and serves that fresh checkpoint, `servekit verify`
checks it against the recorded baseline.

test_correctness_e2e.py serves checkpoints sharded long ago, so it can only fail
on the loading path. Here the checkpoint comes out of the servekit under test,
which puts `prepare` under the same logprob check.

    pytest tests/e2e/test_prepare_serve_e2e.py -m e2e
"""

import json
from pathlib import Path

import pytest

from conftest import SCRIPTSDIR, sbatch_wait

pytestmark = pytest.mark.e2e

FIXTURE = Path(__file__).parent / "fixtures" / "apertus8b-tp4-bf16.json"
MODEL = "/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-8B-Instruct-2509"
TP = 4

# A dump plus a cold start plus the probe, and the queue on top of that.
JOB_TIMEOUT = 3600


def _log(job_id: str, suffix: str) -> dict:
    path = SCRIPTSDIR / "logs" / f"prepared-{job_id}{suffix}"
    assert path.is_file(), f"the job wrote no {suffix} output at {path}"
    return json.loads(path.read_text())


def test_freshly_prepared_apertus8b_serves_the_lustre_baseline():
    if not FIXTURE.is_file():
        pytest.skip(
            f"no baseline capture at {FIXTURE}; "
            f"run: MODE=baseline sbatch tests/e2e/scripts/correctness-apertus8b.sbatch"
        )

    job_id = sbatch_wait(SCRIPTSDIR / "prepare-serve-apertus8b.sbatch", timeout=JOB_TIMEOUT)

    manifest = _log(job_id, ".manifest.json")
    assert manifest["format"] == "sharded_state"
    assert manifest["tp_size"] == TP, f"prepared for tp={manifest['tp_size']}, asked for {TP}"
    assert manifest["source"] == MODEL

    assert _log(job_id, ".report.json")["success"], "the server never reported ready"

    result = _log(job_id, ".json")
    assert result["passed"], "\n".join(result["failures"])
