"""The whole pipeline in one job: `servekit launch` shards Apertus-8B into an
empty artifact dir, stages and serves that fresh checkpoint, and `servekit
verify` checks it against the recorded baseline.

test_e2e.py runs the same scripts against checkpoints sharded long ago, so it
can only fail on the loading path. Here the checkpoint comes out of the
servekit under test, which puts the prepare step under the same logprob check.

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

TP_PP = 2
PP_PP = 2

# A dump plus a cold start plus the probe, and the queue on top of that.
JOB_TIMEOUT = 3600


def _log(job_id: str, suffix: str, prefix: str = "fast") -> dict:
    path = SCRIPTSDIR / "logs" / f"{prefix}-{job_id}{suffix}"
    assert path.is_file(), f"the job wrote no {suffix} output at {path}"
    return json.loads(path.read_text())


def _assert_served_and_benched(report: dict) -> None:
    assert report["success"], "the server never reported ready"
    bench = report["benchmark"]
    assert not bench["errors"], f"bench-level errors: {bench['errors']}"
    assert bench["throughput"]["errors"] == 0, f"{bench['throughput']['errors']} failed requests"


def test_freshly_prepared_apertus8b_serves_the_lustre_baseline():
    if not FIXTURE.is_file():
        pytest.skip(
            f"no baseline capture at {FIXTURE}; "
            f"run: MODE=baseline sbatch tests/e2e/scripts/apertus8b.sbatch"
        )

    job_id = sbatch_wait(SCRIPTSDIR / "apertus8b.sbatch", timeout=JOB_TIMEOUT, env={"FRESH": "1"})

    manifest = _log(job_id, ".manifest.json")
    assert manifest["format"] == "sharded_state"
    assert manifest["tp_size"] == TP, f"prepared for tp={manifest['tp_size']}, asked for {TP}"
    assert manifest["source"] == MODEL

    caches = _log(job_id, ".caches.json")
    for name in ("triton", "tvm-ffi"):
        assert caches[name]["bytes"] > 1024, f"{name} cache is trivially small: {caches}"

    _assert_served_and_benched(_log(job_id, ".report.json"))

    result = _log(job_id, ".json")
    assert result["passed"], "\n".join(result["failures"])


def test_freshly_prepared_apertus8b_pp_serves_its_own_baseline():
    job_id = sbatch_wait(SCRIPTSDIR / "prepare-serve-apertus8b-pp.sbatch", timeout=JOB_TIMEOUT)
    log = lambda suffix: _log(job_id, suffix, prefix="prepared-pp")

    manifest = log(".manifest.json")
    assert manifest["format"] == "sharded_state"
    assert manifest["tp_size"] == TP_PP, f"prepared for tp={manifest['tp_size']}"
    assert manifest["pp_size"] == PP_PP, f"prepared for pp={manifest['pp_size']}"
    assert manifest["source"] == MODEL

    # The shim's whole job: one distinct shard per (pp, tp) rank rather than two
    # stages colliding on model-rank-{tp}-part-*, and no marker left behind.
    shards = log(".shards.json")
    for pp in range(PP_PP):
        for tp in range(TP_PP):
            prefix = f"model-pp-{pp}-rank-{tp}-part-"
            assert any(s.startswith(prefix) for s in shards["shards"]), (
                f"no shard for pp={pp} tp={tp} in {shards['shards']}"
            )
    assert not shards["markers"], f"leftover done-markers: {shards['markers']}"

    _assert_served_and_benched(log(".report.json"))

    result = log(".json")
    assert result["passed"], "\n".join(result["failures"])
