"""Preparing an artifact must refuse the quantization paths a presharded checkpoint
cannot carry, against the real engine rather than a table of flag strings.

gpt-oss-20b with `--moe-runner-backend triton_kernel` is the case from
sgl-project/sglang#34448: the dump silently comes out ~69% short. The unit tests
cover the rule; this covers the wiring -- that servekit reads the checkpoint's
own quantization_config, refuses before the engine loads anything, and leaves
nothing behind for a later `servekit launch` to serve.

    pytest tests/e2e/test_presharding_guard_e2e.py -m e2e
"""

import json

import pytest

from conftest import SCRIPTSDIR, sbatch_wait

pytestmark = pytest.mark.e2e


def test_prepare_refuses_gpt_oss_on_the_triton_kernel_backend():
    # Generous, because the wait is mostly queueing: the job itself only has to
    # get as far as the guard, which runs before the engine.
    job_id = sbatch_wait(SCRIPTSDIR / "presharding-guard-gptoss.sbatch", timeout=3600)
    path = SCRIPTSDIR / "logs" / f"guard-{job_id}.json"
    assert path.is_file(), f"the guard job wrote no result at {path}"
    got = json.loads(path.read_text())

    assert got["rc"] != 0, f"prepare exited 0 on a path that loses weights:\n{got['output']}"
    assert "servekit does not support mxfp4 with --moe-runner-backend triton_kernel" in got["output"], (
        f"prepare failed, but not on the guard -- it has to refuse before the load, "
        f"not crash somewhere later:\n{got['output']}"
    )
    # A refusal that still wrote shards is worse than no refusal: the shards
    # outlive the message and load without complaint.
    assert got["shards"] == 0, f"refused, but left {got['shards']} shards behind"
    assert not got["manifest"], "refused, but left a manifest a later launch would trust"
