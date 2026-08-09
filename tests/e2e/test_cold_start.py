import json

import pytest

from conftest import EXAMPLES, sbatch_wait

pytestmark = pytest.mark.e2e


def _assert_fast_cold_start(
    report: dict, 
    weight_loading_threshold: float = 10 , 
    total_duration_threshold: float = 150
) -> None:
    phases = {p["name"]: p["duration_s"] for p in report["phases"]}
    bench = report["benchmark"]
    assert report["success"], "server never reported ready"
    # Baselines: ~467-553s weight_loading and ~586-667s total reading the HF
    # checkpoint off Lustre with mmap (experiments/multinode-tp-exp).
    assert phases["weight_loading"] < weight_loading_threshold, f"weight_loading {phases['weight_loading']}s"
    assert report["total_duration_s"] < total_duration_threshold, f"total {report['total_duration_s']}s"
    assert not bench["errors"], f"bench-level errors: {bench['errors']}"
    assert bench["throughput"]["errors"] == 0, f"{bench['throughput']['errors']} failed requests"
    assert all(r["output"] for r in bench["correctness"]["results"]), "empty completion in correctness probe"


def test_fast_weight_load_llama70b():
    script = EXAMPLES / "fast-weight-load" / "run_llama70b_sglang.sbatch"
    job_id = sbatch_wait(script)
    out = script.parent / "logs" / f"fast-weight-load-llama70b-{job_id}.json"
    report = json.loads(out.read_text())
    _assert_fast_cold_start(report)


def test_multinode_llama70b():
    script = EXAMPLES / "multinode" / "run_llama70b_sglang.sbatch"
    job_id = sbatch_wait(script)
    rundir = script.parent / "logs" / f"multinode-tp-llama70b-{job_id}"
    for node_rank in (0, 1):
        report = json.loads((rundir / f"run.node{node_rank}.json").read_text())
        assert report["success"], f"node {node_rank} never reported ready"
    _assert_fast_cold_start(json.loads((rundir / "run.node0.json").read_text()))
