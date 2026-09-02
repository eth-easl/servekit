"""One job per (model, topology): each covers both cold-start speed under
--overlap and correctness against a Lustre baseline.

Dev workflow:
(1) MODE=baseline sbatch tests/e2e/scripts/<script>
(2) cp tests/e2e/scripts/logs/baseline-<id>.json tests/e2e/fixtures/<run-name>.json
(3) pytest tests/e2e/test_e2e.py -m e2e (runs MODE=fast; the sbatch job itself
    runs `servekit bench` and `servekit verify --reference` against the
    fixture and writes both results)
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

# The fast arm shards the checkpoint itself when the artifact is missing,
# which is a dump on top of the cold start, and the queue on top of that.
JOB_TIMEOUT = 3600


def _log(job_id: str, suffix: str) -> dict:
    path = SCRIPTSDIR / "logs" / f"fast-{job_id}{suffix}"
    assert path.is_file(), f"the fast arm wrote no {suffix} output at {path}"
    return json.loads(path.read_text())


def _assert_fast_cold_start(report: dict, weight_loading_threshold: float, total_duration_threshold: float) -> None:
    phases = {p["name"]: p["duration_s"] for p in report["phases"]}
    assert report["success"], "server never reported ready"
    assert phases["weight_loading"] < weight_loading_threshold, f"weight_loading {phases['weight_loading']}s"
    assert report["total_duration_s"] < total_duration_threshold, f"total {report['total_duration_s']}s"
    bench = report["benchmark"]
    assert not bench["errors"], f"bench-level errors: {bench['errors']}"
    assert bench["throughput"]["errors"] == 0, f"{bench['throughput']['errors']} failed requests"


@dataclass
class Case:
    script: str
    fixture: str
    id: str
    weight_loading_threshold: float = 10.0
    # Baselines: ~467-667s reading the HF checkpoint off Lustre with mmap
    # (experiments/multinode-tp-exp); 150s leaves headroom over the ~95-127s
    # servekit measured there.
    total_duration_threshold: float = 150.0


CASES = [
    Case(script="llama70b.sbatch", fixture="llama70b-tp4-bf16.json", id="llama70b"),
    Case(
        script="llama70b-multinode-tp8.sbatch",
        fixture="llama70b-tp8-bf16.json",
        id="llama70b_multinode_tp8",
    ),
    # moe, bf16
    Case(script="qwen3coder.sbatch", fixture="qwen3-coder-30b-a3b-tp4-bf16.json", id="qwen3_coder_30b_a3b_tp4"),
    # pp multinode, bf16
    Case(
        script="apertus8b-multinode-pp.sbatch",
        fixture="apertus8b-tp4pp2-bf16.json",
        id="apertus8b_tp4pp2_multinode",
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_fast_weight_load(case):
    fixture_path = FIXTURES / case.fixture
    if not fixture_path.is_file():
        pytest.skip(
            f"no baseline capture at {fixture_path}; run: MODE=baseline sbatch tests/e2e/scripts/{case.script}"
        )

    job_id = sbatch_wait(SCRIPTSDIR / case.script, timeout=JOB_TIMEOUT)

    _assert_fast_cold_start(_log(job_id, ".report.json"), case.weight_loading_threshold, case.total_duration_threshold)

    result = _log(job_id, ".json")
    assert result["passed"], "\n".join(result["failures"])
