"""
Compares logs probs between (no servekit) baseline and serveki) fast weight load.
The logs probs are expected to be equal within a tolerance.

Dev workflow:
(1) ARM=baseline sbatch tests/e2e/scripts/correctness-llama70b.sbatch
(2) cp tests/e2e/scripts/logs/baseline-<id>.json tests/e2e/fixtures/<run-name>.json
(3) pytest tests/e2e/test_correctness_e2e.py -m e2e (runs ARM=fast, compares logprobs)
"""

import json
import os
from pathlib import Path

import pytest

from conftest import SCRIPTSDIR, sbatch_wait

os.environ.setdefault("ARM", "fast")

pytestmark = pytest.mark.e2e

FIXTURE = Path(__file__).parent / "fixtures" / "llama70b-tp4-bf16.json"

# Tight tolerances, because two baselines runs came out exactly equal
TOKEN_TOL = 1e-6
NLL_TOL = 1e-6


def _capture(job_id: str) -> dict:
    path = SCRIPTSDIR / "logs" / f"fast-{job_id}.json"
    assert path.is_file(), f"the fast arm wrote no capture at {path}"
    return json.loads(path.read_text())


def _worst(deltas):
    index, value = max(enumerate(deltas), key=lambda kv: kv[1])
    return index, value


def test_fast_weight_load_matches_the_lustre_baseline():
    if not FIXTURE.is_file():
        pytest.skip(f"no baseline capture at {FIXTURE}; run: ARM=baseline sbatch tests/e2e/scripts/correctness-llama70b.sbatch")

    reference = json.loads(FIXTURE.read_text())
    got = _capture(sbatch_wait(SCRIPTSDIR / "correctness-llama70b.sbatch"))
    by_key = {p["key"]: p for p in got["prompts"]}

    assert sorted(by_key) == sorted(p["key"] for p in reference["prompts"]), (
        "the fast arm captured a different prompt set than the fixture; "
        "the two runs are not comparable"
    )

    for expected in reference["prompts"]:
        key = expected["key"]
        found = by_key[key]

        # A different tokenization means every comparison below is meaningless,
        # so it is its own failure rather than a flood of logprob mismatches.
        assert found["n_tokens"] == expected["n_tokens"], (
            f"{key}: tokenized to {found['n_tokens']} tokens, the baseline got {expected['n_tokens']}"
        )

        if "greedy_tokens" in expected:
            assert found["greedy_tokens"] == expected["greedy_tokens"], (
                f"{key}: greedy continuation diverged from the baseline\n"
                f"  baseline: {expected['greedy_tokens']}\n"
                f"  fast:     {found['greedy_tokens']}"
            )

        deltas = [abs(a - b) for a, b in zip(found["token_logprobs"], expected["token_logprobs"])]
        index, value = _worst(deltas)
        
        over = sum(1 for d in deltas if d > TOKEN_TOL)
        assert value <= TOKEN_TOL, (
            f"{key}: token {index} logprob differs by {value:.6f} > {TOKEN_TOL} "
            f"({over}/{len(deltas)} tokens over tolerance)"
        )

        drift = abs(found["mean_nll"] - expected["mean_nll"])
        assert drift <= NLL_TOL, (
            f"{key}: mean NLL {found['mean_nll']} vs baseline {expected['mean_nll']} "
            f"(differs by {drift:.6f} > {NLL_TOL})"
        )
