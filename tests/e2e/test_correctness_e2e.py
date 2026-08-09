"""The fast path must serve the same model, not just serve something quickly.

Everything else in the suite checks that weights arrived fast and that the bytes
add up. Nothing checks that they are the *right* bytes: the stager's production
gate is a byte sum over files `truncate` already sized before any writer ran, and
the bench's correctness probe captures greedy text without comparing it. Wrong
weights still produce fluent text, so both pass on a broken load.

Per-token logprobs do not. This compares the fast arm against a frozen capture of
a plain HF-off-Lustre load of the same checkpoint -- the reference is the model's
own numbers, and a silently cast dtype, a zero-filled page, or a shard loaded
into the wrong rank moves them.

The fixture is regenerated with examples/correctness/baseline_llama70b.sbatch;
the tolerances below come from running that twice and diffing (see `--compare`
in probe_logprobs.py). Both are recorded in the constants.
"""
import json
from pathlib import Path

import pytest

from conftest import EXAMPLES, sbatch_wait

pytestmark = pytest.mark.e2e

FIXTURE = Path(__file__).parent / "fixtures" / "llama70b-tp4-bf16.json"

# Headroom, not measurement. Two baseline runs on different nodes came out
# bit-identical: max |delta| was 0.000000 over 1178 scored tokens, and none of
# the 256 greedy tokens disagreed. Sequential batch-size-1 requests in a fixed
# order leave nothing for kernel nondeterminism to vary. These bounds are
# therefore slack against a future node or kernel that is less obliging -- a load
# that actually got the weights wrong misses by orders of magnitude more.
TOKEN_TOL = 1e-3
NLL_TOL = 1e-4


def _capture(job_id: str) -> dict:
    path = EXAMPLES / "correctness" / "logs" / f"fast-{job_id}.json"
    assert path.is_file(), f"the fast arm wrote no capture at {path}"
    return json.loads(path.read_text())


def _worst(deltas):
    index, value = max(enumerate(deltas), key=lambda kv: kv[1])
    return index, value


def test_fast_weight_load_matches_the_lustre_baseline():
    if not FIXTURE.is_file():
        pytest.skip(f"no baseline capture at {FIXTURE}; run examples/correctness/baseline_llama70b.sbatch")

    reference = json.loads(FIXTURE.read_text())
    got = _capture(sbatch_wait(EXAMPLES / "correctness" / "fast_llama70b.sbatch"))
    by_key = {p["key"]: p for p in got["prompts"]}

    assert sorted(by_key) == sorted(p["key"] for p in reference["prompts"]), (
        "the fast arm captured a different prompt set than the fixture; "
        "the two runs are not comparable"
    )

    for want in reference["prompts"]:
        key = want["key"]
        mine = by_key[key]

        # A different tokenization means every comparison below is meaningless,
        # so it is its own failure rather than a flood of logprob mismatches.
        assert mine["n_tokens"] == want["n_tokens"], (
            f"{key}: tokenized to {mine['n_tokens']} tokens, the baseline got {want['n_tokens']}"
        )

        if "greedy_tokens" in want:
            assert mine["greedy_tokens"] == want["greedy_tokens"], (
                f"{key}: greedy continuation diverged from the baseline\n"
                f"  baseline: {want['greedy_tokens']}\n"
                f"  fast:     {mine['greedy_tokens']}"
            )

        deltas = [abs(a - b) for a, b in zip(mine["token_logprobs"], want["token_logprobs"])]
        index, value = _worst(deltas)
        # A load that corrupted one block shows up as a run of bad positions, so
        # the count is worth seeing alongside the worst one.
        over = sum(1 for d in deltas if d > TOKEN_TOL)
        assert value <= TOKEN_TOL, (
            f"{key}: token {index} logprob differs by {value:.6f} > {TOKEN_TOL} "
            f"({over}/{len(deltas)} tokens over tolerance)"
        )

        drift = abs(mine["mean_nll"] - want["mean_nll"])
        assert drift <= NLL_TOL, (
            f"{key}: mean NLL {mine['mean_nll']} vs baseline {want['mean_nll']} "
            f"(differs by {drift:.6f} > {NLL_TOL})"
        )
