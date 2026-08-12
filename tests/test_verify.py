"""Tests for `servekit verify` against the stub server from test_bench.py."""
import json
import tempfile
from pathlib import Path

from servekit.cli import main
from servekit.verify import capture, compare

from test_bench import _start_server

REFERENCE = {
    "model": "stub/Model-1B",
    "prompt_set": "builtin-v1",
    "greedy_prompts": 1,
    "greedy_tokens": 3,
    "prompts": [
        {
            "key": "a",
            "text": "hello world",
            "n_tokens": 2,
            "token_logprobs": [-1.0, -2.0],
            "mean_nll": 1.5,
            "greedy_tokens": ["x", "y", "z"],
        },
        {
            "key": "b",
            "text": "another one",
            "n_tokens": 2,
            "token_logprobs": [-0.5, -0.25],
            "mean_nll": 0.375,
        },
    ],
}


def _clone(d):
    return json.loads(json.dumps(d))


def test_compare_clean_capture_passes():
    result = compare(REFERENCE, _clone(REFERENCE))
    assert result.passed
    assert result.worst_token_delta == 0.0
    assert result.worst_nll_delta == 0.0
    assert not result.failures
    assert all(row["status"] == "ok" for row in result.per_prompt)


def test_compare_detects_logprob_drift():
    captured = _clone(REFERENCE)
    captured["prompts"][0]["token_logprobs"][1] = -2.1
    result = compare(REFERENCE, captured, token_tol=1e-6, nll_tol=1e-6)
    assert not result.passed
    assert any("a: token 1" in f for f in result.failures)
    assert result.per_prompt[0]["status"] == "FAIL"
    assert result.per_prompt[1]["status"] == "ok"


def test_compare_detects_token_count_mismatch():
    captured = _clone(REFERENCE)
    captured["prompts"][0]["n_tokens"] = 3
    result = compare(REFERENCE, captured)
    assert not result.passed
    assert any("tokenized to 3 tokens" in f for f in result.failures)
    assert not any("token" in f and "logprob differs" in f for f in result.failures if f.startswith("a:"))


def test_compare_detects_greedy_mismatch():
    captured = _clone(REFERENCE)
    captured["prompts"][0]["greedy_tokens"] = ["x", "Y", "z"]
    result = compare(REFERENCE, captured)
    assert not result.passed
    assert any("greedy continuation diverged" in f for f in result.failures)


def test_compare_model_mismatch_warns_not_fails():
    captured = _clone(REFERENCE)
    captured["model"] = "stub/Different-1B"
    result = compare(REFERENCE, captured)
    assert result.passed
    assert result.model_warning and "Different-1B" in result.model_warning


def test_capture_records_text_and_logprobs():
    srv, url = _start_server()
    try:
        result = capture(url, "stub/Model-1B", [("greet", "hello world")], "builtin-v1")
        entry = result["prompts"][0]
        assert entry["key"] == "greet"
        assert entry["text"] == "hello world"
        assert entry["n_tokens"] == 2
        assert len(entry["token_logprobs"]) == 2
        assert "greedy_tokens" in entry
    finally:
        srv.shutdown()


def test_cli_verify_requires_exactly_one_of_record_or_reference():
    assert main(["verify", "--url", "http://x", "--wait-ready", "0"]) == 2


def test_cli_verify_rejects_both_record_and_reference():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "g.json"
        path.write_text(json.dumps(REFERENCE))
        rc = main(["verify", "--url", "http://x", "--record", str(path), "--reference", str(path)])
        assert rc == 2


def test_cli_verify_rejects_prompts_with_reference():
    with tempfile.TemporaryDirectory() as d:
        ref = Path(d) / "g.json"
        ref.write_text(json.dumps(REFERENCE))
        prompts = Path(d) / "p.json"
        prompts.write_text(json.dumps({"prompt_set": "x", "prompts": []}))
        rc = main(["verify", "--url", "http://x", "--reference", str(ref), "--prompts", str(prompts)])
        assert rc == 2


def test_cli_verify_record_then_check_round_trip():
    srv, url = _start_server()
    try:
        with tempfile.TemporaryDirectory() as d:
            prompts = Path(d) / "prompts.json"
            prompts.write_text(json.dumps({
                "prompt_set": "test-set",
                "prompts": [{"key": "greet", "text": "hello world"}],
            }))
            gold = Path(d) / "gold.json"
            rc = main(["verify", "--url", url, "--record", str(gold), "--prompts", str(prompts)])
            assert rc == 0
            assert json.loads(gold.read_text())["prompts"][0]["key"] == "greet"

            out = Path(d) / "verify.json"
            rc = main(["verify", "--url", url, "--reference", str(gold), "--out", str(out)])
            assert rc == 0
            assert json.loads(out.read_text())["passed"] is True
    finally:
        srv.shutdown()


def test_cli_verify_check_fails_and_still_writes_out():
    srv, url = _start_server()
    try:
        with tempfile.TemporaryDirectory() as d:
            prompts = Path(d) / "prompts.json"
            prompts.write_text(json.dumps({
                "prompt_set": "test-set",
                "prompts": [{"key": "greet", "text": "hello world"}],
            }))
            gold = Path(d) / "gold.json"
            main(["verify", "--url", url, "--record", str(gold), "--prompts", str(prompts)])

            reference = json.loads(gold.read_text())
            reference["prompts"][0]["token_logprobs"][0] -= 1.0
            gold.write_text(json.dumps(reference))

            out = Path(d) / "verify.json"
            rc = main(["verify", "--url", url, "--reference", str(gold), "--out", str(out)])
            assert rc == 1
            result = json.loads(out.read_text())
            assert result["passed"] is False
            assert result["per_prompt"][0]["status"] == "FAIL"
    finally:
        srv.shutdown()
