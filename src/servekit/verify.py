"""Check that a served model produces the same numbers as a recorded reference.

Two modes: `--record` captures per-token logprobs and greedy continuations from
a trusted server into a reference file; `--reference` replays the same prompts
against a later server and compares. The reference is self-contained -- it
carries the prompt texts it was recorded with, so it stays valid even if the
built-in prompt set changes.

Speaks the OpenAI protocol (``GET /v1/models`` + ``POST /v1/completions``), same
as ``bench.py``, and reuses its HTTP helpers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from .bench import _completions, discover_model, wait_for_ready  # noqa: F401

_BUILTIN_PROMPTS = Path(__file__).parent / "_verify" / "prompts.json"

# Not 0. SGLang's OpenAI adapter reads meta_info["input_top_logprobs"] whenever
# echo and logprobs are both set, but only fills it when top_logprobs >= 1, so
# logprobs=0 is a KeyError 500 (seen on v0.5.10).
TOP_LOGPROBS = 1


def load_prompt_set(path: Optional[Path]) -> Tuple[str, List[Tuple[str, str]]]:
    """Return (prompt_set name, [(key, text), ...]) from `path`, or the built-in set."""
    src = path or _BUILTIN_PROMPTS
    data = json.loads(Path(src).read_text())
    prompts = [(p["key"], p["text"]) for p in data["prompts"]]
    return data.get("prompt_set", "custom"), prompts


def score(
    base_url: str, model: str, text: str, timeout: float, seed: Optional[int] = None
) -> Tuple[List[float], List[str]]:
    """Per-token logprobs of `text` itself, under the model that is serving it.

    `max_tokens: 1` rather than 0 because some SGLang builds reject 0; the
    generated token is sliced back off, so the choice does not reach the capture.
    The leading entry is null (nothing predicts the first token) and goes too.

    `seed` pins whatever the engine's `seed` sampling param controls (e.g. MoE
    routing tie-breaks); at temperature=0.0 it does not affect which token is
    picked, only how ties/kernel nondeterminism resolve.
    """
    params = {"max_tokens": 1, "echo": True, "logprobs": TOP_LOGPROBS, "temperature": 0.0}
    if seed is not None:
        params["seed"] = seed
    resp = _completions(base_url, model, text, params, timeout)
    lp = (resp.get("choices") or [{}])[0].get("logprobs") or {}
    token_logprobs = lp.get("token_logprobs")
    tokens = lp.get("tokens")
    if not token_logprobs or not tokens:
        raise RuntimeError(
            f"the server returned no prompt logprobs; echo+logprobs is how verify works. "
            f"Got keys {sorted(lp)} for prompt {text[:40]!r}"
        )
    # Drop the generated token, then the leading null.
    prompt_len = len(tokens) - 1
    logprobs, toks = token_logprobs[:prompt_len], tokens[:prompt_len]
    if logprobs and logprobs[0] is None:
        logprobs, toks = logprobs[1:], toks[1:]
    if any(v is None for v in logprobs):
        raise RuntimeError(f"unexpected None in token_logprobs beyond the leading entry for prompt {text[:40]!r}")
    return logprobs, toks


def greedy(
    base_url: str, model: str, text: str, timeout: float, n_tokens: int, seed: Optional[int] = None
) -> List[str]:
    params = {"max_tokens": n_tokens, "temperature": 0.0, "logprobs": TOP_LOGPROBS}
    if seed is not None:
        params["seed"] = seed
    resp = _completions(base_url, model, text, params, timeout)
    choice = (resp.get("choices") or [{}])[0]
    tokens = ((choice.get("logprobs") or {}).get("tokens")) or []
    if not tokens:
        raise RuntimeError(f"the server returned no generated tokens for prompt {text[:40]!r}")
    return tokens


def capture(
    base_url: str,
    model: str,
    prompts: List[Tuple[str, str]],
    prompt_set: str,
    timeout: float = 600.0,
    greedy_prompts: int = 8,
    greedy_tokens: int = 32,
    seed: Optional[int] = None,
) -> dict:
    out = []
    for i, (key, text) in enumerate(prompts):
        logprobs, tokens = score(base_url, model, text, timeout, seed=seed)
        entry = {
            "key": key,
            "text": text,
            "n_tokens": len(tokens),
            "token_logprobs": [round(v, 6) for v in logprobs],
            "mean_nll": round(-sum(logprobs) / len(logprobs), 6) if logprobs else 0.0,
        }
        if i < greedy_prompts:
            entry["greedy_tokens"] = greedy(base_url, model, text, timeout, greedy_tokens, seed=seed)
        out.append(entry)
    return {
        "model": model,
        "prompt_set": prompt_set,
        "greedy_prompts": greedy_prompts,
        "greedy_tokens": greedy_tokens,
        "seed": seed,
        "prompts": out,
    }


@dataclass
class VerifyResult:
    passed: bool
    per_prompt: List[dict] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    worst_token_delta: float = 0.0
    worst_nll_delta: float = 0.0
    model_warning: Optional[str] = None
    token_tol: float = 1e-6
    nll_tol: float = 1e-6

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "per_prompt": self.per_prompt,
            "failures": self.failures,
            "worst_token_delta": self.worst_token_delta,
            "worst_nll_delta": self.worst_nll_delta,
            "model_warning": self.model_warning,
            "token_tol": self.token_tol,
            "nll_tol": self.nll_tol,
        }


def compare(reference: dict, captured: dict, token_tol: float = 1e-6, nll_tol: float = 1e-6) -> VerifyResult:
    result = VerifyResult(passed=True, token_tol=token_tol, nll_tol=nll_tol)

    ref_model = reference.get("model")
    got_model = captured.get("model")
    if ref_model and got_model and ref_model != got_model:
        result.model_warning = f"served model {got_model!r} differs from the reference's {ref_model!r}"

    by_key = {p["key"]: p for p in captured.get("prompts", [])}

    for expected in reference.get("prompts", []):
        key = expected["key"]
        row = {"key": key, "status": "ok", "n_tokens": expected.get("n_tokens"), "max_token_delta": 0.0, "nll_delta": 0.0, "greedy": "-"}
        found = by_key.get(key)
        if found is None:
            row["status"] = "FAIL"
            msg = f"{key}: absent from the captured run"
            result.failures.append(msg)
            row["error"] = msg
            result.per_prompt.append(row)
            result.passed = False
            continue

        if found.get("n_tokens") != expected.get("n_tokens"):
            row["status"] = "FAIL"
            row["n_tokens"] = found.get("n_tokens")
            msg = (
                f"{key}: tokenized to {found.get('n_tokens')} tokens, the reference got "
                f"{expected.get('n_tokens')} -- nothing below is comparable"
            )
            result.failures.append(msg)
            row["error"] = msg
            result.per_prompt.append(row)
            result.passed = False
            continue

        deltas = [abs(a - b) for a, b in zip(found["token_logprobs"], expected["token_logprobs"])]
        max_delta = max(deltas) if deltas else 0.0
        worst_idx = deltas.index(max_delta) if deltas else -1
        nll_delta = abs(found.get("mean_nll", 0.0) - expected.get("mean_nll", 0.0))

        row["max_token_delta"] = max_delta
        row["nll_delta"] = nll_delta
        result.worst_token_delta = max(result.worst_token_delta, max_delta)
        result.worst_nll_delta = max(result.worst_nll_delta, nll_delta)

        prompt_failed = False
        if max_delta > token_tol:
            over = sum(1 for d in deltas if d > token_tol)
            msg = (
                f"{key}: token {worst_idx} logprob differs by {max_delta:.6f} > {token_tol} "
                f"({over}/{len(deltas)} tokens over tolerance)"
            )
            result.failures.append(msg)
            prompt_failed = True
        if nll_delta > nll_tol:
            msg = f"{key}: mean NLL {found.get('mean_nll')} vs reference {expected.get('mean_nll')} (differs by {nll_delta:.6f} > {nll_tol})"
            result.failures.append(msg)
            prompt_failed = True

        if "greedy_tokens" in expected:
            ga, gb = expected["greedy_tokens"], found.get("greedy_tokens")
            if gb is None:
                row["greedy"] = "missing"
                msg = f"{key}: reference has greedy_tokens but the captured run does not"
                result.failures.append(msg)
                prompt_failed = True
            elif ga != gb:
                n = min(len(ga), len(gb))
                same = sum(1 for x, y in zip(ga, gb) if x == y)
                longest = max(len(ga), len(gb))
                row["greedy"] = f"{same}/{longest}"
                first = next((i for i in range(n) if ga[i] != gb[i]), n)
                msg = (
                    f"{key}: greedy continuation diverged from the reference at token {first} "
                    f"({same}/{longest} match): {ga[first:first + 1]} vs {gb[first:first + 1]}"
                )
                result.failures.append(msg)
                prompt_failed = True
            else:
                row["greedy"] = "ok"

        row["status"] = "FAIL" if prompt_failed else "ok"
        if prompt_failed:
            result.passed = False
        result.per_prompt.append(row)

    return result


def render_verify(url: str, result: VerifyResult) -> str:
    lines = [f"[SERVEKIT] Verify of {url}", "-" * 40]
    if result.model_warning:
        lines.append(f"! {result.model_warning}")
    lines.append(f"{'prompt':<20} {'n_tok':>6} {'max|dlp|':>10} {'|dnll|':>10}  {'greedy':>8}  status")
    for row in result.per_prompt:
        n_tok = row.get("n_tokens")
        n_tok_s = str(n_tok) if n_tok is not None else "?"
        lines.append(
            f"{row['key']:<20} {n_tok_s:>6} {row['max_token_delta']:>10.6f} {row['nll_delta']:>10.6f}  "
            f"{row['greedy']:>8}  {row['status']}"
        )
    lines.append("")
    lines.append(f"max |delta token_logprob| : {result.worst_token_delta:.6f}")
    lines.append(f"max |delta mean_nll|      : {result.worst_nll_delta:.6f}")
    bad = sum(1 for r in result.per_prompt if r["status"] != "ok")
    lines.append(
        f"{bad}/{len(result.per_prompt)} prompts over tolerance "
        f"(--token-tol {result.token_tol:g}, --nll-tol {result.nll_tol:g})"
    )
    for f in result.failures:
        lines.append(f"! {f}")
    return "\n".join(lines)
