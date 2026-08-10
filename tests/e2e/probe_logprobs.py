"""Capture logprobs for a served model over multiple fixed prompts. 

Usage:

  python probe_logprobs.py --url http://127.0.0.1:8080 --out capture.json
  python probe_logprobs.py --compare a.json b.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import List, Optional, Tuple


PROMPTS: List[Tuple[str, str]] = [
    ("capital_of_france", "The capital of France is"),
    ("fibonacci_code", "def fibonacci(n):\n    if n < 2:\n        return n\n    return"),
    ("arithmetic", "Question: A train travels 60 km in 1.5 hours. What is its average speed in km/h? Answer:"),
    ("primes", "Here are the first ten prime numbers, in order: 2, 3, 5, 7,"),
    ("thermodynamics", "Summary of the three laws of thermodynamics. The first law states that"),
    ("french", "Bonjour, je m'appelle Marie et je travaille comme ingénieure logicielle à Paris depuis"),
    ("german", "Guten Tag. Die Bundesrepublik Deutschland besteht aus sechzehn Bundesländern, darunter"),
    ("unicode", "数学における素数とは、1 と自分自身以外に正の約数を持たない、1 より大きな自然数のことである。例えば"),
    ("json_schema", '{"model": "llama", "parameters": {"temperature": 0.0, "max_tokens":'),
    ("sql", "SELECT customer_id, SUM(amount) AS total FROM orders WHERE created_at >= '2024-01-01' GROUP BY"),
    ("repetition", "buffalo buffalo buffalo buffalo buffalo buffalo buffalo buffalo buffalo buffalo buffalo buffalo"),
    ("rare_tokens", "Pneumonoultramicroscopicsilicovolcanoconiosis and floccinaucinihilipilification are both"),
    ("dialogue", "Alice: Did you finish the report?\nBob: Almost. I still need to check the numbers in section"),
    ("chess", "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7."),
    ("numbers", "Counting by sevens from zero: 0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91,"),
    # Long enough to reach positions and layers a short prompt never exercises.
    # Corruption confined to one late block shows up here first.
    (
        "long_prose",
        "Distributed inference systems face a tension between memory locality and "
        "parallel throughput that has no clean resolution. " + (
            "A tensor-parallel rank holds only a slice of every weight matrix, so each "
            "forward pass requires an all-reduce across the group, and the latency of "
            "that collective sets a floor on how small a batch can profitably be. "
            "Pipeline parallelism trades that collective for a point-to-point handoff "
            "between stages, which is cheaper per token but introduces bubbles whenever "
            "the stages are unevenly balanced. Practitioners therefore tune the split "
            "empirically, measuring throughput at several configurations rather than "
            "deriving one from first principles, because the interaction between kernel "
            "launch overhead, collective bandwidth, and memory bandwidth is not "
            "separable in any useful analytic form. "
        ) * 6,
    ),
]


GREEDY_PROMPTS = 8
GREEDY_TOKENS = 32


# Not 0. SGLang's OpenAI adapter reads meta_info["input_top_logprobs"] whenever
# echo and logprobs are both set, but only fills it when top_logprobs >= 1, so
# logprobs=0 is a KeyError 500 (seen on v0.5.10).
TOP_LOGPROBS = 1


def _post(base_url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        # The body carries the engine's own traceback, and losing it costs
        # another ten-minute weight load to find out what went wrong.
        raise RuntimeError(f"HTTP {e.code} from {base_url}/v1/completions: {e.read().decode()[:600]}") from e


def discover_model(base_url: str, timeout: float = 30.0) -> str:
    with urllib.request.urlopen(f"{base_url}/v1/models", timeout=timeout) as r:
        data = (json.loads(r.read()) or {}).get("data") or []
    if not data:
        raise RuntimeError(f"{base_url}/v1/models listed no models")
    return data[0]["id"]


def wait_for_ready(base_url: str, timeout_s: float, interval_s: float = 2.0) -> Tuple[float, str]:
    """Poll until the server completes a request. Returns (seconds waited, model id)."""
    t0 = time.time()
    last: Optional[Exception] = None
    while True:
        try:
            model = discover_model(base_url)
            _post(base_url, {"model": model, "prompt": "ping", "temperature": 0.0, "max_tokens": 1}, 60.0)
            return time.time() - t0, model
        except Exception as e:  # noqa: BLE001
            last = e
        if time.time() - t0 >= timeout_s:
            raise TimeoutError(f"{base_url} not serving after {timeout_s:g}s (last error: {last})")
        time.sleep(interval_s)


def score(base_url: str, model: str, prompt: str, timeout: float) -> Tuple[List[float], List[str]]:
    """Per-token logprobs of `prompt` itself, under the model that is serving it.

    `max_tokens: 1` rather than 0 because some SGLang builds reject 0; the
    generated token is sliced back off, so the choice does not reach the capture.
    The leading entry is null (nothing predicts the first token) and goes too.
    """
    resp = _post(
        base_url,
        {"model": model, "prompt": prompt, "max_tokens": 1, "echo": True, "logprobs": TOP_LOGPROBS, "temperature": 0.0},
        timeout,
    )
    lp = (resp.get("choices") or [{}])[0].get("logprobs") or {}
    token_logprobs = lp.get("token_logprobs")
    tokens = lp.get("tokens")
    if not token_logprobs or not tokens:
        raise RuntimeError(
            f"the server returned no prompt logprobs; echo+logprobs is how this probe works. "
            f"Got keys {sorted(lp)} for prompt {prompt[:40]!r}"
        )
    # Drop the generated token, then the leading null.
    prompt_len = len(tokens) - 1
    return [v for v in token_logprobs[:prompt_len] if v is not None], tokens[:prompt_len]


def greedy(base_url: str, model: str, prompt: str, timeout: float) -> List[str]:
    resp = _post(
        base_url,
        {"model": model, "prompt": prompt, "max_tokens": GREEDY_TOKENS, "temperature": 0.0, "logprobs": TOP_LOGPROBS},
        timeout,
    )
    choice = (resp.get("choices") or [{}])[0]
    tokens = ((choice.get("logprobs") or {}).get("tokens")) or []
    if not tokens:
        raise RuntimeError(f"the server returned no generated tokens for prompt {prompt[:40]!r}")
    return tokens


def capture(base_url: str, model: str, timeout: float = 600.0) -> dict:
    out = []
    for i, (key, text) in enumerate(PROMPTS):
        logprobs, tokens = score(base_url, model, text, timeout)
        entry = {
            "key": key,
            "n_tokens": len(tokens),
            "token_logprobs": [round(v, 6) for v in logprobs],
            "mean_nll": round(-sum(logprobs) / len(logprobs), 6) if logprobs else 0.0,
        }
        if i < GREEDY_PROMPTS:
            entry["greedy_tokens"] = greedy(base_url, model, text, timeout)
        print(f"  {key}: {entry['n_tokens']} tokens, mean_nll={entry['mean_nll']}", flush=True)
        out.append(entry)
    return {"model": model, "greedy_tokens": GREEDY_TOKENS, "prompts": out}


def compare(a: dict, b: dict) -> int:
    """Print how far apart two captures are. This is what sets the tolerances.

    Run it on two captures of the SAME load to read off the noise floor; the test
    then asserts a multiple of it.
    """
    by_key_b = {p["key"]: p for p in b["prompts"]}
    worst_token = 0.0
    worst_nll = 0.0
    greedy_bad = 0
    greedy_total = 0
    shape_problems = []

    print(f"{'prompt':<20} {'n_tok':>6} {'max|dlp|':>10} {'|dnll|':>10}  greedy")
    for pa in a["prompts"]:
        pb = by_key_b.get(pa["key"])
        if pb is None:
            shape_problems.append(f"{pa['key']}: absent from the second capture")
            continue
        if pa["n_tokens"] != pb["n_tokens"]:
            shape_problems.append(
                f"{pa['key']}: {pa['n_tokens']} tokens vs {pb['n_tokens']} -- different tokenization, "
                f"nothing below is comparable"
            )
            continue
        dlp = max((abs(x - y) for x, y in zip(pa["token_logprobs"], pb["token_logprobs"])), default=0.0)
        dnll = abs(pa["mean_nll"] - pb["mean_nll"])
        worst_token = max(worst_token, dlp)
        worst_nll = max(worst_nll, dnll)

        note = "-"
        if "greedy_tokens" in pa and "greedy_tokens" in pb:
            ga, gb = pa["greedy_tokens"], pb["greedy_tokens"]
            n = min(len(ga), len(gb))
            same = sum(1 for x, y in zip(ga, gb) if x == y)
            longest = max(len(ga), len(gb))
            greedy_bad += longest - same
            greedy_total += longest
            note = "ok" if ga == gb else f"{same}/{longest}"
            if ga != gb:
                first = next((i for i in range(n) if ga[i] != gb[i]), n)
                note += f" first differs at {first}: {ga[first:first+1]} vs {gb[first:first+1]}"
        print(f"{pa['key']:<20} {pa['n_tokens']:>6} {dlp:>10.6f} {dnll:>10.6f}  {note}")

    for problem in shape_problems:
        print(f"! {problem}")
    print()
    print(f"max |delta token_logprob| : {worst_token:.6f}")
    print(f"max |delta mean_nll|      : {worst_nll:.6f}")
    print(f"greedy token disagreement : {greedy_bad}/{greedy_total}")
    if not shape_problems and greedy_bad == 0:
        print()
        print(f"suggested TOKEN_TOL = {max(worst_token * 10, 1e-3):.4f}")
        print(f"suggested NLL_TOL   = {max(worst_nll * 10, 1e-4):.5f}")
    return 1 if shape_problems else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="probe_logprobs")
    parser.add_argument("--url", help="server base URL, e.g. http://127.0.0.1:8080")
    parser.add_argument("--out", help="where to write the capture")
    parser.add_argument("--wait-ready", type=float, default=0.0, help="seconds to wait for the server to serve")
    parser.add_argument("--timeout", type=float, default=600.0, help="per-request timeout")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"), help="diff two captures and suggest tolerances")
    args = parser.parse_args(argv)

    if args.compare:
        with open(args.compare[0]) as f:
            a = json.load(f)
        with open(args.compare[1]) as f:
            b = json.load(f)
        return compare(a, b)

    if not args.url or not args.out:
        parser.error("--url and --out are required unless --compare is given")

    if args.wait_ready > 0:
        print(f"[PROBE] waiting up to {args.wait_ready:g}s for {args.url} to serve", flush=True)
        waited, model = wait_for_ready(args.url, args.wait_ready)
        print(f"[PROBE] serving after {waited:.1f}s: {model}", flush=True)
    else:
        model = discover_model(args.url)

    print(f"[PROBE] capturing {len(PROMPTS)} prompts from {args.url}", flush=True)
    result = capture(args.url, model, timeout=args.timeout)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[PROBE] capture written to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
