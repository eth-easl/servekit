"""Benchmark a live inference server (stdlib only).

Works against any reachable server, whether or not ``servekit profile`` launched
it -- readiness is checked over HTTP (see ``wait_for_ready``) rather than by
parsing a startup log, since e.g. a CRIU-restored process emits none.

Runs two checks once the server answers: correctness (greedy completions on a
fixed prompt set, compared qualitatively rather than hashed since SGLang is not
bit-deterministic across runs) and throughput (a fixed concurrent workload,
which also exercises first-call JIT/lazy-init that a single warmup misses).

Speaks the OpenAI protocol (``GET /v1/models`` + ``POST /v1/completions``), which
both SGLang and vLLM serve, so the same numbers are produced for either engine
and no framework flag is needed. No third-party deps.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from random import Random
from typing import List, Optional

CORRECTNESS_PROMPTS = [
    "The capital of France is",
    "Explain in one sentence why the sky is blue.",
    "List the first 10 prime numbers.",
    "Q: If a train travels 60 km in 1.5 hours, what is its average speed? A:",
    "def fibonacci(n):",
    "The three laws of thermodynamics are:",
]

_VOCAB = (
    "the a of and to in is that it for on with as are was be by this from or an "
    "model server token weight memory kernel graph batch request latency system "
    "compute storage network cluster process thread cache buffer stream loader"
).split()


@dataclass
class BenchConfig:
    requests: int = 100
    input_len: int = 512  # approx prompt length in words (~tokens)
    output_len: int = 128  # max_new_tokens per request
    concurrency: int = 16
    seed: int = 42
    correctness: bool = True
    correctness_max_new_tokens: int = 64
    timeout_s: float = 600.0
    wait_ready_s: float = 0.0  # >0: poll until the server serves (see wait_for_ready)
    model: Optional[str] = None  # None: ask the server (see discover_model)


@dataclass
class BenchReport:
    base_url: str
    model: Optional[str] = None  # what the server said it serves
    correctness: Optional[dict] = None
    throughput: Optional[dict] = None
    ready_wait_s: Optional[float] = None  # seconds spent waiting for the first 200
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def discover_model(base_url: str, timeout: float = 30.0) -> str:
    """Ask the server which model it serves, so the CLI needs no --model flag."""
    req = urllib.request.Request(f"{base_url}/v1/models")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = (json.loads(r.read()) or {}).get("data") or []
    if not data:
        raise RuntimeError(f"{base_url}/v1/models listed no models")
    return data[0]["id"]


def _completions(base_url: str, model: str, prompt: str, params: dict, timeout: float) -> dict:
    body = json.dumps({"model": model, "prompt": prompt, **params}).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/completions", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        # The body carries the engine's own traceback, and losing it costs
        # another ten-minute weight load to find out what went wrong.
        raise RuntimeError(f"HTTP {e.code} from {base_url}/v1/completions: {e.read().decode()[:600]}") from e


def wait_for_ready(
    base_url: str,
    timeout_s: float,
    interval_s: float = 1.0,
    probe_timeout_s: float = 30.0,
) -> tuple:
    """Poll until the server completes a request; return (seconds waited, model id).

    Distinct from the profiler's log-based ready signal: this only cares that
    the server can complete a request, the only signal left after a CRIU
    restore. Raises TimeoutError if `timeout_s` elapses first.
    """
    t0 = time.time()
    last_error: Optional[Exception] = None
    while True:
        try:
            model = discover_model(base_url, probe_timeout_s)
            _completions(base_url, model, "ping", {"temperature": 0.0, "max_tokens": 1}, probe_timeout_s)
            return time.time() - t0, model
        except Exception as e:  # noqa: BLE001
            last_error = e
        if time.time() - t0 >= timeout_s:
            raise TimeoutError(
                f"{base_url} not serving after {timeout_s:g}s (last error: {last_error})"
            )
        time.sleep(interval_s)


def _completion_tokens(resp: dict, fallback: int) -> int:
    usage = resp.get("usage") or {}
    val = usage.get("completion_tokens")
    return int(val) if isinstance(val, (int, float)) and val > 0 else fallback


def _text(resp: dict) -> str:
    choices = resp.get("choices") or [{}]
    return choices[0].get("text", "")


def _percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def run_correctness(base_url: str, cfg: BenchConfig, model: str) -> dict:
    results = []
    for p in CORRECTNESS_PROMPTS:
        resp = _completions(
            base_url,
            model,
            p,
            {"temperature": 0.0, "max_tokens": cfg.correctness_max_new_tokens},
            cfg.timeout_s,
        )
        results.append({"prompt": p, "output": _text(resp)})
    return {
        "max_new_tokens": cfg.correctness_max_new_tokens,
        "results": results,
    }


def _make_prompt(rng: Random, n_words: int) -> str:
    return " ".join(rng.choice(_VOCAB) for _ in range(n_words))


def run_throughput(base_url: str, cfg: BenchConfig, model: str) -> dict:
    rng = Random(cfg.seed)
    prompts = [_make_prompt(rng, cfg.input_len) for _ in range(cfg.requests)]
    # ignore_eos: a non-standard extension both engines accept. Without it
    # output_len is only an upper bound and throughput is not comparable.
    sp = {"temperature": 0.0, "max_tokens": cfg.output_len, "ignore_eos": True}

    latencies: List[float] = []
    out_tokens = 0
    errors = 0

    def one(prompt: str):
        t0 = time.time()
        resp = _completions(base_url, model, prompt, sp, cfg.timeout_s)
        return time.time() - t0, _completion_tokens(resp, cfg.output_len)

    wall0 = time.time()
    with ThreadPoolExecutor(max_workers=cfg.concurrency) as ex:
        for fut in [ex.submit(one, p) for p in prompts]:
            try:
                lat, toks = fut.result()
                latencies.append(lat)
                out_tokens += toks
            except Exception:  # noqa: BLE001
                errors += 1
    wall = time.time() - wall0

    latencies.sort()
    completed = len(latencies)
    return {
        "requests": cfg.requests,
        "completed": completed,
        "errors": errors,
        "concurrency": cfg.concurrency,
        "input_len": cfg.input_len,
        "output_len": cfg.output_len,
        "wall_s": round(wall, 3),
        "output_tokens": out_tokens,
        "output_tok_per_s": round(out_tokens / wall, 1) if wall > 0 else 0.0,
        "requests_per_s": round(completed / wall, 3) if wall > 0 else 0.0,
        "latency_s": {
            "mean": round(sum(latencies) / completed, 3) if completed else 0.0,
            "p50": round(_percentile(latencies, 0.50), 3),
            "p99": round(_percentile(latencies, 0.99), 3),
            "max": round(latencies[-1], 3) if latencies else 0.0,
        },
    }


def run_benchmark(base_url: str, cfg: BenchConfig) -> BenchReport:
    report = BenchReport(base_url=base_url)
    model = cfg.model
    if cfg.wait_ready_s > 0:
        try:
            waited, served = wait_for_ready(base_url, cfg.wait_ready_s)
        except TimeoutError as e:
            report.errors.append(str(e))
            return report
        report.ready_wait_s = round(waited, 3)
        model = model or served
    if model is None:
        try:
            model = discover_model(base_url)
        except Exception as e:  # noqa: BLE001
            report.errors.append(f"model discovery failed: {e}")
            return report
    report.model = model
    if cfg.correctness:
        try:
            report.correctness = run_correctness(base_url, cfg, model)
        except Exception as e:  # noqa: BLE001
            report.errors.append(f"correctness failed: {e}")
    try:
        report.throughput = run_throughput(base_url, cfg, model)
    except Exception as e:  # noqa: BLE001
        report.errors.append(f"throughput failed: {e}")
    return report


def render_bench(report: BenchReport) -> str:
    lines = [f"[SERVEKIT] Benchmark of {report.base_url} (model={report.model})", "-" * 40]
    if report.ready_wait_s is not None:
        lines.append(f"ready_wait: {report.ready_wait_s} s (until the server served a request)")
    if report.correctness:
        res = report.correctness["results"]
        lines.append(f"correctness: {len(res)} greedy prompts captured (compare outputs in report JSON)")
        for r in res[:2]:
            out = " ".join(r["output"].split())
            lines.append(f"  {r['prompt'][:34]!r} -> {out[:56]!r}")
    t = report.throughput
    if t:
        lines.append(
            f"throughput: {t['output_tok_per_s']} tok/s  "
            f"({t['completed']}/{t['requests']} reqs, conc={t['concurrency']}, "
            f"in={t['input_len']} out={t['output_len']})"
        )
        lat = t["latency_s"]
        lines.append(
            f"latency_s: mean={lat['mean']} p50={lat['p50']} p99={lat['p99']} max={lat['max']}"
            f"   errors={t['errors']}"
        )
    for e in report.errors:
        lines.append(f"! {e}")
    return "\n".join(lines)
