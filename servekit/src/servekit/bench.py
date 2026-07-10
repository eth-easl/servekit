"""Post-cold-start benchmark for a live inference server (stdlib only).

Two checks, run after the server reports ready:

* correctness - greedy (temperature=0) completions on a fixed prompt set,
  captured verbatim in the report. Different loaders (mmap / no-mmap /
  fastsafetensors / a staged copy) should produce essentially the same, coherent
  text; a loader that corrupted weights would yield garbage or clearly divergent
  output. We do NOT hash-compare: SGLang is not bit-deterministic across runs
  (batched atomic reductions, kernel nondeterminism), so exact equality would
  false-alarm - the outputs are compared qualitatively instead.
* throughput - a fixed concurrent workload (same seed everywhere) to confirm the
  loaded model actually serves at the expected rate, and to exercise the first
  real inference calls (JIT/lazy-init) that a single warmup request misses.

Targets SGLang's / vLLM's native ``POST /generate`` (``{"text", "sampling_params"}``
-> ``{"text", "meta_info": {"completion_tokens": ...}}``). No third-party deps so
servekit stays zero-dependency and engine-agnostic.
"""
from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from random import Random
from typing import List, Optional

# Fixed correctness prompts - stable across runs so the output hash is comparable.
CORRECTNESS_PROMPTS = [
    "The capital of France is",
    "Explain in one sentence why the sky is blue.",
    "List the first 10 prime numbers.",
    "Q: If a train travels 60 km in 1.5 hours, what is its average speed? A:",
    "def fibonacci(n):",
    "The three laws of thermodynamics are:",
]

# Small fixed vocab for synthetic throughput prompts (seeded -> reproducible).
_VOCAB = (
    "the a of and to in is that it for on with as are was be by this from or an "
    "model server token weight memory kernel graph batch request latency system "
    "compute storage network cluster process thread cache buffer stream loader"
).split()


@dataclass
class BenchConfig:
    requests: int = 100
    input_len: int = 512          # approx prompt length in words (~tokens)
    output_len: int = 128         # max_new_tokens per request
    concurrency: int = 16
    seed: int = 42
    correctness: bool = True
    correctness_max_new_tokens: int = 64
    timeout_s: float = 600.0


@dataclass
class BenchReport:
    base_url: str
    correctness: Optional[dict] = None
    throughput: Optional[dict] = None
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _generate(base_url: str, prompt: str, sampling_params: dict, timeout: float) -> dict:
    body = json.dumps({"text": prompt, "sampling_params": sampling_params}).encode()
    req = urllib.request.Request(
        f"{base_url}/generate", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _completion_tokens(resp: dict, fallback: int) -> int:
    meta = resp.get("meta_info") or {}
    val = meta.get("completion_tokens")
    return int(val) if isinstance(val, (int, float)) and val > 0 else fallback


def _percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def run_correctness(base_url: str, cfg: BenchConfig) -> dict:
    results = []
    for p in CORRECTNESS_PROMPTS:
        resp = _generate(
            base_url,
            p,
            {"temperature": 0.0, "max_new_tokens": cfg.correctness_max_new_tokens},
            cfg.timeout_s,
        )
        results.append({"prompt": p, "output": resp.get("text", "")})
    return {
        "max_new_tokens": cfg.correctness_max_new_tokens,
        "results": results,
    }


def _make_prompt(rng: Random, n_words: int) -> str:
    return " ".join(rng.choice(_VOCAB) for _ in range(n_words))


def run_throughput(base_url: str, cfg: BenchConfig) -> dict:
    rng = Random(cfg.seed)
    prompts = [_make_prompt(rng, cfg.input_len) for _ in range(cfg.requests)]
    # temperature=0 + ignore_eos => deterministic, fixed-length decode for a
    # clean, reproducible throughput measurement.
    sp = {"temperature": 0.0, "max_new_tokens": cfg.output_len, "ignore_eos": True}

    latencies: List[float] = []
    out_tokens = 0
    errors = 0

    def one(prompt: str):
        t0 = time.time()
        resp = _generate(base_url, prompt, sp, cfg.timeout_s)
        return time.time() - t0, _completion_tokens(resp, cfg.output_len)

    wall0 = time.time()
    with ThreadPoolExecutor(max_workers=cfg.concurrency) as ex:
        for fut in [ex.submit(one, p) for p in prompts]:
            try:
                lat, toks = fut.result()
                latencies.append(lat)
                out_tokens += toks
            except Exception:  # noqa: BLE001 - count failures, don't abort the run
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
    if cfg.correctness:
        try:
            report.correctness = run_correctness(base_url, cfg)
        except Exception as e:  # noqa: BLE001
            report.errors.append(f"correctness failed: {e}")
    try:
        report.throughput = run_throughput(base_url, cfg)
    except Exception as e:  # noqa: BLE001
        report.errors.append(f"throughput failed: {e}")
    return report


def render_bench(report: BenchReport) -> str:
    lines = ["[SERVEKIT] Post-ready benchmark", "-" * 40]
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
