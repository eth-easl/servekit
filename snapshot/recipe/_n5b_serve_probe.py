#!/usr/bin/env python3
# _n5b_serve_probe.py — N5b Task 8: focused serving-overhead probe.
# Sends CONCURRENCY×ROUNDS greedy completions at a fixed concurrency and reports
# throughput (req/s) + p50/p99 latency (ms). Used for the restored-vs-baseline
# serving-overhead check (identical graphs → expect ~equal throughput/latency).
#
# Usage: _n5b_serve_probe.py <url> [concurrency=8] [rounds=4]
import concurrent.futures
import json
import sys
import time
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8830/v1/completions"
CONCURRENCY = int(sys.argv[2]) if len(sys.argv) > 2 else 8
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 4

PROMPTS = [
    "The capital of France is",
    "Count from one to five:",
    "The reverse of the word hello is",
    "2 plus 2 equals",
    "The primary colors are",
]


def one(i):
    p = PROMPTS[i % len(PROMPTS)]
    body = json.dumps(
        {"model": "cs", "prompt": p, "max_tokens": 16, "temperature": 0}
    ).encode()
    t0 = time.perf_counter()
    try:
        urllib.request.urlopen(
            urllib.request.Request(
                URL, data=body, headers={"Content-Type": "application/json"}
            ),
            timeout=120,
        ).read()
        return time.perf_counter() - t0
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    lats = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = [ex.submit(one, i) for i in range(CONCURRENCY * ROUNDS)]
        for f in concurrent.futures.as_completed(futs):
            l = f.result()
            if l is not None:
                lats.append(l)
    lats.sort()
    n = len(lats)
    total = sum(lats)
    throughput = n / total if total > 0 else 0.0
    p50 = lats[int(n * 0.50)] if n else 0.0
    p99 = lats[min(int(n * 0.99), n - 1)] if n else 0.0
    print(
        f"N5B_SERVE_PROBE: n={n} concurrency={CONCURRENCY} "
        f"throughput_rps={throughput:.1f} p50_ms={p50*1000:.0f} "
        f"p99_ms={p99*1000:.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
