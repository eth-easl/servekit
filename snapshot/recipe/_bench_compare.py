#!/usr/bin/env python3
"""Compare two benchmaker serving run-bundles (baseline vs restore).

Reads summary.json from each phase (latency, saturation) of a MODE=baseline run
and a MODE=restore run, and prints a side-by-side delta table for the metrics
that verify the snapshot interposer adds no serving overhead:

  latency phase : ttft_s (mean/p50/p99), itl_ms_mean (p50/p99), latency_s p50/p99
  saturation    : throughput_rps, goodput_rps, tokens_per_s, ttft p50/p99, latency p99

Usage:
  _bench_compare.py --base-dir snapshot/bench-serving \
      --baseline-tag baseline --restore-tag restore [--run-prefix 20260624-...]

Expects, under --base-dir:
  <baseline-tag>-<prefix>-lat/summary.json
  <baseline-tag>-<prefix>-sat/summary.json
  <restore-tag>-<prefix>-lat/summary.json
  <restore-tag>-<prefix>-sat/summary.json
"""
import argparse
import glob
import json
import os
import sys


def load_summary(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def find_run(base_dir, tag, phase, prefix):
    """Resolve a run dir. If prefix given, use it; else pick the newest match.
    benchmaker writes summary.json under a nested timestamped subdir, so search
    recursively rather than expecting it at the bundle root."""
    if prefix:
        roots = [os.path.join(base_dir, f"{tag}-{prefix}-{phase}")]
    else:
        roots = sorted(glob.glob(os.path.join(base_dir, f"{tag}-*-{phase}")),
                       reverse=True)
    for root in roots:
        if not os.path.isdir(root):
            continue
        cands = sorted(glob.glob(os.path.join(root, "**", "summary.json"),
                                recursive=True), reverse=True)
        # Prefer the deepest/newest, but also accept root-level.
        cands += [os.path.join(root, "summary.json")]
        for c in cands:
            s = load_summary(c)
            if s:
                return os.path.dirname(c), s
    return None, None


def g(s, *path, default=None):
    cur = s
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def fmt(v, nd=4):
    if v is None:
        return "      n/a"
    if isinstance(v, float):
        return f"{v:>10.{nd}f}"
    return f"{v:>10}"


def pct_delta(a, b):
    if a is None or b is None or a == 0:
        return ""
    d = (b - a) / a * 100.0
    return f"({d:+5.1f}%)"


def section(title, base, rest, rows):
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)
    print(f"  {'metric':<26}{'baseline':>12}{'restore':>12}   {'delta':>10}")
    print("  " + "-" * 74)
    for label, path, nd in rows:
        bv = g(base, *path)
        rv = g(rest, *path)
        print(f"  {label:<26}{fmt(bv,nd)}{fmt(rv,nd)}   {pct_delta(bv, rv):>10}")
    # totals
    tb = g(base, "total_requests")
    tr = g(rest, "total_requests")
    print("  " + "-" * 74)
    print(f"  {'requests':<26}{fmt(tb,0)}{fmt(tr,0)}")
    print(f"  {'success':<26}{fmt(g(base,'success'),0)}{fmt(g(rest,'success'),0)}")
    print(f"  {'failed':<26}{fmt(g(base,'failed'),0)}{fmt(g(rest,'failed'),0)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="snapshot/bench-serving")
    ap.add_argument("--baseline-tag", default="baseline")
    ap.add_argument("--restore-tag", default="restore")
    ap.add_argument("--run-prefix", default=None,
                    help="shared run-id prefix (timestamp). If omitted, newest is used.")
    args = ap.parse_args()

    prefix = args.run_prefix
    print(f"# Serving overhead check: {args.baseline_tag} vs {args.restore_tag}")
    print(f"# base-dir: {args.base_dir}")

    # ---- latency phase ----
    bdir, base = find_run(args.base_dir, args.baseline_tag, "lat", prefix)
    rdir, rest = find_run(args.base_dir, args.restore_tag, "lat", prefix)
    if not base or not rest:
        print(f"\n[!] latency phase missing. base={bdir} rest={rdir}")
    else:
        print(f"# latency runs: base={bdir}\n#              rest={rdir}")
        section(
            "PHASE 1 — latency (closed-loop, single-ish stream) — interposer sensitivity",
            base, rest,
            [
                ("ttft_s mean",     ("workload_metrics", "ttft_s", "mean"), 4),
                ("ttft_s p50",      ("workload_metrics", "ttft_s", "p50"), 4),
                ("ttft_s p99",      ("workload_metrics", "ttft_s", "p99"), 4),
                ("itl_ms_mean p50", ("workload_metrics", "itl_ms_mean", "p50"), 3),
                ("itl_ms_mean p99", ("workload_metrics", "itl_ms_mean", "p99"), 3),
                ("tokens_per_s mean",("workload_metrics","tokens_per_s","mean"), 3),
                ("latency_s p50",   ("latency_s", "p50"), 4),
                ("latency_s p99",   ("latency_s", "p99"), 4),
            ],
        )

    # ---- saturation phase ----
    bdir, base = find_run(args.base_dir, args.baseline_tag, "sat", prefix)
    rdir, rest = find_run(args.base_dir, args.restore_tag, "sat", prefix)
    if not base or not rest:
        print(f"\n[!] saturation phase missing. base={bdir} rest={rdir}")
    else:
        print(f"# saturation runs: base={bdir}\n#                rest={rdir}")
        section(
            "PHASE 2 — saturation (closed-loop, high concurrency) — throughput",
            base, rest,
            [
                ("throughput_rps",  ("throughput_rps",), 2),
                ("goodput_rps",     ("goodput_rps",), 2),
                ("tokens_per_s mean",("workload_metrics","tokens_per_s","mean"), 3),
                ("ttft_s p50",      ("workload_metrics", "ttft_s", "p50"), 4),
                ("ttft_s p99",      ("workload_metrics", "ttft_s", "p99"), 4),
                ("itl_ms_mean p99", ("workload_metrics", "itl_ms_mean", "p99"), 3),
                ("latency_s p50",   ("latency_s", "p50"), 4),
                ("latency_s p99",   ("latency_s", "p99"), 4),
            ],
        )

    print()
    print("# Verdict: |delta| < ~5% across all metrics => interposer is overhead-free.")


if __name__ == "__main__":
    main()
