#!/usr/bin/env python3
"""Summarize servekit profile(+bench) JSON reports into one comparable view.

Emits:
  1. a comparison table (loader, node, key phases, throughput, latency, errors)
  2. the full phase breakdown per run
  3. the post-ready benchmark detail per run (throughput + latency)
  4. a cross-run correctness check: for each probe prompt, do all runs produce
     the same greedy output? (loaders must load equivalent weights)

Usage: summarize_e2e.py [results_dir] [glob]
"""
import glob
import json
import os
import re
import sys
from collections import OrderedDict

KEY_PHASES = ("weight_loading", "cuda_graph_capture", "piecewise_cuda_graph_capture")


def parse_name(path):
    """-> (loader, job, node) from ...-<loader>-<job>-<nid>-profile.json"""
    base = os.path.basename(path)
    m = re.search(r"-(\d+)-(nid\d+)-profile\.json$", base)
    job, node = m.groups() if m else ("?", "?")
    head = base.split(f"-{job}-")[0] if job != "?" else base
    loader = head.rsplit("-", 1)[-1]
    return loader, job, node


def load(path):
    d = json.load(open(path))
    d["_phases"] = {p["name"]: p["duration_s"] for p in d["phases"]}
    d["_loader"], d["_job"], d["_node"] = parse_name(path)
    d["_file"] = os.path.basename(path)
    b = d.get("benchmark") or {}
    d["_thr"] = b.get("throughput") or {}
    d["_corr"] = b.get("correctness") or {}
    d["_bench_errors"] = b.get("errors") or []
    return d


def main():
    # scripts/analysis/summarize_e2e.py -> ../../results
    results_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "results")
    pattern = sys.argv[2] if len(sys.argv) > 2 else "phase1.3_e2e/**/phase1_3_e2e-*.json"

    files = sorted(glob.glob(os.path.join(results_dir, pattern), recursive=True))
    if not files:
        print(f"no reports matching {pattern} in {results_dir}")
        return 1
    runs = [load(f) for f in files]
    runs.sort(key=lambda d: d["_phases"].get("weight_loading", 0))

    # 1) comparison table
    print("=== comparison (sorted by weight_loading) ===")
    hdr = (f"{'loader':<17}{'job':>7}{'node':>11}{'weight_s':>10}{'total_s':>9}"
           f"{'graph_s':>9}{'tok/s':>9}{'p50_s':>8}{'reqs':>9}{'errs':>6}")
    print(hdr)
    print("-" * len(hdr))
    for d in runs:
        ph, t = d["_phases"], d["_thr"]
        graph = ph.get("cuda_graph_capture", 0) + ph.get("piecewise_cuda_graph_capture", 0)
        reqs = f"{t.get('completed','-')}/{t.get('requests','-')}" if t else "-"
        lat = (t.get("latency_s") or {}).get("p50", 0) if t else 0
        print(f"{d['_loader']:<17}{d['_job']:>7}{d['_node']:>11}"
              f"{ph.get('weight_loading',0):>10.1f}{d['total_duration_s']:>9.1f}{graph:>9.1f}"
              f"{t.get('output_tok_per_s',0):>9.1f}{lat:>8.2f}{reqs:>9}{t.get('errors','-'):>6}")

    # 2) full phase breakdown
    print("\n=== full phase breakdown ===")
    for d in runs:
        status = "ok" if d["success"] else "FAILED"
        print(f"\n### {d['_file']}  (total={d['total_duration_s']}s, {status})")
        for p in d["phases"]:
            print(f"    {p['name']:32s} {p['duration_s']:8.2f}  ({p['source']})")

    # 3) benchmark detail
    print("\n=== post-ready benchmark ===")
    for d in runs:
        t = d["_thr"]
        print(f"\n### {d['_loader']} ({d['_node']})")
        if not t:
            print("    (no benchmark - run without --bench)")
            continue
        lat = t.get("latency_s") or {}
        print(f"    requests        {t.get('completed')}/{t.get('requests')}  "
              f"conc={t.get('concurrency')} in={t.get('input_len')} out={t.get('output_len')}")
        print(f"    wall_s          {t.get('wall_s')}")
        print(f"    output_tokens   {t.get('output_tokens')}")
        print(f"    output_tok/s    {t.get('output_tok_per_s')}")
        print(f"    requests/s      {t.get('requests_per_s')}")
        print(f"    latency_s       mean={lat.get('mean')} p50={lat.get('p50')} "
              f"p99={lat.get('p99')} max={lat.get('max')}")
        print(f"    errors          {t.get('errors')}")
        if d["_bench_errors"]:
            print(f"    !! bench errors: {d['_bench_errors']}")

    # 4) cross-run correctness: same prompt -> same greedy output across loaders?
    print("\n=== correctness cross-check (greedy outputs across loaders) ===")
    with_corr = [d for d in runs if d["_corr"].get("results")]
    if len(with_corr) < 2:
        print("    need >=2 runs with --bench correctness to compare")
    else:
        by_prompt = OrderedDict()
        for d in with_corr:
            for r in d["_corr"]["results"]:
                by_prompt.setdefault(r["prompt"], {})[d["_loader"]] = r["output"]
        n_agree = 0
        for prompt, outs in by_prompt.items():
            uniq = set(outs.values())
            agree = len(uniq) == 1
            n_agree += agree
            mark = "AGREE " if agree else "DIVERGE"
            print(f"\n  [{mark}] {prompt!r}")
            if agree:
                sample = " ".join(next(iter(uniq)).split())[:88]
                print(f"      all {len(outs)} loaders -> {sample!r}")
            else:
                for loader, out in outs.items():
                    print(f"      {loader:<16} -> {' '.join(out.split())[:78]!r}")
        print(f"\n  {n_agree}/{len(by_prompt)} prompts identical across "
              f"{len(with_corr)} loaders ({', '.join(d['_loader'] for d in with_corr)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
