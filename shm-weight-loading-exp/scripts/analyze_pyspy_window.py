#!/usr/bin/env python3
"""Classify where the weight_loading window's time goes, from a py-spy speedscope trace.

Deliberately does NOT slice by timestamp. py-spy's speedscope exporter gives
every sample a fixed weight of 1/rate, so the file's time axis is *sampled*
time, not wall clock: whenever py-spy drops samples (27% of attempts in the
20Hz --nonblocking run) the axis silently compresses and no longer lines up
with servekit's phase offsets. Instead we select samples by stack content --
anything under `load_model` IS the weight_loading window by construction --
which is immune to that drift.

Usage: analyze_pyspy_window.py <pyspy.speedscope> [...]
"""
import collections
import json
import sys


def load(path):
    # --nonblocking reads process memory without pausing, so a few frame-name
    # strings come back torn and non-UTF8 (4 of 3813 frames in job 75707).
    # Replace rather than fail; the damaged frames are not on the hot path.
    return json.loads(open(path, "rb").read().decode("utf-8", "replace"))


def classify(leaf_name, leaf_file):
    if leaf_name == "weight_loader" and "linear.py" in leaf_file:
        return "H2D copy (param_data.copy_)"
    if leaf_name == "default_weight_loader":
        return "H2D copy (param_data.copy_)"
    if "vocab_parallel_embedding" in leaf_file:
        return "H2D copy (embedding weight_loader)"
    if leaf_name == "load_weights" and "llama.py" in leaf_file:
        return "weights iterator (tmpfs read + deserialize)"
    return "other / python overhead"


def analyze(path):
    d = load(path)
    frames = d["shared"]["frames"]

    # Root of the loading subtree, in both the model_runner and loader modules.
    load_roots = {
        i for i, f in enumerate(frames)
        if f.get("name") == "load_model"
        and ("loader.py" in f.get("file", "") or "model_runner" in f.get("file", ""))
    }

    total = 0.0
    by_cat = collections.Counter()
    by_proc = collections.Counter()
    by_line = collections.Counter()
    collective = 0.0

    for prof in d["profiles"]:
        proc = prof["name"].split(" Thread ")[0]
        for stack, weight in zip(prof["samples"], prof["weights"]):
            if not (load_roots & set(stack)):
                continue
            total += weight
            by_proc[proc] += weight

            # Was a collective anywhere on the stack? (PLAN hypothesis (c):
            # a cross-rank broadcast redistributing weights.)
            if any(
                "distributed" in frames[i].get("file", "")
                or frames[i].get("name") in ("broadcast", "all_reduce", "all_gather")
                for i in stack
            ):
                collective += weight

            leaf = frames[stack[-1]]
            name, file = leaf.get("name", "?"), leaf.get("file", "?")
            by_cat[classify(name, file)] += weight
            by_line[(name, file.split("/")[-1], leaf.get("line", 0))] += weight

    print(f"=== {path.split('/')[-1]} ===")
    print(f"sampled time inside load_model subtree: {total:.2f}s "
          f"across {len(by_proc)} processes")
    print()
    for proc, w in by_proc.most_common():
        print(f"  {w:7.2f}s  {proc}")
    print()
    for cat, w in by_cat.most_common():
        print(f"  {w:7.2f}s  {100 * w / total:5.1f}%  {cat}")
    print(f"  {collective:7.2f}s  {100 * collective / total:5.1f}%  "
          f"[torch.distributed/NCCL frame anywhere in stack]")
    print()
    print("  top source lines:")
    for (name, file, line), w in by_line.most_common(8):
        print(f"    {w:7.2f}s  {name:<34} {file}:{line}")
    print()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
