#!/usr/bin/env python3
"""Recover the weight_loading window's offset inside a py-spy speedscope trace.

servekit's profile JSON only has per-phase `duration_s` (sequential, no
absolute timestamps beyond the top-level `started_at`/`ready_at`), and py-spy
timestamps are relative to when it attached -- not to servekit's spawn time.
This sums the durations of every phase before weight_loading to get its
start offset from `started_at`, then converts to "seconds since py-spy
attached" using the `pyspy_start_epoch` recorded by the sbatch script, so the
window can be sliced out of the speedscope file for analysis.

Usage: compute_weight_loading_window.py <profile.json> <meta.txt>
"""
import json
import re
import sys


def main() -> int:
    profile_path, meta_path = sys.argv[1], sys.argv[2]

    with open(profile_path) as f:
        report = json.load(f)

    meta_text = open(meta_path).read()
    m = re.search(r"pyspy_start_epoch=([\d.]+)", meta_text)
    if not m:
        print(f"no pyspy_start_epoch found in {meta_path}", file=sys.stderr)
        return 1
    pyspy_start_epoch = float(m.group(1))

    started_at = report["started_at"]
    phases = report["phases"]

    offset = 0.0
    window = None
    for p in phases:
        if p["name"] == "weight_loading":
            window = (offset, offset + p["duration_s"])
            break
        offset += p["duration_s"]

    print("phases (name, duration_s, source):")
    for p in phases:
        print(f"  {p['name']:<20} {p['duration_s']:>8.2f}  {p['source']}")
    print(f"  {'total':<20} {report['total_duration_s']:>8.2f}")

    if window is None:
        print("weight_loading phase not found in profile JSON", file=sys.stderr)
        return 1

    wl_start_offset, wl_end_offset = window
    wl_start_epoch = started_at + wl_start_offset
    wl_end_epoch = started_at + wl_end_offset

    rel_start = wl_start_epoch - pyspy_start_epoch
    rel_end = wl_end_epoch - pyspy_start_epoch

    print()
    print(f"servekit started_at (epoch)     = {started_at:.3f}")
    print(f"py-spy attached at (epoch)      = {pyspy_start_epoch:.3f}")
    print(f"weight_loading offset from start = {wl_start_offset:.2f}s -> {wl_end_offset:.2f}s "
          f"(duration {wl_end_offset - wl_start_offset:.2f}s)")
    print(f"weight_loading window in speedscope timeline (seconds since py-spy attach):")
    print(f"  {rel_start:.2f}s -> {rel_end:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
