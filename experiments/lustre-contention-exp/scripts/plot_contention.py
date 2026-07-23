#!/usr/bin/env python3
"""Plot the 24h Lustre contention time series.

Reads every results/contention-<jobid>-<node>.csv (one per 30-min slot, 5 read
repeats each) and renders a bar-per-slot time series of aggregate read bandwidth
(GiB/s): bar height = median of the 5 repeats, whiskers = min..max. Also prints a
text summary table.

Each CSV row: iso_time,run_index,concurrency,total_bytes,wall_s,agg_GiBps

Usage: plot_contention.py [results_dir] [out_png]
  defaults: <exp>/results  and  <results>/contention_timeseries.png
"""
import csv
import glob
import os
import sys
from datetime import datetime
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

BAR = "#3b6fd4"       # one hue, single series (title names it -> no legend)
INK = "#1a1a1a"
MUTED = "#8a8a8a"
GRID = "#e6e6e6"


def parse_node(path):
    # contention-<jobid>-<node>.csv
    base = os.path.basename(path)
    stem = base[len("contention-"):-len(".csv")] if base.startswith("contention-") else base
    parts = stem.split("-")
    return parts[-1] if len(parts) >= 2 else "?"


def load_slot(path):
    """-> dict(time, node, med, lo, hi, n) or None if the file has no data rows."""
    times, gibs = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                gibs.append(float(row["agg_GiBps"]))
                times.append(datetime.fromisoformat(row["iso_time"]))
            except (KeyError, ValueError):
                continue
    if not gibs:
        return None
    # slot wall-clock = middle of the repeats
    times.sort()
    t = times[len(times) // 2]
    return dict(time=t.replace(tzinfo=None), node=parse_node(path),
                med=median(gibs), lo=min(gibs), hi=max(gibs), n=len(gibs))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    results_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, os.pardir, "results")
    results_dir = os.path.abspath(results_dir)
    out_png = sys.argv[2] if len(sys.argv) > 2 else os.path.join(results_dir, "contention_timeseries.png")

    files = sorted(glob.glob(os.path.join(results_dir, "contention-*.csv")))
    slots = [s for s in (load_slot(p) for p in files) if s]
    if not slots:
        print(f"no contention CSVs with data rows in {results_dir}")
        return 1
    slots.sort(key=lambda s: s["time"])

    # text summary
    print(f"{'slot_time':<20}{'node':>11}{'runs':>6}{'median':>9}{'min':>8}{'max':>8}  (GiB/s)")
    print("-" * 70)
    for s in slots:
        print(f"{s['time'].strftime('%Y-%m-%d %H:%M'):<20}{s['node']:>11}{s['n']:>6}"
              f"{s['med']:>9.2f}{s['lo']:>8.2f}{s['hi']:>8.2f}")
    meds = [s["med"] for s in slots]
    print("-" * 70)
    print(f"{len(slots)} slots | median-of-medians {median(meds):.2f} | "
          f"span {min(s['lo'] for s in slots):.2f}..{max(s['hi'] for s in slots):.2f} GiB/s")

    # figure
    xs = [s["time"] for s in slots]
    med = [s["med"] for s in slots]
    lo = [s["med"] - s["lo"] for s in slots]   # asymmetric err: down to min
    hi = [s["hi"] - s["med"] for s in slots]   # up to max

    fig, ax = plt.subplots(figsize=(13, 5.2))
    width = 20.0 / (24 * 60)  # ~20 min, in matplotlib date units (days)
    ax.bar(xs, med, width=width, color=BAR, edgecolor="white", linewidth=0.4,
           zorder=2, yerr=[lo, hi],
           error_kw=dict(ecolor=INK, elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=3))

    ax.set_ylabel("aggregate read bandwidth (GiB/s)", color=INK, fontsize=11)
    ax.set_title("capstor Lustre read bandwidth over 24 h\n"
                 "Llama-3.1-70B (132 GB, 30 shards) · O_DIRECT · 32-worker pool · "
                 "5 reads/slot · fresh node/slot · bar=median, whisker=min..max",
                 fontsize=11, color=INK, loc="left")

    # adaptive ticks: readable whether we have 5 h of partial data or the full 24 h
    loc = mdates.AutoDateLocator(minticks=4, maxticks=12)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
