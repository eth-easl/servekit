#!/usr/bin/env python3
"""Recover the true max-over-ranks weight_loading time from SGLang logs.

servekit reports the first "Load weight end" line it sees, which is whichever
TP rank happened to finish first. The number that actually gates cold start is
the LAST rank to finish, because no rank starts cuda-graph capture until every
rank has loaded.

Each rank logs both a second-resolution timestamp and a sub-second
`elapsed=`, so for every rank:

    begin_r = end_ts_r - elapsed_r      (sub-second, derived)
    gated   = max(end_ts_r) - min(begin_r)

Timestamps are second-resolution, so `gated` carries up to ~1 s of
quantisation. `max_elapsed` is reported alongside as the skew-free lower
bound; when begin-skew is 0 the two agree to within that quantisation.

Usage: gated_weight_loading.py LOG [LOG ...]
"""

import re
import sys
from datetime import datetime

TS = r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
BEGIN = re.compile(TS + r" TP(\d+)\] Load weight begin")
END = re.compile(TS + r" TP(\d+)\] Load weight end\. elapsed=([\d.]+) s")


def parse(path):
    begins, ends = {}, {}
    with open(path, errors="replace") as fh:
        for line in fh:
            if (m := BEGIN.search(line)) and m.group(2) not in begins:
                begins[m.group(2)] = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            elif m := END.search(line):
                rank = m.group(2)
                if rank not in ends:
                    ends[rank] = (
                        datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"),
                        float(m.group(3)),
                        len(ends),  # file order, to identify what servekit saw first
                    )
    return begins, ends


def main(paths):
    print(f"{'log':<44} {'n':>2} {'servekit':>9} {'gated':>7} {'max_el':>7} {'skew':>5}")
    print("-" * 80)
    for path in paths:
        begins, ends = parse(path)
        if not ends:
            continue
        derived = [t.timestamp() - e for t, e, _ in ends.values()]
        gated = max(t.timestamp() for t, _, _ in ends.values()) - min(derived)
        max_el = max(e for _, e, _ in ends.values())
        # what servekit saw: the first "Load weight end" line in the file
        first = min(ends.values(), key=lambda v: v[2])[1]
        skew = (max(begins.values()) - min(begins.values())).total_seconds() if begins else float("nan")
        name = path.rsplit("/", 1)[-1]
        print(f"{name:<44} {len(ends):>2} {first:>9.2f} {gated:>7.2f} {max_el:>7.2f} {skew:>5.0f}")


if __name__ == "__main__":
    main(sys.argv[1:])
