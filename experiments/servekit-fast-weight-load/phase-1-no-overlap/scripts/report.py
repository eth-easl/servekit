"""Read the launch report, the memory series and the second bench into one verdict.

Baselines are experiments/clariden-loading-exp's, not re-measured here.
"""
import json
import sys

# (default total, default weight_loading, overlapped total) from
# experiments/clariden-loading-exp/SUMMARY.md, SGLang, TP=4, fresh nodes.
BASELINE = {
    "llama70b": (586.33, 466.81, 127.06),
    "apertus8b": (172.68, 81.41, 95.53),
}

launch_json, mem_log, bench2_json, preset = sys.argv[1:5]
d = json.load(open(launch_json))
by = {p["name"]: p["duration_s"] for p in d["phases"]}

print("  stage              = %7.2f s" % by.get("stage", float("nan")))
print("  weight_loading     = %7.2f s" % by.get("weight_loading", float("nan")))
print("  TOTAL COLD START   = %7.2f s   (stage included)" % d["total_duration_s"])

if preset in BASELINE:
    default_total, default_wl, overlapped = BASELINE[preset]
    print()
    print("  vs clariden-loading-exp: default %.2f s (wl %.2f), overlapped %.2f s"
          % (default_total, default_wl, overlapped))
    print("  speedup over default = %.2fx" % (default_total / d["total_duration_s"]))
    print("  cost of not overlapping = %+.2f s vs the overlapped run"
          % (d["total_duration_s"] - overlapped))

b = d.get("benchmark", {}).get("throughput", {})
print()
print("  throughput         = %s tok/s, %s/%s ok, errors=%s"
      % (b.get("output_tok_per_s"), b.get("completed"), b.get("requests"), b.get("errors")))
try:
    b2 = json.load(open(bench2_json))["throughput"]
    # Must not DROP. Higher is normal -- the first bench starts the instant the
    # server is ready, with SGLang's lazy init still finishing.
    print("  throughput +60s    = %s tok/s, %s/%s ok, errors=%s   (must not drop: nothing re-read after the free)"
          % (b2.get("output_tok_per_s"), b2.get("completed"), b2.get("requests"), b2.get("errors")))
except OSError:
    print("  throughput +60s    = MISSING")

# Does unlink actually return the RAM? tmpfs pages come back only once nothing
# maps them, which for the mmap loader may be later than the unlink.
print()
print("  --- did the free return the RAM? (MemAvailable GB / shm used GB) ---")
series = []
for line in open(mem_log):
    parts = line.split()
    if len(parts) == 3:
        try:
            # /proc/meminfo and `df -k` report KiB; the stage sizes servekit
            # prints are decimal GB, so convert rather than divide by 1e6.
            series.append((float(parts[0]), int(parts[1]) * 1024 / 1e9, int(parts[2]) * 1024 / 1e9))
        except ValueError:
            pass

if not series:
    print("  no samples")
else:
    ready = d["ready_at"]
    for label, when in (("before stage", d["started_at"]), ("at ready", ready), ("ready+60s", ready + 60)):
        t, mem, shm = min(series, key=lambda s: abs(s[0] - when))
        print("  %-13s  MemAvailable %6.1f GB   /dev/shm used %6.1f GB   (t%+.0fs)"
              % (label, mem, shm, t - d["started_at"]))
    print("  peak shm used %.1f GB, min MemAvailable %.1f GB"
          % (max(s[2] for s in series), min(s[1] for s in series)))
