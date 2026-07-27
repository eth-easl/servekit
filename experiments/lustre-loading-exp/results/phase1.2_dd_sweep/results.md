# Phase 1.2 — raw read ceiling (dd O_DIRECT, full-model workload)

**Status:** done — **initial finding was WRONG, corrected in Phase 3**

## Goal

Establish the achievable raw read bandwidth ceiling for the native (unstriped)
layout, independent of the SGLang loader, before touching any engine code.
`fio` is unavailable on these nodes, so `dd iflag=direct` is the tool.

## Method

`scripts/lib/dd_read_sweep.sh`, run via
`scripts/phase1.2_dd_sweep/phase1_dd_sweep.sbatch`. Constant workload = all 30
shards (141 GB); a worker pool of size N drains the shard list
(`xargs -P N`), so `wall_s` is "time to read the whole model at parallelism
N" and every N is comparable (same bytes read).

## Result

**Full curve, job 73586 (nid002313, COMPLETED), N = 1, 4, 8, 16, 32, 64:**

| N | full-model wall_s | agg GB/s |
|---|---|---|
| 1 | 405 | 0.35 |
| 4 | 175 | 0.80 |
| 8 | 81 | **1.75 (apparent peak)** |
| 16 | 128 | 1.10 |
| 32 | 209 | 0.68 |
| 64 | 202 | 0.70 |

An earlier partial run (job 73582) hit 0.91 GB/s @N=8 and 2.42 GB/s @N=16 —
a completely different curve for the same config.

## ⚠️ CORRECTION — this "ceiling" is wrong

The finding as originally recorded here — "capstor peaks around 1.75 GB/s and
collapses with added concurrency" — is **an artifact of one badly-contended
sample**, and it sent Phase 2/3 planning down the wrong path for a while.

Measured later, with a **native-layout dd probe run inside each Phase-3 job**
(same node, minutes after the load, O_DIRECT, N=30), capstor actually
delivers **6.7–8.6 GB/s on the unmodified native layout** — 4–5× the "ceiling"
above. Five independent samples: 6.66, 7.44, 8.55, 6.91, 7.29 GB/s. See
[`../phase3_loader_concurrency/results.md`](../phase3_loader_concurrency/results.md).

## Verdict

**Do not trust this table as "the ceiling."** capstor bandwidth is
contention-dominated and swings 2–6× depending on whole-cluster load at the
moment of the sample; a curve measured once, in one job, at one point in
time, is one contended sample — not a property of the storage.

The N=32/64 rows here are also capped by shard count: one dd worker per shard
means concurrency tops out at N=30 without striping, which is what actually
motivated Phase 2 (see below) — a motivation that later turned out to be
unnecessary once Phase 1.1 established shards already scatter over 24 OSTs.

## Caveats / methodology lesson (load-bearing for every later phase)

**A bandwidth number from a different job at a different time is worthless on
shared storage.** From this phase onward, every job that needs a bandwidth
comparison runs its own dd probe **in the same job**, after the measurement it
is normalizing, so it cannot be warmed by that measurement and cannot be
contaminated by a different job's contention window. This single lesson is
why Phase 3's loader-concurrency result is trustworthy and this phase's
original "ceiling" was not.
