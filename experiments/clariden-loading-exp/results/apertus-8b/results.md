# Clariden (GH200) — Apertus-8B, default vs preshard+shm+overlap

**Status:** done — `weight_loading` drops **~90x** (81.4–86.9 s → 0.90–0.93 s),
giving a total cold-start speedup of **1.81–1.85x** (which of the two preshard
runs you pair with the one non-slow default). Much smaller than the 70B's
4.61x, and that is the point: the technique's value scales with how much of cold
start is weight loading.

**Second finding, and the more important one for methodology: Clariden compute
nodes are not interchangeable.** Non-loading phases vary up to 1.7x across
nodes, which is large enough to fake or hide a config effect at n=1. See
"Node heterogeneity" below.

## Goal

The 70B showed 4.61x. Does the technique still pay at 8B, where the checkpoint
is 16 GB across 4 shards rather than 141 GB across 28?

## Method

Same scripts, `MODEL_PRESET=apertus8b`; TP=4, ctx 32768, mem-fraction 0.85, all
held identical to the 70B. `n=3` default, `n=2` preshard, on **5 distinct
nodes**. The presharded checkpoint was generated on Clariden by
`../../save_sharded_ckpt.sbatch` (job 2916681): 15 GB, 4 ranks × 1 part.

| job | config | node |
|---|---|---|
| 2916681 | save_sharded_ckpt (one-off) | nid006632 |
| 2916697 / 2916736 / 2916800 | default | nid007244 / nid007013 / nid006653 |
| 2916725 / 2916766 | preshard+shm+overlap | nid007558 / nid006644 |

`weight_loading` below is the **max over TP ranks** — see the servekit note at
the end.

## Headline — one run per config, both on non-slow nodes

Two slow nodes (nid007244, nid007013) drew the default config and are excluded
here; see "Node heterogeneity". The preshard row is the **slower** of the two
preshard runs, so this is the conservative reading — using the faster one
(93.19 s) gives 1.85x.

| config | node | total | weight_load | non-load | stage (hidden) | tok/s |
|---|---|---|---|---|---|---|
| default | nid006653 | 172.68 s | 81.41 s | 91.27 s | — | 2825.7 |
| preshard+shm+overlap | nid006644 | **95.53 s** | **0.90 s** | 94.63 s | 1.40 s @ 12.68 GB/s | 2816.8 |
| | | **1.81x** | **90x** | — | | — |

`non-load` = total − weight_load. It is flat across configs (91.27 vs 94.63),
which is the check that the technique moved only the phase it targets.

## All runs

| config | node | total | weight_loading | **total − weight_loading** | bench1 |
|---|---|---|---|---|---|
| default | nid007244 | 241.22 | 84.38 | **156.84** | 1869.6 |
| default | nid007013 | 237.37 | 86.88 | **150.49** | 1683.7 |
| default | nid006653 | **172.68** | 81.41 | **91.27** | 2825.7 |
| preshard | nid007558 | 93.19 | 0.93 | **92.26** | 2708.5 |
| preshard | nid006644 | 95.53 | 0.90 | **94.63** | 2816.8 |

Stage (hidden): 1.17 s @ 15.20 GB/s and 1.40 s @ 12.68 GB/s, 4 files × 64 slices
= 244 readers. Overlap gate **VALID** both times, 36.41 s and 38.79 s of slack —
the stage finishes ~1 s into a ~40 s window before the loader opens a file.

**Speedup: 172.68 / 94.36 = 1.83x** against the *mean* of the two preshard runs
(the Headline table above pairs single runs instead, giving 1.81x conservatively
or 1.85x). Using the slow-node defaults would give 2.5x, which would be an
artifact of node assignment, not of the technique.

Correctness: 6/6 greedy probes byte-identical across configs, errors=0, 64/64
completed in every run.

## Node heterogeneity

The decomposition above is what makes this legible. **`total − weight_loading`
is 91.27 s for the *default* config on nid006653 and 92.26 / 94.63 s for the
preshard config** — statistically indistinguishable. Meanwhile the same quantity
is 150–157 s on nid007013 / nid007244. So:

- Everything except `weight_loading` tracks **the node**, not the config.
- `process_startup` was 40.6 / 40.9 s on the slow nodes and 15.0 s on the fast
  one *running the same default config*; graph capture 30.1 / 27.1 vs 19.7 s.
- Serving throughput likewise: a **default** run reached 2825.7 tok/s, matching
  the preshard runs' 2708.5 / 2816.8. There is **no throughput benefit** from
  staging; the earlier apparent 1.6x was slow nodes landing in the default arm.

Mechanism not established. The container image is mounted via `squashfuse_ll`
from the 28 GB `.sqsh` on iopsstor, so a node whose page cache is cold for that
image pays for every Python import — a plausible cause for a 2.7x spread in
`process_startup`, but unverified.

**This does not undermine the 70B result.** There, `total − weight_loading` was
119.52 s (default) vs 120.87 s (preshard) — 1.1% apart — so both its nodes were
comparable and its 4.61x is unaffected.

## Verdict

- **The technique works and is size-dependent, as expected.** It removes
  essentially all of weight loading (81 s → 0.9 s) in both models. What changes
  is how much that is worth: 80% of cold start for the 70B, 47% for Apertus-8B.
- **Forecasting from `weight_loading` share is roughly right** — 1.81–1.85x
  measured against 172.68/91.27 = 1.89x predicted by driving loading to zero.
- **The 4 shards were not a problem.** The sliced stager cuts each file into 64
  ranges, so it reached 12.7–15.2 GB/s from only 4 files. A loader relying on
  file-level parallelism would have suffered here; this one does not.
- **The floor is now startup + capture**, ~91 s, of which graph capture is ~41 s
  and `process_startup` ~15 s. That is where any further work belongs.

## Caveats

- **n=3 / n=2, and node assignment is the dominant nuisance variable.** The
  speedup rests on a single fast-node default run. A same-node paired design
  (default first, then preshard — the stage uses O_DIRECT so it is not helped by
  the first run's page cache) would settle it properly; note the second run does
  get a warm container-image cache, which favours it, so both orders are needed.
- **servekit's first benchmark understates steady state by ~30%.** Three
  back-to-back benches against the same live server (job 2916800) gave
  2825.7 → 3641.1 → 3665.0 tok/s. All `bench1` numbers here are therefore
  first-bench figures and comparable to each other, but not steady state. Use
  `BENCH_REPEATS` in `models.sh` to check.
- The presharded checkpoint here was built **on Clariden**, whereas the 70B
  reused a **bristen**-built one. Not a controlled difference, though the 70B
  already showed the artifacts are architecture-portable.

## servekit correction

`weight_loading` in this file is the **max over TP ranks**. servekit originally
recorded the *first* `Load weight end` line, i.e. the **fastest** rank. That was
fixed in `servekit/src/servekit/profile.py` (phases now stay open and take the
max), with regression tests in `servekit/tests/test_profile.py`. The profile
JSONs in this directory predate the fix and still carry the fastest-rank value:

| run | JSON (fastest rank) | correct (max) |
|---|---|---|
| default 2916697 / 2916736 / 2916800 | 84.24 / 86.81 / 81.36 | 84.38 / 86.88 / 81.41 |
| preshard 2916725 / 2916766 | 0.90 / 0.64 | 0.93 / 0.90 |

Totals are unaffected — they are wall-clock from the wrapper, not a sum of
phases.
