# Clariden (GH200) loading experiment — summary

**Question:** does the bristen weight-loading speedup reproduce on Clariden
(aarch64 Grace, GH200), and does it still pay for a small model?

**Answer: yes for the 70B (4.61x, better than bristen's 4.4x), and yes but
smaller for 8B (1.83x).** The technique removes essentially all of weight
loading in both cases; what differs is how much of cold start that was.

| model | default | preshard+shm+overlap | speedup | weight_loading share of default |
|---|---|---|---|---|
| Llama-3.1-70B (141 GB, 28 shards) | 586.33 s | **127.13 s** | **4.61x** | 80% |
| Apertus-8B (16 GB, 4 shards) | 172.68 s | **94.36 s** | **1.83x** | 47% |

`weight_loading` itself, max over TP ranks:

| model | default | preshard+shm+overlap | ratio |
|---|---|---|---|
| Llama-3.1-70B | 466.81 s | 6.19 s | 75x |
| Apertus-8B | 81.4 / 84.4 / 86.9 s | 0.90 / 0.93 s | ~92x |

Stage (always hidden inside startup, gate VALID every run): 70B 8.78 s @
17.02 GB/s with 30.75 s slack; Apertus 1.17 / 1.40 s @ 15.20 / 12.68 GB/s with
36–39 s slack.

TP=4, ctx 32768, mem-fraction 0.85, `lmsysorg/sglang:v0.5.10` (same tag as
bristen; arm64 build here). Every run served correctly: 64/64, errors=0, and
6/6 greedy probes byte-identical between configs.

Side by side with bristen (x86, A100, TP=4, Llama-3.1-70B):

| | bristen default | bristen presh+shm+ovl | clariden default | clariden presh+shm+ovl |
|---|---|---|---|---|
| weight_loading | 634 s (n=4) | 9.81 s | 466.81 s | 6.19 s |
| stage | — | 14.24 s @ 10.59 GB/s | — | 8.78 s @ 17.02 GB/s |
| total | 812 s (n=4) | 183.43 s | 586.33 s | 127.13 s |
| speedup | — | 4.4x | — | **4.61x** |

## What this settles

- **The technique is not bristen-specific**, and not x86-specific. The
  "Clariden/GH200 is unmeasured" caveat in
  `docs/packaging-fast-weight-load/PLAN.md` is discharged.
- **Presharded checkpoints are portable across architectures.** The 70B reused a
  checkpoint written by SGLang on x86/A100 unmodified on aarch64/GH200, with
  byte-identical outputs. TP-size- and engine-version-locked, not ISA-locked.
- **Few shards are not a problem.** Apertus-8B has only 4 files; the sliced
  stager cuts each into 64 ranges and still hit 12.7–15.2 GB/s. A loader relying
  on file-level parallelism would suffer here.
- **The payoff scales with weight loading's share of cold start**, and is
  predictable from it (1.83x measured vs 1.89x predicted for Apertus).
- **More cores help the stager.** 288 cores vs bristen's 64 took the 70B stage
  14.24 → 8.78 s (10.59 → 17.02 GB/s).

## What it does not settle

- **Clariden nodes are not interchangeable.** Non-loading phases vary up to 1.7x
  across nodes — `process_startup` 15.0 s vs 40.9 s for the *same* config. Big
  enough to fake a config effect at n=1. The 70B result survives this (its two
  nodes matched to 1.1% on non-loading time) but the Apertus numbers had to be
  read via the load/non-load decomposition to see through it. Details in
  `results/apertus-8b/results.md`.
- **No throughput benefit.** An earlier reading of the Apertus data suggested
  staging raised serving throughput ~1.6x. It does not: a default run reached
  2825.7 tok/s against the preshard runs' 2708.5 / 2816.8. That was node
  variance. servekit's first bench also understates steady state by ~30%
  (2825.7 → 3641.1 → 3665.0 across three back-to-back benches).
- Sample sizes are small (70B n=1 per config; Apertus n=3 / n=2). A same-node
  paired design would be the right next step for anything finer-grained.
- **The 4-socket NUMA question is open, not closed.** The 70B's four ranks
  loaded in 3.72 / 4.06 / 5.81 / 6.19 s — a 1.7x spread, the shape a
  cross-socket asymmetry would produce. Apertus's ranks agree to 0.03 s,
  consistent with a per-byte effect. A `numactl`-pinned stage would test it.

## Next bottleneck

Graph capture, in both models. For the 70B it is
`piecewise_cuda_graph_capture` 46.11 + `cuda_graph_capture` 22.74 = **68.85 s,
54%** of the 127 s. For Apertus it is ~41 s of a ~91 s floor. Weight loading is
now ~5% and ~1% of what remains respectively, and is not worth further work on
this path. See `../graph-compile-cache-exp/`.

## servekit fix landed here

servekit recorded `weight_loading` from the **first** `Load weight end` line —
the **fastest** TP rank. Harmless when loading is slow (ranks converge: 466.20
vs 466.81 s) but badly wrong when it is fast (3.72 s reported against a true
6.19 s, −40%), i.e. biased against exactly the configuration under test. Fixed
in `servekit/src/servekit/profile.py`: TP-parallel phases stay open across ranks
and take the max, with regression tests in `servekit/tests/test_profile.py`.
Profile JSONs in `results/` predate the fix; each `results.md` tabulates the
corrected values.

Per-model detail: `results/llama-3.1-70b/results.md`,
`results/apertus-8b/results.md`. Design and rationale: `PLAN.md`.
