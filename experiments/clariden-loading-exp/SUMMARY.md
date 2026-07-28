# Clariden (GH200) loading experiment — summary

**Question:** does the bristen weight-loading speedup reproduce on Clariden
(aarch64 Grace, GH200), does it still pay for a small model, and does it hold on
vLLM as well as SGLang?

**Answer: yes on all three.** SGLang 70B 4.61x (better than bristen's 4.4x),
SGLang 8B 1.81-1.85x, vLLM 70B 2.56x. The technique removes essentially all of
weight loading every time; what differs is how much of cold start that was.

One run per config, each on a fresh node (see "What it does not settle"):

| engine / model | config | node | total | weight_load | non-load | tok/s |
|---|---|---|---|---|---|---|
| **SGLang, Llama-3.1-70B**<br>141 GB, 28 shards | default | nid007661 | 586.33 s | 466.81 s | 119.52 s | 822.9 |
| | preshard+shm+overlap | nid007585 | **127.06 s** | **6.19 s** | 120.87 s | 797.7 |
| | | | **4.61x** | **75x** | — | — |
| **SGLang, Apertus-8B**<br>16 GB, 4 shards | default | nid006653 | 172.68 s | 81.41 s | 91.27 s | 2825.7 |
| | preshard+shm+overlap | nid006644 | **95.53 s** | **0.90 s** | 94.63 s | 2816.8 |
| | | | **1.81x** | **90x** | — | — |
| **vLLM, Llama-3.1-70B**<br>132 GB, 28 shards | default | nid007424 | 322.01 s | 199.90 s | 122.11 s | 827.3 |
| | preshard+shm+overlap | nid006918 | **125.80 s** | **7.50 s** | 118.22 s | 798.1 |
| | | | **2.56x** | **27x** | — | — |

**The two engines land within 1% of each other once loading is removed** —
125.80 vs 127.06 s — from baselines of 322 and 586. The speedups differ because
the baselines do, and the baselines differ in `weight_loading`, whose spread on
this storage (430–939 s across the bristen repeats) is wider than the gap. So
"vLLM starts faster than SGLang" is not a claim this data supports; "both engines
have a ~120 s floor here and the technique reaches it" is.

`non-load` = total − weight_load, and it is flat across configs in both models
(119.5 vs 120.9; 91.3 vs 94.6) — the check that the technique moved only the
phase it targets. weight_loading was 80% of the 70B's cold start and 47% of
Apertus's, which is the whole reason the speedups differ. The Apertus preshard
row is the slower of its two runs (the faster gives 1.85x).

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
- **The technique is not SGLang-specific either.** vLLM gets the same treatment
  to the same ~120 s floor, 2.56x, on the same node type and model.
- **Presharded checkpoints are portable across architectures, and NOT across
  engines.** The 70B reused a checkpoint written by SGLang on x86/A100 unmodified
  on aarch64/GH200, with byte-identical outputs — but vLLM rejects that same
  checkpoint outright: `Missing keys ('...attn._k_scale', ...)`, 320 attention
  scale params SGLang never writes (job 2918412). So the artifact is locked to
  TP size, engine and engine version, and free of the ISA. Packaging needs one
  checkpoint per engine.
- **The stager's process-per-slice design is a portability hazard.** In NVIDIA's
  NGC image `BASH_ENV=/etc/bash.bashrc`, so each of the stager's 1704 `bash -c`
  slice workers runs `nvidia-smi` at startup: load average 1658, no `dd` ever
  running, a 348 s stage at 0.41 GB/s, and vLLM's own `process_startup` starved
  to 366 s. Fixed at the call site with `env -u BASH_ENV`; stage 348 → 16.89 s.
  A stager that does not fork 3408 processes to move 141 GB would not have been
  exposed to this at all — see `results/vllm/llama-3.1-70b/results.md`.
- **Few shards are not a problem.** Apertus-8B has only 4 files; the sliced
  stager cuts each into 64 ranges and still hit 12.7–15.2 GB/s. A loader relying
  on file-level parallelism would suffer here.
- **The payoff scales with weight loading's share of cold start**, and is
  predictable from it (1.81-1.85x measured vs 1.89x predicted for Apertus).
- **More cores help the stager.** 288 cores vs bristen's 64 took the 70B stage
  14.24 → 8.78 s (10.59 → 17.02 GB/s).

## What it does not settle

- **Clariden nodes are not interchangeable.** Non-loading phases vary up to 1.7x
  across nodes — `process_startup` 15.0 s vs 40.9 s for the *same* config. Big
  enough to fake a config effect at n=1. The 70B result survives this (its two
  nodes matched to 1.1% on non-loading time) but the Apertus numbers had to be
  read via the load/non-load decomposition to see through it. Details in
  `results/sglang/apertus-8b/results.md`.
- **No throughput benefit.** An earlier reading of the Apertus data suggested
  staging raised serving throughput ~1.6x. It does not: a default run reached
  2825.7 tok/s against the preshard runs' 2708.5 / 2816.8. That was node
  variance. servekit's first bench also understates steady state by ~30%
  (2825.7 → 3641.1 → 3665.0 across three back-to-back benches).
- Sample sizes are small (70B n=1 per config, both engines; Apertus n=3 / n=2). A
  same-node paired design would be the right next step for anything
  finer-grained.
- **The two engines are not version-matched.** SGLang 0.5.10 against vLLM
  0.24.0.dev, because that is the only vLLM build that survives CUDA graph
  capture on GH200. Cross-engine numbers compare working builds, not versions.
  vLLM's total also stops at an earlier event (no warmup request; its
  `ready_wait_s` is 0.072 s). The vLLM-vs-vLLM 2.56x is unaffected.
- **vLLM was only tested on the 70B.** The small-model case is SGLang-only.
- **The 4-socket NUMA question is open, not closed.** The 70B's four ranks
  loaded in 3.72 / 4.06 / 5.81 / 6.19 s — a 1.7x spread, the shape a
  cross-socket asymmetry would produce. Apertus's ranks agree to 0.03 s,
  consistent with a per-byte effect. A `numactl`-pinned stage would test it.

## Next bottleneck

Compile and graph capture, in every configuration measured here. SGLang 70B:
`piecewise_cuda_graph_capture` 46.11 + `cuda_graph_capture` 22.74 = **68.85 s,
54%** of the 127 s. vLLM 70B: `torch_compile` 35.45 + `cuda_graph_capture`
8.00 = **43.45 s, 35%** of the 126 s, with `worker_spawn+dist_init` 44.08 s
alongside it — together 70% of what is left. Apertus: ~41 s of a ~91 s floor.
Weight loading is now ~5%, ~6% and ~1% of what remains, and is not worth further
work on this path. See `../graph-compile-cache-exp/`.

Second, for the packaging work rather than the measurement: **replace the
`dd`-per-slice stager**. It forks 3408 short-lived processes to move 141 GB,
which is what exposed it to the `BASH_ENV` trap above, and it runs at 8.62 GB/s
overlapped against 28.22 GB/s standalone. A single process issuing async reads
(io_uring / threads) would do the same work with one process and a controllable
queue depth.

## servekit fix landed here

servekit recorded `weight_loading` from the **first** `Load weight end` line —
the **fastest** TP rank. Harmless when loading is slow (ranks converge: 466.20
vs 466.81 s) but badly wrong when it is fast (3.72 s reported against a true
6.19 s, −40%), i.e. biased against exactly the configuration under test. Fixed
in `servekit/src/servekit/profile.py`: TP-parallel phases stay open across ranks
and take the max, with regression tests in `servekit/tests/test_profile.py`.
Profile JSONs in `results/` predate the fix; each `results.md` tabulates the
corrected values.

Per-run detail: `results/sglang/llama-3.1-70b/results.md`,
`results/sglang/apertus-8b/results.md`,
`results/vllm/llama-3.1-70b/results.md`. Design and rationale: `PLAN.md`.
