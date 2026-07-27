# Idea 0 — what import time is actually made of

`sglang.srt.server_args` imported under `-X importtime`, three times per
container (A cold / B warm+pyc / C warm+no-pyc), one cold node per job.
Script: [`../../scripts/idea0_split/split_imports.sbatch`](../../scripts/idea0_split/split_imports.sbatch).
Jobs 76482 / 76483 / 76484, nodes nid002293 / nid002301 / nid002312, 2026-07-27.

## Answer

**Cold I/O is ~65-70% of import time. Compile is ~1%. There is a ~9 s exec floor.**

| job / node | A cold | B warm+pyc | C warm, no pyc | **compile** `C-B` | **cold I/O** `A-C` | **exec** `B` |
|---|---:|---:|---:|---:|---:|---:|
| 76482 nid002293 | 30.06 | 9.01 | 9.04 | 0.03 | **21.02** | 9.01 |
| 76483 nid002301 | 25.07 | 8.95 | 9.36 | 0.41 | **15.72** | 8.95 |
| 76484 nid002312 | 26.21 | 8.90 | 8.99 | 0.08 | **17.22** | 8.90 |

The exec floor is remarkably stable (8.90-9.01 s, ±0.6%). Cold I/O carries all
the variance (15.7-21.0 s) — expected, it is Lustre under whatever load the
filesystem happened to be in.

Per-package, cold vs warm self time (job 76484) shows where the I/O lands — it
is not concentrated in sglang at all:

| package | A cold | B warm | I/O share |
|---|---:|---:|---:|
| torch | 5.69 | 1.79 | 3.90 |
| transformers | 5.41 | 1.28 | 4.13 |
| sglang | 2.27 | 0.62 | 1.65 |
| sgl_kernel | 1.03 | 0.84 | 0.19 |
| torchcodec | 1.03 | 0.15 | 0.88 |
| *total (4898 modules)* | *23.92* | *7.50* | *16.42* |

## Verdicts

- **Idea 2 (precompile `.pyc`) — close it.** Compile is 0.03-0.41 s. The image
  audit was right that sglang ships 1663 `.py` and 0 `.pyc`, but that framing
  overestimated the cost: this import touches only **117** of those modules, and
  compiling them costs milliseconds. Even scaling to a full server launch
  (~2x the sglang self-time seen here) this stays well under a second.
  `PYTHONPYCACHEPREFIX` + `compileall` would be free and harmless, but it is not
  worth an experiment — it is not where the time goes.
- **Idea 3 (I/O) — pursue, it is the whole story.** 16-21 s of a 25-30 s cold
  import is reading files off the 30 GB squashfs on Lustre with a cold page
  cache, spread across torch/transformers/dist-info rather than concentrated
  anywhere patchable. This is the same QD1-small-read pathology already priced
  in `lustre-loading-exp` at 0.74-0.77 GB/s vs 19-20 GB/s with fan-out, and it
  is addressable with machinery already built here, at zero correctness risk.
- **Idea 1 (spawn→fork) — still open, and comparable in size to idea 3.** Fork
  would let workers inherit exec *and* I/O — but the workers' I/O is already
  free (see below), so its prize is the workers' ~10 s of warm-but-repeated
  import. That is the same order as the I/O win, and it becomes the dominant
  remaining cost once idea 3 lands. Still gated on the CUDA-pre-fork check.

## Which phase does each fix actually touch?

`tp_worker_spawn` is a *gap*, not an import measurement: `servekit/src/servekit/profile.py:42-49`
defines it as the interval between "Using default HuggingFace chat template"
and the first `TP<n>]` line — 4x spawn and interpreter re-exec, the worker
imports, per-rank CUDA context, and NCCL all fall inside it.

Splitting the older apertus-8B `-X importtime` log by process (the header line
is emitted once per process) shows the workers are **not** paying cold I/O.
Per-module cost is used because interleaving makes per-process attribution
unreliable:

| | modules | self time | ms/module |
|---|---:|---:|---:|
| main process, real launch | 6311 | 26.78 | **4.24** |
| all later (worker) processes | 34518 | 61.65 | **1.79** |
| this experiment, A cold | 4898 | 23.92 | **4.88** |
| this experiment, B warm | 4898 | 7.50 | **1.53** |

The main process imports at cold rates; the workers, same node same run, import
at ~1.2x the warm floor. **The ~20 s of cold I/O is paid once, in
`process_startup`, and never again.**

So on the 70B TP4 baseline (`process_startup` 24.24 s, `tp_worker_spawn`
17.49 s = 41.7 s):

- idea 3 takes `process_startup` toward the ~9 s exec floor and leaves
  `tp_worker_spawn` essentially untouched → ~41.7 s becomes ~26.5 s, a **~15 s
  win in one phase**, not 20 s across both.
- idea 1 then targets what remains, since after idea 3 `tp_worker_spawn` is the
  larger of the two.

## Caveats

- A also *writes* the 117 `.pyc` it compiles, so that write sits inside the
  cold-I/O bucket. Bounded by the compile number itself — negligible.
- `import sglang.srt.server_args` is a proxy for `process_startup`, not the
  phase itself: 4898 modules, 23.9 s of cold self-time against the ~21 s wall
  the real phase shows. It under-covers sglang specifically (2.27 s here vs
  4.4 s measured in-launch), which is exactly the package the compile verdict
  concerns — hence the scaling argument above rather than a flat claim.
- Cold-node discipline held: `pyc before anything: 0` and A >> B in all three
  jobs, so no node was pre-warmed.
