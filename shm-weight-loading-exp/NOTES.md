# shm-weight-loading-exp — experiment log

See `PLAN.md` for context and methodology.

## Phase 1-2 — where the weight_loading window actually goes

**Answer: ~88-95% of it is one line, `param_data.copy_(loaded_weight)` — a
synchronous pageable host-to-device copy. The cross-rank-broadcast hypothesis
is dead (0.0%), and reading the bytes back out of tmpfs is nearly free.**

### Runs

| job | node | rate | weight_loading | notes |
|-----|------|------|----------------|-------|
| 75699 | nid002296 | 100 Hz | 26.99 s | clean UTF-8; total cold start 270.15 s |
| 75707 | nid002313 | 20 Hz, `--nonblocking` | 19.86 s | total cold start 190.96 s |

Both are the sliced-`/dev/shm` launch vehicle (stage 12.1-12.2 s @ ~11.6 GB/s,
native source, mmap loader) from `lustre-loading-exp/scripts/phase4_shm/`.
Both reached ready and benched clean (401.4 tok/s, 0 errors), so the
instrumentation did not perturb correctness.

### Breakdown (`scripts/analyze_pyspy_window.py`)

Two independent runs, different nodes and sample rates, agree:

| category | 75707 | 75699 |
|----------|-------|-------|
| **H2D copy** (`param_data.copy_`) | **87.5%** | **94.8%** |
| weights iterator (tmpfs read + deserialize) | 9.8% | 0.8% |
| embedding `weight_loader` (also a `copy_`) | 1.1% | 1.5% |
| other / Python overhead | 1.6% | 2.8% |
| **torch.distributed / NCCL, anywhere in stack** | **0.0%** | **0.0%** |

Time is split evenly across the 4 TP ranks (17.8-18.0 s each in 75707), and
the sampled time inside `load_model` (~18 s) closely tracks the engine's
reported 19.86 s, so coverage of the window is ~91% — the classification is
not built on a thin slice.

The top source lines are all the *same statement* in different `weight_loader`
overloads:

- `layers/linear.py:1460` — `param_data.copy_(loaded_weight)` (47.25 s)
- `layers/linear.py:714`  — `param_data.copy_(loaded_weight)` (14.00 s)
- `layers/linear.py:1309` — `param_data.copy_(loaded_weight)` (1.25 s)
- `models/llama.py:620`   — `for name, loaded_weight in weights:` (6.90 s), i.e.
  pulling the next tensor from the safetensors iterator

### The copies are not equally expensive: strided slices cost ~5x per byte

Splitting the hot lines by which `weight_loader` they belong to, against each
one's share of the model's bytes (computed from `config.json`: hidden 8192,
intermediate 28672, 80 layers, 8 KV heads):

| loader (hot line) | share of bytes | share of time | time/byte |
|---|---|---|---|
| `RowParallelLinear` — `linear.py:1460` (o_proj, down_proj) | 35.3% | **65.9%** | **1.87** |
| `MergedColumnParallelLinear` — `linear.py:714` (gate_up) | 54.9% | 19.5% | 0.36 |
| `QKVParallelLinear` — `linear.py:1309` | 9.8% | 1.7% | 0.17 |

**Row-parallel slices cost ~5.3x more per byte than column-parallel ones.**
The cause is the narrow axis, and it is visible in the source:

- `RowParallelLinear` narrows on **`input_dim`** (`linear.py:1441-1449`), so
  rank-local `down_proj` is `[8192, 7168]` carved out of `[8192, 28672]` —
  **8192 separate ~14 KB fragments** at stride 28672. cudaMemcpy sees a
  strided pageable source and degenerates into many small staged transfers.
- `MergedColumnParallelLinear`/`QKVParallelLinear` narrow on **dim 0**
  (`linear.py:688-690`, `1279-1284`), yielding **one contiguous ~117 MB block**.

Same tensor sizes (gate/up/down are all 235M params), same storage, same
pageable memory — only the narrow axis differs. So this is a clean
contiguity effect, not a size effect.

Practical consequence: the fix has **two** independent factors, pinning *and*
contiguity, and the microbenchmark below must vary both to tell them apart.
It also means "batch it into one big copy" is right for the wrong reason —
copies already average ~48 MB (35 GB / 723 tensors), so per-call overhead is
negligible; it is one third of the bytes moving as 14 KB fragments that hurts.

### Why that copy is expensive

In vendored SGLang v0.5.10 (`1519acf37c`, tag `v0.5.10` — identical to the
container image, so the line numbers above are trustworthy):

- `model_loader/weight_utils.py`'s `buffered_multi_thread_safetensors_weights_iterator`
  opens shards with `safetensors.safe_open(..., device="cpu")` → `loaded_weight`
  is a **CPU** tensor backed by tmpfs pages.
- `model_loader/loader.py:679-690` builds the model under `with target_device:`
  (cuda) → `param_data` is a **CUDA** tensor.

So every `copy_` is a **pageable** (non-pinned) H2D transfer. Rough rate: ~35 GB
per rank (141 GB staged / TP=4) in ~17.4 s ≈ **2.0 GB/s per rank, ~8 GB/s
aggregate** — in the range expected for pageable transfers contended across 4
processes, and far below the ~20+ GB/s that pinned memory sustains on PCIe.

That also explains why weight_loading (19-27 s) exceeds the 12.1 s stage: the
stage is a Lustre→RAM copy at 11.6 GB/s, while this is a RAM→GPU copy that the
driver must internally stage through a pinned bounce buffer.

The iterator being nearly free (0.8-9.8%) confirms the tmpfs read and
safetensors deserialize are well overlapped by its ThreadPoolExecutor
prefetch — the bottleneck is downstream of the read, as suspected.

### Limitation — what this does NOT resolve

py-spy is a Python-level sampler, so everything C-level inside `copy_` is
attributed to that one frame. Within the 87-95% we **cannot** separate:

1. soft page faults on first touch of the tmpfs pages,
2. the CPU→pinned bounce-buffer `memcpy`,
3. the actual pinned→GPU DMA.

So PLAN Phase 3's hugepage hypothesis is *not* ruled out — page-fault cost is
hiding inside the copy, not beside it. What is ruled out is it being a
separately visible, separately fixable phase.

## Why no off-the-shelf loader fixes this (already measured)

Every alternative `load_format` optimizes **reads**. Profiling says reads are
0.8-9.8% of this window. So none of them can help here — and the existing
measurements confirm it. The regime flips once staging removes the read
bottleneck:

| `load_format` | from capstor/Lustre (read-bound) | from `/dev/shm` (H2D-bound) |
|---|---|---|
| mmap (default) | 429.73-939.05 s | **19.55 s**  ← best |
| `fastsafetensors` | 58.66-114.03 s  ← best | 25.07 s |
| `--weight-loader-disable-mmap` | 161.92-166.28 s | 103.57 s |
| `runai_streamer` | 84.91-105.97 s | not run |

(Lustre rows: `lustre-loading-exp/results/phase1.3_e2e/`. tmpfs rows:
`lustre-loading-exp/results/phase4_shm/`, jobs 73665/73666/73667.)

**`fastsafetensors` from tmpfs is 28% *worse* than plain mmap** (25.07 s vs
19.55 s), and the log says exactly why —
`lustre-loading-exp/results/phase4_shm/p4shm-fastsafetensors-73667.out:65-68`,
on all 4 ranks:

```
GDS file-handle setup failed (raw_gds_file_handle: cuFileHandleRegister
returned an error = 5030); falling back to the nogds copier
```

cuFile/GDS requires a real block device; **tmpfs is not GDS-capable**. So the
one thing fastsafetensors exists to do — DMA storage→GPU, bypassing the host —
is unavailable, it falls back to the `nogds` host-bounce copier, and it still
pays the cross-rank NCCL redistribution its per-rank file assignment requires.
Read savings it cannot cash in, collectives it cannot avoid.

Corollary: "stage to `/dev/shm`, then use a fast loader" is a dead end by
construction. Staging is what *makes* reads free, and free reads are what make
every read-optimizing loader pointless. The two ideas cancel.

This is what leaves a targeted pinned/contiguous patch as the only lever: no
upstream `load_format` exposes control over *how* the H2D copy is issued.

## Recommendation

Both branches the PLAN pre-committed to are the wrong shape:

- **Broadcast removal (Phase 3B): drop it.** There is no broadcast. 0.0% of
  samples touch `torch.distributed` — each TP rank already reads its own
  weights independently, exactly as hoped.
- **Hugepage tmpfs (Phase 3A): don't lead with it.** It targets only
  component (1) of three, needs a root-level remount that likely isn't
  available in the unprivileged container, and we have no evidence page faults
  dominate the copy.

> **Superseded by Phase 3 below.** The microbenchmark was run (jobs 75718,
> 75726) and the answer is: staging works, but only with `OMP_NUM_THREADS`
> capped — 4 ranks x 64 OMP threads on 64 cores was masking the entire
> effect. Projection ~19.9 s -> ~9.4 s. The 5.3x striding attribution below
> also did not survive: measured striding penalty is ~1.35x solo / ~2.8x
> contended, not 5.3x.

**Lead instead with a 2x2 microbenchmark** — the cheap, decisive next step,
and it tests the actually-actionable fix. Standalone script, no engine
changes: move ~35 GB tmpfs→GPU varying **both** factors the profile
implicates, `{pageable, pinned} x {contiguous, strided}`, with the strided
case using the real `[8192, 7168]`-of-`[8192, 28672]` geometry. Run 1 process
and 4 concurrent to capture PCIe contention. That yields the ceiling *and*
tells us whether pinning, contiguity, or only both together is what pays —
which decides the patch. Add `non_blocking=True` double-buffering as a fifth
cell only if pinned alone already wins.

If the pinned/contiguous cells land near PCIe line rate, the fix is a
loader-side staging buffer; if everything sits at ~2 GB/s per rank, the
ceiling is the host path itself and *then* hugepages/nsys become the next
question.

**The patch, if the benchmark supports it**: a `h2d_copy_(param_data,
loaded_weight)` helper in `weight_utils.py` that, when the source is CPU and
the destination CUDA, stages the already-narrowed slice through a reusable
per-process pinned buffer (~117 MB, sized to the largest slice — trivial) and
issues `non_blocking=True`. Apply at the three hot sites (`linear.py` 1460,
714, 1309 = ~87% of the window); there are ~16 `param_data.copy_(loaded_weight)`
sites in total if it generalizes well. This fixes striding and pinning in one
place, at the point where the narrow has already happened.

Rejected alternative: pinning inside the iterator's `_load_file`
(`weight_utils.py:829`) is a smaller diff but worse — it would pin
`(num_threads+2) x shard_size` ≈ 47 GB **per rank** (188 GB across 4 ranks)
and still leave the source strided after the narrow.

Use the existing clone-diff-verify harness
(`lustre-loading-exp/scripts/lib/patch_sglang_in_container.sh`) to apply it.

Ordering matters: that microbenchmark is minutes on one node and needs no
privileges, whereas nsys (PLAN's stated fallback) is a heavier tool that would
mostly confirm what the microbenchmark decides more cheaply.

## Phase 3 — microbenchmark: the fix is staging + thread capping (jobs 75718, 75726)

`scripts/h2d_microbench.py`, real down_proj geometry ([8192, 28672] bf16,
TP=4 slice = 117 MB either layout), one shared tmpfs file with each rank
narrowing its own slice. Mean GB/s across the 4 ranks:

**4 processes, DEFAULT threads (what production does today)**

| layout | pageable | pinned | pinned2x | gather | dma_only | batch | batch_mt |
|---|---|---|---|---|---|---|---|
| contig | 5.50 | 2.47 | 2.72 | 2.61 | 17.30 | 3.04 | 4.84 |
| strided | **2.05** | 2.98 | 2.96 | 3.04 | 18.45 | 3.38 | 5.12 |

**4 processes, 16 threads each (64 cores / 4 ranks)**

| layout | pageable | pinned | pinned2x | gather | dma_only | batch | batch_mt |
|---|---|---|---|---|---|---|---|
| contig | 6.30 | **8.64** | 7.45 | 8.86 | 21.15 | 5.25 | 4.45 |
| strided | **2.23** | **7.70** | 7.09 | 11.16 | 23.83 | 3.87 | 7.29 |

### The first conclusion (job 75718) was wrong

75718 ran only with default threads and concluded "host memory bandwidth is
the wall, staging is dead." The tell that it was wrong: **aggregate** gather
throughput went *down* from 1 proc to 4 (22.56 -> 9.11 GB/s) while the
driver's pageable copy went *up* (7.93 -> 22.00). A real bandwidth ceiling
caps both at a similar aggregate; throughput moving backwards is contention
pathology, not saturation.

Cause: **OMP oversubscription.** torch's CPU `copy_` parallelizes over OMP
threads, and nothing sets `OMP_NUM_THREADS`, so 4 ranks x 64 default threads
compete for 64 cores. Capping to 16/rank recovers it:

| 4-proc strided | default | capped | |
|---|---|---|---|
| `pageable` | 2.05 | 2.23 | 1.09x (unaffected) |
| `gather` | 3.04 | **11.16** | **3.67x** |
| `pinned` | 2.98 | **7.70** | **2.58x** |

`pageable` is nearly unchanged because the driver's internal memcpy does not
use OMP — which is precisely why it appeared to scale while every staged mode
collapsed.

### What actually helps

**Staging + thread capping: `pinned` 7.70 vs `pageable` 2.23 on strided =
3.45x.** Both parts are required; neither works alone. Thread capping by
itself does nothing (`pageable` 1.09x), because today's path never touches
OMP.

**Batching into a large multi-tensor buffer does NOT add anything over
plain per-tensor staging** — `batch_mt` 7.29 vs `pinned` 7.70 (strided), and
clearly worse on contiguous (4.45 vs 8.64). The idea was to parallelize the
gather, but a single `copy_` into pinned memory *already* parallelizes
internally across OMP threads; an explicit ThreadPoolExecutor just fragments
the same work and adds a GPU-side split. Testing it is what exposed the
oversubscription bug, so it earned its keep as a diagnostic, but it is not
the design to ship.

`pinned2x` (double buffering) is also consistently a slight *loss* vs
`pinned` (7.09 vs 7.70): extra memory pressure on a memory-bound workload,
for overlap the driver was already achieving.

### Revised projection

Applying per-tensor pinned staging + `torch.set_num_threads(cores/tp)`:

| path | share of window | today | staged | |
|---|---|---|---|---|
| row-parallel (strided) | 65.9% = 13.1 s | 2.23 GB/s | 7.70 GB/s | -> 3.8 s |
| column-parallel (contig) | 21.2% = 4.2 s | 6.30 GB/s | 8.64 GB/s | -> 3.1 s |

**weight_loading 19.9 s -> ~9.4 s**, i.e. ~10 s off a ~190 s cold start
(~5%). Worth a patch, though not transformative — and see the caveat below.

Still dead: **whole-model pinning.** `cudaHostRegister` measures 2.59-4.56
GB/s across all runs, so 141 GB costs **31-54 s** against a >8 GB/s
break-even. Also settled: `/dev/shm` is 353 GB, mounted **without** `huge=`,
so the hugepage variant would need a remount unavailable in the container.

### Caveat on the projection

The microbenchmark has no concurrent file-reading prefetch threads, while the
real loader runs an 8-thread `ThreadPoolExecutor` reading safetensors at the
same time as the copies. The real CPU budget is therefore more contended than
this measures, and the thread-cap tuning interacts with `num_threads` in
`model_loader_extra_config`. Treat ~10 s as an optimistic ceiling; the
patched end-to-end run is what settles it.

Variance is also high in the capped 4-proc cells (contig `pinned` ranged
5.93-15.22 across ranks), so the means above are directional, not precise.

## Methodology notes / gotchas

Worth recording, since they cost several jobs:

- **The speedscope time axis is *sampled* time, not wall clock.** py-spy's
  exporter gives every sample a fixed weight of `1/rate`, so dropped samples
  silently compress the axis (job 75707: 2417 errors / 8787 attempts = 27%;
  its 4 TP profiles read 73 s when the processes actually lived ~110 s+).
  Mapping servekit's phase offsets onto it therefore does not work.
  `analyze_pyspy_window.py` selects by **stack content** instead — anything
  under `load_model` is the weight_loading window by construction — which is
  immune to the drift. `compute_weight_loading_window.py` (which does the
  offset arithmetic) is kept only as a sanity cross-check, not as the basis
  for the numbers above.
- **100 Hz across ~80 processes cannot keep up**: job 75700 fell 146 s behind
  real time and blew its time limit. 20 Hz + `--nonblocking` is the workable
  setting; `--nonblocking` costs a handful of torn, non-UTF8 frame names
  (4 of 3813 in 75707 — decode with `errors="replace"`).
- **Don't let py-spy's `--duration` expire during the post-ready benchmark.**
  Jobs 75705 and 75706 both died at exactly the `--duration` boundary while
  the bench was hammering the GPUs ("task 0: Terminated", step CANCELLED, no
  OOM or time-limit signature); `setsid` did not help. Ending py-spy well
  before ready (150 s) avoids it. Suspected cause is ptrace-detaching from
  CUDA-active processes under load.
- **The container tears down the instant the sglang process tree exits**, so
  anything the sbatch script writes after that is lost — job 75707 lost its
  `meta.txt` that way. Write artifacts before the server goes down.
