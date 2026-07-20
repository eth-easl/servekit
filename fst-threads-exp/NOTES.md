# Notes — fastsafetensors reader concurrency

## TL;DR — `max_threads` is a NEGATIVE result, and the hypothesis was wrong

`PLAN.md` predicted that raising fastsafetensors' `max_threads` (default 16)
would cut weight_loading, and would compose with phase 3's
`SGLANG_FST_FILES_PER_RANK`. **It does the opposite.**

| tag | files_per_rank | max_threads | weight_loading | in-job capstor (sliced T=8) | node |
|---|---|---|---|---|---|
| `ctl_first` | 1 | default (16) | **76.75 s** | 5.20 GB/s | nid002297 |
| `mt32` | 1 | 32 | **104.01 s** (+35%) | 5.41 GB/s | nid002321 |

Sweep stopped after `mt32`: the source explains the regression, and the
remaining points (`mt64`, `mt128`, `fpr8_mt64`) would only have bought more of
the same at ~8 min of GPU each.

**The null run reproduced the baseline** (76.75 s, inside the established
69–93 s band), so the harness is sound and the regression is real. The in-job
probes were flat (5.20 vs 5.41 GB/s; OST 8 at 25.7 vs 26.7 MB/s), so it is not
storage drift.

## Why — from the v0.3.3 source, not from a guess

1. **`max_threads` is not a read-concurrency knob at this call site.**
   `NoGdsFileCopier.submit_io()` chunks a file by **`max_copy_block_size`** and
   submits **one read per chunk**, one thread each. SGLang calls
   `loader.copy_files_to_device()` with no arguments → the default **16 GiB**.
   Every shard is 5 GB, under 16 GiB, so each file yields **exactly ONE submit →
   ONE thread → one sequential `pread` loop**, regardless of `max_threads`.
2. **So raising it is pure cost.** `submit_read()` eagerly
   `cudaHostAlloc`s `bbuf_size_kb * max_threads` of **pinned** host memory per
   rank: 256 MB at mt16, 512 MB at mt32, ×4 ranks. Page-locking that is slow and
   buys nothing. That is the +27 s.
3. **The real source of fastsafetensors' queue depth is readahead, not
   threads.** It opens files buffered (`os.open(src, O_RDONLY)`, no `O_DIRECT`),
   so the Lustre client reads ahead to `max_read_ahead_per_file_mb=160` ≈ 10 ×
   16 MiB RPCs in flight. Measured ramp on the sick OST-8 shard: buffered starts
   at 30 MB/s (== O_DIRECT) and climbs to 356 as the window opens; O_DIRECT stays
   flat. See `lustre-contention-exp/DD_VS_FASTSAFETENSORS.md` §4a.

## The live candidate: `max_copy_block_size`

Set it to e.g. 256 MB and a 5 GB shard becomes ~20 concurrent reads — real
queue depth, which the sliced `dd` sweep showed is worth 26× to raw reads.

It should help **even on healthy OSTs**, for a reason unrelated to OST 8: each
thread's loop is `pread → cudaMemcpy → pread` **serially**, so with one thread
per file the **H2D copy never overlaps the read**. That is plausibly a large
part of the ~20 s of non-overlapped GPU-side work `lustre-loading-exp` phase 3
measured but could not remove — and it is what PR sgl#28506 (InstantTensor)
claims to attack structurally.

Wired as `SGLANG_FST_MAX_COPY_BLOCK_SIZE_MB` in `scripts/fst_knobs.patch`
(superset: `FILES_PER_RANK` + `MAX_THREADS` + `BBUF_SIZE_KB` + `MAX_COPY_BLOCK_SIZE_MB`).
Driver ready at `scripts/submit_mcbs_sweep.sh`. **Not yet run.**

## Files

```
scripts/
  fst_threads.patch      # fpr + max_threads + bbuf. Produced the mt results above.
  fst_threads.sbatch     #   kept as-is so those numbers stay reproducible.
  submit_sweep.sh
  fst_knobs.patch        # superset, adds max_copy_block_size  <- use this one
  fst_knobs.sbatch
  submit_mcbs_sweep.sh
  preflight.sbatch       # GPU-less: patch applies, kwargs accepted, null path == upstream
results/
  sweep.log, fst-ctl_first-75160.out, fst-mt32-75161.out, *-profile.json
```

Both patches are generated with `git diff` against the pinned SHA
(`1519acf37c23f2189adb93f57ca9cd2db1bebf18`), never hand-written, and with no
env vars set both reduce to **byte-for-byte upstream** (verified in
`preflight.sbatch`: `files_per_rank=1` → `chunk = pg.size()`; empty kwargs →
`SafeTensorsFileLoader(pg, device)` and `copy_files_to_device()`; round-robin
`rank_file_map` == upstream's `enumerate()`, short final chunk included).

**Never pass `fst_threads.patch` and `fst_knobs.patch` together**, or either
with phase 3's `fst_files_per_rank.patch`: they touch the same lines of the same
function, and `patch_sglang_in_container.sh` cannot compose two patches over one
region — it re-clones pristine per call and its version check would trip.

## Method notes worth keeping

- **The in-job storage probe is the SLICED one (T=8), not `dd_read_sweep.sh`.**
  The old probe fans out across files only → queue depth 1 per OST → it reports
  the *worst* OST's latency as bandwidth (0.72 GB/s while the node was doing
  18.9). Normalizing loader runs against that is worse than having no probe.
- **The negative result cost ~2 GPU jobs and was worth it.** The prediction in
  `PLAN.md` was plausible, sourced, and wrong — and only the *measurement* plus
  reading the C++ separated "the knob does nothing" from "the knob backfires".
- `PLAN.md` said a null result would be a real result that pins the floor and
  redirects effort. It half-was: the floor is not where it guessed, because the
  axis it named turned out not to exist.
