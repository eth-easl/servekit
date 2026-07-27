# Hugepage-backed sharded_state weight loading -- running notes

See `/users/yboughizane/.claude/plans/how-are-we-going-floofy-peach.md` for
the full design plan. This file tracks results as they come in.

## Background

`shm-weight-loading-exp/STEP4_PLAN.md` established that a hugetlbfs *mount*
based design is dead (ftruncate only accepts exact 2MB multiples on the one
writable hugetlbfs mount in this container; a padded file makes
`safetensors.safe_open` reject it outright). The fix: stage into an
**anonymous** `memfd_create(MFD_HUGETLB)` buffer instead (no filesystem path,
no ftruncate-exact-size problem), and hand that buffer to consumers directly
rather than through `safe_open`.

Applied here to the **pre-sharded checkpoint path**
(`save_sharded_state_fixed.py` / `--load-format sharded_state`): each TP rank
already reads only its own distinct `model-rank-{rank}-part-*.safetensors`
files (confirmed in `model_loader/loader.py:1392-1434`), and every tensor is
contiguous by construction (the TP narrow was done once, offline, at save
time) -- so no strided-copy handling is needed, and no `layers/linear.py`
change either (`ShardedStateLoader.load_model` does its own inline
`param_data.copy_(tensor)`).

## Components built and validated standalone (no SGLang)

- `hugepage_safetensors.py`, `hugepage_stager.py`, `hugepage_fd_broker.py` --
  moved from `shm-weight-loading-exp/scripts/`, validated in that experiment
  (jobs 76127-76130): fd-sharing across 4 separately-spawned processes works,
  tensors reconstructed bit-exact, `cudaHostRegister` on real data measured
  9.46-17.77 GB/s under 4-way contention (35.71 GB/s isolated) vs. 2.59-4.56
  GB/s on 4KB tmpfs.
- `hugepage_stage_daemon.py` -- external process, stages the WHOLE sharded
  checkpoint (all ranks' files) into one memfd, serves it over the broker.
  Smoke-tested standalone (job 76159, **PASS**): staged 141.16 GB in 18.60s
  (7.59 GB/s), rank-0 filter matched the correct 7 files, all 483 tensors
  across those files checksum-matched the source exactly, `PipelinedRegistrar`
  register/join/unregister cycle completed with no errors.
- `hugepage_client.py` -- broker fetch + per-rank file filtering +
  `PipelinedRegistrar` (register shard N+1 while shard N's copies drain,
  per-rank, matching STEP4_PLAN.md's Phase 2 design).
- `hugepage_sharded_h2d.patch` -- single, self-contained patch to
  `model_loader/loader.py`'s `ShardedStateLoader.load_model`. Independent of
  both `h2d_pinned_staging.patch` and `sharded_pinned_h2d.patch` (never
  applied together). Preflight-verified (job 76160): applies byte-for-byte
  against the real installed sglang v0.5.10, all new modules import cleanly.

## First full e2e run (job 76161) -- real result, with a caveat

`ARM=hugepage`, sharded checkpoint read directly from Lustre (no `/dev/shm`
pre-stage at all -- the daemon's memfd stage replaces it).

- Staged 141.16 GB in 17.85s (7.91 GB/s).
- Benchmark: 64/64 requests OK, 0 errors, correct completions.
- servekit reported `weight_loading: 5.34s` -- **but this is misleading.**
  It captures only the first "Load weight end" log line (TP0). Looking at
  the raw log: all 4 ranks start loading at the same second; TP0 finishes at
  5.34s, but TP1/TP2/TP3 all finish at ~9.93-10.07s, and **"Capture cuda
  graph begin" for every rank doesn't start until all 4 have finished**
  (13:21:51, right after the slowest rank). The true gating weight-loading
  time is **~10.07s**, not 5.34s -- only marginally better than the existing
  `sharded_pin` baseline (10.60s, job 76115), not the ~2x win the raw number
  implied.

**Why TP0 is ~2x faster than TP1-3:** likely `cudaHostRegister` contention
across the 4 concurrently-registering ranks, consistent with the Phase 0 gate
finding (35.71 GB/s isolated vs 9.46-17.77 GB/s under 4-way concurrent
registration -- a 2-4x slowdown). TP0 probably wins a head start (first to
connect to the broker / begin registering) and runs largely uncontended,
while TP1-3 catch up moments later and then contend with each other for the
rest of the run (consistent with them finishing within 0.14s of each other).

Note this run's `hp-hugepage-76161.out` also hit a cosmetic bug in the e2e
sbatch's own result-printing tail (`set -e` + `pipefail` + a non-matching
`grep` on the hugepage arm's stage log, which has no `stage_wall_s=` line
since there's no dd-based stage) -- sbatch reported job state FAILED even
though the real work (staging, serving, weight loading, benchmark) all
completed successfully and correctly. Fixed (`|| true` on that line).

## Rank-imbalance instrumentation (jobs 76204, 76208) -- broker ruled out

Added rank-tagged timing to `hugepage_client.py` and made
`hugepage_fd_broker.py`'s `_serve()` handle connections concurrently
(thread-per-connection, not a serial accept loop) to rule out the broker as
the source of the TP0-vs-TP1-3 gap. Both preflight-verified and smoke-tested
clean. This groundwork turned out not to matter -- the real bug (below) was
elsewhere, and once fixed the rank imbalance disappeared on its own (all 4
ranks now finish within ~0.5s of each other; see final numbers).

## The real bug: cudaMemcpyAsync can't span two separately-registered chunks

Switching `PipelinedRegistrar` to 1 GiB chunked registration (instead of one
`cudaHostRegister` per whole file) immediately started crashing with
`cudaMemcpyAsync failed rc=1` (`cudaErrorInvalidValue`), reproducibly, on
every rank, every run. A long chain of wrong theories, each tested and
disproven in turn:

1. **Cross-thread `cudaHostRegister`/`cudaMemcpyAsync`** (background thread
   racing the copy loop) -- web research found a corroborating NVIDIA forum
   thread describing exactly this as unreliable/driver-version-dependent.
   Fix attempt: made `PipelinedRegistrar` fully single-threaded. **Still
   crashed, identically.**
2. **Fresh `cudaHostRegister` racing a still-draining previous chunk's DMA**
   (same-thread, but temporally concurrent). Fix attempt: full
   `cudaDeviceSynchronize()` before every registration. **Still crashed,
   identically**, and at nearly the same point every time.
3. **Two separate `libcudart` instances loaded in-process** (torch's own
   bundled copy vs. the system CUDA toolkit's copy that
   `ctypes.util.find_library` resolves to -- confirmed as genuinely two
   different files/loaded instances via `/proc/self/maps`, a real
   discrepancy, just not the cause of this bug). Fix attempt: resolve the
   already-loaded library via `/proc/self/maps`. **Still crashed,
   identically.**

Added surgical diagnostics (job 76258): a `cudaPeekAtLastError`/
`cudaGetLastError` sticky-error check plus exact pointer/size/tensor-key
dump at the failure point. This showed the **exact same tensor**
(`model.layers.1.mlp.gate_up_proj.weight`) failing every single time, on
every rank, immediately after the chunk that tensor's byte range straddled
had just been registered.

**Root cause**: `cudaMemcpyAsync`'s source range cannot span two
*separately* `cudaHostRegister`'d regions, even when they cover physically
contiguous, adjacent bytes. Blindly slicing every N bytes into a chunk (as
all versions above did) will eventually cut a chunk boundary through the
middle of some tensor, and copying that tensor then fails. **Fix**: build
chunks by walking each file's tensors in offset order and cutting only at a
tensor's end offset, accumulating toward the target chunk size but never
landing mid-tensor. Verified (job 76259): correct, 0 errors, 64/64 requests
OK, ranks balanced within 0.5s.

## Registration-lookahead tuning, once correctness was fixed

With the crash gone, the earlier "register the next file's first chunk only"
approach turned out to leave the copy loop blocking on registration
constantly -- confirmed by adding an always-on `wait_for()` timing print
(not gated on verbose logging, since this is the one number that answers "is
lookahead deep enough"). Tried, in order:

| design | chunk size | gated weight_loading | wait_for blocks/rank |
|---|---|---|---|
| device sync, 1-chunk lookahead (job 76266) | 1 GiB | 7.54s | not measured |
| **stream sync, 1-chunk lookahead (job 76269)** | **1 GiB** | **7.03s (best)** | 112 (heavy, but see below) |
| stream sync, whole-next-file lookahead (job 76271) | 1 GiB | 8.24s | 20 |
| stream sync, continuous 3-chunk lookahead (job 76272) | 100 MB | 9.40s | 4 |
| stream sync, continuous 3-chunk lookahead (job 76274) | 500 MB | 9.21s | 4 |

Counterintuitive result: the run with the MOST measured blocking (76269,
112 blocks/rank, 1 GiB chunks) is still the FASTEST overall, and the runs
with essentially ZERO blocking (100-500 MB chunks + continuous 3-chunk
lookahead, which does eliminate blocking) are slower. Read together with the
"whole file" result also being worse, the pattern points at per-call
`cudaHostRegister` overhead dominating once chunk count gets high: 1 GiB
chunks mean ~35 registrations/rank total; 100 MB chunks mean ~350/rank.
Fewer, larger registrations -- even if a few of them briefly block the copy
loop -- costs less in aggregate than many small ones that never block at
all. **Not yet tested: 1 GiB chunks WITH the continuous 3-chunk lookahead
design** (the lookahead redesign happened after the chunk-size sweep moved
to smaller sizes) -- that combination might beat 7.03s and is the natural
next data point.

`cudaStreamSynchronize(stream_ptr)` (scoped to just this rank's copy stream)
instead of `torch.cuda.synchronize()` (whole device) was a small, clean,
low-risk win on its own (7.54s -> 7.03s at the same 1-chunk-lookahead
design) and stays regardless of chunk-size conclusions.

## Current best confirmed number

**7.03s gated `weight_loading`** (job 76269: 1 GiB chunks, tensor-boundary-
aligned, stream-scoped sync, register-ahead-one-chunk), down from:
- 20.5s -- default (unsharded) loader baseline
- 10.60s -- today's best known number (`sharded_pin`, job 76115)
- **7.03s -- this design**

0 errors, 64/64 benchmark requests OK, ranks balanced within ~0.3-0.5s of
each other (the original TP0-vs-TP1-3 2x gap is gone -- it was a symptom of
the same registration-granularity issues explored above, not a separate
bug).

## Next steps

1. Test 1 GiB chunks + continuous 3-chunk lookahead (the untested
   combination noted above) -- may beat 7.03s.
2. Confirm with `nsys stats --report cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum`
   that DMA-active time approaches the phase wall time (STEP4_PLAN.md's
   original verification bar).
3. Consider whether `cudaHostRegister` call COUNT vs. TOTAL BYTES REGISTERED
   is the real lever to optimize (current evidence points at count), which
   would argue for staying near or above 1 GiB chunks rather than chasing
   smaller ones further.
