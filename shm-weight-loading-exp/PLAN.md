# Plan: shm-weight-loading-exp — why is weight_loading (18.6-19.6s) slower than the stage (12.1s)?

## Context

The sliced `/dev/shm` staging experiment (`lustre-loading-exp/phase4_shm`,
sliced variant) copies the full 132GB model from Lustre into tmpfs in
**12.1s** (~11.7 GB/s, O_DIRECT, 1744 concurrent slice-readers). Yet SGLang's
own `weight_loading` phase — reading that *same, already-in-RAM* model with
the default mmap loader — takes **18.6-19.6s**, consistently, across all
three sliced-mmap runs (75168, 75196, 75335). That's backwards: touching
bytes already resident in RAM should be far cheaper than reading them from
Lustre, not 1.5-1.6x more expensive.

`lustre-loading-exp/NOTES.md` (Phase 3/4) already attributes tmpfs
weight_loading cost to "non-overlapped H2D copy + NCCL cross-rank broadcast +
param materialization" — but that was inferred from timing arithmetic, never
measured directly. This plan's only goal is to **measure** where the 18-19s
actually goes, using the same "instrument before treating" discipline that
found `SGLANG_FST_FILES_PER_RANK` for the loading bottleneck and is currently
being applied to graph capture in `experiments/graph-compile-cache-exp/`. No fix is
committed here — the deliverable is a breakdown, plus a recommendation on
whether a follow-up (e.g. hugepage-backed tmpfs, or removing an unnecessary
cross-rank broadcast) is worth pursuing.

## Working location

New sibling directory: `shm-weight-loading-exp/` at the repo root (same
convention as `lustre-loading-exp/`, `experiments/graph-compile-cache-exp/`), with
`PLAN.md`, `NOTES.md`, `scripts/`, `results/`.

## Phase 1 — instrument the weight_loading window with py-spy

- Reuse `lustre-loading-exp/scripts/phase4_shm/phase4_shm_sliced_e2e.sbatch`
  (sliced stage + mmap loader) as the launch vehicle unmodified — it's
  already the fastest known /dev/shm baseline, so the profile reflects the
  real bottleneck, not an inflated one.
- py-spy is pre-installed in the container and same-uid `PTRACE_ATTACH`
  already works here (verified previously, no capability changes needed).
  From the same sbatch, after launching the server in the background, run
  `py-spy record --pid <sglang_pid> --subprocesses --rate 100 --output
  results/.../pyspy-<job>-<node>.speedscope --format speedscope` spanning the
  full startup-to-ready window (a few hundred seconds is fine — `--subprocesses`
  follows the TP worker processes too, where the actual H2D/NCCL calls happen).
- `servekit profile`'s phases are sequential and reported as `duration_s`
  only (no absolute per-phase timestamps in the JSON) — recover the
  weight_loading window post-hoc by summing prior phases' durations from
  `started_at` to get its start offset, then adding its own `duration_s` for
  the end offset. Filter the py-spy samples to that window for analysis.

## Phase 2 — classify where the samples land

- Expect one of: (a) time inside the mmap-backed tensor construction /
  numpy-torch copy path (page-fault-driven, since `/dev/shm` is tmpfs and
  every 4KB touch still costs a soft page fault even with zero disk I/O
  behind it), (b) `cudaMemcpyAsync`/`.to(device)` H2D copy calls, (c)
  `torch.distributed` collective calls (broadcast/all-reduce across the 4 TP
  ranks), (d) generic Python-loop overhead in the weight iterator.
- py-spy is a Python-level sampler, so C-level CUDA/NCCL time will show up
  attributed to whichever Python call entered it (e.g. `Tensor.to`,
  `dist.broadcast`) — coarse but sufficient to rank the categories.
- **Fallback if inconclusive**: `nsys profile` (Nsight Systems), if present
  in the container, for an actual CUDA-kernel/NCCL timeline. Only pursue if
  py-spy's attribution is too coarse to distinguish H2D from collective time
  — heavier tool, not the default path.

## Phase 3 (conditional, only if Phase 2 points there) — one confirming test

- **If page-fault/mmap-copy dominates**: tmpfs supports `huge=always` as a
  mount option for transparent hugepages, which would cut the ~34M 4KB page
  faults for a 132GB touch down to ~64K 2MB faults. This likely requires a
  node-level remount (root), which may not be available inside the
  unprivileged container — confirm feasibility before committing a job to
  it; if infeasible, record as a documented ceiling instead of a dead end.
- **If cross-rank broadcast dominates**: since `/dev/shm` is node-local and
  already identically visible to all 4 TP ranks, a broadcast may be
  structurally unnecessary here (unlike Lustre, where fastsafetensors reads
  different files per rank and needs to redistribute) — worth checking
  whether SGLang's default loader path already reads each rank's own shard
  independently or actually round-trips through rank 0. This would be a
  distinct follow-up patch, out of scope for this measurement-only plan;
  just flag it with the evidence for a future experiment.

## Deliverable

`NOTES.md`: py-spy-derived time breakdown of the 18-19s weight_loading
window into the categories above (rough percentages, not exact), with the
supporting speedscope file referenced. A recommendation on whether hugepages
and/or a broadcast-removal patch are worth a dedicated follow-up experiment,
or whether this is close to a floor (analogous to how InstantTensor and
striping were ruled out elsewhere in this project).

## Verification

- The instrumented run still reaches "fired up and ready to roll", emits a
  normal `servekit profile` JSON, and (if bench is enabled) reports 0 errors
  — py-spy attachment must not perturb correctness or throughput.
- Success = a clear ranking of where the 18-19s goes (even if it's "all in
  one bucket, no easy split"), sufficient to make a go/no-go call on Phase 3
  without further guessing.
