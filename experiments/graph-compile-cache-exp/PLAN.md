# Plan: graph-compile-cache-exp — does torch.compile/Inductor cache survive a cold restart?

## Context

`lustre-loading-exp` drove weight-loading cold start from 940s (mmap default)
down to ~19-38s (fastsafetensors+fpr8, or sliced /dev/shm staging). With
loading solved, `piecewise_cuda_graph_capture` (~79s) + `cuda_graph_capture`
(~27s) = **~106s, 57% of the remaining ~185s server-side total** is now by far
the largest phase (see `lustre-loading-exp/NOTES.md` "Phase 6 (next) — graph
capture ⏳", never started). `piecewise_cuda_graph_capture` is SGLang/torch.compile
(Inductor) compiling Triton kernels per batch-size bucket before CUDA-graph
capture — a classically cache-able cost, and near-constant (78.9-79.5s) across
every loader/node measured so far, which smells like a fixed compile cost, not
an I/O one. Nothing has been measured yet; this experiment runs the single
cheapest test before any patch or design work: **does Inductor's on-disk
compile cache actually eliminate recompilation on a second cold start**, and
does it survive across nodes (not just the same one).

Explicitly out of scope (per user decision): reducing the number of captured
batch-size buckets — not to be pursued.

Also folds in one small, free-riding second probe: `weight_loading` from
`/dev/shm` (mmap) is 18.6-19.6s even though the tmpfs stage itself is only
12.1s and storage cost is zero. Phase 4 already attributed this to
non-overlapped H2D + NCCL broadcast + param materialization, but that was
never profiled directly — worth a cheap look using the same launch script
before deciding whether it's worth a dedicated effort later.

## Working location

New sibling directory (same convention as `fst-threads-exp`,
`lustre-contention-exp`, `lustre-loading-exp`): `experiments/graph-compile-cache-exp/` at
the repo root, with `PLAN.md`, `NOTES.md`, `scripts/`, `results/`. This is a
clean break from `lustre-loading-exp` per the user's request — nothing under
`servekit/` or `lustre-loading-exp/` is touched, though its scripts/harnesses
are reused by reference.

## Phase 1 — locate + relocate the Inductor cache (no code changes)

1. Reuse `lustre-loading-exp/scripts/phase4_shm/phase4_shm_sliced_e2e.sbatch`
   as the base launch script (sliced /dev/shm staging, mmap loader — the
   fastest known baseline, so graph-capture is measured against the best
   floor, not an inflated one).
2. Before launching SGLang, set `TORCHINDUCTOR_CACHE_DIR` and
   `TORCHINDUCTOR_FX_GRAPH_CACHE=1` to point at a path that survives across
   job submissions — NOT the ephemeral container `$HOME`. Candidate: a
   bind-mounted scratch path already visible in the container (`/capstor` or
   `/iopsstor` per the `.toml` env-def mounts, e.g. the existing
   `infra01`-group-writable area used in Phase 2 of `lustre-loading-exp`).
3. Confirm the SGLang/torch version in the container actually routes the
   piecewise path through `torch.compile` (grep vendored `sglang` source for
   `torch.compile` / `enable_torch_compile`) so the cache-dir env vars are
   known to apply before spending a job on it.

## Phase 2 — same-node repeat (does the cache work at all)

- One `--exclusive` allocation, run the sliced-shm launch **twice** in the
  same job (same node, same GPU/driver), with the persistent cache dir set
  both times. Record `piecewise_cuda_graph_capture` + `cuda_graph_capture`
  for both runs.
- Bracket with a **third run using the default ephemeral cache location**
  (today's baseline) on the same node, to rule out node-specific variance —
  same "bracket every sweep with a control" rule as `lustre-loading-exp`.
- **Success signal**: run 2 (warm cache) drops materially below run 1 and the
  ephemeral-cache control. A null result (no drop) means the cache isn't
  being hit — check invalidation (PID/tmp-path salting, per-process compile
  keys) before concluding it doesn't help.

## Phase 3 — cross-node repeat (does it actually deploy)

- Only run if Phase 2 shows a real effect. Two **distinct fresh nodes**
  (never `--dependency` chains — the Phase-3 lesson from `lustre-loading-exp`
  about SLURM handing back the same node applies here too), same persistent
  cache dir, same GPU model. First node populates the cache (cold), second
  node consumes it (should be warm if the cache generalizes across
  nodes/processes, not just same-PID reuse).
- This is the number that matters for production: a same-node-only win isn't
  useful for cold starts landing on arbitrary nodes.

## Deliverable

`NOTES.md` table: ephemeral-cache baseline vs same-node warm vs cross-node
warm, for both graph-capture phases, plus a short paragraph on the
weight_loading profile finding. A go/no-go recommendation on whether to build
a real cache-persistence mechanism into the deployment (e.g. shipping a
prewarmed cache dir alongside the model, analogous to the `/dev/shm` staging
story), or whether it's a dead end like InstantTensor/striping.

## Verification

- Each run reaches "fired up and ready to roll", `servekit profile` emits a
  profile JSON, and the bench (`--bench`) reports 0 errors, matching
  throughput to prior runs — the compile cache must not change model
  outputs.
- Success = a reproducible, materially lower `piecewise_cuda_graph_capture`
  time with a warm cache vs the ephemeral-cache control, confirmed on ≥2
  distinct nodes for the cross-node case.
